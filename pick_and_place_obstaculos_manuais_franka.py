import time
from pathlib import Path

import numpy as np

from cinematica_franka import matriz_coppelia_para_homogenea
from pick_and_place_obstaculos_franka import (
    ACTION_PAUSE,
    APPROACH_HEIGHT,
    GRASP_HEIGHT,
    HOME_Q,
    HOST,
    PORT,
    SAVE_RESULTS,
    TRANSPORT_STEPS_PER_SEGMENT,
    densify_cartesian_route,
    enable_collision_for_obstacle,
    fk_world,
    get_alias,
    get_franka,
    get_tip,
    min_route_clearance,
    move_to_q,
    route_length,
    save_execution_csv,
    save_metrics_csv,
    save_route_plot,
    save_waypoints_csv,
    segment_intersects_aabb,
    set_cube_parent,
    warn_if_route_crosses_obstacles,
)


OUTPUT_DIR = Path("resultados") / "obstaculos_manuais"

# Crie os obstaculos manualmente no CoppeliaSim e use um destes prefixos.
OBSTACLE_PREFIXES = (
    "Obstaculo",
    "Obstacle",
    "Parede",
    "Muro",
    "Coluna",
    "Barreira",
)

# Se preferir, coloque nomes exatos aqui. Exemplo: ("CaixaGrande", "Pilar01")
MANUAL_OBSTACLE_NAMES = ()

# Se existir um objeto com um destes nomes, ele vira o destino da entrega.
# Caso contrario, o destino sera Cuboid + DROP_OFFSET.
DESTINATION_OBJECT_NAMES = (
    "Destino_Manual",
    "Destino",
    "Base_Entrega",
)

DROP_OFFSET = np.array([-0.35, 0.30, 0.0])

AUTO_ENABLE_COLLISION = True
AVOIDANCE_MARGIN = 0.18
TOP_CLEARANCE = 0.20
MAX_SAFE_Z = 0.76
ROUTE_MAX_STEP = 0.07
VERTICAL_MAX_STEP = 0.025
MAX_IK_ERROR = 0.018

# "auto" escolhe o lado mais curto; use "negative_y" ou "positive_y" para fixar.
PREFERRED_SIDE = "auto"


def short_name(alias):
    return str(alias).strip("/").split("/")[-1]


def object_matches_manual_obstacle(alias):
    name = short_name(alias)
    if alias in MANUAL_OBSTACLE_NAMES or name in MANUAL_OBSTACLE_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in OBSTACLE_PREFIXES)


def list_scene_shapes(sim):
    for root in (getattr(sim, "handle_scene", None), -1):
        if root is None:
            continue
        try:
            return sim.getObjectsInTree(root, sim.object_shape_type, 0)
        except Exception:
            pass

    shapes = []
    index = 0
    while True:
        try:
            handle = sim.getObjects(index, sim.object_shape_type)
        except Exception:
            break
        if handle == -1:
            break
        shapes.append(handle)
        index += 1
    return shapes


def get_float_param(sim, handle, param):
    value = sim.getObjectFloatParam(handle, param)
    if isinstance(value, (list, tuple)):
        value = value[-1]
    return float(value)


def get_shape_world_aabb(sim, handle):
    min_x = get_float_param(sim, handle, sim.objfloatparam_objbbox_min_x)
    max_x = get_float_param(sim, handle, sim.objfloatparam_objbbox_max_x)
    min_y = get_float_param(sim, handle, sim.objfloatparam_objbbox_min_y)
    max_y = get_float_param(sim, handle, sim.objfloatparam_objbbox_max_y)
    min_z = get_float_param(sim, handle, sim.objfloatparam_objbbox_min_z)
    max_z = get_float_param(sim, handle, sim.objfloatparam_objbbox_max_z)

    local_corners = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])

    t_world = matriz_coppelia_para_homogenea(
        sim.getObjectMatrix(handle, sim.handle_world)
    )
    world_corners = []
    for corner in local_corners:
        world_corner = t_world @ np.r_[corner, 1.0]
        world_corners.append(world_corner[0:3])

    world_corners = np.array(world_corners)
    p_min = np.min(world_corners, axis=0)
    p_max = np.max(world_corners, axis=0)
    return p_min, p_max


def read_manual_obstacles(sim, robot, ignored_handles):
    robot_tree = set(sim.getObjectsInTree(robot, sim.handle_all, 0))
    robot_tree.add(robot)

    obstacles = []
    for handle in list_scene_shapes(sim):
        if handle in ignored_handles or handle in robot_tree:
            continue

        alias = get_alias(sim, handle)
        if not object_matches_manual_obstacle(alias):
            continue

        try:
            p_min, p_max = get_shape_world_aabb(sim, handle)
        except Exception as exc:
            print(f"Aviso: nao consegui ler o tamanho de {alias}: {exc}")
            continue

        size = p_max - p_min
        if np.any(size < 1e-4):
            print(f"Aviso: ignorando {alias}, caixa muito pequena: {np.round(size, 5)}")
            continue

        if AUTO_ENABLE_COLLISION:
            enable_collision_for_obstacle(sim, handle)

        obstacles.append({
            "name": short_name(alias),
            "handle": handle,
            "center": (p_min + p_max) / 2.0,
            "size": size,
        })

    return obstacles


def get_destination(sim, cube_start):
    for name in DESTINATION_OBJECT_NAMES:
        try:
            handle = sim.getObject(f"/{name}")
            return np.array(sim.getObjectPosition(handle, sim.handle_world)), name
        except Exception:
            pass

    return cube_start + DROP_OFFSET, "DROP_OFFSET"


def segment_crosses_obstacles(start, goal, obstacles, margin=AVOIDANCE_MARGIN):
    for obstacle in obstacles:
        inflated_half = obstacle["size"] / 2.0 + margin
        if segment_intersects_aabb(start, goal, obstacle["center"], inflated_half):
            return True
    return False


def choose_side_y(start, goal, obstacles):
    min_y = min(obs["center"][1] - obs["size"][1] / 2.0 for obs in obstacles)
    max_y = max(obs["center"][1] + obs["size"][1] / 2.0 for obs in obstacles)

    negative_y = min_y - AVOIDANCE_MARGIN
    positive_y = max_y + AVOIDANCE_MARGIN

    if PREFERRED_SIDE == "negative_y":
        return negative_y
    if PREFERRED_SIDE == "positive_y":
        return positive_y

    negative_cost = abs(start[1] - negative_y) + abs(goal[1] - negative_y)
    positive_cost = abs(start[1] - positive_y) + abs(goal[1] - positive_y)
    return negative_y if negative_cost <= positive_cost else positive_y


def safe_height(start, goal, obstacles):
    obstacle_top = max(
        obs["center"][2] + obs["size"][2] / 2.0
        for obs in obstacles
    )
    requested_z = max(obstacle_top + TOP_CLEARANCE, start[2] + 0.15, goal[2] + 0.15)

    if requested_z > MAX_SAFE_Z:
        print(
            "Aviso: a altura segura calculada passou do limite. "
            f"Usando z={MAX_SAFE_Z:.3f} m; talvez seja preciso baixar/remover obstaculos."
        )
    return min(requested_z, MAX_SAFE_Z)


def route_avoiding_manual_obstacles(start, goal, obstacles):
    if not obstacles:
        return [start, goal]

    if not segment_crosses_obstacles(start, goal, obstacles):
        return [start, goal]

    lane_y = choose_side_y(start, goal, obstacles)
    z = safe_height(start, goal, obstacles)
    middle_x = (start[0] + goal[0]) / 2.0

    print(
        "Rota direta bloqueada; usando corredor lateral "
        f"y={lane_y:.3f}, z={z:.3f}."
    )

    return [
        start,
        np.array([start[0], start[1], z]),
        np.array([start[0], lane_y, z]),
        np.array([middle_x, lane_y, z]),
        np.array([goal[0], lane_y, z]),
        np.array([goal[0], goal[1], z]),
        goal,
    ]


def vertical_route(above, low):
    return densify_cartesian_route([above, low], max_step=VERTICAL_MAX_STEP)


def solve_world_target_manual(name, target_world, t_base_world, q_seed):
    from inversa import mundo_para_base, resolver_ik

    target_base = mundo_para_base(target_world, t_base_world)
    q_sol, _, error_base = resolver_ik(
        target_base,
        q_inicial=q_seed,
        max_iter=2200,
    )
    pos_world = fk_world(q_sol, t_base_world)
    error_world = float(np.linalg.norm(pos_world - target_world))

    print(f"{name}")
    print(f"  alvo mundo : {np.round(target_world, 4)}")
    print(f"  erro modelo: {error_world:.6f} m")

    if error_world > MAX_IK_ERROR:
        raise RuntimeError(
            f"IK falhou no waypoint '{name}'. "
            f"Erro base={error_base:.6f} m, erro mundo={error_world:.6f} m."
        )

    return q_sol


def solve_cartesian_route_manual(route, route_name, t_base_world, q_seed):
    q_route = []
    current_seed = q_seed

    for i, point in enumerate(route[1:], start=1):
        current_seed = solve_world_target_manual(
            f"{route_name} {i}",
            point,
            t_base_world,
            current_seed,
        )
        q_route.append(current_seed)

    return q_route


def execute_q_route(sim, joints, tip, q_current, q_route, label, execution_log):
    for i, q_target in enumerate(q_route, start=1):
        q_current = move_to_q(
            sim,
            joints,
            q_current,
            q_target,
            f"{label} {i}",
            steps=TRANSPORT_STEPS_PER_SEGMENT,
            logger=execution_log,
            tip=tip,
        )
    return q_current


def print_route(title, route):
    print(f"\n{title}:")
    for i, point in enumerate(route):
        print(f"  p{i}: {np.round(point, 4)}")
    print(f"  pontos principais: {len(route)}")


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient(host=HOST, port=PORT)
    sim = client.require("sim")
    print("Conectado ao CoppeliaSim.")

    robot, joints = get_franka(sim)
    tip = get_tip(sim, robot)
    cube = sim.getObject("/Cuboid")

    print(f"Efetuador final usado: {get_alias(sim, tip)}")
    print(f"Cubo usado: {get_alias(sim, cube)}")

    try:
        sim.setObjectParent(cube, -1, True)
    except Exception:
        pass

    t_base_world = matriz_coppelia_para_homogenea(
        sim.getObjectMatrix(joints[0], sim.handle_world)
    )

    cube_start = np.array(sim.getObjectPosition(cube, sim.handle_world))
    drop_center, destination_source = get_destination(sim, cube_start)
    print(f"Destino usado: {destination_source} -> {np.round(drop_center, 4)}")

    ignored_handles = {cube, tip}
    obstacles = read_manual_obstacles(sim, robot, ignored_handles)
    if not obstacles:
        raise RuntimeError(
            "Nenhum obstaculo manual foi encontrado. "
            "Crie formas no CoppeliaSim com nomes iniciando por "
            f"{OBSTACLE_PREFIXES} ou preencha MANUAL_OBSTACLE_NAMES."
        )

    print("\nObstaculos manuais detectados:")
    for obstacle in obstacles:
        print(
            f"  {obstacle['name']}: centro={np.round(obstacle['center'], 4)} "
            f"tam={np.round(obstacle['size'], 4)}"
        )

    pick_above = cube_start + np.array([0.0, 0.0, APPROACH_HEIGHT])
    pick_grasp = cube_start + np.array([0.0, 0.0, GRASP_HEIGHT])
    drop_above = drop_center + np.array([0.0, 0.0, APPROACH_HEIGHT])
    drop_grasp = drop_center + np.array([0.0, 0.0, GRASP_HEIGHT])

    q_current = np.array([sim.getJointPosition(joint) for joint in joints])
    current_pos = fk_world(q_current, t_base_world)
    home_pos = fk_world(HOME_Q, t_base_world)

    approach_sparse = route_avoiding_manual_obstacles(current_pos, pick_above, obstacles)
    transport_sparse = route_avoiding_manual_obstacles(pick_above, drop_above, obstacles)
    drop_sparse = [drop_above, drop_grasp]
    return_sparse = route_avoiding_manual_obstacles(drop_above, home_pos, obstacles)

    approach_route = densify_cartesian_route(approach_sparse, max_step=ROUTE_MAX_STEP)
    transport_route = densify_cartesian_route(transport_sparse, max_step=ROUTE_MAX_STEP)
    drop_route = vertical_route(drop_above, drop_grasp)
    return_route = densify_cartesian_route(return_sparse, max_step=ROUTE_MAX_STEP)

    warn_if_route_crosses_obstacles(approach_sparse, obstacles)
    warn_if_route_crosses_obstacles(transport_sparse, obstacles)
    warn_if_route_crosses_obstacles(drop_sparse, obstacles)
    warn_if_route_crosses_obstacles(return_sparse, obstacles)

    print_route("Rota de aproximacao", approach_sparse)
    print_route("Rota de transporte", transport_sparse)
    print_route("Descida na entrega", drop_sparse)
    print_route("Rota de retorno", return_sparse)

    planned_sections = [
        ("aproximacao", approach_route),
        ("transporte", transport_route),
        ("descida_entrega", drop_route),
        ("retorno", return_route),
    ]
    metrics = {
        "manual_obstacles": len(obstacles),
        "approach_waypoints": len(approach_route),
        "transport_waypoints": len(transport_route),
        "drop_waypoints": len(drop_route),
        "return_waypoints": len(return_route),
        "approach_length_m": route_length(approach_route),
        "transport_length_m": route_length(transport_route),
        "drop_length_m": route_length(drop_route),
        "return_length_m": route_length(return_route),
        "min_clearance_planned_m": min(
            min_route_clearance(route, obstacles)
            for _, route in planned_sections
        ),
    }

    if SAVE_RESULTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_waypoints_csv(OUTPUT_DIR / "waypoints_obstaculos_manuais.csv", planned_sections)
        save_route_plot(
            OUTPUT_DIR / "trajetoria_obstaculos_manuais.png",
            planned_sections,
            obstacles,
            cube_start,
            drop_center,
        )

    print("\nCalculando IK dos waypoints...")
    q_approach = solve_cartesian_route_manual(
        approach_route,
        "aproximacao",
        t_base_world,
        q_current,
    )
    q_pick_above = q_approach[-1]
    q_pick_grasp = solve_world_target_manual(
        "pegar cubo",
        pick_grasp,
        t_base_world,
        q_pick_above,
    )
    q_transport = solve_cartesian_route_manual(
        transport_route,
        "transporte",
        t_base_world,
        q_pick_above,
    )
    q_drop_descent = solve_cartesian_route_manual(
        drop_route,
        "descida entrega",
        t_base_world,
        q_transport[-1],
    )
    q_return = solve_cartesian_route_manual(
        return_route,
        "retorno",
        t_base_world,
        q_transport[-1],
    )

    print("\nIniciando simulacao com obstaculos manuais...")
    sim.startSimulation()
    execution_log = []
    execution_start = time.perf_counter()
    time.sleep(ACTION_PAUSE)

    try:
        q_current = execute_q_route(
            sim,
            joints,
            tip,
            q_current,
            q_approach,
            "aproximacao",
            execution_log,
        )
        q_current = move_to_q(
            sim,
            joints,
            q_current,
            q_pick_grasp,
            "descer ate o cubo",
            logger=execution_log,
            tip=tip,
        )

        print("Prendendo cubo ao efetuador...")
        set_cube_parent(sim, cube, tip)
        time.sleep(ACTION_PAUSE)

        q_current = move_to_q(
            sim,
            joints,
            q_current,
            q_pick_above,
            "subir com cubo",
            logger=execution_log,
            tip=tip,
        )
        q_current = execute_q_route(
            sim,
            joints,
            tip,
            q_current,
            q_transport,
            "transporte",
            execution_log,
        )
        q_current = execute_q_route(
            sim,
            joints,
            tip,
            q_current,
            q_drop_descent,
            "descida entrega",
            execution_log,
        )

        print("Soltando cubo...")
        set_cube_parent(sim, cube, -1)
        time.sleep(ACTION_PAUSE)

        for i, q_target in enumerate(reversed(q_drop_descent[:-1]), start=1):
            q_current = move_to_q(
                sim,
                joints,
                q_current,
                q_target,
                f"subida entrega {i}",
                steps=TRANSPORT_STEPS_PER_SEGMENT,
                logger=execution_log,
                tip=tip,
            )
        q_current = move_to_q(
            sim,
            joints,
            q_current,
            q_transport[-1],
            "subir acima da entrega",
            steps=TRANSPORT_STEPS_PER_SEGMENT,
            logger=execution_log,
            tip=tip,
        )
        q_current = execute_q_route(
            sim,
            joints,
            tip,
            q_current,
            q_return,
            "retorno",
            execution_log,
        )

        final_cube = np.array(sim.getObjectPosition(cube, sim.handle_world))
        delivery_error = float(np.linalg.norm(final_cube - drop_center))
        metrics["delivery_error_m"] = delivery_error
        metrics["execution_time_s"] = time.perf_counter() - execution_start
        metrics["execution_samples"] = len(execution_log)

        print("\nResultado:")
        print(f"  posicao inicial do cubo: {np.round(cube_start, 4)}")
        print(f"  destino planejado      : {np.round(drop_center, 4)}")
        print(f"  posicao final do cubo  : {np.round(final_cube, 4)}")
        print(f"  erro de entrega        : {delivery_error:.6f} m")

        if SAVE_RESULTS:
            save_execution_csv(OUTPUT_DIR / "trajetoria_executada_manuais.csv", execution_log)
            save_metrics_csv(OUTPUT_DIR / "metricas_obstaculos_manuais.csv", metrics)
            print(f"  resultados salvos em   : {OUTPUT_DIR.resolve()}")
    finally:
        time.sleep(ACTION_PAUSE)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()

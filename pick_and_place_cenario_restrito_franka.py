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
    set_cube_parent,
    warn_if_route_crosses_obstacles,
)


OUTPUT_DIR = Path("resultados") / "cenario_restrito"

# Mantem o robo no lugar e cria uma base/plataforma em volta dele.
# Mudar o objeto /Franka inteiro costuma quebrar cenas ja ajustadas.
DROP_OFFSET = np.array([-0.34, 0.26, 0.0])
LANE_OFFSET_Y = -0.32
ROUTE_MAX_STEP = 0.07
VERTICAL_MAX_STEP = 0.025
MAX_IK_ERROR = 0.018


def remove_if_exists(sim, name):
    try:
        handle = sim.getObject(f"/{name}")
        try:
            sim.removeObject(handle)
        except Exception:
            sim.removeObjects([handle])
    except Exception:
        pass


def color_shape(sim, handle, color):
    try:
        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)
    except Exception:
        pass


def create_box(sim, name, center, size, color, collidable=True):
    remove_if_exists(sim, name)

    handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size.tolist(), 0)
    sim.setObjectAlias(handle, name)
    sim.setObjectPosition(handle, sim.handle_world, center.tolist())
    color_shape(sim, handle, color)

    if collidable:
        enable_collision_for_obstacle(sim, handle)

    return handle


def create_marker(sim, name, center, color):
    return create_box(
        sim,
        name,
        np.asarray(center, dtype=float),
        np.array([0.18, 0.18, 0.018]),
        color,
        collidable=False,
    )


def build_restricted_scene(sim, robot, joints, cube_start, drop_center):
    """Cria uma bancada estreita, paredes e colunas no caminho do movimento."""
    robot_pos = np.array(sim.getObjectPosition(robot, sim.handle_world))
    base_pos = np.array(sim.getObjectPosition(joints[0], sim.handle_world))
    middle = (cube_start + drop_center) / 2.0

    # A plataforma fica logo abaixo do nivel do piso da cena.
    floor_z = min(cube_start[2], drop_center[2]) - 0.05
    create_box(
        sim,
        "Plataforma_Estreita_Franka",
        np.array([robot_pos[0], robot_pos[1], floor_z - 0.025]),
        np.array([0.82, 0.62, 0.05]),
        [0.20, 0.22, 0.26],
    )

    # Guardas perto da base: elas dificultam movimentos largos do ombro,
    # mas ficam fora do caminho principal do efetuador.
    create_box(
        sim,
        "Guarda_Base_Esquerda",
        np.array([base_pos[0], base_pos[1] + 0.42, floor_z + 0.13]),
        np.array([0.72, 0.04, 0.26]),
        [0.22, 0.33, 0.46],
    )
    create_box(
        sim,
        "Guarda_Base_Direita",
        np.array([base_pos[0], base_pos[1] - 0.42, floor_z + 0.13]),
        np.array([0.72, 0.04, 0.26]),
        [0.22, 0.33, 0.46],
    )

    obstacles = [
        {
            "name": "Muro_Central_Restrito",
            "center": middle + np.array([0.00, 0.03, 0.16]),
            "size": np.array([0.17, 0.34, 0.30]),
        },
        {
            "name": "Coluna_Coleta_Restrita",
            "center": cube_start + np.array([-0.12, 0.13, 0.16]),
            "size": np.array([0.10, 0.10, 0.30]),
        },
        {
            "name": "Coluna_Entrega_Restrita",
            "center": drop_center + np.array([0.12, -0.13, 0.16]),
            "size": np.array([0.10, 0.10, 0.30]),
        },
    ]

    for obstacle in obstacles:
        create_box(
            sim,
            obstacle["name"],
            obstacle["center"],
            obstacle["size"],
            [0.85, 0.12, 0.10],
        )

    # Duas colunas formam uma passagem visual estreita no corredor seguro.
    lane_y = min(cube_start[1], drop_center[1]) + LANE_OFFSET_Y
    gate_x = middle[0]
    create_box(
        sim,
        "Portal_Corredor_A",
        np.array([gate_x, lane_y + 0.12, floor_z + 0.18]),
        np.array([0.08, 0.08, 0.36]),
        [0.95, 0.58, 0.08],
    )
    create_box(
        sim,
        "Portal_Corredor_B",
        np.array([gate_x, lane_y - 0.12, floor_z + 0.18]),
        np.array([0.08, 0.08, 0.36]),
        [0.95, 0.58, 0.08],
    )

    create_marker(
        sim,
        "Base_Coleta_Restrita",
        cube_start + np.array([0.0, 0.0, -0.052]),
        [0.1, 0.55, 0.95],
    )
    create_marker(
        sim,
        "Base_Entrega_Restrita",
        drop_center + np.array([0.0, 0.0, -0.052]),
        [0.95, 0.78, 0.10],
    )

    return obstacles


def restricted_safe_z(start, goal, obstacles):
    obstacle_top = max(
        obstacle["center"][2] + obstacle["size"][2] / 2.0
        for obstacle in obstacles
    )
    safe_z = max(obstacle_top + 0.18, start[2] + 0.16, goal[2] + 0.16)
    return min(safe_z, 0.72)


def restricted_lane_y(start, goal):
    return min(start[1], goal[1]) + LANE_OFFSET_Y


def route_through_side_corridor(start, goal, obstacles):
    safe_z = restricted_safe_z(start, goal, obstacles)
    lane_y = restricted_lane_y(start, goal)
    middle_x = (start[0] + goal[0]) / 2.0

    return [
        start,
        np.array([start[0], start[1], safe_z]),
        np.array([start[0], lane_y, safe_z]),
        np.array([middle_x, lane_y, safe_z]),
        np.array([goal[0], lane_y, safe_z]),
        np.array([goal[0], goal[1], safe_z]),
        goal,
    ]


def vertical_route(above, low):
    return densify_cartesian_route([above, low], max_step=VERTICAL_MAX_STEP)


def solve_world_target_restricted(name, target_world, t_base_world, q_seed):
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


def solve_cartesian_route_restricted(route, route_name, t_base_world, q_seed):
    q_route = []
    current_seed = q_seed

    for i, point in enumerate(route[1:], start=1):
        current_seed = solve_world_target_restricted(
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
    drop_center = cube_start + DROP_OFFSET

    # Garante que o cubo esteja no ponto inicial usado pelo roteiro.
    sim.setObjectPosition(cube, sim.handle_world, cube_start.tolist())

    obstacles = build_restricted_scene(sim, robot, joints, cube_start, drop_center)

    pick_above = cube_start + np.array([0.0, 0.0, APPROACH_HEIGHT])
    pick_grasp = cube_start + np.array([0.0, 0.0, GRASP_HEIGHT])
    drop_above = drop_center + np.array([0.0, 0.0, APPROACH_HEIGHT])
    drop_grasp = drop_center + np.array([0.0, 0.0, GRASP_HEIGHT])

    q_current = np.array([sim.getJointPosition(joint) for joint in joints])
    home_pos = fk_world(HOME_Q, t_base_world)
    current_pos = fk_world(q_current, t_base_world)

    approach_sparse = route_through_side_corridor(current_pos, pick_above, obstacles)
    transport_sparse = route_through_side_corridor(pick_above, drop_above, obstacles)
    drop_sparse = [drop_above, drop_grasp]
    return_sparse = route_through_side_corridor(drop_above, home_pos, obstacles)

    approach_route = densify_cartesian_route(approach_sparse, max_step=ROUTE_MAX_STEP)
    transport_route = densify_cartesian_route(transport_sparse, max_step=ROUTE_MAX_STEP)
    drop_route = vertical_route(drop_above, drop_grasp)
    return_route = densify_cartesian_route(return_sparse, max_step=ROUTE_MAX_STEP)

    warn_if_route_crosses_obstacles(approach_sparse, obstacles)
    warn_if_route_crosses_obstacles(transport_sparse, obstacles)
    warn_if_route_crosses_obstacles(drop_sparse, obstacles)
    warn_if_route_crosses_obstacles(return_sparse, obstacles)

    print("\nObstaculos ativos no planejamento:")
    for obstacle in obstacles:
        print(
            f"  {obstacle['name']}: centro={np.round(obstacle['center'], 4)} "
            f"tam={np.round(obstacle['size'], 4)}"
        )

    print_route("Rota de aproximacao pelo corredor", approach_sparse)
    print_route("Rota de transporte pelo corredor", transport_sparse)
    print_route("Descida na entrega", drop_sparse)
    print_route("Rota de retorno pelo corredor", return_sparse)

    planned_sections = [
        ("aproximacao_restrita", approach_route),
        ("transporte_restrito", transport_route),
        ("descida_entrega", drop_route),
        ("retorno_restrito", return_route),
    ]
    metrics = {
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
        save_waypoints_csv(OUTPUT_DIR / "waypoints_cenario_restrito.csv", planned_sections)
        save_route_plot(
            OUTPUT_DIR / "trajetoria_cenario_restrito.png",
            planned_sections,
            obstacles,
            cube_start,
            drop_center,
        )

    print("\nCalculando IK dos waypoints do cenario restrito...")
    q_approach = solve_cartesian_route_restricted(
        approach_route,
        "aproximacao restrita",
        t_base_world,
        q_current,
    )
    q_pick_above = q_approach[-1]
    q_pick_grasp = solve_world_target_restricted(
        "pegar cubo",
        pick_grasp,
        t_base_world,
        q_pick_above,
    )
    q_transport = solve_cartesian_route_restricted(
        transport_route,
        "transporte restrito",
        t_base_world,
        q_pick_above,
    )
    q_drop_descent = solve_cartesian_route_restricted(
        drop_route,
        "descida entrega",
        t_base_world,
        q_transport[-1],
    )
    q_return = solve_cartesian_route_restricted(
        return_route,
        "retorno restrito",
        t_base_world,
        q_transport[-1],
    )

    print("\nIniciando simulacao no cenario restrito...")
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
            "aproximacao restrita",
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
            "transporte restrito",
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
            "retorno restrito",
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
            save_execution_csv(OUTPUT_DIR / "trajetoria_executada_restrita.csv", execution_log)
            save_metrics_csv(OUTPUT_DIR / "metricas_cenario_restrito.csv", metrics)
            print(f"  resultados salvos em   : {OUTPUT_DIR.resolve()}")
    finally:
        time.sleep(ACTION_PAUSE)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()

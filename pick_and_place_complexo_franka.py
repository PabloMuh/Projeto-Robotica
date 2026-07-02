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
    OUTPUT_DIR,
    PORT,
    SAVE_RESULTS,
    TRANSPORT_STEPS_PER_SEGMENT,
    create_or_update_obstacle,
    densify_cartesian_route,
    fk_world,
    get_alias,
    get_franka,
    get_tip,
    min_route_clearance,
    move_to_q,
    plan_route_with_obstacles,
    route_length,
    save_execution_csv,
    save_metrics_csv,
    save_route_plot,
    save_waypoints_csv,
    set_cube_parent,
    solve_cartesian_route,
    solve_world_target,
    warn_if_route_crosses_obstacles,
)


INSPECTION_OFFSET = np.array([-0.22, -0.30, 0.0])
FINAL_OFFSET = np.array([-0.42, 0.28, 0.0])
COMPLEX_CORRIDOR_MARGIN = 0.13
COMPLEX_MAX_SAFE_Z = 0.74
COMPLEX_TOP_CLEARANCE = 0.20
MAX_TRANSIT_IK_ERROR = 0.018


def remove_if_exists(sim, name):
    try:
        handle = sim.getObject(f"/{name}")
        try:
            sim.removeObject(handle)
        except Exception:
            sim.removeObjects([handle])
    except Exception:
        pass


def create_marker(sim, name, center, color):
    """Cria uma base visual fina para indicar estacao/destino."""
    remove_if_exists(sim, name)
    size = np.array([0.18, 0.18, 0.02])
    try:
        handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size.tolist(), 0)
        sim.setObjectAlias(handle, name)
        sim.setObjectPosition(handle, sim.handle_world, center.tolist())
        try:
            sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)
        except Exception:
            pass
        return handle
    except Exception as exc:
        print(f"Nao consegui criar marcador {name}: {exc}")
        return None


def complex_safe_corridor_route(start, goal, obstacles):
    """Corredor seguro menos agressivo para nao gerar pontos fora do alcance."""
    safe_y = min(
        obstacle["center"][1] - obstacle["size"][1] / 2.0 - COMPLEX_CORRIDOR_MARGIN
        for obstacle in obstacles
    )
    safe_z = max(
        obstacle["center"][2] + obstacle["size"][2] / 2.0 + COMPLEX_TOP_CLEARANCE
        for obstacle in obstacles
    )
    safe_z = max(safe_z, start[2] + 0.08, goal[2] + 0.16)
    safe_z = min(safe_z, COMPLEX_MAX_SAFE_Z)

    return [
        start,
        np.array([start[0], start[1], safe_z]),
        np.array([start[0], safe_y, safe_z]),
        np.array([goal[0], safe_y, safe_z]),
        np.array([goal[0], goal[1], safe_z]),
        goal,
    ]


def build_complex_scene(sim, cube_start, inspection_center, final_center):
    """Cria obstaculos e marcadores para um pick-and-place em duas etapas."""
    middle_1 = (cube_start + inspection_center) / 2.0
    middle_2 = (inspection_center + final_center) / 2.0

    obstacles = [
        {
            "name": "Obstaculo_A",
            "center": middle_1 + np.array([0.00, 0.00, 0.13]),
            "size": np.array([0.11, 0.12, 0.20]),
        },
        {
            "name": "Obstaculo_B",
            "center": middle_2 + np.array([0.02, 0.04, 0.13]),
            "size": np.array([0.13, 0.12, 0.22]),
        },
    ]

    for obstacle in obstacles:
        create_or_update_obstacle(
            sim,
            obstacle["name"],
            obstacle["center"],
            obstacle["size"],
        )

    create_marker(
        sim,
        "Estacao_Inspecao",
        inspection_center + np.array([0.0, 0.0, -0.04]),
        [0.1, 0.35, 0.95],
    )
    create_marker(
        sim,
        "Destino_Final",
        final_center + np.array([0.0, 0.0, -0.04]),
        [0.95, 0.75, 0.1],
    )

    return obstacles


def print_route(title, route):
    print(f"\n{title}:")
    for i, point in enumerate(route):
        print(f"  p{i}: {np.round(point, 4)}")
    print(f"  rota densificada: {len(densify_cartesian_route(route))} pontos cartesianos")


def solve_world_target_complex(name, target_world, T_base_world, q_seed, max_error=MAX_TRANSIT_IK_ERROR):
    from inversa import mundo_para_base, resolver_ik

    target_base = mundo_para_base(target_world, T_base_world)
    q_sol, _, error_base = resolver_ik(target_base, q_inicial=q_seed)
    pos_world = fk_world(q_sol, T_base_world)
    error_world = float(np.linalg.norm(pos_world - target_world))

    print(f"{name}")
    print(f"  alvo mundo : {np.round(target_world, 4)}")
    print(f"  erro modelo: {error_world:.6f} m")

    if error_world > max_error:
        raise RuntimeError(
            f"IK falhou no waypoint '{name}'. "
            f"Erro base={error_base:.6f} m, erro mundo={error_world:.6f} m."
        )

    return q_sol


def solve_cartesian_route_complex(route, route_name, T_base_world, q_seed):
    q_route = []
    current_seed = q_seed

    for i, point in enumerate(route[1:], start=1):
        current_seed = solve_world_target_complex(
            f"{route_name} {i}",
            point,
            T_base_world,
            current_seed,
        )
        q_route.append(current_seed)

    return q_route


def execute_route(sim, joints, tip, q_current, q_route, label, execution_log):
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

    T_base_world = matriz_coppelia_para_homogenea(
        sim.getObjectMatrix(joints[0], sim.handle_world)
    )

    cube_start = np.array(sim.getObjectPosition(cube, sim.handle_world))
    inspection_center = cube_start + INSPECTION_OFFSET
    final_center = cube_start + FINAL_OFFSET

    pick_above = cube_start + np.array([0.0, 0.0, APPROACH_HEIGHT])
    pick_grasp = cube_start + np.array([0.0, 0.0, GRASP_HEIGHT])

    inspection_above = inspection_center + np.array([0.0, 0.0, APPROACH_HEIGHT])
    inspection_grasp = inspection_center + np.array([0.0, 0.0, GRASP_HEIGHT])
    inspection_view_1 = inspection_center + np.array([0.14, -0.12, 0.42])
    inspection_view_2 = inspection_center + np.array([-0.12, -0.10, 0.40])

    final_above = final_center + np.array([0.0, 0.0, APPROACH_HEIGHT])
    final_grasp = final_center + np.array([0.0, 0.0, GRASP_HEIGHT])

    obstacles = build_complex_scene(sim, cube_start, inspection_center, final_center)

    q_current = np.array([sim.getJointPosition(joint) for joint in joints])
    current_pos = fk_world(q_current, T_base_world)
    home_pos = fk_world(HOME_Q, T_base_world)

    approach_route = densify_cartesian_route(
        complex_safe_corridor_route(current_pos, pick_above, obstacles)
    )
    pick_to_inspection_route = densify_cartesian_route(
        plan_route_with_obstacles(pick_above, inspection_above, obstacles)
    )
    inspection_drop_route = densify_cartesian_route(
        [inspection_above, inspection_grasp],
        max_step=0.025,
    )
    inspection_scan_route = densify_cartesian_route(
        [inspection_above, inspection_view_1, inspection_view_2, inspection_above]
    )
    inspection_to_final_route = densify_cartesian_route(
        plan_route_with_obstacles(inspection_above, final_above, obstacles)
    )
    final_drop_route = densify_cartesian_route(
        [final_above, final_grasp],
        max_step=0.025,
    )
    return_route = densify_cartesian_route(
        complex_safe_corridor_route(final_above, home_pos, obstacles)
    )

    sparse_routes = [
        ("aproximacao", complex_safe_corridor_route(current_pos, pick_above, obstacles)),
        ("coleta_para_inspecao", plan_route_with_obstacles(pick_above, inspection_above, obstacles)),
        ("descida_inspecao", [inspection_above, inspection_grasp]),
        ("varredura_inspecao", [inspection_above, inspection_view_1, inspection_view_2, inspection_above]),
        ("inspecao_para_destino", plan_route_with_obstacles(inspection_above, final_above, obstacles)),
        ("descida_final", [final_above, final_grasp]),
        ("retorno", complex_safe_corridor_route(final_above, home_pos, obstacles)),
    ]
    for _, route in sparse_routes:
        warn_if_route_crosses_obstacles(route, obstacles)

    print("\nCenario complexo:")
    print(f"  cubo inicial     : {np.round(cube_start, 4)}")
    print(f"  estacao inspecao : {np.round(inspection_center, 4)}")
    print(f"  destino final    : {np.round(final_center, 4)}")
    for obstacle in obstacles:
        print(
            f"  {obstacle['name']}: centro={np.round(obstacle['center'], 4)} "
            f"tam={np.round(obstacle['size'], 4)}"
        )

    for title, route in sparse_routes:
        print_route(title, route)

    planned_sections = [
        ("aproximacao", approach_route),
        ("coleta_para_inspecao", pick_to_inspection_route),
        ("descida_inspecao", inspection_drop_route),
        ("varredura_inspecao", inspection_scan_route),
        ("inspecao_para_destino", inspection_to_final_route),
        ("descida_final", final_drop_route),
        ("retorno", return_route),
    ]

    metrics = {
        "approach_waypoints": len(approach_route),
        "pick_to_inspection_waypoints": len(pick_to_inspection_route),
        "inspection_scan_waypoints": len(inspection_scan_route),
        "inspection_to_final_waypoints": len(inspection_to_final_route),
        "return_waypoints": len(return_route),
        "total_planned_waypoints": sum(len(route) for _, route in planned_sections),
        "total_planned_length_m": sum(route_length(route) for _, route in planned_sections),
        "min_clearance_planned_m": min(
            min_route_clearance(route, obstacles) for _, route in planned_sections
        ),
    }

    if SAVE_RESULTS:
        complex_dir = OUTPUT_DIR / "complexo"
        complex_dir.mkdir(parents=True, exist_ok=True)
        save_waypoints_csv(complex_dir / "waypoints_complexos.csv", planned_sections)
        save_route_plot(
            complex_dir / "trajetoria_complexa.png",
            planned_sections,
            obstacles,
            cube_start,
            final_center,
        )
    else:
        complex_dir = Path(".")

    print("\nCalculando IK dos waypoints do processo complexo...")
    q_approach = solve_cartesian_route_complex(approach_route, "aproximacao", T_base_world, q_current)
    q_pick_above = q_approach[-1]
    q_pick_grasp = solve_world_target("pegar cubo inicial", pick_grasp, T_base_world, q_pick_above)

    q_pick_to_inspection = solve_cartesian_route_complex(
        pick_to_inspection_route,
        "coleta para inspecao",
        T_base_world,
        q_pick_above,
    )
    q_inspection_drop = solve_cartesian_route(
        inspection_drop_route,
        "descida inspecao",
        T_base_world,
        q_pick_to_inspection[-1],
    )
    q_inspection_scan = solve_cartesian_route_complex(
        inspection_scan_route,
        "varredura inspecao",
        T_base_world,
        q_pick_to_inspection[-1],
    )
    q_inspection_repick = solve_cartesian_route(
        inspection_drop_route,
        "recoleta inspecao",
        T_base_world,
        q_inspection_scan[-1],
    )
    q_inspection_to_final = solve_cartesian_route_complex(
        inspection_to_final_route,
        "inspecao para destino",
        T_base_world,
        q_pick_to_inspection[-1],
    )
    q_final_drop = solve_cartesian_route(
        final_drop_route,
        "descida final",
        T_base_world,
        q_inspection_to_final[-1],
    )
    q_return = solve_cartesian_route_complex(return_route, "retorno", T_base_world, q_inspection_to_final[-1])

    print("\nIniciando processo complexo de pick-and-place...")
    sim.startSimulation()
    execution_log = []
    t0 = time.perf_counter()
    time.sleep(ACTION_PAUSE)

    try:
        q_current = execute_route(sim, joints, tip, q_current, q_approach, "aproximacao", execution_log)
        q_current = move_to_q(
            sim,
            joints,
            q_current,
            q_pick_grasp,
            "pegar cubo inicial",
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
            "subir com cubo inicial",
            logger=execution_log,
            tip=tip,
        )
        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            q_pick_to_inspection,
            "levar para inspecao",
            execution_log,
        )
        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            q_inspection_drop,
            "descer na inspecao",
            execution_log,
        )

        print("Soltando cubo na estacao de inspecao...")
        set_cube_parent(sim, cube, -1)
        time.sleep(ACTION_PAUSE)

        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            list(reversed(q_inspection_drop[:-1])),
            "subir da inspecao",
            execution_log,
        )
        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            q_inspection_scan,
            "varredura de inspecao",
            execution_log,
        )
        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            q_inspection_repick,
            "descer para recoleta",
            execution_log,
        )

        print("Prendendo cubo novamente...")
        set_cube_parent(sim, cube, tip)
        time.sleep(ACTION_PAUSE)

        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            list(reversed(q_inspection_repick[:-1])) + [q_pick_to_inspection[-1]],
            "subir com cubo inspecionado",
            execution_log,
        )
        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            q_inspection_to_final,
            "levar para destino final",
            execution_log,
        )
        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            q_final_drop,
            "descer no destino final",
            execution_log,
        )

        print("Soltando cubo no destino final...")
        set_cube_parent(sim, cube, -1)
        time.sleep(ACTION_PAUSE)

        q_current = execute_route(
            sim,
            joints,
            tip,
            q_current,
            list(reversed(q_final_drop[:-1])) + [q_inspection_to_final[-1]],
            "subir do destino final",
            execution_log,
        )
        q_current = execute_route(sim, joints, tip, q_current, q_return, "retorno", execution_log)

        final_cube = np.array(sim.getObjectPosition(cube, sim.handle_world))
        inspection_error = float(np.linalg.norm(final_cube - final_center))
        metrics["execution_time_s"] = time.perf_counter() - t0
        metrics["execution_samples"] = len(execution_log)
        metrics["final_delivery_error_m"] = inspection_error

        print("\nResultado complexo:")
        print(f"  posicao inicial do cubo: {np.round(cube_start, 4)}")
        print(f"  estacao intermediaria  : {np.round(inspection_center, 4)}")
        print(f"  destino final planejado: {np.round(final_center, 4)}")
        print(f"  posicao final do cubo  : {np.round(final_cube, 4)}")
        print(f"  erro final             : {inspection_error:.6f} m")

        if SAVE_RESULTS:
            save_execution_csv(complex_dir / "trajetoria_executada_complexa.csv", execution_log)
            save_metrics_csv(complex_dir / "metricas_complexas.csv", metrics)
            print(f"  resultados salvos em   : {complex_dir.resolve()}")
    finally:
        time.sleep(ACTION_PAUSE)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()

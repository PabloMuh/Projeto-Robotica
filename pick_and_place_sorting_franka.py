import time
from pathlib import Path

import numpy as np

from cinematica_franka import matriz_coppelia_para_homogenea
from pick_and_place_obstaculos_franka import (
    ACTION_PAUSE,
    APPROACH_HEIGHT,
    GRASP_HEIGHT,
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
    solve_world_target,
    warn_if_route_crosses_obstacles,
)
from pick_and_place_complexo_franka import (
    HOME_Q,
    complex_safe_corridor_route,
    create_marker,
    execute_route,
    solve_cartesian_route_complex,
)


PIECE_SIZE = np.array([0.045, 0.045, 0.045])
INSPECTION_OFFSET = np.array([-0.22, -0.30, 0.0])

PIECES = [
    {
        "name": "Peca_Vermelha",
        "color": [0.95, 0.1, 0.1],
        "supply_offset": np.array([0.00, 0.00, 0.0]),
        "destination_offset": np.array([-0.42, 0.28, 0.0]),
    },
    {
        "name": "Peca_Verde",
        "color": [0.1, 0.75, 0.25],
        "supply_offset": np.array([-0.10, -0.14, 0.0]),
        "destination_offset": np.array([-0.50, 0.02, 0.0]),
    },
    {
        "name": "Peca_Azul",
        "color": [0.1, 0.3, 0.95],
        "supply_offset": np.array([-0.16, 0.13, 0.0]),
        "destination_offset": np.array([-0.34, -0.24, 0.0]),
    },
]


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


def create_piece(sim, name, center, color):
    """Cria uma peca pequena para a tarefa de classificacao."""
    remove_if_exists(sim, name)
    handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, PIECE_SIZE.tolist(), 0)
    sim.setObjectAlias(handle, name)
    sim.setObjectPosition(handle, sim.handle_world, center.tolist())
    color_shape(sim, handle, color)
    return handle


def build_sorting_scene(sim, cube_reference):
    """Cria pecas, bases de destino, estacao de inspecao e obstaculos."""
    supply_origin = cube_reference.copy()
    inspection_center = supply_origin + INSPECTION_OFFSET

    pieces = []
    for item in PIECES:
        center = supply_origin + item["supply_offset"]
        handle = create_piece(sim, item["name"], center, item["color"])
        destination = supply_origin + item["destination_offset"]
        create_marker(
            sim,
            f"Destino_{item['name']}",
            destination + np.array([0.0, 0.0, -0.04]),
            item["color"],
        )
        pieces.append({
            **item,
            "handle": handle,
            "start": center,
            "destination": destination,
        })

    create_marker(
        sim,
        "Estacao_Inspecao_Sorting",
        inspection_center + np.array([0.0, 0.0, -0.04]),
        [0.1, 0.35, 0.95],
    )

    obstacles = [
        {
            "name": "Pilar_Sorting_A",
            "center": supply_origin + np.array([-0.18, -0.02, 0.17]),
            "size": np.array([0.11, 0.12, 0.24]),
        },
        {
            "name": "Pilar_Sorting_B",
            "center": supply_origin + np.array([-0.36, 0.12, 0.16]),
            "size": np.array([0.12, 0.12, 0.22]),
        },
    ]

    for obstacle in obstacles:
        create_or_update_obstacle(
            sim,
            obstacle["name"],
            obstacle["center"],
            obstacle["size"],
        )

    return pieces, inspection_center, obstacles


def vertical_route(above, low):
    return densify_cartesian_route([above, low], max_step=0.025)


def plan_leg(start, goal, obstacles, use_safe_corridor=False):
    if use_safe_corridor:
        sparse = complex_safe_corridor_route(start, goal, obstacles)
    else:
        sparse = plan_route_with_obstacles(start, goal, obstacles)
    dense = densify_cartesian_route(sparse)
    warn_if_route_crosses_obstacles(sparse, obstacles)
    return sparse, dense


def solve_and_execute_route(sim, joints, tip, q_current, route, route_name, t_base_world, log):
    q_route = solve_cartesian_route_complex(route, route_name, t_base_world, q_current)
    return execute_route(sim, joints, tip, q_current, q_route, route_name, log), q_route


def process_piece(
    sim,
    joints,
    tip,
    piece,
    inspection_center,
    obstacles,
    t_base_world,
    q_current,
    execution_log,
    planned_sections,
):
    print(f"\n=== Processando {piece['name']} ===")

    pick_above = piece["start"] + np.array([0.0, 0.0, APPROACH_HEIGHT])
    pick_grasp = piece["start"] + np.array([0.0, 0.0, GRASP_HEIGHT])
    inspection_above = inspection_center + np.array([0.0, 0.0, APPROACH_HEIGHT])
    inspection_grasp = inspection_center + np.array([0.0, 0.0, GRASP_HEIGHT])
    inspect_view = inspection_center + np.array([0.12, -0.12, 0.40])
    dest_above = piece["destination"] + np.array([0.0, 0.0, APPROACH_HEIGHT])
    dest_grasp = piece["destination"] + np.array([0.0, 0.0, GRASP_HEIGHT])

    current_pos = fk_world(q_current, t_base_world)

    approach_sparse, approach_route = plan_leg(
        current_pos,
        pick_above,
        obstacles,
        use_safe_corridor=True,
    )
    to_inspection_sparse, to_inspection_route = plan_leg(
        pick_above,
        inspection_above,
        obstacles,
    )
    inspection_drop_route = vertical_route(inspection_above, inspection_grasp)
    inspection_scan_route = densify_cartesian_route(
        [inspection_above, inspect_view, inspection_above]
    )
    inspection_pick_route = vertical_route(inspection_above, inspection_grasp)
    to_destination_sparse, to_destination_route = plan_leg(
        inspection_above,
        dest_above,
        obstacles,
    )
    destination_drop_route = vertical_route(dest_above, dest_grasp)

    planned_sections.extend([
        (f"{piece['name']}_aproximacao", approach_route),
        (f"{piece['name']}_ate_inspecao", to_inspection_route),
        (f"{piece['name']}_descida_inspecao", inspection_drop_route),
        (f"{piece['name']}_scan", inspection_scan_route),
        (f"{piece['name']}_ate_destino", to_destination_route),
        (f"{piece['name']}_descida_destino", destination_drop_route),
    ])

    q_current, _ = solve_and_execute_route(
        sim, joints, tip, q_current, approach_route, f"{piece['name']} aproximacao", t_base_world, execution_log
    )
    q_pick_grasp = solve_world_target(
        f"{piece['name']} pegar",
        pick_grasp,
        t_base_world,
        q_current,
    )
    q_current = move_to_q(
        sim,
        joints,
        q_current,
        q_pick_grasp,
        f"{piece['name']} descer para pegar",
        steps=TRANSPORT_STEPS_PER_SEGMENT,
        logger=execution_log,
        tip=tip,
    )

    print(f"Prendendo {piece['name']} ao efetuador...")
    set_cube_parent(sim, piece["handle"], tip)
    time.sleep(ACTION_PAUSE)

    q_pick_above = solve_world_target(
        f"{piece['name']} subir apos pegar",
        pick_above,
        t_base_world,
        q_current,
    )
    q_current = move_to_q(
        sim,
        joints,
        q_current,
        q_pick_above,
        f"{piece['name']} subir apos pegar",
        steps=TRANSPORT_STEPS_PER_SEGMENT,
        logger=execution_log,
        tip=tip,
    )

    q_current, _ = solve_and_execute_route(
        sim, joints, tip, q_current, to_inspection_route, f"{piece['name']} ate inspecao", t_base_world, execution_log
    )
    q_current, q_inspection_drop = solve_and_execute_route(
        sim, joints, tip, q_current, inspection_drop_route, f"{piece['name']} descer inspecao", t_base_world, execution_log
    )

    print(f"Soltando {piece['name']} na estacao de inspecao...")
    set_cube_parent(sim, piece["handle"], -1)
    time.sleep(ACTION_PAUSE)

    q_current = execute_route(
        sim,
        joints,
        tip,
        q_current,
        list(reversed(q_inspection_drop[:-1])),
        f"{piece['name']} subir inspecao",
        execution_log,
    )
    q_current, _ = solve_and_execute_route(
        sim, joints, tip, q_current, inspection_scan_route, f"{piece['name']} scan inspecao", t_base_world, execution_log
    )
    q_current, q_inspection_pick = solve_and_execute_route(
        sim, joints, tip, q_current, inspection_pick_route, f"{piece['name']} recoletar", t_base_world, execution_log
    )

    print(f"Prendendo {piece['name']} novamente...")
    set_cube_parent(sim, piece["handle"], tip)
    time.sleep(ACTION_PAUSE)

    q_current = execute_route(
        sim,
        joints,
        tip,
        q_current,
        list(reversed(q_inspection_pick[:-1])),
        f"{piece['name']} subir com peca inspecionada",
        execution_log,
    )

    q_current, _ = solve_and_execute_route(
        sim, joints, tip, q_current, to_destination_route, f"{piece['name']} ate destino", t_base_world, execution_log
    )
    q_current, q_destination_drop = solve_and_execute_route(
        sim, joints, tip, q_current, destination_drop_route, f"{piece['name']} descer destino", t_base_world, execution_log
    )

    print(f"Soltando {piece['name']} no destino correto...")
    set_cube_parent(sim, piece["handle"], -1)
    time.sleep(ACTION_PAUSE)

    q_current = execute_route(
        sim,
        joints,
        tip,
        q_current,
        list(reversed(q_destination_drop[:-1])),
        f"{piece['name']} subir destino",
        execution_log,
    )

    final_pos = np.array(sim.getObjectPosition(piece["handle"], sim.handle_world))
    error = float(np.linalg.norm(final_pos - piece["destination"]))
    print(f"{piece['name']} erro de entrega: {error:.6f} m")
    return q_current, error


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient(host=HOST, port=PORT)
    sim = client.require("sim")
    print("Conectado ao CoppeliaSim.")

    robot, joints = get_franka(sim)
    tip = get_tip(sim, robot)

    t_base_world = matriz_coppelia_para_homogenea(
        sim.getObjectMatrix(joints[0], sim.handle_world)
    )

    try:
        cube_reference = np.array(sim.getObjectPosition(sim.getObject("/Cuboid"), sim.handle_world))
    except Exception:
        cube_reference = np.array([1.17, -0.025, 0.05])

    pieces, inspection_center, obstacles = build_sorting_scene(sim, cube_reference)

    print("\nCenario de classificacao:")
    print(f"  referencia abastecimento: {np.round(cube_reference, 4)}")
    print(f"  estacao de inspecao     : {np.round(inspection_center, 4)}")
    for piece in pieces:
        print(
            f"  {piece['name']}: inicio={np.round(piece['start'], 4)} "
            f"destino={np.round(piece['destination'], 4)}"
        )
    for obstacle in obstacles:
        print(
            f"  {obstacle['name']}: centro={np.round(obstacle['center'], 4)} "
            f"tam={np.round(obstacle['size'], 4)}"
        )

    q_current = np.array([sim.getJointPosition(joint) for joint in joints])
    execution_log = []
    planned_sections = []
    delivery_errors = {}

    print("\nIniciando classificacao multipeca...")
    sim.startSimulation()
    t0 = time.perf_counter()
    time.sleep(ACTION_PAUSE)

    try:
        for piece in pieces:
            q_current, error = process_piece(
                sim,
                joints,
                tip,
                piece,
                inspection_center,
                obstacles,
                t_base_world,
                q_current,
                execution_log,
                planned_sections,
            )
            delivery_errors[piece["name"]] = error

        home_route = densify_cartesian_route(
            complex_safe_corridor_route(
                fk_world(q_current, t_base_world),
                fk_world(HOME_Q, t_base_world),
                obstacles,
            )
        )
        planned_sections.append(("retorno_home", home_route))
        q_current, _ = solve_and_execute_route(
            sim,
            joints,
            tip,
            q_current,
            home_route,
            "retorno home",
            t_base_world,
            execution_log,
        )

        metrics = {
            "pieces_processed": len(pieces),
            "execution_time_s": time.perf_counter() - t0,
            "execution_samples": len(execution_log),
            "total_planned_waypoints": sum(len(route) for _, route in planned_sections),
            "total_planned_length_m": sum(route_length(route) for _, route in planned_sections),
            "min_clearance_planned_m": min(
                min_route_clearance(route, obstacles) for _, route in planned_sections
            ),
        }
        for name, error in delivery_errors.items():
            metrics[f"{name}_delivery_error_m"] = error

        print("\nResultado da classificacao:")
        for name, error in delivery_errors.items():
            print(f"  {name}: erro de entrega {error:.6f} m")
        print(f"  tempo total: {metrics['execution_time_s']:.2f} s")

        if SAVE_RESULTS:
            sort_dir = OUTPUT_DIR / "sorting"
            sort_dir.mkdir(parents=True, exist_ok=True)
            save_waypoints_csv(sort_dir / "waypoints_sorting.csv", planned_sections)
            save_route_plot(
                sort_dir / "trajetoria_sorting.png",
                planned_sections,
                obstacles,
                cube_reference,
                pieces[-1]["destination"],
            )
            save_execution_csv(sort_dir / "trajetoria_executada_sorting.csv", execution_log)
            save_metrics_csv(sort_dir / "metricas_sorting.csv", metrics)
            print(f"  resultados salvos em: {sort_dir.resolve()}")
    finally:
        time.sleep(ACTION_PAUSE)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()

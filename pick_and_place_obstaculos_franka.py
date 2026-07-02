import csv
import time
from pathlib import Path

import numpy as np

from cinematica_franka import calcular_cinematica_direta, matriz_coppelia_para_homogenea


HOST = "127.0.0.1"
PORT = 23000
OUTPUT_DIR = Path("resultados")
SAVE_RESULTS = True

DIRECT_MODE = False
FAST_MODE = True

if FAST_MODE:
    STEPS_PER_SEGMENT = 45
    TRANSPORT_STEPS_PER_SEGMENT = 8
    DT = 0.008
    CARTESIAN_WAYPOINT_STEP = 0.08
    ACTION_PAUSE = 0.12
else:
    STEPS_PER_SEGMENT = 100
    TRANSPORT_STEPS_PER_SEGMENT = 24
    DT = 0.02
    CARTESIAN_WAYPOINT_STEP = 0.045
    ACTION_PAUSE = 0.4

APPROACH_HEIGHT = 0.28
GRASP_HEIGHT = 0.10
DROP_OFFSET = np.array([-0.35, 0.30, 0.0])
HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.0])

OBSTACLE_MARGIN = 0.20
OBSTACLE_TOP_CLEARANCE = 0.24
DETOUR_SIDE = -1.0


def get_alias(sim, obj):
    try:
        return sim.getObjectAlias(obj, 1)
    except Exception:
        try:
            return sim.getObjectAlias(obj)
        except Exception:
            return str(obj)


def get_franka(sim):
    robot = sim.getObject("/Franka")
    joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)[:7]
    if len(joints) < 7:
        raise RuntimeError("Nao encontrei as 7 juntas principais do Franka.")
    return robot, joints


def get_tip(sim, robot):
    try:
        return sim.getObject("/Franka/connection")
    except Exception:
        pass

    objects = sim.getObjectsInTree(robot, sim.handle_all, 0)
    candidates = [obj for obj in objects if sim.getObjectType(obj) != sim.object_joint_type]

    def depth(obj):
        value = 0
        current = obj
        while True:
            parent = sim.getObjectParent(current)
            if parent == -1:
                break
            value += 1
            current = parent
        return value

    candidates.sort(key=depth, reverse=True)
    return candidates[0]


def set_joint(sim, joint, angle, direct_mode=False):
    if direct_mode:
        sim.setJointPosition(joint, float(angle))
    else:
        sim.setJointTargetPosition(joint, float(angle))


def smoothstep(alpha):
    return alpha * alpha * (3.0 - 2.0 * alpha)


def interpolate(q_start, q_end, steps):
    for i in range(steps):
        alpha = smoothstep(i / (steps - 1))
        yield (1.0 - alpha) * q_start + alpha * q_end


def log_execution_sample(log, sim, tip, q, segment_name):
    if log is None or tip is None:
        return

    pos = np.array(sim.getObjectPosition(tip, sim.handle_world))
    log.append({
        "t": time.perf_counter(),
        "segment": segment_name,
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "q1": q[0],
        "q2": q[1],
        "q3": q[2],
        "q4": q[3],
        "q5": q[4],
        "q6": q[5],
        "q7": q[6],
    })


def move_to_q(
    sim,
    joints,
    q_start,
    q_end,
    name,
    steps=STEPS_PER_SEGMENT,
    logger=None,
    tip=None,
):
    print(f"Executando trecho: {name}")
    for q in interpolate(q_start, q_end, steps):
        for joint, angle in zip(joints, q):
            set_joint(sim, joint, angle, direct_mode=DIRECT_MODE)
        log_execution_sample(logger, sim, tip, q, name)
        time.sleep(DT)
    return q_end.copy()


def fk_world(q, T_base_world):
    T_base, _, _ = calcular_cinematica_direta(q)
    T_world = T_base_world @ T_base
    return T_world[0:3, 3]


def solve_world_target(name, target_world, T_base_world, q_seed):
    from inversa import mundo_para_base, resolver_ik

    target_base = mundo_para_base(target_world, T_base_world)
    q_sol, success, error_base = resolver_ik(target_base, q_inicial=q_seed)
    pos_world = fk_world(q_sol, T_base_world)
    error_world = float(np.linalg.norm(pos_world - target_world))

    print(f"{name}")
    print(f"  alvo mundo : {np.round(target_world, 4)}")
    print(f"  erro modelo: {error_world:.6f} m")

    if not success or error_world > 0.007:
        raise RuntimeError(
            f"IK falhou no waypoint '{name}'. "
            f"Erro base={error_base:.6f} m, erro mundo={error_world:.6f} m."
        )

    return q_sol


def set_cube_parent(sim, cube, parent):
    sim.setObjectParent(cube, parent, True)


def segment_intersects_aabb(p0, p1, center, half_size):
    """Teste linha-caixa por slab method em 3D."""
    direction = p1 - p0
    t_min = 0.0
    t_max = 1.0

    box_min = center - half_size
    box_max = center + half_size

    for axis in range(3):
        if abs(direction[axis]) < 1e-9:
            if p0[axis] < box_min[axis] or p0[axis] > box_max[axis]:
                return False
        else:
            inv_d = 1.0 / direction[axis]
            t1 = (box_min[axis] - p0[axis]) * inv_d
            t2 = (box_max[axis] - p0[axis]) * inv_d
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)
            t_min = max(t_min, t_enter)
            t_max = min(t_max, t_exit)
            if t_min > t_max:
                return False

    return True


def detour_around_obstacle(p0, p1, obstacle):
    center = obstacle["center"]
    half = obstacle["size"] / 2.0

    side_y = center[1] + DETOUR_SIDE * (half[1] + OBSTACLE_MARGIN)
    safe_z = center[2] + half[2] + OBSTACLE_TOP_CLEARANCE

    return [
        np.array([p0[0], p0[1], safe_z]),
        np.array([p0[0], side_y, safe_z]),
        np.array([center[0], side_y, safe_z]),
        np.array([p1[0], side_y, safe_z]),
        np.array([p1[0], p1[1], safe_z]),
    ]


def plan_route_with_obstacles(start, goal, obstacles):
    route = [start]
    current = start

    for obstacle in obstacles:
        inflated_half = obstacle["size"] / 2.0 + OBSTACLE_MARGIN
        if segment_intersects_aabb(current, goal, obstacle["center"], inflated_half):
            print(f"Rota direta cruza {obstacle['name']}; adicionando desvio.")
            for point in detour_around_obstacle(current, goal, obstacle):
                route.append(point)
            current = route[-1]

    route.append(goal)
    return route


def safe_corridor_route(start, goal, obstacles):
    """Rota conservadora para entrada/saida: sobe, vai para o lado livre e cruza alto."""
    safe_y = min(
        obstacle["center"][1] - obstacle["size"][1] / 2.0 - OBSTACLE_MARGIN
        for obstacle in obstacles
    )
    safe_z = max(
        obstacle["center"][2] + obstacle["size"][2] / 2.0 + OBSTACLE_TOP_CLEARANCE
        for obstacle in obstacles
    )
    safe_z = max(safe_z, start[2] + 0.20, goal[2] + 0.20)

    return [
        start,
        np.array([start[0], start[1], safe_z]),
        np.array([start[0], safe_y, safe_z]),
        np.array([goal[0], safe_y, safe_z]),
        np.array([goal[0], goal[1], safe_z]),
        goal,
    ]


def densify_cartesian_route(route, max_step=CARTESIAN_WAYPOINT_STEP):
    dense = [route[0]]

    for start, goal in zip(route[:-1], route[1:]):
        distance = float(np.linalg.norm(goal - start))
        steps = max(1, int(np.ceil(distance / max_step)))

        for i in range(1, steps + 1):
            alpha = i / steps
            dense.append((1.0 - alpha) * start + alpha * goal)

    return dense


def solve_cartesian_route(route, route_name, T_base_world, q_seed):
    q_route = []
    current_seed = q_seed

    for i, point in enumerate(route[1:], start=1):
        current_seed = solve_world_target(
            f"{route_name} {i}",
            point,
            T_base_world,
            current_seed,
        )
        q_route.append(current_seed)

    return q_route


def warn_if_route_crosses_obstacles(route, obstacles):
    for start, goal in zip(route[:-1], route[1:]):
        for obstacle in obstacles:
            inflated_half = obstacle["size"] / 2.0 + 0.03
            if segment_intersects_aabb(start, goal, obstacle["center"], inflated_half):
                print(
                    f"AVISO: trecho {np.round(start, 3)} -> {np.round(goal, 3)} "
                    f"ainda passa perto de {obstacle['name']}."
                )


def point_aabb_clearance(point, obstacle):
    half = obstacle["size"] / 2.0
    delta = np.abs(point - obstacle["center"]) - half
    outside = np.maximum(delta, 0.0)

    if np.any(delta > 0.0):
        return float(np.linalg.norm(outside))

    return -float(np.min(-delta))


def route_length(route):
    return float(
        sum(np.linalg.norm(goal - start) for start, goal in zip(route[:-1], route[1:]))
    )


def min_route_clearance(route, obstacles):
    if not obstacles:
        return float("inf")

    dense = densify_cartesian_route(route, max_step=0.01)
    best = float("inf")
    for point in dense:
        for obstacle in obstacles:
            best = min(best, point_aabb_clearance(point, obstacle))
    return best


def save_waypoints_csv(path, sections):
    rows = []
    for section_name, route in sections:
        for index, point in enumerate(route):
            rows.append({
                "section": section_name,
                "index": index,
                "x": point[0],
                "y": point[1],
                "z": point[2],
            })

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["section", "index", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)


def save_execution_csv(path, log):
    if not log:
        return

    t0 = log[0]["t"]
    rows = []
    for row in log:
        out = dict(row)
        out["t"] = row["t"] - t0
        rows.append(out)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_csv(path, metrics):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def draw_obstacle(ax, obstacle):
    center = obstacle["center"]
    sx, sy, sz = obstacle["size"] / 2.0
    x = [center[0] - sx, center[0] + sx]
    y = [center[1] - sy, center[1] + sy]
    z = [center[2] - sz, center[2] + sz]

    vertices = np.array([
        [x[0], y[0], z[0]], [x[1], y[0], z[0]],
        [x[1], y[1], z[0]], [x[0], y[1], z[0]],
        [x[0], y[0], z[1]], [x[1], y[0], z[1]],
        [x[1], y[1], z[1]], [x[0], y[1], z[1]],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for start, end in edges:
        pts = vertices[[start, end]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="red", linewidth=2)

    ax.text(center[0], center[1], center[2] + sz + 0.03, obstacle["name"], color="red")


def save_route_plot(path, sections, obstacles, cube_start, drop_center):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Nao foi possivel gerar figura da rota: {exc}")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "aproximacao": "tab:blue",
        "transporte": "tab:green",
        "descida_entrega": "tab:orange",
        "retorno": "tab:purple",
    }

    for section_name, route in sections:
        pts = np.array(route)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            "-o",
            markersize=3,
            linewidth=2,
            label=section_name,
            color=colors.get(section_name, None),
        )

    for obstacle in obstacles:
        draw_obstacle(ax, obstacle)

    ax.scatter(*cube_start, color="black", s=45, label="cubo inicial")
    ax.scatter(*drop_center, color="gold", s=45, label="destino")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Pick-and-place com desvio de obstaculo")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=-55)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def enable_collision_for_obstacle(sim, handle):
    """Ativa propriedades de colisao/dinamica do obstaculo quando disponiveis."""
    try:
        sim.setObjectSpecialProperty(
            handle,
            sim.objectspecialproperty_collidable
            | sim.objectspecialproperty_measurable
            | sim.objectspecialproperty_detectable_all
            | sim.objectspecialproperty_renderable,
        )
    except Exception:
        pass

    # Em muitas versoes do CoppeliaSim, estes parametros tornam a forma
    # estatica e respondable para o motor de fisica.
    for param_name, value in [
        ("shapeintparam_static", 1),
        ("shapeintparam_respondable", 1),
    ]:
        try:
            sim.setObjectInt32Param(handle, getattr(sim, param_name), value)
        except Exception:
            pass


def create_or_update_obstacle(sim, name, center, size):
    """Cria um bloco visual se possivel; se falhar, o obstaculo segue virtual."""
    try:
        handle = sim.getObject(f"/{name}")
        try:
            sim.removeObject(handle)
        except Exception:
            sim.removeObjects([handle])
    except Exception:
        pass

    try:
        handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size.tolist(), 0)
        sim.setObjectAlias(handle, name)
        sim.setObjectPosition(handle, sim.handle_world, center.tolist())
        enable_collision_for_obstacle(sim, handle)
        try:
            sim.setShapeColor(
                handle,
                None,
                sim.colorcomponent_ambient_diffuse,
                [0.9, 0.1, 0.1],
            )
        except Exception:
            pass
        return handle
    except Exception as exc:
        print(f"Nao consegui criar visualmente o obstaculo {name}: {exc}")
        print("O planejamento ainda vai tratar esse obstaculo como virtual.")
        return None


def build_obstacles(sim, cube_start, drop_center):
    middle = (cube_start + drop_center) / 2.0

    # Os obstaculos ficam no caminho direto entre coleta e entrega, mas
    # deslocados para +Y. Assim eles bloqueiam a rota reta sem ocupar a regiao
    # principal da base/ombro do Franka, deixando um corredor livre em -Y.
    obstacles = [
        {
            "name": "Obstaculo_1",
            "center": middle + np.array([-0.02, -0.02, 0.13]),
            "size": np.array([0.13, 0.14, 0.22]),
        },
    ]

    for obstacle in obstacles:
        create_or_update_obstacle(
            sim,
            obstacle["name"],
            obstacle["center"],
            obstacle["size"],
        )

    return obstacles


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
    drop_center = cube_start + DROP_OFFSET

    pick_above = cube_start + np.array([0.0, 0.0, APPROACH_HEIGHT])
    pick_grasp = cube_start + np.array([0.0, 0.0, GRASP_HEIGHT])
    drop_above = drop_center + np.array([0.0, 0.0, APPROACH_HEIGHT])
    drop_grasp = drop_center + np.array([0.0, 0.0, GRASP_HEIGHT])

    obstacles = build_obstacles(sim, cube_start, drop_center)
    q_current = np.array([sim.getJointPosition(joint) for joint in joints])
    current_pos = fk_world(q_current, T_base_world)
    home_pos = fk_world(HOME_Q, T_base_world)

    approach_route_sparse = safe_corridor_route(current_pos, pick_above, obstacles)
    approach_route = densify_cartesian_route(approach_route_sparse)

    transport_route_sparse = plan_route_with_obstacles(pick_above, drop_above, obstacles)
    transport_route = densify_cartesian_route(transport_route_sparse)

    drop_route_sparse = [drop_above, drop_grasp]
    drop_route = densify_cartesian_route(drop_route_sparse, max_step=0.025)

    return_route_sparse = safe_corridor_route(drop_above, home_pos, obstacles)
    return_route = densify_cartesian_route(return_route_sparse)

    warn_if_route_crosses_obstacles(approach_route_sparse, obstacles)
    warn_if_route_crosses_obstacles(transport_route_sparse, obstacles)
    warn_if_route_crosses_obstacles(drop_route_sparse, obstacles)
    warn_if_route_crosses_obstacles(return_route_sparse, obstacles)

    print("\nObstaculos:")
    for obstacle in obstacles:
        print(
            f"  {obstacle['name']}: centro={np.round(obstacle['center'], 4)} "
            f"tam={np.round(obstacle['size'], 4)}"
        )

    print("\nRota de aproximacao segura:")
    for i, point in enumerate(approach_route_sparse):
        print(f"  p{i}: {np.round(point, 4)}")
    print(f"  rota densificada: {len(approach_route)} pontos cartesianos")

    print("\nRota de transporte com desvio:")
    for i, point in enumerate(transport_route_sparse):
        print(f"  p{i}: {np.round(point, 4)}")
    print(f"  rota densificada: {len(transport_route)} pontos cartesianos")

    print("\nRota de retorno segura:")
    for i, point in enumerate(return_route_sparse):
        print(f"  p{i}: {np.round(point, 4)}")
    print(f"  rota densificada: {len(return_route)} pontos cartesianos")

    print("\nDescida cartesiana na entrega:")
    for i, point in enumerate(drop_route_sparse):
        print(f"  p{i}: {np.round(point, 4)}")
    print(f"  rota densificada: {len(drop_route)} pontos cartesianos")

    planned_sections = [
        ("aproximacao", approach_route),
        ("transporte", transport_route),
        ("descida_entrega", drop_route),
        ("retorno", return_route),
    ]
    metrics = {
        "fast_mode": int(FAST_MODE),
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
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_waypoints_csv(OUTPUT_DIR / "waypoints_planejados.csv", planned_sections)
        save_route_plot(
            OUTPUT_DIR / "trajetoria_planejada.png",
            planned_sections,
            obstacles,
            cube_start,
            drop_center,
        )

    print("\nCalculando IK dos waypoints...")
    q_approach = solve_cartesian_route(approach_route, "aproximacao", T_base_world, q_current)
    q_pick_above = q_approach[-1]
    q_pick_grasp = solve_world_target("pegar cubo", pick_grasp, T_base_world, q_pick_above)

    q_transport = solve_cartesian_route(
        transport_route,
        "desvio/transporte",
        T_base_world,
        q_pick_above,
    )

    q_drop_descent = solve_cartesian_route(
        drop_route,
        "descida entrega",
        T_base_world,
        q_transport[-1],
    )
    q_drop_grasp = q_drop_descent[-1]
    q_return = solve_cartesian_route(return_route, "retorno", T_base_world, q_transport[-1])

    print("\nIniciando simulacao com desvio de obstaculos...")
    sim.startSimulation()
    execution_log = []
    execution_start = time.perf_counter()
    time.sleep(ACTION_PAUSE)

    try:
        for i, q_target in enumerate(q_approach, start=1):
            q_current = move_to_q(
                sim,
                joints,
                q_current,
                q_target,
                f"aproximacao segura {i}",
                steps=TRANSPORT_STEPS_PER_SEGMENT,
                logger=execution_log,
                tip=tip,
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

        for i, q_target in enumerate(q_transport, start=1):
            q_current = move_to_q(
                sim,
                joints,
                q_current,
                q_target,
                f"desvio {i}",
                steps=TRANSPORT_STEPS_PER_SEGMENT,
                logger=execution_log,
                tip=tip,
            )

        for i, q_target in enumerate(q_drop_descent, start=1):
            q_current = move_to_q(
                sim,
                joints,
                q_current,
                q_target,
                f"descida entrega {i}",
                steps=TRANSPORT_STEPS_PER_SEGMENT,
                logger=execution_log,
                tip=tip,
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
            "subir ate acima da entrega",
            steps=TRANSPORT_STEPS_PER_SEGMENT,
            logger=execution_log,
            tip=tip,
        )

        for i, q_target in enumerate(q_return, start=1):
            q_current = move_to_q(
                sim,
                joints,
                q_current,
                q_target,
                f"retorno seguro {i}",
                steps=TRANSPORT_STEPS_PER_SEGMENT,
                logger=execution_log,
                tip=tip,
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
            save_execution_csv(OUTPUT_DIR / "trajetoria_executada.csv", execution_log)
            save_metrics_csv(OUTPUT_DIR / "metricas_pick_and_place.csv", metrics)
            print(f"  resultados salvos em   : {OUTPUT_DIR.resolve()}")
    finally:
        time.sleep(ACTION_PAUSE)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()

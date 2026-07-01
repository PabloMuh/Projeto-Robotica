import csv
import re
import time
from pathlib import Path

import numpy as np

from cinematica_franka import matriz_coppelia_para_homogenea
from pick_and_place_obstaculos_franka import (
    HOME_Q,
    HOST,
    PORT,
    fk_world,
    get_alias,
    move_to_q,
    set_cube_parent,
)


OUTPUT_DIR = Path("resultados") / "quatro_frankas"

NUM_ROBOTS = 4
NUM_CYCLES = 3

# Cada Franka i pega o cubo da estacao (i + SOURCE_OFFSET) % 4
# e leva para a propria estacao i. Com offset 1, o fluxo fica:
# F0 <- E1, F1 <- E2, F2 <- E3, F3 <- E0.
SOURCE_OFFSET = 1

CREATE_STATIONS_IF_MISSING = True
RESET_STATIONS_AT_START = True
RESET_CUBES_AT_START = True
GO_HOME_AT_END = True

DIRECT_MODE = False
FAST_MODE = True

if FAST_MODE:
    STEPS_PER_SEGMENT = 36
    GRASP_STEPS = 20
    DT = 0.006
    ACTION_PAUSE = 0.05
else:
    STEPS_PER_SEGMENT = 80
    GRASP_STEPS = 45
    DT = 0.012
    ACTION_PAUSE = 0.15

STATION_CENTER_BLEND = 0.20
DEFAULT_STATION_Z = 0.05
MIN_STATION_Z = 0.03
STATION_SIZE = np.array([0.20, 0.20, 0.025])
CUBE_SIZE = np.array([0.045, 0.045, 0.045])

APPROACH_HEIGHT = 0.30
GRASP_HEIGHT = 0.105
MAX_IK_ERROR = 0.015

STATION_COLORS = [
    [0.10, 0.45, 0.95],
    [0.95, 0.22, 0.12],
    [0.16, 0.70, 0.25],
    [0.95, 0.75, 0.10],
]


def short_name(alias):
    return str(alias).strip("/").split("/")[-1]


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


def set_non_collidable_marker(sim, handle):
    try:
        sim.setObjectSpecialProperty(handle, sim.objectspecialproperty_renderable)
    except Exception:
        pass


def create_or_reuse_box(sim, name, center, size, color, recreate=False, marker=False):
    if recreate:
        remove_if_exists(sim, name)

    try:
        handle = sim.getObject(f"/{name}")
        sim.setObjectPosition(handle, sim.handle_world, center.tolist())
    except Exception:
        handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size.tolist(), 0)
        sim.setObjectAlias(handle, name)
        sim.setObjectPosition(handle, sim.handle_world, center.tolist())

    color_shape(sim, handle, color)
    if marker:
        set_non_collidable_marker(sim, handle)
    return handle


def find_franka_root(sim, index):
    candidates = [
        f"/Franka[{index}]",
        f"/Franka#{index}",
        "/Franka" if index == 0 else None,
    ]
    for path in candidates:
        if not path:
            continue
        try:
            return sim.getObject(path)
        except Exception:
            pass

    try:
        all_objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    except Exception:
        all_objects = sim.getObjectsInTree(-1, sim.handle_all, 0)

    target_patterns = [
        re.compile(rf"^Franka\[{index}\]$"),
        re.compile(rf"^Franka#{index}$"),
        re.compile(r"^Franka$") if index == 0 else None,
    ]
    for obj in all_objects:
        alias = short_name(get_alias(sim, obj))
        if any(pattern and pattern.match(alias) for pattern in target_patterns):
            return obj

    raise RuntimeError(f"Nao encontrei o modelo Franka[{index}] na cena.")


def get_franka_by_index(sim, index):
    robot = find_franka_root(sim, index)
    joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)[:7]
    if len(joints) < 7:
        raise RuntimeError(f"Franka[{index}] nao tem 7 juntas detectadas.")

    tip = find_tip_in_robot_tree(sim, robot)
    t_base_world = matriz_coppelia_para_homogenea(
        sim.getObjectMatrix(joints[0], sim.handle_world)
    )

    return {
        "index": index,
        "robot": robot,
        "joints": joints,
        "tip": tip,
        "t_base_world": t_base_world,
        "q_current": np.array([sim.getJointPosition(joint) for joint in joints]),
    }


def find_tip_in_robot_tree(sim, robot):
    objects = sim.getObjectsInTree(robot, sim.handle_all, 0)
    candidates = [obj for obj in objects if sim.getObjectType(obj) != sim.object_joint_type]

    for obj in candidates:
        alias = short_name(get_alias(sim, obj)).lower()
        if "connection" in alias:
            return obj

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


def default_station_position(sim, owner_robot_info, visitor_robot_info, scene_center_xy):
    owner_base = np.array(sim.getObjectPosition(owner_robot_info["joints"][0], sim.handle_world))
    visitor_base = np.array(sim.getObjectPosition(visitor_robot_info["joints"][0], sim.handle_world))

    pair_mid = (owner_base[0:2] + visitor_base[0:2]) / 2.0
    station_xy = (1.0 - STATION_CENTER_BLEND) * pair_mid + STATION_CENTER_BLEND * scene_center_xy

    station = owner_base.copy()
    station[0:2] = station_xy
    station[2] = DEFAULT_STATION_Z
    return station

def normalize_station_position(pos):
    pos = np.asarray(pos, dtype=float).copy()
    if pos[2] < MIN_STATION_Z:
        pos[2] = DEFAULT_STATION_Z
    return pos


def create_or_read_stations_and_cubes(sim, robots):
    base_positions = [
        np.array(sim.getObjectPosition(robot["joints"][0], sim.handle_world))
        for robot in robots
    ]
    scene_center_xy = np.mean(np.array(base_positions)[:, 0:2], axis=0)

    stations = []
    cubes = []
    for i, robot in enumerate(robots):
        station_name = f"Estacao_Franka_{i}"
        cube_name = f"Cubo_Franka_{i}"

        visitor_index = (i - SOURCE_OFFSET) % NUM_ROBOTS
        default_pos = default_station_position(sim, robot, robots[visitor_index], scene_center_xy)
        station_created = False
        try:
            station_handle = sim.getObject(f"/{station_name}")
            if RESET_STATIONS_AT_START:
                station_pos = default_pos
            else:
                station_pos = np.array(sim.getObjectPosition(station_handle, sim.handle_world))
        except Exception:
            if not CREATE_STATIONS_IF_MISSING:
                raise RuntimeError(
                    f"Estacao {station_name} nao existe. Crie no CoppeliaSim "
                    "ou ative CREATE_STATIONS_IF_MISSING."
                )
            station_pos = default_pos
            station_handle = create_or_reuse_box(
                sim,
                station_name,
                station_pos,
                STATION_SIZE,
                STATION_COLORS[i],
                recreate=True,
                marker=True,
            )
            station_created = True

        original_station_pos = station_pos.copy()
        station_pos = normalize_station_position(station_pos)
        if RESET_STATIONS_AT_START or not np.allclose(original_station_pos, station_pos):
            if not np.allclose(original_station_pos, station_pos):
                print(
                    f"Ajustando {station_name}: "
                    f"{np.round(original_station_pos, 4)} -> {np.round(station_pos, 4)}"
                )
            sim.setObjectPosition(station_handle, sim.handle_world, station_pos.tolist())

        if not station_created:
            color_shape(sim, station_handle, STATION_COLORS[i])
            set_non_collidable_marker(sim, station_handle)

        cube_pos = station_pos.copy()
        try:
            cube_handle = sim.getObject(f"/{cube_name}")
            try:
                set_cube_parent(sim, cube_handle, -1)
            except Exception:
                pass
            if RESET_CUBES_AT_START:
                sim.setObjectPosition(cube_handle, sim.handle_world, cube_pos.tolist())
        except Exception:
            cube_handle = create_or_reuse_box(
                sim,
                cube_name,
                cube_pos,
                CUBE_SIZE,
                STATION_COLORS[i],
                recreate=True,
                marker=False,
            )

        stations.append({
            "index": i,
            "name": station_name,
            "handle": station_handle,
            "position": station_pos,
        })
        cubes.append({
            "index": i,
            "name": cube_name,
            "handle": cube_handle,
        })

    return stations, cubes


def solve_world_target(robot_info, label, target_world, q_seed):
    from inversa import mundo_para_base, resolver_ik

    target_base = mundo_para_base(target_world, robot_info["t_base_world"])
    q_sol, _, error_base = resolver_ik(target_base, q_inicial=q_seed, max_iter=2200)
    pos_world = fk_world(q_sol, robot_info["t_base_world"])
    error_world = float(np.linalg.norm(pos_world - target_world))

    print(f"Franka[{robot_info['index']}] - {label}")
    print(f"  alvo mundo : {np.round(target_world, 4)}")
    print(f"  erro modelo: {error_world:.6f} m")

    if error_world > MAX_IK_ERROR:
        raise RuntimeError(
            f"IK falhou para Franka[{robot_info['index']}] em '{label}'. "
            f"Erro base={error_base:.6f} m, erro mundo={error_world:.6f} m. "
            "Mova as estacoes para uma regiao mais compartilhada entre os robos."
        )

    return q_sol


def precompute_robot_targets(robots, stations):
    targets = []
    for robot in robots:
        i = robot["index"]
        source_station = (i + SOURCE_OFFSET) % NUM_ROBOTS
        source_pos = stations[source_station]["position"]
        own_pos = stations[i]["position"]

        source_above = source_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
        source_grasp = source_pos + np.array([0.0, 0.0, GRASP_HEIGHT])
        own_above = own_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
        own_grasp = own_pos + np.array([0.0, 0.0, GRASP_HEIGHT])

        q_seed = robot["q_current"]
        q_source_above = solve_world_target(
            robot,
            f"acima da estacao {source_station}",
            source_above,
            q_seed,
        )
        q_source_grasp = solve_world_target(
            robot,
            f"pegar na estacao {source_station}",
            source_grasp,
            q_source_above,
        )
        q_own_above = solve_world_target(
            robot,
            f"acima da propria estacao {i}",
            own_above,
            q_source_above,
        )
        q_own_grasp = solve_world_target(
            robot,
            f"soltar na propria estacao {i}",
            own_grasp,
            q_own_above,
        )

        targets.append({
            "source_station": source_station,
            "destination_station": i,
            "source_above": q_source_above,
            "source_grasp": q_source_grasp,
            "own_above": q_own_above,
            "own_grasp": q_own_grasp,
        })

    return targets


def set_joint(sim, joint, angle):
    if DIRECT_MODE:
        sim.setJointPosition(joint, float(angle))
    else:
        sim.setJointTargetPosition(joint, float(angle))


def smoothstep(alpha):
    return alpha * alpha * (3.0 - 2.0 * alpha)


def move_all_to_targets(sim, robots, q_targets, label, steps=STEPS_PER_SEGMENT):
    print(f"Executando fase sincronizada: {label}")
    q_starts = [robot["q_current"].copy() for robot in robots]

    for step in range(steps):
        alpha = smoothstep(step / (steps - 1))
        for robot, q_start, q_target in zip(robots, q_starts, q_targets):
            q = (1.0 - alpha) * q_start + alpha * q_target
            for joint, angle in zip(robot["joints"], q):
                set_joint(sim, joint, angle)
        time.sleep(DT)

    for robot, q_target in zip(robots, q_targets):
        robot["q_current"] = q_target.copy()


def attach_all(sim, robots, station_cube, targets, events, cycle):
    print("Prendendo cubos aos quatro efetuadores...")
    for robot, target in zip(robots, targets):
        source_station = target["source_station"]
        cube = station_cube[source_station]
        set_cube_parent(sim, cube["handle"], robot["tip"])
        events.append({
            "cycle": cycle,
            "robot": robot["index"],
            "action": "pick",
            "station": source_station,
            "cube": cube["name"],
        })


def detach_all(sim, robots, station_cube, targets, events, cycle):
    print("Soltando cubos nas proprias estacoes...")
    new_station_cube = [None] * NUM_ROBOTS

    for robot, target in zip(robots, targets):
        source_station = target["source_station"]
        destination_station = target["destination_station"]
        cube = station_cube[source_station]
        set_cube_parent(sim, cube["handle"], -1)
        new_station_cube[destination_station] = cube
        events.append({
            "cycle": cycle,
            "robot": robot["index"],
            "action": "drop",
            "station": destination_station,
            "cube": cube["name"],
        })

    return new_station_cube


def print_station_state(station_cube, title):
    print(f"\n{title}")
    for station_index, cube in enumerate(station_cube):
        cube_name = cube["name"] if cube else "vazia"
        print(f"  Estacao {station_index}: {cube_name}")


def save_events_csv(path, events):
    if not events:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["cycle", "robot", "action", "station", "cube"],
        )
        writer.writeheader()
        writer.writerows(events)


def save_station_csv(path, stations):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["station", "x", "y", "z"])
        writer.writeheader()
        for station in stations:
            pos = station["position"]
            writer.writerow({
                "station": station["index"],
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
            })


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient(host=HOST, port=PORT)
    sim = client.require("sim")
    print("Conectado ao CoppeliaSim.")

    robots = [get_franka_by_index(sim, i) for i in range(NUM_ROBOTS)]
    for robot in robots:
        print(
            f"Franka[{robot['index']}] encontrado: "
            f"raiz={get_alias(sim, robot['robot'])}, tip={get_alias(sim, robot['tip'])}"
        )

    stations, cubes = create_or_read_stations_and_cubes(sim, robots)
    station_cube = cubes.copy()

    print("\nEstacoes:")
    for station in stations:
        print(f"  {station['name']}: {np.round(station['position'], 4)}")

    print_station_state(station_cube, "Estado inicial dos cubos:")

    print("\nCalculando IK das posicoes de troca...")
    targets = precompute_robot_targets(robots, stations)

    save_station_csv(OUTPUT_DIR / "estacoes_quatro_frankas.csv", stations)

    print("\nIniciando simulacao sincronizada com quatro Frankas...")
    sim.startSimulation()
    events = []
    time.sleep(ACTION_PAUSE)

    try:
        for cycle in range(1, NUM_CYCLES + 1):
            print(f"\n================ CICLO {cycle}/{NUM_CYCLES} ================")

            move_all_to_targets(
                sim,
                robots,
                [target["source_above"] for target in targets],
                "ir acima da estacao de outro robo",
            )
            move_all_to_targets(
                sim,
                robots,
                [target["source_grasp"] for target in targets],
                "descer para pegar cubos",
                steps=GRASP_STEPS,
            )
            attach_all(sim, robots, station_cube, targets, events, cycle)
            time.sleep(ACTION_PAUSE)

            move_all_to_targets(
                sim,
                robots,
                [target["source_above"] for target in targets],
                "subir com cubos",
                steps=GRASP_STEPS,
            )
            move_all_to_targets(
                sim,
                robots,
                [target["own_above"] for target in targets],
                "levar cubos para a propria estacao",
            )
            move_all_to_targets(
                sim,
                robots,
                [target["own_grasp"] for target in targets],
                "descer para soltar cubos",
                steps=GRASP_STEPS,
            )
            station_cube = detach_all(sim, robots, station_cube, targets, events, cycle)
            time.sleep(ACTION_PAUSE)

            move_all_to_targets(
                sim,
                robots,
                [target["own_above"] for target in targets],
                "subir apos soltar",
                steps=GRASP_STEPS,
            )
            print_station_state(station_cube, f"Estado apos ciclo {cycle}:")

        if GO_HOME_AT_END:
            move_all_to_targets(
                sim,
                robots,
                [HOME_Q for _ in robots],
                "voltar para home",
            )

        save_events_csv(OUTPUT_DIR / "eventos_troca_quatro_frankas.csv", events)
        print(f"\nResultados salvos em: {OUTPUT_DIR.resolve()}")
    finally:
        time.sleep(ACTION_PAUSE)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()











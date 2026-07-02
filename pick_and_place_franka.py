import time

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from cinematica_franka import calcular_cinematica_direta, matriz_coppelia_para_homogenea
from inversa import mundo_para_base, resolver_ik


HOST = "127.0.0.1"
PORT = 23000

DIRECT_MODE = False
STEPS_PER_SEGMENT = 120
DT = 0.02

APPROACH_HEIGHT = 0.28
GRASP_HEIGHT = 0.10
TRANSPORT_HEIGHT = 0.36

DROP_OFFSET = np.array([-0.35, 0.30, 0.0])
HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.0])


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


def move_to_q(sim, joints, q_start, q_end, name):
    print(f"Executando trecho: {name}")
    for q in interpolate(q_start, q_end, STEPS_PER_SEGMENT):
        for joint, angle in zip(joints, q):
            set_joint(sim, joint, angle, direct_mode=DIRECT_MODE)
        time.sleep(DT)
    return q_end.copy()


def fk_world(q, T_base_world):
    T_base, _, _ = calcular_cinematica_direta(q)
    T_world = T_base_world @ T_base
    return T_world[0:3, 3]


def solve_world_target(name, target_world, T_base_world, q_seed):
    target_base = mundo_para_base(target_world, T_base_world)
    q_sol, success, error_base = resolver_ik(target_base, q_inicial=q_seed)
    pos_world = fk_world(q_sol, T_base_world)
    error_world = float(np.linalg.norm(pos_world - target_world))

    print(f"{name}")
    print(f"  alvo mundo : {np.round(target_world, 4)}")
    print(f"  solucao q  : {np.round(q_sol, 4)}")
    print(f"  erro modelo: {error_world:.6f} m")

    if not success or error_world > 0.005:
        raise RuntimeError(
            f"IK falhou no waypoint '{name}'. "
            f"Erro base={error_base:.6f} m, erro mundo={error_world:.6f} m."
        )

    return q_sol


def set_cube_parent(sim, cube, parent):
    # keepInPlace=True preserva a pose global no instante de prender/soltar.
    sim.setObjectParent(cube, parent, True)


def main():
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
    transport_mid = (cube_start + drop_center) / 2.0 + np.array([0.0, 0.0, TRANSPORT_HEIGHT])
    drop_above = drop_center + np.array([0.0, 0.0, APPROACH_HEIGHT])
    drop_grasp = drop_center + np.array([0.0, 0.0, GRASP_HEIGHT])

    print("\nWaypoints cartesianos:")
    for name, point in [
        ("acima do cubo", pick_above),
        ("pegar cubo", pick_grasp),
        ("transporte alto", transport_mid),
        ("acima da entrega", drop_above),
        ("soltar cubo", drop_grasp),
    ]:
        print(f"  {name:16s}: {np.round(point, 4)}")

    q_current = np.array([sim.getJointPosition(joint) for joint in joints])

    print("\nCalculando IK dos waypoints...")
    q_home = solve_world_target("home", fk_world(HOME_Q, T_base_world), T_base_world, q_current)
    q_pick_above = solve_world_target("acima do cubo", pick_above, T_base_world, q_home)
    q_pick_grasp = solve_world_target("pegar cubo", pick_grasp, T_base_world, q_pick_above)
    q_transport_mid = solve_world_target("transporte alto", transport_mid, T_base_world, q_pick_grasp)
    q_drop_above = solve_world_target("acima da entrega", drop_above, T_base_world, q_transport_mid)
    q_drop_grasp = solve_world_target("soltar cubo", drop_grasp, T_base_world, q_drop_above)

    print("\nIniciando simulacao do pick-and-place...")
    sim.startSimulation()
    time.sleep(0.5)

    try:
        q_current = move_to_q(sim, joints, q_current, q_home, "ir para home")
        time.sleep(0.4)

        q_current = move_to_q(sim, joints, q_current, q_pick_above, "aproximar acima do cubo")
        q_current = move_to_q(sim, joints, q_current, q_pick_grasp, "descer ate o cubo")

        print("Prendendo cubo ao efetuador...")
        set_cube_parent(sim, cube, tip)
        time.sleep(0.4)

        q_current = move_to_q(sim, joints, q_current, q_pick_above, "subir com cubo")
        q_current = move_to_q(sim, joints, q_current, q_transport_mid, "transporte alto")
        q_current = move_to_q(sim, joints, q_current, q_drop_above, "aproximar entrega")
        q_current = move_to_q(sim, joints, q_current, q_drop_grasp, "descer na entrega")

        print("Soltando cubo...")
        set_cube_parent(sim, cube, -1)
        time.sleep(0.4)

        q_current = move_to_q(sim, joints, q_current, q_drop_above, "subir apos soltar")
        q_current = move_to_q(sim, joints, q_current, q_home, "voltar para home")

        final_cube = np.array(sim.getObjectPosition(cube, sim.handle_world))
        print("\nResultado:")
        print(f"  posicao inicial do cubo: {np.round(cube_start, 4)}")
        print(f"  destino planejado      : {np.round(drop_center, 4)}")
        print(f"  posicao final do cubo  : {np.round(final_cube, 4)}")
        print(f"  erro de entrega        : {np.linalg.norm(final_cube - drop_center):.6f} m")
    finally:
        time.sleep(0.8)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()

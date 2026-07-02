import time

import numpy as np


HOST = "127.0.0.1"
PORT = 23000

DIRECT_MODE = False
STEPS_PER_SEGMENT = 140
DT = 0.02
PAUSE_AT_POSE = 0.8

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.4363, -3.0718])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.6251, 3.0718])

POSES = [
    (
        "home",
        np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.0]),
    ),
    (
        "alto_central",
        np.array([0.0, -1.15, 0.0, -1.25, 0.0, 2.25, 0.0]),
    ),
    (
        "alcance_esquerda",
        np.array([0.65, -0.85, -0.35, -2.05, 0.25, 1.85, 0.65]),
    ),
    (
        "alcance_direita",
        np.array([-0.65, -0.85, 0.35, -2.05, -0.25, 1.85, -0.65]),
    ),
    (
        "baixo_frente",
        np.array([0.0, 0.65, 0.0, -0.8, 0.0, 1.65, 0.0]),
    ),
    (
        "pose_giro_punho",
        np.array([0.9, -0.6, -0.7, -2.35, 0.55, 2.05, 1.2]),
    ),
    (
        "retorno_home",
        np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.0]),
    ),
]


def get_alias(sim, obj):
    try:
        return sim.getObjectAlias(obj, 1)
    except Exception:
        try:
            return sim.getObjectAlias(obj)
        except Exception:
            return str(obj)


def find_end_effector(sim, robot):
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
        alpha = i / (steps - 1)
        alpha = smoothstep(alpha)
        yield (1.0 - alpha) * q_start + alpha * q_end


def validate_poses(poses):
    for name, q in poses:
        if q.shape != (7,):
            raise ValueError(f"Pose {name} precisa ter 7 juntas.")

        below = np.where(q < Q_MIN)[0]
        above = np.where(q > Q_MAX)[0]
        if below.size or above.size:
            raise ValueError(
                f"Pose {name} passa dos limites articulares. "
                f"Abaixo: {below + 1}; acima: {above + 1}"
            )


def print_tip_pose(sim, tip, pose_name):
    pos = np.array(sim.getObjectPosition(tip, sim.handle_world))
    quat = np.array(sim.getObjectQuaternion(tip, sim.handle_world))
    print(f"{pose_name}: pos={np.round(pos, 4)} quat={np.round(quat, 4)}")


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    validate_poses(POSES)

    client = RemoteAPIClient(host=HOST, port=PORT)
    sim = client.require("sim")
    print("Conectado ao CoppeliaSim.")

    robot = sim.getObject("/Franka")
    joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)[:7]
    if len(joints) < 7:
        raise RuntimeError("Nao encontrei as 7 juntas principais do Franka.")

    tip = find_end_effector(sim, robot)
    print(f"Efetuador final usado: {get_alias(sim, tip)}")

    print("Juntas usadas:")
    for i, joint in enumerate(joints, start=1):
        print(f"q{i}: {get_alias(sim, joint)}")

    q_current = np.array([sim.getJointPosition(joint) for joint in joints])

    print("\nIniciando sequencia de poses...")
    sim.startSimulation()
    time.sleep(0.5)

    try:
        for pose_name, q_target in POSES:
            print(f"\nIndo para pose: {pose_name}")

            for q_step in interpolate(q_current, q_target, STEPS_PER_SEGMENT):
                for joint, angle in zip(joints, q_step):
                    set_joint(sim, joint, angle, direct_mode=DIRECT_MODE)
                time.sleep(DT)

            q_current = q_target.copy()
            time.sleep(PAUSE_AT_POSE)
            print_tip_pose(sim, tip, pose_name)

        print("\nSequencia finalizada.")
    finally:
        time.sleep(0.5)
        sim.stopSimulation()
        print("Simulacao parada.")


if __name__ == "__main__":
    main()

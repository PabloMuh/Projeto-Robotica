import time
import numpy as np
from scipy.optimize import least_squares
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# Conexão com o CoppeliaSim
# ============================================================

client = RemoteAPIClient(host="127.0.0.1", port=23000)
sim = client.require("sim")

print("Conectado ao CoppeliaSim.")


# ============================================================
# Funções auxiliares
# ============================================================

def get_alias(obj):
    try:
        return sim.getObjectAlias(obj, 1)
    except Exception:
        return sim.getObjectAlias(obj)


def get_depth(obj):
    """
    Mede a profundidade do objeto na árvore da cena.
    Quanto maior, mais perto do fim da cadeia.
    """
    depth = 0
    current = obj

    while True:
        parent = sim.getObjectParent(current)

        if parent == -1:
            break

        depth += 1
        current = parent

    return depth


def find_end_effector_by_depth(robot):
    """
    Encontra um candidato para o efetuador final escolhendo
    o objeto não-junta mais profundo dentro da árvore do Franka.
    """
    objects = sim.getObjectsInTree(robot, sim.handle_all, 0)

    candidates = []

    for obj in objects:
        obj_type = sim.getObjectType(obj)

        # Ignora juntas
        if obj_type == sim.object_joint_type:
            continue

        try:
            alias = get_alias(obj)
        except Exception:
            alias = str(obj)

        depth = get_depth(obj)

        candidates.append((depth, alias, obj, obj_type))

    if not candidates:
        raise RuntimeError("Não encontrei nenhum objeto candidato a efetuador final.")

    candidates.sort(key=lambda x: x[0], reverse=True)

    print("\nCandidatos a efetuador final:")
    for depth, alias, obj, obj_type in candidates[:15]:
        print(f"depth={depth} | {alias} | handle={obj} | type={obj_type}")

    chosen = candidates[0][2]

    return chosen


def set_robot_q(joints, q):
    """
    Define diretamente as juntas.
    Usado durante a otimização da cinemática inversa.
    """
    for joint, angle in zip(joints, q):
        sim.setJointPosition(joint, float(angle))


def get_tip_position(tip):
    return np.array(sim.getObjectPosition(tip, sim.handle_world), dtype=float)


def move_robot_smooth(joints, q_start, q_end, steps=180, dt=0.025):
    """
    Movimento suave para visualização.
    """
    for i in range(steps):
        alpha = i / (steps - 1)
        q = (1 - alpha) * q_start + alpha * q_end

        for joint, angle in zip(joints, q):
            sim.setJointTargetPosition(joint, float(angle))

        time.sleep(dt)


# ============================================================
# Buscar robô, juntas, bloco e ponta
# ============================================================

robot = sim.getObject("/Franka")
cube = sim.getObject("/Cuboid")

joints = sim.getObjectsInTree(
    robot,
    sim.object_joint_type,
    0
)

if len(joints) < 7:
    raise RuntimeError(f"Foram encontradas apenas {len(joints)} juntas.")

joints = joints[:7]

print("\nJuntas encontradas:")
for i, joint in enumerate(joints, start=1):
    print(f"q{i}: {get_alias(joint)} | handle={joint}")

tip = find_end_effector_by_depth(robot)

print("\nEfetuador final escolhido:")
print(get_alias(tip))


# ============================================================
# Posição do bloco e alvo
# ============================================================

cube_pos = np.array(sim.getObjectPosition(cube, sim.handle_world), dtype=float)

print("\nPosição do bloco:")
print(cube_pos)

# O alvo fica um pouco acima do cubo para não tentar atravessar o bloco
target_pos = cube_pos + np.array([0.0, 0.0, 0.0])

print("\nPosição alvo usada:")
print(target_pos)


# ============================================================
# Configuração inicial do Franka
# ============================================================

q_initial = np.array([
    0.0,
    0.0,
    0.0,
    -1.5,
    0.0,
    1.5,
    0.0
])

q_ref = np.array([
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.6,
    0.0
])
# Limites aproximados do Franka
q_min = np.array([
    -2.90,
    -1.83,
    -2.90,
    -3.07,
    -2.87,
    0.44,
    -3.05
])

q_max = np.array([
    2.90,
    1.83,
    2.90,
    -0.12,
    2.87,
    4.62,
    3.05
])
W_POSTURE = np.array([
    0.03,  # q1
    0.05,  # q2
    0.03,  # q3
    0.08,  # q4
    0.40,  # q5
    0.70,  # q6
    0.60   # q7
])

# ============================================================
# Função de erro da cinemática inversa
# ============================================================

def ik_error(q):
    """
    Erro da cinemática inversa.

    A primeira parte faz a ponta chegar no alvo.
    A segunda parte evita posturas estranhas, principalmente no punho.
    """
    set_robot_q(joints, q)

    tip_pos = get_tip_position(tip)

    # Erro de posição da ponta em relação ao alvo
    position_error = 5.0 * (tip_pos - target_pos)

    # Penalização de postura para evitar link 7 muito inclinado
    posture_error = W_POSTURE * (q - q_ref)

    return np.concatenate([
        position_error,
        posture_error
    ])


# ============================================================
# Resolver cinemática inversa
# ============================================================

print("\nColocando robô na configuração inicial...")
set_robot_q(joints, q_initial)
time.sleep(0.2)

print("\nResolvendo cinemática inversa...")

result = least_squares(
    ik_error,
    q_initial,
    bounds=(q_min, q_max),
    max_nfev=500,
    xtol=1e-5,
    ftol=1e-5,
    gtol=1e-5
)

q_solution = result.x

set_robot_q(joints, q_solution)
time.sleep(0.2)

tip_final = get_tip_position(tip)
erro_final = np.linalg.norm(tip_final - target_pos)

print("\nSolução encontrada q:")
print(q_solution)

print("\nPosição final da ponta:")
print(tip_final)

print("\nErro final até o alvo:")
print(erro_final, "m")


# ============================================================
# Movimento visual na simulação
# ============================================================

print("\nIniciando simulação visual...")

# Volta para posição inicial antes de animar
set_robot_q(joints, q_initial)

sim.startSimulation()
time.sleep(0.5)

for joint, angle in zip(joints, q_initial):
    sim.setJointTargetPosition(joint, float(angle))

time.sleep(1.0)

move_robot_smooth(
    joints,
    q_initial,
    q_solution,
    steps=200,
    dt=0.025
)

time.sleep(2.0)

print("\nResultado final:")
print("Posição do bloco:", cube_pos)
print("Alvo usado:", target_pos)
print("Posição final da ponta:", get_tip_position(tip))

sim.stopSimulation()

print("\nFim.")
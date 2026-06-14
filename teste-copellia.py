import time
import numpy as np
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
    """
    Tenta obter o nome do objeto no CoppeliaSim.
    """
    try:
        return sim.getObjectAlias(obj, 1)
    except Exception:
        try:
            return sim.getObjectAlias(obj)
        except Exception:
            return str(obj)


def set_joint(joint, angle, direct_mode=False):
    """
    Move uma junta.

    direct_mode=False:
        usa setJointTargetPosition, mais correto para simulação com motor.

    direct_mode=True:
        usa setJointPosition, força a posição cinematicamente.
        Use caso o robô não se mova com setJointTargetPosition.
    """
    if direct_mode:
        sim.setJointPosition(joint, float(angle))
    else:
        sim.setJointTargetPosition(joint, float(angle))


def interpolate(q_start, q_end, steps):
    """
    Gera uma trajetória linear no espaço das juntas.
    """
    for i in range(steps):
        alpha = i / (steps - 1)
        yield (1 - alpha) * q_start + alpha * q_end


# ============================================================
# Buscar o robô e suas juntas automaticamente
# ============================================================

robot = sim.getObject("/Franka")

all_joints = sim.getObjectsInTree(
    robot,
    sim.object_joint_type,
    0
)

print("\nJuntas encontradas dentro de /Franka:")

for i, joint in enumerate(all_joints):
    print(f"{i}: {get_alias(joint)} | handle = {joint}")

if len(all_joints) < 7:
    raise RuntimeError(
        f"Foram encontradas apenas {len(all_joints)} juntas. "
        "Verifique se o modelo Franka está carregado corretamente."
    )

# Considera as 7 primeiras juntas como as juntas principais do braço
joints = all_joints[:7]

print("\nUsando estas 7 juntas principais:")

for i, joint in enumerate(joints, start=1):
    print(f"q{i}: {get_alias(joint)}")


# ============================================================
# Tentar encontrar o efetuador final
# ============================================================

end_effector = None

all_objects = sim.getObjectsInTree(robot, -1, 0)

for obj in all_objects:
    alias = get_alias(obj).lower()

    if "connection" in alias or "tip" in alias or "end" in alias:
        end_effector = obj
        print(f"\nPossível efetuador encontrado: {get_alias(obj)}")
        break

if end_effector is None:
    print("\nNão encontrei automaticamente o efetuador final.")
    print("O robô ainda vai se mover, mas não vou imprimir a posição final.")
else:
    print("A posição final do efetuador será impressa ao final.")


# ============================================================
# Configurações articulares
# ============================================================

# Posição inicial segura
q0 = np.array([
    0.0,
    0.0,
    0.0,
    -1.5,
    0.0,
    1.5,
    0.0
])

# Posição intermediária
q1 = np.array([
    0.0,
    -0.6,
    0.0,
    -2.0,
    0.0,
    1.6,
    0.0
])

# Posição final
q2 = np.array([
    0.6,
    -0.8,
    0.4,
    -2.3,
    0.3,
    1.7,
    0.8
])

trajectory = [q0, q1, q2, q0]


# ============================================================
# Configuração de movimento
# ============================================================

# Deixe False primeiro.
# Se o robô não se mexer, mude para True.
DIRECT_MODE = False

STEPS_PER_SEGMENT = 120
DT = 0.025


# ============================================================
# Executar simulação
# ============================================================

print("\nIniciando simulação...")

sim.startSimulation()
time.sleep(0.5)

# Vai para q0
for joint, angle in zip(joints, q0):
    set_joint(joint, angle, direct_mode=DIRECT_MODE)

time.sleep(1.5)

# Executa trajetória
for k in range(len(trajectory) - 1):
    qa = trajectory[k]
    qb = trajectory[k + 1]

    print(f"\nMovendo do ponto {k} para o ponto {k + 1}")

    for q in interpolate(qa, qb, STEPS_PER_SEGMENT):
        for joint, angle in zip(joints, q):
            set_joint(joint, angle, direct_mode=DIRECT_MODE)

        time.sleep(DT)

time.sleep(1.0)

# Lê posição final do efetuador, se encontrado
if end_effector is not None:
    pos = sim.getObjectPosition(end_effector, sim.handle_world)
    quat = sim.getObjectQuaternion(end_effector, sim.handle_world)

    print("\nPosição final do efetuador [x, y, z]:")
    print(pos)

    print("\nQuatérnio final [x, y, z, w]:")
    print(quat)

print("\nParando simulação...")
sim.stopSimulation()

print("Fim.")
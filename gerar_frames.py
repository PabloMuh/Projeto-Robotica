"""
gerar_frames.py
================
Conecta no CoppeliaSim, lê a pose (posição + orientação) de cada frame do
robô Franka e gera a figura "franka_frames.jpg" com os triedros (eixos
x, y, z) de cada junta -- a figura "Posicionamento dos frames do robô"
usada no relatório.

Pré-requisitos (rodar uma vez no terminal):
    pip install coppeliasim-zmqremoteapi-client numpy matplotlib

Como usar:
    1. Abra a cena do Franka no CoppeliaSim e deixe-o rodando.
       (A API remota ZMQ já vem habilitada por padrão na porta 23000.)
    2. Rode:  python gerar_frames.py
    3. O arquivo "franka_frames.jpg" será salvo na mesma pasta.
       Suba esse arquivo no Overleaf para preencher a Figura dos frames.

Obs.: não é preciso iniciar a simulação (play). O script só lê as poses
dos objetos na cena, coloca o robô numa configuração de demonstração para
os frames ficarem bem visíveis e gera a figura.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário p/ projeção 3d)
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ============================================================
# Configurações
# ============================================================

HOST = "127.0.0.1"
PORT = 23000

# Configuração de demonstração das juntas (rad), só para abrir o braço
# e deixar os frames espaçados na figura. Ajuste se quiser outra pose.
Q_DEMO = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.0])

ARROW_LEN = 0.08      # comprimento dos eixos de cada frame (m)
OUT_FILE = "franka_frames.jpg"

# ============================================================
# Conexão com o CoppeliaSim
# ============================================================

client = RemoteAPIClient(host=HOST, port=PORT)
sim = client.require("sim")
print("Conectado ao CoppeliaSim.")


def get_alias(obj):
    try:
        return sim.getObjectAlias(obj, 1)
    except Exception:
        return str(obj)


def matrix_to_Rp(m):
    """
    Converte a matriz 3x4 do CoppeliaSim (lista de 12 valores) em:
      R -> matriz de rotação 3x3
      p -> vetor de posição 3x1
    """
    M = np.array(m, dtype=float).reshape(3, 4)
    R = M[:, :3]
    p = M[:, 3]
    return R, p


# ============================================================
# Buscar robô, juntas e efetuador
# ============================================================

robot = sim.getObject("/Franka")

joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)
if len(joints) < 7:
    raise RuntimeError(f"Foram encontradas apenas {len(joints)} juntas.")
joints = joints[:7]

print("\nJuntas encontradas:")
for i, j in enumerate(joints, start=1):
    print(f"q{i}: {get_alias(j)} | handle={j}")


def find_end_effector(robot):
    """Escolhe o objeto não-junta mais profundo como efetuador final."""
    objs = sim.getObjectsInTree(robot, sim.handle_all, 0)

    def depth(o):
        d, cur = 0, o
        while True:
            par = sim.getObjectParent(cur)
            if par == -1:
                break
            d += 1
            cur = par
        return d

    cand = [o for o in objs if sim.getObjectType(o) != sim.object_joint_type]
    if not cand:
        return None
    cand.sort(key=depth, reverse=True)
    return cand[0]


tip = find_end_effector(robot)
print(f"\nEfetuador final escolhido: {get_alias(tip) if tip else 'nenhum'}")

# ============================================================
# Coloca o robô na configuração de demonstração
# ============================================================

for j, ang in zip(joints, Q_DEMO):
    sim.setJointPosition(j, float(ang))

# ============================================================
# Lê as poses (matriz homogênea) de cada frame em relação ao mundo
# ============================================================

frames = []  # (rótulo, R, p)

# Frame da base
R0, p0 = matrix_to_Rp(sim.getObjectMatrix(robot, sim.handle_world))
frames.append(("base", R0, p0))

# Frames das juntas
for i, j in enumerate(joints, start=1):
    R, p = matrix_to_Rp(sim.getObjectMatrix(j, sim.handle_world))
    frames.append((f"j{i}", R, p))
    # também imprime posição + quatérnio (útil para conferência)
    quat = sim.getObjectQuaternion(j, sim.handle_world)  # [x, y, z, w]
    print(f"\nFrame j{i}: pos={np.round(p,4).tolist()}  quat(xyzw)={np.round(quat,4).tolist()}")

# Frame do efetuador
if tip is not None:
    Re, pe = matrix_to_Rp(sim.getObjectMatrix(tip, sim.handle_world))
    frames.append(("EE", Re, pe))
    quat = sim.getObjectQuaternion(tip, sim.handle_world)
    print(f"\nFrame EE: pos={np.round(pe,4).tolist()}  quat(xyzw)={np.round(quat,4).tolist()}")

# ============================================================
# Desenha a figura 3D com os triedros de cada frame
# ============================================================

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Linhas conectando as origens dos frames (corpo do robô)
origins = np.array([p for (_, _, p) in frames])
ax.plot(origins[:, 0], origins[:, 1], origins[:, 2],
        "-o", color="0.4", linewidth=1.5, markersize=4, zorder=1)

# Triedros: x = vermelho, y = verde, z = azul
axis_colors = ["r", "g", "b"]
for label, R, p in frames:
    for k in range(3):
        d = R[:, k] * ARROW_LEN
        ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2],
                  color=axis_colors[k], linewidth=2, arrow_length_ratio=0.25)
    ax.text(p[0], p[1], p[2] + 0.02, label, fontsize=9, weight="bold")

# Aparência
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Posicionamento dos frames do robô Franka")

# Eixos com mesma escala (aspecto cúbico)
pts = origins
c = pts.mean(axis=0)
r = max(np.ptp(pts, axis=0).max(), 0.5) / 2 + ARROW_LEN
ax.set_xlim(c[0] - r, c[0] + r)
ax.set_ylim(c[1] - r, c[1] + r)
ax.set_zlim(c[2] - r, c[2] + r)
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    pass
ax.view_init(elev=20, azim=45)

# Legenda dos eixos
from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], color="r", lw=2, label="eixo x"),
    Line2D([0], [0], color="g", lw=2, label="eixo y"),
    Line2D([0], [0], color="b", lw=2, label="eixo z"),
]
ax.legend(handles=legend, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
print(f"\nFigura salva em: {OUT_FILE}")

# Mostra a janela (feche para encerrar). Comente se rodar sem interface.
plt.show()

print("\nFim.")

"""MÃ³dulo de CinemÃ¡tica Direta para o robÃ´ Franka (DH Modificado de Craig).

Calcula a matriz homogÃªnea global, posiÃ§Ã£o e orientaÃ§Ã£o em quatÃ©rnios (x, y, z, w).
"""

import numpy as np
from typing import Tuple


def matriz_para_quaternio(R: np.ndarray) -> np.ndarray:
    """Converte matriz de rotacao para quaternio (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    return np.array([qx, qy, qz, qw])


def matriz_coppelia_para_homogenea(matriz_3x4) -> np.ndarray:
    """Converte a matriz 3x4 do CoppeliaSim em matriz homogenea 4x4."""
    M = np.array(matriz_3x4, dtype=float).reshape(3, 4)
    T = np.identity(4, dtype=float)
    T[0:3, 0:3] = M[:, 0:3]
    T[0:3, 3] = M[:, 3]
    return T


def erro_angular_quaternio(q1: np.ndarray, q2: np.ndarray) -> float:
    """Retorna o erro angular entre quaternios, em radianos."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return 2 * np.arccos(dot)


def calcular_jacobiana_posicao(q: np.ndarray, passo: float = 1e-6) -> np.ndarray:
    """Calcula numericamente a Jacobiana de posicao do efetuador.

    Retorna uma matriz 3x7. Cada coluna mostra como a posicao XYZ do
    efetuador varia quando uma junta do Franka sofre uma pequena perturbacao.
    A derivada e estimada por diferenca central.
    """
    q = np.asarray(q, dtype=float)
    if q.size < 7:
        raise ValueError("`q` deve conter ao menos 7 elementos (juntas do Franka)")
    if passo <= 0:
        raise ValueError("`passo` deve ser positivo")

    q = q[:7].copy()
    J = np.zeros((3, 7), dtype=float)

    for i in range(7):
        q_mais = q.copy()
        q_menos = q.copy()
        q_mais[i] += passo
        q_menos[i] -= passo

        _, p_mais, _ = calcular_cinematica_direta(q_mais)
        _, p_menos, _ = calcular_cinematica_direta(q_menos)
        J[:, i] = (p_mais - p_menos) / (2.0 * passo)

    return J


def calcular_jacobiana_geometrica(q: np.ndarray, passo: float = 1e-6) -> np.ndarray:
    """Calcula numericamente a Jacobiana geometrica 6x7.

    As tres primeiras linhas correspondem a velocidade linear do efetuador.
    As tres ultimas aproximam a velocidade angular a partir da derivada da
    matriz de rotacao. A Jacobiana e expressa no frame DH da base.
    """
    q = np.asarray(q, dtype=float)
    if q.size < 7:
        raise ValueError("`q` deve conter ao menos 7 elementos (juntas do Franka)")
    if passo <= 0:
        raise ValueError("`passo` deve ser positivo")

    q = q[:7].copy()
    T0, _, _ = calcular_cinematica_direta(q)
    R0 = T0[0:3, 0:3]

    J = np.zeros((6, 7), dtype=float)
    J[0:3, :] = calcular_jacobiana_posicao(q, passo=passo)

    for i in range(7):
        q_mais = q.copy()
        q_menos = q.copy()
        q_mais[i] += passo
        q_menos[i] -= passo

        T_mais, _, _ = calcular_cinematica_direta(q_mais)
        T_menos, _, _ = calcular_cinematica_direta(q_menos)
        R_mais = T_mais[0:3, 0:3]
        R_menos = T_menos[0:3, 0:3]

        R_ponto = (R_mais - R_menos) / (2.0 * passo)
        omega_chapeu = R_ponto @ R0.T
        omega_chapeu = 0.5 * (omega_chapeu - omega_chapeu.T)
        J[3:6, i] = np.array([
            omega_chapeu[2, 1],
            omega_chapeu[0, 2],
            omega_chapeu[1, 0],
        ])

    return J

def calcular_cinematica_direta(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula a cinemÃ¡tica direta do Franka com base na tabela DH Modificada.

    ParÃ¢metros:
        q: Array-like com pelo menos 7 elementos (Ã¢ngulos das juntas em radianos).
    """
    q = np.asarray(q, dtype=float)
    if q.size < 7:
        raise ValueError("`q` deve conter ao menos 7 elementos (juntas do Franka)")

    # Tabela de DH Modificado: [a_{i-1}, d_i, alpha_{i-1}]
    dh_params = [
        [0,       0.333,  0],           # Junta 1
        [0,       0,     -np.pi/2],     # Junta 2
        [0,       0.316,  np.pi/2],      # Junta 3
        [0.0825,  0,      np.pi/2],      # Junta 4
        [-0.0825, 0.384, -np.pi/2],     # Junta 5
        [0,       0,      np.pi/2],      # Junta 6
        [0.088,   0,      np.pi/2],      # Junta 7
        [0,       0.107,  0]            # Flange (Efetor final)
    ]
    
    T_global = np.identity(4, dtype=float)
    
    for i in range(8):
        a_prev, d_i, alpha_prev = dh_params[i]
        theta_i = q[i] if i < 7 else 0.0
        
        ct, st = np.cos(theta_i), np.sin(theta_i)
        ca, sa = np.cos(alpha_prev), np.sin(alpha_prev)
        
        # EquaÃ§Ã£o (1) - ConvenÃ§Ã£o de Craig
        T_local = np.array([
            [ct,        -st,         0,   a_prev],
            [st*ca,      ct*ca,    -sa,  -d_i*sa],
            [st*sa,      ct*sa,     ca,   d_i*ca],
            [0,          0,          0,   1]
        ])
        
        T_global = np.dot(T_global, T_local)
        
    posicao = T_global[0:3, 3]
    R = T_global[0:3, 0:3]
    
    # ConversÃ£o de Matriz de RotaÃ§Ã£o para QuatÃ©rnio (x, y, z, w)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    quaternio = np.array([qx, qy, qz, qw])
    return T_global, posicao, quaternio


if __name__ == "__main__":
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        
        client = RemoteAPIClient(host="127.0.0.1", port=23000)
        sim = client.require("sim")
        print("[ValidaÃ§Ã£o] Conectado ao CoppeliaSim!")
        
        robot = sim.getObject("/Franka")
        all_joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)
        joints = all_joints[:7]
        
        # IntegraÃ§Ã£o da lÃ³gica robusta do Manoel para encontrar a ponta sem falhas de nome
        objs = sim.getObjectsInTree(robot, sim.handle_all, 0)
        cand = [o for o in objs if sim.getObjectType(o) != sim.object_joint_type]
        
        # FunÃ§Ã£o auxiliar para mapear a profundidade de cada objeto candidato
        def obter_profundidade(o):
            prof, atual = 0, o
            while True:
                pai = sim.getObjectParent(atual)
                if pai == -1:
                    break
                prof += 1
                atual = pai
            return prof

        try:
            tip = sim.getObject("/Franka/connection")
        except Exception:
            cand.sort(key=obter_profundidade, reverse=True)
            tip = cand[0]
        
        try:
            alias_ponta = sim.getObjectAlias(tip, 1)
            print(f"Validando contra o efetuador final detectado: {alias_ponta}")
        except Exception:
            print("Validando contra o efetuador final localizado na Ã¡rvore.")
        
        q_atual = np.array([sim.getJointPosition(j) for j in joints])
        print(f"\n---> q_teste = [ {', '.join(f'{val:.4f}' for val in q_atual)} ]^T\n")

        T_base_calc, _, _ = calcular_cinematica_direta(q_atual)

        # A FK por DH sai no referencial da base da cadeia. No modelo do
        # CoppeliaSim, esse frame coincide com a primeira junta, nao com o
        # objeto raiz /Franka.
        T_base_mundo = matriz_coppelia_para_homogenea(
            sim.getObjectMatrix(joints[0], sim.handle_world)
        )
        T_mundo_calc = T_base_mundo @ T_base_calc
        pos_calc = T_mundo_calc[0:3, 3]
        quat_calc = matriz_para_quaternio(T_mundo_calc[0:3, 0:3])
        
        pos_sim = np.array(sim.getObjectPosition(tip, sim.handle_world))
        quat_sim = np.array(sim.getObjectQuaternion(tip, sim.handle_world))
        erro_posicao = np.linalg.norm(pos_calc - pos_sim)
        erro_orientacao = erro_angular_quaternio(quat_calc, quat_sim)
        
        print("\n================== RESULTADO DA VALIDAÃ‡ÃƒO ==================")
        print(f"Sua PosiÃ§Ã£o XYZ:  {np.round(pos_calc, 4)}")
        print(f"Simulador XYZ:    {np.round(pos_sim, 4)}")
        print(f"Seu QuatÃ©rnio:    {np.round(quat_calc, 4)}")
        print(f"Simu QuatÃ©rnio:   {np.round(quat_sim, 4)}")
        print(f"Erro posicao:     {erro_posicao:.6f} m")
        print(f"Erro orientacao:  {erro_orientacao:.6f} rad")
        print("------------------------------------------------------------")
        
        if erro_posicao < 1e-3 and erro_orientacao < 1e-3:
            print(f"SUCESSO! Erro de precisÃ£o: {erro_posicao:.6f} metros.")
        else:
            print("ATENÃ‡ÃƒO: Houve divergÃªncia. Verifique se a convenÃ§Ã£o DH estÃ¡ idÃªntica.")
        print("============================================================\n")
            
    except ModuleNotFoundError:
        print("\n[Aviso] Cliente CoppeliaSim nÃ£o encontrado. FunÃ§Ã£o pronta para importaÃ§Ã£o.\n")
    except Exception as e:
        print(f"\nErro na validaÃ§Ã£o: {e}\n")

"""Módulo de Cinemática Direta para o robô Franka (DH Modificado de Craig).

Calcula a matriz homogênea global, posição e orientação em quatérnios (x, y, z, w).
"""

import numpy as np
from typing import Tuple


def calcular_cinematica_direta(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula a cinemática direta do Franka com base na tabela DH Modificada.

    Parâmetros:
        q: Array-like com pelo menos 7 elementos (ângulos das juntas em radianos).
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
        
        # Equação (1) - Convenção de Craig
        T_local = np.array([
            [ct,        -st,         0,   a_prev],
            [st*ca,      ct*ca,    -sa,  -d_i*sa],
            [st*sa,      ct*sa,     ca,   d_i*ca],
            [0,          0,          0,   1]
        ])
        
        T_global = np.dot(T_global, T_local)
        
    posicao = T_global[0:3, 3]
    R = T_global[0:3, 0:3]
    
    # Conversão de Matriz de Rotação para Quatérnio (x, y, z, w)
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
        print("🤖 [Validação] Conectado ao CoppeliaSim!")
        
        robot = sim.getObject("/Franka")
        all_joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)
        joints = all_joints[:7]
        
        # Integração da lógica robusta do Manoel para encontrar a ponta sem falhas de nome
        objs = sim.getObjectsInTree(robot, sim.handle_all, 0)
        cand = [o for o in objs if sim.getObjectType(o) != sim.object_joint_type]
        
        # Função auxiliar para mapear a profundidade de cada objeto candidato
        def obter_profundidade(o):
            prof, atual = 0, o
            while True:
                pai = sim.getObjectParent(atual)
                if pai == -1:
                    break
                prof += 1
                atual = pai
            return prof

        cand.sort(key=obter_profundidade, reverse=True)
        tip = cand[0] if cand else sim.getObject("/Franka/connection")
        
        try:
            alias_ponta = sim.getObjectAlias(tip, 1)
            print(f"📍 Validando contra o efetuador final detectado: {alias_ponta}")
        except Exception:
            print("📍 Validando contra o efetuador final localizado na árvore.")
        
        q_atual = np.array([sim.getJointPosition(j) for j in joints])
        _, pos_calc, quat_calc = calcular_cinematica_direta(q_atual)
        
        pos_sim = np.array(sim.getObjectPosition(tip, sim.handle_world))
        quat_sim = np.array(sim.getObjectQuaternion(tip, sim.handle_world))
        erro_posicao = np.linalg.norm(pos_calc - pos_sim)
        
        print("\n================== RESULTADO DA VALIDAÇÃO ==================")
        print(f"Sua Posição XYZ:  {np.round(pos_calc, 4)}")
        print(f"Simulador XYZ:    {np.round(pos_sim, 4)}")
        print(f"Seu Quatérnio:    {np.round(quat_calc, 4)}")
        print(f"Simu Quatérnio:   {np.round(quat_sim, 4)}")
        print("------------------------------------------------------------")
        
        if erro_posicao < 1e-3:
            print(f"✅ SUCESSO! Erro de precisão: {erro_posicao:.6f} metros.")
        else:
            print("⚠️ ATENÇÃO: Houve divergência. Verifique se a convenção DH está idêntica.")
        print("============================================================\n")
            
    except ModuleNotFoundError:
        print("\n💡 [Aviso] Cliente CoppeliaSim não encontrado. Função pronta para importação.\n")
    except Exception as e:
        print(f"\n❌ Erro na validação: {e}\n")
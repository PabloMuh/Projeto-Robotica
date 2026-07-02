"""
cinematica_inversa.py
=====================
Modulo de cinematica inversa para o robo Franka Emika Panda.

A IK e resolvida como um problema de otimizacao nao-linear com limites
articulares. A funcao de custo prioriza o erro cartesiano de posicao e usa
uma pequena penalizacao de postura apenas para escolher uma solucao mais
regular entre as infinitas configuracoes possiveis do robo redundante.
"""

import time

import numpy as np
from scipy.optimize import least_squares

from cinematica_franka import calcular_cinematica_direta, calcular_jacobiana_posicao, matriz_coppelia_para_homogenea


# Limites articulares oficiais do Franka Emika Panda, em radianos.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0.4363, -3.0718])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 4.6251, 3.0718])

# Postura de referencia usada apenas como regularizacao leve.
Q_REF = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.0])

# A posicao deve dominar a otimizacao; a postura so desempata solucoes.
W_POSTURE = np.array([0.002, 0.002, 0.002, 0.003, 0.004, 0.004, 0.004])
ESCALA_POSICAO = 100.0


def _ik_error(q, target_pos_base):
    """Residuos minimizados pelo least_squares.

    target_pos_base deve estar no mesmo referencial da FK, isto e, no frame DH
    da base da cadeia cinematica.
    """
    _, pos_calc, _ = calcular_cinematica_direta(q)
    erro_posicao = ESCALA_POSICAO * (pos_calc - target_pos_base)
    erro_postura = W_POSTURE * (q - Q_REF)
    return np.concatenate([erro_posicao, erro_postura])



def _ik_jacobian(q, target_pos_base):
    """Jacobiana dos residuos usados na IK.

    As tres primeiras linhas correspondem ao erro cartesiano de posicao.
    As sete linhas restantes correspondem a regularizacao de postura.
    """
    del target_pos_base
    J_pos = ESCALA_POSICAO * calcular_jacobiana_posicao(q)
    J_postura = np.diag(W_POSTURE)
    return np.vstack([J_pos, J_postura])

def _sementes_iniciais(q_inicial):
    """Monta tentativas deterministicas para evitar minimos locais ruins."""
    sementes = []
    if q_inicial is not None:
        sementes.append(np.asarray(q_inicial, dtype=float))

    sementes.extend([
        Q_REF,
        np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]),
        np.array([0.0, -1.5, 0.0, -1.5, 0.0, 1.5, 0.0]),
        np.array([0.0, 1.2, 0.0, -1.0, 0.0, 2.4, 0.0]),
        np.array([1.5, -1.4, -1.0, -1.2, 0.0, 1.5, 0.0]),
        np.array([-1.5, -1.4, 1.0, -1.2, 0.0, 1.5, 0.0]),
    ])

    unicas = []
    for semente in sementes:
        semente = np.clip(semente, Q_MIN, Q_MAX)
        if not any(np.allclose(semente, outra) for outra in unicas):
            unicas.append(semente)
    return unicas


def resolver_ik(target_pos_base, q_inicial=None, max_iter=1500, tolerancia=1e-8):
    """Resolve IK de posicao no referencial DH da base.

    Retorna:
        q_solucao: vetor (7,) com os angulos das juntas em radianos.
        sucesso: True quando o erro cartesiano final e menor que 1 mm.
        erro_metros: norma do erro final em metros.
    """
    melhor_q = None
    melhor_erro = np.inf

    for semente in _sementes_iniciais(q_inicial):
        resultado = least_squares(
            fun=_ik_error,
            x0=semente,
            args=(target_pos_base,),
            jac=_ik_jacobian,
            bounds=(Q_MIN, Q_MAX),
            max_nfev=max_iter,
            xtol=tolerancia,
            ftol=tolerancia,
            gtol=tolerancia,
            method="trf",
        )

        q_candidato = resultado.x
        _, pos_final, _ = calcular_cinematica_direta(q_candidato)
        erro = float(np.linalg.norm(pos_final - target_pos_base))

        if erro < melhor_erro:
            melhor_q = q_candidato
            melhor_erro = erro

        if erro < 1e-3:
            break

    sucesso = melhor_erro < 1e-3
    return melhor_q, sucesso, melhor_erro


def encontrar_efetuador_final(sim, robot):
    """Prioriza o frame /Franka/connection, usado tambem na validacao da FK."""
    try:
        return sim.getObject("/Franka/connection")
    except Exception:
        pass

    objs = sim.getObjectsInTree(robot, sim.handle_all, 0)
    candidatos = [o for o in objs if sim.getObjectType(o) != sim.object_joint_type]

    def profundidade(obj):
        d, atual = 0, obj
        while True:
            pai = sim.getObjectParent(atual)
            if pai == -1:
                break
            d += 1
            atual = pai
        return d

    candidatos.sort(key=profundidade, reverse=True)
    return candidatos[0]


def mundo_para_base(pos_mundo, T_base_mundo):
    """Converte uma posicao do mundo do CoppeliaSim para o frame DH da base."""
    pos_base_h = np.linalg.inv(T_base_mundo) @ np.r_[pos_mundo, 1.0]
    return pos_base_h[0:3]


if __name__ == "__main__":
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient

        client = RemoteAPIClient(host="127.0.0.1", port=23000)
        sim = client.require("sim")
        print("Conectado ao CoppeliaSim.")

        robot = sim.getObject("/Franka")
        joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)[:7]
        tip = encontrar_efetuador_final(sim, robot)
        q_atual = np.array([sim.getJointPosition(j) for j in joints])

        T_base_mundo = matriz_coppelia_para_homogenea(
            sim.getObjectMatrix(joints[0], sim.handle_world)
        )

        try:
            cubo = sim.getObject("/Cuboid")
            target_pos_world = np.array(sim.getObjectPosition(cubo, sim.handle_world))
            print(f"Alvo lido do CoppeliaSim (Cuboid): {np.round(target_pos_world, 4)}")
        except Exception:
            target_pos_world = np.array([1.0, -0.05, 0.5])
            print(f"Cuboid nao encontrado. Usando alvo fixo no mundo: {target_pos_world}")

        target_pos_base = mundo_para_base(target_pos_world, T_base_mundo)
        print(f"Alvo convertido para a base DH: {np.round(target_pos_base, 4)}")

        print("Resolvendo cinematica inversa...")
        q_sol, sucesso, erro = resolver_ik(target_pos_base, q_inicial=q_atual)

        print("=================== RESULTADO DA CI ===================")
        print(f"Configuracao solucao (rad): {np.round(q_sol, 4)}")
        print(f"Erro de posicao final     : {erro:.6f} m")
        print(f"Convergiu (<1 mm)         : {'SIM' if sucesso else 'NAO'}")
        print("=======================================================")

        if sucesso:
            print("Movendo robo para a solucao...")
            sim.startSimulation()
            time.sleep(0.5)

            steps = 200
            for i in range(steps):
                alpha = i / (steps - 1)
                q_interp = (1 - alpha) * q_atual + alpha * q_sol
                for joint, angle in zip(joints, q_interp):
                    sim.setJointTargetPosition(joint, float(angle))
                time.sleep(0.025)

            time.sleep(1.0)
            pos_final_sim = np.array(sim.getObjectPosition(tip, sim.handle_world))
            erro_sim = float(np.linalg.norm(pos_final_sim - target_pos_world))
            print(f"Posicao final do efetuador: {np.round(pos_final_sim, 4)}")
            print(f"Alvo                      : {np.round(target_pos_world, 4)}")
            print(f"Erro no CoppeliaSim       : {erro_sim:.6f} m")
            sim.stopSimulation()
        else:
            print("Solucao nao convergiu. O alvo pode estar fora do espaco de trabalho.")
            print("Tente posicionar o Cuboid um pouco mais alto ou mais perto da base.")

    except ModuleNotFoundError:
        print("Cliente CoppeliaSim nao encontrado. Executando teste offline.")

        q_teste = np.array([0.3, -0.4, 0.1, -1.8, 0.2, 1.5, 0.4])
        _, pos_alvo, _ = calcular_cinematica_direta(q_teste)
        print(f"Posicao alvo (via FK):  {np.round(pos_alvo, 4)}")

        q_sol, sucesso, erro = resolver_ik(pos_alvo)
        print(f"Solucao encontrada:    {np.round(q_sol, 4)}")
        print(f"Erro de posicao:       {erro:.6f} m")
        print(f"Convergiu (<1 mm):     {'SIM' if sucesso else 'NAO'}")

    except Exception as e:
        print(f"Erro: {e}")

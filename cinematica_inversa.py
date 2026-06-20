"""
cinematica_inversa.py
=====================
Módulo de Cinemática Inversa para o robô Franka Emika Panda.

Abordagem: Problema de otimização não-linear com restrições de caixa,
resolvido via scipy.optimize.least_squares (algoritmo Trust Region
Reflective). A função de custo minimiza simultaneamente:
  (1) o erro de posição entre o efetuador e o alvo, e
  (2) um erro de postura ponderado em relação a uma configuração de
      referência segura, evitando singularidades e posturas instáveis
      no punho do Franka.

Complexidade: O(k * n), onde k é o número de iterações do solver e
n = 7 (número de juntas). O número de iterações é limitado por
max_nfev, tornando o tempo de execução praticamente constante na prática.

Uso como módulo:
    from cinematica_inversa import resolver_ik
    q_sol, sucesso, erro = resolver_ik(target_pos, q_inicial)

Uso direto (validação com CoppeliaSim):
    python cinematica_inversa.py
"""

import numpy as np
from scipy.optimize import least_squares
from cinematica_franka import calcular_cinematica_direta

# ============================================================
# Constantes do Franka Emika Panda
# ============================================================

# Limites articulares (rad) — especificação oficial do fabricante
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.4363, -3.0718])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  4.6251,  3.0718])

# Postura de referência: configuração segura e natural do Franka
Q_REF = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.0])

# Pesos de penalização de postura por junta.
# Pesos maiores nas juntas do punho (q5, q6, q7) estabilizam
# a orientação do efetuador final.
W_POSTURE = np.array([0.03,   # q1 — base
                      0.05,   # q2 — ombro
                      0.03,   # q3
                      0.08,   # q4
                      0.40,   # q5 — punho
                      0.70,   # q6
                      0.60])  # q7

# Escala do erro de posição em relação ao erro de postura.
ESCALA_POSICAO = 5.0


# ============================================================
# Função de erro (custo) para o otimizador
# ============================================================

def _ik_error(q, target_pos):
    """Vetor de resíduos minimizado pelo least_squares.

    Especificação matemática:
        r(q) = [ ESCALA_POSICAO * (p_FK(q) - p_alvo),
                 W_POSTURE * (q - Q_REF)            ]

    onde p_FK(q) é a posição do efetuador pela cinemática direta.
    O solver minimiza ||r(q)||^2.
    """
    _, pos_calc, _ = calcular_cinematica_direta(q)
    erro_posicao = ESCALA_POSICAO * (pos_calc - target_pos)
    erro_postura = W_POSTURE * (q - Q_REF)
    return np.concatenate([erro_posicao, erro_postura])


# ============================================================
# Função principal de resolução da CI
# ============================================================

def resolver_ik(target_pos, q_inicial=None, max_iter=500, tolerancia=1e-5):
    """Resolve a cinemática inversa para uma posição alvo.

    Parâmetros
    ----------
    target_pos : array (3,)
        Posição desejada do efetuador [x, y, z] em metros (referencial mundo).
    q_inicial : array (7,), opcional
        Configuração inicial das juntas em radianos. Padrão: Q_REF.
    max_iter : int
        Número máximo de avaliações da função de custo.
    tolerancia : float
        Tolerância de convergência (xtol, ftol, gtol).

    Retorno
    -------
    q_solucao : np.ndarray (7,)
        Ângulos das juntas que aproximam o efetuador de target_pos.
    sucesso : bool
        True se o erro de posição final for menor que 1 mm.
    erro_metros : float
        Norma do erro de posição residual em metros.
    """
    if q_inicial is None:
        q_inicial = Q_REF.copy()

    resultado = least_squares(
        fun=_ik_error,
        x0=q_inicial,
        args=(target_pos,),
        bounds=(Q_MIN, Q_MAX),
        max_nfev=max_iter,
        xtol=tolerancia,
        ftol=tolerancia,
        gtol=tolerancia,
        method="trf",  # Trust Region Reflective — suporta restrições de caixa
    )

    q_solucao = resultado.x
    _, pos_final, _ = calcular_cinematica_direta(q_solucao)
    erro_metros = float(np.linalg.norm(pos_final - target_pos))
    sucesso = erro_metros < 1e-3

    return q_solucao, sucesso, erro_metros


# ============================================================
# Bloco de validação com CoppeliaSim
# ============================================================

if __name__ == "__main__":
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        import time

        client = RemoteAPIClient(host="127.0.0.1", port=23000)
        sim = client.require("sim")
        print("Conectado ao CoppeliaSim.")

        robot = sim.getObject("/Franka")
        all_joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)
        joints = all_joints[:7]

        # Localiza efetuador final pelo objeto não-junta mais profundo na árvore
        objs = sim.getObjectsInTree(robot, sim.handle_all, 0)
        cand = [o for o in objs if sim.getObjectType(o) != sim.object_joint_type]

        def profundidade(o):
            d, cur = 0, o
            while True:
                pai = sim.getObjectParent(cur)
                if pai == -1:
                    break
                d += 1
                cur = pai
            return d

        cand.sort(key=profundidade, reverse=True)
        tip = cand[0]

        q_atual = np.array([sim.getJointPosition(j) for j in joints])

        # Tenta usar o Cuboid como alvo; fallback para ponto fixo
        try:
            cubo = sim.getObject("/Cuboid")
            target_pos = np.array(sim.getObjectPosition(cubo, sim.handle_world))
            print(f"Alvo lido do CoppeliaSim (Cuboid): {np.round(target_pos, 4)}")
        except Exception:
            target_pos = np.array([0.4, 0.0, 0.4])
            print(f"Cuboid nao encontrado. Usando alvo fixo: {target_pos}")

        print("Resolvendo cinematica inversa...")
        q_sol, sucesso, erro = resolver_ik(target_pos, q_inicial=q_atual)

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
            print(f"Posicao final do efetuador: {np.round(pos_final_sim, 4)}")
            print(f"Alvo                      : {np.round(target_pos, 4)}")
            sim.stopSimulation()
        else:
            print("Solucao nao convergiu. Verifique se o alvo esta dentro do espaco de trabalho.")

    except ModuleNotFoundError:
        print("Cliente CoppeliaSim nao encontrado. Executando teste offline.")

        # Teste offline: verifica que a CI recupera uma configuracao conhecida
        q_teste = np.array([0.3, -0.4, 0.1, -1.8, 0.2, 1.5, 0.4])
        _, pos_alvo, _ = calcular_cinematica_direta(q_teste)
        print(f"Posicao alvo (via FK):  {np.round(pos_alvo, 4)}")

        q_sol, sucesso, erro = resolver_ik(pos_alvo)
        print(f"Solucao encontrada:    {np.round(q_sol, 4)}")
        print(f"Erro de posicao:       {erro:.6f} m")
        print(f"Convergiu (<1 mm):     {'SIM' if sucesso else 'NAO'}")

    except Exception as e:
        print(f"Erro: {e}")

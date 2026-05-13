import numpy as np

class WhaleOptimizationAlgorithm:
    """
    Whale Optimization Algorithm (WOA)
    Implementação modular para otimização de funções de custo contínuas.
    """
    
    def __init__(self, obj_function, bounds, num_whales=30, max_iter=100, minimize=True):
        """
        Inicializa o otimizador WOA.
        
        :param obj_function: Função de custo a ser otimizada. Deve receber um array 1D e retornar um float.
        :param bounds: Lista de tuplas (min, max) definindo os limites para cada dimensão.
        :param num_whales: Tamanho da população (número de agentes de busca).
        :param max_iter: Número máximo de iterações.
        :param minimize: Booleano indicando se o objetivo é minimizar (True) ou maximizar (False).
        """
        self.obj_function = obj_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.num_whales = num_whales
        self.max_iter = max_iter
        self.minimize = minimize
        
        # Estado do melhor agente encontrado
        self.best_whale_pos = np.zeros(self.dim)
        self.best_whale_score = float('inf') if minimize else float('-inf')
        
        # Histórico de convergência para plotagem futura
        self.convergence_curve = np.zeros(self.max_iter)

    def _initialize_population(self):
        """Gera a população inicial aleatoriamente dentro dos limites."""
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        # Cria matriz de posições (num_whales x dimensões)
        return lb + np.random.rand(self.num_whales, self.dim) * (ub - lb)

    def _check_bounds(self, position):
        """Garante que a baleia não saia do espaço de busca permitido."""
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        return np.clip(position, lb, ub)

    def optimize(self):
        """
        Executa o loop principal do Algoritmo de Otimização das Baleias.
        
        :return: best_whale_pos (array), best_whale_score (float), convergence_curve (array)
        """
        # 1. Inicializa as posições
        whales_pos = self._initialize_population()
        
        for t in range(self.max_iter):
            
            # 2. Avaliação de Fitness
            for i in range(self.num_whales):
                whales_pos[i, :] = self._check_bounds(whales_pos[i, :])
                
                # AVALIAÇÃO: Aqui o seu script chamará o TMM!
                fitness = self.obj_function(whales_pos[i, :])
                
                # Atualiza a "Melhor Baleia" (Líder)
                if (self.minimize and fitness < self.best_whale_score) or \
                   (not self.minimize and fitness > self.best_whale_score):
                    self.best_whale_score = fitness
                    self.best_whale_pos = whales_pos[i, :].copy()
            
            # 3. Atualização do parâmetro 'a' (decai linearmente de 2 para 0)
            a = 2.0 - t * (2.0 / self.max_iter)
            
            # 4. Atualização das Posições (Movimentação)
            for i in range(self.num_whales):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                
                # Parâmetro de probabilidade para escolher a estratégia de caça
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # EXPLOTATION: Cerco à presa (encircling prey)
                        D = abs(C * self.best_whale_pos - whales_pos[i, :])
                        whales_pos[i, :] = self.best_whale_pos - A * D
                    else:
                        # EXPLORATION: Busca aleatória (search for prey)
                        rand_leader_idx = np.random.randint(0, self.num_whales)
                        rand_leader_pos = whales_pos[rand_leader_idx, :]
                        D = abs(C * rand_leader_pos - whales_pos[i, :])
                        whales_pos[i, :] = rand_leader_pos - A * D
                else:
                    # EXPLOTATION: Movimento em espiral (bubble-net attacking)
                    D_prime = abs(self.best_whale_pos - whales_pos[i, :])
                    l = np.random.uniform(-1, 1) # Parâmetro de forma da espiral
                    b = 1.0 # Constante logarítmica da espiral
                    
                    whales_pos[i, :] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_whale_pos
            
            # Registra o progresso
            self.convergence_curve[t] = self.best_whale_score
            
            # Print de progresso (opcional, bom para acompanhar simulações longas)
            print(f"Iteração {t+1}/{self.max_iter} - Melhor MSE: {self.best_whale_score:.6e}")
            
        return self.best_whale_pos, self.best_whale_score, self.convergence_curve

# =====================================================================
# EXEMPLO DE USO PARA O SEU FUTURO main.py
# =====================================================================
if __name__ == "__main__":
    # Esta função de teste será substituída pela sua cachoeira do TMM
    def dummy_fitness(genes):
        # Exemplo: Esfera de Rastrigin (queremos chegar em 0)
        return np.sum(genes**2)

    # Definindo limites: Ex: W, Wc, Lambda, DC, H, alpha, beta, S
    # Aqui coloquei limites genéricos só para teste
    limites = [
        (0.5e-6, 1.2e-6),  # W
        (0.3e-6, 0.6e-6),  # Wc
        (0.2e-6, 0.4e-6),  # Lambda
        (0.1, 0.9)         # DC
    ]

    otimizador = WhaleOptimizationAlgorithm(
        obj_function=dummy_fitness,
        bounds=limites,
        num_whales=20,
        max_iter=50,
        minimize=True
    )

    melhor_baleia, melhor_mse, curva = otimizador.optimize()
    
    print("\nOtimização Concluída!")
    print(f"Melhores parâmetros encontrados: {melhor_baleia}")
import numpy as np
from scipy.stats import poisson
from typing import Dict
from config import log_info

class ProbabilityEngine:
    """
    Motor de cálculo de probabilidades basado en Poisson bivariado independiente.
    
    Modelos:
    1. Poisson 1X2: Calcula P(Home Win), P(Draw), P(Away Win)
    2. Poisson Over/Under: Calcula P(Over N.5), P(Under N.5)
    
    Entrada: Tasas lambda (xG predichos por modelo ML)
    Salida: Probabilidades normalizadas [0, 1]
    """

    @staticmethod
    def calculate_1x2(
        home_xg: float,
        away_xg: float,
        max_goals: int = 8
    ) -> Dict[str, float]:
        """
        Calcula probabilidades de Match Odds (1X2) mediante Poisson bivariado.
        
        Modelo:
        - P(i, j) = Poisson(i | home_xg) * Poisson(j | away_xg)
        - P(Home) = suma de P(i, j) donde i > j
        - P(Draw) = suma de P(i, j) donde i == j
        - P(Away) = suma de P(i, j) donde i < j
        
        Args:
            home_xg: Expected Goals del equipo local
            away_xg: Expected Goals del equipo visitante
            max_goals: Máximo de goles a considerar (default 8)
        
        Returns:
            Dict con claves: home, draw, away (valores en [0, 1], suman a 1.0)
        """
        # Construye matriz de probabilidades bivariadas
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
        
        # Calcula probabilidades por evento
        # Lower triangular (i > j) = Home wins
        p_home = float(np.sum(np.tril(matrix, -1)))
        # Diagonal (i == j) = Draws
        p_draw = float(np.sum(np.diag(matrix)))
        # Upper triangular (i < j) = Away wins
        p_away = float(np.sum(np.triu(matrix, 1)))
        
        # Normaliza (por seguridad numérica)
        total = p_home + p_draw + p_away
        if total <= 0:
            # Fallback: distribución uniforme
            return {"home": 0.333, "draw": 0.333, "away": 0.333}
        
        return {
            "home": round(p_home / total, 4),
            "draw": round(p_draw / total, 4),
            "away": round(p_away / total, 4)
        }

    @staticmethod
    def calculate_over_under(
        rate: float,
        line: float = 2.5
    ) -> Dict[str, float]:
        """
        Calcula probabilidades Over/Under usando CDF de Poisson.
        
        Modelo:
        - P(Under) = CDF(floor(line), rate)  [P(X <= k)]
        - P(Over) = 1 - P(Under)
        
        Args:
            rate: Tasa lambda (xG predicho)
            line: Línea de goles (default 2.5)
        
        Returns:
            Dict con claves: over, under (valores en [0, 1], suman a 1.0)
        """
        k = int(np.floor(line))
        # CDF: P(X <= k)
        under = float(poisson.cdf(k, rate))
        # Complemento: P(X > k) = 1 - P(X <= k)
        over = max(0.0, 1.0 - under)
        
        # Normaliza
        total = over + under
        if total <= 0:
            return {"over": 0.5, "under": 0.5}
        
        return {
            "over": round(over / total, 4),
            "under": round(under / total, 4)
        }

import numpy as np
from scipy.stats import poisson
from typing import Dict

class ProbabilityEngine:
    @staticmethod
    def calculate_1x2_probabilities(home_xg: float, away_xg: float, max_goals: int = 8) -> Dict[str, float]:
        """Calcula probabilidades exactas del resultado 1X2 usando Poisson bivariado."""
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                
        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.sum(np.diag(matrix)))
        p_away = float(np.sum(np.triu(matrix, 1)))
        total = p_home + p_draw + p_away
        
        return {
            "home": p_home / total,
            "draw": p_draw / total,
            "away": p_away / total
        }

    @staticmethod
    def calculate_over_under(expected_rate: float, threshold: float) -> Dict[str, float]:
        """Calcula probabilidades Over/Under para goles, córners o tiros."""
        k = int(np.floor(threshold))
        under_prob = float(poisson.cdf(k, expected_rate))
        over_prob = max(0.0, 1.0 - under_prob)
        return {"over": over_prob, "under": under_prob}

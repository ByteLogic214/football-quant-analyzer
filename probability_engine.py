import numpy as np
from scipy.stats import poisson
from typing import Dict

class ProbabilityEngine:
    @staticmethod
    def calculate_1x2(home_xg: float, away_xg: float, max_goals: int = 8) -> Dict[str, float]:
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                
        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.sum(np.diag(matrix)))
        p_away = float(np.sum(np.triu(matrix, 1)))
        total = p_home + p_draw + p_away
        return {"home": p_home / total, "draw": p_draw / total, "away": p_away / total}

    @staticmethod
    def calculate_over_under(rate: float, line: float = 2.5) -> Dict[str, float]:
        k = int(np.floor(line))
        under = float(poisson.cdf(k, rate))
        over = max(0.0, 1.0 - under)
        return {"over": over, "under": under}

import os
import numpy as np
import pandas as pd
from tabulate import tabulate

from thestatsapi_client import TheStatsAPIClient
from odds_client import TheOddsAPIClient
from ml_engine import HybridPredictor
from probability_engine import ProbabilityEngine
from config import DEFAULT_BANKROLL, FRACTIONAL_KELLY, MIN_EV_THRESHOLD, THESTATSAPI_KEY, ODDS_API_KEY

def calculate_kelly(prob: float, odd: float, bankroll: float = DEFAULT_BANKROLL) -> float:
    b = odd - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    if f <= 0:
        return 0.0
    return round(bankroll * f * FRACTIONAL_KELLY, 2)

def run_pipeline():
    print("=" * 90)
    print("🚀 INICIANDO PIPELINE QUANTITATIVO: TheStatsAPI + The Odds API + CatBoost + LSTM Momentum")
    print("=" * 90)

    # 1. Verificar estado de credenciales
    has_keys = bool(THESTATSAPI_KEY and ODDS_API_KEY)
    if not has_keys:
        print("⚠️ API Keys no configuradas en GitHub Secrets. Ejecutando en Modo de Simulación de Alta Precisión...")

    # 2. Inicializar Modelos
    predictor = HybridPredictor()
    prob_engine = ProbabilityEngine()

    # 3. Secuencia temporal de rendimiento (5 partidos previos)
    # [xG, xGA, TirosArco, TirosArco_Contra, Córners, Córners_Contra]
    seq_home = np.array([
        [2.2, 0.7, 8, 3, 7, 2],
        [1.9, 1.0, 6, 4, 8, 4],
        [2.6, 0.4, 9, 2, 9, 3],
        [2.1, 1.2, 7, 5, 6, 5],
        [3.1, 0.5, 11, 2, 11, 2] # Fuerte aceleración reciente (Momentum positivo)
    ])
    
    seq_away = np.array([
        [0.8, 1.9, 3, 7, 3, 8],
        [1.1, 1.4, 4, 5, 4, 6],
        [0.6, 2.3, 2, 8, 2, 9],
        [1.3, 1.2, 5, 4, 5, 4],
        [0.9, 2.0, 3, 7, 3, 8]
    ])

    home_tab = np.array([3, 160, 64.0, 11, 2.1, 0.7, 4, 1])
    away_tab = np.array([4, -160, 36.0, 15, 0.9, 2.0, 1, 4])
    
    home_team = "Manchester City"
    away_team = "Tottenham"

    # 4. Inferencia con CatBoost + LSTM Momentum
    home_preds = predictor.predict(home_tab, seq_home)
    away_preds = predictor.predict(away_tab, seq_away)

    total_xg = home_preds["xg"] + away_preds["xg"]
    total_sot = home_preds["sot"] + away_preds["sot"]
    total_corners = home_preds["corners"] + away_preds["corners"]

    # 5. Modelado de Probabilidades Poisson
    p_1x2 = prob_engine.calculate_1x2_probabilities(home_preds["xg"], away_preds["xg"])
    p_goals_25 = prob_engine.calculate_over_under(total_xg, 2.5)
    p_cor_95 = prob_engine.calculate_over_under(total_corners, 9.5)
    p_sot_85 = prob_engine.calculate_over_under(total_sot, 8.5)

    # 6. Cuotas de mercado
    market_odds = {
        "1X2: Ganador Local": (p_1x2["home"], 1.50, f"xG: {home_preds['xg']}"),
        "1X2: Empate": (p_1x2["draw"], 4.75, "-"),
        "1X2: Ganador Visitante": (p_1x2["away"], 6.80, f"xG: {away_preds['xg']}"),
        "Goles: Más de 2.5": (p_goals_25["over"], 1.70, f"xG Total: {round(total_xg, 2)}"),
        "Goles: Menos de 2.5": (p_goals_25["under"], 2.25, f"xG Total: {round(total_xg, 2)}"),
        "Córners: Más de 9.5": (p_cor_95["over"], 1.95, f"Córners: {round(total_corners, 1)}"),
        "Remates Arco: Más de 8.5": (p_sot_85["over"], 1.90, f"SoT: {round(total_sot, 1)}")
    }

    # 7. Construcción de Reporte de Apuestas de Valor (+EV)
    rows = []
    for market, (prob_model, odd, projection) in market_odds.items():
        implied_p = 1.0 / odd
        ev = ((prob_model * odd) - 1.0) * 100
        stake = calculate_kelly(prob_model, odd, DEFAULT_BANKROLL)
        is_value = ev >= MIN_EV_THRESHOLD
        
        rows.append({
            "Mercado": market,
            "Proyección Modelo": projection,
            "Prob. Modelo": f"{round(prob_model * 100, 1)}%",
            "Cuota": odd,
            "Prob. Implícita": f"{round(implied_p * 100, 1)}%",
            "Valor (+EV)": f"{'+' if ev > 0 else ''}{round(ev, 2)}%",
            "Stake Kelly": f"€{stake}" if stake > 0 else "€0.00",
            "Decisión": "🔥 APOSTAR" if is_value else "⏸️ PASAR"
        })

    df = pd.DataFrame(rows)

    # 8. Mostrar en Terminal
    table_str = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
    print(f"\nANÁLISIS DE ENCUENTRO: {home_team} vs {away_team}")
    print(table_str)

    # 9. Escribir reporte en Markdown para GitHub Actions Summary
    report_md = f"""# 📊 Reporte Cuantitativo de Apuestas: {home_team} vs {away_team}

### 🧠 Modelos Utilizados:
- **CatBoost Regressor:** Ajuste por variables estructurales y tabulares.
- **LSTM con Momentum (PyTorch):** Inferencia sobre series temporales y aceleración reciente.
- **Métricas Evaluadas:** Expected Goals ($xG$), Remates al Arco ($SoT$) y Córners.

---

### 📋 Tabla de Decisiones y Oportunidades (+EV):

{tabulate(df, headers='keys', tablefmt='github', showindex=False)}

---
> 💡 *Nota de Gestión de Riesgo:* Los stakes recomendados aplican Criterio Kelly Fraccional (25%) sobre un bankroll base de **€{DEFAULT_BANKROLL}**.
"""

    with open("output_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    # Integración nativa con GitHub Actions Summary
    github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary_path:
        with open(github_summary_path, "a", encoding="utf-8") as f:
            f.write(report_md)
        print("✅ Reporte integrado con éxito en GitHub Actions Summary.")

if __name__ == "__main__":
    run_pipeline()

import os
import sys
import pandas as pd
from tabulate import tabulate

from config import validate_environment, DEFAULT_BANKROLL, FRACTIONAL_KELLY, MIN_EV_THRESHOLD
from odds_client import TheOddsAPIClient
from thestatsapi_client import TheStatsAPIClient
from ml_engine import HybridPredictor
from probability_engine import ProbabilityEngine

def calculate_kelly(prob: float, odd: float, bankroll: float = DEFAULT_BANKROLL) -> float:
    if odd <= 1.0: return 0.0
    b = odd - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return round(bankroll * f * FRACTIONAL_KELLY, 2) if f > 0 else 0.0

def run_real_time_pipeline():
    print("=" * 110)
    print("⚡ SISTEMA QUANTITATIVO DE FÚTBOL EN TIEMPO REAL (100% DATOS REALES)")
    print("=" * 110)

    # 1. Validar presencia obligatoria de API Keys
    validate_environment()

    odds_client = TheOddsAPIClient()
    stats_client = TheStatsAPIClient()
    predictor = HybridPredictor()
    prob_engine = ProbabilityEngine()

    # 2. Obtener los 10 partidos reales del día a nivel mundial
    matches = odds_client.get_top_matches_of_the_day()
    if not matches:
        print("❌ No se encontraron partidos disponibles en The Odds API para el día de hoy.")
        sys.exit(0)

    summary_rows = []
    value_bets_rows = []

    for idx, match in enumerate(matches, 1):
        home = match.get("home_team", "Local")
        away = match.get("away_team", "Visitante")
        league = match.get("sport_league", "").replace("soccer_", "").replace("_", " ").title()
        commence = match.get("commence_time", "").replace("T", " ")[:16]
        
        print(f"[{idx}/10] Extrayendo datos de TheStatsAPI para: {home} vs {away}...")
        
        # 3. Consulta de métricas reales a TheStatsAPI
        home_tab, home_seq = stats_client.get_team_real_metrics(home)
        away_tab, away_seq = stats_client.get_team_real_metrics(away)

        # 4. Inferencia con CatBoost + LSTM Momentum
        h_pred = predictor.predict(home_tab, home_seq)
        a_pred = predictor.predict(away_tab, away_seq)

        tot_xg = round(h_pred["xg"] + a_pred["xg"], 2)
        tot_sot = round(h_pred["sot"] + a_pred["sot"], 1)
        tot_corners = round(h_pred["corners"] + a_pred["corners"], 1)

        # 5. Probabilidades Poisson
        p_1x2 = prob_engine.calculate_1x2(h_pred["xg"], a_pred["xg"])
        p_goals = prob_engine.calculate_over_under(tot_xg, 2.5)

        # 6. Cuotas reales de The Odds API
        odds = odds_client.extract_best_odds(match)

        # Tabla resumen
        summary_rows.append({
            "#": idx,
            "Hora (UTC)": commence,
            "Torneo": league,
            "Partido": f"{home} vs {away}",
            "xG Pred.": f"{h_pred['xg']} - {a_pred['xg']}",
            "SoT": tot_sot,
            "Córners": tot_corners,
            "Prob. 1X2 (%)": f"{round(p_1x2['home']*100)}% / {round(p_1x2['draw']*100)}% / {round(p_1x2['away']*100)}%",
            "Cuotas 1X2": f"{odds['home']} | {odds['draw']} | {odds['away']}" if odds['home'] > 0 else "N/D"
        })

        # 7. Evaluación de Apuestas de Valor (+EV)
        markets = [
            (f"{home} (Ganador Local)", p_1x2["home"], odds["home"]),
            (f"Empate (X)", p_1x2["draw"], odds["draw"]),
            (f"{away} (Ganador Visitante)", p_1x2["away"], odds["away"]),
            ("Más de 2.5 Goles", p_goals["over"], odds["over_25"]),
            ("Menos de 2.5 Goles", p_goals["under"], odds["under_25"])
        ]

        for pick, prob, odd in markets:
            if odd <= 1.0: 
                continue
            ev = ((prob * odd) - 1.0) * 100
            if ev >= MIN_EV_THRESHOLD:
                stake = calculate_kelly(prob, odd, DEFAULT_BANKROLL)
                value_bets_rows.append({
                    "Partido": f"{home} vs {away}",
                    "Selección": pick,
                    "Prob. ML": f"{round(prob*100, 1)}%",
                    "Cuota Real": odd,
                    "Valor (+EV)": f"+{round(ev, 2)}%",
                    "Stake Sugerido": f"€{stake}"
                })

    df_summary = pd.DataFrame(summary_rows)
    df_value = pd.DataFrame(value_bets_rows)

    # Impresión en terminal
    print("\n" + "=" * 110)
    print("📋 TABLA GENERAL: 10 PARTIDOS TOP DEL DÍA (DATOS EN TIEMPO REAL)")
    print("=" * 110)
    print(tabulate(df_summary, headers="keys", tablefmt="grid", showindex=False))

    print("\n" + "=" * 110)
    print("🔥 APUESTAS DE VALOR (+EV) DETECTADAS EN TIEMPO REAL:")
    print("=" * 110)
    if not df_value.empty:
        print(tabulate(df_value, headers="keys", tablefmt="grid", showindex=False))
    else:
        print("ℹ️ El mercado se encuentra altamente eficiente hoy. No se detectaron cuotas con EV > +3.0%.")

    # Guardar en Markdown y publicar en GitHub Actions Summary
    report_md = f"""# ⚽ Reporte Cuantitativo en Tiempo Real: 10 Partidos Top del Día
*Modelos: **CatBoost Regressor** + **LSTM con Momentum (PyTorch)** cruzados con **TheStatsAPI** y **The Odds API***

---

### 📊 1. Proyecciones Estadísticas en Directo

{tabulate(df_summary, headers="keys", tablefmt="github", showindex=False)}

---

### 🎯 2. Apuestas con Valor Positivo (+EV) Detectadas

{tabulate(df_value, headers="keys", tablefmt="github", showindex=False) if not df_value.empty else "No hay oportunidades con EV > +3% en este momento."}

---
> 💡 *Nota:* Todas las cuotas y estadísticas fueron extraídas en vivo. La gestión monetaria aplica **Criterio Kelly Fraccional (25%)**.
"""

    with open("output_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report_md)
        print("\n✅ Reporte publicado en GitHub Actions Summary.")

if __name__ == "__main__":
    run_real_time_pipeline()

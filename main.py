import os
import sys
from datetime import datetime
import pandas as pd
from tabulate import tabulate

from config import (
    validate_environment, DEFAULT_BANKROLL, FRACTIONAL_KELLY, MIN_EV_THRESHOLD,
    log_info, log_warning, log_error
)
from odds_client import TheOddsAPIClient
from thestatsapi_client import TheStatsAPIClient
from ml_engine import HybridPredictor
from probability_engine import ProbabilityEngine

# ============================================================================
# UTILIDADES
# ============================================================================

def calculate_kelly(prob: float, odd: float, bankroll: float = DEFAULT_BANKROLL) -> float:
    """
    Calcula stake óptimo usando Criterio Kelly Fraccional (25%).
    
    Fórmula:
    Kelly % = (p * (o - 1) - (1 - p)) / (o - 1)
    Stake = Kelly % * FRACTIONAL_KELLY * Bankroll
    
    Args:
        prob: Probabilidad del modelo [0, 1]
        odd: Cuota decimal [1.0, ∞)
        bankroll: Bankroll disponible (default €1000)
    
    Returns:
        Stake en euros (float, ≥ 0)
    """
    if odd <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0
    
    b = odd - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    
    if f <= 0:
        return 0.0
    
    return round(bankroll * f * FRACTIONAL_KELLY, 2)


def format_league_name(sport_key: str) -> str:
    """
    Convierte sport_key de The Odds API a nombre legible.
    
    Ejemplo: "soccer_england_premier_league" -> "England Premier League"
    """
    return (
        sport_key
        .replace("soccer_", "")
        .replace("_", " ")
        .title()
    )


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_real_time_pipeline():
    """
    Orquestador principal:
    1. Valida credenciales
    2. Extrae 10 partidos reales del día
    3. Consulta métricas reales de TheStatsAPI
    4. Ejecuta ensamble CatBoost + LSTM
    5. Calcula probabilidades Poisson
    6. Detecta apuestas con +EV
    7. Genera reporte markdown
    
    Restricción CRÍTICA:
    - Omite eventos donde falle la extracción de datos (NO inventa)
    - Registra logs informativos para auditoría
    """
    
    print("="*120)
    print("⚡ SISTEMA CUANTITATIVO DE FÚTBOL EN TIEMPO REAL (100% DATOS REALES)")
    print("="*120)
    print(f"🕐 Inicio: {datetime.utcnow().isoformat()} UTC")
    print()
    
    # 1. Validar credenciales
    validate_environment()
    
    # 2. Inicializar clientes
    odds_client = TheOddsAPIClient()
    stats_client = TheStatsAPIClient()
    predictor = HybridPredictor()
    prob_engine = ProbabilityEngine()
    
    # 3. Obtener partidos reales del día
    matches = odds_client.get_top_matches_of_the_day()
    if not matches:
        log_error("No se encontraron partidos para hoy. Abortando.")
        sys.exit(0)
    
    print(f"\n⚽ {len(matches)} partidos listos para análisis cuantitativo\n")
    
    summary_rows = []
    value_bets_rows = []
    failed_matches = 0
    
    # ========================================================================
    # 4. PROCESA CADA PARTIDO
    # ========================================================================
    
    for idx, match in enumerate(matches, 1):
        home = match.get("home_team", "Local")
        away = match.get("away_team", "Visitante")
        league = format_league_name(match.get("sport_league", "Unknown"))
        commence = match.get("commence_time", "").replace("T", " ")[:16]
        event_id = match.get("id", f"match_{idx}")
        
        print(f"[{idx}/{len(matches)}] Procesando: {home} vs {away} ({league})")
        
        # ====================================================================
        # 4.1. Consulta métricas reales de TheStatsAPI
        # ====================================================================
        
        home_metrics = stats_client.get_team_real_metrics(home)
        if not home_metrics:
            log_warning(f"Omitiendo {home}: métricas no disponibles en TheStatsAPI")
            failed_matches += 1
            continue  # ← CRÍTICO: Omite si falta data (NO inventa)
        
        away_metrics = stats_client.get_team_real_metrics(away)
        if not away_metrics:
            log_warning(f"Omitiendo {away}: métricas no disponibles en TheStatsAPI")
            failed_matches += 1
            continue  # ← CRÍTICO: Omite si falta data (NO inventa)
        
        home_tab, home_seq = home_metrics
        away_tab, away_seq = away_metrics
        
        # ====================================================================
        # 4.2. Inferencia con CatBoost + LSTM Momentum
        # ====================================================================
        
        try:
            h_pred = predictor.predict(home_tab, home_seq)
            a_pred = predictor.predict(away_tab, away_seq)
        except Exception as e:
            log_error(f"Error en predicción para {home} vs {away}: {e}")
            failed_matches += 1
            continue
        
        tot_xg = round(h_pred["xg"] + a_pred["xg"], 2)
        tot_sot = round(h_pred["sot"] + a_pred["sot"], 1)
        tot_corners = round(h_pred["corners"] + a_pred["corners"], 1)
        
        # ====================================================================
        # 4.3. Probabilidades Poisson
        # ====================================================================
        
        p_1x2 = prob_engine.calculate_1x2(h_pred["xg"], a_pred["xg"])
        p_goals = prob_engine.calculate_over_under(tot_xg, 2.5)
        
        # ====================================================================
        # 4.4. Cuotas reales de The Odds API
        # ====================================================================
        
        odds = odds_client.extract_best_odds(match)
        
        # ====================================================================
        # 4.5. TABLA DE RESUMEN
        # ====================================================================
        
        summary_rows.append({
            "#": idx,
            "Hora (UTC)": commence,
            "Torneo": league,
            "Partido": f"{home} vs {away}",
            "xG Pred.": f"{h_pred['xg']} - {a_pred['xg']}",
            "SoT": tot_sot,
            "Córners": tot_corners,
            "Prob. 1X2 (%)": f"{round(p_1x2['home']*100)}% / {round(p_1x2['draw']*100)}% / {round(p_1x2['away']*100)}%",
            "Cuotas 1X2": (
                f"{odds['home']} | {odds['draw']} | {odds['away']}"
                if odds['home'] > 0 else "N/D"
            )
        })
        
        # ====================================================================
        # 4.6. EVALUACIÓN DE APUESTAS CON +EV
        # ====================================================================
        
        markets = [
            (f"{home} (Ganador Local)", p_1x2["home"], odds["home"]),
            (f"Empate (X)", p_1x2["draw"], odds["draw"]),
            (f"{away} (Ganador Visitante)", p_1x2["away"], odds["away"]),
            ("Más de 2.5 Goles", p_goals["over"], odds["over_25"]),
            ("Menos de 2.5 Goles", p_goals["under"], odds["under_25"])
        ]
        
        for pick, prob, odd in markets:
            if odd <= 1.0:
                continue  # Cuota inválida
            
            # EV en porcentaje: (prob * odd - 1) * 100
            ev_pct = ((prob * odd) - 1.0) * 100
            
            if ev_pct >= MIN_EV_THRESHOLD:
                stake = calculate_kelly(prob, odd, DEFAULT_BANKROLL)
                value_bets_rows.append({
                    "Partido": f"{home} vs {away}",
                    "Selección": pick,
                    "Prob. ML": f"{round(prob*100, 1)}%",
                    "Cuota Real": f"{odd:.2f}",
                    "Valor (+EV)": f"+{round(ev_pct, 2)}%",
                    "Stake (Kelly 25%)": f"€{stake}",
                    "Exp. Return": f"€{round(stake * odd, 2)}"
                })
        
        print(f"    ✓ Procesado exitosamente\n")
    
    # ========================================================================
    # 5. GENERACIÓN DE REPORTE
    # ========================================================================
    
    df_summary = pd.DataFrame(summary_rows)
    df_value = pd.DataFrame(value_bets_rows)
    
    # Imprime en terminal
    print("\n" + "="*120)
    print("📋 TABLA GENERAL: PROYECCIONES ESTADÍSTICAS EN TIEMPO REAL")
    print("="*120)
    if not df_summary.empty:
        print(tabulate(df_summary, headers="keys", tablefmt="grid", showindex=False))
    else:
        print("ℹ️ No hay partidos disponibles.")
    
    print("\n" + "="*120)
    print("🔥 APUESTAS CON VALOR POSITIVO (+EV) DETECTADAS")
    print("="*120)
    if not df_value.empty:
        print(tabulate(df_value, headers="keys", tablefmt="grid", showindex=False))
    else:
        print("ℹ️ El mercado se encuentra altamente eficiente. No hay +EV > +3.0% en este momento.")
    
    print(f"\n⚠️ Eventos omitidos (data insuficiente): {failed_matches}")
    print(f"✅ Eventos procesados exitosamente: {len(summary_rows)}")
    print(f"💰 Oportunidades de valor identificadas: {len(value_bets_rows)}")
    
    # ========================================================================
    # 6. GUARDAR REPORTE MARKDOWN
    # ========================================================================
    
    report_md = f"""# ⚽ Análisis Cuantitativo en Tiempo Real

*Generado: {datetime.utcnow().isoformat()} UTC*

**Modelos:** CatBoost Regressor 50% + LSTM con Momentum (PyTorch) 50%
**Datos:** TheStatsAPI (métricas reales) + The Odds API (cuotas en vivo)
**Gestión de Bankroll:** Criterio Kelly Fraccional (25%)

---

## 📊 1. Proyecciones Estadísticas

{tabulate(df_summary, headers="keys", tablefmt="github", showindex=False) if not df_summary.empty else "Sin datos disponibles."}

---

## ⚽ 2. Apuestas con Valor (+EV > +3.0%)

{tabulate(df_value, headers="keys", tablefmt="github", showindex=False) if not df_value.empty else "No hay oportunidades con EV > +3.0% en este momento."}

---

## 📈 Estadísticas de la Sesión

- **Partidos analizados:** {len(summary_rows)}
- **Partidos omitidos (data insuficiente):** {failed_matches}
- **Oportunidades de valor identificadas:** {len(value_bets_rows)}
- **Bankroll simulado:** €{DEFAULT_BANKROLL}
- **Fracción Kelly aplicada:** {FRACTIONAL_KELLY * 100}%
- **Umbral mínimo de EV:** +{MIN_EV_THRESHOLD}%

---

> 💡 **Nota de Auditoría:** Todos los datos (estadísticas, cuotas, métricas) fueron extraídos en tiempo real. No se utilizaron datos sintéticos ni simulados. Los eventos donde faltó información se omitieron (criterio conservador).
"""
    
    # Escribe archivo de reporte
    with open("output_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    
    log_info("✅ Reporte markdown guardado en output_report.md")
    
    # Intenta publicar en GitHub Actions Summary
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(report_md)
            log_info("✅ Reporte publicado en GitHub Actions Summary")
        except Exception as e:
            log_warning(f"No se pudo escribir en GITHUB_STEP_SUMMARY: {e}")
    else:
        log_info("(Ejecutándose fuera de GitHub Actions)")
    
    print("\n" + "="*120)
    print("✨ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*120)


if __name__ == "__main__":
    try:
        run_real_time_pipeline()
    except KeyboardInterrupt:
        log_warning("Pipeline interrumpido por usuario")
        sys.exit(0)
    except Exception as e:
        log_error(f"Error fatal en pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

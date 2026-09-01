# ⚽ Football Quantitative Analyzer

**Sistema de análisis cuantitativo de fútbol en tiempo real con apuestas de valor (+EV)**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Objetivo

Identificar **oportunidades de valor (+EV)** en mercados de apuestas de fútbol mediante:
- Extracción de datos **reales en tiempo real** (TheStatsAPI + The Odds API)
- Ensamble híbrido ML: **CatBoost 50% + LSTM con Momentum 50%**
- Cálculo de probabilidades con **modelo Poisson bivariado**
- Gestión de bankroll con **Criterio Kelly Fraccional (25%)**

**Restricción crítica:** CERO datos sintéticos o simulados. 100% datos reales.

---

## 📊 Pipeline Completo

```
┌──────────────────────────────────────────────────────��──────────────┐
│                     SISTEMA EN TIEMPO REAL                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            ┌──────────────────┐      ┌──────────────────┐
            │ The Odds API     │      │ TheStatsAPI      │
            │ - Ligas activas  │      │ - Métricas reales│
            │ - Cuotas en vivo │      │ - Historial      │
            │ - Top 10 partidos│      │ - Equipos        │
            └────────┬─────────┘      └────────┬─────────┘
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                     ┌────────────��───────────────┐
                     │  TeamMatcher             │
                     │  - Fuzzy matching (0.70) │
                     │  - Normalización         │
                     └────────────┬─────────────┘
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │              ML ENGINE (Ensamble)              │
        ├─────────────────────────────────────────────────┤
        │  50% CatBoost Regressor                         │
        │  - Predicción tabular [8 features]              │
        │  - xG, SoT, Corners                             │
        │                                                 │
        │  50% LSTM con Momentum (PyTorch)                │
        │  - Secuencia temporal [5x6 partidos]            │
        │  - Hidden state + momentum mechanism            │
        └────────────────┬───────────────────────────��─────┘
                         ▼
        ┌─────────────────────────────────────────────────┐
        │        PROBABILITY ENGINE (Poisson)             │
        ├─────────────────────────────────────────────────┤
        │  Bivariado 1X2:                                 │
        │  - P(Home Win), P(Draw), P(Away Win)            │
        │                                                 │
        │  Over/Under:                                    │
        │  - P(Over 2.5), P(Under 2.5)                    │
        └────────────────┬─────────────────────────────────┘
                         ▼
        ┌─────────────────────────────────────────────────┐
        │     VALOR DETECTION & BANKROLL MGMT             │
        ├─────────────────────────────────────────────────��
        │  EV Calculation: (prob * odd - 1) * 100         │
        │  Threshold: +3.0%                               │
        │  Kelly Criterion (Fraccional 25%)               │
        │  Stake Calculation & Expected Return            │
        └────────────────┬─────────────────────────────────┘
                         ▼
        ┌─────────────────────────────────────────────────┐
        │     MARKDOWN REPORT & ARTIFACTS                 │
        ├─────────────────────────────────────────────────┤
        │  output_report.md (GitHub Actions Summary)      │
        │  Terminal tables (tabulate)                     │
        │  Audit trail (logging)                          │
        └─────────────────────────────────────────────────┘
```

---

## 🚀 Inicio Rápido

### 1. Requisitos Previos
- Python 3.11+
- Cuentas de API:
  - **The Odds API** (https://the-odds-api.com) → `ODDS_API_KEY`
  - **TheStatsAPI** (https://www.thestatsapi.com) → `THESTATSAPI_KEY`

### 2. Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/ByteLogic214/football-quant-analyzer.git
cd football-quant-analyzer

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o en Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Credenciales

**Opción A: Variables de entorno (LOCAL)**
```bash
export THESTATSAPI_KEY="tu_clave_aqui"
export ODDS_API_KEY="tu_clave_aqui"
```

**Opción B: GitHub Secrets (PRODUCCIÓN)**
1. Ir a: Repository → Settings → Secrets and variables → Actions
2. Crear:
   - `THESTATSAPI_KEY` = tu clave
   - `ODDS_API_KEY` = tu clave

### 4. Ejecutar Análisis

```bash
# Local
python main.py

# GitHub Actions (automático)
# Scheduled: 09:08 UTC Mon-Fri
# O manual: Actions → Football Quantitative Analyzer → Run workflow
```

---

## 📈 Output Esperado

**Terminal:**
```
================================================== 
⚡ SISTEMA CUANTITATIVO DE FÚTBOL EN TIEMPO REAL
==================================================
🕐 Inicio: 2026-09-01T09:08:00 UTC

✓ Credenciales validadas correctamente

⚽ 10 partidos listos para análisis cuantitativo

[1/10] Procesando: Manchester United vs Liverpool (England Premier League)
    ✓ Procesado exitosamente

...

================================================== 
📋 TABLA GENERAL: PROYECCIONES ESTADÍSTICAS
==================================================
┌─────┬──────────┬────────┬───────────────────────┬──────────┐
│   # │ Hora UTC │ Torneo │ Partido               │ xG Pred. │
├─────┼──────────┼────────┼───────────────────────┼──────────┤
│   1 │ 13:30    │ England Premier League │ Man Utd vs Liv │ 1.8 - 2.1 │
└─────┴──────────┴────────┴───────────────────────┴──────────┘

================================================== 
🔥 APUESTAS CON VALOR (+EV) DETECTADAS
==================================================
┌──────────────────────┬──────────��──────────┬────────────┐
│ Partido              │ Selección│ Prob. ML │ +EV        │
├──────────────────────┼──────────┼──────────┼────────────┤
│ Man Utd vs Liverpool │ Over 2.5 │ 62.3%    │ +5.45%     │
└──────────────────────┴──────────┴──────────┴────────────┘

✅ Eventos procesados exitosamente: 9
💰 Oportunidades de valor identificadas: 3
```

---

## 🏗️ Arquitectura Técnica

### Módulos Principales

| Módulo | Responsabilidad |
|--------|-----------------|
| `config.py` | Validación de credenciales, parámetros globales, logging |
| `odds_client.py` | The Odds API: ligas dinámicas, partidos, cuotas reales |
| `thestatsapi_client.py` | TheStatsAPI: métricas reales de equipos, histórico |
| `matcher.py` | TeamMatcher: fuzzy matching para normalización |
| `lstm_momentum.py` | Red LSTM con mecanismo de momentum (PyTorch) |
| `ml_engine.py` | HybridPredictor: ensamble CatBoost + LSTM (50/50) |
| `probability_engine.py` | Poisson bivariado: 1X2 y Over/Under |
| `main.py` | Orquestador: pipeline completo + reporte |

### Features ML

**Vector Tabular [8]:**
- Días de descanso
- Posesión media
- Precisión de pases
- xG promedio
- Faltas promedio
- Victorias / Empates / Derrotas

**Secuencia [5x6]:**
- Últimos 5 partidos reales
- Por partido: [xG, xGA, SoT, SoTA, Corners, Corners_contra]

---

## ⚙️ Configuración Avanzada

### Parámetros en `config.py`

```python
TOP_MATCHES_LIMIT = 10              # Partidos a analizar
DEFAULT_BANKROLL = 1000.0           # Bankroll simulado (€)
FRACTIONAL_KELLY = 0.25             # Kelly fraccional (25%)
MIN_EV_THRESHOLD = 3.0              # Umbral mínimo de +EV (%)
MAX_RETRIES = 3                     # Reintentos en APIs
REQUEST_TIMEOUT = 15                # Timeout en segundos
```

---

## 🔐 Seguridad

- **Nunca** harcodear API keys (usar GitHub Secrets)
- Validación obligatoria de credenciales
- Timeout en requests (15s)
- Rate limit handling automático

---

## 📄 Licencia

MIT License - Ver `LICENSE` para detalles.

---

**⚽ Made with ❤️ for football analytics | 100% Real Data | Zero Synthetic Data**

import os

# Claves obtenidas de GitHub Secrets o Variables de Entorno
THESTATSAPI_KEY = os.getenv("THESTATSAPI_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

THESTATSAPI_BASE_URL = "https://api.thestatsapi.com/api"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Configuración de Bankroll para Criterio Kelly
DEFAULT_BANKROLL = 1000.0  # Euros / Dólares
FRACTIONAL_KELLY = 0.25    # Criterio Kelly Conservador (25%)
MIN_EV_THRESHOLD = 3.0     # Mínimo % de Valor Esperado para apostar (+3.0%)

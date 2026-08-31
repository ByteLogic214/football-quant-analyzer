import os
import sys

# Claves obtenidas de GitHub Secrets / Variables de Entorno
THESTATSAPI_KEY = os.getenv("THESTATSAPI_KEY", "").strip()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

THESTATSAPI_BASE_URL = "https://api.thestatsapi.com/api"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Parámetros operativos
TOP_MATCHES_LIMIT = 10
DEFAULT_BANKROLL = 1000.0
FRACTIONAL_KELLY = 0.25
MIN_EV_THRESHOLD = 3.0

def validate_environment():
    """Valida que existan las credenciales reales antes de iniciar."""
    missing = []
    if not THESTATSAPI_KEY:
        missing.append("THESTATSAPI_KEY")
    if not ODDS_API_KEY:
        missing.append("ODDS_API_KEY")
        
    if missing:
        msg = f"❌ ERROR CRÍTICO: Faltan las siguientes API Keys en GitHub Secrets: {', '.join(missing)}.\n" \
              f"Configúralas en Settings > Secrets and variables > Actions para consultar datos en tiempo real."
        print(msg)
        sys.exit(1)

import os
import sys
from datetime import datetime

# ============================================================================
# API CREDENTIALS & CONFIGURATION
# ============================================================================
THESTATSAPI_KEY = os.getenv("THESTATSAPI_KEY", "").strip()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

THESTATSAPI_BASE_URL = "https://api.thestatsapi.com/api"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# ============================================================================
# OPERATIONAL PARAMETERS
# ============================================================================
TOP_MATCHES_LIMIT = 10
DEFAULT_BANKROLL = 1000.0
FRACTIONAL_KELLY = 0.25
MIN_EV_THRESHOLD = 3.0

# Fuzzy matching umbral para normalización de equipos
FUZZY_MATCH_THRESHOLD = 0.70

# Máximo de reintentos en requests
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15

# ============================================================================
# LOGGING & VALIDATION
# ============================================================================
def validate_environment():
    """
    Valida la presencia obligatoria de credenciales reales.
    Aborta la ejecución si falta alguna clave API.
    """
    missing = []
    if not THESTATSAPI_KEY:
        missing.append("THESTATSAPI_KEY")
    if not ODDS_API_KEY:
        missing.append("ODDS_API_KEY")
    
    if missing:
        msg = (
            f"❌ ERROR CRÍTICO: Faltan las siguientes API Keys en GitHub Secrets:\n"
            f"   {', '.join(missing)}\n\n"
            f"Configúralas en: Settings > Secrets and variables > Actions\n"
            f"para consultar datos en tiempo real de The Odds API y TheStatsAPI."
        )
        print(msg)
        sys.exit(1)
    
    print(f"✅ Credenciales validadas correctamente a las {datetime.utcnow().isoformat()}")

def log_info(msg: str):
    """Registra mensaje informativo."""
    print(f"[INFO] {msg}")

def log_warning(msg: str):
    """Registra advertencia."""
    print(f"⚠️  WARNING: {msg}")

def log_error(msg: str):
    """Registra error crítico."""
    print(f"❌ ERROR: {msg}")

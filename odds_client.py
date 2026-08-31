import requests
from typing import Dict, Any, List
from config import ODDS_API_BASE_URL, ODDS_API_KEY

class TheOddsAPIClient:
    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = ODDS_API_BASE_URL

    def get_odds(self, sport: str = "soccer_epl", regions: str = "eu,uk") -> List[Dict[str, Any]]:
        """Obtiene cuotas del mercado 1X2 y Totales."""
        if not self.api_key:
            return []
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": "h2h,totals",
            "oddsFormat": "decimal"
        }
        try:
            response = requests.get(url, params=params, timeout=12)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"[Aviso The Odds API]: {e}")
            return []

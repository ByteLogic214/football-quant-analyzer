import requests
from typing import Dict, Any
from config import THESTATSAPI_BASE_URL, THESTATSAPI_KEY

class TheStatsAPIClient:
    def __init__(self, api_key: str = THESTATSAPI_KEY):
        self.api_key = api_key
        self.base_url = THESTATSAPI_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }

    def get_match_stats(self, match_id: str) -> Dict[str, Any]:
        """Consulta estadísticas avanzadas en vivo o históricas."""
        if not self.api_key:
            return {}
        url = f"{self.base_url}/football/matches/{match_id}/stats"
        try:
            response = requests.get(url, headers=self.headers, timeout=12)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"[Aviso TheStatsAPI]: {e}")
            return {}

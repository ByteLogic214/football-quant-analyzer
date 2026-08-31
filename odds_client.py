import requests
from datetime import datetime, timezone
from typing import Dict, Any, List
from config import ODDS_API_BASE_URL, ODDS_API_KEY, TOP_MATCHES_LIMIT

class TheOddsAPIClient:
    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = ODDS_API_BASE_URL

    def get_all_active_soccer_sports(self) -> List[str]:
        """Obtiene dinámicamente todos los torneos y ligas de fútbol activos a nivel mundial."""
        url = f"{self.base_url}/sports"
        params = {"apiKey": self.api_key}
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        
        sports = res.json()
        soccer_keys = [
            s["key"] for s in sports 
            if s.get("group") == "Soccer" and s.get("active") is True
        ]
        return soccer_keys

    def get_top_matches_of_the_day(self, limit: int = TOP_MATCHES_LIMIT) -> List[Dict[str, Any]]:
        """
        Escanea todas las ligas de fútbol activas, filtra partidos de hoy y
        selecciona los 10 con mayor liquidez y cobertura de cuotas.
        """
        soccer_keys = self.get_all_active_soccer_sports()
        print(f"🌍 Ligas y torneos de fútbol activos encontrados: {len(soccer_keys)}")
        
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        
        all_matches = []
        for sport_key in soccer_keys:
            url = f"{self.base_url}/sports/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": "eu,uk,us",
                "markets": "h2h,totals",
                "oddsFormat": "decimal"
            }
            try:
                res = requests.get(url, params=params, timeout=12)
                if res.status_code != 200:
                    continue
                events = res.json()
                for ev in events:
                    commence = ev.get("commence_time", "")
                    # Filtrar partidos que se juegan hoy o en las próximas 24 horas
                    if commence.startswith(today_str) or (commence and commence > now.isoformat()):
                        ev["sport_league"] = sport_key
                        all_matches.append(ev)
            except Exception as e:
                print(f"Aviso al consultar {sport_key}: {e}")

        # Ordenar por cantidad de casas de apuestas disponibles (relevancia de mercado)
        all_matches.sort(
            key=lambda x: (len(x.get("bookmakers", [])), x.get("commence_time", "")), 
            reverse=True
        )

        selected = all_matches[:limit]
        print(f"🎯 Partidos del día listos para análisis en tiempo real: {len(selected)}")
        return selected

    @staticmethod
    def extract_best_odds(event: Dict[str, Any]) -> Dict[str, float]:
        """Extrae la cuota más alta del mercado disponible entre todos los operadores."""
        best = {"home": 0.0, "draw": 0.0, "away": 0.0, "over_25": 0.0, "under_25": 0.0}
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                m_key = market.get("key")
                outcomes = market.get("outcomes", [])
                
                if m_key == "h2h":
                    for out in outcomes:
                        name = out.get("name")
                        price = float(out.get("price", 0.0))
                        if name == home_team and price > best["home"]:
                            best["home"] = price
                        elif name == away_team and price > best["away"]:
                            best["away"] = price
                        elif name.lower() in ["draw", "empate"] and price > best["draw"]:
                            best["draw"] = price

                elif m_key == "totals":
                    for out in outcomes:
                        if out.get("point") == 2.5:
                            name = out.get("name", "").lower()
                            price = float(out.get("price", 0.0))
                            if name == "over" and price > best["over_25"]:
                                best["over_25"] = price
                            elif name == "under" and price > best["under_25"]:
                                best["under_25"] = price

        return best

import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from config import (
    ODDS_API_BASE_URL, ODDS_API_KEY, TOP_MATCHES_LIMIT,
    MAX_RETRIES, REQUEST_TIMEOUT, log_info, log_warning, log_error
)

class TheOddsAPIClient:
    """
    Cliente para consumir The Odds API en tiempo real.
    Extrae cuotas y partidos activos de todas las ligas de fútbol.
    
    Restricción crítica: ZERO MOCK DATA
    - Si una API retorna 429/500, registra y continúa
    - Nunca inventa cuotas o partidos
    """

    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = ODDS_API_BASE_URL
        self.session = requests.Session()

    def _get_with_retries(
        self, 
        url: str, 
        params: Optional[Dict] = None,
        retries: int = MAX_RETRIES
    ) -> Optional[Dict]:
        """
        Ejecuta GET con reintentos y manejo de errores HTTP.
        Retorna None si falla, registra excepciones.
        """
        for attempt in range(retries):
            try:
                res = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                
                if res.status_code == 429:
                    log_warning(f"Rate limit (429) en {url}. Reintentando ({attempt+1}/{retries})...")
                    continue
                elif res.status_code == 500:
                    log_warning(f"Error 500 en {url}. Reintentando ({attempt+1}/{retries})...")
                    continue
                elif res.status_code != 200:
                    log_warning(f"HTTP {res.status_code} en {url}")
                    return None
                
                return res.json()
            
            except requests.exceptions.Timeout:
                log_warning(f"Timeout en {url} (intento {attempt+1}/{retries})")
            except requests.exceptions.ConnectionError as e:
                log_warning(f"Conexión rechazada {url}: {e}")
            except ValueError:
                log_error(f"Respuesta no-JSON en {url}")
                return None
        
        return None

    def get_all_active_soccer_sports(self) -> List[str]:
        """
        Obtiene dinámicamente todos los torneos de fútbol ACTIVOS a nivel mundial.
        
        Endpoint: GET /v4/sports/?apiKey={KEY}
        Filtros:
        - group == "Soccer"
        - active == True
        
        Returns:
            Lista de sport_keys activos. Lista vacía si falla la consulta.
        """
        url = f"{self.base_url}/sports"
        params = {"apiKey": self.api_key}
        
        data = self._get_with_retries(url, params)
        if not data:
            log_error("No se pudo obtener ligas de The Odds API")
            return []
        
        sports = data if isinstance(data, list) else data.get("sports", [])
        soccer_keys = [
            s["key"] for s in sports
            if s.get("group") == "Soccer" and s.get("active") is True
        ]
        
        log_info(f"Ligas de fútbol activas encontradas: {len(soccer_keys)}")
        return soccer_keys

    def get_top_matches_of_the_day(self, limit: int = TOP_MATCHES_LIMIT) -> List[Dict[str, Any]]:
        """
        Escanea TODAS las ligas de fútbol activas globalmente.
        Filtra partidos que comienzan hoy (UTC) o en las próximas 24 horas.
        Ordena por liquidez de mercado (cantidad de bookmakers con cuotas activas).
        Retorna los TOP N partidos.
        
        Restricción CRÍTICA:
        - No hardcodea ligas (EPL, LaLiga, etc.)
        - Si una liga falla, continúa con las siguientes
        - NUNCA inventa partidos
        
        Returns:
            Lista de Dict con estructura de The Odds API
        """
        soccer_keys = self.get_all_active_soccer_sports()
        if not soccer_keys:
            log_warning("No se encontraron ligas de fútbol activas")
            return []
        
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_start = today_start + timedelta(days=1)
        
        all_matches = []
        
        for sport_key in soccer_keys:
            url = f"{self.base_url}/sports/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": "eu,uk,us",
                "markets": "h2h,totals",
                "oddsFormat": "decimal"
            }
            
            events = self._get_with_retries(url, params)
            if not events:
                log_warning(f"No se obtuvieron datos para {sport_key}")
                continue
            
            events_list = events if isinstance(events, list) else events.get("events", [])
            
            for ev in events_list:
                commence = ev.get("commence_time", "")
                if not commence:
                    continue
                
                # Parsea ISO 8601
                try:
                    commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    continue
                
                # Filtro temporal: hoy o mañana
                if not (today_start <= commence_dt < tomorrow_start + timedelta(hours=1)):
                    continue
                
                # Requiere equipo local y visitante
                if not ev.get("home_team") or not ev.get("away_team"):
                    continue
                
                # Requiere al menos 1 casa de apuestas con cuotas
                if not ev.get("bookmakers"):
                    continue
                
                ev["sport_league"] = sport_key
                all_matches.append(ev)
        
        if not all_matches:
            log_warning("No se encontraron partidos para hoy tras escanear todas las ligas")
            return []
        
        # Ordena por liquidez: número de bookmakers con cuotas
        all_matches.sort(
            key=lambda x: (len(x.get("bookmakers", [])), x.get("commence_time", "")),
            reverse=True
        )
        
        selected = all_matches[:limit]
        log_info(f"Partidos del día seleccionados: {len(selected)}/{len(all_matches)}")
        return selected

    @staticmethod
    def extract_best_odds(event: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrae la MEJOR CUOTA (máxima) del mercado entre todos los bookmakers.
        
        Mercados:
        - h2h (Match Odds): Home (1), Draw (X), Away (2)
        - totals: Over/Under 2.5 Goles
        
        Restricción: Solo retorna cuotas > 1.0 que existan realmente.
        Si un mercado no tiene cuotas, retorna 0.0.
        
        Returns:
            Dict con claves: home, draw, away, over_25, under_25
        """
        best = {"home": 0.0, "draw": 0.0, "away": 0.0, "over_25": 0.0, "under_25": 0.0}
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                m_key = market.get("key")
                outcomes = market.get("outcomes", [])
                
                if m_key == "h2h":
                    for out in outcomes:
                        name = out.get("name", "")
                        price = float(out.get("price", 0.0))
                        
                        if price <= 1.0:
                            continue
                        
                        if name == home_team and price > best["home"]:
                            best["home"] = price
                        elif name == away_team and price > best["away"]:
                            best["away"] = price
                        elif name.lower() in ["draw", "empate", "tie"] and price > best["draw"]:
                            best["draw"] = price
                
                elif m_key == "totals":
                    for out in outcomes:
                        point = out.get("point")
                        name = out.get("name", "").lower()
                        price = float(out.get("price", 0.0))
                        
                        if price <= 1.0:
                            continue
                        
                        if point == 2.5:
                            if name == "over" and price > best["over_25"]:
                                best["over_25"] = price
                            elif name == "under" and price > best["under_25"]:
                                best["under_25"] = price

        return best

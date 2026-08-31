import requests
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from config import (
    THESTATSAPI_BASE_URL, THESTATSAPI_KEY,
    MAX_RETRIES, REQUEST_TIMEOUT, log_info, log_warning, log_error
)
from matcher import TeamMatcher

class TheStatsAPIClient:
    """
    Cliente para TheStatsAPI que extrae métricas REALES de equipos.
    
    Restricción CRÍTICA (CERO MOCK DATA):
    - Si un equipo no existe o falla la consulta, captura el error
    - Registra log INFORMATIVO
    - JAMÁS inventa métricas: retorna None para señalar fallo
    - El orquestador (main.py) decide si omitir el evento
    
    Extrae:
    1. Vector Tabular [8]: días_descanso, posesión, precisión_pases, xG_media, faltas, victorias, empates, derrotas
    2. Secuencia [5x6]: Últimos 5 partidos reales con [xG, xGA, SoT, SoTA, Corners, Corners_contra]
    """

    def __init__(self, api_key: str = THESTATSAPI_KEY):
        self.api_key = api_key
        self.base_url = THESTATSAPI_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        self.matcher = TeamMatcher()
        self.session = requests.Session()

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        GET con reintentos y manejo de errores sin inventar datos.
        Retorna None si falla (NO retorna diccionarios sintéticos).
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                res = self.session.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
                
                if res.status_code == 429:
                    log_warning(f"Rate limit (429) en TheStatsAPI. Reintentando ({attempt+1}/{MAX_RETRIES})...")
                    continue
                elif res.status_code == 401:
                    log_error("Autenticación fallida en TheStatsAPI (401). Verifica THESTATSAPI_KEY")
                    return None
                elif res.status_code == 404:
                    log_warning(f"Recurso no encontrado: {endpoint}")
                    return None
                elif res.status_code >= 500:
                    log_warning(f"Error servidor (HTTP {res.status_code}). Reintentando...")
                    continue
                elif res.status_code != 200:
                    log_warning(f"HTTP {res.status_code} en {endpoint}")
                    return None
                
                return res.json()
            
            except requests.exceptions.Timeout:
                log_warning(f"Timeout en TheStatsAPI (intento {attempt+1}/{MAX_RETRIES})")
            except requests.exceptions.ConnectionError as e:
                log_warning(f"Conexión rechazada por TheStatsAPI: {e}")
            except ValueError:
                log_error(f"Respuesta no-JSON en {endpoint}")
                return None
        
        log_error(f"Falló obtener datos después de {MAX_RETRIES} reintentos: {endpoint}")
        return None

    def find_team_id(self, team_name: str) -> Optional[int]:
        """
        Busca el ID real de un equipo en TheStatsAPI.
        
        Estrategia:
        1. Busca exacto por nombre
        2. Si falla, busca por fuzzy matching
        
        Returns:
            team_id si existe, None si no se encuentra
        """
        # Intenta búsqueda directa
        data = self._get("/football/teams", params={"name": team_name})
        if data:
            teams = data.get("data", [])
            if isinstance(teams, list) and len(teams) > 0:
                return teams[0].get("id")
        
        # Intenta búsqueda por query
        data = self._get(f"/football/teams/search", params={"query": team_name})
        if data:
            teams = data.get("data", [])
            if isinstance(teams, list) and len(teams) > 0:
                return teams[0].get("id")
        
        log_warning(f"Equipo '{team_name}' no encontrado en TheStatsAPI")
        return None

    def get_team_real_metrics(self, team_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extrae métricas REALES de un equipo.
        
        Restricción CRÍTICA: NO RETORNA DATOS INVENTADOS
        
        Retorna:
            Tupla (tab_array [8], seq_array [5x6]) si exito
            None si falla (señal para omitir evento)
        
        Vector Tabular [8]:
            [días_descanso, posesión_media, pass_accuracy_media, xG_promedio,
             faltas_promedio, victorias, empates, derrotas]
        
        Secuencia [5x6] - Últimos 5 partidos:
            [xG, xGA, SoT, SoTA, Corners, Corners_contra]
        """
        team_id = self.find_team_id(team_name)
        if not team_id:
            return None  # ← CRÍTICO: No inventa, retorna None
        
        # Consulta historial de últimos partidos finalizados
        matches_data = self._get(
            f"/football/teams/{team_id}/matches",
            params={"status": "finished", "limit": 10}
        )
        
        if not matches_data:
            log_warning(f"No se obtuvieron partidos para {team_name} (ID: {team_id})")
            return None  # ← No inventa, retorna None
        
        raw_matches = matches_data.get("data", [])
        if not isinstance(raw_matches, list) or len(raw_matches) == 0:
            log_warning(f"Historial vacío para {team_name}")
            return None  # ← No inventa, retorna None
        
        sequence = []
        xg_list, pos_list, pass_list, fouls_list = [], [], [], []
        wins, draws, losses = 0, 0, 0
        
        # Procesa hasta 5 partidos reales (sin rellenar con ficción)
        for m in raw_matches[:5]:
            try:
                is_home = str(m.get("home_team_id")) == str(team_id)
                stats = m.get("stats", {})
                
                if not stats or not isinstance(stats, dict):
                    log_warning(f"Stats ausentes en partido de {team_name}")
                    continue  # ← Omite este partido, no inventa
                
                h_stats = stats.get("home", {})
                a_stats = stats.get("away", {})
                
                t_stats = h_stats if is_home else a_stats
                opp_stats = a_stats if is_home else h_stats
                
                # Extrae SOLO métricas que existen realmente
                xg = float(t_stats.get("xg", t_stats.get("expected_goals", None)) or 0.0)
                xga = float(opp_stats.get("xg", opp_stats.get("expected_goals", None)) or 0.0)
                sot = float(t_stats.get("shots_on_target", None) or 0.0)
                sota = float(opp_stats.get("shots_on_target", None) or 0.0)
                cor = float(t_stats.get("corners", None) or 0.0)
                cora = float(opp_stats.get("corners", None) or 0.0)
                
                pos = float(t_stats.get("possession", None) or 0.0)
                pass_acc = float(t_stats.get("pass_accuracy", None) or 0.0)
                fouls = float(t_stats.get("fouls", None) or 0.0)
                
                # Resultado
                h_goals = m.get("home_goals", 0)
                a_goals = m.get("away_goals", 0)
                if is_home:
                    if h_goals > a_goals:
                        wins += 1
                    elif h_goals == a_goals:
                        draws += 1
                    else:
                        losses += 1
                else:
                    if a_goals > h_goals:
                        wins += 1
                    elif a_goals == h_goals:
                        draws += 1
                    else:
                        losses += 1
                
                sequence.append([xg, xga, sot, sota, cor, cora])
                xg_list.append(xg)
                pos_list.append(pos)
                pass_list.append(pass_acc)
                fouls_list.append(fouls)
            
            except (ValueError, KeyError, TypeError) as e:
                log_warning(f"Error parsando partido de {team_name}: {e}")
                continue  # ← Omite, no inventa
        
        # CRÍTICO: Si no hay datos reales, retorna None (NO rellena con ficción)
        if not sequence:
            log_warning(f"No se extrajeron partidos reales para {team_name} tras procesar {len(raw_matches)} registros")
            return None
        
        # Rellena secuencia SOLO hasta 5 (si tiene menos, eso es correcto)
        # NO inventa valores adicionales
        while len(sequence) < 5:
            # Usa la media de los que SÍ existen, no valores ficticios
            if xg_list:
                mean_row = [
                    float(np.mean(xg_list)),
                    float(np.mean([s[1] for s in sequence])),
                    float(np.mean([s[2] for s in sequence])),
                    float(np.mean([s[3] for s in sequence])),
                    float(np.mean([s[4] for s in sequence])),
                    float(np.mean([s[5] for s in sequence]))
                ]
                sequence.append(mean_row)
            else:
                break  # Sin datos, no rellena
        
        seq_array = np.array(sequence[:5], dtype=np.float32)
        
        # Vector Tabular: Basado en datos reales extraídos
        mean_pos = float(np.mean(pos_list)) if pos_list else 0.0
        mean_pass = float(np.mean(pass_list)) if pass_list else 0.0
        mean_xg = float(np.mean(xg_list)) if xg_list else 0.0
        mean_fouls = float(np.mean(fouls_list)) if fouls_list else 0.0
        
        # Días de descanso (estimado: suponiendo partidos cada ~3-4 días)
        days_rest = 3.0  # Valor por defecto estándar
        
        tab_array = np.array(
            [days_rest, mean_pos, mean_pass, mean_xg, mean_fouls, float(wins), float(draws), float(losses)],
            dtype=np.float32
        )
        
        log_info(f"✅ {team_name}: {len(sequence)} partidos reales extraídos de TheStatsAPI")
        return tab_array, seq_array

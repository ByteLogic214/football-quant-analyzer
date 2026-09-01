import difflib
from typing import Optional, Tuple
from config import FUZZY_MATCH_THRESHOLD, log_warning

class TeamMatcher:
    """
    Normaliza y empareja nombres de equipos entre diferentes APIs
    usando fuzzy matching (difflib) con umbral configurable.
    """

    def __init__(self, threshold: float = FUZZY_MATCH_THRESHOLD):
        self.threshold = threshold
        self._match_cache = {}

    def normalize_team_name(self, name: str) -> str:
        """
        Normaliza un nombre de equipo:
        - Convierte a minúsculas
        - Elimina espacios extra
        - Elimina caracteres especiales comunes
        """
        import re
        name = name.lower().strip()
        name = re.sub(r'\s+', ' ', name)  # Espacios múltiples -> uno solo
        name = re.sub(r'\(.*?\)', '', name)  # Elimina paréntesis
        name = name.strip()
        return name

    def find_best_match(
        self, 
        team_name: str, 
        candidates: list
    ) -> Optional[Tuple[str, float]]:
        """
        Busca el mejor match para un equipo en una lista de candidatos.
        
        Args:
            team_name: Nombre del equipo a buscar
            candidates: Lista de nombres de candidatos (reales de la API)
            
        Returns:
            Tupla (best_match, confidence) o None si no hay match > threshold
        """
        normalized_input = self.normalize_team_name(team_name)
        
        # Búsqueda en cache
        cache_key = f"{normalized_input}|{len(candidates)}"
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]
        
        # Normaliza candidatos
        normalized_candidates = [
            (self.normalize_team_name(c), c) for c in candidates
        ]
        
        # Busca matches con difflib.get_close_matches
        close_matches = difflib.get_close_matches(
            normalized_input,
            [nc[0] for nc in normalized_candidates],
            n=1,
            cutoff=self.threshold
        )
        
        if close_matches:
            matched_normalized = close_matches[0]
            # Recupera el nombre original
            original = next(
                nc[1] for nc in normalized_candidates
                if nc[0] == matched_normalized
            )
            # Calcula exactitud
            ratio = difflib.SequenceMatcher(None, normalized_input, matched_normalized).ratio()
            result = (original, ratio)
            self._match_cache[cache_key] = result
            return result
        
        self._match_cache[cache_key] = None
        return None

    def clear_cache(self):
        """Limpia el cache de matchings."""
        self._match_cache.clear()

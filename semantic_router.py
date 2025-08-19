# src/orchestrator/semantic_router.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity sin dependencias externas."""
    if not a or not b:
        return 0.0
    s = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        s += x * y
        na += x * x
        nb += y * y
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(s / ((na ** 0.5) * (nb ** 0.5)))


class SemanticRouter:
    """
    Router semántico local.
    Rankeá agentes por similitud entre:
      - embed(query)
      - embed("name + role + domains + intents")

    Requiere un 'embedder' con:
      - embed_documents(list[str]) -> list[list[float]]
      - embed_query(str) -> list[float]
    """
    def __init__(
        self,
        agents_cfg: List[Dict],
        embedder,
        intents_by_agent: Optional[Dict[str, List[str]]] = None,
    ):
        self._embedder = embedder
        self._names: List[str] = []
        self._texts: List[str] = []

        intents_by_agent = intents_by_agent or {}

        for a in agents_cfg or []:
            name = a.get("name", "").strip()
            if not name:
                continue
            role = a.get("role", "").strip()
            domains = " ".join(a.get("domains") or [])
            intents = " ".join(intents_by_agent.get(name, a.get("intents", [])) or [])
            desc = f"{name}. Rol: {role}. Dominios: {domains}. {intents}".strip()
            self._names.append(name)
            self._texts.append(desc)

        # precalculamos embeddings de descripciones
        self._M: List[List[float]] = self._embedder.embed_documents(self._texts) if self._texts else []

    def topk(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Devuelve [(agent_name, score)] por similitud coseno, descendente."""
        if not self._names or not query.strip():
            return []
        qv = self._embedder.embed_query(query)
        sims: List[Tuple[str, float]] = []
        for name, v in zip(self._names, self._M):
            sims.append((name, _cosine(qv, v)))
        sims.sort(key=lambda t: -t[1])
        return sims[:max(1, int(k))]

    @property
    def agent_names(self) -> List[str]:
        return list(self._names)

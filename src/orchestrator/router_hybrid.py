# src/orchestrator/router_hybrid.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Dict as TDict

from semantic_router import SemanticRouter

@dataclass
class RouteDecision:
    """
    Decisión de enrutamiento del HybridRouter:
      - agent: nombre del agente elegido (o None si no hay)
      - confidence: score final (similitud + boost por evidencia)
      - reason: explicación corta
      - candidates: lista (agente, score) ordenada desc para telemetría/debug
      - features: pistas útiles para el supervisor (p. ej. dominios del agente elegido)
      - dom_hits: hits por dominio (si el probe aportó info)
    """
    agent: Optional[str]
    confidence: float
    reason: str
    candidates: List[Tuple[str, float]]
    features: List[str] = field(default_factory=list)
    dom_hits: Optional[TDict[str, int]] = None

class HybridRouter:
    """
    Router híbrido:
      1) Candidatos por router semántico (local embeddings)
      2) Boost por evidencia de RAG (hits por dominio), provista por 'rag_probe'
    """
    def __init__(
        self,
        cfg: Dict,
        agents_cfg: List[Dict],
        embedder,
        rag_probe,  # callable: (query: str) -> Dict[str, int]
        intents_by_agent: Optional[Dict[str, List[str]]] = None,
    ):
        self.cfg = cfg or {}
        self.agents_cfg = agents_cfg or []
        self.semantic = SemanticRouter(agents_cfg, embedder, intents_by_agent=intents_by_agent)
        self.rag_probe = rag_probe

        rcfg = (self.cfg.get("router") or {})
        self.threshold: float = float(rcfg.get("confidence_threshold", 0.55))
        # pesos/boosts
        self.boost_per_hit: float = float(rcfg.get("evidence_boost_per_hit", 0.05))
        self.max_boost: float = float(rcfg.get("max_evidence_boost", 0.20))
        self.candidates_k: int = int(rcfg.get("candidates_k", 3))

    def _agent_domains(self, name: str) -> List[str]:
        ag = next((a for a in self.agents_cfg if a.get("name") == name), None)
        return list(ag.get("domains") or []) if ag else []

    def route(self, query: str) -> RouteDecision:
        # 1) candidatos semánticos
        sem_cand = self.semantic.topk(query, k=self.candidates_k)  # [(name, sim), ...]
        if not sem_cand:
            return RouteDecision(agent=None, confidence=0.0, reason="Sin candidatos semánticos", candidates=[])

        # 2) evidencia por dominio (p. ej. BM25 hits)
        try:
            dom_hits = self.rag_probe(query) or {}
        except Exception:
            dom_hits = {}

        # 3) combinar: score = sim + min(max_boost, hits*boost_per_hit)
        scored: List[Tuple[str, float]] = []
        for name, sim in sem_cand:
            hits = 0
            for d in self._agent_domains(name):
                hits = max(hits, int(dom_hits.get(d, 0)))
            boost = min(self.max_boost, self.boost_per_hit * hits)
            scored.append((name, float(sim + boost)))

        scored.sort(key=lambda t: -t[1])
        best_name, best_score = scored[0]
        reason = (
            f"sem={sem_cand[0][1]:.2f}, "
            f"evidence_boost={best_score - sem_cand[0][1]:.2f}, "
            f"dom_hits={dom_hits}"
        )
        features = self._agent_domains(best_name)  # pistas explícitas para el handoff

        # 4) umbral & fallback
        if best_score < self.threshold:
            # baja confianza → dejá que 'rag_general' maneje
            return RouteDecision(
                agent="rag_general",
                confidence=best_score,
                reason=reason + " (fallback)",
                candidates=scored,
                features=features,
                dom_hits=dom_hits,
            )

        return RouteDecision(
            agent=best_name,
            confidence=best_score,
            reason=reason,
            candidates=scored,
            features=features,
            dom_hits=dom_hits,
        )

# Conveniencia (por si querés un builder rápido)
def build_hybrid_router(
    cfg: Dict,
    agents_cfg: List[Dict],
    embedder,
    rag_probe,
    intents_by_agent: Optional[Dict[str, List[str]]] = None,
) -> HybridRouter:
    return HybridRouter(cfg, agents_cfg, embedder, rag_probe, intents_by_agent=intents_by_agent)

# src/orchestrator/handoff.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json
import time
from pathlib import Path

@dataclass
class Handoff:
    """
    Representa la 'entrega' del supervisor a un agente:
      - worker: agente elegido
      - reason: explicación corta (por qué)
      - confidence: 0..1 (si no la tenés, podés pasar 0.0 o un proxy)
      - features: señales/rasgos que influyeron (p. ej. dominios del agente, keywords)
      - rules: reglas disparadas (opcional)
      - votes: mapa con puntajes de candidatos (para trazabilidad/telemetría)
      - dom_hits: hits por dominio (si se aportó desde un router híbrido/RAG)
    """
    worker: str
    reason: str
    confidence: float
    features: Optional[List[str]] = None
    rules: Optional[List[str]] = None
    votes: Optional[Dict[str, float]] = None
    dom_hits: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def make_handoff(
    worker: str,
    reason: str,
    confidence: float,
    *,
    features: Optional[List[str]] = None,
    rules: Optional[List[str]] = None,
    votes: Optional[Dict[str, float]] = None,
    dom_hits: Optional[Dict[str, int]] = None,
) -> Handoff:
    return Handoff(
        worker=worker,
        reason=reason,
        confidence=confidence,
        features=features,
        rules=rules,
        votes=votes,
        dom_hits=dom_hits,
    )

def log_handoff(handoff: Handoff, trace_id: Optional[str] = None, path: str = "cache/traces.jsonl") -> None:
    """
    Persiste el handoff como JSONL para auditoría básica.
    Best-effort: si falla, no rompe el flujo.
    """
    rec = {
        "ts": int(time.time()),
        "trace_id": trace_id,
        "handoff": handoff.to_dict(),
    }
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
##comentario 
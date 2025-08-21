# src/orchestrator/evidence.py
from __future__ import annotations
from typing import Dict, List, Optional
from contextvars import ContextVar
from threading import Lock

# contextvar con el trace_id actual (opcional)
_CURRENT_TRACE_ID: ContextVar[Optional[str]] = ContextVar("_CURRENT_TRACE_ID", default=None)

# Evidencia recolectada por trace_id
#   trace_id -> List[{"text": str, "source": str, "score": float|None, "domain": str|None}]
_EVIDENCE: Dict[str, List[dict]] = {}
_LOCK = Lock()

# ----------------- API pública ----------------- #
def set_current_trace_id(trace_id: Optional[str]) -> None:
    _CURRENT_TRACE_ID.set(trace_id)

def get_current_trace_id() -> Optional[str]:
    return _CURRENT_TRACE_ID.get()

def record_evidence(results: List[dict], trace_id: Optional[str] = None) -> None:
    """
    Agrega resultados de RAG al bucket del trace dado.
    Si trace_id es None, usa el contextvar actual; si tampoco hay, no hace nada.
    """
    tid = trace_id or get_current_trace_id()
    if not tid or not results:
        return
    with _LOCK:
        bucket = _EVIDENCE.setdefault(tid, [])
        bucket.extend(results)

def get_evidence(trace_id: Optional[str] = None) -> List[dict]:
    """
    Devuelve la evidencia almacenada para ese trace_id (o el actual).
    """
    tid = trace_id or get_current_trace_id()
    if not tid:
        return []
    with _LOCK:
        return list(_EVIDENCE.get(tid, []))

def clear_evidence(trace_id: Optional[str] = None) -> None:
    tid = trace_id or get_current_trace_id()
    if not tid:
        return
    with _LOCK:
        _EVIDENCE.pop(tid, None)

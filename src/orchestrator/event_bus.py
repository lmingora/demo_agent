# src/orchestrator/event_bus.py
from __future__ import annotations
from typing import List, Optional
import warnings

# Usamos evidence.py como fuente de verdad (thread-safe + ContextVar)
from .evidence import (
    set_current_trace_id as _set_current_trace_id,
    get_current_trace_id as _get_current_trace_id,
    record_evidence as _record_evidence_core,
    get_evidence as _get_evidence_core,
)

warnings.warn(
    "src.orchestrator.event_bus está deprecado. Usa src.orchestrator.evidence como API canónica.    ",
    DeprecationWarning,
    stacklevel=2,
)

# ----------------- API pública (compat) ----------------- #
def set_current_trace_id1(tid: Optional[str]) -> None:
    _set_current_trace_id(tid)
    
def set_current_trace_id(tid: Optional[str]) -> None:
    _set_current_trace_id(tid)

def get_current_trace_id() -> Optional[str]:
    return _get_current_trace_id()

def record_evidence(trace_id: Optional[str], results: List[dict]) -> None:
    """
    Compatibilidad con la firma antigua de event_bus:
    - event_bus: record_evidence(trace_id, results)
    - evidence:  record_evidence(results, trace_id=None)
    """
    if not results:
        return
    _record_evidence_core(results, trace_id=trace_id)

def get_evidence_results(trace_id: Optional[str]) -> List[dict]:
    """
    Compat: en event_bus se llamaba get_evidence_results, en evidence es get_evidence.
    """
    return _get_evidence_core(trace_id)

def get_evidence_context_md(trace_id: Optional[str], max_chars: int = 4000) -> str:
    """
    Genera un Markdown breve con la evidencia para ese trace.
    Se mantiene aquí para no forzar cambios en call-sites actuales.
    """
    res = _get_evidence_core(trace_id)
    if not res:
        return ""
    lines, used = [], 0
    for i, r in enumerate(res, 1):
        src = (r.get("source") or "unknown")
        txt = (r.get("text") or "").strip().replace("\n", " ")
        piece = f"- [{i}] ({src}) {txt}"
        if used + len(piece) > max_chars:
            break
        lines.append(piece)
        used += len(piece)
    return "\n".join(lines)

# src/orchestrator/evidence.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from contextvars import ContextVar
from threading import RLock

# Trace actual por contexto (thread/coroutine-safe)
_CURRENT_TRACE_ID: ContextVar[Optional[str]] = ContextVar("_CURRENT_TRACE_ID", default=None)

# Evidencia por trace_id (memoria de proceso)
_EVIDENCE: Dict[str, List[dict]] = {}
_LOCK = RLock()

# Límite de elementos por trace para evitar crecimiento descontrolado
_MAX_ITEMS_PER_TRACE = 500  # ajustable

# ----------------- API pública ----------------- #

def set_current_trace_id(trace_id: Optional[str]) -> None:
    _CURRENT_TRACE_ID.set(trace_id)

def get_current_trace_id() -> Optional[str]:
    return _CURRENT_TRACE_ID.get()

def record_evidence(results: List[dict], trace_id: Optional[str] = None) -> None:
    """
    Agrega resultados de RAG al bucket del trace dado.
    Si trace_id es None, usa el contextvar actual; si tampoco hay, no hace nada.
    Dedup liviano por (source, text[:120]) y cap por trace.
    """
    tid = trace_id or get_current_trace_id()
    if not tid or not results:
        return

    # Deduplicación simple antes de guardar
    seen: set[Tuple[str, str]] = set()
    cleaned: List[dict] = []
    for r in results:
        src = (r.get("source") or "unknown")
        txt = ((r.get("text") or "").strip().replace("\n", " "))[:120]
        key = (src, txt)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(r)

    if not cleaned:
        return

    with _LOCK:
        bucket = _EVIDENCE.setdefault(tid, [])
        bucket.extend(cleaned)
        # recorte para mantener como mucho N
        if len(bucket) > _MAX_ITEMS_PER_TRACE:
            del bucket[:-_MAX_ITEMS_PER_TRACE]

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
    """
    Limpia la evidencia para ese trace_id (o el actual).
    """
    tid = trace_id or get_current_trace_id()
    if not tid:
        return
    with _LOCK:
        _EVIDENCE.pop(tid, None)

def get_evidence_context_md(trace_id: Optional[str] = None, max_chars: int = 4000) -> str:
    """
    Devuelve la evidencia resumida en Markdown (bullets) para mostrar en UI/logs.
    """
    res = get_evidence(trace_id)
    if not res:
        return ""
    lines, used = [], 0
    for i, r in enumerate(res, 1):
        src = (r.get("source") or "unknown")
        txt = (r.get("text") or "").strip().replace("\n", " ")
        piece = f"- [{i}] ({src}) {txt}"
        if used + len(piece) > max_chars:
            break
        lines.append(piece); used += len(piece)
    return "\n".join(lines)

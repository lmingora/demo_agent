# src/orchestrator/event_bus.py
from __future__ import annotations
from typing import Dict, List

# Memoria en-proceso por trace_id
_EVIDENCE_BY_TRACE: Dict[str, List[dict]] = {}
_CURRENT_TRACE_ID: str | None = None

def set_current_trace_id(tid: str | None) -> None:
    global _CURRENT_TRACE_ID
    _CURRENT_TRACE_ID = tid

def get_current_trace_id() -> str | None:
    return _CURRENT_TRACE_ID

def record_evidence(trace_id: str | None, results: List[dict]) -> None:
    if not trace_id or not results:
        return
    _EVIDENCE_BY_TRACE.setdefault(trace_id, []).extend(results)

def get_evidence_results(trace_id: str | None) -> List[dict]:
    if not trace_id:
        return []
    return list(_EVIDENCE_BY_TRACE.get(trace_id, []))

def get_evidence_context_md(trace_id: str | None, max_chars: int = 4000) -> str:
    res = get_evidence_results(trace_id)
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

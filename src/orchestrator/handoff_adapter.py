# src/orchestrator/handoff_adapter.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import re

from src.orchestrator.handoff import Handoff, make_handoff, log_handoff

_THOUGHT_RE = re.compile(r"^Thought:\s*(.+)$", re.IGNORECASE)

def extract_reason_from_supervisor_output(text: str) -> str:
    """
    Intenta extraer la razón desde una línea 'Thought: <...>'.
    Devuelve cadena vacía si no matchea.
    """
    if not isinstance(text, str):
        return ""
    for line in text.splitlines():
        m = _THOUGHT_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return ""

def build_handoff_from_transfer(
    *,
    agent_name: str,
    supervisor_output_text: Optional[str] = None,
    confidence: float = 0.0,
    features: Optional[List[str]] = None,
    rules: Optional[List[str]] = None,
    votes: Optional[Dict[str, float]] = None,
    dom_hits: Optional[Dict[str, int]] = None,
) -> Handoff:
    """
    Construye un Handoff cuando ya sabés que el supervisor transfirió a `agent_name`.
    Si pasás `supervisor_output_text`, tratará de extraer el 'Thought:' como reason.
    """
    reason = extract_reason_from_supervisor_output(supervisor_output_text or "") or "supervisor transfer"
    return make_handoff(
        worker=agent_name,
        reason=reason,
        confidence=confidence,
        features=features,
        rules=rules,
        votes=votes,
        dom_hits=dom_hits,
    )

def inject_handoff_system_message(messages: list, handoff: Handoff) -> None:
    """
    Inserta una SystemMessage con la razón del handoff.
    Si LangChain no está disponible, usa un dict {'role':'system', 'content':...}.
    """
    msg_text = f'[HANDOFF_REASON] worker={handoff.worker} conf={handoff.confidence:.2f} reason="{handoff.reason}"'
    try:
        from langchain_core.messages import SystemMessage
        messages.insert(0, SystemMessage(content=msg_text))
    except Exception:
        messages.insert(0, {"role": "system", "content": msg_text})

def maybe_log_handoff(handoff: Handoff, trace_id: Optional[str]) -> None:
    """
    Persistencia best-effort (no interrumpe el flujo si falla).
    """
    try:
        log_handoff(handoff, trace_id=trace_id)
    except Exception:
        pass

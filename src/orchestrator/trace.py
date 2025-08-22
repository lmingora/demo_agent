# src/orchestrator/trace.py
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage, SystemMessage

from src.rag.toolbox import rag_probe_counts, list_domains


# ------------------------ Trace IDs & Hints ------------------------ #
def make_trace_id() -> str:
    """ID corto para trazar un turno."""
    return uuid4().hex[:12]


def trace_header_message(trace_id: str) -> SystemMessage:
    """Header de traza que viaja en el prompt."""
    return SystemMessage(content=f"[TRACE_ID] {trace_id}")


def router_hint_message(user_text: str) -> Optional[SystemMessage]:
    """
    Pista barata para el supervisor: cuenta hits por dominio con top-k chico.
    No decide la ruta; solo da contexto.
    """
    try:
        doms = list_domains()
        hits: Dict[str, int] = rag_probe_counts(query=user_text, domains=doms, k=3) or {}
    except Exception:
        hits = {}
    if not hits:
        return None
    hint = "[ROUTER_HINT] " + ", ".join(f"{k}={v}" for k, v in hits.items())
    return SystemMessage(content=hint)


# ------------------------ Helpers de texto ------------------------ #
def extract_text_from_messages(messages: List[BaseMessage]) -> str:
    if not messages:
        return ""
    m = messages[-1]
    content = getattr(m, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                txt = p.get("text") or p.get("content")
                if txt:
                    parts.append(txt)
            elif isinstance(p, str):
                parts.append(p)
        if parts:
            return "\n".join(parts)
    try:
        return m[1]  # compat con backends raros
    except Exception:
        return str(m)


def peek_node_preview(node_payload: dict, max_len: int = 120) -> str:
    msgs = node_payload.get("messages", [])
    txt = extract_text_from_messages(msgs)
    txt = txt.replace("\n", " ").strip()
    return (txt[:max_len] + "…") if len(txt) > max_len else txt


# ------------------------ Supervisor routing parse ------------------------ #
_RE_TRANSFER = re.compile(
    r"(?:transfer_to|handoff_to)_([A-Za-z0-9_\-]+)(?:\s*\(\s*reason\s*=\s*['\"](.*?)['\"]\s*\))?",
    re.S,
)


_RE_THOUGHT = re.compile(r"Thought:\s*(.+)")

def parse_supervisor_decision(text: str) -> Tuple[str | None, str | None]:
    """
    Extrae (agent, reason) desde:
      - Action: transfer_to_<agente>  (obligatorio)
      - reason: si viene inline (transfer_to_xxx(reason='...')) o,
                si no, tomamos el contenido del 'Thought:' (1 línea).
    """
    if not text:
        return None, None

    agent = None
    reason = None

    m = _RE_TRANSFER.search(text)
    if m:
        agent = m.group(1) or None
        # si venía inline (no lo esperamos, pero soportamos)
        if m.lastindex and m.group(2):
            reason = m.group(2)

    if reason is None:
        # fallback: tomar la línea del Thought
        mt = _RE_THOUGHT.search(text)
        if mt:
            # una sola línea, ya trimmed
            reason = mt.group(1).strip()

    return agent, reason

# ------------------------ Runner con stream ------------------------ #
def run_with_trace(app, payload, thread_id: str):
    """
    Devuelve (res, last_agent, routed_to)
    - res: último chunk de un agente (no supervisor) o el último visto
    - last_agent: nombre del último agente ejecutado
    - routed_to: a quién dijo el supervisor que debía ir
    """
    trace_on = logging.getLogger("app").getEffectiveLevel() <= logging.DEBUG
    last_agent = None
    routed_to = None
    final_any = None
    final_non_supervisor = None

    try:
        for chunk in app.stream(payload, config={"configurable": {"thread_id": thread_id}}):
            for node_name, data in chunk.items():
                if node_name in ("__start__", "__end__"):
                    continue

                preview = peek_node_preview(data)

                if node_name == "supervisor":
                    agent, reason = parse_supervisor_decision(preview)
                    if agent:
                        routed_to = agent
                    if trace_on:
                        if agent and reason:
                            print(f"[route] supervisor → {agent} (motivo: {reason})")
                        elif agent:
                            print(f"[route] supervisor → {agent}")
                        else:
                            print(f"[route] supervisor → (sin orden válida) :: {preview}")
                else:
                    last_agent = node_name
                    final_non_supervisor = data
                    if trace_on:
                        print(f"[trace] node={node_name} :: {preview}")

                final_any = data

        return (final_non_supervisor or final_any), last_agent, routed_to

    except AttributeError:
        # Fallback para compilaciones sin .stream
        res = app.invoke(payload, config={"configurable": {"thread_id": thread_id}})
        return res, last_agent, None

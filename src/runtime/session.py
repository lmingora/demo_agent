# src/runtime/session.py
from __future__ import annotations
import uuid
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage

from src.memory.store import get_store
from src.rag.toolbox import rag_probe_counts


class Session:
    """
    Estado mínimo por sesión/CLI: user_id, thread_id y trace_id por turno.
    No setea el trace_id global aquí; eso lo hace main antes de invocar el grafo.
    """
    def __init__(self, cfg: Dict, user_id: str = "local", thread_id: str = "cli"):
        self.cfg = cfg
        self.user_id = user_id
        self.thread_id = thread_id
        self.trace_id: str | None = None

    def next_trace_id(self) -> str:
        self.trace_id = uuid.uuid4().hex[:8]
        return self.trace_id

    def _router_hint_system_message(self, user_text: str) -> SystemMessage | None:
        dom_hits = rag_probe_counts(query=user_text, domains=None, k=3) or {}
        if not dom_hits:
            return None
        hint = "[ROUTER_HINT] " + ", ".join(f"{k}={v}" for k, v in dom_hits.items())
        return SystemMessage(content=hint)

    def build_payload(self, user_text: str) -> dict:
        """
        Inserta [TRACEID], [ROUTER_HINT] y [MEMORIA] cuando corresponde.
        """
        msgs = []
        if self.trace_id:
            msgs.append(SystemMessage(content=f"[TRACEID] {self.trace_id}"))

        hint = self._router_hint_system_message(user_text)
        if hint:
            msgs.append(hint)

        # bloque de memoria relevante
        try:
            store = get_store()
            top_k = int((self.cfg.get("memory") or {}).get("top_k", 5))
            mem_block = store.top_text_block(user_id=self.user_id, query=user_text, top_k=top_k)
        except Exception:
            mem_block = ""
        if mem_block:
            msgs.append(SystemMessage(content=f"[MEMORIA]\n{mem_block}"))

        msgs.append(HumanMessage(content=user_text))
        return {"messages": msgs}

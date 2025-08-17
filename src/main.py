# src/main.py
from __future__ import annotations

# --- bootstrap para permitir `python src/main.py` o `python -m src.main` ---
# --- bootstrap robusto para permitir `python -m src.main` y `python src/main.py` ---
import sys
from pathlib import Path

import warnings
warnings.filterwarnings(
    "ignore",
    message="Using default key encoder: SHA-1 is \\*not\\* collision-resistant",
)


ROOT = Path(__file__).resolve().parents[1]   # <- /.../demo_agent
if str(ROOT) not in sys.path:
    # lo insertamos siempre al inicio, sin condicionales con __package__
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------------------------

# ---------------------------------------------------------------------------

import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from config.settings import load_cfg
from src.cli.interactive import (
    banner, handle_cmd_index, handle_cmd_agents, handle_cmd_domains,
    handle_cmd_help, set_trace, handle_cmd_index_docs
)
from src.rag.toolbox import init_rag_tooling
from src.orchestrator.lg_supervisor import (
    build_app as build_supervisor_app,
    get_agent_graph, get_agent_names
)
from src.utils.logging import get_logger

log = get_logger("main")


# ------------------------ Helpers de impresión / trazas --------------------- #
def _extract_text_from_messages(messages) -> str:
    """Devuelve el texto imprimible del último mensaje LangChain/Graph."""
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
        # compat si algún backend dejó tuplas
        return m[1]
    except Exception:
        return str(m)


def _peek_node_preview(node_payload: dict, max_len: int = 120) -> str:
    """Devuelve un preview de texto del último mensaje en el nodo."""
    msgs = node_payload.get("messages", [])
    txt = _extract_text_from_messages(msgs)
    txt = txt.replace("\n", " ").strip()
    return (txt[:max_len] + "…") if len(txt) > max_len else txt


def _parse_supervisor_decision(text: str):
    """Extrae (agent, reason) de:
       transfer_to_<agent>(reason='...')  o  handoff_to_<agent>(...)
       Devuelve (None, None) si no matchea.
    """
    if not text:
        return None, None
    m = re.search(r"transfer_to_([A-Za-z0-9_\-]+)\s*\(\s*reason\s*=\s*['\"](.*?)['\"]\s*\)", text)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"transfer_to_([A-Za-z0-9_\-]+)", text)
    if m:
        return m.group(1), ""
    m = re.search(r"handoff_to_([A-Za-z0-9_\-]+)", text)
    if m:
        return m.group(1), ""
    return None, None


def _run_with_trace(app, payload, thread_id: str):
    """Ejecuta el grafo con stream; imprime trazas si :trace está ON.
    Devuelve (res, last_agent, routed_to):
      - res: último chunk de un agente (no supervisor) si existe, o el último visto
      - last_agent: último nodo-agente ejecutado
      - routed_to: agente que el supervisor dijo (parseado) si se pudo extraer
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

                preview = _peek_node_preview(data)

                if node_name == "supervisor":
                    agent, reason = _parse_supervisor_decision(preview)
                    if agent:
                        routed_to = agent
                    if trace_on:
                        if agent:
                            if reason:
                                print(f"[route] supervisor → {agent} (motivo: {reason})")
                            else:
                                print(f"[route] supervisor → {agent}")
                        else:
                            print(f"[route] supervisor → (sin orden válida) :: {preview}")
                else:
                    # Guardamos qué agente corrió y su último chunk
                    last_agent = node_name
                    final_non_supervisor = data
                    if trace_on:
                        print(f"[trace] node={node_name} :: {preview}")

                final_any = data

        # Preferimos devolver el último chunk NO-supervisor
        return (final_non_supervisor or final_any), last_agent, routed_to

    except AttributeError:
        # Fallback para compilaciones sin .stream
        res = app.invoke(payload, config={"configurable": {"thread_id": thread_id}})
        return res, last_agent, None
# --------------------------------------------------------------------------- #


def main():
    cfg = load_cfg()
    print(banner())

    # Inicializa vectorstore, BM25 por dominio y tool RAG
    init_rag_tooling(cfg)

    # Orquestación (Supervisor + ReAct agents + handoffs)
    app = build_supervisor_app(cfg)

    # Estado CLI
    current_thread = "cli"

    # CLI loop
    while True:
        try:
            msg = input("Pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not msg:
            continue
        low = msg.lower()

        # Comandos
        if low in {"exit", "quit"}:
            break
        if low.startswith(":help"):
            print(handle_cmd_help()); continue
        if low.startswith(":agents"):
            print(handle_cmd_agents(cfg)); continue
        if low.startswith(":domains"):
            print(handle_cmd_domains()); continue
        if low.startswith(":index_docs"):
            print(handle_cmd_index_docs(cfg, msg)); continue
        if low.startswith(":index"):
            print(handle_cmd_index(cfg, msg)); continue
        if low.startswith(":thread"):
            parts = msg.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                current_thread = parts[1].strip()
                print(f"thread_id = {current_thread}")
            else:
                print("Uso: :thread <id>")
            continue
        if low.startswith(":trace"):
            parts = msg.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                print(set_trace(parts[1] == "on"))
            else:
                print("Uso: :trace on|off")
            continue

        # Invocación normal al supervisor (con memoria por thread_id si está habilitado)
         # Invocación normal al supervisor (con memoria por thread_id si está habilitado)
        payload = {"messages": [HumanMessage(content=msg)]}
        res, last_agent, routed_to = _run_with_trace(app, payload, current_thread)

        # --- Fallback/Dispatch robusto ---
        # Si el supervisor eligió un agente (routed_to) pero NO se ejecutó ningún agente (last_agent is None),
        # despachamos nosotros directamente al grafo del agente elegido.
        known = set(get_agent_names())
        if routed_to and last_agent is None:
            target = routed_to if routed_to in known else "rag_general"
            g = get_agent_graph(target)
            if g is not None:
                try:
                    fres = g.invoke(payload, config={"configurable": {"thread_id": current_thread}})
                    messages = fres.get("messages", []) if isinstance(fres, dict) else []
                    answer = _extract_text_from_messages(messages) or str(fres)
                    print(f"[dispatch→{target}] {answer}")
                    continue
                except Exception as fe:
                    print(f"[dispatch error→{target}] {fe}")
                    # si falla el dispatch, seguimos abajo e imprimimos lo que haya

        # --- Respuesta normal (cuando sí hubo agente o no hubo routed_to) ---
        messages = res.get("messages", []) if isinstance(res, dict) else []
        answer = _extract_text_from_messages(messages) or str(res)
        if last_agent:
            print(f"[agente={last_agent}] {answer}")
        else:
            print(answer)


    return 0


if __name__ == "__main__":
    sys.exit(main())

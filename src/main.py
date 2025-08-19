# src/main.py
from __future__ import annotations

# --- bootstrap para permitir `python -m src.main` o `python src/main.py` ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------------

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Using default key encoder: SHA-1 is \*not\* collision-resistant",
)
import logging
import re
import json
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import load_cfg, load_agents_cfg

from src.cli.interactive import (
    banner, handle_cmd_index, handle_cmd_agents, handle_cmd_domains,
    handle_cmd_help, set_trace, handle_cmd_index_docs
)

from src.utils.logging import get_logger
from src.llm.factory import make_chat

# RAG / Tools
from src.rag.toolbox import (
    init_rag_tooling, rag_search,    # seguimos usando rag_search
    rag_probe_counts                 # <- usamos esto para el hint del supervisor
)
from src.agents.tools.structure_tools import summarize_evidence

# Orquestación
from src.orchestrator.lg_supervisor import (
    build_app as build_supervisor_app,
    get_agent_graph, get_agent_names
)

# Memoria
from src.memory.store import init_memory, get_store  # <- importante: import get_store

log = get_logger("main")

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*[\r\n]+Action Input:\s*(\{.*?\})", re.S)

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
        return m[1]  # compat si algún backend dejó tuplas
    except Exception:
        return str(m)


def _peek_node_preview(node_payload: dict, max_len: int = 120) -> str:
    """Devuelve un preview de texto del último mensaje en el nodo."""
    msgs = node_payload.get("messages", [])
    txt = _extract_text_from_messages(msgs)
    txt = txt.replace("\n", " ").strip()
    return (txt[:max_len] + "…") if len(txt) > max_len else txt


def _parse_supervisor_decision(text: str):
    """Extrae (agent, reason) de transfer_to_<agent>(reason='...') / handoff_to_<agent>(...)."""
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
    """Stream del grafo (si existe). Devuelve (res, last_agent, routed_to)."""
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


# ------------------------ Hint para el supervisor --------------------------- #
def _router_hint_system_message(user_text: str) -> SystemMessage | None:
    """
    Pista barata para el supervisor, NO decide ruta:
    cuenta hits por dominio con un top-k chico y lo inyecta como SystemMessage.
    """
    dom_hits = rag_probe_counts(query=user_text, domains=None, k=3) or {}
    if not dom_hits:
        return None
    hint = "[ROUTER_HINT] " + ", ".join(f"{k}={v}" for k, v in dom_hits.items())
    return SystemMessage(content=hint)


# ------------------------ Forzado de tools (anti-alucinación) --------------- #
def _try_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(s.replace("'", '"'))

def _results_to_context_md(results: list[dict], max_chars: int = 4000) -> str:
    """
    Construye un 'context_md' (markdown) a partir de rag_search.results.
    Limita tamaño para no inflar el prompt de cierre.
    """
    if not results:
        return ""
    lines = []
    used = 0
    for i, r in enumerate(results, 1):
        src = r.get("source", "unknown")
        txt = (r.get("text") or "").strip().replace("\n", " ")
        piece = f"- [{i}] ({src}) {txt}"
        if used + len(piece) > max_chars:
            break
        lines.append(piece)
        used += len(piece)
    return "\n".join(lines)

def _maybe_force_tool_execution(answer_text: str, user_text: str, cfg: dict) -> str | None:
    """
    Si el modelo escribió 'Action: rag_search' como texto (sin ejecutar tool),
    ejecutamos rag_search -> summarize_evidence y pedimos al LLM que cierre.
    Devuelve texto final o None si no hizo falta intervenir.
    """
    m = _ACTION_RE.search(answer_text or "")
    if not m:
        return None

    tool, argstr = m.groups()
    if tool != "rag_search":
        return None

    try:
        args = _try_json(argstr)
    except Exception:
        args = {}

    q = (args.get("query") or user_text or "").strip()
    domains = args.get("domains") or ["general"]
    try:
        k = int(args.get("k", cfg.get("retrieval", {}).get("k", 8)))
    except Exception:
        k = cfg.get("retrieval", {}).get("k", 8)

    # 1) Ejecutar RAG
    obs = rag_search.invoke({"query": q, "domains": domains, "k": k}) or {}
    results = obs.get("results") or []
    if not results:
        return ("No encontré evidencia local para esa consulta. "
                "Decime servicio/fecha y vuelvo a intentar con más contexto.")

    # 2) Construir context_md y resumir evidencia
    context_md = _results_to_context_md(results)
    summ = summarize_evidence.invoke({"context_md": context_md}) or ""

    # 3) Cierre con LLM (formato tipo incidente)
    llm = make_chat(cfg)
    close_prompt = (
        "Usá el siguiente resumen de evidencia para responder con el formato:\n"
        "- Cronología breve (≤5)\n- Causa raíz (hipótesis)\n- Mitigaciones (3–5 bullets)\n- Fuentes\n\n"
        f"Resumen:\n{summ}\n"
    )
    out = llm.invoke([HumanMessage(content=close_prompt)])
    return getattr(out, "content", None) or str(out)


# ------------------------ Payload con memoria + hint ------------------------- #
def _build_payload_with_memory(cfg: dict, user_text: str, user_id: str, thread_id: str) -> dict:
    """
    Inserta bloque [ROUTER_HINT] y [MEMORIA] si corresponde.
    Usa get_store() inicializado en main().
    """
    msgs = []

    # 1) Hint para el supervisor (no decisor).
    hint = _router_hint_system_message(user_text)
    if hint:
        msgs.append(hint)

    # 2) Bloque de memoria relevante.
    try:
        store = get_store()
        top_k = int((cfg.get("memory") or {}).get("top_k", 5))
        mem_block = store.top_text_block(user_id=user_id, query=user_text, top_k=top_k)
    except Exception:
        mem_block = ""
    if mem_block:
        msgs.append(SystemMessage(content=f"[MEMORIA]\n{mem_block}"))

    # 3) Mensaje del usuario
    msgs.append(HumanMessage(content=user_text))

    return {"messages": msgs}


# ------------------------ Programa principal ---------------------------------
def main():
    # 1) Cargar configuración completa
    cfg = load_cfg()

    # 2) Inicializar memoria (usará cfg["memory"] si existe)
    init_memory(cfg)

    # 3) Cargar definición de agentes (lista)
    agents_cfg = load_agents_cfg()

    # 4) Inicializar vectorstore/BM25 y tool RAG
    init_rag_tooling(cfg)

    # 5) Orquestación (Supervisor + ReAct agents + handoffs)
    app = build_supervisor_app(cfg)

    current_thread = "cli"
    print("Pipeline LangGraph (Supervisor) — (:help para ver comandos, incluye :index_docs)")

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

        # --- Payload con hint + memoria ---
        payload = _build_payload_with_memory(cfg, msg, user_id="local", thread_id=current_thread)

        # --- SIEMPRE Supervisor (sin pre-router como decisor) ---
        res, last_agent, routed_to = _run_with_trace(app, payload, current_thread)

        # Fallback: el supervisor decidió pero el worker no corrió
        known = set(get_agent_names())
        if routed_to and last_agent is None:
            target = routed_to if routed_to in known else "rag_general"
            g = get_agent_graph(target)
            if g is not None:
                try:
                    fres = g.invoke(payload, config={"configurable": {"thread_id": current_thread}})
                    messages = fres.get("messages", []) if isinstance(fres, dict) else []
                    answer = _extract_text_from_messages(messages) or str(fres)

                    # Forzado de tools (2/3)
                    forced = _maybe_force_tool_execution(answer, msg, cfg)
                    if forced:
                        print(f"[forced-tools→{target}] {forced}")
                        continue

                    print(f"[dispatch→{target}] {answer}")
                    continue
                except Exception as fe:
                    print(f"[dispatch error→{target}] {fe}")

        # Respuesta normal
        messages = res.get("messages", []) if isinstance(res, dict) else []
        answer = _extract_text_from_messages(messages) or str(res)

        # Forzado de tools (3/3)
        forced = _maybe_force_tool_execution(answer, msg, cfg)
        if forced:
            print(f"[forced-tools] {forced}")
            continue

        if last_agent:
            print(f"[agente={last_agent}] {answer}")
        else:
            print(answer)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# src/main.py
from __future__ import annotations

# Anti-alucinación (forzado + saneo de fuentes locales)
from src.orchestrator.anti_hallucination import (
    maybe_force_tool_execution,
    rewrite_sources_to_local,
)

from src.orchestrator.evidence import get_evidence

# --- bootstrap para permitir `python -m src.main` o `python src/main.py` ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Using default key encoder: SHA-1 is \*not\* collision-resistant",
)

import logging
from typing import Any, Dict
import os

# Config
from config.settings import load_cfg, load_agents_cfg

# CLI helpers
from src.cli.interactive import (
    banner, handle_cmd_index, handle_cmd_agents, handle_cmd_domains,
    handle_cmd_help, set_trace, handle_cmd_index_docs
)

# RAG init
from src.rag.toolbox import init_rag_tooling

# Orquestación (Supervisor + workers ReAct)
from src.orchestrator.lg_supervisor import (
    build_app as build_supervisor_app,
    get_agent_graph, get_agent_names
)

# Memoria
from src.memory.store import init_memory

# Estado de sesión + trazas
from src.runtime.session import Session
from src.orchestrator.trace import (
    run_with_trace,
    extract_text_from_messages,
)

# Silenciar telemetría/ruido de librerías (Chroma, Posthog, urllib3)
for _name in ("chromadb.telemetry", "posthog", "urllib3", "urllib3.connectionpool"):
    logging.getLogger(_name).setLevel(logging.ERROR)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY_ANONYMOUS", "false")


# -----------------------------------------------------------------------------
def main() -> int:
    # 1) Config global
    cfg: Dict[str, Any] = load_cfg()

    # 2) Memoria (usa cfg["memory"] si existe)
    init_memory(cfg)

    # 3) Definición de agentes (lista) — útil para CLI (:agents)
    _ = load_agents_cfg()

    # 4) RAG stack (vectorstore + BM25 + tools)
    init_rag_tooling(cfg)

    # 5) Supervisor compilado (prompt con whitelist + few-shots dentro de lg_supervisor.py)
    app = build_supervisor_app(cfg)

    # 6) Estado de sesión (multiusuario + thread + trace_id por turno)
    sess = Session(cfg=cfg, user_id="local", thread_id="cli")

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
        # ---- Comandos de la CLI ----
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
                sess.thread_id = parts[1].strip()
                print(f"thread_id = {sess.thread_id}")
            else:
                print("Uso: :thread <id>")
            continue
        if low.startswith(":user"):
            parts = msg.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                sess.user_id = parts[1].strip()
                print(f"user_id = {sess.user_id}")
            else:
                print("Uso: :user <id>")
            continue
        if low.startswith(":trace"):
            parts = msg.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                print(set_trace(parts[1] == "on"))
            else:
                print("Uso: :trace on|off")
            continue

        # ---- TraceId por turno + payload con hint/memoria/trace ----
        trace_id = sess.next_trace_id()           # ← Session ya setea el trace en el contextvar
        payload = sess.build_payload(msg)

        # ---- Supervisor normal (sin pre-router como decisor) ----
        res, last_agent, routed_to = run_with_trace(app, payload, sess.thread_id)

        # Fallback: supervisor decidió pero el worker no corrió
        known = set(get_agent_names())
        if routed_to and last_agent is None:
            target = routed_to if routed_to in known else "rag_general"
            g = get_agent_graph(target)
            if g is not None:
                try:
                    fres = g.invoke(payload, config={"configurable": {"thread_id": sess.thread_id}})
                    messages = fres.get("messages", []) if isinstance(fres, dict) else []
                    answer = extract_text_from_messages(messages) or str(fres)

                    # Anti-alucinación: si imprimió "Action: rag_search" en texto, forzamos tools
                    forced = maybe_force_tool_execution(answer, msg, cfg)
                    if forced:
                        # Reescribe/inyecta fuentes LOCALES reales del trace actual
                        forced = rewrite_sources_to_local(forced, get_evidence(sess.trace_id))
                        print(f"[forced-tools→{target}] {forced}")
                        continue

                    # Saneamos fuentes (solo locales, basadas en evidencia real)
                    answer = rewrite_sources_to_local(answer, get_evidence(sess.trace_id))
                    print(f"[dispatch→{target}] {answer}")
                    continue
                except Exception as fe:
                    print(f"[dispatch error→{target}] {fe}")

        # ---- Respuesta normal ----
        messages = res.get("messages", []) if isinstance(res, dict) else []
        answer = extract_text_from_messages(messages) or str(res)

        # Anti-alucinación (forzado tools)
        forced = maybe_force_tool_execution(answer, msg, cfg)
        if forced:
            forced = rewrite_sources_to_local(forced, get_evidence(sess.trace_id))
            print(f"[forced-tools] {forced}")
            continue

        # Saneador final: quitar URLs externas/“inventadas” y citar SOLO evidencia local
        answer = rewrite_sources_to_local(answer, get_evidence(sess.trace_id))

        if last_agent:
            print(f"[agente={last_agent}] {answer}")
        else:
            print(answer)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

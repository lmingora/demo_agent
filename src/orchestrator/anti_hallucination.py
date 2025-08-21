# src/orchestrator/anti_hallucination.py
from __future__ import annotations
import json, re
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from src.llm.factory import make_chat
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence
from src.orchestrator.evidence import get_evidence  # ← evidencia real por trace_id

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*[\r\n]+Action Input:\s*(\{.*?\})", re.S)

def _try_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(s.replace("'", '"'))

def _results_to_context_md(results: List[dict], max_chars: int = 4000) -> str:
    if not results:
        return ""
    lines, used = [], 0
    for i, r in enumerate(results, 1):
        src = r.get("source", "unknown")
        txt = (r.get("text") or "").strip().replace("\n", " ")
        piece = f"- [{i}] ({src}) {txt}"
        if used + len(piece) > max_chars:
            break
        lines.append(piece); used += len(piece)
    return "\n".join(lines)

def maybe_force_tool_execution(answer_text: str, user_text: str, cfg: dict) -> Optional[str]:
    """
    Si el modelo escribió 'Action: rag_search' en texto (sin ejecutar tool),
    ejecuta rag_search -> summarize_evidence y cierra con el LLM.
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

    # 1) RAG
    obs = rag_search.invoke({"query": q, "domains": domains, "k": k}) or {}
    results = obs.get("results") or []
    if not results:
        return ("No encontré evidencia local para esa consulta. "
                "Decime servicio/fecha y vuelvo a intentar con más contexto.")

    # 2) Resumen
    context_md = _results_to_context_md(results)
    summ = summarize_evidence.invoke({"context_md": context_md}) or ""

    # 3) Cierre
    llm = make_chat(cfg)
    close_prompt = (
        "Usá el siguiente resumen de evidencia para responder con el formato:\n"
        "- Cronología breve (≤5)\n- Causa raíz (hipótesis)\n- Mitigaciones (3–5 bullets)\n- Fuentes\n\n"
        f"Resumen:\n{summ}\n"
    )
    out = llm.invoke([HumanMessage(content=close_prompt)])
    return getattr(out, "content", None) or str(out)

# --- saneo de fuentes basado en evidencia real del trace --- #
_HTTP_URL_RE = re.compile(r"https?://\S+")

def _only_local_sources(results: List[dict]) -> List[str]:
    paths = []
    for r in results or []:
        src = (r or {}).get("source")
        if not src:
            continue
        if src.startswith("http://") or src.startswith("https://"):
            continue
        paths.append(src)
    # quita duplicados preservando orden
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def rewrite_sources_to_local(answer: str, results_or_trace_evidence: List[dict] | None) -> str:
    """
    Elimina URLs externas y reemplaza la sección de 'Fuentes' por paths locales reales
    provenientes de la evidencia del trace (results).
    """
    text = answer or ""
    # quita URLs externas por las dudas
    text = _HTTP_URL_RE.sub("", text).strip()

    # preparar bloque de fuentes local
    local_paths = _only_local_sources(results_or_trace_evidence or [])
    if local_paths:
        block = "Fuentes (locales):\n" + "\n".join(f"- {p}" for p in local_paths)
    else:
        block = "Fuentes: _No hay evidencia local para citar._"

    # si ya hay una sección 'Fuentes', reemplazarla
    # (heurística sencilla)
    if "Fuentes" in text:
        # cortar desde "Fuentes" al final y reemplazar
        idx = text.rfind("Fuentes")
        if idx >= 0:
            text = text[:idx].rstrip() + "\n\n" + block
        else:
            text = text + "\n\n" + block
    else:
        text = text + "\n\n" + block

    return text

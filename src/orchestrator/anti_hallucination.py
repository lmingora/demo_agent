# src/orchestrator/anti_hallucination.py
from __future__ import annotations
import json, re
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage

from src.llm.factory import make_chat
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence
from src.orchestrator.evidence import get_evidence, get_current_trace_id

# Detecta un bloque ReAct de tool-call
_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*[\r\n]+Action Input:\s*(\{.*?\})", re.S)

# Encabezados de fuentes: "Fuentes", "[Fuentes]", "## Fuentes", "**Fuentes**:", etc.
_FUENTES_HDR_RE = re.compile(r"(?mi)^\s*(\#*\s*)?(\[?\*?\*?)?\s*fuentes(\s*\(.*?\))?\s*(\*?\*\]?)?\s*:\s*$")

def _strip_fuentes_block(md: str) -> str:
    """
    Elimina el heading 'Fuentes...' y las líneas de lista que le sigan.
    Heurística conservadora para no dejar fuentes inventadas.
    """
    lines = (md or "").splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if _FUENTES_HDR_RE.match(ln.strip()):
            i += 1
            # saltar lista siguiente (bullets o vacías)
            while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(("-", "*"))):
                i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out).strip()


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


def rewrite_sources_to_local(
    text: str,
    user_text: str,
    cfg: dict,
    trace_id: str | None = None,
    evidence: list[dict] | None = None,
) -> str:
    """
    Reescribe/normaliza referencias a fuentes para que apunten a rutas locales o alias internos,
    usando (si está disponible) la evidencia recuperada en el turno.
    Retorna el texto saneado. Si no hay nada que reescribir, devuelve `text` igual.
    """
    try:
        new_text = text or ""
        if not isinstance(new_text, str) or not new_text:
            return text

        # Heurística conservadora por ahora (no reemplazamos agresivo)
        # Mantener aquí por si luego se implementan mapeos URL->path local.
        return new_text
    except Exception:
        return text


def verify_and_repair(answer_text: str, user_text: str, cfg: dict, trace_id: str | None) -> str:
    """
    Verifica la respuesta contra la evidencia local registrada en el turno (trace_id).
    - Remueve/ajusta afirmaciones no soportadas por la evidencia.
    - Fuerza sección 'Fuentes' a documentos locales (o '—' si no hay).
    """
    # Construimos la evidencia desde el store real (no event_bus)
    tid = trace_id or get_current_trace_id()
    ev_list = get_evidence(tid) or []
    ev_md = _results_to_context_md(ev_list)

    # Sin evidencia → disclaimer honesto
    if not ev_md.strip():
        fixed = rewrite_sources_to_local(answer_text or "", user_text, cfg, trace_id=trace_id)
        fixed = _strip_fuentes_block(fixed)
        fixed = (fixed + "\n\n" if fixed else "") + "Fuentes: —"
        return fixed

    # Con evidencia → pedimos verificación/rewrite al LLM y luego normalizamos “Fuentes”
    llm = make_chat(cfg)
    prompt = f"""
Vas a verificar y, si hace falta, reescribir una respuesta para que:
1) No incluya afirmaciones factuales que NO estén soportadas por la evidencia de abajo.
2) Las **Fuentes** sean SOLO documentos locales que estén en la evidencia (usa el nombre del archivo/metadata.path). Si no hay, escribe 'Fuentes: —'.
3) Si falta evidencia para alguna parte, di explícitamente que no se encontró evidencia local para eso.

[CONSULTA DEL USUARIO]
{user_text}

[RESPUESTA ORIGINAL]
{answer_text}

[EVIDENCIA LOCAL (fragmentos)]
{ev_md}

Devuelve solo la respuesta reescrita (sin explicaciones).
"""
    out = llm.invoke([HumanMessage(content=prompt)])
    fixed = getattr(out, "content", None) or str(out)

    # Saneo final + “Fuentes (locales)” calculadas desde ev_md
    fixed = rewrite_sources_to_local(fixed, user_text, cfg, trace_id=trace_id)
    fixed = _strip_fuentes_block(fixed)

    fixed = (fixed + "\n\n" if fixed else "") + "**Fuentes (locales)**:\n" + ev_md
    return fixed

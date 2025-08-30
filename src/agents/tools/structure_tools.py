# src/agents/tools/structure_tools.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from src.llm.factory import make_chat

def _md_to_items(md: str) -> List[Dict[str, Any]]:
    """
    Convierte un context_md con lines tipo '- [i] (source) texto...' en items {text, source}.
    Es tolerante a formato; ignora líneas que no matcheen el patrón mínimo.
    """
    items: List[Dict[str, Any]] = []
    if not md:
        return items
    for ln in md.splitlines():
        ln = (ln or "").strip()
        if not ln or not ln.startswith("-"):
            continue
        # Heurística: '- [i] (src) text'
        try:
            # fuente entre paréntesis después del primer ')'
            src = "local"
            text = ln
            # buscar ') '
            if ") " in ln:
                idx = ln.index(") ")
                # buscar '(' que le corresponde
                lpar = ln.rfind("(", 0, idx)
                if lpar != -1:
                    src = ln[lpar+1:idx].strip() or "local"
                    text = ln[idx+2:].strip()
            items.append({"text": text, "source": src})
        except Exception:
            continue
    return items

@tool("summarize_evidence")
def summarize_evidence(
    evidence: Optional[List[Dict[str, Any]]] = None,
    max_bullets: int = 8,
    language: str = "es",
    context_md: Optional[str] = None,
) -> str:
    """
    Resume evidencia en bullets. Compat:
    - Modo A: evidence=[{text,source,...}]
    - Modo B: context_md="- [1] (src) texto..."
    Retorna bullets Markdown. Si no hay nada, '—'.
    """
    items = evidence or _md_to_items(context_md or "")
    if not items:
        return "—"

    # armar prompt determinista, sin CoT
    flat = []
    for i, ev in enumerate(items[: max(1, int(max_bullets) * 3)], 1):
        txt = (ev.get("text") or "").strip().replace("\n", " ")
        src = (ev.get("source") or "local")
        flat.append(f"[{i}] ({src}) {txt}")

    prompt = (
        f"Resume la siguiente evidencia en hasta {int(max_bullets)} bullets claros en {language}.\n"
        "No inventes fuentes ni enlaces. Si hay duplicados, consolídalos.\n"
        "EVIDENCIA:\n" + "\n".join(flat) + "\n"
        "DEVUELVE SOLO BULLETS:\n"
    )

    try:
        llm = make_chat({"llm_defaults": {}})   # usa defaults
        out = llm.invoke([HumanMessage(content=prompt)])
        text = getattr(out, "content", None) or str(out)
    except Exception:
        # fallback sin LLM
        uniq, seen = [], set()
        for ev in items:
            t = (ev.get("text") or "").strip()
            if not t:
                continue
            k = t[:120]
            if k in seen:
                continue
            seen.add(k); uniq.append(t)
            if len(uniq) >= int(max_bullets):
                break
        text = "\n".join(f"- {u}" for u in uniq) if uniq else "—"

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "—"
    if not any(ln.startswith("- ") for ln in lines):
        lines = [f"- {ln}" for ln in lines]
    return "\n".join(lines[: int(max_bullets)])

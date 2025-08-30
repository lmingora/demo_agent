# src/orchestrator/finalizer.py
from __future__ import annotations
import re
from typing import Optional
from src.orchestrator.evidence import get_current_trace_id, get_evidence, get_evidence_context_md

_FUENTES_RE = re.compile(r"(?mi)^\s*[-*]?\s*fuentes\s*\(.*?\)\s*:\s*$")

def _strip_existing_fuentes(md: str) -> str:
    """
    Elimina cualquier bloque 'Fuentes ...' y líneas siguientes tipo lista hasta un corte en blanco doble
    (heurística conservadora, evita duplicar o dejar falsas fuentes).
    """
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if _FUENTES_RE.match(ln.strip()):
            # saltar este heading y la lista que le sigue
            i += 1
            while i < len(lines) and (lines[i].strip().startswith(("-", "*")) or not lines[i].strip()):
                i += 1
            # continúa sin agregar esas líneas
            continue
        out.append(ln)
        i += 1
    return "\n".join(out).strip()

def finalize_answer(draft_md: str, *, force_sources: bool = True, require_evidence: bool = False) -> str:
    """
    - Reemplaza cualquier 'Fuentes (locales): ...' por fuentes derivadas de la evidencia real del trace.
    - Si no hay evidencia y require_evidence=True, agrega un descargo.
    """
    tid = get_current_trace_id()
    ev = get_evidence(tid)
    final_md = draft_md or ""

    if force_sources:
        final_md = _strip_existing_fuentes(final_md)
        fuentes_md = get_evidence_context_md(tid, max_chars=2000)
        if fuentes_md:
            final_md = (final_md + "\n\n" if final_md else "") + "**Fuentes (locales)**:\n" + fuentes_md
        elif require_evidence:
            final_md = (final_md + "\n\n" if final_md else "") + "_No hay evidencia local asociada a esta respuesta._"

    return final_md.strip()

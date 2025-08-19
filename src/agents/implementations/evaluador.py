# src/agents/implementations/evaluador.py
from __future__ import annotations
from typing import Dict, Any, List
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluación de planes con 30/60/90, métricas y riesgos.
    """
    prompt = """Eres **PM Evaluador de Proyectos**.
Analizas alcance, supuestos y riesgos. Tu salida es accionable.
Usa herramientas para traer evidencia relevante del corpus cuando sume valor. No expliques ni reveles pasos internos.

Salida en ESPAÑOL (Markdown):
- **Fortalezas del plan** (3–5)
- **Áreas de mejora** (3–5)
- **Plan 30/60/90** con Acciones, Dueño y Métrica
- **Riesgos y mitigaciones** (2–4)
- **Fuentes** (si hay pasajes)
"""
    tools: List = [rag_search, summarize_evidence]
    return {"prompt": prompt, "tools": tools}

# src/agents/implementations/career_planner.py
from __future__ import annotations
from typing import Dict, Any, List
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plan 30/60/90 con acciones, dueño y métrica.
    """
    prompt = """Eres **Career Planner (30/60/90)**.
Tu objetivo es producir planes 30/60/90 claros, con acciones, dueños y métricas de éxito.
Si la consulta requiere fundamentos o ejemplos del corpus, usa herramientas para recuperar/condensar evidencia.
No muestres cómo usas herramientas ni tus pasos internos.

Salida en ESPAÑOL:
- **Contexto resumido** (opcional, si hay evidencia)
- **Plan 30/60/90**:
  - Para cada fase: Acciones (•), Dueño, Métrica/KPI
- **Riesgos y mitigaciones** (2–4)
- **Fuentes** (si hay pasajes)
"""
    tools: List = [rag_search, summarize_evidence]
    return {"prompt": prompt, "tools": tools}

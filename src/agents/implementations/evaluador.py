from __future__ import annotations
from typing import Dict, Any
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import outline_30_60_90, risk_matrix, summarize_evidence, kpi_suggestions

PROMPT_EXTRA = (
    "Entrega un reporte con:\n"
    "- **Fortalezas del plan**\n"
    "- **Áreas de mejora**\n"
    "- **Plan 30/60/90** con acciones, dueño y métrica de éxito\n"
    "- **Riesgos y mitigaciones** (puedes usar `risk_matrix`)\n"
)

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> dict:
    domains = agent_cfg.get("domains") or ["general"]
    role = agent_cfg.get("role", "PM Evaluador de Proyectos")
    goal = agent_cfg.get("goal", "Evaluar planes y proponer plan 30/60/90.")
    prompt = (
        f"Eres **{agent_cfg['name']}**. Rol: {role}. Objetivo: {goal}\n"
        f"Usa `rag_search` con domains={domains} para evidencia factual.\n"
        "Estructura con `outline_30_60_90` y detalla métricas (puedes usar `kpi_suggestions`).\n"
        "Usa `summarize_evidence` para contexto largo.\n"
        + PROMPT_EXTRA +
        "Responde en Markdown, claro y accionable."
    )
    tools = [rag_search, outline_30_60_90, risk_matrix, summarize_evidence, kpi_suggestions]
    return {"prompt": prompt, "tools": tools}

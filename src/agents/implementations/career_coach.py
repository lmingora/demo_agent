from __future__ import annotations
from typing import Dict, Any
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence, kpi_suggestions

PROMPT_EXTRA = (
    "Entrega:\n"
    "- **Definición breve** del concepto o tema.\n"
    "- **3 bullets accionables** con pasos concretos.\n"
    "- **1 ejemplo** adaptado al contexto.\n"
)

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> dict:
    domains = agent_cfg.get("domains") or ["career","general"]
    role = agent_cfg.get("role", "Coach de carrera")
    goal = agent_cfg.get("goal", "Dar guía práctica de carrera.")
    prompt = (
        f"Eres **{agent_cfg['name']}**. Rol: {role}. Objetivo: {goal}\n"
        f"Usa `rag_search` con domains={domains} para evidencia factual.\n"
        "Si la consulta sugiere métricas, `kpi_suggestions('career', <foco>)`.\n"
        "Usa `summarize_evidence` para comprimir contexto largo.\n"
        + PROMPT_EXTRA +
        "Responde en Markdown, conciso y accionable."
    )
    tools = [rag_search, summarize_evidence, kpi_suggestions]
    return {"prompt": prompt, "tools": tools}

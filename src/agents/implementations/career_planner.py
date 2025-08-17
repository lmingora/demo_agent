from __future__ import annotations
from typing import Dict, Any
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import outline_30_60_90, kpi_suggestions, summarize_evidence

PROMPT_EXTRA = (
    "Devuelve un **Plan 30/60/90** con:\n"
    "- Acciones por cada fase (30/60/90)\n"
    "- **Dueño** y **métrica de éxito** por acción\n"
    "- Si aplica, **KPIs** (usa `kpi_suggestions('career', foco)`)\n"
)

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> dict:
    domains = agent_cfg.get("domains") or ["career","general"]
    role = agent_cfg.get("role", "Career Planner")
    goal = agent_cfg.get("goal", "Generar planes 30/60/90 con métricas.")
    prompt = (
        f"Eres **{agent_cfg['name']}**. Rol: {role}. Objetivo: {goal}\n"
        f"Usa `rag_search` con domains={domains} para evidencia factual.\n"
        "Puedes llamar `outline_30_60_90(<objetivo>)` y enriquecer con KPIs.\n"
        "Usa `summarize_evidence` primero si el contexto es extenso.\n"
        + PROMPT_EXTRA +
        "Responde en Markdown, claro y accionable."
    )
    tools = [rag_search, outline_30_60_90, kpi_suggestions, summarize_evidence]
    return {"prompt": prompt, "tools": tools}

from __future__ import annotations
from typing import Dict, Any
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import five_whys, risk_matrix, summarize_evidence, kpi_suggestions

PROMPT_EXTRA = (
    "Estructura la salida con:\n"
    "- **Cronología breve** (≤ 5 ítems)\n"
    "- **5 Porqués** (usa `five_whys` si ayuda)\n"
    "- **Causa raíz (hipótesis)**\n"
    "- **3 acciones de mitigación** con responsable y ETA\n"
    "Puedes incluir una **Matriz de Riesgos** (`risk_matrix`) si corresponde.\n"
)

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> dict:
    domains = agent_cfg.get("domains") or ["incident","general"]
    role = agent_cfg.get("role", "Incident Analyst (SRE)")
    goal = agent_cfg.get("goal", "Analizar incidentes y mitigaciones.")
    prompt = (
        f"Eres **{agent_cfg['name']}**. Rol: {role}. Objetivo: {goal}\n"
        f"Usa `rag_search` con domains={domains} para evidencia factual.\n"
        "Puedes llamar `five_whys` y `risk_matrix` para estructurar.\n"
        "Si la consulta pide métricas, `kpi_suggestions('incident', foco)`.\n"
        "Usa `summarize_evidence` si el contexto es largo.\n"
        + PROMPT_EXTRA +
        "Responde en Markdown, claro y accionable."
    )
    tools = [rag_search, five_whys, risk_matrix, summarize_evidence, kpi_suggestions]
    return {"prompt": prompt, "tools": tools}

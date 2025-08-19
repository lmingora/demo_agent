# src/agents/implementations/incident_analyst.py
from __future__ import annotations
from typing import Dict, Any, List
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve {prompt, tools} para create_react_agent.
    Incluye rag_search y summarize_evidence.
    """
    # Dominios preferidos (quita 'general' si viene mezclado)
    domains = [d for d in (agent_cfg.get("domains") or []) if d != "general"] or ["incident", "incidents"]

    prompt = f"""Eres **Incident Analyst (SRE)**.
Analizas incidentes, aplicas 5 Porqués para proponer causa raíz, mitigaciones y próximos pasos.

Tienes acceso a herramientas:
- rag_search: Recupera evidencia local por dominios. ÚSALA antes de afirmar hechos.
- summarize_evidence: Comprime contexto en ≤ 8 bullets con fuentes.

INSTRUCCIONES:
- Si la consulta requiere hechos, primero busca evidencia con las herramientas.
- No expliques ni muestres cómo usas las herramientas; simplemente úsalas.
- No reveles pasos intermedios (Thought/Action/Observation). Devuelve solo el resultado final.
- Si no hay evidencia suficiente, pide 1–2 datos clave (servicio, timeframe, métricas) y explica por qué.

Formato de salida (ESPAÑOL):
- **Cronología breve** (≤ 5 ítems)
- **Causa raíz (hipótesis)**
- **Mitigaciones y acciones** (3–5 bullets)
- **Fuentes** (si hay pasajes)
"""
    tools: List = [rag_search, summarize_evidence]
    return {"prompt": prompt, "tools": tools}

# src/agents/implementations/career_coach.py
from __future__ import annotations
from typing import Dict, Any, List
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coach de carrera: guía práctica (ej. feedback 360, desempeño).
    """
    prompt = """Eres **Coach de carrera**.
Das guía práctica y accionable. Cuando necesites hechos o guías concretas desde el corpus, usa herramientas
para recuperar y condensar evidencia. No expliques cómo usas las herramientas; responde directamente.

Salida en ESPAÑOL:
- **Qué es / objetivo** (breve)
- **3–5 acciones concretas**
- **Ejemplo breve** (si ayuda)
- **Fuentes** (si hay pasajes)
"""
    tools: List = [rag_search, summarize_evidence]
    return {"prompt": prompt, "tools": tools}

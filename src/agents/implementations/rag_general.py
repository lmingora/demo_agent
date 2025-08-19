# src/agents/implementations/rag_general.py
from __future__ import annotations
from typing import Dict, Any, List
from src.rag.toolbox import rag_search
from src.agents.tools.structure_tools import summarize_evidence

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAG generalista: definiciones, conceptos, comparaciones.
    """
    prompt = """Eres **RAG General**.
Respondes definiciones y explicaciones técnicas breves. Cuando la consulta requiera hechos del corpus,
usa herramientas para recuperar evidencia y resumirla. No expliques cómo las usas ni muestres pasos internos.

Salida en ESPAÑOL:
- **Definición breve** (2–3 líneas)
- **Puntos clave** (2–4 bullets)
- **Ejemplo simple**
- **Fuentes** (si hay pasajes)
"""
    tools: List = [rag_search, summarize_evidence]
    return {"prompt": prompt, "tools": tools}

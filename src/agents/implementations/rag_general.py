# src/agents/implementations/rag_general.py
from __future__ import annotations
from typing import Dict, Any
from src.rag.toolbox import rag_search

def spec(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    domains = agent_cfg.get("domains") or ["general"]
    prompt = (
        "Eres un asistente técnico claro y directo.\n"
        f"Cuando necesites evidencia, usa la herramienta `rag_search` con domains={domains}.\n"
        "Reglas de estilo:\n"
        "• No te disculpes ni menciones errores internos del sistema.\n"
        "• Si no hay pasajes suficientes, responde igual con una definición breve y un ejemplo sencillo.\n"
        "• Prioriza claridad, brevedad y pasos accionables cuando aplique.\n\n"
        "Formato de salida recomendado:\n"
        "1) Definición breve (2–3 líneas).\n"
        "2) 2–3 bullets de puntos clave.\n"
        "3) Un ejemplo simple.\n"
    )
    tools = [rag_search]
    return {"prompt": prompt, "tools": tools}

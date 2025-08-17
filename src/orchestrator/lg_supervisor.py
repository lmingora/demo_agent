# src/orchestrator/lg_supervisor.py
from __future__ import annotations
from pathlib import Path

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

from src.llm.factory import make_chat
from src.utils.logging import get_logger
from src.rag.toolbox import rag_search  # para el agente genérico cuando no hay spec

# Implementaciones (cada módulo expone spec(agent_cfg, cfg) -> {"prompt": str, "tools": List[Tool]})
from src.agents.implementations import (
    rag_general as _rag_general,
    career_coach as _career_coach,
    career_planner as _career_planner,
    incident_analyst as _incident_analyst,
    evaluador as _evaluador,
)

log = get_logger("orchestrator.supervisor")

# Mapa nombre -> función spec()
_SPEC_BY_NAME = {
    "rag_general": _rag_general.spec,
    "career_coach": _career_coach.spec,
    "career_planner": _career_planner.spec,
    "incident_analyst": _incident_analyst.spec,
    "evaluador": _evaluador.spec,
}

# Para fallback desde la CLI (p. ej., si el supervisor inventara un agente)
_WORKERS_BY_NAME: Dict[str, Any] = {}


def get_agent_graph(name: str) -> Optional[Any]:
    """Devuelve el grafo del agente por nombre, si existe (para fallback directo)."""
    return _WORKERS_BY_NAME.get(name)


def get_agent_names() -> List[str]:
    """Lista de agentes conocidos por el supervisor (para validaciones/CLI)."""
    return list(_WORKERS_BY_NAME.keys())


def _build_worker(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Any:
    """
    Crea un agente ReAct preconstruido (LangGraph) con nombre estable.
    - Si hay spec(): usa su prompt y tools.
    - Si no hay: agente genérico con `rag_search` limitado a domains declarados.
    """
    name = agent_cfg["name"]
    try:
        spec_fn = _SPEC_BY_NAME.get(name)
        if spec_fn is None:
            domains = agent_cfg.get("domains") or ["general"]
            prompt = (
                f"Eres **{name}** (rol: {agent_cfg.get('role','Agente')}).\n"
                f"Para evidencia factual, usa la herramienta `rag_search` con domains={domains}.\n"
                "Si falta información, pide 1–2 aclaraciones. Responde en Markdown claro."
            )
            tools: List[BaseTool] = [rag_search]
        else:
            spec = spec_fn(agent_cfg, cfg)
            prompt = spec["prompt"]
            tools = list(spec["tools"])

        # LLM del agente
        llm = make_chat(cfg)  # NO bind_tools: create_react_agent gestiona tools internamente
        # Agente ReAct preconstruido (name = identificador que usa el supervisor para handoffs)
        worker = create_react_agent(
            model=llm,
            tools=tools,
            prompt=prompt,
            name=name,
        )
        return worker
    except Exception as e:
        log.error(f"Error construyendo worker '{name}': {e}")
        raise


def _make_supervisor_prompt(agent_names: List[str], cfg: Dict[str, Any]) -> str:
    """
    Prompt ESTRICTO para el supervisor con reglas de ruteo (keywords) y few-shots.
    - Usa sólo tools `transfer_to_<agente>`.
    - Formato ReAct: Action ... / Action Input: {}
    """
    agents_str = ", ".join(agent_names)

    # Hints desde settings.yaml -> router.keywords
    rkw = (cfg.get("router", {}) or {}).get("keywords", {}) or {}
    inc_kw = ", ".join(rkw.get("incident", [])) or "(incidente, caída, outage, rca, ...)"
    car_kw = ", ".join(rkw.get("career", []))   or "(feedback, OKR, 360, desempeño, ...)"

    # Few-shots canónicos (1 por agente clave)
    shots = [
        # general
        (
            "¿Qué es un pipeline en ingeniería de software y para qué sirve?",
            "rag_general"
        ),
        # career_coach
        (
            "¿Qué es el feedback 360 y cómo pedirlo? Quiero 3 acciones concretas.",
            "career_coach"
        ),
        # career_planner
        (
            "Quiero un plan 30/60/90 para pasar a SRE II enfocado en fiabilidad.",
            "career_planner"
        ),
        # incident_analyst
        (
            "Tuvimos una caída hoy 14:05 (12 min). Aplicá 5 Porqués y mitigaciones.",
            "incident_analyst"
        ),
        # evaluador
        (
            "Evaluá este plan y devolvé fortalezas, áreas de mejora, 30/60/90 y riesgos.",
            "evaluador"
        ),
    ]
    examples = "\n\n".join(
        (
            f"Thought: El mensaje corresponde a {agente}.\n"
            f"Action: transfer_to_{agente}\n"
            "Action Input: {}"
        ) for (_q, agente) in shots
    )

    return (
        "Eres el SUPERVISOR del orquestador. Tu tarea es ELEGIR el agente adecuado y delegar con una sola tool.\n"
        f"Agentes disponibles (whitelist): {agents_str}\n\n"
        "REGLAS ESTRICTAS:\n"
        "1) NO hables con el usuario ni generes prosa.\n"
        "2) Usa EXCLUSIVAMENTE el formato ReAct de tools:\n"
        "   Thought: <breve razonamiento>\n"
        "   Action: transfer_to_<agente>\n"
        "   Action Input: {}\n"
        "3) NO agregues argumentos (Action Input DEBE ser {}).\n"
        "4) Si dudas, usa 'rag_general'. PROHIBIDO inventar nombres (assistant, chatbot, etc.).\n\n"
        "GUÍAS DE RUTEO (heurísticas):\n"
        f"- Si el mensaje contiene términos de INCIDENTES ({inc_kw}) → 'incident_analyst'.\n"
        f"- Si el mensaje es de carrera (p. ej., {car_kw}) → 'career_coach' o 'career_planner' si pide 30/60/90.\n"
        "- Si pide evaluar un plan con fortalezas/áreas/30-60-90/risgos → 'evaluador'.\n"
        "- Si es una definición o consulta general de tecnología → 'rag_general'.\n\n"
        "EJEMPLOS VÁLIDOS:\n"
        f"{examples}\n\n"
        "Responde SIEMPRE con ese bloque Action/Action Input y nada más."
    )

def build_app(cfg: Dict[str, Any]) -> Any:
    """
    Construye y compila el grafo del supervisor (API oficial):
      - workers: LISTA de agentes (cada uno con name único)
      - supervisor: create_supervisor(agents=workers, model=..., prompt=...)  -> .compile()
    """
    agents_cfg = cfg.get("agents", []) or []
    if not agents_cfg:
        raise ValueError("No hay agentes definidos en config/agents.yaml")

    # 1) Construir workers como LISTA (canónico para create_supervisor)
    workers: List[Any] = []
    names: List[str] = []
    _WORKERS_BY_NAME.clear()

    for ac in agents_cfg:
        worker = _build_worker(ac, cfg)
        workers.append(worker)
        names.append(ac["name"])
        _WORKERS_BY_NAME[ac["name"]] = worker

    # 2) Modelo del supervisor (conservador)
    sup_llm = make_chat(cfg)
    try:
        sup_llm.temperature = 0.0
    except Exception:
        pass

    # 3) Prompt del supervisor
    sup_prompt = _make_supervisor_prompt(names, cfg)


    # 4) Crear supervisor (firma oficial: agents=list, model=..., prompt=...)
    #    handoff_tool_prefix=None -> usa 'transfer_to_<name>' por defecto
    #    include_agent_name="inline" ayuda con providers que no propagan name
    supervisor = create_supervisor(
        agents=workers,
        model=sup_llm,
        prompt=sup_prompt,
        handoff_tool_prefix=None,
        include_agent_name="inline",
    )

    # 5) Compilar SIEMPRE (para disponer de .invoke / .stream)
    compiled = supervisor.compile()

    log.info("Supervisor compilado (prebuilt, lista de workers, handoffs internos).")
    return compiled

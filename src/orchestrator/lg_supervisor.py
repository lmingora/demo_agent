# src/orchestrator/lg_supervisor.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

from src.llm.factory import make_chat
from src.utils.logging import get_logger
from src.rag.toolbox import rag_search  # fallback tool p/ agente genérico

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

# Para fallback desde la CLI
_WORKERS_BY_NAME: Dict[str, Any] = {}


def get_agent_graph(name: str) -> Optional[Any]:
    """Devuelve el grafo del agente por nombre, si existe (para fallback directo)."""
    return _WORKERS_BY_NAME.get(name)


def get_agent_names() -> List[str]:
    """Lista de agentes conocidos por el supervisor (para validaciones/CLI)."""
    return list(_WORKERS_BY_NAME.keys())


def _build_worker(agent_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Any:
    """
    Construye el grafo de un agente usando su spec (prompt + tools).

    Nota:
    - No pasamos extra_tools de handoff: el create_supervisor prebuilt ya genera
      las transfer tools `transfer_to_<name>` y las gestiona él.
    """
    try:
        name = agent_cfg.get("name")
        if not name:
            raise ValueError(f"Spec de agente inválida (sin 'name'): {agent_cfg}")

        spec_fn = _SPEC_BY_NAME.get(name)
        if spec_fn is None:
            # Agente genérico (fallback)
            domains = agent_cfg.get("domains") or ["general"]
            prompt = (
                f"Eres **{name}**. Rol: {agent_cfg.get('role','Agente')}.\n"
                f"Usa `rag_search` con domains={domains} para evidencia factual.\n"
                "No muestres código; usa herramientas si necesitás buscar.\n"
                "Si falta info, pide 1–2 aclaraciones. Responde en Markdown."
            )
            tools: List[BaseTool] = [rag_search]
        else:
            spec = spec_fn(agent_cfg, cfg) or {}
            prompt = spec.get("prompt") or (
                f"Eres **{name}**. Responde en Markdown y usa herramientas cuando necesites evidencia."
            )
            tools = list(spec.get("tools") or [])
            if not tools:
                # Aseguramos al menos RAG básico
                tools = [rag_search]
                log.warning(f"Agente '{name}' sin tools en spec(); usando fallback [rag_search].")

        # LLM + tools
        llm = make_chat(cfg).bind_tools(tools)
        graph = create_react_agent(llm, tools=tools, prompt=prompt, name=name)

        # Logging defensivo para ver qué tools recibió el agente
        try:
            tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
            log.info(f"Agente '{name}' con tools: {tool_names}")
        except Exception:
            pass

        return graph

    except Exception as e:
        log.error(f"Error construyendo worker {agent_cfg.get('name')}: {e}")
        raise


def _make_supervisor_prompt(agent_names: List[str], cfg: Dict[str, Any]) -> str:
    """
    Prompt del supervisor SIN reglas duras.
    - Whitelist derivada de agents.yaml
    - Tabla dinámica con (Agente / Rol / Dominios)
    - Few-shots canónicos (1 por agente) auto-derivados
    """
    # --- 1) Whitelist/tabla (derivada de cfg["agents"]) ---
    agents_cfg = cfg.get("agents", []) or []
    # Mapear name -> (role, domains)
    rows = []
    for a in agents_cfg:
        name = a.get("name", "").strip()
        if not name:
            continue
        role = a.get("role", "").strip() or "Agente"
        domains = a.get("domains") or ["general"]
        rows.append((name, role, ", ".join(domains)))

    whitelist_str = ", ".join([r[0] for r in rows]) if rows else ", ".join(agent_names)

    # tabla markdown
    table_lines = ["| Agente | Rol | Dominios |", "|---|---|---|"]
    for name, role, doms in rows:
        table_lines.append(f"| {name} | {role} | {doms} |")
    table_md = "\n".join(table_lines)

    # --- 2) Few-shots canónicos (derivados por nombre; fallback genérico) ---
    #   Nota: no incrustamos “reglas”; sólo ejemplos de entrada → handoff correcto.
    CANON_Q = {
        "rag_general":       "¿Qué es un pipeline en ingeniería de software y para qué sirve?",
        "career_coach":      "Necesito feedback 360 para mi evaluación. Dame 3 pasos prácticos para pedirlo.",
        "career_planner":    "Armá un plan 30/60/90 para mi onboarding como backend en microservicios.",
        "incident_analyst":  "Tuvimos un incidente sev2 con 12 minutos de downtime hoy. Aplicá 5 porqués y da mitigaciones.",
        "evaluador":         "Evaluá este plan: mejorar release cycle en 6 semanas. Dame fortalezas, áreas de mejora, 30/60/90 y riesgos.",
    }

    def _fallback_q(a: Dict[str, Any]) -> str:
        doms = a.get("domains") or ["general"]
        dom = doms[0]
        if dom in ("incident", "incidents"):
            return "Hay una caída reciente. Necesito análisis, causa raíz y mitigaciones."
        if dom == "career":
            return "Quiero mejorar mi desempeño. Sugiéreme 3 acciones prácticas."
        return "Necesito una explicación breve y un ejemplo concreto."

    # Formato ReAct estricto esperado para el supervisor
    def _supervisor_action(agent: str) -> str:
        return (
            f"Thought: El mensaje corresponde a {agent}.\n"
            f"Action: transfer_to_{agent}\n"
            f"Action Input: {{}}"
        )

    fewshots_blocks: List[str] = []
    # seguimos el orden del grafo (agent_names)
    by_name = {a.get("name"): a for a in agents_cfg}
    for name in agent_names:
        a = by_name.get(name, {"name": name})
        q = CANON_Q.get(name) or _fallback_q(a)
        fewshots_blocks.append(
            f"Usuario: {q}\n{_supervisor_action(name)}"
        )

    fewshots_md = "\n\n".join(fewshots_blocks)

    # --- 3) Prompt final del supervisor (sin reglas heurísticas) ---
    return (
        "Eres el **SUPERVISOR** del orquestador. Tu tarea es elegir el agente adecuado "
        "y delegar con una sola herramienta de handoff.\n\n"
        f"Agentes disponibles (whitelist): {whitelist_str}\n\n"
        "Descripción de agentes:\n"
        f"{table_md}\n\n"
        "REGLAS ESTRICTAS:\n"
        "1) NO hables con el usuario ni generes prosa.\n"
        "2) Usa EXCLUSIVAMENTE el formato ReAct de tools:\n"
        "   Thought: <breve razonamiento>\n"
        "   Action: transfer_to_<agente>\n"
        "   Action Input: {}\n"
        "3) NO agregues argumentos (Action Input DEBE ser {}).\n"
        "4) Si dudas, usa 'rag_general'. PROHIBIDO inventar nombres.\n\n"
        "EJEMPLOS (uno por agente):\n"
        f"{fewshots_md}\n\n"
        "Responde SIEMPRE con ese bloque Action/Action Input y nada más."
    )



def _ensure_unique_names(agents_cfg: List[Dict[str, Any]]) -> None:
    names: List[str] = []
    dups: List[str] = []
    for a in agents_cfg:
        n = a.get("name")
        if not n:
            continue
        if n in names:
            dups.append(n)
        names.append(n)
    if dups:
        raise ValueError(f"Nombres de agentes duplicados: {sorted(set(dups))}. Deben ser únicos.")


def build_app(cfg: Dict[str, Any]) -> Any:
    """
    Construye y compila el grafo del supervisor (API oficial):
      - workers: LISTA de agentes (cada uno con name único)
      - supervisor: create_supervisor(agents=workers, model=..., prompt=...) -> .compile()
    """
    if not isinstance(cfg, dict):
        raise TypeError(
            f"build_supervisor_app esperaba cfg (dict), recibió {type(cfg).__name__}. "
            "Pasá el dict que retorna load_cfg()."
        )

    agents_cfg = cfg.get("agents", []) or []
    if not agents_cfg:
        raise ValueError("No hay agentes definidos en config/agents.yaml")

    _ensure_unique_names(agents_cfg)

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

    # 4) Crear supervisor (firma oficial: lista de workers)
    supervisor = create_supervisor(
        agents=workers,
        model=sup_llm,
        prompt=sup_prompt,
        handoff_tool_prefix=None,     # 'transfer_to_<name>' por defecto
        include_agent_name="inline",  # ayuda a preservar el name
    )

    # 5) Compilar (con o sin checkpointer SQLite)
    compiled = None
    if (cfg.get("features", {}) or {}).get("use_checkpointer_sqlite", False):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            cp_path = (cfg.get("features", {}) or {}).get("sqlite_checkpoint_path", "cache/checkpoints.sqlite3")
            cp = SqliteSaver(cp_path)
            compiled = supervisor.compile(checkpointer=cp)
            log.info(f"Supervisor compilado con SqliteSaver (checkpoint={cp_path}).")
        except Exception as e:
            log.warning(f"No se pudo activar checkpointer SQLite: {e}. Compilo sin checkpoint.")
            compiled = supervisor.compile()
    else:
        compiled = supervisor.compile()
        log.info("Supervisor compilado sin checkpointer.")

    return compiled

# src/agents/tools/handoffs.py
from __future__ import annotations
from typing import List
from langchain_core.tools import tool, BaseTool
from src.utils.logging import get_logger
from typing import Any, Dict, List, Optional

__all__ = ["create_handoff_tool", "create_handoff_tools"]

log = get_logger("agents.handoff")

#ultima version

 
@dataclass
class Handoff:
    worker: str
    reason: str
    confidence: float = 0.0
    features: Optional[List[str]] = None
    votes: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _sanitize_reason(reason: str, max_len: int = 240) -> str:
    """Limpia y limita el texto del motivo para evitar logs ruidosos o inyección rara."""
    if not reason:
        return ""
    r = reason.replace("\n", " ").strip()
    r = r.replace("'", "’")  # evita romper el string de retorno
    if len(r) > max_len:
        r = r[: max_len - 1] + "…"
    return r


def create_handoff_tool(agent_name: str) -> BaseTool:
    """
    Crea una tool para que un agente solicite transferencia al `agent_name`.

    La tool devuelve una cadena con la “orden” de transferencia, que el
    supervisor interpretará (p.ej. `transfer_to_career_planner(reason='...')`).

    Args:
        agent_name: Nombre del agente destino (debe existir en el supervisor).

    Returns:
        BaseTool registrada con nombre `handoff_to_<agent_name>`.
    """

    @tool(name=f"handoff_to_{agent_name}", infer_schema=True)
    def _handoff(reason: str = "") -> str:
        """Pedir al supervisor transferir la conversación a otro agente.

        Args:
            reason: Motivo breve de la transferencia (opcional).

        Returns:
            Cadena con la orden de transferencia para el supervisor.
        """
        clean = _sanitize_reason(reason)
        log.info(f"[handoff] → {agent_name}  reason='{clean}'")
        return f"transfer_to_{agent_name}(reason='{clean}')"

    return _handoff


def create_handoff_tools(agent_names: List[str], self_name: str | None = None) -> List[BaseTool]:
    """
    Helper para crear varias handoff-tools de una sola vez.

    Args:
        agent_names: Lista de agentes disponibles en el supervisor.
        self_name: (opcional) nombre del agente actual para NO crearse a sí mismo.

    Returns:
        Lista de tools `handoff_to_<agent>`.
    """
    tools: List[BaseTool] = []
    for name in agent_names:
        if self_name and name == self_name:
            continue
        tools.append(create_handoff_tool(name))
    return tools





def log_handoff(h: Handoff, trace_id: Optional[str] = None, path: str = "cache/traces.jsonl") -> None:
    rec = {
        "ts": int(time.time()),
        "trace_id": trace_id,
        "handoff": h.to_dict(),
    }
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort
        pass

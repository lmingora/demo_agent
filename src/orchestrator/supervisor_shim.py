# src/orchestrator/supervisor_shim.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable

def _fallback_create_supervisor(
    *,
    agents: List[Dict[str, Any]],
    model: Any,
    prompt: str,
    # hook opcional para telemetría: on_handoff(handoff_dict)
    on_handoff: Optional[Callable[[Dict[str, Any]], None]] = None,
    **kw,
):
    """
    Fallback mínimo si no existe langgraph_supervisor:
    - Elige agente por similitud cruda contra su prompt
    - Inyecta SystemMessage con [HANDOFF_REASON]
    - Loguea el handoff vía callback
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage, SystemMessage
    import math
    import time

    # Construimos workers reales y memoria de prompts
    names = [a.get("name") or f"agent_{i}" for i, a in enumerate(agents)]
    prompts = [a.get("prompt") or "" for a in agents]
    tools_lists = [a.get("tools") or [] for a in agents]
    workers = [a.get("worker") or create_react_agent(model, t, prompt=a.get("prompt") or "", name=names[i]) for i, t in enumerate(tools_lists)]

    def _best_agent_and_reason(query: str) -> tuple[int, str, float, Dict[str, float]]:
        # Heurística (LCS) para fallback: suficiente para routing básico
        def lcs(a: str, b: str) -> int:
            dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
            for i, ca in enumerate(a, 1):
                for j, cb in enumerate(b, 1):
                    dp[i][j] = dp[i-1][j-1]+1 if ca==cb else max(dp[i-1][j], dp[i][j-1])
            return dp[-1][-1]
        scores = [(lcs(query.lower(), p.lower())/max(1, len(query))) for p in prompts]
        idx = max(range(len(scores)), key=lambda i: scores[i])
        # Reason simple (puede mejorarse con keywords/domains)
        reason = "similaridad con la descripción del agente"
        votes = {names[i]: float(scores[i]) for i in range(len(scores))}
        return idx, reason, float(scores[idx]), votes

    class _Compiled:
        def invoke(self, payload: Dict[str, Any], config: Dict[str, Any] | None = None):
            messages = payload.get("messages") or []
            # Sacamos último mensaje humano para estimar intención
            last = ""
            for m in reversed(messages):
                role = getattr(m, "type", getattr(m, "role", ""))
                if role in ("human", "user"):
                    last = getattr(m, "content", "") or ""
                    break
            idx, reason, conf, votes = _best_agent_and_reason(last)
            worker = workers[idx]
            name = names[idx]

            # Inyectar SystemMessage con la razón
            try:
                messages.insert(0, SystemMessage(content=f'[HANDOFF_REASON] worker={name} conf={conf:.2f} reason="{reason}"'))
            except Exception:
                messages.insert(0, {"role": "system", "content": f'[HANDOFF_REASON] worker={name} conf={conf:.2f} reason="{reason}"'})

            # Telemetría via callback (si existe)
            if callable(on_handoff):
                try:
                    on_handoff({
                        "ts": int(time.time()),
                        "worker": name,
                        "reason": reason,
                        "confidence": conf,
                        "votes": votes,
                    })
                except Exception:
                    pass

            # Ejecutar el agente elegido
            res = worker.invoke({"messages": messages}, config=config)
            
            try:
                from src.orchestrator.anti_hallucination import verify_and_repair
                from src.orchestrator.event_bus import get_current_trace_id
                # extraer texto de respuesta
                answer_text = None
                try:
                    answer_text = getattr(res, "content", None)
                except Exception:
                    pass
                if answer_text is None and isinstance(res, dict):
                    # LangGraph a veces devuelve dict con 'messages' o 'output'
                    msgs = res.get("messages") or []
                    if msgs:
                        last = msgs[-1]
                        answer_text = getattr(last, "content", None) or last.get("content")
                    if answer_text is None:
                        answer_text = res.get("output", None)
                if not isinstance(answer_text, str):
                    answer_text = str(answer_text or "")

                fixed = verify_and_repair(
                    answer_text=answer_text,
                    user_text=last,                  # 'last' ya lo calculamos como el último mensaje del usuario
                    cfg={},                          # el verificador usa make_chat(cfg); si querés, podés pasar el cfg real
                    trace_id=get_current_trace_id(),
                )
                # devolver el mismo tipo que 'res' traía, pero con el texto arreglado
                try:
                    # si 'res' es mensaje de LangChain
                    res.content = fixed
                except Exception:
                    if isinstance(res, dict):
                        if "output" in res:
                            res["output"] = fixed
                        elif res.get("messages"):
                            try:
                                res["messages"][-1]["content"] = fixed
                            except Exception:
                                res["final_text"] = fixed
                    else:
                        res = fixed
            except Exception:
                pass
            return res

    class _Supervisor:
        def compile(self, **_: Any) -> _Compiled:
            return _Compiled()

    return _Supervisor()

# Si existe la lib real, la usamos tal cual.
try:
    from langgraph_supervisor import create_supervisor as _real_create_supervisor
    create_supervisor = _real_create_supervisor  # type: ignore
except Exception:
    create_supervisor = _fallback_create_supervisor  # type: ignore

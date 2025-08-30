# src/orchestrator/supervisor_shim.py  (reemplazo completo recomendado)
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable

    
def _fallback_create_supervisor(
    *,
    agents: List[Dict[str, Any]],
    model: Any,
    prompt: str,
    on_handoff: Optional[Callable[[Dict[str, Any]], None]] = None,
    agent_domain_map: Optional[Dict[str, List[str]]] = None,
    tools: Optional[List[Any]] = None,   # ← NUEVO: supervisor puede recibir tools (p.ej. probe_domains)
    **kw,
):
    """
    Fallback local del supervisor cuando no está la impl oficial.
    - Acepta 'tools' (e.g., probe_domains) y las usa para decidir el handoff por recuperación real.
    - Mantiene compat: on_handoff, agent_domain_map y el fallback LCS previo.
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage, SystemMessage
    import time

    # workers a partir de agentes (compat con tu código existente)
    names = [a.get("name") or f"agent_{i}" for i, a in enumerate(agents)]
    prompts = [a.get("prompt") or "" for a in agents]
    tools_lists = [a.get("tools") or [] for a in agents]
    workers = [
        a.get("worker") or create_react_agent(model, t, prompt=a.get("prompt") or "", name=names[i])
        for i, (a, t) in enumerate(zip(agents, tools_lists))
    ]

    # (opcional) bind tools al LLM del supervisor si existen (paridad con API oficial)
    sup_llm = model
    try:
        if tools:
            sup_llm = model.bind_tools(tools)
    except Exception:
        pass

    # helper LCS (fallback)
    def _lcs(a: str, b: str) -> int:
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i, ca in enumerate(a, 1):
            for j, cb in enumerate(b, 1):
                dp[i][j] = dp[i-1][j-1]+1 if ca==cb else max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
    def _ensure_min_evidence(agent_name: str, user_query: str, messages: list):
        """
        Garantiza que haya evidencia registrada en el trace actual antes de invocar al worker.
        - Si no hay trace_id, crea uno e inserta [TRACEID] en messages.
        - Ejecuta rag_search con dominios del agente (o todos si no hay mapping).
        """
        try:
            from uuid import uuid4
            from langchain_core.messages import SystemMessage
            from src.orchestrator.evidence import (
                get_current_trace_id, set_current_trace_id, get_evidence,
            )
            from src.rag.toolbox import rag_search
        except Exception:
            return

        # 1) trace_id
        tid = get_current_trace_id()
        if not tid:
            tid = uuid4().hex
            try:
                set_current_trace_id(tid)
            except Exception:
                pass
            # Inyectar TRACEID en el hilo de mensajes del turno
            try:
                messages.insert(0, SystemMessage(content=f"[TRACEID] {tid}"))
            except Exception:
                messages.insert(0, {"role": "system", "content": f"[TRACEID] {tid}"})

        # 2) ¿ya hay evidencia?
        try:
            if get_evidence(tid):
                return
        except Exception:
            pass

        # 3) RAG mínimo con dominios del agente (o todos)
        doms = (agent_domain_map or {}).get(agent_name) or ["*"]
        q = (user_query or "").strip()
        if not q:
            return
        try:
            rag_search.invoke({"query": q, "domains": doms, "k": 6, "trace_id": tid})
        except Exception:
            pass


        
    # intentar usar la tool 'probe_domains' si está disponible
    def _probe_hits(query: str) -> Optional[Dict[str, int]]:
        if not tools:
            return None
        probe = None
        try:
            # buscar por nombre estándar; si usás otra convención, adaptalo
            for t in tools:
                name = getattr(t, "name", getattr(t, "__name__", "")).lower()
                if name == "probe_domains":
                    probe = t
                    break
        except Exception:
            probe = None
        if probe is None:
            return None

        try:
            # invocación directa de la tool (equivalente a un tool-call del LLM)
            out = probe.invoke({"query": query, "domains": ["*"], "k_per_domain": 3}) or {}
            hits = out.get("hits") or {}
            # log opcional de diagnóstico
            print(f"[router] probe_domains hits: {hits}")
            return {str(k): int(v) for k, v in hits.items()}
        except Exception:
            return None

    def _best_agent_and_reason(query: str) -> tuple[int, str, float, Dict[str, float]]:
        """
        1) Si hay tool probe_domains → elegir agente por dominio con más hits (recuperación real).
        2) Si falla/empata → fallback LCS con descripciones/prompts de agentes.
        """
        # 1) routing por evidencia real
        try:
            hits = _probe_hits(query)
            if hits and agent_domain_map:
                # dominio ganador
                best_dom = max(hits.keys(), key=lambda d: hits.get(d, 0))
                # candidatos: agentes que contienen ese dominio
                candidates = [i for i, nm in enumerate(names) if best_dom in (agent_domain_map.get(nm) or [])]
                if candidates:
                    idx = candidates[0]
                    reason = f"probe_domains: dominio '{best_dom}' con {hits.get(best_dom,0)} pasajes"
                    votes = {nm: float(hits.get(best_dom, 0)) for nm in names}
                    conf = float(hits.get(best_dom, 0))
                    return idx, reason, conf, votes
        except Exception:
            pass

        # 2) fallback LCS
        scores = [_lcs(query.lower(), p.lower()) / max(1, len(query)) for p in prompts]
        idx = max(range(len(scores)), key=lambda i: scores[i])
        reason = "similaridad con la descripción del agente"
        votes = {names[i]: float(scores[i]) for i in range(len(names))}
        return idx, reason, float(scores[idx]), votes

    class _Compiled:
        def invoke(self, payload: Dict[str, Any], config: Dict[str, Any] | None = None):
            messages = payload.get("messages") or []
            last = ""
            for m in reversed(messages):
                role = getattr(m, "type", getattr(m, "role", ""))
                if role in ("human", "user"):
                    last = getattr(m, "content", "") or ""
                    break

            idx, reason, conf, votes = _best_agent_and_reason(last)
            worker = workers[idx]
            name = names[idx]

            # Inyectar razón como SystemMessage
            try:
                messages.insert(0, SystemMessage(content=f'[HANDOFF_REASON] worker={name} conf={conf:.2f} reason="{reason}"'))
            except Exception:
                messages.insert(0, {"role": "system", "content": f'[HANDOFF_REASON] worker={name} conf={conf:.2f} reason="{reason}"'})

            # Telemetría del handoff
            if callable(on_handoff):
                try:
                    on_handoff({"ts": int(time.time()), "worker": name, "reason": reason, "confidence": conf, "votes": votes})
                except Exception:
                    pass
            _ensure_min_evidence(name, last)

            res = worker.invoke({"messages": messages}, config=config)

            # Post-proceso anti-alucinación (tu finalizador)
            try:
                from src.orchestrator.anti_hallucination import verify_and_repair
                from src.orchestrator.event_bus import get_current_trace_id
                answer_text = None
                try:
                    answer_text = getattr(res, "content", None)
                except Exception:
                    pass
                if answer_text is None and isinstance(res, dict):
                    msgs = res.get("messages") or []
                    if msgs:
                        lastm = msgs[-1]
                        answer_text = getattr(lastm, "content", None) or lastm.get("content")
                    if answer_text is None:
                        answer_text = res.get("output", None)
                if not isinstance(answer_text, str):
                    answer_text = str(answer_text or "")

                fixed = verify_and_repair(answer_text=answer_text, user_text=last, cfg={}, trace_id=get_current_trace_id())
                try:
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


try:
    from langgraph_supervisor import create_supervisor as _real_create_supervisor
    create_supervisor = _real_create_supervisor  # type: ignore
except Exception:
    create_supervisor = _fallback_create_supervisor  # type: ignore
# src/orchestrator/probe_tools.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

from src.rag.toolbox import rag_search, _all_domains

@tool("probe_domains", infer_schema=True)
def probe_domains(
    query: str,
    domains: Optional[List[str]] = None,
    k_per_domain: int = 3,
) -> Dict[str, Any]:
    """
    Sondea recuperación por dominio usando el RAG actual.
    - Si 'domains' es None/[]/['*'], usa todos los dominios de docs.yaml.
    - Devuelve conteo de pasajes por dominio y una muestra de fuentes.
    """
    try:
        if not domains or domains == ["*"]:
            domains = _all_domains({})
    except Exception:
        domains = domains or ["general"]

    out: Dict[str, Any] = {"hits": {}, "samples": {}}
    for dom in domains:
        try:
            obs = rag_search.invoke({"query": query, "domains": [dom], "k": int(k_per_domain)}) or {}
            results = obs.get("results") or []
            out["hits"][dom] = int(len(results))
            # sample: hasta 2 fuentes (source)
            samples: List[str] = []
            for r in results[:2]:
                src = (r or {}).get("source") or (r or {}).get("path") or "unknown"
                if src not in samples:
                    samples.append(src)
            out["samples"][dom] = samples
        except Exception:
            out["hits"][dom] = 0
            out["samples"][dom] = []
    return out

# src/observability/metrics.py
from __future__ import annotations
from typing import Callable
try:
    from prometheus_client import Counter, Histogram, start_http_server
except Exception:
    Counter = Histogram = None
    def start_http_server(*args, **kwargs): ...

RAG_SEARCH_TOTAL = Counter("rag_search_calls_total", "Total de llamadas a rag_search") if Counter else None
ROUTER_DECISIONS = Counter("router_decisions_total", "Rutas del router por agente", ["agent"]) if Counter else None
VERIFY_REPAIRS = Counter("answer_verify_repairs_total", "Total de reparaciones de respuesta") if Counter else None
RAG_LATENCY = Histogram("rag_latency_seconds", "Latencia de rag_search (s)") if Histogram else None
LLM_LATENCY = Histogram("llm_latency_seconds", "Latencia de llamadas LLM (s)") if Histogram else None

def start_metrics_server(port: int = 9099):
    try:
        start_http_server(port)
    except Exception:
        pass

def inc(counter, *labels):
    try:
        if counter is None: return
        (counter.labels(*labels) if labels else counter).inc()
    except Exception:
        pass

def observe(hist, value: float):
    try:
        if hist is None: return
        hist.observe(value)
    except Exception:
        pass

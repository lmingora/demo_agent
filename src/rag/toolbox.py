# src/rag/toolbox.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from src.utils.logging import get_logger
from src.rag.vectorstores import make_vectorstore
from src.rag import retrievers as _retr

log = get_logger("rag.toolbox")

__all__ = [
    "init_rag_tooling",
    "rag_search",
    "get_vectorstore",
    "list_domains",
    "refresh_bm25",
]

# VectorStore global (se setea en init_rag_tooling)
_VECTORSTORE = None


def get_vectorstore():
    """Devuelve el vectorstore actual (o None si no se inicializó)."""
    return _VECTORSTORE


def list_domains() -> List[str]:
    """Devuelve los dominios disponibles detectados por el subsistema de retrievers (BM25)."""
    try:
        return _retr.list_domains()
    except Exception:
        return []


def init_rag_tooling(cfg: Dict[str, Any]) -> None:
    """
    Inicializa el stack de RAG:
      1) Crea el VectorStore (Chroma, etc.) con embeddings cacheados.
      2) Inyecta el vectorstore en el módulo de retrievers.
      3) Inicializa BM25 por dominio (según tus datos/docs).
    """
    global _VECTORSTORE

    # 1) Vectorstore (usa embeddings cacheados dentro de make_vectorstore)
    _VECTORSTORE = make_vectorstore(cfg)

    # 2) Entregar el vectorstore al módulo de retrievers
    try:
        _retr.set_vectorstore(_VECTORSTORE)
    except Exception as e:
        log.warning(f"No se pudo setear el vectorstore en retrievers: {e}")

    # 3) Inicializar BM25 por dominio (el módulo imprime sus propios logs por dominio)
    try:
        _retr.init_bm25(cfg)
    except Exception as e:
        log.warning(f"No se pudo inicializar BM25: {e}")

    # Info de dominios disponibles
    try:
        doms = _retr.list_domains()
        if doms:
            log.info(f"Dominios detectados: {doms}")
    except Exception:
        pass


def refresh_bm25(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Refresca/reinicializa los índices BM25.

    Estrategia:
      - Si el módulo retrievers expone `refresh_bm25()`, úsalo.
      - Si no, y tenemos `cfg`, invocá `init_bm25(cfg)` como reinicialización.
      - Si no hay ninguna de las dos opciones, loguea warning y no hace nada.
    """
    try:
        if hasattr(_retr, "refresh_bm25") and callable(_retr.refresh_bm25):
            _retr.refresh_bm25()
            log.info("BM25 refrescado (via retrievers.refresh_bm25).")
            return
        if cfg is not None:
            _retr.init_bm25(cfg)
            log.info("BM25 reinicializado (via retrievers.init_bm25(cfg)).")
            return
        log.warning("refresh_bm25: no hay retrievers.refresh_bm25 y no se pasó cfg; BM25 no se actualizó.")
    except Exception as e:
        log.error(f"No se pudo refrescar BM25: {e}")


# ---------------------------- Herramientas (Tools) ----------------------------

def _run_retriever(retr, query: str):
    """
    Usa la API moderna .invoke si existe; si no, fallback a get_relevant_documents().
    Devuelve una lista de langchain.schema.Document (o similar).
    """
    # Camino “Runnable” moderno
    if hasattr(retr, "invoke"):
        try:
            return retr.invoke(query)
        except Exception as e:
            log.warning(f"invoke(query) falló en retriever; probando fallback: {e}")

    # Compatibilidad con retrievers legacy
    if hasattr(retr, "get_relevant_documents"):
        try:
            return retr.get_relevant_documents(query)
        except Exception as e:
            log.error(f"get_relevant_documents(query) falló: {e}")
            return []

    # Último intento: invocar como Runnable con payload dict
    try:
        return retr.invoke({"query": query})
    except Exception as e:
        log.error(f"Retriever sin interfaz compatible (invoke/query): {e}")
        return []


def _pack_results(docs) -> Dict[str, Any]:
    """
    Normaliza resultados de recuperación a un payload estable para el LLM:
      {"results":[{"text","source","score","domain"},...], "count":N}
    """
    results: List[Dict[str, Any]] = []
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}
        results.append({
            "text": getattr(d, "page_content", "") or "",
            "source": md.get("source") or md.get("path") or md.get("file") or md.get("id") or "unknown",
            "score": md.get("score") or md.get("relevance") or None,
            "domain": md.get("domain"),
        })
    return {"results": results, "count": len(results)}


@tool("rag_search", infer_schema=True)
def rag_search(query: str, domains: Optional[List[str]] = None, k: Optional[int] = None) -> Dict[str, Any]:
    """
    Busca pasajes relevantes combinando recuperación lexical (BM25) y vectorial (VectorStore).

    Args:
      query: Texto de consulta.
      domains: Limitar búsqueda a uno o más dominios (p.ej. ["career","incident"]).
      k: Override del top-k (si se omite, usa el de settings.yaml).

    Returns:
      dict: {"results":[{"text","source","score","domain"},...], "count":N}
    """
    try:
        retr = _retr.get_ensemble_retriever(domains=domains, k_override=k)
    except Exception as e:
        log.error(f"No se pudo construir el ensemble retriever: {e}")
        return {"results": [], "count": 0}

    docs = _run_retriever(retr, query)
    payload = _pack_results(docs)
    log.debug(f"RAG({domains or '*'}) '{query[:60]}…' → {payload['count']} pasajes")
    return payload

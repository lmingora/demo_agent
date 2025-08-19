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
    "get_embedder",
    "list_domains",
    "refresh_bm25",
    "rag_probe_counts",
]

# VectorStore global (se setea en init_rag_tooling)
_VECTORSTORE = None


def get_vectorstore():
    """Devuelve el vectorstore actual (o None si no se inicializó)."""
    return _VECTORSTORE


def get_embedder():
    """
    Devuelve el objeto Embeddings usado por el VectorStore (si está disponible).
    Útil para el router semántico / similares.
    """
    vs = get_vectorstore()
    if vs is None:
        raise RuntimeError("Vectorstore no inicializado. Llamá init_rag_tooling(cfg) primero.")
    emb = getattr(vs, "embedding_function", None)
    if emb is None:
        raise RuntimeError("El VectorStore actual no expone .embedding_function")
    return emb


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
      3) Inicializa BM25 por dominio.
    """
    global _VECTORSTORE

    _VECTORSTORE = make_vectorstore(cfg)

    try:
        _retr.set_vectorstore(_VECTORSTORE)
    except Exception as e:
        log.warning(f"No se pudo setear el vectorstore en retrievers: {e}")

    try:
        _retr.init_bm25(cfg)
    except Exception as e:
        log.warning(f"No se pudo inicializar BM25: {e}")

    try:
        doms = _retr.list_domains()
        if doms:
            log.info(f"Dominios detectados: {doms}")
    except Exception:
        pass


def refresh_bm25(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Refresca/reinicializa los índices BM25.
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


# ---------------------------- Internals --------------------------------------

def _run_retriever(retr, query: str):
    """
    Usa la API moderna .invoke si existe; si no, fallback a get_relevant_documents().
    Devuelve una lista de langchain.schema.Document (o similar).
    """
    if hasattr(retr, "invoke"):
        try:
            return retr.invoke(query)
        except Exception as e:
            log.warning(f"invoke(query) falló en retriever; probando fallback: {e}")

    if hasattr(retr, "get_relevant_documents"):
        try:
            return retr.get_relevant_documents(query)
        except Exception as e:
            log.error(f"get_relevant_documents(query) falló: {e}")
            return []

    try:
        return retr.invoke({"query": query})
    except Exception as e:
        log.error(f"Retriever sin interfaz compatible (invoke/query): {e}")
        return []


def _pack_results(docs) -> Dict[str, Any]:
    """
    Normaliza resultados de recuperación a un payload estable:
      {"results":[{"text","source","score","domain"},...], "count":N}
    """
    results: List[Dict[str, Any]] = []
    for d in docs or []:
        md = getattr(d, "metadata", None)
        if md is None and isinstance(d, dict):
            md = d.get("metadata", {}) or {}
            text = d.get("page_content") or d.get("text") or ""
        else:
            md = md or {}
            text = getattr(d, "page_content", "") or ""
        results.append({
            "text": text,
            "source": md.get("source") or md.get("path") or md.get("file") or md.get("id") or "unknown",
            "score": md.get("score") or md.get("relevance") or None,
            "domain": md.get("domain"),
        })
    return {"results": results, "count": len(results)}


def _results_to_context_md(results: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    Construye un 'context_md' compacto para consumo de LLMs/tooling.
    """
    if not results:
        return ""
    lines: List[str] = []
    used = 0
    for i, r in enumerate(results, 1):
        src = r.get("source", "unknown")
        txt = (r.get("text") or "").strip().replace("\n", " ")
        piece = f"- [{i}] ({src}) {txt}"
        if used + len(piece) > max_chars:
            break
        lines.append(piece)
        used += len(piece)
    return "\n".join(lines)


# ---------------------------- Herramientas (Tools) ----------------------------

@tool("rag_search", infer_schema=True)
def rag_search(query: str, domains: Optional[List[str]] = None, k: Optional[int] = None) -> Dict[str, Any]:
    """
    Recupera evidencia combinando recuperación lexical (BM25) y vectorial.

    Args:
      query: Texto de consulta.
      domains: Limitar búsqueda a dominios (p.ej. ["career","incident"]).
      k: top-k; si se omite usa el de settings.yaml.

    Returns:
      dict con:
        - "results": lista de pasajes normalizados
        - "count": cantidad de pasajes
        - "context_md": resumen markdown compacto listo para `summarize_evidence`
    """
    try:
        retr = _retr.get_ensemble_retriever(domains=domains, k_override=k)
    except Exception as e:
        log.error(f"No se pudo construir el ensemble retriever: {e}")
        return {"results": [], "count": 0, "context_md": ""}

    docs = _run_retriever(retr, query)
    payload = _pack_results(docs)
    payload["context_md"] = _results_to_context_md(payload["results"])
    log.debug(f"RAG({domains or '*'}) '{query[:60]}…' → {payload['count']} pasajes")
    return payload


# ---------------------------- Probe para Router -------------------------------

def rag_probe_counts(query: str, domains: Optional[List[str]] = None, k: int = 3) -> Dict[str, int]:
    """
    Conteo simple de evidencia por dominio para enrutado.
    """
    try:
        retr = _retr.get_ensemble_retriever(domains=domains, k_override=k)
    except Exception as e:
        log.warning(f"rag_probe_counts: no se pudo construir retriever: {e}")
        return {}

    docs = _run_retriever(retr, query) or []
    counts: Dict[str, int] = {}
    for d in docs:
        md = getattr(d, "metadata", None)
        dom = None
        if md is None and isinstance(d, dict):
            md = d.get("metadata") or {}
            dom = md.get("domain") or d.get("domain")
        else:
            md = md or {}
            dom = md.get("domain")
        dom = dom or "general"
        counts[dom] = counts.get(dom, 0) + 1

    return counts

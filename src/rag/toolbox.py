# src/rag/toolbox.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable

from langchain_core.tools import tool
from src.utils.logging import get_logger
from src.rag.vectorstores import make_vectorstore
from src.rag import retrievers as _retr
from src.orchestrator.evidence import record_evidence, get_current_trace_id
from src.observability.metrics import inc

import os, json, re, glob
from pathlib import Path   # ← ESTA LÍNEA ES LA QUE FALTA


log = get_logger("rag.toolbox")

__all__ = [
    "init_rag_tooling",
    "rag_search",
    "get_vectorstore",
    "get_embedder",
    "list_domains",
    "refresh_bm25",
    "rag_probe_counts",
    "set_request_context",   # NUEVO
]

# ---- Contexto por turno (user/trace) ----
_CURRENT_USER_ID: Optional[str] = None
_CURRENT_TRACE_ID: Optional[str] = None

def set_request_context(user_id: Optional[str], trace_id: Optional[str]) -> None:
    """Setea el contexto del turno para que las tools lo usen por defecto."""
    global _CURRENT_USER_ID, _CURRENT_TRACE_ID
    _CURRENT_USER_ID = user_id
    _CURRENT_TRACE_ID = trace_id



# VectorStore global (se setea en init_rag_tooling)
_VECTORSTORE = None


def get_vectorstore():
    """Devuelve el vectorstore actual (o None si no se inicializó)."""
    return _VECTORSTORE


def get_embedder(cfg: Optional[Dict[str, Any]] = None):
    """
    Devuelve el objeto Embeddings usado por el VectorStore (si está disponible).
    Fallback: intenta crear embeddings desde config si el VS no expone .embedding_function.
    """
    vs = get_vectorstore()
    emb = getattr(vs, "embedding_function", None)
    if emb is not None:
        return emb

    if cfg is not None:
        try:
            # Fallback elegante: usar el factory de embeddings del proyecto
            from src.llm.embeddings import make_embeddings  # type: ignore
            emb = make_embeddings(cfg)
            if emb is not None:
                log.info("get_embedder: usando fallback make_embeddings(cfg).")
                return emb
        except Exception as e:
            log.warning(f"get_embedder: fallback make_embeddings(cfg) no disponible: {e}")

    # Último recurso: mensaje claro
    raise RuntimeError("El VectorStore actual no expone .embedding_function y no se pudo crear un embedder desde cfg.")


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
      3) Inicializa BM25 por dominio (según tus datos/docs o JSONL).
    """
    global _VECTORSTORE

    # 1) Vectorstore
    _VECTORSTORE = make_vectorstore(cfg)

    # 2) Entregar el vectorstore al módulo de retrievers
    try:
        _retr.set_vectorstore(_VECTORSTORE)
    except Exception as e:
        log.warning(f"No se pudo setear el vectorstore en retrievers: {e}")

    # 3) Inicializar BM25
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


def refresh_bm25(cfg: Optional[dict] = None, domain: Optional[str] = None) -> None:
    """
    Refresca los índices BM25. Tolerante a firmas antiguas:
    - Si el impl soporta domain: lo usamos.
    - Si no, llamamos sin domain.
    - Nunca rompe el flujo (loggea warning).
    """
    try:
        from src.rag import retrievers as _ret
    except Exception as e:
        log.warning(f"BM25 refresh no disponible (no se pudo importar retrievers): {e}")
        return

    try:
        # preferir firma con dominio si existe
        if hasattr(_ret, "refresh_bm25"):
            import inspect
            sig = inspect.signature(_ret.refresh_bm25)
            if "domain" in sig.parameters:
                _ret.refresh_bm25(cfg=cfg, domain=domain)
            else:
                _ret.refresh_bm25(cfg)  # firma vieja
        else:
            log.warning("BM25 refresh no disponible: retrievers.refresh_bm25 no encontrado.")
    except TypeError:
        # fallback duro: intentar sin args
        try:
            _ret.refresh_bm25()
        except Exception as e:
            log.warning(f"BM25 refresh fallo (fallback): {e}")
    except Exception as e:
        log.warning(f"BM25 refresh fallo: {e}")

# ---------------------------- Internals --------------------------------------

def _run_retriever(retr, query: str):
    """Invoca retriever moderno (invoke) o legacy (get_relevant_documents)."""
    if hasattr(retr, "invoke"):
        try:
            return retr.invoke(query)
        except Exception as e:
            log.warning(f"invoke(query) falló en retriever; fallback: {e}")

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
    Normaliza resultados de recuperación a un payload estable para el LLM:
      {"results":[{"text","source","score","domain","owner"},...], "count":N}
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
            "owner": md.get("owner"),
        })
    return {"results": results, "count": len(results)}


def _results_to_context_md(results: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """Construye un markdown compacto a partir de los results para el cierre."""
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
def rag_search(
    query: str,
    domains: Optional[List[str]] = None,
    k: Optional[int] = None,
    user_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Busca pasajes relevantes combinando recuperación vectorial y, como respaldo,
    recuperación lexical on-disk (lee cache/bm25/<dom>.jsonl) para completar top-K.
    """
    # helper local para armar context_md corto y estable
    def _results_to_context_md(results: List[dict], max_chars: int = 4000) -> str:
        if not results:
            return ""
        lines, used = [], 0
        for i, r in enumerate(results, 1):
            src = r.get("source", "unknown")
            txt = (r.get("text") or "").strip().replace("\n", " ")
            piece = f"- [{i}] ({src}) {txt}"
            if used + len(piece) > max_chars:
                break
            lines.append(piece); used += len(piece)
        return "\n".join(lines)

    from src.utils.logging import get_logger
    log = get_logger("rag.toolbox")

    # top-k por config si no vino override (conserva tu lógica)
    try:
        k_final = int(k) if k is not None else int((_retr.get("cfg") or {}).get("retrieval", {}).get("k", 8))
    except Exception:
        k_final = 8

    # Normalizar dominios: si None/[]/["*"] → usar todos
    try:
        cfg = _retr.get("cfg") or {}
    except Exception:
        cfg = {}
    doms_req = (domains or [])
    if not doms_req or doms_req == ["*"]:
        doms_req = _all_domains(cfg)

    # Construir retriever híbrido (respetando tu contrato)
    try:
        retr = _retr.get_ensemble_retriever(domains=doms_req, k_override=k_final, user_id=user_id)
    except Exception as e:
        log.error(f"rag_search: no se pudo construir el ensemble retriever: {e}")
        return {"results": [], "count": 0, "context_md": ""}

    # 1) Vector / ensemble principal
    docs = _run_retriever(retr, query)
    payload = _pack_results(docs)
    results: List[Dict[str, Any]] = payload.get("results", [])
    count_before = len(results)

    # 2) Completar con fallback lexical on-disk por dominio si falta para top-K
    try:
        # dedup por (source, text[:120])
        seen = set(( (r.get("source") or ""), (r.get("text") or "")[:120]) for r in results)
        if len(results) < k_final:
            for dom in doms_req:
                # si ya llenamos, cortamos
                if len(results) >= k_final:
                    break
                fallback = _bm25_disk_search(query, dom, k=k_final)
                for r in fallback:
                    key = (r.get("source") or "", (r.get("text") or "")[:120])
                    if key in seen:
                        continue
                    results.append(r)
                    seen.add(key)
                    if len(results) >= k_final:
                        break
        payload["results"] = results
        payload["count"] = len(results)
    except Exception as e:
        log.warning(f"rag_search: fallback lexical on-disk fallo: {e}")

    # 3) context_md y evidencia
    payload["context_md"] = _results_to_context_md(payload["results"])

    tid = trace_id or get_current_trace_id()
    if payload["results"]:
        try:
            record_evidence(payload["results"], trace_id=tid)
        except Exception:
            pass

    log.debug(
        f"RAG({doms_req or '*'}) uid={user_id or '-'} trace={tid or '-'} "
        f"'{query[:60]}…' → {payload['count']} pasajes (vec={count_before}, +lex={payload['count']-count_before})"
    )
    inc(RAG_SEARCH_TOTAL)
    return payload

# ---------------------------- Probe para Router -------------------------------

def rag_probe_counts(query: str, domains: Optional[List[str]] = None, k: int = 3) -> Dict[str, int]:
    """
    Conteo simple de evidencia por dominio para enrutado:
      - Ejecuta el ensemble retriever con top-k bajo (rápido).
      - Devuelve dict {domain -> hits} usando metadata.domain
    """
    try:
        retr = _retr.get_ensemble_retriever(domains=domains, k_override=k, user_id=_CURRENT_USER_ID)
    except Exception as e:
        log.warning(f"rag_probe_counts: no se pudo construir retriever: {e}")
        return {}

    docs = _run_retriever(retr, query) or []
    counts: Dict[str, int] = {}
    for d in docs:
        md = getattr(d, "metadata", None)
        if md is None and isinstance(d, dict):
            md = d.get("metadata") or {}
        dom = (md or {}).get("domain") or "general"
        counts[dom] = counts.get(dom, 0) + 1
    return counts

# === NUEVO: dominios desde docs.yaml ===
def _all_domains(cfg: Optional[dict] = None) -> List[str]:
    try:
        # intentar desde cfg ya cargado
        docs = (cfg or {}).get("docs") or {}
        if docs:
            return sorted([k for k, v in docs.items() if isinstance(v, list) and v])
        # fallback: leer config/docs.yaml si existe
        import yaml
        p = Path("config/docs.yaml")
        if p.exists():
            y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return sorted([k for k, v in (y or {}).items() if isinstance(v, list) and v])
    except Exception:
        pass
    return ["general"]  # mínimo

# === NUEVO: búsqueda lexical on-disk (fallback no-BM25) ===
import json, re
_WORD = re.compile(r"\w+", re.U)

def _tokenize(q: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(q or "") if len(w) > 1]

def _bm25_disk_search(query: str, domain: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Fallback lexical: lee cache/bm25/<dominio>.jsonl y rankea por coincidencia de tokens.
    No requiere un BM25 en memoria; evita depender del refresh.
    """
    path = Path(f"cache/bm25/{domain}.jsonl")
    if not path.exists():
        return []
    toks = set(_tokenize(query))
    if not toks:
        return []
    results = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                txt = (rec.get("text") or "")
                src = rec.get("source") or rec.get("path") or "unknown"
                # score simple: conteo de tokens presentes
                text_toks = set(_tokenize(txt))
                score = len(toks & text_toks)
                if score <= 0:
                    continue
                results.append({"text": txt, "source": src, "score": float(score), "domain": domain})
    except Exception:
        return []
    # rankear por score desc, cortar k
    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return results[: max(1, int(k))]

# === NUEVO: probe de hits por dominio ===
def rag_probe_counts(query: str, domains: List[str], k: int = 5) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for dom in domains:
        # intentar vector primero
        try:
            obs = rag_search.invoke({"query": query, "domains": [dom], "k": k}) or {}
            cnt = len(obs.get("results") or [])
        except Exception:
            cnt = 0
        # si no hay nada, fallback lexical on-disk
        if cnt == 0:
            cnt = len(_bm25_disk_search(query, dom, k))
        counts[dom] = cnt
    return counts

def rag_probe_counts(
    query: str,
    domains: Optional[List[str]] = None,
    k: int = 5,
) -> Dict[str, int]:
    """
    Cuenta hits por dominio usando rag_search (con fallback on-disk).
    - Si domains es None/[]/["*"], usa todos los dominios conocidos desde config/docs.yaml.
    """
    try:
        cfg = _retr.get("cfg") or {}
    except Exception:
        cfg = {}

    # Normalizar dominios
    doms = domains or []
    if not doms or doms == ["*"]:
        doms = _all_domains(cfg)

    counts: Dict[str, int] = {}
    for dom in doms:
        try:
            obs = rag_search.invoke({"query": query, "domains": [dom], "k": k}) or {}
            cnt = int(len(obs.get("results") or []))
        except Exception:
            # Último recurso: solo lexical disco
            cnt = len(_bm25_disk_search(query, dom, k))
        counts[dom] = cnt
    return counts
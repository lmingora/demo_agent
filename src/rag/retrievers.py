# src/rag/retrievers.py
from __future__ import annotations

import os
import re
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from src.utils.logging import get_logger

log = get_logger("rag.retrievers")

# ---------------------- Estado de módulo ----------------------
_VS: Any = None            # VectorStore (LangChain), seteado vía set_vectorstore()
_CFG: Dict[str, Any] = {}  # settings completos, seteados en init_bm25()

_BM25_READY = False

# Índice BM25 en memoria
_BM25_DOCS: List[Document] = []
_BM25_TOKS: List[List[str]] = []
_BM25_IDF: Dict[str, int] = {}
_BM25_DF: Dict[str, int] = {}
_BM25_AVGDL: float = 0.0
_BM25_K1 = 1.5
_BM25_B = 0.75

# Dominios detectados (basado en metadata domain o path)
_DOMAINS: List[str] = []

# --- BM25 store path (compat con código viejo) ---
_BM25_STORE_DIR = "cache/bm25"

def bm25_store_path(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """
    Devuelve el directorio donde se podría persistir/cachar BM25 (si aplica).
    Lee cfg['bm25']['store_dir'] si está disponible; si no, usa _BM25_STORE_DIR.
    """
    global _BM25_STORE_DIR
    if cfg:
        try:
            _BM25_STORE_DIR = cfg.get("bm25", {}).get("store_dir", _BM25_STORE_DIR)
        except Exception:
            pass
    p = Path(_BM25_STORE_DIR)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

# Alias compat para imports antiguos
_bm25_store_path = bm25_store_path

# ---------------------- Utilidades ----------------------
def _tokenize(text: str) -> List[str]:
    """Tokenización unicode simple (lower y \w+)."""
    return [t for t in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE) if t]

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunking por caracteres con solapamiento (robusto y determinista)."""
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - max(0, overlap))
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]

def _hash_key(source: str, text: str) -> str:
    h = hashlib.sha256()
    h.update((source or "").encode("utf-8"))
    h.update(b"||")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()[:16]

def _detect_domains_from_docs(docs: List[Document]) -> List[str]:
    vals = set()
    for d in docs:
        md = d.metadata or {}
        dom = md.get("domain")
        if not dom:
            # inferir de ruta: data/<domain>/...
            src = (md.get("source") or md.get("path") or "")
            parts = Path(src).parts
            for p in parts:
                if p in {"career", "general", "incident", "incidents", "training"}:
                    dom = p
                    break
        if dom:
            vals.add(dom)
    return sorted(vals) if vals else []

# ---------------------- API pública requerida ----------------------
def set_vectorstore(vs: Any) -> None:
    """Recibe el VectorStore (Chroma u otro) para búsquedas densas."""
    global _VS
    _VS = vs
    log.info("Vectorstore recibido en retrievers.")

def list_domains() -> List[str]:
    """Dominios conocidos por el índice BM25 o detectados en los docs."""
    return list(_DOMAINS)

def init_bm25(cfg: Dict[str, Any]) -> None:
    """
    Construye un BM25 en memoria leyendo archivos de `paths.data_dir`.
    Chunking según retrieval.chunk_size/overlap y metadata:
      - source (ruta)
      - domain (carpeta de primer nivel bajo data_dir)
    Guarda cfg global para poder leer k y ensemble_weights luego.
    """
    global _BM25_DOCS, _BM25_TOKS, _BM25_IDF, _BM25_DF, _BM25_AVGDL, _BM25_READY, _DOMAINS, _CFG
    _CFG = cfg or {}

    _BM25_DOCS = []
    _BM25_TOKS = []
    _BM25_IDF = {}
    _BM25_DF = {}
    _BM25_AVGDL = 0.0
    _BM25_READY = False
    _DOMAINS = []

    data_dir = Path(_CFG.get("paths", {}).get("data_dir", "data")).resolve()
    if not data_dir.exists():
        log.warning(f"BM25: data_dir no existe: {data_dir}")
        return

    chunk_size = int(_CFG.get("retrieval", {}).get("chunk_size", 1200))
    overlap = int(_CFG.get("retrieval", {}).get("overlap", 200))

    total_chunks = 0
    domain_counts: Dict[str, int] = {}
    for root, _, files in os.walk(data_dir):
        root_path = Path(root)
        try:
            rel = root_path.relative_to(data_dir)
            parts = rel.parts
            domain = parts[0] if parts else "general"
        except Exception:
            domain = "general"

        for fname in files:
            if not fname.lower().endswith((".txt", ".md", ".markdown")):
                continue
            path = root_path / fname
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                log.warning(f"No se pudo leer {path}: {e}")
                continue

            chunks = _chunk_text(text, chunk_size, overlap)
            for ch in chunks:
                if not ch.strip():
                    continue
                doc = Document(
                    page_content=ch,
                    metadata={"source": str(path), "domain": domain},
                )
                _BM25_DOCS.append(doc)
                _BM25_TOKS.append(_tokenize(ch))
                total_chunks += 1
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

    if total_chunks == 0:
        log.info("BM25 vacío (no se encontraron chunks).")
        return

    # DF / IDF
    for toks in _BM25_TOKS:
        for t in set(toks):
            _BM25_DF[t] = _BM25_DF.get(t, 0) + 1

    N = len(_BM25_TOKS)
    _BM25_AVGDL = sum(len(t) for t in _BM25_TOKS) / max(1, N)
    for t, df in _BM25_DF.items():
        _BM25_IDF[t] = math.log(1 + (N - df + 0.5) / (df + 0.5))

    _DOMAINS = sorted(domain_counts.keys())
    _BM25_READY = True
    for d, n in domain_counts.items():
        log.info(f"BM25[{d}] cargado con {n} chunks")

def refresh_bm25() -> None:
    """
    Refresca BM25 usando el último cfg conocido (_CFG).
    Si no hay _CFG, informa no-op.
    """
    if _CFG:
        log.info("Refrescando BM25 con cfg actual…")
        init_bm25(_CFG)
    else:
        log.info("BM25: no hay cfg cargado; llamá init_bm25(cfg).")

def get_ensemble_retriever(domains: Optional[List[str]] = None,
                           k_override: Optional[int] = None) -> Runnable:
    """
    Devuelve un retriever híbrido (BM25 + Vector) como Runnable.invoke(query)->List[Document].
    Lee pesos de settings.yaml: retrieval.ensemble_weights = [w_lex, w_dense]
    Lee top-k: retrieval.k (con env override RAG_TOPK) y parámetro k_override.
    """
    # --- config: k y pesos ---
    def _read_k() -> int:
        if isinstance(k_override, int) and k_override > 0:
            return k_override
        try:
            env_k = int(os.environ.get("RAG_TOPK", "0"))
        except Exception:
            env_k = 0
        if env_k > 0:
            return env_k
        try:
            return int((_CFG.get("retrieval") or {}).get("k", 8))
        except Exception:
            return 8

    def _read_weights() -> Tuple[float, float]:
        w = (_CFG.get("retrieval") or {}).get("ensemble_weights")
        if isinstance(w, (list, tuple)) and len(w) == 2:
            try:
                w_lex, w_dense = float(w[0]), float(w[1])
            except Exception:
                w_lex, w_dense = 0.5, 0.5
        else:
            # env override: RAG_WEIGHTS="0.6,0.4"
            env = os.environ.get("RAG_WEIGHTS")
            if env and "," in env:
                try:
                    a, b = env.split(",", 1)
                    w_lex, w_dense = float(a), float(b)
                except Exception:
                    w_lex, w_dense = 0.5, 0.5
            else:
                w_lex, w_dense = 0.5, 0.5
        # normalizar si no suman ~1
        s = w_lex + w_dense
        if s <= 0:
            return 0.5, 0.5
        return (w_lex / s, w_dense / s)

    k = _read_k()
    w_lex, w_dense = _read_weights()

    # --- wrappers ---
    class _VectorWrapper(Runnable):
        def __init__(self, vs, k: int, domains: Optional[List[str]]):
            self.vs = vs
            self.k = k
            self.domains = domains or None

        def invoke(self, query: str, config: Optional[dict] = None) -> List[Document]:
            if self.vs is None:
                return []
            search_kwargs = {"k": self.k}
            if self.domains:
                # Chroma admite filtros de la forma {"domain":{"$in":[...]}}
                search_kwargs["filter"] = {"domain": {"$in": self.domains}}
            try:
                retr = self.vs.as_retriever(search_kwargs=search_kwargs)
            except Exception:
                retr = self.vs.as_retriever(search_kwargs={"k": self.k})
            # Camino moderno
            if hasattr(retr, "invoke"):
                return retr.invoke(query)
            # Legacy
            if hasattr(retr, "get_relevant_documents"):
                return retr.get_relevant_documents(query)
            # Último recurso
            try:
                return retr.invoke({"query": query})
            except Exception:
                return []

    class _BM25Wrapper(Runnable):
        def __init__(self, k: int, domains: Optional[List[str]]):
            self.k = k
            self.domains = set(domains) if domains else None

        def _score(self, q_toks: List[str], d_toks: List[str]) -> float:
            score = 0.0
            freqs: Dict[str, int] = {}
            for t in d_toks:
                freqs[t] = freqs.get(t, 0) + 1
            dl = len(d_toks)
            for t in q_toks:
                df = _BM25_DF.get(t)
                if df is None:
                    continue
                idf = math.log(1 + (len(_BM25_TOKS) - df + 0.5) / (df + 0.5))
                f = freqs.get(t, 0)
                if f == 0:
                    continue
                num = f * (_BM25_K1 + 1)
                den = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(1.0, _BM25_AVGDL))
                score += idf * (num / max(1e-9, den))
            return score

        def invoke(self, query: str, config: Optional[dict] = None) -> List[Document]:
            if not _BM25_READY:
                return []
            q_toks = _tokenize(query)
            scored: List[Tuple[float, int]] = []
            for i, (doc, toks) in enumerate(zip(_BM25_DOCS, _BM25_TOKS)):
                if self.domains:
                    dom = (doc.metadata or {}).get("domain")
                    if dom not in self.domains:
                        continue
                s = self._score(q_toks, toks)
                if s > 0:
                    scored.append((s, i))
            scored.sort(reverse=True, key=lambda x: x[0])
            out: List[Document] = []
            for s, i in scored[: self.k]:
                d = _BM25_DOCS[i]
                md = dict(d.metadata or {})
                md["score"] = float(s)
                out.append(Document(page_content=d.page_content, metadata=md))
            return out

    vec = _VectorWrapper(_VS, k, domains)
    lex = _BM25Wrapper(k, domains)

    class _Ensemble(Runnable):
        def __init__(self, vec_r: Runnable, lex_r: Runnable, k: int, w_lex: float, w_dense: float):
            self.vec_r = vec_r
            self.lex_r = lex_r
            self.k = k
            self.w_lex = w_lex
            self.w_dense = w_dense

        def _norm_scores(self, docs: List[Document], key: str) -> Dict[str, float]:
            vals = [float(d.metadata.get(key)) for d in docs if d.metadata and isinstance(d.metadata.get(key), (int, float))]
            if not vals:
                return {}
            mx = max(vals) or 1.0
            out: Dict[str, float] = {}
            for d in docs:
                s = d.metadata.get(key)
                if isinstance(s, (int, float)):
                    out[_hash_key(d.metadata.get("source", "unk"), d.page_content)] = float(s) / mx
            return out

        def invoke(self, query: str, config: Optional[dict] = None) -> List[Document]:
            # Obtener listas
            try:
                vdocs = self.vec_r.invoke(query) if self.vec_r else []
            except Exception as e:
                log.warning(f"Vector retriever falló: {e}")
                vdocs = []
            try:
                ldocs = self.lex_r.invoke(query) if self.lex_r else []
            except Exception as e:
                log.warning(f"BM25 retriever falló: {e}")
                ldocs = []

            # Normalización / proxies
            vkeys = {}  # rank inverso como proxy (vector stores no traen score)
            for rank, d in enumerate(vdocs, start=1):
                vkeys[_hash_key(d.metadata.get("source", "unk"), d.page_content)] = 1.0 / rank

            lnorm = self._norm_scores(ldocs, "score")

            # Fusión + de-dup
            scores: Dict[str, float] = {}
            pool: Dict[str, Document] = {}

            def _merge(docs: List[Document], weight: float, prox_scores: Dict[str, float], meta_score_key: Optional[str] = None):
                for d in docs:
                    key = _hash_key(d.metadata.get("source", "unk"), d.page_content)
                    pool[key] = d
                    s = prox_scores.get(key)
                    if s is None and meta_score_key and d.metadata.get(meta_score_key) is not None:
                        s = float(d.metadata[meta_score_key])
                    if s is None:
                        s = 0.0
                    scores[key] = scores.get(key, 0.0) + weight * s

            _merge(ldocs, self.w_lex, lnorm, meta_score_key="score")
            _merge(vdocs, self.w_dense, vkeys, meta_score_key=None)

            # Ordenar y cortar top-k
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            out: List[Document] = []
            for key, _ in ranked[: self.k]:
                d = pool[key]
                md = dict(d.metadata or {})
                md["score"] = float(scores[key])  # score combinado
                out.append(Document(page_content=d.page_content, metadata=md))
            return out

    return _Ensemble(vec, lex, k, w_lex, w_dense)

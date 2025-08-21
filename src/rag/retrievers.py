# src/rag/retrievers.py
from __future__ import annotations

import os
import re
import math
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from src.utils.logging import get_logger

log = get_logger("rag.retrievers")

# ---------------------- Estado de módulo ----------------------
_VS = None  # VectorStore (LangChain) seteado vía set_vectorstore
_BM25_READY = False

# Índice BM25 en memoria
_BM25_DOCS: List[Document] = []
_BM25_TOKS: List[List[str]] = []
_BM25_IDF: Dict[str, int] = {}
_BM25_DF: Dict[str, int] = {}
_BM25_AVGDL: float = 0.0
_BM25_K1 = 1.5
_BM25_B = 0.75

# Dominios detectados
_DOMAINS: List[str] = []

# --- BM25 store base dir ---
_BM25_STORE_DIR = "cache/bm25"


def bm25_store_path(cfg: Optional[Dict[str, Any]] = None, domain: Optional[str] = None) -> Path:
    """
    Devuelve:
      - si domain=None: carpeta base para JSONL
      - si domain=str : archivo JSONL de ese dominio
    Crea la carpeta si no existe.
    """
    global _BM25_STORE_DIR
    if cfg:
        _BM25_STORE_DIR = cfg.get("bm25", {}).get("store_dir", _BM25_STORE_DIR)
    base = Path(_BM25_STORE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    return base if domain is None else (base / f"{domain}.jsonl")

# Alias compat
_bm25_store_path = bm25_store_path


# ---------------------- Utilidades ----------------------
def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [t for t in toks if t]


def _hash_key(source: str, text: str) -> str:
    h = hashlib.sha256()
    h.update((source or "").encode("utf-8"))
    h.update(b"||")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------- API pública requerida ----------------------
def set_vectorstore(vs: Any) -> None:
    global _VS
    _VS = vs
    log.info("Vectorstore recibido en retrievers.")


def list_domains() -> List[str]:
    return list(_DOMAINS)


def _load_bm25_from_jsonl(cfg: Dict[str, Any]) -> bool:
    """Intenta cargar BM25 desde JSONL en cache/bm25/*.jsonl (si existen)."""
    global _BM25_DOCS, _BM25_TOKS, _DOMAINS

    base = bm25_store_path(cfg, None)
    files = sorted([p for p in base.glob("*.jsonl") if p.is_file()])
    if not files:
        return False

    _BM25_DOCS.clear()
    _BM25_TOKS.clear()
    doms = set()

    for fp in files:
        domain = fp.stem
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get("text") or ""
                md = rec.get("metadata") or {}
                # aseguramos metadata coherente
                md.setdefault("domain", domain)
                doc = Document(page_content=text, metadata=md)
                _BM25_DOCS.append(doc)
                _BM25_TOKS.append(_tokenize(text))
                doms.add(domain)
        except Exception as e:
            log.warning(f"No se pudo leer BM25 JSONL {fp}: {e}")

    _DOMAINS[:] = sorted(doms)
    return len(_BM25_DOCS) > 0


def init_bm25(cfg: Dict[str, Any]) -> None:
    """
    Construye un BM25 en memoria. Preferencia:
      1) Si hay JSONL en cache/bm25/*.jsonl (producido por indexing.ingest) → usar eso (contiene owner).
      2) Si no hay JSONL → fallback a recorrer data_dir y chunkear (sin owner).
    """
    global _BM25_DOCS, _BM25_TOKS, _BM25_IDF, _BM25_DF, _BM25_AVGDL, _BM25_READY, _DOMAINS

    _BM25_DOCS = []
    _BM25_TOKS = []
    _BM25_IDF = {}
    _BM25_DF = {}
    _BM25_AVGDL = 0.0
    _BM25_READY = False
    _DOMAINS = []

    # 1) Intentar JSONL (incluye owner)
    if _load_bm25_from_jsonl(cfg):
        log.info(f"BM25: cargado desde JSONL ({len(_BM25_TOKS)} chunks, doms={_DOMAINS})")
    else:
        # 2) Fallback a filesystem
        data_dir = Path(cfg.get("paths", {}).get("data_dir", "data")).resolve()
        if not data_dir.exists():
            log.warning(f"BM25: data_dir no existe: {data_dir}")
            return

        rcfg = (cfg.get("retrieval", {}) or {})
        chunk_size = int(rcfg.get("chunk_size", 1200))
        overlap = int(rcfg.get("overlap", 200))

        def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
            if not text:
                return []
            if chunk_size <= 0:
                return [text]
            step = max(1, chunk_size - max(0, overlap))
            return [text[i : i + chunk_size] for i in range(0, len(text), step)]

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
                    md = {"source": str(path), "domain": domain}
                    _BM25_DOCS.append(Document(page_content=ch, metadata=md))
                    _BM25_TOKS.append(_tokenize(ch))
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

        _DOMAINS[:] = sorted(domain_counts.keys())
        for d, n in domain_counts.items():
            log.info(f"BM25[{d}] cargado con {n} chunks (filesystem)")

    # Estadísticas BM25
    if not _BM25_TOKS:
        log.info("BM25 vacío")
        return

    # DF por término
    for toks in _BM25_TOKS:
        seen = set(toks)
        for t in seen:
            _BM25_DF[t] = _BM25_DF.get(t, 0) + 1

    N = len(_BM25_TOKS)
    _BM25_AVGDL = sum(len(t) for t in _BM25_TOKS) / max(1, N)
    for t, df in _BM25_DF.items():
        _BM25_IDF[t] = math.log(1 + (N - df + 0.5) / (df + 0.5))

    _BM25_READY = True


def refresh_bm25() -> None:
    if _BM25_READY:
        log.info("BM25: ya estaba inicializado (no-op).")
    else:
        log.info("BM25: no inicializado; llamá init_bm25(cfg).")


def get_ensemble_retriever(
    domains: Optional[List[str]] = None,
    k_override: Optional[int] = None,
    user_id: Optional[str] = None,
) -> Runnable:
    """
    Devuelve un retriever híbrido (BM25 + Vector) como Runnable.invoke(query)->List[Document].
    Filtra privados: owner == user_id; públicos: owner ausente.
    """
    class _VectorWrapper(Runnable):
        def __init__(self, vs, k: int, domains: Optional[List[str]], user_id: Optional[str]):
            self.vs = vs
            self.k = k
            self.domains = domains or None
            self.user_id = user_id

        def _post_filter_owner(self, docs: List[Document]) -> List[Document]:
            if not self.user_id:
                # sólo públicos o sin owner
                return [d for d in docs if (d.metadata or {}).get("owner") in (None, "",)]
            # públicos + míos
            return [d for d in docs if (d.metadata or {}).get("owner") in (None, "", self.user_id)]

        def invoke(self, query: str, config: Optional[dict] = None) -> List[Document]:
            if self.vs is None:
                return []
            search_kwargs = {"k": self.k}
            # Intento de filtro por dominio (y tal vez owner si backend lo soporta)
            filt: Dict[str, Any] = {}
            if self.domains:
                filt["domain"] = {"$in": self.domains}
            # Muchos backends no soportan OR; por eso hacemos post-filter
            if filt:
                search_kwargs["filter"] = filt
            try:
                retr = self.vs.as_retriever(search_kwargs=search_kwargs)
            except Exception:
                retr = self.vs.as_retriever(search_kwargs={"k": self.k})

            # Camino moderno
            docs: List[Document] = []
            if hasattr(retr, "invoke"):
                docs = retr.invoke(query)
            elif hasattr(retr, "get_relevant_documents"):
                docs = retr.get_relevant_documents(query)
            else:
                try:
                    docs = retr.invoke({"query": query})
                except Exception:
                    docs = []

            return self._post_filter_owner(docs)

    class _BM25Wrapper(Runnable):
        def __init__(self, k: int, domains: Optional[List[str]], user_id: Optional[str]):
            self.k = k
            self.domains = set(domains) if domains else None
            self.user_id = user_id

        def _score(self, q_toks: List[str], d_toks: List[str]) -> float:
            score = 0.0
            freqs: Dict[str, int] = {}
            for t in d_toks:
                freqs[t] = freqs.get(t, 0) + 1
            dl = len(d_toks)
            for t in q_toks:
                if t not in _BM25_IDF:
                    continue
                f = freqs.get(t, 0)
                if f == 0:
                    continue
                idf = _BM25_IDF[t]
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
                # Filtro por dominio
                if self.domains:
                    dom = (doc.metadata or {}).get("domain")
                    if dom not in self.domains:
                        continue
                # Filtro por owner (público o mío)
                owner = (doc.metadata or {}).get("owner")
                if self.user_id:
                    if owner not in (None, "", self.user_id):
                        continue
                else:
                    if owner not in (None, ""):
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

    # --- Pesos y top-k ---
    k = int(k_override or 0) or int(os.environ.get("RAG_TOPK", "0")) or 8
    w_lex, w_dense = 0.5, 0.5  # simple (si querés, leelos de cfg fuera de este módulo)

    vec = _VectorWrapper(_VS, k, domains, user_id)
    lex = _BM25Wrapper(k, domains, user_id)

    class _Ensemble(Runnable):
        def __init__(self, vec_r: Runnable, lex_r: Runnable, k: int, w_lex: float, w_dense: float):
            self.vec_r = vec_r
            self.lex_r = lex_r
            self.k = k
            self.w_lex = w_lex
            self.w_dense = w_dense

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

            # Normalizar scores
            def _norm_scores(docs: List[Document], key: str) -> Dict[str, float]:
                vals = []
                for d in docs:
                    s = (d.metadata or {}).get(key)
                    if isinstance(s, (int, float)):
                        vals.append(float(s))
                if not vals:
                    return {}
                mx = max(vals) or 1.0
                out = {}
                for d in docs:
                    s = (d.metadata or {}).get(key)
                    if isinstance(s, (int, float)):
                        out[_hash_key((d.metadata or {}).get("source", "unk"), d.page_content)] = float(s) / mx
                return out

            # Vector: proxy por rank inverso
            vkeys = {}
            for rank, d in enumerate(vdocs, start=1):
                key = _hash_key((d.metadata or {}).get("source", "unk"), d.page_content)
                vkeys[key] = 1.0 / rank

            lnorm = _norm_scores(ldocs, "score")

            # Fusión
            scores: Dict[str, float] = {}
            pool: Dict[str, Document] = {}

            def _merge(docs: List[Document], weight: float, prox_scores: Dict[str, float], meta_score_key: Optional[str] = None):
                for d in docs:
                    key = _hash_key((d.metadata or {}).get("source", "unk"), d.page_content)
                    pool[key] = d
                    s = prox_scores.get(key)
                    if s is None and meta_score_key and (d.metadata or {}).get(meta_score_key) is not None:
                        s = float((d.metadata or {})[meta_score_key])
                    if s is None:
                        s = 0.0
                    scores[key] = scores.get(key, 0.0) + weight * s

            _merge(ldocs, w_lex, lnorm, meta_score_key="score")
            _merge(vdocs, w_dense, vkeys, meta_score_key=None)

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            out: List[Document] = []
            for key, _ in ranked[: self.k]:
                d = pool[key]
                md = dict(d.metadata or {})
                md["score"] = float(scores[key])
                out.append(Document(page_content=d.page_content, metadata=md))
            return out

    return _Ensemble(vec, lex, k, w_lex, w_dense)

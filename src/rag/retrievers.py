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
_VS = None  # VectorStore (LangChain) seteado vía set_vectorstore
_BM25_READY = False

# Índice BM25 en memoria
_BM25_DOCS: List[Document] = []
_BM25_TOKS: List[List[str]] = []
_BM25_IDF: Dict[str, float] = {}
_BM25_DF: Dict[str, int] = {}
_BM25_AVGDL: float = 0.0
_BM25_K1 = 1.5
_BM25_B = 0.75

# Dominios detectados (basado en metadata domain o path)
_DOMAINS: List[str] = []


from pathlib import Path
from typing import Optional  # si no está ya importado

# --- BM25 store path (compat con código viejo) ---
_BM25_STORE_DIR = "cache/bm25"

def bm25_store_path(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """
    Devuelve el directorio donde se persiste/cacha BM25 (si aplica).
    Lee cfg['bm25']['store_dir'] si está disponible; si no, usa _BM25_STORE_DIR.
    Crea la carpeta si no existe.
    """
    global _BM25_STORE_DIR
    if cfg:
        _BM25_STORE_DIR = cfg.get("bm25", {}).get("store_dir", _BM25_STORE_DIR)
    p = Path(_BM25_STORE_DIR)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

# Alias de compatibilidad para código que importaba el nombre privado
_bm25_store_path = bm25_store_path


# ---------------------- Utilidades ----------------------
def _tokenize(text: str) -> List[str]:
    """Tokenizador simple unicode (lower, sin signos)."""
    # Mantiene letras unicode y números, separa por no-alfanumérico
    # Python re con Unicode: \w incluye letras con acento.
    toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [t for t in toks if t]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunking por caracteres con solapamiento (simple y robusto)."""
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - max(0, overlap))
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), step)]
    return chunks


def _hash_key(source: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"||")
    h.update(text.encode("utf-8"))
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
    Realiza chunking (retrieval.chunk_size/overlap) y anota metadata:
      - source (ruta)
      - domain (carpeta de primer nivel bajo data_dir)
    """
    global _BM25_DOCS, _BM25_TOKS, _BM25_IDF, _BM25_DF, _BM25_AVGDL, _BM25_READY, _DOMAINS

    _BM25_DOCS = []
    _BM25_TOKS = []
    _BM25_IDF = {}
    _BM25_DF = {}
    _BM25_AVGDL = 0.0
    _BM25_READY = False
    _DOMAINS = []

    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data")).resolve()
    if not data_dir.exists():
        log.warning(f"BM25: data_dir no existe: {data_dir}")
        return

    chunk_size = int(cfg.get("retrieval", {}).get("chunk_size", 1200))
    overlap = int(cfg.get("retrieval", {}).get("overlap", 200))

    # Recorrer subcarpetas como dominios
    total_chunks = 0
    domain_counts: Dict[str, int] = {}
    for root, dirs, files in os.walk(data_dir):
        root_path = Path(root)
        # El dominio lo tomamos como el nombre de la carpeta inmediatamente bajo data_dir
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
                    metadata={
                        "source": str(path),
                        "domain": domain,
                    },
                )
                _BM25_DOCS.append(doc)
                toks = _tokenize(ch)
                _BM25_TOKS.append(toks)
                total_chunks += 1
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Estadísticas
    if total_chunks == 0:
        log.info(f"BM25[{domain}] vacío")
        return

    # DF por término
    for toks in _BM25_TOKS:
        seen = set(toks)
        for t in seen:
            _BM25_DF[t] = _BM25_DF.get(t, 0) + 1

    N = len(_BM25_TOKS)
    _BM25_AVGDL = sum(len(t) for t in _BM25_TOKS) / max(1, N)
    # IDF clásico (BM25)
    for t, df in _BM25_DF.items():
        _BM25_IDF[t] = math.log(1 + (N - df + 0.5) / (df + 0.5))

    _DOMAINS = sorted(domain_counts.keys())
    _BM25_READY = True
    for d, n in domain_counts.items():
        log.info(f"BM25[{d}] cargado con {n} chunks")


def refresh_bm25() -> None:
    """Compat: si necesitás refrescar y no pasás cfg, sólo marca estado (no-op aquí)."""
    # Proveemos esta función porque algunas partes de la app la llaman
    # sin cfg; si necesitás reindex real, llamá init_bm25(cfg).
    if _BM25_READY:
        log.info("BM25: ya estaba inicializado (no-op).")
    else:
        log.info("BM25: no inicializado; llamá init_bm25(cfg).")


def get_ensemble_retriever(domains: Optional[List[str]] = None, k_override: Optional[int] = None) -> Runnable:
    """
    Devuelve un retriever híbrido (BM25 + Vector) como Runnable.invoke(query)->List[Document].
    Usa pesos de settings.yaml: retrieval.ensemble_weights = [w_lex, w_dense]
    """
    class _VectorWrapper(Runnable):
        def __init__(self, vs, k: int, domains: Optional[List[str]]):
            self.vs = vs
            self.k = k
            self.domains = domains or None

        def invoke(self, query: str, config: Optional[dict] = None) -> List[Document]:
            if self.vs is None:
                return []
            # Intentar usar filtros por domain si el backend lo soporta (Chroma)
            search_kwargs = {"k": self.k}
            if self.domains:
                # LangChain Chroma usa "filter" con where {...}
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
            # Último intento
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

    # --- Pesos y top-k ---
    k = int(k_override or 0) or  int(os.environ.get("RAG_TOPK", "0")) or 8
    w_lex, w_dense = 0.5, 0.5  # defaults seguros; el llamador puede ajustar post-función si quiere

    def _read_weights_from_cfg() -> Tuple[float, float]:
        # Tratamos de leer desde settings.yaml ya cargado en memoria (no tenemos cfg acá)
        # Por simplicidad, respetamos los defaults; la fusión pondera linealmente.
        return w_lex, w_dense

    w_lex, w_dense = _read_weights_from_cfg()

    vec = _VectorWrapper(_VS, k, domains)
    lex = _BM25Wrapper(k, domains)

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
                    s = d.metadata.get(key) if d.metadata else None
                    if isinstance(s, (int, float)):
                        vals.append(float(s))
                if not vals:
                    return {}
                mx = max(vals) or 1.0
                out = {}
                for d in docs:
                    s = d.metadata.get(key) if d.metadata else None
                    if isinstance(s, (int, float)):
                        out[_hash_key(d.metadata.get("source","unk"), d.page_content)] = float(s) / mx
                return out

            # Vector retrievers en LangChain suelen no exponer score; usamos rank inverso como proxy
            vkeys = {}
            for rank, d in enumerate(vdocs, start=1):
                key = _hash_key(d.metadata.get("source","unk"), d.page_content)
                vkeys[key] = 1.0 / rank  # simple proxy [1, 1/2, 1/3, ...]

            lnorm = _norm_scores(ldocs, "score")
            # Fusión
            scores: Dict[str, float] = {}
            pool: Dict[str, Document] = {}

            def _merge(docs: List[Document], weight: float, prox_scores: Dict[str, float], meta_score_key: Optional[str] = None):
                for d in docs:
                    key = _hash_key(d.metadata.get("source","unk"), d.page_content)
                    pool[key] = d
                    s = prox_scores.get(key)
                    if s is None and meta_score_key and d.metadata.get(meta_score_key) is not None:
                        s = float(d.metadata[meta_score_key])
                    if s is None:
                        s = 0.0
                    scores[key] = scores.get(key, 0.0) + weight * s

            _merge(ldocs, w_lex, lnorm, meta_score_key="score")
            _merge(vdocs, w_dense, vkeys, meta_score_key=None)

            # Ordenar y cortar top-k
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            out: List[Document] = []
            for key, _ in ranked[: self.k]:
                d = pool[key]
                # Pasamos score combinado
                md = dict(d.metadata or {})
                md["score"] = float(scores[key])
                out.append(Document(page_content=d.page_content, metadata=md))
            return out

    # Si no hay VS, el ensemble sigue funcionando (solo BM25)
    return _Ensemble(vec, lex, k, w_lex, w_dense)

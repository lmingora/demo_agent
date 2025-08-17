from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from src.utils.logging import get_logger

log = get_logger("llm.embeddings")

def make_embeddings(cfg: Dict[str, Any]):
    emb_cfg = cfg.get("embeddings", {}) or {}
    model = emb_cfg.get("model", "mxbai-embed-large")
    log.info(f"Embeddings Ollama(model={model})")
    return OllamaEmbeddings(model=model)

def make_cached_embeddings(cfg: Dict[str, Any]):
    base = make_embeddings(cfg)
    cache_dir = Path(cfg["paths"]["cache_dir"]) / "emb_cache"
    store = LocalFileStore(str(cache_dir))
    log.info(f"CacheBackedEmbeddings -> {cache_dir}")
    return CacheBackedEmbeddings.from_bytes_store(base, store, namespace="emb_v1")

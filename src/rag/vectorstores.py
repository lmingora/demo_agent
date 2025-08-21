from __future__ import annotations
from typing import Dict, Any, List
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.llm.embeddings import make_cached_embeddings
from src.utils.logging import get_logger
from chromadb.config import Settings as ChromaSettings

log = get_logger("rag.vectorstores")

_VALID = re.compile(r"[a-zA-Z0-9._-]+")

def _sanitize_collection_name(name: str) -> str:
    """Normaliza el nombre de colección para cumplir las reglas de Chroma:
    - 3..512 chars
    - sólo [a-zA-Z0-9._-]
    - empieza y termina con alfanumérico
    """
    raw = (name or "").strip()
    # Reemplaza caracteres inválidos por '-'
    s = "".join(ch if _VALID.fullmatch(ch) else "-" for ch in raw)
    # Quita separadores al inicio/fin
    s = s.strip("._-")
    # Garantiza inicio/fin alfanuméricos
    if not s or not s[0].isalnum():
        s = f"k{s}"
    if not s[-1].isalnum():
        s = f"{s}0"
    # Longitud mínima 3
    if len(s) < 3:
        s = (s + "-main")
        if len(s) < 3:
            s = "kb-main"
    # Limita longitud para evitar problemas
    s = s[:128]
    return s

def make_vectorstore(cfg: Dict[str, Any]):
    vs_cfg = cfg.get("vectorstore", {}) or {}
    persist_directory = vs_cfg.get("persist_directory", "cache/chroma")
    raw_collection = vs_cfg.get("collection", "kb_main")
    collection = _sanitize_collection_name(raw_collection)

    if collection != raw_collection:
        log.warning(f"Nombre de colección inválido '{raw_collection}', usando '{collection}'")

    emb = make_cached_embeddings(cfg)
    log.info(f"Vectorstore Chroma(collection={collection}, dir={persist_directory})")
    try:
        client_settings = ChromaSettings(anonymized_telemetry=False)
        db = Chroma(
            collection_name=collection,
            embedding_function=emb,
            persist_directory=persist_directory,
            client_settings=client_settings,
        )
        log.info(f"Vectorstore Chroma(collection={collection}, dir={persist_directory})")

        return db
    except Exception as e:
        log.error(f"Error creando Chroma: {e}")
        raise

def add_documents(vs, docs: List[Document]):
    if not docs:
        return
    try:
        vs.add_documents(docs)
    except Exception as e:
        log.error(f"Error agregando documentos al vectorstore: {e}")
        raise

def delete_by_path_and_domain(vs, path: str, domain: str):
    """Borra documentos del vectorstore por path (y dominio, si el backend lo permite).
    Chroma 0.5+ prefiere operadores lógicos en 'where' ($and, $or).
    """
    try:
        # intento con $and (preferido)
        vs.delete(where={"$and": [{"path": path}, {"domain": domain}]})
    except Exception:
        try:
            # fallback: solo por path
            vs.delete(where={"path": path})
        except Exception as e:
            log.warning(f"No se pudo borrar docs previos (path={path}, domain={domain}): {e}")

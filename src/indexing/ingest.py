# src/indexing/ingest.py
from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
import shutil
import json
import os
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document

from src.rag.splitters import make_splitter_for
from src.rag.vectorstores import (
    make_vectorstore,
    add_documents,
    delete_by_path_and_domain,
)
from src.rag.toolbox import get_vectorstore
from src.rag.retrievers import bm25_store_path
from src.utils.logging import get_logger

log = get_logger("indexing.ingest")


# ----------------------------- helpers ------------------------------------- #
def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.error(f"Error leyendo {path}: {e}")
        raise


def _split_text(path: Path, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Splitter tolerante: soporta splitters que devuelven strings o Documents.
    """
    splitter = make_splitter_for(path, chunk_size=chunk_size, overlap=overlap)
    try:
        pieces = splitter.split_text(text)
    except AttributeError:
        # Algunos splitters podrían devolver Documents vía otra interfaz
        pieces = []
    if not pieces:
        return []

    # Si devuelve Documents, convertir a strings
    if isinstance(pieces, list) and pieces and hasattr(pieces[0], "page_content"):
        return [d.page_content for d in pieces]
    return pieces


def _atomic_write_jsonl(target: Path, records: List[dict]) -> None:
    """
    Escritura atómica de JSONL para evitar corrupción si hay procesos concurrentes.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=str(target.parent), encoding="utf-8") as tmp:
            for rec in records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)  # atomic on POSIX
    except Exception as e:
        log.error(f"Error escritura atómica JSONL {target}: {e}")
        raise


# ----------------------------- API pública --------------------------------- #
def index_path(
    cfg: Dict[str, Any],
    src_path: str,
    domain: str = "general",
    owner: str | None = None,
    private: bool = False,
) -> str:
    """
    Upsertea un archivo en data/<domain>/ y actualiza:
      1) VectorStore (borrando lo previo por (path, domain [, owner]) y reinsertando).
      2) Cache BM25 en JSONL: cache/bm25/<domain>.jsonl (por (path, owner)).

    Reglas:
      - Si el archivo YA está en data/<domain>/, indexa in-place (no copia).
      - chunk_size/overlap se leen de settings.yaml > retrieval.
      - Si el archivo rinde 0 chunks, borra entradas previas del mismo (path, owner) y retorna sin error.
      - Se persisten metadata.owner y metadata.private en VS y BM25 JSONL.

    Args:
      cfg: configuración completa (load_cfg()).
      src_path: ruta del archivo origen.
      domain: dominio lógico (carpeta bajo data/).
      owner: identificador del usuario dueño del documento (o None si es global).
      private: si True, marca el doc como privado para el owner.

    Returns:
      Mensaje de resultado.
    """
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    dst_dir = (data_dir / domain).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = Path(src_path).expanduser().resolve()
    if not src.exists():
        return f"No existe: {src}"

    # ¿ya está bajo data/<domain>?
    try:
        is_under_domain = dst_dir in src.parents
    except Exception:
        is_under_domain = False

    if is_under_domain:
        dst = src
    else:
        dst = (dst_dir / src.name).resolve()
        try:
            if src.exists() and dst.exists() and src.samefile(dst):
                pass  # mismo archivo físico
            else:
                shutil.copy2(src, dst)
        except FileNotFoundError:
            shutil.copy2(src, dst)
        except Exception as e:
            log.error(f"No se pudo copiar {src} -> {dst}: {e}")
            return f"Error copiando archivo: {e}"

    try:
        text = _read_text(dst)

        # chunking desde config si existe
        rcfg = (cfg.get("retrieval", {}) or {})
        chunk_size = int(rcfg.get("chunk_size", 1200))
        overlap = int(rcfg.get("overlap", 200))

        chunks = _split_text(dst, text, chunk_size=chunk_size, overlap=overlap)
        mtime = dst.stat().st_mtime

        # clave de path estable y legible (relativo a data/ si aplica)
        if data_dir in dst.parents:
            path_key = str(dst.relative_to(data_dir))
        else:
            path_key = str(dst)

        # 1) Vectorstore: eliminar versiones previas de este archivo (por (path, domain [, owner]))
        vs = get_vectorstore() or make_vectorstore(cfg)
        try:
            # nueva firma con owner (si está implementada en tu vectorstore)
            delete_by_path_and_domain(vs, path_key, domain, owner=owner)
        except TypeError:
            # compat: firma vieja sin owner
            delete_by_path_and_domain(vs, path_key, domain)

        # 2) BM25 JSONL por dominio
        bm25_dir = bm25_store_path(cfg)           # directorio cache/bm25
        jsonl = bm25_dir / f"{domain}.jsonl"      # archivo por dominio

        old: List[dict] = []
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    md = rec.get("metadata", {}) or {}
                    # conservar registros que no sean de este (path, owner)
                    same_path = (md.get("path") == path_key)
                    same_owner = ((md.get("owner") or None) == (owner or None))
                    if not (same_path and same_owner):
                        old.append(rec)
                except Exception:
                    continue

        # Si no hay chunks, persistimos la limpieza y salimos
        if not chunks:
            _atomic_write_jsonl(jsonl, old)
            msg = f"Indexado: dom={domain}, archivo={Path(path_key).name}, chunks=0 (sin contenido indexable)"
            log.info(msg)
            return msg

        # Construir documentos y upsert
        docs: List[Document] = []
        for i, ch in enumerate(chunks):
            md = {
                "path": path_key,
                "domain": domain,
                "chunk_id": f"{path_key}#{i}",
                "mtime": mtime,
                "owner": owner,
                "private": bool(private),
            }
            docs.append(Document(page_content=ch, metadata=md))
            old.append({"text": ch, "metadata": md})

        add_documents(vs, docs)
        _atomic_write_jsonl(jsonl, old)

        msg = f"Indexado: dom={domain}, archivo={Path(path_key).name}, chunks={len(chunks)}"
        log.info(msg)
        return msg

    except Exception as e:
        log.exception(f"Fallo indexando {dst}: {e}")
        return f"Error indexando: {e}"

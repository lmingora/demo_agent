# src/indexing/ingest.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import shutil, json, os
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document

from src.rag.splitters import make_splitter_for
from src.rag.vectorstores import make_vectorstore, add_documents, delete_by_path_and_domain
from src.rag.toolbox import get_vectorstore  # reusar si ya está inicializado
from src.rag.retrievers import bm25_store_path  # ← devuelve directorio, no archivo
from src.utils.logging import get_logger

log = get_logger("indexing.ingest")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.error(f"Error leyendo {path}: {e}")
        raise


def _split_text(path: Path, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Devuelve lista de strings con los chunks. Soporta splitters que devuelven strings
    o Document objects.
    """
    splitter = make_splitter_for(path, chunk_size=chunk_size, overlap=overlap)
    try:
        out = splitter.split_text(text)
    except TypeError:
        # Algunos splitters exponen firma distinta (poco común), intentamos fallback
        out = splitter.split_text(text=text)

    if not out:
        return []

    # Normalizar a lista de strings
    if isinstance(out, list) and out and hasattr(out[0], "page_content"):
        # p.ej., cuando el splitter devuelve Documents
        return [d.page_content for d in out]  # type: ignore[attr-defined]
    return list(out)


def _atomic_write_jsonl(target: Path, records: List[dict]):
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=str(target.parent), encoding="utf-8") as tmp:
            for rec in records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)  # atomic en POSIX
    except Exception as e:
        log.error(f"Error escritura atómica JSONL {target}: {e}")
        raise


def index_path(cfg: Dict[str, Any], src_path: str, domain: str = "general") -> str:
    """
    Upsertea un archivo en data/<domain>/ y actualiza:
      1) VectorStore (borrando lo previo por (path, domain) y reinsertando).
      2) Cache BM25 en JSONL: cache/bm25/<domain>.jsonl

    Reglas:
      - Si el archivo YA está en data/<domain>/, indexa in-place (no copia).
      - chunk_size/overlap se leen de settings.yaml > retrieval.
      - Si el archivo rinde 0 chunks, borra entradas previas y retorna sin error.
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
        overlap    = int(rcfg.get("overlap", 200))

        chunks = _split_text(dst, text, chunk_size=chunk_size, overlap=overlap)
        mtime = dst.stat().st_mtime

        # clave de path estable y legible (relativo a data/ si aplica)
        if data_dir in dst.parents:
            path_key = str(dst.relative_to(data_dir))
        else:
            path_key = str(dst)

        # 1) Vectorstore: eliminar versiones previas de este archivo
        vs = get_vectorstore() or make_vectorstore(cfg)
        delete_by_path_and_domain(vs, path_key, domain)

        # 2) BM25 JSONL por dominio
        bm25_dir = bm25_store_path(cfg)                 # ← directorio
        jsonl    = bm25_dir / f"{domain}.jsonl"         # ← archivo por dominio

        old: List[dict] = []
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    if rec.get("metadata", {}).get("path") != path_key:
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

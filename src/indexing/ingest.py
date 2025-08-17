from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import shutil, json, os
from tempfile import NamedTemporaryFile
from langchain_core.documents import Document
from src.rag.splitters import make_splitter_for
from src.rag.vectorstores import make_vectorstore, add_documents, delete_by_path_and_domain
from src.rag.retrievers import _bm25_store_path
from src.utils.logging import get_logger

log = get_logger("indexing.ingest")

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.error(f"Error leyendo {path}: {e}")
        raise

def _split_text(path: Path, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    splitter = make_splitter_for(path, chunk_size=chunk_size, overlap=overlap)
    if hasattr(splitter, "split_text"):
        return splitter.split_text(text)
    docs = splitter.split_text(text)
    return [d.page_content for d in docs]

def _atomic_write_jsonl(target: Path, records: List[dict]):
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=str(target.parent), encoding="utf-8") as tmp:
            for rec in records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)  # atomic on POSIX
    except Exception as e:
        log.error(f"Error escritura atomica JSONL {target}: {e}")
        raise

def index_path(cfg: Dict[str, Any], src_path: str, domain: str = "general") -> str:
    """Upsertea un archivo en data/<domain>/ y actualiza vectorstore + BM25 JSONL.
    - Si el archivo YA está en data/<domain>/, indexa in-place.
    - Lee chunk_size/overlap desde settings.yaml > retrieval si están.
    - Si el archivo rinde 0 chunks, elimina entradas previas y retorna sin error.
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
                pass  # mismo archivo
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

        # Vectorstore: eliminar versiones previas de este archivo
        vs = make_vectorstore(cfg)
        delete_by_path_and_domain(vs, path_key, domain)

        # BM25 JSONL actual: cargar y filtrar lo previo por path
        p = _bm25_store_path(cfg, domain)
        old = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    if rec["metadata"].get("path") != path_key:
                        old.append(rec)
                except Exception:
                    continue

        # Si no hay chunks, persistimos la limpieza y salimos
        if not chunks:
            _atomic_write_jsonl(p, old)
            msg = f"Indexado: dom={domain}, archivo={Path(path_key).name}, chunks=0 (sin contenido indexable)"
            log.info(msg)
            return msg

        # Construir documentos y upsert
        docs = []
        for i, ch in enumerate(chunks):
            md = {"path": path_key, "domain": domain, "chunk_id": f"{path_key}#{i}", "mtime": mtime}
            docs.append(Document(page_content=ch, metadata=md))
            old.append({"text": ch, "metadata": md})

        add_documents(vs, docs)
        _atomic_write_jsonl(p, old)

        msg = f"Indexado: dom={domain}, archivo={Path(path_key).name}, chunks={len(chunks)}"
        log.info(msg)
        return msg
    except Exception as e:
        log.exception(f"Fallo indexando {dst}: {e}")
        return f"Error indexando: {e}"

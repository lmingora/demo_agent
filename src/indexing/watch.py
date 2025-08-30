# src/indexing/watch.py
from __future__ import annotations
import os, time, hashlib, threading, glob, yaml
from typing import Dict, Any, List, Tuple
from src.indexing.ingest import index_path
from src.utils.logging import get_logger

log = get_logger("ingest.watch")

def _hash_file(p: str) -> Tuple[int, int]:
    try:
        st = os.stat(p)
        return (int(st.st_mtime), int(st.st_size))
    except Exception:
        return (0, 0)

def _iter_patterns(base: str, patterns: List[str]) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(base, pat), recursive=True))
    return sorted(set([f for f in files if os.path.isfile(f)]))

def run_watcher(cfg: Dict[str, Any]) -> None:
    wcfg = (cfg.get("ingest", {}) or {}).get("watch", {}) or {}
    if not wcfg.get("enabled", False):
        log.info("Watcher deshabilitado.")
        return
    base = (cfg.get("paths", {}) or {}).get("data", "data")
    polling_ms = int(wcfg.get("polling_ms", 5000))
    patterns = wcfg.get("patterns") or ["**/*.md", "**/*.txt", "**/*.pdf"]
    domains = wcfg.get("domains") or ["general"]

    state: Dict[str, Tuple[int, int]] = {}
    log.info(f"Watcher activado (base={base}, patterns={patterns}, domains={domains}).")
    while True:
        try:
            files = _iter_patterns(base, patterns)
            for f in files:
                h = _hash_file(f)
                if state.get(f) != h:
                    state[f] = h
                    # indexar por archivo (incremental)
                    for dom in domains:
                        try:
                            index_path(cfg, f, domain=dom, owner=None, replace=False)
                            log.info(f"Reindex incremental: {f} -> domain={dom}")
                        except Exception as e:
                            log.warning(f"Reindex falló para {f}: {e}")
        except Exception as e:
            log.warning(f"Watcher loop error: {e}")
        time.sleep(polling_ms/1000.0)

def start_in_background(cfg: Dict[str, Any]) -> None:
    t = threading.Thread(target=run_watcher, args=(cfg,), daemon=True)
    t.start()

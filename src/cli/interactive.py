from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import logging
import glob

from src.indexing.ingest import index_path
from src.rag.toolbox import refresh_bm25, list_domains
from src.utils.logging import get_logger

log = get_logger("cli")

# YAML es opcional: si no está, devolvemos mensaje claro
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

_HELP = """Comandos disponibles:
  :help                         Mostrar esta ayuda
  :index <ruta-archivo> [dominio]
                               Indexar/actualizar un archivo en un dominio (default: general)
  :index_docs                  Indexar en masa según config/docs.yaml (por dominio y glob)
  :agents                      Listar agentes configurados
  :domains                     Listar dominios disponibles
  :thread <id>                 Fijar thread_id para memoria conversacional
  :trace on|off                Activar/Desactivar logs DEBUG
  exit | quit                  Salir
"""

def banner():
    return "Pipeline LangGraph (Supervisor) — (:help para ver comandos, incluye :index_docs)"

def handle_cmd_index(cfg: Dict[str, Any], msg: str) -> str:
    """
    Uso:
        :index <ruta_o_archivo> [owner=<user_id>]
    Si se pasa owner=..., se guarda en metadata.owner.
    """
    parts = msg.split()
    if len(parts) < 2:
        return "Uso: :index <ruta_o_archivo> [owner=<user_id>]"
    path = parts[1].strip()
    owner = None

    # parsea argumentos extra como owner=xxx
    for token in parts[2:]:
        if token.startswith("owner="):
            owner = token.split("=", 1)[1].strip()

    from src.indexing.index import index_path
    try:
        n_chunks = index_path(path, cfg, owner=owner)
        if owner:
            return f"Indexados {n_chunks} chunks desde {path} (owner={owner})"
        else:
            return f"Indexados {n_chunks} chunks desde {path}"
    except Exception as e:
        return f"Error indexando {path}: {e}"

def handle_cmd_index_docs(cfg: Dict[str, Any], msg: str) -> str:
    """
    Indexa archivos según config/docs.yaml.
    Formato esperado:
      career:
        - "./data/career/*.txt"
      incident:
        - "./data/incidents/*.txt"
      general:
        - "./data/general/*"
    """
    if yaml is None:
        return "PyYAML no está instalado; ejecuta `pip install pyyaml`."

    root = Path(__file__).resolve().parents[2]
    docs_yaml = root / "config" / "docs.yaml"
    if not docs_yaml.exists():
        return f"No existe {docs_yaml}"

    try:
        spec = yaml.safe_load(docs_yaml.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return f"Error leyendo docs.yaml: {e}"

    if not isinstance(spec, dict):
        return "docs.yaml inválido: la raíz debe ser un mapa {dominio: [globs...]}"

    total = 0
    per_dom: Dict[str, int] = {}

    for domain, patterns in spec.items():
        if not patterns:
            continue
        if isinstance(patterns, str):
            patterns = [patterns]
        for pat in patterns:
            for fp in glob.glob(pat, recursive=True):
                res = index_path(cfg, fp, domain)
                log.info(res)
                total += 1
                per_dom[domain] = per_dom.get(domain, 0) + 1

    # refresca BM25 de dominios tocados
    for d in per_dom.keys():
        try:
            refresh_bm25(cfg, d)
        except Exception as e:
            log.warning(f"BM25 refresh fallo para dominio {d}: {e}")

    if total == 0:
        return "docs.yaml no produjo coincidencias. Verifica rutas y patrones."
    det = ", ".join(f"{d}={n}" for d, n in per_dom.items())
    return f"Indexación por docs.yaml finalizada. Archivos: {total} ({det})"

def handle_cmd_agents(cfg: Dict[str, Any]) -> str:
    agents = cfg.get("agents", [])
    if not agents:
        return "No hay agentes definidos en config/agents.yaml"
    lines = ["Agentes:"]
    for a in agents:
        d = ", ".join(a.get("domains") or [])
        lines.append(f"- {a['name']}: {a.get('role','')} (domains: {d})")
    return "\n".join(lines)

def handle_cmd_domains() -> str:
    doms = list_domains()
    if not doms:
        return "No hay dominios detectados."
    return "Dominios: " + ", ".join(doms)

def handle_cmd_help() -> str:
    return _HELP

def set_trace(on: bool):
    lvl = logging.DEBUG if on else logging.INFO
    logging.getLogger("app").setLevel(lvl)
    for h in logging.getLogger("app").handlers:
        h.setLevel(lvl)
    return f"Trace {'ON' if on else 'OFF'}"

# config/settings.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import os
import yaml
import logging

_LOGGER_NAME = "app.config"
log = logging.getLogger(_LOGGER_NAME)
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | " + _LOGGER_NAME + " | %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(logging.INFO)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        log.warning(f"YAML no encontrado: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            log.warning(f"YAML inválido (raíz no es mapa): {path}")
            return {}
        return data
    except Exception as e:
        log.error(f"Error leyendo YAML {path}: {e}")
        return {}


def _normalize_agents(raw: Any) -> List[Dict[str, Any]]:
    """
    Acepta lista de agentes o variantes y devuelve lista normalizada.
    Reglas:
    - Debe quedar una lista de dicts con 'name' obligatorio.
    - Si falta 'domains', setea ['general'].
    - Filtra entradas vacías.
    """
    agents: List[Dict[str, Any]] = []

    if not raw:
        return agents

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        # Soportar un formato viejo: {"agents": [...]} o {"agente1": {...}, ...}
        if "agents" in raw and isinstance(raw["agents"], list):
            items = raw["agents"]
        else:
            # mapa nombre->spec → convertir a lista
            items = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    v = dict(v)
                    v.setdefault("name", k)
                    items.append(v)
    else:
        return agents

    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if not name or not isinstance(name, str):
            continue
        d = dict(it)
        if not d.get("domains"):
            d["domains"] = ["general"]
        agents.append(d)

    return agents


def load_cfg() -> Dict[str, Any]:
    """
    Carga y fusiona:
      - config/settings.yaml (base)
      - config/agents.yaml   (agents, llm_defaults, embeddings, router opcional)
      - config/docs.yaml     (opcional, no se fusiona en cfg; lo dejamos a la CLI)
    Prioridad:
      - settings.yaml domina en llm_defaults/embeddings si hay duplicado.
      - agents.yaml provee 'agents' (lista final).
    """
    # Este archivo vive en .../config/settings.py
    config_dir = Path(__file__).resolve().parent
    repo_root = config_dir.parent

    settings_path = config_dir / "settings.yaml"
    agents_path   = config_dir / "agents.yaml"
    docs_path     = config_dir / "docs.yaml"  # opcional

    log.info(f"settings.yaml: {settings_path}")
    log.info(f"agents.yaml  : {agents_path}")
    if docs_path.exists():
        log.info(f"docs.yaml    : {docs_path}")

    settings = _load_yaml(settings_path)
    agents_y = _load_yaml(agents_path)

    # base: settings.yaml
    cfg: Dict[str, Any] = dict(settings or {})

    # merge suaves desde agents.yaml
    # (si en el futuro querés que agents.yaml overridee llm_defaults/embeddings, invertí la prioridad)
    if "llm_defaults" not in cfg and "llm_defaults" in agents_y:
        cfg["llm_defaults"] = agents_y["llm_defaults"]
    if "embeddings" not in cfg and "embeddings" in agents_y:
        cfg["embeddings"] = agents_y["embeddings"]
    if "router" not in cfg and "router" in agents_y:
        cfg["router"] = agents_y["router"]

    # AGENTS: siempre toman los de agents.yaml (normalizados)
    cfg["agents"] = _normalize_agents(agents_y.get("agents"))

    # Defaults sanos por si faltan bloques en settings.yaml
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("data_dir", "data")
    cfg["paths"].setdefault("cache_dir", "cache")
    cfg["paths"].setdefault("log_dir", "cache/logs")

    cfg.setdefault("vectorstore", {})
    cfg["vectorstore"].setdefault("provider", "chroma")
    cfg["vectorstore"].setdefault("collection", "kb_main")
    cfg["vectorstore"].setdefault("persist_directory", "cache/chroma")

    cfg.setdefault("retrieval", {})
    cfg["retrieval"].setdefault("k", 8)
    cfg["retrieval"].setdefault("ensemble_weights", [0.5, 0.5])
    cfg["retrieval"].setdefault("compression", False)
    cfg["retrieval"].setdefault("chunk_size", 1200)
    cfg["retrieval"].setdefault("overlap", 200)

    cfg.setdefault("bm25", {})
    cfg["bm25"].setdefault("store_dir", "cache/bm25")

    cfg.setdefault("features", {})
    cfg["features"].setdefault("use_langgraph_supervisor", True)
    cfg["features"].setdefault("use_checkpointer_sqlite", False)
    cfg["features"].setdefault("sqlite_checkpoint_path", "cache/checkpoints.sqlite3")

    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("log_level", "INFO")

    # Validación dura de agentes
    if not cfg["agents"]:
        raise ValueError(
            "No hay agentes definidos en config/agents.yaml -> clave 'agents'.\n"
            "Ejemplo mínimo:\n"
            "agents:\n"
            "  - name: rag_general\n"
            "    domains: [\"general\"]\n"
        )

    names = [a["name"] for a in cfg["agents"]]
    log.info(f"Agentes cargados: {', '.join(names)}")

    return cfg

# config/settings.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import yaml
import logging

# ───────────────────────── Logger ─────────────────────────
_LOGGER_NAME = "app.config"
log = logging.getLogger(_LOGGER_NAME)
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | " + _LOGGER_NAME + " | %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(logging.INFO)

# ───────────────────── Helpers de paths ───────────────────
def _repo_root() -> Path:
    # Raíz del repo: .../demo_agent/
    return Path(__file__).resolve().parents[1]

def _cfg_path(name: str) -> Path:
    # Archivo dentro de /config
    return _repo_root() / "config" / name

# ────────────────────── Helpers YAML ──────────────────────
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
    - Devuelve lista de dicts con 'name' obligatorio.
    - Si falta 'domains', setea ['general'].
    - Filtra entradas vacías o inválidas.
    """
    agents: List[Dict[str, Any]] = []
    if not raw:
        return agents

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        # Soportar formato viejo: {"agents": [...]} o {"agente1": {...}, ...}
        if "agents" in raw and isinstance(raw["agents"], list):
            items = raw["agents"]
        else:
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

# ─────────────────────── Cargas main ──────────────────────
def load_cfg() -> Dict[str, Any]:
    """
    Carga y fusiona:
      - config/settings.yaml (base)
      - config/agents.yaml   (agents, llm_defaults, embeddings, router opcional)
      - config/docs.yaml     (no se fusiona en cfg; se consulta aparte si hace falta)
    Prioridad:
      - settings.yaml domina en llm_defaults/embeddings si hay duplicado.
      - agents.yaml provee 'agents' (lista final normalizada).
    """
    settings_path = _cfg_path("settings.yaml")
    agents_path   = _cfg_path("agents.yaml")
    docs_path     = _cfg_path("docs.yaml")

    log.info(f"settings.yaml: {settings_path}")
    log.info(f"agents.yaml  : {agents_path}")
    if docs_path.exists():
        log.info(f"docs.yaml    : {docs_path}")

    settings = _load_yaml(settings_path)
    agents_y = _load_yaml(agents_path)

    # base: settings.yaml
    cfg: Dict[str, Any] = dict(settings or {})

    # merge suave desde agents.yaml (solo si faltan en settings)
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

    # Validación de agentes
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

# ────────────── Helpers adicionales (para imports) ─────────
def load_agents_cfg() -> List[Dict[str, Any]]:
    """
    Devuelve la lista de agentes leyendo directamente config/agents.yaml,
    normalizada con _normalize_agents. Útil para componentes que sólo
    necesitan los agentes (p. ej., el pre-router).
    """
    agents_path = _cfg_path("agents.yaml")
    agents_y = _load_yaml(agents_path)
    agents = _normalize_agents(agents_y.get("agents"))
    names = [a.get("name") for a in agents if a.get("name")]
    if names:
        log.info(f"Agentes cargados: {', '.join(names)}")
    else:
        log.warning("No se encontraron agentes en config/agents.yaml (clave 'agents' vacía)")
    return agents

def load_docs_cfg() -> Dict[str, Any]:
    """
    Devuelve el mapeo de dominios → rutas desde config/docs.yaml.
    Si no existe, devuelve {}.
    """
    docs_path = _cfg_path("docs.yaml")
    return _load_yaml(docs_path)

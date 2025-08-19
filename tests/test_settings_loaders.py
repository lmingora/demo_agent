# tests/test_settings_loaders.py
from config.settings import load_cfg, load_agents_cfg

def test_load_cfg_and_agents():
    cfg = load_cfg()
    assert isinstance(cfg, dict)
    assert "agents" in cfg and isinstance(cfg["agents"], list)
    agents = load_agents_cfg()
    assert isinstance(agents, list)
    names = [a.get("name") for a in agents]
    assert "rag_general" in names

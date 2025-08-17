from __future__ import annotations
from typing import Dict, Any
from langchain_ollama import ChatOllama
from src.utils.logging import get_logger

log = get_logger("llm.factory")

def make_chat(cfg: Dict[str, Any]):
    llm_cfg = cfg.get("llm_defaults", {}) or {}
    model = llm_cfg.get("model", "mistral:instruct")
    temperature = float(llm_cfg.get("temperature", 0.1))
    log.info(f"LLM ChatOllama(model={model}, temperature={temperature})")
    try:
        return ChatOllama(model=model, temperature=temperature)
    except Exception as e:
        log.error(f"Error inicializando ChatOllama: {e}")
        raise

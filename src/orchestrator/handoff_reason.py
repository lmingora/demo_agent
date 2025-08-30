from __future__ import annotations
from typing import List, Optional

def extract_handoff_reason(messages: List) -> Optional[str]:
    if not messages:
        return None
    for m in messages:
        try:
            text = getattr(m, "content", None) or m.get("content")
            if not text or "[HANDOFF_REASON]" not in text:
                continue
            start = text.find('reason="'); 
            if start == -1: 
                continue
            start += len('reason="')
            end = text.find('"', start)
            if end == -1:
                continue
            return text[start:end]
        except Exception:
            continue
    return None

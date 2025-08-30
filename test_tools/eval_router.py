#!/usr/bin/env python
"""
Evalúa la precisión del router (planner_node).

Uso:
    python tools/eval_router.py tests/router_testset.json

El JSON de test debe ser:
[
  {"q": "Necesito feedback 360 sobre mi rol", "expected": "career"},
  {"q": "Hubo un outage en prod",            "expected": "incident"},
  …
]
Imprime accuracy y ejemplos mal clasificados.
"""
import json, sys
from pathlib import Path

from pipeline_v4_3 import classify_query   # re-usa helper

def main(fp: str):
    cases = json.loads(Path(fp).read_text("utf-8"))
    total = ok = 0
    wrong = []
    for c in cases:
        total += 1
        pred = classify_query(c["q"])
        if pred == c["expected"]:
            ok += 1
        else:
            wrong.append((c["q"], pred, c["expected"]))
    acc = ok / total if total else 0
    print(f"Accuracy: {ok}/{total} = {acc:.2%}")
    if wrong:
        print("\nMal clasificados:")
        for q, p, e in wrong:
            print(f"- {q!r} → {p} (esperado {e})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: eval_router.py <ruta_json>")
        sys.exit(1)
    main(sys.argv[1])

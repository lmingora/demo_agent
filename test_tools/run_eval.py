#!/usr/bin/env python
"""
Wrapper simple para correr todos los evals disponibles.
Añade más llamadas si creas otras suites (p.ej. eval_rerank.py).
"""

import subprocess, sys, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
tests = root / "tests"

def run_router():
    fp = tests / "router_testset.json"
    if not fp.exists():
        print("⚠️  No hay tests/router_testset.json")
        return
    subprocess.run([sys.executable, str(root / "tools" / "eval_router.py"), str(fp)])

if __name__ == "__main__":
    run_router()

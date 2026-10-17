"""Rebuild every graph and report from scratch, in order. Stops at the first failure.

1. Moltbook: ingest, detect, rank blocks, agent answers (scripts/pipeline.py)
2. AI Village: ingest (agent messages only), memory benchmark
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
steps = [["scripts/pipeline.py"], ["scripts/ingest_village.py"], ["scripts/bench_memory.py"]]
for s in steps:
    t0 = time.perf_counter()
    print(f">>> {' '.join(s)}", flush=True)
    r = subprocess.run([sys.executable, "-u", *s], cwd=ROOT)
    print(f"<<< {s[0]} rc={r.returncode} in {time.perf_counter() - t0:.0f}s", flush=True)
    if r.returncode:
        sys.exit(r.returncode)
print("rebuild complete")

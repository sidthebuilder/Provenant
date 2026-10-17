"""Load the AI Village export into FalkorDB graph ``village``."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.db import Graph  # noqa: E402
from provenant.village.ingest import build, load  # noqa: E402

t0 = time.perf_counter()
r = build(log=lambda m: print(m, flush=True))
out = load(Graph("village"), r, log=lambda m: print(m, flush=True))
out["timing_s"]["total"] = round(time.perf_counter() - t0, 1)
print(json.dumps(out, indent=1))
Path("reports").mkdir(exist_ok=True)
Path("reports/ingest_village.json").write_text(json.dumps(out, indent=1))

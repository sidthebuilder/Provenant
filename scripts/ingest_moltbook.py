"""Load the Moltbook launch-week slice into FalkorDB graph ``moltbook_w1``."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.moltbook.ingest import ingest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "swarmchaser" / "data" / "moltbook"))
ap.add_argument("--graph", default="moltbook_w1")
ap.add_argument("--pattern", default="*.parquet")
a = ap.parse_args()
out = ingest(a.data, a.graph, a.pattern, log=lambda m: print(m, flush=True))
print(json.dumps(out, indent=1))
Path("reports").mkdir(exist_ok=True)
Path(f"reports/ingest_{a.graph}.json").write_text(json.dumps(out, indent=1))

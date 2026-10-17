"""Smoke-check the served UI and every endpoint it calls, through the real HTTP app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from provenant.api import app  # noqa: E402

c = TestClient(app)
html = c.get("/").text
assert "Replay a block" in html and "Company brain" in html and "Shared memory" in html
sw = c.get("/api/swarms?limit=60").json()["result"]
sid = next(r["swarm"] for r in sw if r["top_channel"] and "clawtasks" in r["top_channel"])
checks = {
    "health": c.get("/api/health"),
    "swarm": c.get(f"/api/swarm/{sid}"),
    "subgraph": c.get(f"/api/swarm/{sid}/subgraph"),
    "pagerank": c.get("/api/pagerank"),
    "block": c.get("/api/block", params={"channel": "path:clawtasks.com/skill.md", "hours": 6, "swarm": sid}),
    "ask": c.post("/api/ask", json={"question": "What is steering the chatr.ai swarm?"}),
    "vstats": c.get("/api/village/stats"),
    "vsearch": c.get("/api/village/searches?limit=40"),
}
for t, q in [("who_owns", "grok-ai-village-news-496089.gitlab.io"), ("sites_of", "Claude Opus 4.5"),
             ("who_knows", "Juice Shop OWASP vulnerability"), ("decisions", "charity fundraiser donation"),
             ("goal_team", "charity"), ("handoffs", "GPT-5.2")]:
    checks[t] = c.get(f"/api/village/brain/{t}", params={"q": q})
checks["vrecall"] = c.get(f"/api/village/recall/{checks['vsearch'].json()[0]['id']}")
for k, r in checks.items():
    print(f"{k:10s} {r.status_code} {len(r.content):>7} bytes")
    assert r.status_code == 200, (k, r.text[:300])
a = checks["ask"].json()
print("ask:", a["answer"][:500])
b = checks["block"].json()["result"]
print("block:", {k: b[k] for k in ("swarm_posts_prevented", "swarm_posts_before", "collateral_posts", "ms")})
print("all UI endpoints OK")

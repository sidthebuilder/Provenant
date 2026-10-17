"""Track 1 tests.

Unit tests build a tiny graph whose correct answers are known by hand, run the
real detection and block replay against it, and check exact counts. API tests
hit the real Moltbook graph and are skipped if it is not loaded.
"""

import pytest

from provenant.db import Graph, client, tracing
from provenant.moltbook.detect import detect
from provenant.moltbook.impact import block_impact
from provenant.moltbook.ingest import load_graph

try:
    client().list_graphs()
    HAVE_DB = True
except Exception:  # noqa: BLE001
    HAVE_DB = False

pytestmark = pytest.mark.skipif(not HAVE_DB, reason="FalkorDB not reachable")

H = 3600.0


def _fixture_rows():
    """Swarm A: 4 agents (4 owners) post template K1 linking chan X, two posts each, at t=0h and t=10h.
    Swarm B: 3 agents share payload K2, no channel. Noise: 6 agents post unrelated text; 2 of them link X."""
    agents, posts, has_key, links = [], [], [], []
    for i in range(4):
        agents.append({"id": f"a{i}", "name": f"a{i}", "owner": f"o{i}", "created_ts": 0.0})
        for j, t in enumerate((0.0, 10 * H)):
            pid = f"pa{i}_{j}"
            posts.append({"id": pid, "agent": f"a{i}", "ts": t + i, "venue": "", "text": "mint now", "injection": False})
            has_key.append({"p": pid, "k": "tmpl:K1"})
            links.append({"p": pid, "c": "dom:x.example"})
    for i in range(3):
        agents.append({"id": f"b{i}", "name": f"b{i}", "owner": "same", "created_ts": 0.0})
        posts.append({"id": f"pb{i}", "agent": f"b{i}", "ts": 5 * H, "venue": "", "text": "{json}", "injection": False})
        has_key.append({"p": f"pb{i}", "k": "payload:K2"})
    for i in range(6):
        agents.append({"id": f"n{i}", "name": f"n{i}", "owner": None, "created_ts": None})
        posts.append({"id": f"pn{i}", "agent": f"n{i}", "ts": 12 * H, "venue": "", "text": f"hello {i}", "injection": False})
        if i < 2:
            links.append({"p": f"pn{i}", "c": "dom:x.example"})
    return {
        "agents": agents, "posts": posts, "has_key": has_key, "links": links,
        "keys": [{"id": "tmpl:K1", "family": "tmpl", "sample": "mint now", "sim": 1.0},
                 {"id": "payload:K2", "family": "payload", "sample": "{json}", "sim": 1.0}],
        "channels": [{"id": "dom:x.example", "kind": "dom"}],
    }


@pytest.fixture(scope="module")
def g():
    graph = Graph("provenant_test_t1")
    load_graph(graph, _fixture_rows(), dataset="fixture", log=lambda m: None)
    from provenant.moltbook.detect import DetectConfig
    detect(graph, DetectConfig(max_key_agent_share=1.0, alpha=1.0, min_owners=2), log=lambda m: None)
    yield graph
    graph.delete()


def test_detects_both_swarms_with_exact_members(g):
    rows = g.read("MATCH (s:Swarm) RETURN s.n_agents, s.n_posts ORDER BY s.n_agents DESC")
    assert rows == [[4, 8], [3, 3]]


def test_owner_spread_and_kind(g):
    a = g.read_dicts("MATCH (s:Swarm {n_agents: 4}) RETURN s.distinct_owners AS o, s.kind AS k")[0]
    b = g.read_dicts("MATCH (s:Swarm {n_agents: 3}) RETURN s.distinct_owners AS o, s.top_owner_share AS t, s.kind AS k")[0]
    assert a == {"o": 4, "k": "copy-paste wave"}
    assert b["o"] == 1 and b["t"] == 1.0 and b["k"] == "payload campaign"


def test_steering_channel_found_only_for_swarm_a(g):
    rows = g.read("MATCH (s:Swarm)-[r:STEERED_BY]->(c:Channel) RETURN s.n_agents, c.id, r.share, r.owners")
    assert rows == [[4, "dom:x.example", 1.0, 4]]


def test_block_impact_exact_counts_and_original_untouched(g):
    before = g.read("MATCH (p:Post) RETURN count(p)")[0][0]
    # Block from t = 6h: the four 10h swarm posts and the two 12h noise posts are rejected.
    r = block_impact(g, "dom:x.example", at_ts=6 * H)
    assert r["blocked_posts"] == 6
    assert r["blocked_swarm_posts"] == 4
    assert r["collateral_posts"] == 2
    assert r["swarm_posts_prevented"] == 4
    assert r["swarm_agents_cut_off"] == 0          # every agent already posted at 0h
    assert g.read("MATCH (p:Post) RETURN count(p)")[0][0] == before
    assert "provenant_test_t1__whatif" not in " ".join(client().list_graphs())


def test_block_from_start_cuts_off_all_agents(g):
    r = block_impact(g, "dom:x.example", at_ts=-1.0)
    s = next(x for x in r["swarms"] if x["agents_before"] == 4)
    assert s["agents_after"] == 0 and s["posts_after"] == 0


def test_every_query_is_traced_and_reads_are_read_only(g):
    with tracing() as tr:
        g.read("MATCH (s:Swarm) RETURN count(s)")
        with pytest.raises(Exception):
            g.read("CREATE (:Bad)")
    assert tr.queries and tr.queries[0].mode == "read" and tr.queries[0].ms >= 0


# ------------------------------------------------------------- real graph
REAL = "moltbook_w1"
need_real = pytest.mark.skipif(not HAVE_DB or REAL not in (client().list_graphs() if HAVE_DB else []),
                               reason="real Moltbook graph not loaded")


@need_real
def test_real_graph_reproduces_mint_campaign():
    r = Graph(REAL).read("MATCH (s:Swarm) RETURN s.n_agents, s.n_posts, s.distinct_owners ORDER BY s.n_agents DESC LIMIT 1")
    assert r == [[442, 5942, 278]]


@need_real
def test_api_end_to_end():
    from fastapi.testclient import TestClient

    from provenant.api import app
    c = TestClient(app)
    h = c.get("/api/health").json()
    assert h["posts"] == 125885 and h["swarms"] > 100
    sw = c.get("/api/swarms?limit=5").json()
    assert sw["result"][0]["agents"] == 442 and sw["evidence"]
    a = c.post("/api/ask", json={"question": "What is steering the skill-file swarm, and what should we block?",
                                 "mode": "planner"}).json()
    assert a["recommendation"]["action"] == "block"
    assert "clawtasks" in a["recommendation"]["channel"]
    cited = {int(n) for n in __import__("re").findall(r"\[s(\d+)\]", a["answer"])}
    assert cited and cited <= {s["n"] for s in a["steps"]}
    assert c.get("/").status_code == 200

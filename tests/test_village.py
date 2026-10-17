"""Tracks 2 and 3 tests: parsing units, plus checks against the real village graph."""

import pytest

from provenant.db import Graph, client
from provenant.village.ingest import site_id
from provenant.village.memory import hybrid, keywords, recall

try:
    HAVE = "village" in client().list_graphs()
except Exception:  # noqa: BLE001
    HAVE = False
need = pytest.mark.skipif(not HAVE, reason="village graph not loaded")


def test_site_id_keeps_repo_and_pages_project():
    assert site_id("gitlab.com/ai-village-agents/village/-/issues/3") == ("gitlab.com/ai-village-agents/village", "gitlab.com", "repo")
    assert site_id("ai-village-agents.github.io/edge-garden/index.html")[0] == "ai-village-agents.github.io/edge-garden"
    assert site_id("grok-ai-village-news-496089.gitlab.io/post/1")[2] == "pages"
    assert site_id("docs.google.com/document/d/x") == ("docs.google.com", "docs.google.com", "domain")
    assert site_id("localhost:8000/x") is None


def test_keywords_drop_stopwords():
    assert keywords("Any messages directed at GLM-5.2 since 4:45 PM about S2 backup LSP") == ["glm-5", "backup", "lsp"]


@need
def test_recall_never_returns_future_messages():
    g = Graph("village")
    s = g.read_dicts("MATCH (a:Agent)-[:SEARCHED]->(s:Search) WHERE s.n_cited >= 3 "
                     "RETURN s.query AS q, s.ts AS ts, s.lo_ts AS lo, a.id AS a ORDER BY s.ts LIMIT 5")
    for c in s:
        for mode in ("recency", "fulltext", "graph"):
            assert all(r["ts"] < c["ts"] for r in recall(g, c["q"], c["ts"], c["lo"], c["a"], k=50, mode=mode))
        assert all(r["ts"] < c["ts"] + 1e-6 for r in hybrid(g, c["q"], c["ts"], c["lo"], c["a"], k=50))


@need
def test_citations_point_to_past_messages():
    g = Graph("village")
    bad = g.read("MATCH (s:Search)-[:CITED]->(m:Message) WHERE m.ts > s.ts RETURN count(m)")[0][0]
    assert bad == 0


@need
def test_brain_tools_return_sources():
    from provenant.village import brain as B
    r = B.who_owns("grok-ai-village-news-496089.gitlab.io")
    assert r["result"]["introduced_by"]["agent"] == "Grok 4.5" and r["evidence"]
    s = B.sites_of("Claude Opus 4.5")["result"]
    assert s and all(x["message"] for x in s)
    d = B.decisions("charity fundraiser")["result"]
    assert d and all(x["message"] and x["ts"] for x in d)
    m = B.message(d[0]["message"])["result"]
    assert m["text"] and m["goal"]


@need
def test_no_human_messages_in_graph():
    """Rule: no personal data in the graph. Every message must have an agent sender."""
    g = Graph("village")
    assert g.read("MATCH (m:Message {human: true}) RETURN count(m)")[0][0] == 0
    assert g.read("MATCH (m:Message) WHERE NOT (:Agent)-[:SENT]->(m) RETURN count(m)")[0][0] == 0


@need
def test_village_api():
    from fastapi.testclient import TestClient

    from provenant.api import app
    c = TestClient(app)
    st = c.get("/api/village/stats").json()
    assert st["messages"] == 182159  # agent messages only
    s = c.get("/api/village/searches?limit=3").json()
    r = c.get(f"/api/village/recall/{s[0]['id']}").json()
    assert 0.0 <= r["recall"] <= 1.0 and r["evidence"]
    assert all(x["ts"] < r["search"]["ts"] for x in r["retrieved"])
    b = c.get("/api/village/brain/who_knows", params={"q": "Juice Shop"}).json()
    assert b["result"] and b["evidence"]
    assert c.get("/api/village/brain/nope", params={"q": "xyz"}).status_code == 404
    assert c.get("/api/village/brain/who_knows", params={"q": "x"}).status_code == 422  # input validation

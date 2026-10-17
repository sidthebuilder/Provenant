"""FalkorDB access layer.

Every query an agent tool runs goes through :class:`Graph`, which

- uses ``GRAPH.RO_QUERY`` for reads, so a tool can never write by accident,
- applies a timeout to every call,
- records the exact Cypher, parameters, row count and wall time in a trace,
  so the UI and the agent's answer can show the evidence behind each claim.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()  # reads .env if present (FALKORDB_HOST, LLM_API_KEY, ...); never committed

HOST = os.environ.get("FALKORDB_HOST", "localhost")
PORT = int(os.environ.get("FALKORDB_PORT", "6379"))
READ_TIMEOUT_MS = int(os.environ.get("PROVENANT_READ_TIMEOUT_MS", "15000"))
WRITE_TIMEOUT_MS = int(os.environ.get("PROVENANT_WRITE_TIMEOUT_MS", "600000"))


@dataclass
class QueryRecord:
    graph: str
    cypher: str
    params: dict
    rows: int
    ms: float
    mode: str  # "read" or "write"


@dataclass
class Trace:
    queries: list[QueryRecord] = field(default_factory=list)

    def to_list(self) -> list[dict]:
        return [asdict(q) for q in self.queries]


_trace: ContextVar[Trace | None] = ContextVar("provenant_trace", default=None)


@contextmanager
def tracing():
    """Collect every query run inside the block."""
    t = Trace()
    token = _trace.set(t)
    try:
        yield t
    finally:
        _trace.reset(token)


_client: FalkorDB | None = None


def client() -> FalkorDB:
    global _client
    if _client is None:
        _client = FalkorDB(host=HOST, port=PORT)
    return _client


def _clip(params: dict | None) -> dict:
    """Keep traces small: long lists are summarised, not copied."""
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, list) and len(v) > 5:
            out[k] = f"<list of {len(v)}>"
        else:
            out[k] = v
    return out


class Graph:
    def __init__(self, name: str):
        self.name = name
        self._g = client().select_graph(name)

    def _record(self, cypher, params, rows, ms, mode):
        t = _trace.get()
        if t is not None:
            t.queries.append(QueryRecord(self.name, " ".join(cypher.split()), _clip(params), rows, round(ms, 2), mode))

    def read(self, cypher: str, params: dict | None = None, timeout_ms: int = READ_TIMEOUT_MS) -> list[list]:
        t0 = time.perf_counter()
        rs = self._g.ro_query(cypher, params or {}, timeout=timeout_ms).result_set
        self._record(cypher, params, len(rs), (time.perf_counter() - t0) * 1000, "read")
        return rs

    def read_dicts(self, cypher: str, params: dict | None = None, timeout_ms: int = READ_TIMEOUT_MS) -> list[dict]:
        t0 = time.perf_counter()
        res = self._g.ro_query(cypher, params or {}, timeout=timeout_ms)
        cols = [c[1] if isinstance(c, (list, tuple)) else c for c in res.header]
        rows = [dict(zip(cols, r)) for r in res.result_set]
        self._record(cypher, params, len(rows), (time.perf_counter() - t0) * 1000, "read")
        return rows

    def write(self, cypher: str, params: dict | None = None, timeout_ms: int = WRITE_TIMEOUT_MS):
        t0 = time.perf_counter()
        res = self._g.query(cypher, params or {}, timeout=timeout_ms)
        self._record(cypher, params, len(res.result_set), (time.perf_counter() - t0) * 1000, "write")
        return res

    def exists(self) -> bool:
        return self.name in client().list_graphs()

    def delete(self):
        if self.exists():
            self._g.delete()

    def copy_to(self, target: str) -> "Graph":
        """Fork this graph with GRAPH.COPY (used for what-if simulation)."""
        t0 = time.perf_counter()
        if target in client().list_graphs():
            client().select_graph(target).delete()
        # GRAPH.COPY forks the server process; back-to-back copies can fail
        # with "could not fork" while the previous child is still exiting.
        for attempt in range(8):
            try:
                client().connection.execute_command("GRAPH.COPY", self.name, target)
                break
            except Exception as e:  # noqa: BLE001
                if "could not fork" not in str(e) or attempt == 7:
                    raise
                time.sleep(0.5 * (attempt + 1))
        fork = Graph(target)
        fork.wait_indexes()
        self._record(f"GRAPH.COPY {self.name} {target}", {}, 0, (time.perf_counter() - t0) * 1000, "write")
        return fork

    def wait_indexes(self, timeout_s: float = 30.0):
        """Block until every index is OPERATIONAL. A copied graph rebuilds its
        indexes in the background; querying earlier can silently miss rows."""
        deadline = time.time() + timeout_s
        while True:
            rows = self._g.ro_query("CALL db.indexes() YIELD status RETURN status").result_set
            if all(str(r[0]).upper() == "OPERATIONAL" for r in rows):
                return
            if time.time() > deadline:
                raise TimeoutError(f"indexes on {self.name} not ready after {timeout_s}s: {rows}")
            time.sleep(0.05)


def batched(rows: list, size: int = 2000):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]

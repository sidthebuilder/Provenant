"""Check which FalkorDB procedures and commands this server supports."""
import json

from falkordb import FalkorDB

db = FalkorDB(host="localhost", port=6379)
g = db.select_graph("probe")
try:
    g.delete()
except Exception:
    pass
g.query("CREATE (:N {i:1})-[:R]->(:N {i:2})-[:R]->(:N {i:3}), (:N {i:4})-[:R]->(:N {i:5})")
out = {"server": db.connection.execute_command("INFO", "server").get("redis_version")}
out["modules"] = str(db.connection.execute_command("MODULE", "LIST"))
procs = g.query("CALL dbms.procedures() YIELD name RETURN name ORDER BY name").result_set
out["procedures"] = [p[0] for p in procs]

tests = {
    "wcc": "CALL algo.WCC({nodeLabels:['N'], relationshipTypes:['R']}) YIELD node, componentId RETURN node.i, componentId ORDER BY node.i",
    "pagerank": "CALL algo.pageRank('N','R') YIELD node, score RETURN node.i, score ORDER BY node.i",
    "betweenness": "CALL algo.betweenness({nodeLabels:['N'], relationshipTypes:['R']}) YIELD node, score RETURN node.i, score ORDER BY node.i",
    "cdlp": "CALL algo.labelPropagation({nodeLabels:['N'], relationshipTypes:['R']}) YIELD node, communityId RETURN node.i, communityId ORDER BY node.i",
    "bfs": "MATCH (a:N {i:1}) CALL algo.BFS(a, 0, 'R') YIELD nodes RETURN size(nodes)",
    "shortest": "MATCH (a:N {i:1}),(b:N {i:3}) RETURN length(shortestPath((a)-[:R*]->(b)))",
}
for name, q in tests.items():
    try:
        out[name] = g.query(q).result_set
    except Exception as e:  # noqa: BLE001
        out[name] = f"ERR {e}"

try:
    db.connection.execute_command("GRAPH.COPY", "probe", "probe_fork")
    out["copy"] = db.select_graph("probe_fork").query("MATCH (n) RETURN count(n)").result_set
    db.select_graph("probe_fork").delete()
except Exception as e:  # noqa: BLE001
    out["copy"] = f"ERR {e}"
try:
    out["ro_query"] = g.ro_query("MATCH (n) RETURN count(n)", timeout=1000).result_set
    out["ro_write"] = str(g.ro_query("CREATE (:X)"))
except Exception as e:  # noqa: BLE001
    out["ro_write"] = f"ERR (expected) {e}"
g.delete()
print(json.dumps(out, indent=1, default=str))

"""Delete leftover what-if fork graphs (e.g. after a crash mid-replay)."""
from falkordb import FalkorDB

db = FalkorDB()
gone = [n for n in db.list_graphs() if "__whatif_" in n or n in ("dbg_fork", "probe_fork")]
for n in gone:
    db.select_graph(n).delete()
print("deleted", gone, "remaining", db.list_graphs())

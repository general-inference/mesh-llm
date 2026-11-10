"""Top sampled CPU functions (leaf frames and inclusive) from an nsys SQLite export."""
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
if "SAMPLING_CALLCHAINS" not in tables:
    print("no CPU samples in", sys.argv[1], sorted(t for t in tables if "SAMPL" in t)); sys.exit(0)
names = dict(db.execute("select id, value from StringIds"))
leaf, incl = collections.Counter(), collections.Counter()
total = 0
cur_id, seen = None, set()
for cid, depth, sym in db.execute("select id, stackDepth, symbol from SAMPLING_CALLCHAINS order by id, stackDepth"):
    name = names.get(sym, str(sym))[:140]
    if cid != cur_id:
        cur_id, seen = cid, set(); total += 1
    if depth == 0:
        leaf[name] += 1
    if name not in seen:
        incl[name] += 1; seen.add(name)
print(f"CPU samples: {total}")
print("-- leaf"); [print(f"{100*n/total:5.1f}%  {k}") for k, n in leaf.most_common(25)]
print("-- inclusive"); [print(f"{100*n/total:5.1f}%  {k}") for k, n in incl.most_common(40)]

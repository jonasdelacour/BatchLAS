import collections
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)
evs = d["traceEvents"]
c = collections.Counter(e["name"] for e in evs)
for name, n in c.most_common(12):
    print(f"  {n:6d}  {name}")

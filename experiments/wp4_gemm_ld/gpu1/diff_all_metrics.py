import csv, sys
base = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_gemm_ld/gpu1/"


def load(f):
    rows = list(csv.reader(open(base + f)))
    hdr = rows[0]
    units = rows[1]
    data = rows[-1]
    d = {}
    for h, u, v in zip(hdr, units, data):
        try:
            val = float(v.replace(",", ""))
        except ValueError:
            continue
        d[h] = (val, u.strip())
    return d


a = load(sys.argv[1])
b = load(sys.argv[2])
# ratio of durations, for identifying "scales with time" metrics
out = []
for k in a:
    if k not in b:
        continue
    va, ua = a[k]
    vb, ub = b[k]
    if ua != ub:
        out.append((float('inf'), k, va, ua, vb, ub, "UNIT-CHANGE"))
        continue
    if abs(va) < 1e-9 and abs(vb) < 1e-9:
        continue
    if abs(va) < 1e-9:
        out.append((float('inf'), k, va, ua, vb, ub, ""))
        continue
    r = vb / va
    out.append((r, k, va, ua, vb, ub, ""))

thr = float(sys.argv[3]) if len(sys.argv) > 3 else 1.15
sel = [o for o in out if o[0] == float('inf') or o[0] > thr or o[0] < 1 / thr]
sel.sort(key=lambda o: -(o[0] if o[0] != float('inf') else 1e18))
print(f"{len(sel)} of {len(out)} metrics differ by more than {thr}x")
for r, k, va, ua, vb, ub, note in sel:
    rs = "inf" if r == float('inf') else f"{r:.3f}"
    print(f"{rs:>8}  {k}  {va:g} {ua} -> {vb:g} {ub} {note}")

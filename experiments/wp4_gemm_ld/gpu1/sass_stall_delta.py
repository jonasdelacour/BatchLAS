import csv, sys
T = "/home/jonaslacour/.claude/jobs/20812aa0/tmp/"


def load(f):
    rows = list(csv.reader(open(T + f)))
    hdr = rows[1]
    return hdr, [dict(zip(hdr, r)) for r in rows[2:] if r]


h0, a = load("sass-p0.csv")
h1, b = load("sass-p384.csv")
assert len(a) == len(b), (len(a), len(b))
mism = sum(1 for x, y in zip(a, b) if x["Source"] != y["Source"])
print(f"{len(a)} SASS lines paired by index; {mism} instruction-text mismatches")


def num(x):
    try:
        return float((x or "").replace(",", ""))
    except Exception:
        return 0.0


rows = list(zip(a, b))
col = "Warp Stall Sampling (All Samples)"
tot0 = sum(num(r[col]) for r, q in rows)
tot1 = sum(num(q[col]) for r, q in rows)
print(f"total warp-stall samples: p0={tot0:.0f}  p384={tot1:.0f}  ratio={tot1/tot0:.2f}")
d_tot = tot1 - tot0
print()
print("Top 12 SASS instructions by INCREASE in warp-stall samples:")
rows.sort(key=lambda t: -(num(t[1][col]) - num(t[0][col])))
print(f"{'p0':>8} {'p384':>8} {'delta':>8} {'%dtot':>7}  {'bar_p0':>7} {'bar_p384':>8} {'lsb_p0':>7} {'lsb_p384':>8}  instruction")
for r, q in rows[:12]:
    d = num(q[col]) - num(r[col])
    print(f"{num(r[col]):8.0f} {num(q[col]):8.0f} {d:8.0f} {100*d/d_tot:6.1f}%  "
          f"{num(r['stall_barrier']):7.0f} {num(q['stall_barrier']):8.0f} "
          f"{num(r['stall_long_sb']):7.0f} {num(q['stall_long_sb']):8.0f}  {r['Source'].strip()[:64]}")

print()
print("All BAR / LDG / STG instructions:")
print(f"{'p0':>8} {'p384':>8} {'delta':>8}  {'l2sec_p0':>10} {'l2sec_p384':>10} {'excess_p0':>10} {'excess_384':>10}  instruction")
rows2 = list(zip(a, b))
for r, q in rows2:
    s = r["Source"].strip()
    op = s.split()[0].lstrip("@").split(".")[0]
    if s.startswith("@"):
        op = s.split()[1].split(".")[0]
    if op in ("BAR", "BSSY", "BSYNC", "LDG", "STG", "LDGSTS", "LDS", "STS"):
        if op in ("LDS", "STS", "BSSY", "BSYNC"):
            continue
        print(f"{num(r[col]):8.0f} {num(q[col]):8.0f} {num(q[col])-num(r[col]):8.0f}  "
              f"{num(r['L2 Theoretical Sectors Global']):10.0f} {num(q['L2 Theoretical Sectors Global']):10.0f} "
              f"{num(r['L2 Theoretical Sectors Global Excessive']):10.0f} {num(q['L2 Theoretical Sectors Global Excessive']):10.0f}  {s[:58]}")

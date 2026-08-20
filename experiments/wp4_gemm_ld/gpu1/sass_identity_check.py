import csv
T = "/home/jonaslacour/.claude/jobs/20812aa0/tmp/"


def load(f):
    rows = list(csv.reader(open(T + f)))
    hdr = rows[1]
    return [dict(zip(hdr, r)) for r in rows[2:] if r]


a = load("sass-p0.csv")
b = load("sass-p384.csv")


def num(x):
    try:
        return float((x or "").replace(",", ""))
    except Exception:
        return 0.0


print("=== identity check: per-instruction traffic counters, p0 vs p384 ===")
cols = ["Instructions Executed", "L1 Tag Requests Global", "L2 Theoretical Sectors Global",
        "L2 Theoretical Sectors Global Excessive", "L1 Wavefronts Shared",
        "L1 Wavefronts Shared Excessive", "L1 Conflicts Shared N-Way"]
diff = 0
for i, (r, q) in enumerate(zip(a, b)):
    for c in cols:
        if num(r[c]) != num(q[c]):
            diff += 1
            print(f"  idx {i} {c}: {num(r[c]):.0f} -> {num(q[c]):.0f}   {r['Source'].strip()[:50]}")
print(f"  -> {diff} per-instruction traffic-counter differences over {len(a)} instructions x {len(cols)} counters")

print()
print("=== SASS window 143..166 (staging + barrier + first fragment read) ===")
print(f"{'idx':>4}  {'exec':>9} {'stallAll_p0':>11} {'stallAll_p384':>13} {'bar_p0':>7} {'bar_p384':>8} {'lsb_p0':>7} {'lsb_p384':>8}  instruction")
for i in range(143, 167):
    r, q = a[i], b[i]
    print(f"{i:4d}  {num(r['Instructions Executed']):9.0f} "
          f"{num(r['Warp Stall Sampling (All Samples)']):11.0f} {num(q['Warp Stall Sampling (All Samples)']):13.0f} "
          f"{num(r['stall_barrier']):7.0f} {num(q['stall_barrier']):8.0f} "
          f"{num(r['stall_long_sb']):7.0f} {num(q['stall_long_sb']):8.0f}  {r['Source'].strip()[:52]}")

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


print("=== main-loop global/shared memory instructions (p0 counters; identical in p384 unless shown) ===")
print(f"{'idx':>5} {'instr-exec':>12} {'L1req':>12} {'L2sec':>12} {'L2ideal':>12} {'L2excess':>10} {'sec/req':>8} {'shWF':>12} {'shWFideal':>10} {'shWFexc':>10}  instruction")
for i, (r, q) in enumerate(zip(a, b)):
    s = r["Source"].strip()
    asp = r["Address Space"]
    if asp in ("-", ""):
        continue
    ie = num(r["Instructions Executed"])
    req = num(r["L1 Tag Requests Global"])
    sec = num(r["L2 Theoretical Sectors Global"])
    idl = num(r["L2 Theoretical Sectors Global Ideal"])
    exc = num(r["L2 Theoretical Sectors Global Excessive"])
    swf = num(r["L1 Wavefronts Shared"])
    swi = num(r["L1 Wavefronts Shared Ideal"])
    swe = num(r["L1 Wavefronts Shared Excessive"])
    if ie < 100000:
        continue
    spr = sec / req if req else 0
    print(f"{i:5d} {ie:12.0f} {req:12.0f} {sec:12.0f} {idl:12.0f} {exc:10.0f} {spr:8.2f} {swf:12.0f} {swi:10.0f} {swe:10.0f}  [{asp}] {s[:46]}")

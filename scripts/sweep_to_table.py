#!/usr/bin/env python3
"""Convert the potrf route sweeps into seed tuned tables.

Spec: docs/design/flat-kernel-selection.md section 7 (conversion rules), section 6.3
(tie rule) and section 5.4 (file format and nearest-point lookup). Inputs are the
forced-route sweeps in benchmarks/results/routing/ (see the README there); outputs
are tuned/potrf.<dtype>.<device>.txt.

    scripts/sweep_to_table.py                 # write tuned/*.txt, print a summary
    scripts/sweep_to_table.py --check         # re-derive and diff, check nearest(), exit 1 on failure
    scripts/sweep_to_table.py --check --check-points pts.txt

--check is the pure-data half of the acceptance gate (section 10.2): every table must
equal what the sweeps produce, and the section 5.4 lookup must return each measured
cell's own row with the measured winner first. It also prints where the section 10.3
off-grid points land, for review. A points file holds one "n" or "n batch" per line.

Python 3 standard library only.
"""

import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTING = "benchmarks/results/routing"
TUNED = "tuned"

# Candidate-list order of choice.hh (section 4.2): the tie-break order.
CANDIDATE_ORDER = ["tiny", "cta", "lpanel:panel=8", "lpanel:panel=16", "blocked", "vendor"]

ARM_SPELLING = {
    "route:native:tiny": "tiny",
    "route:native:cta": "cta",
    "route:native:lpanel": "lpanel:panel=8",
    "route:native:blocked": "blocked",
    "vendor": "vendor",
}
# The coverage spellings (chosen_origin:chosen_algo) each arm may have reached (rule 1):
# the RouteTable era's, then flat selection's, whose chosen_algo is the choice spelling.
ARM_ROUTE = {
    "route:native:tiny": ("native:tiny",),
    "route:native:cta": ("native:cta",),
    "route:native:lpanel": ("native:lpanel", "native:lpanel:panel=8"),
    "route:native:blocked": ("native:blocked",),
    "vendor": ("vendor:auto", "vendor:vendor"),
}
DTYPES = {
    "float": "float", "double": "double", "cfloat": "cfloat", "cdouble": "cdouble",
    "complex<float>": "cfloat", "complex<double>": "cdouble",
}

TIE = 0.03
NOISY = 0.10

SOURCES = {
    "sm_120": {
        "files": ["sm120_potrf_sweep.jsonl", "sm120_potrf_sweep_edges.jsonl"],
        "batchlas": "a1063892",
        "note": "converted, passes 1+2",
        "current_only": False,
    },
    "sm_89": {
        "files": ["sm89_potrf_archive.jsonl"],
        "batchlas": "unknown",
        "note": "converted, kernel_current rows only; repeated medians averaged",
        "current_only": True,
    },
}

DEFAULT_OFFGRID_N = [44, 72, 104, 144, 208, 272, 352, 480, 704, 1152]
DEFAULT_OFFGRID_BATCH = [512, 8192, 32768]


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def uplo_of(r):
    if "uplo" in r:
        return r["uplo"]
    return "U" if r["op"] == "potrf_upper" else "L"


def collect(device, stats):
    """Return {(dtype, uplo, n, batch): {spelling: [median, ...]}} for one device.

    Each list entry is one independent median: a pass on sm_120, an archived
    measurement on sm_89. Rule 2's drop of sm_89 cells with < 2 arms happens here.
    """
    spec = SOURCES[device]
    cells = defaultdict(lambda: defaultdict(list))
    seen = set()
    for fname in spec["files"]:
        for r in load_rows(os.path.join(REPO, ROUTING, fname)):
            if r["op"] not in ("potrf", "potrf_upper"):
                stats["dropped_op"] += 1
                continue
            if spec["current_only"] and not r.get("kernel_current"):
                stats["dropped_not_current"] += 1
                continue
            arm = r["arm"]
            if arm not in ARM_SPELLING:
                stats["dropped_unpinned_arm"] += 1
                continue
            if not r["ok"]:
                stats["dropped_not_ok"] += 1
                continue
            if r["route"] not in ARM_ROUTE[arm]:
                stats["dropped_route_mismatch"] += 1
                continue
            if r["m"] != r["n"] or not r.get("time_ms"):
                stats["dropped_malformed"] += 1
                continue
            key = (DTYPES[r["dtype"]], uplo_of(r), int(r["n"]), int(r["batch"]))
            if "pass" in r:
                ident = (fname, key, arm, r["pass"])
                if ident in seen:
                    raise SystemExit(f"duplicate pass row: {ident}")
                seen.add(ident)
            cells[key][ARM_SPELLING[arm]].append(float(r["time_ms"]))
    if spec["current_only"]:
        for key in [k for k, arms in cells.items() if len(arms) < 2]:
            stats["dropped_cells_lt2_arms"] += 1
            del cells[key]
    return cells


def rank(times):
    """Section 6.3: within 3% of the best ties, ties in list order, the rest by time."""
    best = min(times.values())
    tied = [c for c in times if times[c] <= best * (1.0 + TIE)]
    rest = [c for c in times if c not in tied]
    tied.sort(key=CANDIDATE_ORDER.index)
    rest.sort(key=lambda c: (times[c], CANDIDATE_ORDER.index(c)))
    return tied + rest


def fmt_ms(t):
    """Four significant figures in plain decimal (no exponent), for any parser."""
    if t <= 0:
        raise SystemExit(f"non-positive time {t}")
    decimals = max(0, 3 - int(math.floor(math.log10(t))))
    return f"{t:.{decimals}f}"


def build_tables(date, stats_by_device):
    """Return {(dtype, device): (text, rows)} where rows maps key -> (ranked, noisy)."""
    out = {}
    for device, spec in SOURCES.items():
        stats = defaultdict(int)
        stats_by_device[device] = stats
        cells = collect(device, stats)
        by_dtype = defaultdict(dict)
        for (dtype, uplo, n, batch), arms in cells.items():
            times = {c: sum(v) / len(v) for c, v in arms.items()}
            noisy = any(max(v) / min(v) - 1.0 > NOISY for v in arms.values())
            by_dtype[dtype][(uplo, n, batch)] = (times, noisy)
        for dtype, rows in by_dtype.items():
            source = " + ".join(f"{ROUTING}/{f}" for f in spec["files"])
            lines = [
                f"# op=potrf dtype={dtype} device={device} batchlas={spec['batchlas']} "
                f"kernels=unknown date={date}",
                f"# source={source} ({spec['note']})",
                "# keys: uplo:exact n:log batch:log",
            ]
            for (uplo, n, batch) in sorted(rows):
                times, noisy = rows[(uplo, n, batch)]
                entries = " | ".join(f"{c} {fmt_ms(times[c])}" for c in rank(times))
                line = f"uplo={uplo} n={n} batch={batch} | {entries}"
                lines.append(line + ("   # noisy" if noisy else ""))
            out[(dtype, device)] = ("\n".join(lines) + "\n", rows)
    return out


def table_path(dtype, device):
    return os.path.join(REPO, TUNED, f"potrf.{dtype}.{device}.txt")


def parse_table(text):
    """Parse the section 5.4 format: header dict and [((uplo, n, batch), [(choice, ms)])]."""
    header, rows = {}, []
    for line in text.splitlines():
        if line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    header.setdefault(k, v)
            continue
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        parts = [p.strip() for p in body.split("|")]
        keys = dict(tok.split("=", 1) for tok in parts[0].split())
        ranked = []
        for p in parts[1:]:
            choice, ms = p.split()
            ranked.append((choice, float(ms)))
        rows.append(((keys["uplo"], int(keys["n"]), int(keys["batch"])), ranked))
    return header, rows


def nearest(rows, key):
    """Section 5.4 lookup: exact uplo, min sum |log2(row/key)|, ties to smaller n then batch."""
    uplo, n, batch = key
    pool = [r for r in rows if r[0][0] == uplo] or rows
    # Same arithmetic and 1e-9 tie window as Table::nearest (src/select/select.cc): with
    # log2(rn / n) the two sides of a geometric midpoint differ by an ulp and the tie is lost.
    ln, lb = math.log2(max(1, n)), math.log2(max(1, batch))
    best, best_d = None, 0.0
    for r in pool:
        _, rn, rb = r[0]
        d = abs(math.log2(rn) - ln) + abs(math.log2(rb) - lb)
        tie = best is not None and abs(d - best_d) <= 1e-9
        if best is None or (not tie and d < best_d) or (tie and (rn, rb) < best[0][1:]):
            best, best_d = r, d
    return best


def support_envelope(rows):
    """Largest measured n per (uplo, choice): a stand-in for can_run when reviewing guesses."""
    env = defaultdict(int)
    env[("L", "blocked")] = env[("L", "vendor")] = env[("U", "vendor")] = sys.maxsize
    for (uplo, n, _), ranked in rows:
        for c, _ in ranked:
            env[(uplo, c)] = max(env[(uplo, c)], n)
    return env


def read_points(path):
    pts = []
    with open(path) as f:
        for line in f:
            tok = line.split("#", 1)[0].split()
            if tok:
                pts.append((int(tok[0]), int(tok[1]) if len(tok) > 1 else None))
    return pts


def check(tables, points):
    failures = []
    on_disk = sorted(f for f in os.listdir(os.path.join(REPO, TUNED))
                     if f.startswith("potrf.") and f.endswith(".txt"))
    expected = sorted(os.path.basename(table_path(d, dev)) for d, dev in tables)
    if on_disk != expected:
        failures.append(f"tuned/ files {on_disk} != derived {expected}")
    for (dtype, device), (_, rows) in sorted(tables.items()):
        path = table_path(dtype, device)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            disk = f.read()
        date = parse_table(disk)[0].get("date", "")
        derived = build_tables(date, {})[(dtype, device)][0]
        if disk != derived:
            failures.append(f"{path}: differs from the sweeps (re-run without --check)")
        _, parsed = parse_table(disk)
        if len({k for k, _ in parsed}) != len(parsed):
            failures.append(f"{path}: duplicate row keys")
        for key, (times, _) in rows.items():
            hit = nearest(parsed, key)
            if hit is None or hit[0] != key:
                failures.append(f"{path}: nearest{key} -> {hit and hit[0]}")
                continue
            if hit[1][0][0] != rank(times)[0]:
                failures.append(f"{path}: {key} first {hit[1][0][0]} != winner {rank(times)[0]}")
        print(f"check {os.path.relpath(path, REPO)}: {len(rows)} measured cells looked up")
    print_offgrid(tables, points)
    return failures


def print_offgrid(tables, points):
    print("\noff-grid lookup (uplo=L): n batch -> row n/batch : first entry"
          " [first entry within the measured n envelope, if different]")
    for (dtype, device) in sorted(tables, key=lambda k: (k[1], k[0])):
        _, parsed = parse_table(tables[(dtype, device)][0])
        env = support_envelope(parsed)
        print(f"-- {dtype} {device}")
        for n, b in points:
            batches = [b] if b else DEFAULT_OFFGRID_BATCH
            cells = []
            for batch in batches:
                (_, rn, rb), ranked = nearest(parsed, ("L", n, batch))
                first = ranked[0][0]
                fit = next((c for c, _ in ranked if env[("L", c)] >= n), "none")
                extra = f" [{fit}]" if fit != first else ""
                cells.append(f"b{batch}->{rn}/{rb}:{first}{extra}")
            print(f"   n={n:<5} " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="diff tuned/ against the sweeps")
    ap.add_argument("--check-points", help="file of off-grid 'n [batch]' points for --check")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args()

    stats = {}
    tables = build_tables(args.date, stats)
    if args.check:
        points = (read_points(args.check_points) if args.check_points
                  else [(n, None) for n in DEFAULT_OFFGRID_N])
        failures = check(tables, points)
        for f in failures:
            print("FAIL:", f)
        print(f"\n--check: {'FAILED, ' + str(len(failures)) + ' problems' if failures else 'OK'}")
        return 1 if failures else 0

    os.makedirs(os.path.join(REPO, TUNED), exist_ok=True)
    for device, s in stats.items():
        print(f"{device}: " + ", ".join(f"{k}={v}" for k, v in sorted(s.items())))
    for (dtype, device), (text, rows) in sorted(tables.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        path = table_path(dtype, device)
        with open(path, "w") as f:
            f.write(text)
        noisy = sum(1 for _, nz in rows.values() if nz)
        upper = sum(1 for k in rows if k[0] == "U")
        print(f"wrote {os.path.relpath(path, REPO)}: {len(rows)} rows ({upper} Upper), {noisy} noisy")
    return 0


if __name__ == "__main__":
    sys.exit(main())

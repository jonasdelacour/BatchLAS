#!/usr/bin/env python3
"""Normalise archived RTX 4090 (sm_89) potrf timings into benchviz-style JSONL rows.

The sm_89 routing profile cannot be re-measured (no 4090 is available), so it is fitted
from what the repository already holds:

  * benchmarks/results/*.csv               factor_bench grids, Git LFS (run `git lfs pull`)
  * perf-evidence/vendor-independence      WP4 harness rows, read with `git show`
  * one distilled table (phase2/README.md) the only post-fix large-n blocked timings

Every row says HOW its route is known (`route_basis`), whether the route's own kernel is
still the shipped one (`kernel_current`), and why it is unusable when it is (`ok`/`reason`).
A pinned arm is resolved against the route table AS IT WAS when the grid ran: a pin that
supports() refused fell through to automatic(), so the arm name alone does not say what ran.

    python3 evaluation/routing/import_archive.py                  # write the JSONL
    python3 evaluation/routing/import_archive.py --report         # ...and print coverage
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = "perf-evidence/vendor-independence"
OUT = "benchmarks/results/routing/sm89_potrf_archive.jsonl"
DEVICE = "NVIDIA GeForce RTX 4090"
VENDOR_LIB = "cuSOLVER, CUDA 13.2"
RELSD_GATE = 0.10
TYPES = ("float", "double", "cfloat", "cdouble")

# Capacity facts per era. wp4 = whole-budget CTA ceiling (factor_baseline readback agrees);
# from P7 (c6b5463f) the advertised ceiling at 4 blocks/SM. evidence: docs/perf/potrf.md#what-ships
CTA_WP4 = dict(float=155, double=109, cfloat=109, cdouble=77)
CTA_P7 = dict(float=77, double=54, cfloat=54, cdouble=38)
TINY = dict(float=32, double=32, cfloat=32, cdouble=16)
LPANEL = dict(float=744, double=368, cfloat=368, cdouble=180)

ERAS = {
    "wp4":    dict(cta=CTA_WP4, tiny=None, lpanel=None),
    "p7":     dict(cta=CTA_P7, tiny=TINY, lpanel=None),
    "p3":     dict(cta=CTA_P7, tiny=TINY, lpanel=LPANEL),
    "p9post": dict(cta=CTA_P7, tiny=TINY, lpanel=LPANEL),
    "p10":    dict(cta=CTA_P7, tiny=TINY, lpanel=LPANEL),
}

KERNEL_NOTES = {
    "lpanel": "LPanel loads rewritten after this grid (eb916865 vector sB reads, 43a15df7 "
              "local-space stores; +<=1.1x cfloat on sm_120, unmeasured on sm_89)",
    "blocked": "sub-op drift: trailing gemm (df8ff2b8, 12a8ef8b, 7ff1e3a3) and panel trsm "
               "(44c1b49a) changed after this grid",
    "pre_p7": "pre-P7 kernel: CTA G-packing and blocked nb clamp changed in c6b5463f",
}


def supports(era, algo, t, n, uplo):
    caps = ERAS[era]
    if algo == "tiny":
        return caps["tiny"] is not None and n <= caps["tiny"][t]
    if algo == "cta":
        return n <= caps["cta"][t]
    if algo == "lpanel":
        return uplo == "L" and caps["lpanel"] is not None and n <= caps["lpanel"][t]
    if algo == "blocked":
        return uplo == "L"
    return False


def tiny_window(era, t, n, uplo):
    if era == "p9post":  # a6f6294b: Lower, fp32 only
        return uplo == "L" and t in ("float", "cfloat") and n <= TINY[t]
    if era == "p10":     # ce6909fc: every type on Upper, two bands for fp64 Lower
        if n > TINY[t] or n > 32:
            return False
        if uplo == "U" or t in ("float", "cfloat"):
            return True
        return 2 <= n <= 8 or 12 <= n <= 16
    return False


def automatic(era, t, n, uplo):
    """What Auto ran in a vendor-present build of that era."""
    if era in ("wp4", "p7"):
        return "vendor:auto"
    if tiny_window(era, t, n, uplo):
        return "native:tiny"
    if uplo == "L" and t in ("float", "cfloat") and 32 < n <= 256:
        cta_last = 35 if t == "float" else 32
        return "native:cta" if n <= cta_last else "native:lpanel"
    return "vendor:auto"


def resolve(era, pin, t, n, uplo):
    """(route, basis) for an env pin. A refused pin falls through to automatic()."""
    if pin == "vendor":
        return "vendor:auto", "pin"
    if pin == "auto":
        return automatic(era, t, n, uplo), "era_predicate"
    if pin == "native":  # bare origin; both eras that used it have an all-false preferred()
        assert era in ("wp4", "p7"), era
        algo = "cta" if supports(era, "cta", t, n, uplo) else "blocked"
        return f"native:{algo}", "bare_native_walk"
    if supports(era, pin, t, n, uplo):
        return f"native:{pin}", "pin"
    return automatic(era, t, n, uplo), "pin_fallthrough"


def kernel_state(route, era):
    algo = route.split(":")[1]
    if route.startswith("vendor"):
        return True, None
    if era == "wp4":
        return False, KERNEL_NOTES["pre_p7"]
    if algo == "lpanel":
        return False, KERNEL_NOTES["lpanel"]
    if algo == "blocked":
        return True, KERNEL_NOTES["blocked"]
    return True, None


def arm_of(pin, route):
    if pin == "auto":
        return "auto"
    return "vendor" if route.startswith("vendor") else f"route:{route}"


def row(**kw):
    base = dict(op="potrf", nrhs=0, cc=89, device=DEVICE, vendor_lib=VENDOR_LIB, stat="median",
                reps=None, residual=None, notes="")
    base.update(kw)
    base["m"] = base["n"]
    return base


# --------------------------------------------------------------- factor_bench grids
# (file, commit that archived it, era, uplo rule, guard, extra note)
FACTOR_FILES = [
    ("factor_smoke.csv", "02939e2f", "wp4", "L", "real", ""),
    ("factor_baseline_potrf_float.csv", "ea515cc9", "wp4", "L", "real", ""),
    ("factor_baseline_potrf_double.csv", "ea515cc9", "wp4", "L", "real", ""),
    ("factor_baseline_potrf_cfloat.csv", "ea515cc9", "wp4", "L", "real", ""),
    ("factor_baseline_potrf_cdouble.csv", "ea515cc9", "wp4", "L", "real", ""),
    ("p7_occupancy_potrf.csv", "c6b5463f", "p7", "L", "real", ""),
    ("p7_occupancy_potrf_band.csv", "c6b5463f", "p7", "L", "real", ""),
    ("p3_lpanel_ab.csv", "1d352e54", "p3", "L", "real", ""),
    ("p3_lpanel_edges.csv", "1d352e54", "p3", "L", "real", ""),
    ("p3_auto_after_flip.csv", "1d352e54", "p3", "L", "real", ""),
    ("p6_e2e_before.csv", "1d352e54", "p3", "L", "real", "gemm routes as shipped at 1d352e54"),
    ("p6_e2e_with_refused_window.csv", "1d352e54", "p3", "L", "real",
     "SUB-ROUTE NOT SHIPPED: trailing gemm under P6's withdrawn window"),
    # The p9 auto arm of the tiny grid is the BEFORE picture of a6f6294b's flip (its message).
    ("p9_potrf_tiny.csv", "a6f6294b", "p3", "L", "unrecorded", "auto = pre-flip"),
    ("p9_potrf_low.csv", "a6f6294b", "p3", "L", "unrecorded", ""),
    ("p9_potrf_band.csv", "a6f6294b", "p3", "L", "unrecorded", ""),
    ("p9_e2e.csv", "a6f6294b", "p9post", "L", "unrecorded", "auto = post-flip"),
    ("p10_potrf_tiny_fp64.csv", "ce6909fc", "p10", "L", "stand-in",
     "NVML down: /proc-fd stand-in guard, blind to other users' jobs (2b045ab4)"),
    ("p10_potrf_tiny_upper.csv", "ce6909fc", "p10", "U", "stand-in",
     "NVML down: /proc-fd stand-in guard, blind to other users' jobs (2b045ab4)"),
    # No uplo column: lines 2-13 are the Upper spot-checks 2b045ab4 quotes (float 16 10.786x,
    # double 16 2.335x, cdouble 10 1.388x); the later potrf lines are the fp64 Lower re-cut.
    ("p10_recut_real_guard.csv", "2b045ab4", "p10", lambda line: "U" if line <= 13 else "L",
     "real", ""),
]


def import_factor_file(name, commit, era, uplo_rule, guard, extra):
    path = os.path.join(REPO, "benchmarks/results", name)
    with open(path) as fh:
        text = fh.read()
    if text.startswith("version https://git-lfs"):
        sys.exit(f"{path} is an LFS pointer; run `git lfs pull` first")
    out = []
    for line, r in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        if r["op"] != "potrf":
            continue
        t, n, batch = r["type"], int(r["n"]), int(r["batch"])
        uplo = uplo_rule(line) if callable(uplo_rule) else uplo_rule
        pin = r["pin"]
        route, basis = resolve(era, pin, t, n, uplo)
        notes = [extra] if extra else []
        readback = r.get("resolved_route")
        if readback:
            if readback != route:
                notes.append(f"readback {readback} disagrees with era model {route}")
            route, basis = readback, "readback"
        if basis == "pin_fallthrough":
            notes.append(f"pin '{pin}' refused by supports(); ran {route}")
        current, knote = kernel_state(route, era)
        if knote:
            notes.append(knote)
        ok, reason = True, "ok"
        rel_sd = float(r["rel_sd"])
        if r["info_nonzero"] != "0":
            ok, reason = False, f"wrong answer (info_nonzero={r['info_nonzero']})"
        elif r["bad"] != "0":
            ok, reason = False, f"harness gate: {r['reason']}"
        elif r["pin_parsed"] != "1":
            ok, reason = False, "pin not parsed"
        elif rel_sd > RELSD_GATE:
            ok, reason = False, f"rel_sd {rel_sd:.3f} > {RELSD_GATE}"
        elif extra.startswith("SUB-ROUTE NOT SHIPPED") and route == "native:blocked":
            ok, reason = False, "blocked under a gemm sub-route that never shipped"
        out.append(row(dtype=t, n=n, batch=batch, uplo=uplo, arm=arm_of(pin, route), route=route,
                       route_basis=basis, time_ms=float(r["median_ms"]), rel_sd=rel_sd,
                       reps=int(r["reps"]), residual=float(r["residual"]), ok=ok, reason=reason,
                       kernel_current=current, era=era, guard=guard, commit=commit,
                       source=f"benchmarks/results/{name}#L{line}", notes="; ".join(notes)))
    return out


# ----------------------------------------------------- WP4 phase2_bench (direct calls)
# Direct calls into potrf_vendor / potrf_cta_dispatch / potrf_blocked_dispatch, so the
# route is known by construction. Its blocked rows ran the racing V1 trsm at float W=32,
# which phase2/README.md section 5 supersedes; they are kept, marked unusable.
P2B = "experiments/wp4_potrf/phase2_bench/"
P2B_FILES = ["main.csv", "recheck.csv", "wins.csv", "allvendor.csv", "batchdep.csv",
             "novendor.csv"]
SUPERSEDED = ("superseded: blocked driver on the racing V1 trsm at float W=32 "
              "(tag phase2/README.md sections 2, 4, 5)")


def git_show(path):
    return subprocess.run(["git", "-C", REPO, "show", f"{TAG}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def tag_commit():
    return subprocess.run(["git", "-C", REPO, "rev-list", "-n1", TAG], check=True,
                          capture_output=True, text=True).stdout.strip()[:8]


def import_phase2_bench(commit):
    out = []
    for name in P2B_FILES:
        for line, r in enumerate(csv.DictReader(io.StringIO(git_show(P2B + name))), start=2):
            variant = r["variant"]
            route = {"vendor": "vendor:auto", "cta": "native:cta", "blocked": "native:blocked",
                     "facade": "native:blocked"}[variant]
            cfg = r.get("cfg", "facade" if variant == "facade" else "")
            rel_sd = float(r["rel_sd"])
            ok, reason = True, "ok"
            if variant in ("blocked", "facade"):
                ok, reason = False, SUPERSEDED
            elif r["info_nonzero"] != "0" or r["nonfinite"] != "0":
                ok, reason = False, f"wrong answer (info_nonzero={r['info_nonzero']})"
            elif rel_sd > RELSD_GATE:
                ok, reason = False, f"rel_sd {rel_sd:.3f} > {RELSD_GATE}"
            current, knote = kernel_state(route, "wp4")
            notes = [f"cfg={cfg}" if cfg else "", f"pass={r['pass']}" if "pass" in r else "",
                     "build-novendor" if name == "novendor.csv" else "", knote or "",
                     "guard: CUDA_VISIBLE_DEVICES=1 + nvidia-smi spot checks only"]
            out.append(row(dtype=r["type"], n=int(r["n"]), batch=int(r["batch"]), uplo="L",
                           arm=arm_of("x", route), route=route, route_basis="direct_call",
                           time_ms=float(r["med_ms"]), rel_sd=rel_sd,
                           residual=float(r["residual"]), ok=ok, reason=reason,
                           kernel_current=current, era="wp4", guard="cvd_only", commit=commit,
                           source=f"{TAG}:{P2B}{name}#L{line}",
                           notes="; ".join(x for x in notes if x)))
    return out


# --------------------------------- phase2/README.md section 5: distilled, but post-fix
P2_README = "experiments/wp4_potrf/phase2/README.md"
P2_ROW = re.compile(r"^\| (float|double|cfloat|cdouble) \| (\d+) \| (\d+) \| ([\d.]+) \| "
                    r"([\d.]+) \| ([\d.]+) \|")


def import_phase2_table(commit):
    out = []
    lines = git_show(P2_README).splitlines()
    start = next(i for i, s in enumerate(lines) if s.startswith("## 5. The performance picture"))
    end = next(i for i, s in enumerate(lines) if i > start and s.startswith("### What to take"))
    for i in range(start, end):
        m = P2_ROW.match(lines[i])
        if not m:
            continue
        t, n, batch = m.group(1), int(m.group(2)), int(m.group(3))
        arms = (("vendor:auto", m.group(4), "BATCHLAS_POTRF_ROUTE=vendor"),
                ("native:blocked", m.group(5), "sub-ops nn: gemm+trsm forced native "
                                               "(= vendor-free build)"),
                ("native:blocked", m.group(6), "sub-ops routed: gemm+trsm on their own routes"))
        for route, ms, cfg in arms:
            current, knote = kernel_state(route, "wp4")
            out.append(row(dtype=t, n=n, batch=batch, uplo="L", arm=arm_of("x", route),
                           route=route, route_basis="distilled", time_ms=float(ms), rel_sd=None,
                           ok=True, reason="ok (distilled table: no raw CSV; worst rel_sd 5.3%)",
                           kernel_current=current, era="wp4", guard="cvd_only", commit=commit,
                           source=f"{TAG}:{P2_README}#L{i + 1}",
                           notes="; ".join(x for x in (cfg, knote) if x)))
    return out


# ------------------------------------------------------------------- coverage report
NBANDS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512), (513, 4096))
BBANDS = ((1, 255), (256, 1023), (1024, 4095), (4096, 16383), (16384, 1 << 30))
ROUTES = ("native:tiny", "native:cta", "native:lpanel", "native:blocked", "vendor:auto")


def band(v, bands):
    return next(f"{lo}-{hi}" if hi < (1 << 30) else f">={lo}" for lo, hi in bands if lo <= v <= hi)


def report(rows):
    usable = [r for r in rows if r["ok"]]
    print(f"rows {len(rows)}, ok {len(usable)}, "
          f"ok+kernel_current {sum(r['kernel_current'] for r in usable)}")
    for reason in sorted({r["reason"] for r in rows if not r["ok"]}):
        print(f"  rejected {sum(r['reason'] == reason for r in rows):4d}  {reason}")
    for current_only in (False, True):
        print(f"\n== ok rows{' with kernel_current' if current_only else ''}: "
              "dtype uplo n-band | batch-band: tiny cta lpanel blocked vendor")
        cells = defaultdict(lambda: defaultdict(int))
        for r in usable:
            if current_only and not r["kernel_current"]:
                continue
            key = (r["dtype"], r["uplo"], band(r["n"], NBANDS), band(r["batch"], BBANDS))
            cells[key][r["route"]] += 1
        for key in sorted(cells, key=lambda k: (TYPES.index(k[0]), k[1], int(k[2].split("-")[0]),
                                               int(k[3].lstrip(">=").split("-")[0]))):
            print("  %-7s %s %-8s | %-11s: " % key +
                  " ".join("%3d" % cells[key][rt] for rt in ROUTES))
    for current_only in (False, True):
        print("\n== same-shape cells (dtype, uplo, n, batch) timed on >=2 routes, ok rows" +
              (", kernel_current only" if current_only else ""))
        shapes = defaultdict(set)
        for r in usable:
            if not current_only or r["kernel_current"]:
                shapes[(r["dtype"], r["uplo"], r["n"], r["batch"])].add(r["route"])
        for t in TYPES:
            for uplo in ("L", "U"):
                multi = {k: v for k, v in shapes.items()
                         if k[0] == t and k[1] == uplo and len(v) > 1}
                if not multi:
                    continue
                combo = defaultdict(lambda: [0, set()])
                for k, v in multi.items():
                    name = "+".join(x.split(":")[1] if x[0] == "n" else "vendor" for x in sorted(v))
                    combo[name][0] += 1
                    combo[name][1].add(k[2])
                print(f"  {t:7s} {uplo}: {len(multi):3d} shapes")
                for name, (c, ns) in sorted(combo.items()):
                    print(f"      {c:3d} x {name:30s} n={min(ns)}..{max(ns)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, OUT))
    ap.add_argument("--report", action="store_true", help="print the coverage report")
    a = ap.parse_args()
    rows = []
    for spec in FACTOR_FILES:
        rows += import_factor_file(*spec)
    commit = tag_commit()
    rows += import_phase2_bench(commit)
    rows += import_phase2_table(commit)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {os.path.relpath(a.out, REPO)}", file=sys.stderr)
    if a.report:
        report(rows)


if __name__ == "__main__":
    main()

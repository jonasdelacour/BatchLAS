#!/usr/bin/env python3
"""Fit the routing cost model's constants from benchviz-schema timings, and score it.

    python3 evaluation/routing/fit.py --results R.jsonl [R2.jsonl ...] --profile sm_89 \
        --plan-dump build/tests/potrf_plan_dump --out OUT/sm_89 \
        [--borrow sm_120=OUT/sm_120/profile.json] [--compare-windows] [--grid] [--include-stale]

Device facts come from profiles/device_facts.json. A (route, dtype, uplo) key is fitted only
from >= --min-rows rows over >= --min-batches batch sizes; otherwise it borrows (see
`fallback`). Held-out regret is leave-one-n-out within (dtype, uplo), and the ship gate
(`ship_gate`) is printed as PASS/FAIL.

Model (docs/design/routing-cost-model.md section 3, src/util/launch_plan.hh):

    t = t_launch*T0 + max(s_per_flop*T1, s_per_byte*T2) + t_step*T3

T0..T3 are the plan's cost terms, printed by potrf_plan_dump from the C++ launch plan --
waves, occupancy and per-group shares are never recomputed here. `combine` below is the one
line of arithmetic mirrored from launch_plan::combine; potrf_plan_tests and test_fit.py pin
the two to the same value.

Objective: least squares on log(time), with the constants parameterised as exp(theta) so
they stay non-negative. Cells span five decades (0.01 ms to 100 ms); an absolute-error fit
is decided by the slowest few cells, while routing compares RATIOS at every scale, which is
what a log residual weighs equally.

Outputs, under --out: profile.json (constants, measured support regions, fallbacks,
provenance) and report.md (held-out regret, extrapolated regions, route diff).
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROUTES = ("native:tiny", "native:cta", "native:lpanel", "native:blocked", "vendor")
DTYPES = ("float", "double", "cfloat", "cdouble")
PARAMS = ("t_launch", "s_per_flop", "s_per_byte", "t_step")

# Device facts per profile live in ONE file; see its _doc for where each number comes from.
FACTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", "device_facts.json")
FACT_KEYS = ("local_mem", "max_wg", "cus", "max_threads_per_cu", "max_groups_per_cu",
             "fp64_rate", "fp32_tflops", "fp64_tflops", "dram_gbps")


def load_facts(profile, path=FACTS_FILE):
    with open(path) as f:
        return dict(json.load(f)[profile])

PRECISION_PARTNER = {"float": "double", "double": "float", "cfloat": "cdouble", "cdouble": "cfloat"}
COMPLEXITY_PARTNER = {"float": "cfloat", "cfloat": "float", "double": "cdouble", "cdouble": "double"}
IS_FP64 = {"float": False, "cfloat": False, "double": True, "cdouble": True}


def combine(terms, c):
    """launch_plan::combine. terms = (T0..T3), c = (t_launch, s_per_flop, s_per_byte, t_step)."""
    return c[0] * terms[0] + max(c[1] * terms[1], c[2] * terms[2]) + c[3] * terms[3]


# ---- data -------------------------------------------------------------------------------

def norm_route(row):
    r = row.get("route") or ""
    arm = row.get("arm") or ""
    if not r and arm.startswith("route:"):
        r = arm[len("route:"):]
    if r.startswith("vendor"):
        return "vendor"
    return r if r in ROUTES else None


def load_rows(paths, op, include_stale):
    """Measured cells: {(dtype, uplo, n, batch, route): {"times": [...], "sources": set}}."""
    cells = defaultdict(lambda: {"times": [], "sources": set()})
    skipped = defaultdict(int)
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("op") != op:
                    continue
                if not row.get("ok", False) or not row.get("time_ms"):
                    skipped["not ok"] += 1
                    continue
                if not include_stale and row.get("kernel_current") is False:
                    skipped["kernel_current=false"] += 1
                    continue
                route = norm_route(row)
                dtype = row.get("dtype")
                if route is None or dtype not in DTYPES:
                    skipped["unknown route/dtype"] += 1
                    continue
                n = int(row.get("n") or row.get("m"))
                key = (dtype, row.get("uplo", "L") or "L", n, int(row["batch"]), route)
                cells[key]["times"].append(float(row["time_ms"]) * 1e-3)
                src = row.get("era") or row.get("source") or os.path.basename(path)
                cells[key]["sources"].add(str(src))
    out = {k: {"t": statistics.median(v["times"]), "reps": len(v["times"]),
               "sources": sorted(v["sources"])} for k, v in cells.items()}
    return out, dict(skipped)


def plan_features(plan_dump, preset, shapes):
    """{(dtype, uplo, n, batch): tool JSON} for every shape, from the C++ launch plans."""
    shapes = sorted(set(shapes))
    cmd = [plan_dump, "--local-mem", str(preset["local_mem"]), "--max-wg", str(preset["max_wg"]),
           "--cus", str(preset["cus"]), "--max-threads-per-cu", str(preset["max_threads_per_cu"]),
           "--max-groups-per-cu", str(preset["max_groups_per_cu"])]
    stdin = "".join(f"{d} {n} {b} {u}\n" for (d, u, n, b) in shapes)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/dpcpp-cuda/lib:" + env.get("LD_LIBRARY_PATH", "")
    res = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env, check=True)
    feats = {}
    for line in res.stdout.splitlines():
        j = json.loads(line)
        feats[(j["dtype"], j["uplo"], j["n"], j["batch"])] = j
    missing = [s for s in shapes if s not in feats]
    if missing:
        raise RuntimeError(f"potrf_plan_dump returned no plan for {missing[:5]}")
    return feats


# ---- the fit ----------------------------------------------------------------------------

def _solve(A, b):
    """Gaussian elimination with partial pivoting; A is small and square."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def _loss(theta, data):
    c = [math.exp(t) for t in theta]
    s = 0.0
    for terms, t in data:
        m = combine(terms, c)
        if m <= 0:
            return float("inf")
        s += (math.log(m) - math.log(t)) ** 2
    return s


def _lm(theta, data, lo, hi, iters=80):
    mu = 1e-3
    loss = _loss(theta, data)
    for _ in range(iters):
        c = [math.exp(t) for t in theta]
        JTJ = [[0.0] * 4 for _ in range(4)]
        JTr = [0.0] * 4
        for terms, t in data:
            tf, tb = c[1] * terms[1], c[2] * terms[2]
            m = c[0] * terms[0] + max(tf, tb) + c[3] * terms[3]
            r = math.log(m) - math.log(t)
            g = [c[0] * terms[0] / m, (tf / m) if tf >= tb else 0.0,
                 (tb / m) if tb > tf else 0.0, c[3] * terms[3] / m]
            for i in range(4):
                JTr[i] += g[i] * r
                for k in range(4):
                    JTJ[i][k] += g[i] * g[k]
        improved = False
        while mu < 1e12:
            A = [[JTJ[i][k] + (mu * (JTJ[i][i] + 1e-12) if i == k else 0.0) for k in range(4)]
                 for i in range(4)]
            step = _solve(A, [-v for v in JTr])
            if step is None:
                mu *= 10
                continue
            cand = [min(hi[i], max(lo[i], theta[i] + step[i])) for i in range(4)]
            cl = _loss(cand, data)
            if cl < loss:
                done = loss - cl < 1e-12 * max(1.0, loss)
                theta, loss, mu, improved = cand, cl, max(mu / 10, 1e-9), True
                break
            mu *= 10
        if not improved or done:
            break
    return theta, loss


def fit_constants(data):
    """data: [(terms, seconds)]. Returns (constants, stats). Inactive terms are set to 0."""
    scale = []
    for j in range(4):
        ratios = [t / terms[j] for terms, t in data if terms[j] > 0]
        scale.append(statistics.median(ratios) if ratios else 1.0)
    lo = [math.log(s) - 40 for s in scale]
    hi = [math.log(s) + 3 for s in scale]
    best = None
    levels = (0.5, 0.03, 0.002)
    for a in levels:
        for b in levels:
            for c in levels:
                for d in levels:
                    th0 = [math.log(scale[0] * a), math.log(scale[1] * b),
                           math.log(scale[2] * c), math.log(scale[3] * d)]
                    th, ls = _lm(th0, data, lo, hi)
                    if best is None or ls < best[1]:
                        best = (th, ls)
    const = [math.exp(t) for t in best[0]]
    base = best[1]
    inactive = []
    # Parsimony: a term the data cannot see is 0, not whatever value the optimiser left.
    for j in range(4):
        trial = const[:]
        trial[j] = 0.0
        if all(combine(terms, trial) > 0 for terms, _ in data):
            ls = sum((math.log(combine(terms, trial)) - math.log(t)) ** 2 for terms, t in data)
            if ls <= base * 1.001 + 1e-9:
                const, base = trial, ls
                inactive.append(PARAMS[j])
    resid = [math.log(combine(terms, const)) - math.log(t) for terms, t in data]
    stats = {"rows": len(data), "rms_log": math.sqrt(sum(r * r for r in resid) / len(resid)),
             "max_abs_log": max(abs(r) for r in resid), "inactive": inactive}
    return const, stats


def training_set(cells, feats, keep):
    """{(route, dtype, uplo): [(terms, t, n, batch, sources)]}, rows whose plan fits."""
    out = defaultdict(list)
    dropped = defaultdict(int)
    for (dtype, uplo, n, batch, route), v in cells.items():
        if not keep(dtype, uplo, n, batch):
            continue
        plan = feats[(dtype, uplo, n, batch)]["routes"][route]
        if not plan["fits"] or not plan["supported"]:
            dropped[route] += 1   # measured under an older capacity, or refused today
            continue
        out[(route, dtype, uplo)].append((plan["terms"], v["t"], n, batch, v["sources"]))
    return out, dict(dropped)


def fit_key(rows):
    const, stats = fit_constants([(r[0], r[1]) for r in rows])
    ns = [r[2] for r in rows]
    bs = [r[3] for r in rows]
    return {"constants": const, "fit": stats, "fallback": None,
            "support": {"n_min": min(ns), "n_max": max(ns), "batch_min": min(bs),
                        "batch_max": max(bs), "rows": len(rows),
                        "sources": sorted({s for r in rows for s in r[4]})}}


def eligible(rows, cfg):
    """The minimum-data rule: >= min_rows rows over >= min_batches distinct batch sizes."""
    return len(rows) >= cfg["min_rows"] and len({r[3] for r in rows}) >= cfg["min_batches"]


def fit_measured(train, cfg):
    return {key: fit_key(rows) for key, rows in train.items() if eligible(rows, cfg)}


def complete(measured, cfg, unfittable=None, train=None):
    """Every (route, dtype, uplo) without a fit of its own gets the documented fallback;
    keys the fallback cannot price are recorded in `unfittable` and left out."""
    model = dict(measured)
    for route in ROUTES:
        for dtype in DTYPES:
            for uplo in ("L", "U"):
                if (route, dtype, uplo) not in model:
                    fb = fallback(measured, route, dtype, uplo, cfg,
                                  (train or {}).get((route, dtype, uplo), []))
                    if fb and "unfittable" in fb:
                        if unfittable is not None:
                            unfittable[(route, dtype, uplo)] = fb
                    elif fb:
                        model[(route, dtype, uplo)] = fb
    return model


def fit_all(train, cfg):
    return complete(fit_measured(train, cfg), cfg, None, train)


def fallback(measured, route, dtype, uplo, cfg, own_rows=()):
    """The rule for a (route, dtype, uplo) without a fit of its own. Sources, in this order:
    1. the same route and dtype on the other uplo, copied;
    2. the precision partner (float<->double, cfloat<->cdouble), s_per_flop scaled by the
       profile's FP64:FP32 PEAK rate, the rest copied (latency-bound kernels violate this:
       sm_89 tiny double measures ~2x float, not 64x);
    3. the complexity partner (float<->cfloat, double<->cdouble), copied -- the plan already
       counts a complex multiply-add as four;
    4. the same (route, dtype, uplo) MEASURED in another profile (--borrow), s_per_flop scaled
       by the per-CU peak FP rate of that precision (source/target) and s_per_byte by the
       per-CU DRAM bandwidth (source/target); t_launch and t_step copied.
    PER TERM: each constant comes from the first source whose fit SAW that term (it is not in
    the source's `inactive` list); an inactive 0 is never borrowed. A term no source saw leaves
    the key unfittable, so the route is not a candidate for it. Only measured entries are
    sources, never another fallback. Returns the entry, or {"unfittable": [terms]}.
    5. LAST, the key's OWN THIN FIT: its own rows (>= 3) fitted although below the minimum-data
       rule, flagged `thin`. It supplies every term the sources above did not -- including a
       term it saw as 0 -- so a key's own data is never discarded in favour of dropping it."""
    other = "U" if uplo == "L" else "L"
    cands = []
    src = measured.get((route, dtype, other))
    if src:
        cands.append((src, "other_uplo", (route, dtype, other), 1.0, 1.0))
    p = PRECISION_PARTNER[dtype]
    rate = cfg["facts"]["fp64_rate"]
    for u in (uplo, other):
        src = measured.get((route, p, u))
        if src:
            f = (1 / rate) if IS_FP64[dtype] else rate
            cands.append((src, "precision_scaled", (route, p, u), f, 1.0))
            break
    q = COMPLEXITY_PARTNER[dtype]
    for u in (uplo, other):
        src = measured.get((route, q, u))
        if src:
            cands.append((src, "complexity_borrow", (route, q, u), 1.0, 1.0))
            break
    tgt = cfg["facts"]
    for name, prof, sf in cfg["borrows"]:
        e = prof["routes"].get(route, {}).get(dtype, {}).get(uplo)
        if not e or e.get("fallback"):
            continue
        peak = "fp64_tflops" if IS_FP64[dtype] else "fp32_tflops"
        ff = (sf[peak] / sf["cus"]) / (tgt[peak] / tgt["cus"])
        bf = (sf["dram_gbps"] / sf["cus"]) / (tgt["dram_gbps"] / tgt["cus"])
        src = {"constants": [e["constants"][k] for k in PARAMS], "fit": e.get("fit")}
        cands.append((src, "other_profile", (name, route, dtype, uplo), ff, bf))
    thin = None
    if len(own_rows) >= 3:
        thin = fit_key(list(own_rows))
        thin_src = {"constants": thin["constants"], "fit": {"inactive": []}}
        cands.append((thin_src, "thin", (route, dtype, uplo), 1.0, 1.0))
    if not cands:
        return None
    const, terms, missing = [0.0] * 4, {}, []
    for j, name in enumerate(PARAMS):
        for src, rule, frm, ff, bf in cands:
            if name in ((src.get("fit") or {}).get("inactive") or []):
                continue
            const[j] = src["constants"][j] * (ff if j == 1 else bf if j == 2 else 1.0)
            terms[name] = {"rule": rule, "from": "/".join(frm),
                           "factor": ff if j == 1 else bf if j == 2 else 1.0}
            break
        else:
            missing.append(name)
    if missing:
        return {"unfittable": missing, "tried": ["/".join(c[2]) for c in cands]}
    first = terms[PARAMS[0]]
    used_thin = any(v["rule"] == "thin" for v in terms.values())
    return {"constants": const, "fit": thin["fit"] if used_thin else None,
            "support": thin["support"] if used_thin else None, "thin": used_thin,
            "fallback": {"rule": first["rule"], "from": first["from"],
                         "flop_factor": terms["s_per_flop"]["factor"],
                         "byte_factor": terms["s_per_byte"]["factor"], "terms": terms}}


# ---- choice and regret ------------------------------------------------------------------

def extrapolated(entry, n, batch):
    s = entry.get("support")
    if not s:
        return True
    return not (s["n_min"] <= n <= s["n_max"] and s["batch_min"] <= batch <= s["batch_max"])


def choose(model, feat, margin, vendor=True):
    """argmin cost over supported routes with constants; native must beat vendor by margin."""
    dtype, uplo = feat["dtype"], feat["uplo"]
    costs = {}
    for route in ROUTES:
        if route == "vendor" and not vendor:
            continue
        plan = feat["routes"][route]
        entry = model.get((route, dtype, uplo))
        if not plan["supported"] or not plan["fits"] or entry is None:
            continue
        costs[route] = combine(plan["terms"], entry["constants"])
    native = {r: c for r, c in costs.items() if r != "vendor"}
    if not native:
        return ("vendor" if "vendor" in costs else None), costs
    best = min(native, key=native.get)
    if "vendor" in costs and not native[best] < (1 - margin) * costs["vendor"]:
        return "vendor", costs
    return best, costs


def group_cells(cells, feats):
    by_cell = defaultdict(dict)
    for (dtype, uplo, n, batch, route), v in cells.items():
        plan = feats[(dtype, uplo, n, batch)]["routes"][route]
        if plan["supported"] and plan["fits"]:
            by_cell[(dtype, uplo, n, batch)][route] = v["t"]
    return by_cell


def regret_of(pick, measured):
    if pick is None or pick not in measured:
        return None
    return measured[pick] / min(measured.values())


def summarise(regrets):
    if not regrets:
        return {"cells": 0}
    rs = sorted(regrets)
    gm = math.exp(sum(math.log(r) for r in rs) / len(rs))
    p95 = rs[min(len(rs) - 1, int(math.ceil(0.95 * len(rs))) - 1)]
    return {"cells": len(rs), "geomean": gm, "p95": p95, "max": rs[-1],
            "over_1.1": sum(1 for r in rs if r > 1.1)}


def cross_validate(cells, feats, train_full, measured_full, cfg, margin):
    """Leave-one-n-out within each (dtype, uplo): every cell is scored by a model that never
    saw its order. Only that (dtype, uplo)'s keys hold rows at that order, so only they are
    refitted; the rest of the full fit is reused unchanged."""
    by_cell = group_cells(cells, feats)
    folds = sorted({(c[0], c[1], c[2]) for c in by_cell})
    scored = []
    for (dtype, uplo, n) in folds:
        measured = {k: v for k, v in measured_full.items() if not (k[1] == dtype and k[2] == uplo)}
        for key, rows in train_full.items():
            if key[1] != dtype or key[2] != uplo:
                continue
            kept = [r for r in rows if r[2] != n]
            if eligible(kept, cfg):
                measured[key] = fit_key(kept)
        train_fold = {k: ([r for r in v if r[2] != n] if (k[1], k[2]) == (dtype, uplo) else v)
                      for k, v in train_full.items()}
        model = complete(measured, cfg, None, train_fold)
        for cell, meas in by_cell.items():
            if cell[:3] != (dtype, uplo, n):
                continue
            pick, costs = choose(model, feats[cell], margin)
            entry = model.get((pick, cell[0], cell[1])) if pick else None
            scored.append({"cell": cell, "measured": meas, "pick": pick,
                           "predicted": costs, "regret": regret_of(pick, meas),
                           "extrapolated": entry is None or extrapolated(entry, cell[2], cell[3])})
    return scored


# ---- report -----------------------------------------------------------------------------

def fmt_cell(c):
    return f"{c[0]} {c[1]} n={c[2]} b={c[3]}"


def ship_gate(cv, full_by_cell, feats, max_regret=1.25):
    """PASS only if, on the cells where BOTH today's pick and the held-out model's pick were
    measured, the model beats today's windows on geomean AND p95, and its max <= max_regret."""
    held = {s["cell"]: s for s in cv}
    pt, pm, cells = [], [], []
    for cell, measured in full_by_cell.items():
        s = held.get(cell)
        rt = regret_of(feats[cell]["auto"], measured)
        if len(measured) >= 2 and s and s["regret"] is not None and rt is not None:
            pt.append(rt)
            pm.append(s["regret"])
            cells.append(cell)
    t, m = summarise(pt), summarise(pm)
    ok = bool(t.get("cells")) and m["geomean"] < t["geomean"] and m["p95"] < t["p95"] \
        and m["max"] <= max_regret
    return {"verdict": "PASS" if ok else "FAIL", "paired_cells": len(cells), "today": t,
            "model": m, "rule": ship_gate.__doc__.strip()}


def write_report(path, args, cv, full_by_cell, feats, full_model, skipped, dropped,
                 grid_rows, train, gate, unfittable):
    L = []
    L.append(f"# potrf cost-model fit: profile `{args.profile}`\n")
    L.append(f"Sources: {', '.join(args.results)}. Rows skipped: {skipped or 'none'}. "
             f"Measured rows whose plan does not fit or is refused today (dropped): "
             f"{dropped or 'none'}.\n")
    t, m = gate["today"], gate["model"]
    L.append(f"## Ship gate: **{gate['verdict']}**\n")
    L.append(gate["rule"] + "\n")
    if t.get("cells"):
        L.append(f"* {gate['paired_cells']} paired cells. Today: geomean {t['geomean']:.4f}, "
                 f"p95 {t['p95']:.4f}, max {t['max']:.4f}. Model (held-out): geomean "
                 f"{m['geomean']:.4f}, p95 {m['p95']:.4f}, max {m['max']:.4f}.\n")
    else:
        L.append("* no paired cells: nothing to compare, so the gate fails.\n")
    L.append("## Fitted vs borrowed keys\n")
    L.append(f"Minimum-data rule: >= {args.min_rows} rows over >= {args.min_batches} distinct "
             f"batch sizes. Below it a key borrows (see the fallback rule in profile.json).\n")
    fitted = sorted(k for k, e in full_model.items() if not e["fallback"])
    L.append("* fitted: " + (", ".join("/".join(k) for k in fitted) or "none"))
    short = sorted(k for k, rows in train.items() if not eligible(rows, vars(args)))
    if short:
        L.append("* measured but below the rule (borrowed instead): " + ", ".join(
            f"{'/'.join(k)} ({len(train[k])} rows, {len({r[3] for r in train[k]})} batches)"
            for k in short))
    borrowed = sorted((k, e["fallback"]) for k, e in full_model.items() if e["fallback"])
    L.append("* borrowed (per term: the first source whose fit saw it): " + (", ".join(
        f"{'/'.join(k)} <- " + "; ".join(f"{t}:{v['rule']} {v['from']}"
                                         for t, v in fb["terms"].items())
        for k, fb in borrowed) or "none"))
    L.append("* UNFITTABLE (a term no source saw; the route is not a candidate here): " + (
        ", ".join(f"{'/'.join(k)} (missing {','.join(v['unfittable'])}; tried "
                  f"{', '.join(v['tried'])})" for k, v in sorted(unfittable.items())) or "none"))
    L.append("* precision_scaled uses the PEAK FP64:FP32 rate; latency-bound kernels violate it "
             "(sm_89 tiny double measures ~2x float, not 64x).\n")
    L.append("## Constants (full fit)\n")
    L.append("| route | dtype | uplo | t_launch s | s/flop | s/byte | t_step s | rows | rms log | inactive | fallback |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (route, dtype, uplo), e in sorted(full_model.items()):
        c = e["constants"]
        fit = e["fit"] or {}
        fb = e["fallback"]
        L.append(f"| {route} | {dtype} | {uplo} | {c[0]:.3g} | {c[1]:.3g} | {c[2]:.3g} | {c[3]:.3g} "
                 f"| {fit.get('rows', 0)} | {fit.get('rms_log', float('nan')):.3f} "
                 f"| {','.join(fit.get('inactive', [])) or '-'} "
                 f"| {(fb['rule'] + ' <- ' + fb['from']) if fb else '-'} |")
    multi = [s for s in cv if len(s["measured"]) >= 2]
    single = [s for s in cv if len(s["measured"]) < 2]
    L.append("\n## Held-out regret (leave one n out)\n")
    L.append(f"Leave-one-n-out within each (dtype, uplo). Regret = time(model pick) / "
             f"time(best measured route the table supports). Margin {args.margin}.\n")
    st = summarise([s["regret"] for s in multi if s["regret"] is not None])
    L.append(f"* cells with >= 2 measured routes: {len(multi)}; scored {st.get('cells', 0)}; "
             + (f"geomean {st['geomean']:.4f}, p95 {st['p95']:.4f}, max {st['max']:.4f}, "
                f"{st['over_1.1']} over 1.1" if st.get("cells") else "nothing to score"))
    unscored = [s for s in multi if s["regret"] is None]
    L.append(f"* multi-route cells where the pick was NOT measured (unscorable): {len(unscored)}")
    L.append(f"* single-route cells: {len(single)} (regret is 1 by construction when the pick "
             f"is the measured route; {sum(1 for s in single if s['regret'] is None)} picked an "
             f"unmeasured route)")
    bad = sorted([s for s in multi if s["regret"] and s["regret"] > 1.1], key=lambda s: -s["regret"])
    if bad:
        L.append("\n| cell | pick | regret | measured | extrapolated |")
        L.append("|---|---|---|---|---|")
        for s in bad:
            meas = ", ".join(f"{r}={t * 1e3:.4g}ms" for r, t in sorted(s["measured"].items()))
            L.append(f"| {fmt_cell(s['cell'])} | {s['pick']} | {s['regret']:.3f} | {meas} "
                     f"| {s['extrapolated']} |")
    if unscored:
        L.append("\nUnscorable picks: " + "; ".join(
            f"{fmt_cell(s['cell'])} -> {s['pick']}" for s in unscored[:40]))

    L.append("\n## Extrapolated regions\n")
    L.append("Every (route, dtype, uplo) the profile prices, with its measured box. A prediction "
             "outside the box, or from a fallback entry, is extrapolated and flagged.\n")
    L.append("| route | dtype | uplo | measured n | measured batch | rows | sources / fallback |")
    L.append("|---|---|---|---|---|---|---|")
    for (route, dtype, uplo), e in sorted(full_model.items()):
        s = e["support"]
        if s:
            L.append(f"| {route} | {dtype} | {uplo} | {s['n_min']}..{s['n_max']} "
                     f"| {s['batch_min']}..{s['batch_max']} | {s['rows']} | {', '.join(s['sources'])}"
                     + (" **THIN** (own rows below the rule; other terms: " + ", ".join(
                         f"{t}:{v['rule']}" for t, v in e['fallback']['terms'].items()) + ")"
                        if e.get('thin') else "") + " |")
        else:
            fb = e["fallback"]
            L.append(f"| {route} | {dtype} | {uplo} | none | none | 0 "
                     f"| fallback {fb['rule']} <- {fb['from']} (x{fb['flop_factor']:.3g} on s/flop, "
                     f"x{fb['byte_factor']:.3g} on s/byte) |")
    absent = [(r, d, u) for r in ROUTES for d in DTYPES for u in ("L", "U")
              if (r, d, u) not in full_model]
    if absent:
        L.append("\nNo constants and no fallback (never a candidate): "
                 + ", ".join("/".join(a) for a in absent))
    if grid_rows:
        ext = [g for g in grid_rows if g["extrapolated"]]
        L.append(f"\nOn the --grid sweep, {len(ext)} of {len(grid_rows)} model picks are "
                 f"extrapolated: " + "; ".join(f"{fmt_cell(g['cell'])} -> {g['pick']}"
                                                for g in ext[:60]))

    if args.compare_windows:
        L.append("\n## Today's route_potrf.hh windows on the same cells\n")
        tw = []
        diffs = []
        for cell, measured in sorted(full_by_cell.items()):
            today = feats[cell]["auto"]
            pick, costs = choose(full_model, feats[cell], args.margin)
            tw.append(regret_of(today, measured))
            if pick != today:
                diffs.append((cell, today, pick, measured, costs))
        tst = summarise([r for r, cell in zip(tw, sorted(full_by_cell))
                         if r is not None and len(full_by_cell[cell]) >= 2])
        L.append("Today's choice is RouteTable<Op::potrf,T> resolved by potrf_plan_dump on the "
                 "same shape builder the library uses (vendor present).\n")
        L.append(f"* today, multi-route cells: " + (
            f"scored {tst['cells']}, geomean {tst['geomean']:.4f}, p95 {tst['p95']:.4f}, "
            f"max {tst['max']:.4f}, {tst['over_1.1']} over 1.1" if tst.get("cells") else "none scorable"))
        L.append(f"* model (held-out), its own scorable cells: " + (
            f"geomean {st['geomean']:.4f}, p95 {st['p95']:.4f}, max {st['max']:.4f}"
            if st.get("cells") else "none scorable"))
        L.append(f"* paired, gate population: see 'Ship gate' above ({gate['verdict']})")
        L.append(f"\n### Route diff: {len(diffs)} measured cells where the full-fit model "
                 f"differs from today\n")
        L.append("| cell | today | model | t(today) | t(model) | best measured | extrapolated |")
        L.append("|---|---|---|---|---|---|---|")
        for cell, today, pick, measured, costs in diffs:
            e = full_model.get((pick, cell[0], cell[1])) if pick else None
            ex = e is None or extrapolated(e, cell[2], cell[3])
            tt = f"{measured[today] * 1e3:.4g}ms" if today in measured else "unmeasured"
            tm = f"{measured[pick] * 1e3:.4g}ms" if pick in measured else "unmeasured"
            bm = min(measured, key=measured.get)
            L.append(f"| {fmt_cell(cell)} | {today} | {pick} | {tt} | {tm} | {bm} | {ex} |")
        if grid_rows:
            gd = [g for g in grid_rows if g["pick"] != g["today"]]
            L.append(f"\n### Route diff on the --grid sweep: {len(gd)} of {len(grid_rows)} shapes\n")
            for g in gd[:200]:
                L.append(f"* {fmt_cell(g['cell'])}: today {g['today']} -> model {g['pick']}"
                         + (" (extrapolated)" if g["extrapolated"] else ""))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def provenance(paths, plan_dump):
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:16]

    def git(*a):
        try:
            return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None

    campaigns = []
    for p in paths:
        cj = os.path.join(os.path.dirname(os.path.abspath(p)), "campaign.json")
        if os.path.exists(cj):
            with open(cj) as f:
                campaigns.append({"results": p, "provenance": json.load(f).get("provenance")})
    return {"sources": [{"path": p, "sha256_16": sha(p)} for p in paths],
            "campaigns": campaigns, "fit_git_sha": git("rev-parse", "--short", "HEAD"),
            "fit_git_dirty": bool(git("status", "--porcelain")), "plan_dump": plan_dump,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}


def grid_shapes(dtypes, uplos):
    ns = sorted({1, 2, 3, 4, 6, 8, 9, 12, 16, 17, 24, 32, 33, 36, 40, 48, 54, 55, 64, 77, 78,
                 96, 128, 160, 192, 256, 257, 320, 384, 512, 744, 745, 1024})
    bs = (64, 512, 4096, 32768)
    return [(d, u, n, b) for d in dtypes for u in uplos for n in ns for b in bs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--profile", required=True, help="a key of profiles/device_facts.json")
    ap.add_argument("--facts-file", default=FACTS_FILE)
    ap.add_argument("--facts", help="JSON object overriding the profile's device facts")
    ap.add_argument("--plan-dump", default=os.path.join(REPO, "build/tests/potrf_plan_dump"))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--op", default="potrf")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-rows", type=int, default=10)
    ap.add_argument("--min-batches", type=int, default=2)
    ap.add_argument("--borrow", action="append", default=[], metavar="PROFILE=profile.json",
                    help="a fitted profile of ANOTHER device to borrow unmeasured routes from")
    ap.add_argument("--include-stale", action="store_true",
                    help="keep rows with kernel_current=false (archive rows of an older kernel)")
    ap.add_argument("--compare-windows", action="store_true")
    ap.add_argument("--grid", action="store_true", help="also price a synthetic grid")
    args = ap.parse_args(argv)

    with open(args.facts_file) as f:
        all_facts = json.load(f)
    if args.profile not in all_facts and not args.facts:
        ap.error(f"profile {args.profile} not in {args.facts_file}; pass --facts")
    preset = dict(all_facts.get(args.profile, {}))
    if args.facts:
        preset.update(json.loads(args.facts))
    for k in FACT_KEYS:
        if k not in preset:
            ap.error(f"device fact {k} unknown for profile {args.profile}")
    borrows = []
    for spec in args.borrow:
        name, path = spec.split("=", 1)
        with open(path) as f:
            borrows.append((name, json.load(f), dict(all_facts[name])))
    cfg = {"facts": preset, "min_rows": args.min_rows, "min_batches": args.min_batches,
           "borrows": borrows}

    cells, skipped = load_rows(args.results, args.op, args.include_stale)
    if not cells:
        ap.error("no usable rows")
    shapes = {(d, u, n, b) for (d, u, n, b, _r) in cells}
    dts = sorted({s[0] for s in shapes})
    uls = sorted({s[1] for s in shapes})
    gshapes = grid_shapes(dts, uls) if args.grid else []
    feats = plan_features(args.plan_dump, preset, list(shapes) + gshapes)

    train, dropped = training_set(cells, feats, lambda *a: True)
    measured_full = fit_measured(train, cfg)
    unfittable = {}
    full_model = complete(measured_full, cfg, unfittable, train)
    cv = cross_validate(cells, feats, train, measured_full, cfg, args.margin)
    full_by_cell = group_cells(cells, feats)
    gate = ship_gate(cv, full_by_cell, feats)

    grid_rows = []
    for g in gshapes:
        pick, _ = choose(full_model, feats[g], args.margin)
        e = full_model.get((pick, g[0], g[1])) if pick else None
        grid_rows.append({"cell": g, "pick": pick, "today": feats[g]["auto"],
                          "extrapolated": e is None or extrapolated(e, g[2], g[3])})

    os.makedirs(args.out, exist_ok=True)
    prof = {"schema": 1, "op": args.op, "profile": args.profile, "device_facts": preset,
            "margin": args.margin, "objective": "least squares on log(time); constants = exp(theta)",
            "model": "t = t_launch*T0 + max(s_per_flop*T1, s_per_byte*T2) + t_step*T3 "
                     "(launch_plan::combine over launch_plan::cost_terms)",
            "fallback_rule": fallback.__doc__.strip(),
            "min_data_rule": {"rows": args.min_rows, "batches": args.min_batches},
            "borrowed_profiles": [b[0] for b in borrows], "gate": gate,
            "unfittable": {"/".join(k): v for k, v in sorted(unfittable.items())},
            "routes": defaultdict(lambda: defaultdict(dict)),
            "provenance": provenance(args.results, args.plan_dump)}
    for (route, dtype, uplo), e in sorted(full_model.items()):
        prof["routes"][route][dtype][uplo] = {
            "constants": dict(zip(PARAMS, e["constants"])), "fit": e["fit"],
            "support": e["support"], "fallback": e["fallback"], "thin": e.get("thin", False)}
    with open(os.path.join(args.out, "profile.json"), "w") as f:
        json.dump(prof, f, indent=1, sort_keys=True)
    write_report(os.path.join(args.out, "report.md"), args, cv, full_by_cell, feats,
                 full_model, skipped, dropped, grid_rows, train, gate, unfittable)
    multi = [s["regret"] for s in cv if len(s["measured"]) >= 2 and s["regret"] is not None]
    st = summarise(multi)
    print(json.dumps({"profile": args.profile, "cells": len(full_by_cell), "heldout": st,
                      "gate": gate["verdict"], "out": args.out}))
    print(f"ship gate {args.profile}: {gate['verdict']}")
    st["gate"] = gate
    return st


if __name__ == "__main__":
    main()

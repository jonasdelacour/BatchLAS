#!/usr/bin/env python3
"""Compile potrf's routing policy into first-match rules files, one per source.

    python3 evaluation/routing/compile_rules.py --plan-dump build/tests/potrf_plan_dump \
        [--nmax 2048] [--bexp 18] [--steps 1] [--eps 0] [--fidelity 10000] [--jobs 8]

Two sources, one format (routing/rules/<arch>/potrf.rules):

  model  a profile whose ship gate passed (profiles/<arch>.json): every grid cell is priced
         with fit.combine over potrf_plan_dump's cost terms, exactly as the C++ model_pick
         does; the vendor keeps a cell unless the best native beats it by the margin.
  hand   today's RouteTable windows, transcribed by sampling potrf_plan_dump's "auto" and
         "auto_vendor_free" over the same grid (routing/rules/hand/potrf.rules).

Each cell's action is a ranked list: the vendor-present pick, then the vendor-free pick, then
the vendor. The engine takes the first LEGAL entry, so the vendor-free build (vendor illegal)
gets the native argmin from the same rule. Cells are run-length merged in n per batch band,
then identical adjacent bands merge. Batch bands are the grid's batches, `--steps` per octave,
each band covering [b_i, b_{i+1} - 1] with b_i's label; --fidelity measures what that costs on
random off-grid shapes against the C++ oracle.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fit  # noqa: E402

ROUTES = fit.ROUTES
NATIVE = ROUTES[:-1]
DTYPES = fit.DTYPES
UPLOS = ("L", "U")


def key_index(d, u):
    return DTYPES.index(d) * 2 + UPLOS.index(u)


def dump_flags(arch):
    facts = fit.load_facts(arch)
    flags = ["--local-mem", str(facts["local_mem"]), "--max-wg", str(facts["max_wg"]),
             "--cus", str(facts["cus"]), "--max-threads-per-cu", str(facts["max_threads_per_cu"]),
             "--max-groups-per-cu", str(facts["max_groups_per_cu"])]
    for d, r in sorted(facts["regs"].items()):
        flags += ["--regs", d + "=" + ",".join(f"{k}:{v}" for k, v in sorted(r.items()))]
    return flags, facts


def run_dump(dump, flags, shapes, jobs):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/dpcpp-cuda/lib:" + env.get("LD_LIBRARY_PATH", "")
    chunks = [shapes[i::jobs] for i in range(jobs)]

    def one(chunk):
        stdin = "".join(f"{d} {n} {b} {u}\n" for (d, u, n, b) in chunk)
        r = subprocess.run([dump] + flags, input=stdin, capture_output=True, text=True,
                           env=env, check=True)
        return [json.loads(x) for x in r.stdout.splitlines()]

    out = {}
    with cf.ThreadPoolExecutor(jobs) as ex:
        for rows in ex.map(one, chunks):
            for j in rows:
                out[(j["dtype"], j["uplo"], j["n"], j["batch"])] = j
    return out


class ModelPricer:
    def __init__(self, arch, candidacy):
        self.prof = json.load(open(os.path.join(HERE, "profiles", arch + ".json")))
        self.margin = self.prof["margin"]
        self.cand = candidacy

    def consts(self, route, d, u):
        if route not in self.cand.get(f"{d}/{u}", ROUTES):
            return None
        e = self.prof["routes"].get(route, {}).get(d, {}).get(u)
        if not e or not e.get("constants"):
            return None
        return [e["constants"].get(k, 0.0) for k in fit.PARAMS]

    def costs(self, j):
        cs = {}
        for r, p in j["routes"].items():
            c = self.consts(r, j["dtype"], j["uplo"])
            if c is None or not p["supported"] or not p["fits"]:
                continue
            cs[r] = fit.combine(p["terms"], c)
        return cs

    def picks(self, j):
        """(vendor-present pick, vendor-free pick or None), model_pick's exact semantics."""
        cs = self.costs(j)
        nat = [(cs[r], i, r) for i, r in enumerate(NATIVE) if r in cs]
        best = min(nat)[2] if nat else None
        if "vendor" in cs and (best is None or not cs[best] < (1 - self.margin) * cs["vendor"]):
            pv = "vendor"
        else:
            pv = best if best else "vendor"
        if best is None:
            sup = [r for r in NATIVE if j["routes"][r]["supported"]]
            best = sup[0] if sup else None
        return pv, best, cs


def hand_picks(j):
    nv = j["auto_vendor_free"]
    return j["auto"], (None if nv == "vendor" else nv)


def rank_of(pv, pnv):
    out = [pv]
    if pnv and pnv not in out:
        out.append(pnv)
    if "vendor" not in out:
        out.append("vendor")
    return tuple(out)


def batches(bexp, steps):
    bs = set()
    for i in range(bexp * steps + 1):
        bs.add(int(round(2 ** (i / steps))))
    return sorted(bs)


def compress(label, ns, bs, cost=None, eps=0.0):
    """label(n, b) -> rank tuple. Returns [(b_lo, b_hi, [(n0, n1, rank)])] merged bands."""
    bands = []
    worst = 1.0
    for bi, b in enumerate(bs):
        runs = []
        for n in ns:
            lab = label(n, b)
            if runs:
                cur = runs[-1][2]
                ok = cur == lab
                if not ok and eps > 0 and cost is not None:
                    r = cost(n, b, cur, lab)
                    if r is not None and r <= 1 + eps:
                        ok = True
                        worst = max(worst, r)
                if ok:
                    runs[-1][1] = n
                    continue
            runs.append([n, n, lab])
        b_hi = bs[bi + 1] - 1 if bi + 1 < len(bs) else None
        bands.append([b, b_hi, [tuple(r) for r in runs]])
    merged = []
    for b, b_hi, runs in bands:
        if merged and merged[-1][2] == runs:
            merged[-1][1] = b_hi
        else:
            merged.append([b, b_hi, runs])
    return merged, worst


def write_rules(path, header, keyed):
    lines = list(header)
    rid = 0
    for d in DTYPES:
        for u in UPLOS:
            for b0, b1, runs in keyed[(d, u)]["bands"]:
                for i, (n0, n1, rank) in enumerate(runs):
                    rid += 1
                    nhi = "inf" if i == len(runs) - 1 else str(n1)
                    bhi = "inf" if b1 is None else str(b1)
                    lines.append(f"R{rid:04d} {d} {u} n [{n0},{nhi}] batch [{b0},{bhi}] -> "
                                 f"{' > '.join(rank)} ; src={keyed[(d, u)]['src']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return rid


def lookup(keyed, d, u, n, b):
    for b0, b1, runs in keyed[(d, u)]["bands"]:
        if b < b0 or (b1 is not None and b > b1):
            continue
        for i, (n0, n1, rank) in enumerate(runs):
            if n >= n0 and (n <= n1 or i == len(runs) - 1):
                return rank
    return None


def first_legal(rank, j, vendor):
    for r in rank:
        if r == "vendor":
            if vendor:
                return r
        elif j["routes"][r]["supported"]:
            return r
    return "vendor"


def compile_getrs(a):
    """getrs: features (n, nrhs, batch), key (dtype, trans). Hand source only: the ranks come
    from route_oracle, already legality-free, so capacity never fragments a box."""
    out = subprocess.run([a.oracle, "getrs"], capture_output=True, text=True, check=True).stdout
    data = {}
    for line in out.splitlines():
        d, t, b, r, n0, n1, rank = line.split(" ", 6)
        data.setdefault((d, t), {}).setdefault(int(b), {}).setdefault(int(r), []).append(
            (int(n0), n1, rank))
    cand = json.load(open(a.candidacy))["getrs"]["candidates"]
    lines = ["# GENERATED by evaluation/routing/compile_rules.py --op getrs -- never edit.",
             "# op=getrs arch=hand source=hand provenance=RouteTable<getrs>::preferred+native_tier_"
             "preferred-sampled-by-route_oracle",
             "# rank = preferred hits > vendor > tie-break hits > natives, each asked with unbounded",
             "# capacity: legal() (fused n*nrhs ceiling) decides the rest at run time.",
             "# id     dtype trans  n [lo,hi]  nrhs [lo,hi]  batch [lo,hi]  -> ranked tiers"]
    rid = 0
    for d in DTYPES:
        for t in ("N", "T", "C"):
            per_b = data[(d, t)]
            bs = sorted(per_b)
            bands = []
            for bi, b in enumerate(bs):
                rs = sorted(per_b[b])
                nb = []
                for ri, r in enumerate(rs):
                    runs = tuple(per_b[b][r])
                    r_hi = rs[ri + 1] - 1 if ri + 1 < len(rs) else None
                    if nb and nb[-1][2] == runs:
                        nb[-1][1] = r_hi
                    else:
                        nb.append([r, r_hi, runs])
                b_hi = bs[bi + 1] - 1 if bi + 1 < len(bs) else None
                key = [(x[0], x[1], x[2]) for x in nb]
                if bands and bands[-1][2] == key:
                    bands[-1][1] = b_hi
                else:
                    bands.append([b, b_hi, key])
            for b0, b1, nbands in bands:
                for r0, r1, runs in nbands:
                    for n0, n1, rank in runs:
                        bad = set(rank.split(" > ")) - set(cand[f"{d}/{t}"])
                        if bad:
                            sys.exit(f"compile_rules: getrs ranks non-candidates {bad} for {d}/{t}")
                        rid += 1
                        fmt = lambda v: "inf" if v is None else str(v)
                        lines.append(f"R{rid:04d} {d} {t} n [{n0},{n1}] nrhs [{r0},{fmt(r1)}] "
                                     f"batch [{b0},{fmt(b1)}] -> {rank} ; src=hand")
    path = os.path.join(a.out, "hand", "getrs.rules")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"getrs hand: {rid} rules -> {os.path.relpath(path, REPO)}", file=sys.stderr)


def main():
    if "--op" in sys.argv and sys.argv[sys.argv.index("--op") + 1] == "getrs":
        ap = argparse.ArgumentParser()
        ap.add_argument("--op")
        ap.add_argument("--oracle", required=True)
        ap.add_argument("--out", default=os.path.join(REPO, "routing", "rules"))
        ap.add_argument("--candidacy", default=os.path.join(REPO, "routing", "candidacy.json"))
        compile_getrs(ap.parse_args())
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-dump", required=True)
    ap.add_argument("--arch", default="sm_120")
    ap.add_argument("--nmax", type=int, default=2048)
    ap.add_argument("--bexp", type=int, default=18)
    ap.add_argument("--steps", type=int, default=1, help="batch grid points per octave")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--fidelity", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(REPO, "routing", "rules"))
    ap.add_argument("--candidacy", default=os.path.join(REPO, "routing", "candidacy.json"))
    a = ap.parse_args()

    cand_all = json.load(open(a.candidacy))["potrf"]
    cand = cand_all["candidates"]
    flags, facts = dump_flags(a.arch)
    pricer = ModelPricer(a.arch, cand)
    gate = pricer.prof["gate"]["verdict"]
    ns = list(range(1, a.nmax + 1))
    bs = batches(a.bexp, a.steps)
    shapes = [(d, u, n, b) for d in DTYPES for u in UPLOS for n in ns for b in bs]
    print(f"grid: {len(shapes)} cells ({len(ns)} n x {len(bs)} batch x 8 keys)", file=sys.stderr)
    cells = run_dump(a.plan_dump, flags + ["--profile", a.arch], shapes, a.jobs)

    agree = mism = 0
    model_lab, hand_lab, model_cost = {}, {}, {}
    for k, j in cells.items():
        pv, pnv, cs = pricer.picks(j)
        if pv == j.get("auto_model"):
            agree += 1
        else:
            mism += 1
        model_lab[k] = rank_of(pv, pnv)
        model_cost[k] = cs
        hand_lab[k] = rank_of(*hand_picks(j))
    print(f"python model pick == C++ auto_model: {agree}/{agree + mism}", file=sys.stderr)
    if mism:
        sys.exit("compile_rules: the Python pricing no longer reproduces the C++ model")

    def regret(d, u):
        def f(n, b, cur, lab):
            cs = model_cost[(d, u, n, b)]
            if cur[0] not in cs or lab[0] not in cs:
                return None
            if len(cur) > 1 and len(lab) > 1 and cur[1] != lab[1]:
                if cur[1] not in cs or lab[1] not in cs:
                    return None
                r2 = cs[cur[1]] / cs[lab[1]]
            else:
                r2 = 1.0
            return max(cs[cur[0]] / cs[lab[0]], r2)
        return f

    # Candidacy: a ranked tier outside the candidate set, or a native tier one strategy ranks
    # first and the other never does, fails the compile unless waived.
    waived = {(w["key"], w["tier"]) for w in cand_all.get("waivers", [])}
    errors, splits = [], []
    for d in DTYPES:
        for u in UPLOS:
            ks = [k for k in cells if k[0] == d and k[1] == u]
            for name, lab in (("model", model_lab), ("hand", hand_lab)):
                used = {r for k in ks for r in lab[k]}
                bad = used - set(cand[f"{d}/{u}"])
                if bad:
                    errors.append(f"{name} ranks non-candidates {sorted(bad)} for {d}/{u}")
            m_first = {model_lab[k][0] for k in ks} - {"vendor"}
            h_first = {hand_lab[k][0] for k in ks} - {"vendor"}
            for t in sorted(m_first ^ h_first):
                splits.append((f"{d}/{u}", t, "model" if t in m_first else "hand"))
    for key, t, only in splits:
        tag = "waived" if (key, t) in waived else "UNWAIVED"
        print(f"candidacy split {key} {t}: only the {only} strategy ranks it first ({tag})",
              file=sys.stderr)
        if (key, t) not in waived:
            errors.append(f"candidacy split {key} {t} (only {only}) has no waiver")
    for key, t in sorted(waived - {(k, t) for k, t, _ in splits}):
        print(f"stale waiver {key} {t}: no split left to excuse", file=sys.stderr)
    if errors:
        sys.exit("compile_rules: " + "; ".join(errors))

    psha = hashlib.sha256(open(os.path.join(HERE, "profiles", a.arch + ".json"), "rb").read())
    grid = f"n 1..{a.nmax} x batch 2^0..2^{a.bexp} ({a.steps}/octave)"
    compiled = (f"# compiled_for cus={facts['cus']} local_mem={facts['local_mem']} "
                f"max_wg={facts['max_wg']} sg=32")
    results = {}
    for src, lab, arch_dir in (("model", model_lab, a.arch), ("hand", hand_lab, "hand")):
        if src == "model" and gate != "PASS":
            continue
        keyed, worst_all = {}, 1.0
        for d in DTYPES:
            for u in UPLOS:
                f = (lambda n, b, d=d, u=u, lab=lab: lab[(d, u, n, b)])
                bands, worst = compress(f, ns, bs, regret(d, u) if src == "model" else None,
                                        a.eps if src == "model" else 0.0)
                worst_all = max(worst_all, worst)
                keyed[(d, u)] = {"bands": bands, "src": src}
        prov = (f"profiles/{a.arch}.json@{psha.hexdigest()[:12]};gate={gate};eps={a.eps}"
                if src == "model" else "RouteTable<potrf>::preferred-windows-sampled-by-potrf_plan_dump")
        header = ["# GENERATED by evaluation/routing/compile_rules.py -- never edit; corrections",
                  "# go in routing/overrides/<arch>/potrf.rules with an evidence: pointer.",
                  f"# op=potrf arch={arch_dir} source={src} provenance={prov}",
                  f"# grid {grid}; rank = vendor-present pick > vendor-free pick > vendor",
                  compiled,
                  "# id     dtype uplo  n [lo,hi]  batch [lo,hi]  -> ranked tiers (first legal wins)"]
        path = os.path.join(a.out, arch_dir, "potrf.rules")
        count = write_rules(path, header, keyed)
        per = {f"{d}/{u}": sum(len(r[2]) for r in keyed[(d, u)]["bands"])
               for d in DTYPES for u in UPLOS}
        results[src] = keyed
        print(f"{src}: {count} rules -> {os.path.relpath(path, REPO)}; per key {per}; "
              f"worst predicted regret {worst_all:.3f}", file=sys.stderr)

    if a.fidelity:
        random.seed(7)
        probe = set()
        while len(probe) < a.fidelity:
            probe.add((random.choice(DTYPES), random.choice(UPLOS), random.randint(1, a.nmax),
                       max(1, int(2 ** random.uniform(0, a.bexp)))))
        probe = sorted(probe)
        truth = run_dump(a.plan_dump, flags + ["--profile", a.arch], probe, a.jobs)
        for src, keyed in results.items():
            bad = []
            for (d, u, n, b) in probe:
                j = truth[(d, u, n, b)]
                rank = lookup(keyed, d, u, n, b)
                if src == "model":
                    pv, pnv, _ = pricer.picks(j)
                else:
                    pv, pnv = hand_picks(j)
                got_v = first_legal(rank, j, True)
                got_nv = first_legal(rank, j, False)
                if got_v != pv or got_nv != (pnv or "vendor"):
                    bad.append((d, u, n, b, got_v, pv, got_nv, pnv))
            print(f"fidelity {src}: {len(bad)}/{len(probe)} off-grid shapes disagree "
                  f"({len(bad) / len(probe):.3%})", file=sys.stderr)
            for x in bad[:12]:
                print("   ", x, file=sys.stderr)


if __name__ == "__main__":
    main()

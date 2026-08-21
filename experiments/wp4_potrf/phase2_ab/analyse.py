#!/usr/bin/env python3
"""Summaries of the Phase 2 CSVs. Reads only, writes nothing."""
import csv, os, sys, collections

D = os.path.dirname(os.path.abspath(__file__))


def rows(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("mode") is None:
                continue
            out.append(r)
    return out


def panel():
    d = {}
    for r in rows("panel.csv"):
        k = (r["type"], int(r["n"]), int(r["j"]), int(r["m2"]), int(r["ib"]))
        d.setdefault(k, {})[r["route"]] = (float(r["med_ms"]), float(r["rel_sd"]))
    print("\n== OQ5: panel trsm(Right,Lower,ConjTrans,NonUnit), sub-views at parent ld, batch 128")
    print(f"{'type':8}{'n':>6}{'j':>6}{'m2':>6}{'ib':>5}"
          f"{'default':>10}{'vendor':>10}{'native':>10}{'ven/nat':>9}{'ven/def':>9}{'maxsd':>8}")
    for k in sorted(d):
        v = d[k]
        dm, ve, na = (v.get(x, (float('nan'), 0)) for x in ("default", "vendor", "native"))
        sd = max(dm[1], ve[1], na[1])
        print(f"{k[0]:8}{k[1]:>6}{k[2]:>6}{k[3]:>6}{k[4]:>5}"
              f"{dm[0]:>10.4f}{ve[0]:>10.4f}{na[0]:>10.4f}"
              f"{ve[0]/na[0]:>9.2f}{ve[0]/dm[0]:>9.2f}{sd:>8.3f}")


def trail():
    d = {}
    for r in rows("trail.csv"):
        k = (r["type"], int(r["m"]), int(r["n"]), int(r["k"]), r["tB"], r["store"])
        d.setdefault(k, {})[r["route"]] = (float(r["med_ms"]), float(r["rel_sd"]))
    keys = sorted({k[:5] for k in d})
    print("\n== OQ6: trailing gemm C -= L21 L21^H, alpha=-1 beta=1, batch 128")
    print(f"{'type':8}{'m':>6}{'n':>6}{'k':>5} tB"
          f"{'sub:def':>10}{'sub:ven':>10}{'sub:nat':>10}"
          f"{'flat:def':>10}{'flat:ven':>10}{'flat:nat':>10}"
          f"{'nat/ven':>9}{'sub/flat(n)':>12}{'sub/flat(v)':>12}{'maxsd':>8}")
    for k in keys:
        s = d.get(k + ("sub",), {})
        f = d.get(k + ("flat",), {})
        g = lambda x, r: x.get(r, (float('nan'), 0))[0]
        sd = max([v[1] for v in list(s.values()) + list(f.values())] or [0])
        print(f"{k[0]:8}{k[1]:>6}{k[2]:>6}{k[3]:>5} {k[4]} "
              f"{g(s,'default'):>10.4f}{g(s,'vendor'):>10.4f}{g(s,'native'):>10.4f}"
              f"{g(f,'default'):>10.4f}{g(f,'vendor'):>10.4f}{g(f,'native'):>10.4f}"
              f"{g(s,'vendor')/g(s,'native'):>9.2f}"
              f"{g(s,'native')/g(f,'native'):>12.2f}{g(s,'vendor')/g(f,'vendor'):>12.2f}{sd:>8.3f}")


def blocked():
    rs = [r for r in rows("blocked.csv") if r["mode"] == "blocked"]
    print("\n== nb sweep: whole blocked driver, W=128, batch 128")
    print("   CORRECT? = residual within 100x the type's eps*n AND info_nonzero == 0")
    print(f"{'mode':9}{'type':8}{'n':>6}{'nb':>5}{'W':>5}"
          f"{'total_ms':>10}{'leaf':>8}{'panel':>8}{'trail':>8}"
          f"{'panel%':>8}{'leaf%':>7}{'trail%':>8}{'gflops':>9}{'resid':>11}{'bad':>5}{'sd':>7}")
    for r in rs:
        stg = float(r["leaf_ms"]) + float(r["panel_ms"]) + float(r["trail_ms"])
        if stg <= 0:
            continue
        lf, pn, tr = (float(r[x]) / stg * 100 for x in ("leaf_ms", "panel_ms", "trail_ms"))
        ok = float(r["residual"]) < (1e-4 if r["type"] in ("float", "cfloat") else 1e-11) \
             and int(r["info_nonzero"]) == 0
        print(f"{r['routemode']:9}{r['type']:8}{int(r['n']):>6}{int(r['nb']):>5}{int(r['W']):>5}"
              f"{float(r['med_ms']):>10.3f}{float(r['leaf_ms']):>8.3f}"
              f"{float(r['panel_ms']):>8.3f}{float(r['trail_ms']):>8.3f}"
              f"{pn:>8.1f}{lf:>7.1f}{tr:>8.1f}{float(r['gflops']):>9.1f}"
              f"{float(r['residual']):>11.2e}{int(r['info_nonzero']):>5}"
              f"{float(r['rel_sd']):>7.3f}"
              + ("" if ok else "   <-- WRONG ANSWER"))

    print("\n-- best correct nb per (routemode, type, n), by total_ms")
    best = {}
    for r in rs:
        ok = float(r["residual"]) < (1e-4 if r["type"] in ("float", "cfloat") else 1e-11) \
             and int(r["info_nonzero"]) == 0
        if not ok or int(r["W"]) != 128:
            continue
        k = (r["routemode"], r["type"], int(r["n"]))
        t = float(r["med_ms"])
        if k not in best or t < best[k][0]:
            best[k] = (t, int(r["nb"]))
    for k in sorted(best):
        print(f"   {k[0]:9}{k[1]:8} n={k[2]:<6} nb={best[k][1]:<5} {best[k][0]:.3f} ms")

    print("\n-- vendorpotrf (cuSOLVER) reference")
    with open(os.path.join(D, "blocked.csv")) as f:
        for line in f:
            if line.startswith("vendorpotrf,"):
                print("   " + line.strip())


def blocked2():
    rs = [r for r in rows("blocked2.csv") if r["mode"] == "blocked"]
    if not rs:
        return
    print("\n== supplement: correct-answer nb and W (vendor trsm, native gemm), batch 128")
    print(f"{'mode':8}{'type':8}{'n':>6}{'nb':>5}{'W':>5}"
          f"{'total_ms':>10}{'leaf':>8}{'panel':>8}{'trail':>8}"
          f"{'gflops':>9}{'resid':>11}{'bad':>5}{'sd':>7}")
    for r in rs:
        ok = float(r["residual"]) < (1e-4 if r["type"] in ("float", "cfloat") else 1e-11) \
             and int(r["info_nonzero"]) == 0
        print(f"{r['routemode']:8}{r['type']:8}{int(r['n']):>6}{int(r['nb']):>5}{int(r['W']):>5}"
              f"{float(r['med_ms']):>10.3f}{float(r['leaf_ms']):>8.3f}"
              f"{float(r['panel_ms']):>8.3f}{float(r['trail_ms']):>8.3f}"
              f"{float(r['gflops']):>9.1f}{float(r['residual']):>11.2e}"
              f"{int(r['info_nonzero']):>5}{float(r['rel_sd']):>7.3f}"
              + ("" if ok else "   <-- WRONG ANSWER"))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "panel"):
        panel()
    if which in ("all", "trail"):
        trail()
    if which in ("all", "blocked"):
        blocked()
    if which in ("all", "blocked", "blocked2"):
        blocked2()

"""Build comparison: two campaigns, paired cell by cell.

    python3 benchmarks/benchviz compare <baseline> <candidate>

A comparison is a campaign directory like any other, so the dashboard, the
figures and the export all work on it unchanged. Its results.jsonl is not
stored: rows() derives it from the two source campaigns on every read, so a
comparison against a campaign still running fills in as that campaign does.

The pairing puts the candidate's BatchLAS arm in the `batchlas` slot and the
baseline's BatchLAS arm in the `vendor` slot. A speedup above 1x therefore
means the candidate build is faster. The ops no vendor library has (stedc,
steqr, sytrd, ...) get a speedup here too, which a single campaign cannot give
them.

Both campaigns ran the vendor arm too, from the same vendor library. Its
ratio between the two runs is the CONTROL: the vendor did not change, so
anything far from 1x is the machine (clocks, contention, driver) and biases
every speedup in the comparison by the same amount. README.md, "Comparing
builds".
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ops import OPS
from store import Campaign

KIND = "compare"
_KEY = ("op", "dtype", "m", "n", "nrhs", "batch")


def is_compare(cfg: dict) -> bool:
    return cfg.get("kind") == KIND


def _latest_ok(camp: Campaign, arm: str) -> Dict[Tuple, dict]:
    """The newest verified row per cell of one arm, as paired() counts it."""
    rs = [r for r in camp.rows() if r.get("arm") == arm and r.get("ok")]
    return {tuple(r.get(f) for f in _KEY): r for r in sorted(rs, key=lambda r: r.get("t", 0))}


def build_label(cfg: dict) -> str:
    """What a figure calls a campaign's build: the commit its binaries were
    built from, else the checkout's HEAD when it ran, else its name."""
    prov = cfg.get("provenance", {})
    builds = [b for b in prov.get("builds") or [] if b.get("built_from")]
    if builds:
        return f"Build {builds[0]['built_from']}"
    if prov.get("git_sha"):
        return f"Build {prov['git_sha']}"
    return cfg.get("name", "?")


def _same_build(a: dict, b: dict) -> bool:
    ba = [(x.get("dir"), x.get("built")) for x in a.get("provenance", {}).get("builds") or []]
    bb = [(x.get("dir"), x.get("built")) for x in b.get("provenance", {}).get("builds") or []]
    return bool(ba) and ba == bb


def create(root: Path, name: str, base: str, new: str) -> Campaign:
    root = Path(root)
    srcs = {}
    for role, src in (("base", base), ("new", new)):
        c = Campaign(root, src)
        if not (c.dir / "campaign.json").exists():
            raise ValueError(f"no campaign {src!r} under {root}")
        cfg = c.config
        if is_compare(cfg):
            raise ValueError(f"{src} is itself a comparison")
        srcs[role] = {**cfg, "name": src}
    if srcs["base"].get("backend") != srcs["new"].get("backend"):
        raise ValueError("the two campaigns ran on different backends")
    lb, ln = build_label(srcs["base"]), build_label(srcs["new"])
    if lb == ln:  # the same commit, e.g. a rebuild with a different compiler: say which campaign
        lb, ln = f"{lb} ({base})", f"{ln} ({new})"
    pb, pn = srcs["base"].get("provenance", {}), srcs["new"].get("provenance", {})
    warnings = []
    if pb.get("device") and pn.get("device") and pb["device"] != pn["device"]:
        warnings.append(f"different devices: {pb['device']} vs {pn['device']}")
    if _same_build(srcs["base"], srcs["new"]):
        warnings.append("both campaigns measured the same binaries: every speedup here is run-to-run noise")
    cfg = {
        "kind": KIND, "base": base, "new": new, "preset": KIND, "backend": srcs["new"].get("backend", "cuda"),
        "ops": [o for o in srcs["new"].get("ops", []) if o in srcs["base"].get("ops", [])],
        "types": [t for t in srcs["new"].get("types", []) if t in srcs["base"].get("types", [])],
        "provenance": {
            "kind": KIND, "ref_label": lb, "new_label": ln, "warnings": warnings,
            "device": pn.get("device"), "driver": pn.get("driver"), "vendor_label": pn.get("vendor_label", ""),
            "base": {"campaign": base, **{k: pb.get(k) for k in ("builds", "git_sha", "started", "device", "driver")}},
            "new": {"campaign": new, **{k: pn.get(k) for k in ("builds", "git_sha", "started", "device", "driver")}},
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    grid = srcs["new"].get("grid") or srcs["base"].get("grid")
    if grid:  # absent in campaigns from before grids; Grid.from_config then uses the preset
        cfg["grid"] = grid
    camp = Campaign(root, name)
    camp.dir.mkdir(parents=True, exist_ok=True)
    camp.figures.mkdir(exist_ok=True)
    cfg_path = camp.dir / "campaign.json"
    if cfg_path.exists() and not is_compare(json.loads(cfg_path.read_text())):
        raise ValueError(f"{name} is a measured campaign; pick another name")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    camp.set_status(KIND)
    return camp


def sources(camp: Campaign) -> Tuple[Campaign, Campaign]:
    cfg = camp.config
    return Campaign(camp.root, cfg["base"]), Campaign(camp.root, cfg["new"])


def rows(camp: Campaign) -> List[dict]:
    """The pairable cells only: a cell measured in one campaign has no ratio."""
    base, new = sources(camp)
    b, n = _latest_ok(base, "batchlas"), _latest_ok(new, "batchlas")
    out = []
    for k in n.keys() & b.keys():
        out.append(n[k])
        out.append({**b[k], "arm": "vendor", "t": n[k].get("t", 0)})
    return out


def _geomean(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def control(camp: Campaign) -> dict:
    """Coverage of the pairing, and the vendor arm's ratio between the two runs."""
    base, new = sources(camp)
    bb, nb = _latest_ok(base, "batchlas"), _latest_ok(new, "batchlas")
    bv, nv = _latest_ok(base, "vendor"), _latest_ok(new, "vendor")
    per_op: Dict[str, List[float]] = {}
    for k in nv.keys() & bv.keys():
        t_new, t_base = float(nv[k]["time_ms"]), float(bv[k]["time_ms"])
        if t_new > 0 and t_base > 0:
            per_op.setdefault(k[0], []).append(t_base / t_new)
    allr = [x for v in per_op.values() for x in v]
    return {
        "matched": len(nb.keys() & bb.keys()), "only_base": len(bb.keys() - nb.keys()),
        "only_new": len(nb.keys() - bb.keys()),
        "vendor": {"geomean": _geomean(allr) if allr else None, "cells": len(allr),
                   "per_op": {o: {"geomean": _geomean(v), "cells": len(v)} for o, v in per_op.items()}},
    }


def data_key(camp: Campaign) -> tuple:
    """Changes whenever either source campaign gains a row."""
    key = []
    for c in sources(camp):
        try:
            st = c.results.stat()
            key.append((st.st_size, st.st_mtime))
        except FileNotFoundError:
            key.append((0, 0))
    return tuple(key)


def report(camp: Campaign) -> str:
    """The comparison as text: per op and precision, the geomean speedup at saturation."""
    import numpy as np
    from plots import paired, saturated

    prov = camp.config["provenance"]
    w = paired(camp.rows())
    lines = [f"{prov['new_label']} vs {prov['ref_label']}: speedup > 1x means {prov['new_label']} is faster"]
    for msg in prov.get("warnings", []):
        lines.append(f"  WARNING: {msg}")
    ctl = control(camp)
    lines.append(f"  {ctl['matched']} cells in both; {ctl['only_new']} only in {camp.config['new']}, "
                 f"{ctl['only_base']} only in {camp.config['base']}")
    v = ctl["vendor"]
    if v["geomean"]:
        lines.append(f"  vendor control: {v['geomean']:.3f}x over {v['cells']} cells (1x = same machine state)")
    s = saturated(w) if not w.empty else w
    if s.empty:
        lines.append("  no cell measured in both campaigns")
        return "\n".join(lines)
    lines.append(f"  {'op':10s} {'dtype':8s} {'geomean':>8s} {'min':>7s} {'max':>7s}  cells")
    for op in [o for o in OPS if (s.op == o).any()]:
        for t in s[s.op == op].dtype.unique():
            sp = s[(s.op == op) & (s.dtype == t)].speedup.astype(float).to_numpy()
            lines.append(f"  {op:10s} {t:8s} {np.exp(np.log(sp).mean()):7.3f}x {sp.min():6.3f}x {sp.max():6.3f}x  {sp.size}")
    return "\n".join(lines)

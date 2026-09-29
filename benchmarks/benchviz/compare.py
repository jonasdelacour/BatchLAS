"""Compare any two logs: two campaigns' results, paired cell by cell.

    python3 benchmarks/benchviz logs                             # every log on this box
    python3 benchmarks/benchviz compare <baseline> <candidate>   # names, dirs or results.jsonl paths
    python3 benchmarks/benchviz compare old new --base-arm vendor --new-arm vendor   # e.g. a driver update

A source is a campaign name (looked up in --root, then in every checkout's
benchviz_runs/), a campaign directory, or its results.jsonl. The two need not
share a grid, a checkout or a purpose; neither was necessarily run to be compared.

A comparison is a campaign directory like any other, so the dashboard, the
figures and the export work on it unchanged. It stores no results: rows()
derives them from the two sources on every read, so a comparison against a
campaign still running fills in as that campaign does.

Pairing puts the candidate's chosen arm (BatchLAS by default) in the `batchlas`
slot and the baseline's chosen arm in the `vendor` slot; above 1x the
candidate is faster.

Matching, in order:
  1. The same cell (op, precision, shape, batch) in both logs.
  2. A shape both logs measured, but never at a common batch (two grids, two
     memory budgets): each log's largest batch, compared by time per matrix.
     That assumes both points are saturated, which is what the largest batch of
     a ladder is for. These pairs are counted as `rescaled`, and --exact drops them.

When both sides compare BatchLAS, the vendor arm that both logs also ran is the
CONTROL: the vendor did not change, so its ratio is the machine (clocks,
contention, driver) and biases every speedup by the same amount. README.md,
"Comparing logs".
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ops import OPS
from store import Campaign, discover_logs

KIND = "compare"
_SHAPE = ("op", "dtype", "m", "n", "nrhs")
ARMS = ("batchlas", "vendor")


def is_compare(cfg: dict) -> bool:
    return cfg.get("kind") == KIND


def resolve(root: Path, spec: str) -> Path:
    """A campaign directory from a name, a directory, or a results.jsonl path."""
    p = Path(spec).expanduser()
    if p.name == "results.jsonl":
        p = p.parent
    if p.is_absolute() or p.exists():
        if (p / "campaign.json").is_file():
            return p.resolve()
        raise ValueError(f"{spec}: not a campaign directory (no campaign.json)")
    if (Path(root) / spec / "campaign.json").is_file():
        return (Path(root) / spec).resolve()
    hits = [x for x in discover_logs() if x["name"] == spec and x["kind"] != KIND]
    if len(hits) == 1:
        return Path(hits[0]["path"])
    if hits:
        raise ValueError(f"{spec!r} names {len(hits)} campaigns; pass one of these paths:\n  "
                         + "\n  ".join(h["path"] for h in hits))
    raise ValueError(f"no campaign {spec!r} in {root} or any checkout's benchviz_runs/ (see `benchviz logs`)")


def _open(p: str) -> Campaign:
    p = Path(p)
    return Campaign(p.parent, p.name)


def _latest_ok(camp: Campaign, arm: str) -> Dict[Tuple, dict]:
    """The newest verified row per cell of one arm, as paired() counts it."""
    rs = [r for r in camp.rows() if r.get("arm") == arm and r.get("ok") and r.get("time_ms")]
    return {tuple(r.get(f) for f in _SHAPE) + (r.get("batch"),): r for r in sorted(rs, key=lambda r: r.get("t", 0))}


def match(b: Dict[Tuple, dict], n: Dict[Tuple, dict], exact: bool = False) -> Tuple[List[Tuple[dict, dict]], dict]:
    """Pairs of (baseline row, candidate row) on the candidate's cell key, and
    what was matched how. A rescaled baseline row carries the candidate's batch
    and a time scaled by the batch ratio; its own batch is kept as batch_orig."""
    pairs = [(b[k], n[k]) for k in n.keys() & b.keys()]
    shapes_b: Dict[Tuple, List[Tuple]] = {}
    shapes_n: Dict[Tuple, List[Tuple]] = {}
    for k in b:
        shapes_b.setdefault(k[:-1], []).append(k)
    for k in n:
        shapes_n.setdefault(k[:-1], []).append(k)
    exact_shapes = {k[:-1] for k in n.keys() & b.keys()}
    rescaled = 0
    if not exact:
        for s in (shapes_b.keys() & shapes_n.keys()) - exact_shapes:
            kb, kn = max(shapes_b[s], key=lambda k: k[-1]), max(shapes_n[s], key=lambda k: k[-1])
            rb = b[kb]
            scaled = {**rb, "batch": kn[-1], "batch_orig": kb[-1],
                      "time_ms": float(rb["time_ms"]) * kn[-1] / kb[-1]}
            pairs.append((scaled, n[kn]))
            rescaled += 1
    matched_shapes = exact_shapes | (set() if exact else shapes_b.keys() & shapes_n.keys())
    return pairs, {
        "matched": len(pairs), "rescaled": rescaled,
        "only_base": len(shapes_b.keys() - matched_shapes), "only_new": len(shapes_n.keys() - matched_shapes),
    }


def build_label(cfg: dict) -> str:
    """What a figure calls a log's build: the commit its binaries were built
    from; else, for logs from before builds were recorded, the HEAD of the
    checkout benchviz ran from, which is not necessarily what was measured."""
    prov = cfg.get("provenance", {})
    builds = [b for b in prov.get("builds") or [] if b.get("built_from")]
    if builds:
        return f"Build {builds[0]['built_from']}"
    if prov.get("git_sha"):
        return f"HEAD {prov['git_sha']}"
    return cfg.get("name", "?")


def _same_build(a: dict, b: dict) -> bool:
    ba = [(x.get("dir"), x.get("built")) for x in a.get("provenance", {}).get("builds") or []]
    bb = [(x.get("dir"), x.get("built")) for x in b.get("provenance", {}).get("builds") or []]
    return bool(ba) and ba == bb


def _label(cfg: dict, arm: str, other_arm: str) -> str:
    """The build, plus the arm whenever it is not BatchLAS on both sides."""
    lab = build_label(cfg)
    if arm != other_arm or arm == "vendor":
        lab += " vendor" if arm == "vendor" else " BatchLAS"
    return lab


def create(root: Path, name: str, base: str, new: str, base_arm: str = "batchlas", new_arm: str = "batchlas",
           exact: bool = False) -> Campaign:
    root = Path(root)
    if base_arm not in ARMS or new_arm not in ARMS:
        raise ValueError(f"an arm is one of {', '.join(ARMS)}")
    srcs, paths = {}, {}
    for role, spec in (("base", base), ("new", new)):
        p = resolve(root, spec)
        cfg = json.loads((p / "campaign.json").read_text())
        if is_compare(cfg):
            raise ValueError(f"{spec} is itself a comparison")
        srcs[role], paths[role] = {**cfg, "name": p.name}, str(p)
    if paths["base"] == paths["new"] and base_arm == new_arm:
        raise ValueError("that compares a log's arm with itself")
    if srcs["base"].get("backend", "cuda") != srcs["new"].get("backend", "cuda"):
        raise ValueError("the two logs ran on different backends")
    lb, ln = _label(srcs["base"], base_arm, new_arm), _label(srcs["new"], new_arm, base_arm)
    if lb == ln:  # the same commit, e.g. a rerun or a rebuild: say which log
        lb, ln = f"{lb} ({srcs['base']['name']})", f"{ln} ({srcs['new']['name']})"
    pb, pn = srcs["base"].get("provenance", {}), srcs["new"].get("provenance", {})
    warnings = []
    if pb.get("device") and pn.get("device") and pb["device"] != pn["device"]:
        warnings.append(f"different devices: {pb['device']} vs {pn['device']}")
    if base_arm == new_arm == "batchlas" and _same_build(srcs["base"], srcs["new"]):
        warnings.append("both logs measured the same binaries: every speedup here is run-to-run noise")
    gb, gn = (srcs[r].get("grid") or {} for r in ("base", "new"))
    if (gb.get("batch_mode"), gb.get("mem_gib")) != (gn.get("batch_mode"), gn.get("mem_gib")):
        warnings.append("the two logs used different batch grids; shapes with no common batch are compared "
                        "by time per matrix at each log's largest batch")
    cfg = {
        "kind": KIND, "base": paths["base"], "new": paths["new"], "base_arm": base_arm, "new_arm": new_arm,
        "exact": exact, "preset": KIND, "backend": srcs["new"].get("backend", "cuda"),
        "ops": [o for o in srcs["new"].get("ops", []) if o in srcs["base"].get("ops", [])],
        "types": [t for t in srcs["new"].get("types", []) if t in srcs["base"].get("types", [])],
        "provenance": {
            "kind": KIND, "ref_label": lb, "new_label": ln, "warnings": warnings,
            "base_arm": base_arm, "new_arm": new_arm,
            "device": pn.get("device"), "driver": pn.get("driver"), "vendor_label": pn.get("vendor_label", ""),
            "base": {"campaign": srcs["base"]["name"], "path": paths["base"],
                     **{k: pb.get(k) for k in ("builds", "git_sha", "started", "device", "driver")}},
            "new": {"campaign": srcs["new"]["name"], "path": paths["new"],
                    **{k: pn.get(k) for k in ("builds", "git_sha", "started", "device", "driver")}},
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
    """Comparisons made before sources were paths store bare names, relative to the comparison's root."""
    cfg = camp.config
    return tuple(_open(p) if Path(p).is_absolute() else Campaign(camp.root, p) for p in (cfg["base"], cfg["new"]))


def _arms(cfg: dict) -> Tuple[str, str]:
    return cfg.get("base_arm", "batchlas"), cfg.get("new_arm", "batchlas")


def rows(camp: Campaign) -> List[dict]:
    """The paired cells only: a shape measured in one log has no ratio."""
    cfg = camp.config
    base, new = sources(camp)
    ab, an = _arms(cfg)
    pairs, _ = match(_latest_ok(base, ab), _latest_ok(new, an), cfg.get("exact", False))
    out = []
    for rb, rn in pairs:
        out.append({**rn, "arm": "batchlas"})
        out.append({**rb, "arm": "vendor", "t": rn.get("t", 0)})
    return out


def _geomean(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def control(camp: Campaign) -> dict:
    """How the pairing matched, and, when both sides are BatchLAS, the vendor
    arm's ratio between the two logs."""
    cfg = camp.config
    base, new = sources(camp)
    ab, an = _arms(cfg)
    _, stats = match(_latest_ok(base, ab), _latest_ok(new, an), cfg.get("exact", False))
    per_op: Dict[str, List[float]] = {}
    if ab == an == "batchlas":
        pairs, _ = match(_latest_ok(base, "vendor"), _latest_ok(new, "vendor"), cfg.get("exact", False))
        for rb, rn in pairs:
            t_new, t_base = float(rn["time_ms"]), float(rb["time_ms"])
            if t_new > 0 and t_base > 0:
                per_op.setdefault(rn["op"], []).append(t_base / t_new)
    allr = [x for v in per_op.values() for x in v]
    return {**stats,
            "vendor": {"geomean": _geomean(allr) if allr else None, "cells": len(allr),
                       "per_op": {o: {"geomean": _geomean(v), "cells": len(v)} for o, v in per_op.items()}}}


def data_key(camp: Campaign) -> tuple:
    """Changes whenever either source gains a row."""
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

    cfg = camp.config
    prov = cfg["provenance"]
    w = paired(camp.rows())
    lines = [f"{prov['new_label']} vs {prov['ref_label']}: speedup > 1x means {prov['new_label']} is faster"]
    for msg in prov.get("warnings", []):
        lines.append(f"  WARNING: {msg}")
    ctl = control(camp)
    lines.append(f"  {ctl['matched']} cells paired ({ctl['rescaled']} by time per matrix at unequal batches); "
                 f"shapes only in the candidate {ctl['only_new']}, only in the baseline {ctl['only_base']}")
    v = ctl["vendor"]
    if v["geomean"]:
        lines.append(f"  vendor control: {v['geomean']:.3f}x over {v['cells']} cells (1x = same machine state)")
    s = saturated(w) if not w.empty else w
    if s.empty:
        lines.append("  no shape measured in both logs")
        return "\n".join(lines)
    lines.append(f"  {'op':10s} {'dtype':8s} {'geomean':>8s} {'min':>7s} {'max':>7s}  cells")
    for op in [o for o in OPS if (s.op == o).any()]:
        for t in s[s.op == op].dtype.unique():
            sp = s[(s.op == op) & (s.dtype == t)].speedup.astype(float).dropna().to_numpy()
            if sp.size:
                lines.append(f"  {op:10s} {t:8s} {np.exp(np.log(sp).mean()):7.3f}x {sp.min():6.3f}x "
                             f"{sp.max():6.3f}x  {sp.size}")
    return "\n".join(lines)

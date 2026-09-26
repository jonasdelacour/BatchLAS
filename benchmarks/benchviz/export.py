"""A static, phone-friendly snapshot of one or more campaigns: one HTML page
plus downscaled figure PNGs, readable anywhere without the live server.

    python3 benchmarks/benchviz export smoke blas-smoke --out benchviz_runs/_export

`--fragment` writes the page without its <html>/<head>/<body> skeleton, for
hosts that wrap the page themselves (e.g. a claude.ai artifact).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import List

from ops import OPS, TYPES, Grid, vendor_name
from store import Campaign

FIG_WIDTH = 1200  # px; the dashboard PNGs are ~2000 px, far more than a phone needs


def _shrink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        im = Image.open(src)
        if im.width > FIG_WIDTH:
            im = im.resize((FIG_WIDTH, round(im.height * FIG_WIDTH / im.width)), Image.LANCZOS)
        im.save(dst, optimize=True)
    except ImportError:
        shutil.copyfile(src, dst)


def campaign_data(camp: Campaign, figdir: Path, rel: str) -> dict:
    import server  # reuses the dashboard's pairing and per-op statistics
    an = server._analysis(camp)
    cfg = camp.config
    backend = cfg.get("backend", "cuda")
    ops = []
    for op, st in an["per_op"].items():
        if op not in OPS:
            continue
        figs = {}
        for name in ("speedup_n", "throughput_n", "heatmap"):
            src = camp.figures / op / f"{name}.png"
            if src.exists():
                _shrink(src, figdir / op / f"{name}.png")
                figs[name] = f"{rel}/{op}/{name}.png"
        rows = server.op_table(camp.root, camp.name, op)["rows"]
        ops.append({"name": op, "title": OPS[op].title, "group": OPS[op].group, "notes": OPS[op].notes,
                    "vendor": vendor_name(op, backend), "stats": {k: st[k] for k in ("ok", "failed", "planned")},
                    "geomean": st.get("geomean"), "prec": st.get("prec", {}), "figs": figs, "rows": rows})
    ops.sort(key=lambda o: list(OPS).index(o["name"]))
    summaries = {}
    for t in TYPES:
        src = camp.figures / "_summary" / f"summary_{t}.png"
        if src.exists():
            _shrink(src, figdir / "_summary" / f"summary_{t}.png")
            summaries[t] = f"{rel}/_summary/summary_{t}.png"
    fails = [f for f in an["failures"] if f.get("op") in OPS]
    return {"name": camp.name, "provenance": cfg.get("provenance", {}), "grid": Grid.from_config(cfg).to_dict(),
            "status": camp.status().get("state", ""), "rows": an["rows"], "ops": ops,
            "summaries": summaries, "failures": fails[-100:]}


def export(root: Path, names: List[str], out: Path, fragment: bool = False) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    data = []
    for name in names:
        camp = Campaign(root, name)
        rel = f"figs/{name}"
        data.append(campaign_data(camp, out / rel, rel))
    page = TEMPLATE.replace("/*DATA*/null", json.dumps({"campaigns": data, "exported": time.strftime("%Y-%m-%d %H:%M")}))
    if not fragment:
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                "</head><body>" + page + "</body></html>")
    p = out / "index.html"
    p.write_text(page)
    return p


TEMPLATE = (Path(__file__).with_name("export_template.html")).read_text()

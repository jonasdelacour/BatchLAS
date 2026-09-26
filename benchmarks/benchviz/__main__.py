"""benchviz: BatchLAS vs vendor LAPACK -- run, plot, watch.

    python3 benchmarks/benchviz serve                      # dashboard on :8765
    python3 benchmarks/benchviz run  --ops potrf,syev --types float,double
    python3 benchmarks/benchviz plot <campaign>            # (re)render every figure
    python3 benchmarks/benchviz import <campaign> benchmarks/results/factor_baseline_*.csv
    python3 benchmarks/benchviz list-ops
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ops import OPS, PRESETS, TYPES, Grid, parse_list, plan_cells  # noqa: E402
from store import DEFAULT_ROOT, REPO, Campaign, provenance  # noqa: E402


def _csv_list(s: str, allowed) -> list:
    if s == "all":
        return list(allowed)
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise SystemExit(f"unknown: {', '.join(bad)} (choose from {', '.join(allowed)})")
    return out


def default_build_dirs() -> list:
    cands = [REPO / "build", REPO / "build" / "presets" / "benchmarks", REPO / "build" / "presets" / "dev-tests"]
    # A worktree's own build first, then the main checkout's.
    main = Path(os.environ.get("BATCHLAS_MAIN_CHECKOUT", REPO))
    if main != REPO:
        cands.append(main / "build")
    return [c for c in cands if c.exists()]


def cmd_list_ops(a):
    for name, s in OPS.items():
        print(f"{name:7s} {s.harness:13s} {s.binary:24s} types={','.join(s.types):28s} n={list(s.orders)}")
        if s.notes:
            print(f"        {s.notes}")


def grid_from_args(a) -> Grid:
    """Preset first, then --grid-json, then each explicit flag."""
    import json
    g = PRESETS[a.preset].to_dict()
    if getattr(a, "grid_json", None):
        g.update(json.loads(a.grid_json))
    for flag, key, conv in (("orders", "orders", parse_list), ("batches", "batches", parse_list),
                            ("batch_mode", "batch_mode", str), ("batch_min", "batch_min", int),
                            ("batch_max", "batch_max", int), ("batch_step", "batch_step", int),
                            ("reps", "reps", int), ("mem_gib", "mem_gib", float)):
        v = getattr(a, flag, None)
        if v is not None:
            g[key] = conv(v)
            g["name"] = "custom"
    if getattr(a, "batches", None) and not getattr(a, "batch_mode", None):
        g["batch_mode"] = "list"
    return Grid.from_dict(g)


def make_campaign(a) -> Campaign:
    ops = _csv_list(a.ops, OPS)
    types = _csv_list(a.types, TYPES)
    name = a.campaign or time.strftime(f"{a.backend}-%Y%m%d-%H%M%S")
    cfg = {
        "ops": ops, "types": types, "preset": a.grid.name, "backend": a.backend,
        "gpu": a.gpu, "grid": a.grid.to_dict(),
        "provenance": provenance(a.backend, a.gpu),
    }
    return Campaign.create(Path(a.root), name, cfg)


def cmd_run(a):
    from plots import render_op
    from runner import Runner

    a.grid = grid_from_args(a)
    cells = plan_cells(_csv_list(a.ops, OPS), _csv_list(a.types, TYPES), a.grid)
    if a.dry_run:
        print(f"grid: {a.grid.to_dict()}")
        print(f"{len(cells)} cells x 2 arms")
        for c in cells:
            print(f"  {c.op:6s} {c.dtype:8s} n={c.n:5d} batch={c.batch}")
        return
    camp = make_campaign(a)
    print(f"campaign {camp.name}: {len(cells)} cells x 2 arms -> {camp.dir}")
    build_dirs = [Path(p) for p in a.build_dir] if a.build_dir else default_build_dirs()
    # The dashboard tails run.log. A dashboard-started run already has its
    # stdout redirected there; a terminal run appends as well as printing.
    logf = None if os.environ.get("BENCHVIZ_STDOUT_IS_LOG") else open(camp.dir / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    runner = Runner(camp, build_dirs, a.gpu, guard=not a.no_guard, backend=a.backend, log=log)

    # Replot in the background as rows land, throttled per op: a usetex render
    # of three figures is a couple of seconds, a cell is several.
    stop = threading.Event()

    def replotter():
        last_rows, last_t = 0, 0.0
        while not stop.is_set():
            stop.wait(2.0)
            rows = camp.rows()
            if len(rows) != last_rows and (time.time() - last_t > a.replot_s or stop.is_set()):
                ops_touched = list(dict.fromkeys(r["op"] for r in rows[last_rows:]))
                last_rows, last_t = len(rows), time.time()
                for op in ops_touched:
                    try:
                        render_op(camp, op)
                    except Exception as e:  # a figure bug must not kill the campaign
                        log(f"  plot {op} failed: {e!r}")
                camp.set_status(camp.status().get("state", "running"), plotted=time.time())

    th = threading.Thread(target=replotter, daemon=True)
    if not a.no_plot:
        th.start()
    try:
        runner.run()
    finally:
        stop.set()
        if not a.no_plot:
            th.join(timeout=5)
            from plots import render_all
            render_all(camp)
            camp.set_status(camp.status().get("state", "finished"), plotted=time.time())
            print(f"figures: {camp.figures}")


def cmd_plot(a):
    from plots import render_all
    import style
    if a.no_tex:
        style.apply(usetex=False)
    camp = Campaign(Path(a.root), a.campaign)
    out = render_all(camp, ops=_csv_list(a.ops, OPS) if a.ops else None)
    for p in out:
        print(p.relative_to(camp.dir))
    camp.set_status(camp.status().get("state", "idle"), plotted=time.time())


def cmd_import(a):
    """Load factor_bench CSVs (run_factor_grid.sh output) into a campaign."""
    cfg = {"ops": [], "types": [], "preset": "imported", "backend": "cuda", "gpu": None,
           "grid": {**PRESETS["quick"].to_dict(), "name": "imported"}, "provenance": {**provenance("cuda", 1), "imported_from": a.files}}
    rows = []
    for f in a.files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r.get("arm") not in ("vendor", "native"):
                    continue
                route = r.get("resolved_route", "unknown")
                rec = dict(op=r["op"], dtype=r["type"], m=int(r["m"]), n=int(r["n"]), nrhs=int(r["nrhs"]),
                           batch=int(r["batch"]), arm="batchlas" if r["arm"] == "native" else "vendor",
                           backend="cuda", t=os.path.getmtime(f), time_ms=float(r["median_ms"]),
                           rel_sd=float(r["rel_sd"]), residual=float(r["residual"]),
                           ok=r["bad"] == "0", reason=r["reason"], route=route, subroutes=[],
                           vendor_free=not route.startswith("vendor"))
                if rec["ok"] and rec["arm"] == "batchlas" and route.startswith("vendor"):
                    rec.update(ok=False, reason=f"batchlas arm resolved to {route}")
                rows.append(rec)
                if rec["op"] not in cfg["ops"]:
                    cfg["ops"].append(rec["op"])
                if rec["dtype"] not in cfg["types"]:
                    cfg["types"].append(rec["dtype"])
    camp = Campaign.create(Path(a.root), a.campaign, cfg)
    for r in rows:
        camp.append(r)
    camp.set_status("imported", total=len(rows), done=len(rows))
    print(f"imported {len(rows)} rows into {camp.dir}")


def cmd_serve(a):
    import server
    server.serve(Path(a.root), a.host, a.port, default_build_dirs())


def main():
    p = argparse.ArgumentParser(prog="benchviz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=str(DEFAULT_ROOT), help="campaign directory (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common], help="measure a campaign (resumes if it exists)")
    r.add_argument("--campaign", help="name; default <backend>-<timestamp>")
    r.add_argument("--ops", default="all", help=f"comma list or 'all' ({','.join(OPS)})")
    r.add_argument("--types", default="all", help="comma list of float,double,cfloat,cdouble or 'all'")
    r.add_argument("--preset", default="quick", choices=list(PRESETS), help="starting grid; flags below override it")
    r.add_argument("--orders", help="n values: '16,32,64', '4:512' (doubling), '8:128:8' (step 8)")
    r.add_argument("--batch-mode", choices=["ladder", "list", "saturated"],
                   help="ladder: batch-min * step^k up to the cap; list: --batches; saturated: one batch per n at the cap")
    r.add_argument("--batches", help="explicit batch list (implies --batch-mode list), same syntax as --orders")
    r.add_argument("--batch-min", type=int)
    r.add_argument("--batch-max", type=int, help="upper cap on batch, before the memory cap")
    r.add_argument("--batch-step", type=int, help="ladder factor (2 = every power of two)")
    r.add_argument("--reps", type=int, help="timed repetitions per arm-cell")
    r.add_argument("--mem-gib", type=float, help="per-arm device-memory budget that caps batch")
    r.add_argument("--grid-json", help="a whole grid as JSON (what the dashboard sends)")
    r.add_argument("--backend", default="cuda", choices=["cuda", "rocm"])
    r.add_argument("--gpu", type=int, default=1, help="GPU index (default 1: the benchmark card)")
    r.add_argument("--build-dir", action="append", help="where the benchmark binaries are (repeatable)")
    r.add_argument("--no-guard", action="store_true", help="skip gpu_guard.sh (exclusive-GPU check)")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--replot-s", type=float, default=8.0)
    r.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    r.set_defaults(fn=cmd_run)

    pl = sub.add_parser("plot", parents=[common], help="render all figures of a campaign")
    pl.add_argument("campaign")
    pl.add_argument("--no-tex", action="store_true", help="mathtext instead of LaTeX")
    pl.add_argument("--ops", help="only these ops (comma list)")
    pl.set_defaults(fn=cmd_plot)

    im = sub.add_parser("import", parents=[common], help="import factor_bench / run_factor_grid.sh CSVs")
    im.add_argument("campaign")
    im.add_argument("files", nargs="+")
    im.set_defaults(fn=cmd_import)

    s = sub.add_parser("serve", parents=[common], help="the live dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)

    lo = sub.add_parser("list-ops", parents=[common])
    lo.set_defaults(fn=cmd_list_ops)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

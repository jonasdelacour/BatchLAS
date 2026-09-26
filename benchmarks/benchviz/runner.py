"""Run a campaign: one process per (cell, arm), results appended as JSONL.

ONE PROCESS PER ARM PER CELL, for the reasons run_factor_grid.sh gives: the SLM
carve-out attribute is sticky per CUfunction, so an earlier launch in the same
process can change whether a later one succeeds; and dispatch coverage is
emitted at exit, so one process is exactly one coverage file. Unlike that
script, coverage is on DURING the timed run rather than in a second untimed
run: resolve_route records once per op invocation behind a predicted branch
(coverage.hh), which is noise next to a batched LAPACK call, and it halves the
process count.
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from ops import OPS, Cell, Grid, plan_cells
from store import Campaign

GUARD_CONTAMINATED = 5


def find_binary(name: str, build_dirs: List[Path]) -> Optional[Path]:
    for d in build_dirs:
        for cand in (d / "benchmarks" / name, d / name):
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
    return None


def parse_coverage(path_base: str, op: str) -> Dict[str, object]:
    """The `reached` rows of every coverage file this process wrote.

    Returns the op's own route plus every sub-op route, because a native
    driver that calls cuBLAS gemm inside is native at the top and vendor
    underneath -- and a figure titled "vendor-free" must know the difference.
    """
    d = os.path.dirname(path_base)
    stem = os.path.basename(path_base)
    top, subs = set(), set()
    for f in os.listdir(d):
        if not f.startswith(stem + "."):
            continue
        with open(os.path.join(d, f)) as fh:
            for row in csv.reader(fh):
                if len(row) < 11 or row[0] != "reached":
                    continue
                route = f"{row[9]}:{row[10]}"
                (top if row[1] == op else subs).add(route if row[1] == op else f"{row[1]}={route}")
    return {
        "route": "|".join(sorted(top)) or "unknown",
        "subroutes": sorted(subs),
        "vendor_free": not any(r.startswith("vendor") for r in top)
        and not any("=vendor" in s for s in subs),
    }


def classify(rec: dict, op, arm_key: str) -> dict:
    """The gate the plots rely on: an arm whose route is not the one it is named
    for is not that library's measurement, whatever the pin said."""
    effective = [rec["route"]]
    if op.composed_of:
        mine = [s for s in rec.get("subroutes", []) if s.split("=", 1)[0] in op.composed_of]
        effective = [s.split("=", 1)[1] for s in mine] or ["unknown"]
        rec["route"] = "+".join(mine) or "unknown"
    want = "vendor" if arm_key == "vendor" else "native"
    stray = [r for r in effective if not r.startswith(want)]
    if rec["ok"] and stray:
        rec.update(ok=False, reason=f"{arm_key} arm resolved to {'|'.join(stray)}")
    return rec


class Runner:
    def __init__(self, camp: Campaign, build_dirs: List[Path], gpu: int, guard: bool,
                 backend: str, log=print):
        self.camp = camp
        self.build_dirs = build_dirs
        self.gpu = gpu
        self.guard = guard and shutil.which("nvidia-smi") is not None and backend == "cuda"
        self.backend = backend
        self.log = log
        self.repo = Path(__file__).resolve().parents[2]
        self._stop = False

    def stop(self):
        self._stop = True

    # ------------------------------------------------------------ one process
    def _exec(self, argv: List[str], env: Dict[str, str], timeout: float) -> subprocess.CompletedProcess:
        full_env = {**os.environ, **env}
        if self.guard:
            argv = [str(self.repo / "benchmarks" / "gpu_guard.sh"), str(self.gpu), *argv]
        elif self.backend == "cuda":
            full_env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        else:
            full_env["ROCR_VISIBLE_DEVICES"] = str(self.gpu)
        return subprocess.run(argv, env=full_env, capture_output=True, text=True, timeout=timeout)

    def _run_arm(self, cell: Cell, arm_key: str, reps: int, tmp: str) -> dict:
        op = OPS[cell.op]
        arm = next(a for a in op.arms if a.key == arm_key)
        binary = find_binary(op.binary, self.build_dirs)
        base = dict(asdict(cell), arm=arm_key, backend=self.backend, t=time.time())
        if binary is None:
            return {**base, "ok": False, "reason": f"binary {op.binary} not built"}
        csv_path = os.path.join(tmp, "out.csv")
        cov = os.path.join(tmp, "cov")
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        env = dict(arm.env)
        env["BATCHLAS_COVERAGE_OUT"] = cov
        if op.harness == "factor_bench":
            argv = [str(binary), cell.op, cell.dtype, str(cell.m), str(cell.n), str(cell.nrhs),
                    str(cell.batch), str(reps), f"--arms={arm.fb_arm}", f"--csv={csv_path}"]
        else:
            argv = [str(binary), f"--backend={self.backend.upper()}", f"--type={cell.dtype}",
                    f"--name={arm.bench_name}", "--warmup=3", f"--min_iters={reps}",
                    f"--max_iters={max(reps, 20)}", "--min_time=200", f"--csv={csv_path}",
                    *map(str, op.cell_args(cell.n, cell.batch))]
        for attempt in range(3):
            t0 = time.time()
            try:
                p = self._exec(argv, env, timeout=900)
            except subprocess.TimeoutExpired:
                return {**base, "ok": False, "reason": "timeout (900 s)"}
            if p.returncode == GUARD_CONTAMINATED:
                self.log(f"  GPU {self.gpu} contaminated during the run; retrying ({attempt + 1}/3)")
                continue
            break
        else:
            return {**base, "ok": False, "reason": "GPU never exclusive"}
        wall = time.time() - t0
        rows = []
        if os.path.exists(csv_path):
            with open(csv_path) as fh:
                rows = list(csv.DictReader(fh))
        if not rows:
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
            return {**base, "ok": False, "wall_s": wall,
                    "reason": f"no output (rc={p.returncode}): " + " / ".join(tail)[:300]}
        rec = {**base, "wall_s": wall}
        if op.harness == "factor_bench":
            r = rows[-1]
            rec.update(time_ms=float(r["median_ms"]), rel_sd=float(r["rel_sd"]),
                       residual=float(r["residual"]), ok=r["bad"] == "0", reason=r["reason"])
        else:
            if len(rows) != 1:
                return {**rec, "ok": False, "reason": f"--name matched {len(rows)} benchmarks, need 1"}
            r = rows[0]
            avg, sd = float(r["avg_ms"]), float(r["stddev_ms"])
            rec.update(time_ms=avg, rel_sd=sd / avg if avg > 0 else 0.0,
                       residual=None, ok=True, reason="ok (timing only; not verified)")
        if arm.fixed_route:
            rec.update(route=arm.fixed_route, subroutes=[], vendor_free=arm.key != "vendor")
        else:
            rec.update(parse_coverage(cov, cell.op))
            if op.setup_ops:
                rec["vendor_free"] = None
        return classify(rec, op, arm_key)

    # ------------------------------------------------------------ campaign
    def run(self):
        cfg = self.camp.config
        grid = Grid.from_config(cfg)
        req = cfg.get("request", cfg)
        cells = plan_cells(req["ops"], req["types"], grid)
        done = self.camp.done_keys()
        todo = [(c, a.key) for c in cells for a in OPS[c.op].arms if c.key(a.key) not in done]
        total = len(cells) * 2
        self.camp.set_status("running", total=total, done=total - len(todo), pid=os.getpid())
        self.log(f"campaign {self.camp.name}: {len(todo)} of {total} arm-cells to run")
        tmp = tempfile.mkdtemp(prefix="benchviz_")
        failed_ops = {}
        try:
            for i, (cell, arm) in enumerate(todo):
                if self._stop or self.camp.stop_requested():
                    self.camp.set_status("stopped")
                    self.log("stopped on request")
                    return
                self.camp.set_status("running", current=f"{cell.op} {cell.dtype} n={cell.n} "
                                     f"batch={cell.batch} [{arm}]", done=total - len(todo) + i)
                # An op whose binary is missing would fail every cell the same way.
                if failed_ops.get(cell.op, "").startswith("binary"):
                    rec = dict(asdict(cell), arm=arm, ok=False, reason=failed_ops[cell.op])
                else:
                    rec = self._run_arm(cell, arm, grid.reps, tmp)
                    # factor_bench's noise gate: re-measure rather than leave a hole.
                    for _ in range(2):
                        if rec.get("ok") or "relsd" not in str(rec.get("reason")):
                            break
                        self.log(f"  noisy ({rec['reason']}); re-measuring")
                        rec = self._run_arm(cell, arm, grid.reps, tmp)
                    if not rec["ok"] and rec["reason"].startswith("binary"):
                        failed_ops[cell.op] = rec["reason"]
                self.camp.append(rec)
                status = f"{rec['time_ms']:.4f} ms {rec.get('route', '')}" if rec.get("ok") else rec["reason"]
                self.log(f"[{total - len(todo) + i + 1}/{total}] {cell.op} {cell.dtype} "
                         f"n={cell.n} batch={cell.batch} {arm}: {status}")
            self.camp.set_status("finished", done=total, current="")
        except Exception as e:  # surface in the dashboard rather than die silently
            self.camp.set_status("failed", error=repr(e))
            raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

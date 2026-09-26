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
import signal
import tempfile
import threading
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


def detect_gpus(backend: str = "cuda") -> List[dict]:
    """The GPUs this box actually has: [{index, name}]. Empty when it cannot tell."""
    if backend != "cuda" or not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        idx, _, name = line.partition(",")
        if idx.strip().isdigit():
            gpus.append({"index": int(idx), "name": name.strip()})
    return gpus


class Stopped(Exception):
    pass


class Runner:
    """Runs a campaign on one or more GPUs. With several, each GPU gets its own
    worker pulling whole cells from one shared queue: both arms of a cell always
    run on the same card, so every ratio compares like with like."""

    def __init__(self, camp: Campaign, build_dirs: List[Path], gpus, guard: bool,
                 backend: str, log=print):
        self.camp = camp
        self.build_dirs = build_dirs
        self.gpus = [int(g) for g in (gpus if isinstance(gpus, (list, tuple)) else [gpus])]
        self.guard = guard and shutil.which("nvidia-smi") is not None and backend == "cuda"
        self.backend = backend
        self.log = log
        self.repo = Path(__file__).resolve().parents[2]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        known = detect_gpus(backend)
        if known:
            bad = [g for g in self.gpus if g not in {k["index"] for k in known}]
            if bad:
                raise ValueError(f"GPU {bad} does not exist; this box has "
                                 + ", ".join(f"{k['index']} ({k['name']})" for k in known))

    def stop(self):
        self._stop.set()

    def stopping(self) -> bool:
        if not self._stop.is_set() and self.camp.stop_requested():
            self._stop.set()
        return self._stop.is_set()

    # ------------------------------------------------------------ one process
    def _exec(self, gpu: int, argv: List[str], env: Dict[str, str], timeout: float):
        """One benchmark process in its own process group, polled for a stop
        request so Stop takes effect within half a second, mid-cell."""
        full_env = {**os.environ, **env}
        if self.guard:
            argv = [str(self.repo / "benchmarks" / "gpu_guard.sh"), str(gpu), *argv]
        elif self.backend == "cuda":
            full_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            full_env["ROCR_VISIBLE_DEVICES"] = str(gpu)
        p = subprocess.Popen(argv, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
        t0 = time.time()
        while p.poll() is None:
            if self.stopping() or time.time() - t0 > timeout:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(3)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                p.wait()
                if self._stop.is_set():
                    raise Stopped()
                raise subprocess.TimeoutExpired(argv, timeout)
            time.sleep(0.5)
        out, err = p.communicate()
        return subprocess.CompletedProcess(argv, p.returncode, out, err)

    def _run_arm(self, gpu: int, cell: Cell, arm_key: str, reps: int, tmp: str) -> dict:
        op = OPS[cell.op]
        arm = next(a for a in op.arms if a.key == arm_key)
        binary = find_binary(op.binary, self.build_dirs)
        base = dict(asdict(cell), arm=arm_key, backend=self.backend, gpu=gpu, t=time.time())
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
                p = self._exec(gpu, argv, env, timeout=900)
            except subprocess.TimeoutExpired:
                return {**base, "ok": False, "reason": "timeout (900 s)"}
            if p.returncode == GUARD_CONTAMINATED:
                self.log(f"  GPU {gpu} contaminated during the run; retrying ({attempt + 1}/3)")
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
    def _measure(self, gpu: int, cell: Cell, arm: str, reps: int, tmp: str) -> dict:
        with self._lock:
            missing = self._failed_ops.get(cell.op, "")
        if missing:  # an op whose binary is missing fails every cell the same way
            return dict(asdict(cell), arm=arm, gpu=gpu, ok=False, reason=missing)
        rec = self._run_arm(gpu, cell, arm, reps, tmp)
        for _ in range(2):  # factor_bench's noise gate: re-measure rather than leave a hole
            if rec.get("ok") or "relsd" not in str(rec.get("reason")):
                break
            self.log(f"  GPU {gpu}: noisy ({rec['reason']}); re-measuring")
            rec = self._run_arm(gpu, cell, arm, reps, tmp)
        if not rec["ok"] and rec["reason"].startswith("binary"):
            with self._lock:
                self._failed_ops[cell.op] = rec["reason"]
        return rec

    def _worker(self, gpu: int, reps: int):
        tmp = tempfile.mkdtemp(prefix=f"benchviz_gpu{gpu}_")
        try:
            while not self.stopping():
                with self._lock:
                    if not self._queue:
                        return
                    cell, arms = self._queue.pop(0)
                for arm in arms:
                    with self._lock:
                        self._current[gpu] = f"{cell.op} {cell.dtype} n={cell.n} batch={cell.batch} [{arm}]"
                        self._status()
                    rec = self._measure(gpu, cell, arm, reps, tmp)
                    with self._lock:
                        self.camp.append(rec)
                        self._done += 1
                        n = self._done
                    status = f"{rec['time_ms']:.4f} ms {rec.get('route', '')}" if rec.get("ok") else rec["reason"]
                    self.log(f"[{n}/{self._total}] GPU {gpu} {cell.op} {cell.dtype} n={cell.n} "
                             f"batch={cell.batch} {arm}: {status}")
        except Stopped:
            pass
        except Exception as e:  # surface in the dashboard rather than die silently
            self._errors.append(f"GPU {gpu}: {e!r}")
            self._stop.set()
        finally:
            with self._lock:
                self._current.pop(gpu, None)
            shutil.rmtree(tmp, ignore_errors=True)

    def _status(self, state: str = "running", **kw):
        if len(self.gpus) > 1:
            cur = "  |  ".join(f"GPU {g}: {c}" for g, c in sorted(self._current.items()))
        else:
            cur = next(iter(self._current.values()), "")
        self.camp.set_status(state, done=self._done, total=self._total, current=cur, gpus=self.gpus, **kw)

    def run(self):
        cfg = self.camp.config
        grid = Grid.from_config(cfg)
        req = cfg.get("request", cfg)
        cells = plan_cells(req["ops"], req["types"], grid)
        done = self.camp.done_keys()
        self._queue = []
        for c in cells:
            arms = [a.key for a in OPS[c.op].arms if c.key(a.key) not in done]
            if arms:
                self._queue.append((c, arms))
        self._total = len(cells) * 2
        self._done = self._total - sum(len(a) for _, a in self._queue)
        self._current, self._failed_ops, self._errors = {}, {}, []
        self.camp.set_status("running", total=self._total, done=self._done, pid=os.getpid(),
                             gpus=self.gpus, error=None, current="")
        self.log(f"campaign {self.camp.name}: {self._total - self._done} of {self._total} arm-cells to run "
                 f"on GPU {', '.join(map(str, self.gpus))}")
        threads = [threading.Thread(target=self._worker, args=(g, grid.reps), daemon=True) for g in self.gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with self._lock:
            self._current.clear()
            if self._errors:
                self._status("failed", error="; ".join(self._errors))
                raise RuntimeError(self._errors[0])
            if self._stop.is_set() and self._queue:
                self._status("stopped")
                self.log("stopped on request")
            else:
                self._status("finished")

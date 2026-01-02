#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Case:
    bench: str  # "stedc" | "steqr" | "steqr_cta"
    args: List[int]


@dataclass(frozen=True)
class MeasurementKey:
    bench: str
    backend: str
    dtype: str
    args: Tuple[int, ...]
    metric: str


@dataclass
class Measurement:
    key: MeasurementKey
    value: float
    avg_ms: float
    stddev_ms: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_build_dir(repo_root: Path) -> Path:
    return repo_root / "build"


def _default_benchmark_path(build_dir: Path, exe_name: str) -> Path:
    return build_dir / "benchmarks" / exe_name


def _run_cmd(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _now_us() -> float:
    # Chrome trace expects microseconds.
    return time.perf_counter_ns() / 1000.0


def _append_trace_event(
    trace_events: Optional[List[Dict[str, Any]]],
    *,
    name: str,
    cat: str,
    ts_us: float,
    dur_us: float,
    args: Optional[Dict[str, Any]] = None,
    pid: int = 1,
    tid: int = 1,
) -> None:
    if trace_events is None:
        return
    trace_events.append(
        {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": ts_us,
            "dur": dur_us,
            "pid": pid,
            "tid": tid,
            "args": args or {},
        }
    )


def _guess_backend_and_type_from_name(name: str) -> Tuple[str, str]:
    # Name looks like: "(BM_STEQR<float, batchlas::Backend::CUDA>)"
    backend = "UNKNOWN"
    for b in ("CUDA", "ROCM", "MKL", "NETLIB"):
        if f"Backend::{b}" in name:
            backend = b
            break

    dtype = "unknown"
    if "<float" in name or "float," in name:
        dtype = "float"
    elif "<double" in name or "double," in name:
        dtype = "double"
    elif "complex<float>" in name:
        dtype = "complex<float>"
    elif "complex<double>" in name:
        dtype = "complex<double>"

    return backend, dtype


def _metric_is_higher_better(metric: str) -> bool:
    # Keep this simple and explicit.
    m = metric.strip().lower()
    if "gflops" in m or "throughput" in m:
        return True
    return False


def _metric_is_lower_better(metric: str) -> bool:
    return not _metric_is_higher_better(metric)


def _parse_minibench_csv(csv_path: Path, *, expected_metric: str, bench: str) -> List[Measurement]:
    out: List[Measurement] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        # Confirm metric exists (helpful error)
        if expected_metric not in reader.fieldnames:
            raise RuntimeError(
                f"Expected metric column '{expected_metric}' not found in {csv_path}. "
                f"Columns: {', '.join(reader.fieldnames)}"
            )

        for row in reader:
            name = (row.get("name") or "").strip().strip('"')
            backend, dtype = _guess_backend_and_type_from_name(name)

            # Collect args arg0..argN (ignore trailing empties)
            args: List[int] = []
            i = 0
            while True:
                k = f"arg{i}"
                if k not in row:
                    break
                v = (row.get(k) or "").strip()
                if v == "":
                    break
                args.append(int(v))
                i += 1

            metric_val = float(row[expected_metric])
            avg_ms = float(row.get("avg_ms") or 0.0)
            stddev_ms = float(row.get("stddev_ms") or 0.0)

            out.append(
                Measurement(
                    key=MeasurementKey(
                        bench=bench,
                        backend=backend,
                        dtype=dtype,
                        args=tuple(args),
                        metric=expected_metric,
                    ),
                    value=metric_val,
                    avg_ms=avg_ms,
                    stddev_ms=stddev_ms,
                )
            )

    return out


def _run_one_benchmark(
    *,
    exe: Path,
    bench: str,
    cases: List[Case],
    backend: str,
    dtype: str,
    warmup: int,
    min_time_ms: float,
    min_iters: int,
    max_iters: int,
    trace_events: Optional[List[Dict[str, Any]]] = None,
    kernel_trace_dir: Optional[Path] = None,
) -> List[Measurement]:
    matching_cases = [c for c in cases if c.bench == bench]
    if not matching_cases:
        return []

    if not exe.exists():
        raise FileNotFoundError(
            f"Benchmark executable not found: {exe}. "
            f"Build first (e.g., cmake --build build -j)."
        )
    if not os.access(exe, os.X_OK):
        raise PermissionError(f"Benchmark executable is not runnable: {exe}")

    expected_metric = {
        "stedc": "Time (µs) / Batch",
        "steqr": "T(µs)/Batch",
        "steqr_cta": "T(µs)/Batch",
    }[bench]

    measurements: List[Measurement] = []

    for case in matching_cases:

        with tempfile.TemporaryDirectory(prefix="batchlas-eval-") as td:
            csv_path = Path(td) / f"{bench}.csv"

            cmd = [
                str(exe),
                f"--backend={backend}",
                f"--type={dtype}",
                f"--csv={csv_path}",
                f"--warmup={warmup}",
                f"--min_time={min_time_ms}",
                f"--min_iters={min_iters}",
                f"--max_iters={max_iters}",
            ] + [str(x) for x in case.args]

            child_env: Optional[Dict[str, str]] = None
            kernel_trace_path: Optional[Path] = None
            if kernel_trace_dir is not None:
                kernel_trace_dir.mkdir(parents=True, exist_ok=True)
                safe_args = "_".join(str(a) for a in case.args)
                kernel_trace_path = kernel_trace_dir / f"kernels_{bench}_{backend}_{dtype}_{safe_args}.trace.json"
                child_env = dict(os.environ)
                child_env["BATCHLAS_KERNEL_TRACE"] = "1"
                child_env["BATCHLAS_KERNEL_TRACE_PATH"] = str(kernel_trace_path)

            t0 = _now_us()
            _run_cmd(cmd, env=child_env)
            t1 = _now_us()
            _append_trace_event(
                trace_events,
                name=f"{bench}_benchmark",
                cat="minibench",
                ts_us=t0,
                dur_us=(t1 - t0),
                tid=1 if bench == "stedc" else 2,
                args={
                    "exe": str(exe),
                    "bench": bench,
                    "backend": backend,
                    "type": dtype,
                    "case_args": list(case.args),
                    "cmd": cmd,
                    "kernel_trace": str(kernel_trace_path) if kernel_trace_path else "",
                },
            )

            parsed = _parse_minibench_csv(csv_path, expected_metric=expected_metric, bench=bench)
            # With explicit ARGS we expect exactly one result row.
            if len(parsed) != 1:
                raise RuntimeError(
                    f"Expected 1 CSV row for {bench} args={case.args}, got {len(parsed)}"
                )

            measurements.extend(parsed)

    return measurements


def _load_cases(path: Path) -> List[Case]:
    data = json.loads(path.read_text())
    cases: List[Case] = []
    for item in data.get("cases", []):
        bench = item["bench"]
        args = list(item["args"])
        cases.append(Case(bench=bench, args=args))
    return cases


def _save_results_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save_chrome_trace(path: Path, trace_events: List[Dict[str, Any]], *, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "metadata": meta,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _measurements_to_payload(measurements: List[Measurement], *, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "meta": meta,
        "results": [
            {
                "bench": m.key.bench,
                "backend": m.key.backend,
                "type": m.key.dtype,
                "args": list(m.key.args),
                "metric": m.key.metric,
                "value": m.value,
                "avg_ms": m.avg_ms,
                "stddev_ms": m.stddev_ms,
            }
            for m in measurements
        ],
    }


def _payload_to_measurements(payload: Dict[str, Any]) -> List[Measurement]:
    out: List[Measurement] = []
    for r in payload.get("results", []):
        key = MeasurementKey(
            bench=r["bench"],
            backend=r["backend"],
            dtype=r["type"],
            args=tuple(r["args"]),
            metric=r["metric"],
        )
        out.append(
            Measurement(
                key=key,
                value=float(r["value"]),
                avg_ms=float(r.get("avg_ms", 0.0)),
                stddev_ms=float(r.get("stddev_ms", 0.0)),
            )
        )
    return out


def _index_measurements(measurements: Iterable[Measurement]) -> Dict[MeasurementKey, Measurement]:
    return {m.key: m for m in measurements}


def _compare(
    *,
    baseline: List[Measurement],
    current: List[Measurement],
    tolerance: float,
) -> Tuple[bool, List[str]]:
    base = _index_measurements(baseline)
    cur = _index_measurements(current)

    ok = True
    lines: List[str] = []

    missing_in_current = [k for k in base.keys() if k not in cur]
    extra_in_current = [k for k in cur.keys() if k not in base]

    if missing_in_current:
        ok = False
        lines.append(f"Missing {len(missing_in_current)} baseline measurement(s) in current run")
        for k in missing_in_current[:20]:
            lines.append(f"  - missing: {k}")
        if len(missing_in_current) > 20:
            lines.append("  - ...")

    if extra_in_current:
        lines.append(f"Note: {len(extra_in_current)} extra measurement(s) not present in baseline")

    for k, b in base.items():
        c = cur.get(k)
        if c is None:
            continue

        if b.value <= 0:
            # Avoid divide-by-zero / nonsense comparisons.
            continue

        ratio = c.value / b.value
        pct = (ratio - 1.0) * 100.0

        if _metric_is_lower_better(k.metric):
            regressed = ratio > (1.0 + tolerance)
            direction = "higher (worse)"
        else:
            regressed = ratio < (1.0 - tolerance)
            direction = "lower (worse)"

        if regressed:
            ok = False
            lines.append(
                f"REGRESSION {k.bench} {k.backend} {k.dtype} args={list(k.args)} metric='{k.metric}': "
                f"baseline={b.value:.6g}, current={c.value:.6g} ({pct:+.2f}%, {direction})"
            )

    return ok, lines


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="BatchLAS perf eval (CUDA FP32, STEDC/STEQR/STEQR_CTA)")
    p.add_argument("--build-dir", default="", help="Build directory (default: repo_root/build)")
    p.add_argument(
        "--cases",
        default="",
        help="Path to cases JSON (default: evaluation/perf_cases.json)",
    )
    p.add_argument(
        "--baseline",
        default="evaluation/baselines/perf_cuda_fp32.json",
        help="Baseline JSON path",
    )
    p.add_argument("--record", action="store_true", help="Record baseline (overwrite baseline file)")
    p.add_argument(
        "--check",
        action="store_true",
        help="Compare current vs baseline and exit nonzero on regression",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed relative regression (default: 0.05 = 5%%)",
    )
    p.add_argument(
        "--trace",
        default="",
        help="Write a Chrome trace JSON file for the run (wall-time around each benchmark invocation)",
    )
    p.add_argument(
        "--kernel-trace-dir",
        default="",
        help=(
            "Enable BatchLAS SYCL event profiling and write per-case kernel traces into this directory "
            "(Chrome trace JSON)."
        ),
    )

    # Benchmark knobs
    p.add_argument("--backend", default="CUDA", help="minibench backend filter (default CUDA)")
    p.add_argument("--type", dest="dtype", default="float", help="minibench type filter (default float)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--min-time", type=float, default=200.0)
    p.add_argument("--min-iters", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=100)

    args = p.parse_args(argv)

    if not args.record and not args.check:
        p.error("Must specify --record or --check")

    repo_root = _repo_root()
    build_dir = Path(args.build_dir) if args.build_dir else _default_build_dir(repo_root)
    cases_path = Path(args.cases) if args.cases else (repo_root / "evaluation" / "perf_cases.json")
    baseline_path = (repo_root / args.baseline).resolve() if not Path(args.baseline).is_absolute() else Path(args.baseline)

    cases = _load_cases(cases_path)

    stedc_exe = _default_benchmark_path(build_dir, "stedc_benchmark")
    steqr_exe = _default_benchmark_path(build_dir, "steqr_benchmark")
    steqr_cta_exe = _default_benchmark_path(build_dir, "steqr_cta_benchmark")

    measurements: List[Measurement] = []
    trace_events: Optional[List[Dict[str, Any]]] = [] if args.trace else None
    kernel_trace_dir = Path(args.kernel_trace_dir) if args.kernel_trace_dir else None
    measurements += _run_one_benchmark(
        exe=stedc_exe,
        bench="stedc",
        cases=cases,
        backend=args.backend,
        dtype=args.dtype,
        warmup=args.warmup,
        min_time_ms=args.min_time,
        min_iters=args.min_iters,
        max_iters=args.max_iters,
        trace_events=trace_events,
        kernel_trace_dir=kernel_trace_dir,
    )
    measurements += _run_one_benchmark(
        exe=steqr_exe,
        bench="steqr",
        cases=cases,
        backend=args.backend,
        dtype=args.dtype,
        warmup=args.warmup,
        min_time_ms=args.min_time,
        min_iters=args.min_iters,
        max_iters=args.max_iters,
        trace_events=trace_events,
        kernel_trace_dir=kernel_trace_dir,
    )
    measurements += _run_one_benchmark(
        exe=steqr_cta_exe,
        bench="steqr_cta",
        cases=cases,
        backend=args.backend,
        dtype=args.dtype,
        warmup=args.warmup,
        min_time_ms=args.min_time,
        min_iters=args.min_iters,
        max_iters=args.max_iters,
        trace_events=trace_events,
        kernel_trace_dir=kernel_trace_dir,
    )

    meta = {
        "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "build_dir": str(build_dir),
        "cases": str(cases_path),
        "backend": args.backend,
        "type": args.dtype,
        "minibench": {
            "warmup": args.warmup,
            "min_time_ms": args.min_time,
            "min_iters": args.min_iters,
            "max_iters": args.max_iters,
        },
    }

    payload = _measurements_to_payload(measurements, meta=meta)

    if args.trace and trace_events is not None:
        _save_chrome_trace(Path(args.trace), trace_events, meta=meta)
        print(f"Wrote trace: {args.trace}")

    if args.record:
        _save_results_json(baseline_path, payload)
        print(f"Wrote baseline: {baseline_path}")
        return 0

    # --check
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}", file=sys.stderr)
        print("Run with --record to create one.", file=sys.stderr)
        return 2

    baseline_payload = json.loads(baseline_path.read_text())
    baseline_measurements = _payload_to_measurements(baseline_payload)

    ok, lines = _compare(baseline=baseline_measurements, current=measurements, tolerance=args.tolerance)

    if ok:
        print("OK: no performance regressions detected")
        return 0

    print("FAIL: performance regressions detected", file=sys.stderr)
    for ln in lines:
        print(ln, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

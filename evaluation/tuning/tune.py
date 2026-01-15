#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import itertools
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BenchSpace:
    bench: str
    exe: str
    metric: str
    direction: str  # "min" | "max"
    arg_spec: List[str]
    cases: List[Dict[str, Any]]  # each has {"fixed": {..}, "tune": {..}}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_build_dir(repo_root: Path) -> Path:
    return repo_root / "build"


def _default_benchmark_path(build_dir: Path, exe_name: str) -> Path:
    return build_dir / "benchmarks" / exe_name


def _run_cmd(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _parse_minibench_csv(csv_path: Path, *, expected_metric: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        if expected_metric not in reader.fieldnames:
            raise RuntimeError(
                f"Expected metric column '{expected_metric}' not found in {csv_path}. "
                f"Columns: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            name = (row.get("name") or "").strip().strip('"')
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

            rows.append(
                {
                    "name": name,
                    "args": args,
                    "metric": expected_metric,
                    "value": float(row[expected_metric]),
                    "avg_ms": float(row.get("avg_ms") or 0.0),
                    "stddev_ms": float(row.get("stddev_ms") or 0.0),
                }
            )

    return rows


def _score(values: Sequence[float], direction: str) -> float:
    # Aggregate across cases: simple mean.
    m = sum(values) / max(1, len(values))
    return m if direction == "min" else -m


def _iter_grid(tune: Dict[str, List[int]]) -> Iterable[Dict[str, int]]:
    keys = list(tune.keys())
    if not keys:
        yield {}
        return
    value_lists = [tune[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield {k: int(v) for k, v in zip(keys, combo)}


def _args_from_spec(arg_spec: List[str], fixed: Dict[str, int], params: Dict[str, int]) -> List[int]:
    out: List[int] = []
    for name in arg_spec:
        if name in params:
            out.append(int(params[name]))
        elif name in fixed:
            out.append(int(fixed[name]))
        else:
            raise KeyError(f"Missing value for arg '{name}' (spec={arg_spec}, fixed={fixed}, params={params})")
    return out


def _run_one_case(
    *,
    exe: Path,
    backend: str,
    dtype: str,
    metric: str,
    warmup: int,
    min_time_ms: float,
    min_iters: int,
    max_iters: int,
    args: List[int],
) -> float:
    with tempfile.TemporaryDirectory(prefix="batchlas-tune-") as td:
        csv_path = Path(td) / "out.csv"
        cmd = [
            str(exe),
            f"--backend={backend}",
            f"--type={dtype}",
            f"--csv={csv_path}",
            f"--warmup={warmup}",
            f"--min_time={min_time_ms}",
            f"--min_iters={min_iters}",
            f"--max_iters={max_iters}",
        ] + [str(x) for x in args]

        _run_cmd(cmd)
        rows = _parse_minibench_csv(csv_path, expected_metric=metric)
        if not rows:
            raise RuntimeError(
                f"No rows produced by {exe} for backend={backend} type={dtype} args={args}. "
                "This usually means the benchmark wasn't registered for that backend/type (compiled out)"
            )

        # Usually there is exactly one row after backend/type filtering.
        # If multiple appear, choose the first.
        return float(rows[0]["value"])


def _load_spaces(path: Path) -> List[BenchSpace]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "spaces" not in raw:
        raise RuntimeError(f"Invalid tuning space file (expected object with 'spaces'): {path}")

    spaces: List[BenchSpace] = []
    for s in raw["spaces"]:
        spaces.append(
            BenchSpace(
                bench=str(s["bench"]),
                exe=str(s.get("exe") or f"{s['bench']}_benchmark"),
                metric=str(s["metric"]),
                direction=str(s.get("direction") or "min"),
                arg_spec=list(s["arg_spec"]),
                cases=list(s["cases"]),
            )
        )
    return spaces


def _tune_one_bench(
    *,
    space: BenchSpace,
    exe: Path,
    backend: str,
    dtype: str,
    warmup: int,
    min_time_ms: float,
    min_iters: int,
    max_iters: int,
    topk: int,
    verbose: bool,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    leaderboard: List[Dict[str, Any]] = []

    # For now, we assume all cases share the same tuning keys.
    tune_keys: Dict[str, List[int]] = {}
    for case in space.cases:
        for k, vs in (case.get("tune") or {}).items():
            if k not in tune_keys:
                tune_keys[k] = list(vs)
            else:
                # If repeated, keep intersection if possible; otherwise just keep the union.
                existing = set(tune_keys[k])
                new = set(vs)
                inter = existing.intersection(new)
                tune_keys[k] = sorted(inter) if inter else sorted(existing.union(new))

    for params in _iter_grid(tune_keys):
        values: List[float] = []
        per_case: List[Dict[str, Any]] = []

        for case in space.cases:
            fixed = {k: int(v) for k, v in (case.get("fixed") or {}).items()}
            args = _args_from_spec(space.arg_spec, fixed=fixed, params=params)
            v = _run_one_case(
                exe=exe,
                backend=backend,
                dtype=dtype,
                metric=space.metric,
                warmup=warmup,
                min_time_ms=min_time_ms,
                min_iters=min_iters,
                max_iters=max_iters,
                args=args,
            )
            values.append(v)
            per_case.append({"fixed": fixed, "args": args, "value": v})

        s = _score(values, space.direction)
        entry = {"params": params, "score": s, "values": values, "cases": per_case}

        if verbose:
            # Print minimal progress; keep stdout readable.
            pretty = ", ".join(f"{k}={v}" for k, v in params.items())
            avg = sum(values) / max(1, len(values))
            print(f"[{space.bench}] {pretty} -> avg={avg:.6g} ({space.metric})")

        leaderboard.append(entry)
        leaderboard.sort(key=lambda e: e["score"])  # lower score is better
        if len(leaderboard) > max(1, topk):
            leaderboard = leaderboard[:topk]

        if best is None or entry["score"] < best["score"]:
            best = entry

    assert best is not None
    return {
        "bench": space.bench,
        "exe": space.exe,
        "metric": space.metric,
        "direction": space.direction,
        "arg_spec": space.arg_spec,
        "best": best,
        "top": leaderboard,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BatchLAS bottom-up tuning harness (grid search)")
    parser.add_argument("--build-dir", type=Path, default=None, help="Build directory (default: <repo>/build)")
    parser.add_argument("--space", type=Path, required=True, help="Path to tuning space JSON")
    parser.add_argument("--backend", type=str, required=True, help="Backend passed to benchmarks (e.g., CUDA, ROCM, NETLIB, MKL)")
    parser.add_argument("--type", dest="dtype", type=str, required=True, help="Type passed to benchmarks (e.g., float, double)")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON profile path")

    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-time", type=float, default=25.0)
    parser.add_argument("--min-iters", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=20)

    parser.add_argument("--topk", type=int, default=10, help="Keep top-K candidates per bench")
    parser.add_argument("--skip-missing", action="store_true", help="Skip benches whose executables are missing")
    parser.add_argument("--skip-failed", action="store_true", help="Skip benches that fail to run (e.g., unsupported backend)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    repo_root = _repo_root()
    build_dir = args.build_dir or _default_build_dir(repo_root)

    spaces = _load_spaces(args.space)

    results: List[Dict[str, Any]] = []
    for space in spaces:
        exe = _default_benchmark_path(build_dir, space.exe)
        if not exe.exists() or not os.access(exe, os.X_OK):
            msg = f"Missing benchmark executable: {exe}"
            if args.skip_missing:
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)

        print(f"Tuning {space.bench} via {exe.name}...")
        try:
            r = _tune_one_bench(
                space=space,
                exe=exe,
                backend=args.backend,
                dtype=args.dtype,
                warmup=args.warmup,
                min_time_ms=args.min_time,
                min_iters=args.min_iters,
                max_iters=args.max_iters,
                topk=args.topk,
                verbose=args.verbose,
            )
            results.append(r)
        except Exception as e:
            if args.skip_failed:
                print(f"[skip] {space.bench} failed: {e}")
                continue
            raise

    profile = {
        "meta": {
            "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "backend": args.backend,
            "dtype": args.dtype,
            "build_dir": str(build_dir),
        },
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(f"Wrote tuning profile: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared minibench process and CSV plumbing for evaluation scripts."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_build_dir(repo_root: Path) -> Path:
    return repo_root / "build"


def default_benchmark_path(build_dir: Path, exe_name: str) -> Path:
    return build_dir / "benchmarks" / exe_name


def build_minibench_command(
    exe: Path,
    *,
    backend: str,
    dtype: str,
    csv_path: Path,
    warmup: int,
    min_time_ms: float,
    min_iters: int,
    max_iters: int,
    args: List[int],
) -> List[str]:
    return [
        str(exe),
        f"--backend={backend}",
        f"--type={dtype}",
        f"--csv={csv_path}",
        f"--warmup={warmup}",
        f"--min_time={min_time_ms}",
        f"--min_iters={min_iters}",
        f"--max_iters={max_iters}",
    ] + [str(value) for value in args]


def parse_minibench_csv(csv_path: Path, *, expected_metric: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as stream:
        reader = csv.DictReader(stream)
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
            index = 0
            while f"arg{index}" in row:
                value = (row.get(f"arg{index}") or "").strip()
                if not value:
                    break
                args.append(int(value))
                index += 1
            rows.append({
                "name": name,
                "args": args,
                "metric": expected_metric,
                "value": float(row[expected_metric]),
                "avg_ms": float(row.get("avg_ms") or 0.0),
                "stddev_ms": float(row.get("stddev_ms") or 0.0),
            })
    return rows


def run_minibench_csv(
    exe: Path,
    *,
    backend: str,
    dtype: str,
    metric: str,
    warmup: int,
    min_time_ms: float,
    min_iters: int,
    max_iters: int,
    args: List[int],
    prefix: str,
    env: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary_dir:
        csv_path = Path(temporary_dir) / "out.csv"
        command = build_minibench_command(
            exe,
            backend=backend,
            dtype=dtype,
            csv_path=csv_path,
            warmup=warmup,
            min_time_ms=min_time_ms,
            min_iters=min_iters,
            max_iters=max_iters,
            args=args,
        )
        subprocess.run(command, env=env, check=True)
        return parse_minibench_csv(csv_path, expected_metric=metric)

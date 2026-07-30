#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.benchmark_runner import default_benchmark_path, default_build_dir, run_minibench_csv


@dataclass(frozen=True)
class BenchSpace:
    bench: str
    exe: str
    metric: str
    direction: str  # "min" | "max"
    arg_spec: List[str]
    cases: List[Dict[str, Any]]  # each has {"fixed": {..}, "tune": {..}}
    pre_tune: List[Dict[str, Any]]  # optional bench-level phases: {"params": {..}, "cases": [..]}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_space_path(repo_root: Path) -> Path:
    return repo_root / "evaluation" / "tuning" / "spaces" / "default.json"


def _default_output_path(build_dir: Path) -> Path:
    return build_dir / "tuning" / "profile.json"


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


def _representative_case_params(case: Dict[str, Any], *, exclude: Optional[Sequence[str]] = None) -> Dict[str, int]:
    representative: Dict[str, int] = {}
    excluded = set(exclude or [])
    for key, values in (case.get("tune") or {}).items():
        if key in excluded:
            continue
        if not isinstance(values, list) or not values:
            continue
        representative[str(key)] = int(values[0])
    return representative


def _normalize_case_indices(total_cases: int, selected: Optional[Sequence[Any]]) -> List[int]:
    if selected is None:
        return list(range(total_cases))

    indices: List[int] = []
    for raw in selected:
        idx = int(raw)
        if idx < 0 or idx >= total_cases:
            raise IndexError(f"Case index {idx} out of range for {total_cases} cases")
        indices.append(idx)

    if not indices:
        raise ValueError("pre_tune phase must reference at least one case")

    return indices


def _collect_tune_keys(cases: Sequence[Dict[str, Any]], *, exclude: Optional[Sequence[str]] = None) -> Dict[str, List[int]]:
    tune_keys: Dict[str, List[int]] = {}
    excluded = set(exclude or [])
    for case in cases:
        for k, vs in (case.get("tune") or {}).items():
            if k in excluded:
                continue
            if k not in tune_keys:
                tune_keys[k] = list(vs)
            else:
                existing = set(tune_keys[k])
                new = set(vs)
                inter = existing.intersection(new)
                tune_keys[k] = sorted(inter) if inter else sorted(existing.union(new))
    return tune_keys


def _tune_pre_phases(
    *,
    space: BenchSpace,
    exe: Path,
    backend: str,
    dtype: str,
    warmup: int,
    min_time_ms: float,
    min_iters: int,
    max_iters: int,
    verbose: bool,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    resolved: Dict[str, int] = {}
    summaries: List[Dict[str, Any]] = []

    for phase_idx, phase in enumerate(space.pre_tune):
        raw_params = phase.get("params") or {}
        if not isinstance(raw_params, dict) or not raw_params:
            raise ValueError(f"Invalid pre_tune phase at index {phase_idx}: expected non-empty 'params' object")

        case_indices = _normalize_case_indices(len(space.cases), phase.get("cases"))
        candidate_grid = {str(k): [int(v) for v in vs] for k, vs in raw_params.items()}

        best: Optional[Dict[str, Any]] = None
        for params in _iter_grid(candidate_grid):
            effective_params = {**resolved, **params}
            values: List[float] = []

            for case_idx in case_indices:
                case = space.cases[case_idx]
                fixed = {k: int(v) for k, v in (case.get("fixed") or {}).items()}
                case_defaults = _representative_case_params(case, exclude=effective_params.keys())
                args = _args_from_spec(
                    space.arg_spec,
                    fixed=fixed,
                    params={**case_defaults, **effective_params},
                )
                values.append(
                    _run_one_case(
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
                )

            score = _score(values, space.direction)
            entry = {
                "phase_index": phase_idx,
                "params": dict(params),
                "resolved_params": dict(effective_params),
                "cases": case_indices,
                "values": values,
                "score": score,
            }
            if verbose:
                pretty = ", ".join(f"{k}={v}" for k, v in effective_params.items())
                avg = sum(values) / max(1, len(values))
                print(f"[{space.bench}:pre_tune:{phase_idx}] {pretty} -> avg={avg:.6g} ({space.metric})")

            if best is None or entry["score"] < best["score"]:
                best = entry

        assert best is not None
        resolved.update({k: int(v) for k, v in best["params"].items()})
        summaries.append(best)

    return resolved, summaries


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
    rows = run_minibench_csv(
        exe,
        backend=backend,
        dtype=dtype,
        metric=metric,
        warmup=warmup,
        min_time_ms=min_time_ms,
        min_iters=min_iters,
        max_iters=max_iters,
        args=args,
        prefix="batchlas-tune-",
    )
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
                pre_tune=list(s.get("pre_tune") or []),
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
    per_case_best: List[Optional[Dict[str, Any]]] = [None for _ in space.cases]

    resolved_pre_tune, pre_tune_summary = _tune_pre_phases(
        space=space,
        exe=exe,
        backend=backend,
        dtype=dtype,
        warmup=warmup,
        min_time_ms=min_time_ms,
        min_iters=min_iters,
        max_iters=max_iters,
        verbose=verbose,
    )

    tune_keys = _collect_tune_keys(space.cases, exclude=resolved_pre_tune.keys())

    for params in _iter_grid(tune_keys):
        effective_params = {**resolved_pre_tune, **params}
        values: List[float] = []
        per_case: List[Dict[str, Any]] = []

        for case_idx, case in enumerate(space.cases):
            fixed = {k: int(v) for k, v in (case.get("fixed") or {}).items()}
            args = _args_from_spec(space.arg_spec, fixed=fixed, params=effective_params)
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

            current = per_case_best[case_idx]
            better = False
            if current is None:
                better = True
            elif space.direction == "min":
                better = v < float(current["value"])
            else:
                better = v > float(current["value"])

            if better:
                per_case_best[case_idx] = {
                    "case_index": case_idx,
                    "fixed": fixed,
                    "args": args,
                    "value": v,
                    "params": dict(effective_params),
                }

        s = _score(values, space.direction)
        entry = {"params": effective_params, "score": s, "values": values, "cases": per_case}

        if verbose:
            # Print minimal progress; keep stdout readable.
            pretty = ", ".join(f"{k}={v}" for k, v in effective_params.items())
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
        "pre_tune": pre_tune_summary,
        "best": best,
        "top": leaderboard,
        "per_case_best": [x for x in per_case_best if x is not None],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BatchLAS bottom-up tuning harness (grid search)")
    parser.add_argument("--build-dir", type=Path, default=None, help="Build directory (default: <repo>/build)")
    parser.add_argument(
        "--space",
        type=Path,
        default=None,
        help="Path to tuning space JSON (default: <repo>/evaluation/tuning/spaces/default.json)",
    )
    parser.add_argument("--backend", type=str, required=True, help="Backend passed to benchmarks (e.g., CUDA, ROCM, NETLIB, MKL)")
    parser.add_argument("--type", dest="dtype", type=str, required=True, help="Type passed to benchmarks (e.g., float, double)")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON profile path (default: <build-dir>/tuning/profile.json)")

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
    space_path = args.space or _default_space_path(repo_root)
    out_path = args.out or _default_output_path(build_dir)

    spaces = _load_spaces(space_path)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(f"Wrote tuning profile: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

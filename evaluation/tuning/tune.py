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
    # {param_name: ENV_VAR}. Params listed here are passed to the benchmark as
    # environment variables instead of positional arguments, which is the only
    # way to reach knobs no benchmark exposes in its arg list (gebrd's panel
    # width, the sy2sb ormqr hint, the sb2st block size...). They must NOT
    # appear in arg_spec.
    env: Dict[str, str]


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


def _env_from_spec(env_spec: Dict[str, str],
                   fixed: Dict[str, int],
                   params: Dict[str, int]) -> Dict[str, str]:
    """Environment overrides for this combo, as {ENV_VAR: "value"}.

    A param with no value in either params or fixed is simply left unset, which
    means the benchmark keeps its compiled default.
    """
    out: Dict[str, str] = {}
    for name, env_var in env_spec.items():
        if name in params:
            out[str(env_var)] = str(int(params[name]))
        elif name in fixed:
            out[str(env_var)] = str(int(fixed[name]))
    return out


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
    """Union of every case's grid for each tuned key.

    Deliberately a union, not an intersection. The generated header is
    bucket-first: it reads `per_case_best`, so each case must be free to search
    its own declared grid. Intersecting first silently shrinks that search --
    with the shipped default space, ormqr's per-case grids intersect to the
    single value {16}, so every ORMQR_BLOCK_SIZE_* bucket was "tuned" without
    ever comparing a second candidate.

    Cross-case aggregation (`best`/`top`) is unaffected: `_case_allows` keeps a
    case out of any combo its own grid excludes, and only combos that cover
    every case are scored -- which is exactly the old intersection.
    """
    tune_keys: Dict[str, List[int]] = {}
    excluded = set(exclude or [])
    for case in cases:
        for k, vs in (case.get("tune") or {}).items():
            if k in excluded:
                continue
            if k not in tune_keys:
                tune_keys[k] = list(vs)
            else:
                tune_keys[k] = sorted(set(tune_keys[k]).union(vs))
    return tune_keys


def _case_allows(case: Dict[str, Any], params: Dict[str, int]) -> bool:
    """True when `params` lies inside this case's own declared grid.

    A key the case does not declare is unconstrained -- it is resolved from
    `fixed` (or a pre-tuned value) exactly as before.
    """
    case_tune = case.get("tune") or {}
    for key, value in params.items():
        allowed = case_tune.get(key)
        if allowed is None:
            continue
        if int(value) not in {int(v) for v in allowed}:
            return False
    return True


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
                phase_params = {**case_defaults, **effective_params}
                args = _args_from_spec(
                    space.arg_spec,
                    fixed=fixed,
                    params=phase_params,
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
                        env_overrides=_env_from_spec(space.env, fixed=fixed, params=phase_params),
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
    env_overrides: Optional[Dict[str, str]] = None,
) -> float:
    # Merge, never replace: run_minibench_csv hands this straight to
    # subprocess.run, and a bare dict would drop PATH / LD_LIBRARY_PATH and the
    # CUDA and SYCL runtime variables the benchmark needs to start at all.
    env = {**os.environ, **env_overrides} if env_overrides else None

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
        env=env,
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
                env={str(k): str(v) for k, v in (s.get("env") or {}).items()},
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
    partial_best: Optional[Dict[str, Any]] = None
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

    # Widening the grid to a union means different combos can collapse to the
    # same argument vector for a case that does not tune every key. Measure each
    # distinct vector once.
    measured: Dict[Tuple[Any, ...], float] = {}

    for params in _iter_grid(tune_keys):
        effective_params = {**resolved_pre_tune, **params}
        values: List[float] = []
        per_case: List[Dict[str, Any]] = []
        covers_all_cases = True

        for case_idx, case in enumerate(space.cases):
            if not _case_allows(case, effective_params):
                covers_all_cases = False
                continue

            fixed = {k: int(v) for k, v in (case.get("fixed") or {}).items()}
            args = _args_from_spec(space.arg_spec, fixed=fixed, params=effective_params)
            env_overrides = _env_from_spec(space.env, fixed=fixed, params=effective_params)
            # The env vars are part of the measured configuration, so they must
            # be part of the cache key -- otherwise every env-tuned combo would
            # collide on an identical argument vector and return the first
            # measurement for all of them.
            key = (tuple(args), tuple(sorted(env_overrides.items())))
            if key in measured:
                v = measured[key]
            else:
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
                    env_overrides=env_overrides,
                )
                measured[key] = v
            values.append(v)
            entry_case: Dict[str, Any] = {"fixed": fixed, "args": args, "value": v}
            if env_overrides:
                entry_case["env"] = env_overrides
            per_case.append(entry_case)

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

        if not values:
            continue

        s = _score(values, space.direction)
        entry = {"params": effective_params, "score": s, "values": values, "cases": per_case}

        # Only combos legal for every case are comparable across cases, so the
        # cross-case leaderboard still ranks exactly the old intersected grid.
        # Per-case winners above are already recorded either way. Partial-coverage
        # combos are kept aside purely so a space with fully disjoint per-case
        # grids still yields a profile instead of tripping the assert below.
        if not covers_all_cases:
            if partial_best is None or s < partial_best["score"]:
                partial_best = entry
            continue

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

    if best is None:
        best = partial_best
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Run one cheap invocation per bench to check the executable exists and the "
            "metric column parses, then exit without tuning. Do this before a real "
            "sweep: benches run sequentially, so a wrong metric name in the last one "
            "otherwise surfaces after every earlier bench has already been measured."
        ),
    )

    args = parser.parse_args()

    repo_root = _repo_root()
    build_dir = args.build_dir or default_build_dir(repo_root)
    space_path = args.space or _default_space_path(repo_root)
    out_path = args.out or _default_output_path(build_dir)

    spaces = _load_spaces(space_path)

    if args.validate:
        problems: List[str] = []
        for space in spaces:
            exe = default_benchmark_path(build_dir, space.exe)
            if not exe.exists() or not os.access(exe, os.X_OK):
                problems.append(f"{space.bench}: missing executable {exe}")
                continue
            case = space.cases[0]
            fixed = {k: int(v) for k, v in (case.get("fixed") or {}).items()}
            # pre_tune params are resolved before the main sweep and so are
            # absent from any case's `tune` grid; take their first candidate.
            pre: Dict[str, int] = {}
            for phase in space.pre_tune:
                for key, values in (phase.get("params") or {}).items():
                    if isinstance(values, list) and values:
                        pre[str(key)] = int(values[0])
            params = {**pre, **_representative_case_params(case)}
            try:
                _run_one_case(
                    exe=exe,
                    backend=args.backend,
                    dtype=args.dtype,
                    metric=space.metric,
                    warmup=0,
                    min_time_ms=0.0,
                    min_iters=1,
                    max_iters=1,
                    args=_args_from_spec(space.arg_spec, fixed=fixed, params=params),
                    env_overrides=_env_from_spec(space.env, fixed=fixed, params=params),
                )
            except Exception as exc:  # noqa: BLE001 - report every bench, not just the first
                problems.append(f"{space.bench}: {exc}")
            else:
                print(f"[ok] {space.bench} via {exe.name} (metric {space.metric!r})")

        for problem in problems:
            print(f"[FAIL] {problem}")
        print(f"\n{len(spaces) - len(problems)}/{len(spaces)} benches OK")
        return 1 if problems else 0

    results: List[Dict[str, Any]] = []
    for space in spaces:
        exe = default_benchmark_path(build_dir, space.exe)
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

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


NSYS_FALLBACKS = [
    "/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/nsys",
]
NCU_FALLBACKS = [
    "/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/ncu",
]
NVPROF_FALLBACKS = [
    "/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/nvprof",
]


def repo_root_from(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "benchmarks" / "CMakeLists.txt").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate BatchLAS repository root")


def shell_join(items: Iterable[str]) -> str:
    return " ".join(shlex.quote(item) for item in items)


def resolve_tool(name: str, fallbacks: List[str]) -> str:
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if Path(candidate).exists():
            return candidate
    return name


def base_benchmark_command(args: argparse.Namespace, benchmark_path: Path, case_args: List[str]) -> List[str]:
    cmd = [str(benchmark_path)]
    if args.name:
        cmd.append(f"--name={args.name}")
    cmd.extend(
        [
            f"--backend={args.backend}",
            f"--type={args.dtype}",
            f"--warmup={args.warmup}",
            f"--min_iters={args.min_iters}",
            f"--max_iters={args.max_iters}",
            f"--min_time={args.min_time}",
        ]
    )
    cmd.extend(case_args)
    return cmd


def prefixed_command(env_items: List[str], command: List[str]) -> str:
    parts = [*env_items, *command]
    return shell_join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print BatchLAS profiling commands for trace, nsys, ncu, or nvprof.")
    parser.add_argument("--benchmark", required=True, help="Benchmark target name, for example ormqr_blocked_benchmark")
    parser.add_argument("--build-dir", default=None, help="Build directory, default: <repo>/build")
    parser.add_argument("--tool", choices=("trace", "nsys", "ncu", "nvprof", "all"), default="all")
    parser.add_argument("--backend", default="CUDA")
    parser.add_argument("--type", dest="dtype", default="float")
    parser.add_argument("--name", default=None, help="Optional minibench name filter")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-iters", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--min-time", type=int, default=0)
    parser.add_argument("--output-dir", default=None, help="Output directory, default: <repo>/output/profiling")
    parser.add_argument("--output-stem", default=None, help="Base name for profiling artifacts")
    parser.add_argument("--kernel-regex", default=None, help="Kernel regex for ncu")
    parser.add_argument("--ncu-mode", choices=("basic", "detailed"), default="basic")
    parser.add_argument("--env", action="append", default=[], help="Extra env assignment, repeatable, for example KEY=VALUE")
    args, extra = parser.parse_known_args()

    if extra and extra[0] == "--":
        extra = extra[1:]

    repo_root = repo_root_from(Path(__file__))
    build_dir = Path(args.build_dir) if args.build_dir else repo_root / "build"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "output" / "profiling"
    benchmark_path = build_dir / "benchmarks" / args.benchmark
    output_stem = args.output_stem or args.benchmark.removesuffix("_benchmark")

    base_cmd = base_benchmark_command(args, benchmark_path, extra)
    env_items = list(args.env)

    nsys_bin = resolve_tool("nsys", NSYS_FALLBACKS)
    ncu_bin = resolve_tool("ncu", NCU_FALLBACKS)
    nvprof_bin = resolve_tool("nvprof", NVPROF_FALLBACKS)

    print(f"mkdir -p {shlex.quote(str(output_dir))}")
    print()

    if args.tool in ("trace", "all"):
        trace_path = output_dir / f"{output_stem}.trace.json"
        env_prefix = [*env_items, "BATCHLAS_KERNEL_TRACE=1", f"BATCHLAS_KERNEL_TRACE_PATH={trace_path}"]
        print("# BatchLAS kernel trace")
        print(prefixed_command(env_prefix, base_cmd))
        print()

    if args.tool in ("nsys", "all"):
        nsys_out = output_dir / f"{output_stem}_nsys"
        nsys_cmd = [
            nsys_bin,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--stats=true",
            "--force-overwrite=true",
            "-o",
            str(nsys_out),
            *base_cmd,
        ]
        stats_cmd = [
            nsys_bin,
            "stats",
            "--report",
            "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum",
            f"{nsys_out}.nsys-rep",
        ]
        print("# Nsight Systems capture")
        print(prefixed_command(env_items, nsys_cmd))
        print("# Nsight Systems summary replay")
        print(shell_join(stats_cmd))
        print()

    if args.tool in ("ncu", "all"):
        kernel_regex = args.kernel_regex or "TODO_KERNEL_REGEX"
        ncu_out = output_dir / f"{output_stem}_ncu"
        if args.ncu_mode == "basic":
            section_args = ["--set", "basic"]
        else:
            section_args = [
                "--section",
                "SchedulerStats",
                "--section",
                "WarpStateStats",
                "--section",
                "SourceCounters",
                "--section",
                "MemoryWorkloadAnalysis",
                "--section",
                "ComputeWorkloadAnalysis",
            ]
        ncu_cmd = [
            ncu_bin,
            *section_args,
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{kernel_regex}",
            "--launch-count",
            "1",
            "--target-processes",
            "all",
            "--force-overwrite",
            "--export",
            str(ncu_out),
            *base_cmd,
        ]
        import_cmd = [
            ncu_bin,
            "--import",
            f"{ncu_out}.ncu-rep",
            "--page",
            "details",
            "--print-summary",
            "per-kernel",
        ]
        print("# Nsight Compute capture")
        if args.kernel_regex is None:
            print("# Replace TODO_KERNEL_REGEX with the hotspot kernel from the trace or nsys pass.")
        print(prefixed_command(env_items, ncu_cmd))
        print("# Nsight Compute report replay")
        print(shell_join(import_cmd))
        print()

    if args.tool in ("nvprof", "all"):
        nvprof_cmd = [
            nvprof_bin,
            "--print-gpu-summary",
            "--print-api-summary",
            *base_cmd,
        ]
        print("# nvprof summary")
        print("# Use only on older CUDA toolkits where nvprof still runs. This machine reports CUDA 13 deprecation.")
        print(prefixed_command(env_items, nvprof_cmd))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

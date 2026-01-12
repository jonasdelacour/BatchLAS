#!/usr/bin/env python3
"""bandr1_evolution

One-command driver to:
  1) generate BANDR1 dump CSVs (via build/benchmarks/bandr1_dump)
  2) plot a sweep×step mosaic PNG

This exists because the full plot script has a lot of knobs. This wrapper is
opinionated and always regenerates into a parameter-derived dump directory, so
changing flags actually changes the generated data.

Example:
  python3 playground/bandr1_evolution.py \
      --n 64 --kd 16 --kd-work 24 --block-size 16 --d 2 \
      --sweep-max 0 --step-max 24 \
      --out output/bandr1_plots/evolution64_kd16_d2.png

Requirements:
  - Configure with benchmarks enabled: cmake -S . -B build -DBATCHLAS_BUILD_BENCHMARKS=ON
  - Build driver: cmake --build build -j --target bandr1_dump
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", required=True, help="Output PNG path")

    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--kd", type=int, default=8)
    ap.add_argument("--kd-work", type=int, default=0, help="0 => auto (min(n-1, 3*kd))")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--d", type=int, default=0, help="0 => impl default")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--type", dest="scalar_type", default="f32", choices=["f32", "f64", "c64", "c128"])
    ap.add_argument("--device", default="gpu")

    ap.add_argument("--which", default="after", choices=["before", "after"])
    ap.add_argument("--mode", default="nz", choices=["abs", "real", "imag", "nz"])
    ap.add_argument("--eps", type=float, default=0.0, help="Threshold for --mode nz")

    ap.add_argument("--layout", default="sweep-step", choices=["sweep-step", "linear"])
    ap.add_argument("--sweep-max", type=int, default=0, help="Max sweep index to plot (0 => first sweep only)")
    ap.add_argument("--step-max", type=int, default=24, help="Max step index to plot")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-cols", type=int, default=None)

    ap.add_argument(
        "--driver",
        default="build/benchmarks/bandr1_dump",
        help="Path to bandr1_dump executable",
    )
    ap.add_argument(
        "--dump-root-base",
        default="output/bandr1_dumps/auto",
        help="Base directory under which run-specific dump dirs are created",
    )
    ap.add_argument(
        "--keep-dumps",
        action="store_true",
        help="Do not delete existing dump directory (default: always regenerate)",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    driver = Path(args.driver)
    if not driver.is_absolute():
        driver = (Path.cwd() / driver).resolve()
    if not driver.exists():
        raise SystemExit(
            f"bandr1_dump not found at {driver}.\n"
            "Build with: cmake -S . -B build -DBATCHLAS_BUILD_BENCHMARKS=ON\n"
            "Then:       cmake --build build -j --target bandr1_dump"
        )

    # Make a deterministic dump directory name from parameters.
    kdw_tag = str(args.kd_work) if args.kd_work and args.kd_work > 0 else "auto"
    run_dir = (
        Path(args.dump_root_base)
        / f"n{args.n}_kd{args.kd}_kdw{kdw_tag}_bs{args.block_size}_d{args.d}_seed{args.seed}_{args.scalar_type}"
    )

    if run_dir.exists() and not args.keep_dumps:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Generate enough sweeps to cover what we will plot.
    max_sweeps = int(args.sweep_max) + 1 if args.sweep_max is not None and args.sweep_max >= 0 else 1

    gen_cmd = [
        str(driver),
        "--dump-dir",
        str(run_dir),
        "--n",
        str(args.n),
        "--kd",
        str(args.kd),
        "--kd-work",
        str(args.kd_work),
        "--block-size",
        str(args.block_size),
        "--d",
        str(args.d),
        "--max-sweeps",
        str(max_sweeps),
        "--batch",
        "1",
        "--seed",
        str(args.seed),
        "--type",
        str(args.scalar_type),
        "--device",
        str(args.device),
        "--abw-only",
        "--dump-step",
    ]

    print("Generating dumps:\n  " + " ".join(gen_cmd))
    subprocess.run(gen_cmd, check=True)

    plot_cmd = [
        "python3",
        str((Path.cwd() / "playground/plot_bandr1_evolution.py").resolve()),
        "--dump-root",
        str(run_dir),
        "--mosaic-out",
        str(out_path),
        "--layout",
        str(args.layout),
        "--which",
        str(args.which),
        "--mode",
        str(args.mode),
        "--eps",
        str(args.eps),
        "--batch",
        "0",
        "--scale",
        "global",
        "--sweep-max",
        str(args.sweep_max),
        "--step-max",
        str(args.step_max),
    ]

    # Only apply filters if the user requested them (avoid the exact footgun you hit).
    plot_cmd += ["--n", str(args.n)]
    if args.kd_work and args.kd_work > 0:
        plot_cmd += ["--kd-work", str(args.kd_work)]

    if args.max_rows is not None:
        plot_cmd += ["--max-rows", str(args.max_rows)]
    if args.max_cols is not None:
        plot_cmd += ["--max-cols", str(args.max_cols)]

    print("Plotting mosaic:\n  " + " ".join(plot_cmd))
    subprocess.run(plot_cmd, check=True)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# BatchLAS tuning harness (bottom-up)

This directory contains a minimal grid-search tuner that reuses the existing benchmark executables under `build/benchmarks/`.

## What it does

- Runs one or more benchmark executables for a fixed set of problem sizes.
- Sweeps a small discrete search space of tuning parameters (block sizes, thresholds, etc.).
- Chooses the parameter set that minimizes the selected timing metric averaged across the configured cases.
- Writes a JSON “tuning profile” you can later consume in the library (compile-time or runtime).

## Run manually

From the repo root, after building benchmarks:

- Build benchmarks: `cmake -B build -DBATCHLAS_BUILD_BENCHMARKS=ON && cmake --build build -j`
- Tune with CUDA float (example):

  `python3 evaluation/tuning/tune.py --space evaluation/tuning/spaces/default.json --backend CUDA --type float --out build/tuning/profile_cuda_float.json --skip-missing`

Notes:
- Some benchmarks are only built/active for certain backends (e.g. `sytrd_blocked_benchmark` is CUDA-only today). Use `--skip-missing` to ignore unavailable executables.
- You can change problem sizes and search ranges by editing the JSON space file.

## Output format (high level)

The output JSON contains:
- `meta`: environment info (backend/type/build dir)
- `results`: per-benchmark best parameters and a small top-K leaderboard

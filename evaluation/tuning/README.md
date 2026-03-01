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

## Generating compile-time tuning constants

BatchLAS now provides a generated header at build time:

- `build/include/batchlas/tuning_params.hh`

By default, this header is generated with safe fallback values at CMake configure time.

To generate benchmark-derived constants from a profile:

- One-shot from an existing profile:

  `python3 evaluation/tuning/generate_tuning_header.py --profile build/tuning/profile.json --out build/include/batchlas/tuning_params.hh`

- Through CMake target (when `BATCHLAS_ENABLE_TUNING=ON`):

  `cmake --build build --target batchlas_tuning_header`

This target depends on `batchlas_tune`, so it will run the tuning harness first and then regenerate the header.

## Current model (size-aware only)

The tuning header is now bucket-first (no single global ORMQR/SYTRD block-size constant in the generation flow).

Generated constants:

- `ORMQR_BLOCK_SIZE_TINY`, `ORMQR_BLOCK_SIZE_SMALL`, `ORMQR_BLOCK_SIZE_MEDIUM`, `ORMQR_BLOCK_SIZE_LARGE`, `ORMQR_BLOCK_SIZE_XLARGE`
- `SYTRD_BLOCK_SIZE_TINY`, `SYTRD_BLOCK_SIZE_SMALL`, `SYTRD_BLOCK_SIZE_MEDIUM`, `SYTRD_BLOCK_SIZE_LARGE`, `SYTRD_BLOCK_SIZE_XLARGE`

STEDC constants are bucketed:

- `STEDC_*_{TINY,SMALL,MEDIUM,LARGE,XLARGE}`

Runtime selection helpers:

- `batchlas::tuning::ormqr_block_size_for_n(n)`
- `batchlas::tuning::sytrd_block_size_for_n(n)`
- `batchlas::tuning::stedc_recursion_threshold_for_n(n)`
- `batchlas::tuning::stedc_merge_variant_for_n(n)`
- `batchlas::tuning::stedc_threads_per_root_for_n(n)`
- `batchlas::tuning::stedc_wg_multiplier_for_n(n)`

Bucket boundaries are currently:

- `n <= 64` -> `tiny`
- `65..128` -> `small`
- `129..256` -> `medium`
- `257..512` -> `large`
- `> 512` -> `xlarge`

When tuning data includes multiple `n` cases, bucketed values are derived from each case winner (`per_case_best`) in the profile.

## STEDC bottom-up cases

STEDC tuning cases start at `n=64` and above. Leaf sizes `n <= 32` are intentionally not tuned separately.

At runtime, recursion thresholds are clamped to local subproblem size (`threshold <= n`) at each recursion level.

## Practical workflow

1) Configure with benchmarks+tuning enabled:

`cmake -B build -DBATCHLAS_BUILD_BENCHMARKS=ON -DBATCHLAS_ENABLE_TUNING=ON`

2) Run tuning + regenerate header:

`cmake --build build --target batchlas_tuning_header -j`

3) Inspect outputs:

- Profile JSON: `build/tuning/profile.json`
- Generated header: `build/include/batchlas/tuning_params.hh`

4) Verify the selected parameters:

- Check `results[].per_case_best` for per-`n` winners.
- Check header constants and `*_for_n` selectors match your expected ranges.

## Notes

- `syev` does not carry independent child block-size knobs; it always uses `ormqr_block_size_for_n` and `sytrd_block_size_for_n` from child-owned tuning.
- If your profile does not cover some size ranges, configured fallback bucket values are used for those missing ranges.

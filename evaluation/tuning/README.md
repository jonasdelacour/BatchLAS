# BatchLAS tuning harness (bottom-up)

This directory contains a minimal grid-search tuner that reuses the existing benchmark executables under `build/benchmarks/`.

## What it does

- Runs one or more benchmark executables for a fixed set of problem sizes.
- Sweeps a small discrete search space of tuning parameters (block sizes, thresholds, etc.).
- Optionally pre-tunes selected parameters on a representative subset of cases before the main cross-case search.
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

## Tuning space format

Each bench entry contains:

- `arg_spec`: positional benchmark arguments.
- `cases`: problem sizes, each with `fixed` args and a case-local `tune` grid.
- Optional `pre_tune`: one or more bench-level phases of the form `{ "params": { ... }, "cases": [ ... ] }`.

`pre_tune` phases run before the main search. The selected values are then injected into every case as fixed parameters for the main sweep and are still recorded in the final profile's `best`, `top`, and `per_case_best` parameter sets.

## Output format (high level)

The output JSON contains:
- `meta`: environment info (backend/type/build dir)
- `results`: per-benchmark best parameters and a small top-K leaderboard

## Generating compile-time tuning constants

**The header the library actually compiles is `include/batchlas/tuning_params.hh`,
not the one under `build/include/`.** Both files exist and declare the same
symbols, and the compile line lists `-I<repo>/include` *before*
`-I<repo>/build/include`, so the checked-in header always shadows the generated
one. Verify with:

```
g++ -std=c++17 -fsyntax-only -I include -I build/include -x c++ - <<'EOF'
#include <batchlas/tuning_params.hh>
static_assert(batchlas::tuning::ORMQR_BLOCK_SIZE_MEDIUM == 16);
EOF
```

Consequences, both confirmed on this tree:

- `cmake --build build --target batchlas_tuning_header` writes
  `build/include/batchlas/tuning_params.hh` and therefore **changes nothing**.
  It is a no-op for the library.
- CMake's `configure_file` rewrites that same path from hardcoded defaults on
  every reconfigure, so even its contents are transient.

To actually move the constants, regenerate and port the values into the
checked-in header:

```
python3 evaluation/tuning/generate_tuning_header.py \
    --profile build/tuning/profile.json \
    --out /tmp/tuning_params.hh
diff include/batchlas/tuning_params.hh /tmp/tuning_params.hh
```

Copy across only the `inline constexpr` values you intend to change. Do not
overwrite the file wholesale: its comments are hand-maintained (see the
`StedcMergeVariant` note), the generator's template does not carry them, and
several shipped constants are outside the default space's grid — regenerating
blindly would silently downgrade e.g. `STEDC_WG_MULTIPLIER_*` from 8 to 4
because the space only sweeps `[1,2,4]`.

### Faster: A/B a candidate with no rebuild

Every accessor consults an environment variable first, so a candidate can be
measured against the compiled default in the existing build:

```
BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=48 ./build/benchmarks/gesvd_blocked_benchmark \
    --backend=CUDA --type=float 512 256
```

The variables are `BATCHLAS_TUNE_{ORMQR_BLOCK_SIZE, SYTRD_BLOCK_SIZE,
LATRD_WG_HINT, STEDC_RECURSION_THRESHOLD, STEDC_MERGE_VARIANT,
STEDC_THREADS_PER_ROOT, STEDC_WG_MULTIPLIER}`. Each overrides *all* size
buckets at once, so test one `n` at a time. Do not change one mid-process: the
same accessor feeds `*_buffer_size()` queries and the matching solve.

## Validate a winner at the consumer before adopting it

A per-kernel winner is not automatically an end-to-end win, and can be a loss.
Measured 2026-08-06 on CUDA/float after fixing the grid-intersection bug:

| block size | ormqr kernel, n=1024 | syev n=1024 | gesvd_blocked n=512 |
|---|---|---|---|
| 16 (shipped) | 536 µs | 899 µs | 940 µs |
| 48/56 (tuned) | 238 µs (**2.16x faster**) | 905 µs (no change) | 1045 µs (**11% slower**) |

So the 2.16x kernel win was not adopted. `ORMQR_BLOCK_SIZE_*` is read by
`syev_blocked`, `syev_two_stage`, `syevx_direct_subset` *and* by
`gesvd_blocked.cc`, where it also sets `gebrd_block_size` — a different kernel
with a different optimum. `ormqr_blocked_benchmark` cannot see that coupling
because it takes the block size as an explicit argument.

Always A/B the consumer benchmarks with the env override before editing a
constant.

## Current model (size-aware only)

The tuning header is now bucket-first (no single global ORMQR/SYTRD block-size constant in the generation flow).

Generated constants:

- `ORMQR_BLOCK_SIZE_TINY`, `ORMQR_BLOCK_SIZE_SMALL`, `ORMQR_BLOCK_SIZE_MEDIUM`, `ORMQR_BLOCK_SIZE_LARGE`, `ORMQR_BLOCK_SIZE_XLARGE`
- `SYTRD_BLOCK_SIZE_TINY`, `SYTRD_BLOCK_SIZE_SMALL`, `SYTRD_BLOCK_SIZE_MEDIUM`, `SYTRD_BLOCK_SIZE_LARGE`, `SYTRD_BLOCK_SIZE_XLARGE`
- `LATRD_LOWER_PANEL_WG_HINT_TINY`, `LATRD_LOWER_PANEL_WG_HINT_SMALL`, `LATRD_LOWER_PANEL_WG_HINT_MEDIUM`, `LATRD_LOWER_PANEL_WG_HINT_LARGE`, `LATRD_LOWER_PANEL_WG_HINT_XLARGE`
- `SYTRD_FUSE_PANEL_UPDATE_TINY`, `SYTRD_FUSE_PANEL_UPDATE_SMALL`, `SYTRD_FUSE_PANEL_UPDATE_MEDIUM`, `SYTRD_FUSE_PANEL_UPDATE_LARGE`, `SYTRD_FUSE_PANEL_UPDATE_XLARGE`

STEDC constants are bucketed:

- `STEDC_*_{TINY,SMALL,MEDIUM,LARGE,XLARGE}`

Runtime selection helpers:

- `batchlas::tuning::ormqr_block_size_for_n(n)`
- `batchlas::tuning::sytrd_block_size_for_n(n)`
- `batchlas::tuning::latrd_lower_panel_wg_hint_for_n(n)`
- `batchlas::tuning::sytrd_fuse_panel_update_for_n(n)`
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

The default STEDC space pre-tunes `recursion_threshold` on the `n=64` case only, then reuses that threshold for the cross-size sweep of merge variant and workgroup settings.

At runtime, recursion thresholds are clamped to local subproblem size (`threshold <= n`) at each recursion level.

## Practical workflow

Do not use the `batchlas_tuning_header` CMake target -- see above, it writes a
header nothing compiles. Drive the scripts directly.

1) Build the five benchmarks the default space needs (they are ordinary
   benchmark targets; `BATCHLAS_ENABLE_TUNING` is not required):

```
cmake -B build -DBATCHLAS_BUILD_BENCHMARKS=ON
cmake --build build -j --target stedc_benchmark steqr_benchmark \
      sytrd_blocked_benchmark ormqr_blocked_benchmark syev_benchmark
```

2) Run the sweep. **~12 minutes** for the full default space on CUDA/float
   (RTX 4090, measured 2026-08-06: 535 benchmark invocations):

```
python3 evaluation/tuning/tune.py \
    --space evaluation/tuning/spaces/default.json \
    --backend CUDA --type float \
    --out build/tuning/profile.json --skip-missing
```

   Prefer `--skip-missing` alone. Adding `--skip-failed` (which the CMake target
   passes) converts a broken bench into a silent omission and still writes a
   profile that looks successful.

   To go faster, cut the space file rather than the iteration counts: drop the
   large-`n` cases, which dominate. `sytrd_blocked` at n=1024 is ~7.4 s per
   invocation against ~0.6 s for the small cases.

3) Inspect the profile at `build/tuning/profile.json`:

- `results[].per_case_best` -- per-`n` winners, what the bucketed header reads.
- `results[].best` -- the single parameter set best *averaged* across cases.
  Only combos legal for every case appear here, so for a bench whose per-case
  grids barely overlap this is a much smaller search than `per_case_best`.

4) A/B any candidate with the env override (no rebuild), at the *consumer*
   benchmark, then port the constant by hand into
   `include/batchlas/tuning_params.hh` and rebuild.

## Notes

- `syev` now supports coupled tuning of SYTRD internals (`nb`, LATRD lower-panel `wg`, and fused panel update) so the selected tuple is optimized for end-to-end eigensolver time.
- If your profile does not cover some size ranges, configured fallback bucket values are used for those missing ranges.

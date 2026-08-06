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

## Why a kernel winner can be an end-to-end loss

Measured 2026-08-06 on CUDA/float, after fixing the grid-intersection bug:

| block size | ormqr kernel, n=1024 | syev n=1024 | gesvd_blocked n=512 |
|---|---|---|---|
| 16 (shipped) | 536 µs | 899 µs | 940 µs |
| 48/56 (tuned) | 238 µs (**2.16x faster**) | 905 µs (inert) | 1045 µs (**11% slower**) |

The 2.16x kernel win was real and was still not adopted. Three separate
mechanisms produce that, and none is visible from `ormqr_blocked_benchmark`.

### 1. Aliasing — one constant drives unrelated kernels

`gesvd_blocked.cc` reads `ormqr_block_size_for_n` three times: twice for genuine
ormbr backtransforms, and once at line 753 as `gebrd_block_size` — the
bidiagonal reduction, a completely different kernel. With
`BATCHLAS_GESVD_PROFILE=1` at n=512, batch=256, vectors on:

| stage | nb=16 | nb=48 |
|---|---|---|
| `gesvd.gebrd` | 229.7 ms | 259.2 ms (**+12.8%**) |
| `gesvd.apply_left_backtransform` | 18.35 ms | 14.87 ms (−19%) |
| `gesvd.apply_right_backtransform` | 18.25 ms | 15.12 ms (−17%) |

The two uses have **opposite gradients**, and gebrd is 6.3x the bigger term, so
its loss (+29.5 ms) swamps the backtransform win (−6.6 ms). Worse, the two
curves have very different shapes: sweeping the knob against the `gesvd.gebrd`
stage alone gives 8:234.9  12:232.1  **16:230.7**  24:235.5  32:240.9  48:256.6 —
gebrd's own optimum is 16, and its curve is flat (±2%), while ormqr's is steep
(2.16x). Aliasing therefore pins the steep knob at the flat knob's optimum.

(With vectors *off*, which is the default in `gesvd_blocked_benchmark`, the
backtransforms do not run at all — 95% of that call is gebrd. So the original
"11% slower" measurement was 100% gebrd, with ormqr never invoked.)

### 2. Shadowing — the hot path never reads the constant

syev looked insensitive, which is not the same as the parameter not mattering.
`sytrd_sy2sb.cc`'s `sy2sb_ormqr_block_size_hint` returns `kd` outright when
`n >= 1024 && batch >= 32`, passing ormqr an explicit hint that bypasses the
tuning table. Flipping the tuning constant leaves the kernel trace
*bit-identical* (124 `ormqr_blocked.larft` calls at both 16 and 56).

Disable that local gate and the constant comes alive:

| | nb=16 | nb=56 |
|---|---|---|
| `BATCHLAS_SY2SB_ORMQR_NB=off` | 1094.9 µs | 905.4 µs (**1.21x**) |
| default (gate on) | 898.6 µs | 905.0 µs (inert) |

So the win is genuine — a hand-written local override had simply already
claimed it. **"No change" from a global knob is not evidence the parameter is
unimportant; it can mean the knob is dead code on that path.**

### 3. The bucket is keyed on the wrong dimension

`ormqr_block_size_for_n` keys on `A.rows()`, but the WY block width is bounded
by `k`, the reflector count. In sy2sb those differ by orders of magnitude
(rows in the thousands, `k = kd = 32`). The long comment at `sytrd_sy2sb.cc:26`
documents this, and mechanism 2 exists precisely to work around it.

### What to do about it

- **A/B at the consumer with the env override before editing any constant.**
  Cheap, no rebuild. This is the one non-negotiable step.
- **If flipping a knob changes nothing, check whether it is even read** —
  `BATCHLAS_KERNEL_TRACE=1` and compare call counts. A bit-identical trace means
  shadowing, not insignificance.
- **Do not tune a shared constant through a benchmark that takes it as an
  explicit argument.** `ormqr_blocked_benchmark` is structurally blind to all
  three mechanisms above.
- **Split aliased constants.** Giving gebrd its own `GEBRD_BLOCK_SIZE_*` would
  let gebrd keep 16 while ormqr consumers take 48 — worth ~2.3% on
  gesvd-with-vectors by the stage timings above, and it unblocks the 2.16x
  everywhere else. This is the highest-value follow-up, and it is a code change,
  not a tuning change.
- **Note what is not tuned at all.** `gesvd.gebrd` is ~79% of gesvd-with-vectors
  and ~95% without, and no bench in the default space tunes it.

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

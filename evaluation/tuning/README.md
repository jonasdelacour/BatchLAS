# BatchLAS tuning harness (bottom-up) {#tuning_harness}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2

The tuning harness grid-searches tuning parameters by running the benchmark executables in
`build/benchmarks/`. It writes a JSON profile (`meta`: backend, type, build dir; `results`: per-bench
best and top-K), and `generate_tuning_header.py` turns the profile into constants for
`include/batchlas/tuning_params.hh`. The numbers below are sm_89 measurements.

## Quick start

From the repository root, after building the benchmarks:

```sh
cmake -B build -DBATCHLAS_BUILD_BENCHMARKS=ON && cmake --build build -j
python3 evaluation/tuning/tune.py --space evaluation/tuning/spaces/default.json \
    --backend CUDA --type float --out build/tuning/profile.json --skip-missing
```

- A full default-space run takes about 12 minutes on CUDA/float (535 invocations, RTX 4090, 2026-08-06).
- Use `--skip-missing` alone. `--skip-failed` turns a broken bench into a silent omission and
  still writes a profile that looks successful.
- To go faster, cut the space file (drop large-n cases, which dominate), not the iteration counts.
  `sytrd_blocked` at n = 1024 takes about 7.4 s per invocation, against about 0.6 s for small cases.
- Some benches exist only on some backends (`sytrd_blocked_benchmark` is CUDA-only). `--skip-missing`
  ignores unavailable executables.

## Space files

| File | Purpose |
|---|---|
| `spaces/default.json` | Benches that feed a constant in `tuning_params.hh`. Run this to retune. |
| `spaces/unwired.json` | Parameters that matter but have no constant. The generator ignores them; the run produces a profile for a human to read. |

A bench belongs in `default.json` only if the generator reads it. Today those are `stedc`,
`sytrd_blocked`, `ormqr_blocked`, `syev`, `gesvd`, `sb2st` and `sy2sb`. A bench outside that list is
measured and then silently dropped. `steqr` was in that state until it was removed.

### Entry format

| Key | Meaning |
|---|---|
| `arg_spec` | Positional benchmark arguments |
| `cases` | Problem sizes, each with `fixed` args and a case-local `tune` grid |
| `pre_tune` | Optional phases `{ "params": {...}, "cases": [...] }` run before the main search. Winners are injected as fixed params into every case and recorded in the profile. |
| `env` | Optional `{param: ENV_VAR}`. Passed as environment variables instead of positional args; must not appear in `arg_spec`. Merged onto the parent environment and part of the measurement cache key. |
| `row` | Substring selecting the benchmark row to read. **Required** when an executable registers several benchmarks; the harness refuses to guess. |

Rules:

- Per-case grids are searched as a **union**; each case sweeps only its own grid. Only combos legal
  for every case enter the cross-case `best` and `top`. Per-case winners (`per_case_best`) are
  always recorded.
- **A grid must contain the value currently shipped in the header.** Otherwise a retune can "win"
  with the best of a grid that never held the incumbent and silently downgrade it. `stedc`'s
  `wg_multiplier` swept `[1,2,4]` while `STEDC_WG_MULTIPLIER_*` shipped 8.
- `sb2st_hh_benchmark` emits `BM_SB2ST_HH_CHASE` and `BM_SB2ST_HH_BACK`. Reading the first row
  measured a kernel the parameters cannot affect; the noise (1.4 % spread) looked like a winner
  where the intended kernel spans 3.6×. Always set `row`; `--validate` catches a missing one in seconds.
- `env` exists because several knobs have no positional argument: gebrd's panel width, the LATRD
  wg hint, the sb2st back-transform tiling, the sy2sb ormqr hint.

## Pruning: `sensitivity.py`

```sh
python3 evaluation/tuning/sensitivity.py build/tuning/profile.json
```

For each `(bench, parameter)` it sweeps that parameter with everything else fixed and reports how
far the metric moves. Read the **profile**, not a log: env-only parameters never appear in the
benchmark's CSV columns.

The 2026-08-07 pass removed the `latrd_lower_panel` bench (its only parameter moved the metric by
at most 0.38 %) and trimmed values that never won. The default space went from 609 to 265
invocations.

## The consumer can overrule the bench

An ordinary constant can also lose at the consumer. The `stedc` bench picked `wg_multiplier` 2–4
and `threads_per_root` 4–8. Adopting those cost syev 2.7 % at n = 256. Measured through syev with
everything else pinned, 8/8 wins nearly everywhere (7.6 % at n = 64; noise at n ≥ 512), so both ship
as 8.

The stedc bench measures the merge in isolation; syev pays for the whole tridiagonal solve. A
parameter whose owning bench is not its dominant consumer must be confirmed at the consumer. The
isolated sensitivity (0.56 % median for `wg_multiplier`) understated the real cost by an order of
magnitude.

## Generating constants

The header the library compiles is the checked-in `include/batchlas/tuning_params.hh`. There is no
generated copy.

```sh
python3 evaluation/tuning/generate_tuning_header.py \
    --profile build/tuning/profile.json --out /tmp/tuning_params.hh
diff include/batchlas/tuning_params.hh /tmp/tuning_params.hh
```

Port only the `inline constexpr` values you intend to change, by hand. Do not overwrite the file:

- its comments are hand-maintained (for example the `StedcMergeVariant` note), and the generator's
  template does not carry them;
- several shipped constants lie outside the default grid. Blind regeneration would downgrade
  `STEDC_WG_MULTIPLIER_*` from 8 to 4, because the space sweeps only `[1,2,4]`.

Read `results[].per_case_best` for the per-n winners the bucketed header uses. Read
`results[].best` for the single set best averaged across cases; it is a much smaller search when
the per-case grids barely overlap.

### A/B a candidate without a rebuild

Every accessor reads an environment variable first:

```sh
BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=48 ./build/benchmarks/gesvd_blocked_benchmark \
    --backend=CUDA --type=float 512 256
```

Variables: `BATCHLAS_TUNE_{ORMQR_BLOCK_SIZE, SYTRD_BLOCK_SIZE, LATRD_WG_HINT,
STEDC_RECURSION_THRESHOLD, STEDC_MERGE_VARIANT, STEDC_THREADS_PER_ROOT, STEDC_WG_MULTIPLIER}`.
Each overrides **all** size buckets, so test one n at a time. Do not change one mid-process: the
same accessor feeds `*_buffer_size()` and the matching solve.

## Workflow

1. Build the five benches the default space needs. `BATCHLAS_ENABLE_TUNING` is not required.

   ```sh
   cmake -B build -DBATCHLAS_BUILD_BENCHMARKS=ON
   cmake --build build -j --target stedc_benchmark steqr_benchmark \
         sytrd_blocked_benchmark ormqr_blocked_benchmark syev_benchmark
   ```

2. Run the sweep (see Quick start).
3. Inspect `build/tuning/profile.json` (`per_case_best`, `best`).
4. A/B each candidate with the env override at the **consumer** benchmark, then port the constant
   into `include/batchlas/tuning_params.hh` and rebuild.

There is no CMake target that writes the header. Drive the scripts directly.

## Why a kernel win can be an end-to-end loss

Measured 2026-08-06, CUDA/float:

| Block size | ormqr kernel, n = 1024 | syev n = 1024 | gesvd_blocked n = 512 |
|---|---|---|---|
| 16 (shipped) | 536 µs | 899 µs | 940 µs |
| 48/56 (tuned) | 238 µs (2.16× faster) | 905 µs (inert) | 1045 µs (11 % slower) |

Three mechanisms explain it. None is visible from `ormqr_blocked_benchmark`.

1. **Aliasing.** `gesvd_blocked.cc` reads `ormqr_block_size_for_n` three times: twice for the
   ormbr backtransforms and once as `gebrd_block_size`, a different kernel. At n = 512, batch 256,
   vectors on:

   | Stage | nb = 16 | nb = 48 |
   |---|---|---|
   | `gesvd.gebrd` | 229.7 ms | 259.2 ms (+12.8 %) |
   | `gesvd.apply_left_backtransform` | 18.35 ms | 14.87 ms (−19 %) |
   | `gesvd.apply_right_backtransform` | 18.25 ms | 15.12 ms (−17 %) |

   The uses have opposite gradients, and gebrd is 6.3× the larger term. Aliasing pins the steep
   knob at the flat knob's optimum.
2. **Shadowing.** `sytrd_sy2sb.cc`'s `sy2sb_ormqr_block_size_hint` returns `kd` when
   `n >= 1024 && batch >= 32`, bypassing the table. The kernel trace is bit-identical at nb 16 and 56.

   | | nb = 16 | nb = 56 |
   |---|---|---|
   | `BATCHLAS_SY2SB_ORMQR_NB=off` | 1094.9 µs | 905.4 µs (1.21×) |
   | default (gate on) | 898.6 µs | 905.0 µs (inert) |

3. **Wrong bucket key.** `ormqr_block_size_for_n` keys on `A.rows()`, but the WY block width is
   bounded by the reflector count `k`. In sy2sb they differ by orders of magnitude (`k = kd = 32`).

What to do:

- A/B at the consumer with the env override before editing any constant.
- If a knob changes nothing, check whether it is read: compare `BATCHLAS_KERNEL_TRACE=1` call
  counts. A bit-identical trace means shadowing, not insignificance.
- Do not tune a shared constant through a benchmark that takes it as an explicit argument.
- Split aliased constants. A `GEBRD_BLOCK_SIZE_*` would keep gebrd at 16 while ormqr takes 48:
  about 2.3 % on gesvd-with-vectors, and it unblocks the 2.16× elsewhere. This is a code change.

## Bucketed model

The header is bucket-first; there is no single global block-size constant.

| Constant family | Accessor |
|---|---|
| `ORMQR_BLOCK_SIZE_<BUCKET>` | `ormqr_block_size_for_n(n)` |
| `SYTRD_BLOCK_SIZE_<BUCKET>` | `sytrd_block_size_for_n(n)` |
| `LATRD_LOWER_PANEL_WG_HINT_<BUCKET>` | `latrd_lower_panel_wg_hint_for_n(n)` |
| `SYTRD_FUSE_PANEL_UPDATE_<BUCKET>` | `sytrd_fuse_panel_update_for_n(n)` |
| `STEDC_{RECURSION_THRESHOLD, MERGE_VARIANT, THREADS_PER_ROOT, WG_MULTIPLIER}_<BUCKET>` | `stedc_*_for_n(n)` |

`<BUCKET>` is `TINY`, `SMALL`, `MEDIUM`, `LARGE` or `XLARGE`:

| Bucket | n |
|---|---|
| `TINY` | ≤ 64 |
| `SMALL` | 65–128 |
| `MEDIUM` | 129–256 |
| `LARGE` | 257–512 |
| `XLARGE` | > 512 |

Bucketed values come from each case winner (`per_case_best`). Missing ranges use the configured
fallback values.

STEDC tuning cases start at n = 64. Leaf sizes n ≤ 32 are not tuned separately. The default space
pre-tunes `recursion_threshold` at n = 64 only, then reuses it for the merge-variant and workgroup
sweep. At runtime the threshold is clamped to the local subproblem size at each level.

`syev` tunes `nb`, the LATRD lower-panel wg hint and the fused panel update as one tuple, against
end-to-end eigensolver time.

## Dependencies between benches

- **syev overrides sytrd_blocked.** When a `syev` entry exists, `SYTRD_BLOCK_SIZE_*` is derived
  from syev's coupled `nb`. Syev's `nb` grid must be at least as wide as sytrd_blocked's, or the
  standalone winner is unreachable.
- **stedc's constants are global.** Tune stedc first; syev's and gesvd's tridiagonal solves inherit
  them.
- **ormqr and gebrd must stay separate.** They shared a constant until 2026-08-06; see the split
  note in `tuning_params.hh`.
- **latrd_lower_panel is a fallback** for sizes syev's coupled wg hint does not reach.

## Untuned parameters

Shares on CUDA/float, RTX 4090 (from `BATCHLAS_KERNEL_TRACE=1` and `BATCHLAS_GESVD_PROFILE=1`):

| Kernel | Share | Tuned by |
|---|---|---|
| `syev_two_stage.sb2st_hh` | 52.9 % of syev at n = 1024 | `sb2st` bench; the heuristic already wins, so the constants stay 0 |
| `gesvd.gebrd` | 79–95 % of gesvd | `gesvd` bench |
| `ormqr_blocked.larft` + `pack_v_panel` | about 20 % of syev at n = 1024 | `ormqr_blocked`, plus `sy2sb` for the width that shadows it |
| Two-stage band width `kd` | sets the stage-1/chase balance | nothing (hand table); in `unwired.json` |
| Chase blocking factor | hardcoded 32 | nothing; in `unwired.json` |
| steqr CTA knobs | leaf of every stedc merge | nothing; in `unwired.json` |

To promote a parameter out of `unwired.json`, in order:

1. Add `<NAME>_{TINY..XLARGE}` constants and a `*_for_n` accessor to `tuning_params.hh`, with a
   `BATCHLAS_TUNE_*` env override.
2. Teach `generate_tuning_header.py` to derive them from the bench.
3. Make the production call site read the accessor instead of its hardcoded default. Skipping this
   step makes the whole exercise a no-op.

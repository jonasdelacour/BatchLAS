# potrs {#perf_potrs}

> **Status:** current · sm_89 float and complex<float> rows measured (RTX 4090, CUDA 13.2, DPC++ `/opt/dpcpp-cuda`, 2026-10-26); double, complex<double> and every sm_120 row transcribed

`potrs(A, B, uplo)` solves \f$AX = B\f$ from a Cholesky factor (contract:
[potrs and potri](potri.md#potrs-and-potri)). Two native families, no vendor family.

## potrs selection {#potrs-selection}

| family | driver | `can_run` (correctness only) |
|---|---|---|
| `fused` | `ops::getrs::potrs_fused_dispatch`: both substitutions in one launch, B in SLM (`getrs_fused.cc`) | `ops::potrs::fused_fits`: GPU, sub-group 32, `nrhs <= 8`, `n * nrhs <= getrs_fused_max_rhs_elems` of the SLM budget |
| `trsm` | two public `trsm` calls, each choosing its own kernel | always; the children refuse what they cannot take |

- Keys (`src/ops/potrs/choice.hh`): `dtype`, `uplo:exact` (the triangles run different bodies),
  `n:log:2`, `nrhs:log`, `batch:log`. Last resort `trsm`. A heterogeneous batch or `n`/`nrhs < 1`
  throws before selection; batch 0 is a no-op.
- `BATCHLAS_POTRS_ROUTE=fused|trsm` pins (a `fused` pin it cannot run throws); `BATCHLAS_SELECT_TRACE=1`
  prints the choice. The earlier "fused whenever it fits" rule cost up to 4.6x on sm_120 and 2.5x on
  sm_89.
- Tables: `tuned/potrs.{sm_89,sm_120}.txt` from `benchmarks/results/tuning/ledger/potrs.<device>.jsonl`
  via `scripts/sweep_to_table.py`. sm_89 float/cfloat are `deep` (spec `tools/tune/potrs_spec.cc`,
  factor made once by potrf, every arm verified by `solve_residual`); double from float, cdouble from
  cfloat and every sm_120 row are `transcribed`.
- **posv.** posv's `cta`/`blocked` are `potrf` + `fused`/`trsm`, bit for bit, but posv keeps its own
  table, which still ranks `cta` first where potrs measures `trsm` faster (float \f$n = 16\f$, \f$\mathrm{nrhs} = 8\f$).
  **Open debt:** the packed block solve moved posv's `cta` hash; posv needs a retune on both devices.

**Measured window** (sm_89 deep, 1252 rows, both triangles, batches 32-16384; full grid in the table
file):

- float: `fused` wins from \f$n = 17\f$ up (1.1-2.8x; one exception, Upper \f$n = 32\f$, nrhs 8, batch 2048,
  `trsm` 1.05x). Up to \f$n = 16\f$ `trsm` wins at \f$\mathrm{nrhs} \ge 3\f$ from \f$n = 4\f$ at batch \f$\ge 2048\f$ (up to 1.8x);
  `fused` takes every nrhs 1-2 cell.
- complex<float>: `fused` wins \f$n \ge 17\f$ at nrhs 1-4; `trsm` wins \f$24 \le n \le 128\f$ at nrhs 5-8 in a
  batch-dependent band (up to 1.39x), and at \f$\mathrm{nrhs} \ge 2\f$ starting at some \f$2 \le n \le 4\f$, up to \f$n = 16\f$ (up to 2.6x).
- Bracketing cells, float Lower, batch 16384 (slower/faster): \f$n = 16\f$, nrhs 8 `trsm` 1.78; \f$n = 17\f$,
  nrhs 8 `fused` 1.42. complex<float>: \f$n = 16\f$, nrhs 1 `trsm` 1.14; \f$n = 17\f$, nrhs 1 `fused` 2.72.
- The \f$n = 16 \to 17\f$ step is trsm's: its register-resident `cta` kernel moves from the 16 to the 32 bucket
  (`smallest_bucket_ge`, `trsm_native.cc`) and doubles its per-item time; the fused solve is smooth.

## The packed block solve {#the-packed-block-solve}

The fused kernel puts one work-group (64-1024 lanes, `getrs_fused_wg`) on each matrix and solves the
`nb = 16` diagonal blocks (below \f$n = 1024\f$) as a sub-group shuffle recurrence. Each right-hand side takes
a `w`-lane slot (`w` = a power of two \f$\ge\f$ `jb`, at least 4), up to `32 / w` columns at once, and only as
many slots as the round-robin needs (`potrs_slots`), so `nrhs <= nsg` keeps the unpacked schedule and
`NR <= 2` never packs. Arithmetic order is unchanged, so results are bit-identical to the unpacked
kernel. Gain over unpacked: 1.1-1.8x at nrhs 3-8, unchanged at nrhs 1-2 (sm_89, batch 8192,
2026-10-26).

Traps:

- **Idle lanes must not divide.** `0 / d` fails the fast-division check and takes the slow path every
  step (1.4-1.6x at \f$\mathrm{nrhs} = 1\f$); the guard `on && l == s` keeps idle lanes out.
- **Registers are occupancy.** Packing code in \f$\mathrm{NR} \le 2\f$ pushed complex<float> \f$\mathrm{NR} = 1\f$ from 40 to 44
  registers (12 to 10 warps per sub-partition, -9-10%), hence no packing there.
- `GetrsFusedRegs<T>::potrs` ([blackwell.md](blackwell.md)) under-charges the device image
  (complex<double> \f$\mathrm{NR} = 8\f$: 86 against 116 measured) but yields the same `sm89_max_work_group` cap at
  every admitted sm_89 launch; not re-probed on sm_120.

**Open debt:** `fused` still loses at \f$n \le 16\f$ for complex<float> and \f$n \le 32\f$ for \f$\mathrm{nrhs} \ge 6\f$: one
work-group per matrix caps residency at 24 matrices per SM, where `trsm`'s `cta` kernel puts several
matrices in a group. Closing it needs a thread-per-column fused solve (not written); the table routes
those cells to `trsm`.

## What the tests pin {#potrs-what-the-tests-pin}

`tests/potrs_tests.cc` (heavy cases in `potrs_tests_heavy`), every backend and type: each family pinned
and Auto on graded SPD factors, both triangles, \f$n\f$ across block edges (16/17, 33, 130), nrhs across
accumulator buckets and past the width ceiling, padded `ld`/stride with a poisoned triangle, factor
read-only, B's pads untouched; `fused`/`trsm` bit-equal to posv `cta`/`blocked`; a non-SPD item
isolated; the fused pin at the SLM ceiling and refused one row above; `reached` coverage rows; a
synthetic table whose winner moves with each key field; Auto against the shipped table; 1024 identical
items bit-identical. Each test's deliberate break (2026-10-26) is written above it.

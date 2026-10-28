# potrs {#perf_potrs}

> **Status:** current · sm_89 float and complex<float> rows measured (RTX 4090, CUDA 13.2, DPC++ `/opt/dpcpp-cuda`, 2026-10-26); double, complex<double> and every sm_120 row transcribed

`potrs(A, B, uplo)` solves `A X = B` from the Cholesky factor in the `uplo` triangle of `A` (the API and
its contract: [potrs and potri](potri.md#potrs-and-potri)). It is a routed op with two native families
and no vendor family, so the vendor-free build serves it.

## potrs selection {#potrs-selection}

| family | driver | `can_run` (correctness only) |
|---|---|---|
| `fused` | `sycl_getrs::potrs_fused_dispatch`: both substitutions in one launch, B resident in SLM (`getrs_fused.cc`, potrs body) | `ops::potrs::fused_fits`: GPU, sub-group size 32, `nrhs <= 8` (the widest accumulator), `n * nrhs` within the SLM capacity (`getrs_fused_max_rhs_elems` of the device's budget) |
| `trsm` | two public `trsm` calls (`L` then `L^H`, or `U^H` then `U`), each choosing its own kernel | always; the children refuse what they cannot take (a CPU queue without CBLAS has no route) |

- `src/ops/potrs/choice.hh`: keys `dtype`, `uplo:exact`, `n:log:2`, `nrhs:log`, `batch:log` (work
  ~ n^2 nrhs batch; `uplo` is a key because the two triangles run different kernel bodies and different
  trsm transposes). Last resort `trsm`. A heterogeneous batch and `n` or `nrhs` below 1 throw before
  selection; batch 0 is a no-op under any pin.
- Pin with `BATCHLAS_POTRS_ROUTE=fused|trsm` (a `fused` pin on a shape it cannot take throws "cannot run
  this shape"); `BATCHLAS_SELECT_TRACE=1` prints the choice. Before 2026-10-26 `potrs` had no `Op`: it
  took `fused` wherever `fused_fits` held, a capacity predicate deciding speed, which cost up to 4.6x on
  sm_120 (posv deep rows below) and 2.5x on sm_89 (float n = 16, nrhs = 8).
- Tables: `tuned/potrs.sm_89.txt` and `tuned/potrs.sm_120.txt`, generated from
  `benchmarks/results/tuning/ledger/potrs.<device>.jsonl` by `scripts/sweep_to_table.py`. The sm_89
  float and cfloat rows are a `batchlas_tune potrs --tier deep` run (spec `tools/tune/potrs_spec.cc`:
  the factor made once, untimed, by the public potrf; every arm verified with `solve_residual`). The
  double and cdouble rows (FP64 is not tuned on a 1:64 part, agent guide §10) and every sm_120 row
  (no sm_120 box measured) are `transcribed` from those runs: double from float, cdouble from cfloat,
  each sm_120 dtype from the same-precision-class sm_89 dtype. A retune on the device replaces them.
- **posv.** posv's `cta` and `blocked` are `potrf` + this `fused` solve and `potrf` + this `trsm` solve,
  bit for bit (`FusedMatchesPosvCtaAndTrsmMatchesPosvBlockedBitForBit`), so the measured choice applies to
  posv's `cta` against `blocked` too. posv keeps its own families and table (its rows time the whole
  composition, potrf included, and add `tiny`); its sm_89 table is still the transcribed old order,
  which ranks `cta` before `blocked` at shapes where the rows below measure `trsm` faster (float
  n = 16, nrhs = 8). The kernel change below moves posv's `cta` family hash, so posv's sm_120 deep table
  now reads stale: a posv retune on both devices is the follow-up, not a potrs table read from posv.

The measured window on sm_89 (deep rows, batch at the CU ladder 32, 256, 2048, 16384):

Each cell: the winner (`f` fused, `t` trsm) and the slower family's time over the faster one's, Lower,
batch 16384 (the ladder's saturating class; n >= 128 there is over the tuner's 4 GiB cell cap, the batch
2048 rows cover it). Off-lattice n (13-15, 17-20, 26-39, ...) are the tuner's frontier refinement around
each flip. The Upper rows and the other batches (32, 256, 2048) are in the table file.

float, Lower, batch 16384:

| n \ nrhs | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 1 | f 2.02 |  | f 1.70 |  |  | f 1.50 |  | f 1.38 |
| 2 |  |  | f 1.60 | f 1.52 |  | f 1.37 |  | f 1.34 |
| 3 |  |  |  |  |  |  |  | f 1.06 |
| 4 | f 1.57 | f 1.51 | f 1.37 | f 1.25 |  | f 1.10 | f 1.02 | t 1.03 |
| 5 |  |  |  |  |  | t 1.12 |  |  |
| 6 |  |  |  | f 1.03 | t 1.13 | t 1.17 |  |  |
| 7 |  |  |  | t 1.09 |  |  |  |  |
| 8 | f 1.29 | f 1.17 | t 1.03 | t 1.11 |  | t 1.35 |  | t 1.57 |
| 9 |  |  |  | f 1.50 |  |  |  |  |
| 10 |  |  |  | f 1.39 |  |  |  |  |
| 12 | f 1.73 | f 1.57 | f 1.37 | f 1.23 | t 1.19 | t 1.24 |  | t 1.42 |
| 16 | f 1.48 | f 1.32 | f 1.11 | f 1.00 | t 1.49 | t 1.56 |  | t 1.78 |
| 17 |  |  |  |  |  | f 1.63 |  | f 1.42 |
| 18 |  |  |  |  |  | f 1.59 |  | f 1.40 |
| 20 |  |  |  |  |  | f 1.49 |  | f 1.30 |
| 24 | f 2.84 |  | f 2.17 | f 1.99 |  | f 1.33 |  | f 1.20 |
| 32 |  |  | f 1.82 | f 1.65 |  | f 1.07 |  | t 1.01 |
| 48 | f 2.30 |  | f 1.95 |  |  | f 1.33 |  | f 1.27 |
| 64 |  |  |  |  |  |  |  | f 1.52 |
| 96 | f 2.18 |  | f 2.41 |  |  | f 1.85 |  | f 1.82 |

complex<float>, Lower, batch 16384:

| n \ nrhs | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 1 | f 1.69 |  | f 1.37 |  |  | f 1.21 |  | f 1.16 |
| 2 | f 1.44 | f 1.35 | f 1.19 | f 1.17 |  | f 1.01 |  | t 1.02 |
| 3 |  | f 1.11 | t 1.05 | t 1.12 |  | t 1.17 |  | t 1.42 |
| 4 | f 1.04 | t 1.06 | t 1.20 | t 1.26 |  | t 1.46 |  | t 1.67 |
| 5 | t 1.09 |  |  |  |  |  |  |  |
| 6 | t 1.19 |  |  |  |  |  |  |  |
| 8 | t 1.35 | t 1.53 | t 1.71 |  |  |  |  |  |
| 9 | f 1.32 |  |  |  |  |  |  |  |
| 10 | f 1.22 |  |  |  |  |  |  |  |
| 12 | f 1.06 | t 1.06 | t 1.21 | t 1.31 |  | t 2.18 |  | t 2.41 |
| 13 | t 1.00 |  |  |  |  |  |  |  |
| 14 | t 1.06 |  |  |  |  |  |  |  |
| 16 | t 1.14 | t 1.29 | t 1.46 | t 1.55 |  | t 2.36 |  | t 2.44 |
| 17 | f 2.72 | f 2.46 | f 2.15 | f 2.00 |  | f 1.25 |  | f 1.17 |
| 18 | f 2.62 | f 2.41 | f 2.08 | f 1.89 |  | f 1.20 |  | f 1.13 |
| 20 | f 2.46 | f 2.18 | f 1.88 | f 1.71 |  | f 1.12 |  | f 1.06 |
| 24 | f 2.19 | f 1.94 | f 1.68 | f 1.55 |  | f 1.09 |  | t 1.01 |
| 26 |  |  |  |  |  | t 1.01 | t 1.07 | t 1.11 |
| 28 |  |  |  |  |  | t 1.10 |  | t 1.18 |
| 32 | f 1.97 | f 1.73 | f 1.50 | f 1.38 | t 1.10 | t 1.13 |  | t 1.18 |
| 35 |  |  |  |  |  | f 1.39 |  | f 1.32 |
| 39 |  |  |  |  |  | f 1.40 |  | f 1.33 |
| 48 | f 2.06 |  | f 1.73 | f 1.63 |  | f 1.12 |  | f 1.05 |
| 64 |  |  |  |  |  | f 1.19 |  | f 1.09 |
| 96 | f 2.07 |  | f 2.01 |  |  | f 1.46 |  | f 1.24 |

Reading, over all 1252 deep rows of the table (both triangles, every batch): `fused` wins from n = 17 up
(1.1-2.8x) in float, where one of its 368 rows ranks `trsm` first (Upper n = 32, nrhs 8, batch 2048,
1.05x), and in complex<float> at nrhs 1-4. In complex<float> at nrhs 5-8 `trsm` holds the other 86 of
the 87 deep rows with n >= 17 that rank it first: n = 28-128 at batch 32 and 256, 32-96 at 2048 and
24-32 at 16384, up to 1.39x (Upper n = 128, nrhs 8, batch 32). Up to n = 16, `trsm` wins float at
nrhs >= 3 from n = 4 at batch 2048 and 16384 only (up to 1.8x; `fused` takes every float cell at nrhs
1-2), and complex<float> at nrhs >= 2 from n = 2-4 (up to 2.6x at n = 12-16, nrhs 6-8, batch 16384) and
at nrhs = 1 for n = 5, 6, 8, 14 and 16 at batch 16384 and Upper n = 16 at 2048 (up to 1.41x). The step at n = 16/17 is trsm's: both orders run its
register-resident `cta` kernel (`BATCHLAS_SELECT_TRACE`), which moves from the 16 to the 32 bucket there
(`smallest_bucket_ge`, `trsm_native.cc`) and doubles its time per item, while the fused solve is smooth in n.

## The packed block solve {#the-packed-block-solve}

The fused kernel puts one work-group (64 to 1024 lanes, `getrs_fused_wg`) on each matrix and solves the
diagonal blocks (`nb = 16` below n = 1024) as a sub-group shuffle recurrence, lane `i` owning row `i`.
Before 2026-10-26 the right-hand sides were dealt round-robin over the sub-groups and each sub-group
solved its columns one after another with only `jb` of its 32 lanes busy: at `nrhs = 8` and `wg = 64`
that is four serial 16-step recurrences per sub-group per block, half the lanes idle. That, not the
capacity, is why `fused` lost at `nrhs = 8`.

Now each column takes a `w`-lane slot of the sub-group (`w` the power of two >= `jb`, at least 4), up to
`32 / w` columns at once, and only as many slots as the round-robin needs (`potrs_slots`), so
`nrhs <= nsg` keeps the old schedule. Each column's arithmetic and its order are unchanged, so the
solution is the old kernel's bit for bit (the posv `cta` comparison, and every residual and saturating
guard, pass). Two measured traps on the way, both kept out:

- **An idle lane must not divide.** The first version let lanes of an unused slot run the recurrence on
  zeros. `0 / d` fails the fast-division check (`FCHK`) and takes the slow subroutine every step:
  fused at `nrhs = 1` ran 1.4-1.6x slower (float n = 16, 32, 64). The guard `on && l == s` keeps idle
  lanes out of the division.
- **Registers are occupancy.** Packing code in the NR = 1 and NR = 2 kernels raised complex<float> NR = 1
  Lower from 40 to 44 registers, which drops a 64-lane work-group from 12 to 10 warps per SM sub-partition
  (`resident::sm89_max_work_group`) and cost 9-10% at nrhs = 1. `NR <= 2` never packs (every work-group
  has at least two sub-groups), so those kernels fold to the unpacked loop; their ptxas counts are the
  old ones again.

ptxas counts (sm_89, `cuobjdump -res-usage` of the device image, Lower / Upper, NR 1, 2, 4, 8), before:
float 37/40, 40/40, 55/45, 80/60; complex<double> 54/56, 52/58, 58/60, 92/116. After: float 37/40, 40/40,
45/45, 64/58; complex<double> 56/56, 50/54, 56/56, 90/116. The `GetrsFusedRegs<T>::potrs` row in
`getrs_fused.cc` ([blackwell.md](blackwell.md)) charges less than this build's device image holds, before
and after alike (complex<double> NR = 8: 86 against 116); it still gives
the same `sm89_max_work_group` cap as the measured counts at every launch the capacity admits on sm_89
(complex<double> NR = 8 tops out at n = 1004 for nrhs = 5, `wg = 512`, the 120-register cap), and it was
not re-probed on sm_120.

Fused before and after against `trsm`, sm_89 GPU 1 (headless), Lower, batch 8192, `batchlas_tune`
custom protocol (16 reps x 2 passes, L2 scrubbed, interleaved arms, NVML guard; medians of pass 2, ms).
The before arm is the same binary with the parent commit's libraries first on `LD_LIBRARY_PATH`.

float:

| n | nrhs | fused before (ms) | fused after (ms) | trsm (ms) | before / after | trsm / fused after | faster |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.0171 | 0.0163 | 0.0270 | 1.05 | 1.65 | fused |
| 4 | 2 | 0.0170 | 0.0181 | 0.0267 | 0.94 | 1.48 | fused |
| 4 | 3 | 0.0222 | 0.0194 | 0.0258 | 1.14 | 1.33 | fused |
| 4 | 4 | 0.0236 | 0.0206 | 0.0259 | 1.15 | 1.26 | fused |
| 4 | 6 | 0.0361 | 0.0238 | 0.0267 | 1.52 | 1.12 | fused |
| 4 | 8 | 0.0418 | 0.0262 | 0.0266 | 1.59 | 1.02 | fused |
| 8 | 1 | 0.0200 | 0.0199 | 0.0262 | 1.00 | 1.32 | fused |
| 8 | 2 | 0.0217 | 0.0222 | 0.0264 | 0.98 | 1.19 | fused |
| 8 | 3 | 0.0314 | 0.0256 | 0.0265 | 1.22 | 1.04 | fused |
| 8 | 4 | 0.0344 | 0.0277 | 0.0264 | 1.24 | 0.95 | trsm |
| 8 | 6 | 0.0554 | 0.0348 | 0.0274 | 1.59 | 0.79 | trsm |
| 8 | 8 | 0.0688 | 0.0399 | 0.0273 | 1.73 | 0.68 | trsm |
| 16 | 1 | 0.0320 | 0.0326 | 0.0481 | 0.98 | 1.47 | fused |
| 16 | 2 | 0.0366 | 0.0366 | 0.0491 | 1.00 | 1.34 | fused |
| 16 | 3 | 0.0556 | 0.0442 | 0.0495 | 1.26 | 1.12 | fused |
| 16 | 4 | 0.0619 | 0.0493 | 0.0504 | 1.26 | 1.02 | fused |
| 16 | 6 | 0.1050 | 0.0754 | 0.0514 | 1.39 | 0.68 | trsm |
| 16 | 8 | 0.1323 | 0.0871 | 0.0525 | 1.52 | 0.60 | trsm |
| 32 | 1 | 0.0663 | 0.0673 | 0.1595 | 0.98 | 2.37 | fused |
| 32 | 2 | 0.0738 | 0.0762 | 0.1603 | 0.97 | 2.10 | fused |
| 32 | 3 | 0.1139 | 0.0930 | 0.1620 | 1.23 | 1.74 | fused |
| 32 | 4 | 0.1264 | 0.1025 | 0.1623 | 1.23 | 1.58 | fused |
| 32 | 6 | 0.2163 | 0.1553 | 0.1646 | 1.39 | 1.06 | fused |
| 32 | 8 | 0.2703 | 0.1798 | 0.1683 | 1.50 | 0.94 | trsm |
| 64 | 1 | 0.1612 | 0.1627 | 0.4005 | 0.99 | 2.46 | fused |
| 64 | 2 | 0.1780 | 0.1782 | 0.4002 | 1.00 | 2.25 | fused |
| 64 | 3 | 0.2507 | 0.2173 | 0.4228 | 1.15 | 1.95 | fused |
| 64 | 4 | 0.2757 | 0.2348 | 0.4253 | 1.17 | 1.81 | fused |
| 64 | 6 | 0.4652 | 0.3396 | 0.4890 | 1.37 | 1.44 | fused |
| 64 | 8 | 0.5682 | 0.3873 | 0.5375 | 1.47 | 1.39 | fused |
| 128 | 1 | 0.5815 | 0.5747 | 1.1290 | 1.01 | 1.96 | fused |
| 128 | 2 | 0.5965 | 0.5990 | 1.1920 | 1.00 | 1.99 | fused |
| 128 | 3 | 0.6864 | 0.6507 | 1.2580 | 1.05 | 1.93 | fused |
| 128 | 4 | 0.7330 | 0.6851 | 1.2890 | 1.07 | 1.88 | fused |
| 128 | 6 | 1.0700 | 0.8561 | 1.4210 | 1.25 | 1.66 | fused |
| 128 | 8 | 1.2710 | 0.9542 | 1.5560 | 1.33 | 1.63 | fused |

complex<float>:

| n | nrhs | fused before (ms) | fused after (ms) | trsm (ms) | before / after | trsm / fused after | faster |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.0283 | 0.0241 | 0.0275 | 1.17 | 1.14 | fused |
| 4 | 2 | 0.0258 | 0.0303 | 0.0265 | 0.85 | 0.87 | trsm |
| 4 | 3 | 0.0379 | 0.0351 | 0.0280 | 1.08 | 0.80 | trsm |
| 4 | 4 | 0.0396 | 0.0303 | 0.0265 | 1.31 | 0.87 | trsm |
| 4 | 6 | 0.0545 | 0.0347 | 0.0276 | 1.57 | 0.80 | trsm |
| 4 | 8 | 0.0683 | 0.0385 | 0.0267 | 1.77 | 0.69 | trsm |
| 8 | 1 | 0.0359 | 0.0349 | 0.0296 | 1.03 | 0.85 | trsm |
| 8 | 2 | 0.0391 | 0.0402 | 0.0290 | 0.97 | 0.72 | trsm |
| 8 | 3 | 0.0615 | 0.0457 | 0.0283 | 1.35 | 0.62 | trsm |
| 8 | 4 | 0.0658 | 0.0479 | 0.0287 | 1.37 | 0.60 | trsm |
| 8 | 6 | 0.0956 | 0.0578 | 0.0286 | 1.66 | 0.50 | trsm |
| 8 | 8 | 0.1232 | 0.0690 | 0.0306 | 1.78 | 0.44 | trsm |
| 16 | 1 | 0.0677 | 0.0679 | 0.0645 | 1.00 | 0.95 | trsm |
| 16 | 2 | 0.0737 | 0.0739 | 0.0616 | 1.00 | 0.83 | trsm |
| 16 | 3 | 0.1188 | 0.0864 | 0.0623 | 1.38 | 0.72 | trsm |
| 16 | 4 | 0.1271 | 0.0920 | 0.0632 | 1.38 | 0.69 | trsm |
| 16 | 6 | 0.1863 | 0.1492 | 0.0664 | 1.25 | 0.45 | trsm |
| 16 | 8 | 0.2400 | 0.1685 | 0.0718 | 1.42 | 0.43 | trsm |
| 32 | 1 | 0.1320 | 0.1320 | 0.2424 | 1.00 | 1.84 | fused |
| 32 | 2 | 0.1546 | 0.1533 | 0.2449 | 1.01 | 1.60 | fused |
| 32 | 3 | 0.2435 | 0.1789 | 0.2470 | 1.36 | 1.38 | fused |
| 32 | 4 | 0.2615 | 0.1965 | 0.2469 | 1.33 | 1.26 | fused |
| 32 | 6 | 0.3938 | 0.3106 | 0.2581 | 1.27 | 0.83 | trsm |
| 32 | 8 | 0.5022 | 0.3554 | 0.2768 | 1.41 | 0.78 | trsm |
| 64 | 1 | 0.3281 | 0.3217 | 0.6471 | 1.02 | 2.01 | fused |
| 64 | 2 | 0.3557 | 0.3569 | 0.6789 | 1.00 | 1.90 | fused |
| 64 | 3 | 0.5132 | 0.4109 | 0.7017 | 1.25 | 1.71 | fused |
| 64 | 4 | 0.5626 | 0.4478 | 0.7201 | 1.26 | 1.61 | fused |
| 64 | 6 | 0.8273 | 0.6775 | 0.7676 | 1.22 | 1.13 | fused |
| 64 | 8 | 1.1040 | 0.8064 | 0.8354 | 1.37 | 1.04 | fused |
| 128 | 1 | 1.2330 | 1.2310 | 2.1520 | 1.00 | 1.75 | fused |
| 128 | 2 | 1.3000 | 1.2910 | 2.2650 | 1.01 | 1.75 | fused |
| 128 | 3 | 1.4570 | 1.3190 | 2.3470 | 1.10 | 1.78 | fused |
| 128 | 4 | 1.6120 | 1.4150 | 2.4330 | 1.14 | 1.72 | fused |
| 128 | 6 | 2.4860 | 2.0950 | 2.6200 | 1.19 | 1.25 | fused |
| 128 | 8 | 3.3760 | 2.5180 | 2.8320 | 1.34 | 1.12 | fused |

Reading: nrhs 3-8 gain 1.1-1.8x, nrhs 1-2 are unchanged within the n = 4 noise (about 25 us cells).
`fused` still loses at n <= 16 for complex<float> and at n <= 32 for nrhs >= 6: one work-group per matrix
caps residency at 24 matrices per SM (64-lane groups), while `trsm`'s `cta` kernel puts one thread on each
column and several matrices in a group. Closing that is a different kernel (a thread-per-column fused
solve), not a schedule change, and is not written; the table routes those cells to `trsm`.

The sm_120 evidence that started this (`tuned/posv.sm_120.txt`, deep, posv = potrf + solve, ms):
float n = 16, nrhs = 8, batch 8192: `blocked` 0.02951 against `cta` 0.1350 (4.6x, line 349); complex<double>
n = 8, nrhs = 8, batch 8192: 0.1537 against 2.374 (15x, line 7015); float n = 64, nrhs = 1, batch 8192:
`cta` 0.3881 against `blocked` 0.5036 (line 753). Those rows predate the packed block solve.

## What the tests pin {#potrs-what-the-tests-pin}

`tests/potrs_tests.cc` (the potrs cases moved out of `potri_tests.cc`), every backend and type, heavy
cases in `potrs_tests_heavy`:

- each family pinned, and Auto, on graded SPD factors (`fill_graded_hermitian`, condition 1e6 double,
  1e3 single), both triangles, n through the block edge (16/17, 33, 130) and nrhs through the accumulator
  buckets and past the width ceiling, padded `ld` and stride with a poisoned unreferenced triangle; the
  factor read-only and B's pads untouched; a second right-hand side on the same factor;
- `fused` bit for bit equal to posv `cta`, `trsm` to posv `blocked`; a non-SPD item flagged by posv with
  the other items undisturbed;
- the fused pin straddling both ceilings, launched at the SLM ceiling (`n = cap / 8`, nrhs 8) and refused
  by selection one row above; `reached` coverage rows naming each pinned family; a synthetic table whose
  winner moves with each key field; Auto against the shipped table; 1024 identical items bit-identical to
  item 0 for each family at one block, several blocks and a four-sub-group work-group.

The deliberate breaks, each run on 2026-10-26 with its red set, are written above the tests they turned
red: the launch ignoring the choice, `key_of` dropping `nrhs`, `fused_fits` dropping the width cap, the
Upper block staged from the poisoned triangle, the forward barrier dropped, and the packed slot losing its
column offset.

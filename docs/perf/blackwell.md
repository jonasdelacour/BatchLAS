# Blackwell (sm_120) retune {#perf_blackwell}

> **Status:** current · RTX PRO 6000 Blackwell Max-Q (sm_120), threadripper02, icpx 2026.0, CUDA 13.2 · measured 2026-09-30 to 2026-10-05

The sm_120-specific facts that still drive code: the icpx fp-model fix and its guard, the device
key, and the kernel changes (spmm pair loads, potrf LPanel vector sB, fused getrs, trsm ladder cap and
sub-group Left kernel, small wide gemm tiles) that came out of the sm_120 campaign. sm_120 has 188 SMs,
128 MB L2, ~1.8 TB/s GDDR7 and a 300 W cap; cuBLAS SGEMM reaches 43.9-45.8 TFLOP/s, about 43% of the
FP32 peak.

## What sm_120 selects

Routing on sm_120 has no architecture predicates. Every op reads `tuned/<op>.sm_120.txt`
([tuned/README.md](../../tuned/README.md), "The sm_120 deep tables"); gemm float and complex<float>
were re-measured 2026-10-21 ([gemm.md](gemm.md)). The windows the campaign measured on the
`worktree-blackwell-tuning` branch are not code; where a section below quotes one, the deep table
decides instead. Branch results that never shipped:

| Branch change | Result on sm_120 | Status |
|---|---|---|
| syev float blocked/two_stage edge 768 (sm_89: 448); sytrd nb 16 and ormqr WY 32 above \f$n = 512\f$; per-type latrd grid min-n (float 321, double/cfloat 257, cdouble 129) | float 512 b512 148.1 -> 103.9 ms; cfloat 1024 b64 365.5 -> 264.2 ms | not on main; the deep `syev.sm_120` table decides; benchviz `sytrd_nb` still mirrors sm_89's 48 |
| potrf Blocked \f$W = 64\f$, \f$n_b = 64\f$ up to \f$n = 512\f$ (sm_89: 128/128) | float vendor/blocked 0.78 -> 0.95 at n320 b1024, 1.26 -> 1.30 at n1024 b128 | not on main; `potrf_blocked_params` has no architecture term |
| getrs `Algorithm::Tiny` (one sub-group partition per matrix, \f$n \le 32\f$, single precision) | tiny/vendor 0.20-0.58 at b32768 | not on main (`getrs_tiny.cc` absent) |
| geqrf skinny register leg (\f$n \le 8\f$) and tile-bytes leaf width (float 72 B, cfloat 96 B per item, tiles \f$\le 16\f$ KiB, batch \f$\ge 2048\f$) | float \f$64 \times 8\f$ native/vendor 0.34; 106-cell native geomean float 1.04 -> 0.54 | not on main (`geqrf_skinny.cc` absent) |

## icpx fast fp-model

icpx 2026.0 defaults to `-fp-model=fast`. Device functions then carry a preserve-sign denormal mode,
while the libspirv builtins linked through `-mlink-builtin-bitcode` are IEEE. LLVM does not inline
across that mismatch, so `__spirv_ocl_fma` (every `sycl::fma`), `__spirv_BuiltIn*Id` and
`__spirv_ControlBarrier` become out-of-line calls.

Fix: `-ffp-model=precise` for IntelLLVM DPC++ builds (`cmake/BatchLASOptions.cmake`, the
`batchlas_build_options` block). `-fdenormal-fp-math=ieee` alone restores inlining but leaves reassoc
and afn on, which agent-guide §11 rules out. The self-built dpcpp-cuda compiler already defaults to
precise.

- `GemmRegister64x64K16WideKernel<complex<float>,true>` SASS: 1031 `CALL.REL` under the icpx default,
  0 calls and 1152 inline FFMA under precise. 4472 of 4539 kernels in the icpx images contained calls.
- Only the flag toggled (fast -> precise): cfloat wide kernel \f$128^3\f$ b8192 29.0 -> 3.75 ms; batchlas/vendor
  cfloat gemm \f$128^3\f$ 6.54 -> 0.98; potrf cfloat LPanel \f$n = 256\f$ 13.87 -> 2.49 ms.

### cfloat spmm under the precise fp model

Precise slowed one kernel: the native CSR gather (`SpmmGatherKernel`,
`src/ops/spmm/kernels/spmm_native.cc`) for cfloat, 5.51 -> 6.49 ms at \f$n = 16384\f$, 16 nnz/row, nrhs=64,
batch=32 (cuSPARSE 5.76). The kernel is L2-bound; precise if-converts the 8-column block and issues the
two 4-byte halves of every complex element at once, so both halves miss in L1 (ncu: L2 sectors 637M
fast, 791M precise).

Fix: load and store a complex element as one `sycl::vec<R, 2>` (`LDG.E.64` for cfloat, `LDG.E.128` for
cdouble). That needs `2*sizeof(R)` alignment, which `std::complex<R>` does not promise, so the launcher
checks A's values, B and C and falls back to the scalar kernel for a view one R off. cuSPARSE rejects
that case, so the misaligned tests skip unless the native route is pinned. Not gated by architecture.
Result: cfloat 16384/64/b32 6.49 -> 5.23 ms (cuSPARSE 5.74); cdouble 16384/64/b16 4.55 -> 3.81 (8.83);
geomean over 16 cells cfloat 0.85, cdouble 0.84, float 1.00.

## Device-call guard

`scripts/check_device_calls.py` (ctest `device_calls_tests`, label `util`; also
`BATCHLAS_BUILD_DIR=build .github/ci/run_local_checks.sh`) guards the fp-model fix. A regression changes
only timings, not test results, so this check is the only signal.

It walks each `build/src/libbatchlas_*.so` for embedded CUDA fatbins (magic `0xBA55ED50`), runs
`cuobjdump -ptx` on each, and fails if any `.entry` calls a `__spirv_*` or `__clc_*` function, directly
or through the local `.func` helpers it calls. It also fails if one of
`GemmRegister64x64K16WideKernel<complex<float>,true>`, `PotrfLpanelKernel` or `TrsmCtaKernel` is
missing. It exits 77 (ctest skip) when there is no NVPTX image or no `cuobjdump`. Calls to non-inlined
library device functions (`larfg`, `sec_solve_roc`, gesvdj and syrk/trmm tile lambdas) are reported but
do not fail.

> **Note:** The check reads PTX, not SASS. A correct precise build still has SASS `CALL.REL` in
> `TrsmCtaKernel` (ptxas's IEEE division and sqrt slow paths); a PTX `call` names its callee.

Proof that it can fail: `trsm_native.cc`, `potrf_lpanel.cc` and `gemm_kernels.cc` device-linked with
`-ffp-model=fast` appended give 248 of 251 kernels with builtin calls, exit 1; as built, 0 calls, exit 0.

| Alternative | Result | Verdict |
|---|---|---|
| `__builtin_fma{,f}` in `src/sycl/device_scalar.hh` instead of `sycl::fma` | under fast, wide-cfloat calls 1031 -> 7, but barrier and id calls remain; under precise it changes the SASS of 158 kernels | rejected; the flag is the fix |

## Compute-capability key

`Device::cuda_compute_capability()` returns `major*10+minor`, parsed from the CUDA device's SYCL version
string and memoized per device; 0 means "not CUDA". `select::describe` names a CUDA device
`sm_<cc>` (`src/select/select.cc`), which picks `tuned/<op>.sm_<cc>.txt`; a device with no table of its
own borrows the nearest one with a one-time warning.

`is_sm120_family` (\f$120 \le \mathrm{cc} < 130\f$, `include/batchlas/util/sycl-device-queue.hh`) is not a routing
predicate. Its one use is the per-architecture register charge of the fused getrs launch
([LU getrs fused](#lu-getrs-fused)).

`tests/util_device_queue_tests.cc` checks the value against the CUDA runtime's
`prop.major*10+prop.minor` per NVIDIA GPU, paired by ordinal (`DeviceTest.GetDevices` requires
`devices[i].idx == i`); CPU and non-NVIDIA devices must return 0; the family window is tested at
119/120 and 129/130. Deliberate break (cc = major): exactly the two `CudaComputeCapability*` cases go red.

## potrf LPanel vector sB

`potrf_lpanel_body`'s left-looking update reads each sB column as 16-byte `sycl::vec<float,4>` loads
through a local-space pointer (it was one scalar shared load per FMA; cfloat compiled to generic
`LD.E`). Each `rS[i]` still accumulates `kk` in ascending order, so the arithmetic is unchanged; the
change is architecture-independent. LPanel pinned, vendor/LPanel BASE -> new: float n256 0.95 -> 1.15,
n512 0.76 -> 0.90; cfloat n128 1.00 -> 1.18, n256 0.78 -> 0.99.

> **Warning:** sB is stored as the scalar type and read back as `sycl::vec<float,4>`. This type punning
> is correct only because barrier B1 sits between the stores and the loads. A double-buffered sB that
> drops or moves B1 must store through the vector type too.

## LU getrs fused

`getrs_fused.cc` deals the \f$n_b \times n_b\f$ block-solve columns round-robin over the sub-groups (they all ran in
sub-group 0 before). Architecture-independent. CTA arm, before -> after, ms (vendor): float n64 b16384
nrhs 8 1.04 -> 0.77 (0.75); cfloat n512 b256 nrhs 8 1.92 -> 1.20 (1.40).

At NR == 1 the three fused kernels (getrs NoTrans, Trans, potrs) keep the sub-group-0 form, guard
`c_active = (NR != 1) || sgid == 0`: round-robin was 1-3% slower with one column to spread.

> **Warning:** The sub-group-0 guard is not covered by the suite. With `c_active = true` at NR == 1
> every sub-group solves column 0 at once and all four binaries stay green, because the race writes
> identical values. The guard exists because of the data race, not an observed failure.

The launch cap charges registers per architecture: `getrs_fused_regs_for` reads `GetrsFusedRegs120<T>`
on the sm_120 family and `GetrsFusedRegs<T>` (the sm_89 rows) elsewhere. In-library counts exceed the
sm_89 rows (float Trans nrhs 5..8 uses 86 registers), so the sm_89 row handed sm_120 a 768-lane launch
the driver refused (`FusedGetrsLaunchHoleAt48KiB`, `tests/getrf_tests.cc`).

### LU getrs fused sm89 rows

The round-robin change altered the kernels behind the sm_89 `GetrsFusedRegs` rows, which were probed on
the 4090 and could not be re-probed on sm_120. Standalone sm_89 ptxas counts, NR > 1, before -> now
(icpx 2026.0 / DPC++ 7.2): cdouble NoTrans nrhs 2 50 -> 54 / 50 -> 51; double NoTrans nrhs 5-8 55 -> 61 /
54 -> 54; double potrs nrhs 5-8 88 -> 92 / 70 -> 71; everything else moved by 0 or less. potrs has its own
sm_89 row (it was charged the max of NoTrans and Trans); a `static_assert` keeps it >= the old max.

Effect on the sm_89 cap (margin +8, allocation rounded to 8), for every non-sm_120 device including
`cuda_cc == 0`: cdouble NoTrans nrhs 2..4 1024 -> 896 lanes, double NoTrans nrhs 5..8 896 -> 768, double
potrs nrhs 5..8 768 -> 640; the fused width reaches a cap only at \f$n \ge 1026\f$. Only the potrs narrowing fixes
a launch the driver would refuse (icpx double potrs nrhs 5-8: 92 registers, \f$\lceil 24/4 \rceil \cdot 32 \cdot 96 = 18432 > 16384\f$ at 768 lanes).

Open: under icpx 2026.0 some standalone sm_89 counts already exceed the rows (double potrs nrhs 5-8 88 vs
72 + 8; cfloat Trans nrhs 5-8 70 vs 56 + 8). An icpx build for a 4090 needs its own probe.

## trsm

Measured on GPU 0, icpx 2026.0 precise, cuBLAS as the vendor; `trsm_benchmark` (`BM_TRSM<`: Left,
Lower, NoTrans, ld = n), `--warmup=3 --min_iters=8`, 3 alternating reps, medians.

### trsm V1 ladder cap

The V1 CTA kernel gives each lane one right-hand-side column, but lanes without a column still run the
whole \f$N\f$-step recurrence. Its work-group ladder {256, 128, 64, 32} took the first rung with
`batch * ceil(q / wg) >= 4 * CU`, which at saturated batch is always 256: a q=8 solve ran 248 dead lanes
per group. `trsm_v1_ladder_wg` (`src/ops/trsm/trsm_native.hh`) now skips a rung more than half of whose
lanes would have no column (`cand > 32 && cand / 2 >= q`). Architecture-independent and
correctness-neutral (the group barrier stays).

V1 alone, BASE -> capped, ms, batch 32768: float n32 q8 Left 0.994 -> 0.284; cfloat n32 q8 Left 1.551 ->
0.44; float n32 q8 Right 1.042 -> 0.306 (cuBLAS 0.165). The \f$q \ge 256\f$ shapes never reach a capped rung.

> **Note:** Measured on sm_120 only. On the 4090 small-q V1 geometry changes at saturated batch too;
> re-measure before quoting a 4090 number.

### trsm sub-group Left kernel

`ops::trsm::trsm_native_sg_left_dispatch` (`src/ops/trsm/kernels/trsm_sg_left.cc`) is the trsm
candidate `sg_left`: Side::Left, orders 1..32, sub-group 32 (`src/ops/trsm/choice.hh`); a pin above
order 32 throws. Each lane is a (matrix, canonical row \f$r\f$) pair, so a sub-group holds \f$32/N\f$ matrices for \f$N \in \{4, 8, 16, 32\}\f$. Each lane carries QC right-hand sides: 4 for \f$q \le 4\f$, 8 for \f$q \le 8\f$, otherwise 16
(complex<double> caps at 8). Lane \f$r\f$ holds row \f$r\f$ of the canonical Lc; step \f$s\f$ broadcasts \f$x_s\f$ from lane \f$s\f$
with `select_from_group`. No local memory, no barrier. The 24 (side, uplo, trans, diag) cases use V1's
canonical fold (`trsm_canonical.hh`); the non-finite-reciprocal fallback to division is per matrix, an
xor-butterfly over its N lanes. `blocked`'s diagonal solves run V1 (`trsm_native_v1_buckets`), not
`sg_left`.

Register decisions (`scripts/register_probe.sh`):

- The step loop stops at the runtime order \f$n\f$ (rows \f$n, \dots, N-1\f$ are identity padding). For complex \f$N \ge 16\f$ and
  for \f$N = 32\f$ with QC = 16 the loop is rolled (`#pragma unroll 1` with `break`): cfloat N=32 QC=16 drops
  from 168 to 77 registers, n32 q64 1.91 -> 1.39 ms. Elsewhere it is fully unrolled with `continue`. No
  variant spills.
- The TU is built with `-pragma-unroll-threshold=262144`; without it the unrolled variants put x[] on the stack.

V1-capped time over sub-group time at batch 32768 (above 1: sub-group faster); the deep
`tuned/trsm.sm_120.txt` reproduces this shape (float `side=L trans=N` ranks `sg_left` first for \f$q \le 83\f$ at
order 8, \f$q \le 38\f$ at 16, \f$q \le 32\f$ at 32):

| n \ q | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| float 4 | 3.18 | 2.67 | 2.15 | 1.89 | 1.81 |
| float 8 | 2.29 | 1.84 | 1.45 | 1.09 | 0.97 |
| float 32 | 2.49 | 1.43 | 1.15 | 0.81 | 0.76 |
| cfloat 12 | 1.31 | 0.87 | 0.64 | 0.73 | 0.79 |
| cfloat 16 | 1.26 | 1.08 | 1.61 | 1.79 | 1.66 |
| cfloat 32 | 1.32 | 1.36 | 1.32 | 1.37 | 1.12 |

The cfloat n=16 island is V1's (V1 at cfloat n16 q64 takes 0.91 ms, against 0.31 at n=12).

Tests (`TrsmNativeSgLeft.*`) call the kernel directly with a padded ld, a batch stride and large finite
poison. Planted breaks over `trsm_tests`: reading A with ld = n turns 10 tests red; dropping the store
guard `c0 + j < q` turns 13 red before the saturating-batch case aborts; ending the step loop one row
early turns 15 red, all Left.

### trsm on sm_120: result

`BM_TRSM<`, BASE (before the ladder cap and `sg_left`) / new / cuBLAS, ms, batch 32768 unless given,
2026-10-05:

| type | n | q | BASE | new | cuBLAS |
|---|---|---|---|---|---|
| float | 8 | 8 | 0.128 | 0.015 | 0.043 |
| float | 32 | 8 | 0.994 | 0.115 | 0.171 |
| float | 32 | 64 | 1.083 | 0.575 | 1.238 |
| float | 128 | 64 | 7.777 | 5.259 | 9.566 |
| float | 512 | 8 (b2048) | 1.893 | 1.091 | 1.125 |
| float | 512 | 512 (b1024) | 7.594 | 7.603 | 12.780 |
| cfloat | 32 | 8 | 1.551 | 0.333 | 1.391 |
| cfloat | 128 | 64 (b16384) | 8.613 | 5.921 | 140.6 |

Over the 62-cell grid (\f$n = 8, \dots, 512\f$, \f$q \in \{8, 16, 32, 64, n\}\f$) the float geomean time/cuBLAS went 1.44 ->
0.57 (cells above 1: 21 -> 0; closest float \f$n = 512\f$ q=8/16 and \f$n = 256\f$ q=8 at 0.96-0.97), cfloat 0.18 -> 0.09.
Consumers followed: every posv cell (\f$n = 4, \dots, 128\f$, nrhs 16 and 64) and every getrs `native:blocked` cell
(\f$n = 64, \dots, 512\f$, nrhs 16 and 64) beats the vendor composition.

Open: Side::Right has no sub-group kernel and keeps the capped V1, which loses to cuBLAS when a solve has
few rows (float n32 q8 b32768 0.305 vs 0.164 ms; n64 q8 1.007 vs 0.481). The deep table ranks `vendor`
first there (orders 20-64 at \f$q \le 8\f$, order 32 from batch 8192). Rows 9..16 at orders 17..24 still lose
5-18%. A Side::Right sub-group kernel (each row of B is a Left solve with op(A)^T) would fix both.

## gemm (native register-tiled selector)

The \f$16 \times 16\f$ (\f$2 \times 2\f$ thread tile) and \f$32 \times 32\f$ (\f$4 \times 4\f$) wide tiles, `wide:m=16:n=16:k=16` and
`wide:m=32:n=32:k=16` in `wide_configs` (`src/ops/gemm/choice.hh`), came from this campaign: on the
4090 the only complex register tile was \f$64 \times 64\f$, which at \f$m = n = 32\f$ computes \f$3/4\f$ padding. They are
ordinary candidates in every form; the sm_120 table times them, and gemm measurements are owned by
[gemm.md](gemm.md). Tests: `SmallWideNNTilesMatchTiled16` (ragged m, n, k from 1 to 257, sub-view ld,
both betas), `SmallWideTilesRunTheirTransposedForms`, `SmallWideBetaZeroNeverReadsC` and
`SmallWideSaturatingBatchIsBitIdentical` in `tests/gemm_tests.cc`.

The branch's selector findings, for reading the deep table (batch 32768, time/vendor):

- One 64-thread group per tile walks k serially, so at deep k the small tiles need more tiles in flight
  than the older kernels: float 16 tile wins from 2048 tiles at \f$k \ge 768\f$ (\f$16 \times 16 \times 1024\f$: 0.92 at 2048 tiles,
  1.71 at 1024), the float 32 tile from 1536 tiles at \f$k \ge 512\f$; cfloat wins at 1024.
- complex<double> NN favours the tile with the most groups in flight (\f$16 \times 16 \ge 32 \times 32 \ge 64 \times 64\f$) whenever
  \f$\max(m,n,k) > 8\f$ (\f$64 \times 64 \times 16\f$ b16384: 0.82 / 0.81 / 0.90); Direct wins at \f$8^3\f$.
- k-split (2-4 slices of k with an SLM reduction) never beat the unsplit tiles, and a \f$128 \times 64\f$ or \f$K = 32\f$
  \f$64 \times 64\f$ tile for large cfloat was 1.28-1.55 against 1.11-1.17. Rejected.

Open: large cfloat squares (\f$m = n = 128, \dots, 1024\f$) stay 1.10-1.18x behind cuBLAS; there is no large complex
tile.

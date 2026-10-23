# LDL^T: sytrf, sytrs, sysv {#perf_sytrf}

> **Status:** current · routes and `can_run` windows only, nothing timed · threadripper02 (RTX PRO 6000, sm_120), CUDA 13.2 · 2026-10-22

Batched symmetric indefinite factorization by Bunch-Kaufman pivoting with LAPACK `?sytrf` semantics,
its solve `?sytrs`, and the composed `?sysv`. Consumers: KKT systems \f$[[H, A^T], [A, 0]]\f$ (optimize)
and `dsysv` (gfn2). Complex is symmetric, not Hermitian (`csytrf`).

## sytrf: contract

* **Storage.** Only the `uplo` triangle is read and written. It receives D and the multipliers of L
  (Lower) or U (Upper), exactly as LAPACK stores them; the other triangle is never touched.
* **Pivots.** LAPACK's `ipiv`, 1-based int32 packed into the first half of the int64 span, item b at
  `as_span<int>()[b n + k]` (getrf's packing). `ipiv[k] > 0`: 1 x 1 block, rows/columns k and
  `ipiv[k] - 1` interchanged. Lower `ipiv[k] = ipiv[k+1] = -p`: 2 x 2 block at (k, k+1), rows k+1 and
  p-1 interchanged. Upper `ipiv[k-1] = ipiv[k] = -p`: 2 x 2 block at (k-1, k), rows k-1 and p-1
  interchanged.
* **info.** Per item: 0, or i > 0 when D(i,i) (1-based) is exactly zero (`?sytf2`'s rule: the
  first column whose `max(|a_kk|, colmax)` is zero, in loop order, so the zero matrix gives 1 for
  Lower and n for Upper). The factorization still completes.
* **Pivot metric.** `cabs1 = |re| + |im|`, as `csytf2`.

## sysv and sytrs: contract

* `sytrs` reads A, `uplo` and `ipiv` exactly as `sytrf` left them and overwrites B with X. No
  singularity check: a singular item gives infinities or NaNs.
* `sysv` is `sytrf` then `sytrs`. An item with `info > 0` keeps its B (LAPACK `?sysv`); the other
  items are solved. Every route honours this, the fused `cta` kernel included.

## Selection

| op | family | driver | `can_run` |
|---|---|---|---|
| `sytrf` | `cta` | `sytrf_kernels.cc` `cta_launch`: one sub-group per matrix, logical lower triangle in local memory | native term, `n <= 32` (`sytrf.cc`, `can_run`) |
| `sytrf` | `wg` | `sytrf_kernels.cc` `sytrf_wg_dispatch`: one work-group (32..256) per matrix, unblocked `?sytf2` in place in global memory | native term |
| `sytrf` | `vendor` | `cusolverDn<t>sytrf` per item (`cusolver_sytrf.cc`), LAPACKE `?sytrf` (`netlib_lapack.cc`) | CUDA with cuSOLVER or NETLIB with LAPACKE, `d.has_vendor`, homogeneous batch. rocSOLVER is not wired |
| `sysv` | `cta` | `sysv_cta_dispatch`: the `cta` factor and both solves in one launch | GPU, sub-group 32, homogeneous, `1 <= n <= 32` |
| `sysv` | `compose` | public `sytrf` (routed), then `sytrs` with the per-item status as a skip list | homogeneous batch |

* **Native term:** `B != NETLIB && d.is_gpu && d.has_sg32 && max_wg >= 32`, homogeneous batch,
  `n, batch >= 1`.
* **sytrs is not routed** (no `Op`, no route variable): the native kernel (one work-group per
  right-hand side, `sytrs_native_dispatch`) on every GPU backend, LAPACKE on NETLIB.
* **Last resort** (no table): sytrf `wg`, then `vendor`; sysv `compose`.
* **Pins:** `BATCHLAS_SYTRF_ROUTE` (`cta`, `wg`, `vendor`, `native`, `auto`),
  `BATCHLAS_SYSV_ROUTE` (`cta`, `compose`). A `cta` pin above n = 32 throws.

### sytrf: the transcribed tables

`tuned/{sytrf,sysv}.{sm_89,sm_120}.txt` are `transcribed`, **untimed**, generated from
`benchmarks/results/tuning/ledger/{sytrf,sysv}.<device>.jsonl` (run `transcribed-30cc8e06`). Grid: n in
{1, 2, 4, 8, 12, 16, 24, 32, 33, 48, ..., 2048}, batch in {128, ..., 32768}, sysv nrhs in {1, 4, 16, 64}.

| op | n <= 32 | n > 32 |
|---|---|---|
| `sytrf` | `cta`, `wg`, `vendor` | `wg`, `vendor` |
| `sysv` | `cta`, `compose` | `compose` |

The order is a hypothesis: native first because the vendor arm is a serial per-item loop. Above
n = 32 the unblocked `wg` kernel streams the trailing matrix from global memory once per column, so
the per-item cuSOLVER loop may win at large n and small batch. There is no tuner spec
(`tools/tune/sytrf_spec.cc`) yet; a measured table needs one.

## sytrf: the reversal map

The kernels implement `?sytf2` and `?sytrs` once, in lower form (`sytrf_device.hh`). Upper runs the
same code on the index-reversed matrix, logical i = physical n-1-i. Under the reversal `?sytf2`'s
upper loop (k = n down to 1, 2 x 2 block at (k-1, k), interchange with k-1) is its lower loop step for
step (block at (k, k+1), interchange with k+1), the stored factor lands in the upper triangle, and
`ipiv` is written at the physical index. The one asymmetry is `I?AMAX`'s tie-break (first maximum in
storage order): the upper form takes the largest logical index on a tie.

Pivot decisions are taken redundantly by every lane from values read after a barrier; reductions are
a sub-group butterfly plus sub-groups in fixed order, so every lane agrees and the result is
deterministic. A NaN magnitude is folded to +inf before any compare so the lanes cannot disagree.

## sytrf: vendor

cuSOLVER 13.2 (`cusolverDn.h`) offers `cusolverDn<t>sytrf` (+ `_bufferSize`; legacy API, int32
`ipiv`, one matrix per call) and the 64-bit `cusolverDnXsytrs` (int64 `ipiv`). There is no
`cusolverDnXsytrf` and no batched sytrf or sytrs. The vendor family is therefore a per-item loop on one
stream with one shared work buffer; cuSOLVER writes LAPACK's `ipiv`, which the native `sytrs`
consumes directly (`VendorFactorSolvesWithNativeSytrs`). `Xsytrs` is not wired: it would need an int64
copy of the pivots and is still per item.

## sytrf: tests

`tests/sytrf_tests.cc`, every route pinned, padded `ld` and batch stride, both pads and the
unreferenced triangle poisoned with large finite values, complex with nonzero imaginary parts:

* `GeneratorsForceTheirPivotKinds`: random data has interchanges and 2 x 2 blocks, the dominant
  generator neither, every KKT item a 2 x 2 block.
* `FactorThenSolveEveryRoute`, `SysvEveryRoute`: residual through `EXPECT_VERIFY(Check::solve)`, both
  uplo, n straddling 32 | 33 and 256 | 257, `info == 0`, untouched triangle and pads.
* `SingularItemReportsInfoWithoutDisturbingTheOthers`: the zero matrix in slot 2 gives `info` 1
  (Lower) or n (Upper); every other item is bit-identical to a batch without it; sysv keeps its B.
* `WgPivotsMatchVendor`: on random data (no near ties) the `wg` kernel's ipiv equals cuSOLVER's entry
  for entry at n = 24, 100, 200; a residual cannot see a wrong pivot choice, and above n = 32 the
  argmax spans several sub-groups.
* `CtaAndWgAgreeBitForBit`, `SaturatingBatchIsBitIdenticalToItemZero` (batch 1024, identical KKT
  items), `VendorFactorSolvesWithNativeSytrs`, `CtaCeilingStraddled`, `ArgumentChecks`.

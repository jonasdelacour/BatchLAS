# Vendor-free status

> **Status:** current · test counts are the last recorded runs (end of WP8) · op table snapshot 2026-10-06

The vendor-free build (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) configures, builds, links, loads and
runs. Milestone M1 (full `ctest` suite passing vendor-free) is not reached. Every public dense op
and `spmm` has a native SYCL kernel. The remaining gaps are the host (`Backend::NETLIB`) path,
routing defects and shapes that native `can_run` refuses.

Performance evidence is in [`../perf/`](../perf/README.md). The vendor seam is in
[`vendor-independence.md`](vendor-independence.md). Open bugs are in [`known-defects.md`](known-defects.md).

## Build configurations

| | vendor-present | vendor-free |
|---|---|---|
| configure | default | `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` |
| result | `BATCHLAS_HAS_CUBLAS` / `CUSOLVER` / `CUSPARSE` = 1 | `BATCHLAS_HAS_CUDA_BACKEND 1`, every CUDA math library at 0 |
| `ctest -LE slow` | **56 / 57** | **35 / 57** |

The one vendor-present failure, `lanczos_tests`, predates the campaign. It never calls `gemv`, its
only `linked` rows and no `reached` rows.

`BATCHLAS_ENABLE_VENDOR_BLAS` is a master switch over the per-library options in
`BATCHLAS_VENDOR_LIBRARIES` (`cmake/BatchLASOptions.cmake`).

## Measuring vendor-free status

Do not use the `ctest` pass count. Level-3 and factorization suites also run against the host
backend, which a vendor-free build cannot serve, so a suite can fail on host rows while every CUDA
case passes. Vendor-free `trsm_tests` is 59 passing / 32 failing; all 32 failures are host rows.

The native families' `can_run` requires `d.is_gpu` for `geqrf`, `orgqr`, `ormqr`, `getrf`, `getrs`,
`getri`, `potrf`, `trsm`, `syev` and `gesvd` (`src/ops/<op>/<op>.cc`; geqrf's in `can_run.hh`).
Exceptions: `gemv`'s `direct` arm and `spmm`'s gather, which run on a `native_cpu` queue, and `gemm`,
which accepts `is_gpu || !has_vendor` (`src/ops/gemm/gemm.cc`).

### The metric: the `NoRouteError` census

Every vendor-free failure is a `NoRouteError` naming an op, scalar type and the switch that would
restore it (`include/batchlas/no_route.hh`). Count them with:

```
ctest --test-dir build-novendor -LE slow --rerun-failed --output-on-failure \
  | grep -o 'no route for [a-z0-9_]*' | sort | uniq -c | sort -rn
```

Diff the census and the failing suite names, not the pass count. Recorded after WP8:
`syev` 87, `geqrf` 44, `trsm` 32, `ormqr` 24, `trmm` 16, `herk` 16, `getri` 16, `syrk` 12,
`her2k` 12, `hemm` 12, `syr2k` 10, `symm` 8, `spmm` 0 (was 2).

For routes, read the `reached` rows of the coverage dump (`BATCHLAS_COVERAGE_OUT`). The static
`linked` table is not evidence: it was stale in both directions (`src/select/coverage.cc`).

## Per-op status

Route = the first runnable native entry of the op's table (`tuned/<op>.<dtype>.<device>.txt`,
selected in `src/ops/<op>/<op>.cc`). "Transcribed" means the rows reproduce the deleted
`preferred()` window (@ref tuned_tables_readme).

| op | vendor-free route | gaps |
|---|---|---|
| `gemm` | `direct`, `tiled`, `small`, `reg`, `wide`. Transcribed: GPU, homogeneous, `batch >= 64`; `double` at `k >= 2`; `float` NN square `max_dim <= 48`. | Complex is never native-first: no register kernel, so it falls to `direct`/`tiled`, 3.2–7.1× slower than cuBLAS. |
| `gemv` | `cta`, `direct`. Transcribed: `complex<double>`, transposed, `64 <= red <= 352`, `out >= 256`, `batch >= 320`. | Vendor first almost everywhere: cuBLAS runs at 94–105% of the DRAM roof on 90 of 92 cells ([`gemv.md`](../perf/gemv.md)). |
| `trsm` | `cta`, `sg_left`, `blocked`. Transcribed: native from `batch >= 8`; `float`/`Side::Right` also need `batch >= 128 \|\| order <= 32`. | Census 32, all host rows. |
| `potrf` | `tiny`, `cta`, `lpanel`, `blocked`. Measured per cell. | `Uplo::Upper` refused by the blocked driver (`src/ops/potrf/potrf.cc`). Complex (0.311–0.509×) and `n <= 256` are vendor-first ([`potrf.md`](../perf/potrf.md)). |
| `posv` | `tiny`, `cta`, `blocked`. No vendor family. | None recorded. |
| `geqrf` | `tiny`, `cta`, `blocked`. Transcribed: above a per-type order floor (`float` 64, `cfloat` 48, `double` 76, `cdouble` 256); tall panels `n >= 32 && aspect >= 4` (8 for 64-bit types). | Census 44, host rows. Below the floor the vendor wins, to 0.02× at `double` n = 8. Tall clause `rows >= 128` is not a bracketed edge; `double` 128x32 is 0.68× ([`small-n-baseline.md`](../perf/small-n-baseline.md#geqrf)). |
| `orgqr` | `blocked`. Transcribed: `rows <= 512 && cols <= 512`. | Vendor wins above n = 512. Host rows refused. |
| `ormqr` | `blocked`, then `vendor` on every row. | Census 24, host rows. |
| `getrf` | `tiny`, `cta`, `blocked`. Transcribed: `float` order >= 256; `cfloat` order >= 512 (>= 256 at `batch >= 256`). | `double` and `complex<double>` earn nothing at any order ([`lu.md`](../perf/lu.md)). |
| `getrs` | `cta`, `blocked`. Transcribed: `cta` at `nrhs <= 2` (all types), `nrhs <= 4` (`float`); `blocked` at `batch >= 128` with `float` `nrhs >= 64` or `double` `nrhs >= 128`. | Clauses A and B hand 84 winning cells to the vendor (largest 3.944×). |
| `getri` | `blocked`. Transcribed: `float` order >= 128, `cfloat` order >= 256. | Census 16, host rows. Unrouted win at `batch <= 32`, every type (see levers below). |
| `gesv` | `tiny`, `blocked`. No vendor family. | None recorded. |
| `spmm` | `direct`. Transcribed: CSR, `transA == NoTrans`, minus `complex<float>` with `transB != NoTrans`. | Census 0. Transposed scatter is vendor-first: 169 of 458 saturated cells lose, worst 3.011, and no clause recovers a window ([`spmm.md`](../perf/spmm.md)). |
| `syev` | `cta`, `cta_fused`, `jacobi`, `blocked`, `two_stage`. Transcribed from the old CUDA grid. | Census 87. Four call sites demand the vendor `syev_vendor_or_throw` and throw vendor-free: `src/extra/cond.cc:54`, `src/extra/norm.cc:46`, `src/extensions/syevx_lobpcg.cc:524` and `:1076`. Fix: call the public `syev` ([known-defects #2](known-defects.md)). |
| `gesvd` | `jacobi`, `cta`, `blocked`. Transcribed: real 33..64 is `blocked\|vendor\|jacobi` ([evidence](../perf/gesvd.md#gesvd-the-wide-band-33-to-64)). | Host rows refused. |
| `symm` | Hand-rolled `if` chain. `expand` (mirrored expansion + public `gemm`; needs `expansion_fits`). No tile kernel. | Census 8. At the snapshot, `double` had no expansion route; the flat-selection spec lists `expand` for both types. Reconcile before relying on either. |
| `syrk` | Hand-rolled. `gram` (`n <= 128`, float and double), `triangular` (float). | Census 12. Non-float tiles are reachable only from `cublas.cc`. |
| `syr2k` | `triangular` (float only). Real ConjTrans stays on the vendor. | Census 10. No non-float tile route; `syr2k_triangular_tiles` has one call site, in the float-only dispatcher. |
| `trmm` | `triangular` (`Side::Left`), `expand`. | Census 16. The `double`/complex tile branch is reachable only from `cublas.cc`. |
| `hemm`, `herk`, `her2k` | None: vendor or `NoRouteError`. | Census 12, 16, 12. No native arm in the facade (`src/ops/level3/level3.cc:68-115`). |

Rows that stay vendor-first by measurement are the vendor's wins; kernel work will not change them.

## M1: self-sufficient build (not reached)

**Definition.** `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` configures, builds and passes the full `ctest`
suite. No performance claim; the vendor stays first in every Auto order.

Blockers, by census:

1. **Host (`Backend::NETLIB`) path (WP9).** Most of the residue. Covers all 32 vendor-free `trsm`
   failures, all 12 `sytrd_blocked`, 8 `ortho_tests` and 20 `cond_tests`.
2. **`syev`, 87.** Partly a routing-vocabulary defect (see the table).
3. **`hemm`, `herk`, `her2k`, 40.** No native arm.
4. **Level-3 non-float.** `trmm` 16, `syrk` 12, `syr2k` 10, `symm` 8.
5. **`geqrf` 44, `ormqr` 24, `getri` 16.** Host rows and refused shapes.
6. **`potrf` `Uplo::Upper`.** A correctness refusal. Fix by mirroring the upper triangle and running
   the Lower pipeline, as `syev` does.

## M2: vendor-free by default, cell by cell

**Definition.** For each (op, type, shape class) at saturated, large-batch shapes,

```
t_native <= 1.10 * t_vendor
```

and within the op's accuracy tolerance, native moves ahead of the vendor in Auto. A failing cell
stays vendor-first and is published as such.

Reached as measured windows: `getrf`, `getrs`, `getri`, `gemv`, `spmm`'s gather. Not reached by
measurement: `potrf`, `spmm`'s scatter, complex `gemm`. Not reached for want of an end-to-end
measurement: `geqrf`, `orgqr`.

## Remaining work

### WP9: the CPU story (not started)

Open question: must a CPU SYCL device be fast, or only correct? The standing answer is correct only,
since MKL and OpenBLAS already serve the CPU market. Dropping the `is_gpu` clause from `can_run`
is enough for a family to serve the host queue.

Traps: `Backend::INTEL` is hard-wired false and oneMKL cannot be tested on this box. A CUDA-off
`ctest` shows about 30 failures that are artefacts of the CPU-only build.

### Measured but unrouted

| lever | measured | where |
|---|---|---|
| `geqrf` / `orgqr` default flip | 3.24× / 7.85× geomean | [`qr.md`](../perf/qr.md) |
| `getri` at `batch <= 32`, every type | 1.7–28× over cuBLAS (per-item loop there) | [`lu.md`](../perf/lu.md) |
| `potrf` `float` at `n >= 1024` | 1.13–1.40× over cuSOLVER | [`potrf.md`](../perf/potrf.md) |
| `getrs` clauses A and B | 84 winning cells, largest 3.944× | [`lu.md`](../perf/lu.md) |
| `gemv` `out_len >= 768 && batch >= 128`; batch floor 320 → ~288 | ~18 cells at 2.26–2.91×; six wins at 1.27–4.45× | [`gemv.md`](../perf/gemv.md) |
| `getrs` clause-C batch floor 128 → 32 | float 3.87–5.96×, double 3.56–4.31× at `nrhs = 128` | [`lu.md`](../perf/lu.md) |
| `spmm` gather narrowed to `nrhs >= 16` | 183 cells, worst 0.968; refused, the axis is the column pattern | [`spmm.md`](../perf/spmm.md) |
| complex register-tiled GEMM | ~2.7× on vendor-free `cdouble potrf`; unblocks complex `gemm`, `potrf`, level-3 complex | [`gemm.md`](../perf/gemm.md) |

### Open defects

`ortho.cc`'s transposed `gemv` view, the `syev` resolver bypasses above, and `lanczos.cc`'s two-column
`gemm`, whose second column is discarded ([`known-defects.md`](known-defects.md)). Closed: the
`BATCHLAS_SYRK_ROUTE=native` wrong answer, the `symm` `expansion_fits` gap and the `trsm`
heterogeneous-batch gate.

### Do not re-attempt

| Alternative | Result | Verdict |
|---|---|---|
| `syrk`/`herk` for `ortho`'s Gram matrix | 73–96× slower at `ortho`'s shapes | Dead end; wins only at `k >= 512`, square-ish. |
| `trmm` for the WY block factor | Loses at every shape | Dead end; complex still takes GEMM. |
| Complex Gram tiles (`herk`) | Loses to GEMM plus Hermitian fold | Dead end; compute bound. |
| `syr2k` for `sytrd_blocked` trailing update, `double` | 7.7× slower where it matters | Route stays CUDA + float. |
| Cooperative TRSM solve | 0.39× at order 64 | Dead end; the traffic model missed the serial recurrence. |
| Transcribing level-3 gate thresholds into a table | Sends `129 <= n <= 383` to a route that writes both triangles | Dead end. |
| `potrf` fold-free trailing update | 11% cheaper, and wrong | Dead end. |

## How to re-derive this page

```
cmake -S . -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON
cmake --build build-novendor -j"$(nproc)"
ctest --test-dir build-novendor -LE slow
ctest --test-dir build-novendor -LE slow --rerun-failed --output-on-failure \
  | grep -o 'no route for [a-z0-9_]*' | sort | uniq -c | sort -rn
```

The superseded root-level plan documents are kept at the tag `perf-evidence/vendor-independence`.

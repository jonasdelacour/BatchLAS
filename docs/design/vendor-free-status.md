# Vendor-free status {#design_vendor_free_status}

> **Status:** current · checked 2026-10-21 · CUDA test counts and the `NoRouteError` census are from 2026-09-06 (end of vendor-independence WP8, before flat selection) and have not been re-run

Two vendor-free configurations exist: a CUDA build with `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`, and the
AdaptiveCpp Metal build (`Backend::SYCL`, no vendor library by construction). Both configure, build,
link, load and run. Milestone M1 (the full suite passing vendor-free) is not reached on CUDA. Every
public dense op except hemm, herk and her2k, and `spmm`, has a native SYCL kernel; the remaining gaps
are the host (`Backend::NETLIB`) path, hemm/herk/her2k, and shapes that native `can_run` refuses.

Mechanism: [vendor-independence.md](vendor-independence.md). Performance: [../perf/](../perf/README.md).
Open bugs: [known-defects.md](known-defects.md).

## Build configurations

| | vendor-present (CUDA) | vendor-free (CUDA) | Metal (acpp, Apple GPU) |
|---|---|---|---|
| configure | default | `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` | preset `acpp-metal-tests` |
| result | `BATCHLAS_HAS_CUBLAS` / `CUSOLVER` / `CUSPARSE` = 1 | `BATCHLAS_HAS_CUDA_BACKEND 1`, every vendor library 0 | `BATCHLAS_HAS_SYCL_BACKEND 1`; GPU float and complex<float>; double on the host (NETLIB, Homebrew OpenBLAS) |
| last recorded `ctest` | `-LE slow` **56 / 57** (2026-09-06) | `-LE slow` **35 / 57** (2026-09-06) | full, serial: **99 passed, 5 failed, 1 skipped of 105** (2026-10-14, M5 Max) |
| failure ledger | `tests/known-failures.txt` | none (the CI step reports, never gates) | `tests/known-failures-acpp-metal.txt` |

- The one vendor-present failure in the 2026-09 count, `lanczos_tests`, predates the
  vendor-independence work; it still fails on every tree (known-defects, "lanczos returns a spectrum
  about three times too wide").
- Metal's five failing tests: `hemm_tests`, `herk_tests`, `her2k_tests` (no native family),
  `posv_tests` (Upper complex<float> n = 33: potrf Upper above the `cta` ceiling has no vendor-free
  route, known-defects #19) and `lanczos_tests` (shared with every tree).
- `BATCHLAS_ENABLE_VENDOR_BLAS` is the only vendor switch (`cmake/BatchLASOptions.cmake:317-326`);
  there are no per-library options. A partial CUDA set fails configure.
- CI runs the vendor-free CUDA configure, build and `ctest -LE slow` with `continue-on-error`
  (`.github/workflows/ci.yml`, "Vendor-free configure, build and ctest"): it reports, it does not gate.

## Measuring vendor-free status

Do not use the `ctest` pass count. Level-3 and factorization suites also run against the host
backend, which a vendor-free build cannot serve, so a suite can fail on host rows while every CUDA
case passes. Vendor-free `trsm_tests` (2026-09-06): 59 passing, 32 failing; all 32 failures are
host rows.

The native families' `can_run` requires `d.is_gpu` for `geqrf`, `orgqr`, `ormqr`, `getrf`, `getrs`,
`getri`, `potrf`, `trsm`, `syev`, `gesvd`, the level-3 four, and `posv` and `gesv` `tiny`/`cta`
(their `blocked` delegates to children that select for themselves) (`src/ops/<op>/<op>.cc`; geqrf's
in `can_run.hh`). Exceptions: `gemv`'s `direct` arm and `spmm`'s
`direct`, which run on a host queue, and `gemm`, which accepts `is_gpu || !has_vendor`
(`src/ops/gemm/gemm.cc:97`).

### The metric: the `NoRouteError` census

Every vendor-free failure is a `NoRouteError` naming an op, scalar type and the switch that would
restore it (`include/batchlas/no_route.hh`; message `no route for <op><scalar>`). Count them with:

```
ctest --test-dir build-novendor -LE slow --rerun-failed --output-on-failure \
  | grep -o 'no route for [a-z0-9_]*' | sort | uniq -c | sort -rn
```

Diff the census and the failing test names, not the pass count. Recorded 2026-09-06 (before flat
selection and the level-3 wave):

| op | census | op | census |
|---|---|---|---|
| `syev` | 87 | `getri` | 16 |
| `geqrf` | 44 | `syrk` | 12 |
| `trsm` | 32 | `her2k` | 12 |
| `ormqr` | 24 | `hemm` | 12 |
| `trmm` | 16 | `syr2k` | 10 |
| `herk` | 16 | `symm` | 8 |
| `spmm` | 0 (was 2) | | |

For routes, read the `reached` rows of the coverage dump (`BATCHLAS_COVERAGE_OUT`;
`ctest -R`-scoped, or the `batchlas_coverage` target). The `linked` rows are not evidence that a
route ran.

## Per-op status

Route = the first runnable native entry of the op's table (`tuned/<op>.<device>.txt`, selected in
`src/ops/<op>/<op>.cc`). "sm_89" windows are the transcribed rows, which reproduce the deleted
`preferred()` windows (@ref tuned_tables_readme); sm_120 is deep-measured.

| op | native families (vendor-free route) | gaps |
|---|---|---|
| `gemm` | `direct`, `tiled`, `small`, `reg` (float), `wide`, `splitk`. sm_89: GPU, homogeneous, `batch >= 64`; `double` at `k >= 2`; `float` NN square `max_dim <= 48`. | sm_89: complex is never native-first (3.2–7.1× slower than cuBLAS on `direct`/`tiled`). sm_120: wide v2 runs complex<float> at geomean 0.858× cuBLAS ([gemm.md](../perf/gemm.md)). |
| `gemv` | `cta`, `direct`. sm_89: `complex<double>`, transposed, `64 <= red <= 352`, `out >= 256`, `batch >= 320`. | Vendor first almost everywhere: cuBLAS runs at 94–105% of the DRAM roof on 90 of 92 cells ([gemv.md](../perf/gemv.md)). |
| `trsm` | `cta`, `sg_left`, `blocked`. sm_89: native from `batch >= 8`; `float`/`Side::Right` also need `batch >= 128 \|\| order <= 32`. | Census 32, all host rows. |
| `potrf` | `tiny`, `cta`, `lpanel`, `blocked`. | `Uplo::Upper` above the `cta` ceiling has no route: `lpanel` and `blocked` are Lower-only (known-defects #19). Complex (0.311–0.509×) and `n <= 256` are vendor-first ([potrf.md](../perf/potrf.md)). |
| `posv` | `tiny`, `cta`, `blocked`. No vendor family. | Upper inherits potrf's gap (Metal ledger). |
| `geqrf` | `tiny`, `cta`, `blocked`. sm_89: above a per-type order floor (`float` 64, `cfloat` 48, `double` 76, `cdouble` 256); tall panels `n >= 32 && aspect >= 4` (8 for 64-bit types). | Census 44, host rows. Below the floor the vendor wins, to 0.02× at `double` n = 8. Tall clause `rows >= 128` is not a bracketed edge; `double` 128x32 is 0.68× ([small-n-baseline.md](../perf/small-n-baseline.md#geqrf)). |
| `orgqr` | `blocked`. sm_89: `rows <= 512 && cols <= 512`. | Vendor wins above n = 512. Host rows refused. |
| `ormqr` | `blocked`, then `vendor` on every row. | Census 24, host rows. |
| `getrf` | `tiny`, `cta`, `blocked`. sm_89: `float` order >= 256; `cfloat` order >= 512 (>= 256 at `batch >= 256`). | `double` and `complex<double>` earn nothing at any order ([lu.md](../perf/lu.md)). |
| `getrs` | `cta`, `blocked`. sm_89: `cta` at `nrhs <= 2` (all types), `nrhs <= 4` (`float`); `blocked` at `batch >= 128` with `float` `nrhs >= 64` or `double` `nrhs >= 128`. | Clauses A and B hand 84 winning cells to the vendor (largest 3.944×). |
| `getri` | `blocked`. sm_89: `float` order >= 128, `cfloat` order >= 256. | Census 16, host rows. Unrouted win at `batch <= 32`, every type (levers below). |
| `gesv` | `tiny`, `blocked`. No vendor family. | None recorded. |
| `spmm` | `direct`. sm_89: CSR, `transA == NoTrans`, minus `complex<float>` with `transB != NoTrans`. | Census 0. Transposed scatter is vendor-first: 169 of 458 saturated cells lose, worst 3.011, and no clause recovers a window ([spmm.md](../perf/spmm.md)). |
| `syev` | `cta`, `cta_fused`, `jacobi`, `blocked`, `two_stage`. | Census 87. `cond` and `norm` call `syev_vendor_or_throw` directly (`src/util/cond.cc:67`, `src/util/norm.cc:55`; `Backend::SYCL` uses the public `syev`), so they throw vendor-free on CUDA (known-defects #2). LOBPCG's projected solves (`src/ops/syevx/kernels/syevx_lobpcg.cc:548`, `:1107`) do so only under the opt-in `BATCHLAS_SYEVX_PROJECTED_VENDOR`. Fix: call the public `syev`. |
| `gesvd` | `jacobi`, `cta`, `blocked`. sm_89: real 33..64 is `blocked\|vendor\|jacobi` ([evidence](../perf/gesvd.md#gesvd-the-wide-band-33-to-64)). | Host rows refused. |
| `symm` | `expand` (mirrored expansion + public `gemm`; float and double; needs `expansion_fits`). | Census 8 (2026-09-06, before double `expand`). |
| `syrk` | `gram` (`n <= 128`, float and double), `triangular` (float). | Census 12 (before double `gram`). Double above n = 128 has no native family. |
| `syr2k` | `triangular` (float only). A real ConjTrans runs as Trans. | Census 10. No non-float native family. |
| `trmm` | `triangular` (`Side::Left`), `expand`; all four types. | Census 16 (before every dtype got both families). |
| `hemm`, `herk`, `her2k` | None: vendor or `NoRouteError` (`src/ops/level3/level3.cc:33-102`). | Census 12, 16, 12. Throws on Metal too. |

Native level-3 families need a native GPU backend (CUDA or SYCL) and `batch <= max_groups_yz`.
Rows that stay vendor-first by measurement are the vendor's wins; kernel work will not change them.

## M1: self-sufficient build (not reached)

**Definition.** `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` configures, builds and passes the full `ctest`
suite. No performance claim; the vendor stays first wherever the table ranks it first.

Blockers, by the 2026-09-06 census (re-run to re-rank):

1. **Host (`Backend::NETLIB`) path (WP9).** Most of the residue: all 32 vendor-free `trsm` failures,
   all 12 `sytrd_blocked`, 8 `ortho_tests` and 20 `cond_tests`.
2. **`syev`, 87.** Partly the `cond`/`norm` vendor bypass (table above).
3. **`hemm`, `herk`, `her2k`, 40.** No native family (@ref design_flat_selection §11).
4. **Level-3 non-float.** `trmm` 16, `syrk` 12, `syr2k` 10, `symm` 8 at the census. Since then
   (`docs/perf/dispatch.md`, "Vendor-free baseline"): float syr2k at batch 1, double symm, double
   syrk `n <= 128` and every trmm dtype on both sides have native routes. Remaining: double syrk above
   128, non-float syr2k.
5. **`geqrf` 44, `ormqr` 24, `getri` 16.** Host rows and refused shapes.
6. **`potrf` `Uplo::Upper`** above the `cta` ceiling (known-defects #19). Fix: an Upper arm for
   `lpanel` or `blocked`, or mirror into the Lower pipeline as `syev` does.

## M2: vendor-free by default, cell by cell

**Definition.** For each (op, type, shape class) at saturated, large-batch shapes,
\f$t_\mathrm{native} \le 1.10\,t_\mathrm{vendor}\f$ within the op's accuracy tolerance moves native
ahead of the vendor in Auto. A failing cell stays vendor-first and is published as such. Under flat
selection this is the tables' ranking (3% tie band, native first on ties).

Reached as measured windows: `getrf`, `getrs`, `getri`, `gemv`, `spmm`'s gather. Not reached by
measurement: `potrf`, `spmm`'s scatter, complex `gemm` (sm_89; sm_120 complex<float> is now
native-first on most cells). Not reached for want of an end-to-end measurement: `geqrf`, `orgqr`.

## Remaining work

### WP9: the CPU story

Open question: must a CPU SYCL device be fast, or only correct? The standing answer is correct only,
since MKL and OpenBLAS already serve the CPU market. Dropping the `is_gpu` clause from `can_run` is
enough for a family to serve the host queue, if its kernel is correct there.

Traps: there is no Intel or oneMKL backend (`Backend::MKL` is reserved). A CUDA-off `ctest` shows
about 30 failures that are artefacts of the CPU-only build. On AdaptiveCpp a data-dependent loop
around barriers never ends on the OpenMP host device unless its condition is a group collective
(@ref dev_agent_guide, Metal traps).

### Measured but unrouted

| lever | measured | where |
|---|---|---|
| `geqrf` / `orgqr` default flip | 3.24× / 7.85× geomean | [qr.md](../perf/qr.md) |
| `getri` at `batch <= 32`, every type | 1.7–28× over cuBLAS (per-item loop there) | [lu.md](../perf/lu.md) |
| `potrf` `float` at `n >= 1024` | 1.13–1.40× over cuSOLVER | [potrf.md](../perf/potrf.md) |
| `getrs` clauses A and B | 84 winning cells, largest 3.944× | [lu.md](../perf/lu.md) |
| `gemv` `out_len >= 768 && batch >= 128`; batch floor 320 → ~288 | ~18 cells at 2.26–2.91×; six wins at 1.27–4.45× | [gemv.md](../perf/gemv.md) |
| `getrs` clause-C batch floor 128 → 32 | float 3.87–5.96×, double 3.56–4.31× at `nrhs = 128` | [lu.md](../perf/lu.md) |
| `spmm` gather narrowed to `nrhs >= 16` | 183 cells, worst 0.968; refused, the axis is the column pattern | [spmm.md](../perf/spmm.md) |
| complex register-tiled GEMM | ~2.7× on vendor-free `cdouble potrf`; unblocks complex `gemm`, `potrf`, level-3 complex | [gemm.md](../perf/gemm.md) |

These are sm_89 windows. "Unrouted" means the sm_89 transcribed table does not rank them first; the
sm_120 deep tables rank by measurement and may already route them. Re-check against the table before
acting.

### Open defects

`ortho`'s transposed `gemv` view (#1), the `cond`/`norm` vendor bypass (#2), and `lanczos`'s
two-column `gemm`, whose second column is discarded (#3) ([known-defects.md](known-defects.md)).
Closed: the `BATCHLAS_SYRK_ROUTE=native` wrong answer, the `symm` `expansion_fits` gap and the
`trsm` heterogeneous-batch gate.

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
cmake --build build-novendor -j8
ctest --test-dir build-novendor -LE slow
ctest --test-dir build-novendor -LE slow --rerun-failed --output-on-failure \
  | grep -o 'no route for [a-z0-9_]*' | sort | uniq -c | sort -rn
```

Record the failing test names (not the count) with the date and commit. Superseded root-level plan
documents are at the tag `perf-evidence/vendor-independence`.

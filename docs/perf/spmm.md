# SPMM: batched CSR kernel and routing

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda

Batched CSR `spmm` has two families: the native `direct` kernel and the cuSPARSE `vendor` call. This page gives
the routing rule, the measurements the `tuned/spmm.*` tables rest on, the boundaries of the native window, and the
vendor defects the test suite found. Selection is flat (@ref design_flat_selection). Every number is from one
RTX 4090 (device 1, 128 SMs, 72 MB L2, 1008 GB/s DRAM roof), CUDA backend, in-order SYCL queue.

## Choices

| spelling | implementation | `can_run` (correctness only) |
|---|---|---|
| `direct` | `sycl_spmm::spmm_native_csr` | CSR; the body for this `transA` compiled; `one_spmm()` shape checks; no heterogeneous B or C; batch >= 1 |
| `vendor` | `backend::spmm_vendor` (cuSPARSE, rocSPARSE, netlib) | a sparse vendor library, minus three known-bad shapes: netlib with any transpose, and on CUDA complex `transB = ConjTrans` with one row of B, or `complex<double>` N/N with one column ([known-defects #13](../design/known-defects.md)) |

Selection is the first runnable entry of the nearest row of `tuned/spmm.<dtype>.<device>.txt`, keyed
`transA:exact transB:exact m:log nrhs:log batch:log`. `ConjTrans` folds to `T` (`src/ops/spmm/spmm.cc:33-39`). The
last resort is `vendor`, then `direct`. `BATCHLAS_SPMM_ROUTE` takes `auto`, `native`, `vendor` or `direct`. Any
other spelling, or one `can_run` refuses for the call, throws.

`direct` has three bodies in `src/sycl/spmm_native.cc`: the `NoTrans` gather (body 1), and a scale plus atomic
scatter pair (bodies 0 and 2) for `Trans` and `ConjTrans`. The body is chosen in the launcher on `transA`. `transA`
is part of the coverage `variant_key` (`src/select/coverage.cc:39-45`), so gather and scatter stay separable in
`scripts/route_diff.sh`. Scale and scatter do not.

### Ranking in the tables

Every row of `tuned/spmm.<dtype>.<device>.txt` on sm_89, sm_120 and `cpu` is transcribed from the predicate that
preceded flat selection (untimed; provenance in `tuned/README.md`):

| rows | float, double, complex\<double\> | complex\<float\> |
|---|---|---|
| `transA=N transB=N` | `direct`, then `vendor` | `direct`, then `vendor` |
| `transA=N transB=T` | `direct`, then `vendor` | `vendor`, then `direct` |
| `transA=T`, either `transB` | `vendor`, then `direct` | `vendor`, then `direct` |

The ranking is identical at all 125 `(m, nrhs, batch)` cells of each block (`m` 1..65536, `nrhs` 1..64,
`batch` 1..16384). `SpmmTranscribedTable.RowsHoldTheOldPreference` (`tests/spmm_candidates_tests.cc:874`) pins it.

The tables have no batch, extent, `is_gpu` or `nnz` term. Each absence is a measured decision: the ranking is
constant along that axis.

In a vendor-present build the change against the old router moves 65 decisions. All 65 are `spmm`, all
`transA = NoTrans`, and all `vendor:auto` to `native:direct`. `complex<float>` moves only at `transB = NoTrans`.
The other three types move at all three `transB` spellings.

> **Note:** `scripts/route_diff.sh compare` reports 240 changed `reached` lines for this change. About 110 are
> `backend = AUTO` rows from unit-level shapes. `experiments/sparse_spmm/route_census.py` keys on the decision tuple
> and gives the 65.

## supports(), and what is deliberately not in it

The `direct` `can_run` terms (`src/ops/spmm/spmm.cc:71-84`) are correctness gates only: CSR format (`:76`), the
capability flag of the body that will run (`:79-80`), `one_spmm()` (`:44-59`: matching extents and batch, positive
`ld`s, an offset stride of at least `m + 1`), no heterogeneous dense batch (`:81-82`), and no empty batch (`:82`).
A CSR view is never heterogeneous in the `active_rows_` sense. Per-item `nnz` variation is handled exactly through
the row offsets.

Deliberate absences:

* **No `is_gpu` gate.** Every body is a plain loop with no local memory, group or sub-group collective, or required
  sub-group size. The vendor-free build runs `direct` on the host backend too.
* **No transpose refusal.** All nine `(transA, transB)` spellings are served. Passing the dense block as
  `transB = Trans` cuts the gather's `op(B)` traffic from `nrhs` 32-byte sectors per nonzero to
  `ceil(nrhs*sizeof(T)/32)`.
* **No `nnz` key** (`key_names` in `src/ops/spmm/choice.hh`). `MatrixView::nnz()` is the batch-maximum capacity. The
  per-item count is in device memory (`row_offsets`), and selection must not read device memory (`spmm.cc:6-7`).

`direct` needs no workspace (`spmm.cc:114-120`). `spmm_buffer_size` runs the same choice as the call, so a
`direct` call never asks the vendor sizer.

## Measurement harness and hygiene

`benchmarks/spmm_benchmark.cc` times K back-to-back `spmm` calls on one in-order queue, closed by one `wait`, on
the host clock. Named cells are `(m, nnz/row, nrhs, batch)`: **L** = (1024, 3, 2, 512) lanczos, **M** = (1024, 16,
12, 512) LOBPCG, **S** = (2048, 16, 25, 128) LOBPCG.

* **No event timing.** Event timing costs about 0.36 ms per call. Cell L's ideal-traffic roof is 22.9 µs.
* **The vendor's per-call chain is inside the timed region for both arms**: `setStream`, the host walk over every
  item's row offsets, the `cusparseSpMM_bufferSize` re-query, and the `BumpAllocator` carve. Lanczos pays it on every
  call. `nsys` separates kernel time from host time ([The nsys split](#the-nsys-split)). Descriptors are created on
  the first, untimed call, so every ratio is against the warm vendor arm.
* **Warm-up is wall-clock, not call-count.** The SM clock idles at 210 MHz. Cell L's first row runs 2.3 % slow
  until the clock ramps. A fixed call count cannot price that: 250 calls take 40 ms on a cheap cell and 13.5 s on the
  54 ms `cdouble, m=4096, nrhs=50, b=512` cell. The default is `BATCHLAS_SPMM_WARM_MS` = 400 ms per row. With it, a
  fresh process's first row reads 0.161916 ms at rel_sd 0.0018.
* **Check the route.** Read the route from `BATCHLAS_COVERAGE_OUT` in every run. `BATCHLAS_SPMM_ROUTE=vendor`
  bypasses the window entirely, so it is a pin, not a control for the window.
* **Admission rule.** A row is admitted when either arm's rel_sd is <= 0.02 in both passes, **or** the two passes'
  ratios agree within 5 %. An rel_sd-only filter deleted a reproducible loss (see
  [Negative results](#negative-results)).
* Every sweep ran twice in independent processes, one route per process, device 1 pinned by the runner. Total: 7,536
  timed rows over 9 sweeps (6,512 main, 1,024 small-batch).

Reproducibility, worst ratio spread across two passes (`t_native/t_vendor`):

| pass pair | gate rows in both | worst spread | median spread | rows crossing 1.10 |
|---|---|---|---|---|
| `pass1`/`pass2` | 380 | 1.114 | 1.0034 | 4 (all `transA=Trans`) |
| `bnd1`/`bnd2` gather plane | 120 | 1.094 | 1.0041 | 0 |
| `bnd1`/`bnd2` banded family | 144 | 1.147 | 1.0040 | 0 |
| `bnd1`/`bnd2` scatter planes | 240 | 1.153 | 1.008 | 7 |

> **Warning:** `BATCHLAS_SPMM_ROUTE=vendor` over `spmm_tests` gives 276 passed and **92 failed**, all
> `Backend::NETLIB`. The 92 are the netlib transpose refusals, which the suite skips unless the pin is native. On
> `Backend::CUDA` the same pinned run is 184/184.

## The gather window

**Acceptance gate:** a clause may move a cell only if worst-of-two-passes `t_native/t_vendor <= 1.10` on every cell
it moves, at saturation (batch >= 128), with every boundary bracketed by measured rows on both sides.

`verdict.txt`, over 644 saturated, chk-agreeing rows (626 quiet, 18 admitted on cross-pass reproduction):

| clause | rows moved | verdict | worst-of-two | median | best |
|---|---|---|---|---|---|
| `transA == NoTrans`, unconditional | 186 | **FAILS** 1/186 | 1.934 | 0.446 | 0.032 |
| **shipped**: `AND NOT (cfloat && transB != NoTrans)` | 176 | passes | **0.968** | **0.445** | **0.032** |
| `AND NOT (cfloat && transB != NoTrans && nrhs >= 16)` | 183 | passes | 0.968 | 0.444 | 0.032 |

The two `nrhs`-narrowed variants pass their own grid but were rejected. Their boundary depends on the banded column
pattern, which the selection key cannot see. The 468 rows the shipped clause does not move contain 170 measured
non-winners.

### The cfloat transB exclusion

The one measured non-winner on the gather arm: `complex<float>`, `transB = Trans`, banded pattern, m=2048,
16 nnz/row, batch=512 (`cfedge1`, `cfedge2`). Ratios `t_native/t_vendor`, pass 1:

| nrhs | 8 | 9 | 12 | 16 | 17 | 20 | 25 | 32 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| cfloat banded | 0.630 | 0.713 | 0.689 | 1.087 | 1.315 | 1.218 | **1.731** | 1.159 | 1.695 |
| cfloat scattered | 0.670 | 0.856 | 0.888 | 0.963 | 0.979 | 0.793 | 1.000 | 1.019 | 0.953 |
| float banded (control) | 0.406 | 0.404 | 0.404 | 0.545 | 0.360 | 0.368 | 0.385 | 0.934 | 0.571 |

The loss is banded-exclusive: the scattered pattern is at parity (0.79-1.02). That is why the exclusion is stated by
type and `transB`, not by `nrhs`. By batch (banded, m=2048, 16 nnz/row, nrhs=25, worst of two passes), the loss
switches on between batch 4 and 8:

| type | b=1 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 |
|---|---|---|---|---|---|---|---|
| float | 0.655 | 0.485 | 0.402 | 0.409 | 0.403 | 0.406 | 0.471 |
| double | 0.341 | 0.221 | 0.503 | 0.634 | 0.625 | 0.621 | 0.613 |
| **complex\<float\>** | 0.520 | 0.581 | **1.447** | **2.072** | **2.182** | **2.092** | **1.944** |
| complex\<double\> | 0.335 | 0.432 | 0.688 | 0.676 | 0.681 | 0.681 | 0.669 |

At batch 1-4 the excluded family is a 1.7-1.9x native win that the clause declines. The cost is accepted. The
mechanism is unconfirmed: `kNCmax<Cx<float>>` is 8, so `nrhs=25` needs 4 passes with 7 idle lanes, while `nrhs=32`
needs 4 with none. `nrhs=32` measures 1.157-1.159 against 1.714-1.731 at `nrhs=25`.

### The batch axis has no floor

The window applies at every batch rung (1 to 16384). The acceptance gate is stated at batch >= 128. A separate
small-batch sweep (`run_smallbatch.sh`; 1,024 timed rows, 210 admitted) covers batch 1-128:

| batch | rows | worst | median | best | over 1.10 | max Δ µs/call (native − vendor) |
|---|---|---|---|---|---|---|
| 1 | 22 | 0.992 | 0.592 | 0.186 | 0 | −0.23 |
| 2 | 27 | 0.981 | 0.287 | 0.170 | 0 | −0.44 |
| 4 | 21 | **1.078** | 0.473 | 0.173 | 0 | **+13.46** |
| 8 | 25 | 0.956 | 0.460 | 0.154 | 0 | −3.64 |
| 16 | 25 | 0.966 | 0.499 | 0.154 | 0 | −11.04 |
| 32 | 29 | 0.978 | 0.373 | 0.134 | 0 | −16.17 |
| 64 | 25 | 0.967 | 0.412 | 0.099 | 0 | −23.51 |
| 128 | 30 | 0.964 | 0.305 | 0.066 | 0 | −36.00 |

The one row that costs the caller time: `complex<float>`, m=4096, 16 nnz/row, nrhs=50, scattered, `transB=NoTrans`,
batch 4, at 1.078 (+13.46 µs per call). It is non-monotonic on its own cell, so it is a launch artefact rather than a
structural loss. Inside the gate, this grid contains no measured non-winner at any batch rung. So no batch floor is
justified. `SpmmTranscribedTable.RowsHoldTheOldPreference` checks the ranking at batch 1, 300 and 100000.

Below batch ~64 the timed region is launch latency plus the vendor's per-call host chain. The small-batch ratios show
no harm and nothing more. The headline comes from the saturated grid alone.

### DRAM roof

Lanczos shape, batch 4096 and 8192, `transA = NoTrans`, footprints 151-1074 MB (2x to 15x the L2, so residency
cannot inflate the numbers). Fraction of the roof in brackets:

| type | pattern | cuSPARSE | native gather |
|---|---|---|---|
| float | either | 120-147 GB/s (0.12-0.15) | 910-928 GB/s (0.90-0.92) |
| double | either | 185-235 GB/s (0.18-0.23) | 906-931 GB/s (0.90-0.92) |
| complex\<float\> | either | 185-236 GB/s (0.18-0.23) | 918-931 GB/s (0.91-0.92) |
| complex\<double\> | banded | 306-366 GB/s (0.30-0.36) | 915-926 GB/s (0.91-0.92) |
| complex\<double\> | scattered | 308-360 GB/s (0.31-0.36) | 714-850 GB/s (0.71-0.84) |

The gather runs at the roof. cuSPARSE is 2.3-7.7x below it: 6.2-7.7x for `float`, 3.9-5.0x for `double` and
`complex<float>`, 2.3-3.0x for `complex<double>`.

In the LOBPCG regime (footprint > 288 MB, `transA=0`) neither arm is near the roof: vendor 0.06-0.16, native
0.09-0.52. The column-major `op(B)` gather binds both arms. The native kernel is still 1.3-4x faster there (LOBPCG
`transA=0` ratios 0.253-0.804 on the four types).

### Saturation and the L2 cliff

Saturation means that a wider batch buys < 10 % per item. The gather at `(float, m=1024, 16 nnz/row, nrhs=12)` runs
2.369, 0.788, 0.309, then 0.485 µs per item at batch 8/32/128/512. It rises at 512 because the 119 MB footprint leaves
the 72 MB L2. Per-item µs, float, m=1024, 3 nnz/row, nrhs=1, banded:

| batch | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| footprint | 37.8 MB (L2) | 75.5 MB | 151 MB | 302 MB |
| vendor | 0.308 | 0.304 | 0.306 | 0.306 |
| native | 0.010 | 0.020 | 0.040 | 0.040 |
| ratio | **0.032** | 0.066 | **0.131** | **0.130** |

cuSPARSE is flat from batch 256. The native gather is not saturated until about batch 4096. The batch-512 lanczos
ratios (0.032-0.072 for float) are L2-resident. Quote the DRAM-resident figure, **0.13-0.43** (2.3x-7.7x), instead.

### The transposed refusal

`transA != NoTrans` moves 458 saturated cells. **169 are over the 1.10 gate**. Median 1.030, worst 3.011.

| clause | rows moved | verdict | worst | refuting cell |
|---|---|---|---|---|
| `transA != NoTrans` | 458 | FAILS 169/458 | 3.011 | cdouble m=4096 nnz/row=16 nrhs=50 b=512 tB=0 scattered |
| `AND nrhs <= 4` | 204 | FAILS 11/204 | 1.208 | cdouble m=2048 nnz/row=16 nrhs=2 b=512 |
| `AND nrhs <= 1` | 60 | FAILS 2/60 | 1.132 | cdouble m=2048 nnz/row=16 nrhs=1 b=1024 |
| `AND nrhs <= 2 AND type != cdouble` | 111 | passes | 1.023 | fitted to `nnz/row` (below); rejected |

Every refuting cell is on the scattered pattern. The scatter's `nrhs` boundary moves with the type and with
`nnz/row` (`bnd_scatter_a`, m=1024, batch=512, scattered, worst of two passes):

| type \ nrhs | 1 | 2 | 4 | 8 | 12 | 25 |
|---|---|---|---|---|---|---|
| float (3 nnz/row) | 0.180 | 0.323 | 0.590 | 0.779 | 0.880 | 1.095 |
| double (3 nnz/row) | 0.194 | 0.350 | 0.636 | 0.996 | 1.076 | 1.103 |
| cfloat (3 nnz/row) | 0.344 | 0.632 | 0.820 | 0.989 | 1.137 | 1.107 |
| cdouble (3 nnz/row) | 0.390 | 0.683 | 0.968 | 1.118 | 1.105 | 1.289 |
| **cdouble (16 nnz/row)** | **1.043** | 1.078 | 1.112 | 1.164 | 1.206 | 1.307 |

At m=2048, cdouble with 16 nnz/row loses at `nrhs = 1` (1.098/1.101 at b=512, 1.130/1.132 at b=1024). The selection
key has no `nnz`, so no shipped row separates this loss from the cdouble win at 3 nnz/row. The one passing clause
needs a type exclusion. It moves 111 cells and has no in-tree C++ caller.

The scatter stays runnable. `direct` admits every `transA`, so `BATCHLAS_SPMM_ROUTE=direct` reaches it. In a
vendor-free build the second entry, `direct`, runs.

### The nsys split

Both arms spend > 93 % of wall time in GPU kernels, so the win is a kernel result (`nsys_split.txt`). Per call, float:

| cell | arm | wall | GPU kernel | host share |
|---|---|---|---|---|
| m=1024 nnz/row=3 nrhs=2 b=512 | native | 0.01196 ms | 0.01124 ms | 6.0 % |
| same | vendor | 0.16239 ms | 0.16461 ms | −1.4 % |
| m=1024 nnz/row=3 nrhs=2 b=4096 | native | 0.20227 ms | 0.20087 ms | 0.7 % |
| same | vendor | 1.25856 ms | 1.26342 ms | −0.4 % |
| m=2048 nnz/row=16 nrhs=25 b=128 | native | 0.33853 ms | 0.34594 ms | −2.2 % |
| same | vendor | 0.86439 ms | 0.91118 ms | −5.4 % |
| same, `transA=Trans` | native (scatter + scale) | 2.10142 ms | 2.17493 ms | −3.5 % |
| same, `transA=Trans` | vendor | 1.87837 ms | 1.99502 ms | −6.2 % |

Negative host shares are profiler accounting skew. They mean the per-call host chain is below the comparison's
resolution.

cuSPARSE launches three kernels per call (`csrmm_alg1_kernel`, `csr_partition_kernel`,
`matrix_scalar_multiply_kernel`) and re-partitions the CSR rows every call. `csr_partition_kernel` is 36 % of the
vendor time at batch 512. The native gather launches one kernel. The transposed arm launches two (scale, then
scatter). That row is 1.12x the vendor on the same cell.

## SpMM: the kernel contract

The operation is \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C\f$, with `A` batched CSR (one strided slab
per item) and `B`, `C` dense column-major. The three bodies sit behind the `direct` family.

* **CSR indexing** (`src/matrix.cc`): row offsets are item-local, indexed `b*offset_stride()`. Values and column
  indices are indexed `b*matrix_stride()`. `A.nnz()` is a capacity, so the only legal bound on the nonzero loop is
  `row_offsets[ro+i+1]`. Slots above an item's own nnz are uninitialised. This is correct at batch 1 and wrong at
  batch 2.
* **`beta == 0` must not read `C`.** Callers pass never-zeroed `BumpAllocator` memory, and `0 * NaN` is NaN.
  `alpha == 0` leaves `A` and `B` unread but still computes `C = beta*C`. This does not take the `?GEMV`
  quick return.
* **No `__restrict__` on any pointer, and no pointer arrays.** LOBPCG passes `X`, `P`, `R` as element-disjoint slices
  of one buffer, which alias at the object level.
* **The transposed arm scatters through global atomics.** Summation order varies between runs, so no test may
  compare two runs bitwise. Its FP64 instantiations require the `atomic64` aspect.
* **`B` and `C` carry their own `ld` and batch stride.** The bodies read them from the view.

## SpMM: correctness findings

### Three vendor defects, found here and fixed {#three-vendor-defects-found-here-and-fixed}

The three are shipping wrong answers that predate `direct`. `tests/spmm_tests.cc` (368 cases: four types, all nine
transpose spellings, heterogeneous `nnz`, empty rows, padded strides, alpha/beta corners) is the first suite to cover
these axes.

1. **`netlib_lapack.cc:248,272`: `spmm` read `A` at `alpha == 0` and `C` at `beta == 0`.** `0 * NaN` poisons the
   result. The host arm now skips the alpha term and substitutes `T(0)` for the beta term.
2. **`cusparse.cc` mapped `ConjTrans` to `CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE` for real scalars.** On a real
   scalar `ConjTrans` is `Trans`, and `CUDA_R_32F` and `CUDA_R_64F` returned wrong results. It applied to both
   operands. The complex arms, which use the enum correctly, hid it.
3. **Heterogeneous `nnz` and padding over-read.** `cusparseCreateCsr` takes one `nnz`, and
   `backend_handle_impl.hh:63` passed `A.nnz()`, the per-item capacity. Short items' descriptors covered values the
   conversion never wrote. The result was wrong last rows, and `CUDA_ERROR_ILLEGAL_ADDRESS` with a dead process on the
   padding case. The fix takes `nnz` from each item's row offsets. A uniform batch keeps one batched call. A
   non-uniform batch issues one `cusparseSpMM` per item.

### The eleventh blind guard

Four deliberate breaks were run against the transposed path. `B4` (`scatterBound`, a transposed `nnz` bound that
reads padding) was **green on all 352 cases** the first time.

The cause: `PaddingAboveNnzIsNotReadTrans` poisoned padding with NaN at an **out-of-range** column index (2^30). The
scatter's range guard (`spmm_native.cc:367-375`) skips that entry before it is multiplied. The test passed because
of the kernel's guard, not the property it names. Controls showed the over-read is real: with the bound broken and
the guard removed, the case segfaults (exit 139, `double`). With the bound correct, 352/352 pass.

The test now poisons with an **in-range** column and a large finite value, so the over-read lands in `C` where the
backward-error comparison names the `(batch, col, row)`. The out-of-range case is kept as
`PaddingAboveNnzOutOfRangeIsNotReadTrans`, the range guard's only regression test. It does not cover the bound.
Rules that followed:

* A poison must reach the assertion. Check what each body does with it, since a defensive predicate in between makes
  the case vacuous.
* Coverage rows prove that some shape resolved to a route, not that this shape ran that body. Break the body.
* A break must be as narrow as the contract it denies. A break that also turns homogeneous batches red identifies
  nothing.
* `git diff` cannot verify the revert of an untracked file. `spmm_native.cc` is untracked in this package. Verify the
  revert with `md5sum` against a copy taken before the first break.

The transposed tolerance denominator is a backward-error scale (sum of `|a|*|b|`, floored at 1), not `|expected|`. A
transposed case whose backend throws is skipped unless the pin is native (`tests/spmm_tests.cc:366-373`). A
vendor returning a status code instead of throwing is not covered by the skip. `cusparse.cc` checks no status, so
`CUSPARSE_STATUS_NOT_SUPPORTED` leaves `C` untouched and is reported as a wrong answer.

### Two defects filed, not fixed

* **`rocsparse.cc:30-31` and `:62-63`** carry cuSPARSE defect 2 unchanged. A real-scalar `ConjTrans` becomes
  `rocsparse_operation_conjugate_transpose`. **Unmeasured:** there is no AMD device here.
* **`netlib_lapack.cc:508,520,537,549`**: `trsm` reads `B` at `alpha == 0` (`0 * NaN`). Outside this op.

## Negative results

1. **No shippable window exists for the transposed scatter.** 169 of 458 saturated cells lose, and the one passing
   clause is fitted to `nnz/row`, which the key cannot see.
2. **The design prediction of parity in the LOBPCG regime was refuted.** The gather is 1.3-4x faster there. The
   margin comes from cuSPARSE paying the `op(B)` wall and re-partitioning the CSR every call, not from beating the
   wall. Neither arm exceeds 0.52 of the roof.
3. **The lanczos ratios shrink 4x at true saturation** (0.032 at batch 1024, 0.131 at batch 4096).
4. **An rel_sd-only filter manufactured a passing clause** by deleting the reproducible non-winner (1.934 / 1.872,
   pass-2 rel_sd 0.033).
5. **`complex<double>` transposed at 16 nnz/row loses at `nrhs = 1`** for m >= 2048.
6. **The `cfloat` + `transB=Trans` loss is not a saturation effect.** It is 0.581 at b=4, 1.447 at b=8, 2.182 at
   b=32 and 1.71-1.73 at b=512.
7. **The `cfloat` gather margin is about 3 %, not 2x.** Across the batch ladder it is 0.910-1.078.
8. **Reversing the route-era order array did not send admitted shapes back to cuSPARSE.** Exactly one unit test went
   red. Under flat selection the candidate order is only the tie-break (`choice.hh:19-20`). The ranking is in each
   table row.
9. **Vendor-free suite counts are the wrong metric for this op.** The metric that moved is the per-op `NoRouteError`
   count: `spmm` went from 2 to 0. Six recovered cases are the type conditional: `transA = NoTrans` with
   `transB = Trans`/`ConjTrans` on `Backend::NETLIB` now takes the native gather for `float`, `double` and
   `complex<double>`. `complex<float>` keeps its 23 skips. Vendor-present, `spmm_tests` is 282 passed, 86 skipped, 0
   failed, unpinned.

## Open debts

* **The transposed scatter has no winning row, and zero in-tree C++ callers.** Nothing exercises it in practice.
* **The tables are transcribed, not timed.** The grid above is sm_89 only. The sm_120 and CPU rows carry the same
  clause without measurement. The next retune would replace them with timed rows.
* **The `cfloat` gather margin is about 3 %.** A kernel change costing 5 % on `complex<float>` turns admitted cells
  into losses, and no test notices.
* **The `nrhs >= 16` narrowing** passes `verdict.txt` at worst 0.968 and would move 183 cells instead of 176. It is
  refused because its axis is the column pattern. Both it and the scatter's `nrhs <= 2 && !cdouble` clause become
  re-arguable if the key gains an honest pattern or `nnz` signal.
* **The `kNCmax` mechanism for the `cfloat` loss is a hypothesis**, not a profile.
* **`FP64` scatter needs `atomic64`.** Both development devices have it. On a device without it the failure is a
  launch-time kernel-selection error. Untested.
* **Coverage cannot distinguish scale from scatter**, and rows are keyed on a power-of-two `shape_class`,
  first-writer-wins. A CSR and a Dense `spmm` at the same extents would share a row. This is unobservable today. Do
  not add a format bit: it invalidates every stored `.routes` baseline.
* **One GPU only.** There is no second GPU generation, no AMD device, and no CPU-queue timing, so "no `is_gpu` gate"
  is a correctness decision, not a measured claim about `native_cpu`. The two 4090s share a NUMA node and one UVM
  driver. A sweep on the other card has inflated a cell 5.5x at a stable rel_sd. Verify both devices on re-runs.

## Raw evidence

Raw data is at the git tag `perf-evidence/vendor-independence`: `git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| clause table, every candidate and its refuting cell | `experiments/sparse_spmm/verdict.txt` (from `verdict.py`) |
| main grid, two passes | `experiments/sparse_spmm/pass1/`, `pass2/`, `report_pass12.txt` |
| DRAM roof, footprint > 288 MB | `experiments/sparse_spmm/roof.txt`, `roof.py` |
| kernel and host split | `experiments/sparse_spmm/nsys_split.txt`, `nsys_split.sh`, `nsys.log` |
| saturation, batch 1024-8192 | `experiments/sparse_spmm/sat1/`, `sat2/`, `run_satext.sh`, `tables.txt` |
| cfloat nrhs walk | `experiments/sparse_spmm/cfedge1/`, `cfedge2/`, `run_cfloat_edge.sh` |
| nnz/row x nrhs boundaries, scatter batch ladder | `experiments/sparse_spmm/bnd1/`, `bnd2/`, `scl1/`, `scl2/`, `run_boundary.sh`, `run_scatter_ladder.sh` |
| small-batch corner, batch 1-128 | `experiments/sparse_spmm/sb1/`, `sb2/`, `run_smallbatch.sh`, `smallbatch.txt`, `sb_report.py` |
| warm-up ramp, route-pin probes | `experiments/sparse_spmm/probe/warmup_probe.sh`, `probe/order_probe.sh` |
| route census (65 decisions) | `experiments/sparse_spmm/route_census.py`, `scripts/route_diff.sh` captures |
| campaign summary and vendor-defect list | `VENDOR_INDEPENDENCE_PLAN.md`, "WP8" sections |
| coverage-instrument limits | `VENDOR_FREE_BASELINE.md` |

Reproduction (device 1 exclusive; `run_all.sh` takes about 21 min per pass, `run_smallbatch.sh` about 25 min):

```bash
cmake --build build --target spmm_benchmark -j16
cd experiments/sparse_spmm
./run_all.sh pass1; ./run_all.sh pass2   # then run_boundary / run_cfloat_edge / run_satext /
./run_nsys.sh                            # run_scatter_ladder / run_smallbatch, each twice
python3 verdict.py pass{1,2}/joined.csv scl{1,2}/joined.csv
```

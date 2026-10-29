# SPMM: batched CSR kernel and routing {#perf_spmm}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda

Batched CSR `spmm` has two families: the native `direct` kernel and the cuSPARSE `vendor` call. This page gives
the routing rule, the measurements behind the native window, the kernel contract and the vendor defects the suite
found. Selection is flat (@ref design_flat_selection). Numbers are from one RTX 4090 (128 SMs, 72 MB L2, 1008 GB/s
DRAM roof), in-order SYCL queue.

## Choices

| spelling | implementation | `can_run` (correctness only) |
|---|---|---|
| `direct` | `sycl_spmm::spmm_native_csr` | CSR; the body for this `transA` compiled; `one_spmm()` shape checks; no heterogeneous B or C; batch >= 1 |
| `vendor` | `backend::spmm_vendor` (cuSPARSE, rocSPARSE, netlib) | a sparse vendor library, no heterogeneous B or C, not netlib with any transpose, not CUDA complex `transB = ConjTrans` with one row of B ([known-defects #17](../design/known-defects.md#defect-17-cusparse-spmm-shapes-refused-in-can_run-and-the-alignment-waiver)) |

Both refuse a masked batch. Selection takes the first runnable entry of the nearest row of
`tuned/spmm.<device>.txt`, keyed `dtype:exact transA:exact transB:exact m:log nrhs:log batch:log`; `ConjTrans`
folds to `T`. The last resort is `vendor`, then `direct`. `BATCHLAS_SPMM_ROUTE` takes `auto`, `native`, `vendor`
or a spelling; `native`/`vendor` fall back to auto with a warning when nothing in the class can run, and a spelling
that is invalid or refused by `can_run` throws.

`direct` has three bodies in `src/ops/spmm/kernels/spmm_native.cc`: the `NoTrans` gather (body 1), and a scale plus
atomic scatter pair (bodies 0 and 2) for `Trans` and `ConjTrans`, chosen in the launcher on `transA`. `transA` is in
the coverage `variant_key`, so gather and scatter stay separable in `scripts/route_diff.sh`; scale and scatter do not.

### Ranking in the tables

`tuned/spmm.sm_89.txt` and `tuned/spmm.cpu.txt` are transcribed (untimed; provenance in `tuned/README.md`).
`tuned/spmm.sm_120.txt` is deep-measured, with 24 transcribed rows. The transcribed ranking:

| rows | float, double, complex\<double\> | complex\<float\> |
|---|---|---|
| `transA=N transB=N` | `direct`, then `vendor` | `direct`, then `vendor` |
| `transA=N transB=T` | `direct`, then `vendor` | `vendor`, then `direct` |
| `transA=T`, either `transB` | `vendor`, then `direct` | `vendor`, then `direct` |

It is identical at all 125 `(m, nrhs, batch)` cells of each block; no batch, extent, `is_gpu` or `nnz` term
changes it. `SpmmTranscribedTable.RowsHoldTheTranscribedPreference` (`tests/spmm_candidates_tests.cc`) pins it.

## supports(), and what is deliberately not in it

The `direct` terms in `src/ops/spmm/spmm.cc` are correctness gates. `one_spmm()` checks matching extents and batch,
positive `ld`s and an offset stride of at least `m + 1`. An empty batch returns before selection. Per-item `nnz`
variation is handled through the row offsets, so a CSR view is never heterogeneous.

* **No `is_gpu` gate.** No body uses local memory, group or sub-group collectives; the vendor-free build runs
  `direct` on the host backend.
* **No transpose refusal.** All nine `(transA, transB)` spellings are served. `transB = Trans` cuts the gather's
  `op(B)` traffic from `nrhs` 32-byte sectors per nonzero to `ceil(nrhs*sizeof(T)/32)`.
* **No `nnz` key.** `MatrixView::nnz()` is the batch-maximum capacity; the per-item count is in device memory, and
  selection must not read device memory.

`direct` needs no workspace; `spmm_buffer_size` runs the same choice, so a `direct` call never asks the vendor sizer.

## Measurement harness and hygiene

`benchmarks/spmm_benchmark.cc` times K back-to-back calls on one in-order queue, closed by one `wait`, on the host
clock. Named cells `(m, nnz/row, nrhs, batch)`: **L** = (1024, 3, 2, 512) lanczos, **M** = (1024, 16, 12, 512) and
**S** = (2048, 16, 25, 128) LOBPCG.

* **No event timing.** It costs about 0.36 ms per call; cell L's ideal-traffic roof is 22.9 µs.
* **The vendor's per-call host chain is timed** (`setStream`, the host walk of row offsets, the buffer-size
  re-query, the `BumpAllocator` carve). Descriptors are created on the first, untimed call. `nsys` shows both arms
  spend > 93 % of wall time in GPU kernels, so the win is a kernel result.
* **Warm-up is wall-clock.** The SM clock idles at 210 MHz; a call count cannot price the ramp. Default
  `BATCHLAS_SPMM_WARM_MS` = 400 ms per row.
* **Check the route** from `BATCHLAS_COVERAGE_OUT` in every run. `BATCHLAS_SPMM_ROUTE=vendor` is a pin, not a control.
* **Admission rule.** A row is admitted when either arm's rel_sd is <= 0.02 in both passes, **or** the two passes'
  ratios agree within 5 %. An rel_sd-only filter deleted a reproducible loss (1.934 / 1.872) and manufactured a
  passing clause.
* Every sweep ran twice in independent processes, one route per process. Cross-pass median spread is 0.3-0.8 %.

> **Warning:** `BATCHLAS_SPMM_ROUTE=vendor` over `spmm_tests` fails 92 `Backend::NETLIB` cases: the netlib
> transpose refusals, which the suite skips only when the pin is native.

## The gather window

**Acceptance gate:** a clause may move a cell only if worst-of-two-passes `t_native/t_vendor <= 1.10` on every cell
it moves, at batch >= 128, with every boundary bracketed by measured rows. Over 644 saturated rows (`verdict.txt`):

| clause | rows moved | verdict | worst | median |
|---|---|---|---|---|
| `transA == NoTrans` | 186 | fails 1/186 | 1.934 | 0.446 |
| **shipped**: `AND NOT (cfloat && transB != NoTrans)` | 176 | passes | **0.968** | **0.445** |
| `AND NOT (cfloat && transB != NoTrans && nrhs >= 16)` | 183 | passes; rejected | 0.968 | 0.444 |

The `nrhs`-narrowed variant was rejected because its boundary depends on the banded column pattern, which the key
cannot see.

The gather runs at 0.90-0.92 of the DRAM roof at batch 4096-8192 (footprint 2-15x L2); cuSPARSE reaches 0.12-0.36,
so the DRAM-resident native/vendor ratio is 0.13-0.43. Batch-512 lanczos ratios (0.032) are L2-resident; do not
quote them. In the LOBPCG regime neither arm exceeds 0.52 of the roof and the gather is still 1.3-4x faster.

### The cfloat transB exclusion

`complex<float>`, `transB = Trans`, banded, m=2048, 16 nnz/row, batch 512: the ratio is 0.63-0.71 at `nrhs` 8-12
and 1.09-1.73 at `nrhs` 16-50 (worst 1.731 at 25); the scattered pattern stays at 0.79-1.02. By batch (nrhs=25) the
loss switches on between batch 4 (0.581) and 8 (1.447), peaking at 2.18. Hence the exclusion is by type and
`transB`, not `nrhs`, and it declines a 1.7-1.9x native win at batch 1-4. The mechanism is a hypothesis:
`kNCmax<Cx<float>>` is 8, so `nrhs=25` runs 4 passes with 7 idle lanes.

### The batch axis has no floor

A small-batch sweep (batch 1-128, 210 admitted rows) has no row over 1.10 at any rung; the worst is 1.078 at batch 4
(+13.5 µs/call, non-monotonic on its cell). Below batch ~64 the timed region is launch latency plus the vendor's host
chain, so these ratios show only absence of harm.

### The transposed refusal

`transA != NoTrans` moves 458 saturated cells and fails the gate on 169 (worst 3.011):

| clause | rows moved | verdict | worst | refuting cell |
|---|---|---|---|---|
| `transA != NoTrans` | 458 | fails 169/458 | 3.011 | cdouble m=4096 nnz/row=16 nrhs=50 b=512 scattered |
| `AND nrhs <= 4` | 204 | fails 11/204 | 1.208 | cdouble m=2048 nnz/row=16 nrhs=2 b=512 |
| `AND nrhs <= 1` | 60 | fails 2/60 | 1.132 | cdouble m=2048 nnz/row=16 nrhs=1 b=1024 |
| `AND nrhs <= 2 AND type != cdouble` | 111 | passes; rejected | 1.023 | fitted to `nnz/row` |

Every refuting cell is scattered, and the `nrhs` boundary moves with type and `nnz/row` (cdouble at 16 nnz/row
loses already at `nrhs = 1`), which the key cannot see. The scatter stays runnable: `BATCHLAS_SPMM_ROUTE=direct`
reaches it, and a vendor-free build runs it as the second entry.

### The nsys split

cuSPARSE launches three kernels per call and re-partitions the CSR every call (`csr_partition_kernel` is 36 % of
vendor time at batch 512); the gather launches one, the transposed arm two (scale, then scatter).

## SpMM: the kernel contract

\f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C\f$, with `A` batched CSR (one strided slab per item) and
`B`, `C` dense column-major.

* **CSR indexing** (`src/matrix.cc`): row offsets are indexed `b*offset_stride()`, values and column indices
  `b*matrix_stride()`. `A.nnz()` is a capacity, so the only legal bound on the nonzero loop is
  `row_offsets[ro+i+1]`; slots above an item's nnz are uninitialised.
* **`beta == 0` must not read `C`.** Callers pass never-zeroed `BumpAllocator` memory. `alpha == 0` leaves `A` and
  `B` unread but still computes `C = beta*C`.
* **No `__restrict__`, no pointer arrays.** LOBPCG passes element-disjoint slices of one buffer.
* **The transposed arm scatters through global atomics.** Summation order varies, so no test compares two runs
  bitwise. Its FP64 instantiations require the `atomic64` aspect.
* **`B` and `C` carry their own `ld` and batch stride.**

## SpMM: correctness findings

### Three vendor defects, found here and fixed {#three-vendor-defects-found-here-and-fixed}

`tests/spmm_tests.cc` (four types, nine transpose spellings, heterogeneous `nnz`, empty rows, padded strides,
alpha/beta corners) found three shipping wrong answers:

1. **netlib `spmm` read `A` at `alpha == 0` and `C` at `beta == 0`.** The host arm now skips the alpha term and
   substitutes `T(0)` for the beta term.
2. **`cusparse.cc` mapped `ConjTrans` to `CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE` for real scalars**, on both
   operands, giving wrong `CUDA_R_32F`/`CUDA_R_64F` results.
3. **Heterogeneous `nnz` over-read.** `cusparseCreateCsr` was given the per-item capacity, so short items covered
   unwritten values (wrong last rows; `CUDA_ERROR_ILLEGAL_ADDRESS` on padding). `nnz` now comes from each item's row
   offsets: a uniform batch keeps one batched call, a non-uniform one issues one `cusparseSpMM` per item.

The rocSPARSE analogue of defect 2 and netlib `trsm` reading `B` at `alpha == 0` are open:
[known-defects 4 and 5](../design/known-defects.md).

### The eleventh blind guard

The break `scatterBound` (a transposed `nnz` bound that reads padding) was green on all 352 cases:
`PaddingAboveNnzIsNotReadTrans` poisoned padding at an out-of-range column, which the scatter's range guard skips
before multiplying. The test now poisons with an in-range column and a large finite value, so the over-read reaches
`C`. `PaddingAboveNnzOutOfRangeIsNotReadTrans` keeps the out-of-range case as the range guard's only regression
test. Rules:

* A poison must reach the assertion; a defensive predicate in between makes the case vacuous.
* Coverage rows prove that some shape resolved to a route, not that this shape ran that body. Break the body.
* A break must be as narrow as the contract it denies.

The transposed tolerance denominator is a backward-error scale (sum of `|a|*|b|`, floored at 1). A transposed case
whose backend throws is skipped unless the pin is native. `cusparse.cc` checks no status, so a
`CUSPARSE_STATUS_NOT_SUPPORTED` leaves `C` untouched and shows as a wrong answer, not a skip.

## Open debts

* **The transposed scatter has no winning sm_89 row and no in-tree C++ caller.** The deep `tuned/spmm.sm_120.txt`
  ranks `direct` first in 1055 of its 1823 `transA=T` rows; the refusal above is sm_89 only.
* **The sm_89 and CPU tables are transcribed, not timed.**
* **The `cfloat` gather margin is about 3 %** (0.910-1.078 across the batch ladder). A 5 % kernel regression on
  `complex<float>` turns admitted cells into losses, and no test notices.
* **The `nrhs >= 16` narrowing and the scatter's `nrhs <= 2 && !cdouble` clause** become re-arguable if the key
  gains an honest pattern or `nnz` signal.
* **FP64 scatter needs `atomic64`.** Without it the failure is a launch-time kernel-selection error. Untested.
* **Coverage cannot distinguish scale from scatter**, and rows are keyed on a power-of-two `shape_class`,
  first-writer-wins, so a CSR and a Dense `spmm` at the same extents would share a row. Do not add a format bit: it
  invalidates every stored `.routes` baseline.
* **No `is_gpu` gate is a correctness decision**, not a measured claim about `native_cpu`: there is no CPU-queue
  timing. The two 4090s share one UVM driver; a sweep on the other card inflated a cell 5.5x at a stable rel_sd.

## Raw evidence

`git show perf-evidence/vendor-independence:<path>`, all under `experiments/sparse_spmm/`:

| topic | path |
|---|---|
| clause table, every candidate and its refuting cell | `verdict.txt` (from `verdict.py`) |
| main grid, two passes | `pass1/`, `pass2/`, `report_pass12.txt` |
| DRAM roof | `roof.txt`, `roof.py` |
| kernel and host split | `nsys_split.txt`, `nsys_split.sh` |
| saturation, batch 1024-8192 | `sat1/`, `sat2/`, `run_satext.sh` |
| cfloat nrhs walk | `cfedge1/`, `cfedge2/`, `run_cfloat_edge.sh` |
| nnz/row x nrhs boundaries, scatter ladder | `bnd1/`, `bnd2/`, `scl1/`, `scl2/`, `run_boundary.sh`, `run_scatter_ladder.sh` |
| small batch, 1-128 | `sb1/`, `sb2/`, `run_smallbatch.sh`, `smallbatch.txt` |

Reproduce on an exclusive device (`run_all.sh` about 21 min per pass):

```bash
scripts/b spmm_benchmark
cd experiments/sparse_spmm
./run_all.sh pass1; ./run_all.sh pass2   # likewise run_boundary / run_cfloat_edge / run_satext /
./run_nsys.sh                            # run_scatter_ladder / run_smallbatch, each twice
python3 verdict.py pass{1,2}/joined.csv scl{1,2}/joined.csv
```

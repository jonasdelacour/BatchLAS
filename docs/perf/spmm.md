# SPMM: the native batched CSR kernel and the three vendor defects it found (WP8)

Batched CSR `spmm` had no native kernel, no route table and no measurement in this repository before WP8. It now has one native
route with three kernel bodies, a `preferred()` window that takes the `transA == NoTrans` gather away from cuSPARSE in **every**
build, and a measured refusal for the transposed arm. Hardware for every number below: one RTX 4090 (device 1, 128 SMs, 72 MB L2,
1008 GB/s DRAM roof), CUDA backend, SYCL in-order queue.

## What ships

### Route arms

`kSpmmOrder` (`include/batchlas/blas/dispatch/route_spmm.hh:33-36`) has exactly two entries:

| Origin | Algorithm | bodies behind it |
|---|---|---|
| `Native` | `Direct` | three, in `src/sycl/spmm_native.cc`: the `NoTrans` gather (body 1), and the scale + atomic scatter pair (bodies 0 and 2) that together serve `Trans`/`ConjTrans` |
| `Vendor` | `Auto` | `cusparseSpMM` / rocSPARSE / netlib |

Body selection is in the launcher, on `transA` (`spmm_native.cc:386-417`, under the note at `:560-565`), deliberately below the
routing vocabulary — the decomposition-not-algorithm rule `gemv` already uses, so no `Algorithm` enumerator,
`to_string(Algorithm)` case or `parse_algorithm_word` case ships. `transA` *is* in `variant_key`, so gather-vs-scatter stays
separable in `scripts/route_diff.sh`; scale-vs-scatter does not.

The env variable is `BATCHLAS_SPMM_ROUTE`, read through the shared `origin[:algorithm]` grammar
(`route_env.hh:50-70`): `native:direct`, `native`, `direct`, `vendor`, `auto`. A misspelling is **silent** —
`ParsedRouteEnv::unparsed` is discarded and every decision goes to the vendor with no message; reproduced deliberately, see
[Measurement harness and hygiene](#measurement-harness-and-hygiene).

### The preferred window, as implemented

`route_spmm.hh:65-75`, stripped of its ~140 lines of evidence comments:

```cpp
static bool preferred(Route r, const SpmmShape& s) {
    if (!is_native(r) || r.algo != Algorithm::Direct) return false;   // :66
    if (s.format != MatrixFormat::CSR) return false;                  // :67
    if (s.transA != Transpose::NoTrans) return false;                 // :69
    if constexpr (std::is_same_v<T, std::complex<float>>) {           // :71
        if (s.transB != Transpose::NoTrans) return false;             // :72
    }
    return true;                                                      // :74
}
```

**No batch term, no extent term, no `is_gpu` term, no `nnz`/density term** — each absence is a measured decision. The exploration
notes label this clause "recommended"; it is what shipped, verbatim, including the small-batch caveat's closure in favour of no
floor. `preferred()` is consulted by `automatic()`'s **first** walk, which runs regardless of `vendor_available`, so it moves the
default in a vendor-present build too. Route census after the flip: **65 moved decisions, every one `spmm`, every one `vendor:auto`
→ `native:direct`, every one `transA = 0`; zero non-`spmm` decisions moved and none disappeared** (4,107 → 4,217 distinct
decisions). The per-type split *is* the clause, read back out of the library: `complex<float>` moves at `transB = NoTrans` only, the
other three types move at all three `transB` spellings, and nothing at all moves at `transA != NoTrans`.

**That number is not readable off `route_diff.sh compare`.** The raw before/after capture diff shows **65 removed and 175 added
`reached` lines**, of which **110 additions are fabricated pure-layer shapes recorded with `backend = AUTO`** by this pass's new
`route_vocabulary_tests` cases. `compare` applies no `backend != AUTO` filter, so it turns a clean 65-decision move into 240 lines
of apparent churn; `experiments/sparse_spmm/route_census.py` keys on the decision tuple and splits `AUTO` from real backends first.
`VENDOR_FREE_BASELINE.md` files that as arguably a **fifth defect of the coverage instrument**.

### supports(), and what is deliberately not in it

`route_spmm.hh:42-61`. Correctness gates only: CSR format (`:198`); no heterogeneous *dense* batch (`:211` — a CSR view is never
heterogeneous in the `active_rows_` sense, and per-item `nnz` variation is handled exactly through the row offsets); no negative
extent, no empty batch (`:217`); then the capability flag for the body that would actually run (`:250-251`). Three absences are WP8
deliverables:

* **No `is_gpu` gate.** Every body is a plain loop — zero local memory, no group or sub-group collective, no required sub-group
  size. `build-novendor` has `BATCHLAS_HAS_HOST_BACKEND 1` with `BATCHLAS_HAS_LAPACKE` and `BATCHLAS_HAS_CBLAS` both 0, so
  `Backend::NETLIB` `spmm` symbols exist and threw `NoRouteError`; a GPU gate would move that half of the burn-down by exactly zero.
* **No transpose refusal.** All nine `(transA, transB)` spellings are served. Refusing `transB` would foreclose the caller-side
  layout lever: handing the dense block as `transB = Trans` collapses the gather's `op(B)` touch from `nrhs` 32-byte sectors per
  nonzero to `ceil(nrhs*sizeof(T)/32)`.
* **No `nnz` field on `SpmmShape`** (`route_spmm.hh:30-31`). `MatrixView::nnz()` is the per-item *capacity* (the batch maximum);
  the honest per-item `nnz(b)` reads `row_offsets`, device memory the same builder touches from `spmm_buffer_size`, where a read is
  a segfault rather than a wrong route. This single constraint kills both rejected wider clauses below.

Zero workspace: `spmm_native_csr` takes no `Span<std::byte>`, so query and call agree by construction. `spmm_buffer_size` folds a
named `kSpmmNativeDirectNeed = 0` through the usual `max(native, vendor)` and gates its consistency check on `native_fired`, never
on `native_need != 0` — the need is exactly zero on every shape, so the `!= 0` spelling would throw on every call the route table
had just accepted.

## Measurement harness and hygiene

`benchmarks/spmm_benchmark.cc` times K back-to-back `spmm` calls on one in-order queue, closed by a single `wait`, on the host
clock. Named cells are `(m, nnz/row, nrhs, batch)`: **L** = (1024, 3, 2, 512) lanczos, **M** = (1024, 16, 12, 512) LOBPCG, **S** =
(2048, 16, 25, 128) LOBPCG.

* **Event timing is deliberately not installed.** `make_event_timed_kernel_ms` costs a recorded ~0.36 ms per call
  (recorded at `VENDOR_INDEPENDENCE_PLAN.md:1949`; `spmm_benchmark.cc:6` cites this section back); cell L's
  ideal-traffic roof is 22.9 µs, so event timing would be 15x the thing being measured.
* **The vendor's per-call chain is inside the timed region for both arms**: `setStream`, the `SpmmCsrBatchPlan` host walk over every
  item's row offsets, the `cusparseSpMM_bufferSize` **re-query on every call**, the `BumpAllocator` carve. No caller can hoist it —
  it lives inside `spmm_vendor` and lanczos pays it `n` times per solve. Because an end-to-end ratio that is really a host-overhead
  ratio must not be sold as a kernel win, `nsys` decomposes both arms ([The nsys split](#the-nsys-split)). `MatrixView`s are built
  once, so the cuSPARSE descriptors are created on the first (untimed) call: every ratio here is against the *warm* vendor arm.
* **The clock ramp was priced, then budgeted.** The SM clock idles at 210 MHz (logged in every `probe/*.log` header) against a
  boost clock the notes give as 2805 MHz *(unverified: no source in this tree records it; the 2.3 % ramp below does not depend on
  it)*. Cell L, vendor: with minibench's
  default 2 warm-up calls a process's first row reads 0.16544 ms at rel_sd 0.0495 and its second 0.16196 ms at rel_sd 0.0019; 250
  warm-up calls bring the first row to 0.16179 ms at rel_sd 0.0016, and 2000 calls to 0.16175. The 2.3 % ramp lands entirely on
  whichever row runs first, and a CALL-counted warm-up cannot price it uniformly — 250 calls is 40 ms on a cheap cell and 13.5 s of
  dead time on the 54 ms `cdouble, m=4096, nrhs=50, b=512` cell. Hence a wall-clock budget (`BATCHLAS_SPMM_WARM_MS`, default 400
  ms/row); with it a fresh process's first row reads 0.161916 ms at rel_sd 0.0018, converged.
* **The route pin is proved, not asserted.** Same cell (float, m=1024, 3 nnz/row, nrhs=2, b=512), three processes: `vendor` →
  0.162654 ms, coverage says `vendor:auto`; `native:direct` → 0.011745 ms, coverage says `native:direct` (13.8x apart); `bogus_typo`
  → 0.162707 ms, coverage says **`vendor:auto`**. The typo is indistinguishable from `vendor`. Cross-arm `chk` (L1 norm of batch
  item 0 of C) is 2211 for all three.
* **The filter that manufactured a win.** An rel_sd-only admission rule silently deleted the single most important negative result
  in the sweep: `(cfloat, tA=0, tB=1, banded, m=2048, nnz/row=16, nrhs=25, b=128)` measures **1.934 / 1.872** across two passes —
  reproducing to 3 % — but its pass-2 rel_sd is 0.033. With that row gone the unconditional gather clause "passed" at worst 1.019.
  `verdict.py` therefore admits a row when either arm's rel_sd ≤ 0.02 in **both** passes **OR** the two passes' ratios agree within
  5 %.
* **`BATCHLAS_SPMM_ROUTE=vendor` is not the control it looks like.** Run over `spmm_tests` it gives 276 passed / **92 FAILED**,
  all `Backend::NETLIB` (the same pinned run on `Backend::CUDA` is 184/184), because 92 is precisely the old unpinned *skip* count
  and a vendor pin converts those refusals into failures. The clause provably does not participate: `BATCHLAS_COVERAGE_OUT` on that
  run shows **144 `spmm` `reached` rows, all `vendor:auto`** — a forced route returns before `preferred()` is ever consulted
  (`VENDOR_FREE_BASELINE.md`, "turns 92 skips into 92 FAILURES").
* Every sweep ran twice in independent processes, one route per process, device 1 pinned by the runner, route read off
  `BATCHLAS_COVERAGE_OUT`. 7,536 timed rows over 9 sweeps (6,512 main + 1,024 small-batch).

| pass pair | gate rows in both | worst ratio spread | median spread | rows crossing the 1.10 line |
|---|---|---|---|---|
| `pass1`/`pass2` | 380 / 380 | 1.114 | 1.0034 | 4 (all `transA=Trans`, both ratios in 1.09-1.12) |
| `bnd1`/`bnd2` gather plane | 120 | 1.094 | 1.0041 | 0 |
| `bnd1`/`bnd2` banded family | 144 | 1.147 | 1.0040 | 0 |
| `bnd1`/`bnd2` scatter planes | 240 | 1.153 | 1.008 | 7 (both ratios in 1.08-1.15) |

## The evidence for each boundary

Acceptance gate throughout: a clause may move a cell only if worst-of-two-passes `t_native/t_vendor <= 1.10` on **every** cell it
moves, at saturation, batch >= 128, on the `(m, nnz/row, nrhs, batch, pattern, beta, transB)` grid, with every boundary bracketed by
measured rows on both sides.

### The gather window

`verdict.txt`, over 644 rows that are saturated, batch >= 128 and chk-agreeing in both passes of their grid (626 quiet, 18 admitted
on cross-pass reproduction alone; 9 dropped as noisy *and* non-reproducing):

| clause | rows moved | verdict | worst-of-two | median | best |
|---|---|---|---|---|---|
| `transA == NoTrans`, unconditional | 186 | **FAILS** 1/186 | 1.934 | 0.446 | 0.032 |
| **shipped**: `... AND NOT (cfloat && transB != NoTrans)` | 176 | PASSES | **0.968** | **0.445** | **0.032** |
| `... AND NOT (cfloat && transB != NoTrans && nrhs >= 16)` | 183 | PASSES | 0.968 | 0.444 | 0.032 |
| `... AND NOT (cfloat && transB != NoTrans && nrhs >= 13)` | 183 | PASSES | 0.968 | 0.444 | 0.032 |

The 468 rows the shipped clause does **not** move contain 170 measured non-winners, so the refusals are bracketed rather than
untested. The two `nrhs`-narrowed variants **pass their own grid and were still rejected**: their boundary rides on the banded
column pattern, which `SpmmShape` cannot see and cannot acquire (it would have to read `col_indices` on the device). A clause whose
true axis the shape cannot express is fitted, not measured. Refusing the family whole costs at most 2 % on the scattered pattern,
and 7 cells against the best *passing* alternative (183 − 176; 10 against the unconditional clause, which fails).

### The DRAM roof

Lanczos shape, batch 4096 and 8192, `transA = NoTrans`, footprints 151-1074 MB — i.e. 2x to 15x the 72 MB L2, so partial
residency cannot inflate the number (`roof.txt`, second block; the **separate** 4 x L2 = 288 MB restriction defines the 220-row
set the LOBPCG-regime sentence below quotes, not this table):

| type | pattern | cuSPARSE | native gather |
|---|---|---|---|
| float | either | 120-147 GB/s (0.12-0.15 x roof) | 910-928 GB/s (0.90-0.92 x) |
| double | either | 185-235 GB/s (0.18-0.23 x) | 906-931 GB/s (0.90-0.92 x) |
| complex\<float\> | either | 185-236 GB/s (0.18-0.23 x) | 918-931 GB/s (0.91-0.92 x) |
| complex\<double\> | banded | 306-366 GB/s (0.30-0.36 x) | 915-926 GB/s (0.91-0.92 x) |
| complex\<double\> | scattered | 308-360 GB/s (0.31-0.36 x) | 714-850 GB/s (0.71-0.84 x) |

The gather is AT the roof and cuSPARSE is **2.3-7.7x** below it on these rows — 6.2-7.7x for `float`, 3.9-5.0x for `double` and
`complex<float>`, and only 2.3-3.0x for `complex<double>`, which is the type cuSPARSE handles best (`roof.txt`'s per-row `ratio`
column; the "3-8x" in the exploration notes rounds the `complex<double>` end the wrong way). It is still the one op in the campaign
where the vendor was found nowhere near its own hardware.

In the LOBPCG regime **neither** arm is near the roof (`roof.txt`'s first block, the 220 rows with footprint > 288 MB;
`transA=0`: vendor 0.06-0.16 x, native 0.09-0.52 x): that is the column-major `op(B)` gather wall, one 32-byte sector per
(nonzero, column) touch for `sizeof(T)` useful bytes, and it binds both arms. The native kernel is still 1.3-4x faster there
(LOBPCG `transA=0` ratios run 0.253-0.804 on the four types, `report_pass12.txt`), and *not* by beating the wall.

### The nsys split

Both arms are > 93 % GPU-kernel time, so the win is a kernel result, not host overhead (`nsys_split.txt`; negative "host" shares are
profiler accounting skew — kernel sums measured under `nsys`, wall times from unprofiled CSVs — and mean the per-call host chain is
below the resolution of the comparison):

| cell (float) | arm | wall/call | GPU kernel/call | host share |
|---|---|---|---|---|
| m=1024 nnz/row=3 nrhs=2 b=512 | native | 0.01196 ms | 0.01124 ms | 6.0 % |
| " | vendor | 0.16239 ms | 0.16461 ms | −1.4 % |
| m=1024 nnz/row=3 nrhs=2 b=4096 | native | 0.20227 ms | 0.20087 ms | 0.7 % |
| " | vendor | 1.25856 ms | 1.26342 ms | −0.4 % |
| m=2048 nnz/row=16 nrhs=25 b=128 | native | 0.33853 ms | 0.34594 ms | −2.2 % |
| " | vendor | 0.86439 ms | 0.91118 ms | −5.4 % |
| " `transA=Trans` | native (scatter + scale) | 2.10142 ms | 2.17493 ms | −3.5 % |
| " `transA=Trans` | vendor | 1.87837 ms | 1.99502 ms | −6.2 % |

cuSPARSE launches **three kernels per `spmm` call** — `csrmm_alg1_kernel`, `csr_partition_kernel`, `matrix_scalar_multiply_kernel` —
and re-partitions the CSR rows on **every** call: `csr_partition_kernel` is 0.0591 ms of 0.1646 ms (36 %) at batch 512, 0.4481 ms of
1.2634 ms at batch 4096. The native gather launches **one** kernel; the transposed arm launches **two** (scale, then scatter —
2.1527 ms + 0.0222 ms on the last row above), and that row is also the refusal measured independently under the profiler: 1.12x
the vendor on the same cell.

### Saturation and the L2 cliff

Saturation here means "a wider batch buys < 10 % per item", not "within 10 % of the fastest batch in the ladder". The naive rule
breaks on exactly these cells: the gather at `(float, m=1024, 16 nnz/row, nrhs=12)` runs 2.369, 0.788, 0.309, **0.485** µs per item
over batch 8/32/128/512 — it *rises* at 512 because the 119 MB footprint leaves the 72 MB L2. Per-item µs, vendor/native, float,
m=1024, 3 nnz/row, nrhs=1, banded (`sat1`/`sat2`, agreeing to three decimals):

| batch | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| footprint | 37.8 MB (L2) | 75.5 MB | 151 MB | 302 MB |
| vendor | 0.308 | 0.304 | 0.306 | 0.306 |
| native | 0.010 | 0.020 | 0.040 | 0.040 |
| ratio | **0.032** | 0.066 | **0.131** | **0.130** |

**cuSPARSE is flat from batch 256; the native gather is not saturated until batch ~4096**, because it is fast enough to stay
latency-bound for longer. The spectacular batch-512 lanczos ratios (0.032-0.072 for float) are L2-resident numbers and **must not be
quoted as the algorithmic result**. The saturated, DRAM-resident figure is **0.13-0.43**, i.e. 2.3x-7.7x.

### The cfloat transB exclusion

The one measured non-winner on the gather arm. `complex<float>`, `transB = Trans`, **banded** column pattern, m=2048, nnz/row=16,
batch=512 (`cfedge1`, `cfedge2`):

| nrhs | 8 | 9 | 12 | 16 | 17 | 20 | 25 | 32 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| cfloat banded p1 | 0.630 | 0.713 | 0.689 | 1.087 | 1.315 | 1.218 | **1.731** | 1.159 | 1.695 |
| cfloat banded p2 | 0.663 | 0.737 | 0.761 | 1.040 | 1.274 | 1.162 | **1.714** | 1.157 | 1.703 |
| cfloat scattered p1 | 0.670 | 0.856 | 0.888 | 0.963 | 0.979 | 0.793 | 1.000 | 1.019 | 0.953 |
| cfloat scattered p2 | 0.671 | 0.927 | 0.850 | 0.909 | 0.938 | 0.792 | 0.993 | 1.016 | 0.941 |
| float banded p1 (control) | 0.406 | 0.404 | 0.404 | 0.545 | 0.360 | 0.368 | 0.385 | 0.934 | 0.571 |

Bracketed on three axes. **nrhs**: 12 wins at 0.689-0.761, 17 loses at 1.274-1.315. **Pattern**: on the scattered pattern — the one
a filtered eigensolve actually has — the same family is 0.79-1.02, parity; the loss is banded-exclusive, which is why the exclusion
is stated by type and `transB` rather than by `nrhs`. **Type and batch** (`sb1`/`sb2`, banded, m=2048, 16 nnz/row, nrhs=25, worst of
two passes):

| type | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 |
|---|---|---|---|---|---|---|---|---|
| float | 0.655 | 0.569 | 0.485 | 0.402 | 0.409 | 0.403 | 0.406 | 0.471 |
| double | 0.341 | 0.275 | 0.221 | 0.503 | 0.634 | 0.625 | 0.621 | 0.613 |
| **complex\<float\>** | 0.520 | 0.524 | 0.581 | **1.447** | **2.072** | **2.182** | **2.092** | **1.944** |
| complex\<double\> | 0.335 | 0.287 | 0.432 | 0.688 | 0.676 | 0.681 | 0.681 | 0.669 |

The loss switches on between batch 4 and 8 and is **worst in the middle of the ladder** (2.182 at b=32 against 1.714-1.731 at
b=512), so no batch floor fixes it; only `complex<float>` crosses, so the type conditional is exactly as narrow as the data. **The
cost is recorded rather than hidden**: at batch 1-4 the excluded family is a 1.7-1.9x native win the clause now declines, and buying
it back would take a `batch <= 4` sub-clause on a type-conditional, on a shape with no in-tree caller, one ladder rung wide.
**Mechanism, hypothesis only, not confirmed by profile**: `kNCmax<Cx<float>>` is 8 (`spmm_native.cc:49-52`), so `nrhs=25` needs
`ceil(25/8) = 4` passes over A with 7 idle accumulator lanes while `nrhs=32` needs 4 with none — and `nrhs=32` measures 1.157-1.159
against `nrhs=25`'s 1.714-1.731.

This family's batch-128 cell is deliberately shared with the main grid's `cellsC0, tB=1, beta=0, pat=0` — four independent processes
across two sweeps ~90 min apart, the harness's own self-check: float 0.458 / 0.471 / 0.460 / 0.453; double 0.613 / 0.609 / 0.612 /
0.599; cfloat 1.826 / 1.944 / 1.934 / 1.872; cdouble 0.663 / 0.669 / 0.662 / 0.666.

### The batch axis has no floor

`preferred()` is consulted on every call while the acceptance gate is stated at batch >= 128. That mismatch was a real outstanding
caveat, closed by a separate sweep (`run_smallbatch.sh`, `sb1`, `sb2`, `smallbatch.txt`): 5 shape families x 4 types x 2 patterns x
2 betas x both `transB` x batch {1,2,4,8,16,32,64,128}, twice, one route per process — 1,024 timed rows, 256 cells present and
chk-agreeing in both passes, 210 admitted. Under the shipped clause, worst of two passes:

| batch | rows | worst | median | best | over the 1.10 gate | max Δ µs/call (native − vendor) |
|---|---|---|---|---|---|---|
| 1 | 22 | 0.992 | 0.592 | 0.186 | 0 | −0.23 |
| 2 | 27 | 0.981 | 0.287 | 0.170 | 0 | −0.44 |
| 4 | 21 | **1.078** | 0.473 | 0.173 | 0 | **+13.46** |
| 8 | 25 | 0.956 | 0.460 | 0.154 | 0 | −3.64 |
| 16 | 25 | 0.966 | 0.499 | 0.154 | 0 | −11.04 |
| 32 | 29 | 0.978 | 0.373 | 0.134 | 0 | −16.17 |
| 64 | 25 | 0.967 | 0.412 | 0.099 | 0 | −23.51 |
| 128 | 30 | 0.964 | 0.305 | 0.066 | 0 | −36.00 |

Over `batch <= 64` (174 admitted rows): worst 1.078, median 0.440, best 0.099, and **1 of 174 rows costs the caller any time at
all** — `complex<float>`, m=4096, 16 nnz/row, nrhs=50, scattered, `transB=NoTrans`, batch 4, at **1.078 / 1.078** (rel_sd 0.011 /
0.009; vendor 173.4 µs, native 186.8 µs, +13.46 µs/call). It is non-monotonic on its own cell (b=2 0.977, b=8 0.956), i.e. an
unsaturated launch/occupancy artefact rather than a structural loss. The median admitted row saves the caller 18.1 µs per call.

**That row is the bracketing evidence for having no floor**, because it sits *inside* the gate. A floor must be justified by a
measured non-winner on the wrong side of it, and this grid contains none at any rung for any type the clause admits.
`RouteSpmm.PreferredHasNoBatchFloor` in `tests/route_vocabulary_tests.cc` pins the absence so that adding `s.batch >= N` goes red
with a message saying where to look.

**Honesty label**: below batch ~64 the timed region is launch latency plus the vendor's unhoistable per-call host chain — on the
cheapest family (`sbL`, lanczos) the native per-call floor is ~2.9-3.4 µs against the vendor's 13.3-16.6 µs. That ~2.9 µs is a
*launch* floor, not a per-call time for every shape: at batch 1 the native arm already costs 11.6-17.5 µs on `sbM`, 15.9-22.7 on
`sbS` and 39.1-82.1 on `sbB` (`smallbatch.txt`), which is work, not overhead. What is shape-independent is the *vendor's* extra
~13 µs, and that is what the small-batch ratios are measuring. These ratios are evidence of **no harm** and nothing else; the
0.968 / 0.445 headline comes from the saturated grid alone. Filter check repeated here: 46 of 256
cells were not admitted and every dropped row was inspected, not counted — exactly two have either pass over the gate, both `cfloat`
+ `transB=Trans` (batch 16: 1.958/2.072; batch 128: 1.826/1.944), the family already excluded; every other dropped row is a native
win, largest 0.955.

### The transposed refusal

Measured on the same grid at the same saturation. `transA != NoTrans` moves 458 saturated cells with **169 over the 1.10 gate**,
median 1.030, worst **3.011**.

| clause | rows moved | verdict | worst | refuting cell |
|---|---|---|---|---|
| `transA != NoTrans` | 458 | FAILS 169/458 | 3.011 | cdouble m=4096 nnz/row=16 nrhs=50 b=512 tB=0 **scattered**, p1=3.000 p2=3.011 |
| `... AND nrhs <= 4` | 204 | FAILS 11/204 | 1.208 | cdouble m=2048 nnz/row=16 nrhs=2 b=512, p1=1.208 p2=1.207 |
| `... AND nrhs <= 2` | 151 | FAILS 5/151 | 1.208 | same cell |
| `... AND nrhs <= 1` | 60 | FAILS 2/60 | 1.132 | cdouble m=2048 nnz/row=16 nrhs=1 b=1024, p1=1.130 p2=1.132 |
| `... AND nrhs <= 2 AND type != cdouble` | 111 | PASSES | 1.023 | **rejected anyway** — see below |
| `... AND nrhs <= 4 AND type != cdouble` | 150 | FAILS 1/150 | 1.101 | double m=2048 nnz/row=16 nrhs=4 b=512, p1=1.101 p2=1.101 |

Every refuting cell in that table is on the **scattered** pattern (`pat=1`), the worst one included. `route_spmm.hh:69-73` and
`VENDOR_INDEPENDENCE_PLAN.md`'s negative-results list both label the 3.011 cell "banded"; that is a mislabel — `verdict.txt` gives
it `pat=1`, its row in `pass{1,2}/joined.csv` carries the tag `lobpcg_ta1`, and that sweep passes pattern 1 explicitly
(`run_all.sh:34`, `LOBPCG_ARGS="... 0 0 1"`; `spmm_benchmark.cc:31-143` fixes `kBanded = 0` / `kRandom = 1`, and `:99` says so in
words). The gather arm's refuting cell *is* banded (`pat=0`), which is presumably where the label came from.

The scatter's `nrhs` boundary moves with the type **and** with `nnz/row`, and it is the `nnz/row` axis that the shape cannot
express (`bnd_scatter_a`, m=1024, batch=512, scattered, pass 1):

| type \ nrhs | 1 | 2 | 4 | 8 | 12 | 25 |
|---|---|---|---|---|---|---|
| float (3 nnz/row) | 0.180 | 0.323 | 0.590 | 0.779 | 0.880 | 1.095 |
| double (3 nnz/row) | 0.194 | 0.350 | 0.636 | 0.996 | 1.076 | 1.103 |
| cfloat (3 nnz/row) | 0.344 | 0.632 | 0.820 | 0.989 | 1.137 | 1.107 |
| cdouble (3 nnz/row) | 0.390 | 0.683 | 0.968 | 1.118 | 1.105 | 1.289 |
| **cdouble (16 nnz/row)** | **1.043** | 1.078 | 1.112 | 1.164 | 1.206 | 1.307 |

`complex<double>` at 16 nnz/row is at or over the gate at `nrhs = 1` **once `m` grows**: at m=2048 it is 1.098/1.101 at b=512 and
1.130/1.132 at b=1024 (`scl1`/`scl2`, both passes), against 0.390 for the same type at 3 nnz/row and the same width. At the
table's own m=1024 the 16-nnz/row cell is 1.043 — inside the gate, and still 2.7x the 3-nnz/row cell beside it, which is the point:
the axis that separates them is `nnz/row`, not `nrhs`. `SpmmShape` carries no `nnz` field and cannot acquire one, so **no predicate
expressible in the routing shape separates the cdouble win from the cdouble loss**. The one clause that passes needs an explicit
type exclusion, moves 111 cells of a decomposition with **zero in-tree C++ callers today**, and is fitted to an invisible axis.

The scatter stays **supported** — `BATCHLAS_SPMM_ROUTE=native` still reaches it and the vendor-free build still routes it.
Un-preferred is not unsupported.

## Negative results

1. **No shippable window exists for the transposed scatter.** 169 of 458 saturated cells over the gate, worst 3.011; every
   `nrhs`-only clause fails and the only passing one is fitted to `nnz/row`, which the shape cannot see. A refusal with 458 measured
   cells behind it, not an omission.
2. **The design phase's "parity at best in the LOBPCG regime" prediction was refuted — but the mechanism it gave was correct.** The
   gather beats cuSPARSE 1.3-4x at LOBPCG shapes and *not* by beating the column-major `op(B)` gather wall: neither arm exceeds 0.52
   x roof there. The margin comes from cuSPARSE paying that wall **and** re-partitioning the CSR every call. A correct mechanism can
   predict a wrong outcome.
3. **The spectacular lanczos ratios are an L2 artefact and shrink 4x at true saturation** (0.032 at batch 1024 → 0.131 at batch
   4096). Any headline off the batch-512 grid alone would have been 4x too flattering.
4. **An rel_sd-only hygiene filter manufactured a passing unconditional clause** by deleting the one reproducible non-winner (1.934
   / 1.872, pass-2 rel_sd 0.033).
5. **`complex<double>` transposed at 16 nnz/row loses at `nrhs = 1`** (m=2048: 1.098/1.101 at b=512, 1.130/1.132 at b=1024) — the
   narrowest possible transposed shape, which is where a scatter should have been strongest.
6. **The `cfloat` + `transB=Trans` loss is not a saturation effect**, the opposite of what batch-512-only evidence implied: 0.581 at
   b=4, 1.447 at b=8, 2.182 at b=32, 1.944 at b=128, 1.71-1.73 at b=512. Had it been a large-batch artefact, a batch floor would
   have been the cheap fix.
7. **The `cfloat` gather margin is ~3 %, not the 2x the median suggests** — 0.910 to 1.078 across the whole batch ladder on the
   largest LOBPCG family, against a saturated worst-of-two of 0.968.
8. **Reversing `kSpmmOrder` does not send admitted shapes back to cuSPARSE.** Written into the header as a claim, then applied,
   rebuilt and run: exactly one case goes red (`RouteSpmm.OrderIsExactlyTwoEntries`), structurally rather than through any decision,
   because `preferred()` is false for `{Vendor, Auto}` and the first walk skips that entry wherever it sits. The mistake this array
   *can* make — a `preferred()` true for the vendor entry — is pinned by `RouteSpmm.PreferredIsFalseForEveryOtherRouteAndFormat`.
9. **The suite-count burn-down is the wrong instrument for this op.** Vendor-free `ctest -LE slow` went 34/56 → 35/57 and the
   joining suite is `spmm_tests` itself; the 22 failing names are byte-identical to the post-WP7 set. The metric that moved is the
   per-op `NoRouteError` census: `spmm` **2 → 0**, every other op unchanged digit for digit. Vendor-present, `spmm_tests` is 282
   passed / 86 skipped / 0 failed unpinned (was 276 / 92 / 0 before the clause) and 368/368 pinned native. Those **six** recovered
   cases are the type conditional showing up as a test-count delta: `transA = NoTrans` with `transB = Trans`/`ConjTrans` on
   `Backend::NETLIB` now takes the native gather instead of netlib's blanket transpose refusal — for `float`, `double` and
   `complex<double>` only, while `complex<float>` keeps all 23 of its skips, because the clause refuses exactly that family.
   And the flip was checked for collateral rather than assumed: `ctest -LE slow` is 56/57 vendor-present, the one failure
   `lanczos_tests` — which *does* consume the moved gather — was re-run under `BATCHLAS_SPMM_ROUTE=vendor` and produced the same
   two failing cases (`LanczosTestBase.LanczosTest`, `LanczosTestBase.ToeplitzEigenpairs`), so it is pre-existing and not WP8's.

## Correctness findings

### Three vendor defects, found here and fixed

None was introduced by WP8; all three were shipping wrong answers, or a dead process, before it. They surfaced because
`tests/spmm_tests.cc` (368 cases, four types, all nine `(transA, transB)` spellings, heterogeneous `nnz`, empty rows, padded
strides, alpha/beta corners) is the first suite to cover their axes. Prior coverage was two calls in `lanczos_tests.cc`, both
`NoTrans/NoTrans`, square, at natural strides, checked by an *eigenvalue* not the product.

1. **`netlib_lapack.cc:248,272` — `spmm` read `A` at `alpha == 0` and `C` at `beta == 0`.** Callers hand `spmm` a `BumpAllocator`
   allocation that is not zeroed, and `0 * NaN` is `NaN`, so an unread operand poisons the result instead of dropping out of it. The
   host arm now skips the alpha term and substitutes `T(0)` for the beta term — the guarantee the native bodies already made.
2. **`cusparse.cc` mapped `ConjTrans` to `CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE` for REAL scalars.** On a real scalar `ConjTrans`
   *is* `Trans`, and `CUDA_R_32F`/`CUDA_R_64F` silently produced wrong results under the conjugating enum. It applied to **both**
   operands (`transA = NoTrans, transB = ConjTrans` was wrong on its own), and survived because the complex arms — where that enum
   is the distinct and correct operation — always passed.
3. **The heterogeneous-`nnz` / padding over-read.** `cusparseCreateCsr` takes one `nnz` and `cusparseCsrSetStridedBatch` adds no
   per-item one, while `backend_handle_impl.hh:63` handed it `A.nnz()` — the per-item **capacity**, sized by the batch maximum, in a
   conversion that zeroes only `row_offsets`. Every short item's descriptor claimed values and column indices the conversion never
   wrote, and cuSPARSE read them: wrong last rows on `HeterogeneousNnzAcrossBatch`, and `CUDA_ERROR_ILLEGAL_ADDRESS` with a dead
   process on the padding case. The fix takes `nnz` from the items' own row offsets for a uniform batch and issues **one
   `cusparseSpMM` per item** for a non-uniform one, the only shape the API offers. A homogeneous batch keeps the single batched call
   — which is why every in-tree caller was unharmed and why this survived until a suite covered the axis.

### The eleventh blind guard

The four deliberate breaks are logged at `tests/spmm_tests.cc:1356-1510` **in the archived
revision** — the comment pass removed the log block from the tree, so read it with
`git show perf-evidence/vendor-independence:tests/spmm_tests.cc`. The tests it names are still
in-tree (`PaddingAboveNnzIsNotReadTrans` and its controls). **B4 (`scatterBound`) came back GREEN over all 352 cases
the first time it was run** — the transposed `nnz` bound was completely unguarded, and a kernel reading uninitialised padding on
every transposed call would have shipped.

What made the guard blind: `PaddingAboveNnzIsNotReadTrans` poisoned the padding with a NaN value at an **out-of-range** column index
(2^30), and the scatter's own range guard (`spmm_native.cc:367-375`) `continue`s **before** `av` is multiplied — so both halves of
the poison went in the bin together and the test was green because of a kernel guard, not the property it named. Two control runs
established that rather than argued it: broken bound + range guard deleted → the case **segfaults** (exit 139, on the `double`
instantiation), so the over-read is real and lethal; correct bound + range guard deleted → 352/352 green, so deleting the guard is
harmless alone and the crash is attributable to the bound. The `float` instantiation survived even the first of those — a second
reason not to trust an out-of-range index as a detector: where it lands is not the fixture's to decide.

**The fix was to the test, not the kernel.** The range guard is correct and stays: in the gather a bad column index is an
out-of-range *read*, in the scatter an out-of-range *atomic write*, i.e. heap corruption. `PaddingAboveNnzIsNotReadTrans` now
poisons with an **in-range** column and a large finite sentinel — an entry the scatter accepts and accumulates — so the over-read
lands in C where the backward-error comparison names the `(batch, col, row)` it hit. Finite rather than NaN because NaN is absorbing
under atomic addition (one spurious entry is indistinguishable from a whole slab), says nothing about *where* it landed, and is the
one assertion a fast-math device build may fold away. The out-of-range configuration is kept under an honest name,
`PaddingAboveNnzOutOfRangeIsNotReadTrans`, as a vendor-fault and missing-guard detector that is **not** coverage of the bound —
though it is the range guard's only regression test. `HeterogeneousNnzAcrossBatchTrans` was added at the same time: the *unpoisoned*
form of the transposed over-read had no twin at all. Four rules this produced:

* Every contract here has two independent implementations, and a poison tuned to one body's failure mode can be inert against the
  other's. Before trusting a `*Trans` twin, ask not "does it exist" but "what does the kernel do with the poison": a defensive
  predicate between poison and assertion makes the case vacuous.
* A coverage row cannot substitute for a break. Rows are keyed on a power-of-two `shape_class` and are first-writer-wins, so a row
  proves *some* shape resolved to a route, never that *this* shape ran *that* body.
* A break must be as narrow as the contract it denies. B2 (`gatherBound`) written the obvious way (`re = a_nnz_cap` for every row)
  falsifies its own named controls — a homogeneous batch goes red too — so "stays green" identifies nothing and the red set no
  longer isolates the padding.
* **`git diff` cannot verify the revert of an untracked file.** `src/sycl/spmm_native.cc` is new in this package, so after a
  deliberate break `git diff` reports *nothing* — an assertion that cannot fail, the same defect class as the blind guard it was
  being used to police. The recipe recorded in the file is an `md5sum` of the pristine source taken **before** the first break and
  compared after the last.

Two further deliberate properties: the tolerance denominator is a backward-error scale (sum of `|a|*|b|` over the contributions to
that element, floored at 1), never `|expected|`, because the transposed path is an atomic scatter and is **not** bitwise
reproducible run to run; and a transposed case whose backend *throws* is SKIPPED, not failed — disabled when the pin names
**native** (`tests/spmm_tests.cc:339-343`: `pin_text` containing `native`, or equal to `direct`/`cta`/`blocked`), so pinned-route
break runs cannot be silently skipped past. Note the narrowness, which is itself a finding: the file's own header comment at `:92`
still says "disabled when `BATCHLAS_SPMM_ROUTE` is set", and keying it that way is what **turned 92 pre-existing `Backend::NETLIB`
skips into 92 failures** under `BATCHLAS_SPMM_ROUTE=vendor` — netlib hard-throws on any transpose, so a vendor pin must leave the
skip armed. A wrong answer is never skipped, and a vendor returning a *status code* instead of throwing is not covered by the skip
at all: `cusparse.cc` checks no cuSPARSE status, so
`CUSPARSE_STATUS_NOT_SUPPORTED` leaves C untouched and the suite calls it a wrong answer — correctly.

### Two defects filed, not fixed

* **`rocsparse.cc:30-31` and `:62-63` carry cuSPARSE defect 2 unchanged** — `transA` and `transB` go through
  `enum_convert<BackendLibrary::ROCSPARSE>` unconditionally, so a real-scalar `ConjTrans` becomes
  `rocsparse_operation_conjugate_transpose`. **Unmeasured: there is no AMD device on this machine.** That rocSPARSE mishandles it
  the way cuSPARSE did is inferred from the cuSPARSE finding, not observed.
* **`netlib_lapack.cc:508,520,537,549` — `trsm` reads `B` at `alpha == 0`** (`T x = alpha * Bb.at(i, j, 0) - sum;`). Same `0 * NaN`
  family, different op, out of WP8's scope.

## Open debts

* **The transposed scatter is vendor-first and has no route** — 169 of 458 saturated cells lose, no shape-expressible clause
  recovers a window, and it has **zero in-tree C++ callers**, so nothing exercises it in anger.
* **The `cfloat` gather margin is ~3 %, not 2x.** The clause admits the tightest cell in the campaign (1.078 at batch 4, 0.910-0.978
  elsewhere on its ladder). A kernel change costing 5 % on `complex<float>` turns admitted cells into losses with no test noticing.
* **The `nrhs >= 16` narrowing is left on the table.** It passes `verdict.txt` at worst 0.968 and would move 183 cells instead of
  176; refused only because its axis is the column pattern. If `SpmmShape` ever gains an honest pattern or `nnz` signal, both this
  and the scatter's `nrhs <= 2 && !cdouble` clause are re-arguable.
* **The `kNCmax` register-block mechanism for the `cfloat` loss is a hypothesis, not a profile.** The non-monotonicity (nrhs=32 at
  1.157-1.159 vs nrhs=25 at 1.714-1.731) is consistent with it and nothing more.
* **Three in-tree comments are stale or wrong, and all three are the kind a reader trusts.** (a)
  `src/dispatch/entry_points/sparse.cc:59-110`'s route-neutrality comment states that `preferred()` "is false for every route, every
  type and every shape (route_spmm.hh:65)" and derives byte-identical vendor routing from that; the shipped `preferred()` is not
  all-false, the code is still correct, but the justification no longer holds and the line reference now points into a block that
  says the opposite. (b) is **closed**: the comment saying the transposed refusal-skip was "DISABLED when `BATCHLAS_SPMM_ROUTE` is set" no longer
  exists — the comment pass removed it, and the skip at `tests/spmm_tests.cc:339-353` tests for a **native** pin, which is the
  correct form. (The loose wording is what turned 92 NETLIB skips into 92 failures; read it at
  `git show perf-evidence/vendor-independence:tests/spmm_tests.cc` line 27.)
  (c) `route_spmm.hh:69-73` (and the plan's negative-results list) call the worst transposed cell "banded"; `verdict.txt` and the
  sweep's own arguments make it `pat=1`, scattered.
* **Coverage blindness, known and accepted**: `variant_key` packs only `uplo/side/diag/transA/transB` and `shape_class` buckets
  `max(m,n,k)` and batch by power of two, so a CSR and a Dense `spmm` at the same extents would collapse into one first-writer-wins
  row. Unobservable today because only CSR is instantiated; do **not** add a format bit to fix it — renumbering invalidates every
  stored `.routes` baseline. Coverage likewise cannot distinguish scale from scatter; only a break red for that body alone can.
* **The scatter's FP64 instantiations require the `atomic64` aspect**, which the FP32 ones do not. Both development devices have it;
  on a device without it the failure is a kernel-selection error at launch, not a compile error. Untested on such a device.
* **The whole grid is one RTX 4090** — no second GPU generation, no AMD device, no CPU-queue timing, so "no `is_gpu` gate" is a
  correctness and burn-down decision, not a measured claim about `native_cpu`. The two 4090s here share a NUMA node, a CPU affinity
  mask and one UVM driver while `nvidia-smi --query-compute-apps` is per device, so a sweep on the other card is invisible to
  `run_spmm.sh`'s guard and has been measured inflating a cell 5.5x at a stable rel_sd. Re-runs must verify both devices.

## Raw evidence

Raw data is preserved at the git tag `perf-evidence/vendor-independence`, retrievable with `git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| the distilled write-up all of this is drawn from | `experiments/sparse_spmm/README.md` |
| the clause table: every candidate, its verdict and its refuting cell | `experiments/sparse_spmm/verdict.txt` (built by `verdict.py` from `pass{1,2}/joined.csv` and `scl{1,2}/joined.csv`) |
| the main grid, two independent passes | `experiments/sparse_spmm/pass1/`, `pass2/`, summarised in `report_pass12.txt` |
| DRAM-roof rows, footprint > 288 MB | `experiments/sparse_spmm/roof.txt`, `roof.py` |
| kernel/host decomposition, both arms | `experiments/sparse_spmm/nsys_split.txt`, `nsys_split.sh`, `nsys.log` |
| saturation extension, batch 1024-8192 | `experiments/sparse_spmm/sat1/`, `sat2/`, `run_satext.sh`, ladders in `tables.txt` |
| the `cfloat` nrhs walk that places the gather-arm loss | `experiments/sparse_spmm/cfedge1/`, `cfedge2/`, `run_cfloat_edge.sh` |
| the (nnz/row x nrhs) boundary planes, the banded family, and the scatter arm's nrhs boundary with a batch ladder under every cell | `experiments/sparse_spmm/bnd1/`, `bnd2/`, `run_boundary.sh`, `scl1/`, `scl2/`, `run_scatter_ladder.sh` |
| the small-batch corner, batch 1-128 | `experiments/sparse_spmm/sb1/`, `sb2/`, `run_smallbatch.sh`, `smallbatch.csv`, `smallbatch.txt`, `sb_report.py` |
| warm-up ramp, ordering and route-pin probes | `experiments/sparse_spmm/probe/warmup_probe.sh`, `probe/order_probe.sh` |
| the route census behind "65 moved decisions" | `experiments/sparse_spmm/route_census.py`, with `scripts/route_diff.sh` captures `wp8-before` / `wp8-after` |
| campaign-level WP8 summary and the vendor-defect list | `VENDOR_INDEPENDENCE_PLAN.md`, sections "WP8 has landed" and "WP8 — sparse: `spmm`" |
| coverage-instrument limits (first-writer-wins rows) | `VENDOR_FREE_BASELINE.md` |

Reproduction, one sweep at a time, device 1 exclusive (`run_all.sh` ~21 min per pass, `run_smallbatch.sh` ~25 min):

```bash
cmake --build build --target spmm_benchmark -j16
cd experiments/sparse_spmm
./run_all.sh pass1; ./run_all.sh pass2   # then run_boundary / run_cfloat_edge / run_satext /
./run_nsys.sh                            # run_scatter_ladder / run_smallbatch, each twice
python3 verdict.py pass{1,2}/joined.csv scl{1,2}/joined.csv
```

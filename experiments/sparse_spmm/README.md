# WP8 — native batched CSR SpMM vs cuSPARSE: the first `spmm` baseline in this tree

## The one-paragraph result

**The native gather (`transA == NoTrans`) beats cuSPARSE on every one of 186
saturated, batch ≥ 128 cells measured twice, except one family.** Median
worst-of-two-passes ratio **0.446**, best **0.032**. At unambiguous DRAM
residency the native gather runs at **90–92 % of the 1008 GB/s roof for all four
scalar types**, while cuSPARSE runs at **12 % (float), 18–23 % (double,
complex\<float\>) and 30–36 % (complex\<double\>)** — so this is **not** a case of
both arms sitting at the roof with host overhead between them, and `nsys`
confirms both arms are > 93 % GPU-kernel time. **The transposed arm
(`transA == Trans`) is the opposite story**: it wins the narrow-`nrhs` corner and
**loses the whole LOBPCG regime**, median ratio 1.08–1.24 with a worst of 3.01.
One `preferred()` clause is recommended, **with no batch term**: the batch 1-64
corner was swept separately and contains **zero** rows over the gate under that
clause. The transposed arm stays vendor-first.

---

## Why this directory exists

Before this pass there was **no `spmm` measurement anywhere in the repository** —
no row under `benchmarks/results/`, and
`include/batchlas/blas/dispatch/route_spmm.hh` says so in the comment above its
all-false `preferred()`. The deliberate-break exercise that armed WP8's other
axes found this one **vacuous**: nothing in the tree could tell whether the
native kernels were fast, slow, or a wash. This directory is that measurement:
**7 536 timed rows over 9 sweeps, each run as 6 independent pass-pairs plus an `nsys` decomposition.**

---

## What is inside the timed region — and what is not

`benchmarks/spmm_benchmark.cc` times **K back-to-back `spmm` calls on one
in-order queue, closed by a single `wait`, on the host clock**. The queue is
in-order (`sycl-device-queue.hh:254`, `in_order = true` by default), so the
native kernels serialise exactly as cuSPARSE's calls on one CUDA stream do —
neither arm gets free inter-call overlap.

INSIDE, for both arms alike: the `spmm` facade, route resolution, and the
kernel/library launch. For the vendor arm that means **everything
`backend::spmm_vendor` does per call**: `handle.setStream`, the
`SpmmCsrBatchPlan` walk over every batch item's row offsets on the host (added
by this session's heterogeneous-`nnz` fix), the **`cusparseSpMM_bufferSize`
re-query on every call**, the `BumpAllocator` carve, and the `cusparseSpMM`
launch.

OUTSIDE, for both arms alike: matrix generation, USM prefetch, the one-time
`spmm_buffer_size` query that sizes the workspace, and the warm-up.

**The fairness decision, stated explicitly.** The vendor's per-call re-query is
*not* hoisted out of the timed region, because **no caller can hoist it** — it
lives inside `spmm_vendor`, runs on every `spmm` call, and `lanczos` pays it `n`
times per solve. But an end-to-end ratio that is really a host-overhead ratio
must never be sold as a kernel win, so `nsys_split.sh` decomposes both arms.
**The decomposition says host overhead is not the story** (`nsys_split.txt`):

| cell (float) | arm | wall/call | GPU kernel/call | host share |
|---|---|---|---|---|
| m=1024 nnz/row=3 nrhs=2 b=512 | native | 0.01196 ms | 0.01124 ms | 6.0 % |
| " | vendor | 0.16239 ms | 0.16461 ms | ≈ 0 (−1.4 %) |
| m=1024 nnz/row=3 nrhs=2 b=4096 | native | 0.20227 ms | 0.20087 ms | 0.7 % |
| " | vendor | 1.25856 ms | 1.26342 ms | ≈ 0 (−0.4 %) |
| m=2048 nnz/row=16 nrhs=25 b=128 | native | 0.33853 ms | 0.34594 ms | ≈ 0 (−2.2 %) |
| " | vendor | 0.86439 ms | 0.91118 ms | ≈ 0 (−5.4 %) |

(Negative "host" shares are the profiler's own accounting skew — the kernel sum
is measured under `nsys`, the wall time from an unprofiled CSV. They mean the
per-call host chain is below the resolution of the comparison, not that it is
negative.)

What the profile *does* show is where cuSPARSE's time goes: **three kernels per
`spmm` call** — `csrmm_alg1_kernel`, `csr_partition_kernel` and
`matrix_scalar_multiply_kernel` — and the partition kernel is **36 % of the
vendor's GPU time** at the lanczos cell (0.0591 ms of 0.1646 ms at batch 512;
0.4481 ms of 1.2634 ms at batch 4096). cuSPARSE re-partitions the CSR rows on
**every** call. The native arm launches **one** kernel.

`MatrixView`s are built **once** and reused, so the cuSPARSE descriptors are
created on the first (untimed) call. This is the **warm** vendor arm, the only
one a kernel-vs-kernel ratio may be quoted against.

---

## Measurement hygiene actually performed

| rule | what was done | evidence |
|---|---|---|
| one measuring harness on the box | every sweep runs **sequentially** from one script; `run_spmm.sh` refuses to start if the target GPU already holds > 200 MiB or if another `spmm_benchmark` is live | `run_spmm.sh` guard |
| device 1, never device 0 | `CUDA_VISIBLE_DEVICES=1` is exported by the runner, not left to the caller. Device 0 held ~1.0 GB of Xorg/gnome-shell throughout | `*/*.log` (clock + temp before and after every run) |
| cold clocks | minibench's warm-up is counted in CALLS and cannot price a ramp measured in ms. A **wall-clock warm-up budget** (default 400 ms/row, `BATCHLAS_SPMM_WARM_MS`) was added to `spmm_benchmark.cc`'s `SetPrepare` | see below |
| rel_sd | reported on every row; noisy rows are dropped, with one deliberate exception (see "the filter that manufactured a win") | `analyse.py`, `verdict.py` |
| route pins verified, never assumed | every run writes `BATCHLAS_COVERAGE_OUT` beside its CSV and the runner prints the resolved `origin:algo` | `*/*.routes.csv` |
| a typo must not pass silently | measured directly | see below |
| saturation | measured off the batch ladder, never assumed | `report.py::mark_saturation` |
| L2 residency | whole-batch footprint computed per row and compared against the 4090's 72 MB L2 | `analyse.py`, `roof.py` |

### The clock ramp is a per-PROCESS cost (`probe/warmup_probe.sh`, `probe/order_probe.sh`)

Cell L, vendor, device 1:

| minibench warm-up | first row of the process | second row of the same process |
|---|---|---|
| 2 calls | 0.16544 ms, rel_sd 0.0495 | 0.16196 ms, rel_sd 0.0019 |
| 50 calls | 0.16303 ms, rel_sd 0.0278 | 0.16169 ms, rel_sd 0.0016 |
| 250 calls | 0.16179 ms, rel_sd 0.0016 | 0.16187 ms, rel_sd 0.0095 |
| 2000 calls | 0.16175 ms, rel_sd 0.0009 | — |

The 210 MHz idle clock costs **2.3 %** and lands entirely on whichever row runs
first. A CALL-counted warm-up cannot price it uniformly: 250 calls is 40 ms of
ramp on a cheap cell and **13.5 s of dead time** on the 54 ms
`cdouble, m=4096, nrhs=50, batch=512` cell. Hence the wall-clock budget. With it,
the first row of a fresh process reads 0.161916 ms at rel_sd 0.0018 — the
converged value.

### The route pin, proved rather than asserted (`probe/`)

Three processes, same cell (float, m=1024, nnz/row=3, nrhs=2, batch=512):

| `BATCHLAS_SPMM_ROUTE` | avg_ms | rel_sd | coverage table says | chk |
|---|---|---|---|---|
| `vendor` | 0.162654 | 0.0011 | `vendor:auto`, 1263 calls | 2211 |
| `native:direct` | 0.011745 | 0.0020 | `native:direct`, 10363 calls | 2211 |
| `bogus_typo` | 0.162707 | 0.0025 | **`vendor:auto`**, 1263 calls | 2211 |

The two real pins differ by 13.8×, and the deliberate typo is
indistinguishable from `vendor` — the documented silent fallback, reproduced
here so that no ratio in this directory rests on an environment variable alone.
Identical `chk` across arms is the cross-arm answer check.

### The filter that manufactured a win

`verdict.py` admits a row when either arm's rel_sd ≤ 0.02 **in both passes**, OR
the two passes' ratios agree to within 5 %. The second clause is not laxity — an
rel_sd-only filter **silently deleted the single most important negative result
in this sweep**: `(cfloat, transA=NoTrans, transB=Trans, banded, m=2048,
nnz/row=16, nrhs=25, batch=128)` measures **1.934** in pass 1 and **1.872** in
pass 2, reproducing to 3 %, but its pass-2 rel_sd is 0.033. With that row gone,
the unconditional gather clause "passed" with a worst ratio of 1.019. Cross-pass
reproduction is stronger evidence about a row than its within-pass spread, and a
hygiene filter that can delete a reproducible non-winner is a filter that
manufactures wins.

---

## Saturation, and the L2 cliff that changes the headline

**Saturation is defined as "a wider batch buys < 10 % per item"**, not "within
10 % of the fastest batch in the ladder". The naive rule breaks on exactly the
cells this sweep is about: the native gather at
`(float, m=1024, 16 nnz/row, nrhs=12)` runs **2.369, 0.788, 0.309, 0.485 µs per
item** over batch 8/32/128/512 — it *rises* at 512 because the footprint
(119 MB) leaves the 72 MB L2. That is an L2 cliff, not a return to
launch-latency territory.

`run_satext.sh` walks the lanczos ladder past the cliff (`tables.txt`,
`sat1/`, `sat2/`). Per-item µs, vendor / native, float, m=1024, 3 nnz/row,
nrhs=1, banded:

| batch | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| footprint | 37.8 MB (L2) | 75.5 MB | 151 MB | 302 MB |
| vendor | 0.308 | 0.304 | 0.306 | 0.306 |
| native | 0.010 | 0.020 | 0.040 | 0.040 |
| ratio | 0.032 | 0.066 | **0.131** | **0.130** |

**The vendor is flat from batch 256 onward; the native gather is not saturated
until batch ≈ 4096**, because it is fast enough to stay latency-bound longer.
So the spectacular batch-512 lanczos ratios (0.032–0.072 for float) are
**L2-resident numbers and must not be quoted as the algorithmic result**. The
saturated, DRAM-resident ratio is **0.13–0.43**, i.e. a 2.3×–7.7× win. Both
passes agree to three decimal places.

---

## The DRAM roof

Restricting to footprints above **4 × L2 = 288 MB**, so partial residency cannot
inflate the number (`roof.txt`), lanczos shape, batch 4096 and 8192:

| type | pattern | cuSPARSE | native gather |
|---|---|---|---|
| float | either | 120–147 GB/s (**0.12–0.15 × roof**) | 910–928 GB/s (**0.90–0.92 ×**) |
| double | either | 185–233 GB/s (0.18–0.23 ×) | 906–930 GB/s (0.90–0.92 ×) |
| complex\<float\> | either | 185–236 GB/s (0.18–0.23 ×) | 918–931 GB/s (0.91–0.92 ×) |
| complex\<double\> | banded | 306–366 GB/s (0.30–0.36 ×) | 915–926 GB/s (0.91–0.92 ×) |
| complex\<double\> | scattered | 308–360 GB/s (0.31–0.36 ×) | 714–848 GB/s (0.71–0.84 ×) |

**The native gather is AT the DRAM roof and cuSPARSE is 3–8× below it.** The
difference is bandwidth utilisation in the kernel, not host overhead — which is
what the `nsys` split independently confirms.

In the LOBPCG regime **neither** arm is near the roof (transA=NoTrans, footprint
> 288 MB: vendor 0.06–0.16 × roof, native 0.09–0.52 ×). That is the predicted
column-major `op(B)` gather wall — one 32-byte sector per (nonzero, column) touch
for `sizeof(T)` useful bytes — and it binds both arms. The native kernel is still
1.3–4 × faster there, but the design phase's "parity at best" prediction was
**wrong in the native kernel's favour**, and the reason is not that it beats the
gather wall; it is that cuSPARSE pays the wall *and* re-partitions the CSR every
call.

---

## The grid

| sweep | script | grid | rows |
|---|---|---|---|
| lanczos ladder | `run_all.sh` | m=1024, nnz/row=3, nrhs 1–2, **batch 8…1024**, both patterns, transA 0 and 1 | 2 × 4 types × 56 |
| LOBPCG ladder | `run_all.sh` | m 1024/2048/4096, nnz/row=16, nrhs 12/25/50, **batch 8/32/128/512**, scattered, transA 0 and 1 | 2 × 4 × 72 |
| named cells | `run_all.sh` | L/M/S × transB{0,1} × beta{0,1} × pattern{0,1}, transA 0 and 1 | 2 × 4 × 48 |
| scatter boundary | `run_boundary.sh` | nnz/row {3,6,8,12,16} × nrhs {1,2,4,8,12,25}, two (m,batch) points, transA 1; the same plane at transA 0 | 2 × 4 × 360 |
| banded boundary | `run_boundary.sh` | m {1024,2048,4096} × nrhs {12,25,50} × batch {128,512} × transB {0,1}, banded, transA 0 | 2 × 4 × 72 |
| cfloat edge | `run_cfloat_edge.sh` | nrhs {8,9,12,16,17,20,25,32,50}, both patterns, transB=Trans | 2 × 2 × 36 |
| saturation extension | `run_satext.sh` | lanczos shape, **batch 1024/2048/4096/8192** | 2 × 4 × 32 |
| scatter ladder | `run_scatter_ladder.sh` | m {1024,2048} × nnz/row {3,16} × nrhs {1,2,4,8,12} × **batch 128/256/512/1024**, transA 1 | 2 × 4 × 160 |
| **small batch** | `run_smallbatch.sh` | 5 families × 4 types × **batch 1/2/4/8/16/32/64/128**, transA 0 only; `sbL` adds beta{0,1} × pattern{0,1}, `sbT` is transB=Trans banded | 2 × 4 × 64 |

Every sweep runs **twice, in independent processes**, and every route runs in its
own process.

### Reproduction across passes

| pass pair | gate rows in both | worst ratio spread | median spread | rows that flipped side of the 1.10 line |
|---|---|---|---|---|
| `pass1` / `pass2` | 380 / 380 | 1.114 | **1.0034** | 4 (all `transA=Trans`, all with both ratios in 1.09–1.12) |
| `bnd1` / `bnd2` — gather plane | 120 | 1.094 | 1.0041 | **0** |
| `bnd1` / `bnd2` — banded family | 144 | 1.147 | 1.0040 | **0** |
| `bnd1` / `bnd2` — scatter planes | 240 | 1.153 | 1.008 | 7 (all with both ratios in 1.08–1.15) |

---

## VERDICT

### The gather arm, `transA == NoTrans` — a window, and it is large

`verdict.txt`, over 644 rows that are saturated, batch ≥ 128 and chk-agreeing in
**both** passes of their grid:

```
transA == NoTrans (unconditional)
   moves 186 rows; FAILS (1 of 186 over the gate)
   worst-of-two 1.934, median 0.446, best 0.032
   OVER GATE cfloat tA=0 m=2048 nnz/row=16 nrhs=25 b=128 tB=1 beta=0 pat=0
             p1=1.934 p2=1.872

transA == NoTrans AND NOT (complex<float> && transB != NoTrans)
   moves 176 rows; PASSES; worst-of-two 0.968, median 0.445, best 0.032
   the 468 rows it does not move contain 170 measured non-winners (bracketed)
```

The single non-winning family is mapped in `cfedge1/`, `cfedge2/` — `complex<float>`,
`transB = Trans`, **banded** column pattern, m=2048, nnz/row=16, batch=512:

| nrhs | 8 | 9 | 12 | 16 | 17 | 20 | 25 | 32 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| banded, pass 1 | 0.630 | 0.713 | 0.689 | 1.087 | 1.315 | 1.218 | **1.731** | 1.159 | 1.695 |
| banded, pass 2 | 0.663 | 0.737 | 0.761 | 1.040 | 1.274 | 1.162 | **1.714** | 1.157 | 1.703 |
| scattered, pass 1 | 0.670 | 0.856 | 0.888 | 0.963 | 0.979 | 0.793 | 1.000 | 1.019 | 0.953 |
| scattered, pass 2 | 0.671 | 0.927 | 0.850 | 0.909 | 0.938 | 0.792 | 0.993 | 1.016 | 0.941 |

The boundary is bracketed on both sides by measured rows: `nrhs = 12` wins
(0.69–0.76), `nrhs = 17` loses (1.27–1.32). `float` on the identical cells runs
0.36–0.94 and never loses, so this is `complex<float>`-exclusive. It is also
**pattern-exclusive**: on the scattered pattern — the one a filtered eigensolve
actually has — the same family is 0.79–1.02, parity. The suspected mechanism is
the register block: `kNCmax<Cx<float>>` is 8 (`spmm_native.cc:88`), so `nrhs=25`
needs `ceil(25/8) = 4` passes over A with 7 idle accumulator lanes, while `nrhs=32`
needs 4 passes with none — and `nrhs=32` measures 1.16 against `nrhs=25`'s 1.73.
**Not confirmed by profile; recorded as a hypothesis with the non-monotonicity
that suggests it.**

### The scatter arm, `transA == Trans / ConjTrans` — no shippable window

```
transA != NoTrans (unconditional)
   moves 458 rows; FAILS (169 of 458 over the gate)
   worst-of-two 3.011, median 1.030
transA != NoTrans AND nrhs <= 4                FAILS (11 of 204), worst 1.208
transA != NoTrans AND nrhs <= 2                FAILS ( 5 of 151), worst 1.208
transA != NoTrans AND nrhs <= 1                FAILS ( 2 of  60), worst 1.132
transA != NoTrans AND nrhs <= 2 AND !cdouble   PASSES; worst 1.023, median 0.366
```

The scatter's boundary is `nrhs`, and it is **type-dependent**
(`bnd_scatter_a`, m=1024, batch=512, scattered):

| type \ nrhs | 1 | 2 | 4 | 8 | 12 | 25 |
|---|---|---|---|---|---|---|
| float (nnz/row 3) | 0.180 | 0.323 | 0.590 | 0.779 | 0.880 | 1.095 |
| double (nnz/row 3) | 0.194 | 0.350 | 0.636 | 0.996 | 1.076 | 1.103 |
| cfloat (nnz/row 3) | 0.344 | 0.632 | 0.820 | 0.989 | 1.137 | 1.107 |
| cdouble (nnz/row 3) | 0.390 | 0.683 | 0.968 | 1.118 | 1.105 | 1.289 |
| cdouble (nnz/row 16) | 1.043 | 1.078 | 1.112 | 1.164 | 1.206 | 1.307 |

`complex<double>` at 16 nnz/row is already over the gate at `nrhs = 1`
(1.098–1.132 at batch 512/1024, `scl1`/`scl2`), and **`SpmmShape` carries no
`nnz` field** (`route_spmm.hh`, and deliberately: `MatrixView::nnz()` is a
per-item *capacity* and the honest per-item spelling reads device memory in a
path where that is a segfault). So no predicate expressible in the routing shape
separates the `cdouble` win from the `cdouble` loss. The only clause that passes
carries an explicit type exclusion, moves 111 rows, and — per
`spmm_native.cc`'s own note — the transposed arm **has zero in-tree C++ callers
today**. It buys nothing real and adds a type-conditional to a table that has
none.

### Recommended clause — UNCONDITIONAL; the small-batch caveat below is now closed

```cpp
static bool preferred(Route r, const SpmmShape& s) {
    if (!is_native(r) || r.algo != Algorithm::Direct) return false;
    if (s.format != MatrixFormat::CSR) return false;

    // NO BATCH TERM, AND THAT IS MEASURED RATHER THAN ASSUMED. preferred() is
    // consulted on every call, so the batch 1..64 corner -- which the saturated
    // grid does not cover -- was swept separately over 5 shape families x 4
    // types x 2 patterns x 2 betas x both transB, twice
    // (experiments/sparse_spmm/sb{1,2}, smallbatch.txt). Under this clause 0 of
    // 174 admitted rows at batch <= 64 exceed the 1.10 gate, and 1 of 174 costs
    // the caller any time at all: the worst cell anywhere in that region is
    // 1.078 (cfloat, m=4096, 16 nnz/row, nrhs=50, scattered, batch 4,
    // +13.5 us/call), reproduced 1.078 / 1.078. A batch floor would need a
    // measured non-winner outside the gate to bracket it and there is none.
    // The small-batch ratios are launch-latency ratios and are NOT quoted as
    // kernel results; they are quoted only as evidence of no harm.

    // THE GATHER ONLY. transA == NoTrans selects spmm_gather
    // (src/sycl/spmm_native.cc body 1). Measured over 176 saturated,
    // batch >= 128 cells reproduced across two independent passes:
    // worst-of-two-passes t_native/t_vendor = 0.968, median 0.445,
    // best 0.032 -- experiments/sparse_spmm/verdict.txt, built from
    // experiments/sparse_spmm/pass{1,2}/joined.csv and scl{1,2}/joined.csv.
    // At DRAM residency the gather reads 906-931 GB/s, 90-92% of this
    // part's 1008 GB/s roof, for all four scalar types, against cuSPARSE's
    // 120-366 GB/s (experiments/sparse_spmm/roof.txt).
    //
    // The transposed bodies are EXCLUDED and that is a measured refusal, not
    // an omission: 169 of 458 saturated transposed cells are above the 1.10
    // gate, median 1.030, worst 3.011, and the boundary is nnz/row-dependent
    // in a way SpmmShape cannot express.
    if (s.transA != Transpose::NoTrans) return false;

    // THE ONE MEASURED NON-WINNER ON THE GATHER ARM. complex<float> with a
    // transposed dense operand runs 1.71-1.73x SLOWER than cuSPARSE on a
    // strongly banded column pattern at nrhs >= 17 (experiments/
    // sparse_spmm/cfedge{1,2}, both passes). Bracketed: nrhs=12 is 0.69-0.76
    // and nrhs=17 is 1.27-1.32. Excluded whole rather than by nrhs because
    // the shape cannot see the column pattern, and on the scattered pattern
    // the same family is only 0.79-1.02 -- so refusing it costs at most 2%.
    // Bracketed on the BATCH axis too (experiments/sparse_spmm/sb{1,2}): the
    // same cell runs 0.581 at batch 4 and 1.447 at batch 8, peaks at 2.18 at
    // batch 32 and is 1.94 at 128 -- so the loss is not a saturation artefact
    // and no batch-conditional narrows it usefully. float, double and
    // complex<double> on the identical cells stay at 0.22-0.69 across the
    // whole ladder, so the type conditional is exactly as narrow as the data.
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        if (s.transB != Transpose::NoTrans) return false;
    }
    return true;
}
```

**Caveat that was outstanding when the clause above was written — now CLOSED, in
favour of NO batch floor.** Every number above the line is `batch >= 128` at
saturation, per the gate, while `preferred()` is consulted on **every** call.
That corner has now been measured directly (`run_smallbatch.sh`, `sb1/`, `sb2/`,
`smallbatch.csv`, `smallbatch.txt`) and the answer is that **no batch floor is
warranted**: under the recommended clause, zero of 174 admitted rows at
`batch <= 64` are over the 1.10 gate in both passes, and exactly **one** of those
174 rows costs the caller any time at all. The section below is that measurement.
**Nothing in this directory ships a route header change; `route_spmm.hh` is
untouched.**

---

## The small-batch corner of the gather arm — the caveat, closed

### What this sweep is, and what it deliberately is not

`run_smallbatch.sh` walks **batch 1, 2, 4, 8, 16, 32, 64, 128** under five
`transA = NoTrans` families drawn from the cells the recommended clause admits,
for all four scalar types, twice, one route per process — 1 024 timed rows,
**256 vendor/native cells present and chk-agreeing in both passes**.

This is **not** an algorithm comparison and no number in it is quoted as one. At
batch 1–16 the timed region is launch latency, route dispatch and — on the vendor
arm — the `cusparseSpMM_bufferSize` re-query `spmm_vendor` performs on every call.
The campaign rule against quoting unsaturated ratios stands, and the tables below
therefore carry the **absolute per-call microseconds of both arms** beside every
ratio.

The question a `preferred()` clause actually needs answered at low batch is not
"is native faster at batch 4". It is **"is native ever materially SLOWER below
batch 128, and where"** — because a region where both arms are within noise is a
region where either routing is fine, and that licenses an unconditional clause
just as a win does.

| family | shape | why it is here |
|---|---|---|
| `sbL` | m=1024, 3 nnz/row, nrhs=2, tB=N, **both patterns, beta 0 and 1** | lanczos; the beta and pattern axes had *no* low-batch coverage before |
| `sbM` | m=1024, 16 nnz/row, nrhs=12, tB=N, scattered | LOBPCG M |
| `sbS` | m=2048, 16 nnz/row, nrhs=25, tB=N, scattered | LOBPCG S |
| `sbB` | m=4096, 16 nnz/row, nrhs=50, tB=N, scattered | the largest LOBPCG shape; the tightest ratios in the whole campaign |
| `sbT` | m=2048, 16 nnz/row, nrhs=25, **tB=T, banded** | the known large-batch loser family, measured for **all four types** so the exclusion is widened or kept on evidence |

What the main grid already covered below batch 128 was `lanczos_ta0` and
`lobpcg_ta0` at batch 8/32/64 only, `transB = NoTrans` and `beta = 0` only — 120
rows per pass, none over the gate, max 0.996. This sweep adds batch **1, 2, 4 and
16**, `beta = 1`, and `transB = Trans`, which had **no** low-batch row anywhere.

### Route pins and the cross-sweep anchor

Both passes: 160 `reached` coverage rows per route per pass, **all**
`native:direct tA=0` on the native arm and **all** `vendor:auto tA=0` on the
vendor arm, read off `BATCHLAS_COVERAGE_OUT`, never off the environment.

`sbT` at batch 128 is deliberately the *same cell* as `cellsC0, tB=1, beta=0,
pat=0` in `pass1`/`pass2`. Four independent processes across two different
sweeps, run about 90 min apart on the same box:

| type | sb1 | sb2 | pass1 | pass2 |
|---|---|---|---|---|
| float | 0.458 | 0.471 | 0.460 | 0.453 |
| double | 0.613 | 0.609 | 0.612 | 0.599 |
| complex\<float\> | 1.826 | 1.944 | 1.934 | 1.872 |
| complex\<double\> | 0.663 | 0.669 | 0.662 | 0.666 |

That is the harness validating itself, including on the one cell that loses.

### The result, per batch rung, under the recommended clause

`transA = NoTrans` minus `complex<float> && transB != NoTrans`, worst of two
independent passes, admitted by the same rel_sd-**or**-reproduction rule
`verdict.py` uses:

| batch | rows | worst | median | best | over the 1.10 gate | max Δ µs/call (native − vendor) |
|---|---|---|---|---|---|---|
| 1 | 22 | 0.992 | 0.592 | 0.186 | **0** | −0.23 |
| 2 | 27 | 0.981 | 0.287 | 0.170 | **0** | −0.44 |
| 4 | 21 | **1.078** | 0.473 | 0.173 | **0** | **+13.46** |
| 8 | 25 | 0.956 | 0.460 | 0.154 | **0** | −3.64 |
| 16 | 25 | 0.966 | 0.499 | 0.154 | **0** | −11.04 |
| 32 | 29 | 0.978 | 0.373 | 0.134 | **0** | −16.17 |
| 64 | 25 | 0.967 | 0.412 | 0.099 | **0** | −23.51 |
| 128 | 30 | 0.964 | 0.305 | 0.066 | **0** | −36.00 |

Over the whole unmeasured region (`batch <= 64`, 174 admitted rows): worst
**1.078**, median 0.440, best 0.099, and **1 of 174 rows has a positive per-call
delta at all**. The median row saves the caller 18.1 µs per `spmm` call.

**There is no batch at which routing native harms a caller the clause admits.**
The ratio is not flat across the ladder — it degrades monotonically toward 1 as
batch falls, exactly as a launch-latency-bound region should — but it degrades to
**parity, not to a loss**, and it never crosses.

### Why the small-batch win is real for the caller even though it is not a kernel win

At batch 1 the two arms' per-call floors are:

| family | vendor µs/call | native µs/call |
|---|---|---|
| `sbL` (all four types) | 15.1 – 16.6 | 2.95 – 3.43 |
| `sbM` | 16.2 – 26.6 | 11.6 – 17.5 |
| `sbS` | 23.2 – 42.1 | 15.9 – 22.7 |
| `sbB` | 48.5 – 123.8 | 39.1 – 82.1 |

The native arm's floor is ~2.9 µs and does not depend on the shape; the vendor's
is ~13–17 µs on the cheapest cell in the sweep. That gap is **not** a kernel
result — it is `spmm_vendor`'s per-call chain (`setStream`, the
`SpmmCsrBatchPlan` host walk, the `cusparseSpMM_bufferSize` re-query, the
`BumpAllocator` carve) against the native arm's single launch. It is nonetheless
a cost a caller genuinely pays and **cannot hoist**, which is why it is charged to
both arms in the timed region (see "What is inside the timed region"). It is
reported here as *the reason the small-batch region is safe*, not as evidence
about either kernel.

### The one measured non-winner, and it is what brackets the recommendation

`complex<float>, sbB (m=4096, 16 nnz/row, nrhs=50, scattered, transB=NoTrans),
batch = 4`: **1.078 in pass 1 and 1.078 in pass 2**, rel_sd 0.011 / 0.009,
vendor 173.4 µs, native 186.8 µs, **Δ = +13.46 µs per call**.

This is the *only* row of 174 in the small-batch corner where the native arm
costs the caller anything, it reproduces to three decimal places across two
independent processes, and it sits **below** the 1.10 gate. It is also
non-monotonic in batch — `b=2` is 0.977 and `b=8` is 0.956 on the same cell —
which is what an unsaturated launch/occupancy artefact looks like rather than a
structural loss.

**This row is the bracketing evidence for recommending NO batch floor.** A floor
must be justified by a measured non-winner on the wrong side of it; the worst
measured cell anywhere in the region a floor would cut off is 1.078, i.e. inside
the gate. There is no floor that this grid supports, and inventing one would
forfeit the 0.099–0.5 region below batch 64 for nothing — the mistake the
campaign already made in the opposite direction with `getri` at batch ≤ 32.

The `complex<float>` `sbB` family is the tightest in the whole sweep at every
batch (0.910–0.978 from batch 1 to 128), so the clause's margin there is thin at
*all* batches, not just small ones. That is a property of the cell, not of the
batch axis, and it is already inside the saturated grid's 0.968 worst-of-two.

### The `complex<float>` + `transB=Trans` loser, all the way down

The family the clause excludes, measured across the full ladder for all four
types (banded, m=2048, 16 nnz/row, nrhs=25). `float`, `double` and
`complex<double>` on the identical cells are the controls:

| type | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 |
|---|---|---|---|---|---|---|---|---|
| float | 0.655 | 0.569 | 0.485 | 0.402 | 0.409 | 0.403 | 0.406 | 0.471 |
| double | 0.341 | 0.275 | 0.221 | 0.503 | 0.634 | 0.625 | 0.621 | 0.613 |
| **complex\<float\>** | 0.520 | 0.524 | 0.581 | **1.447** | **2.072** | **2.182** | **2.092** | **1.944** |
| complex\<double\> | 0.335 | 0.287 | 0.432 | 0.688 | 0.676 | 0.681 | 0.681 | 0.669 |

Three things this settles, and one it costs:

1. **The loss is not a large-batch phenomenon.** It switches on between batch 4
   and batch 8 and is then worst in the *middle* of the ladder (2.18 at batch 32),
   not at the top. The exclusion is therefore right at every batch it covers,
   which it would not have been if the family had only lost at saturation.
2. **No widening is needed.** Only `complex<float>` crosses; the other three types
   stay at 0.22–0.69 across the entire ladder on the identical cells, so the
   type-conditional exclusion is exactly as narrow as the evidence.
3. **It is bracketed on the batch axis too**, not just the `nrhs` axis
   (`cfedge1`/`cfedge2`): batch 4 wins at 0.581 and batch 8 loses at 1.447.
4. **The cost of stating the exclusion unconditionally**: at batch 1–4 the excluded
   family is a 1.7–1.9× native **win** that the clause now refuses. That is a real
   forfeit and it is recorded rather than hidden. It is not worth buying back — it
   would take a `batch <= 4` sub-clause on a type-conditional, on a shape that has
   no in-tree caller, whose boundary is a single ladder rung wide.

### The filter check, repeated — because this campaign has been burned by it once

46 of the 256 cells were not admitted by the rel_sd-or-reproduction rule (small
batch is jittery; that rate is expected). The `README` records an earlier episode
where an rel_sd-only filter deleted the one real loss and manufactured a passing
clause, so **every dropped row was inspected**, not merely counted:

- Exactly **two** dropped rows have either pass over the gate, and both are
  `complex<float>` + `transB=Trans` (batch 16: 1.958/2.072; batch 128:
  1.826/1.944) — i.e. the family the clause already excludes, corroborated by the
  *admitted* batch 8/32/64 rows of the same family.
- **Every other dropped row is a native win**, the largest of them at 0.955.

So the filter is not concealing a loss in this sweep. The dropped rows argue for
the exclusion the clause already makes, and for nothing else.

### Verdict on the caveat

**Recommend the clause UNCONDITIONALLY — no `batch` term.**

- Measured region: batch 1 → 128, 4 types, 5 families, 2 patterns, 2 betas,
  both `transB`, two independent passes, routes proved from the coverage
  instrument.
- Harms under the clause: **0 of 174** admitted rows at `batch <= 64`, **0 of 30**
  at batch 128.
- Worst measured cell in the region a floor would cut off: **1.078**
  (`complex<float>`, m=4096, 16 nnz/row, nrhs=50, scattered, batch 4,
  +13.5 µs/call), reproduced at 1.078 / 1.078 — **inside** the 1.10 gate, so it
  brackets the recommendation instead of contradicting it.
- A batch floor would have to be justified by a non-winner outside the gate. This
  grid contains none, at any rung, for any type the clause admits.

The clause body above is therefore unchanged; only its documentation gains the
small-batch line.

---

## Negative results

1. **The transposed arm has no shippable window.** 169 of 458 saturated
   transposed cells fail the gate, worst 3.011 (`cdouble`, m=4096, nnz/row=16,
   nrhs=50, batch=512, reproducing 3.000 / 3.011). Every `nrhs`-only clause
   fails; the only passing one needs a `complex<double>` exclusion and moves an
   arm with no in-tree caller.
2. **The design phase's "parity at best in the LOBPCG regime" prediction is
   refuted — but so is the reason it gave.** The gather beats cuSPARSE 1.3–4× at
   LOBPCG shapes, and *not* by beating the column-major `op(B)` gather wall:
   neither arm exceeds 0.52 × roof there. The margin comes from cuSPARSE
   re-running a `csr_partition_kernel` on every call (36 % of its GPU time).
3. **The spectacular lanczos ratios are an L2 artefact and shrink by 4× at true
   saturation.** float, m=1024, 3 nnz/row, nrhs=1: 0.032 at batch 1024 (37.8 MB,
   L2-resident) → **0.131 at batch 4096** (151 MB, DRAM-resident), stable to
   batch 8192. Any headline quoted from the batch-512 grid alone would have been
   4× too flattering.
4. **`complex<float>` + `transB = Trans` + banded + `nrhs >= 17` is a genuine
   NoTrans loss, 1.71–1.73×**, and it was nearly lost to an rel_sd filter (see
   "the filter that manufactured a win").
5. **`complex<double>` transposed at 16 nnz/row loses at `nrhs = 1`**
   (1.098–1.132) — the narrowest possible transposed shape, which is where the
   scatter should have been strongest.
6. **The `complex<float>` + `transB=Trans` + banded loss is NOT a saturation
   effect, which is the opposite of what the batch-512-only evidence suggested.**
   It switches on between batch 4 and batch 8 (0.581 → 1.447) and is worst in the
   *middle* of the ladder — 2.18 at batch 32, 1.94 at batch 128, 1.71–1.73 at
   batch 512. Had it been a large-batch artefact, a batch-floored clause would
   have been the cheap fix; it is not, so the type/`transB` exclusion is the only
   correct one.
7. **The `complex<float>` LOBPCG-XL family is thin at EVERY batch, not just small
   ones** (m=4096, 16 nnz/row, nrhs=50, scattered, `transB=NoTrans`): 0.910,
   0.977, **1.078**, 0.956, 0.966, 0.978, 0.967, 0.964 at batch 1→128, against a
   saturated worst-of-two of 0.968. The single row over 1.0 reproduces to three
   decimals and is the tightest cell in the campaign — inside the gate, but the
   clause's real margin on `complex<float>` gather is ~3 %, not the 2× the median
   suggests.
8. **No batch floor is supportable, in either direction.** 0 of 174 admitted rows
   at `batch <= 64` are over the gate under the recommended clause, so a floor has
   nothing to bracket it; and the ratios degrade toward parity, never through it,
   as batch falls.

---

## Reproducing

```bash
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cmake --build build --target spmm_benchmark -j16
./experiments/sparse_spmm/run_all.sh            pass1   # ~21 min, device 1, exclusive
./experiments/sparse_spmm/run_all.sh            pass2
./experiments/sparse_spmm/run_boundary.sh       bnd1
./experiments/sparse_spmm/run_boundary.sh       bnd2
./experiments/sparse_spmm/run_cfloat_edge.sh    cfedge1
./experiments/sparse_spmm/run_cfloat_edge.sh    cfedge2
./experiments/sparse_spmm/run_satext.sh         sat1
./experiments/sparse_spmm/run_satext.sh         sat2
./experiments/sparse_spmm/run_scatter_ladder.sh scl1
./experiments/sparse_spmm/run_scatter_ladder.sh scl2
./experiments/sparse_spmm/run_smallbatch.sh      sb1   # ~25 min each
./experiments/sparse_spmm/run_smallbatch.sh      sb2
./experiments/sparse_spmm/run_nsys.sh

for p in pass1 pass2 bnd1 bnd2 cfedge1 cfedge2 sat1 sat2 scl1 scl2 sb1 sb2; do
  python3 experiments/sparse_spmm/analyse.py experiments/sparse_spmm/$p
done
python3 experiments/sparse_spmm/report.py  experiments/sparse_spmm/pass{1,2}/joined.csv
python3 experiments/sparse_spmm/verdict.py experiments/sparse_spmm/pass{1,2}/joined.csv \
                                           experiments/sparse_spmm/scl{1,2}/joined.csv
python3 experiments/sparse_spmm/roof.py   pass1 pass2
python3 experiments/sparse_spmm/tables.py
python3 experiments/sparse_spmm/nsys_table.py
python3 experiments/sparse_spmm/sb_report.py experiments/sparse_spmm/sb1 experiments/sparse_spmm/sb2
```

Run one at a time. The two RTX 4090s here share a NUMA node, a CPU affinity mask
and one UVM driver, and `nvidia-smi --query-compute-apps` is **per device**, so a
second sweep on the other card is invisible to the guard and has been measured
inflating a cell 5.5× at a beautifully stable rel_sd.

## Files

| file | what it holds |
|---|---|
| `pass1/`, `pass2/` | the main grid, two independent passes; `joined.csv` is the vendor/native join |
| `bnd1/`, `bnd2/` | the (nnz/row × nrhs) boundary planes and the banded family |
| `cfedge1/`, `cfedge2/` | the `complex<float>` nrhs walk that places the one gather-arm loss |
| `sat1/`, `sat2/` | the batch 1024–8192 saturation extension |
| `scl1/`, `scl2/` | the scatter arm's nrhs boundary **with a batch ladder under every cell** |
| `sb1/`, `sb2/` | the **small-batch corner** of the gather arm, batch 1-128, two independent passes |
| `nsys/` | the kernel/host decomposition traces |
| `probe/` | the warm-up, ordering and route-pin probes |
| `smallbatch.csv`, `smallbatch.txt` | the small-batch join and its ladders / clause arithmetic (`sb_report.py`) |
| `report_pass12.txt`, `verdict.txt`, `roof.txt`, `tables.txt`, `nsys_split.txt` | the saved tables every number above is quoted from |

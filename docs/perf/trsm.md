# TRSM: the CTA tier, the blocked driver, and the barrier that was missing (WP3)

Native batched `trsm`: two SYCL kernels, a routing window measured on an RTX 4090, and a wrong-answer bug that the whole work package's benchmark grid was measured on top of. Read `## open-debts` before trusting any ratio here. Shipped code is authoritative for *what*; `WP3_TRSM_SPEC.md`, `WP3_TRSM_SPEC_CORRECTIONS.md` and `experiments/wp3*` for *why*.

---

## what-ships

### route-arms

`kTrsmOrder`, `include/batchlas/blas/dispatch/route_trsm.hh:121-125`, in walk order:

| Origin | Algorithm | implementation | serves |
|---|---|---|---|
| `Native` | `CTA` | `sycl_trsm::trsm_native_v1_dispatch` (V1) | triangular order <= 32 |
| `Native` | `Blocked` | `sycl_trsm::trsm_native_blocked` (V2) | every order; calls V1 per diagonal block |
| `Vendor` | `Auto` | `backend::trsm_vendor` | everything |

A capability ladder, not a preference: CTA cannot serve order > 32, so the vendor-off fallback (`route_resolve.hh:113-127`) tries the tighter route first. That fallback is now **two** passes: the first honours an optional `native_tier_preferred` tie-break hook, the second (`:125-126`) re-walks testing `supports()` alone. `RouteTable<Op::trsm>` declares no such hook, so for trsm the two passes are identical and the behaviour is the single supports()-only walk the route header describes. (Both `route_trsm.hh`'s header comment and `route_resolve.hh`'s own cross-references still say `:60-63`, which is now a comment about the hook, not the walk.) Dispatch happens once, in the facade at `src/dispatch/entry_points/level3.cc:197-276`, **before** the vendor-available test.

`supports()` (`route_trsm.hh:134-186`) holds correctness gates only — `is_gpu`, `!heterogeneous_batch` (which trsm's shape builder never populates, so it cannot fire — debt 12), `order >= 1 && q >= 1 && batch >= 1`, per-arm capacity. Nothing type-dependent, no speed number: a threshold in `supports()` makes a vendor-free `trsm` **throw**, not run slower. `Algorithm::Auto` is deliberately unsupported for native, since two native arms mean a bare "native" names neither. Capacities, all four types (`src/sycl/trsm_native.cc:939-942`, `:964-967`): `trsm_cta_max_n<T>() == 32` and `trsm_blocked_available<T>() == true`.

### the-preferred-window-as-implemented

Quoted from `include/batchlas/blas/dispatch/route_trsm.hh:224-325`:

```cpp
if (!is_native(r)) return false;                      // :230
const int64_t order = s.tri_order();
if (s.batch < 8) return false;                        // :248
if constexpr (std::is_same_v<T, float>) {
    if (s.side == Side::Left) {
        return true;                                  // :296
    }
    return s.batch >= 128 || order <= 32;             // :304  (float, Side::Right)
} else {
    return true;                                      // :323  (double, cfloat, cdouble)
}
```

**Native is preferred for every type, both sides, every order, at batch >= 8**, with one exception: `float` + `Side::Right` at batch in `[8,127]` is preferred only for order <= 32. There is **no upper order bound anywhere**; above 32 the blocked driver takes over and `supports()` has already routed it.

Two windows in the exploration notes are **not** what ships:

* the notes record `float && Side::Left -> order <= 16` (step 9), then `order <= 128` (step 12), then `order <= 128 || q*batch < 524288` (step 13). The shipped predicate is unconditional `return true`; step 16 deleted the work threshold.
* spec §10 proposed one `trsm_use_native()` predicate carrying `batch*q < 8*CU*32 -> vendor`. Nothing like it ships — see `### rejected-the-starvation-guard`.

### the-shape-builder-and-the-field-mapping

`src/backends/trsm_route.hh` builds `TrsmShape` and is the only place touching the device or the environment (the table must stay pure). The mapping is trmm's, not the spec's, and getting it wrong is silent: `s.m = B.rows()`, `s.n = B.cols()`, `s.k = A.rows() == A.cols() ==` **the triangular order**; `tri_order() == s.k`, `rhs_count() == (side == Left ? s.n : s.m)`. `trsm_op_shape` returns `nullopt` when `A.batch_size() != B.batch_size()` or A is not square — the only place batch disagreement is caught, since `trsm_validate_params` (`functions/trsm.hh:39`) does not compare the two batch counts.

### tuning-knobs-and-environment

* `BATCHLAS_TRSM_ROUTE` — route override (`cta` / `blocked` / `native` / `vendor`). **`BATCHLAS_TRSM_VARIANT` is read by nothing**; `legacy_variable_for` has no `Op::trsm` case. The spec instructs pinning the native path with that variable, which would pin nothing.
* `BATCHLAS_TRSM_OUTER_NB` — V2's outer block width; a **tuning** knob, never a routing one (`trsm_native.cc:692-703`). Default 128 for `Side::Left`, `cta_max_n` (32) for `Side::Right`, rounded down to a whole number of CTA blocks. The parse is cached in a function-local static, so the first blocked call in a process fixes it.

---

## design-v1-v2-and-the-canonical-fold

**V1** (`Algorithm::CTA`): one work-group per matrix, one work-item per independent solve, the solution vector resident in that thread's registers as `T x[N]`, the canonical triangle staged once into SLM and broadcast (every thread reads the same `Lc(s,t)` at each step, so bank layout is irrelevant). The 24 canonical `(side, uplo, transA, diag)` cases fold into one recurrence via `canonicalise()` (`trsm_native.cc:83`); the index map is `rho(s) = fwd ? s : order-1-s`.

**V2** (`Algorithm::Blocked`): a host-side two-level driver. The outer level blocks at `OUTER_NB` and issues a trailing GEMM; each outer panel is then solved by the inner `nb = cta_max_n = 32` loop against its own, much shorter prefix. V1 is literally V2's panel solve, so the crossover is a capacity, not a tuned guess.

The grid is `batch * ceil(q/WG)`, never batch alone — the guard against this repo's recurring batch-only-parallelism defect. The work-group ladder walks `{256,128,64,32}` and takes the first `cand` with `bs*ceil(q/cand) >= 4*CU` (`trsm_native.cc:265-274`). It cannot exceed 256: the worst instantiation is `complex<double>` N=32 at 226 registers, and `226*256 = 57,856` against the hard 65,536-registers-per-block limit — 12% headroom. That is a `static_assert` at `:262`, not a comment.

**Diagonal-block inversion is rejected at every tier** (spec §2.4, survived the verification pass). The "free for ortho" licence compares a trsm *residual* bound against a CholQR *orthogonality* bound; restated in orthogonality currency the inverted variant contributes at the same order as the existing term with an unbounded constant, and any constant above ~2 flips Chol2 from recovering to not. This retires the plan's Risk 4 for `trsm` rather than measuring it.

### the-register-gate-and-the-cta-capacity

The gate is `stack frame == 0` **AND** `0 spill bytes` **AND** `registers * WG <= 65536`, measured with `scripts/register_probe.sh`, which replays the shared library's `link.txt`. A per-TU `-Xcuda-ptxas -v` is reported "argument unused" — device code is AOT-compiled to a cubin at the *shared-library device link* — so grepping such a log for "spill" finds nothing and reads as "no spill". That is a phantom measurement whichever gate you use.

| type | N=8 | N=16 | N=32 | regs*256 at N=32 |
|---|---|---|---|---|
| float | 44 | 76 | 114 | 29,184 |
| double | 59 | 101 | 153 | 39,168 |
| complex\<float\> | 50 | 86 | 148 | 37,888 |
| complex\<double\> | 74 | 138 | 226 | **57,856** |

Zero frame and zero spill in all 24 kernels (4 types x 3 buckets x 2 sides). N=64 fails: float 119 regs / **256 B frame**, double 145 regs / **512 B frame**, both with **zero spill**. 256 B is 64 floats and 512 B is 64 doubles — `x[]` itself in local memory, which voids V1's entire thesis. **Read the frame column, not the spill column**: a spill-only check passes N=64 while the design is void. The spec's "256 B/thread register cliff" is falsified (`gemm_kernels.cc:869-876`: an 8x8 double tile at 208 registers, `complex<float>` at 247, both spill-free; only `complex<double>` spills there, and only 3.4 KB), and the corrections document's "gate on spill, not frame" is wrong *for this kernel*, where the only thing that can be on the stack is the accumulator. The spec predicted `n_cta(float) = 64` and reached the right answer (32) only through a fallback instruction whose stated mechanism does not occur.

### the-side-left-staging-tile

For `Side::Left` thread `u` owns column `u`, so at step `s` the lanes of a warp read `B(rho(s), u0+lane)` — addresses `ldb` apart. ncu, float, n=32, q=1024, batch=512:

| | load sec/req | store sec/req | DRAM vs analytic floor | kernel time |
|---|---|---|---|---|
| Left, before | **31.39** (7.85x) | **32.00** (8.00x) | 0.85x | 0.517 ms |
| Left, after | **5.13** | **4.00** | — | **0.145 ms** |
| Right (never staged) | 5.13 (1.28x) | 4.00 | 0.75x | 0.141 ms |

The spec named the right factor at the wrong level. It predicted "8x over-fetch on read and write-allocate", which reads as DRAM traffic; DRAM measures **0.75-0.85x of the analytic floor `2*q*n*sizeof(T)*batch`, i.e. below it** — the bytes a lane skips at step `s` are the bytes it wants at steps `s+1..s+7`, still in cache when it gets there. The defect is entirely LSU/L1 transaction count. Same fix either way, but "short of DRAM bandwidth" aims the obvious responses (vectorised loads, wider tiles) at the wrong resource, and that misreading has already cost this repo one panel kernel.

**Staging is gated to the real types, by measurement** (`trsm_stage_left`, `trsm_native.cc:214`). Applied to complex it costs register residency: `complex<float>` N=32 gains a **464 B** frame and `complex<double>` a **232 B** one, both zero-spill, because the nested round loop stops fully unrolling for wide bodies. And complex cannot benefit — over-fetch is `32/sizeof(T)` lanes per sector, so a 16-byte scalar is capped at 2x. **float is the only type that loses precisely because 32/4 = 8.** With the gate, complex measures 1.00-1.01x either way and all 24 kernels are back to zero frame.

Tile height is 16 for `sizeof(T) <= 4` and 8 otherwise; row stride is `NB_STAGE + 1`, and that padding is what makes the read-out conflict-free. The spec's §4.1 size formula allocated `NB_STAGE * WG`, which at `WG=128, NB_STAGE=16` indexes `15 + 127*17 = 2174` into a 2048-element allocation: **127 elements past the end**.

### the-two-level-blocked-driver

V2's outer block used to be `nb = trsm_cta_max_n<T>() = 32`. At n=512 that is 16 blocks, so every trailing GEMM had one dimension pinned at 32. GEMM arithmetic intensity with a dimension pinned at `w` tends to `2w/sizeof(T)` flop/byte = 16 for float, against an RTX 4090 machine balance of ~42 — **93.75% of the solve's flops ran bandwidth-bound by construction** on a problem that at n=512 is intrinsically compute-bound (51 flop/byte). The left-looking re-read factor `(p-1)/2` compounds it: 7.5x at n=512. Traffic model, B elements per batch item in units of q at n=512, ideal 1024: NB=32 -> 5824 (5.7x); **NB=128, nb=32 -> 4096 (4.0x, ships)**; NB=128, nb=64 -> 3328; NB=128, nb=128 -> 2560.

The width is **side-dependent, measured not aesthetic**. `OUTER_NB` sweep on float, worst cell per order, `vendor_ms/native_ms`:

| order | Left nb32 | nb64 | nb128 | nb256 | Right nb32 | nb64 | nb128 | nb256 |
|---|---|---|---|---|---|---|---|---|
| 128 | 1.18 | 1.10 | 1.20 | 1.17 | 1.00 | 0.96 | 1.00 | 0.98 |
| 256 | 0.75 | 0.78 | **0.87** | 0.75 | 1.01 | 0.94 | **0.83** | 1.01 |
| 512 | 0.58 | 0.74 | **0.76** | 0.75 | 1.07 | 0.92 | **0.82** | 0.91 |

Widening helps Left at every large order and **hurts Right at every large order**, turning two winning Right cells into losses: Left's update is `C(nb x q)`, Right's is `C(q x nb)`, so they land in different `select_kernel_variant` clauses, and widening also shortens the inner updates' `k` below the `k >= 128` gate float's transposed fast paths require. One number for both sides would have to be 32, discarding everything the two-level driver buys. `nb256` degenerating to `nb32` at order 256 (0.75 vs 0.75; 1.01 vs 1.01) is the internal check that the knob does what it says.

The notes present this sweep as "on float", but `nb_sweep.csv` also carries `complex<float>`, and recomputed it is **nearly insensitive to the knob**: 6.91-6.97x at order 128, 7.82-8.10x at 256, 17.67-18.59x at 512 (one 27.41x outlier at nb256). So the side split is a float-only effect, and the shipped side-dependent default is applied to all four types on float's evidence alone. `double` and `complex<double>` were never in this sweep.

---

## the-measured-grid

All ratios are `vendor_ms / native_ms`; **>1 means native is faster**. RTX 4090, one card held exclusive by `experiments/gpu_guard.sh`, warmup ahead of every timed iteration (no cold-JIT number), `bench::pristine(B)` restored between iterations because trsm is in place, and the two legs of every pair differ **only** in `BATCHLAS_TRSM_ROUTE`.

### the-step-9-grid

The grid is `benchmarks/trsm_benchmark.cc`'s `TrsmOrthoSizes`: n in {8,16,32,64,128,256} x q in {256,1024,4096} x batch in {128,512,2048}, all four types, both sides. **Not** a square RHS — the library never issues one. The two real call sites (`ortho.cc:202`, `:289`) pass a k x k Cholesky factor as A and an m x k basis as B, so the triangular order is small and the other extent large. Coverage capture confirms it against what the suite issues: `n=10 q=256 batch=1` (4880 calls), `n=10 q=20` (4800), `n=12 q=36 batch=3` (2392), `n=5 q=64 batch=3` (3258) — every one `Side::Right, Lower, Trans, NonUnit`, every one inside V1's capacity of 32.

**Nine of the spec's 54 cells are dropped**, by a 6 GB cap rather than by the card: recomputed from `trsm_grid_bytes()` the nine ask 6.4-70.9 GB, and only two of them (70.9 and 34.9 GB) actually exceed this box's 24 GB. (Both the benchmark comment and the step-9 README say "do not fit in 24 GB"; the arithmetic says the cap is what drops the other seven.) The grid *prints* every dropped cell; a grid that shrinks quietly reads exactly like one that covered everything. The cap is computed for `complex<double>` and applied to all types so the type columns stay comparable. An earlier draft of the cap table omitted the harness's pristine copy of B and understated every row by ~2x — read the figures off `trsm_grid_bytes()`, do not re-derive them.

Saturation is enforced per `(type, side, n, q)`: a ratio is quoted only if the top batch's GFLOP/s is within 1.15x of the batch below. Step 9, min-max over all measured cells at each order, recomputed from the committed CSVs:

| type | side | order 8 | 32 | 128 | 256 |
|---|---|---|---|---|---|
| double | Right | 4.46-9.62 | 3.65-4.30 | 1.97-2.04 | 1.58-1.62 |
| double | Left | 3.38-8.46 | 2.39-3.52 | 1.62-1.86 | **1.39**-1.51 |
| complex\<double\> | Right | 3.96-9.87 | 3.76-14.35 | 1.33-5.30 | **1.20**-4.74 |
| complex\<double\> | Left | 2.27-6.07 | 2.86-10.39 | 3.52-5.12 | 3.49-4.66 |
| complex\<float\> | Right | **1.01**-1.52 | 1.42-3.84 | 4.99-6.30 | 5.51-8.01 |
| complex\<float\> | Left | 1.06-1.40 | 2.71-3.50 | 6.54-15.85 | 8.20-**21.91** |
| float | Right | 1.62-4.59 | 2.21-3.61 | **0.97**-1.58 | 1.02-1.63 |
| float | Left | 1.61-3.58 | **0.70**-0.87 | 0.71-0.79 | **0.57**-0.63 |

Saturated subsets as ranked in the notes: double 32/32 cells won (1.39x-9.62x), `complex<double>` 30/30 (1.20x-4.66x), `complex<float>` 30/30 (1.01x-21.91x), float/Right 18/18 (1.54x-4.59x), float/Left **6 of 16**. The one losing region is exactly where §3.4 predicted, and `double` at the same shapes wins 1.39-6.37x — same kernel, same access pattern, opposite verdict, because cuBLAS's double triangular path is weak enough that the over-fetch never decides the race. That is why this is a per-type predicate and not one number.

### the-batch-floor

`if (s.batch < 8) return false;` — from `starved.sh`, batch in {1,8,32} x q in {32,128}, n in {8,32,128}, run on GPU 1 while the saturated sweep owned GPU 0:

All rows below are the **q = 32** leg (the q = 128 leg is in the same CSVs and tells the same story, except where noted):

| type, side, order | batch=1 | batch=8 | batch=32 |
|---|---|---|---|
| float Right n=32 | **0.462** | 2.108 | 2.063 |
| float Right n=128 | **0.401** | **0.779** | **0.778** |
| float Left n=32 | **0.415** | 1.223 | 1.232 |
| float Left n=128 | **0.229** | **0.756** | **0.756** |
| double Right n=32 | **0.803** | 1.325 | 1.329 |
| double Right n=128 | **0.852** | 1.215 | 1.213 |
| double Right n=8 | 1.131 | 2.932 | 2.915 |
| complex\<double\> Right n=32 | 3.536 | 9.975 | 39.767 |

The boundary sits at the **first measured win**, not at a round number: batch=1 loses at every order >= 32 for both real types, and batch=8 wins at every order **except float at 128**, which stays a loss (0.740-0.810x on both sides at batch 8 and 32) — that residue is exactly what the float/`Side::Right` order clause below encodes. Float/`Side::Left` has **no** such clause and prefers native there, on the strength of the saturated grid at batch >= 128; the only evidence in `[8,127]` is this profile, and it says 0.756-0.780x. Bracketed on both sides for float and double at orders 32 and 128.

Two caveats. (1) `starved.sh` says in its own header: *"PROFILE ONLY, NOT FOR RANKING… every number this produces is dominated by launch overhead; a ratio read off it is an overhead ratio and must not be quoted as an algorithm result."* The shipped batch floor is nevertheless derived from exactly these numbers. (2) The floor is **type-blind and demonstrably over-broad**: recomputed over both q legs, `double` at order 8 wins **1.09-1.15x** at batch=1, `complex<float>` wins 1.29-12.8x at batch=1 at every order measured, and `complex<double>` wins **2.1-11.1x** at batch=1 (2.1-2.9x at order 8, rising to 7.0-11.1x at order 128) — and `preferred()` hands all of them to the vendor.

### float-side-right-the-only-order-clause

`return s.batch >= 128 || order <= 32;`

* **Bracketed below:** at batch 8 and 32, order 128 measures 0.740-0.810 (loss) while order 32 measures 1.157-2.108 (win). The clause keeps the winner and drops the loser.
* **Unbracketed:** **order 64 was never measured at any batch below 128.** The 32/64 cut point interpolates between a measured win at 32 and a measured loss at 128 — treat it as unverified.
* Inside the window the code knowingly accepts one small loss (below).

### the-final-grid-after-the-routed-trailing-gemm

Step 16's grid (`experiments/wp3_s16/baseline.csv`) covers orders 8..512, both sides, **float and `complex<float>` only**. Worst clean cell per order (relative sd <= 10%):

| | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| float Left | 1.60 | 1.72 | 1.69 | 1.65 | 1.42 | 1.25 | 1.21 |
| float Right | 1.62 | 2.37 | 2.23 | 2.01 | 1.60 | 1.23 | **1.00** |
| complex\<float\> Left | 1.05 | 1.41 | 2.64 | 4.69 | 11.21 | 16.62 | 51.70 |
| complex\<float\> Right | 1.01 | 1.03 | 1.35 | 1.90 | 8.49 | 15.31 | 19.64 |

Recomputed from the committed CSV, the file holds **224 vendor/native pairs, all clean, of which 223 win**. The route header and the plan both say "167 of 168": the substantive claim (exactly one losing cell) reproduces, **the cell count does not**, and no committed artefact explains the 168.

The single non-winner is float / `Side::Right` / order 512 / q=256 / batch=128, at **0.9787, 0.9776, 0.9832** over three explicit repeats with a longer window (`recheck-{native,vendor}-{1,2,3}.csv`). It is the smallest-work cell at that order (~1.0 ms total) and its neighbours win 1.30-1.38x. No router clause is fitted to it — the clause would be narrower than the noise floor of most of this table.

Progression of the float/`Side::Left` losing region, from the committed baselines:

| grid | clean pairs | losing cells | worst |
|---|---|---|---|
| step 13, before two-level blocking | 192 | 14 (float/Left, orders 256 and 512) | 0.583 |
| step 13, after | 179 | 8 (float/Left, orders 256 and 512) | 0.760 |
| step 16, after routing the trailing GEMM | 224 | 1 (float/Right 512/256/128) | 0.995 |

The step-13 residue is **exactly** `q*batch >= 524288`: all 8 losing cells satisfy it and no cell below it loses. That is why step 13's predicate was a *work* threshold and not an order cap — order 512 wins at `q*batch = 32768` (1.23x) while order 256 loses at `q*batch = 524288` (0.90x). Neither side is bandwidth-bound there (11-26% of DRAM peak), so it was re-read amplification escaping L2, not a bandwidth wall. Step 16 deleted the threshold by fixing the cause.

**The cause was not in trsm.** V2 called `sycl_gemm::gemm_custom` — the native kernel entry point — which bypasses `RouteTable<Op::gemm>` entirely, so every trailing update took the native GEMM whether or not it was better. The facade now injects the **routed** gemm through `TrsmTrailingGemm<T>` (`src/sycl/trsm_native.hh:105-111`, wired at `level3.cc:255-266`). At n=512, q=1024, batch=512 the solve goes **18.8 ms -> 11.19 ms** against the vendor's 14.28 ms. Injection rather than an include keeps the kernel TU free of the dispatch layer; an empty callable means `gemm_custom`, so tests and the vendor-free build are unaffected (the vendor-off fallback returns the native GEMM anyway). No per-call cost: cuBLAS GEMM uses `cublasGemmStridedBatchedEx`, so unlike the trsm vendor path there are no pointer arrays to build and no device drain, at 15 GEMM calls per solve.

**And the reason it mattered is the leading dimension.** Every operand trsm hands GEMM is a sub-view carrying its parent's `ld` — a 128-row C with `ld = 512`. The six shapes V2 issues at order 512 (float, q=1024, batch=512):

| shape | native, ld==rows | native, real ld | vendor, real ld | vendor/native |
|---|---|---|---|---|
| outer m=128 n=1024 k=128 | 0.98 ms | 1.53 | 0.96 | **0.62x** |
| outer k=256 | 2.35 | 2.73 | 1.31 | **0.48x** |
| outer k=384 | 3.49 | 3.78 | 1.63 | **0.43x** |
| inner m=32 n=1024 k=32 | 0.248 | 0.406 | 0.235 | **0.58x** |
| inner k=64 | 0.356 | 0.680 | 0.335 | **0.49x** |
| inner k=96 | 0.487 | 0.887 | 0.426 | **0.48x** |

cuBLAS barely moves. **Strided is the only case trsm ever issues, so a square-matrix GEMM benchmark structurally could not have found this.** Caveat: padded operands were allocated uninitialized while unpadded ones used `::Random`; after that was fixed the reference cell moved 0.34%, so the effect is not a data artefact.

### end-to-end-through-ortho

A kernel win is not a library win: a 2.16x kernel win in this repo once turned into an 11% gesvd loss. Both A/Bs run with the route **unset**, so `preferred()` is what selects.

* **`Side::Right` (step 9)** — `ortho` at m in {1024,4096}, k in {16..256}, batch in {128,512}, Chol2 and ShiftChol3: **80 cells, 80 at or above parity, 1.147x-2.719x**, within 4.4% of the forced-native leg. (The route header rounds the top to 2.69x; the committed CSVs give 2.719x.)
* **`Side::Left` (step 12)** — `ortho_benchmark` hardcoded `Transpose::NoTrans` and `ortho.cc:205,289` select the trsm side from exactly that flag, so **the whole `Side::Left` half of the table had never been exercised through a real caller**. `arg4` now selects it. Route unset: 80 cells, best 2.385x, worst **0.986x**, 7 cells fractionally below parity (all >= 0.986). Forced native at order 256 loses 0.783x, and the default correctly tracks the vendor there (4.08 ms default vs 4.07 vendor, 4.35 native) — the predicate declining native is visible end to end. *The notes report "80/80 at or above parity, worst 0.99x"; that is a rounding of 0.986.*

Neither A/B has been re-run since step 12, and steps 13 and 16 changed V2 for every type.

---

## negative-results

### rejected-the-cooperative-cta-solve-v3

Built, measured, **rejected**; kept as `experiments/wp3_s14/v3_cooperative_kernel.patch`, not in the tree, because the device link is this project's long pole. W=8 work-items cooperate on one solve, thread `w` owning canonical rows `{w, w+W, …}` and holding `NL = N/W` accumulators, each `x_s` exchanged by a sub-group shuffle. **The register premise was right**: N=128, W=8, zero frame and zero spill — float **106 registers**, fewer than V1 needs at N=32 (114); double 136, `complex<float>` 139, `complex<double>` 174. The loop order is the whole trick: a runtime `acc[t/W]` forces local memory, a scan costs 2x, and block distribution is 7x load-imbalanced at W=8; cyclic distribution with the local index outermost makes the owner index compile-time and executes only needed FMAs. Coalescing came from the lane map (`w = lane % W`, so eight lanes read eight consecutive rows of one column — one 32 B sector), so V3 needed no staging tile. That same map is **wrong for `Side::Right`**, which wants consecutive columns: V3 was deliberately `Side::Left`-only, which is also the only side with a measured gap. Re-applying the patch also needs `dev_select` and `fma_acc_neg` in `src/sycl/device_scalar.hh`.

| float `Side::Left`, worst cell clean in both runs | 8 | 16 | 32 | **64** | **128** | 256 | 512 |
|---|---|---|---|---|---|---|---|
| step 13 (V1 + two-level blocking) | 1.59 | 1.72 | 1.77 | **1.48** | **1.18** | 0.86 | 0.76 |
| step 14 (V3 cooperative) | 1.59 | 1.72 | 1.80 | **0.39** | **0.80** | 0.84 | 0.77 |

`Side::Right` is unaffected (1.61->1.62, 3.36->3.44, 1.55->1.55, 1.23->1.24), confirming the side gating held. The decisive cell is order 128, where V3 fits exactly with zero padding waste and still goes 1.18x -> 0.80x: the kernel is intrinsically ~1.5x slower at equal order. Order 64, padded to 128 (4x the arithmetic), collapses to 0.39x. At 256 and 512, where V3 removes the entire inner blocking level and the traffic model predicts 4096 -> 2560 q-units (1.6x), the measurement moves by **0.02x**. **The traffic model counts bytes and does not count the critical path**: V3's recurrence is N dependent shuffle-scale-FMA steps, while V1 fills 32 steps with independent FMAs and lets well-tuned parallel GEMMs carry the rest. Trading parallel GEMM work for serial in-kernel recurrence loses even when it removes DRAM traffic — the same shape as the earlier `cta-large-n` rejection (85-211x slower).

The comparison is restricted to the **81 cells clean (relative sd <= 10%) in both runs**, so it is like-for-like. The per-cell CSVs for this run **did not survive** (deleted after analysis, before aggregation; the summary was never written because a rebuild interrupted the run). The table above is the record, not a derivation from committed data.

### rejected-n-64-cta-bucket

Re-tested rather than assumed, because the staging tile had cut float `Side::Left` from 114 registers to 53 and the arithmetic that killed N=64 no longer described the kernel. **It still fails, and by more**: float N=64 Left 72 registers / **456 B frame**, N=64 Right 119 registers / **256 B frame**, zero spill in both. Left is worse than Right because the tile's own live state competes with the accumulator rather than paying for it. There is deliberately no N=64 bucket, and `smallest_bucket_ge` returns **0** above 32 rather than the next power of two.

### rejected-outer_nb-128-for-side-right

See the sweep in `### the-two-level-blocked-driver`: `OUTER_NB = 128` on `Side::Right` regresses orders 256 and 512 from 1.01 and 1.07 to 0.83 and 0.82. Right keeps the single-level schedule.

### rejected-the-starvation-guard

Spec §10's `batch*q < 8*CU*32 -> vendor` is **refuted by measurement, not merely unimplementable**. At batch=8, q=32 the product is 256 against its own threshold of 32,768, and native wins those cells 2.2-2.4x — the guard would have handed back every one. It is *also* unimplementable as written: `OpShape::compute_units` (`route.hh:240`) has zero writers and zero readers and reads 0. It dies on the measurement first. The kill criterion stated in advance — *"if native real trsm exceeds 1.10x vendor at the saturated ortho shape, real stays vendor-first and only complex flips"* — **did not fire**: double won every cell on both sides and float won every `Side::Right` cell.

### rejected-diagonal-block-inversion

See `## design-v1-v2-and-the-canonical-fold`. Rejected on the accuracy argument before anything was built.

### refuted-mechanisms

Two mechanisms this work package published were later **measured wrong**.

**1. Step 16 blamed `src/sycl/gemm/register_tiled_common.hh`** for the strided-`ld` collapse — odd tile strides `TileM+1`/`TileK+1` defeating 16-byte alignment, B staged `[n][k]`, a read-modify-write epilogue writing columns 4096 B apart, and `is_contiguous_dense_matrix`, which every sub-view fails. **Those shapes never execute that file.** `select_kernel_variant` (`gemm_kernels.cc:519-521`) sends the outer shapes to `Tiled128x128RegisterK8`, and the dispatcher re-evaluates the same predicate to pick `AlignedFastPath = true` in *both* columns of the table above (`:878-886`; the in-file comment at `:533-536` still cites the pre-drift `:737-741`). `can_use_128x128_fast_path` (`src/sycl/gemm/register_128x128.hh:71-91`) never tests contiguity — only `m%128`, `n%128`, `k%8`, a 16-byte-aligned base, `ld%4==0` and `stride%4==0`, all of which a strided sub-view satisfies. ncu: every transaction counter is byte-identical between packed and strided (16.00 load sectors/request, identical DRAM sectors, identical instructions, 119 registers, zero spill; `dram__cycles_active` differs by 0.3%). The loss is **entirely exposed global-load latency** — barrier stall 1.552 -> 7.703, long_scoreboard 8.755 -> 11.740 — it belongs to operand **B alone** (padding A costs 1.003x, C 1.056x, B **1.552x**), it is a slope monotonic in stride from 512 B to 4096 B rather than a cliff, and it is beta-independent (+0.564 ms at beta=1, +0.603 ms at beta=0), which refutes the epilogue story directly. Two fixes were **built and measured dead**: double-buffering the k-loop (127 registers, occupancy preserved, barriers halved, incidentally fixing a split-`LDG` defect — 33.55M -> 25.17M global load sectors, exactly cuBLAS's count — recovering **no time at all**, 1.564 ms against a 1.547 baseline); and packing B into contiguous scratch (the kernel is at 89% of roofline when packed, so the pack is paid at that same roofline; it loses everywhere and harder as m grows). What worked was **routing**: `can_use_128x128_fast_path` is a *leg* predicate the dispatcher evaluates again, so using it as a *routing* gate did not demote the call to the predicated leg — it handed it to an entirely different, much slower kernel. The shipped fix is the shape-only gate at `gemm_kernels.cc:577-580`, which does not consult the alignment predicate at all (and so also subsumes the `ld%4 != 0` cliff). Routing by what the kernel can run is worth geomean **1.74x/1.75x** (native 0.58x -> 0.99x of cuBLAS packed, 0.54x -> 0.93x strided) — *unverified: the 12-shape subset those geomeans average over is named in `gemm_kernels.cc:544-549` and in `docs/perf/gemm.md` but is not identified in the preserved `experiments/wp4_gemm_ld/routing/` data, so the figure cannot be recomputed* and with cuBLAS present **changes no runtime at all**: `route_gemm.hh`'s float NN window requires `m==n==k`, so 79 native float gemm calls against 102,791 vendor. A vendor-free and ROCm win.

**2. Step 14 concluded the inner blocking level did not matter**, because replacing it wholesale changed nothing. Wrong, and masked: the cooperative solve was slow in its own way, so removing the inner level and adding a slower diagonal solve cancelled out. nsys puts the inner GEMMs at **7.83 ms, 42% of the solve for 20% of the flops**.

**The general lesson, twice in one work package: confirm which kernel runs before theorising about why it is slow.**

---

## correctness-findings

### the-missing-group-barrier

**WP3's trsm returned wrong answers**, found during WP4 Phase 2 while looking for something else. V1 stages the canonical triangle into SLM with a loop strided by `lane`, so element `idx` is written by lane `idx % wg`; the loop immediately after has lane `s` read `sLc[tri_idx(s,s)]` — a *different lane's write* for nearly every `s` — with nothing in between. `sDiv[0]` had the same problem: lane 0 zeroes it before the staging loop and any lane may store 1 into it after. **One `sycl::group_barrier` is the entire fix** (`src/sycl/trsm_native.cc:412`).

A/B, barrier deleted and the shared library rebuilt (the `.so` relink *is* the AOT device compile, so this is a real rebuild):

| | max relative diff vs vendor | items wrong | native residual |
|---|---|---|---|
| deleted | **6.05e+16** | 127 / 128 | 8.0e+05 |
| restored | 4.27e-07 | 0 / 128 | 2.38e-07 (= vendor) |

Vendor-free potrf, n=1024 batch=256, float and double: before the barrier, 61-75 of 256 items came back `info != 0` non-deterministically, the failing column always `== 1 (mod nb)` — the first column of a panel, i.e. a diagonal block the previous panel's bad L21 had already destroyed. After, 0/256 over every rep at batch up to 1024.

**How it hid.** The ladder picks the first `cand` in `{256,128,64,32}` with `bs*ceil(q/cand) >= 4*CU` (512 on this box). Every trsm test uses `bs <= 3, q <= 257`, so every one lands on **wg = 32** — a single sub-group, executing the two loops in lock step, the one width where the race cannot express itself. An unsynchronised cross-sub-group read-after-write in SLM is UB that NVIDIA hardware hides at exactly the width every small-batch test picks. The blocked potrf panel solve does not sit there: n=1024, batch=256 gives q=896 and wg=256, eight sub-groups, and the race fires.

### the-bucket-ladder-that-truncated

`smallest_bucket_ge` used to return 64 for any `n > 32`, and the dispatch switch's `default:` label collapsed 64 onto the N=32 instantiation — so a **33-order solve silently solved the leading 32x32 system and left the last row of B untouched**. Nothing caught it: the staging pad test (`s >= n`) cannot fire when `N < n`, the recurrence simply stops early, and the store loop writes only the rows it computed. It was unreachable through the facade because `supports(CTA)` caps the order, but the direct entry is exactly what V2 calls on its diagonal blocks. It now returns 0 and V1 **throws** rather than truncating (`TrsmNativeCta.OverCapacityThrowsRatherThanTruncating`).

### blind-and-vacuous-guards

* **The regression test shipped with the barrier fix was itself vacuous.** It called V1 directly at n=16, q=1024, bs=128, cleared the work-group ladder, *asserted* that it had — and still passed **green, twice**, with the barrier deleted and the library rebuilt. Clearing the ladder is necessary and **not sufficient**. The reproducing configuration goes through V2: order **48** (so the final V1 block is order 16), **q=976, batch=128**. Orders whose final V1 block lands in the N=16 bucket (48, 77, 80, 109) failed 90-128 of 128 items deterministically while 32, 33, 64, 65, 96 and 155 were clean, so an order dividing evenly would have been another silent pass. That is `TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly` (`tests/trsm_tests.cc:630`), which asserts the rung *and* drives the reproducing shape, and was verified red with the barrier deleted. Fifth recorded blind guard in this repository, and the first written in the same change as the fix it guards.
* **Every blocked test stopped at order 100.** With `OUTER_NB = 128` that is a *single* panel, so all of them took `LO == 0` and the outer level never ran — they passed unchanged against the two-level driver while proving nothing about it. `TwoLevelPanelStructure` now uses 129 (two panels, second one element wide), 256 (two full), 300 (two plus a ragged 44) and 384 (three full).
* **Alpha is applied exactly once, and there are two distinct ways to get it wrong.** For blocks `i > 0` alpha arrives through the trailing GEMM's **beta**, not through V1; the natural `beta = 1` computes `B_i - sum` where `alpha*B_i - sum` is required — correct at block 0, wrong at every later block, and invisible to any `alpha == 1` test. With two levels a block in panel `p > 0` is touched by the outer gemm, an inner gemm, and the solve: three chances, exactly one right. Both levels have their own test; mutation-tested, outer-beta breaks 4 tests and inner-beta 9.
* **Complex is where `ConjTrans` and the complex reciprocal first become visible.** For a real scalar `ConjTrans` is identical to `Trans`, so every real cell is blind to it, and `Canonical::do_conj` was written by `canonicalise()` and read by nothing until complex arrived. The test *data* is what makes it visible: `tri_fill` gives every element a non-zero imaginary part that is a different function of `(r,c)` than the real part, so the triangle is neither real, nor symmetric, nor Hermitian. The reciprocal is Smith's overflow-safe form — the textbook `conj(d)/|d|^2` silently returns 0 for inputs whose true reciprocal is representable.
* **The oracle is an independent multiply-back, not the in-tree reference.** `netlib_lapack.cc:470-473` and `cublas.cc:1134-1137` fold the 24 canonical cases identically, i.e. they are *one* implementation; checking against either would validate the fold against itself.
* **The documented test command runs zero tests and exits 0.** `ctest -L blas -L ortho` — repeated `-L` is an AND and no test carries two component labels: **Total Tests: 0, exit 0**, a silent false green under a section whose entire correctness argument rests on those targets running. One `-L` with a backslash-escaped pipe is *also* 0. The working form is one `-L` with a bare quoted pipe: `ctest -L 'blas|ortho'` -> 20 tests (15 + 5).
* **`BATCHLAS_TRSM_OUTER_NB` is tested for correctness under whatever value is live**, not for a particular schedule, because `trsm_outer_block` caches the parse in a function-local static and the first blocked call in the process fixes it — a schedule assertion would pass or fail on gtest's ordering, not on the code.

Suite state at the end of WP3: `trsm_tests` 91/91 vendor-present; vendor-free 59 passing with the failing set byte-identical to the WP2 baseline (the remainder are the NETLIB parameterisations — the CTA kernel is a GPU kernel and `supports()` correctly reports `is_gpu == false` as unsupported).

---

## what-the-spec-got-wrong

`WP3_TRSM_SPEC.md` was written against a pre-WP1/WP2 tree; `WP3_TRSM_SPEC_CORRECTIONS.md` records 27 findings that survived adversarial refutation. Beyond the items already covered (the SLM size overrun, the `n_cta` derivation, the starvation guard, the ctest command):

* **The three routing hook points no longer exist.** The spec routes at `cublas.cc:1594`, `rocblas.cc:138`, `netlib_lapack.cc:404`. WP1 left exactly one public `trsm`, the facade; the backends own `trsm_vendor` only. One hook, in the facade, **before** the vendor-available test — anything after it is unreachable in the vendor-free build WP3 exists for. That hoist also fixed netlib's long-missing `trsm_validate_params` for every backend in one edit.
* **`parse_cublasdx_variant_request` and `TrsmVariant` were deleted by WP0.** `src/backends/route_common.hh:35-41` is their tombstone.
* **A single `trsm_use_native()` bool cannot express what the vendor-free build needs.** Mixing env read, correctness and speed into one predicate means every real-type cell and everything below the starvation cut has *no route at all* without a vendor, and the facade throws. Pinned by `RouteTrsm.SupportedButNotPreferredIsTheWholePoint` (`tests/route_vocabulary_tests.cc:407`), which names a supported-and-un-preferred cell and asserts it still routes natively when `vendor_available == false`.
* **`TriangularTransform` is in `batchlas::device`, not `batchlas::device::detail`** — a compile error if the spec's §6.1 citation is transcribed.
* **The link-time budget fires with zero trsm code** and names a target with no link step: `batchlas_sycl_obj` is an OBJECT library; the link unit is the shared library, and it measured **43.9 s** before WP3 added anything. Make any such budget a delta.

---

## open-debts

1. **The entire WP3 performance grid was measured on a kernel that could return wrong answers.** The barrier landed in WP4 Phase 2, after step 16. The recorded caveat is that *"`preferred()` windows above `q*batch ~ 65k` were measured on a racing kernel and have not been re-run"*. It is likely worse: the smallest cell in `TrsmOrthoSizes` is q=256, batch=128, i.e. `q*batch = 32768`, and the ladder rule (`bs*ceil(q/cand) >= 4*CU`, `4*CU = 512`) selects **wg = 64** there — two sub-groups. By that rule **no cell in the shipped grid ran at the single-sub-group width where the race provably cannot fire.** That is an inference from the ladder arithmetic, not a re-run. Nothing on this page except `### the-batch-floor` has been re-measured post-fix.
2. **`double` and `complex<double>` were last measured at step 9** — orders 8..256 only, before the `Side::Left` staging tile, before two-level blocking, and before the routed trailing GEMM, all three of which change V2 for every type. `preferred()` returns unconditional `true` for both, at every order, on that evidence. Steps 13, 14 and 16 measured **float and `complex<float>` only**.
3. **No order above 512 has ever been measured, for any type**, and there is no upper bound in the predicate. The step-9 grid stopped at 256 because of the harness's 6 GB per-cell cap (not because anything changes in kind); steps 13-16 reached 512 for float and `complex<float>` only.
4. **float at batch in `[8,127]` is barely measured at all above order 32.** For `Side::Right`, order 64 was never run at any batch below 128, so the `order <= 32` cut is an interpolation between a measured win at 32 and a measured loss at 128. For `Side::Left` the predicate has no batch-dependent clause and prefers native throughout, while the only data in that band is the starved profile — which puts order 128 at **0.756-0.780x** (and forbids ranking from itself). The saturated grid starts at batch 128.
5. **The batch floor of 8 rests on a profile its own script forbids ranking from**, and is type-blind: `double` at order 8 and both complex types measurably win at batch=1 and go to the vendor anyway.
6. **Every complex "vendor" ratio on this page is native against another BatchLAS kernel.** `src/backends/cublas.cc:1111-1214` diverts *both* complex types to a hand-written sequential per-RHS SYCL substitution; `cublasCtrsmBatched`/`cublasZtrsmBatched` at `:1220` are unreachable. The diversion rests on an uncited comment about NaNs under SYCL/USM interop. This is why complex shows 8-52x at large order. **It is not a vendor-independence result**, and settling whether the diversion is still warranted has been open since step 13.
7. **`complex<float>`/`Side::Right` at orders 8-16 (1.01-1.05x) are roofline ties, not defects** — native runs at **88.5-90.5% of the 1008 GB/s DRAM peak** and the ceiling is 1.12-1.18x. The same orders at small `q*batch` fit in L2, are not DRAM-bound, and win 1.25-2.04x.
8. **The end-to-end `ortho` A/B has not been re-run since step 12**, i.e. before two-level blocking and before the routed trailing GEMM.
9. **The step-16 cell count does not reconcile.** Code and plan say "167 of 168"; the committed `baseline.csv` holds 224 clean pairs with 1 loser. The conclusion is unchanged; the published count is not reproducible from committed data.
10. **`experiments/wp3_s14`'s per-cell CSVs were deleted before aggregation.** The V3 rejection table is a written record, not a derivation from data.
11. **The residual native-GEMM strided-`ld` slope is unexplained.** Routing recovered 1.74x/1.75x geomean and got native to 0.93x of cuBLAS strided; the rest is exposed load latency on operand B, monotonic in stride, with two candidate fixes measured dead. At 0.93x a `preferred()` flip is arguable, not winning.
12. **trsm's `heterogeneous_batch` correctness gate can never fire.** `supports()` rejects a heterogeneous batch at `route_trsm.hh:151` — correctly, since one launch covers the whole batch with a single `(order, q, ld, stride)` tuple — but `trsm_op_shape` (`src/backends/trsm_route.hh:40-56`) never writes the field, so it keeps `OpShape`'s default `false` (`route.hh:236`). `MatrixView::is_heterogeneous()` exists (`matrix.hh:1034`) and the `getrf`/`getrs`/`geqrf` shape builders call it; trsm's does not. The gate is therefore a documented intention, not an enforced one. *(Code observation, not something the WP3 notes record; no measurement either way.)*
13. **`OpShape::compute_units` is still dead** (declared, zero writers, zero readers, reads 0). Any future occupancy clause needs the *shape builder* to populate it; the table must stay pure.
14. **`MatrixView::operator()(Slice,Slice)` passing the parent pointer array** (`matrix.hh:1140`) — reported, deliberately untouched since step 13. V2 works around it by passing the parent's `ld` **and** `stride` explicitly at every sub-view construction, because the constructor defaults `stride` to `ld*cols` when 0 is passed and every batch item after the first would otherwise read the wrong matrix.
15. **WP3 makes no extension vendor-free.** `ortho_tests` is blocked by `potrf`, `geqrf`, `orgqr` and `syev`; `cond_tests` and `inverse_tests` by `syev`, `getrf` and `getri`. The honest claim is that WP3 removes `trsm` from the vendor-dependency list. There is no CPU trsm.
16. **`preferred()` moves almost nothing in the test suite.** Every trsm call the suite issues runs at batch <= 5, below the floor of 8, so the route diffs across steps 9, 12 and 13 show zero moved library decisions — the only changed rows are `route_vocabulary_tests` recording its own `resolve_trsm_route` calls. Any future flip must be validated by an A/B through a real caller.

---

## raw-evidence

Raw data is preserved at the git tag `perf-evidence/vendor-independence`, retrievable with `git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| step-9 routing grid, all four types, both sides | `experiments/wp3_s9/{left,right}-{vendor,native}.csv`, `sweep.sh`, `analyse.py`, `README.md` |
| the batch floor (profile only, never ranked) | `experiments/wp3_s9/starved-*.csv`, `starved.sh` |
| end-to-end `ortho` A/B, `Side::Right`, route unset | `experiments/wp3_s9/ortho-{vendor,native,default}.csv`, `ortho_ab.sh` |
| `Side::Left` staging tile: ncu profile, before/after grid, register exclusion | `experiments/wp3_s12/README.md`, `left-{vendor,native}.csv`, `profile.sh`, `left_sweep.sh` |
| end-to-end `ortho` A/B, `Side::Left` | `experiments/wp3_s12/ortho-left-{vendor,native,default}.csv`, `ortho_left_ab.sh` |
| two-level blocking, the `OUTER_NB` sweep, orders to 512 | `experiments/wp3_s13/{baseline-before,baseline,nb_sweep}.csv`, `README.md` |
| the rejected cooperative CTA solve (V3) | `experiments/wp3_s14/README.md`, `v3_cooperative_kernel.patch`, `measure.py` |
| routed trailing GEMM; isolated GEMM shapes at `ld==rows` vs the real `ld` | `experiments/wp3_s16/baseline.csv`, `{outer,inner}{,pad}-{vendor,native}.csv`, `trailing_shapes.sh`, `outer_ld.sh`, `recheck-*` |
| the strided-`ld` re-diagnosis and the routing fix | `experiments/wp4_gemm_ld/` |
| the GPU exclusivity guard used by every sweep | `experiments/gpu_guard.sh` |
| verification pass behind `WP3_TRSM_SPEC_CORRECTIONS.md` | `experiments/wp3/verification_pass_raw.md` |

Sweep CSVs are `name,arg0,arg1,arg2,iterations,avg_ms,stddev_ms,GFLOPS,Time (us) / matrix` with `arg0 = n` (triangular order), `arg1 = q`, `arg2 = batch`. **GFLOPS uses the real-arithmetic convention `n^2 q` for all four types**, so complex understates by 4x by construction — compare `avg_ms` across types, never GFLOPS. The step-13/14/16 `baseline.csv` files are aggregated: `type,side,route,n,q,batch,ms,sd_pct`, one row per leg, `sd_pct <= 10` the cleanliness gate.

Two measurement traps that cost real time. `gpu_guard.sh` samples *foreign* processes at the start and end of a run, so **two copies of your own sweep script** queued behind a co-tenant are invisible to it — that produced 22 of 180 cells at 10-103% relative sd while the guard reported the run clean, and the drivers now take a `flock` and delete any leg with a cell above 10% sd (a clean leg has 0 of 180). And a contaminated profile once nearly added a bogus gate: the first post-tile profile showed order 8 regressing 0.023 -> 0.028 ms and a staging gate on order was about to be written; on the clean grid order 8 is **1.01-1.04x** and there was nothing to protect.

Three more that are designed *around* rather than fixed, all visible in the harness:

* **`--name` matches by substring.** The starved rows are registered as `BM_TRSM_StarvedRight`, not the obvious `BM_TRSM_OrthoRightStarved`, because the latter would be selected by every `--name=BM_TRSM_OrthoRight` run and fold profile-only rows into the saturated grid they exist to be excluded from (`benchmarks/trsm_benchmark.cc:236-244`).
* **An unparsed route variable measures the default twice.** `trsm_announce_route_env()` (`:164-180`) prints the parse of `BATCHLAS_TRSM_ROUTE` once per process and shouts when it was not understood, because a typo otherwise runs both legs on the same route and reports a ratio of 1.0 as a finding — the precedent is `BATCHLAS_SYEV_PROVIDER=TWOSTAGE` silently parsing as `Auto`.
* **`ncu --kernel-name` matches the mangled name.** `regex:TrsmCtaKernel` matches nothing and ncu then profiles the run without emitting a single metric row, which looks exactly like a kernel that does no memory traffic; `--kernel-name-base demangled` is what makes the readable name the thing being matched (`experiments/wp3_s12/profile.sh`).

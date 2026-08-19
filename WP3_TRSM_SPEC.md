# FINAL SPECIFICATION — Native batched TRSM for BatchLAS

>  **READ `WP3_TRSM_SPEC_CORRECTIONS.md` FIRST.** This spec was written at `aa827f5`, which
>  predates WP1 and WP2. Its three routing hook points no longer exist, its `TrsmVariant` enum is
>  the vocabulary WP0 deleted, its SLM size formula writes 127 elements out of bounds, its
>  `-Xcuda-ptxas -v` gate is not executable per-TU, and its documented `ctest` command runs zero
>  tests while exiting 0. The corrections document lists what survives and what to do instead.

All paths relative to `/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/`. Every line citation below was re-read from source at `aa827f5`. Nothing was edited.

Notation used throughout: **`n`** is the triangular order (`A.rows() == A.cols()`); **`q`** is the independent extent — `q = B.cols()` for `Side::Left`, `q = B.rows()` for `Side::Right`. `q` is the number of *independent solves*.

---

## 1. Decision, in five sentences

The two designs compose and both ship: a CTA-resident kernel **V1** that gives one thread one complete independent solve with the solution vector in registers and the triangle broadcast from SLM, and a host-blocked driver **V2** for `n > n_cta(T)` that calls V1 as its diagonal-block solver and `sycl_gemm::gemm_custom` for the trailing update — V1 is literally V2's panel factorisation, so the crossover is a single number `n_cta(T)` fixed by register residency, not a tuned guess. From design "trsm-substitution" I take the 24-case canonicalisation table (verified correct against both in-tree references by all three critiques) and the packed-triangle SLM staging; from design "trsm-inverse-gemm" I take the one-thread-per-solve register decomposition and the blocked driver, corrected for the `Side::Right` GEMM operand order that was dimensionally impossible as written. I reject design 1's LDS-resident RHS panel (it costs 41.6 KB/CTA at `n=64` float for 17% occupancy and introduces a diagonal-phase serial bubble that the register form does not have), I reject design 2's V2 "panel-blocked" tier outright (its `nb = 8·T_c` geometry splits the dimension its own recurrence is serial in, giving `≥ n` barriers where it budgeted `n/nb`), and I reject **diagonal-block inversion at every tier** — the "free for ortho" licence claimed for it does not survive being restated in orthogonality currency (§7.5). Nothing is guessed that can be measured: `n_cta(T)` is derived from the 256 B/thread register rule and must be *confirmed* with `-Xcuda-ptxas -v` before any other code is written, and the routing predicate ships with real types vendor-first until the grid in §10 says otherwise. The complex cells are the only ones claimed with confidence, because their incumbent is not a vendor kernel at all but `src/backends/cublas.cc:1122-1225`, a fully serial per-column substitution in which every work-item re-reads the whole triangle from global memory.

---

## 2. The algorithm and blocking, mathematically

### 2.1 Canonicalisation — 24 cases folded into one recurrence

From `(side, uplo, transA, diag)` define, exactly as `src/backends/cublas.cc:1145-1148` and `src/backends/netlib_lapack.cc:418-421` do:

```
do_trans    = (transA != Transpose::NoTrans)
do_conj     = (transA == Transpose::ConjTrans)          // and T complex
op_is_lower = (uplo == Uplo::Lower) ? !do_trans : do_trans
unit        = (diag == Diag::Unit)
opA(r,c)    = do_trans ? conj_if(A(c,r)) : A(r,c)
n           = A.rows()
q           = (side == Side::Left) ? B.cols() : B.rows()
```

Define the **canonical direction** and index map

```
fwd  = (side == Side::Left) ? op_is_lower : !op_is_lower
ρ(s) = fwd ? s : (n - 1 - s)                            // s ∈ [0, n)
```

and the **canonical unit-lower factor** `Lc(s,t)`, `0 ≤ t ≤ s < n`:

| side | `Lc(s,t)` |
|---|---|
| Left | `opA(ρ(s), ρ(t))` |
| Right | `opA(ρ(t), ρ(s))` — **operand order swapped** |

and the **canonical RHS accessor**, affine in both indices:

| side | `B̃(s,u)` | `b0` | `ds` | `du` |
|---|---|---|---|---|
| Left | `B(ρ(s), u)` | `fwd ? 0 : (n-1)` | `fwd ? +1 : -1` | `ldb` |
| Right | `B(u, ρ(s))` | `fwd ? 0 : (n-1)·ldb` | `fwd ? +ldb : -ldb` | `1` |

so `&B̃(s,u) = B.data_ptr() + b·B.stride() + b0 + s·ds + u·du`.

The recurrence is then unconditionally, for each `u ∈ [0,q)` **independently**:

```
for s = 0 .. n-1:
    acc  = Σ_{t<s} Lc(s,t) · x[t]            // fresh accumulator, ascending t
    x[s] = alpha · B̃(s,u) - acc
    if (!unit) x[s] = x[s] · rd[s]           // rd[s] = 1/Lc(s,s), guarded — §7
store B̃(s,u) := x[s]
```

**This is operation-for-operation the reference loop nest.** `acc` is a separate accumulator summed in ascending `t`, then subtracted from `alpha·b` — identical to `sum` / `x = alpha*B[i] - sum` at `cublas.cc:1170-1178` and `netlib_lapack.cc:452-459`. The *only* arithmetic deviation is `· rd[s]` vs `/ Lc(s,s)`, and §7 specifies a guard that reverts to division. This is deliberate: it makes the accuracy prediction in §9 sharp instead of a false-alarm detector, which is the defect critique 1 (design 1) and critique 3 (design 1) both raised.

`Lc` is never read from A directly. It is produced by the existing runtime-transform helper

```cpp
batchlas::device::detail::triangular_matrix_entry<T>(a_kv, row, col,
    TriangularTransform{side, uplo, transA, diag})
```
(`include/batchlas/blas/device/detail/group_blas_common.hh:705-728`, verified), called with `(row,col) = (ρ(s),ρ(t))` for Left and `(ρ(t),ρ(s))` for Right. That helper returns `T(1)` at `src_row == src_col` under `Diag::Unit` **before any load** (`:716-718`), returns `T(0)` outside the referenced triangle **before any load** (`:721-723`), and applies `conj` for `ConjTrans` (`:726`). Consequence: the unreferenced triangle is provably never read, which is the production precondition in `ortho` — the upper half of `C` is uninitialised workspace (`src/extensions/ortho.cc:156-161`).

### 2.2 V1 — CTA-resident, register-resident, `n ≤ n_cta(T)`

One thread owns one `u`. `x[0..N-1]` lives in that thread's registers, where `N` is a **compile-time** bucket `≥ n`; the loop over `s` and the inner loop over `t` are fully unrolled so every register index is a compile-time constant. Rows `n..N-1` are zero-padded during staging (`Lc(s,t) = 0`, `Lc(s,s) = 1`, `rd[s] = 1`, `B̃` not touched), the `sytrd_cta` idiom at `src/extensions/sytrd_cta.cc:123-137`, so the unrolled tail computes zeros instead of branching.

**This compile-time-`N` requirement is not optional.** A per-thread array indexed by a runtime induction variable is placed in `.local` by ptxas, not registers; that turns a DRAM-bound kernel into an L1-bound one and voids the entire design. Critique 2 (design 2) is correct and this is the fix.

SLM per work-group holds only:
* `Lc` packed row-major by `s`: `idx(s,t) = s(s+1)/2 + t`, `N(N+1)/2` elements. All threads read the same `Lc(s,t)` at the same step → SLM **broadcast**, no bank conflicts, layout irrelevant to conflicts but sequential in `t` for locality.
* `rd[0..N-1]`: `N` elements.
* `Side::Left` only: a transpose staging tile, `NB_stage × WG` elements, `NB_stage = min(N, 16)` (§3.4).

### 2.3 V2 — host-blocked driver, `n > n_cta(T)`

Canonical block `i` covers `s ∈ [i·nb, min(n,(i+1)·nb))`. In **stored** indices that is a contiguous range

```
R_i = fwd ? [i·nb, min(n,(i+1)·nb))
          : [n - min(n,(i+1)·nb), n - i·nb)
```
and the already-solved set is also contiguous:
```
S_i = fwd ? [0, i·nb) : [n - i·nb, n)
```

For `i = 0 .. ⌈n/nb⌉-1`:

**Side::Left** (`X`, `B` are `n × q`):
```
i>0:  B[R_i, :] := alpha·B[R_i, :] - op(A)[R_i, S_i] · X[S_i, :]
      gemm_custom(ctx, A_sub, X_sub, B_sub, T(-1), alpha, transA, Transpose::NoTrans, prec)
        A_sub = do_trans ? A(S_i, R_i) : A(R_i, S_i)
        X_sub = B(S_i, all)          B_sub = B(R_i, all)
then: V1 on op(A)[R_i,R_i] against B[R_i,:], with alpha_eff = (i==0 ? alpha : T(1))
```

**Side::Right** (`X`, `B` are `q × n`):
```
i>0:  B[:, R_i] := alpha·B[:, R_i] - X[:, S_i] · op(A)[S_i, R_i]
      gemm_custom(ctx, X_sub, A_sub, B_sub, T(-1), alpha, Transpose::NoTrans, transA, prec)
        X_sub = B(all, S_i)
        A_sub = do_trans ? A(R_i, S_i) : A(S_i, R_i)
        B_sub = B(all, R_i)
then: V1 on X[:,R_i]·op(A)[R_i,R_i] = B[:,R_i], alpha_eff = (i==0 ? alpha : T(1))
```

`sycl_gemm::gemm_custom` is `(ctx, A, B, C, alpha, beta, transA, transB, precision)` computing `C := alpha·op(A)·op(B) + beta·C` (`src/sycl/gemm_kernels.hh:61-70`, verified). **The `Side::Right` form must put `X_sub` in the `A` position.** Design 2's single form `gemm_custom(ctx, A_blk, X_prev, B_i, ...)` produces a `C` of at most `nb` rows against a required `q` rows and does not conform for any `transA_eff`; critique 3 (design 2) is correct and this is the fix.

`C = B(·, R_i)` and the `X` operand `B(·, S_i)` occupy **disjoint** index ranges of the same buffer, so `gemm_custom` reads no element it writes.

**Slices must be built by explicit constructor, not `operator()(Slice,Slice)`.** `MatrixView::operator()(Slice,Slice)` at `include/batchlas/blas/matrix.hh:1129-1141` carries a comment saying it does not propagate the parent pointer array but then passes `data_ptrs_.data()` anyway; `init_data_ptr_array` (`src/matrix.cc:2364-2382`) would regenerate *into the parent's storage*, silently corrupting it — and in `ortho` the Gram matrix `C` is constructed **with** a pointer array (`ortho.cc:71-76`), so this is live. Build every sub-view as `MatrixView<T,Dense>(base + offset, rows, cols, ld, stride, batch)` (6-arg, `data_ptrs = nullptr`). `gemm_custom` uses `data_ptr()+stride` and never touches the pointer array, so this is sufficient.

`beta = alpha ≠ 0` on every one of those GEMMs. The repo has a recorded case where a `beta ≠ 0` epilogue cost 15 TFLOP/s and the standalone GEMM harness (which defaults `beta = 0`) could not see it. Benchmark V2 at `beta ≠ 0` from the first measurement.

### 2.4 Explicitly rejected: diagonal-block inversion, at every tier

Reasons, in order of force:
1. **The "free for ortho" argument does not hold.** Design 2 §7.3 compared `u·κ(A)` (a trsm *residual*) against `u·κ(A)²` (CholQR *orthogonality*). Pushed through the consumer — `X̂(L̂+ΔL)^H = A` gives `‖E‖ ≤ (‖ΔL‖/‖L̂‖)·κ(A)` — the inverted variant contributes `c·u·κ(A)²` to `‖X̂^H X̂ − I‖`, the *same order* as the existing term, with an unbounded constant `c`. Since `potrf` is allowed to succeed to `κ(A) ≈ u^{-1/2}` where `u·κ(A)² ≈ 1`, any `c > ~2` flips Chol2 from recovering to not. Critique 1 (design 2) is correct.
2. **There is nothing to buy.** In V1 the diagonal apply is `n` multiplies against `n²/2` FMAs; in V2 it is `nb/n ≤ 1/2` of the flops and is done by V1, which is register-resident. Inversion changes ≤ 6% of the work at `n=1024, nb=64`.
3. It degrades the guarantee from componentwise to normwise with a `max_i κ(U_ii)` constant, and `tests/trsm_tests.cc:74-105` is a residual test — the exact quantity that degrades.

**Not rejected because of `cond_tests`.** Both designs cited `tests/cond_tests.cc:371-380` as proof that ortho depends on a NaN-propagation pattern. Read at source, `:366-387` is a post-mortem on a bug that was **fixed** ("The default is now Householder, which is unconditionally backward stable"), heading `CondTest.RandomMatrixGeneratorIsAlwaysFinite`, which asserts the *absence* of non-finite output. `grep -n "Chol2\|OrthoAlgorithm" tests/cond_tests.cc` returns exactly one hit — line 370, inside that comment. That citation is withdrawn from this spec entirely; see §5.6 for the argument that actually supports the no-`info` contract.

---

## 3. Parallel decomposition and the exact nd_range

### 3.1 Ownership

| level | owns |
|---|---|
| work-group | one batch item × `WG` consecutive independent solves. Stages `Lc` (packed, canonicalised) and `rd` into SLM once, then never touches A again. |
| sub-group | nothing special. No `sg_compat` partitions. |
| thread | **one complete independent solve**: `x[0..N-1]` in registers, the whole serial `s`-recurrence, zero cross-thread dependence, **zero barriers after the staging barrier** (Side::Right) |

The parallel axis is `batch × q`. The serial axis is `s`, a true data dependency.

### 3.2 The exact launch

```cpp
// Compile-time bucket
//   float          N ∈ {8,16,32,64}      n_cta = 64
//   double         N ∈ {8,16,32}         n_cta = 32
//   complex<float> N ∈ {8,16,32}         n_cta = 32
//   complex<double>N ∈ {8,16}            n_cta = 16
const int N  = smallest_bucket_ge<T>(n);            // n <= n_cta(T) guaranteed by the router
const int q  = (side == Side::Left) ? B.cols() : B.rows();
const int bs = A.batch_size();

const auto dev  = ctx.device();
const int  CU   = int(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));   // 128 here
const int  MAXWG= int(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
const size_t LMEM = ctx->get_device().get_info<sycl::info::device::local_mem_size>();
// Reserve as cmake/BatchLASDetectSYCL.cmake:57-67 does; build/include/batchlas/device_limits.hh:28
// records 45056 for this GPU. Never budget against the raw 49152.
const size_t E_BYTES = std::max<size_t>(16384, LMEM - 4096);

const int NB_STAGE = (side == Side::Left) ? std::min(N, 16) : 0;

auto lds_bytes = [&](int wg) {
    return (size_t(N)*(N+1)/2 + size_t(N) + size_t(NB_STAGE)*wg) * sizeof(T);
};

// Largest WG in {256,128,64,32} that still yields >= 4*CU work-groups AND fits SLM.
// Descending, because a larger WG amortises the triangle staging over more solves:
// A traffic / B traffic = n / (4*WG).
int WG = 32;
for (int cand : {256, 128, 64, 32}) {
    if (cand > MAXWG) continue;
    if (lds_bytes(cand) > E_BYTES) continue;
    if (int64_t(bs) * ceil_div(q, cand) >= int64_t(4) * CU) { WG = cand; break; }
    WG = cand;                                  // remember the largest feasible as the floor case
}

const int groups = ceil_div(q, WG);
sycl::nd_range<1>(sycl::range<1>(size_t(bs) * groups * WG), sycl::range<1>(WG));

// inside:
const int wg_id  = int(it.get_group_linear_id());
const int b      = wg_id / groups;
const int u      = (wg_id % groups) * WG + int(it.get_local_linear_id());
const bool live  = (u < q);
```

`live` is a **mask, not an early return**. Threads with `u >= q` must still enter the cooperative staging loop and hit `sycl::group_barrier`; on `Side::Left` they must also participate in every staging-tile barrier for the whole kernel. Only their loads and stores are predicated.

The blocked tier V2 issues `2·⌈n/nb⌉ - 1` submissions: one V1 launch per diagonal block (shape above with `n → nb`) and one `gemm_custom` per `i > 0`.

### 3.3 Why it does not starve at small batch

Work-groups = `batch × ⌈q/WG⌉`, and `WG` *falls* to 32 when the product is short. Worked on this box (128 SMs), `n = 32` float, `Side::Right`:

| batch | q | WG chosen | work-groups | threads | warps/SM |
|---|---|---|---|---|---|
| 2048 | 1024 | 256 | 8 192 | 2.10 M | saturated |
| 512 | 1024 | 256 | 2 048 | 524 k | saturated |
| 128 | 1024 | 128 | 1 024 | 131 k | 32 |
| 32 | 1024 | 32 | 1 024 | 32 768 | 8 |
| 8 | 1024 | 32 | 256 | 8 192 | 2 |
| 1 | 1024 | 32 | 32 | 1 024 | 0.25 |
| 128 | 32 | 32 | 128 | 4 096 | 1 |

Three things this table says honestly:

1. **This is not the batch-only-parallelism shape.** The grid is `batch × ⌈q/WG⌉` and the `q` term carries it at `batch = 1`. The distinguishing test is mechanical — read the nd_range — and it passes.
2. **It does not degenerate the way design 1's launcher did.** Critique 2 (design 1) showed that maximising the RHS panel `QB` against the SLM ceiling drives `panels → 1` and collapses the grid to exactly `batch` work-groups. There is no panel here; the decomposition granularity is one *thread* per solve, so `⌈q/WG⌉ ≥ q/256` always, and `WG` is chosen *from* the CTA-count target rather than from the SLM ceiling.
3. **It genuinely runs out of parallelism at `batch·q ≲ 32 k`**, and no TRSM implementation escapes that with `s` serial. The routing predicate (§10) therefore contains an explicit `batch·q ≥ 8·CU·32` guard, not a batch threshold. Per the standing rule the benchmark sweep still runs `batch ∈ {1,8,32,128,512,2048}` — profiling only at saturation is exactly what hid this class of defect in this repo before — but only `batch ≥ 128` is used for ranking.

### 3.4 Coalescing — the mitigation design 2 promised in a section it never wrote

`B` is column-major.

* **`Side::Right`** (`du = 1`): thread `u` owns row `u` of `B`; at step `s` lanes read `B(u0+lane, ρ(s))` — **consecutive addresses, natively coalesced**, `float4`-vectorisable when `ldb` and `u0` are 4-aligned. No staging tile. Ortho's default cell.
* **`Side::Left`** (`ds = ±1`): thread `u` owns column `u`; at step `s` lanes read `B(ρ(s), u0+lane)` — stride `ldb`. For float, `ldb = 32`: 32 lanes × 4 B used out of 32 sectors × 32 B fetched = **8× over-fetch on both the read and the write-allocate**. Against a design whose entire thesis is "touch B exactly twice" this is fatal, and it is not a corner: `ortho.cc:112,197,281` selects `Side::Left` whenever the public `ortho` is called with `transA = Trans/ConjTrans`.

  **Mitigation (specified, not deferred):** an SLM transpose staging tile of `NB_STAGE × WG` elements, `NB_STAGE = min(N,16)`. The `s`-loop is chunked into `⌈N/NB_STAGE⌉` rounds. Per round, lanes are remapped `r = lane % NB_STAGE`, `c = lane / NB_STAGE`, so a warp reads `NB_STAGE = 16` consecutive floats (64 B = two fully-used 32-B sectors — 32 B is the hardware sector granularity, so 64 B contiguous is zero over-fetch) per column; then each thread pulls its own `NB_STAGE` values out of SLM into its register chain. The store path is the mirror image. Tile stride is `NB_STAGE + 1` elements to keep the SLM column reads conflict-free.

  Cost: `NB_STAGE·WG·sizeof(T)` of SLM and `2⌈N/NB_STAGE⌉` extra work-group barriers. `NB_STAGE ∈ {8,16,32}` is the one tuning knob here; `16` is the starting value and MUST be measured.

### 3.5 Reuse decision on `kernel-heuristics.hh` — explicit rejection

`include/batchlas/util/kernel-heuristics.hh:172-247` (`compute_batched_nd_range_sizes`) already implements a `workgroups_per_matrix = ceil(CU/batch)` decomposition with an L2 `footprint_per_problem` term. It is **not** reused. Reason: it models "one work-group per matrix, work-group covers `problem_size` elements" and derives `local_size` from `compute_optimal_wg_size(device, kernel_type, ...)`. This kernel's `WG` is constrained by register residency and by `WG ≤ q`-driven CTA count, neither of which that helper models, and its `min(batch, problems_in_L2)` clamp would cap the grid below what `q` supports. Re-deriving 12 lines is cheaper than bending it.

---

## 4. Local memory budget, largest resident `n`, occupancy on sm_89

### 4.1 The formula

```
lds_elements(N, WG, side) = N(N+1)/2            // packed Lc
                          + N                    // rd
                          + (side == Left ? NB_STAGE * WG : 0)
lds_bytes = lds_elements * sizeof(T)
E_BYTES   = max(16384, local_mem_size - 4096)    // 45056 on this GPU
```

The `- 4096` reserve is the repo's own convention (`cmake/BatchLASDetectSYCL.cmake:57-67`; `build/include/batchlas/device_limits.hh:24,28` records `local_mem_bytes = 49152`, `subgroup_workspace_budget_bytes = 45056`). Critique 2 (design 1) is correct that budgeting against the raw 49152 produces configurations that will not launch.

### 4.2 Budget vs `n` and scalar type (`WG = 128`, `NB_STAGE = min(N,16)`)

| T | N | `Lc + rd` | Right total | Left total | vs 45056 B |
|---|---|---|---|---|---|
| float | 8 | 176 B | 176 B | 4.2 KB | ✓ |
| float | 16 | 608 B | 608 B | 8.6 KB | ✓ |
| float | 32 | 2 240 B | 2.2 KB | 10.4 KB | ✓ |
| float | **64** | 8 320 B | 8.1 KB | 16.4 KB | ✓ |
| double | 16 | 1 216 B | 1.2 KB | 17.2 KB | ✓ |
| double | **32** | 4 480 B | 4.4 KB | 20.4 KB | ✓ |
| complex\<float\> | **32** | 4 480 B | 4.4 KB | 20.4 KB | ✓ |
| complex\<double\> | 8 | 704 B | 0.7 KB | 16.7 KB | ✓ |
| complex\<double\> | **16** | 2 432 B | 2.4 KB | 34.4 KB | ✓ (tight) |

**SLM is never the binding constraint for `Side::Right`, and is binding only for `complex<double>` on `Side::Left`** — where the launcher will drop `WG` to 64 (`NB_STAGE·WG` halves to 16 KB) and the cell is flagged for measurement in §10.

### 4.3 Largest resident `n` per type

Set by **registers**, not SLM: `N` accumulators of `sizeof(T)/4` registers each, against the repo's measured 64-accumulator / 256 B-per-thread cliff.

```
n_cta(float) = 64      n_cta(double) = 32
n_cta(complex<float>) = 32     n_cta(complex<double>) = 16
```

`n > n_cta(T)` goes to V2. **These four numbers are a prediction and step 2 of §8 exists to falsify them** — the very first thing built must be compiled with `-Xcuda-ptxas -v` and the register count read off. If `N = 64` float spills, `n_cta(float)` drops to 32 and V2 takes `33..64`; the design survives the demotion because V2 already exists.

### 4.4 Occupancy on sm_89 — ESTIMATES, to be confirmed with `ncu`

sm_89: 65 536 regs/SM, 1 536 threads/SM (48 warp slots), 100 KB shared/SM under opt-in, 48 KB per-CTA via SYCL. Register estimate = `N·(sizeof(T)/4) + 24` for addressing, loop state and the staged value.

| T | N | side | regs/thd | WG | CTA limit (reg) | CTA limit (SLM, 100 KB) | CTAs | warps/SM | occupancy |
|---|---|---|---|---|---|---|---|---|---|
| float | 16 | Right | 40 | 128 | 12 (thread cap) | — | 12 | 48 | **100 %** |
| float | 16 | Left | 40 | 128 | 12 (thread cap) | 11 | 11 | 44 | 92 % |
| float | 32 | Right | 56 | 128 | 9 | — | 9 | 36 | 75 % |
| float | 32 | Left | 56 | 128 | 9 | 9 | 9 | 36 | 75 % |
| float | 64 | Right | 88 | 128 | 5 | 12 | 5 | 20 | 42 % |
| float | 64 | Left | 88 | 128 | 5 | 6 | 5 | 20 | 42 % |
| double | 32 | Right | 88 | 128 | 5 | 22 | 5 | 20 | 42 % |
| double | 32 | Left | 88 | 128 | 5 | 4 | 4 | 16 | 33 % |
| complex\<float\> | 32 | Right | 88 | 128 | 5 | 22 | 5 | 20 | 42 % |
| complex\<double\> | 16 | Right | 88 | 128 | 5 | 41 | 5 | 20 | 42 % |
| complex\<double\> | 16 | Left | 88 | **64** | 11 | 6 | 6 | 12 | **25 %** ⚠ |

**The carveout caveat, stated rather than assumed.** DPC++ lowers `local_accessor` to dynamic shared memory and does not call the equivalent of `cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout, …)`. Ada's carveout is quantised; a CTA requesting 16.4 KB may be served from a 32 KB carveout and be limited to 2 CTAs/SM regardless of the register arithmetic. **Every occupancy number in the table above is unverified and must be read off `ncu`'s `launch__occupancy_limit_shared_mem` / `sm__warps_active.avg.pct_of_peak_sustained_active` before it is quoted.** Critique 2 (design 1) raised this and it is accepted, not answered: the mitigation is that this design's SLM footprint is 2–10× smaller than the LDS-panel alternative it replaces, so the carveout question binds in fewer cells.

**The arithmetic-intensity ceiling that licenses this structure.** For a triangular order `n` the work is `n²q/2` FMA against a minimum traffic of `2nq` elements, i.e. `n/2` FMA per element — `n/(2·sizeof(T))` FLOP/byte. In float at `n=32` that is 4 FLOP/B against a machine balance of ~41 FLOP/B (41 TFLOP/s strict FP32 / ~1 TB/s — the *real* FP32 target, not the TF32 number). So `Side::Right` V1 is DRAM-bound by ~10× and the only thing that matters is touching `B` exactly twice; that is why §3.4's staging tile is a requirement and not an optimisation, and why a 42 %-occupancy kernel is not automatically disqualified. It is *not* an argument that 42 % is sufficient — a Little's-law budget (≈500 KB in flight for 1 TB/s at ~500 ns ⇒ ~4 KB/SM ⇒ ~8 outstanding 128 B loads per warp at 4 warps/SM) has to come out of `ncu`, not out of a ceiling.

FP64 inverts this: ~1.3 TFLOP/s on this part gives balance ~1.3 FLOP/B, so `double` and `complex<double>` are compute-bound from `n ≈ 21` and their crossovers must be derived separately. Tuning the whole surface on float alone has already cost this repo once, on `syev`.

---

## 5. Exact public contract

### 5.1 Signature — unchanged, no header edits

```cpp
template <Backend Back, typename T>
Event batchlas::trsm(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const MatrixView<T, MatrixFormat::Dense>& B,
                     T alpha,
                     Side side, Uplo uplo, Transpose transA, Diag diag);
```
`include/batchlas/blas/functions/trsm.hh:85-95`. `alpha` is in position 4 to match `trmm`; the trailing-`alpha` spelling is a deleted overload at `:109-125`. Option-struct spellings `options.hh:486-496`, `TrsmOptions<T>` defaults `{alpha=1, Side::Left, Uplo::Lower, NoTrans, NonUnit}` at `options.hh:257-264`.

**Hook points are the three PUBLIC entry points, not `trsm_vendor`:**
* `src/backends/cublas.cc:1594` — `trsm<Back,T>(ctx, A, B, alpha, side, uplo, transA, diag)`, which forwards to `backend::trsm_vendor` at `:1602` with `alpha` **last**. Hooking `trsm_vendor` (`:1103`) as both designs proposed would sit *above* the `trsm_validate_params` call at `:1115` and would silently drop the throw contract.
* `src/backends/rocblas.cc:138` — public `trsm`, validates at `:150`.
* `src/backends/netlib_lapack.cc:404` — public `trsm`, and **it does not call `trsm_validate_params` at all**. `grep -rn trsm_validate_params src/` returns exactly `cublas.cc:1115` and `rocblas.cc:150`. Adding the missing call to netlib is step 5 of §8; the router must call `trsm_validate_params` itself before reading any extent, because the predicate reads `B.rows()/B.cols()` and `A.rows()` and would otherwise index a non-conforming shape.

### 5.2 The 24 combinations, enumerated

`n = A.rows() = A.cols()`. `ρ(s) = fwd ? s : n-1-s`. `Lc(s,t)` is given in **stored A coordinates**; `conj` column says whether `conj()` is applied to the loaded element. Every listed coordinate lies inside the referenced triangle by construction.

| # | Side | Uplo | Transpose | Diag | `op_is_lower` | `fwd` | `Lc(s,t)` (stored) | conj | diag of A read? | `B̃(s,u)` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Left | Lower | NoTrans | NonUnit | T | T | `A(s,t)` | no | yes → `rd[s]=1/A(s,s)` | `B(s,u)` |
| 2 | Left | Lower | NoTrans | Unit | T | T | `A(s,t)`, `Lc(s,s)≡1` | no | **no** | `B(s,u)` |
| 3 | Left | Lower | Trans | NonUnit | F | F | `A(n-1-t, n-1-s)` | no | yes | `B(n-1-s,u)` |
| 4 | Left | Lower | Trans | Unit | F | F | `A(n-1-t, n-1-s)`, `Lc(s,s)≡1` | no | **no** | `B(n-1-s,u)` |
| 5 | Left | Lower | ConjTrans | NonUnit | F | F | `A(n-1-t, n-1-s)` | **yes** | yes | `B(n-1-s,u)` |
| 6 | Left | Lower | ConjTrans | Unit | F | F | `A(n-1-t, n-1-s)`, `Lc(s,s)≡1` | **yes** | **no** | `B(n-1-s,u)` |
| 7 | Left | Upper | NoTrans | NonUnit | F | F | `A(n-1-s, n-1-t)` | no | yes | `B(n-1-s,u)` |
| 8 | Left | Upper | NoTrans | Unit | F | F | `A(n-1-s, n-1-t)`, `Lc(s,s)≡1` | no | **no** | `B(n-1-s,u)` |
| 9 | Left | Upper | Trans | NonUnit | T | T | `A(t,s)` | no | yes | `B(s,u)` |
| 10 | Left | Upper | Trans | Unit | T | T | `A(t,s)`, `Lc(s,s)≡1` | no | **no** | `B(s,u)` |
| 11 | Left | Upper | ConjTrans | NonUnit | T | T | `A(t,s)` | **yes** | yes | `B(s,u)` |
| 12 | Left | Upper | ConjTrans | Unit | T | T | `A(t,s)`, `Lc(s,s)≡1` | **yes** | **no** | `B(s,u)` |
| 13 | Right | Lower | NoTrans | NonUnit | T | F | `A(n-1-t, n-1-s)` | no | yes | `B(u, n-1-s)` |
| 14 | Right | Lower | NoTrans | Unit | T | F | `A(n-1-t, n-1-s)`, `Lc(s,s)≡1` | no | **no** | `B(u, n-1-s)` |
| 15 | Right | Lower | Trans | NonUnit | F | T | `A(s,t)` | no | yes | `B(u,s)` |
| 16 | Right | Lower | Trans | Unit | F | T | `A(s,t)`, `Lc(s,s)≡1` | no | **no** | `B(u,s)` |
| **17** | **Right** | **Lower** | **ConjTrans** | **NonUnit** | F | T | `A(s,t)` | **yes** | yes | `B(u,s)` |
| 18 | Right | Lower | ConjTrans | Unit | F | T | `A(s,t)`, `Lc(s,s)≡1` | **yes** | **no** | `B(u,s)` |
| 19 | Right | Upper | NoTrans | NonUnit | F | T | `A(t,s)` | no | yes | `B(u,s)` |
| 20 | Right | Upper | NoTrans | Unit | F | T | `A(t,s)`, `Lc(s,s)≡1` | no | **no** | `B(u,s)` |
| 21 | Right | Upper | Trans | NonUnit | T | F | `A(n-1-s, n-1-t)` | no | yes | `B(u, n-1-s)` |
| 22 | Right | Upper | Trans | Unit | T | F | `A(n-1-s, n-1-t)`, `Lc(s,s)≡1` | no | **no** | `B(u, n-1-s)` |
| 23 | Right | Upper | ConjTrans | NonUnit | T | F | `A(n-1-s, n-1-t)` | **yes** | yes | `B(u, n-1-s)` |
| 24 | Right | Upper | ConjTrans | Unit | T | F | `A(n-1-s, n-1-t)`, `Lc(s,s)≡1` | **yes** | **no** | `B(u, n-1-s)` |

Row **17** is `ortho`'s dominant cell for `transA = NoTrans` on complex (`ortho.cc:112-114, 194-198`; real uses row 15). Rows **1** and **2**'s neighbourhood — specifically **row 1**, `Side::Left, Uplo::Lower, NoTrans, NonUnit` — is `ortho`'s cell whenever the public `ortho` is called with `transA = Trans/ConjTrans` (`is_A_trans ? Side::Left : Side::Right` with `inv_trans = NoTrans`). Both must be in the test ladder and both must be in the benchmark grid; `tests/ortho_tests.cc:249,293` pins `transposes = {Transpose::NoTrans}` so the Left cell has **no** end-to-end coverage today.

For real `T`, `ConjTrans ≡ Trans` (rows 5/3, 11/9, 17/15, 23/21 coincide) — `conj()` is a no-op for real scalars, so no special-casing is needed, but the tests must still pass `ConjTrans` on complex data to exercise it.

### 5.3 In-place semantics

`X` overwrites `B`. There is no output matrix. Within V1 each thread reads its `n` inputs and writes its `n` outputs with no cross-thread dependence, so there is no intra-kernel hazard; work-groups own disjoint `u`-ranges. Within V2, `gemm_custom` reads `B(·,S_i)` and writes `B(·,R_i)`, disjoint ranges of the same buffer, in separate submissions ordered by the queue.

`A` and `B` must not overlap. Debug-only assert via `batchlas::device::detail::views_overlap(A.kernel_view(), B.kernel_view())` (`group_blas_common.hh:299-319`; note it takes `KernelMatrixView`, so host-side use needs `.kernel_view()` — the spelling both designs got wrong).

### 5.4 `alpha`

`alpha` scales the RHS **arithmetically**, before substitution: `x[s] = alpha·B̃(s,u) - acc`. This is exactly `cublas.cc:1174` and `netlib_lapack.cc:457`.

**`alpha == 0` explicitly does NOT get a reference-BLAS quick return.** Design 2 proposed adopting the early zero-fill for the native kernel only. That is rejected: `trsm_use_native` selects per shape, per type and per env var, so within one binary the same call would return zeros on one route and `NaN` on another — a three-way divergence keyed on a *tuning constant*, and one that a finite-`B` residual test is structurally incapable of detecting (`‖op(A)X‖ = 0` under both semantics). The native path matches all three existing backends bit-for-bit in intent here. If the inconsistency with reference BLAS is to be fixed, it is a separate atomic commit touching `cublas.cc:1174/1204`, `netlib_lapack.cc:454/469/483/497` and the native path together, gated on a test that seeds `B` with `NaN`.

### 5.5 Complex conjugation

`ConjTrans` conjugates **once per staged element**, inside `triangular_matrix_entry` during the staging pass — `n(n+1)/2` conjugations per work-group, never in the `n²q/2` inner loop. The hot loop (the `acc += Lc·x` accumulation and the `x·rd` scaling) writes the complex multiply **explicitly** as `(ac−bd, bc+ad)` on the real components; `std::complex operator*` emits an `isnan` branch and a `__mulsc3` call in device code and is worth 1.2–1.3× here. Staging, load and store keep `std::complex`.

### 5.6 Failure reporting and `buffer_size`

| aspect | contract |
|---|---|
| `trsm_buffer_size` | **Does not exist and is not added.** `grep -rn trsm_buffer_size` over the tree returns nothing. V1's scratch is `sycl::local_accessor` sized host-side from `(N, WG, sizeof(T))`; V2's operands are views into the caller's `A` and `B`. No `ctx.workspace()` lease, no `Matrix<T>` per call, no global allocation anywhere. |
| singular / zero diagonal | **No exception, no `info`, no event failure.** `inf`/`nan` propagate silently, as `cublas.cc`, `rocblas.cc` and `netlib_lapack.cc` all do. |
| validation | `trsm_validate_params(A, B, side, uplo, transA, diag)` (`trsm.hh:26-80`) runs **before** the routing predicate reads any extent, and still throws `std::invalid_argument`. |
| batch | Strided (`data_ptr() + b·stride()`). Batch count comes from `A.batch_size()`, matching all three vendors (`cublas.cc:1114`, `rocblas.cc:147`, `netlib_lapack.cc:434`). The native path additionally requires `A.batch_size() == B.batch_size()` and refuses otherwise — note that `trsm_validate_params` does **not** currently compare the two, so a batch-1 `A` against batch-N `B` validates today; propose adding that check upstream in a separate commit. |
| heterogeneous batches | Refused (`is_heterogeneous()`, `matrix.hh:1034`). Strictly more conservative than cuBLAS, which ignores per-item extents. |
| return | `ctx.get_event()` (`sycl-device-queue.hh:255`) — the SYCL-submission convention. `create_event_after_external_work()` exists for non-SYCL vendor calls only. |

**The argument for shipping without `info`, on its own merits.** It is not pinned by `cond_tests` — that citation is withdrawn (§2.4). It rests on three facts: (i) all three existing backends report nothing, so adding a status to one route would make the observable behaviour route-dependent; (ii) `trsm` is called from exactly one place in the library, `src/extensions/ortho.cc:194,281` (`grep -rn trsm src/extensions/`), which reads no status; (iii) the public `TrsmOptions<T>` (`options.hh:257-264`) has no `info` member, so surfacing one is an API change. The `use_div` flag of §7 is computed and deliberately discarded for exactly these reasons; if a future `TrsmOptions::info` lands, that flag is where it plugs in.

---

## 6. Reused primitives by exact verified signature, and the new code

### 6.1 Reused — device level, all read from source and confirmed present

```cpp
// include/batchlas/blas/device/detail/group_blas_common.hh:705-728   — the entire
// uplo/trans/conj/unit fold, staging pass ONLY.
template <typename T>
inline constexpr T batchlas::device::detail::triangular_matrix_entry(
        const KernelMatrixView<T, MatrixFormat::Dense>& a,
        int row, int col, TriangularTransform transform);

// group_blas_common.hh:102-107
struct batchlas::device::detail::TriangularTransform {
    Side side = Side::Left;  Uplo uplo = Uplo::Upper;
    Transpose trans = Transpose::NoTrans;  Diag diag = Diag::NonUnit;
};

// group_blas_common.hh:675-677  — debug asserts on the staged coordinates
inline constexpr bool batchlas::device::detail::triangular_storage_contains(
        Uplo uplo, int row, int col);

// group_blas_common.hh:299-319  — debug-only A/B aliasing assert (KernelMatrixView!)
template <typename T>
inline constexpr bool batchlas::device::detail::views_overlap(
        const KernelMatrixView<T, MatrixFormat::Dense>& lhs,
        const KernelMatrixView<T, MatrixFormat::Dense>& rhs);
```

### 6.2 Reused — non-`group_blas`, all verified

```cpp
// include/batchlas/blas/functions/trsm.hh:26-80
template <typename T> inline void batchlas::trsm_validate_params(
        const MatrixView<T,MatrixFormat::Dense>& A, const MatrixView<T,MatrixFormat::Dense>& B,
        Side side, Uplo uplo, Transpose transA, Diag diag);

// src/sycl/gemm_kernels.hh:61-70  — V2's trailing update
template <typename T> Event batchlas::sycl_gemm::gemm_custom(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A, const MatrixView<T,MatrixFormat::Dense>& B,
        const MatrixView<T,MatrixFormat::Dense>& C, T alpha, T beta,
        Transpose transA, Transpose transB, ComputePrecision precision);

// include/batchlas/util/sycl-local-accessor-helpers.hh:23
template <typename T, int Dims> inline T* batchlas::util::get_raw_ptr(
        const sycl::local_accessor<T, Dims>& accessor);

// src/backends/route_common.hh:43-67  — BATCHLAS_TRSM_VARIANT parsing; NOTE the default
// for an UNSET variable is auto_variant, not vendor_variant.
template <typename Variant> Variant batchlas::backend::detail::parse_cublasdx_variant_request(
        const char* env_var, Variant vendor_variant, Variant custom_variant, Variant auto_variant);

// src/backends/route_common.hh:70-72
inline bool batchlas::backend::detail::is_gpu_queue(const Queue& ctx);

// src/math-helpers.hh:19-26
template <typename T> struct batchlas::internal::is_complex;    // NOT `is_complex_v<T>` — that
                                                                // spelling does not exist in this tree

// include/batchlas/util/sycl-device-queue.hh:314, :176, :122-135
Device Queue::device() const;                      // then `.type` is a DATA MEMBER, not a call
size_t Device::get_property(DeviceProperty) const; // DeviceProperty::MAX_COMPUTE_UNITS / MAX_WORK_GROUP_SIZE

// src/util/template-instantiations.hh
BATCHLAS_FOR_EACH_SCALAR_TYPE_1, BATCHLAS_UNPAREN   // instantiation, as src/extensions/sytrd_cta.cc:359-378
```

### 6.3 Deliberately NOT reused, with reasons

* **`batchlas::device::gemm`** (`group_blas_gemm.hh:424-453`) for the trailing update. Its `C` write goes through `write_matrix_output`'s `alpha/beta` contract (`group_blas_common.hh:773-779`) over a `KernelMatrixView`; our left operand is a *packed* triangle with an `s`-dependent base. Additionally `can_use_matrix_fast_path` (`group_blas_subgroup_common.hh:433-452`) and `can_use_matrix_register_fast_path` (`:455-478`, requiring `group_local_linear_range == 256` exactly) are `float`-only, so `double` and complex would land on `generic::gemm`'s `reduce_sum_group` collective — strictly worse than a per-thread register chain.
* **`batchlas::device::trmv`** (`group_blas_trmv.hh:47-56`). Its `effective_lower` output ordering (`:16-19`) is the right *idea* for aliased in-place work and is why the ordering here is safe, but it multiplies rather than divides, is vector-only, and is a work-group collective producing one output at a time.
* **`sg_compat.hh` sub-group partitions.** `sytrd_cta`'s one-matrix-per-partition mapping is wrong here: the parallelism is in `q`, not in the batch.
* **`kernel-heuristics.hh:172-247`** — see §3.5.
* **`MatrixView::operator()(Slice,Slice)`** for V2's sub-views — see §2.3.

### 6.4 New code to write

1. `src/sycl/trsm_native.hh` — declarations, `trsm_native_max_cta_n<T>()`, `trsm_native_supported<T>(...)`, `TrsmVariant` + `trsm_variant_request()`.
2. `src/sycl/trsm_native.cc`:
   a. `trsm_stage_canonical<T,N>(item, a_kv, Lc, rd, n, transform, fwd, &use_div)` — cooperative packed staging with the §2.1 index map, guarded reciprocals, zero padding to `N`, and a work-group reduction of the `use_div` flag.
   b. `TrsmCtaKernel<T, N, SideV>` — the V1 body: staging, barrier, per-thread unrolled recurrence, store. `Side::Left` adds the §3.4 transpose staging tile.
   c. `trsm_native_v1<Back,T>(...)` host launcher implementing §3.2.
   d. `trsm_native_blocked<Back,T>(...)` implementing §2.3.
   e. Explicit instantiation over `BATCHLAS_FOR_EACH_SCALAR_TYPE_1` × `{CUDA, ROCM, NETLIB}`.
3. Routing hooks in `src/backends/cublas.cc:1594`, `src/backends/rocblas.cc:138`, `src/backends/netlib_lapack.cc:404`, plus the missing `trsm_validate_params` call in netlib.
4. Test additions in `tests/trsm_tests.cc` (§9) and benchmark additions in `benchmarks/trsm_benchmark.cc`.

**Kernel-object count.** `Side` and `N` are compile-time; `fwd`, `op_is_lower`, `conj`, `unit` and `alpha` are all runtime, consumed by the staging pass or by work-group-uniform branches. So the count is `|N buckets| × 2 (Side) × 4 (types)` = `(4+3+3+2) × 2 = 24` per backend. That is comparable to `sytrd_cta`'s 32 (`4 P × 2 uplo × 4 types`). `src/sycl/` currently contains one source (`gemm_kernels.cc`, `src/sycl/CMakeLists.txt:1-3` → `batchlas_sycl_obj`), so this lands in a small, isolated device-link unit and does not touch the `sytrd` group whose device link is ~190 s.

---

## 7. Numerical stability

### 7.1 The bound

TRSM with multiple right-hand sides has no single `ΔA`; the classical result is **columnwise** (Higham, *ASNA* Thm 8.5), and here that is exactly right because §2.1's `u`-solves are genuinely independent. For each independent solve `u`:

```
(op(A) + ΔA_u) x̂_u = α b_u ,    |ΔA_u| ≤ γ_{n+5} |op(A)|   componentwise,   γ_k = ku/(1-ku)
```

for real `T` in **V1**. Reference substitution gives `γ_n`; the `+5` is the reciprocal-multiply on the diagonal (§7.2), and it perturbs the diagonal entries only.

For complex `T` in V1: `|ΔA_u| ≤ γ_{√2(n+10)} |op(A)|`, the extra term being the complex multiply that replaces the complex division (`√2·γ₆` rather than `√2·γ₄` per diagonal step, plus the reciprocal's own rounding). This is stated separately because critique 1 (design 1) is right that a real-arithmetic "one extra rounding, |δ| ≤ 2u" estimate does not cover complex — and complex is the cell §10 ships first.

For **V2** the off-diagonal sum is regrouped into `⌈n/nb⌉` blocks, so `|ΔA_u| ≤ γ_{nb + n/nb + 5}|op(A)|` — strictly *better* than `γ_{n+5}` for `1 < nb < n`. The critical path of roundings is `Θ(nb + n/nb) ≥ 2√n`. **Do not claim `γ_{log n}`**: the recurrence is sequential in the block index and `gemm_custom` accumulates over `k` linearly.

Two claims that appeared in the input designs and are **deleted**: "the sequence of floating-point operations is identical to the unblocked algorithm" (false for V2, and false for any register-tiled trailing update), and "FMA contraction strictly reduces error" (not a theorem — `a·b − c·d` is the standard counterexample).

### 7.2 How it differs from LAPACK, and the guard

Two deviations, both bounded and both switchable off:

**(a) Reciprocal-and-multiply on the diagonal.** `rd[s] = 1/Lc(s,s)` is computed **once per matrix during staging** (`n` reciprocals per work-group, amortised over `WG·groups` solves) and each of the `n·q` diagonal steps becomes a multiply. Motivation: `n` divides against `n²/2` FMAs is *not* negligible per-thread — at `n=32`, 32 divides at ~15 FMA-equivalents each against 512 FMAs would roughly double the kernel. Amortising into staging removes that entirely.

**The guard is a direct verification, not an exponent-range argument.** After computing `rd[s]`, evaluate `chk = rd[s]·Lc(s,s)` and require `|chk − 1| ≤ 4u`. Reduce `use_div = OR_s (test failed)` across the work-group; when set, the recurrence divides instead of multiplying. One multiply and one compare per diagonal entry.

This test covers, in one expression, every failure mode the critiques enumerated separately:
* `rd = inf` or `nan` (`|Lc(s,s)|` denormal-small) → `chk` non-finite → caught.
* `rd` **subnormal** (`|Lc(s,s)| > 1/FLT_MIN ≈ 8.5e37`) → `chk` loses significance → caught. `!isfinite` alone does **not** catch this; critique 1 (design 1) and critique 3 (design 1) both flagged it.
* `rd == 0` (complex `|d|²` overflowed under a naive `conj(d)/|d|²`) → `chk = 0` → caught.
* complex `|d|²` **subnormal** (`|d| ∈ (3.7e-23, 1.09e-19)` for `complex<float>`) → `rd` finite but garbage → caught.
* `std::isfinite` does not accept `std::complex` and there is no such overload in this tree (existing uses are on real scalars: `src/extensions/stedc_secular.cc:685`, `sycl::isfinite` at `stedc_merge_cta.cc:871`), so the naive guard as both designs spelled it **would not compile for two of the four types**. This one does.

Additionally: for complex, form the reciprocal explicitly with Smith's scaled algorithm in the staging pass rather than delegating to `std::complex::operator/`, whose device lowering under DPC++ is not pinned down. The staging pass runs `n` times per work-group; the cost is irrelevant.

The guard's branch is **work-group uniform** and evaluated once per matrix, so the hot loop is unaffected.

**(b) `alpha` handling** is identical to all three backends (§5.4) — no deviation.

**Consequence, and this is the test hook:** with `use_div` forced on (`BATCHLAS_TRSM_DIAG=div`, §8 step 2), V1's arithmetic is operation-for-operation the reference loop nest at `netlib_lapack.cc:439-505`. Modulo compiler FMA contraction, results are **bitwise comparable** to the netlib backend on real types. That is a much stronger correctness signal than any residual tolerance and it must be exercised in CI.

### 7.3 For which inputs the difference is material

| input class | material? | why |
|---|---|---|
| well-scaled `A`, all types | no | `+5u` on a bound already `O(n)u` and ~100× pessimistic |
| `|diag(op(A))|` near `FLT_MAX`/`FLT_MIN` (real) | **caught** — falls to division, bit-identical to reference | §7.2 guard |
| complex `|d|` outside `[~1e-19, ~1.8e19]` (float) / `[~1e-154, ~1.3e154]` (double) | **caught** — falls to division | §7.2 guard. Live for `ortho`: `L = chol(A^H A)` scales like `‖A‖`, so column norms of `1e-20` land in the band |
| singular / zero diagonal | no detection, `inf`/`nan` propagate | contract parity, §5.6 |
| `n > n_cta(T)` (V2) | last bits differ from the vendor; bound *improves* | blocked regrouping |
| any bitwise comparison against another backend | **yes, always** | FMA contraction and the reciprocal both move the last bits. All tests must be residual tests. |

### 7.4 Block size and accuracy

`nb` in V2 is fixed by V1's capacity, not by accuracy: `nb = n_cta(T)` (64/32/32/16), with `nb/2` as the alternative rung of a `{n_cta/2, n_cta}` ladder resolved by `detail::tuning_env_override("BATCHLAS_TUNE_TRSM_NB", …)` (`include/batchlas/tuning_params.hh:33-40`). Because no `trsm_buffer_size` exists, the hazard spelled out at `tuning_params.hh:27-32` — the size query and the run path resolving a ladder differently — **cannot arise here**. That is a real benefit of keeping the workspace at zero.

### 7.5 Inversion, one more time

Rejected at every tier; see §2.4. The specific correction to design 2's licence: restated in orthogonality currency, inversion contributes `c·u·κ(A)²` to `‖X̂^H X̂ − I‖`, the same order as CholQR's own error, with `c` unbounded — not "smaller by a factor `κ(A)`". If it is ever revisited, it needs (a) a measured `c`, (b) a conditioning sweep that grades **inside** the diagonal block (see §9), and (c) a `trsm_buffer_size` + `Span<std::byte>` overload following the `potrf` two-overload pattern (`options.hh:539-568`) — which is a third independent reason to keep it out.

---

## 8. File-by-file implementation plan

Each step compiles and links on its own.

| # | File(s) | What | Kind |
|---|---|---|---|
| 1 | **new** `src/sycl/trsm_native.hh` | Declarations: `enum class TrsmVariant {Vendor,Native,Auto}`, `trsm_variant_request()` via `parse_cublasdx_variant_request`, `trsm_native_max_cta_n<T>()`, `trsm_native_supported<T>(...)`, `trsm_native<Back,T>(...)`. No definitions. | mechanical |
| 2 | **new** `src/sycl/trsm_native.cc`; `src/sycl/CMakeLists.txt` | `TrsmCtaKernel<T,N,Side::Right>` only, plus staging, the §7.2 guard, `BATCHLAS_TRSM_DIAG=div`. Instantiate float `N ∈ {8,16,32,64}`. **Build with `-Xcuda-ptxas -v` and read the register count. If `x[64]` spills, stop and reduce `n_cta(float)` to 32 before writing anything else.** Not routed; exercised by a temporary direct-call test. | **judgement** |
| 3 | `src/sycl/trsm_native.cc` | `Side::Left` with the §3.4 transpose staging tile. All four types, all `N` buckets. | **judgement** |
| 4 | `src/sycl/trsm_native.cc` | The §3.2 `WG`/`N` ladder, SLM clamp against `max(16384, local_mem_size-4096)`, `trsm_native_supported` structural checks. | mechanical |
| 5 | `src/backends/cublas.cc:1594`, `rocblas.cc:138`, `netlib_lapack.cc:404` | Route: `trsm_validate_params(...)` first (**adding the missing call to netlib**), then `if (trsm_use_native(...)) return trsm_native<Back,T>(...)`. Default the `Auto` heuristic to **vendor for every cell** at this step; only `BATCHLAS_TRSM_VARIANT=native` reaches the new kernel. Every existing test must stay green unchanged. | mechanical |
| 6 | `src/sycl/trsm_native.cc` | `trsm_native_blocked` — the §2.3 driver, both `gemm_custom` forms, explicit 6-arg sub-view construction, `beta = alpha` on the first update of each block. | **judgement** |
| 7 | `tests/trsm_tests.cc` | Fix `verifyTrsmResult` (§9), then add the three new suites. Run them against `BATCHLAS_TRSM_VARIANT=native`. | **judgement** |
| 8 | `benchmarks/trsm_benchmark.cc` | Add the ortho-shaped grid of §10. | mechanical |
| 9 | `src/sycl/trsm_native.cc` (predicate only) | Flip `Auto` to native for the complex cells. Real stays vendor-first pending the §10 table. | mechanical |
| 10 | predicate | Flip individual real cells from the measured table, or leave them vendor and record why. | **judgement** |

---

## 9. Test plan

### 9.1 Exact ctest targets

| target | label (`tests/CMakeLists.txt:129-134`) | role |
|---|---|---|
| `trsm_tests` | `blas` | the direct guard; extended here |
| `ortho_tests` | `ortho` | end-to-end: an incorrect solve shows up as a failed orthogonality residual |
| `cond_tests` | `blas` | must stay green (it does **not** pin NaN propagation — §2.4) |
| `options_api_tests`, `linalg_layer_tests` | `util` (smoke) | the documented-spellings guard |

Selective run: `ctest -R "trsm_tests|ortho_tests|cond_tests"` or `ctest -L blas -L ortho`. None carry the `slow` label. **Do not run the full suite by default.** New cases go into the existing `trsm_tests` target rather than a new one, so `BATCHLAS_TEST_LABELS_blas` needs no edit.

### 9.2 What exists proves less than either design claimed

`tests/trsm_tests.cc`: fixed `rows = cols = ld = 8`, `batch_size = 3`, `alpha = 1` (`:28-32`). **Eight** `TYPED_TEST`s (`:194-234`, not four), covering `{Lower,Upper} × {NoTrans,Trans} × {single, host-loop}`. `Side::Right` never passed. `Diag::Unit` never passed. `ConjTrans` never passed. The four "batched" tests issue `batch_size` separate **unbatched** calls in a host loop (`:167-176`) and never reach a batched kernel. `ld > rows` never exercised.

**Prerequisite fix before `trsm_tests` can guard anything:** `verifyTrsmResult` (`:87-95`) accumulates `A.at(a_row,a_col,b) * B.at(k,j,b)` with **no `conj`**, and `performTrsmTest` fills `A` with real `1.0`/`0.5` even for `std::complex<T>` (`:120-133`). So the conjugate path has zero coverage today and cannot acquire it by adding a case.

### 9.3 The single test that catches the most likely failure mode

The most likely defect, by a wide margin, is a wrong cell in the §5.2 table — a swapped `Lc(s,t)`/`Lc(t,s)` on the `Side::Right` rows, a missed `ρ` reversal, or a dropped conjugate. Twenty-four cases collapse into arithmetic evaluated once during staging; one wrong cell is wrong *only* in that cell. `ortho_tests` exercises one of the 24 (row 17 or 15). Current `trsm_tests` exercises four.

> **`TrsmCanonicalCrossProduct`** — for every one of the 24 rows of §5.2, and for all four scalar types:
> * `A` **complex-valued with non-zero imaginary parts** and **non-symmetric** (`A ≠ Aᵀ`, `A ≠ Aᴴ`), distinct well-separated entries. Asymmetry and non-real entries are both load-bearing: with a symmetric `A` half the transpose errors are invisible, and with a real `A` every conjugate error is invisible.
> * The **unreferenced triangle filled with `NaN`**, and for the `Diag::Unit` rows the diagonal also `NaN`. A finite result proves no unreferenced element was read — which is `ortho`'s production precondition (`ortho.cc:156-161`), not a synthetic one.
> * `n = 5`, `q = 3`, `batch = 2`, `ldb > B.rows()`, `stride > ld·cols`, `alpha = 0.5 + 0.25i`.
> * **Oracle: the multiply-back residual `‖op(A)X̂ − αB_orig‖ / (‖op(A)‖·‖X̂‖·n·u)`**, with `op(A)` reconstructed independently from the `(uplo, trans, diag)` triple and accumulated in `double`/`long double`.
>
> The oracle must **not** be a transcription of `netlib_lapack.cc:416-508`. Read side by side, `netlib_lapack.cc:439-505` and `cublas.cc:1156-1220` are one implementation, not two — identical fold, identical four branches, identical `x = alpha*B − sum` with `x /= opA(·,·)`, differing only in `Ab.at(r,c,0)` vs `Ab[c*lda+r]`. A shared uplo or index error in that pair would be *confirmed* by such a test, which is precisely the `trmm` uplo/diag "test that could not fail by construction" pattern.
>
> **Validate that it can fail**: transpose one cell of the table by hand and confirm the corresponding row goes red — and only that row.

Secondly, a **bitwise** cross-check that is cheap and decisive: with `BATCHLAS_TRSM_DIAG=div` and `Backend::NETLIB` as reference, V1's arithmetic is the reference loop nest (§7.2). Assert agreement to within a handful of ulp on real types (not exact — FMA contraction).

### 9.4 The rest of the ladder

* **Boundary sweep.** `n ∈ {1,2,3,7,8,15,16,17,31,32,33,63,64,65,96,128,257}` × `q ∈ {1,7,31,32,33,127,128,129,1024}` × `batch ∈ {1,3,128}`, both sides, residual-checked. Covers the `N`-bucket padding tail, `q` not a multiple of `WG`, and both V1 and V2 (`n > n_cta`). One run with `-UNDEBUG` for the device-side `KernelMatrixView` bounds asserts (`matrix.hh:135-152`, compiled out in release).
* **Tier ladder.** At least one shape per tier per type, so a routing change cannot silently skip a kernel.
* **In-placeness and aliasing.** `B` before/after differs (the shape at `:51-72`); a debug `views_overlap(A.kernel_view(), B.kernel_view())` assert.
* **The two `ortho` cells explicitly**: row 17/15 (`Right, Lower, ConjTrans/Trans, NonUnit`) and row 1 (`Left, Lower, NoTrans, NonUnit`). The latter is reached whenever public `ortho` is called with `transA = Trans/ConjTrans` and has no coverage anywhere today (`ortho_tests.cc:249,293` pin `NoTrans`). Add a `Transpose::Trans` sweep to `ortho_tests`.
* **Guard coverage.** A diagonal entry at `1e38` (float) / `1e308` (double) and a complex diagonal at `|d| = 1e-21` (`complex<float>`): assert the result matches the divide path.

### 9.5 Accuracy harnesses (not ctest — `EXCLUDE_FROM_ALL`, `--target batchlas_benchmarks`)

`orthogonality_miniacc` (`ACC_ORTHO_CHOL2 / CHOLESKY / SHIFTCHOL3`, with `ACC_ORTHO_HOUSEHOLDER` as the conditioning-independent control) and `orthogonality_accuracy --impl ortho_chol2 --log10-cond-max 10 --samples 4096`.

**The prediction, stated so it is actually falsifiable.** Because V1 performs the reference summation in the reference order and does no inversion, the cond-vs-orthogonality curve should track the vendor's to within the `+5u`-on-the-diagonal term — i.e. **the curves should agree to within a factor of ~2 in `‖Q^HQ−I‖` across the whole conditioning range, not be indistinguishable.** A small movement is *expected* (reciprocal rounding, FMA contraction, and for `n > n_cta` the blocked regrouping); a movement of more than ~2× at moderate conditioning, or any qualitative change in where the curve turns up, indicates a canonicalisation or diagonal-handling bug. Both designs asserted "indistinguishable, and if it moves at all something is wrong"; that is a false-alarm detector and it is corrected here.

---

## 10. Routing predicate — **REQUIRES MEASUREMENT BEFORE IT IS TRUSTED**

```cpp
namespace batchlas::sycl_trsm {

enum class TrsmVariant { Vendor, Native, Auto };

inline TrsmVariant trsm_variant_request() {
    return backend::detail::parse_cublasdx_variant_request(
        "BATCHLAS_TRSM_VARIANT", TrsmVariant::Vendor, TrsmVariant::Native, TrsmVariant::Auto);
}   // NOTE: unset -> Auto (route_common.hh:36-42)

template <typename T> constexpr int n_cta();   // float 64, double 32, cfloat 32, cdouble 16

// Called AFTER trsm_validate_params.
template <typename T>
bool trsm_use_native(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const MatrixView<T, MatrixFormat::Dense>& B,
                     Side side, Uplo uplo, Transpose transA, Diag diag)
{
    const auto req = trsm_variant_request();
    if (req == TrsmVariant::Vendor) return false;

    // ---- structural: never negotiable ------------------------------------
    if (!backend::detail::is_gpu_queue(ctx))                  return false;  // .type is a member
    if (A.is_heterogeneous() || B.is_heterogeneous())         return false;
    if (A.batch_size() != B.batch_size())                     return false;

    const int n     = A.rows();
    const int q     = (side == Side::Left) ? B.cols() : B.rows();
    const int batch = A.batch_size();
    if (n < 1 || q < 1)                                       return false;

    if (req == TrsmVariant::Native) return true;              // forced; structurally supported

    // ---- heuristic: EVERY NUMBER BELOW IS A HYPOTHESIS --------------------
    const int CU = int(ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS));

    // Starvation guard. Threads == batch*q; below ~8 CTAs/SM of 32-wide work
    // there is nothing for the q axis to give. MEASURE: the constant 8.
    if (int64_t(batch) * q < int64_t(8) * CU * 32)            return false;

    // Complex: the incumbent is cublas.cc:1122-1225, a serial per-column
    // substitution in which every work-item re-reads the whole triangle from
    // global memory. Claimed with confidence at every n; V2 handles n > n_cta.
    if constexpr (internal::is_complex<T>::value)             return true;

    // Real: PROPOSED, NOT ESTABLISHED. Ships false until the §10 grid says so.
    return false;                                             // <-- step 9 leaves this; step 10 edits it
}
} // namespace
```

### What is claimed, and how confident

| cell | claim | confidence |
|---|---|---|
| `complex<float>` / `complex<double>`, any `n`, `batch·q ≥ 32 k` | native wins | **high** — the incumbent has ~`batch·q·n²/2` elements of A traffic against this design's `batch·⌈q/WG⌉·n²/2`; at `n=64, q=1024, batch=512` that is ~8.6 GB vs ~8.5 MB. There is no mechanism by which a CTA-resident kernel loses. |
| real, `n ≤ n_cta(T)`, `Side::Right`, large batch | native may win via the two `init_data_ptr_array` device drains per call | **medium** — `cublasXtrsmBatched` needs `A.data_ptrs(ctx)` and `B.data_ptrs(ctx)` (`cublas.cc:1231`), and `MatrixView::init_data_ptr_array` (`src/matrix.cc:2364-2382`) submits a kernel and calls `.wait()` **with no caching**, re-running on every invocation. `ortho`'s default `Chol2` issues two TRSMs per call inside LOBPCG's inner loop. Confirm with a profile before quoting it. |
| real, `n ≤ n_cta(T)`, `Side::Left` | unknown — the staging tile is new code with no analogue in the tree | **low** |
| real, `n > n_cta(T)` (V2) | likely loses until WP2 widens `gemm_custom` | **low** — `select_kernel_variant` needs `m ≥ 128, k ≥ 128` for the good float register kernels (`src/sycl/gemm/…`), and V2 feeds it `n_gemm = nb ≤ 64`. Vendor-first and say so. |
| any type, `batch·q < 32 k` | vendor | **high** |

### The grid that settles it

Three shapes, because `trsm_benchmark` today measures a shape the library never issues (`benchmarks/trsm_benchmark.cc:13-21`: `Side::Left`, square `n×n` RHS, `SquareBatchSizes` = `n ∈ {64…1024} × batch ∈ {1…512}`, `minibench.hh:788-796`):

1. **Ortho-Right** (rows 15/17): `Side::Right, Uplo::Lower, Trans` (real) / `ConjTrans` (complex)`, NonUnit`; `A` is `n×n`, `B` is `q×n`.
2. **Ortho-Left** (row 1): `Side::Left, Uplo::Lower, NoTrans, NonUnit`; `A` is `n×n`, `B` is `n×q`.
3. **Legacy square** — the existing `SquareBatchSizes` grid, unchanged, so the regression is comparable to history.

Sweep for (1) and (2): `n ∈ {8,16,32,64,128,256}` × `q ∈ {256,1024,4096}` × **`batch ∈ {128,512,2048}`** × all four types × `{native, vendor}`. Rank only at saturation and state the saturation level with every ratio. Additionally profile (do not rank) `batch ∈ {1,8,32}` and `q ∈ {32,128}` — profiling only at saturation is exactly what hid the batch-only-parallelism defect in this repo for months.

Protocol, non-negotiable:
* Warm the JIT before the first timed iteration (a cold first run once fabricated a 3.7× loss here).
* `bench::pristine(B)` between iterations — `trsm` is in-place.
* Watch for contention from the second RTX 4090; check clocks.
* **Acceptance is not wall-clock alone.** Capture `dram__bytes.sum` in `ncu` against the analytic floor `2·q·n·sizeof(T)·batch`. A ratio above ~1.3 means the kernel is not doing what §3.4 claims — most likely the `Side::Left` staging tile is mis-mapped — and the cell must be diagnosed, not shipped.
* Also capture `launch__occupancy_limit_shared_mem` and `sm__warps_active.avg.pct_of_peak_sustained_active` to replace §4.4's estimates with measurements.
* **Validate every routing flip end-to-end through `ortho`**, never at the kernel level alone. A prior 2.16× kernel win in this repo turned into an 11 % `gesvd` loss.

**Kill criterion, stated in advance:** if native real TRSM exceeds `1.10 × vendor` at the saturated ortho shape, real stays vendor-first and only complex flips. That is a legitimate outcome — vendor independence (a native path *exists* and is correct for every one of the 24 cells) is satisfied either way — and it is written into the predicate, not defended afterwards.

---

## 11. The three biggest risks

**1. `x[N]` does not stay in registers, and the whole structure collapses to an L1-bound kernel.**
This is the load-bearing bet. `T x[N]` is register-allocatable only under full unroll with compile-time `N`; if ptxas spills, `n_cta(float) = 64` means 256 B/thread of `.local` frame and every one of the `n²/2` updates becomes an LDL/STL pair, at which point §4.4's arithmetic-intensity argument says nothing about the kernel that actually runs.
*Mitigation:* step 2 of §8 exists solely to falsify this, before any other code is written — compile with `-Xcuda-ptxas -v`, read `Used N registers, M bytes stack frame`, and require `stack frame == 0`. If `N = 64` float spills, `n_cta(float)` drops to 32 and V2 takes `33..64` unchanged; if `N = 32` double spills, `n_cta(double)` drops to 16. The design degrades gracefully because V2 is not optional.

**2. Real types tie, and the effort buys vendor independence rather than speed.**
At `n ≤ 64` float, `n/(2·sizeof(T)) = 8` FLOP/B against a machine balance of ~41: TRSM is DRAM-bound by ~5×, cuBLAS is already near the same `2qn` roof, and the native kernel's ceiling is that same roof. The plausible real-type win is the two uncached `init_data_ptr_array` device drains per batched call, which matter most at *small* `q` and large batch — precisely what the existing square-shaped benchmark cannot see.
*Mitigation:* complex first (where the incumbent is the serial kernel and the traffic gap is ~1000×), real vendor-first with the `1.10×` kill criterion above, and the ortho-shaped grid added *before* any conclusion is drawn. Sunk cost is not value: if a real cell's first measurement is bad it stays vendor and the code path stays reachable only through `BATCHLAS_TRSM_VARIANT=native`.

**3. A wrong cell in the 24-way canonicalisation table survives into production.**
All three critiques hand-verified the table against both references and it holds — but it is exactly the kind of thing that is right on paper and wrong in code, `ortho_tests` covers one of the 24 cells, and current `trsm_tests` covers four (none of them `Right`, `Unit` or `ConjTrans`, and with a verifier that omits `conj` entirely and fills complex `A` with real values).
*Mitigation:* §9.3's `TrsmCanonicalCrossProduct` is built **first**, with an independent multiply-back oracle rather than a transcription of the reference (which would confirm a shared error rather than catch it), non-symmetric complex `A`, `NaN` in the unreferenced triangle, `ld > rows`, `alpha ∉ {0,1}`, and a deliberate one-cell transposition to prove it can fail. The `BATCHLAS_TRSM_DIAG=div` bitwise cross-check against `Backend::NETLIB` backs it up.

**Runner-up risks, recorded:** `complex<double>` + `Side::Left` at `WG = 128` needs 34.4 KB of SLM and lands at ~25 % occupancy (§4.4) — measure before enabling; `gemm_custom`'s narrow-`nb` envelope makes V2 provisional for real types (§10); and the 24 kernel objects per backend must be checked against the device-link budget after step 3 — if the `batchlas_sycl_obj` link grows past ~30 s, cut the bucket ladders to `{16,32,64}` / `{16,32}` / `{16,32}` / `{16}`.
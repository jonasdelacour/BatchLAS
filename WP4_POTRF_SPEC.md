# Native batched POTRF — final implementation specification

*Supersedes both candidate designs. Every defect raised in the six critiques is resolved below and marked **[FIX-n]** (changed) or **[ACCEPT-n]** (kept, with reason). Read-only survey; nothing was modified. Repo root: `/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/`.*

Facts re-verified for this spec (not taken from the brief): `build/include/batchlas/device_limits.hh:23,28-29` (49152 / 45056), `cmake/BatchLASDetectSYCL.cmake:57-67` (budget = local_mem − 4096), `src/backends/syrk_custom_dispatch.cc:121-201` and `src/backends/syrk_gram_tiles.hh:65,319` (syrk routing holes), `src/backends/cublas.cc:403-409,485-570,684-782` (herk gemm+fold, syrk host loop), `src/backends/cublas.cc:107-178` (`cublasGemmStridedBatchedEx` — genuinely batched for all four types), `include/batchlas/blas/matrix.hh:1128-1141` (slice **does** propagate `data_ptrs_` despite the comment saying it must not), `include/batchlas/blas/dispatch/{provider,context,env}.hh`, `include/batchlas/blas/functions/{potrf,ormqr}.hh`, `tests/options_api_tests.cc:455-514`, `src/extensions/sytrd_cta.cc:305-340`, `src/extensions/symmetric_product_fold.hh:29-72`, `src/expansion_budget.hh:26-80`, and `sycl::select_from_group` usage in `group_blas_{gemm,trmm,symm}.hh`.

---

## 1. Decision, in five sentences

The two designs are **not** alternatives: the CTA-resident kernel is the correct leaf factorisation for the blocked driver, so this spec defines one device function used at two scales, and specifies the crossover as a measured routing predicate rather than a guess. **Phase 1** ships `potrf_cta` — one matrix per sub-group (small `n`) or per work-group (larger `n`), whole triangle SLM-resident, no vendor dependency at all — covering `n ≤ 105/74/74/52` (float/double/`complex<float>`/`complex<double>`). **Phase 2** ships `potrf_blocked`, whose leaf is literally the Phase-1 device function applied to the `nb × nb` diagonal block, whose panel solve is a new global-memory kernel, and whose trailing update is **strided-batched GEMM only** — never `syrk`/`herk`, because `syrk_vendor` degenerates to a per-batch-item host loop for every non-float type above `n = 128` and for float in `m2 ∈ [129, 256]`, which would put 9 µs × batch of launch latency into the 90 %-of-flops stage [FIX-B2.1, FIX-B3.2]. Both providers ship **behind `Provider::Auto` = Vendor**: they are reachable only by `BATCHLAS_POTRF_PROVIDER=cta|blocked` or `DispatchPolicy::forced` until the §10 grid is measured, so no user-visible route changes on merge. If the first honest measurement of either path is not a win at `batch ≥ 128` at saturation, that path is deleted, not defended.

---

## 2. The algorithm and blocking, mathematically

### 2.1 One body, both triangles

Lower: `A = L Lᴴ`, `L` lower triangular, in place, strict upper untouched and never read.
Upper: `A = Uᴴ U`. This is **not a second algorithm**: `A = Uᴴ U` with `U` upper is `Ā = L Lᴴ` with `L = Uᴴ`, so `Uplo::Upper` is a load/store transform on the SLM tile and the factorisation body is compiled once per `(T, NB, TS, Scope)`, not once per `(…, Uplo)`.

| | load into SLM tile `S` (lower-stored, `LDA = n\|1`) | store back |
|---|---|---|
| `Uplo::Lower` | `S(i,c) = (i ≥ c) ? A(i,c) : 0` | `if (i ≥ c) A(i,c) = S(i,c)` |
| `Uplo::Upper` | `S(i,c) = (i ≥ c) ? conj(A(c,i)) : 0` | `if (i ≥ c) A(c,i) = conj(S(i,c))` |

For complex `T` the diagonal is loaded as `T(real(A(c,c)), 0)`, matching LAPACK/cuSOLVER ("imaginary parts of the diagonal need not be set and are assumed zero"). `conj` is identity for real `T`.

### 2.2 Right-looking blocked factorisation, block width `nb`

For `j = 0, nb, 2nb, …`, `ib = min(nb, n − j)`, `m2 = n − j − ib`:

```
       ib     m2
   ┌───────┬───────┐
ib │  A11  │       │   (P1)  A11 = L11 L11ᴴ                 unblocked, lane-per-row
   ├───────┼───────┤   (P2)  L21 = A21 L11⁻ᴴ                row-parallel substitution
m2 │  A21  │  A22  │   (P3)  A22 ← A22 − L21 L21ᴴ           TS×TS register tiles
   └───────┴───────┘
```

`A22` is fully up to date when step `j` begins, so the diagonal entry (P1) tests at global column `c` is the **fully updated Schur diagonal** — identical to LAPACK's leading-minor test, which is what makes `info` match LAPACK exactly.

### 2.3 (P1) — diagonal block, one lane per row. **The stale-pivot defect and its fix**

Both critique passes independently found the same fatal bug in the candidate: the pivot was read from SLM `Sd(k,k)`, which is written by the initial tile load and then only at iteration `k` *after* the read, so `a_kk` was the original diagonal, not `a_kk − Σ_{p<k}|l_kp|²`. Every column from 1 on was wrong, and `info` returned 0 for any matrix whose first failing leading minor is > 1 — which is every realistic Gram-matrix failure in `ortho`.

The updated diagonal **does** exist correctly: lane `k`'s register `d[k]` accumulates `d[k] -= d[p]·conj(Sd(k,p))` at every earlier step `p < k`, i.e. `a_kk − Σ_{p<k}|l_kp|²`. It is in the wrong lane, not missing. **[FIX-A1.1 / FIX-A3.1]** — broadcast it with a sub-group shuffle:

```cpp
// Lane r < ib owns row r of the ib x ib block in registers d[0..NB).
// diag[] and the fail flag live in SLM. rinv is gone -- see §7.
#pragma unroll
for (int k = 0; k < NB; ++k) {
    if (k >= ib) break;                                  // uniform: ib is uniform
    const T    dk  = sycl::select_from_group(sg, d[k], static_cast<uint32_t>(k));
    const real_t akk = real_part(dk);                    // uniform in every lane
    if (!(akk > real_t(0))) {                            // catches NaN, == LAPACK
        if (lane == 0) { slm_fail = j + k + 1; }         // 1-based GLOBAL column
        break;                                           // UNIFORM branch, no orphan barrier
    }
    const real_t dkk = sycl::sqrt(akk);
    const real_t r   = real_t(1) / dkk;                  // NOT rsqrt -- see §7
    if      (lane == k) d[k] = T(dkk);
    else if (lane >  k) d[k] = d[k] * r;
    if (lane < ib && lane >= k) Sd(lane, k) = d[k];      // publish column k
    if (lane == 0)              diag[k] = dkk;           // real; consumed by (P2)
    sycl::group_barrier(sg);                             // one barrier per k
    #pragma unroll
    for (int c = k + 1; c < NB; ++c)
        if (c < ib && lane < ib && lane >= c) d[c] -= d[k] * conj(Sd(c, k));
}
```

Four secondary fixes are folded in:

* **[FIX-A1.4 / FIX-A3.2]** the publish predicate is `lane < ib && lane >= k`, not `lane >= k`. Lanes `ib..31` hold uninitialised `d[]`; unguarded they wrote `S(j+ib .. j+31, j+k)` — i.e. into the `A21` panel (P2) is about to read, on *every* panel where `NB < 32`, and past the tile entirely on the ragged last panel, landing in a neighbouring matrix under `G > 1`. This was not the ragged-tail-only bug the candidate nominated.
* **[FIX-A1.3 / FIX-A3.3]** `fail` is an SLM word, not a register, and the branch that sets it is uniform across the sub-group (`akk` is a shuffled value; `ib` is uniform). No `rinv` array is consumed after failure because `rinv` no longer exists.
* **[FIX-A1.1b]** the `SLM broadcast + extra barrier` alternative is rejected in favour of the shuffle; the candidate's claim of "no scalar hand-off" is **retracted** — there *is* a hand-off, it is one `shfl.sync` per column, and the barrier count is one per `k`, not two. The WAR hazard that would need a second barrier does not exist: step `k+1` writes column `k+1` while step `k`'s update reads column `k`.
* **[ACCEPT-A2.2 / ACCEPT-B2.2]** (P1)'s critical path is real and is *not* 2 % of the panel. Per matrix, summed over panels, it is ≈ `n·NB/2` dependent register FMAs plus ≈ `n` sub-group barriers; at `n = 64, NB = 16` that is ~512 issues + 64 barriers against ~683 lane-issues for the ideal parallel part — roughly **40 % of the per-matrix critical path**. The candidate's `nb²` vs `m2·nb²/2` comparison compared work against work and divided only one side by `L`. This is accepted, not argued away, and it is mitigated by *throughput*, not latency: at `L = 32` every lane of the matrix's single sub-group is busy during (P1) (no idle sub-groups at all), and multiple resident work-groups cover the barrier stalls. It is the reason the routing gate is a batch threshold and the reason §10 requires `smsp__thread_inst_executed_per_inst_executed` and `sm__warps_active` across the whole `n` range, not just at saturation.

### 2.4 (P2) — the panel solve

`L21 = A21 L11⁻ᴴ`, i.e. `Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, alpha = 1`. Every **row** of `A21` is independent.

```cpp
for (int row = tid; row < m2; row += L) {
    const int i = j + ib + row;
    T x[NB];
    #pragma unroll for (int c = 0; c < NB; ++c) x[c] = (c < ib) ? S(i, j + c) : T(0);
    #pragma unroll
    for (int c = 0; c < NB; ++c) {
        if (c >= ib) continue;                       // predicate, never `break`
        T s = x[c];
        #pragma unroll for (int p = 0; p < NB; ++p)
            if (p < c) s -= mul_conj_b(x[p], Sd(c, p));   // Sd(c,p): uniform -> broadcast
        x[c] = div_by_real(s, diag[c]);              // DIVIDE, not reciprocal-multiply
    }
    #pragma unroll for (int c = 0; c < NB; ++c) if (c < ib) S(i, j + c) = x[c];
}
```

* **[FIX-A1-numerics]** `x[c] = s / diag[c]` (real divide), **not** `s * rinv[c]`. Reference `STRSM` divides; the candidate's claim to reproduce LAPACK "bit for bit" while multiplying by a precomputed reciprocal was false for the majority of scaled elements. `diag[c]` is `real_t` and the divisor of a complex numerator is written `T(re/d, im/d)`. Cost: `ib` real divides per row against `ib²/2` FMAs — irrelevant. `(P1)`'s *column* scaling keeps the reciprocal-multiply, because that is exactly what LAPACK `spotf2`'s `sscal(1/ajj)` does.
* **`NB` must be a template parameter** and both loops fully unrolled, or `x[]` acquires a dynamic index and spills. This costs performance, not correctness, so no test catches it: inspect once with `-Rpass-analysis=kernel-resource-usage` / SASS.
* **[FIX-A3.5]** the helper is named `potrf_panel_solve_rows<T, NB>` and takes `const real_t* diag`. The candidate's claim that WP3's general `trsm` would adopt it is **withdrawn**: a real diagonal array encodes a Cholesky-only assumption (`L(c,c)` real), a general `Diag::NonUnit` trsm must divide by `conj(L(c,c))`, and `Diag::Unit` must not divide at all. WP3 may copy the *loop shape*; it cannot adopt the signature.

### 2.5 (P3) — trailing update, `TS × TS` register tiles

Thread tile `TS × TS`, triangle handled at **tile granularity**:

```cpp
// Rt = ceil_div(m2, TS); Ntiles = Rt*(Rt+1)/2; off[] is an SLM prefix table (Rt <= 27 entries).
for (int t = tid; t < Ntiles; t += L) {
    const int ct = upper_bound_minus_one(off, Rt, t);     // binary search, <=5 steps
    const int rt = ct + (t - off[ct]);
    const int r0 = rt * TS, c0 = ct * TS;
    T acc[TS][TS] = {};                                   // local array, never passed by reference
    for (int k = 0; k < ib; ++k) {
        T va[TS], vb[TS];
        #pragma unroll for (int a = 0; a < TS; ++a) va[a] = (r0+a < m2) ? V(r0+a, k) : T(0);
        #pragma unroll for (int b = 0; b < TS; ++b) vb[b] = (c0+b < m2) ? V(c0+b, k) : T(0);
        #pragma unroll for (int a = 0; a < TS; ++a)
        #pragma unroll for (int b = 0; b < TS; ++b) acc[a][b] += mul_conj_b(va[a], vb[b]);
    }
    #pragma unroll for (int a = 0; a < TS; ++a)
    #pragma unroll for (int b = 0; b < TS; ++b) {
        const int ra = r0 + a, cb = c0 + b;
        if (ra < m2 && cb < m2 && ra >= cb) {
            T v = C(ra, cb) - acc[a][b];
            if constexpr (ComplexScalar<T>) if (ra == cb) v = T(real_part(v), real_t(0));
            C(ra, cb) = v;
        }
    }
}
```

* **New fix, not raised by the critiques:** the candidate's tile-index inverse used `sycl::sqrt(double(...))` plus a `while` fixup. That is a floating-point inversion of an integer map with a hand-written correction loop — a latent correctness hazard for no gain. Replaced by a **binary search over an SLM prefix table `off[ct] = ct·Rt − ct(ct−1)/2`** (`Rt ≤ 27` for every resident `n`, so the table is ≤ 108 bytes and the search is ≤ 5 steps amortised over `ib·TS² ≥ 128` FMAs).
* `mul_conj_b(a,b) = (aᵣbᵣ + aᵢbᵢ, aᵢbᵣ − aᵣbᵢ)` written longhand: `std::complex operator*` emits an isnan branch and a `__mulsc3` call in device code. Convert this loop and (P2)'s inner loop only.
* The hermitian diagonal is forced real here. This is the line most easily omitted and it is why test T7 exists.
* **`device::herk` is retained as the differential oracle**, not as a shipping path: `BATCHLAS_POTRF_UPDATE=herk` swaps (P3) for `device::herk<Uplo::Lower, Transpose::NoTrans>(g, V, C, T(-1), T(1), nullptr)` on the same SLM views (verified: a bare `sycl::group` is not `NdItemLike` per `group_blas_common.hh:181-187`, so dispatch is deterministically `generic::rankk`, and `herk_workspace_elements` returns 0 for a `Group` launch). It costs ~15 lines and turns the tiled kernel into an A/B against a primitive already in production in `sytrd_blocked.cc:242-285`. `generic::rankk` does 2 SLM loads per FMA and has **no barriers of its own** — see §3.

### 2.6 Blocked driver (Phase 2), and why its trailing update is GEMM

Outer width `NB_o`; per panel `j`: leaf = §2.3–2.5 device function on the `ib × ib` diagonal block (SLM-resident, one kernel); panel solve = global-memory kernel (§3.4); trailing update over `A22` (`m2 × m2`).

The trailing update is **not** `syrk`/`herk` **[FIX-B2.1, FIX-B3.2]**. Measured from the source:

* `syrk_vendor` for any `T != float` reaches only `syrk_gram_tiles`, gated `C.rows() ≤ kGramMaxTile = 128` (`syrk_gram_tiles.hh:65,319`); everything else falls to `syrk_vendor_impl`'s `for (batch) launch_single(...)` host loop (`cublas.cc:745-750`), priced in-tree at ~9 µs/launch.
* For `float`, `syrk_use_cuda_custom` is the OR of `C.rows() ≤ 128`, `triangular_tiles_per_side(n) ≥ 3 && k ≥ 8` (needs `m2 ≥ 257`), and `min·2 ≥ max` — all three false for `m2 ∈ [129, 256]` at `k = nb ≤ 64`, which every blocked run passes through on its way down.
* `herk` has no batched cuBLAS call at all; its GEMM+fold path leases `expanded_workspace_bytes` from the **queue arena invisibly** (`cublas.cc:538-545`), so `potrf_buffer_size` would stop bounding the memory `potrf` draws, and it is capped at `n ≤ 768` (`cublas.cc:403-409`).

Instead, for each column panel `[c0, c0+W)` of `A22` (`W = kPotrfFoldTile = 128`):

1. **Sub-diagonal rectangle** — rows `c0+W … m2`, cols `c0 … c0+W`. This rectangle lies **entirely inside the lower triangle**, so it is a plain strided-batched GEMM straight into `A`, no scratch, no fold, no waste:
   `gemm<B>(ctx, L21_rows, L21_cols, A22_block, {.alpha = T(-1), .beta = T(1), .transA = NoTrans, .transB = ConjTrans})`.
2. **Diagonal block** — `W × W`. GEMM into pool scratch, then `detail::fold_symmetric_product_into_triangle(ctx, C_blk, product, T(1), uplo)` (`symmetric_product_fold.hh:29-72`, which already guards `total_elements == 0` and takes `beta` so an uninitialised `C` cannot NaN).

Consequences, all of them good: every launch is `cublasGemmStridedBatchedEx` (`cublas.cc:137`) and therefore genuinely batched for all four scalar types; the only redundant arithmetic is the discarded half of each diagonal block, `W/(2·m2)` of the update (12.5 % at `m2 = 512`), not the 100 % a naïve gemm-into-scratch costs; scratch is `W²·batch·sizeof(T)` — **bounded independent of `n`** (67 MB at `W=128, batch=512, double`) and drawn from the caller's pool so `potrf_buffer_size` bounds it; and `BATCHLAS_SYRK_VARIANT` can no longer route us into `syrk_cublasdx_fallback_gemm`, which writes **both** triangles [FIX-B1-secondary].

Sub-views for the GEMM operands are constructed **explicitly** from `A.data_ptr() + off`, parent `ld`/`stride`, `data_ptrs = nullptr` — never `MatrixView::operator()(Slice, Slice)`. Verified at `matrix.hh:1128-1141`: the comment says the parent pointer array must not propagate, and the very next line passes `data_ptrs_.data()` anyway; `ortho.cc:75` builds `C` *with* a pointer array, so a slice of it carries stale base addresses [FIX-B-trap].

`m2 == 0` on the last panel skips (P2), the panel-solve launch and the trailing update entirely [FIX-B1-secondary, FIX-B2.5] — `accumulate_hermitian` does not guard a zero extent and a zero-extent `nd_range<3>` is not benign.

**V2 (two-level blocking) is out of scope.** It is named here only so nobody re-derives it: `nb_outer = 4·nb_inner`, outer diagonal block recursing into the same driver. Trigger is a *measured* miss of the §10 gate at `n ≥ 1024`, not taste.

---

## 3. Parallel decomposition and the exact nd_range

### 3.1 Constants

| `T` | `sizeof(T)` | `TS` | `acc` regs | `NB` ladder | default `NB` | notes |
|---|---|---|---|---|---|---|
| `float` | 4 | 4 | 16 | {8, 16, 32} | 16 | |
| `double` | 8 | 4 | 32 | {8, 16} | 16 | |
| `complex<float>` | 8 | 4 | 32 | {8, 16} | 16 | |
| `complex<double>` | 16 | 2 | 8 | {8} | 8 | |

**[FIX-A2.3b]** `NB` narrows with `sizeof(T)` exactly as `TS` does. The candidate left `NB ∈ {4,8,16,32}` type-independent; `x[32]` for `complex<double>` is 32 × 16 B = **128 registers for one array** and spills into the same SLM the tile is saturating. The budget is 32 32-bit registers for `x[NB]` and ≤ 32 for `acc[TS][TS]`; the two live ranges are disjoint ((P2) and (P3) never overlap), so peak is `max(NB·sizeof(T)/4, TS²·sizeof(T)/4 + 2·TS·sizeof(T)/4)`.

### 3.2 CTA kernel nd_range as a function of `(n, batch)`

```
nb      = resolve_potrf_nb<T>(n, hint)          // clamped into the ladder above
m2_0    = n - min(nb, n)                        // FIRST TRAILING UPDATE, not n
Rt_0    = ceil_div(m2_0, TS)
Ntiles_0= Rt_0*(Rt_0+1)/2

L (work-items per matrix) =  32   if Ntiles_0 <=  64
                             64   if Ntiles_0 <= 256
                            128   otherwise                       // hard cap 128

G (matrices per work-group) = (L == 32)
                            ? clamp(prev_pow2(kSlmSoftTarget / slm_per_matrix), 1, 8)
                            : 1                                   // kSlmSoftTarget = 24576
wg_size = G * L                                                   // <= 256
num_wg  = ceil_div(batch, G)                                      // == batch when G == 1

cgh.parallel_for<PotrfCtaKernel<T, NB, TS, Scope>>(
    sycl::nd_range<1>(sycl::range<1>(num_wg * wg_size), sycl::range<1>(wg_size)),
    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] { ... });
```

**[FIX-A2.1]** `L` is derived from `m2_0 = n − nb`, the actual first trailing update — the candidate used `Rt_0 = ceil(n/TS)`, i.e. a triangle that is never updated, which guaranteed `Ntiles < L` in *every* panel and made its own anti-starvation argument arithmetically backwards. The cap at 128 is new: `L = 256` costs `1536/256 = 6` blocks/SM for tile counts that never materialise. Worked values (float, `TS=4`, `nb=16`): `n=32 → Ntiles_0=10 → L=32`; `n=48 → 21 → L=32`; `n=64 → 78 → L=64`; `n=105 → 276 → L=128`.

**[FIX-A3.3b]** `[[sycl::reqd_sub_group_size(32)]]` is mandatory — the lane-per-row (P1) and the `G` packing both hardcode a 32-wide sub-group and the candidate carried no such attribute, unlike every comparable kernel here (`syev_cta_fused.cc:185`, `gesvdj_cta.cc:297`, `sytrd_sb2st_cta.cc:403`). The host gate additionally enumerates `device::sub_group_sizes` and requires **32 to be present** — copying `sytrd_cta.cc:319-333` verbatim — rather than testing `caps.max_sub_group < 32`, which is a *maximum* and is 64 on ROCm, where a 128-item work-group holds two 64-wide sub-groups and `G = 4` would silently map four matrices onto two of them.

### 3.3 Synchronisation scope — the second fatal defect, fixed

All three critiques of the CTA design found it: the candidate specified `group_barrier(sg)` as its only synchronisation while `G = 1, L = 64…256` puts 2–8 sub-groups on one matrix, so (P1)→(P2)→(P3)→next-(P1) were straight races, and `generic::rankk` (`group_blas_rankk.hh:49-81`) and `device::fill` contain no barriers of their own. **[FIX-A1.2 / FIX-A2.2b / FIX-A3.2b]** The kernel body is templated on `Scope`:

| `Scope` | when | `sync()` is | matrix id | out-of-range item |
|---|---|---|---|---|
| `SubGroup` | `L == 32`, `G ≥ 1` | `sycl::group_barrier(it.get_sub_group())` | `wg_id·G + sg_id` | early `return` is legal (partitions are independent — `sytrd_cta.cc:101-111` precedent) |
| `WorkGroup` | `L > 32`, `G == 1` | `sycl::group_barrier(it.get_group())` | `wg_id` | cannot occur: `num_wg == batch` |

Barriers per panel, explicitly: **(a)** after the zero-fill + triangular load, **(b)** after (P1), **(c)** after (P2), **(d)** after (P3) before the next panel. Under `Scope::SubGroup` these are sub-group barriers and (P1)'s internal per-`k` barrier is the same object. Under `Scope::WorkGroup`, (P1) runs in sub-group 0 with its own per-`k` `group_barrier(sg)` (legal — only that sub-group's items enter) and every other sub-group waits at barrier **(b)**. The candidate's "**zero barriers**" claim for (P2) is **retracted**; it was only ever true of (P2)'s *interior*.

The fill/load race is also closed: `device::fill` and the triangular load use different index maps, so a fill store can land after a load store. Specify a **single fused load loop** that writes every element of the `LDA × n` tile exactly once (triangle value or `T(0)`), with no separate fill pass.

**[FIX-A1.3b]** Failure propagation is work-group-visible: `slm_fail` is written under barrier (b) and read by every work-item after it; (P2), (P3) and every later panel are wrapped in `if (slm_fail == 0) { … }` — a **predicated skip**, never a `return`, so the phase barriers are still reached by every work-item under `Scope::WorkGroup`. Under `Scope::SubGroup` a `return` would also be safe but the predicate is used in both for one code path.

### 3.4 Blocked driver kernels

**Leaf** — the CTA kernel launched on the `ib × ib` diagonal sub-view, at `Scope::SubGroup` with `G` matrices per work-group. **[FIX-B2.2]** The candidate blocked design used one matrix per 32–64-thread work-group and defended it as "0.4 % of flops" — a flop fraction, not a time fraction, at ~4 % of the machine. Reusing the packed CTA kernel makes the leaf's occupancy identical to the Phase-1 kernel's, which is the whole point of the composition. Packing is safe here precisely because the failure path is a predicated skip and not a `return` (§3.3), which is also why **[FIX-B2.3]** the candidate's appeal to `group_reduce_sum_select_from_group` is dropped: that function is built on `permute_group_by_xor(SubGroupPartition<P>, …)` (`sg_compat.hh:104-123`) and **cannot cross sub-groups**, so it could not have made a work-group-wide flag uniform. Uniformity here comes from a barrier plus an SLM read, which is uniform by construction.

**Panel solve** (`PotrfPanelSolveKernel<T, NB_o, Uplo::Lower>`), `TR = 256`:

```
tiles  = ceil_div(m2, TR)                      // driver skips the launch when m2 == 0
nd_range<1>( range<1>(batch * tiles * TR), range<1>(TR) )
b = wg / tiles ;  t = wg % tiles ;  row = t*TR + lid   (guard row < m2)
if (info[b] != 0) { zero_quench(); return; }   // UNCONDITIONAL -- see §5.4
```

1-D, not `nd_range<3>`: the 3-D group-dim tile decomposition brings the "generic fallback must be restricted to group (0,0) or work-groups race" trap (`group_blas_gemm.hh:408-416`) for no benefit. `L11` is staged into SLM as a packed lower triangle, `idx(j,p) = j(j+1)/2 + p` (valid because access is only ever `p ≤ j`); all threads read the same `L11(j,p)` → broadcast, no bank conflicts. Column-major `A` with thread-per-row → fully coalesced read and write. Short final panel: pad `L11` to the **identity** beyond `ib` (diagonal 1, off-diagonal 0) and zero `x[j]` for `j ≥ ib`; padding the diagonal with zeros instead manufactures a spurious `info` failure.

`NB_o` (outer width) per type, ladder `{16, 32, 64}` closed, no runtime `-1` fallback: float 64, double 32, `complex<float>` 32, `complex<double>` 16 — the same 64-register accumulator budget as §3.1. `float, NB_o = 64` sits exactly on the documented spill edge and **must be confirmed spill-free before any other measurement**.

**Upper** in the blocked driver is a genuinely different kernel, not a flag [FIX-B3.3]: the panel is `A12` (`ib × m2`), the solve is `Side::Left` and the independent unit is a *column*, whose elements are `ld` apart — the candidate asserted the Lower access pattern ("consecutive threads, consecutive addresses") for a kernel it templated on `UPLO`. The Upper variant stages an `ib × Wc` tile through SLM (coalesced *down* columns, `ib ≥ 32`), solves column-per-thread in SLM, writes back coalesced. It is **Phase 3**; until it lands `potrf_supports_blocked` returns false for `Uplo::Upper` **unconditionally** — not conditioned on `vendor_available`, which made the Upper path unreachable-therefore-untested in every real build.

### 3.5 Why it does not starve at small batch — and where it honestly does

Total concurrency is `batch × L` (CTA) or `batch × ceil(m2/256) × 256` (blocked panel), never `batch` alone. `L` is derived from the actual first trailing update (§3.2) so the largest (P3) presents at least ~`L` tiles. At `batch = 1, n = 64` there are still 64 work-items doing intra-matrix work; at `n = 32, batch = 128` there are 128/G ≈ 32 work-groups of 128 threads.

Three honest starvation terms, all in `n`, all bounded, all to be **profiled** rather than argued:

1. **(P1) idles `L/32 − 1` sub-groups** under `Scope::WorkGroup` (i.e. `n ≳ 56`), and is ~40 % of the per-matrix critical path (§2.3). Zero idle under `Scope::SubGroup`.
2. **Panel tail**: as `j` grows, `Ntiles` falls below `L`. Look-ahead would fix it and is out of scope for v1.
3. **Blocked last panels**: `ceil(m2/256)` decays to 1, so the panel kernel becomes `batch` work-groups — in the panel where the work is proportionally tiny.

**[ACCEPT-A2.4]** `batch ≥ 128` does **not** fill this machine at `n = 64` (you need ~`128 SMs × 4–6 blocks` of concurrent matrices), and `ortho` legitimately calls `potrf` with `batch = 1` for single-matrix LOBPCG. That is why the gate threshold is a **measured** number in §10 and not `128` by assertion, and why `batch = 1` stays on the vendor.

---

## 4. Local memory budget, resident `n`, occupancy on sm_89

### 4.1 Budget

```
LDA            = n | 1                              // odd: stride-LDA row reads are conflict-free
slm_per_matrix = LDA*n*sizeof(T) + NB*sizeof(real_t) + 64      // tile + diag[] + fail/pad
slm_per_wg     = G * slm_per_matrix
slm_budget     = runtime_local_mem_size - 4096                 // = 45056 on this box
```

**[FIX-A2.3a / FIX-A3.4]** The budget is **45056**, not 49152. `cmake/BatchLASDetectSYCL.cmake:57-67` defines `subgroup_workspace_budget_bytes = local_mem − 4096` for GPUs and every existing device-BLAS sizing decision uses it (`group_blas_subgroup_common.hh:60-61`). The candidate's ceilings (110/77/77/55) all exceed it. The gate uses the **runtime** `q.device().get_property(DeviceProperty::LOCAL_MEM_SIZE)` minus the same 4096 reserve, not the configure-time `device_limits.hh` constant, which is a minimum over detected devices and need not describe the device actually in use. The 99 KB sm_89 opt-in carveout is not exposed through SYCL `local_accessor`, so 48 KB stays the hard per-work-group ceiling and 45056 is what we spend.

### 4.2 Largest resident `n` per type (`G = 1`)

| budget/WG | `float` | `double` | `complex<float>` | `complex<double>` |
|---|---|---|---|---|
| **45056 (fit ceiling, `potrf_cta_max_n<T>()`)** | **105** | **74** | **74** | **52** |
| 25600 (≥4 blocks/SM) | 79 | 56 | 56 | 39 |
| 17066 (≥6 blocks/SM) | 65 | 45 | 45 | 32 |
| 12800 (≥8 blocks/SM) | 56 | 39 | 39 | 27 |

Checks: float `n=105` → `105·105·4 = 44100` ✓, `n=106` → `107·106·4 = 45368` ✗. double `n=74` → `75·74·8 = 44400` ✓, `n=75` → `45000 + 192 > 45056` ✗. `complex<double>` `n=52` → `53·52·16 = 44096` ✓, `n=53` → `44944 + 192 > 45056` ✗.

### 4.3 Occupancy

`blocks/SM = min( floor(shared_per_SM / slm_per_wg), floor(1536 / wg_size), floor(65536 / (regs_per_thread · wg_size)), 24 )`.

**[FIX-A2.3b / FIX-B2.4]** Three corrections to the candidates' tables. (i) `shared_per_SM ≈ 102400` on sm_89 — the per-**SM** pool, not the per-**block** cap; the blocked candidate computed `49152 / 8320 = 5` blocks/SM, which divides a per-block limit by per-block usage and is not an occupancy calculation. (ii) A **register term** is present: the CTA kernel's `max(NB, TS²+2TS)` accumulators plus addressing, loop counters and the divide temporary put it realistically at 64–96 registers/thread, so `65536/(80·256) = 3` blocks at `wg_size = 256`, not 6. (iii) `shared_per_SM` **must be read from `ncu --metrics launch__occupancy_limit_shared_mem, launch__registers_per_thread`** before any table in this section is treated as fact.

Resulting expectations (float, `TS=4`, `NB=16`): `n=32, G=4, wg=128, slm/wg = 17.7 KB` → 5 blocks/SM = 20 matrices/SM, 640 threads. `n=64, G=1, L=64, slm/wg = 16.7 KB` → 6 blocks SLM-limited, ~4 register-limited = 256 threads/SM. `n=105, G=1, L=128, slm/wg = 44.3 KB` → **2 blocks/SM**, almost no latency hiding.

**The fit ceiling is not the useful ceiling**, and this is stated as a design property rather than discovered later: `n = 105` float *fits* and will very plausibly lose to cuSOLVER. §10's grid is written so that the fit test is a hard gate and the useful ceiling is a measured artefact recorded in `tuning_params.hh`.

The blocked panel kernel: packed triangle `NB_o(NB_o+1)/2 · sizeof(T)` = 8320 B (float, `NB_o=64`), 4224 B (double, 32), **4224 B** (`complex<float>`, 32 — the blocked candidate's table used `sizeof(complex<double>)` here and printed 8448 [FIX-B1-secondary]), 2176 B (`complex<double>`, 16). SLM is not the limit there; registers are.

---

## 5. Exact public contract

### 5.1 Signature — unchanged, byte for byte

```cpp
template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

template <Backend B, typename T>
Event  potrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& descrA,
             Uplo uplo, Span<std::byte> workspace, Span<int32_t> info);
```
(`include/batchlas/blas/functions/potrf.hh:28-49`, plus the 4-arity forwarder at `:56-62` and the `Matrix<T>` overloads at `:64-84`.) Option-struct spellings (`options.hh:539-568`) and the `= delete` bare-`{}` guard (`options.hh:570-595`) are untouched: `potrf_dispatch` is a distinct name behind the existing entry points, so **no new same-arity positional overload is introduced** and the `EmptyBracesAreAmbiguous` guard needs no sibling.

### 5.2 Semantics table

| aspect | contract | this implementation |
|---|---|---|
| in-place | `A` is overwritten by its factor; there is no output matrix | same |
| batch | uniform `n`, `ld`, `stride`; `batch_size() ≥ 1` | addressed through `stride`; `data_ptrs(ctx)` is **never** consulted by either native provider |
| workspace | same size with or without `info` (`potrf.hh:40-43`) | same (§5.5) |
| `alpha`, `Side`, `Diag`, `Transpose` | **potrf has none** | n/a; the internal solves are tabulated in §5.4 |
| exceptions | none for a non-PD input; no failed event | same |

### 5.3 Every `Uplo` combination, enumerated

| `Uplo` | reads | writes | produces | other triangle | diagonal | supported by |
|---|---|---|---|---|---|---|
| `Uplo::Lower` | lower incl. diagonal of `A` | lower incl. diagonal | `A = L Lᴴ`, `L` lower | **not read, not written, bit-identical on return** | real; for complex `T`, `imag` of the input diagonal is ignored and `imag` of the output diagonal is exactly `0` | CTA (all `n ≤ ceiling`), Blocked (Phase 2), Vendor |
| `Uplo::Upper` | upper incl. diagonal of `A` | upper incl. diagonal | `A = Uᴴ U`, `U` upper | **not read, not written, bit-identical on return** | as above | CTA (all `n ≤ ceiling`, free via §2.1's transform), Vendor. Blocked: **Phase 3**; until then `potrf_supports_blocked` returns false for Upper unconditionally |

"Not read" is stronger than "not written" and is load-bearing: `ortho.cc:156-161` forms only the lower triangle of the `k×k` Gram matrix and the other half is **uninitialised workspace**. Test T2 poisons it with NaN specifically to prove the read never happens.

### 5.4 Every triangular-solve combination the implementation contains

`potrf` exposes no `Side`/`Trans`/`Diag`, but the internal solves do, and enumerating them is what lets WP3's `trsm` land as a substitution rather than a redesign. All have `alpha = 1` (the right-hand side is not scaled; `alpha` in `TrsmOptions` sits in position 4 to match `trmm`, `trsm.hh:87-95`).

| Side | Uplo | Transpose (real / complex) | Diag | used by | implemented in |
|---|---|---|---|---|---|
| Right | Lower | `Trans` / `ConjTrans` | NonUnit | (P2), Lower CTA; blocked Lower panel solve | `potrf_panel_solve_rows<T,NB>` (§2.4), `PotrfPanelSolveKernel` (§3.4) |
| Left | Upper | `Trans` / `ConjTrans` | NonUnit | blocked Upper panel solve (`U11ᴴ X = A12`) | `PotrfPanelSolveKernelUpper` (Phase 3) |
| Right | Lower | `NoTrans` | NonUnit | — | **not implemented** |
| Right | Upper | `NoTrans` / `Trans` / `ConjTrans` | NonUnit | — | **not implemented** |
| Left | Lower | `NoTrans` / `Trans` / `ConjTrans` | NonUnit | — | **not implemented** |
| Left | Upper | `NoTrans` | NonUnit | — | **not implemented** |
| any | any | any | **Unit** | — | **not implemented**; `Diag::Unit` never divides and these helpers always divide |

Conjugation, spelled out: for complex `T` the (P2) inner product is `s -= x[p] · conj(L11(c,p))` and the divisor is `diag[c]`, which is **real** by construction (it is `sqrt` of a real). For real `T` every `conj` is identity and the `Transpose` column above reads `Trans`. (P3) computes `C -= V Vᴴ` (`mul_conj_b`), and the hermitian diagonal is forced real on every write.

### 5.5 `potrf_buffer_size`

```cpp
template <typename T> Span<int32_t> potrf_cta_layout(Queue& ctx, BumpAllocator& p, int batch) {
    return p.allocate<int32_t>(ctx, batch);                       // info fallback only
}
template <typename T> auto potrf_blocked_layout(Queue& ctx, BumpAllocator& p, int n, int batch) {
    auto info = p.allocate<int32_t>(ctx, batch);                  // info fallback
    const int W = std::min(kPotrfFoldTile, std::max(1, n));       // kPotrfFoldTile = 128
    auto prod = p.allocate<T>(ctx, size_t(W) * W * batch);        // diagonal-block fold scratch
    return std::pair{info, prod};
}
```
Both replayed by `workspace_bytes(...)` (`mempool.hh:185-190`) for the query and against the caller's span for the run — one layout function, two passes, per `sytrd_blocked.cc:655-687`.

`potrf_buffer_size` calls the **same** `choose_potrf_provider` the call uses and returns `max(chosen_provider_size, vendor_size)`. Three properties this buys, each addressing a specific critique:

* **[FIX-B3.2b]** Every byte the blocked driver spends is in this number. The candidate's "byte-identical to the vendor" claim was false the moment the trailing update was `herk`, which leases `expanded_workspace_bytes` from the queue arena invisibly (`cublas.cc:538-545`) — up to 604 MB at `complex<float>, m2 = 768, batch = 128`, against a reported 512 B. Our GEMM+fold draws from the caller's pool.
* Neither native size depends on `nb`, `NB_o`, `L` or `G`, so `BATCHLAS_TUNE_POTRF_*` cannot desynchronise the query from the call. **Preserve this property**: if a future variant needs `nb`-dependent scratch, the `tuning_params.hh:27-32` hazard returns and the doubled-ladder discipline of `sytrd_blocked.cc:978-990, 1008-1020` becomes mandatory.
* The residual hazard is `BATCHLAS_POTRF_PROVIDER` changing between query and call (vendor → blocked). The failure is **loud**: `BumpAllocator`'s capacity check rejects the second allocation. Document it next to the query; do not paper over it.

### 5.6 Failure reporting — `info`

| case | behaviour |
|---|---|
| success | `info[i] = 0`, written for every item, always |
| non-PD | `info[i] = j0 + k + 1`, the **1-based global** column at which the updated diagonal was not `> 0` (`!(d > 0)`, which also catches NaN — LAPACK's `AJJ.LE.ZERO .OR. SISNAN(AJJ)`) |
| multiple failures in one item | **first wins** (sticky), enforced by the guard in §5.7 |
| empty `info` span | "not requested": `detail::info_target(ctx, pool, info_out, batch)` (`src/linalg-impl.hh:714-718`) returns pool scratch and the status is discarded. Workspace size is unchanged either way |
| a failed item's `A` | **undefined** — same as LAPACK and cuSOLVER. See §5.8 |
| exceptions / event | none, ever |

### 5.7 The `info` protocol across multiple kernel launches (blocked only)

`info` becomes an **input as well as an output**, which no existing kernel in this tree does. Three rules:

1. **Zeroing is a separate pre-pass**, before the panel loop — a trivial `parallel_for` (or `ctx->fill`) over `batch`. `info_target`'s pool branch returns **uninitialised** memory; if it is not zeroed, every kernel takes the "already failed" path and `potrf` returns `A` **unmodified with `info == 0` and no error**. The existing `options_api_tests.cc:507-514` passes a real span and would not see it. Test T8 exists solely for this.
2. **Every kernel's guard is unconditional: `if (info[b] != 0) return;`** — never `if (j0 > 0 && info[b] != 0)`. **[FIX-B1.1 / FIX-B3.1]** All three blocked-design critiques found this independently: the `j0 > 0` clause existed because the candidate zeroed `info` *inside* the `j0 == 0` leaf, but it was then copied into the panel-solve kernel, which runs *after* that leaf, so the `j0 == 0` panel had no guard and divided by the very pivot the leaf had just rejected — exactly the `d == 0` rank-deficient Gram case that `tests/cond_tests.cc:371-380` records. Moving the zeroing to a pre-pass makes the unconditional guard correct everywhere.
3. **Only the leaf writes `info`**, one work-group per item per panel → exactly one writer, no atomics. The diagonal test is uniform by construction (§2.3), and the failure exit is a predicated skip, not a `return` from a divergent branch, so no barrier is ever orphaned. `Queue` is in-order by default; the driver **asserts** `ctx.in_order()` (or carries the event chain) rather than assuming it — `DispatchPolicy::require_in_order` is the existing hook.

### 5.8 What happens to a failed item's `A` — the "finite, not NaN" claim, withdrawn and then re-earned

**[FIX-B1.2 / FIX-B3.1b]** The blocked candidate claimed three times that a failed item's `A` comes out *finite*. That is false as it stood: the shared trailing update cannot skip a failed item, so with `m_j` the largest magnitude in the trailing block and inner dimension `nb`, `m_{j+1} ≈ nb·m_j²` compounds quadratically — from `m_0 = 10²` at `nb = 64`, float overflows by panel 4–5, and the next update computes `Inf − Inf = NaN`, over essentially the whole claimed `n` range.

Resolution: a **zero-quench**, three lines, which makes the claim true instead of deleting it. When the panel-solve kernel sees `info[b] != 0` it **writes zeros to its output rows** and returns. `L21 = 0` makes the trailing GEMM a no-op on that item, so `A22` stays exactly as it was; the remaining panels' leaves skip on the same flag. The result: a failed item's `A` is undefined per contract but **finite and bounded by the input**, and no `sqrt` of a negative and no division by a non-positive pivot ever executes.

Callers must not depend on this — the contract says undefined — but it is asserted by test T9, which is the test the candidates lacked entirely (nothing in either §9 checked a *failed* item's `A` at all).

The downstream consequence documented at `tests/cond_tests.cc:371-380` (ortho discards `info`, trsm back-substitutes through a garbage diagonal, two gemms smear NaN) therefore changes shape on the native path: the smear becomes finite garbage. `cond_tests` asserts orthogonality, not NaN-ness, so it is unaffected; the difference is noted in the coverage table.

### 5.9 Cross-provider `info` is zero/non-zero only

**[FIX-B1.3]** `tests/options_api_tests.cc:463-466` states the API's position explicitly: *"Values are asserted only as zero / non-zero. LAPACK, cuSOLVER, cuBLAS and rocSOLVER agree on the sign convention but not always on which index they name first, and the API's contract is the zero/non-zero distinction."* T6 (vendor cross-check) therefore asserts **zero/non-zero agreement only**. The exact index is pinned in T5, which is self-referential (our implementation against a planted `LDLᴴ` pivot). §7 explains why an exact cross-provider match is not even achievable in principle near the `κ²u` cliff.

---

## 6. Reused primitives, by exact verified signature

Every signature below was read in-tree for this spec.

| symbol | file:line | role |
|---|---|---|
| `template <Uplo UploV, Transpose TransV, typename Group, typename T> void herk(const Group&, const KernelMatrixView<T,Dense>& a, RankKOperand<T>, T* ws = nullptr)` and the `(a, c, alpha, beta, ws)` convenience | `group_blas_rankk.hh:528-534, 565-573` | (P3) **differential oracle** under `BATCHLAS_POTRF_UPDATE=herk`. One spelling covers real and complex (`hermitian = ComplexScalar<T>`); `rankk_rhs_transform` returns `ConjTrans` for hermitian/NoTrans (`group_blas_common.hh:835-841`); it forces `imag(diag)=0` (`rankk.hh:71-79`) |
| `template <typename T, Uplo, Transpose> size_t herk_workspace_elements(const DeviceBlasLaunchInfo&, int extent, int contract_extent)` | `group_blas_rankk.hh:548-563` | host-side query; **asserted == 0** for a `Group` launch (every fast path gates on `is_nd_item_3d_launch`, `subgroup_common.hh:336-338`, and `make_group_launch_info` sets `kind = Group`) |
| `inline constexpr DeviceBlasLaunchInfo make_group_launch_info(int local_size)` | `group_blas_common.hh:40` | host stand-in for the device `Group` |
| `template <typename Tag> constexpr bool triangular_storage_contains(int row, int col)` | `group_blas_common.hh:667-677` | triangle predicate available for the (P3) epilogue (`row >= col` for Lower) |
| `KernelMatrixView<T, MatrixFormat::Dense>(T* p, int rows, int cols, int ld, int stride)` | `matrix.hh:243-247` | views built directly over SLM, per `sytrd_blocked.cc:540-548` |
| `sycl::select_from_group(sg, value, uint32_t lane)` | in use at `group_blas_gemm.hh:77`, `trmm.hh:118,132`, `symm.hh:81,92` | the (P1) pivot broadcast |
| `detail::info_target(Queue&, BumpAllocator&, Span<int32_t>, int count)` | `src/linalg-impl.hh:714-718` | empty-span-means-not-requested |
| `BumpAllocator::measuring()`, `required_bytes()`, `workspace_bytes(Fn&&)` | `util/mempool.hh:38, 52-58, 185-190` | the two-pass workspace protocol |
| `detail::fold_symmetric_product_into_triangle<T>(Queue&, const MatrixView<T,Dense>& C, const MatrixView<T,Dense>& product, T beta, Uplo)` | `src/extensions/symmetric_product_fold.hh:29-72` | blocked diagonal-block fold; guards `total_elements == 0`; `beta == 0` means `C` is not read |
| `gemm<B>(ctx, A, B, C, GemmOptions<T>{...})` → `cublasGemmStridedBatchedEx` | `options.hh:179-185`; `cublas.cc:107-178` | blocked trailing update, genuinely batched for all four types |
| `choose_*_provider` / `policy_from_env` / `query_caps` / `DispatchPolicy` | `functions/ormqr.hh:161-176`; `dispatch/{env,context,provider}.hh` | routing scaffolding; `default_order_cta_blocked_vendor_netlib` at `env.hh:57-64` is reused as-is — **no new `std::array<Provider,6>`** |
| `tuning_env_override(const char*, int32_t)` | `tuning_params.hh:33-40` | env overrides for `nb`, `L`, `G` |
| device sub-group-size-32 presence check | `sytrd_cta.cc:319-333` | copied verbatim into the CTA gate |
| `util::get_raw_ptr(local_accessor)` | `sycl-local-accessor-helpers.hh:23` | accessor → `T*` |

**Deliberately not used:** `device::fill` on the SLM tile (replaced by a single fused load loop, §3.3 — the two index maps race); `device::gemm` for any trailing update (every level-3 fast path is `float`-only or `complex<float>`-only, `subgroup_common.hh:433-452`, so `double`/`complex<double>` would land on the generic scalar loop); `group_reduce_sum_select_from_group` (`sytrd_cta_device.hh:79` — sub-group-partition scope only, cannot cross sub-groups); `syrk`/`herk` at host level (§2.6); `MatrixView::operator()(Slice,Slice)` for anything that may reach a pointer-array backend (`matrix.hh:1128-1141`).

### New device code to write

1. `potrf_diag_block_subgroup<T, NB>(sg, T* sd, int lda, int ib, int j, real_t* diag, int32_t& slm_fail)` — §2.3.
2. `potrf_panel_solve_rows<T, NB>(sync, tid, L, T* S, int lda, int j, int ib, int m2, const real_t* diag)` — §2.4.
3. `potrf_trailing_tiles<T, TS>(sync, tid, L, T* S, int lda, int j, int ib, int m2, const int* off)` — §2.5.
4. `potrf_cta_body<T, NB, TS, Scope>` — load/store transform (§2.1), panel loop, barriers, failure predicate.
5. `PotrfLeafKernel` / `PotrfPanelSolveKernel` / `PotrfPanelSolveKernelUpper` / `PotrfInfoZeroKernel` — blocked driver kernels.
6. Host launchers, provider choosers, tuning entries.

---

## 7. Numerical stability

**The bound.** Cholesky is unconditionally backward stable: the computed `L̂` satisfies `L̂ L̂ᴴ = A + ΔA` with `|ΔA| ≤ γ_{n+1} |L̂||L̂ᴴ|`, `γ_k = ku/(1−ku)` (Higham, *ASNA* Thm 10.3). The bound depends on the **length of the longest inner product, not on the ordering**, so right-looking, left-looking, and blocked at any `nb` all satisfy it with the same constant. The panel solve is substitution, so each computed `L21` row satisfies `(L11 + ΔL11)ᴴ x̂ᵀ = a` with `|ΔL11| ≤ γ_{ib} |L11|` (*ASNA* Thm 8.5), which composes into the same `γ_{n+1}` overall. `nb`, `NB_o`, `TS`, `L` and `G` are pure performance parameters that do not move the **bound**.

**Explicit inversion is rejected on purpose.** Inverting `L11` and making the panel a plain GEMM would run at near-full efficiency, but its forward error carries `κ(L11) = κ(A11)^{1/2} ≤ κ(A)^{1/2}` (Cauchy interlacing). `ortho`'s Cholesky-QR already squares the condition number forming the Gram matrix — that is precisely why `ShiftChol3` exists — so `κ(A)` near `1e8` in single precision is the *normal* operating point. Buying ~5–10 % of runtime for lost digits there is a bad trade.

**Five differences from LAPACK, each checkable rather than trusted:**

1. **Rank-1-update ordering** in (P1) vs LAPACK `potf2`'s dot-product form: different summation order, same bound. Results will **not** be bit-identical to cuSOLVER. Every test asserts a residual norm; none compares entries.
2. **Reciprocal in (P1), division in (P2).** (P1)'s column scaling is `d[k] * (1/dkk)`, exactly `spotf2`'s `sscal(one/ajj, …)`. (P2) **divides** by `diag[c]`, exactly reference `STRSM`'s `B(i,j)/A(j,j)`. The candidate multiplied by a reciprocal in both and claimed LAPACK equivalence for both, which was false for the majority of scaled elements (it would have cost `γ_{n+2}` instead of `γ_{n+1}` — negligible in magnitude, but it should be stated, not claimed away).
3. **`sqrt` then divide, never `rsqrt`.** `rsqrt.approx.f32` is 2 ULP and not correctly rounded. Use `real_t(1)/sycl::sqrt(a)`; **inspect the PTX/SASS once** to confirm no fast-math pass fuses them back into `rsqrt.approx`, and add `-fno-fast-math` to this translation unit if it does. Overflow is impossible: `1/sqrt(FLT_TRUE_MIN) ≈ 1.5e22 < FLT_MAX`.
4. **FMA contraction** in (P2)/(P3): strictly helpful, covered by the same bound.
5. **Hermitian diagonal is forced real** at three points — on load, in (P3)'s epilogue, and (if the oracle path is used) by `device::herk` (`rankk.hh:71-79`). If the imaginary part is allowed to drift, `sqrt(a_kk)` is being taken of a complex number and the factor is silently wrong in a way small-`n` residual tests may not catch. T7 exists for exactly this.

**Where the difference is material.** **[FIX-A1.5]** The candidate's claim that "a tuning change can never move the residual" conflated the *bound* with the *residual*. It moves the residual, and near `κ(A)·u ≈ 1` it moves the **pass/fail decision**: `cond_tests.cc:365-378` records that at `n = 64`, float, the squared Gram sits exactly on the cliff where whether potrf fails at all is decided by O(u) differences ("failed on seed 1 and not on 123 or 2024"). Since §10 routes on `(n, batch)`, the same `ortho` call can factor cleanly on one provider and go indefinite on another. That is a real, user-visible behaviour split; it is the reason §5.9 forbids cross-provider `info` equality, and it belongs in the coverage table.

**Accuracy gate before any Auto-order change** (none of these are ctest targets; all `EXCLUDE_FROM_ALL`, run by hand): `benchmarks/orthogonality_miniacc` — `ACC_ORTHO_CHOL2`, `ACC_ORTHO_CHOLESKY`, `ACC_ORTHO_SHIFTCHOL3`, with `ACC_ORTHO_HOUSEHOLDER` as the conditioning-independent control — and `benchmarks/orthogonality_accuracy --impl ortho_all --samples 4096 --log10-cond-max 10`.

---

## 8. File-by-file implementation plan

Each step compiles and passes the existing suite on its own. `[M]` mechanical, `[J]` judgement.

**Phase 0 — scaffolding, no behaviour change**

| # | file | change | |
|---|---|---|---|
| 0.1 | `src/backends/{cusolver,rocsolver,netlib_lapack}.cc` | rename `potrf` → `backend::potrf_vendor`, `potrf_buffer_size` → `backend::potrf_vendor_buffer_size`; bodies unchanged (the WP0 shape, mirroring `ormqr`) | [M] |
| 0.2 | `include/batchlas/blas/functions/potrf.hh` | add `detail::potrf_cta_max_n<T>(size_t slm_budget)`, `detail::potrf_supports_cta/blocked`, `detail::choose_potrf_provider<T>`, `potrf_dispatch<B,T>`; the public `potrf`/`potrf_buffer_size` forward to it. **Chooser returns `Provider::Vendor` for everything at this step.** | [J] |
| 0.3 | `src/linalg-impl.hh` / instantiation sites | route the public entry points through `potrf_dispatch` | [M] |

*Gate:* `ctest -R "options_api_tests|linalg_layer_tests|ortho_tests"` green, zero behaviour change.

**Phase 1 — CTA kernel, force-only**

| # | file | change | |
|---|---|---|---|
| 1.1 | `src/extensions/potrf_cta_device.hh` | `real_part`, `mul_conj_b`, `div_by_real`; `potrf_diag_block_subgroup`, `potrf_panel_solve_rows`, `potrf_trailing_tiles`, `potrf_cta_body<T,NB,TS,Scope>` | [J] |
| 1.2 | `src/extensions/potrf_cta.cc` | `L`/`G`/`wg_size` selection (§3.2), SLM sizing against `runtime_local_mem − 4096`, sub-group-32 presence check, `[[sycl::reqd_sub_group_size(32)]]`, `NB` ladder + `Scope` ladder, `potrf_cta_buffer_size`, `BATCHLAS_FOR_EACH_SCALAR_TYPE_1 × {CUDA, ROCM, NETLIB}` instantiation | [J] |
| 1.3 | `include/batchlas/tuning_params.hh` | `POTRF_CTA_NB_*` constants **for all four scalar types from the start** (the routing grid being float-only has already cost this repo once), `potrf_cta_nb_for_n<T>(n)`, `BATCHLAS_TUNE_POTRF_CTA_NB/L/G` | [M] |
| 1.4 | `functions/potrf.hh` | `potrf_supports_cta` returns true only under `policy.forced == BatchLAS_CTA` or `BATCHLAS_POTRF_PROVIDER=cta` | [M] |
| 1.5 | `tests/potrf_tests.cc` + `tests/CMakeLists.txt` | §9, **provider pinned** | [J] |
| 1.6 | `benchmarks/potrf_benchmark.cc` + `benchmarks/CMakeLists.txt` | §10 grid, via `batchlas_register_benchmark` | [M] |

*Gate:* T1–T9 green with the provider pinned; `-Rpass-analysis=kernel-resource-usage` shows no spill of `x[]`/`acc[]`; SASS shows `sqrt` + `div`, not `rsqrt.approx`.

**Phase 2 — blocked driver, force-only, Lower only**

| # | file | change | |
|---|---|---|---|
| 2.1 | `src/extensions/potrf_blocked.cc` | `potrf_blocked_layout` (§5.5), info-zero pre-pass, panel loop, `PotrfPanelSolveKernel`, GEMM trailing update (§2.6), `m2 == 0` guards, explicit sub-view construction | [J] |
| 2.2 | `functions/potrf.hh` | `potrf_supports_blocked` (force-only, `Uplo::Lower` only, `n > potrf_cta_max_n<T>()`) | [M] |
| 2.3 | `tests/potrf_tests.cc` | extend T1–T9 to blocked sizes and to `Provider::BatchLAS_Blocked` | [M] |

**Phase 3 — gated on measurement**

| # | change | |
|---|---|---|
| 3.1 | Run the §10 grid; record the envelope **with provenance** in `tuning_params.hh` the way the `syev` routing grid is recorded | [J] |
| 3.2 | Flip `Auto` for exactly the measured cells; leave the rest on Vendor | [J] |
| 3.3 | `PotrfPanelSolveKernelUpper` (§3.4) + enable Upper for blocked | [J] |
| 3.4 | Coverage-table entry: which `(T, n, batch, uplo)` cells are native, which are vendor-by-default, and the §5.8 / §7 behaviour notes | [M] |

---

## 9. Test plan

### 9.1 Existing ctest targets (exact `add_test` NAMEs, `tests/CMakeLists.txt:243`)

| target | label | what it covers | caveat |
|---|---|---|---|
| `options_api_tests` | `util` (smoke) | `info` (`:507-514`), Lower/Upper distinguishability (`:401-426`), bare-`{}` ill-formedness (`:430`), arena-lease vs explicit-span (`:248-255`) | runs at `n = 8, batch = 2` — **outside every proposed native envelope**, so without pinning it keeps testing the vendor forever |
| `ortho_tests` | `ortho` | end-to-end `Chol2`/`ShiftChol3` orthogonality residual | same pinning caveat |
| `cond_tests` | `blas` | ortho Cholesky path under controlled conditioning; the discarded-`info` smear | |
| `linalg_layer_tests` | `util` (smoke) | documented spellings incl. `linalg::cholesky` | |
| `trsm_tests` | `blas` | unrelated to potrf, but the only trsm coverage: `Side::Right` and `Diag::Unit` are **never tested** and the "batched" tests issue `batch_size` separate unbatched calls | relevant to WP3, not here |
| `syevx_tests`, `lanczos_tests` | | LOBPCG/Lanczos consumers of `ortho` | |

Per the repo's selective-testing policy: scope with `ctest -L blas -L ortho` during development; full suite pre-push only.

**The single most important line in this section:** every new test **and** the potrf cases in `options_api_tests` must pin the provider (`DispatchPolicy{.forced = Provider::BatchLAS_CTA}` or `BATCHLAS_POTRF_PROVIDER=cta|blocked`). Without it the native code ships untested by construction, which is exactly how the blocked candidate's `Uplo::Upper` path would have shipped with zero coverage.

### 9.2 New target `tests/potrf_tests.cc`, label `blas`, not `slow`

* **T1 — residual.** All four types × both `Uplo` × both providers × `n ∈ {1,2,3,7,8,9,15,16,17,31,32,33,52,63,64,65,74,105}` (CTA) and `{128,129,192,256,257,512}` (blocked) × `batch ∈ {1,3,128}`. `A = M Mᴴ + n·I`; assert `‖A − L Lᴴ‖_F / ‖A‖_F ≤ c·n·eps`. **`n = 2` and `n = 3` with a non-zero off-diagonal are mandatory and non-negotiable** — the stale-pivot defect fails at `n = 2` and nothing in the existing tree would have caught it.
* **T2 — untouched triangle.** Poison the non-`uplo` triangle with a recognisable pattern; assert bit-identical afterwards. Repeat with a quiet NaN there and assert the produced factor is NaN-free — this proves the triangle is never **read**, which is what `ortho.cc:156-161`'s uninitialised upper half requires.
* **T3 — packed-batch ragged panel (see §9.3).**
* **T4 — `info` at scale.** `batch = 256`, items 0, 37, 255 non-PD at *different* columns; assert every item's exact value and `0` elsewhere. Also assert the PD items' factors match those items factored alone (catches a shared-flag / cross-item write).
* **T5 — `info` index across panel boundaries, exact.** `A = L₀ D L₀ᴴ`, `L₀` unit lower triangular with modest entries, `D = diag(1,…,1,−δ,1,…,1)` with `−δ` at global column `c`. The `k`-th updated diagonal equals `D_kk`, so failure must occur at exactly `c` and `info == c+1`. Sweep `c ∈ {0, 1, NB−1, NB, NB+1, 2NB+7, n−1}` with `n = 4·NB + 5`. Catches: panel-local instead of global index; 0-based instead of 1-based; last-wins instead of first-wins (plant two failures); **and the stale-pivot class, because every `c > 0` case returns `info = 0` under it.**
* **T6 — provider equivalence.** Same input through Vendor / CTA / Blocked: compare **residuals**, never entries (§7.1), and `info` as **zero/non-zero only** (§5.9).
* **T7 — Hermitian diagonal.** Complex input with a deliberately non-zero imaginary diagonal; assert the result equals the same input with that part zeroed, and that the factor's diagonal is exactly real.
* **T8 — empty `info` span.** Call with `Span<int32_t>{}` and assert the factor is **correct**. If the blocked driver forgets to zero the pool scratch, every item early-outs and `A` returns unmodified with no error, and no other test notices (§5.7).
* **T9 — failed item finiteness.** Blocked provider, `n = 4·NB_o + 5`, one item non-PD at column `NB_o + 3`. Assert `std::isfinite` over the whole of that item's `A`. This is the test the candidates lacked entirely — nothing in either plan asserted anything about a failed item's `A`, while the property was advertised three times.

### 9.3 The single test that catches the most likely failure mode

**T3 — packed-batch ragged panel with distinct matrices and an SLM canary.**

With the stale pivot fixed, the highest-probability remaining silent failure is an out-of-range write in the compile-time-unrolled, runtime-predicated panel code: (P1)'s `lane < ib && lane >= k` publish, (P2)'s `c < ib`/`p < c`, (P3)'s `r0+a < m2`. A missed guard writes past column `n` of an `LDA × n` SLM tile which, under `G > 1`, **lands in a neighbouring matrix's tile** and yields a plausible wrong answer with no crash and no NaN. This repo has produced exactly this class once already, silently, in `sy2sb` stage 1.

The test:

* For **every instantiated `NB`**, include `n ≡ 1 (mod NB)` and `n ≡ NB−1 (mod NB)`: `n ∈ {9,15,17,31,33,63,65}` and the type-specific ceilings.
* `batch ≥ 8` with `n` small enough that `G > 1` is selected, so the packed layout is exercised.
* **Every matrix in the batch numerically distinct** (different diagonal scale per item), so a cross-matrix read cannot coincidentally agree.
* Built with `-UNDEBUG` so `KernelMatrixView::operator()`'s bounds asserts fire on device (the repo's established GPU-OOB recipe).
* **Plus an explicit SLM canary**, because `-UNDEBUG` alone is not sufficient here: the (P1) publish and the (P2)/(P3) tile accesses are raw `Sd(r,c)` pointer arithmetic, not `KernelMatrixView::operator()`, so `matrix.hh:135-149`'s asserts never fire, and an in-tile cross-matrix write satisfies them anyway. Pad each matrix's SLM allocation with a 16-element guard filled with a known bit pattern and `assert` it intact at write-back under `-UNDEBUG`.

---

## 10. Routing predicate — **REQUIRES MEASUREMENT BEFORE IT IS TRUSTED**

### 10.1 Hard gate (correctness/fit — evaluated identically by the query and the call)

```cpp
template <typename T>
inline int potrf_cta_max_n(size_t slm_budget) {          // 45056 on this box
    // largest n with (n|1)*n*sizeof(T) + NB*sizeof(real_t) + 64 <= slm_budget
}
template <typename T>
inline bool potrf_supports_cta(const DeviceCaps& caps, size_t slm_budget,
                               bool has_sg32, int n, int batch) {
    if (!caps.is_gpu)  return false;                     // CPU SYCL: correct, not fast
    if (!has_sg32)     return false;                     // enumerated sub_group_sizes, not max_sub_group
    if (n < 1)         return false;
    return n <= potrf_cta_max_n<T>(slm_budget) && batch >= kPotrfCtaMinBatch;
}
template <typename T>
inline bool potrf_supports_blocked(const DeviceCaps& caps, int n, int batch,
                                   Uplo uplo, size_t slm_budget) {
    if (!caps.is_gpu)            return false;
    if (uplo == Uplo::Upper)     return false;           // until Phase 3, unconditionally
    if (n <= potrf_cta_max_n<T>(slm_budget)) return false;
    return batch >= kPotrfBlockedMinBatch;
}
```
`slm_budget` = runtime `DeviceProperty::LOCAL_MEM_SIZE` − 4096. `has_sg32` from `device::sub_group_sizes` per `sytrd_cta.cc:319-333`. `DeviceCaps` needs no new field beyond what `query_caps` already provides plus these two queries.

### 10.2 Auto order

The existing `default_order_cta_blocked_vendor_netlib` (`env.hh:57-64`) is reused; **no new `std::array<Provider,6>`**, so the WP0 silent-truncation hazard is untouched. **At merge, `kPotrfCtaMinBatch = kPotrfBlockedMinBatch = INT_MAX`**, i.e. both native providers are reachable only by force. The thresholds and the per-type `n` ceilings become finite **only** as a recorded consequence of §10.3, entered into `tuning_params.hh` with the measurement provenance in the comment, the way the `syev` routing grid is recorded.

Cells and my honest prior (these are hypotheses; the two strongest precedents in this repo both say measure — the 3.0–3.7× SYEVX win was *routing*, and a prior research document's top item measured 85–211× **slower** than what it replaced):

| cell | prior | confidence |
|---|---|---|
| `n ≤ 32`, `batch ≥ 512`, any type — CTA | tie with `cusolverDnXpotrfBatched`, which is itself a CTA kernel | low |
| `32 < n ≤ 64`, `batch ≥ 512`, float/double — CTA | **the target cell**: `ortho`'s dominant shape (`k ≲ 64` Gram, 2–3 potrf per `ortho`, `ortho` inside LOBPCG's inner loop) | medium |
| `64 < n ≤ 105` float — CTA | 2–3 blocks/SM, negligible latency hiding | **low — most likely cell to be cut** |
| `complex<float>`, `complex<double>` — CTA | best relative odds anywhere: cuSOLVER's complex batched path is weakest, and this library already abandoned cuBLAS for complex `trsm` (`cublas.cc:1122-1225`) | medium |
| `n ∈ (ceiling, 512]`, `batch ≥ 128`, float/`complex<float>` — Blocked | plausible; `cusolver.cc:20-22` carries a build-time warning about batches above 128×128 | low-medium |
| `n > 1024` — Blocked | the traffic model says it loses; that cell needs V2, not an argument | very low |
| `batch < 128`, `batch == 1` | **Vendor, unconditionally** | — |

### 10.3 The grid that settles it

`benchmarks/potrf_benchmark.cc`, `EXCLUDE_FROM_ALL` via `batchlas_register_benchmark`, run by hand (`--target potrf_benchmark`); `ctest` will never catch a regression here.

* **Providers:** `Vendor` × `BatchLAS_CTA` × `BatchLAS_Blocked`, pinned by env, same process, same input.
* **`n`:** `{8,16,24,32,40,48,56,64,72,80,96,105}` (CTA range, plus 52/74 exactly at the per-type ceilings) and `{128,192,256,384,512,768,1024}` (blocked range).
* **`batch`:** `{128, 512, 2048}` for the comparison; **additionally `{1, 8, 32}` for profiling only** — comparing only at saturation is exactly what hid a batch-only-parallelism defect in this repo for months, so profile everywhere and compare at saturation.
* **Types:** all four. **Both `Uplo`.**
* **Ortho-shaped grid:** `k ∈ {16,32,64,128}` square Gram × `batch ∈ {512, 2048}` — `ortho` is the only real consumer and its `k` is the parameter that sizes potrf, not `ortho_benchmark`'s `m`.
* **End-to-end:** `ortho_benchmark` (`m,n ∈ {64…1024}`, `batch` to 512) — **a potrf win that does not show up in `ortho` is not a win worth routing to.** `bench::pristine(A)` between iterations; potrf is in-place.
* **Hygiene, all mandatory:** warm the JIT before the first timed iteration (a first-run SYCL JIT fabricated an entire 3.7× regression here); pin one GPU (`CUDA_VISIBLE_DEVICES=0` — this box has two 4090s); warm clocks; beware the substring `--name` trap.
* **Profiles to collect across the whole `n` range, not just at saturation:** `smsp__thread_inst_executed_per_inst_executed` (the (P1)/(P2)/(P3) lane-efficiency claims in §2.3/§3.5), `sm__warps_active.avg.pct_of_peak_sustained_active`, `launch__occupancy_limit_shared_mem`, `launch__registers_per_thread` (§4.3 is unverified without these), and for the blocked path a kernel-count check that no launch resolved to a per-batch-item host loop.
* **Gate to flip a cell to Auto:** `t_native ≤ 0.90 · t_vendor` at that cell at saturation (a 10 % margin, not parity — a route change must pay for its risk), **and** the §7 accuracy harnesses show no regression against `ACC_ORTHO_HOUSEHOLDER`, **and** `ortho_benchmark` shows the win end to end.
* **Kill rule:** if a path's first honest measurement is not a win, it is deleted, not defended. The tiled (P3) in particular ships only after it beats the `device::herk` oracle on the same shapes; its ~60 lines are not an asset.

---

## 11. The three biggest risks

**Risk 1 — (P1) is 30–45 % of the per-matrix critical path, so the CTA kernel may simply lose to `cusolverDnXpotrfBatched` in the target cell.** The diagonal-block factorisation is `n` dependent `sqrt`s with ~`n·NB/2` dependent FMAs and ~`n` barriers, and no amount of tiling removes it. *Mitigations:* (a) `Scope::SubGroup` with `G` matrices per work-group wherever SLM allows, which is the only configuration with **zero** idle lanes during (P1), and which the `L`-selection rule (§3.2) now reaches for every `n ≲ 56`; (b) `NB` and `L` are a tuned table with env overrides, not constants, so the (P1)-vs-(P3) balance is measurable rather than assumed; (c) the §10.3 profile metrics make the term visible instead of arguable; (d) the merge default is Vendor, so a loss costs nothing but the review time. *Kill criterion:* if no `(T, n, batch)` cell clears `0.90 × vendor` at saturation and no cell shows up in `ortho_benchmark`, Phase 1 is reverted rather than tuned indefinitely.

**Risk 2 — the blocked driver is memory-bound and the GEMM trailing update pays 2× flops on every diagonal block, so it may lose everywhere above the CTA ceiling.** Trailing traffic is `2n³/(3·nb)` elements read+written (the candidate's `n³/(3nb)` counted one side), giving `nb/(2·sizeof(T))` = 8 flop/byte for float at `nb = 64` against a ~47 flop/byte machine balance — memory-bound by ~6×, not ~3×. On top of that the diagonal-block fold recomputes `W/(2·m2)` of the update. *Mitigations:* (a) the column-panel decomposition (§2.6) keeps the waste at `W/m2` instead of the 100 % a naïve gemm-into-scratch costs, and keeps scratch bounded by `W²·batch` independent of `n`; (b) it is the only formulation that is batched for all four types — the alternative, public `syrk`/`herk`, is a per-item host loop at ~9 µs × batch for every non-float type above `n = 128` and for float in `m2 ∈ [129,256]`, which is not a tuning problem but a dead cell; (c) V2 (two-level blocking, trailing traffic ÷ 4) is pre-specified so the response to a measured miss at `n ≥ 1024` is a known work item rather than an improvisation; (d) Phase 2 is gated on Phase 1 succeeding, so the cheap half is validated before the expensive half is built. *Kill criterion:* if Phase 1 does not clear its gate, Phase 2 is not started.

**Risk 3 — a silent wrong answer in the ragged/packed path, or a silent behaviour split in `info` near the `κ²u` cliff.** The unrolled-with-predicates structure has three independent guard sites, an out-of-range write under `G > 1` lands in a neighbouring matrix's SLM and produces a plausible wrong answer with no crash, and neither `-UNDEBUG` device asserts nor `KernelMatrixView`'s bounds checks can see it (raw pointer arithmetic, in-tile address). Separately, `nb`/`L`/`TS` do move the residual, and at `κ(A)·u ≈ 1` they move potrf's pass/fail, which the routing switch makes user-visible. *Mitigations:* (a) T3 — every `NB`'s `n ≡ ±1 (mod NB)`, `G > 1`, numerically distinct matrices, `-UNDEBUG`, **plus an SLM canary** that catches the in-tile cross-matrix case the asserts cannot; (b) T2's NaN poison proves the untouched triangle is never read, which is the highest-consequence silent bug because `ortho`'s upper half is uninitialised workspace and the symptom would be intermittent LOBPCG garbage months later; (c) T9 asserts a failed item's `A` is finite, closing the failure path that no test in either candidate plan touched; (d) §5.9 forbids exact cross-provider `info` comparison and the coverage table records the split explicitly, so a legitimate difference is documented rather than discovered as a flake.
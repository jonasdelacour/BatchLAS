# Blocked Jacobi at n = 64–256: implementation and measured results

Executes **WP8 / C1** of `SYEV_PERF_IMPLEMENTATION_PLAN.md` ("block Jacobi at n = 64–256"),
which defers to Tiers B and C of `JACOBI_EIGENSOLVER_PLAN.md` §8.5. Tier A (`syev_jacobi_cta`,
n ≤ 32) was already in tree; Tiers B and C were design-only. Both now exist as one kernel.

Ships as `syev_jacobi_blocked` / `syev_jacobi_blocked_buffer_size`
(`src/extensions/syev_jacobi_blocked.cc`), **not routed from `Auto`**. Why not is §4.

---

## 1. What was built

One work-group per matrix, block-cyclic two-sided Jacobi over block-column pairs. Only the
current pivot block is resident; `A` and the eigenvector accumulator `Z` stay in global memory,
so `n` is bounded by global memory rather than by local memory as in Tier A.

```
for each sweep
  for each block pair (p, q), cyclically            l(l-1)/2 pairs, l = ceil(n/nb)
    S <- A[idx, idx]                                idx = cols(p) ++ cols(q), |idx| = m <= 2nb
    skip the pair unless some |S_ij| > tol*sqrt(|S_ii S_jj|)
    U <- inner two-sided Jacobi on S                local memory, rotation-based
    A[:, idx] <- A[:, idx] * U                      panel update, coalesced
    A[idx, idx] <- S                                already U^H A U
    A[idx, r]   <- conj(A[r, idx])                  mirror, r outside idx
    Z[:, idx] <- Z[:, idx] * U
```

Three things in that loop are not in the plan's sketch and are worth keeping:

**The row update is a transpose, not a second GEMM.** §8.5 has
`A[(p,q), :] <- U^T A[(p,q), :]` as an m × n product. It never has to be computed. With
`V` block-diagonal and `B = A V`, the rows outside the pivot block satisfy
`A'[idx, r] = conj(A'[r, idx])` because `A'` is Hermitian and `A'[r, idx]` is exactly what the
*column* update already produced. The second product therefore collapses to the m × m pivot
block — which the inner solve has already computed, as `S` — plus a transposing copy. That
removes `n*m^2` flops per pivot pair and, more importantly, removes the strided-read panel the
row update would have needed.

**The mirror is staged through local memory.** Its destination is transposed, so writing it
directly turns one coalesced store into m scattered ones. It reuses the `S` buffer, which is
dead by that point, so the staging costs no extra local memory.

**l == 2 degenerates into the fully resident solve.** With two block columns the pivot block is
the whole matrix, the panel update and the mirror are both empty, and the kernel never touches
global memory between the initial load and the final store. That is the plan's Tier B
(`32 < n <~ 128`) and it falls out of the Tier C structure rather than needing its own kernel.
It is also, by a wide margin, where the kernel is fastest — see §3.

Supported: real symmetric and complex Hermitian, `Uplo::Lower` and `Uplo::Upper`, `2 <= n <= 1024`,
both job modes. 25 tests in `tests/syev_jacobi_blocked_tests.cc`, all passing on all four scalar
types. `A` is destroyed in both job modes.

---

## 2. Accuracy — the payoff holds at n = 64 and n = 128

This is the result the solver exists for, and it survives blocking.

Graded SPD input `A = D M D`, `D_ii = 2^-e_i` with the exponents spread linearly over
`[0, 58]`, so `kappa(A) ~ 1e35` while the column-equilibrated condition number stays `O(1)`.
Entries are dyadic, so float and double see bit-identical input. Reference is an independent
double-precision CPU Jacobi. Solver and reference both in **float**:

| n | spectrum | `syev_jacobi_blocked` | routed `syev` |
|---|----------|-----------------------|---------------|
| 64  | 1.1e-35 … 1.06 | **3.0e-06** | 1.6e+27 |
| 128 | 9.0e-36 … 1.11 | **1.2e-05** | 4.8e+26 |

Thirty-three orders of magnitude, on the same input in the same precision. The blocked kernel
resolves every eigenvalue to a small multiple of float eps (25x and 104x eps respectively);
the tridiagonalizing path returns garbage for the small end, exactly as the theory predicts.

Note the test-construction trap that cost a debugging cycle: the fixed per-index grading used
at n ≤ 32 (`D_ii = 2^{-2i}`) does not survive to n = 128 — it puts the smallest entry at
`2^{-4*127}`, which in float is not inaccurate but *zero*. The first version of this test was
measuring underflow, not the solver. Spreading a fixed total exponent span keeps the smallest
eigenvalue three orders above `FLT_MIN` at every n.

Backward error on random symmetric input is pinned by the residual and orthogonality tests at
n ∈ {33, 47, 64, 65, 97, 128, 129}.

---

## 3. Performance — it loses, and the cost model says it had to

RTX 4090 (device 1, idle), float, saturating batch, shipped defaults, µs per matrix.
Ratio > 1 means blocked Jacobi is slower.

| n | batch | job | `syev_jacobi_blocked` | routed `syev` | ratio |
|---|-------|-----|----------------------|---------------|-------|
| 64  | 4096 | vectors | 2.92 | 1.44 | **2.03x** |
| 64  | 4096 | values  | 2.95 | 0.50 | 5.88x |
| 128 | 2048 | vectors | 44.7 | 6.54 | 6.83x |
| 128 | 2048 | values  | 36.4 | 2.88 | 12.7x |
| 256 | 1024 | vectors | 370  | 32.1 | 11.5x |
| 256 | 1024 | values  | 276  | 15.6 | 17.7x |

`complex<float>` is roughly 2x worse again at every shape, because the register budget of the
panel update caps the pivot block at m = 32 instead of 64, which doubles the global traffic
(§3.2).

### 3.1 The plan's cost model was 3x optimistic, and this is why

WP8 estimated "~8–10 sweeps × ~4n³ ≈ 30–40n³ against our ~4n³ — a 10x flop premium — so it
breaks even at 17–21 TFLOP/s".

The `4n³` per sweep counts **one** of the three block updates a round performs. Right-multiplying
`A` by the block-diagonal `V` costs `4n²·nb` per round and there are `l-1 ≈ n/nb` rounds, so
`A·V` is `4n³` per sweep — but `V^H·A` is another `4n³` and `Z·V` a third. The real figure is
**12n³ per sweep with eigenvectors, 8n³ without**, hence ~100n³ over 8–10 sweeps, a **~25x**
premium over syev's ~4n³, not 10x.

Break-even at n = 256 therefore needs ~51 TFLOP/s, which is *above* the ~47 TFLOP/s this card
sustains on cuBLAS SGEMM (see `sycl-matches-cuda-gemm`). **WP8 could not have won at n = 256 at
any implementation quality.** The transpose trick in §1 removes one of the three terms, taking
the premium to ~17x; that is a real saving and it is not enough.

### 3.2 What actually binds is memory traffic, and it explains the shape of the table

Global traffic per sweep is `6n³/nb` elements with vectors (`4n³/nb` without), because every
pivot-pair update reads and writes the whole n × m panel three times over — panel, mirror,
eigenvectors. `nb` is the only term that moves it, and `nb` is capped by local memory: the
pivot block needs two m × (m+1) arrays with m = 2nb, which at 48 KB and float allows m ≤ 64.

That predicts a **cliff, not a slope**, at the point where the matrix stops fitting in local
memory — and that is what the table shows. n = 64 gives l = 2, the fully resident case with
*zero* global traffic between load and store, and costs 2.03x. n = 128 gives l = 4 and costs
6.83x. The degradation is not gradual; it is the loss of residency.

Two independent confirmations that the implementation sits on its own cost model rather than
leaving easy factors on the table:
- Dropping eigenvectors at n = 256 saves 34% (370 → 276 µs). Z is 1/3 of the traffic terms.
  A compute-bound kernel would not have shown that.
- Widening the pivot block is worth exactly what the `1/nb` term says: forcing nb = 8 instead of
  the auto-selected 32 costs 1.9x at n = 64 and 1.4x at n = 128.

### 3.3 Two defaults that are measured, not guessed

**Inner sweeps.** 0 (the default) selects: diagonalize the pivot block exactly when it is the
whole matrix, one inexact sweep otherwise. Measured spread is ~1.3x either way, and the
crossover has a cause — when `l == 2` an exact inner solve finishes the problem in one outer
sweep and there is no outer cost left to amortize, whereas when `l > 2` the inner solve costs
`O(m³)` per block against the outer update's `O(n m²)`, so extra inner sweeps buy convergence
at a worse rate than another outer sweep does. The latter is why MAGMA's batched SVD and
Novaković's block-oriented variant both use a single inexact sweep; the former is why the plan's
blanket `M_s = 1` recommendation is wrong for the resident case.

**Work-group size** (µs per matrix, eigenvectors):

| | wg=256 | wg=512 | wg=768 | wg=1024 |
|---|---|---|---|---|
| n=64  | 3.36 | 3.00 | **2.94** | 3.96 |
| n=128 | 55.4 | 51.5 | 52.0 | **44.8** |
| n=256 | 398 | 398 | 421 | **372** |

Same `l == 2` split. A resident solve is bound by local-memory traffic and barriers and wants
two work-groups per SM to hide them, which under Ada's 1536-thread-per-SM ceiling caps the
work-group at 768 — going to 1024 halves residency and costs 35%, the cliff in the n = 64 row.
A blocked solve is bound by global-memory latency on the panel update instead, where one wide
work-group issuing more concurrent loads beats two narrow ones. Shipped default follows that
rule; `BATCHLAS_JACOBI_BLOCKED_WG` overrides it.

---

## 4. Routing decision: opt-in, not `Auto`

`syev_jacobi_blocked` is **not** wired into `detail::choose_syev_provider`. It is slower than
the routed path at every measured shape, so routing it on speed would be a pure regression, and
routing it on accuracy would require an input property the API does not carry (the
relative-accuracy theorem is SPD-only, and nothing in the signature asserts SPD).

This is the framing `JACOBI_EIGENSOLVER_PLAN.md` already recommends and WP8 repeats: Jacobi is a
*second* backend for accuracy-critical and graded input, not a replacement. What changed is that
the accuracy backend now covers n = 64–256 instead of stopping at n = 32.

---

## 5. If someone wants to close the gap

Recorded so it is not re-derived. The binding constraint is `6n³/nb` and `nb` is capped by local
memory, so the only structural lever is **Novaković's hierarchical blocking**: run the outer loop
at nb = 64 (l = 4 at n = 256 instead of 8) and solve the 128 × 128 pivot block with a nested
blocked Jacobi over a global scratch rather than in local memory. That cuts traffic ~2.3x.

Against ratios of 6.8x and 11.5x, 2.3x does not change the conclusion — which is why it was not
built. It is worth revisiting only on hardware with a materially larger local memory per SM, or
if a future caller needs relative accuracy badly enough to pay a 3-5x time premium instead of a
7-12x one.

Two smaller items measured or identified and deliberately not taken:
- When `l == 2` and the solve converges in one outer sweep, `Z` is exactly `U`, so the `Z = I`
  seed and the first `Z <- Z·U` product are both waste. Worth ~11% at n = 64 (2.95 → ~2.6 µs
  measured as the values-only/vectors gap), and nothing anywhere else.
- Raising the panel-update register cap from 64 to 76 floats would let nb reach 37 at n = 256.
  Worth ~14% of the traffic there, nothing at n = 64 or 128 where the block partition does not
  change.

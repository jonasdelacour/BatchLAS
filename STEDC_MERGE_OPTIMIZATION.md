# STEDC Merge Kernel — Optimization Research Notes

Research log for speeding up the STEDC merge path. Captures the profiling picture,
the code-level findings, the relevant literature, and a prioritized plan.

Status legend: **[done]** implemented · **[fixed]** bug resolved · **[open]** not started.

---

## 1. Profiling picture

From `.github/skills/nvidia-kernel-profiling/references/batchlas-stedc.md`
(RTX 4090, sm_89, `stedc_benchmark --backend=CUDA --type=float 256 256 16 0 32 1`):

| Metric | `StedcFusedCtaMerge<CUDA, float, 32>` |
| --- | --- |
| Duration | 101.50 us |
| SM throughput | 8.80% |
| Memory throughput | 8.80% |
| DRAM throughput | 0.14% |
| Registers/thread | 78 |
| Block size | 32 |
| Grid size | 256 |
| Theoretical occupancy | 50.00% |
| Achieved occupancy | 3.96% |
| Waves per SM | 0.08 |

Kernel share of total GPU time: `SteqrCTAKernel` 35.7%, `StedcFusedCtaMerge` 30.3%,
`Matrix::Identity` 13.9%, `PermutedCopyKernel` 5.4%, `ampere_sgemm_128x128_nn` 4.9%.

**Conclusion: the merge kernel is neither compute- nor bandwidth-bound.** It is
serialization/latency-bound. Achieved occupancy is ~1/12 of theoretical, which
points at intra-kernel serialization rather than launch geometry alone.

---

## 2. Code-level findings

### 2.1 Rescale and normalize were serialized over `dd` — **[done]**

> **Both loops are now partition-parallel.** End-to-end gain on the CTA path is
> 4-9% depending on n (§6). Landing this required first fixing an unrelated
> pre-existing bug that the change exposed — see §2.1a.

`stedc_merge_cta.cc`, `maybe_rescale_vectors` and `normalize_vectors`.

Both looped `for eid = 0 .. dd-1` performing a **whole-work-group**
`reduce_over_group` plus a `group_barrier` per index. The root solve directly
above them is already partition-parallel (`for root_ix = part_id; root_ix < dd;
root_ix += parts_per_wg`), so the kernel went wide for the secular solve and then
collapsed to one column at a time for `2*dd` barrier-separated steps.

Both outer loops are embarrassingly parallel:

- **rescale**, index `eid`: reads row `eid` of `Q` (not written in this phase),
  writes only `v(eid)`. No cross-index dependency.
- **normalize**, index `eig`: reads all of `v`, reads and writes only column
  `eig` of `Q`. No cross-index dependency.

Fix applied: both are now templated on a reduction adapter plus a
`(first, stride)` index range.

- CTA variant passes `PartitionAdapter` with `(part_id, parts_per_wg)` for
  **both** loops — one index per sub-group partition, all running concurrently,
  **zero work-group barriers inside either loop**. A single `group_barrier(wg)`
  separates the two phases (rescale reads `Q` / writes `v`; normalize overwrites
  `Q` / reads `v`).
- WG variant passes `WorkgroupAdapter` with `(0, 1)` for both — bit-identical to
  the previous behaviour, retained as the reference path for A/B comparison.

Supporting changes: `reduce_product` and a no-op/barrier `sync()` added to both
adapters; `nrm2_column` added as an adapter-generic Blue's-scaled 2-norm
mirroring `internal::nrm2` in `src/math-helpers.hh` (the three accumulators are
reduced as scalars rather than a `sycl::vec<R,3>`, which is componentwise
identical).

### 2.1a Root cause: `dd == 1` read `d_prob(-1)` — **[fixed]**

Making `normalize_vectors` partition-parallel surfaced
`CUDA_ERROR_ILLEGAL_ADDRESS` inside the merge kernel. The cause turned out to be
**pre-existing and unrelated to the parallelization**.

`solve_root_ext_generic` computed

```cpp
const int32_t last = dd - 1;
const int32_t prev = dd - 2;
const T d_last = d_prob(last);
const T d_prev = d_prob(prev);   // dd == 1  ->  d_prob(-1)
```

and dereferenced `prev` unconditionally. When a merge subproblem deflates all the
way down to `dd == 1`, `prev` is `-1`. `d_prob` aliases the shared-memory
`d_local` through a generic pointer, so a negative offset resolves to an address
outside the shared window and **faults** rather than reading harmless garbage.

Why it only showed up with partition-parallel normalize: the out-of-range access
was already happening on the work-group path (confirmed by device assert), it
just happened not to fault under that code's register and shared-memory layout.
The partition version changed the layout enough to make the same bad address
fatal. It was latent, not introduced.

Fix: handle the single-pole case in closed form. With one pole the secular
equation `1/rho + z0^2/(d0 - x) = 0` solves exactly as `x = d0 + rho*z0^2`, so
the CTA solver returns `{d_prob(0), rho * z0 * z0}` before computing `prev`.

The identical bug exists in the non-CTA path, `sec_solve_ext_roc` in
`stedc_secular.cc` (`prev_index = dd - 2`, dereferenced unconditionally). It is
fixed the same way there; that path updates `D` in place, so the guard also
writes the denominator `D0 - x = -rho*z0^2`, matching what the CTA path's
`(d - origin) - tau` produces and what `apply_shift_to_poles` leaves behind in
the general case.

**How it was found.** `compute-sanitizer` cannot attach here ("CUDA initialized
before the Sanitizer") because SYCL initializes CUDA first. What worked was
rebuilding with device-side asserts enabled — the bounds checks in
`KernelMatrixView`/`VectorView::at` are compiled out by `NDEBUG`, so:

```bash
cmake -B build-assert -S . -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG" ...
```

CMake puts `-DNDEBUG` in `$DEFINES` and the rule is `$DEFINES $INCLUDES $FLAGS`,
so `-UNDEBUG` in the flags wins. The device assert then names the exact function
and index. This is the tool of choice for any future OOB in these kernels.

Two robustness fixes were made along the way. Neither was the root cause, but
both are genuine hazards created by narrowing the reduction group, and both are
kept:

- **Löwner product range.** A lane multiplies `dd/width` factors before any
  reduction, so going from width 32 to width 4 lengthens each serial product 8x
  and it can leave float range. The product is now accumulated as a
  (mantissa, exponent) pair with `frexp` renormalization. Because the rescaling
  is by exact powers of two, this reproduces the naive product bit-for-bit
  wherever the naive product was in range.
- **Uniform trip counts.** `dd` is rarely a multiple of `parts_per_wg` (deflation
  makes it arbitrary), so a plain `eid < dd` bound lets some partitions exit the
  loop while others are still inside it — and the reductions are sub-group
  shuffles that need every lane of the warp. Both loops now iterate a uniform
  number of times and mask the work instead.

### 2.1b Test coverage gap — **[fixed]**

The gap was **not** partition width: `StedcTest.FusedCtaPartitionWidths` already
covers P = 4, 8, 16, 32 at n = 64. The gap was the **deflation regime** — it uses
plain random tridiagonals, which essentially never deflate a subproblem down to
`dd == 1`, so it passed throughout the failure above.

Added `StedcTest.FusedCtaConditionedHeavyDeflation`, which builds matrices with
`random_hermitian_tridiagonal_with_log10_cond_metric` at log10cond 1 / 3 / 5 and
runs all four partition widths — mirroring the `stedc_acc` case that exposed the
bug. Verified to reproduce `CUDA_ERROR_ILLEGAL_ADDRESS` with the fix reverted and
to pass with it in place.

**Numerical note.** Narrowing the reduction group reassociates both reductions.
This is safe here because neither involves cancellation: the Löwner reduction is
a *product* of ratios, and `nrm2` is a *sum of squares*. Only benign relative
error accumulates. This is materially different from the failed "fast path"
documented in `stedc_merge_kernels.cc:44-52`, which changed *where the poles were
stored* during the solve and produced a bimodal orthogonality distribution.
Regardless, orthogonality must be validated, not assumed — see §5.

### 2.2 Recomputing denominators instead of reading `Q` — **[tried, reverted: regression]**

The idea: `maybe_rescale_vectors` read `Q_bid(eid, j)` with `j` strided across
lanes — an `ld`-stride row walk of a column-major matrix, the only uncoalesced
access in the kernel. That value is exactly
`secular_denominator(d_eid, origin_j, tau_j)`, so it can be recomputed from
shared memory instead, which also lets `write_denominator_column`'s global store
disappear and the 2-norm fold into the write pass.

**It was implemented, verified bit-identical, and reverted — it is 5-10% slower.**

| n | item 1 | with §2.2 | control: item 1 + 2 unused shared arrays |
| --- | --- | --- | --- |
| 64 | 0.840 | 0.981 | 0.990 |
| 256 | 4.698 | 5.112 | 5.190 |
| 512 | 12.704 | 13.807 | 14.105 |

The control column is the decisive measurement: item 1's code with the two extra
`local_accessor`s allocated and written but **never read** is just as slow as the
full §2.2 change. So the entire regression is the shared-memory footprint, not
the recompute — the recompute itself is worth ~1-2% (§2.2 beats the control at
every size), it just cannot pay for the occupancy it costs.

Storing `origin` and `tau` per root doubles shared usage from `2*nloc*sizeof(T)`
to `4*nloc*sizeof(T)`. At 32 threads per work-group that halves the blocks
resident per SM.

**The lesson generalizes:** this kernel runs at **0.14% DRAM throughput**.
Removing global traffic buys essentially nothing, while anything that grows the
shared-memory footprint directly costs occupancy — which is the binding
constraint. Optimizations here should *reduce* resident state, not trade memory
for arithmetic. That is what pointed at §2.3, which turned out to be the real
win.

If revisited, it must not add shared memory: keep `tau` plus a 1-byte origin-pole
selector, or stage the shifts in global scratch (coalesced, L1-resident, no
occupancy cost). Expected upside is only the ~1-2% measured above.

### 2.3 Re-tune work-group size — **[done]**

`STEDC_WG_MULTIPLIER_*` was 1 / 1 / 2 / 2 / 4. It is now **8** across the board.

This is the single largest win in this work, and it is only available *because*
§2.1 removed the barriers — which the measurements show directly:

| multiplier | baseline (serial rescale/normalize) | with §2.1 |
| --- | --- | --- |
| 1 | 0.919 | 0.880 |
| 4 | 0.850 | 0.740 |
| 8 | 0.877 (**worse than 4**) | 0.729 |

(n = 64, batch 256, float, ms.) On baseline, widening past 4 *hurts*: a wide
work-group reducing over only `dd` elements wastes most of its lanes. With the
loops partition-parallel the extra width becomes extra concurrent columns, so it
keeps improving. 16 was also tried — marginally better at n = 64, worse at
n >= 256, so 8 is the pick.

Also fixed here: `choose_wg_size` clamped only against
`device::max_work_group_size` (1024), not against what the kernel can actually
launch. At ~80 registers per work-item, 1024 work-items need 81920 registers
against a 65536 limit, and the launch **throws**:

```
Exceeded the number of registers available on the hardware.
The kernel uses 80 registers per work-item for a total of 1024 work-items.
```

It now also clamps to
`kernel_device_specific::work_group_size` for the specific kernel, so an
aggressive multiplier degrades to the largest launchable size instead of
aborting. Pre-existing, but it matters much more now that wide work-groups are
the default.

Note on the benchmark: `stedc_benchmark` used to substitute fixed fallbacks
(`threads_per_root = 32`, `wg_multiplier = 1`) when those args were 0, so it
silently measured a configuration the library never runs by default. It now
passes non-positive values through, where `StedcParams` resolves them from the
tuning tables — so `... <n> 256 16 2 0 0` measures what a real caller such as
`syev` actually gets.

### 2.4 Deflation-aware back-transform GEMM — **[done]**

`stedc.cc` issued a full dense `n x n x n` GEMM per merge level. The structure it
ignored: `Qprime` is identity-filled and only its first `n_reduced` columns carry
secular eigenvectors, so as a block it is `M = [W | I]`.

The algebra that makes this exploitable without a scatter:

```
eigvects = A * M[:, perm] = (A * M)[:, perm]
A * M    = [ A*W | A(:, dd:) ]
```

Permuting `M`'s columns permutes the product's columns identically, so the sort
can be applied *after* the multiply — and then the deflated columns of the
product are literally columns of `A`, needing no multiply at all. Only the first
`dd` columns need a GEMM.

**Keeping it on cuBLAS.** `dd` varies per batch item, and a per-item GEMM would
be ragged — which would drop off the vendor batched kernel onto the homemade
heterogeneous path and almost certainly lose. Instead a single batch-wide
`dd_max` is used, which keeps one uniform batched call. This is still exact: for
an item with `dd < dd_max`, columns `dd..dd_max-1` of `M` genuinely are identity
columns, so the GEMM reproduces `A` there.

No extra workspace: `A` goes into `temp_Q`, which frees `eigvects` to receive the
narrow GEMM result, which is then folded back over `A`'s head.

**How much deflation is there?** Measured on random tridiagonals (batch 64):

| merge size | mean `dd` | kept | min–max |
| --- | --- | --- | --- |
| 256 | 43.8 | 17.1% | 25–61 |
| 128 | 43.9 | 34.3% | 19–69 |
| 64 | 41.3 | 64.5% | 21–57 |
| 32 | 29.0 | 90.6% | 10–32 |

Deflation is heaviest exactly where the GEMM is most expensive, and the batch is
homogeneous enough that `dd_max` is close to the mean — so the uniform-width
compromise costs little.

**The catch: it needs a host sync.** `dd_max` must be known host-side to size the
GEMM, and that stalls the enqueue pipeline. Measured (feature off vs on,
interleaved, idle GPU, batch 256, float, ms):

| n | off | on | delta |
| --- | --- | --- | --- |
| 256 | 3.998 | 4.020 | -0.5% (noise) |
| 512 | 11.166 | 10.525 | **+5.8%** |
| 1024 | 39.388 | 29.937 | **+24.0%** |

So it is gated at `n >= 512` (`stedc_deflation_gemm_min_n`). Below that the
recursion has many small merge nodes and the syncs cost more than the saved
flops — at a threshold of 128 the n=256 case regressed ~5%. A second guard skips
the narrow path unless deflation removed at least 25% of the columns, so weak
deflation cannot make it a loss beyond the single sync.

The remaining structure LAPACK exploits (`dlaed2`/`dlaed3`'s `CTOT` split of the
non-deflated columns into two blocks, for a further ~2x) is **not** implemented:
it would split the one uniform GEMM into two smaller ones with per-item widths,
which is exactly the ragged shape that would leave the vendor kernel.

### 2.5 Eigenvalues-only still pays for eigenvectors — **[open]**

`jobz` is threaded through `stedc_impl` but between the recursion
(`stedc.cc:110-111`) and the GEMM (`:424`) there is no `NoEigenVectors` branch —
the merge appears to build and back-transform eigenvectors unconditionally.
Worth confirming; if true it is free money for that path, and see §3 for the
matching algorithm.

### 2.6 What is already good — leave alone

The secular root solver (`stedc_merge_cta.cc`, `solve_root_roc_generic`) is the
rocSOLVER middle-way / fixed-weight hybrid with bound-clamped steps and a proper
`|f| <= eps * err` convergence exit. That is the literature-recommended scheme
(Li's middle way; Gu–Eisenstat stability). Do not touch it.

---

## 3. Literature

- **Gu & Eisenstat, "A Stable and Efficient Algorithm for the Rank-One
  Modification of the Symmetric Eigenproblem", SIMAX 1994.**
  <https://epubs.siam.org/doi/10.1137/S089547989223924X>
  The foundation of the Löwner-based eigenvector formula used here: computes
  eigenvectors stably without extended precision. Confirms the current approach.

- **Jakovčević Stor, Slapničar & Barlow, "Forward stable eigenvalue decomposition
  of rank-one modifications of diagonal matrices", arXiv:1405.7537.**
  <https://arxiv.org/pdf/1405.7537>
  Computes each eigenvalue *and all eigenvector components* to high relative
  accuracy in O(n). Could let us drop the `enable_rescale` pass entirely rather
  than optimize it — strictly better than §2.2 if it holds up.

- **Liao, Li, Xia et al., "New fast divide-and-conquer algorithms for the
  symmetric tridiagonal eigenvalue problem", arXiv:1510.04591.**
  <https://arxiv.org/abs/1510.04591>
  The rank-one eigenvector matrix is Cauchy-like with off-diagonally low rank;
  HSS approximation gives O(N^2 r) instead of O(N^3), reported >6x over MKL on
  large matrices with few deflations. Relevant only to a large-n path, not to
  batched small n.

- **Zhan & Zhang, "Reducing Internal State in Eigenvalue-Only Divide-and-Conquer
  Tridiagonal Eigensolvers", arXiv:2605.26599 (May 2026).**
  <https://arxiv.org/pdf/2605.26599>
  Boundary-row D&C: the conquer phase needs only selected boundary rows/columns,
  not the accumulated eigenvector matrix. Quadratic to linear memory, and the
  matrix update disappears. Directly targets §2.5.

- **FMM-accelerated secular evaluation** (Gu–Eisenstat lineage). Evaluating the
  secular function at all roots simultaneously is a Cauchy matvec — O(dd log dd)
  instead of O(dd^2). Our per-iteration `evaluate_roc_secular` is O(dd) per root.
  Probably not worth it below dd ~ 512, but it is the asymptotic answer.

- **LAPACK reference:** [dlaed2](https://netlib.org/lapack/explore-3.2-html/dlaed2.f.html),
  [dlaed3](https://www.netlib.org/lapack/explore-3.2-html/dlaed3.f.html) — the
  `CTOT` block partitioning behind §2.4.

- **[rocSOLVER tuning guide](https://rocm.docs.amd.com/projects/rocSOLVER/en/develop/reference/tuning.html)**
  — our solver follows their STEDC scheme; useful for the minimum-block-size
  heuristics.

---

## 4. Prioritized plan

1. **[done]** Partition-parallelize rescale + normalize — all `2*dd` work-group
   barriers removed. Required fixing the pre-existing `dd == 1` out-of-range
   read in §2.1a.
2. **[tried, reverted]** Recompute denominators from shared memory. Bit-identical
   but 5-10% slower; the extra shared memory costs more occupancy than the saved
   global traffic is worth (§2.2). Do not retry without shrinking the footprint.
3. **[done]** Raise `STEDC_WG_MULTIPLIER_*` to 8, unlocked by item 1, plus a
   kernel-register clamp in `choose_wg_size` (§2.3). Largest single win.
4. **[done]** Deflation-aware back-transform GEMM (§2.4). Gated at n >= 512;
   +24% at n = 1024. The further `CTOT` two-block split is deliberately not
   done — it would make the GEMM ragged and lose the vendor kernel.
5. **[open]** `NoEigenVectors` short-circuit (+ boundary-row D&C from
   arXiv:2605.26599).

**Where the remaining headroom is.** Items 1 and 3 addressed the merge kernel's
serialization and occupancy; item 4 addressed the back-transform. §2.2's control
experiment shows the merge kernel is constrained by resident state, not memory
traffic, so further work there should reduce registers/shared memory rather than
restructure data flow. The largest remaining item is §2.5 (eigenvalues-only).
`SteqrCTAKernel` is now the biggest single kernel at ~25% of GPU time and has not
been looked at.

## 5. Validation protocol

Any change in this area must be checked for **orthogonality**, not just
residuals. The comment at `src/extensions/stedc_merge_kernels.cc:44-52` records a
prior regression that passed residual checks while producing a bimodal
orthogonality distribution for float STEDC at n <= 64.

- `tests/stedc_tests.cc` — correctness gate.
- `benchmarks/orthogonality_accuracy.cc`, `benchmarks/eigensolver_accuracy.cc` —
  compare the *distribution*, not just the max, against the WG variant
  (`StedcMergeVariant::FusedWg` / the baseline 3-kernel path), which is
  deliberately left bit-identical as the reference.
- `benchmarks/stedc_benchmark.cc` — timing, plus re-profile with the `ncu`
  recipe in the skill reference to confirm occupancy actually moved.

---

## 6. Measurements (RTX 4090, sm_89, CUDA backend)

**Measurement hygiene.** This box has two RTX 4090s and other benchmarks may be
running. A contended GPU inflated an early reading of this work from 5.5% to
8.6%. Pin to an idle device (`CUDA_VISIBLE_DEVICES=1`) and check
`nvidia-smi --query-compute-apps=pid,process_name --format=csv` before trusting
any number here.

### Correctness

`stedc_acc --backend=CUDA --samples=64`, float and double, n = 16..512:
**0.00000 Fail%** everywhere, residual and orthogonality at machine-epsilon
levels. Every result in this document is **bit-identical** across §2.1, §2.2 and
§2.3 — the refactors are numerically exact, and the work-group width does not
enter the per-column math because the reductions are partition-local.

Orthogonality `||Z^T Z - I||_F / n`, float, `--samples=256`, vs baseline:

| n | baseline | with change |
| --- | --- | --- |
| 16 | 1.21899e-07 | 1.21899e-07 |
| 32 | 9.36432e-08 | 9.38560e-08 |
| 64 | 7.41322e-08 | 7.44386e-08 |

`stedc_tests` CUDA: 44 passed. `syev_tests` 8/8, `syev_cta_tests` 28/28,
`syev_blocked_tests` 32/32.

Pre-existing failures, verified to reproduce identically with these changes
reverted: 8 `FlatTraceCollapsesByDepth[Ragged]` in `stedc_tests`, and 4
`steqr_tests` failures on Backend 6 (host/CPU).

### Performance

End-to-end `stedc`, batch 256, float, idle GPU, **as a real caller gets it**
(tuning-resolved parameters — `stedc_benchmark ... <n> 256 16 2 0 0`):

| n | baseline (main) | this work | speedup |
| --- | --- | --- | --- |
| 64 | 0.930 | 0.782 | **15.9%** |
| 128 | 2.042 | 1.633 | **20.0%** |
| 256 | 4.885 | 3.967 | **18.8%** |
| 512 | 13.144 | 11.162 | **15.1%** |

Attribution at matched settings (multiplier 1, so §2.1 alone): 0.933 -> 0.882 at
n = 64 and 4.948 -> 4.676 at n = 256, i.e. ~5.5%. The rest comes from §2.3, which
§2.1 is what makes possible.

Caution: **arg3 selects the merge variant** (`-1` Auto, `0` Baseline, `1` Fused,
`2` FusedCta). An early measurement used `arg3=0`, the Baseline 3-kernel path,
which does not exercise this code at all and produced a spurious result.

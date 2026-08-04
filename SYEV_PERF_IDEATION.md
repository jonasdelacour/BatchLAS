# SYEV / SYEVX: current status and where the next performance is

Status: **exploration + ideation only.** Nothing here is implemented and nothing here is
measured by this document. Every number quoted is copied from an existing in-tree
measurement, with its source cited. Items marked *unmeasured* are hypotheses.

Written against `main` at `e2ff635` (merge of PR #51).

---

## 1. Where the two solvers stand today

### 1.1 SYEV — six providers, one hand-written routing table

| Provider | File | Shape | Routed by `Auto`? |
|---|---|---|---|
| `Vendor` (cuSOLVER) | `src/backends/cusolver.cc` | any | yes — the default nearly everywhere |
| `BatchLAS_CTA` | `syev_cta.cc` | n ≤ 32 | yes, all n ≤ 32 |
| `BatchLAS_Blocked` | `syev_blocked.cc` | any, `Uplo::Lower` | yes, only 320 ≤ n ≤ 640 & batch ≥ 128 |
| `BatchLAS_TwoStage` | `syev_two_stage.cc` | any, `Uplo::Lower` | yes, only eigenvalues-only & n ≥ 512 & batch ≥ 256 |
| `syev_cta_fused` | `syev_cta_fused.cc` | n ≤ 32 | **no — unreachable from `Auto`** |
| `syev_jacobi_cta` | `syev_jacobi_cta.cc` | n ≤ 32 | **no — unreachable from `Auto`** |

Routing lives in `detail::choose_syev_provider` (`include/blas/functions/syev.hh:296`) and is
gated `if constexpr (B == Backend::CUDA)` — rocSOLVER keeps the historical provider order,
which is an ordering rather than a decision.

The measured basis for that routing, all RTX 4090 / float:

- **Eigenvectors** (`syev.hh:154`): vendor wins everywhere except one connected box,
  320 ≤ n ≤ 640 with batch ≥ 128 (blocked ahead by up to 1.37×). Outside it the vendor
  margin reaches **15.3×** (n=1024, batch=1).
- **Eigenvalues only** (`syev.hh:181`): two-stage beats the vendor by **2.0–2.9×** at
  n ≥ 512, batch ≥ 256, after the stage-2 chase fix (`27851a6`).
- **n ≤ 32 with vectors** (`syev.hh:246`): moving the projected Rayleigh–Ritz solve off CTA
  onto the vendor is worth 1.10–1.15× on LOBPCG, **but ships disabled** — it flips one
  already-converged case in `ILUKTests.SyevxInstrumentationAndPreconditioner`.

### 1.2 SYEVX — five algorithms, routed by measurement since PR #49

`syevx_select_algorithm` picks between `Direct`, `DirectSubset`, `Filtered`, `LOBPCG`.
The measured outcome (`SYEVX_PLAN.md` §13) overturned the plan's own cost model:

- **`Direct` unless eigenvectors are wanted *and* n ≥ 1024.** The predicted ~3× ceiling for
  `DirectSubset` was never approached; best measured is **1.46×**.
- `DirectSubset` is a **3–5× pessimisation in eigenvalues-only mode** at every measured
  shape, and `Auto` used to route into that loss.
- `Filtered` wins only at n ≥ 1024, k/n ≈ 1%, small batch — under 2×, and it is the only
  path with a convergence failure mode, so it stays opt-in.
- LOBPCG is 10–100× behind `Direct` on dense input and remains the sparse-path solver.

### 1.3 The recurring defect

Four of the last six SYEV performance commits are the same bug in different kernels: the
work decomposition takes **all** its parallelism from the batch dimension, one work-group per
matrix, so at small batch the GPU sits idle. `87f6887` fixed it for the blocked path's panel
factorization with a co-resident multi-work-group grid barrier, worth **1.9–4.1× at n ≥ 1024**
(`latrd_lower_panel.cc:45`).

**That fix is not yet applied anywhere else.** `stedc.cc` launches every one of its merge
kernels as `nd_range(batch_size * 128, 128)` — lines 140, 176, 318, 350, 396, 423 — i.e. one
work-group per matrix, for a phase that is ~1/3 of the total flops of a full eigensolve.

---

## 2. Ideas, ranked by expected value per unit of work

### #1 — The eigenvector routing table is stale. Re-measure it.

**This is the highest-value item on the list and it requires no new code.**

The blocked-vs-vendor eigenvector grid in `syev.hh:154-180` was measured at commit `27851a6`.
The grid-barrier `latrd` path landed *after* it (`87f6887`, defaulted on at n ≥ 768 in
`5401f63`) and is worth 1.4–4.1× in exactly the region where that table records the vendor's
largest margins. The 15.3× at n=1024/batch=1 in the table **is** the panel-starvation cliff
that the grid path exists to remove; the table's own comment names the mechanism.

Separately, the 768 gate itself (`latrd_grid_min_n`) was measured **eigenvalues-only**. It is
applied in both modes.

Actions:
1. Re-run the blocked/vendor eigenvector grid with the current default (grid latrd on).
2. Re-run the `latrd_grid_min_n` sweep in eigenvector mode; split the constant per mode if
   the crossovers differ.
3. Widen or move the 320–640 carve-out to wherever the new numbers put it.

Risk: low. Worst case the table is confirmed and the comment gets a date.

### #2 — Apply the grid-barrier pattern to `stedc`

Once `latrd` stops dominating at small batch, `stedc` is the obvious next term, and it has
the identical decomposition (`nd_range(batch*128, 128)`, six kernels). The machinery already
exists and is proven: sense-reversing device-scope barrier, work-group count capped at
`MAX_COMPUTE_UNITS` so co-residency is guaranteed, fixed group order so reductions stay
run-to-run deterministic, `G == 1` dispatching to the legacy kernel bit-for-bit.

Sequence matters: **profile first** to establish that `stedc` is actually the next dominant
term at batch 1 with grid latrd enabled, rather than assuming it from the flop count. The
`BATCHLAS_KERNEL_TRACE_SCOPE` markers are already in `syev_two_stage.cc`.

*Unmeasured.* The barrier cost is O(nodes) per merge level rather than O(n) per panel column,
so the crossover will sit somewhere different from 768 — possibly much lower, since there are
far fewer barriers.

### #3 — Add a routing-audit benchmark so tables cannot go stale silently

Idea #1 exists because a hand-written grid in a header aged out of date without anything
noticing. Fix the class, not the instance: a `BM_SYEV_RoutingAudit` that, for each shape in
the routing table, runs both the provider `Auto` chose *and* its nearest runner-up, and
reports every cell where the routed choice loses.

This turns "the table is stale" from an invisible condition into a benchmark row. It is also
the natural harness for ideas #1, #4 and #5 — each of them is one column of the same sweep.

Cheap to build (the provider is already forceable via `BATCHLAS_SYEV_PROVIDER`), and it makes
every subsequent routing change a diff against a machine-generated table rather than a
hand-typed one.

### #4 — Re-tune the two-stage `kd`, then reconsider two-stage for eigenvectors

Two facts that have not been composed:

- `choose_two_stage_kd` (`two_stage_common.hh:38`) measured **kd = 32 optimal at every
  n ≥ 256**, with two-stage beating blocked by 1.13× at n=1024.
- `f7f3c57` then changed how the panel back-transform blocks: `ormqr` was keying its WY block
  width on the panel *height*, so a kd=32 panel was split into two k=16 WY blocks. Passing
  `nb = kd` was worth **1.19–1.36× at n ≥ 1024, batch ≥ 32**.

The kd table was measured under the split-WY behaviour, which **structurally penalises wide
kd** — a wider band bought nothing extra because `ormqr` chopped it back to 16 regardless.
With that removed, the optimum should move up, and the tuning literature agrees: the file's
own comment cites Gates/Tomov/Dongarra measuring 32/64 without vectors → 96/128 with, and
MAGMA using band nb = 128.

Chain: re-sweep kd with the nb hint on → if the optimum moves to 64–128, two-stage's
eigenvector performance improves by more than the 1.13× currently on record → at n = 1024 the
gap to the vendor is 1.46/1.13 ≈ 1.29×, which is close enough that a real kd win could flip
it. Two-stage is currently **never** routed in eigenvector mode.

*Unmeasured, and the chain has three links.* But each link is individually plausible and the
first one is a sweep of an existing environment variable.

### #5 — Ship the small-n projected-solve win by fixing the test, not the perf

`BATCHLAS_SYEV_CTA_MAX_N` is a measured 1.10–1.15× on LOBPCG at batch 8 and 1.03–1.06× at
batch 64, monotone in the threshold, and it ships **off** because at threshold 16 one of eight
cases in `ILUKTests.SyevxInstrumentationAndPreconditioner` crosses to ratio 1.25 — at a point
where the baseline has already converged to 4.2e-06.

The blocker is a test-tolerance question, not a performance one, and it is explicitly flagged
in `syev.hh:264` as "someone else's correctness assertion, not mine to relax". It needs an
owner decision: either the assertion tolerates a near-tie on an already-converged case, or
that test pins its projected-solve provider so the two concerns stop colliding.

This is the cheapest already-measured win available.

### #6 — Decide what `syev_cta_fused` and `syev_jacobi_cta` are for

Both are implemented, tested (30 tests for Jacobi, all passing) and **unreachable from
`Auto`**. They are currently carrying build and test cost with zero routed users.

For Jacobi the measured picture (`JACOBI_EIGENSOLVER_PLAN.md` §13.2) actually supports a
routing rule already:

- **double**: Jacobi is faster at every n ≤ 32 (1.2–3.8×).
- **float**: wins to n = 16 (1.1–3.9×), loses at n = 32 (0.4–0.6×).
- **accuracy**: on graded SPD input, max relative eigenvalue error 4.5e-07 against `syev_cta`'s
  2.7e+28. That is the whole point of the solver and it is confirmed.

So `Auto` could route `double && n ≤ 32 → Jacobi` and `float && n ≤ 16 → Jacobi` on speed
alone, with the accuracy win free on top. Caveat the plan states honestly: part of the double
margin is that this is a 1/64-FP64 consumer card, so **the float column is the better
predictor for a datacenter GPU** — the double rule should not ship without a re-measure on
1:2 FP64 hardware, or should be gated on measured FP64 throughput rather than on the type.

For `syev_cta_fused` there is no comparable measurement in tree. Measure it against
`syev_cta` at n ≤ 32; route it or retire it.

### #7 — Look at *why* `Filtered` collapses at large batch

`Filtered` wins at n=1024, k/n≈1%, batch 1 (62 ms vs 114 ms Direct) and loses catastrophically
at batch 64 (23.5 ms vs 3.5 ms per matrix). A path that wins when the GPU is starved and loses
when it is saturated is a suspicious shape — it suggests the filter itself (two GEMMs per
Chebyshev step, which should saturate beautifully) is not what costs, and that the
`ortho` / Rayleigh–Ritz tail is. Worth one profile before any more tuning goes into it.

If the tail is the cost, the same fix helps `Filtered` and LOBPCG at once — both spend their
non-GEMM time in `ortho` and a small projected `syev`.

### #8 — Big swing: a grid-resident whole-solve for small batch

`syev_cta_fused` proves a whole symmetric eigensolve fits in one kernel when it fits in one
work-group (n ≤ 32). The grid barrier from `87f6887` is the missing primitive for doing the
same across the *device* rather than across a work-group — one persistent kernel running
reduction → tridiagonal solve → back-transform for a single matrix on all 128 SMs, with no
launch or drain between phases.

That is the natural endpoint of the batch-starvation work: batch = 1, n = 256–1024 is where
BatchLAS is furthest behind the vendor (up to 15.3×) and where every fix so far has been
incremental. Speculative, expensive, and should not start until #1 and #2 have said how much
of that 15.3× is left after the cheap fixes.

---

## 3. Known blind spots in the current measurements

Not tasks — things to state whenever these numbers are quoted:

- **Everything is float.** `syev.hh:175` says so explicitly for the provider grid, and the
  carve-out box is a 1.16–1.37× margin that could sit elsewhere in double or complex.
- **Everything is CUDA / RTX 4090.** rocSOLVER routing is untuned by construction, and the
  4090's 1/64 FP64 rate distorts every double-precision comparison against a vendor library
  that is compute-bound in FP64.
- **The machine has two idle RTX 4090s.** `SYEVX_PLAN.md` §13 lists §7.11 (the projected-`syev`
  provider sweep) and the §7.5 soft-locking A/B under Chol2 as "now possible" and still not
  done. Both are sweeps, not code.
- **Measurement hygiene applies to all of the above**: compare only at saturation, match
  benchmark names exactly (`--name` is a substring filter — that is what corrupted the
  earlier eigenvector grid), and watch for a second process on the GPU.

---

## 4. Suggested order

1. **#1** re-measure the eigenvector routing grid (no code; may move a whole region)
2. **#5** resolve the `ILUKTests` collision and ship the small-n win (already measured)
3. **#3** build the routing-audit benchmark (makes #1 permanent, harnesses #4 and #6)
4. **#4** re-sweep two-stage `kd` with the nb hint on
5. **#2** profile at batch 1, then port the grid barrier to `stedc` if it is the next term
6. **#6 / #7** measure `syev_cta_fused`; profile `Filtered`'s large-batch tail
7. **#8** only after 1–5 have bounded what is left

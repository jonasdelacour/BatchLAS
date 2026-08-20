# Complex GEMM — the routing defect, and the merge gate that reshaped the fix

## The defect (real, and the same one float had)

`can_use_64x64_k16_wide_fast_path` is the **aligned LEG's** predicate. The
dispatcher at `src/sycl/gemm_kernels.cc:803` re-evaluates it and picks
`<true>`/`<false>` itself, so a call that fails it still runs the wide kernel on
its predicated leg. The kernel's only hard predicate is **NN**, at `:802`.

But `:631` consulted that predicate as a **ROUTING gate**, and failing it did not
demote the call to the predicated leg — it handed the call to **Tiled16**
(`:638`). This is verbatim the defect fixed for the float 128x128 kernel in
`3f0afbd`.

Trace-confirmed over 120 rows (`gpu0/kernels.txt`), not inferred: `auto` names
`gemm_sycl_tiled16` on every refused shape.

Forced-wide against the route it replaces, at **saturation**, both betas,
geomean over 116 refused cells: **cfloat 3.98x, cdouble 2.90x**, null controls
1.000. That number is real. It is also measured in a regime these call sites
never enter — which is the whole point of this document.

## The merge gate: it FIRED, and it changed what shipped

Every win above was measured at **batch 128-8192**. Every demand shape the
relaxation newly captures runs at **batch 1-8** (`gpu1/complex_gemm_demand.csv`:
the Step-1 set is `{1:78, 2:84, 3:32, 4:68, 5:80, 8:44, 128:6}` — nothing else).
**The measured regime and the demand regime do not overlap.**

The launch arithmetic says why: the wide kernel launches
`ceil(m/64)*ceil(n/64)*batch` CTAs against Tiled16's `ceil(m/16)*ceil(n/16)*batch`
— up to 16x fewer. At 129x96x129 b5 that is **30 CTAs against 270** on a 128-SM
part, so wide cannot fill the machine while Tiled16 can.

`smallbatch/`, median of 9, `tiled16_ms / wide_ms`, **>1 means wide wins**:

| type | shape | captured batch | ratio there | ratio at b256 |
|---|---|---|---|---|
| cfloat | 33x61x33 | 5 | **0.630** | 1.969 |
| cfloat | 64x48x64 | 2 | **0.651** | 3.288 |
| cfloat | 96x64x96 | 3 | **0.659** | 3.808 |
| cfloat | 129x96x129 | 5 | **0.803** | 2.713 |
| cfloat | 300x32x300 | 1 | **0.625** | 2.498 |
| cfloat | 129x48x129 | 2 | 0.598 * | 2.738 |
| cdouble | 33x61x33 | 5 | **0.174** | 1.820 |
| cdouble | 64x48x64 | 2 | 0.242 * | 2.611 |
| cdouble | 96x64x96 | 3 | 0.238 * | 2.606 |
| cdouble | 129x96x129 | 5 | **0.601** | 1.754 |
| cdouble | 300x32x300 | 1 | **0.230** | 1.643 |
| cdouble | 129x48x129 | 2 | 0.209 * | 1.800 |

`*` = the Tiled16 arm had relative sd > 10% at that tiny batch; direction agrees
with every clean cell, so it is reported but not relied on.

**Wide loses at the captured batch in 12 of 12 cells and wins at b256 in 12 of
12.** cdouble is far worse small — 0.174x is a 5.7x regression. A bare
`min(m,n) >= 32` floor, which is exactly what the 2.90-3.98x geomean argues for,
would have regressed every shape it newly captured. The relaxation still
shipped, but gated on work-group count rather than on `min_dim` alone.

Nothing in `ctest` asserts on kernel choice or throughput, and `route_diff.sh`
records resolver **Routes**, not `select_kernel_variant`'s **KernelVariant** — so
this regression would have been **completely silent**, visible only as the suite
getting slower.

## What ships instead: a CTA-count gate

The 180-cell ladder in `batchsweep/` (batch 1..256 x 5 shapes x 2 types) shows
the crossover is **not a constant batch** -- it moves 8 -> 128 depending on shape:

| type | shape | b1 | b8 | b16 | b32 | b64 | b128 | b256 | crossover |
|---|---|---|---|---|---|---|---|---|---|
| cfloat | 1024x1024x32 | 4.18 | 4.72 | 2.35 | 2.38 | 2.41 | 2.41 | 2.37 | **b>=1** |
| cfloat | 129x96x129 | 0.58 | 1.02 | 1.60 | 1.93 | 2.65 | 2.75 | 2.73 | b>=8 |
| cfloat | 96x64x96 | 0.64 | 0.77 | 0.96 | 1.64 | 2.75 | 3.68 | 3.88 | b>=32 |
| cfloat | 33x61x33 | 0.60 | 0.56 | 0.65 | 0.75 | 1.08 | 1.55 | 1.98 | b>=64 |
| cdouble | 1024x1024x32 | 3.34 | 3.34 | 3.35 | 3.33 | 3.35 | 3.36 | 3.36 | **b>=1** |
| cdouble | 129x96x129 | 0.21 | 0.79 | 1.37 | 1.38 | 1.79 | 1.77 | 1.76 | b>=16 |
| cdouble | 96x64x96 | 0.24 | 0.45 | 0.67 | 1.31 | 2.59 | 2.60 | 2.58 | b>=32 |
| cdouble | 33x61x33 | 0.17 | 0.18 | 0.33 | 0.48 | 0.93 | 1.82 | 1.82 | b>=128 |

**But it IS very nearly a constant number of work-groups.** The kernel launches
`ceil(m/64)*ceil(n/64)` CTAs of 256 threads per batch item. Re-indexing every
clean cell by that count instead of by batch collapses the eight different
crossovers onto two:

    cfloat   min_dim >= 32 AND ctas >= 64   -> 26 cells, WORST 1.08x, zero losses
    cdouble  min_dim >= 32 AND ctas >= 128  -> 24 cells, WORST 1.08x, zero losses

cdouble needs twice the CTAs because its 32 KB of shared memory caps it at
3 blocks/SM (`register_64x64_k16_wide.hh:271-272` allocates `2*1024*sizeof(D)`),
so it takes more work-groups to fill the same 128 SMs.

Boundary counterexamples, all measured:

* cfloat just below 64 CTAs: `129x96x129 b8` = 48 CTAs, **0.79x LOSS**.
* cdouble just below 128: `33x61x33 b64` = 64 CTAs, **0.93x LOSS**.
* **64 CTAs is genuinely ambiguous for cdouble** -- it holds both that 0.93x
  loss and a 1.31x win (`96x64x96 b32`). 128 is chosen to admit no loss, and it
  knowingly gives up a real 1.37x (`129x96x129 b16`). Conservative on purpose.
* `min_dim >= 32` is needed independently: the CTA gate alone admits tiny shapes
  at huge batch, and `16x16x16` loses 0.71x (cfloat) / 0.28x (cdouble).

Landed in `7d84208`. The pre-existing `min_dim >= 256` arm is kept ahead of the
new one so nothing that routes to this kernel today stops doing so; `256^3 b4`
and `512^3 b1` were verified unchanged by trace.

## What did ship from this investigation, separately

A wrong-answer bug, unrelated to routing and found while reading the dispatch:
nine transposed launchers ran unconditionally for any transpose form, so a
`ConjTrans` call silently computed **unconjugated**. Fixed in `f236575`, with a
test verified to fail without the guard. See that commit.

## Caveat on what the demand table is evidence OF

The batch 1-8 figures come from a `ctest` coverage capture. Project policy is
that batch=1 is not an optimisation target, so this table is evidence about
**test-suite runtime**, not about user workloads — **no capture of user
workloads exists**. The honest reading is: the relaxation must not REGRESS the
small-batch population, and it stands to help a large-batch population whose
size here is unknown.

## Data

`gpu0/` — the forced-vs-auto ratio campaign at saturation, and `kernels.txt`
(trace confirmation of which kernel actually ran). `gpu1/` — the demand capture
with probe rows removed two independent ways that agree exactly (4527 real
complex calls), the cost-of-vendor-freedom measurements, and the standalone
harness `cx_gemm_bench.cpp` (the tree's `gemm_benchmark` is NN-only and
structurally cannot measure transposed shapes). `smallbatch/` — the merge gate
above. `batchsweep/` — the crossover ladder. `routing_proposal/` — the patch as
first proposed, kept unapplied because this gate refuted its floor.

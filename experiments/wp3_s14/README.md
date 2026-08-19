# WP3 step 14 — the cooperative CTA solve: built, measured, REJECTED

## Verdict first

V3 works exactly as designed at the level it was designed for, and loses at the
level that matters. It is **not** in the tree; the tree stays on the step-13
schedule. This directory is the record of why, so the next person does not
rebuild it.

| float `Side::Left`, worst of cells clean in both runs | 8 | 16 | 32 | **64** | **128** | 256 | 512 |
|---|---|---|---|---|---|---|---|
| step 13 (V1 + two-level blocking) | 1.59 | 1.72 | 1.77 | **1.48** | **1.18** | 0.86 | 0.76 |
| step 14 (V3 cooperative) | 1.59 | 1.72 | 1.80 | **0.39** | **0.80** | 0.84 | 0.77 |

`Side::Right` is byte-for-byte unaffected (1.61→1.62, 3.36→3.44, 1.55→1.55,
1.23→1.24), which confirms the side gating held.

## What was built

V1 gives one work-item a whole solve, so the solution vector is `x[N]` in that
thread's registers and the order is capped at 32 — N=64 produces a 456 B stack
frame. That cap pins V2's block width, and the block width sets the traffic:
at n=512, B elements touched per batch item in units of q are 5824 (nb=32),
4096 (two-level, shipping), 3328 (nb=64), 2560 (nb=128), against an ideal 1024.

V3 splits the array: W work-items cooperate on one solve, thread `w` owning the
canonical rows `{w, w+W, w+2W, …}` and holding `NL = N/W` accumulators, with each
`x_s` exchanged by a sub-group shuffle.

Three design points, each of which mattered:

**The loop order is the whole trick.** A runtime `acc[t/W]` to reach the owner's
element puts `acc` in local memory — the exact failure V3 exists to avoid.
Scanning the local array costs a predicated pass per step (`N·NL` against an
ideal `N²/2W`, i.e. 2×). *Block* distribution makes the owner index
compile-time but is 7× load-imbalanced at W=8, since row `i` costs `i` FMAs.
Cyclic distribution with the **local index outermost** is exact: writing
`t = wblk + m·W` and unrolling `m`, the owner's local index *is* `m`, and
`i > t ⟺ nn > m, or nn == m and w > wblk` — a compile-time loop bound plus one
uniform runtime predicate. Every FMA executed is a needed one.

**Coalescing comes from the lane map, not a staging tile.** With `w = lane % W`
and W=8, the eight lanes of a `Side::Left` solve read eight consecutive rows of
one column — 32 B, exactly one sector — so a warp covers four solves in four
sectors, what a fully coalesced 128 B access costs. V3 therefore needed none of
the §3.4 staging tile that V1 requires. (The same map is wrong for
`Side::Right`, which wants consecutive columns; V3 was Left-only, which is also
the only side with a measured gap.)

**And the register mechanism worked.** `scripts/register_probe.sh`, N=128, W=8,
zero stack frame and zero spill in every row:

| type | registers | max work-group |
|---|---|---|
| float | **106** | 512 |
| double | 136 | 256 |
| complex\<float\> | 139 | 256 |
| complex\<double\> | 174 | 256 |

float holds a **4× larger order in fewer registers than V1 holds at N=32** (106
vs 114). The premise of the redesign was correct.

## Why it still lost

The decisive cell is **order 128**: V3 fits it exactly, with zero padding waste,
and still goes 1.18× → 0.80×. So the regression is not the padding — the kernel
is intrinsically ~1.5× slower than V1-plus-blocking at the same order. Order 64,
which V3 pads from 64 to 128 and so does 4× the arithmetic, collapses to 0.39×,
consistent with that waste on top.

And at orders 256 and 512 — where V3 replaces the entire inner blocking level and
the traffic model says 4096 → 2560 q-units, a predicted 1.6× — the measurement
moves by 0.02×. Nothing.

The mechanism the model missed: V3's recurrence is `N` sequential steps each
gated by a shuffle, so the critical path is 128 dependent
`shuffle → scale → FMA` chains. V1 at N=32 has 32 such steps and fills each with
32 independent FMAs, and the blocked driver's trailing GEMMs between blocks are
fully parallel, well-tuned kernels. **V3 trades parallel GEMM work for serial
in-kernel recurrence, and that is a bad trade** even when it removes DRAM
traffic. The traffic model counts bytes; it does not count the critical path.

This is the same shape of result as the earlier CTA-large-n rejection, where a
CTA-resident large-order solver measured 85–211× slower than blocking.

## What this leaves

The remaining float/`Side::Left` gap at order ≥ 256 with `q·batch ≥ 524288`
(0.76–0.86×) is therefore **not** in the diagonal solve, and not in the inner
blocking level — removing that level entirely changed nothing. It is in the
outer trailing GEMM or in the algorithm's fundamental traffic, and any further
attempt should measure there first rather than at the diagonal.

Routing already covers those cells: `preferred()` sends them to the vendor, so
BatchLAS's trsm is not slower than cuBLAS anywhere in the grid.

## Data, and an honest note about it

The sweep used `measure.py` — the same driver and protocol as
`experiments/wp3_s13` — over the full 8..512 grid with V3 wired in, and the
comparison above is restricted to the 81 cells that were clean (relative
sd <= 10%) in *both* runs, so it is like-for-like.

**The per-cell CSVs did not survive.** They were deleted after the analysis and
before being aggregated, and the run had not written its summary file because it
was interrupted when the shared library was rebuilt underneath it. The table at
the top of this file is therefore the record, not a derivation from committed
data. That is a process failure on my part and it is written down rather than
papered over; anyone who wants the raw numbers back must re-run.

`v3_cooperative_kernel.patch` is the rejected kernel, as a diff against the
shipping `src/sycl/trsm_native.cc`. It is kept so the design is reproducible
without living in the build — the device link is this project's long pole and a
dead kernel in it costs every developer time. Re-applying it also needs
`dev_select` and `fma_acc_neg` in `src/sycl/device_scalar.hh` (a sub-group
shuffle specialised component-wise for complex, and an `acc -= a*b`).
`measure.out` is the run log.

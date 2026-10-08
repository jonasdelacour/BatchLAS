# STEQR (CTA solver) {#perf_steqr}

> **Status:** current · RTX 4090 (sm_89)

`steqr_cta` solves one tridiagonal problem per chunk of `P` lanes, `P` in {4, 8, 16, 32}, so a
warp holds `32/P` problems. The device solve `steqr_cta_solve` (`src/extensions/steqr_cta_device.hh`)
also runs inside the fused small-n SYEV kernel and at the stedc leaves. This page records the
lane-scheduling choices that shipped, the ones that did not, and their numbers.

**Setup:** RTX 4090 (sm_89), GPU 1, float and double, eigenvectors on (`V`) or off (`N`), Lapack
shift, EXP update, 30 sweeps per eigenvalue, random normal `d` and `e`. Times are the kernel alone,
JIT runs discarded, medians of 3 interleaved rounds, batch 65,536 for n <= 8 and 32,768 above.

## Lockstep flat solver

The nested solver (`steqr_cta_solve_nested`) realigns the chunks of a warp only at loop exits, so
a chunk that deflates early waits and the warp pays the slowest chunk's sweep count per eigenvalue.
`steqr_cta_solve_flat` moves the sweep out of the nest: every chunk settles (advances past deflated
eigenvalues, solves 2x2 blocks), then all chunks with a sweep `[l, m]` chase in one pass. The
per-chunk operations equal the nested solver's, so results are bitwise equal (checked on 384 dump
cases: n = 1 to 32, EXP and PG, both shifts, multipliers 1 and 2, graded, split, zero, NaN and
1e±18-scaled items).

### What ships: `kSteqrCtaFlatPays`

The flat solver is chosen per (type, `P`) at compile time:

```cpp
template <typename T, size_t P>
inline constexpr bool kSteqrCtaFlatPays = (P == 16) || (P == 8 && std::is_same_v<T, double>);
```

| case | nested (us) | flat (us) | nested/flat |
|---|---|---|---|
| float n=4 V (P=4) | 32.8 | 38.9 | 0.84 |
| float n=8 V (P=8) | 176.1 | 170.0 | 1.04 |
| float n=16 V (P=16) | 489.5 | 450.9 | 1.09 |
| double n=3 V (P=4) | 379.9 | 498.7 | 0.76 |
| double n=8 V (P=8) | 5894.1 | 5262.3 | 1.12 |
| double n=12 V (P=16) | 11502.7 | 9736.2 | 1.18 |

- **`P = 4` is a measured negative.** Sweeps are one to three rotations long, so the warp pays the
  longest sweep of each pass, and the 2x2 solve runs once per chunk instead of once per warp.
- **Float at `P = 8` breaks even** (0.88 to 1.04 for n = 5 to 8).
- **`P = 32` keeps the nested loops.** One chunk per warp has nothing to realign.
- Divergence cost (warp instructions over the lockstep ideal, n = 4 / 8 / 16): nested 1.31 / 1.29 /
  1.17, flat 1.46 / 1.23 / 1.08 (Nsight Compute, float V, batch 8,192).
- Public API, batch 16,384 (32,768 for n = 8), bitwise equal: `steqr` float n=12 1.02-1.04x,
  n=16 1.06-1.07x; `steqr` double n=8 1.07-1.11x, n=16 1.09-1.13x; fused `syev` float n=12/16
  1.03-1.06x. Float n=4, n=8 and n=32 are within noise.

### Rejected: flat-solver variants

| Alternative | Result | Verdict |
|---|---|---|
| Gated flat loop on the native partition (every phase under a vote) | +5-10% instructions in R mode; float n=4 0.82x, n=8 0.98x, n=16 1.02x | Rejected |
| One settle pass per chase (no settle loop) | float n=8 V 0.90x, n=16 V 0.91x against the gated flat loop | Rejected |
| Defer each chunk's final close to the loop exit | No change at F; +1.5-3% instructions in R | Rejected |

### The sub-group-wide form (`steqr_cta_solve_lockstep`)

The emulated partition shuffles over the whole sub-group, so the nested and flat solvers are legal
on it only while chunks never diverge around a collective. `steqr_cta_solve_lockstep` is the legal
form: every phase runs under a sub-group vote (`lockstep_any`) with gated updates, and the sweep is
the padded chase (`implicit_ql_step<..., Pad = true>`). On sm_89 it matches the native result
bitwise on all 192 EXP cases, runs 1.2-1.25x slower in float and equal in double. Non-NVPTX targets
use this solver today.

## Full-warp partition

On NVPTX, the native `chunked_partition` takes a runtime member mask, so ptxas wraps every
collective in a MATCH.ANY + VOTEU convergence check with a WARPSYNC slow path. The emulated
partition (`make_partition<P, false>`) shuffles over the whole warp with a constant mask and has no
wrapper, but every lane must reach every collective.

`steqr_cta` and the fused `syev` kernel therefore hand the solve a full-warp partition, clamping
dead tail chunks onto a zero problem instead of returning. What the solve does then depends on `P`:

- **`P = 32`** runs the nested loops unchanged. MATCH.ANY and VOTEU drop from 10-11 to 0 per kernel.
- **`P < 32` on NVPTX** rebuilds the native partition and runs the flat or nested form unchanged.
- **Other targets** have no native partition and run the lockstep solver for EXP.

Kernel alone, float and double, eigenvectors `V` or `N`, "step 3" is the native partition everywhere
and "shipped" is `P = 32` on the full-warp partition:

| case | step 3 (us) | shipped (us) | step 3 / shipped |
|---|---|---|---|
| float n=16 V / N | 455.7 / 416.8 | 464.9 / 423.9 | 0.98 / 0.98 |
| float n=24 N | 752.6 | 690.3 | **1.09** |
| float n=32 V / N | 1485.8 / 1260.9 | 1390.6 / 1156.1 | **1.07 / 1.09** |
| double n=16 V | 18156.6 | 16725.1 | 1.09 (code-generation swing, see below) |
| double n=32 V / N | 60970.1 / 56590.4 | 60899.5 / 56609.8 | 1.00 / 1.00 |

At `P < 32` "shipped" runs the step-3 code with the same instruction counts, so its ratios there are
noise. The double n=16 1.09 is a code-generation swing between two builds of the same flat solver,
not a gain. Double at `P = 32` is FP64-pipe bound, so removing integer and vote instructions does
nothing for it. Public `steqr` float n=24 and n=32 gain 1.05-1.10x; fused `syev` float n=32 gains
1.07-1.09x; everything else is within about 1%. All three forms matched step 3 bitwise on the dump
harness.

### Rejected: full-warp variants

| Alternative | Result | Verdict |
|---|---|---|
| Warp chase at `P < 32` (native settle, warp vote, padded chase) | 0.72-0.93x float, 0.94-1.0x double; padding adds about ten selects per rotation | Rejected |
| Lockstep solver on NVPTX (every phase voted, padded chase) | float n=4 0.69x, n=8 0.81x, n=12 0.78x, n=16 0.78x | Rejected |
| NVPTX convergence fast path (`BATCHLAS_SGP_NVPTX_CONVERGENCE_FAST_PATH`, `src/extensions/sg_partition/backend_nvptx.hh`) | Slower than the shared check in collective-dense CTA solvers; off by default | Rejected |
| `redux.sync` with a per-chunk mask | ptxas serialises the warp; slower than the butterfly | Rejected; the backend uses `redux.sync` only with the full mask |

> **Note:** For a non-immediate member mask, ptxas guards each basic block's collectives with
> MATCH.ANY + REDUX.OR + VOTEU.ANY + BRA.DIV, whatever the mask's source. The fast path tests
> `activemask` first, which wins only where collectives are sparse.

Partition-primitive changes (`src/extensions/sg_partition/`) are judged by real-kernel A/B (`steqr`
and `syev_cta` at n = 5..16, batch 16,384), never by microbenchmark. Full-warp and maskless rewrites
that won microbenchmarks lost in these kernels.

## Interleaved Q tile (measured negative)

`steqr_cta` gives each chunk a private `P x P` Q tile at `part_id * P * P`. The chunks of a warp
share banks (up to 4-way at `P = 4` and `P = 8`, 2-way at `P = 16`). The interleaved layout
(`LDQ = 32`, chunk `k` at `sg_id * 32 * P + k * P`) removes the conflicts and matches the reference
bitwise on 1,160 cases, but buys no time:

| case (Nsight Compute, V, batch 8,192) | shared ld / st conflicts, `LDQ = P` | shared ld / st conflicts, `LDQ = 32` | kernel time `LDQ = P` / `LDQ = 32` (us) |
|---|---|---|---|
| float n=4 (P=4) | 84,920 / 84,920 | 0 / 0 | 11.0 / 10.9 |
| float n=16 (P=16) | 1,224,216 / 1,172,533 | 6,906 / 1,601 | 156.8 / 160.7 |

Library A/B (`steqr`, `stedc`, `syev`, 5 rounds plus a 7-round recheck): float n=4 and n=8 0.988-1.005x;
float n=12 and n=16 0.978-1.002x; double n=8 to 32 0.996-1.001x. The kernel is latency-bound, and the
plan's gate (at least 2% at `P = 4` or `P = 8`) was not met.

## Work-group multiplier

`cta_wg_size_multiplier` sets how many warps share a work-group. At 1, every work-group is one warp,
and sm_89 runs at most 24 work-groups per SM, so one-warp groups stop at 24 of the 48 warp slots.
Float `steqr_cta` needs 55-68 registers; two warps per group reaches the register limit. The
multiplier changes only the launch shape, never the arithmetic (multiplier 1 matched 0, 2 and 4 byte
for byte).

A caller's 0 means "tuned": the default in `SteqrParams` and `syev_cta_fused`, and what `syev` passes
to both CTA arms. An explicit value is taken as given.

| kernel | multiplier 2 | everything else |
|---|---|---|
| `steqr_cta` (`kSteqrCtaAutoWgMultiplier`) | float, `P = 8` and `P = 16` | 1 |
| fused `syev` (`kSyevCtaFusedAutoWgMultiplier`) | real float, `P = 8` and `P = 16` | 1 |

Wall clock, ratio = multiplier 1 / multiplier 2, 5-8 interleaved rounds (spreads at most 2.5%):

| case | saturated batch | batch 4,096-8,192 |
|---|---|---|
| `steqr` float n=4 V (P=4) | 0.993-0.995 | 1.009 (n=4) |
| `steqr` float n=8 V (P=8) | 1.023 | 0.998-0.999 |
| `steqr` float n=12 V / N (P=16) | 1.042 / 1.063 | 0.994 (4,096), 1.100 (8,192) |
| `steqr` float n=16 V / N (P=16) | 1.033 / 1.061 | 1.009 (4,096), 1.085 / 1.124 (8,192) |
| `steqr` float n=24 V / N (P=32) | 1.012 / 1.029 | - |
| `steqr` float n=32 V / N (P=32) | 1.010-1.013 / 1.024-1.025 | 1.007-1.016 / 1.046 |
| `steqr` double, all sizes | 0.995-0.999 | - |
| fused real float n=8 / 16 / 32 V | 1.050 / 1.063 / 1.045 | 1.005-1.008 (n=8), 0.996-1.125 (n=16), 0.990-1.009 (n=32) |
| fused complex float n=16 / 32 V | 1.019 / 0.891 | - |

Multiplier 4 was never better than 2 by more than noise, and it lost up to 28% where local memory
binds (complex float n=32).

**Decision rule:** take 2 where it gains at least 3% at saturation and loses no more than 1% at
batch 4,096-8,192.

- Float `P = 16`, and real-float fused at `P = 8` and `P = 16`: pass on every row, random and graded.
- Real-float fused at `P = 32` gains on random input (1.045) but loses on graded input (`syev` float
  n=32 graded 0.978). It stays at 1.
- Float `P = 8` gains 5.5% on eigenvalues and 2.3% with vectors. Kept at 2.
- Float `P = 4` and `P = 32` gain under 3%, and double loses 0.2-1.8%. They stay at 1.
- Complex float stays at 1. Public `syev` runs at batch 16,384 did not reproduce the 1.061
  kernel-sweep reading at `P = 8` with vectors (0.988-0.994). At `P = 32` it is local-memory
  bound and multiplier 2 costs 11%.

## Small-n syev routing (measured, not shipped)

With its tuned multiplier, the fused kernel at `P = 8` beats Jacobi on real float from n = 7 up.
Moving the float `jacobi` | `cta_fused` boundary from 8 | 9 to 6 | 7 would pay, but it did not ship.
Jacobi's relative off-diagonal threshold keeps small eigenvalues of graded SPD input to relative
accuracy; the fused kernel is only normwise accurate. The A/B improved normwise error (2.4-3.3e-6 to
1.3-1.7e-6) but not relative error on graded SPD input. The boundary stays at 8 | 9 until that trade
is measured and accepted.

Public `syev`, batch 65,536, Jacobi / fused (above 1 means fused is faster):

| case | random V | graded V | random N | graded N |
|---|---|---|---|---|
| real float n=5 | 1.077 | - | 1.164 | 1.289 |
| real float n=6 | 0.962-0.974 | 1.154 | 0.991 | 1.241 |
| real float n=7 | 1.148-1.169 | 1.477 | 1.203 | 1.578 |
| real float n=8 | 1.132-1.153 | 1.546 | 1.111 | 1.526 |

- **n = 5** favours the fused kernel but stays on Jacobi: n = 6 with random input and vectors loses
  3-4%, and a boundary that flips twice would route on noise.
- Below saturation (n = 8, batch 8,192) the two are within noise. Complex float n = 2 is faster on
  Jacobi (0.62), launch-bound, and left alone.
- Complex float keeps the fused kernel for n <= 8.

The current boundary lives in the n <= 32 rows of `tuned/syev.float.*.txt`. Select the kernel with
`BATCHLAS_SYEV_ROUTE=jacobi|cta_fused`; `BATCHLAS_SYEV_SMALL_KERNEL` is retired.

## CTA STEQR: chase micro-structure decisions

The device building blocks live in `src/extensions/steqr_cta_device.hh`, shared by `steqr_cta.cc` and
`syev_cta_fused.cc`, so a fused-versus-partitioned comparison measures fusion rather than two solvers.
None of these choices has a recorded before/after figure; read them as reasons.

- **Givens rotation.** The in-range path forms \f$1/\sqrt{f^2+g^2}\f$ with a hardware reciprocal
  square root plus one Newton step. The range guard keeps \f$|f|, |g|\f$ inside
  \f$(\sqrt{\mathrm{safmin}}, \sqrt{\mathrm{safmax}/2})\f$; anything else, NaN included, takes the
  scaled reference. \f$g = 0\f$ returns \f$(1, 0, f)\f$ through a final select, not an early return,
  so identity rotations do not diverge.
- **Q cache leading dimension.** `LDQ == P` suits the standalone solver. The fused SYEV back-transform
  reads the tile by column and wants `P + 1`.
- **Padding rows are zeroed, not skipped,** so the chase runs unguarded on all P lanes.
- **Streaming rotation apply.** Successive rotations share a column, which stays in a register,
  halving shared-memory traffic of the eigenvector update.
- **Butterfly all-reduce.** Boundary searches use XOR-shuffle butterflies (log2 P shuffles, no local
  memory, no barriers).
- **Registers carried along the chase.** `d(lo)`, `e(lo)` and `e(lo-1)` are carried and written once
  after the chase, as selects. An `if` would diverge on every rotation.
- **Snapshots before the chase.** Broadcasts shuffle from immutable snapshots. Reading `diag` and
  `offdiag` directly chains each SHFL onto the previous write and onto the `lartg` path.
- **Unconditional shuffles.** Shuffles run with a clamped source and the value is selected after. A
  shuffle under a non-warp-uniform condition costs a MATCH/VOTE/BRA.DIV wrapper per call.
- **Seeded running pair.** The first rotation uses \f$(d_m - \mu, e_{m-1})\f$, which removes a
  loop-carried bool, two selects and a branch per iteration.
- **QL versus QR as LAPACK `dsteqr` chooses** (QL if \f$|D_l| \le |D_{lend}|\f$), so a graded block
  converges its small end first. The inverted rule took about 2x the steps and lost relative accuracy
  on graded input. QR runs as QL on the mirrored block
  ([Algorithm: choosing QR versus QL](../algorithms/steqr.md#steqr-choosing-qr-versus-ql)).

## Cumulative result

Measured against baseline `main` at `3df4e99`, the lockstep series comprises direction unification by
per-block mirroring, the warp-legality prep, the flat solver, the full-warp partition at `P = 32`, and
the tuned multiplier. The interleaved Q tile and small-n routing did not ship. Every eigenvalue is
bitwise identical to `main`; eigenvectors match up to sign and the order inside mirrored blocks.

Public API, library defaults, 3 interleaved rounds, medians. Ratio = `main` / final (above 1 is faster).
Spreads are at most 3% except where a parenthesised spread is given (the n = 4 graded V cell, 14-18%);
the n = 4 rows sit near launch overhead. The last column is this section's multiplier alone:

| case | batch | random N | graded N | random V | graded V | multiplier alone (V, random / graded) |
|---|---|---|---|---|---|---|
| `steqr` float n=4 (P=4) | 32,768 | 1.17 | 1.12 | 1.14 | 0.99 (14-18%) | 1 (not changed) |
| `steqr` float n=8 (P=8) | 32,768 | 1.50 | 1.36 | 1.35 | 1.27 | 1.07 / 1.00 |
| `steqr` float n=12 (P=16) | 16,384 | 1.50 | 1.56 | 1.46 | 1.49 | 1.04 / 1.04 |
| `steqr` float n=16 (P=16) | 16,384 | 1.59 | 1.67 | 1.50 | 1.51 | 1.04 / 1.01 |
| `steqr` float n=32 (P=32) | 16,384 | 1.17 | 1.15 | 1.18 | 1.15 | 1 (not changed) |
| `steqr` double n=8 / 16 / 32 | 32,768 / 16,384 | - | - | 1.85 / 1.58 / 1.00 | 1.90 / 1.92 / 1.00 | 1 (not changed) |
| `syev` float n=12 / 16, fused | 16,384 | - | - | 1.54 / 1.59 | 1.55 / 1.59 | 1.08 / 1.07 |
| `syev_cta` float n=16 (forced) | 16,384 | - | - | 1.36 | 1.34 | 1.03 / 1.02 |
| `stedc` float n=64 (P=32 leaves) | 8,192 | - | - | 1.14 | 1.09 | 1.00 |

Most of the gain is direction unification: before it, QL and QR chunks ran two disjoint loop nests
back to back. At `n = 16` (`P = 16`) warp instructions fell from 42,647 to 27,406 and kernel time
from 288.3 to 161.2 us (Nsight Compute, batch 8,192, below saturation). At `n = 32` the full-warp
partition cut instructions by 6.1% and kernel time by 1.095x. What remains at `P <= 8` is ordinary
trip-count divergence, 10-12% over the ideal; the flat solver closes it at `P = 16`, and at `P = 4`
it cannot pay for itself.

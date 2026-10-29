# STEQR (CTA solver) {#perf_steqr}

> **Status:** current · RTX 4090 (sm_89)

`steqr_cta` solves one tridiagonal problem per chunk of `P` lanes, `P` in {4, 8, 16, 32}, so a
warp holds `32/P` problems. The device solve `steqr_cta_solve` (`src/ops/syev/steqr/steqr_cta_device.hh`)
also runs inside the fused small-n SYEV kernel and at the stedc leaves. The algorithm is
@ref algo_steqr; this page holds the lane-scheduling decisions and their numbers.

**Setup:** RTX 4090 (sm_89), float and double, eigenvectors on (`V`) or off (`N`), random normal `d`
and `e`, kernel alone, medians of 3 interleaved rounds, batch 65,536 for n <= 8 and 32,768 above.
Against `main` at `3df4e99` the whole lockstep series (mirrored QR-as-QL, flat solver, full-warp
partition, multiplier) gained 1.15-1.67x on float `steqr`, 1.58-1.92x on double n = 8 and 16, and
1.54-1.59x on fused `syev` float n = 12/16, with every eigenvalue bitwise identical.

## Lockstep flat solver

The nested solver (`steqr_cta_solve_nested`) realigns the chunks of a warp only at loop exits, so
the warp pays the slowest chunk's sweep count per eigenvalue. `steqr_cta_solve_flat` moves the sweep
out of the nest: every chunk settles, then all chunks with a sweep `[l, m]` chase in one pass. The
per-chunk operations equal the nested solver's, so results are bitwise equal (384 dump cases).

### What ships: `kSteqrCtaFlatPays`

```cpp
template <typename T, size_t P>
inline constexpr bool kSteqrCtaFlatPays = (P == 16) || (P == 8 && std::is_same_v<T, double>);
```

| case | nested (us) | flat (us) | nested/flat |
|---|---|---|---|
| float n=4 V (P=4) | 32.8 | 38.9 | 0.84 |
| float n=8 V (P=8) | 176.1 | 170.0 | 1.04 (0.88-1.04 for n = 5-8: break-even) |
| float n=16 V (P=16) | 489.5 | 450.9 | 1.09 |
| double n=3 V (P=4) | 379.9 | 498.7 | 0.76 |
| double n=8 V (P=8) | 5894.1 | 5262.3 | 1.12 |
| double n=12 V (P=16) | 11502.7 | 9736.2 | 1.18 |

`P = 4` loses: sweeps are one to three rotations, so the warp pays the longest of each pass. `P = 32`
keeps the nested loops: one chunk per warp has nothing to realign.

### Rejected: flat-solver variants

| Alternative | Result | Verdict |
|---|---|---|
| Gated flat loop on the native partition (every phase under a vote) | float n=4 0.82x, n=8 0.98x, n=16 1.02x | Rejected |
| One settle pass per chase | float n=8 V 0.90x, n=16 V 0.91x | Rejected |
| Defer each chunk's final close to the loop exit | no change; +1.5-3% instructions | Rejected |

### The sub-group-wide form (`steqr_cta_solve_lockstep`)

The emulated partition shuffles over the whole sub-group, so the nested and flat solvers are legal
on it only while chunks never diverge around a collective. `steqr_cta_solve_lockstep` runs every
phase under a sub-group vote with gated updates and the padded chase
(`implicit_ql_step<..., Pad = true>`). On sm_89 it is bitwise equal to the native form, 1.2-1.25x
slower in float and equal in double. Non-NVPTX targets use it.

## Full-warp partition

On NVPTX, the native `chunked_partition` takes a runtime member mask, so ptxas wraps every
collective in a MATCH.ANY + VOTEU convergence check with a WARPSYNC slow path. The emulated
partition (`make_partition<P, false>`) shuffles over the whole warp with a constant mask and has no
wrapper, but every lane must reach every collective.

`steqr_cta` and the fused `syev` kernel therefore hand the solve a full-warp partition, clamping
dead tail chunks onto a zero problem instead of returning:

- **`P = 32`** runs the nested loops unchanged; MATCH.ANY and VOTEU drop from 10-11 to 0 per kernel.
  Float n = 24 and n = 32 gain 1.07-1.09x (kernel), public `steqr` 1.05-1.10x; double is FP64-pipe
  bound and unchanged.
- **`P < 32` on NVPTX** rebuilds the native partition and runs the flat or nested form.
- **Other targets** have no native partition and run the lockstep solver.

### Rejected: full-warp variants

| Alternative | Result | Verdict |
|---|---|---|
| Warp chase at `P < 32` (native settle, warp vote, padded chase) | 0.72-0.93x float, 0.94-1.0x double | Rejected |
| Lockstep solver on NVPTX | float n=4 0.69x, n=8-16 0.78-0.81x | Rejected |
| NVPTX convergence fast path (`__nvvm_activemask` check) | slower in collective-dense CTA solvers | Rejected; code removed |
| `redux.sync` with a per-chunk mask | ptxas serialises the warp | Rejected; `redux.sync` only with the full mask |
| Interleaved Q tile (`LDQ = 32`, no bank conflicts) | 0.978-1.005x: the kernel is latency-bound | Rejected |

Partition-primitive changes (`src/kernels/common/sg_partition/`) are judged by real-kernel A/B (`steqr`
and `syev_cta` at n = 5..16, batch 16,384), never by microbenchmark: full-warp and maskless rewrites
that won microbenchmarks lost in these kernels.

## Work-group multiplier

`cta_wg_size_multiplier` sets how many warps share a work-group. sm_89 runs at most 24 work-groups
per SM, so one-warp groups stop at 24 of the 48 warp slots; float `steqr_cta` needs 55-68 registers,
so two warps per group reaches the register limit. The multiplier changes only the launch shape,
never the arithmetic. A caller's 0 means "tuned" (the default in `SteqrParams` and `syev_cta_fused`,
and what `syev` passes); an explicit value is taken as given.

| kernel | multiplier 2 | everything else |
|---|---|---|
| `steqr_cta` (`kSteqrCtaAutoWgMultiplier`) | float, `P = 8` and `P = 16` | 1 |
| fused `syev` (`kSyevCtaFusedAutoWgMultiplier`) | real float, `P = 8` and `P = 16` | 1 |

**Decision rule:** take 2 where it gains at least 3% at saturation and loses no more than 1% at
batch 4,096-8,192. Wall clock, multiplier 1 / multiplier 2:

| case | saturated batch | batch 4,096-8,192 | verdict |
|---|---|---|---|
| `steqr` float n=12/16 (P=16) | 1.033-1.063 | 0.994-1.124 | 2 |
| fused real float n=8 / 16 V | 1.050 / 1.063 | 0.996-1.125 | 2 |
| `steqr` float n=8 V (P=8) | 1.023 (N: 1.055) | 0.998-0.999 | 2 |
| fused real float n=32 V | 1.045 random, 0.978 graded | 0.990-1.009 | 1 |
| `steqr` float P=4, P=32 | 0.993-1.029 | - | 1 |
| `steqr` double, all sizes | 0.995-0.999 | - | 1 |
| fused complex float n=16 / 32 V | 1.019 / 0.891 (local-memory bound) | - | 1 |

Multiplier 4 was never better than 2 by more than noise and lost up to 28% where local memory binds.

## Small-n syev routing (measured, not shipped)

The fused kernel at `P = 8` beats Jacobi on real float from n = 7 up (public `syev`, batch 65,536,
Jacobi / fused: n = 7 1.15-1.58x, n = 8 1.11-1.55x; n = 6 random V 0.96-0.97x). The float
`jacobi` | `cta_fused` boundary nevertheless stays at 8 | 9: Jacobi's relative off-diagonal
threshold keeps small eigenvalues of graded SPD input to relative accuracy, and the fused kernel is
only normwise accurate. Moving it waits on accepting that trade. Complex float keeps the fused kernel
for n <= 8. The boundary lives in the float n <= 32 rows of `tuned/syev.*.txt`; pin with
`BATCHLAS_SYEV_ROUTE=jacobi|cta_fused`.

## CTA STEQR: chase micro-structure decisions

The device building blocks live in `src/ops/syev/steqr/steqr_cta_device.hh`, shared by `steqr_cta.cc` and
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

# STEQR (CTA solver) {#perf_steqr}

> **Status:** current · RTX 4090 (sm_89)

`steqr_cta` solves one tridiagonal problem per chunk of `P` lanes, \f$P \in \{4, 8, 16, 32\}\f$, so a
warp holds \f$32/P\f$ problems. The device solve `detail::steqr_cta_solve` (`include/batchlas/device/lapack/steqr.hh`)
also runs inside the fused small-n SYEV kernel and at the stedc leaves. The algorithm is
@ref algo_steqr; this page holds the lane-scheduling decisions and their numbers.

**Setup:** RTX 4090 (sm_89), float and double, eigenvectors on (`V`) or off (`N`), random normal `d`
and `e`, kernel alone, medians of 3 interleaved rounds, batch 65,536 for \f$n \le 8\f$ and 32,768 above.
Against `main` at `3df4e99` the whole lockstep series (mirrored QR-as-QL, flat solver, full-warp
partition, multiplier) gained 1.15-1.67x on float `steqr`, 1.58-1.92x on double n = 8 and 16, and
1.54-1.59x on fused `syev` float n = 12/16, with every eigenvalue bitwise identical.

## Lockstep flat solver

The nested solver (`steqr_cta_solve_nested`) realigns the chunks of a warp only at loop exits, so
the warp pays the slowest chunk's sweep count per eigenvalue. `steqr_cta_solve_flat` moves the sweep
out of the nest: every chunk settles, then all chunks with a sweep \f$[l, m]\f$ chase in one pass. The
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
- **`P < 32`** rebuilds the chunk-masked partition (`make_partition<P, true>`) on every backend that
  masks by default (NVPTX, AMDGCN, SPIR-V, acpp SSCP) and runs the flat or nested form.
- **The generic backend** (DPC++ Native CPU; acpp with `BATCHLAS_SGP_FORCE_GENERIC`) has no device
  partition and runs the lockstep solver.

### Rejected: full-warp variants

| Alternative | Result | Verdict |
|---|---|---|
| Warp chase at `P < 32` (native settle, warp vote, padded chase) | 0.72-0.93x float, 0.94-1.0x double | Rejected |
| Lockstep solver on NVPTX | float n=4 0.69x, n=8-16 0.78-0.81x | Rejected |
| NVPTX convergence fast path (`__nvvm_activemask` check) | slower in collective-dense CTA solvers | Rejected; code removed |
| `redux.sync` with a per-chunk mask | ptxas serialises the warp | Rejected; `redux.sync` only with the full mask |
| Interleaved Q tile (`LDQ = 32`, no bank conflicts) | 0.978-1.005x: the kernel is latency-bound | Rejected |

Partition-primitive changes (`include/batchlas/device/detail/sg_partition/`) are judged by real-kernel A/B (`steqr`
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
for \f$n \le 8\f$. The boundary lives in the float \f$n \le 32\f$ rows of `tuned/syev.*.txt`; pin with
`BATCHLAS_SYEV_ROUTE=jacobi|cta_fused`.

## CTA STEQR: chase micro-structure decisions

The device building blocks live in `include/batchlas/device/lapack/steqr.hh` (`batchlas::device::detail`), shared by `steqr_cta.cc` and
`syev_cta_fused.cc`, so a fused-versus-partitioned comparison measures fusion rather than two solvers.
None of these choices has a recorded before/after figure; read them as reasons.

- **Givens rotation.** The in-range path forms \f$1/\sqrt{f^2+g^2}\f$ with a hardware reciprocal
  square root plus one Newton step. The range guard keeps \f$|f|, |g|\f$ inside
  \f$(\sqrt{\mathrm{safmin}}, \sqrt{\mathrm{safmax}/2})\f$; anything else, NaN included, takes the
  scaled reference. \f$g = 0\f$ returns \f$(1, 0, f)\f$ through a final select, not an early return,
  so identity rotations do not diverge.
- **Q cache leading dimension.** `LDQ == P` suits the standalone solver. The fused SYEV back-transform
  reads the tile by column and wants `P + 1`.
- **Padding rows are zeroed, not skipped,** so the chase runs unguarded on all \f$P\f$ lanes.
- **Streaming rotation apply.** Successive rotations share a column, which stays in a register,
  halving shared-memory traffic of the eigenvector update.
- **Butterfly all-reduce.** Boundary searches use XOR-shuffle butterflies (\f$\log_2 P\f$ shuffles, no local
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

## STEQR: one QL step

Until 2026-11 the CTA solver carried two forms of the QL step, selected by
`SteqrParams::cta_update_scheme`: `EXP` (the default), the explicit bulge chase, and `PG`, the
`dsteqr` scalar recurrence ([Algorithm: the CTA QL step](../algorithms/steqr.md#steqr-the-cta-ql-step)).
The same A/B also tried one state machine for the three solver bodies (nested, flat, lockstep).

**Setup:** RTX 4090 GPU 0, both GPUs otherwise idle, public `steqr_cta` (real `d`, `e`, `V` and `N`)
and `syev_cta_fused` (random Hermitian `A`, real and complex, `V` and `N`), n = 4, 6, 8, 12, 16, 24,
32, batch 2,048-65,536 (float `steqr` 65,536 for \f$n \le 8\f$, 32,768 above; fused half that;
double and complex smaller). Median of 8 timed calls after 3 warm-ups per run, inputs restored
outside the timing, 14 alternating reps after a discarded first.

### `EXP` stays: `PG` does not converge on a stedc leaf

`PG` against `EXP` (time ratio, under 1 means `PG` is faster) won 52 of the 56 real cells:

| op | float `N` | float `V` | double `N` | double `V` |
|---|---|---|---|---|
| `steqr` | 0.974-0.990 | 0.982-1.019 (n = 8-16: 1.013-1.019) | 0.943-0.973 | 0.945-0.996 |
| fused `syev` | 0.963-0.996 | 0.973-1.025 (n = 8: 1.025, noise 0.89-1.09) | 0.929-0.975 | 0.937-0.975 |

It is about a quarter fewer instructions (four carried scalars instead of the band update). It is
also not robust: `StedcTest.NoEigenVectorsMatchesReferenceOnPoisonedWorkspace` (double, n = 600,
`StedcAlgorithm::Recursive`, recursion threshold 32, item `d(i) = |10 - i mod 21|`, `e(i) = 1` with
every 21st `1e-5`) reports `info = 1` under `PG` at every leaf sweep cap tried (30 to 400) and 0
under `EXP` at 30. A faster step that stagnates is not a candidate, so `EXP` became the only step;
`PG` returns only with a fix for that stagnation and a re-run of this A/B.

Deleting `PG` leaves the `EXP` code as it was, inside its lambda: unwrapping the lambda cost float
\f$P = 32\f$ and double \f$P = 16\f$ `V` 1.02-1.035. With it, against the two-step kernels: float
0.941-1.015, complex float 0.983-1.012, complex double 0.977-1.011, double 0.995-1.026 (`steqr` `N`
n = 24, 32: 1.023-1.026; the float analog 0.99-1.00).

### Rejected: one schedule state machine

The three solver bodies (nested, flat, lockstep) were merged into one state machine,
`steqr_cta_sweeps<T, P, Gated, Realign>` (the flat solver's settle / chase loop; without `Realign` a
tight sweep loop at \f$l\f$ for the nested schedule; `Gated` the lockstep votes and padded step), with
outputs bitwise equal to the three bodies on all 84 cells and the gated schedule bitwise equal at
\f$P = 4, 8, 16\f$. Time ratio, merged / three bodies:

| step | worst cells | verdict |
|---|---|---|
| `PG` | 0.938-1.016 (fused float `V` n = 12, 16: 1.016) | passes the 1.02 bar |
| `EXP` | float \f$P = 32\f$ 1.024-1.062 (fused `N` and `V`, `steqr` `N`), `steqr` float `V` n = 12, 16 1.059-1.061; double \f$\le\f$ 1.020 | Rejected |

A first form that sent every unrealigned sweep through the open and close checks lost up to 1.09
at \f$P = 4\f$ and float \f$P = 8\f$ (with `PG`); the tight loop recovered that. With the shipped `EXP`
step the merge stays out: the register-heavier chase does not fit the state machine at float
\f$P = 16, 32\f$. It is worth retrying together with a stagnation-free `PG` (merged `PG` against the
shipped `EXP` bodies: 0.906-1.014).

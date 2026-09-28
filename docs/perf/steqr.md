# STEQR: the CTA tridiagonal solver (`steqr_cta`)

`steqr_cta` solves one tridiagonal problem per chunk of `P` lanes, `P` in {4, 8, 16, 32}, so a
warp holds `32/P` problems. The same device solve (`steqr_cta_solve`, in
`src/extensions/steqr_cta_device.hh`) runs inside the fused small-n SYEV kernel and at the
stedc leaves. This page records how the chunks of a warp are kept in step, and what was
measured worse.

Measurement context, unless stated otherwise: RTX 4090 (sm_89), GPU 1 pinned by UUID, float
and double, eigenvectors on (`V`) or off (`N`), Lapack shift, EXP update scheme, 30 sweeps per
eigenvalue, random normal `d` and `e`. Times are the kernel alone, JIT runs discarded, medians
of 3 interleaved rounds, batch 65,536 for n <= 8 and 32,768 above.

## Lockstep flat solver

The nested solver (`steqr_cta_solve_nested`: split loop, then a loop over `l`, then a sweep
loop) lets the chunks of a warp realign only at loop exits. A chunk that deflates an eigenvalue
waits at the sweep-loop exit while its neighbours finish sweeping theirs. The warp therefore
pays, for each eigenvalue, the slowest chunk's sweep count, summed over eigenvalues.

`steqr_cta_solve_flat` hoists the sweep out of the nest. Each chunk *settles*: it opens a
block, tests, advances past deflated eigenvalues, 2x2-solves, and closes, until it either has
a sweep `[l, m]` to run or is done. Then every chunk with a sweep chases in the same pass. The
per-chunk operation sequence is the nested solver's, so the results are bitwise equal. A dump
harness compared d, e, Q and info on 384 cases and found them identical. The cases covered:

- n from 1 to 32 and batch 4,099;
- EXP and PG, Lapack and Wilkinson shifts, work-group multiplier 1 and 2;
- graded, split, zero, NaN and 1e+-18-scaled items.

### What ships: `kSteqrCtaFlatPays`

The flat solver is chosen per (type, `P`) at compile time:

```cpp
template <typename T, size_t P>
inline constexpr bool kSteqrCtaFlatPays = (P == 16) || (P == 8 && std::is_same_v<T, double>);
```

| case | nested (us) | flat (us) | nested/flat |
|---|---|---|---|
| float n=4 V / N | 32.8 / 30.0 | 38.9 / 32.8 | 0.84 / 0.92 |
| float n=5 V | 85.0 | 90.1 | 0.94 |
| float n=6 V | 100.4 | 114.7 | 0.88 |
| float n=7 V | 144.4 | 141.3 | 1.02 |
| float n=8 V / N | 176.1 / 151.7 | 170.0 / 154.6 | 1.04 / 0.98 |
| float n=9 V | 191.5 | 188.4 | 1.02 |
| float n=10 V / N | 226.3 / 199.7 | 221.2 / 198.7 | 1.02 / 1.01 |
| float n=12 V / N | 302.1 / 267.3 | 285.7 / 259.1 | 1.06 / 1.03 |
| float n=16 V / N | 489.5 / 435.2 | 450.9 / 412.7 | 1.09 / 1.05 |
| double n=3 V | 379.9 | 498.7 | 0.76 |
| double n=4 V | 799.7 | 835.6 | 0.96 |
| double n=5 / 6 V | 2289.7 / 3333.1 | 2235.7 / 3127.6 | 1.02 / 1.07 |
| double n=8 V | 5894.1 | 5262.3 | 1.12 |
| double n=10 / 12 V | 8162.3 / 11502.7 | 6928.4 / 9736.2 | 1.18 / 1.18 |
| double n=16 V | 18450.4 | 16675.8 | 1.11 |

The flat solver ships at `P = 16` for both types, and at `P = 8` for double only.

- **`P = 4` is a measured negative.** Its sweeps are one to three rotations long. Chunks at
  different `l` make the warp pay the longest sweep of each pass, and a 2x2 solve that the
  nested loops ran once per warp now runs at each chunk's own pass.
- **Float at `P = 8` breaks even** over n = 5..8 (0.88 to 1.04).
- **Double gains more than float.** The kernel is FP64-pipe bound (about 4% issue-active), so
  fewer warp-level FP64 instructions is the whole cost, and the extra integer bookkeeping is
  free.
- **`P = 32` keeps the nested loops.** One chunk per warp has nothing to realign.

The public API shows the same result, measured before/after with library defaults at batch
16,384 (32,768 for n=8), random and graded inputs, eigenvalues bitwise equal:

- `steqr` float: n=12 1.02-1.04x, n=16 1.06-1.07x;
- `steqr` double: n=8 1.07-1.11x, n=16 1.09-1.13x;
- fused `syev` float: n=12 1.03-1.04x, n=16 1.05-1.06x;
- `syev_cta` float n=16: 1.04x;
- float n=4, float n=8 and n=32: within noise.

Nsight Compute, float, eigenvectors, batch 8,192, warp instructions (millions). "R" gives every
chunk of a warp the same problem, so R is the lockstep ideal and F/R is the cost of divergence:

| n | nested F | nested R | nested F/R | flat F | flat R | flat F/R |
|---|---|---|---|---|---|---|
| 4 | 3.26 | 2.50 | 1.31 | 3.67 | 2.51 | 1.46 |
| 8 | 20.57 | 15.95 | 1.29 | 19.32 | 15.77 | 1.23 |
| 16 | 123.8 | 105.4 | 1.17 | 110.8 | 103.0 | 1.08 |

### Measured worse

- **A gated flat loop on the native partition.** Every phase ran under a vote with per-chunk
  gates, in the shape the sub-group-wide solver needs. It costs 5-10% more instructions in the
  R mode than the nested loops, and that ate the lockstep gain: float n=4 0.82x, n=8 0.98x,
  n=16 1.02x. The shipped flat solver uses plain chunk-local branches and costs nothing in R
  mode.
- **One settle pass per chase** (no settle loop). A chunk that advances loses that pass's chase
  slot. float n=8 V 0.90x and n=16 V 0.91x against the gated flat loop.
- **Deferring each chunk's final close to the loop exit.** This lets all chunks mirror back
  together. It made no change at F, and cost 1.5-3% more instructions in R.

### The sub-group-wide form (`steqr_cta_solve_lockstep`)

The emulated partition shuffles over the whole sub-group. The nested loops, and the flat
solver above, are only legal on it while chunks never diverge around a collective.
`steqr_cta_solve_lockstep` is the legal form:

- every phase runs under a sub-group vote (`lockstep_any`), with per-chunk gated updates;
- the sweep is the padded chase (`implicit_ql_step<..., Pad = true>`). A short or idle chunk
  runs identity rotations, its Q index takes stride 0, and all of its state is discarded by
  select.

Forcing the emulated partition on sm_89 matched the native result bitwise on all 192 EXP
cases. It was 1.2-1.25x slower in float and equal in double, while its `P = 32` nested solve
was 1.07x faster. It is what an emulated partition runs today (the non-NVPTX targets).

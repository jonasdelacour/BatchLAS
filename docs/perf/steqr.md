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

## Full-warp partition

On NVPTX the native `chunked_partition` passes a runtime member mask, so ptxas wraps every
collective in a MATCH.ANY + VOTEU convergence check with a WARPSYNC slow path. The emulated
partition (`make_partition<P, false>`) shuffles over the whole warp with a constant mask and
has no such wrapper, but every lane of the warp must then reach every collective.

`steqr_cta` and the fused `syev` kernel now hand the solve a full-warp partition. Both clamp
their dead tail chunks onto a zero problem instead of returning, so every lane reaches the
solve. What the solve does with that promise depends on `P`:

- **`P = 32`** is one chunk per warp. The nested loops are already warp-uniform, so they run on
  the full-warp partition as they are. In the SASS, MATCH.ANY and VOTEU drop from 10-11 to 0
  per kernel. The BRA.DIV convergence branches stay: ptxas emits them for any `.sync`
  collective it cannot prove converged.
- **`P < 32` on NVPTX** rebuilds the native partition and runs the step-3 forms (flat or nested)
  unchanged, because every full-warp form measured slower (below).
- **Other targets** have no native partition. They run the lockstep solver for EXP, as before.

Kernel alone, float and double, eigenvectors `V` or not `N`, 3 interleaved rounds (spread in
parentheses). "step 3" is the native partition everywhere; "shipped" is `P = 32` on the
full-warp partition; "warp chase" also runs the flat solver's chase on the full-warp partition
at every `P < 32` (padded, as in the lockstep solver), with the settle phase left native:

| case | step 3 (us) | shipped | step 3 / shipped | warp chase | step 3 / warp chase |
|---|---|---|---|---|---|
| float n=4 V / N | 33.8 / 31.9 | 36.9 / 32.8 | 0.92 (13%) / 0.97 | 47.2 / 39.9 | 0.72 / 0.80 |
| float n=6 V | 112.6 | 112.7 | 1.00 | 131.1 | 0.86 |
| float n=8 V / N | 178.2 / 154.6 | 179.2 / 156.9 | 0.99 / 0.99 | 196.6 / 169.0 | 0.91 / 0.91 |
| float n=12 V | 293.0 | 291.8 | 1.00 | 343.0 | 0.85 |
| float n=16 V / N | 455.7 / 416.8 | 464.9 / 423.9 | 0.98 / 0.98 (4%) | 541.7 / 448.5 | 0.84 / 0.93 |
| float n=24 N | 752.6 | 690.3 | **1.09** | 692.2 | 1.09 |
| float n=32 V / N | 1485.8 / 1260.9 | 1390.6 / 1156.1 | **1.07 / 1.09** | 1391.6 / 1155.4 | 1.07 / 1.09 |
| double n=4 V | 797.7 | 795.6 | 1.00 | 847.9 | 0.94 |
| double n=8 / 16 V | 5282.6 / 18156.6 | 5274.4 / 16725.1 | 1.00 / 1.09 | 5292.0 / 16727.0 | 1.00 / 1.09 |
| double n=32 V / N | 60970.1 / 56590.4 | 60899.5 / 56609.8 | 1.00 / 1.00 | 60901.4 / 56564.5 | 1.00 / 1.00 |

At `P < 32` "shipped" runs the step-3 code with the same instruction counts, so its ratios
there are noise. The one exception is double n=16, whose 1.09 is a code-generation swing
between two builds of the same flat solver, not a gain of this change. Double at `P = 32` is
FP64-pipe bound (about 4% issue-active), so removing integer and vote instructions does
nothing for it.

Through the public API (library defaults, batch 16,384, random and graded, 3-5 interleaved
rounds, eigenvalues bitwise equal):

- `steqr` float n=24 and n=32: 1.05-1.10x;
- fused `syev` float n=32: 1.07-1.09x;
- `steqr` double n=32, and everything at `P < 32` (including fused float n=12/16 and complex
  n=8/16): within about 1%.

All three forms matched step 3 bitwise on the 384-case dump harness: n from 1 to 32, batch
4,099, EXP and PG, both shifts, work-group multiplier 1 and 2, graded, split, zero, NaN and
scaled items.

### Measured worse

- **The warp chase at `P < 32`** (settle on the native partition, then a warp vote and the
  padded chase on the full-warp partition). It is 0.72-0.93x in float and 0.94-1.0x in
  double. The padded chase adds about ten selects per rotation: the idle-chunk inputs, five
  carries and the Q index and store. The chase has only two shuffles per rotation, and their
  mask wrappers cost less than the padding that replaces them.
- **The lockstep solver on NVPTX** (every phase voted, the padded chase), in one round:
  float n=4 0.69x, n=8 0.81x, n=12 0.78x, n=16 0.78x. Step 3 measured it at 1.2-1.25x slower
  in float too.

## Interleaved Q tile (measured negative)

`steqr_cta` gives each chunk a private `P x P` Q tile in shared memory, `LDQ = P`, at
`part_id * P * P`. Lanes index it only by row, so a chunk alone never has a bank conflict. The
chunks of one warp do: their tiles start `P * P` words apart, a multiple of 32 at `P >= 8`, so
chunks touching columns that are equal mod `32/P` hit the same banks. That is up to 4-way at
`P = 4` and `P = 8`, and 2-way at `P = 16`. Interleaving the chunks (`LDQ = 32`, chunk `k` at
`sg_id * 32 * P + k * P`) makes the bank the warp lane for any column. It is a pure address
change: the dump harness matched step 4 bitwise on 1,160 cases, including work-group
multiplier 4, and the tests passed. It removes the conflicts and buys nothing, so it did not
ship.

Nsight Compute, kernel alone, eigenvectors on, batch 8,192, one launch per layout. `P = 32`
has the same layout either way:

| case | shared ld / st conflicts, `LDQ = P` | `LDQ = 32` | kernel time `LDQ = P` / `LDQ = 32` (us) |
|---|---|---|---|
| float n=4 (P=4) | 84,920 / 84,920 | 0 / 0 | 11.0 / 10.9 |
| float n=8 (P=8) | 513,210 / 513,215 | 0 / 0 | 35.0 / 34.6 |
| float n=12 (P=16) | 719,513 / 732,532 | 3,049 / 1,278 | 98.9 / 100.4 |
| float n=16 (P=16) | 1,224,216 / 1,172,533 | 6,906 / 1,601 | 156.8 / 160.7 |
| double n=8 (P=8) | 407,092 / 386,283 | 0 / 0 | 693.5 / 693.6 |
| float n=32 (P=32) | 25,580 / 16,788 | 25,656 / 16,705 | 776.3 / 779.6 |

Library A/B through the public `steqr`, `stedc` and `syev` calls, step 4 against the
interleaved build, eigenvectors on, 5 interleaved rounds and a 7-round recheck of float
`P <= 16`, all eigenvalues bitwise equal:

- float n=4 and n=8, random and graded: 0.988-1.005x. One graded n=8 round read 1.037x at a
  5.5% spread; the recheck gave 0.997x.
- float n=12 and n=16: 0.978-1.002x. Double n=8, n=16 and n=32: 0.996-1.001x.
- The `P = 32`, complex `syev` pipeline and `stedc` n=64 rows are within their noise.

The conflicts cost nothing measurable because the kernel is latency-bound. Each rotation does
one shared load and one store per lane among about 80 instructions, and a replayed wavefront
is hidden behind the rotation's dependent arithmetic chain. Halving the shared wavefronts at
n=16 moved time by less than the run-to-run spread. The plan's gate was at least 2% at
`P = 4` or `P = 8` with no regression, and it was not met.

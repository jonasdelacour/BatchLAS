# SYTRD: tridiagonal and band reduction evidence {#perf_sytrd}

> **Covers:** the reductions under `syev`: the blocked one-stage reduction (`sytrd_blocked` and
> its `latrd` panel), and the two-stage pair (`sytrd_sy2sb` dense to band, `sytrd_sb2st_hh`
> band to tridiagonal with retained reflectors, plus its \f$Q_2\f$ back-transform).
> **Status:** current. Distilled from the comments that used to sit at each call site; the
> solver-level consequences (which `syev` tier wins where, the two-stage band width) stay on
> @ref perf_syev and are linked, not repeated.
> **Machine:** RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda, unless a section says otherwise.
> **Measured:** 2026-08 (grid `latrd`, the sb2st back-transform geometry, the rank-2k trailing
> update); several source comments carried no date, and those sections say so.

The panel (`latrd`) dominates the blocked solve, and its known remaining headroom is the
single-read symv ([syev: open debt, the single-read panel symv](syev.md#syev-open-debt-the-single-read-panel-symv)).
Two-stage is decided by the chase and the \f$Q_2\f$ back-transform, both of which are
occupancy problems before they are arithmetic problems.

## sytrd: the latrd implementation selector

`BATCHLAS_LATRD_IMPL` picks the panel kernel in `src/extensions/latrd_lower_panel.cc`:

| value | kernel |
| --- | --- |
| unset, or anything else | Legacy (the default below `latrd_grid_min_n`; bit-for-bit unchanged) |
| `%device` | Device: the device-BLAS variant (`device::hemv`, `her2k`, ...). Measured slower; no figures were recorded with the claim |
| `grid` | Grid: several work-groups per matrix (below) |

It is re-read on every call, not latched, so one process can A/B the kernels by changing the
variable between runs; a `static` there would pin the first answer and make the A/B vacuous.
Its twin `BATCHLAS_SYTRD_IMPL` (`sytrd_blocked.cc`) *is* latched in a static. The asymmetry
predates the settings rework and was preserved.

## sytrd: the latrd grid-path threshold

The grid path is the fix for one-work-group-per-matrix starvation of the panel. It is not free:
the software grid barrier costs five device-scope syncs per panel column. Legacy/grid time ratio
(> 1 means grid wins), float, eigenvalues-only, blocked provider, undated (before 2026-08-04):

| n | b=1 | b=4 | b=8 | b=16 | b=64 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.71 | 0.74 | 0.75 | 0.76 | 0.77 |
| 256 | 0.79 | 0.74 | 0.74 | 0.75 | 0.75 |
| 384 | 0.95 | - | 0.95 | - | - |
| 512 | 1.03 | 1.02 | 1.03 | 1.02 | 0.95 |
| 768 | 1.43 | - | 1.41 | - | - |
| 1024 | 1.94 | 1.91 | 1.90 | 1.82 | 1.24 |
| 2048 | 4.10 | - | 3.63 | - | - |

The win grows with n because the barrier count is \f$O(n)\f$ per panel while the work it
parallelises is \f$O(n^2)\f$ per column. The loss below 512 is uniform in batch, the signature of
a fixed per-column overhead. **Gated at `latrd_grid_min_n = 768`:** every measured point at or
above it wins by at least 1.4×, and 512 is only neutral. It is a knob rather than a formula
because the crossover is a barrier-latency property of the device, not of the algorithm; re-measure
on new hardware.

The gate is applied in both modes; the eigenvector re-measure (2026-08-04, build `12963a8`) found
the same crossover and is tabulated at
[syev: latrd grid gate confirmed in eigenvector mode](syev.md#syev-latrd-grid-gate-confirmed-in-eigenvector-mode).
The win shrinks as batch grows (n = 768: 1.38× at batch 1, 1.09× at batch 64; n = 1024 / batch
128 is a dead heat, 475.5 against 475.7 ms): once the batch alone saturates the SMs there is no
starvation left for extra work-groups to absorb. `BATCHLAS_LATRD_IMPL` still forces either kernel
at any size, which is what makes the two an intra-run A/B.

## sytrd: the latrd grid kernel and its co-residency cap

The trailing row range \f$[i+1, n)\f$ of the current panel column is split into G contiguous
blocks, one per work-group. Every per-row loop of the legacy kernel (column update, reflector
scaling, symv, the rank-2k corrections, the W write-back) is load-balanced over rows and needs no
communication. The four whole-range reductions (sumsq for the reflector norm, gamma and delta per
previous panel column, the final dot) are reduced per work-group and combined through global
scratch **in a fixed group order**, so the result is run-to-run deterministic. Five grid barriers
per panel column: after the column update and sumsq partials; after scaling the reflector; after
the gamma/delta partials; after the dot partials; at the end of the column (the next column reads
row i+1 of W).

The barrier is a sense-reversing software barrier, so it only terminates if every participating
work-group is resident at once. `choose_grid_launch` guarantees that by never launching more than
`MAX_COMPUTE_UNITS` work-groups in total, one block per SM, which is schedulable for any
work-group size ≤ 256 and any local footprint the legacy kernel accepts (about
\f$2n\cdot\text{sizeof}(T)\f$). Blocks go to distinct SMs while the block count does not exceed
the SM count. Beyond that cap, the useful parallelism is one work-item per trailing row, so
\f$G \cdot wg\f$ is kept near n. G = 1 means "run the legacy kernel verbatim". The grid path is
GPU-only: the CPU and host SYCL runtimes multiplex work-groups onto a thread pool and give no
co-residency guarantee. Grid scratch (barrier words and partial slots) is process-lifetime and
deliberately not taken from the caller's `BumpAllocator`, so no `*_buffer_size` query changes
meaning.

Proposed, unbuilt: a cap of `SMs × achievable blocks per SM` from the kernel's real footprint
([syev: remaining unmeasured items](syev.md#syev-remaining-unmeasured-items)).

## sytrd: the vacuous latrd grid-groups A/B

`BATCHLAS_LATRD_GRID_GROUPS` was meant as an escape hatch to force G, but the residency cap made
the grid path unreachable at batch ≥ `MAX_COMPUTE_UNITS` (128 SMs here): at batch 128 the cap
clamps the variable to 1, and above 128 the cap is 0 and the variable is never read. The recorded
L2-residency A/B, taken at batch 512–1024, therefore compared legacy against legacy:

| shape | "legacy" (ms) | "grid" (ms) |
| --- | --- | --- |
| float n=256, b=1024 | 31.06 | 32.19 |
| float n=512, b=512 | 327.3 | 325.9 |
| cfloat n=512, b=512 | 698.8 | 698.3 |

Identical to three digits: the signature of the fallback, not of a fast grid kernel.
`BATCHLAS_LATRD_GRID_FORCE_UNSAFE` bypasses the cap, and is gated behind the
`BATCHLAS_ALLOW_UNSAFE_ENV` build option (off: the field is false whatever the environment says).
The barrier is per matrix, so only the G groups of one matrix must be co-resident, which small G
plausibly satisfies at large batch, but that rests on an in-order block dispatch the spec does not
promise. A forced launch that is not co-resident **hangs rather than fails**, and a hang looks
exactly like slow JIT: run forced-unsafe measurements under `timeout`, and do not use this to relax
the default cap without its own argument. The override was added but never exercised
([syev: harness unblockers (WP0)](syev.md#syev-harness-unblockers-wp0)).

## sytrd: complex multiply in the latrd symv

`std::complex<float> * std::complex<float>` is not four multiplies. C99 Annex G, which C++
inherits and clang implements unless `-fcx-limited-range` or `-ffast-math` is passed (this build
passes neither), requires a product that comes out (NaN, NaN) to be retried by a recovery routine
that rescues infinities. clang emits the four multiplies, then a branch on
`isnan(re) && isnan(im)`, then a call to `__mulsc3` (`__muldc3` for double); both symbols are in
the library's device code. In a matvec inner loop the per-element branch is opaque to the
unroller and holds the call's operands live, so the loop cannot keep several loads in flight,
which is the one thing this memory-latency-bound kernel depends on.

`latrd_lower_panel.cc` therefore forms `acc += a * b` and `acc += conj(a) * b` in explicit real
arithmetic in the symv. Every dense BLAS makes this trade (it is what `-fcx-limited-range`
gives, and what cuBLAS, MKL and MAGMA use); the semantic difference is confined to operands that
already hold Inf or NaN, and a Hermitian input whose reflectors hold Inf or NaN has no meaningful
eigensolve anyway. The product is formed first and then added, the same association as the
`acc += a * b` it replaces; writing `acc.real() + ar*br - ai*bi` would re-associate and move the
last ulp for no gain (the compiler contracts to fma either way). Real types take the plain
multiply-add and are unaffected: float and double panel timings at n = 512, batch 1024 were
50.86 µs/matrix before and 50.87 after.

**Applied to the symv only, deliberately.** The same escape at every other complex multiply in the
kernel (the rank-2 column update, the gamma/delta reductions, the tau/alpha scalings, the fused
trailing update) made the panel 1.16× *slower*, reproducibly:

| shape | symv only (µs/matrix) | all sites converted |
| --- | --- | --- |
| n=512, batch=1024 | 108.4 | 125.8 |
| n=512, batch=512 | 103.3 | 122.1 |

Those sites are about 6% of the panel's work, so this is not their arithmetic: inlining the
expanded form everywhere costs enough registers to lose occupancy in the one loop that matters.
`AGENTS.md` records the general rule ("write out the real arithmetic in the hot loop only").

## sytrd: rejected latrd symv loop shapes

The legacy symv splits \f$\mathrm{acc} = \sum_c A_h(r,c)\,v(c)\f$ at \f$c = r\f$ into a row walk
and a column walk; ascending c crosses the boundary once, so the accumulation order and the
results are unchanged, and removing the per-element branch lets the compiler unroll and keep
loads in flight. Measured alternatives, all slower and rejected (undated):

- unroll 8, and processing two rows per iteration: both cost more registers than the added
  parallelism returns;
- computing the \f$c > r\f$ term with one sub-group per column: fully coalesced (20 → 9.7 sectors
  per request), but the extra barrier destroys reuse between the two passes, and short columns
  near \f$r \to n\f$ leave most lanes idle, so DRAM traffic and runtime both rose.

The access-pattern change did not change the *number* of reads; the open single-read design is at
[syev: open debt, the single-read panel symv](syev.md#syev-open-debt-the-single-read-panel-symv).

## sytrd: the rank-2k trailing update in the blocked reduction

\f$A_{22} \mathrel{-}= V W^H + W V^H\f$ is one `syr2k`, which touches only the triangle the
panel loop goes on to read. Against the two full \f$n_2 \times n_2\f$ GEMMs it replaced (RTX
4090, float, undated):

- the update alone, over the shapes the panel loop produces: 3.4–3.6× faster;
- end to end, n = 512 batch 1024: 264 / 253 / 248 ms → 228 / 227 / 232 at nb = 16 / 24 / 32;
  n = 256 batch 2048: 34.3 / 34.6 / 37.0 → 27.0 / 30.6 / 34.0.

It is CUDA and float only, and not out of conservatism: `syrk`/`syr2k` reach a batched kernel only
through the custom float route. Everything else falls to `syr2k_vendor_impl`, a host loop issuing
one `cublasXsyr2k` per batch member, which in double measured **7.8× slower** than the GEMM pair
at n = 256, batch 1024: the whole win inverts. `B == Backend::CUDA` at that site meant "the
syr2k tile kernel is wired on this route"; it is now asked as a library question, so a
vendor-free build (same backend, tile TU absent) answers correctly.

complex<float> is admitted through `her2k`, a different function with a different route: one
batched `gemm_vendor` into scratch followed by `accumulate_hermitian<TwoSided=true>`, half the
arithmetic of the pair because \f$\alpha A B^H\f$ and \f$\bar\alpha B A^H\f$ are conjugate
transposes of one another. The motivation was that vendor GEMM is 34.6% of the cfloat solve at
n = 256 and 14.4% at n = 512, about half of it in this update. The source comment carried an
**OPEN** item: `her2k_gemm_preferred`'s crossover was swept on square rank-k shapes, while the
panel issues a narrow one (\f$k = i_b \in \{16,24,32\}\f$, \f$n_2\f$ up to 480) where the fold's
extra \f$n_2^2\cdot\text{batch}\f$ write and read might eat the halved arithmetic. **Resolved
2026-08-08:** her2k beats the pair 1.32–1.33× at \f$n_2 = 480\f$, batch 512, and the host-loop
fallback is 1.23× slower than the pair; numbers and the end-to-end 1.043× at
[syev: her2k trailing update for complex float (WP3)](syev.md#syev-her2k-trailing-update-for-complex-float-wp3).
complex<double> is deliberately excluded: it would reach the same fast route, but its scratch is
16 bytes per element (half the fit headroom below) and none of it has been measured. Guessing is
how the 7.8× inversion was first written down.

## sytrd: her2k scratch fit headroom

her2k's fast route needs an \f$n_2 \times n_2 \times \text{batch}\f$ scratch expansion; when it
does not fit, the backend drops to a host loop over `cublasCher2k`, structurally the route that
measured 7.8× slower than the GEMM pair. The call site therefore asks the backend's own fit
predicate first and keeps the GEMM pair when it says no. The check is **per panel, not hoisted**:
\f$n_2\f$ shrinks every iteration, so an early panel can fail to fit while later ones fit.

`expanded_ld<complex<float>>(n2)` rounds \f$n_2\f$ up to a multiple of 2, so the scratch is about
\f$8 n_2^2 \cdot \text{batch}\f$ bytes against a `GLOBAL_MEM_SIZE / 4` budget, about 6.0 GiB on a
24 GiB 4090:

| shape | scratch |
| --- | --- |
| n=448, batch=585 (the cfloat blocked/vendor crossover in `syev.hh`) | 0.75 GiB |
| n=512, batch=1024 | 1.76 GiB (≥ 3.4× headroom) |
| forced blocked n=1024, batch=1024 | 7.51 GiB (does not fit) |
| forced blocked n=2048, batch=256 | 7.75 GiB (does not fit) |

The ceiling is crossed around \f$n_2^2 \cdot \text{batch} > 8.0\times 10^8\f$ elements: outside
the routed region but reachable by pinning the provider, and exactly where an unguarded call would
invert. The lease is taken per panel inside `her2k_vendor` and released before the next, so the
peak is one panel's scratch. On an out-of-order `Queue` that route also drains the device between
its GEMM and its fold, once per panel; the benchmarks all use in-order queues and never see it.

## sytrd: no symmetrize after the rank-2k update

After the triangle-only update A's upper triangle is stale, and nothing symmetrizes it. That was
checked across every reader rather than assumed:

- `latrd_lower_panel`, all three variants: the symmetric matvec is the only place tempted to cross
  the diagonal, and all three split at \f$c = r\f$, taking \f$A_b(r,c)\f$ for \f$c \le r\f$ and
  \f$\overline{A_b(c,r)}\f$ for \f$c > r\f$. The device variant reaches it through
  `device::hemv<Uplo::Lower>`, which mirrors the same way. The fused trailing update guards with
  `if (r < c) continue` (legacy, grid) or `device::her2k<Uplo::Lower>`.
- `restore_tridiag_lower` reads the diagonal and only writes the superdiagonal.

The GEMM pair happened to leave a valid upper triangle as a side effect; nothing depended on it.
her2k additionally forces \f$\mathrm{imag}(\mathrm{diag}) = 0\f$ on the block it writes, which the
pair does not (it leaves the accumulated roundoff). That is the correct value for a Hermitian
operand and unobservable downstream (`syev_blocked` takes `D(i,b).real()`, and the
\f$n_2 \le 128\f$ device path already does the same through `device::her2k`), but `latrd`'s hemv
consumes the diagonal, so **cfloat results drift in the last bits** between the two routes: expect
drift, not bitwise equality, in an A/B.

## sytrd: the dense-to-band ormqr block-width hint

`sytrd_sy2sb` factors a panel of width kd (default 32) and applies it with `ormqr`, whose dispatch
keys the WY block width on `A.rows()`, the panel *height* (hundreds to thousands), while the
dimension that matters is \f$k = k_d\f$. Every `ORMQR_BLOCK_SIZE_*` constant was 16, so a kd = 32
panel split into two WY blocks: 2× `pack_v`, 2× `larft` and 6 GEMMs per `ormqr` instead of 1, 1
and 3, every GEMM at k = 16 instead of 32. Passing nb = kd (commit `f7f3c57`), interleaved A/B,
median of 15 rounds, idle GPU, on the sy2sb panel loop (undated):

| shape | nb = kd vs table |
| --- | --- |
| n=1024, kd=32, batch=64 | 1.19–1.20× faster |
| n=2048, kd=32, batch=32 | 1.36× faster |
| n=512, kd=32, batch=128 | 0.90× (regression) |
| n=1024, kd=32, batch=8 | 0.67× (large regression) |

The win comes from GEMM k-depth, not from the lower launch count. LARFT work is
\f$O(m k\,n_b)\f$ and doubles with \f$n_b\f$, which dominates once the GEMMs are too small to
benefit. So the hint is gated to where it was measured, n ≥ 1024 and batch ≥ 32, and returns 0
(the tuning-table behaviour) elsewhere. `BATCHLAS_SY2SB_ORMQR_NB`: unset uses the gate, `0`/`off`
never hints, a positive integer forces that width. What this did to the two-stage band-width
optimum (nothing good for wide kd) is at
[syev: the two-stage band width kd](syev.md#syev-the-two-stage-band-width-kd).

## sytrd: the Householder chase against the Givens chase

Stage 2 has two chases: the Givens chase (`sytrd_sb2st`, LAPACK `DSBTRD`/`ZHBTRD`, writes
`tau = 0` and discards \f$Q_2\f$) and the Householder chase (`sytrd_sb2st_hh`, retains the
reflectors). The Householder chase was written to remove the kd = 1 clamp that the Givens chase
forced on eigenvector mode, which degenerated stage 1 into an unblocked BLAS-2 reduction. The
eigenvalues-only path kept the Givens chase on the belief that it was cheaper. It is not: it is
about **5× more expensive** on this GPU, and that belief is why eigenvalues-only, which does
strictly less work, measured 3.7–4× *slower* than eigenvector mode at n = 1024.

The reason is occupancy, not arithmetic. Both chases are sequential per matrix and parallel only
over the batch, but `sytrd_sb2st_hh` has a 256-thread 2D lane mapping while the Givens path runs
one 32-lane sub-group per matrix (`sytrd_sb2st_cta.cc`), with a mostly serial `lid == 0` spine in
its kd > 32 fallback (`sytrd_sb2st.cc`): 8× fewer lanes per matrix, and at batch 1, 32 threads on
a 128-SM device. Measured, float, n = 1024, kd = 32 (undated): Givens chase about **366 ms**,
Householder chase **67.5 ms**, the latter essentially flat to batch 128.

Both modes now use the Householder chase (`syev_two_stage`, `syevx_direct_subset`).
`BATCHLAS_SYEV_TWO_STAGE_CHASE=givens` restores the old values-only behaviour for an intra-run
A/B; it has no effect with eigenvectors. The cost of the switch is memory: values mode now also
allocates the stage-2 reflectors V and tau, which it discards.

## sytrd: the Householder chase work-group shape

The chase is sequential per matrix, so a work-group owns one problem and the only parallelism
inside it is the ≤ kd × kd window of the current step. With a 32-thread work-group and kd = 32
each lane looped 32× serially, and at batch 128 the kernel occupied about 4096 threads on a GPU
with about 196k slots: **about 2% occupancy**, which made the chase cost as much as the
back-transform. `kWg = 256` with a 2D (row, column-chunk) mapping puts 8 lanes on each window row
instead of 1.

The schedule is the plain sequential one (per sweep, eliminate, then chase the bulge to the
bottom), not LAPACK's pipelined `THGRSIZ`/`GRSIZ`/`SHIFT=3` order: the pipelining exists for
multi-core CPU parallelism, and here the parallelism comes from the batch and from lanes inside
each window. It was validated in `playground/sb2st_hh_sequential.py` against tridiagonality, d,
signed e, orthogonality of Q and the reference spectrum.

## sytrd: the Q2 back-transform design

\f$Z := Q_2 Z\f$ with \f$Q_2 = H_1 H_2 \cdots H_m\f$ in generation order. Reflectors act on rows,
so columns of Z are independent. Three kernels exist, in order of preference.

**Streaming (the first design).** Each work-group takes one (batch item, column chunk) and walks
the whole reflector list, no inter-work-group synchronisation and no launch per sweep; lanes map
to reflector rows so Z accesses are contiguous. It is flop-optimal (\f$2n^3\f$, against about
\f$4n^3\f$ for a `larft`/`larfb` formulation whose V panels are zero-padded) but memory-bound.
Wang et al. (PPoPP'25) found a hand-written BLAS-2 back-transform beat MAGMA's BLAS-3 one by 1.5×
on A100 for this reason.

**Resident tile.** The bottleneck is traffic on Z: every reflector reloads its kd rows, and each
row of Z is touched about n/2 times, so the naive form moves about \f$n^3\f$ elements for
\f$2n^3\f$ flops. `larft`/`larfb` cannot fix that here. Two reflectors commute iff their row
ranges are disjoint (\f$|u - u'| \ge k_d\f$, \f$u\f$ = start − 1). The BLAS-3-groupable sets
(consecutive u, one per sweep at a fixed chase step) interleave: for n = 32, kd = 4, generation
order gives u = 0, 4, 8, 1, ..., so the groups {0..3} and {4..7} cannot be linearised as contiguous
units. In any valid schedule consecutive reflectors are mutually disjoint (that is what makes them
schedulable), so row-sharing ones are always far apart in the product. **Rejected: reordering into
larfb blocks.** Instead one work-group keeps a C-column tile of Z in local memory, applies every
reflector in the original order and writes it back once, cutting traffic from about \f$n^3\f$ to
\f$2 n \cdot n_{cols}\f$ per matrix.

**Wave (the default).** The resident tile fixed the traffic but not the serial chain: one
work-group walked all m reflectors with 32 threads. At n = 1024, kd = 64 that is 8687 dependent
steps, and the tile's local footprint capped occupancy near one block per SM. The reflectors of
one sweep act on disjoint rows, so they commute; `build_sb2st_hh_wave_offsets` recovers those runs
from the schedule. At n = 1024, kd = 64 the 8687 reflectors form **1022 waves** averaging 8.5
reflectors, an 8.5× shorter chain. S sub-groups share a tile and take reflectors round-robin
within a wave, with one work-group barrier between waves. Since the tile costs \f$n\cdot C\f$
regardless of S, going from 32 to 32·S threads multiplies threads per byte of local memory by S.
With this landed \f$Q_2\f$ became about 3× cheaper, which moved the two-stage kd optimum back down
([syev: the two-stage band width kd](syev.md#syev-the-two-stage-band-width-kd)).

The eigenvector path applies \f$Z := Q_1 (Q_2 Z)\f$, about \f$4n^3\f$. Forming \f$Q_1 Q_2\f$
first would be about \f$5.3 n^3\f$ (it needs \f$Q_1\f$ materialised) and pays only if the extra
work can overlap `stedc`, which the in-order queue does not currently allow.

## sytrd: sb2st back-transform tile width

For the single-sub-group resident-tile kernel, two costs set the tile width C. Holding a column of
Z resident already gives the full ~n/2 reuse, so reuse does not grow with C. What grows is the
amortisation of V reads: V is re-read once per column tile, and at C = 1 that alone is about
292 GB at n = 1024, batch 128. Larger C shrinks that but grows the footprint
(\f$n \cdot C \cdot \text{sizeof}(T)\f$) and cuts occupancy (32-thread work-groups, so a 32 KB tile
is about one block per SM). Back-transform alone, ms (undated):

| shape | C=0 (stream) | C=1 | C=2 | C=4 | C=8 | C=16 |
| --- | --- | --- | --- | --- | --- | --- |
| n=256, b=1024 | 39.9 | 109.2 | 44.8 | **25.9** | 27.1 | 42.8 |
| n=512, b=512 | 107.6 | 184.6 | **107.0** | 109.7 | 174.2 | 238.3 |
| n=1024, b=128 | 420.7 | 379.5 | **332.2** | 396.2 | 553.6 | 1391.4 |
| n=1024, b=256 | 913.5 | 752.0 | **660.5** | 786.1 | 1105.7 | 2782.9 |

The optimum tracks a footprint of about 8 KB (`kTargetLocalBytes = 8192`), capped at 4 columns.
`BATCHLAS_SB2ST_BACK_TILE` overrides it with a *raw* value, because 0 is meaningful (it selects
the streaming kernel); it is not an abbreviation of `BATCHLAS_SB2ST_BACK_TILE_W`, which drives the
wave kernel.

## sytrd: sb2st back-transform sub-groups per work-group

For the wave kernel, more sub-groups than a wave holds leave sub-groups idle at every barrier, and
fewer serialise the wave, so `subs` tracks the mean wave width (about \f$n / 2k_d\f$, read off the
schedule). Back-transform alone, float, tile = 8, ms (undated):

| subs | 256 / 1024 | 512 / 512 | 1024 / 128 | 1024 / 256 |
| --- | --- | --- | --- | --- |
| 8 | **17.1** | **51.0** | 123.8 | 245.0 |
| 16 | 20.6 | 58.5 | **103.5** | **207.1** |
| mean wave | 8.5 | 8.5 | 16.5 | 16.5 |

8 wins where waves hold about 8 reflectors, 16 where they hold about 16. With S sub-groups the
tile is shared by 32·S threads, so a wider tile no longer costs occupancy the way it did at S = 1;
float's best tile is 8 at every n tried. Tile and subs are a **measured pair**: forcing only one of
`BATCHLAS_SB2ST_BACK_TILE_W` / `BATCHLAS_SB2ST_BACK_SUBS` yields a geometry (for example (8,4))
that was never measured, and a sweep that pins one and reads the other from the default is
measuring neither optimum.

## sytrd: sb2st per-type wave geometry

`sb2st_back_tile_for` / `sb2st_back_subs_for` return a per-type geometry, and 0 ("no opinion") for
every unswept cell and every real type, so real scalars take byte for byte the path they took
before. The budget heuristic is not type-blind (its budget is in bytes, and per-column cost is
\f$n\cdot\text{sizeof}(T)\f$): replayed at kd = 32 it gives float (8,4) / (8,8) / (8,16) but
cfloat (8,4) / (8,8) / (4,16) at n = 256 / 512 / 1024. What complex inherits from float is the
32 KB *budget*, the wrong budget for a kernel that is register-bound (stage 2 at n = 512 cfloat
issues on 35.3% of SM cycles at 49.8% occupancy). Measured, batch 512, two-stage eigenvectors,
µs/matrix:

| type, n | heuristic (tile 8, subs 8) | (2, 4) | gain |
| --- | --- | --- | --- |
| cfloat, 512 | 972.41 | **855.75** | 1.136× |

That is the entire table. cfloat at n = 256 and 1024, and both double types at every n, are
**unswept** and return 0 rather than inheriting 512's answer, since the float subs table shows the
optimum moving with n; the bucket is an equality (`n == 512`) for the same reason. The end-to-end
check is [syev: per-type sb2st back-transform geometry (WP4)](syev.md#syev-per-type-sb2st-back-transform-geometry-wp4).
To fill the missing rows, sweep n ∈ {256, 512, 1024} with kd passed positionally, for example
`sb2st_hh_benchmark 256 1024 32`: the benchmark's default n = 256 row is registered with kd = 16,
which gives subs = 8 where the solver's kd = 32 gives 4, so sweeping it measures a geometry `syev`
never runs. This is the cheap fraction of the complex stage-2 occupancy problem and is **not
additive** with a type-aware retiling of `unmqr_hb2st_wave`
([syev: open debt, complex stage-2 occupancy](syev.md#syev-open-debt-complex-stage-2-occupancy)).

## sytrd: the sb2st tuning-precedence trap

Precedence for the wave tile and subs is: the legacy env knob, then the per-type constant, then
`tuning::`, then the budget heuristic, with the local-memory clamp applied to all four. The
per-type constant sits **above** `tuning::` deliberately. `tuning::` is generated by
`evaluation/tuning/generate_tuning_header.py` from the sb2st cases in `spaces/default.json`
(back_tile × back_subs at n = 256 / 512 / 1024), but the only registration those cases could
measure was `BM_SB2ST_HH_BACK<float>` until the complex one was added, so a retune bakes float's
n = 512 optimum (8,8) into `SB2ST_BACK_*_LARGE`. With `tuning::` first, that constant is non-zero,
the per-type helper is never reached, and cfloat at n = 512 silently goes from 856 back to
972 µs/matrix, with nothing failing, because no test reaches n = 512
(`sytrd_sb2st_hh_tests.cc` sweeps n ∈ {16, 32, 48, 160}). The per-type constant returns 0 for every
real type, so `tuning::` still owns every cell the float bench can measure.

## sytrd: the sb2st wave-knob spelling history

`BATCHLAS_SB2ST_BACK_WAVE=0` falls through from the wave kernel to the single-sub-group tiled
kernel (kept for comparison, and as the fallback when the tile does not fit). The knob was once
parsed with a local `atoi`, under which every non-numeric spelling was 0, so `=off` and `=false`
disabled the wave path. Routing it through the shared `env_falsy` / `env_int_or` helpers inverted
that: `env_falsy` matches only {0, false, FALSE, off, OFF}, and `env_int_or` hands an unparseable
value back as the fallback 1, so `=False`, `=Off` and `=no` silently turned the wave path *on*. A
wave-versus-tiled A/B driven by such a spelling measures the wave kernel against itself and reports
the two paths identical. The fix is a local case-folded disable set (`sb2st_wave_disabled`);
`env_falsy` was deliberately not widened, because its contract is "exactly the spellings the six
parsers it replaced accepted", and broadening it would change every other call site. Anything not
in the set enables the wave path, including a typo: fail-open, so a mistyped value cannot silently
cost the fast path.

## sytrd: open debts

- **The 25% local-memory headroom** in `sytrd_sb2st_cta.cc` (`lmem_budget = lmem_bytes * 3 / 4`,
  "for compiler/runtime usage") is an unmeasured guess.
- **`BATCHLAS_LATRD_IMPL=%device`** is recorded as slower with no figures; re-measure before
  deleting or reviving it.
- <b>Overlapping \f$Q_1 Q_2\f$ with `stedc`</b> (see the Q2 design above) needs an out-of-order
  queue; unmeasured.
- The complex chase `__mulsc3` check and the retiling are tracked on the syev page
  ([syev: open debt, complex stage-2 occupancy](syev.md#syev-open-debt-complex-stage-2-occupancy)).

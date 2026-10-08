# SYTRD and band reduction {#perf_sytrd}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · measured 2026-08 unless a section says otherwise

Measurements for the tridiagonal reductions under `syev`: the blocked one-stage reduction
(`sytrd_blocked` with its `latrd` panel) and the two-stage pair (`sytrd_sy2sb`, dense to band;
`sytrd_sb2st_hh`, band to tridiagonal, plus the \f$Q_2\f$ back-transform). Solver-level tier and
band-width decisions are on @ref perf_syev. The panel dominates the blocked solve; the two-stage
path is limited by the chase and the \f$Q_2\f$ back-transform, both occupancy problems.

## sytrd: the latrd implementation selector

`BATCHLAS_LATRD_IMPL` selects the panel kernel in `src/extensions/latrd_lower_panel.cc`. It is read
on every call, so one process can A/B the kernels.

| Value | Kernel |
| --- | --- |
| unset, or any other value | Legacy (default below `latrd_grid_min_n`) |
| `%device` | Device-BLAS variant (`device::hemv`, `her2k`). Slower; no figures recorded |
| `grid` | Grid kernel |

`BATCHLAS_SYTRD_IMPL` (`sytrd_blocked.cc`) is latched in a `static` and does not behave this way.

## sytrd: the latrd grid-path threshold

The grid path gives several work-groups per matrix, which fixes panel starvation at large n. It
costs five device-scope grid barriers per panel column. Legacy/grid time ratio (> 1: grid wins;
float, eigenvalues only, blocked provider, undated, before 2026-08-04):

| n | b=1 | b=8 | b=64 |
| --- | --- | --- | --- |
| 256 | 0.79 | 0.74 | 0.75 |
| 512 | 1.03 | 1.03 | 0.95 |
| 768 | 1.43 | 1.41 | - |
| 1024 | 1.94 | 1.90 | 1.24 |
| 2048 | 4.10 | 3.63 | - |

n = 128 loses (0.71–0.77); n = 384 is neutral (0.95).

**Gate: `latrd_grid_min_n = 768`.** Every measured point at or above it wins: at least 1.24× at any
batch, at least 1.41× at batch ≤ 8. The crossover is a barrier-latency property of the device;
re-measure on new hardware before changing it. The gate also holds in eigenvector mode
([syev: latrd grid gate confirmed in eigenvector mode](syev.md#syev-latrd-grid-gate-confirmed-in-eigenvector-mode)).
The win shrinks with batch (eigenvector mode, n = 768: 1.38× at batch 1, 1.09× at batch 64; the table above is eigenvalues only and has no batch-64 cell at n = 768).

## sytrd: the latrd grid kernel and its co-residency cap

Each panel column's trailing row range \f$[i+1, n)\f$ is split into G contiguous blocks, one per
work-group. The whole-range reductions (reflector sumsq, gamma and delta, final dot) combine through
global scratch in fixed group order, so results are run-to-run deterministic. Five grid barriers per
column.

The barrier is a sense-reversing software barrier, which terminates only if every participating
work-group is resident. `choose_grid_launch` therefore launches at most `MAX_COMPUTE_UNITS`
work-groups (one per SM), schedulable for any work-group size ≤ 256. G = 1 runs the legacy kernel.
The grid path is GPU-only. Its scratch is process-lifetime and not taken from the caller's
`BumpAllocator`, so no `*_buffer_size` query changes meaning.

Proposed, unbuilt: a cap of `SMs × achievable blocks per SM` from the kernel's real footprint
([syev: remaining unmeasured items](syev.md#syev-remaining-unmeasured-items)).

## sytrd: the vacuous latrd grid-groups A/B

`BATCHLAS_LATRD_GRID_GROUPS` was meant to force G, but the residency cap clamps it to 1 at batch ≥ 128
and it is never read above that. The recorded L2-residency A/B (batch 512–1024) therefore compared
legacy with legacy. Identical timings are the signature: float n = 256, batch 1024 gave 31.06 and
32.19 ms; cfloat n = 512, batch 512 gave 698.8 and 698.3 ms.

`BATCHLAS_LATRD_GRID_FORCE_UNSAFE` bypasses the cap. It compiles in only with
`BATCHLAS_ALLOW_UNSAFE_ENV` and has never been exercised.

> **Warning:** a forced launch that is not co-resident hangs without failing, and looks like slow JIT.
> Run such measurements under `timeout`. Co-residency rests on in-order block dispatch, which the spec
> does not promise ([syev: harness unblockers (WP0)](syev.md#syev-harness-unblockers-wp0)).

## sytrd: complex multiply in the latrd symv

clang implements C99 Annex G: a complex product with a NaN result calls `__mulsc3` (`__muldc3`) unless
`-fcx-limited-range` or `-ffast-math` is passed, and this build passes neither. In the matvec inner loop
the recovery branch blocks unrolling, so several loads cannot stay in flight. The kernel is
memory-latency bound.

The symv in `latrd_lower_panel.cc` writes `acc += a * b` and `acc += conj(a) * b` in explicit real
arithmetic. The only semantic change is for operands already holding Inf or NaN. Real types are
unaffected (float and double at n = 512, batch 1024: 50.86 and 50.87 µs per matrix, before and after).
Converting the other complex multiplies in the kernel made it slower, because the expanded form costs
registers and occupancy: symv only 108.4 µs per matrix against 125.8 with every site converted
(n = 512, batch 1024). General rule:
[agent guide, GPU kernel facts](../developer/agent-guide.md#11-gpu-kernel-design-facts-sm_89-mostly-general).

## sytrd: rejected latrd symv loop shapes

The legacy symv splits \f$\mathrm{acc} = \sum_c A_h(r,c)\,v(c)\f$ at \f$c = r\f$ into a row walk and a column
walk. Rejected after measurement: unrolling 8 and two rows per iteration (more registers than the extra
parallelism returns), and a \f$c > r\f$ term with one sub-group per column. The last one coalesces
(20 → 9.7 sectors per request), but its extra barrier breaks reuse between passes, short columns idle
lanes, and DRAM traffic and runtime rose. The access-pattern change did not reduce the number of reads. The open single-read design is at
[syev: open debt, the single-read panel symv](syev.md#syev-open-debt-the-single-read-panel-symv).

## sytrd: the rank-2k trailing update in the blocked reduction

\f$A_{22} \mathrel{-}= V W^H + W V^H\f$ is one `syr2k`, which touches only the triangle the panel loop
reads, replacing two full \f$n_2 \times n_2\f$ GEMMs. The update alone is 3.4–3.6× faster (RTX 4090, float,
undated). Times for nb = 16 / 24 / 32 (ms): n = 512, batch 1024: 264 / 253 / 248 becomes 228 / 227 / 232.

- **Route:** CUDA and float only. Other types fall to `syr2k_vendor_impl`, a host loop of one
  `cublasXsyr2k` per batch member, which is **7.8× slower** than the GEMM pair in double (n = 256,
  batch 1024).
- **Complex float** goes through `her2k` (one batched `gemm_vendor` into scratch, then
  `accumulate_hermitian<TwoSided=true>`). At \f$n_2 = 480\f$, batch 512 it beats the pair 1.32–1.33×; end to
  end 1.043× ([syev: her2k trailing update for complex float (WP3)](syev.md#syev-her2k-trailing-update-for-complex-float-wp3)).
- **Complex double** is excluded: 16 bytes per element of scratch, and unmeasured.
- The route asks the backend's library question, not the device family, so a vendor-free build answers it correctly.

## sytrd: her2k scratch fit headroom

The her2k fast route needs about \f$8 n_2^2 \cdot \text{batch}\f$ bytes of scratch against a
`GLOBAL_MEM_SIZE / 4` budget (about 6.0 GiB on a 24 GiB 4090). If it does not fit, the backend falls to
the 7.8× host loop. The call site asks the backend's fit predicate first and keeps the GEMM pair on a
no. The check runs per panel, since \f$n_2\f$ shrinks each iteration. Shapes: n = 512, batch 1024 needs
1.76 GiB (≥ 3.4× headroom); n = 448, batch 585 (the cfloat blocked/vendor crossover in `syev.hh`) needs
0.75 GiB; forced blocked n = 1024, batch 1024 needs 7.51 GiB and does not fit.

The ceiling is about \f$8.0\times 10^8\f$ elements of \f$n_2^2 \cdot \text{batch}\f$. That is outside the routed
region but reachable by pinning the provider. On an out-of-order `Queue` this route also drains the device
between GEMM and fold, once per panel; the benchmarks use in-order queues and do not see it.

## sytrd: no symmetrize after the rank-2k update

After the triangle-only update the upper triangle of A is stale, and nothing symmetrizes it. Every
reader uses only the lower triangle: `latrd_lower_panel` (all three variants) splits the symv at
\f$c = r\f$; the device variant's `device::hemv<Uplo::Lower>` mirrors that; the fused trailing update
guards with `if (r < c) continue` or `device::her2k<Uplo::Lower>`; `restore_tridiag_lower` writes only the
superdiagonal.

> **Note:** her2k forces \f$\mathrm{imag}(\mathrm{diag}) = 0\f$ on its block; the GEMM pair does not. Because
> `latrd`'s hemv reads the diagonal, **cfloat results drift in the last bits between the two routes**.
> Expect drift, not bitwise equality, in an A/B.

## sytrd: the dense-to-band ormqr block-width hint

`sytrd_sy2sb` applies each kd-wide panel (default 32) with `ormqr`. Its WY block width was keyed on
`A.rows()`, but the dimension that matters is \f$k = k_d\f$; the table value of 16 split a kd = 32 panel into
two WY blocks. The hint passes nb = kd (commit `f7f3c57`). Interleaved A/B, median of 15 rounds, idle GPU
(undated):

| Shape | nb = kd vs table |
| --- | --- |
| n=1024, kd=32, batch=64 | 1.19–1.20× faster |
| n=2048, kd=32, batch=32 | 1.36× faster |
| n=512, kd=32, batch=128 | 0.90× (regression) |
| n=1024, kd=32, batch=8 | 0.67× (large regression) |

The gain is GEMM k-depth. LARFT work, \f$O(m k\,n_b)\f$, dominates once the GEMMs are too small to benefit, so
the hint is gated to n ≥ 1024 and batch ≥ 32 and returns 0 (table behaviour) elsewhere.
`BATCHLAS_SY2SB_ORMQR_NB`: unset uses the gate, `0` or `off` never hints, a positive integer forces that
width. Effect on the band-width optimum:
[syev: the two-stage band width kd](syev.md#syev-the-two-stage-band-width-kd).

## sytrd: the Householder chase against the Givens chase

Stage 2 has two chases. The Givens chase (`sytrd_sb2st`, LAPACK `DSBTRD`/`ZHBTRD`) writes `tau = 0` and
discards \f$Q_2\f$. The Householder chase (`sytrd_sb2st_hh`) retains the reflectors.

The Givens chase is about **5× more expensive** on this GPU. The cause is occupancy: the Householder chase
maps 256 threads in 2D, while the Givens path runs one 32-lane sub-group per matrix (`sytrd_sb2st_cta.cc`),
with a serial spine for kd > 32 (`sytrd_sb2st.cc`). Float, n = 1024, kd = 32 (undated): Givens **366 ms**,
Householder **67.5 ms**, flat to batch 128. Before the switch, values-only ran 3.7–4× slower than
eigenvector mode at n = 1024.

Both modes use the Householder chase (`syev_two_stage`, `syevx_direct_subset`).
`BATCHLAS_SYEV_TWO_STAGE_CHASE=givens` restores the old values-only path for an intra-run A/B; it has no
effect with eigenvectors.

## sytrd: the Householder chase work-group shape

The chase is sequential per matrix, and the only parallelism inside a matrix is the ≤ kd × kd window of the
current step. With a 32-thread work-group and kd = 32 each lane looped 32 times serially: about 4096 threads
at batch 128 on a device with about 196k slots, roughly **2% occupancy**. That made the chase cost as much as
the back-transform.

`kWg = 256` with a 2D (row, column-chunk) mapping puts 8 lanes on each window row. The schedule is plain
sequential (per sweep: eliminate, then chase the bulge), not LAPACK's pipelined order. Parallelism comes from
the batch and the window lanes. Validated in `playground/sb2st_hh_sequential.py`.

## sytrd: the Q2 back-transform design

\f$Z := Q_2 Z\f$ with \f$Q_2 = H_1 H_2 \cdots H_m\f$ in generation order. Reflectors act on rows, so columns of Z are
independent.

- **Streaming** (first design): one work-group per (batch item, column chunk) walks all reflectors.
  \f$2n^3\f$ flops, memory-bound.
- **Resident tile:** keeps a C-column tile of Z in local memory. Rejected as a `larfb` reorder; kept as
  the tile kernel.
- **Wave** (default): reflectors of one sweep act on disjoint rows and commute. `build_sb2st_hh_wave_offsets`
  recovers the runs, and S sub-groups take them round-robin with one barrier per wave. At n = 1024,
  kd = 64 the 8687 reflectors form **1022 waves** averaging 8.5 reflectors, an 8.5× shorter dependent chain. \f$Q_2\f$ became about 3× cheaper, which moved the two-stage kd optimum down
([syev: the two-stage band width kd](syev.md#syev-the-two-stage-band-width-kd)).

The eigenvector path applies \f$Z := Q_1 (Q_2 Z)\f$, about \f$4n^3\f$. Forming \f$Q_1 Q_2\f$ first costs about
\f$5.3 n^3\f$ and pays only if it overlaps `stedc`, which the in-order queue does not allow.

## sytrd: sb2st back-transform tile width

For the single-sub-group tiled kernel, the tile width C trades V re-reads against footprint. V is re-read once
per column tile: at C = 1 that alone is about 292 GB at n = 1024, batch 128. The footprint is
\f$n \cdot C \cdot \text{sizeof}(T)\f$. Back-transform alone, ms (undated; bold = best):

| Shape | C=0 (stream) | C=1 | C=2 | C=4 | C=8 |
| --- | --- | --- | --- | --- | --- |
| n=256, b=1024 | 39.9 | 109.2 | 44.8 | **25.9** | 27.1 |
| n=512, b=512 | 107.6 | 184.6 | **107.0** | 109.7 | 174.2 |
| n=1024, b=128 | 420.7 | 379.5 | **332.2** | 396.2 | 553.6 |

C = 16 is slowest in every row (n = 1024, batch 256: 2782.9 ms, against 660.5 ms at C = 2). The optimum tracks a footprint of about 8 KB (`kTargetLocalBytes = 8192`),
capped at 4 columns. `BATCHLAS_SB2ST_BACK_TILE` takes a raw value, where 0 selects the streaming kernel. It
is not an abbreviation of `BATCHLAS_SB2ST_BACK_TILE_W`, which drives the wave kernel.

## sytrd: sb2st back-transform sub-groups per work-group

For the wave kernel, `subs` tracks the mean wave width, about \f$n / 2k_d\f$. Extra sub-groups idle at every
barrier and too few serialise the wave. Float, tile = 8, ms (undated):

| subs | 256 / 1024 | 512 / 512 | 1024 / 128 | 1024 / 256 |
| --- | --- | --- | --- | --- |
| 8 | **17.1** | **51.0** | 123.8 | 245.0 |
| 16 | 20.6 | 58.5 | **103.5** | **207.1** |
| Mean wave width | 8.5 | 8.5 | 16.5 | 16.5 |

> **Note:** tile and subs are a measured pair. Forcing only one of `BATCHLAS_SB2ST_BACK_TILE_W` or
> `BATCHLAS_SB2ST_BACK_SUBS` gives a geometry (for example (8,4)) that was never measured.

## sytrd: sb2st per-type wave geometry

`sb2st_back_tile_for` and `sb2st_back_subs_for` return a per-type geometry, and 0 ("no opinion") for every
unswept cell and every real type, so real types take the old path. The budget heuristic is not type-blind:
cfloat inherits float's 32 KB budget while the kernel is register-bound. The one measured override is
cfloat, n = 512, batch 512 (µs per matrix): heuristic (8,8) 972.41, measured (2,4) **855.75**, a 1.136× gain.
Everything else in the cfloat and double rows is unswept and returns 0. The bucket is an equality
(`n == 512`) because the float subs optimum moves with n. End-to-end check:
[syev: per-type sb2st back-transform geometry (WP4)](syev.md#syev-per-type-sb2st-back-transform-geometry-wp4).

To fill the gaps, sweep n ∈ {256, 512, 1024} with kd passed positionally (`sb2st_hh_benchmark 256 1024 32`).
The default row registers kd = 16, which gives subs = 8, while the solver's kd = 32 gives 4. The gain is not
additive with a type-aware retiling of `unmqr_hb2st_wave`
([syev: open debt, complex stage-2 occupancy](syev.md#syev-open-debt-complex-stage-2-occupancy)).

## sytrd: the sb2st tuning-precedence trap

Precedence for the wave tile and subs: the legacy env knob, the per-type constant, `tuning::`, then the
budget heuristic. The local-memory clamp applies to all four.

> **Warning:** the per-type constant sits **above** `tuning::` on purpose. `tuning::` is generated by
> `evaluation/tuning/generate_tuning_header.py` from float-only sb2st cases. A retune bakes float's n = 512
> optimum (8,8) into `SB2ST_BACK_*_LARGE`, and with `tuning::` first, cfloat at n = 512 silently goes from
> 856 back to 972 µs per matrix. No test catches it: `sytrd_sb2st_hh_tests.cc` sweeps n ∈ {16, 32, 48, 160}.

## sytrd: the sb2st wave-knob spelling history

`BATCHLAS_SB2ST_BACK_WAVE=0` selects the single-sub-group tiled kernel. Spellings are matched through the
case-folded disable set `sb2st_wave_disabled`; any other value, including a typo, enables the wave path.

> **Warning:** do not route this knob through `env_falsy` or `env_int_or`. `env_falsy` matches only
> {0, false, FALSE, off, OFF}, and `env_int_or` turns an unparseable value into 1. Either way `=False`,
> `=Off` or `=no` silently enables the wave path, and a wave-versus-tiled A/B compares the wave kernel with itself.

## sytrd: open debts

- The 25% local-memory headroom in `sytrd_sb2st_cta.cc` (`lmem_budget = lmem_bytes * 3 / 4`) is unmeasured.
- `BATCHLAS_LATRD_IMPL=%device` is recorded as slower with no figures; re-measure before deleting or reviving it.
- Overlapping \f$Q_1 Q_2\f$ with `stedc` needs an out-of-order queue; unmeasured.
- The complex stage-2 retiling is tracked in
  [syev: open debt, complex stage-2 occupancy](syev.md#syev-open-debt-complex-stage-2-occupancy).

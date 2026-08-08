# SYEV performance: implementation plan

Derived from `SYEV_PERF_RESEARCH.md` (branch `worktree-syev-perf-research`, commit
`faa4f39`), which profiled `syev` for `float` and `complex<float>` at saturating batch
against `ad0fae7`. That document says *where the time is*. This one says *what to change*,
in what order, and what each change has to prove before it ships.

Every code reference below was verified against the tree at `ad0fae7`. Where reading the
source changed the picture the research document painted, that is called out under
**Trap** — those are the parts most likely to waste a day.

**Ground rules, inherited and non-negotiable.**

* Measure at saturating batch only. Ratios taken below saturation measure overhead, not
  algorithms.
* One process on an idle device (`nvidia-smi` first; this box has two 4090s and contention
  has manufactured 3.6× "wins" before). Warm the clocks; discard the first run — SYCL JIT
  has fabricated a 3.7× loss before.
* `--name` is a *substring* filter. `BATCHLAS_SYEV_PROVIDER` spellings are parsed in
  `include/blas/dispatch/env.hh:36` — `two_stage`/`two-stage`, not `TWOSTAGE`, which
  silently degrades to `Auto`.
* A kernel-level win is not a solver-level win. Every work package below has an
  **end-to-end** accept criterion measured through `syev_benchmark`, because a 2.16×
  kernel win has already turned into an 11% `gesvd` loss in this repo.

---

## Sequence

The research document's §7 order is kept, with two deviations, both stated:

| # | Package | Item | Est. size | Expected |
|---|---|---|---|---|
| **WP0** | Harness unblockers | (prereq for A3, B2, A5) | ~40 lines | 0× — enables measurement |
| **WP1** | Values-only tridiagonal solve | A2 | ~80 lines | **up to 1.35×**, values, n ≤ 320, both types |
| **WP2** | Benchmark `cta-large-n` | B3 | rebase + measure | unknown; decides its own fate |
| **WP3** | Complex trailing update | A1 | ~30 lines | 1.05–1.12× cfloat blocked |
| **WP4** | Per-type back-transform constants | A3 | ~40 lines | measured 1.14× cfloat two-stage |
| **WP5** | Single-read panel symv | B1 | new kernel | **1.22–1.82×** end to end, both types |
| **WP6** | Complex stage-2 occupancy | B2 | kernel rework | 1.49× cfloat two-stage; opens n ≥ 512 |
| **WP7** | Measurement debt | A4, A5, B4 | investigation | routing corrections |
| **WP8** | Block Jacobi | C1 | large | 1.5–2.5× at n = 128–256 (speculative) |

**Deviation 1 — WP0 goes first.** Two of the packages below cannot be measured with the
harness as it stands (§WP0). Fixing that is ~40 lines and removes the need to tune A3 and
B2 through the full solver.

**Deviation 2 — A2 (WP1) is promoted above B3.** §7 puts B3 first because its
implementation cost is sunk. That is true and WP2 keeps it near the front, but B3 has
*zero* performance data and may not pay at all, whereas A2 is a near-certain 1.35× over a
whole routing region. A2 is also the only package that reduces workspace, which matters to
A4.

---

## WP0 — Harness unblockers

Two recorded negative results are really harness defects, and both block a package below.

### WP0.1 `sb2st_hh_benchmark` registers only `float`

`benchmarks/sb2st_hh_benchmark.cc:97-98` registers `BM_SB2ST_HH_CHASE<float, CUDA>` and
`BM_SB2ST_HH_BACK<float, CUDA>` and nothing else, so `--type=cfloat` silently yields zero
rows. Both kernels this document most wants to tune (A3, B2) are the complex ones.

**Change.** Add the two `std::complex<float>` registrations next to the existing pair. The
benchmark bodies are already templated on `T` (`:27`, `:55`) and use
`base_type<T>::type`, so this is a registration change, not a port.

**Accept.** `sb2st_hh_benchmark --type=cfloat` produces rows, and the back-transform row at
n = 512 / batch 512 reproduces the 972 µs/matrix figure that §4 of the research document
measured through the full solver, to within noise.

### WP0.2 The grid `latrd` path cannot be forced

`latrd_lower_panel.cc:1163-1166`:

```cpp
const int forced_g = env_positive_int_or("BATCHLAS_LATRD_GRID_GROUPS", 0);
if (forced_g > 0) {
    G = std::min(forced_g, cap);           // never exceed the residency cap
}
```

`cap = MAX_COMPUTE_UNITS / batch` (`:1157`, integer division), and `:1158` returns the
legacy launch when `cap < 1`. So on a 128-SM card **the escape hatch is itself clamped by
the cap it is trying to escape**: at batch ≥ 128 there is no value of
`BATCHLAS_LATRD_GRID_GROUPS` that reaches the grid kernel. This is precisely why the
recorded L2-residency A/B was legacy-against-legacy.

**Change.** Add an explicitly-unsafe override that bypasses the residency cap — e.g.
`BATCHLAS_LATRD_GRID_FORCE_UNSAFE=1` gating `forced_g` past `cap` — documented as
deadlock-capable and for measurement only. Do **not** relax `cap` itself here; that is
WP7/A5, and it needs its own residency argument.

**Accept.** At n = 1024, batch = 128, `BATCHLAS_LATRD_IMPL=grid` with the override produces
a measurably *different* time from `legacy` (either direction). Today they are identical to
three digits, which is the signature of the path not being taken.

**Trap.** The barrier is a real software grid barrier. If the forced launch exceeds
residency it hangs rather than fails. Run forced-unsafe measurements with a timeout, and
recall the house rule for identifying a stuck kernel (`stedc-fusedcta-merge-hangs`): a hang
here looks exactly like slow JIT.

---

## WP1 — A2: stop computing eigenvectors the caller did not ask for

**Evidence.** 28.3% of the float eigenvalues-only solve at n = 256 batch 1024 is a full
eigenvector divide-and-conquer whose output is discarded (§2.3). `blocked` owns all of
n ≤ 320 in values mode, so this is the whole small-n values regime. stedc runs in real
arithmetic for both scalar types, so the saving is identical for `float` and `cfloat`.

**Where.** `src/extensions/syev_blocked.cc`, four sites in two mirrored branches:

| Site | Branch | What it does today |
|---|---|---|
| `:239` | complex | `const JobType internal_jobz = JobType::EigenVectors;` |
| `:244` | complex | `stedc<B, Real>(..., internal_jobz, ..., z_view)` |
| `:340` | real | same constant |
| `:345` | real | `stedc<B, T>(..., internal_jobz, ..., z_view)` |
| `:450` | complex | `stedc_workspace_size<B, Real>(..., JobType::EigenVectors, ...)` |
| `:478` | real | `stedc_workspace_size<B, T>(..., JobType::EigenVectors, ...)` |

The comment at `:236` explains the constant: *"The current STEDC implementation relies on
eigenvectors during recursion/merge, so we always run it in EigenVectors mode."* That is
accurate — do not simply flip the flag.

**Change.** In values mode, do not call `stedc` at all. Call `stebz`, exactly as the
two-stage path already does. The template to copy is `syev_two_stage.cc:262-278`:

```cpp
if (!want_eigvecs) {
    BATCHLAS_KERNEL_TRACE_SCOPE("syev_blocked.stebz_evals");
    auto m_span = pool.allocate<int32_t>(ctx, batch);
    StebzParams<Real> bp;
    bp.range = EigenRangeType::Index;
    bp.il = 0;
    bp.iu = n - 1;
    bp.order = SortOrder::Ascending;
    auto stebz_ws = pool.allocate<std::byte>(
        ctx, stebz_buffer_size<B, Real>(ctx, n, batch, bp));
    stebz<B, Real>(ctx, d_view, e_view, evals_view, m_span, stebz_ws, bp);
    return ctx.get_event();
}
```

It drops in unmodified in both branches: `d_view`/`e_view`/`evals_view` are already
`Real`-typed in the complex branch too (the phase-similarity kernel at `:200-232` produces
a real tridiagonal), which is the same precondition two-stage relies on.

There is **no `sterf` in this tree** (grepped: zero hits). LAPACK's `COMPZ='N'` answer is
not available; `stebz` is, and it is already proven in exactly this role for n ≥ 384.

**Secondary win, larger than it looks.** Values mode can also skip the workspace that only
the eigenvector path reads:

* `:181-182` (complex): `z_span` (`Real`, n²·batch) **and** `zc_span` (`T`, n²·batch)
* `:308` (real): `z_span` (`T`, n²·batch)
* `:450`/`:478`: the `stedc` workspace itself

At n = 256, batch = 1024, cfloat that is 268 MB + 537 MB of scratch that never gets
touched, plus the stedc workspace. Feeding this back into `syev_blocked_buffer_size` is
part of the change, not a follow-up — and it is the single most likely explanation to test
first in WP7/A4, where n = 2048 is suspected of being a capacity artifact.

In the complex branch the phase vector `S` (`:181`, `phase_view`) is also dead in values
mode; only `Dr = Re(D)` and `Er = |E|` are needed. Keep the loop, drop the `S` writes and
its allocation.

**Validation.**
* `ctest -R syev` scoped to the blocked provider. Values-mode eigenvalues must match the
  existing reference to the current tolerance — `stebz` is bisection, so expect *better*
  agreement, not worse, but the tolerance is two-sided in some tests.
* Explicitly cover n ≤ 32 (where `blocked` is not routed but is reachable by force) and
  n = 320 (the top of its region).
* Confirm ordering: `stedc` and `stebz` must both return ascending. `bp.order =
  SortOrder::Ascending` above matches what the blocked path's `stedc` produces.

**Measurement.**

```bash
CUDA_VISIBLE_DEVICES=1 build/benchmarks/syev_benchmark --backend=CUDA \
    --type=float,cfloat --warmup=2 --min_iters=5 256 1024 0 0 0 0   # jobz=0 = values
```
over n ∈ {64, 128, 192, 256, 320} with `BATCHLAS_SYEV_PROVIDER=blocked`.

**Accept.** ≥ 1.2× on float values-only at n = 256 (the arithmetic says 1.39× if the whole
28.3% goes and `stebz` is free; 1.35× is the honest target), no regression in eigenvector
mode, no accuracy regression. **Re-check routing afterwards:** if blocked gets 1.35×
faster in values mode, the 384–448 boundary where values mode currently switches to
two-stage may move up. That re-measurement is part of this package.

---

## WP2 — B3: rebase and benchmark `cta-large-n`

**Status found.** Branch `cta-large-n` (`04101dc`) is based on `27851a6`, which is **130
commits** behind `main`. That number is misleading in the encouraging direction:

| File | Commits on main since base | Churn |
|---|---|---|
| `include/blas/cta_limits.hh` | 0 | new file |
| `include/blas/dispatch/context.hh` | 0 | — |
| `src/extensions/sg_compat.hh` | 0 | — |
| `src/extensions/ormqr_cta.cc` | 0 | — |
| `src/extensions/sytrd_cta.cc` | 0 | — |
| `src/extensions/syev_cta.cc` | 1 | 1+/5− |
| `tests/ormqr_cta_tests.cc` | 1 | 6+/6− |
| `tests/syev_cta_tests.cc` | 1 | 18+/10− |
| `include/blas/functions/syev.hh` | **16** | **649+/20−** |

Eight of nine files replay essentially clean. The rebase is one file, and the branch's
change to it is **11 lines**, confined to `syev_supports_cta`: it replaces the hard
`n > 32` rejection with `n > cta_max_partition(sizeof(T), caps.local_mem_size)`.

**Trap — the rebase alone will measure nothing.** `syev_supports_cta` is a *functional
support* predicate. Since the branch was cut, `main` grew a separate *performance* gate,
`syev_cta_max_n_for_vectors()` (`syev.hh:349-380`), which defaults to 32 and rejects any
env value `> 32` outright:

```cpp
if (end == v || parsed < 0 || parsed > 32) return kDefault;
```

So after a clean rebase, `Auto` still cannot route above n = 32 and the branch appears to
do nothing. Two consequences:

1. **Benchmark through the forced provider, not `Auto`.** `syev.hh:318` records that "an
   explicitly forced `Provider::BatchLAS_CTA` still wins, since the forced branch returns
   before this is consulted" — so `BATCHLAS_SYEV_PROVIDER=cta` reaches the lifted kernel
   with no routing change at all. Get the numbers before touching any gate.
2. Only if the numbers justify it, raise the clamp and `kDefault` in
   `syev_cta_max_n_for_vectors`. That is a routing change and carries the LOBPCG caveat
   documented in that same comment block (`ILUKTests.SyevxInstrumentationAndPreconditioner`
   asserts `lose_count == 0` and is sensitive to which solver the projected Rayleigh-Ritz
   step uses).

**Steps.**
1. `git rebase main cta-large-n` in a worktree; hand-port the 11-line
   `syev_supports_cta` change into the current `syev.hh`.
2. Run the branch's own tests (it carries 36/36 + 6/6 + 5/5 passing).
3. Measure `BATCHLAS_SYEV_PROVIDER=cta` against `blocked` at n ∈ {33, 48, 64, 96, 128}
   (float) and {33, 48, 64} (cfloat) — the branch's measured local-memory limits — at
   saturating batch, both job modes.
4. Report the cost table. `blocked` at these n runs 2.2–11 µs/matrix, so this is a
   launch-overhead regime: ~15 kernel launches and all global round-trips disappear.

**Accept.** CTA beats `blocked` by ≥ 1.15× at any n in 33–128 → proceed to the routing
gate. Otherwise record the numbers and close the branch; the hypothesis will have been
answered for the price of a rebase, which is the point.

---

## WP3 — A1: give the complex trailing update a level-3 path

**Evidence.** `sytrd_blocked.cc:783`:

```cpp
constexpr bool syr2k_trailing_update_supported =
    (B == Backend::CUDA) && std::is_same_v<T, float>;
```

so `cfloat` takes the `else` branch at `:864-881` and issues **two full n₂×n₂ GEMMs** where
float issues one triangle-only `syr2k`. Vendor GEMM is 34.6% of the cfloat solve at n = 256
and 14.4% at n = 512; the trailing update is roughly half of it.

**The gate is not arbitrary, and this is the part to get right.** The comment at `:778-782`
explains it: `syrk`/`syr2k` reach a batched kernel *only* through the custom float route;
everything else falls to `syr2k_vendor_impl`, a host loop issuing one `cublasXsyr2k` per
batch member, measured **7.8× slower** than the GEMM pair in double. Repeating that mistake
in complex is the failure mode this package must avoid.

**Why complex is nevertheless different.** `her2k` is a *different function* with a
*different backend route*. `cublas.cc:644-665`:

```cpp
if (her2k_gemm_preferred(n, batch) && detail::expansion_fits(ctx, n, batch, product_bytes)) {
    ... gemm_vendor<Back, T>(ctx, A, B, product, alpha, T(0), ...);
    return accumulate_hermitian<T, /*TwoSided=*/true>(ctx, C, product, beta, uplo);
}
```

One batched GEMM into scratch, then a Hermitian fold that adds the product to its own
mirrored transpose — because `alpha·A·Bᴴ` and `conj(alpha)·B·Aᴴ` are conjugate transposes
of one another. That is **half** the arithmetic of the two GEMMs it replaces, not twice it.
`her2k_gemm_preferred` (`:422`) is `batch >= 2 || n >= 128` — true throughout our regime.

**Change.** In `sytrd_blocked.cc`, replace the `is_same_v<T, float>` gate with a condition
admitting complex-on-CUDA, and in the admitted complex case call
`her2k<B>(ctx, V2, W2, A22, {.alpha = T(-1), .beta = 1})` in place of the GEMM pair.

Semantics line up exactly: `her2k` with `alpha = -1` computes
`C ← −(V·Wᴴ + W·Vᴴ) + β·C`, which is the required `A22 -= V Wᴴ + W Vᴴ`. `beta` is
`float_t<T>` (real) — pass `1`, not `T(1)`.

The "no symmetrize" argument at `:829-845` already covers this case: it enumerates every
downstream reader of `A`'s upper triangle and explicitly names
`device::her2k<Uplo::Lower>` among the paths that stay below the diagonal. No new
correctness argument is needed — but re-read that comment block before editing, and extend
it rather than leaving it float-flavoured.

**Trap 1 — the fallback is a host loop, same as the one the float comment warns about.**
If `expansion_fits` returns false, `her2k_vendor` drops to `for (int b = 0; b < batch; ++b)
launch_single(...)` (`:687-691`). The budget is `GLOBAL_MEM_SIZE / 4`
(`triangular_expand.hh:73`) ≈ 6 GB on a 4090, and the scratch is `n₂²·batch·sizeof(T)`:

| shape | n₂ (max) | scratch | fits under 6 GB? |
|---|---|---|---|
| n = 256, batch = 1024 | 224 | 411 MB | yes |
| n = 512, batch = 512 | 480 | 944 MB | yes |
| n = 2048, batch = 64 | 2016 | 2.08 GB | yes, but see WP7/A4 |

So it fits today at the shapes that matter — but this must be *checked at the call site*,
not assumed, because the failure is silent and catastrophic. Guard the `her2k` call with
the same predicate the backend uses, and fall back to the existing GEMM pair when it fails.
A `BATCHLAS_EXPAND_MAX_BYTES=1` run is the cheap way to prove the guard works.

**Trap 2 — the crossover backing `her2k_gemm_preferred` was not measured at these
shapes.** The comment at `:415-421` describes a sweep over rank-k shapes generally; the
panel loop produces a *narrow* update — `k = ib = nb ∈ {16, 24, 32}` against
`n₂` up to 480. At that aspect ratio the GEMM is close to bandwidth-bound, and the fold
adds a full `n₂²·batch` write plus read that the two direct GEMMs never pay. The halved
arithmetic may not survive it.

**Therefore: measure the primitive before touching the solver.** `benchmarks/her2k_benchmark`
already exists. A/B `her2k` against the GEMM pair at exactly the panel shapes
(n₂ ∈ {224, 480}, k ∈ {16, 24, 32}, batch ∈ {512, 1024}, cfloat). If `her2k` does not win
*there*, this package stops and costs one afternoon instead of a regression.

**Validation.** `ctest -R sytrd` and `ctest -R syev` with cfloat, n spanning both the
`n₂ ≤ 128` small-update branch (`:812`) and the large branch. The routing between them is
unchanged.

**Accept.** ≥ 1.05× end-to-end on cfloat `blocked` at n = 256, batch 1024, eigenvectors,
and no float regression (float takes the same code path as before — assert that by
diffing the kernel trace, not by assuming).

---

## WP4 — A3: per-type back-transform tile/subs

**Evidence.** Measured in §4: shipped constants are *exactly* optimal for float
(332.65 swept vs 332.46 shipped) and cost cfloat **1.14×** (972.41 shipped vs 855.75 at
`tile = 2, subs = 4`). The complex optimum is at a smaller tile and fewer sub-groups, which
is what the occupancy collapse in §3 predicts (35.3% SM throughput, occupancy halved from
82.8% to 49.8%).

**Where.** `src/extensions/sytrd_sb2st_hh.cc:786` and `:808`:

```cpp
if (tile <= 0) tile = tuning::sb2st_back_tile_for_n(n);
...
if (subs <= 0) subs = tuning::sb2st_back_subs_for_n(n);
```

**Trap — `tuning_params.hh` has no per-type facility at all.** Every accessor there is
`_for_n(int32_t n)`; grepping for `is_complex`/`ScalarKind`/`base_type` in that header
returns nothing. The generated table comes from a float-only harness. Do not try to make
the generator type-aware as part of this package — that is a harness project, and the
CMake header target is a known no-op (`tuning-harness-traps`).

**Change.** Follow the precedent already set for per-type constants in `syev.hh:322` and
`:434`, where the call site is templated and decides with `constexpr`:

```cpp
using Real = typename base_type<T>::type;
constexpr bool kReal = std::is_same_v<T, Real>;
```

`sytrd_sb2st_hh.cc`'s call site is already templated on `T`, so add a small
`sb2st_back_geometry_for<T>(n)` helper local to that file which returns the tuned
(float) values for real `T` and the complex-measured values otherwise, keeping
`tuning::` as the source for the real case and the env knobs on top. Precedence stays:
env → per-type constant → budget heuristic.

**Trap — only n = 512 was measured.** Shipping `tile = 2, subs = 4` for all n in complex is
not supported by the data. The `subs` comment at `:801-806` shows the float optimum
*changes with n* (8 wins where waves hold ~8, 16 where they hold ~16); there is no reason
complex is flat. Sweep n ∈ {256, 512, 1024} for cfloat and bucket accordingly — and this is
exactly what WP0.1 makes cheap, since after it the sweep runs against
`sb2st_hh_benchmark` directly instead of the whole solver.

**Trap — the instantiation list is finite.** `:821-829` instantiates `BL_WAVE_CASE(C, S)`
for C ∈ {1,2,4,8} × S ∈ {4,8,16}; anything else silently falls through to the tiled
kernel. `(2, 4)` is in the list. Any bucket chosen outside it must be added, and adding
combinations costs device-link time — the build here is device-link-bound
(`build-is-device-link-bound`).

**Accept.** ≥ 1.10× on cfloat two-stage eigenvectors at n = 512, batch 512, and
bit-identical float behaviour (same instantiation selected — verify from the kernel trace).

**Do not oversell it.** As §4 records, 855.75 µs/matrix is still behind blocked (698) and
the vendor (707) at that shape, so this flips no routing decision by itself. It is worth
taking because it is free and because it is a prerequisite for WP6 mattering. **A3 and B2
are not additive** — A3 is the cheap fraction of the same occupancy problem.

---

## WP5 — B1: a single-read, shared-memory-staged panel symv

**The single biggest item.** The panel is 35–71% of every blocked solve. §3's counters, on
`LatrdLowerPanel`, one panel, ib = 32, j₀ = 0:

| | float n=256 b=1024 | float n=512 b=256 | cfloat n=512 b=256 |
|---|---|---|---|
| DRAM | 0.14× ideal | 1.13× | 1.43× |
| L2 | 1.95× | **2.34×** | **2.50×** |
| L1TEX | **11.9×** | **12.2×** | **15.8×** |
| SM throughput | 52.7% | **10.8%** | **10.7%** |

DRAM is already near-optimal; L2 at 2.3× is the triangle being read twice; L1 at 12–16× is
the uncoalesced column walk. Ceiling from the DRAM floor at n = 512: **2.7×**.

**Where.** `src/extensions/latrd_lower_panel.cc:493-535`. The two reads the counters are
describing are literally adjacent:

```cpp
// c in [i+1, r]: walk row r of the lower triangle.   <- coalesced across the warp
for (int c = i + 1; c <= c_split; ++c)  mac(acc, Ab(r, c), v_local[c]);

// c in (r, n): walk column r of the lower triangle.  <- stride lda across the warp
for (int c = c_split + 1; c < n; ++c)   mac_conj(acc, Ab(c, r), v_local[c]);
```

Thread *r* reads `A(r,c)` on its row walk; thread *c* reads that same element as `A(c,r)`
on its column walk. Hence 2.3× L2. And in the second loop consecutive lanes are `lda`
apart, so one warp request touches 32 sectors for 4 useful bytes each. Hence 12–16× L1.

**Structural constraint the research document does not state.** This symv is not a
standalone kernel. It sits inside the per-reflector loop of a **single work-group per
matrix**, with `v_local[n]` and `wcol_local[n]` in local memory, and it runs `ib` times per
panel over a shrinking trailing triangle. Any redesign has to keep the fusion — the
reflectors are sequentially dependent, so the loop cannot be hoisted or batched. A
MAGMA-style multi-block symv is therefore *not* a drop-in; the tiling has to happen
*within* the work-group. (The two loops quoted above are at `:519` and `:527`.)

**Design.** Tile the trailing triangle in square blocks (start at 32×32) and process them
inside the existing work-group:

* **Off-diagonal tile** (row-block R below col-block C): stage `A[R,C]` into local memory
  with a coalesced, column-major-friendly load (consecutive rows are contiguous, so a
  thread-per-row load is coalesced for *both* uses). From the one staged copy compute both
  `y[R] += A[R,C]·v[C]` **and** `y[C] += A[R,C]ᴴ·v[R]`. One load, two updates — this is the
  whole point, and it is what distinguishes this from the attempt the kernel's own comment
  at `:507-513` records as rejected ("one sub-group per column … the extra barrier destroys
  reuse"). That experiment changed the *access pattern* without changing the *number of
  reads*, so it paid a barrier for no extra work. Here the barrier is paid once for twice
  the work.
* **Diagonal tile:** stage the lower half; the transposed contribution folds into the same
  tile.
* **Accumulation:** `y` (`wcol_local`, n elements ≤ 2 KB at n = 512, float) is already
  entirely in local memory, so the transposed partials can be reduced there — via
  per-sub-group private partials plus a reduction, or local atomics. Measure both; local
  atomics on a hot 2 KB array may serialise.

**Budget check before writing any of it.** Local memory currently holds `v_local[n]` +
`wcol_local[n]`. A 32×32 `cfloat` tile is 8 KB on top of that. At n = 512 cfloat that is
4 KB + 4 KB + 8 KB = 16 KB, which still allows ≥ 2 blocks/SM on sm_89 (100 KB). Confirm
against `local_mem_size` before committing to a tile width, and remember the house rule
that the thread tile must shrink as the scalar widens — but not too far
(`register-residency-traps`).

**Staging.**
1. Prototype against `benchmarks/latrd_lower_panel_benchmark` alone. Its default grid is
   n ∈ {64,128,256,512} × batch 1024, ib = 32, j₀ = 0, fuse ∈ {0,1}
   (`latrd_lower_panel_benchmark.cc:13-19`) — the same shape the counters were taken at.
2. Re-run `ncu` with the metric list from §8 and confirm the *mechanism*, not just the
   time: L1TEX must fall from ~12× toward ~2×, and L2 from 2.3× toward ~1×. If the time
   improves but the traffic does not, something else changed and the win will not
   generalise.
3. Only then wire it into the solver, behind `BATCHLAS_LATRD_IMPL` so the old kernel stays
   A/B-able.

**Accept.** ≥ 1.6× on the panel kernel alone at n = 512 (against the 2.7× ceiling), *and*
the end-to-end figures below. Predicted from the measured phase shares:

| | at 2.7× | at 2.0× |
|---|---|---|
| cfloat n=512, vectors (panel 71.5%) | 1.82× | 1.56× |
| cfloat n=256, vectors (panel 41.7%) | 1.36× | 1.26× |
| float n=256, vectors (panel 35.5%) | 1.29× | 1.22× |
| float n=256, values (panel 52.7%) | 1.50× | 1.36× |

**Re-route afterwards.** This is large enough to move the cfloat blocked/vendor crossover
well past 512 and to change the blocked/two-stage boundary for float. Re-running the
routing grid is part of this package, not a follow-up — and note that WP1 will already have
changed the values-mode boundary, so do the routing sweep once, after both.

---

## WP6 — B2: fix the complex stage-2 occupancy

**Evidence.** §3, n = 512, batch 512:

| | float | cfloat |
|---|---|---|
| chase — SM / occupancy | 45.0% / 65.5% | 29.8% / 45.1% |
| back-transform — SM / occupancy | **93.7% / 82.8%** | **35.3% / 49.8%** |

The float back-transform is genuinely saturated. The complex one is not, and occupancy is
halved — a register-pressure signature, not an arithmetic one. So the complex kernel's
3.90× cost is *not* the price of complex arithmetic.

**Two independent sub-items.**

**WP6.1 — type-aware tiling in `unmqr_hb2st_wave`.** A3 (WP4) tunes the two constants that
already exist; this changes the per-thread working set itself so that the complex kernel
gets its registers back. Same lesson as `register-residency-traps`. Do WP4 first — it
establishes how much of the gap is reachable by constants alone, and the remainder is what
WP6.1 has to justify.

**WP6.2 — the Annex G escape in the chase.** The chase is at **4.85×** float, *above* the
~4× arithmetic ratio, on a latency-bound profile. That is the signature of
`std::complex` `operator*` emitting the C99 Annex G `isnan` branch plus a `__mulsc3` call
in device code, which cost the latrd panel 1.22–1.29× when it was found there
(`complex-multiply-annex-g-trap`). The research document records that `__mulsc3` is still
present in `libbatchlas_extensions_sytrd.so`.

Confirm before changing anything:

```bash
nm -C build/lib/libbatchlas_extensions_sytrd.so | grep mulsc3
# and disassemble the chase kernel to see whether the call is on its hot path
```

The fix is the established one: replace `*` with an explicit multiply in the hot loop
**only**. The house rule from that trap is to convert the hot loop and not the file — a
blanket conversion has costs elsewhere.

**Accept.** The prize is specific and worth stating: if stage 2 came down to the ~2.2×
complex/float ratio the panel reached after its own fix, the two stage-2 kernels fall from
1353 ms to 704 ms — **1.49×** on the complex two-stage solve, ≈650 µs/matrix against the
973.8 baseline. That is ahead of both blocked (698) and the vendor (707), which would open
n ≥ 512 cfloat eigenvectors to two-stage and close **the one region where we still lose to
cuSOLVER**. Any routing change is contingent on measuring that, not on predicting it.

---

## WP7 — Measurement debt

**A4 — re-measure the n = 2048 row.** The only place float loses badly (1.65×) and the only
row in the routing table taken at batch 64, i.e. unsaturated. Both `blocked` and
`two_stage` carry larger workspaces than the vendor, so this may be a capacity artifact
rather than a verdict. **Do this after WP1**, which removes `2·n²·batch` of scratch in
values mode and part of it in vector mode — that alone may change the answer. Measure at
the largest batch that fits, and report the batch, not just the ratio.

**A5 — raise the grid residency cap.** `latrd_lower_panel.cc:1155-1158`. The cap exists
because the software grid barrier deadlocks unless all participating work-groups are
co-resident, and "total work-groups ≤ SM count" is the conservative guarantee. These are
≤ 256-thread groups with a modest local footprint, so several are resident per SM; the cap
could be `SMs × achievable_blocks_per_SM`, computed from the kernel's actual local-memory
and register footprint rather than assumed. That would let G ≥ 2 up to batch 256–512.

*Honest expectation: this probably does not pay by itself.* The grid path's known mechanism
is curing batch starvation, and there is no starvation at batch ≥ 128. Its value is that it
makes the L2-residency question answerable. WP0.2 gives the unsafe override for
measurement; A5 is the safe version, and it should only be built if the override shows
there is something to win.

**B4 — profile stedc internally.** Every merge kernel is one work-group per matrix:
`stedc.cc:148, 184, 318, 349, 395, 422` are `nd_range<1>(batch_size*128, 128)`, and `:712`
likewise. At large batch that is not starvation, so the grid-barrier treatment that fixed
`latrd` is not obviously right here. stedc is 20.7% of the float eigenvector solve at
n = 256 and has never been profiled internally. **Profile before designing** — and note
that WP1 removes stedc from the values path entirely, so B4's remaining scope is
eigenvector mode only.

---

## WP8 — C1: block Jacobi at n = 64–256

Kept last deliberately: highest ceiling among the conventional-alternative items, entirely
unmeasured, and it should only start once WP1–WP6 have bounded what the current path can
still give.

**The case.** We run at 1.65–2.1 TFLOP/s, ~3.5% of the ~47 TFLOP/s cuBLAS SGEMM sustains on
this card. Block Jacobi costs ~8–10 sweeps × ~4n³ ≈ 30–40n³ against our ~4n³ — a 10× flop
premium — so it breaks even at 17–21 TFLOP/s, and a good GEMM-based block rotation should
reach 30–40. The repo's own strongest evidence points the same way: `gesvdj_cta` (one-sided
Jacobi, the SVD analogue) beats the tridiagonalizing CTA path by 4.1× at n = 16 and 23× at
n = 8 with vectors, at *better* accuracy, and was just extended to n = 64.

**Note the existing plan.** `JACOBI_EIGENSOLVER_PLAN.md` in the working tree already covers
this ground in depth — 18 sources, 25 claims adversarially verified — and records **Tier A
as implemented** (`src/extensions/syev_jacobi_cta.cc`), with Tiers B and C design-only.
That document, not this one, is the starting point for WP8; its framing is also the right
one to keep, namely that Jacobi is a *second* backend for accuracy-critical and graded
input rather than a replacement. Start with the cost model and a single-shape prototype,
and compose it with WP2 — both want the same shared-memory residency budget.

**Not pursued** (from §5, recorded so they are not re-derived): C5 spectral D&C (the
cluster-parallelism argument does not apply at saturating batch) and C6 real embedding of
the Hermitian problem (2× worse in flops, and it doubles memory).

---

## Cross-cutting notes

**Routing is measured, not derived, and this plan invalidates parts of it.** WP1, WP5 and
WP6 each move a boundary in `include/blas/functions/syev.hh`. Re-run the routing grid
**once**, after WP5, rather than after each — and re-run it per scalar type and per job
mode. The whole class of defect fixed in PR #65 was a constant measured on float and
applied to every type; the same mistake is available here at every step.

**Order of accuracy checks.** WP1 changes which algorithm computes eigenvalues, WP3 changes
the arithmetic of a trailing update, and WP8 changes the algorithm outright. Those three
need accuracy runs (`eigensolver_accuracy`, `syev_blocked_acc`), not just correctness
tests. WP4, WP5 and WP6 are geometry and scheduling changes that must be bit-neutral or
near it — if accuracy moves there, something is wrong rather than merely different.

**Test scope.** Do not run the full `ctest` by default (`selective-testing-policy`); scope
with `-R syev|sytrd|stedc` and `-LE slow`, and save the full suite for pre-push. The
baseline is not green — `stedc_flat` is a known recurring failure and is being deprecated.

## Reproducing / measuring

```bash
# end to end, one shape (n, batch, nb=0 -> shipped default, fuse, jobz, uplo)
CUDA_VISIBLE_DEVICES=1 build/benchmarks/syev_benchmark --backend=CUDA --type=float,cfloat \
    --warmup=2 --min_iters=5 512 512 0 0 1 0

# force a provider / an implementation
BATCHLAS_SYEV_PROVIDER=blocked|two_stage|vendor|cta
BATCHLAS_LATRD_IMPL=legacy|grid          BATCHLAS_LATRD_GRID_MIN_N=<n>
BATCHLAS_SB2ST_BACK_TILE_W=<1,2,4,8>     BATCHLAS_SB2ST_BACK_SUBS=<4,8,16>
BATCHLAS_EXPAND_MAX_BYTES=<bytes>        # WP3: force the her2k host-loop fallback

# kernel attribution (sees cuBLAS; the SYCL trace does not)
nsys profile -t cuda -s none -o out build/benchmarks/syev_benchmark ...
nsys stats --report cuda_gpu_kern_sum --format csv out.nsys-rep

# hardware counters on one kernel
ncu -k regex:LatrdLowerPanel -c 2 --metrics dram__bytes_read.sum,lts__t_bytes.sum,\
l1tex__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active --csv \
    build/benchmarks/latrd_lower_panel_benchmark --backend=CUDA --type=float 512 256 32 0 0
```

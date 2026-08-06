# GESVD implementation spec

Executable companion to `GESVD_PLAN.md`. The plan says *what* and *why*; this
says *where*, *in what order*, and *exactly what code shape*. Every claim below
is cited to `file:line` in this worktree
(`/home/jonaslacour/BatchLAS/.claude/worktrees/gesvd-batched-plan`) at
`c76a01f` plus the uncommitted working tree.

**Read this first — the state of the tree is not what `GESVD_PLAN.md` describes.**
`git status` shows uncommitted work that already lands most of Tier 0 and the
accuracy harness:

| File | State |
|---|---|
| `include/blas/functions/gesvd.hh` | modified — `backend::gesvd_vendor` is now **declaration-only** (`include/blas/functions/gesvd.hh:183`), with a `namespace sig` block at `:29-45` |
| `src/backends/cusolver.cc` | modified — `gesvdjBatched` route implemented (`src/backends/cusolver.cc:293-461`) |
| `src/backends/netlib_lapack.cc` | modified — LAPACKE body moved here (`src/backends/netlib_lapack.cc:838`) + instantiation (`:1263`) |
| `src/backends/rocsolver.cc` | modified — throwing stub + instantiation (`src/backends/rocsolver.cc:354-381`) |
| `benchmarks/gesvd_vendor_benchmark.cc` | new, untracked — head-to-head perf, 2 arms |
| `benchmarks/gesvd_relacc.cc` | new, untracked — relative-accuracy harness, 2 arms |
| `benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv` | new, untracked — measured head-to-head |

So Parts A and B below are **mostly audit + finish**, not greenfield. Part C is
the real work.

### The measured head-to-head, which reframes Tier 1

From `benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv`, float, n=32, batch=16384,
`avg_ms`:

| jobu/jobvh | `gesvd_cta` | `gesvdjBatched` | ratio |
|---|---|---|---|
| None/None | 3.230 | 12.148 | **3.8x faster** |
| None/All  | 4.608 | 12.654 | **2.7x** |
| All/None  | 6.920 | 12.245 | **1.8x** |
| All/All   | 7.691 | 12.680 | **1.6x** |

**We already win on speed and lose on accuracy.** That inverts the framing in
`GESVD_PLAN.md` §7 risk 1. The bar for `gesvdj_cta` is therefore not
"beat `gesvdjBatched`" — that is already done — it is:

> **fix Defect A without regressing below `gesvd_cta`.**

Concretely: `gesvdj_cta` must land under **7.69 ms** at float/n=32/batch=16384/All/All
to be a strict improvement, and under **12.68 ms** to still beat the vendor. The
design in Part C projects 5.5-8 ms into that window; the *accuracy* gain is the
deliverable, and any regression against `gesvd_cta` must be paid for by a
measured relative-accuracy curve, not asserted.

---

# Part A — Tier 0: cuSOLVER `gesvd_vendor`

## A.1 The mechanism question — resolved, and already applied

The header used to **define** the primary template `backend::gesvd_vendor` with a
throwing body for every non-NETLIB backend. That made a definition in
`cusolver.cc` a hard redefinition error (the header reaches that TU via
`src/backends/cusolver.cc:9` → `blas/linalg.hh` → `blas/functions.hh:16`), and
function templates cannot be partially specialised on `Backend B`.

The house idiom — what `syev_vendor` (`include/blas/functions/syev.hh:82`) and
`ormqr_vendor` (`include/blas/functions/ormqr.hh:119`) do — is: **the header
declares only**, and each backend TU defines the primary template inside
`namespace backend` and explicitly instantiates it for its own `Backend` value.
Three TUs define the same primary template; it links because each instantiates
only its own value and the header never implicitly instantiates.

That conversion is **done**:

* `include/blas/functions/gesvd.hh:168-201` is now declaration-only, with the
  comment naming the mechanism.
* `include/blas/functions/gesvd.hh:29-45` adds the missing
  `namespace sig { gesvd_vendor, gesvd_vendor_buffer_size }` aliases required by
  `BATCHLAS_INSTANTIATE` (`src/util/template-instantiations.hh:31`).
* The LAPACKE body moved **verbatim** to `src/backends/netlib_lapack.cc:838`
  (with the "moved verbatim from" comment at `:832`) and is instantiated at
  `src/backends/netlib_lapack.cc:1263-1264` via the `sig::` macro form. This was
  mandatory: `tests/gesvd_tests.cc` builds `Config<T, Backend::NETLIB>` for all
  four scalar types, and `gesvd` is `inline` with no explicit instantiation, so a
  missing NETLIB symbol is a **test link error**, not a library one.
* `src/backends/rocsolver.cc:354-373` supplies a throwing stub + instantiation at
  `:379-381`, replacing what the header-throw used to give for free.

**Nothing to do here except not undo it.** If you touch `gesvd.hh`, keep it
declaration-only.

## A.2 What the cuSOLVER binding does today

`src/backends/cusolver.cc:361-461` (call) and `:293-357` (sizing). Shape:

```
op_external("cusolver.gesvd_vendor", [&]{
    static LinalgHandle<B> handle;        // cusolver.cc:395
    handle.setStream(ctx);                //   :396  -- binds the SYCL stream
    BumpAllocator pool(workspace);        //   :397
    ... bufferSize query, allocate, solve ...
    return ctx.create_event_after_external_work();   // :454
});
```

`create_event_after_external_work()` (`src/util/queue-impl.cc:248`) is required,
not `get_event()`: external library calls never update the queue's cached
`last_event_`.

**Argument order, read from
`/usr/local/cuda-13.3/targets/x86_64-linux/include/cusolverDn.h:4014` (S) and
`:3950` (S bufferSize):**

```c
cusolverDnSgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const float *A, int lda, const float *S,
    const float *U, int ldu, const float *V, int ldv,
    int *lwork, gesvdjInfo_t params, int batchSize);

cusolverDnSgesvdjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    float *A, int lda, float *S,
    float *U, int ldu, float *V, int ldv,
    float *work, int lwork, int *info,
    gesvdjInfo_t params, int batchSize);
```

The complex forms take a **real** `S` and complex `A/U/V/work`
(`cusolverDn.h:4050` for C, `:4068` for Z) — that is why
`base_float_ptr_convert` (`src/linalg-impl.hh:513`) is applied to the singular
value pointer at `src/backends/cusolver.cc:337` and `:418`. There are **no
stride arguments**: A, U and V are each implicitly tightly packed.

Four facts the binding encodes and you must not regress:

1. **`SvdVectors` is not in `is_linalg_enum`** (`src/linalg-impl.hh:63` lists
   only Side/Uplo/Transpose/Diag/Layout/JobType). It is mapped by hand to
   `CUSOLVER_EIG_MODE_VECTOR` / `NOVECTOR` at `src/backends/cusolver.cc:407-408`.
2. **cuSOLVER returns V, BatchLAS's contract is Vh.** The out-of-place
   conjugate-transpose kernel is `gesvd_detail::write_vh_from_v`
   (`src/backends/cusolver.cc:250-269`); it is called at `:455`, and the returned
   `Event` is switched to `ctx.get_event()` at `:456` because that last stage is
   a SYCL kernel, not external work. An in-place transpose would race.
3. **Tight packing is checked on every view.** `gesvd_detail::packed`
   (`src/backends/cusolver.cc:277`) is the `stride() == ld() * cols()` guard,
   copied from the `syev_vendor` precedent at `src/backends/cusolver.cc:86`;
   `batched_route_ok` (`:283`) additionally caps both dims at 32.
4. **The two-call workspace contract is exact.** `gesvd_dispatch` asks
   `gesvd_vendor_buffer_size` for a byte count
   (`include/blas/functions/gesvd.hh:311`), throws
   `"gesvd: insufficient workspace for chosen provider"` at `:323`, then hands the
   **whole, untrimmed** span to `gesvd_vendor` (`:342`). The sizing side therefore
   re-runs the same `bufferSize` query and sums
   `BumpAllocator::allocation_size<T>(ctx, n)` **once per allocation the call side
   makes** (`src/backends/cusolver.cc:342-354`), in the same order as the
   allocations at `:422-438`. Summing a raw byte total under-provisions in two
   independent ways (`include/util/mempool.hh:80-113`).

`gesvdjInfo_t` lifecycle: `cusolverDnCreateGesvdjInfo` (`cusolverDn.h:3927`) →
`cusolverDnXgesvdjSetSortEig(params, 1)` (`:3938`, used at
`src/backends/cusolver.cc:405` so cuSOLVER's own sort produces the descending
order BatchLAS requires) → the two calls → `cusolverDnDestroyGesvdjInfo`
(`:3929`). Note the destroy leaks if `check_status` throws in between; that
matches the existing `syev_vendor` behaviour (`src/backends/cusolver.cc:155/163`)
and `op_external` is a bare tag, not a scope guard
(`include/blas/dispatch/op.hh:11`).

## A.3 What is still missing — the remaining Tier 0 work

### A.3.1 Empirically establish the 32x32 limit (blocking, ~30 min)

`kGesvdjBatchedMaxDim = 32` at `src/backends/cusolver.cc:281` is currently taken
from the programming guide. `GESVD_PLAN.md` §Tier 0 step 3 requires this be
measured, because it defines the boundary of the contested region.

Write a throwaway probe (do **not** commit it as a benchmark target — see F.1 on
per-TU compile cost). Call `cusolverDnSgesvdjBatched_bufferSize` directly at
`m = n = 32, 33, 48, 64` with `batchSize = 2` and record the raw
`cusolverStatus_t` **without** routing it through `check_status`
(`src/linalg-impl.hh:602`), which discards the code into a string. Record the
result as a comment next to `kGesvdjBatchedMaxDim`. Expect
`CUSOLVER_STATUS_INVALID_VALUE` above 32; if it instead succeeds, raise the
constant and re-run the head-to-head at n=48/64 before anything else, because
that would move the whole contested region.

### A.3.2 Route the shapes the batched call cannot take

Today both entry points throw for anything outside the batched route
(`src/backends/cusolver.cc:303-311` and `:370-374`), with a comment explaining
that silently substituting a looped `gesvdj` would corrupt the comparison. That
was the right call for Tier 0; it is the wrong long-term API. Add the two
remaining routes **behind an explicit label**, so a benchmark can never print an
approximate or looped result under the exact result's name:

Add to `gesvd_detail`:

```cpp
enum class VendorRoute { GesvdjBatched, GesvdaStrided, GesvdjLooped };

template <typename T>
inline VendorRoute route_for(const MatrixView<T, MatrixFormat::Dense>& A) {
    if (A.rows() <= kGesvdjBatchedMaxDim && A.cols() <= kGesvdjBatchedMaxDim && packed(A))
        return VendorRoute::GesvdjBatched;
    if (A.rows() >= A.cols()) return VendorRoute::GesvdaStrided;   // APPROXIMATE
    return VendorRoute::GesvdjLooped;
}
```

* **`gesvdaStridedBatched`** — `cusolverDn.h:4304` (call), `:4224`
  (bufferSize). Unlike `gesvdjBatched` it *does* take strides, so no packing
  guard is needed; it takes a `rank` argument (pass `min(m,n)` for a full
  decomposition) and an extra **host** output `double* h_R_nrmF` of length
  `batchSize` (`cusolverDn.h:4324`). It is restricted to `m >= n` and is an
  **approximate** solver. It must be labelled: expose
  `bool gesvd_vendor_is_approximate(...)` alongside, or (preferred) set a
  `state.SetTag("vendor_route", "gesvda_approx")` in the benchmark. Never let it
  share a benchmark name with the exact routes — `--name` is a substring filter
  (`include/util/minibench.hh:591`) and the two will be conflated.
* **looped `gesvdj`** — `cusolverDn.h:4150` (call), `:4086` (bufferSize). Note
  the extra `int econ` argument in position 3, which the batched form does not
  have. `SvdVectors` has no economy mode (`include/blas/enums.hh:89`), so pass
  `econ = 0`. Loop over `A.batch_size()` advancing by `A.stride()`; one
  `bufferSize` query suffices since every item has the same shape.

Both routes still need the V→Vh transpose; reuse `write_vh_from_v`
(`src/backends/cusolver.cc:250`) unchanged — it already takes a raw `const T*`
with `ld == n, stride == n*n`, which is what both routes produce.

Mirror **every** new allocation in `gesvd_vendor_buffer_size`. Because
`allocation_size<T>(ctx, 0) == 0` (`include/util/mempool.hh:69`), you can sum all
three routes' terms unconditionally and let the untaken ones contribute zero —
that is exactly what `syev_vendor_buffer_size` does at
`src/backends/cusolver.cc:220-223`.

### A.3.3 Match the tunables and report sweeps

`GESVD_PLAN.md` §6 requires the comparison match tolerance, `max_sweeps` and sort
order, and report cuSOLVER's executed sweeps. Today only `SetSortEig` is called
(`src/backends/cusolver.cc:405`).

Add an optional `GesvdVendorParams { double tolerance = 0; int max_sweeps = 0; }`
that, when non-zero, calls `cusolverDnXgesvdjSetTolerance` (`cusolverDn.h:3932`)
and `cusolverDnXgesvdjSetMaxSweeps` (`:3935`) before the solve. Defaults of 0
mean "leave cuSOLVER's default", so the existing measurements stay valid.

`cusolverDnXgesvdjGetSweeps` (`cusolverDn.h:3945`) reports the executed sweeps
for the *last* solve on that `gesvdjInfo_t`. Do **not** call it from
`gesvd_vendor` — it is a host-blocking read in the middle of an async path.
Expose it instead as a separate `size_t gesvd_vendor_last_sweeps(Queue&)` used
only by `benchmarks/gesvd_relacc.cc`, which already synchronises
(`benchmarks/gesvd_relacc.cc:194`).

---

# Part B — the accuracy harness

## B.1 What already exists, and why it is the model to copy

`benchmarks/gesvd_relacc.cc` (250 lines, untracked) fixes Defect C. Its header
comment (`benchmarks/gesvd_relacc.cc:1-24`) states all three failure modes of the
old harness; it is the reference implementation for the two mechanisms the task
asks about.

**Mechanism 1 — plumbing `--log10-cond`.** Three things must line up, and the
old `benchmarks/gesvd_cta_acc.cc` misses all three:

1. **Read the target.** `benchmarks/gesvd_relacc.cc:147-149`:
   ```cpp
   const double target_log10_raw = state.target_log10_cond();
   const double target_log10 = std::isfinite(target_log10_raw) ? target_log10_raw : 1.0;
   ```
   The default with no flag is NaN (`include/util/miniacc.hh:590`), and
   `format_value` renders NaN as the empty string (`:388`) — that is the blank
   `log10cond` column.
2. **Feed it to the generator.** `benchmarks/gesvd_relacc.cc:166-167`:
   ```cpp
   auto A = random_with_log10_cond_metric<B, Real>(
       *q, n, static_cast<Real>(target_log10), NormType::Spectral, cur_batch, seed);
   ```
   `NormType::Spectral`, not `Frobenius`: `build_spectrum_kappaF` throws when
   `log10_kappaF < log10(n)` (`src/extra/random_cond.cc:98`), so a sweep starting
   near κ=10 dies on the Frobenius path.
3. **Record a metric literally named `"log10_cond"`.**
   `benchmarks/gesvd_relacc.cc:226`. The printed cell is
   `mean(metric "log10_cond")` when finite, else the target
   (`include/util/miniacc.hh:497`), so recording it also survives the
   no-flag case. The failure path records it too
   (`benchmarks/gesvd_relacc.cc:204`) so a throwing case does not blank the
   column.

**Mechanism 2 — emitting numbers instead of only `Fail%`.** miniacc's terminal
summary prints only a hard-coded whitelist of metric names
(`include/util/miniacc.hh:425`); anything else reaches only `--csv`. That is why
`gesvd_cta_acc.cc`'s `u_ortho` / `vh_ortho` / `recon_rel` / `sv_max_abs_err` are
invisible. `gesvd_relacc.cc:223-226` chooses whitelisted names deliberately:

| metric | meaning | column |
|---|---|---|
| `max_relerr` | `max_i \|σ_i − σ_i^ref\| / σ_i^ref` | `max_relerr` |
| `R` | `‖A − U S V^H‖_F / ‖A‖_F` | `R` |
| `O` | `max(‖U^H U − I‖, ‖V V^H − I‖)` | `O` |
| `log10_cond` | conditioning | `log10cond` |

**The reference is exact, not another solve.** `random_with_log10_cond_metric`
builds `A = U·S·V^H` from an explicitly constructed geometric spectrum
(`src/extra/random_cond.cc:46`, `:276`), so the true singular values are known in
closed form. `reference_spectrum_descending`
(`benchmarks/gesvd_relacc.cc:57-70`) reconstructs that spectrum on the host and
**reverses it** to descending, matching the BatchLAS output convention.
Comparing against a second float solve would put the reference at the same error
floor as the thing being measured (`GESVD_PLAN.md` §5.3).

## B.2 What is still missing

1. **The `ok` predicate is only a finiteness check**
   (`benchmarks/gesvd_relacc.cc:222`). That is defensible today — the point is
   the curve, not a pass/fail — but once Tier 1 lands, `Fail%` should mean
   something. Add `GesvdRelAccThresholds` with `max_relerr <= C * eps * kappa`,
   `R <= C * n * eps`, `O <= C * n * eps`, `C` a CLI-overridable constant.
   Until then, always run with `--csv=` (per-sample rows are retained *only*
   when `--csv` is passed, `include/util/miniacc.hh:595`).
2. **Graded matrices** (`GESVD_PLAN.md` §5.4). `random_with_log10_cond_metric`
   produces `A = U S V^H` with random orthogonal factors, for which
   `kappa(A_c) ≈ kappa(A)` — precisely the case where one-sided Jacobi has the
   *least* advantage. Add a second generator arm producing `A = D · G` with `D`
   a diagonal of geometrically spread scales and `G` well-conditioned random, so
   `kappa(A_c) << kappa(A)`. Reference σ are no longer closed-form there, so
   compute them with a **double** LAPACKE solve on the host — reuse
   `lapacke_gesvd_values_only_any<Scalar>` (`tests/gesvd_tests.cc:150-217`),
   which already handles all four types. Tag the arm
   `state.SetTag("input", "graded")`.
3. **Sweep counts.** Record `{"sweeps", ...}` (CSV-only; not on the whitelist)
   for both the BatchLAS kernel (Part C.9) and cuSOLVER (Part A.3.3).
   `GESVD_PLAN.md` §7 risk 2 asks for the *distribution*, so read it from the CSV,
   not from the printed mean.
4. **A third arm for `gesvdj_cta`.** `benchmarks/gesvd_relacc.cc:139` already
   has `enum class RelAccImpl { BatchlasCta, CusolverJacobi }` and dispatches on
   it at `:184-193`. Add `BatchlasJacobi` and a third
   `BATCHLAS_ACC_CUDA(ACC_GESVD_RELACC_JACOBI, GesvdRelAccSizes)` next to
   `:256-257`. **Extend this TU rather than adding a new one** — benchmarks are
   `EXCLUDE_FROM_ALL` with a ~12 s per-TU compile floor
   (`benchmarks/CMakeLists.txt:74`).
5. **Do not "fix" `gesvd_cta_acc.cc` / `gesvd_blocked_acc.cc`.**
   `gesvd_relacc.cc` supersedes them. Leave them; delete them in the same commit
   that tightens `tests/gesvd_tests.cc` (Part E.3).

Invocation (note miniacc uses `--benchmark_filter=GLOB`, not minibench's
substring `--name`, `include/util/miniacc.hh:326`):

```
./build/benchmarks/gesvd_relacc --samples=512 --log10-cond=1:7:13 \
    --benchmark_filter='*RELACC*' --csv=relacc_float.csv 32
```

---

# Part C — Tier 1: `gesvdj_cta`

## C.0 The design decision

Three candidate mappings were reviewed by three independent judges. **All three
judges ranked Candidate 2 first, and all three disqualified Candidate 3.** The
chosen design is **Candidate 2 — lane = row, one warp per problem,
round-batched reduce-scatter Gram** — with mandatory grafts from Candidate 1
listed in C.10.

**Why lane = row.** `syev_jacobi_cta`'s Phase 1 (`src/extensions/syev_jacobi_cta.cc:413-450`)
already uses lane = row, precisely because both operands of a rotation then live
in the *same* lane. Under lane = column (Candidate 1) the update must re-fetch
the partner element with one shuffle **per matrix element** — 64 shuffles per
round for A and V — which is larger than the traffic it saves. Recounted
per-round LDS+shuffle cost at m=n=32/P=32/float/vectors, against
`syev_jacobi_cta`'s 266 (verified from `src/extensions/syev_jacobi_cta.cc:326-503`):

| design | per-round | ratio vs syev |
|---|---|---|
| Candidate 2 (chosen) | ~181 | **0.68** |
| Candidate 1 (lane = column) | ~251-264 | 0.94-1.00 |
| Candidate 3 (Gram-resident) | 334 + refresh | 1.26-1.39 |

Two further structural advantages of lane = row that decided it:

* **`ap[k]/aq[k]` survive from the Gram phase into the update phase**, saving 32
  LDS loads per round that no other mapping gets.
* **Converged pairs are genuinely free.** The `if (sk == 0) continue;` skip is
  *warp-uniform* (every lane reads the same `Rcs_local` slot, exactly as
  `src/extensions/syev_jacobi_cta.cc:418`). Under lane = column the skip
  predicate is lane-varying, so a round with 3 of 16 pairs live still issues the
  full update under predication. Jacobi's last three or four sweeps are almost
  entirely partial rounds; this is worth a further 10-20% of wall clock.

**Why Candidate 3 is rejected outright.** Maintaining `G = A^H A` by congruence
accumulates *normwise* error `~eps·σ_max²`, not the columnwise-relative error a
freshly computed Gram has. The relative threshold at
`src/extensions/syev_jacobi_cta.cc:351` then degenerates into an absolute test
for every pair with `σ_p σ_q << σ_max²` — the exact substitution the reference
kernel's own comment (`src/extensions/syev_jacobi_cta.cc:233-237`) says forfeits
the point of the kernel, and structurally Defect A one level removed.
Independently, `complex<double>` with vectors needs
`3·1056·16 + 256 + 256 = 51,200` B per problem plus the 992 B pair table against
the 49,152 B that `local_mem_size` reports (the query at
`src/extensions/syev_jacobi_cta.cc:187`) — a hard launch failure for a type
`GESVD_PLAN.md` Tier 4 puts in scope. `GESVD_PLAN.md:287-288` already gates it
behind "(a) or (b) correct and measured"; it stays gated.

## C.1 File location and build group

**File:** `src/extensions/gesvdj_cta.cc`.

**Add to `EXTENSIONS_CTA_SOURCES`** (`src/extensions/CMakeLists.txt:1-10`),
after `syev_jacobi_cta.cc`. Two independent reasons:

1. `batchlas_extensions_cta_obj` is the **only** object library built with
   `NO_CPU_TARGETS` (`src/CMakeLists.txt:59-65`), i.e. `-fsycl-targets` with
   cpu/spir64 filtered out (`cmake/BatchLASDetectSYCL.cmake:453-462`). A 32-lane
   sub-group kernel must not be AOT-compiled for the CPU target.
2. The device-link unit is the shared library
   (`add_library(batchlas_extensions_cta SHARED ...)`, `src/CMakeLists.txt:146`),
   and device symbols resolve only across TUs *within* a library
   (`src/extensions/CMakeLists.txt:46-55`).

**The failure mode is asymmetric and there is no safety net.** Putting the file
in `EXTENSIONS_FACTORIZATION_SOURCES` next to `gesvd_blocked.cc` produces **no
error** — it just silently acquires CPU AOT compilation. The `ptxas fatal:
Unresolved extern function` net only catches genuine cross-TU device symbols, and
the only ones in the tree are `stedc_secular`'s two `SYCL_EXTERNAL` functions
(`src/extensions/stedc_secular.hh:9-19`); all CTA device sharing is header-inline
(`src/extensions/steqr_cta_device.hh:1-11`).

**Include set** — copy `src/extensions/syev_jacobi_cta.cc:1-12` verbatim:
`blas/matrix.hh`, `blas/functions.hh`, `blas/extensions.hh`, `blas/extra.hh`,
`util/kernel-heuristics.hh`, `util/mempool.hh`, `util/group-invoke.hh`,
`"sg_compat.hh"`, `batchlas/backend_config.h`, `"../math-helpers.hh"`,
`"../queue.hh"`, `"../util/template-instantiations.hh"`.

**Kernel name tag must live outside the anonymous namespace**
(`src/extensions/syev_jacobi_cta.cc:20-23`), so it does not depend on
internal-linkage entities:

```cpp
template <typename T, size_t P, bool ComputeV> class GesvdjCTAKernel;
```

## C.2 Public API

Declare in `namespace batchlas` in `include/blas/extensions.hh`, immediately
after the `gesvd_cta` block that ends at `:2050`:

```cpp
    template <typename T>
    struct GesvdjParams {
        using Real = typename base_type<T>::type;

        // |a_pq| > tol_multiplier * n * eps * sqrt(|a_pp| * |a_qq|)  -- the
        // RELATIVE test. See JacobiParams (extensions.hh:1706-1712).
        Real   tol_multiplier = Real(1);
        size_t max_sweeps     = 30;

        // Descending, unconditionally -- see C.7. Deliberately NOT a SortOrder
        // field: gesvd's contract admits exactly one order and an Ascending
        // setting would silently return reversed output.

        size_t cta_wg_size_multiplier = 1;

        // sigma_j <= zero_sigma_multiplier * eps * sigma_max  =>  U_j is not
        // determined by A and is filled from the orthogonal complement (C.8).
        Real   zero_sigma_multiplier = Real(1);

        // de Rijk pre-ordering. OFF until measured to save >= 1 sweep (C.10.6).
        bool   derijk = false;

        // Recompute a column norm from A mid-sweep when the analytic update
        // shrinks it by more than this factor (C.5 step R5).
        Real   drift_refresh_ratio = Real(1) / Real(256);
    };

    template <Backend B, typename T>
    Event gesvdj_cta(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                     Span<typename base_type<T>::type> singular_values,
                     const MatrixView<T, MatrixFormat::Dense>& u_out,
                     const MatrixView<T, MatrixFormat::Dense>& vh_out,
                     SvdVectors jobu,
                     SvdVectors jobvh,
                     const Span<std::byte>& ws = Span<std::byte>(),
                     GesvdjParams<T> params = GesvdjParams<T>());

    template <Backend B, typename T>
    size_t gesvdj_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                  Span<typename base_type<T>::type> singular_values,
                                  const MatrixView<T, MatrixFormat::Dense>& u_out,
                                  const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                  SvdVectors jobu,
                                  SvdVectors jobvh,
                                  GesvdjParams<T> params = GesvdjParams<T>());
```

**New `GesvdjParams`, not `JacobiParams` reuse.** `JacobiParams::sort_order`
defaults to `SortOrder::Ascending` (`include/blas/extensions.hh:1719`) while
gesvd's contract is descending; reusing it invites the silent-reversal bug, and
changing its default would alter `syev_jacobi_cta`'s behaviour. The two structs
share the `tol_multiplier` / `max_sweeps` / `cta_wg_size_multiplier` semantics
verbatim — keep the field names identical so the comment at
`include/blas/extensions.hh:1706-1712` remains the single explanation.

**Add the queue-deducing overloads** in the block at
`include/blas/extensions.hh:2290-2371`, next to the existing gesvd lines at
`:2361-2362`:

```cpp
BATCHLAS_DISPATCH_ON_QUEUE(gesvdj_cta)
BATCHLAS_DISPATCH_ON_QUEUE(gesvdj_cta_buffer_size)
```

The declarations must precede that block, because the macro's `requires`-clause
names the function (`include/blas/queue-dispatch.hh:103-113`). Do not remove the
`requires` clause: an unconstrained variadic pack beats every specific overload
and then fails inside its own body.

**No Hermitian overload.** One-sided Jacobi has no use for the symmetry
shortcut, and adding a second overload with the same arity + convertibility as
the positional one is the option-struct trap.

**Workspace: `gesvdj_cta_buffer_size` returns 0** and the impl does `(void)ws;`
— everything is LDS-resident for the lifetime of the kernel, exactly as
`syev_jacobi_cta` (`src/extensions/syev_jacobi_cta.cc:620-636`). Keep the
argument: the dispatcher passes one.

## C.3 Launch geometry

`P` is compile-time in `{4,8,16,32}`, selected by the ladder at
`src/extensions/syev_jacobi_cta.cc:606-614`, but keyed on **`max(m,n)`**, not
`n`: lane = row means rows must fit in the partition, and the pair-slot phase
uses `P/2` lanes for columns.

```cpp
auto launch = [&](auto P_tag) {
    constexpr size_t P = decltype(P_tag)::value;
    if (want_vh) gesvdj_cta_impl<T, P, true >(ctx, a, s_ptr, u_out, vh_out, jobu, m, n, params);
    else         gesvdj_cta_impl<T, P, false>(ctx, a, s_ptr, u_out, vh_out, jobu, m, n, params);
};
const int md = std::max(m, n);
if      (md <= 4)  launch(std::integral_constant<size_t, 4>{});
else if (md <= 8)  launch(std::integral_constant<size_t, 8>{});
else if (md <= 16) launch(std::integral_constant<size_t, 16>{});
else               launch(std::integral_constant<size_t, 32>{});
```

Two instantiations per `(T, P)`, not four: `ComputeV` is the only compile-time
flag. `jobu` is a **runtime** bool — `U_j = A_j / σ_j` is a normalisation of a
tile that is resident regardless, so `jobu` changes only the epilogue. Total
kernels: 4 types × 4 P × 2 = **32**, matching `syev_jacobi_cta`'s count.

Preconditions, checked in the public entry before any launch:

* `validate_gesvd_dims` semantics (`src/extensions/gesvd_blocked.cc:74-99`):
  `batch >= 1`, `rows >= 1`, `cols >= 1`,
  `singular_values.size() >= min(m,n)*batch`; `U` exactly `m x m` when
  `jobu == All`; `Vh` exactly `n x n` when `jobvh == All`. The `None` side is not
  validated at all and may be a dummy view. Call the existing helper if you
  hoist it to a header; otherwise replicate it exactly — the error strings are
  asserted by `tests/gesvd_tests.cc`.
* `max(m,n) <= 32`, mirroring `src/extensions/gesvd_blocked.cc:1232`.
* Device reports 32 in `sub_group_sizes`, mirroring
  `src/extensions/syev_jacobi_cta.cc:573-587`. `sg_size` is hard-coded to 32
  (`:155`).
* `ctx.in_order()` is **not** required — this is a single launch. Do not copy
  the `": requires an in-order Queue"` guard from
  `src/extensions/gesvd_blocked.cc:651`.

Return `ctx.get_event()` (`src/extensions/syev_jacobi_cta.cc:615`) — native
kernel, not external work.

The input view arrives `const`; `const_cast` it as at
`src/extensions/syev_jacobi_cta.cc:589`. **A is destroyed** (it becomes the
rotated matrix). This is the existing gesvd contract and is why benchmarks must
wrap it in `bench::pristine` (`include/util/bench_structured.hh:117`).

## C.4 LDS layout and the exact budget formula

```cpp
using Real = typename base_type<T>::type;
constexpr int32_t LD          = static_cast<int32_t>(P) + 1;        // 33 at P=32; ODD
constexpr size_t  kTileElems  = static_cast<size_t>(LD) * P;        // 1056
constexpr size_t  kRotSlots   = (P / 2 > 0) ? (P / 2) : 1;          // 16
constexpr size_t  kPairSlots  = (P - 1) * kRotSlots;                // 496
constexpr size_t  kPairWords  = (kPairSlots + 1) / 2;               // 248 uint32 words
constexpr bool    kNeedPhase  = internal::is_complex<T>::value;

auto A_local    = sycl::local_accessor<T,1>(               sycl::range<1>(probs_per_wg * kTileElems), cgh);
auto V_local    = sycl::local_accessor<T,1>(               sycl::range<1>(ComputeV ? probs_per_wg * kTileElems : 1), cgh);
auto Nrm_local  = sycl::local_accessor<Real,1>(            sycl::range<1>(probs_per_wg * P), cgh);
auto Rcs_local  = sycl::local_accessor<sycl::vec<Real,2>,1>(sycl::range<1>(probs_per_wg * kRotSlots), cgh);
auto Rd_local   = sycl::local_accessor<T,1>(               sycl::range<1>(kNeedPhase ? probs_per_wg * kRotSlots : 1), cgh);
auto Rank_local = sycl::local_accessor<int16_t,1>(         sycl::range<1>(probs_per_wg * P), cgh);   // column -> rank
auto Inv_local  = sycl::local_accessor<int16_t,1>(         sycl::range<1>(probs_per_wg * P), cgh);   // rank   -> column
auto Pair_local = sycl::local_accessor<uint32_t,1>(        sycl::range<1>(kPairWords), cgh);         // WORK-GROUP shared
```

Conditionally-unused arrays are sized **1, never 0**, and their base forced to 0
— the trap avoided at `src/extensions/syev_jacobi_cta.cc:207-208,216-219,289`.

Per-problem bases: `base_a = part_id*kTileElems`,
`base_v = ComputeV ? part_id*kTileElems : 0`, `base_n = part_id*P`,
`base_r = part_id*kRotSlots`, `base_p = part_id*P`.

**Why `LD = P+1`.** Under lane = row every hot access is `base + lane + c*LD`
with lanes differing by 1, which is conflict-free at any `LD`. The pad is
required by exactly one phase: the **Vh writeback**, where lane `i` owns output
row `i` and must read `V_local[base_v + r + c_i*LD]` with `c_i` its source
column. Banks are then `(r + c_i·LD) mod 32`; with `LD = 33` this is
`(r + c_i) mod 32` over a permutation `c_i` — all 32 distinct. With `LD = 32` it
would be `r mod 32` for every lane, a 32-way serialisation. Oddness is what buys
this (`gcd(LD,32) = 1` makes `c ↦ c·LD mod 32` a bijection); `P+1` is odd for
every `P` in the ladder.

**Pair table packing.** `uint32_t`, four `uint8` fields per word: word `w` holds
slots `2w` and `2w+1` as `p0 | q0<<8 | p1<<16 | q1<<24`. This halves the
broadcast loads (8 instead of 16 per round, since lane = row means every lane
needs the whole round), and unsigned fields remove the sign-bit hazard that
`syev_jacobi_cta`'s `int16_t` packing (`src/extensions/syev_jacobi_cta.cc:271`)
carries if `P` is ever widened past 32.

### Budget formula (mandatory shape)

```cpp
bytes_per_prob = (1 + (ComputeV ? 1 : 0)) * kTileElems * sizeof(T)
               + P            * sizeof(Real)        // Nrm_local
               + 2*kRotSlots  * sizeof(Real)        // Rcs_local (vec<Real,2>)
               + (kNeedPhase ? kRotSlots * sizeof(T) : 0)
               + 2 * P        * sizeof(int16_t);    // Rank_local + Inv_local
wg_fixed       = kPairWords * sizeof(uint32_t);     // 992 B at P=32
```

At `P = 32`, `ComputeV = true`:

| T | A+V | Nrm | Rcs | Rd | Rank+Inv | **bytes_per_prob** |
|---|---|---|---|---|---|---|
| float | 8448 | 128 | 128 | 0 | 128 | **8,832** |
| double | 16896 | 256 | 256 | 0 | 128 | **17,536** |
| complex\<float\> | 16896 | 128 | 128 | 128 | 128 | **17,408** |
| complex\<double\> | 33792 | 256 | 256 | 256 | 128 | **34,688** |

`ComputeV = false` removes `kTileElems * sizeof(T)`: float → 4,608 B.

Sanity check: `syev_jacobi_cta` at the same point budgets
`(1056+1056)*4 + 2*16*4 = 8,576` B (`src/extensions/syev_jacobi_cta.cc:190-192`).
Ours is **3% more**, so the occupancy of the measured 2.11 ms reference carries
over essentially unchanged. That is only true because **U is never materialised
in LDS** — A + V replaces syev's A + Z, one tile for one tile. Holding A, U and
V resident would be 12,832 B/prob and cost ~20% of occupancy.

### The clamp — fix the inherited bug

`src/extensions/syev_jacobi_cta.cc:193-201` clamps `wg_size_multiplier`, but
actual usage is `probs_per_wg * bytes_per_prob` with
`probs_per_wg = wg_size/P = multiplier * (lcm(P,32)/P)`. Since
`base_wg_size = lcm(P,32) = 32` for every supported `P`
(`src/extensions/syev_jacobi_cta.cc:167`), the clamp **under-counts by `32/P`** —
up to 8x at `P=4`. Harmless there only because `bytes_per_prob` is small and the
default multiplier is 1. With two resident tiles it becomes a shape-dependent
launch failure. **Clamp `probs_per_wg` directly:**

```cpp
const int32_t probs_per_warp = 32 / static_cast<int32_t>(P);            // 8,4,2,1
const size_t  local_mem = dev.get_info<sycl::info::device::local_mem_size>();   // 49152 on a 4090
const size_t  avail     = (local_mem > wg_fixed) ? (local_mem - wg_fixed) : 1;
const int32_t max_probs = std::max<int32_t>(1, static_cast<int32_t>(avail / bytes_per_prob));

int32_t pw = std::min<int32_t>(probs_per_warp * static_cast<int32_t>(params.cta_wg_size_multiplier),
                               max_probs);
pw = std::max(1, (pw / probs_per_warp) * probs_per_warp);   // keep wg_size a multiple of 32
const int32_t wg_size    = pw * static_cast<int32_t>(P);
const int32_t num_wg     = (batch_size + pw - 1) / pw;
const int32_t global_size = num_wg * wg_size;
```

`local_mem_size` bounds the **per-block** request; occupancy is decided by the
SM's shared pool, which SYCL does not report. Verify the resulting warps/SM with
`ncu` rather than assuming — the critical-path analysis puts the
issue-bound/latency-bound transition at ~3 warps per sub-partition, right between
the two possible answers.

**Default `probs_per_wg = probs_per_warp` (i.e. `cta_wg_size_multiplier = 1`).**
At `P=32`/float more problems per block does *not* buy warps/SM: 1 prob → 9,824 B
→ 10 blocks → 10 warps; 2 probs → 18,656 B → 5 blocks → 10 warps; 4 probs →
36,320 B → 2 blocks → **8 warps** (worse). At double, 1 prob → 5 warps but 2
probs → 4 warps. The multiplier is a tuning knob, not a default.

**`complex<double>` with vectors is the known cliff:** 34,688 + 992 = 35,680 B →
2 blocks/SM → 2 warps/SM. Do not pretend otherwise. At 1/64 FP64 rate with 4x
complex multiply cost, two warps roughly saturate the FP64 pipe, so accept it for
now, tag the benchmark row, and revisit only if `complex<double>` becomes a real
workload.

## C.5 The sweep body

Notation: `lane` = row index `r`; `nn = n`, `mm_rows = m` (or swapped, see C.9);
`mp = n + (n & 1)` is the padded **column** count, `pairs_per_round = mp/2`,
`rounds = mp - 1` (padding rule from `src/extensions/syev_jacobi_cta.cc:147`).
`Real = base_type<T>::type`.

### Prologue (once)

**L0. Pair table.** Grid-stride over the **whole work-group** filling
`Pair_local` from `round_robin_pair` (`src/extensions/syev_jacobi_cta.cc:115-129`),
ending in `sycl::group_barrier(wg)`. **This must precede the
`if (prob_id >= nb) return;` early exit** (`src/extensions/syev_jacobi_cta.cc:263-273`
vs `:284`) or the tail work-group of a non-multiple batch deadlocks.

> **Stride bug to avoid.** The table is *filled* with the runtime stride
> `pairs_per_round = mp/2`, but the round loop reads it with the compile-time
> stride `kRotSlots = P/2`. These differ whenever `n < P`, and the unrolled `k`
> loop then dereferences garbage column indices into `A_local`. **Pad the table
> to the compile-time stride** `kRotSlots` and write the sentinel `0xFF` into
> slots `k >= pairs_per_round`, so the fill stride and the read stride are both
> `kRotSlots`.

**L1. Lane/problem mapping**, verbatim from
`src/extensions/syev_jacobi_cta.cc:275-290`:
`sg_id`, `parts_per_sg = part.get_group_linear_range()`,
`part_id = sg_id*parts_per_sg + part.get_group_linear_id()`,
`lane = part.get_local_linear_id()`, `prob_id = wg_id*probs_per_wg + part_id`,
early return if `prob_id >= nb`.

**L2. Load A, lane = ROW.** For `c` in `[0,P)`:
`A_local[base_a + lane + c*LD] = (lane < mm_rows && c < nn) ? A_prob(lane, c) : T(0)`.
Lane-as-row makes the global read `A_prob(lane,c)` (address `c*ld + lane`)
**coalesced**; lane-as-column would stride by `ld`. Pad region written as **exact
zero** so a padded pair's Gram is identically 0 and falls below any threshold
(`src/extensions/syev_jacobi_cta.cc:292-312`).

If `ComputeV`: `V_local[base_v + lane + c*LD] = (lane == c && lane < nn) ? T(1) : T(0)`
(`src/extensions/syev_jacobi_cta.cc:313-315`).

**L3. Global rescale — see C.6.** Compute `beta` (a power of two), scale
`A_local`, remember `1/beta` exactly.

**L4.** `group_barrier(part);`

### Per-sweep preamble

**S0. Exact column norms by reduce-scatter over P values.** Lane `r` forms
`x[c] = |A_local[base_a + lane + c*LD]|²` for `c` in `[0,P)` (P LDS loads, P
multiplies), then a reduce-scatter of P values over P lanes (masks 16,8,4,2,1 →
31 shuffles at P=32) leaving lane `c` holding `‖A_c‖²`. Store
`Nrm_local[base_n + lane] = norm_lane`. Cost ≈ 1.1% of a sweep, and it is what
bounds the drift of the analytic updates in R5.

**S1.** `rot_count = 0;`

### Per-round body, `t = 0 .. rounds-1`

**G1.** Read the round's `kRotSlots/2` packed pair words into registers:
`pw[w] = Pair_local[t*(kRotSlots/2) + w]` for `w` in `[0, kRotSlots/2)`. Unpack
to **compile-time-indexed** `pk[k]`, `qk[k]`. Held live for the whole round —
used again in A1.

**G2.** `#pragma unroll for k in [0, kRotSlots):`
```
ap[k] = A_local[base_a + lane + pk[k]*LD];
aq[k] = A_local[base_a + lane + qk[k]*LD];
g[k]  = conj_if_complex_j(ap[k]) * aq[k];
```
Each column is read exactly once (the round's pairs are disjoint and cover every
column). **`ap`/`aq` stay live into A1** — that is worth 32 LDS loads per round
and is the single largest saving of this mapping.

**G3. Reduce-scatter: 16 dot products in 16 shuffles.**

```cpp
// Scatter V=kRotSlots values across L=P lanes: log2(V) scatter steps, then
// log2(L/V) all-reduce steps. At P=32, V=16 that is 4 + 1 -- NOT 5 + 0.
#pragma unroll
for (int step = 0; step < 4; ++step) {                 // masks 16, 8, 4, 2
    const uint32_t mask = 16u >> step;
    const bool hi   = (lane & mask) != 0;
    const int  half = (16 >> step) / 2;                // 8, 4, 2, 1
    #pragma unroll
    for (int j = 0; j < half; ++j) {
        const T own  = hi ? g[j + half] : g[j];
        const T send = hi ? g[j]        : g[j + half];
        g[j] = own + permute_group_by_xor(part, send, mask);
    }
}
// Final stage is an ALL-REDUCE over the surviving pair of lanes, not a halving.
g[0] = g[0] + permute_group_by_xor(part, g[0], 1u);
// Now g[0] == a_{p_k q_k} for k = lane >> 1, replicated in lanes 2k and 2k+1.
```

> **This is the one place the original design was wrong and it produces a silent
> wrong answer.** Written as `half = (16 >> step) / 2` across five steps, `half`
> is 0 at step 4, the inner loop never runs, and every dot product is summed over
> only 16 of the 32 rows. `max(1, ...)` is **not** a fix — at step 4 it makes the
> `hi` lane read a stale `g[1]`. The general rule: scattering `V` values over `L`
> lanes needs `log2(V)` scatter steps **plus** `log2(L/V)` all-reduce steps.
> Note S0's variant (32 values over 32 lanes) is 5 + 0 and is correct as written.

Shuffle count: 8+4+2+1+1 = **16 for all sixteen length-32 dot products**, versus
80 for sixteen independent 5-step butterflies. All masks are `< P`, so the XOR
address stays inside the chunk (`src/extensions/sg_compat.hh:123-129`).

For complex, run two interleaved reduce-scatters (real and imaginary parts) → 32
shuffles.

**Accuracy note:** the tree sum gives dot-product error `~log2(m)·eps` against
the threshold `tol = n·eps` — a clean ~6x margin. A lane-sequential accumulation
would give `~m·eps`, i.e. a noise floor the same order as the convergence
threshold itself, which shows up as threshold churn and an inflated,
data-dependent sweep count.

**R1.** `const int k = lane >> 1;`
Re-read the packed word for pair `k` **directly from `Pair_local`** — do **not**
index `pk[]`/`qk[]` here. `k` is a runtime index and touching the register arrays
with it spills the whole thing to local memory, which is the recorded
register-residency trap in this repo. One extra LDS load per round.
`active = (kp < nn) && (kq < nn) && (k < pairs_per_round)`; only `q` strictly
needs testing since `round_robin_pair` swaps so `p < q`
(`src/extensions/syev_jacobi_cta.cc:124-128`), but the sentinel makes both cheap.

**R2.** `app = Nrm_local[base_n + kp]`, `aqq = Nrm_local[base_n + kq]`. Sixteen
distinct addresses read by two lanes each → same-address broadcast, conflict-free.

**R3. Rotation — byte-for-byte `src/extensions/syev_jacobi_cta.cc:349-385`**, with
`(apq, app, aqq)` now meaning the Gram of A's columns rather than entries of a
stored Hermitian matrix:

```
g_abs  = abs_if_complex_j(apq);
thresh = tol * sycl::sqrt(sycl::fabs(app) * sycl::fabs(aqq));   // RELATIVE  (:351)
if (!(g_abs > thresh && g_abs > tiny)) active = false;
else {
    complex: g = g_abs; d = T(apq.real()/g_abs, -apq.imag()/g_abs);
    real   : g = apq;   d = T(1);
    tau = (aqq - app) / (Real(2)*g);
    tt  = (fabs(tau) > tau_big) ? Real(1)/(Real(2)*tau)
                                : copysign(Real(1),tau)/(fabs(tau)+sqrt(Real(1)+tau*tau));
    c   = Real(1)/sqrt(Real(1)+tt*tt);   s = tt*c;
    if (s == Real(0)) active = false;                            // (:378-381)
}
```

Constants, all from `src/extensions/syev_jacobi_cta.cc:233-249`:
`tol = params.tol_multiplier * nn * numeric_limits<Real>::epsilon()`,
`tiny = numeric_limits<Real>::min()`,
`tau_big = 1/sqrt(eps)`.

The `s == 0 → inactive` guard is **mandatory**. Without it a rotation that rounds
to the identity is counted in `rot_count`, `rot_count` never reaches 0, and every
problem burns all `max_sweeps` — a silent 30x slowdown, not a wrong answer.

**R4. Analytic norm update**, applied when `active`, by lane `2k` only:
```
Nrm_local[base_n + kp] = fmax(app - tt*g, Real(0));
Nrm_local[base_n + kq] =      aqq + tt*g;
```
These are exact: with `t² + 2τt − 1 = 0` the identity
`c²a_pp − 2cs·a_pq + s²a_qq = a_pp − t·a_pq` reduces to the annihilation relation
`t(a_qq − a_pp) = a_pq(1 − t²)`, which is the very equation `τ`/`tt` solve. LAPACK
`?GESVJ` maintains `SVA` by this recurrence.

**R5. Mid-sweep drift guard (grafted from Candidate 1; the original design has
none).** `nrm_new = app − tt·g` is a difference of like-signed quantities and
cancels catastrophically when the two columns have near-equal norms at ~45°. If
`|nrm_new| < params.drift_refresh_ratio * app`, set a per-problem
`refresh_pending` flag; at the end of the round, if any lane set it (one
`partition_reduce_sum_j`), re-run S0. Up to `rounds−1` rounds of drift otherwise
accumulate between S0 refreshes.

> **The consequence of drift is an OUTPUT error, not merely a schedule error, and
> both original write-ups got this wrong.** A drifted `app` inflates
> `thresh = tol·sqrt(|app|·|aqq|)`, a genuinely non-negligible `a_pq` is skipped,
> `rot_count` reaches 0, the sweep loop breaks, and σ is then a column norm of a
> **non-converged** A. Sigma coming from A does not save you here.

**R6.** `if (!active) { c = 1; s = 0; d = 1; }` — outside the `active` branch, so
stored coefficients are always defined (`src/extensions/syev_jacobi_cta.cc:387-391`).
Lane `2k` writes `Rcs_local[base_r + k] = sycl::vec<Real,2>(c, s)` and, if
`kNeedPhase`, `Rd_local[base_r + k] = d`.

**R7.** `group_barrier(part);`

**R8.** `round_active = partition_reduce_sum_j(part, (active && (lane % 2 == 0)) ? 1 : 0);`
— counted on even lanes only, so no division is needed.
`rot_count += round_active; if (round_active == 0) continue;`
**Called by ALL P lanes, never inside a predicate.** It is a butterfly XOR
reduction (`src/extensions/syev_jacobi_cta.cc:96-103`) and a non-participating
lane poisons the result (`:406`).

**A1. Column update, lane = row.** `#pragma unroll for k in [0, kRotSlots):`
```
cs = Rcs_local[base_r + k]; ck = cs[0]; sk = cs[1];
if (sk == Real(0)) continue;                                 // WARP-UNIFORM skip (:418)
u11 = T(ck); u12 = T(sk); u21 = T(-sk); u22 = T(ck);
if constexpr (kNeedPhase) { dk = Rd_local[base_r+k]; u21 = -(dk*T(sk)); u22 = dk*T(ck); }
A_local[base_a + lane + pk[k]*LD] = ap[k]*u11 + aq[k]*u21;    // operands in registers
A_local[base_a + lane + qk[k]*LD] = ap[k]*u12 + aq[k]*u22;
if constexpr (ComputeV) {
    vp = V_local[base_v + lane + pk[k]*LD];
    vq = V_local[base_v + lane + qk[k]*LD];
    V_local[base_v + lane + pk[k]*LD] = vp*u11 + vq*u21;
    V_local[base_v + lane + qk[k]*LD] = vp*u12 + vq*u22;
}
```

This is exactly `src/extensions/syev_jacobi_cta.cc:424-448`'s Phase 1, applied to
A and V. The complex branch is **unchanged from the working kernel** — that is a
deliberate design goal: complex needs no new derivation.

**There is no Phase 2.** `src/extensions/syev_jacobi_cta.cc:453-501` (the
`A ← U^H A` row update) is deleted outright. That deletion is the whole
difference between two-sided and one-sided Jacobi, and it is a third of the
reference kernel's LDS traffic.

**A2.** `group_barrier(part);`

> **A2 is deletable, but not in the first version.** Under lane = row, lane `r`
> exclusively reads and writes row `r` of both A and V, so neither tile carries a
> cross-lane hazard; the only hand-offs are `Rcs_local`/`Rd_local` (ordered by
> R7) and `Nrm_local` (written in R4 before R7, read in the next round's R2 after
> R7). Land the kernel with both barriers, then delete A2 as a **separate,
> measured** change with the invariant recorded in a comment.

### Sweep termination

`if (rot_count == 0) break;` (`src/extensions/syev_jacobi_cta.cc:505-507`).

**Then one verification sweep, unconditionally.** After the break, run the round
loop once more with S0's exact norms and **no updates** — Gram + threshold +
reduce only. If it reports any active pair, resume the normal loop. Cost is
~17% of a full sweep. This converts the "threshold fired early on drifted norms"
failure from a silent wrong answer into one extra sweep, and the existing float
test tolerances (5e-2 / 2e-1 / 3e-1, `tests/gesvd_tests.cc:312-323`) cannot
detect it any other way.

## C.6 Scaling — global only, geometric centre, power of two

**`GESVD_PLAN.md` §4.4.1 is wrong as an algorithm step and must not be
implemented as written.** All three candidate designs and two of three judges
independently refuted it. If you scale columns by `D`, you compute the SVD of
`A·D = U Σ_c W^H`, hence `A = U Σ_c (D^{-1}W)^H`, and `D^{-1}W` is **not
orthogonal** — the result is not a factorisation of `A`. The `κ(A_c)` of
Demmel–Veselic is an *analysis* quantity; it is delivered by the rotation and
threshold formulas already being per-pair scale-invariant
(`src/extensions/syev_jacobi_cta.cc:349-377`), not by performing a scaling.
LAPACK `?GESVJ` likewise scales only by a scalar. **Amend the plan.**

What is admissible and required is a **single global power-of-two rescale**:

```
nmax = partition_reduce_max(part, nrm);        // max_c ||A_c||^2
nmin = partition_reduce_min(part, nrm over c < nn, ignoring exact zeros);
if (nmax == 0) { write zeros; return; }
beta = exp2( -0.5 * round(0.5 * (log2(nmax) + log2(nmin))) );   // power of two, exact
```

**Centre on the geometric mean, not the maximum.** In float you need
`max_norm < 1.84e19` (so `max²` stays finite) and `min_norm > 1.09e-19` (so
`min²` stays normal). Centring on the max tolerates a column-norm ratio of
`9.2e18`; centring geometrically tolerates `1.69e38` — a factor of `1.8e19` more
headroom, for one extra min-reduction. Graded matrices, exactly the class the
accuracy argument rests on, are where that ratio is large.

`beta` must be a power of two so `beta · (1/beta) == 1` exactly and the
scale/unscale round-trip is lossless.

The column-norm epilogue additionally needs a **scaled sum-of-squares guard**:
without it a column whose norm underflows in the squaring gives `σ = 0`, which
trips the rank-deficiency path and **fabricates** a U column — a wrong answer
wearing the costume of a rank-deficient input.

## C.7 σ, U, V extraction and sort order

**E1. σ comes from A, always.** Repeat S0 one final time, unconditionally, then
`sigma_lane = (1/beta) * sqrt(Nrm_local[base_n + lane])`.

> **This single rule is the line between this kernel and Defect A.** The
> incrementally maintained `Nrm_local` exists only to *choose rotations*, where an
> error perturbs the schedule. Reading σ from it instead — a one-line shortcut
> that will pass every existing test — reintroduces the defect through the side
> door.

**E2. Sort DESCENDING. This is definitive.** Three independent sites in the tree
assume it:

* `finalize_values_only` produces it by an index reversal
  `src = (n-1) - tgt` over ascending eigenvalues
  (`src/extensions/gesvd_blocked.cc:305-312`) — it is not a sort flag;
* `has_tiny_singular_values` reads `sb[0]` as σ_max
  (`src/extensions/gesvd_blocked.cc:531`);
* `patch_zero_left_vectors` reads `sb[0]` as σ_max
  (`src/extensions/gesvd_blocked.cc:564`).

`tests/gesvd_tests.cc:474` compares element-for-element against LAPACKE `?gesvd`,
which is descending, and `expect_sorted_singular_values` takes a parameter
literally named `expected_desc`.

Implement as the **parallel rank sort** of
`src/extensions/syev_jacobi_cta.cc:514-527` with `ascending = false`: lane `c`
computes `rank = #{ j : σ_j > σ_c || (σ_j == σ_c && j < c) }`, ties broken on
index so the permutation is a bijection. Write both directions:
`Rank_local[base_p + c] = rank` and `Inv_local[base_p + rank] = c`. The writeback
loops index by the **inverse**. `group_barrier(part);`

**Sort descending directly and do NOT also reverse indices.**

**E3.** `S[prob_id*k + dst] = sigma(Inv_local[base_p + dst])`, one lane per output
index.

**E4. U writeback, lane = ROW.** For output column `dst`, `c = Inv_local[base_p+dst]`:
`U_prob(lane, dst) = A_local[base_a + lane + c*LD] * (1/sigma_c)`.
Global address `dst*ld + lane` → coalesced; LDS bank `(lane + c·LD) mod 32` over
distinct `lane` → conflict-free.

**E5. Vh writeback, lane = OUTPUT ROW `i`.** `c = Inv_local[base_p + i]`:
`for r in [0,nn): Vh_prob(lane, r) = conj_if_complex_j(V_local[base_v + r + c*LD]);`
The LDS read is conflict-free because `LD` is odd and `c` is a permutation (C.4);
the global write `r*ld + lane` is contiguous in lane → coalesced. **The two
writebacks use opposite lane roles** — each is the one that coalesces for its
output's orientation.

Complex helpers `conj_if_complex_j`, `abs_if_complex_j`, `force_real_j`,
`real_part_j` are at `src/extensions/syev_jacobi_cta.cc:54-91`; copy them (they
are file-local statics, not a shared header).

## C.8 Rank deficiency and `m > n`

Columns `n..m-1` of a full `m x m` U, and any column whose σ is below the zero
threshold, are **not determined by A**. This is the semantic that
`patch_zero_left_vectors` (`src/extensions/gesvd_blocked.cc:544-573`) implements
— but note it does **not** orthogonalise anything: it copies columns from a
separately computed left-eigenvector matrix produced by a whole second
tridiagonal eigensolve, selected by `sb[tgt] > tol_zero`. That mechanism is not
available inside a fused kernel, so this is genuinely new code.

**Threshold.** `src/extensions/gesvd_blocked.cc:317-324` defines
`tol_zero = eps * fmax(T(1), sigma_max)`. Adopt it **with one correction**: drop
the `fmax(T(1), ·)`. As written, a uniformly small input (σ_max = 1e-10) has
*every* singular value declared zero and every U column fabricated. Use:

```
tol_zero = params.zero_sigma_multiplier * eps * sigma_max;     // relative to sigma_max only
```

applied in the **unscaled** domain (after the `1/beta` multiply), and expose
`zero_sigma_multiplier` so the accuracy harness can sweep it. This keeps the
tree's definition of "not determined by A" while fixing the demonstrable bug.

**Algorithm.** Gate on a warp-uniform
`any(sigma_c <= tol_zero) || m > n` (one `partition_reduce_sum_j` over a
predicate) — a well-conditioned square input pays one compare and skips the whole
branch. When it fires, for each deficient output column `dst` in increasing
order:

1. Seed with the canonical basis vector `e_j` whose accumulated overlap with the
   already-accepted columns is smallest (one pass over `U_local`, which is
   `A_local` normalised in place — this is safe because A has been fully consumed
   into σ and U by E4).
2. **Two passes of classical Gram–Schmidt** against every accepted column. Each
   dot product is one 5-step butterfly with lane = row. *Two* passes, not one
   pass of MGS: a single pass against an ill-conditioned accepted set loses the
   orthogonality the patch exists to provide, in exactly the near-deficient case
   that triggers it.
3. Accept when the residual norm exceeds 1/2; otherwise advance `j` and retry.
4. Normalise, write to `U_prob(lane, dst)`.

This is the least-tested path in the kernel by construction — it is off the
`m == n == 32` benchmark shape. **Write the tests first** (E.2: exactly-rank-1,
rank-`n/2`, and `m = 2n` cases).

## C.9 The `m < n` path

Do **not** use `gesvd_blocked.cc`'s out-of-place transpose + recursion
(`src/extensions/gesvd_blocked.cc:706-747`). The tile is ~4 KB of global traffic
per problem, so transposing at **load time** is free:

* In L2, read `A_prob(c, lane)` instead of `A_prob(lane, c)` and swap the
  `mm_rows`/`nn` bounds. The kernel then solves `A^T` (`n x m`, `n > m`), which
  satisfies the rows ≥ columns requirement.
* `A^T = U' Σ V'^H` gives `A = V' Σ U'^H`, so **`U = V'` and `Vh = U'^H`**.
  Swap the two writeback roles: E4's target becomes `Vh` (transposed as in E5)
  and E5's target becomes `U`.
* Sizes work out: `U'` is `n x n` → our `Vh` (`n x n`) ✓; `V'` is `m x m` → our
  `U` (`m x m`) ✓; `min(m,n) = m` singular values ✓.
* The transposed problem has `n` rows and `m < n` columns, so `n − m` columns of
  `U'` are undetermined — the C.8 completion path runs on the **Vh** side. One
  code path, role-swapped.

The job swap is the same one `src/extensions/gesvd_blocked.cc:715-716` performs
(`trans_jobu = want_vh ? All : None`); reuse that as the semantic reference.

`P = next_pow2(max(m,n))` already covers both orientations.

## C.10 Grafts, deferrals and traps

1. **Register pressure is a hard wall for wide scalars.** `ap[16] + aq[16] +
   g[16]` is 48 live `T` values — ~60-75 registers in float (fine at the 10
   warps/SM that LDS already fixes, budget 204/thread), but ~200+ for
   `complex<double>`, where a spill goes to global-backed local memory. **For
   `sizeof(T) >= 16`, process each round as two half-rounds of 8 pairs**, which
   drops the live arrays to 24 values at the cost of one extra `Rcs` round-trip
   and one barrier. Verify with `-Xcuda-ptxas -v` before believing any timing.
2. **Every index into `pk[]`, `qk[]`, `g[]`, `ap[]`, `aq[]` must be
   compile-time.** The unrolled `k` loop, the `(j, j+half)` reduce-scatter
   indexing, and the `hi ? x : y` selects between *named* registers are
   correctness-of-performance requirements, not style.
3. **de Rijk pre-ordering — deferred, `params.derijk = false`.** Candidate 1's
   implementation is the one to use when it is enabled: a **permuted reload from
   global** (`A_prob(lane, Perm[c])`, still coalesced under lane = row) with the
   permutation folded into V's initialisation as
   `V[i,j] = (i == Perm[j]) ? 1 : 0`. That removes the objection that an in-LDS
   column permutation needs a scratch tile there is no budget for. It also has a
   direction trap: inverting the permutation returns a valid-looking but wrong V
   that reconstruction tests on symmetric-ish input will not catch. Ship it off
   until it is *measured* to save at least one sweep.
4. **Candidate 1's closed-form partner inversion** (`partner(0) = (t mod ring)+1`;
   for `j>=1`, `u=j-1`, `v=((2t-u) mod ring + ring) mod ring`,
   `partner(j) = (v==u) ? 0 : v+1`) is correct, and correct **only because
   `ring = mp-1` is odd** (the padding at `src/extensions/syev_jacobi_cta.cc:147`
   guarantees it). It is not useful under lane = row — every lane needs all 16
   pairs, so 8 packed broadcast loads beat 16 closed-form evaluations — but keep
   it as a comment for any future lane = column variant.
5. **Off-CUDA correctness rests on sub-group lockstep.**
   `group_barrier(SubGroupPartition<P>)` is a **no-op** unless
   `__SYCL_DEVICE_ONLY__ && __NVPTX__` (`src/extensions/sg_compat.hh:32-34`,
   `:154-158`). Under lane = row the update phase has *no* cross-lane LDS hazard
   at all — that is a genuine portability advantage of this mapping — but the
   `Rcs_local`/`Nrm_local` hand-offs still depend on it, exactly as
   `syev_jacobi_cta` already does. The kernel is instantiated for ROCM and
   NETLIB by the standard footer; that arm is the least-defended.
6. **A fully-unrolled round loop would put V's row in registers** (32 floats at
   P=32, since `pk/qk` become compile-time constants), deleting 64 of ~181
   LDS ops per round. Do not attempt it: 31 rounds × 16 pairs × ~12 instructions
   is ~6k instructions, 48-96 KB, against a 32 KB Ada I-cache. Recorded so it is
   not re-tried without the cache arithmetic.
7. **`n < P` wastes lanes twice** (padded rows and inactive pair slots) and the
   reduce-scatter is compile-time sized to `P/2` so it does not shrink.
   `gesvdjBatched` has the same cliff, so it is unlikely to lose the comparison,
   but it caps absolute throughput in the 17-31 range.

## C.11 Instantiation footer

Copy the shape of `src/extensions/syev_jacobi_cta.cc:639-664` exactly:

```cpp
#define GESVDJ_CTA_INSTANTIATE(back, fp) \
    template Event gesvdj_cta<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, SvdVectors, \
        const Span<std::byte>&, \
        GesvdjParams<BATCHLAS_UNPAREN fp>); \
    template size_t gesvdj_cta_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, SvdVectors, \
        GesvdjParams<BATCHLAS_UNPAREN fp>);

#define GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(GESVDJ_CTA_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef GESVDJ_CTA_INSTANTIATE_FOR_BACKEND
#undef GESVDJ_CTA_INSTANTIATE
```

`BATCHLAS_FOR_EACH_SCALAR_TYPE_1` passes each type **parenthesised**
(`src/util/template-instantiations.hh:41-53`), so every use must be spelled
`BATCHLAS_UNPAREN fp` (no parentheses after `UNPAREN`) — otherwise
`std::complex<float>`'s embedded comma breaks the expansion. **Every overload
needs its own `template ...;` line**; a missing one is an undefined symbol at
final link, not a compile error in the `.cc`
(`src/extensions/gesvd_blocked.cc:1328-1400` needed six).

---

# Part D — dispatch wiring

## D.1 `Provider::BatchLAS_Jacobi` does not exist and must be added

`include/blas/dispatch/provider.hh:7-14` has exactly six enumerators:
`Auto, Vendor, BatchLAS_CTA, BatchLAS_Blocked, BatchLAS_TwoStage, Netlib`.
**Three declarations carry a hard-coded array size 5** and all three must be
widened:

1. `include/blas/dispatch/provider.hh:7-14` — add the enumerator.
2. `include/blas/dispatch/provider.hh:18-24` — `std::array<Provider, 5> order`
   → `6`.
3. `include/blas/dispatch/env.hh:53-59` —
   `default_order_cta_blocked_vendor_netlib`, `std::array<Provider, 5>` → `6`.
   (`default_order_for_op` at `env.hh:62-65` returns it by value and needs no
   further edit.)

## D.2 Edits

**`include/blas/dispatch/provider.hh`:**
```cpp
enum class Provider {
    Auto, Vendor,
    BatchLAS_Jacobi,          // <-- new
    BatchLAS_CTA, BatchLAS_Blocked, BatchLAS_TwoStage, Netlib,
};

struct DispatchPolicy {
    Provider forced = Provider::Auto;
    std::array<Provider, 6> order = {          // was 5
        Provider::BatchLAS_CTA,
        Provider::BatchLAS_Blocked,
        Provider::BatchLAS_TwoStage,
        Provider::BatchLAS_Jacobi,             // NOT first -- see D.3
        Provider::Vendor,
        Provider::Netlib,
    };
    ...
};
```

**`include/blas/dispatch/env.hh:27-44`** — add to `parse_provider_value`, next to
the `"cta"` line at `:34`:
```cpp
if (s == "jacobi" || s == "batchlas_jacobi" || s == "batchlas-jacobi")
    return Provider::BatchLAS_Jacobi;
```
An unknown value falls back to `Provider::Auto` **silently** (`env.hh:43`), so a
typo'd `BATCHLAS_GESVD_PROVIDER=jacobi` before this edit runs the default path
and looks like the new kernel is being measured. Add the string in the same
commit as the enumerator.

**`include/blas/dispatch/env.hh:53-59`** — widen
`default_order_cta_blocked_vendor_netlib` to `std::array<Provider, 6>` with the
same contents as `DispatchPolicy::order` above.

**`include/blas/functions/gesvd.hh`** — add the predicate after
`gesvd_supports_cta` (which ends at `:240`):

```cpp
template <typename T>
inline bool gesvd_supports_jacobi(const DeviceCaps& caps,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  SvdVectors jobu,
                                  SvdVectors jobvh,
                                  std::optional<Uplo> hermitian_uplo = std::nullopt) {
    if (hermitian_uplo.has_value()) return false;   // no Hermitian shortcut; see C.2
    if (!caps.is_gpu) return false;
    if (caps.max_sub_group < 32) return false;
    if (A.rows() < 1 || A.cols() < 1 || A.batch_size() < 1) return false;
    if (std::max(A.rows(), A.cols()) > 32) return false;
    if (jobu  != SvdVectors::None && jobu  != SvdVectors::All) return false;
    if (jobvh != SvdVectors::None && jobvh != SvdVectors::All) return false;
    return true;    // real AND complex -- no RealScalar gate
}
```

Note the deliberate asymmetry with the two existing predicates: both
`gesvd_supports_cta` (`include/blas/functions/gesvd.hh:229-233`) and
`gesvd_supports_blocked` (`:249-252`) `return false` for non-real `T` outside the
Hermitian branch. `gesvdj_cta` supports complex **general** input natively, which
is the Tier 4 coverage gap (`GESVD_PLAN.md` §Tier 4). Do not copy the
`RealScalar` gate.

**`choose_gesvd_provider`** (`include/blas/functions/gesvd.hh:257-288`) — add one
line to each of the two loops, before the CTA line:

```cpp
    // forced branch, before gesvd.hh:265
    if (chosen == Provider::BatchLAS_Jacobi &&
        gesvd_supports_jacobi(caps, A, jobu, jobvh, hermitian_uplo)) return chosen;

    // order-walk branch, before gesvd.hh:277
    if (p == Provider::BatchLAS_Jacobi &&
        gesvd_supports_jacobi(caps, A, jobu, jobvh, hermitian_uplo)) return p;
```

**`gesvd_dispatch`** (`include/blas/functions/gesvd.hh:290-353`) — add an
explicit branch in **both** the sizing chain (before `:312`) and the run chain
(before `:345`):

```cpp
    } else if (chosen == Provider::BatchLAS_Jacobi) {
        need_ws = gesvdj_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    ...
    if (chosen == Provider::BatchLAS_Jacobi) {
        return gesvdj_cta<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace);
    }
```

**`gesvd_buffer_size_dispatch`** (`include/blas/functions/gesvd.hh:357-386`) —
the same sizing branch, before `:377`.

> **The explicit branch is not optional.** `gesvd_dispatch`'s tail
> (`include/blas/functions/gesvd.hh:351-353`) is an unguarded
> `return gesvd_blocked<B,T>(...)` — there is no `Provider::BatchLAS_Blocked`
> test. Add the enumerator without adding the branch and `gesvdj_cta` **silently
> executes the blocked normal-equation path**, i.e. the exact defect it exists to
> remove, while every benchmark label says otherwise.

## D.3 Ordering policy — deliberate, and it is not first

`GESVD_PLAN.md` Tier 1 step 5 says to place Jacobi "ahead of the current CTA
path for n <= 32", but the same section also says "**keep the old CTA path behind
a provider flag until Tier 1 is measured faster *and* more accurate**". The
measured head-to-head (front matter) shows `gesvd_cta` is already 1.6-3.8x faster
than the vendor, so making Jacobi the default before it is measured risks a
regression on the axis we currently win.

**Call: land `BatchLAS_Jacobi` in the default order *after* `BatchLAS_TwoStage`
and before `Vendor`, where it is unreachable for any shape `BatchLAS_CTA`
accepts** (CTA precedes it and accepts everything Jacobi accepts, for real `T`).
It therefore becomes reachable in exactly two useful ways:

* **forced**, via `BATCHLAS_GESVD_PROVIDER=jacobi` — which
  `choose_gesvd_provider`'s forced branch honours ahead of the order walk;
* **automatically for complex general input**, because `gesvd_supports_cta` and
  `gesvd_supports_blocked` both return false there and today it falls through to
  `Vendor` and throws. That is a strict improvement with no regression risk.

Promoting it ahead of `BatchLAS_CTA` is a **separate one-line commit**, gated on
the Part F.6 exit criteria.

**Forcing remains fully general.** `parse_provider_env` reads
`"BATCHLAS_" + uppercase(op) + "_PROVIDER"` (`include/blas/dispatch/env.hh:46-51`)
and `gesvd_dispatch` calls `policy_from_env("GESVD")`
(`include/blas/functions/gesvd.hh:302`), so `BATCHLAS_GESVD_PROVIDER` ∈
{`auto`, `vendor`, `cta`, `blocked`, `two_stage`, `jacobi`, `netlib`} all work
after D.2.

> **Forcing degrades silently.** A forced provider whose `*_supports_*`
> predicate returns false resets `chosen = Provider::Auto`
> (`include/blas/functions/gesvd.hh:271`) and falls through the default order
> with **no diagnostic**. Forcing `jacobi` on a 64x64 matrix runs `blocked`.
> Every benchmark and test that forces a provider must verify the route was
> actually taken — see E.1.

---

# Part E — tests and benchmarks

## E.1 New test binary

**`tests/gesvdj_cta_tests.cc`.** Register in **two** lists:

* `TEST_TARGETS` (`tests/CMakeLists.txt:18`), which is the whole registration —
  `batchlas_add_test_target` does `add_executable(${name} ${name}.cc)`
  (`tests/CMakeLists.txt:165-204`).
* `BATCHLAS_TEST_LABELS_eig` (`tests/CMakeLists.txt:136-139`), next to
  `gesvd_tests`. Omitting it is **not an error**: `batchlas_test_component`
  returns `"unlabelled"` (`tests/CMakeLists.txt:162`), the test still runs under
  bare `ctest`, but it vanishes from `ctest -L eig` and from every scoped command
  in `tests/README.md`.

**Do not add it to `BATCHLAS_SLOW_TESTS`** (`tests/CMakeLists.txt:143-151`)
initially. The stated rule is >~15 s (`tests/CMakeLists.txt:118-120`); a single
fused kernel at small batch should be far under. Measure and add it only if it
crosses.

**Scaffolding.** Copy the Config/type-list/fixture shape from
`tests/gesvd_tests.cc:55-129`, **not** `test_utils::backend_types`: that file
rolls its own `backend_real_types` / `backend_complex_types`
(`tests/gesvd_tests.cc:61-99`) because gesvd's complex support is Hermitian-only.
For `gesvdj_cta` the complex list should use the **general** (non-Hermitian) path,
so define a single `backend_all_types` covering all four scalar types, guarded
per backend by `BATCHLAS_HAS_*_BACKEND` and terminated with `std::tuple<>{}`.

Direct-kernel GPU tests wrap the whole body in
`#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND` and call with an
explicit backend (`tests/syev_jacobi_cta_tests.cc:324-360`), because the kernel
requires a 32-wide sub-group. Use `if constexpr (B != Backend::CUDA) GTEST_SKIP()`
for CUDA-only cases (`tests/gesvd_tests.cc:767-773`) — a compile-time branch, not
a runtime skip inside a shared body.

Fill matrices by hand with the plain `Matrix<T,F>(n, n, batch)` constructor:
`Matrix::Zeros`/`Matrix::Diagonal` launch device kernels asynchronously and race
with subsequent host writes to the same USM memory
(`tests/syev_jacobi_cta_tests.cc:105-108`).

**Route verification.** Any test that goes through `gesvd` rather than calling
`gesvdj_cta` directly must confirm the route, because forcing degrades silently
(D.3). Use the RAII `ScopedEnvVar` at `tests/gesvd_tests.cc:31-51` to set
`BATCHLAS_GESVD_PROVIDER=jacobi`, and assert on an observable that only the
Jacobi path produces (e.g. `‖U^H U − I‖` at a tolerance the normal-equation path
cannot meet at κ=1e5).

## E.2 Required cases

| # | case | asserts |
|---|---|---|
| 1 | `n = 1,2,3,4,5,8,16,17,31,32`, square, random, all four types | σ vs `lapacke_gesvd_values_only_any<Scalar>` (`tests/gesvd_tests.cc:150-217`) |
| 2 | σ ordering | strictly descending (E2 of C.7) |
| 3 | identity, zero, and diagonal-with-repeats | exact σ; U, Vh orthonormal |
| 4 | **graded**, κ = 1e2/1e4/1e6 (float), 1e12 (double) | **relative** σ error ≤ `C·eps·κ(A_c)` — this is the test that fails on `gesvd_cta` |
| 5 | exactly rank-1 and rank-`n/2` | C.8 completion: U orthonormal, no NaN |
| 6 | `m > n` (`m = 2n`), `jobu = All` | `U` is `m x m` orthonormal; last `m−n` columns from the complement |
| 7 | `m < n` | C.9 role swap; compare against the same matrix solved by `gesvd_blocked` |
| 8 | all four `jobu`/`jobvh` combinations | `None` side untouched; `All` side correct |
| 9 | complex general, `n = 8,16,32` | reconstruction + orthogonality; **currently throws on every GPU path** |
| 10 | batch = 1, 3, 33, 1000 | ragged tail work-groups (`prob_id >= nb`) |

## E.3 Tolerance policy

`tests/gesvd_tests.cc` uses, for float, `gesvd_sv_tol = 5e-2`
(`tests/gesvd_tests.cc:312`), `gesvd_ortho_tol = 2e-1` (`:317`),
`gesvd_recon_tol = 3e-1` (`:322`). Those are the tolerances a normal-equations
SVD needs in order to pass; a relative reconstruction tolerance of 0.3 admits a
result with essentially no accuracy.

**`tests/gesvdj_cta_tests.cc` defines its own, tighter constants** — do not
`#include` or copy the gesvd ones. Starting point, to be calibrated against the
first measured run:

```cpp
template <typename Real> constexpr Real gesvdj_sv_rel_tol()   { return std::is_same_v<Real,float> ? Real(1e-5f) : Real(1e-13); }
template <typename Real> constexpr Real gesvdj_ortho_tol()    { return std::is_same_v<Real,float> ? Real(1e-5f) : Real(1e-13); }
template <typename Real> constexpr Real gesvdj_recon_tol()    { return std::is_same_v<Real,float> ? Real(1e-5f) : Real(1e-13); }
```

Case 4 uses a κ-scaled bound, not a constant.

**Tightening `tests/gesvd_tests.cc` is a SEPARATE, SEPARATELY-REVIEWED COMMIT**
(`GESVD_PLAN.md` §5.5, §8 step 8). It will break the existing CTA and blocked
paths — that is the point. It must not be folded into any kernel change, or a
kernel regression and a tolerance change become indistinguishable in the log.

## E.4 Benchmarks

**Extend the two existing TUs, do not add new ones.** Benchmarks are
`EXCLUDE_FROM_ALL` with a ~12 s per-TU compile floor
(`benchmarks/CMakeLists.txt:74`), and `benchmarks/CMakeLists.txt` already carries
`gesvd_vendor_benchmark` and `gesvd_relacc` (added in the working tree at
`benchmarks/CMakeLists.txt:12` and `:45`).

* **`benchmarks/gesvd_vendor_benchmark.cc`** — add `BM_GESVD_BATCHLAS_JACOBI`
  next to the two arms at `:135-136`. Wrap A in `bench::pristine`
  (`include/util/bench_structured.hh:117`): structured-mode setup runs **once**
  (`include/util/minibench.hh:317`) and gesvd destroys A, so without it every
  iteration after the first measures a different problem. Compute the workspace
  size **before** `SetKernel` — `SetKernel` moves from every argument
  (`include/util/bench_structured.hh:211`).
* **`benchmarks/gesvd_relacc.cc`** — add `RelAccImpl::BatchlasJacobi` to the enum
  at `:139`, a branch at `:184-193`, and a third `BATCHLAS_ACC_CUDA` at `:256`.

**Sweep at saturation only** (`GESVD_PLAN.md` §6, and the repo-wide measurement
rule): `batch ∈ {1024, 4096, 16384, 65536}`. The old
`benchmarks/gesvd_cta_benchmark.cc:20` stops at 64, which measures launch
overhead, not the algorithm.

**Name collision warning.** minibench's `--name` is a plain substring filter
(`include/util/minibench.hh:591`), so `--name=GESVD` selects every arm and
`--name=JACOBI` is the only unambiguous token for the new one. Also: any
positional CLI arg **replaces the entire registered sweep for every benchmark in
the binary** (`include/util/minibench.hh:765`), and `state.range(i)` returns 0 for
a missing arg (`:159`) — so always pass all four (`32 16384 1 1`), never just
`32 16384`.

**One GPU at a time.** This box has two 4090s; a concurrent run cost 7% in the
plan's own measurement (`GESVD_PLAN.md` §6).

---

# Part F — build order and verification

Builds here are **device-link-bound**; the shared library is the link unit
(`src/extensions/CMakeLists.txt:46-55`). Compile the narrowest target that can
fail, never the whole project.

| Target | What it checks |
|---|---|
| `cmake --build build --target batchlas_backends_cuda` | Part A (`src/backends/cusolver.cc`) |
| `cmake --build build --target batchlas_extensions_cta` | Part C (`src/extensions/gesvdj_cta.cc`) — **every edit to the kernel redoes this group's device link** |
| `cmake --build build --target gesvdj_cta_tests` | Part E.1 + all header/instantiation wiring |
| `cmake --build build --target gesvd_relacc` | Part B |
| `cmake --build build --target gesvd_vendor_benchmark` | Part E.4 |
| `cmake --build build --target batchlas_benchmarks` | all benchmarks (~61 TUs, ~710 s — avoid) |

Header-only edits (Part D) do not have their own target: build
`gesvdj_cta_tests`, which pulls `include/blas/extensions.hh`,
`include/blas/functions/gesvd.hh`, `provider.hh` and `env.hh` through
`blas/linalg.hh` (`include/blas/linalg.hh:14-16`).

**Test scoping** (`tests/README.md`, and the repo-wide policy): do **not** run
full `ctest`. Use `ctest -R '^gesvdj_cta_tests$'`, `ctest -L eig -LE slow`.
`-R` is a substring regex, so anchor it.

## The order

**F.1 — Finish Tier 0 (Part A.3).** ~1 day.
*Verify:* the 32x32 probe records a status code in a comment next to
`kGesvdjBatchedMaxDim` (`src/backends/cusolver.cc:281`); `gesvd_vendor` no longer
throws for `n = 48` (takes `gesvda`, labelled) or `m < n` (takes the loop,
labelled); `gesvd_relacc --benchmark_filter='*CUSOLVER*'` still produces the same
numbers it does today at n ≤ 32.
*Build:* `batchlas_backends_cuda`, then `gesvd_relacc`.

**F.2 — Finish the harness (Part B.2).** ~1 day, **before** any kernel code.
*Verify:* a κ sweep on the **current** paths reproduces Defect A —
`max_relerr` for `gesvd_cta` grows like `κ²·eps` while `gesvdjBatched` grows like
`κ·eps`. If that curve does not appear, the instrument is wrong and nothing
downstream is measurable. This is `GESVD_PLAN.md` §8 step 1 and it is a gate, not
a preliminary.
*Build:* `gesvd_relacc`.
```
./build/benchmarks/gesvd_relacc --samples=256 --log10-cond=1:7:13 --csv=defectA.csv 32
```

**F.3 — `gesvdj_cta`, real, values-only (`jobu = jobvh = None`).**
Skip C.5 step A1's V branch, C.7 E4/E5, and all of C.8. This is the smallest
thing that can be correct.
*Verify:* test cases 1-4. Case 4 is the deliverable — relative σ error tracking
`eps·κ(A_c)` where `gesvd_cta` tracks `eps·κ²`. Also dump the sweep-count
distribution: if the mean exceeds ~14, stop and reconsider before adding V (the
whole timing projection is linear in sweeps).
*Build:* `batchlas_extensions_cta`, then `gesvdj_cta_tests`.
*Check before believing any timing:* `-Xcuda-ptxas -v` for spills (C.10.1), and
`ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared` — the design
claims conflict-free on every hot access.

**F.4 — Add U and V (C.7 E4/E5, the `ComputeV` branch).**
*Verify:* test cases 5-8, at the tightened tolerances of E.3. Reconstruction and
orthogonality are what this step buys.
*Build:* `gesvdj_cta_tests`.

**F.5 — Dispatch wiring (Part D).**
*Verify:* `BATCHLAS_GESVD_PROVIDER` ∈ {auto, vendor, cta, blocked, jacobi} each
reach the intended provider — assert on a route-observable, not on the absence of
a throw (D.3: forcing degrades silently). Confirm complex general GPU input now
routes to Jacobi automatically instead of throwing.
*Build:* `gesvdj_cta_tests`, then `ctest -L eig -LE slow` to catch collateral
damage from the widened `std::array<Provider, 6>`.

**F.6 — Benchmarks and the head-to-head (Part E.4).**
*Verify:* the three-arm table at `n ∈ {8,16,24,32}`,
`batch ∈ {4096, 16384, 65536}`, `{float, double}`, all four job combinations,
against the front-matter baseline. **Exit criteria for promoting Jacobi ahead of
`BatchLAS_CTA` in the default order (D.3):** `gesvdj_cta` ≤ `gesvd_cta` in time
at every measured point **and** strictly better `max_relerr` at every κ ≥ 1e3.
If it is slower but more accurate, keep the order and let users force it — a
silent time regression on the axis we currently win is not an acceptable price
for an accuracy gain nobody asked for by name.
*Build:* `gesvd_vendor_benchmark`, `gesvd_relacc`.

**F.7 — Complex (C.5's `kNeedPhase` branches are already written; enable and
test).**
*Verify:* test case 9. Watch `complex<double>` occupancy (C.4: 2 warps/SM) and
register spilling (C.10.1: half-round splitting for `sizeof(T) >= 16`).
*Build:* `batchlas_extensions_cta`, `gesvdj_cta_tests`.

**F.8 — Tighten `tests/gesvd_tests.cc` (E.3), separate commit.**
*Verify:* expect the existing CTA and blocked paths to fail. Decide per-path
whether to fix, deprecate, or exempt with a documented reason. Run the full
suite once here (`ctest`), since this is the pre-push gate.

**F.9 — Tier 2 (QR preconditioning), Tier 3 (`bdsqr`), Tier 4 — unchanged from
`GESVD_PLAN.md` §8 steps 6-9.** Do not start any of them before F.6 produces the
head-to-head table.

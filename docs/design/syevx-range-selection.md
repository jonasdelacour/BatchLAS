# syevx range selection: design record {#design_syevx_range}

> **Status:** current · implemented 2026-08-05. The Python layer and the range benchmark are not yet verified (see the last section).

`syevx` supports two LAPACK-style ranges. `Index` selects eigenpairs `il..iu` by position in the
ascending spectrum (LAPACK `range = 'I'`). `Value` selects eigenpairs with \f$v_l < \lambda \le v_u\f$
(LAPACK `range = 'V'`). Both work from C++ and Python on `Direct` and `DirectSubset`, for every
instantiated scalar type. Interior ranges on the iterative paths are deferred. The algorithms are on
@ref design_syevx.

## syevx range: starting point

`stebz` (`src/extensions/stebz.cc`) already supports all three LAPACK ranges. A Value range becomes an
index range per item through two Sturm counts, and the count is written to `m[bid]`. Sturm counts are
exact (sign changes of the LDL pivots), so \f$m = \mathrm{count}(v_u) - \mathrm{count}(v_l)\f$ is correct
even inside a tight cluster, and an empty interval gives \f$m = 0\f$.

An interior index range costs the same as the extremal one on both dense paths. `stebz`, `stein` and the
back-transforms scale with \f$k\f$, not with the block's position.

## syevx range: the count and the output contract

For a Value range the count is known only after the reduction, and it differs per item. Three options were
weighed:

- **Count, then solve in two calls.** Rejected. For dense input the count needs the full \f$O(n^3)\f$
  reduction, and caching it needs a stateful handle.
- **Per-item variable layout.** Rejected. It breaks the fixed `n x k` layout of `MatrixView`.
- **Caller-declared capacity with a per-item true count.** Chosen, as LAPACK does for `RANGE = 'V'`. The caller
  passes `neigs` as the per-item capacity. The library writes \f$\min(m[b], \mathrm{neigs})\f$ eigenpairs, lowest
  first, and reports the true \f$m[b]\f$. The library never truncates silently: \f$m[b] > \mathrm{neigs}\f$ is the
  overflow signal.

### syevx range: the m output and what neigs means

| `select` | `neigs` means | `m[b]` is |
|---|---|---|
| `Extremal` | number wanted | always `neigs` |
| `Index` | must equal `iu - il + 1` | always `neigs` |
| `Value` | capacity of `W` and `V` per item | the true count; may exceed `neigs` |

`W` past `m[b]` is untouched. `V` past `m[b]` is exactly zero. This asymmetry is deliberate: the `DirectSubset`
back-transforms run over a uniform column count and need inert columns, and both dense paths must agree. `W` has no
such consumer, so its tail stays usable as a "was this slot written" sentinel.

Explicit ranges return ascending order (LAPACK). `SyevxParams::order` overrides it.
`Extremal` keeps descending order under `find_largest`. `stein` requires ascending input, so any descending output
comes from a reversal in the finalize kernel.

## syevx range: API design

### syevx range: SyevxSelect rather than EigenRangeType

`SyevxSelect { Extremal, Index, Value }` (`include/batchlas/blas/enums.hh`). `Extremal` is the historical default. It
is normalized to an Index range internally, but it stays a separate value so no caller depends on a changed default.
`EigenRangeType` is not reused, because its default `All` means every eigenvalue.

`SyevxParams` gains `select`, `il`, `iu` (inclusive, 0-based; `iu = -1` means \f$n-1\f$), `vl`, `vu` (the interval
\f$(v_l, v_u]\f$), `abstol` (non-positive means \f$\epsilon\|T\|\f$) and `order`. All have defaults. `vl`, `vu` and
`abstol` are `float_type`, since eigenvalues are real.

### syevx range: overloads

`syevx`, `syevx_direct` and `syevx_direct_subset` gain an overload that takes `Span<int32_t> m` after `W`. Overloads
without `m` remain, and are legal only when the count is static (`Extremal` and `Index`). With `Value` they throw
`std::invalid_argument` on the host.

- The `m` overload is told apart by parameter 5 (`size_t neigs` against `Span<std::byte>` or `JobType`). Position 4
  does not discriminate, because `Span` converts implicitly from `T&`.
- A bare `{}` picks a positional overload. Each new overload takes `params` in the same trailing position, and the
  tests call each one with `{}`.
- Do not edit exported overloads in place. The `m`-less ones are inline forwarders, and
  `SolverEntryPointsKeepTheirMLessOverloads` fails if they go missing.

## syevx range: normalization and validation

`syevx_resolve_range(n, neigs, params)` returns a `SyevxResolvedRange { value_range, il, iu, vl, vu, max_count, reverse }`.
It is the single normalization point for every solve and every `buffer_size`.

| Input | Result |
|---|---|
| `Extremal`, `find_largest = true` | `il = n-neigs`, `iu = n-1`, `reverse = true` |
| `Extremal`, `find_largest = false` | `il = 0`, `iu = neigs-1`, `reverse = false` |
| `Index` | `il`, `iu` (`iu < 0` means \f$n-1\f$), `reverse = (order == Descending)` |
| `Value` | `vl`, `vu`, `max_count = neigs`, `reverse = (order == Descending)` |

`reverse` for `Extremal` comes from `find_largest`, and `order` is ignored.

Validation (`validate_syevx_range_params`) throws for: Index with \f$il < 0\f$, \f$iu \ge n\f$, \f$il > iu\f$, or
`neigs != iu - il + 1`; Value with \f$v_l \ge v_u\f$; and Value on an overload without `m`. Value with \f$v_l \ge v_u\f$
throws rather than returning an empty answer, because a swapped pair costs a full \f$O(n^3)\f$ reduction.

Not enforced, deliberately:

- **Extremal with a contradicting `order`.** `SortOrder` has no "unset" value, so an explicit `Ascending` looks like the
  default. Python sends every field on every call, so rejecting this would break `bl.syevx(a, k)`.
- **Capacity above \f$n\f$.** Clamped, not rejected.

An out-of-range or inverted Index block resolves to the empty block (`il = 0`, `iu = -1`, `max_count = 0`), so
\f$iu - il + 1 = \mathrm{max\_count}\f$ holds for every resolved range.

## syevx range: path implementations

### Direct

`Direct` is the universal fallback: dense, every scalar type, every range and `jobz`.

- Index: a copy from `src = reverse ? iu - i : il + i` of the sorted `syev` output.
- Value: \f$(v_l, v_u]\f$ maps to the block \f$[\mathrm{lower}(v_l), \mathrm{lower}(v_u))\f$, where \f$\mathrm{lower}(x)\f$ counts
  eigenvalues \f$\le x\f$. Two binary searches run in local memory, then a barrier, using the same predicate as `stebz`.

### DirectSubset

- **Workspace.** `stebz` throws if `w.size() < max_wanted`, which is \f$n\f$ for a Value range and \f$k\f$ otherwise. Every
  allocation is mirrored in `syevx_direct_subset_buffer_size` through `BumpAllocator::allocation_size<>`, in the same
  order. An exactly computed size is too small, because the allocator advances by the raw size.
- **Uniform-column back-transforms.** `stein` zeroes columns at or beyond `m[b]`. The back-transforms keep the capacity column
  count, and an orthogonal transform maps zero to zero. Cost: one item with 3 eigenpairs and another with 200 means every
  item pays for 200 columns. Use an Index range or split the batch.
- The finalize kernel reverses only the valid prefix \f$\min(m[b], \mathrm{capacity})\f$ and copies the true \f$m[b]\f$.
- **Not built:** using \f$\max_b m[b]\f$ as the column count. It needs a host sync per call. Nothing has been measured.

Interior ranges cost `DirectSubset` nothing extra. The reduction, the band width, and the reflector schedule do not depend on
which eigenvalues are wanted. The bisection takes the same number of steps at any index, and the back-transforms act on the same
\f$n \times k\f$ slice wherever the block sits.

## syevx range: routing

`syevx_select_algorithm` sends every non-extremal request to `Direct` or `DirectSubset`.

### syevx range: throw, do not degrade

- **Sparse input with a non-extremal range throws.** Returning extremal eigenpairs would be the worst outcome.
- **Dense input with explicit `LOBPCG` or `Filtered` and a non-extremal range throws.** Substituting an algorithm changes
  only performance. Substituting part of the spectrum changes the answer.
- **`BATCHLAS_SYEVX_ALGORITHM=lobpcg` or `filtered` with a non-extremal range degrades to `Direct`**, with a once-per-process
  warning. The environment wins over params so a whole suite can be forced onto one algorithm. The substitute is always `Direct`.

### syevx range: why the thresholds carry over unchanged

The measured crossovers depend on \f$(n, \mathrm{batch}, \mathrm{jobz})\f$, not on \f$k\f$, and position cannot change the cost of
either dense path. So no re-measurement was done. For a Value range the selector is fed `max_count = neigs`, the capacity, because
both paths scale with it. A routing unit test pins the decision. The timing check `BM_SYEVX_RangePosition` has not been run. If
the argument is wrong, the symptom is a slower correct answer, never a wrong one.

### syevx range: iterative paths cannot answer an interior range

- **LOBPCG** converges to whichever extreme its trial block favours. It has no \f$il/iu\f$.
- **Filtered** is a high-pass. It maps the unwanted interval into \f$[-1, 1]\f$ and lets the wanted end fall outside. An interior
  interval has unwanted spectrum on both sides, which the construction cannot express.

Neither would fail on an interior request. Both would silently answer a different question. `Auto` never routes there, and both
entry points reject a non-extremal range themselves.

## syevx range: stein workspace scaling

`stein` allocates five length-\f$n\f$ scratch arrays and pivot flags per (batch item, vector), because each work-item owns a private
tridiagonal LU. That is \f$21\,nk \cdot \mathrm{batch}\f$ bytes for float. A Value caller who passes `neigs = n` at \f$n = 1024\f$,
batch 128, float, needs about 2.8 GB of scratch on a 24 GB card, plus 0.5 GB for A.

1. **Done:** document it and require a realistic capacity, in the C++ declarations and in `bl.syevx_range`.
2. **Open:** index the scratch by resident work-item rather than by \f$(b, j)\f$. Scratch would fall from \f$O(nk \cdot \mathrm{batch})\f$
   to \f$O(n \cdot \mathrm{resident})\f$ and benefit every `stein` caller.

## syevx range: stein's per-item count bound

`stein` takes one uniform \f$k\f$. With a Value range, item \f$b\f$ may have \f$m[b] < k\f$ valid eigenvalues, and the rest holds stale workspace.

- **Phase 1 needs the bound.** Without it, inverse iteration runs on garbage shifts. The `kb` bound keeps them out.
- **Phase 2 cannot corrupt valid columns.** Modified Gram-Schmidt writes column \f$j\f$ and reads only \f$i < j\f$, and `cluster_start`
  depends only on \f$w(0..j)\f$. Its bound is for cost: without it a stale tail joins the last real eigenvalue into one bogus cluster.

Each bound was verified by removing it separately.

## syevx range: tests

### syevx range: the host reference oracle

A full host `syev`, sorted ascending, then selection by index or by \f$(v_l, v_u]\f$. The Python tests use `numpy.linalg.eigvalsh` as the same oracle.

### syevx range: the stein poisoned-tail test

`SteinTest.PerItemCountsIgnorePoisonedTail` (`tests/stein_tests.cc`) fills item 1's tail with a bogus cluster. Its first assertion, orthonormal vectors
with a small residual, does not discriminate: with the bound removed it reports zero failures. The second assertion does: item 1's unused columns,
pre-poisoned with a sentinel, must come back exactly zero. Without the bound that reports 639 failures. Keep both.

### Index and Value cases

- **Index:** an interior block (\f$n = 64\f$, \f$il = 20\f$, \f$iu = 27\f$); a block touching each end, which must be bit-identical to the `Extremal` call; \f$il = iu\f$; the
  full range; Descending on an interior block.
- **Value:** `Matrix::TriDiagToeplitz` has closed-form eigenvalues \f$d + 2\sqrt{s_l s_u}\cos(j\pi/(n+1))\f$, so counts are exact. Boundaries sit in the gaps. Cases: an empty
  interval, an interval holding everything, a batch whose items disagree (`m = {n, 0}`), and overflow (20 in the interval, `neigs = 5`: `m == 20`, the 5 lowest written, the rest untouched).
- **Rejection:** one `EXPECT_THROW` per validator rule, plus sparse with an interior range, explicit LOBPCG with an interior range, and Value on an `m`-less overload.
- Every case is at \f$n \le 128\f$ and batch \f$\le 4\f$. Larger ones carry the `slow` label.

### syevx range: Direct and DirectSubset may disagree on m by one

The two paths count by different means: a sorted search against a Sturm count. A boundary within rounding distance of an eigenvalue can make them differ by one. This is a documented tolerance.
The consistency tests put boundaries in spectral gaps. One test puts a boundary on an eigenvalue and asserts \f$|m_{\mathrm{direct}} - m_{\mathrm{subset}}| \le 1\f$.

## syevx range: Python surface

- `SyevxOptions` (`python/batchlas/_options.py`) gains `select` (default `"extremal"`), `il`, `iu`, `vl`, `vu`, `abstol` and `order`.
- `bl.syevx` returns `m` only when a range is requested, so existing unpacking is unchanged.
- `bl.syevx_range` returns per-item arrays sliced to \f$\min(m[b], \mathrm{capacity})\f$, plus `counts`, `truncated` and `capacity`. It returns lists, because a rectangular array would misstate the slots past \f$m[b]\f$.
  A Value range requires `neigs`. The only safe default is \f$n\f$, the [stein workspace](#syevx-range-stein-workspace-scaling) worst case.

## syevx range: deferred and out of scope

Deferred by design. Do not build them speculatively, because `DirectSubset` already answers interior ranges at no extra cost.

- **Folded-spectrum LOBPCG.** Apply LOBPCG to \f$(A - \sigma I)^2\f$ and recover \f$\lambda\f$ from the Rayleigh quotient of \f$A\f$. Squaring the operator halves the correct digits of the residual criterion. This is inherent, so it must be documented. Start unpreconditioned.
- **Band-pass polynomial filter.** A Chebyshev step expansion with Jackson damping in place of the high-pass. Worth building only for interior eigenpairs of a matrix too large for `DirectSubset`.
- **Non-goals:** MRRR; a `syevr` entry point (its range vocabulary is the useful half, and MRRR is the other); generalized `sygvx`; different ranges per batch item; any change to the default behaviour of an existing call.

## syevx range: what is not verified

As recorded on 2026-08-05:

- **`BM_SYEVX_RangePosition` has not been run.** The claim that the thresholds carry over is structural, backed by a routing unit test.
- **The Python layer was not compiled** (`BATCHLAS_BUILD_PYTHON=OFF` in the working build). `_options.py`, `_api.py`, `bindings/support.hh`, `bindings/ops_spectral.cc` and the new tests were syntax-checked only.
- **The benchmark was not compiled** (`BATCHLAS_BUILD_BENCHMARKS=OFF`).

```sh
# Python bindings and the range tests
cmake -S . -B build-py -DBATCHLAS_BUILD_PYTHON=ON
cmake --build build-py -j 32
python -m pytest python/tests/test_batchlas.py -k syevx

# The one benchmark point. Quiet GPU, saturation only.
cmake -S . -B build-bench -DBATCHLAS_BUILD_BENCHMARKS=ON
cmake --build build-bench -j 32 --target syevx_benchmark
./build-bench/benchmarks/syevx_benchmark --name BM_SYEVX_RangePosition
```

Reading the benchmark: `Direct` cannot depend on position, so its spread across bottom, middle and top (equal block width) is the noise floor. `DirectSubset` is flat if its spread stays inside that band. If not, the thresholds argument fails, and the `kSyevxSubsetMinN` / `kSyevxSubsetMinWork` gate in `src/extensions/syevx.cc` needs a range-aware term.

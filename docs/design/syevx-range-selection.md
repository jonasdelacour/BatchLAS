# syevx range selection: design record {#design_syevx_range}

**Covers:** LAPACK-style range selection for `syevx`: `Index` (eigenpairs `il..iu` by position in
the ascending spectrum, LAPACK `range = 'I'`) and `Value` (every eigenpair with
\f$v_l < \lambda \le v_u\f$, LAPACK `range = 'V'`). It records the one hard problem, the API chosen,
where the implementation departed from the plan, what is unverified, and what was deferred.
**Status:** implemented (plan written 2026-08-04, landed 2026-08-05). Index and Value ranges work
end to end from C++ and Python on both dense paths (`Direct`, `DirectSubset`) for every
instantiated scalar type. Interior ranges on the iterative paths are deferred by design. One
routing claim is argued structurally and its benchmark has not been run (see
[what is not verified](#syevx-range-what-is-not-verified)).
Performance of the partial eigensolve is covered by @ref perf_syevx and @ref design_syevx.

## syevx range: the tridiagonal kernel already had it

`stebz` (`src/extensions/stebz.cc`) is a Sturm-sequence bisection that supported all three LAPACK
ranges from Tier 1 on: `EigenRangeType { All, Index, Value }`, `StebzParams { range, il, iu, vl,
vu, abstol, order, max_iterations }`, host-side resolution of an Index range, device-side
conversion of a Value range to an index range per batch item via two Sturm counts
(`[count(vl), count(vu) - 1]`), and a per-item count written to `m[bid]`.

Sturm counting is the right primitive: `count(x)`, the number of eigenvalues \f$\le x\f$, is exact
(from sign changes of the LDL pivots), not a tolerance, so `m = count(vu) - count(vl)` is
combinatorially correct even when a boundary lands inside a tight cluster. Empty intervals give
`m = 0` for free.

So this was mostly plumbing plus one design decision, not a new algorithm. **An interior index
range costs the same as the extremal one on both dense paths**: `stebz` is \f$O(nk)\f$ wherever the
block sits, `stein` likewise, and the back-transforms scale with `k`, not with position.

## syevx range: what was missing above stebz

Before this work (file references as of 2026-08-04):

- `SyevxParams` had no range vocabulary, only `bool find_largest` and `neigs`: a top-k/bottom-k
  interface.
- `syevx_direct_subset` hard-coded the extremal block in a `wanted_range(n, k, find_largest)`
  helper and pinned `StebzParams::range = Index`. That was the narrowest point in the stack.
- `syevx_direct` selected by mapping output slot `i` to `find_largest ? n-1-i : i`.
- `syevx` and `syevx_buffer_size` had no `m` output.
- The Python `SyevxOptions` exposed only `find_largest`, and the binding sized outputs from `neigs`.

### syevx range: stein's per-item count bound

`stein` took one uniform `k` for the whole batch. With a Value range, item `b` may have only
`m[b] < k` valid eigenvalues, and the slots past that hold stale workspace. Phase 1 would run
inverse iteration on garbage shifts. The plan also claimed that phase 2's cluster walk would join
a real eigenvalue to a garbage neighbour and corrupt a genuine vector, making it a
silent-wrong-answer path.

**Correction (verified in the implementation):** the phase-2 claim is wrong. The modified
Gram-Schmidt writes only column `j` while reading columns `i < j`, and `cluster_start` is derived
only from `w(0..j)`. So for every `j < kb` the result depends only on valid data, and nothing
past `kb` can flow back into a valid column. The phase-2 bound is a **cost and hygiene** bound:
without it, a monotone-looking stale tail joins the last real eigenvalue into one large bogus
cluster and phase 2 degenerates into an \f$O(nk^2)\f$ barrier-synchronized pass over columns phase
1 already knows are zero. What protects the valid prefix is **phase 1's** `kb` bound, which keeps
garbage shifts out of inverse iteration. That was verified by reverting each bound separately.

### syevx range: the stein poisoned-tail test

The plan asked for a test that fills item 1's tail of `w` with values forming a bogus cluster
with its last real eigenvalue, and asserts that item 1's three vectors are orthonormal with a
small residual. **That assertion does not discriminate.** With the counts bound removed from
both phases, it reports zero failures. `SteinTest.PerItemCountsIgnorePoisonedTail`
(`tests/stein_tests.cc`) therefore also asserts that item 1's unused columns (pre-poisoned with a
sentinel) come back **exactly zero**, which reports 639 failures without the bound. Keep that
second assertion. It is what makes the test fail when the bound regresses.

### syevx range: iterative paths cannot answer an interior range

- **LOBPCG** converges to whichever *extreme* its trial block is biased toward. It has no
  `il/iu` to honour.
- **Filtered**: the Chebyshev filter is a *high-pass*. It maps the unwanted interval into
  \f$[-1, 1]\f$, where \f$|T_m| \le 1\f$, and lets the wanted **end** fall outside. An interior
  interval has unwanted spectrum on both sides, which that construction cannot express.

Neither would fail on an interior request. Both would quietly answer a different question. So
`Auto` never routes a non-extremal request to either, and an explicit request for one throws.
Both `syevx_lobpcg` and `syevx_filtered` also reject a non-extremal range themselves, because they
are public entry points too. Real interior algorithms for them are
[deferred](#syevx-range-deferred-interior-ranges-for-the-iterative-paths).

## syevx range: a data-dependent count in a batched API

For `Value`, **the number of eigenpairs is not known until the matrix has been reduced and
counted, and it differs from one batch item to the next.** Everything awkward in the design
comes from that.

### syevx range: options for the count

- **(a) Two-call protocol, count then solve.** Rejected. Counting needs the tridiagonal form,
  which for dense input is the full \f$O(n^3)\f$ reduction, so a "cheap count" costs about as much
  as the solve. Caching the tridiagonal between calls would need a stateful handle the rest of
  BatchLAS does not have.
- **(b) Per-item variable output layout** (CSR-style offsets + packed values). Rejected for now.
  It composes badly with `MatrixView`, whose eigenvector output is a fixed `n x k` per item with
  a uniform stride, and it would ripple into every consumer. Worth revisiting only if a user hits
  the [stein workspace wall](#syevx-range-stein-workspace-scaling).
- **(c) Caller-declared capacity + per-item true count.** **Chosen.** The caller accepts at most
  `neigs` per item. The library writes `min(m[b], neigs)` eigenpairs into slots
  `[0, min(m[b], neigs))` and reports the **true** `m[b]`. This is what LAPACK does (for
  `RANGE = 'V'` the caller dimensions `W` and `Z` for the worst case, because "the exact value of
  M is not known in advance"), and it is the contract `stebz` already had.

### syevx range: the overflow policy

If `m[b] > neigs`, the request cannot be answered in full. Throwing is impossible (the count is
only known on the device). Silent truncation is unacceptable: it is the "silently returned wrong
numbers" failure the option-struct traps in this repo have produced twice. **Chosen: report and
let the caller check.** `m[b]` carries the true count, so `m[b] > neigs` is the overflow signal,
one comparison per item with no extra sync beyond reading `m`, which a Value caller must read
anyway. Truncation is deterministic: **keep the lowest `neigs` of the interval, ascending.**

### syevx range: output order

`find_largest` keeps implying descending order, and every explicit Index/Value range defaults to
**ascending** (LAPACK's order), with `SyevxParams::order` to override. No existing caller's
output changes. Internal invariant: `stein`'s cluster detection requires **ascending** input, so
any descending output is produced by the reversal in the finalize kernel, never by asking `stebz`
for descending mid-chain. `params.order` is honoured only by that finalize kernel.

## syevx range: API design

### syevx range: SyevxSelect rather than EigenRangeType

`SyevxSelect { Extremal, Index, Value }` (`include/batchlas/blas/enums.hh`), where `Extremal` is
the historical default: `neigs` from one end chosen by `find_largest`. It is a special case of
Index and is normalized to one internally, but it is a distinct value so that no existing
caller's behaviour depends on a default that changed meaning.

`EigenRangeType` was not reused because its natural default `All` means "every eigenvalue",
which is not what `syevx`'s default does. Making `All` mean "the `neigs` extremal ones" inside
`SyevxParams` would give a member whose meaning depends on the struct it sits in.
`EigenRangeType` stays the tridiagonal-layer vocabulary, and normalization converts between the
two in one place.

### syevx range: SyevxParams members

`select`, `il`, `iu` (inclusive, 0-based, into the ascending spectrum; `iu = -1` means `n - 1`),
`vl`, `vu` (the half-open interval \f$(v_l, v_u]\f$, matching LAPACK), `abstol` (forwarded to
`StebzParams::abstol`; non-positive means \f$\epsilon\|T\|\f$; ignored by paths that get
eigenvalues from a full solve), and `order`. All are defaulted, so existing aggregate
initialization by field name is unaffected.

`vl`, `vu` and `abstol` are the **real** type (`float_type`), not `T`: eigenvalues of a Hermitian
matrix are real and `W` is already `Span<base_type<T>>`. Using `T` would force complex callers to
write `std::complex<float>(vl)` for a real quantity. The older `absolute_tolerance` /
`relative_tolerance` members are typed `T`, which is a pre-existing wart that was deliberately
not propagated.

### syevx range: the m output and what neigs means

`syevx` (and `syevx_direct`, `syevx_direct_subset`) gained an overload taking `Span<int32_t> m`
right after `W`. Every existing overload stays. **The `m`-less overloads are legal only when the
count is statically known** (Extremal and Index, where `m[b] == neigs` by construction). Calling
one with `Value` throws `std::invalid_argument` on the host, before device work. So no existing
call site can reach the new path, and no new call site can ignore a count it needs.

| `select` | `neigs` means | `m[b]` is |
|---|---|---|
| `Extremal` | number wanted (unchanged) | always `neigs` |
| `Index` | must equal `iu - il + 1` | always `neigs` |
| `Value` | **capacity** of `W` and `V` per item | true count, may exceed `neigs` |

Validating `neigs == iu - il + 1`, rather than deriving `neigs`, makes a mismatched pair a loud
host error instead of a silently under- or over-filled buffer. The `syevx` declarations in
`include/batchlas/blas/extensions.hh` state the capacity meaning of `neigs`. Their old
`@param neigs` "Number of eigenvalues to compute" (the most misleading line once capacity
semantics landed) and the `@brief` "of a sparse matrix" (wrong since dense input routes to
Direct/DirectSubset) were corrected while the routing landed.

**`W` past `m[b]` is untouched, and `V` past `m[b]` is exactly zero.** The asymmetry is
deliberate and documented at the declaration. The subset path's back-transforms run over a
uniform column count and need something inert there, and since `Auto` routes between the two
dense paths on `(n, batch)` alone, both paths must agree. `W` has no such consumer, so its tail
stays usable as a "was this slot written" sentinel.

### syevx range: overload traps

- **Unconstrained variadic overloads**: the new `m` overload is a plain distinct signature. The
  plan argued it is told apart by `Span<int32_t>` vs `size_t` in position 4. **That mechanism is
  wrong**: `Span` has a non-explicit `Span(T&)` constructor, so `Span<int32_t>` *is* implicitly
  constructible from an `int32_t` lvalue. What discriminates is parameter **5**: `size_t neigs`
  against `Span<std::byte>` / `JobType`, which are mutually non-convertible. The conclusion (the
  forms are unambiguous) survived, and the code comment states the correct reason.
- **Option-struct overloads**: a bare `{}` for params previously picked a positional overload and
  silently returned different numbers. Every new overload takes `params` in the same trailing
  position with the same default, and the tests make one call per overload with `{}`.
- **Do not edit exported overloads in place.** The `m`-less `syevx_direct` /
  `syevx_direct_subset` overloads were briefly edited in place rather than added alongside. That
  broke source and ABI compatibility for two exported symbols nothing in-tree calls, which is why
  it was invisible. They were restored as inline forwarders distinguished by arity, and
  `SolverEntryPointsKeepTheirMLessOverloads` fails if they go missing again.

## syevx range: normalization and validation

`syevx_resolve_range(n, neigs, params)` produces a `SyevxResolvedRange { value_range, il, iu,
vl, vu, max_count, reverse }`. It is **the single normalization point, shared by every solve and
every `buffer_size`**, so the two cannot disagree about what was asked for or how large the
workspace must be. This is the same discipline as `syevx_select_algorithm`.

| input | output |
|---|---|
| Extremal, `find_largest = true` | `il = n-neigs`, `iu = n-1`, `reverse = true` |
| Extremal, `find_largest = false` | `il = 0`, `iu = neigs-1`, `reverse = false` |
| Index | `il`, `iu` (`iu < 0` -> `n-1`), `reverse = (order == Descending)` |
| Value | `vl`, `vu`, `max_count = neigs`, `reverse = (order == Descending)` |

For Extremal, `reverse` comes from `find_largest`, **not** `order`, and `order` is ignored.

Host-side throws (`validate_syevx_range_params`, a sibling of the preconditioner validator):
Index with `il < 0`, `iu >= n` or `il > iu`; Index with `neigs != iu - il + 1`; Value with
`vl >= vu`; Value on an `m`-less overload. `vl >= vu` could be read as a valid `m = 0` answer, but
a caller who writes it has almost certainly swapped the arguments, and being wrong costs a full
\f$O(n^3)\f$ reduction that returns nothing, so it throws and says which.

Departures from the plan:

- **Index gained a clamp as well as the throw.** The throw tells the caller about a mistake. The
  clamp makes the "`max_count` is already clamped to `n`" claim true for any future consumer that
  indexes `[il, il + max_count)` without its own bound. An out-of-range or inverted Index block
  resolves to the canonical empty block (`il = 0`, `iu = -1`, `max_count = 0`), so
  `iu - il + 1 == max_count` holds for every resolved range.
- **"Extremal + a contradicting `order` throws" was not implemented, deliberately.** The Python
  `SyevxOptions` sends every field on every call, so an ordinary `bl.syevx(a, k)` carries
  `order = "ascending"` next to `find_largest = True` and would have started throwing. Ignoring
  `order` for Extremal keeps "no existing caller's output changes" true.
- A capacity above `n` is clamped, not rejected.

## syevx range: Direct selection

`Direct` is the **universal fallback**: dense, every scalar type including complex, every range,
every `jobz`. It was built first, so `Auto` always has somewhere correct to send an interior
request, and it gave the subset path a reference to test against.

- Index: `syev` already produced the ascending spectrum, so selection is a copy with
  `src = reverse ? iu - i : il + i`.
- Value: `(vl, vu]` maps to the contiguous block `[lower(vl), lower(vu))` of the sorted
  eigenvalues, where `lower(x)` counts eigenvalues \f$\le x\f$. That is two binary searches by one
  work-item into local memory followed by a barrier, the same shape `stebz` uses. The predicate
  matches `stebz`'s exactly (\f$\lambda \le x\f$ counts), so both dense paths return the same `m`
  for the same input except within rounding distance of a boundary (see
  [the cross-path tolerance](#syevx-range-direct-and-directsubset-may-disagree-on-m-by-one)).
- `syevx_direct_buffer_size` sizes on `n` and batch only (a full `syev` plus a private copy of
  A), so it needed no change.

## syevx range: DirectSubset

`wanted_range` was replaced by the resolver. `StebzParams` takes the resolved `il/iu` or `vl/vu`
and `abstol`, with `order` **always** Ascending, because `stein` requires it.

- **Workspace.** `stebz` throws if `w.size() < max_wanted`, and for a Value range
  `max_wanted = n`. So the `w_sub` allocation is `n * batch` reals for a Value range and
  `k * batch` otherwise, negligible next to the `n^2 * batch` copy of A. The identical expression
  appears in `syevx_direct_subset_buffer_size`. Every allocation is mirrored there through
  `BumpAllocator::allocation_size<>` in the same order: an exactly computed size is too small.
- **Uniform-column back-transforms.** `m` is passed to `stein`, which zeros columns at or beyond
  `m[b]`. `unmqr_hb2st` and `ormqr_blocked` stay at the uniform capacity column count. An
  orthogonal transform maps zero to zero, so nothing propagates, and both back-transforms stay
  shape-uniform, which is what makes them fast. The cost is real: if one item finds 3 and another
  200, every item pays for 200 columns. A caller who cares should use an Index range or split the
  batch.
- **Not built: the `max_m` refinement.** Using `max_b m[b]` as the column count needs `max_m` on
  the host to shape the `ormqr` call, which is one sync per call, the same defect as
  [LOBPCG's per-iteration host synchronization](../perf/syevx.md#lobpcg-host-synchronization-only-at-convergence-checks).
  The plan said to measure first. Nothing has been measured, so nothing changed.
- The finalize kernel reverses only the valid prefix `min(m[b], capacity)` and copies the true
  `m[b]` to the caller.

### syevx range: why interior ranges cost DirectSubset nothing

`kd` selection, `sy2sb`, `sb2st_hh`, the phase vector and the reflector schedule do not depend on
which eigenvalues are wanted. The bisection does the same number of steps for every index, and
both back-transforms act on the same fixed `n x k` slice wherever the block sits.

## syevx range: routing rules

`syevx_select_algorithm` takes the `SyevxSelect` and enforces that `select != Extremal` resolves
only to Direct or DirectSubset.

### syevx range: throw, do not degrade

- **Sparse input + non-extremal range throws.** LOBPCG is the only default sparse path and cannot
  answer. Returning the extremal eigenpairs instead would be the worst available outcome.
- **Dense + explicit `method = LOBPCG` or `Filtered` + non-extremal range throws**, naming the
  limitation. The precedent right below it in the dispatcher *degrades* an unavailable algorithm
  to its nearest neighbour. That precedent deliberately does not apply: substituting an algorithm
  changes only the performance the caller asked for, while substituting the requested part of the
  spectrum changes the answer.
- **`BATCHLAS_SYEVX_ALGORITHM=lobpcg` (or `filtered`) + non-extremal range degrades to Direct**,
  with a once-per-process warning. The environment wins over params, and it exists so that a
  whole suite can be forced onto one algorithm for diagnosis. Aborting on the first interior call
  would make that sweep impossible. `syevx_select_preconditioner` degrades an environment default
  for the same reason. The substitute is Direct, not the shape heuristics, because a diagnostic
  sweep wants one substitute.

### syevx range: why the thresholds carry over unchanged

The measured crossovers are functions of `(n, batch, jobz)` and explicitly not of `k` (flat from
0.8 % to 25 % of the spectrum, see
[the DirectSubset batch crossover](../perf/syevx.md#syevx-directsubset-batch-crossover-with-eigenvectors)).
Position within the spectrum cannot enter the cost of either dense path: Direct always runs a
full `syev` and copies a block, and
[DirectSubset's cost does not depend on position](#syevx-range-why-interior-ranges-cost-directsubset-nothing).
So **no re-measurement was done for Index or Value ranges**. For Value ranges the selector is fed
`max_count = neigs` (the capacity), the conservative choice, because capacity is what both paths
do work proportional to. A routing unit test pins the *decision*. The timing check
(`BM_SYEVX_RangePosition`) exists but has not been run (see below). If the argument is wrong, the
symptom is a mis-routed interior request: a slower correct answer, never a wrong one.

## syevx range: test design

The suites extend the existing per-algorithm parameterized tests in `tests/syevx_tests.cc`, which
check against a reference `syev` and skip when the environment forces another algorithm.

### syevx range: the host reference oracle

One helper shared by every range test: run a full `syev` on the host, sort ascending, then select
by index or by \f$(v_l, v_u]\f$ in plain host code. It is trivially correct, which is what makes it
a trustworthy reference for four solver paths. The Python tests use `numpy.linalg.eigvalsh` as
the same oracle.

### syevx range: the index and value cases

- **Index:** an interior block (`n = 64`, `il = 20`, `iu = 27`); a block touching each end, which
  must be **bit-identical** to the Extremal call (same path, same inputs, so any difference is a
  normalization bug); `il == iu`; the full range, which must match `syev`; Descending on an
  interior block.
- **Value:** `Matrix::TriDiagToeplitz` has closed-form eigenvalues
  \f$d + 2\sqrt{s_l s_u}\cos(j\pi/(n+1))\f$, \f$j = 1..n\f$, so the exact count in any interval is
  known; boundaries go in the *gaps*. Also an empty interval (`m == 0`, nothing written), an
  interval containing everything (`m == n`), **a batch whose items disagree** (`m = {n, 0}`, run
  with eigenvectors), and **capacity overflow** (20 in the interval, `neigs = 5`: `m == 20`, the 5
  lowest written, slot 5 onward untouched, checked with a sentinel).
- **Rejection:** one `EXPECT_THROW` per validator rule, plus sparse + interior, explicit LOBPCG +
  interior, and Value on an `m`-less overload.
- Cost control: every case at `n <= 128` and `batch <= 4`, with anything larger behind the `slow`
  label.

### syevx range: Direct and DirectSubset may disagree on m by one

For the same `(A, range)`, Direct and DirectSubset must agree on `m` and every eigenvalue. They
compute `m` by different means (sorted search vs Sturm count), so a boundary within rounding
distance of an eigenvalue can make them differ by one. That is a documented tolerance, not a bug.
The consistency tests put boundaries in spectral gaps, and one test deliberately puts a boundary
on an eigenvalue and asserts only `|m_direct - m_subset| <= 1`.

## syevx range: Python surface

`SyevxOptions` (`python/batchlas/_options.py`) gained `select` (default `"extremal"`), `il`, `iu`,
`vl`, `vu`, `abstol`, `order`. `bl.syevx` returns `m` as an extra element **only** when a range
was requested, so no existing caller's unpacking changes. `bl.syevx_range` is the NumPy-shaped
wrapper: per-item arrays already sliced to `min(m[b], capacity)`, plus `counts`, `truncated` and
`capacity`. It returns lists because a rectangular array cannot express a ragged Value answer
without lying about the slots past `m[b]`. For a Value range `neigs` is a required capacity: there
is no safe default short of `n`, and `n` is the
[stein workspace worst case](#syevx-range-stein-workspace-scaling). The binding reuses the
existing queue plumbing, because a per-call `Queue` defeats the workspace arena.

## syevx range: deferred interior ranges for the iterative paths

Deferred by design. Both are real algorithms, not plumbing, and DirectSubset answers interior
ranges at no extra cost, which removes most of the motivation. **Do not build them
speculatively.**

### syevx range: folded-spectrum LOBPCG

The only way `syevx` would answer an interior query on CSR input. Apply LOBPCG to
\f$(A - \sigma I)^2\f$, whose smallest eigenvalues are those of A closest to \f$\sigma\f$, and recover
\f$\lambda\f$ from the Rayleigh quotient of A (not of the squared operator) so the eigenvalues come
back at full accuracy. It costs two matvecs per application. Squaring the operator squares the
condition number and roughly **halves the correct digits of the residual criterion**, which is
inherent and must be documented, not tuned away. It fits the existing matvec abstraction. The
ILU(k) and Jacobi validity arguments are framed around approximating \f$A^{-1}\f$ for the smallest
eigenpairs and do not transfer to \f$(A - \sigma I)^2\f$, so it should start unpreconditioned.

### syevx range: band-pass polynomial filter

For interior slices, `syevx_filtered`'s high-pass construction would be replaced by a polynomial
approximation to the indicator of \f$[v_l, v_u]\f$ (the EVSL approach): a Chebyshev expansion of a
step function with Jackson damping against Gibbs oscillation, or the least-squares/Zolotarev
family. Only worth building if a user needs interior eigenpairs of a matrix too large for
DirectSubset, a constituency that may be empty.

## syevx range: stein workspace scaling

`stein_buffer_size` (`src/extensions/stein.cc`) allocates five length-`n` scratch arrays plus
pivot flags **per (batch, vector)**, because each work-item owns a private tridiagonal LU:
`5 * sizeof(T) * n * k * batch + n * k * batch` bytes, that is \f$21\,nk\cdot\mathrm{batch}\f$
bytes for float. With an Index range `k` is what the caller wants, which is fine. With a Value
range `k` is the capacity, and a defensive caller who passes `neigs = n` at `n = 1024`,
`batch = 128`, float, gets about 2.8 GB of scratch on a 24 GB card, on top of the
\f$n^2 \cdot \mathrm{batch}\f$ copy of A (0.5 GB) and V itself.

Responses, in order of preference:

1. Document it and make callers pass a realistic capacity. That is done, in the C++ declaration
   and in `bl.syevx_range`.
2. **Re-index `stein`'s scratch by work-item rather than by `(b, j)`.** The number of
   concurrently resident work-items is bounded by the launch geometry, not by `k * batch`, so
   this would cut scratch from \f$O(nk\cdot\mathrm{batch})\f$ to \f$O(n \cdot \mathrm{resident})\f$
   and benefit every `stein` caller, including the extremal path. It is a contained change to one
   kernel and the right fix. **Open**, not a blocker, and not to be lost.
3. Per-item packed output (option (b) above), only if 1 and 2 are insufficient.

## syevx range: risks, ranked

1. **`stein` over invalid slots.** Mitigated by per-item counts landing before the DirectSubset
   range work, and by the disagreeing-batch test. The severity was overstated (see
   [the stein bound correction](#syevx-range-steins-per-item-count-bound)), but the bound is still
   needed for phase 1.
2. **Capacity semantics silently under-filling output.** Mitigated by the `m`-less overload
   throwing for Value and by the sentinel test on untouched slots.
3. **Direct and DirectSubset disagreeing on `m` by one.** Inherent. Matched predicates, a
   documented tolerance and an explicit test.
4. **`BumpAllocator` under-sizing the new Value-range allocation.** The sizing function is far
   from the allocation it mirrors. Mitigated by mirroring through `allocation_size<>` and a test
   that runs a Value range through the real `buffer_size` path with an exactly sized buffer.
5. **`Auto` routing an interior request to a path that cannot answer.** Mitigated by the
   throw-not-degrade rule and the rejection tests.
6. **Scope creep into the iterative interior algorithms.** Deferred explicitly.

## syevx range: non-goals

- MRRR: the bisection-over-MRRR argument in @ref design_syevx is unchanged by range selection.
- A `syevr` entry point. The range vocabulary is the useful half of `syevr`'s API, and the other
  half is MRRR.
- Generalized `sygvx` ranges.
- Different ranges per batch item (item 0 wants `[0, 5]`, item 1 wants `[10, 20]`). `stebz` takes
  scalar `il/iu`, so this would need a device-side range vector through every layer. No demand.
- Changing the default behaviour of any existing call.

## syevx range: what is not verified

As recorded on 2026-08-05:

- **`BM_SYEVX_RangePosition` has never been run.** So the claim that
  [the thresholds carry over unchanged](#syevx-range-why-the-thresholds-carry-over-unchanged) is
  a structural argument plus a routing unit test that pins the decision, not a timing.
- **The Python layer was not verified by compilation.** `BATCHLAS_BUILD_PYTHON` was `OFF` in the
  working build, so `_options.py`, `_api.py`, `bindings/support.hh`, `bindings/ops_spectral.cc`
  and the new tests in `python/tests/test_batchlas.py` were syntax-checked only.
- **The benchmark was not verified by compilation** (`BATCHLAS_BUILD_BENCHMARKS=OFF`).

The commands that would close these:

```sh
# Python bindings + the range tests
cmake -S . -B build-py -DBATCHLAS_BUILD_PYTHON=ON
cmake --build build-py -j 32
python -m pytest python/tests/test_batchlas.py -k syevx

# The one benchmark point. Quiet GPU, saturation only.
cmake -S . -B build-bench -DBATCHLAS_BUILD_BENCHMARKS=ON
cmake --build build-bench -j 32 --target syevx_benchmark
./build-bench/benchmarks/syevx_benchmark --name BM_SYEVX_RangePosition
```

Reading the benchmark: `Direct` (algorithm 1) is the control and cannot depend on position, so
its spread across the three positions (bottom, middle, top, identical block width) is the noise
floor. DirectSubset (algorithm 2) is flat if its spread is inside that band. If it is not, the
thresholds argument is wrong and the `kSyevxSubsetMinN` / `kSyevxSubsetMinWork` gate in
`src/extensions/syevx.cc` needs a range-aware term.

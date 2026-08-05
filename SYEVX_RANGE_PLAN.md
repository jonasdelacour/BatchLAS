# SYEVX Range Selection: Implementation Plan

Status: **phases 1-7 implemented; phase 8 deferred by design (see §12 and §17).**
Index and value ranges work end to end from C++ and from Python, on both dense
paths, for every instantiated scalar type. One claim the routing rests on is
argued structurally and has a benchmark written for it that has not been run;
§17 says exactly which. Companion to [SYEVX_PLAN.md](SYEVX_PLAN.md),
which covers *performance* of the partial eigensolve. This document covers the
*capability* gap: BatchLAS `syevx` can currently return only the top-`k` or
bottom-`k` of the spectrum, while a LAPACK-conformant `?syevx` also offers

* `range = 'I'` — eigenpairs `il..iu` by position in the ascending spectrum, and
* `range = 'V'` — every eigenpair with `vl < λ ≤ vu`.

---

## Table of contents

1. [The headline: most of this already exists](#1-the-headline)
2. [Exactly what is missing, per file](#2-what-is-missing)
3. [The one hard problem: a data-dependent count in a batched API](#3-the-hard-problem)
4. [API design](#4-api-design)
5. [Phase 1 — vocabulary, normalization, validation](#5-phase-1--vocabulary-normalization-validation)
6. [Phase 2 — `syevx_direct`: the universal fallback](#6-phase-2--syevx_direct)
7. [Phase 3 — `stein` per-item counts](#7-phase-3--stein-per-item-counts)
8. [Phase 4 — `syevx_direct_subset`: the fast path](#8-phase-4--syevx_direct_subset)
9. [Phase 5 — routing and the public entry point](#9-phase-5--routing-and-the-public-entry-point)
10. [Phase 6 — tests](#10-phase-6--tests)
11. [Phase 7 — bindings, benchmarks, docs](#11-phase-7--bindings-benchmarks-docs)
12. [Phase 8 (later) — interior ranges for the iterative paths](#12-phase-8-later--interior-ranges-for-the-iterative-paths)
13. [Workspace scaling: the `stein` blow-up](#13-workspace-scaling)
14. [Risks, ranked](#14-risks-ranked)
15. [Non-goals](#15-non-goals)
16. [Sequencing and effort](#16-sequencing)
17. [What actually landed](#17-what-actually-landed)

---

## 1. The headline

**The tridiagonal kernel already does everything asked for.** `stebz`
(`src/extensions/stebz.cc`) is a Sturm-sequence bisection that supports all three
LAPACK ranges today:

* `include/blas/extensions.hh:603` — `EigenRangeType { All, Index, Value }`
* `include/blas/extensions.hh:613` — `StebzParams { range, il, iu, vl, vu, abstol, order, max_iterations }`
* `src/extensions/stebz.cc:86-93` — host-side resolution of an `Index` range
* `src/extensions/stebz.cc:155-184` — device-side conversion of a `Value` range into
  an index range via two Sturm counts, `[count(vl), count(vu)-1]`, **per batch item**
* `src/extensions/stebz.cc:216-218` — per-item count written to `m_ptr[bid]`

Sturm counting is exactly the right primitive here: `count(x)` is the number of
eigenvalues `≤ x`, computed exactly (not to within a tolerance) from sign changes of
the LDL pivots, so `m = count(vu) − count(vl)` is *combinatorially* correct even when
a boundary lands inside a tight cluster. Empty intervals fall out for free
(`count_wanted = 0`, loop does not execute, `m = 0`).

SYEVX_PLAN.md:749 already recorded the intent — Tier 1 was specified with
"`IndexRange` (`il..iu`) and `ValueRange` (`vl..vu`) selection" — and Tier 1 shipped
with both. What never happened is exposing them above the tridiagonal layer.

The consequence for effort: **this is mostly plumbing plus one genuinely new design
decision** (§3). It is not a new algorithm. The cost of an interior index range is
*identical* to the cost of the equivalent extremal range on both dense paths —
`stebz` is `O(n·k)` wherever the block sits, `stein` likewise, and the back-transforms
scale with `k` and not with position. Interior support on `DirectSubset` is
effectively free.

---

## 2. What is missing

### 2.1 `SyevxParams` has no range at all

`include/blas/extensions.hh:33-77`. The entire selection vocabulary is:

```cpp
bool find_largest = true;   // line 39
```

plus the positional `size_t neigs`. That is a top/bottom-`k` interface and nothing
more.

### 2.2 `syevx_direct_subset` hardcodes the extremal block

`src/extensions/syevx_direct_subset.cc:77-84`:

```cpp
struct WantedRange { int64_t il; int64_t iu; };

inline WantedRange wanted_range(int64_t n, int64_t k, bool find_largest) {
    return find_largest ? WantedRange{n - k, n - 1} : WantedRange{0, k - 1};
}
```

and `:242-246` pins `bp.range = EigenRangeType::Index`. **This function is the single
narrowest point in the whole stack.** Generalizing it and threading a resolved range
into `bp` is the bulk of the work on the fast path.

### 2.3 `syevx_direct` selects by reversing the ascending order

`src/extensions/syevx_direct.cc:111-125` — the selection kernel maps output slot `i`
to source index `find_largest ? (n-1-i) : i`. An index range is a one-line change to
that mapping (`src = il + i`, with an optional reverse). A value range needs a new
device-side step: a search over the already-sorted eigenvalues for the half-open
interval `(vl, vu]`.

### 2.4 `stein` takes one uniform `k` for the whole batch

`src/extensions/stein.cc:112` (`size_t k_in`) and `:282-284`:

```cpp
for (int64_t j = 0; j < k; ++j) {
    if (j > 0 && (w(j, bid) - w(j - 1, bid)) > gap_tol) cluster_start = j;
```

Phase 1 runs inverse iteration on all `k` slots; phase 2 walks all `k` consecutively
for cluster detection. With a value range, item `b` may have only `m[b] < k` valid
eigenvalues and the remaining slots hold whatever the workspace last contained.
Phase 1 would iterate on garbage shifts, and — worse — phase 2's cluster walk would
join a real eigenvalue to a garbage neighbour and reorthogonalize a genuine vector
against noise. **This is a silent-wrong-answer path, not a crash path.** Fixing it
requires an optional per-item count.

### 2.5 Iterative paths cannot do interior at all

* `syevx_lobpcg` — LOBPCG's Rayleigh–Ritz converges to whichever *extreme* the trial
  block is biased toward. There is no `il/iu` for it to honour.
* `syevx_filtered` — `src/extensions/syevx_filtered.cc:10-21`: the Chebyshev filter is
  a *high-pass*, built by mapping the unwanted interval into `[-1,1]` where `|T_m| ≤ 1`
  and letting the wanted **end** fall outside. An interior interval has unwanted
  spectrum on *both* sides, which that construction cannot express.

Both need real new algorithms (§12), so the near-term answer is that `Auto` must
never route an interior request to either, and an explicit request for one must be
rejected rather than silently answered with the wrong eigenpairs.

### 2.6 `syevx` and `syevx_buffer_size` have no `m` output

`include/blas/extensions.hh:259-336`. There is nowhere to report how many eigenpairs
were actually found.

### 2.7 Python bindings

`python/batchlas/_options.py:18-36` (`SyevxOptions`) exposes `find_largest` only;
`python/batchlas/bindings/ops_spectral.cc:590-620` sizes outputs from `neigs`
unconditionally.

---

## 3. The hard problem

For `range = Value`, **the number of eigenpairs is not known until the matrix has been
reduced and counted, and it differs from one batch item to the next.** Everything
awkward in this plan descends from that one fact.

### 3.1 Options considered

**(a) Two-call protocol — count, then solve.** Rejected. Counting requires the
tridiagonal form, which for dense input means the full `O(n³)` reduction. A "cheap
count" pass would cost essentially the same as the solve, and caching the tridiagonal
between calls would mean a stateful handle that the rest of BatchLAS does not have.

**(b) Per-item variable output layout (CSR-style, offsets + packed values).**
Rejected for now. It composes badly with `MatrixView`, whose eigenvector output `V`
is a fixed `n × k` per item with a uniform stride, and it would ripple into every
consumer. Worth revisiting only if a user actually hits the memory wall in §13.

**(c) Caller-declared capacity + per-item true count.** **Chosen.** The caller says
"I will accept at most `neigs` per item"; the library writes `min(m[b], neigs)`
eigenpairs into slots `[0, min(m[b], neigs))` of item `b` and reports the **true**
`m[b]`. Slots at or beyond that index are left undefined.

This is exactly what LAPACK does — for `RANGE='V'` LAPACK requires the caller to
dimension `W` and `Z` for the worst case (`N`) because "the exact value of M is not
known in advance" — and it is exactly the contract `stebz` already implements with its
`Span<int32_t> m` output. Reusing it keeps the whole stack consistent.

### 3.2 The overflow policy

If `m[b] > neigs` for some `b`, the request cannot be answered. Three sub-options:

* throw — impossible, the count is only known on the device;
* silently truncate — unacceptable, this is the "silently returned wrong numbers"
  failure mode the option-struct traps in this repo have already produced twice;
* **report and let the caller check.** Chosen: `m[b]` carries the *true* count, so
  `m[b] > neigs` is the caller's overflow signal, detectable with one comparison per
  item and no extra sync beyond reading `m` (which a `Value`-range caller must read
  anyway).

Document it in one sentence at the `syevx` declaration, and make the truncation
deterministic: **keep the lowest `neigs` of the interval, in ascending order.**

### 3.3 Ordering

`stebz` returns ascending by default and `SortOrder::Descending` reverses within the
selected block (`src/extensions/stebz.cc:210`). The existing `syevx` contract
returns **descending when `find_largest`** and ascending otherwise
(`syevx_direct.cc:88`, `syevx_direct_subset.cc:312`).

Rule: preserve it. `find_largest` continues to imply descending; every explicit
`Index`/`Value` range defaults to **ascending** (LAPACK's order), with an explicit
`SyevxParams::order` to override. This means no existing caller's output changes.

Note the internal invariant that must not be broken: `stein`'s cluster detection
requires **ascending** input (`stein.cc:282-284`), so any descending output is produced
by the reversal pass at the end (`syevx_direct_subset.cc:328-345`), never by asking
`stebz` for descending mid-chain. The current code already gets this right; the comment
at `:234-236` says so.

---

## 4. API design

### 4.1 A syevx-specific selector enum

Add to `include/blas/enums.hh`, next to `SyevxAlgorithm`:

```cpp
// How `syevx` chooses which part of the spectrum to return.
//
// `Extremal` is the historical behaviour and the default: `neigs` eigenpairs from
// one end, chosen by SyevxParams::find_largest, returned descending for the largest
// and ascending for the smallest. It is a special case of `Index` --
// [n-neigs, n-1] or [0, neigs-1] -- and is normalized to one internally; it exists
// as a distinct value so that no existing caller's behaviour depends on a default
// that changed meaning.
enum class SyevxSelect {
    Extremal,  // neigs from one end; SyevxParams::find_largest picks the end
    Index,     // SyevxParams::il .. iu inclusive, 0-based, ascending spectrum
    Value      // every eigenvalue in the half-open interval (vl, vu]
};
```

**Why not reuse `EigenRangeType`?** Because its `All` member — the natural default —
means "every eigenvalue", which is *not* what `syevx`'s current default does. Making
`EigenRangeType::All` mean "the `neigs` extremal ones" inside `SyevxParams` would be a
member whose meaning depends on which struct it is embedded in. `EigenRangeType`
stays the tridiagonal-layer vocabulary; `SyevxSelect` is the user-facing one, and
normalization (§5) converts between them in exactly one place.

### 4.2 `SyevxParams` additions

Appended to `include/blas/extensions.hh:33`, all defaulted so existing aggregate
initialization by field name is unaffected:

```cpp
    // Which part of the spectrum to return. Default reproduces the historical
    // top-k / bottom-k behaviour exactly; see SyevxSelect.
    SyevxSelect select = SyevxSelect::Extremal;

    // select == Index: inclusive 0-based bounds into the ASCENDING spectrum.
    // il > iu is an empty request and is rejected; use neigs == 0 for that.
    int64_t il = 0;
    int64_t iu = -1;              // -1 means n-1

    // select == Value: the half-open interval (vl, vu], matching LAPACK. The count
    // is data-dependent and per batch item; see the `m` output of syevx.
    float_type vl = float_type(0);
    float_type vu = float_type(0);

    // Absolute tolerance on each eigenvalue for the bisection-based paths.
    // Non-positive means eps * ||T||, i.e. full working precision. Forwarded to
    // StebzParams::abstol; ignored by paths that get eigenvalues from a full solve.
    float_type abstol = float_type(0);

    // Output order within the selected block. Extremal + find_largest defaults to
    // Descending; everything else defaults to Ascending.
    SortOrder order = SortOrder::Ascending;
```

`vl`/`vu`/`abstol` are `float_type`, not `T`: eigenvalues of a Hermitian matrix are
real, and `W` is already `Span<typename base_type<T>::type>`. Using `T` here would
force complex callers to write `std::complex<float>(vl)` for a real quantity. Note
this differs from the existing `absolute_tolerance`/`relative_tolerance` members,
which are typed `T` — that is a pre-existing wart in the LOBPCG parameters and is
deliberately not propagated.

### 4.3 The `m` output

Add an overload of `syevx` (and of `syevx_direct`/`syevx_direct_subset`/etc.) taking
`Span<int32_t> m` immediately after `W`:

```cpp
template <Backend B, typename T, MatrixFormat MFormat>
Event syevx(Queue& ctx,
            const MatrixView<T, MFormat>& A,
            Span<typename base_type<T>::type> W,
            Span<int32_t> m,                 // NEW: per-item count found
            size_t neigs,                    // now: CAPACITY of W and V per item
            Span<std::byte> workspace,
            JobType jobz = JobType::NoEigenVectors,
            const MatrixView<T, MatrixFormat::Dense>& V = {},
            const SyevxParams<T>& params = {});
```

and keep every existing overload, with the rule:

> **The `m`-less overloads are legal only when the count is statically known** — that
> is, for `SyevxSelect::Extremal` and `SyevxSelect::Index`, where `m[b] == neigs` for
> every `b` by construction. Calling one with `SyevxSelect::Value` throws
> `std::invalid_argument` on the host, before any device work.

That rule is what makes the compatibility story airtight: no existing call site can
reach the new code path, and no new call site can accidentally ignore a count it needs.

Meaning of `neigs` under the new API, to be stated in the doc comment:

| `select`   | `neigs` means                          | `m[b]` is                    |
|------------|----------------------------------------|------------------------------|
| `Extremal` | number wanted (unchanged)              | always `neigs`               |
| `Index`    | must equal `iu - il + 1`               | always `neigs`               |
| `Value`    | **capacity** of `W` and `V` per item   | true count, may exceed `neigs` |

Validating `neigs == iu - il + 1` rather than deriving `neigs` from `il`/`iu` is
deliberate: it makes a mismatched pair a loud host-side error instead of a silently
under- or over-filled output buffer.

### 4.4 Beware the overload trap

Two entries in this repo's memory bear directly on §4.3, and the design above is
shaped by them:

* *Unconstrained variadic overload trap* — do not add a forwarding overload whose
  parameter pack can out-rank the specific ones. The new `m` overload must be a
  plain distinct signature, and the `Span<int32_t>` in position 4 must not be
  reachable by implicit conversion from anything a caller would plausibly pass there.
  `Span<int32_t>` and `size_t` are not mutually convertible, so `syevx(ctx, A, W, m,
  neigs, ...)` and `syevx(ctx, A, W, neigs, ...)` are unambiguously distinguished by
  the type of argument 4. Add a static test that both resolve as intended.
* *Option-struct overload traps* — a bare `{}` for the params argument previously
  picked a positional overload and silently returned different numbers. Every new
  overload here takes `params` in the same trailing position with the same default,
  and the test matrix (§10) must include one call per overload with `{}` for params.

---

## 5. Phase 1 — vocabulary, normalization, validation

**Deliverable: no behavioural change.** Everything below is additive and every
existing test must still pass unmodified at the end of this phase.

### 5.1 Files

* `include/blas/enums.hh` — add `SyevxSelect` (§4.1).
* `include/blas/extensions.hh` — add the `SyevxParams` members (§4.2) and declare the
  resolver below.
* `src/extensions/syevx.cc` — implement the resolver and the validator.

### 5.2 The resolver

One function, used by *every* path and by *every* `buffer_size`, so that the solve and
the sizing call can never disagree — the same discipline `syevx_select_algorithm`
already follows (`extensions.hh:338-346`: "Deterministic in its inputs so that `syevx`
and `syevx_buffer_size` always agree on the choice").

```cpp
// Resolved, algorithm-independent description of what the caller asked for.
struct SyevxResolvedRange {
    bool     value_range;   // true: (vl, vu]; false: [il, iu]
    int64_t  il, iu;        // valid iff !value_range; 0-based inclusive, ascending
    float_type vl, vu;      // valid iff  value_range
    int64_t  max_count;     // upper bound on m[b]: iu-il+1, or the capacity
    bool     reverse;       // write the block in descending order
};

template <typename T>
SyevxResolvedRange syevx_resolve_range(int64_t n, size_t neigs,
                                       const SyevxParams<T>& params);
```

Normalization rules:

| input                                      | output                                              |
|--------------------------------------------|-----------------------------------------------------|
| `Extremal`, `find_largest = true`          | `il = n-neigs`, `iu = n-1`, `reverse = true`        |
| `Extremal`, `find_largest = false`         | `il = 0`, `iu = neigs-1`, `reverse = false`         |
| `Index`                                    | `il`, `iu` (with `iu < 0` → `n-1`), `reverse = (order == Descending)` |
| `Value`                                    | `vl`, `vu`, `max_count = neigs`, `reverse = (order == Descending)` |

Note `reverse` for `Extremal` comes from `find_largest`, **not** from `order`: that is
what preserves the historical contract. If a caller sets both `Extremal` and a
contradicting `order`, throw — a silent winner there is exactly the class of bug §4.4
guards against.

### 5.3 The validator

Extend `validate_syevx_preconditioner_params` (`syevx.cc:92-152`) — or better, add a
sibling `validate_syevx_range_params` called from the same two sites
(`syevx.cc:272` and `:296`) so the two concerns stay separable. Host-side throws:

1. `select == Index` and (`il < 0` or `iu >= n` or `il > iu`).
2. `select == Index` and `neigs != iu - il + 1`.
3. `select == Value` and `vl >= vu` — an empty interval. (Arguably `m = 0` is a valid
   answer; but a caller who writes `vl >= vu` has almost certainly swapped the
   arguments, and the cost of being wrong here is a full `O(n³)` reduction that
   returns nothing. Throw, and say which.)
4. `select == Value` and the `m`-less overload was used (§4.3).
5. `select == Extremal` and `order` contradicts `find_largest` (§5.2).
6. `select != Extremal` and `method` resolves to `LOBPCG` or `Filtered` — deferred to
   Phase 5, where routing knows the resolved algorithm.

Also relax the existing check at `syevx_direct.cc:69-71` / `syevx_direct_subset.cc:111`
(`neigs > n` / `k < 1 || k > n`) to account for `neigs` now meaning capacity: a
capacity above `n` is harmless and should be clamped, not rejected.

### 5.4 Test for this phase

A pure host-side unit test over `syevx_resolve_range` with a table of inputs and
expected outputs, plus a `EXPECT_THROW` per validator rule. No device needed, so it
runs in milliseconds and belongs in the fast test label.

---

## 6. Phase 2 — `syevx_direct`

Do this first among the solvers. It is the **universal fallback**: dense, every scalar
type including complex, every range, every `jobz`. Once it works, `Auto` always has
somewhere correct to route an interior request, and every subsequent phase is a
performance optimization with an existing reference to test against.

### 6.1 Index ranges

`src/extensions/syevx_direct.cc:111-125`. `syev` has already produced the full
ascending spectrum in `lambdas`; selection is a copy. Replace the two `find_largest`
mappings with the resolved range:

```cpp
// eigenvalues
const int64_t src = reverse ? (iu - i) : (il + i);
// eigenvector columns
const int64_t src_col = reverse ? (iu - col) : (il + col);
```

with `count = iu - il + 1`. That is the entire change. Set `m[b] = count` for all `b`
in the same kernel.

### 6.2 Value ranges

Two extra steps, both cheap relative to the `syev` that precedes them.

**Step 1 — locate the interval.** `lambdas` is sorted ascending, so the half-open
interval `(vl, vu]` maps to the contiguous index block
`[lower(vl), lower(vu))` where `lower(x) = ` the number of eigenvalues `≤ x`.
Two binary searches, `O(log n)`, done by `tid == 0` into work-group local memory
followed by a barrier — the same shape `stebz` already uses at
`src/extensions/stebz.cc:157-166`.

Match `stebz`'s predicate exactly (`λ ≤ x` counts) so that `Direct` and `DirectSubset`
return the same `m` for the same input. This is worth a dedicated cross-path test
(§10.4): the two compute `m` by genuinely different means — sorted search versus Sturm
count — and a boundary within rounding distance of an eigenvalue can make them differ
by one. That is not a bug in either, but it *is* a documented tolerance.

**Step 2 — write out with truncation.** `write = min(count, capacity)`; copy `write`
eigenvalues and columns; store the **true** `count` in `m[b]`.

### 6.3 Buffer size

`syevx_direct_buffer_size` (`syevx_direct.cc:134-162`) already ignores `neigs`
entirely and sizes on `n` and `batch` — a full `syev` plus a private copy of `A`. **No
change needed.** Worth an explicit comment saying so, because a reader chasing a
value-range memory question will look here first.

### 6.4 Complex support

`syev` handles complex; the selection kernel is a pure copy of `T`. Value ranges
compare `float_type` eigenvalues. So `Direct` supports every range for every scalar
type — which is what makes it the fallback for complex input, where `DirectSubset` is
unavailable (`extensions.hh:443-446`).

---

## 7. Phase 3 — `stein` per-item counts

Prerequisite for Phase 4, and independently useful.

### 7.1 Signature

Additive overload, existing one preserved:

```cpp
template <Backend B, typename T>
Event stein(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            const VectorView<T>& w,
            size_t k,                    // capacity: columns of Z, entries of w
            Span<const int32_t> counts,  // NEW: per-item valid prefix; empty == all k
            const MatrixView<T, MatrixFormat::Dense>& Z,
            const Span<std::byte>& ws,
            SteinParams<T> params = {});
```

`counts` is a device-readable span — the same `m` that `stebz` wrote — so no host sync
is introduced between the two calls.

### 7.2 Kernel changes

* Phase 1 (`stein.cc:184`): `for (int64_t j = tid; j < k; ...)` becomes
  `j < kb` where `kb = counts.empty() ? k : min(k, counts[bid])`. Work-items whose
  `j >= kb` **must still zero their column** `Z(:, j, bid)`, not skip it — the
  back-transforms in Phase 4 run over a uniform column count and would otherwise
  propagate uninitialized workspace. Zero columns are inert under an orthogonal
  transform and cost `O(n)` each.
* Phase 2 (`stein.cc:282`): the cluster walk becomes `for (j = 0; j < kb; ++j)`.
  This is the correctness-critical one — see §2.4.
* `stein_buffer_size` is unchanged: it sizes on capacity `k`, which is what is
  allocated regardless.

### 7.3 Tests

Extend `tests/stein_tests.cc`: build a batch where item 0 wants 8 vectors and item 1
wants 3, deliberately fill the tail of `w` for item 1 with values that would form a
*bogus cluster* with its last real eigenvalue, and assert item 1's three vectors are
orthonormal and satisfy `‖Tx − λx‖` to tolerance. Without the phase-2 fix this test
fails; that is the point of writing it that way.

---

## 8. Phase 4 — `syevx_direct_subset`

The payoff phase: this is the path with the measured 2.4× win at large `n` and large
batch (`syevx.cc:212-224`), and interior ranges cost it nothing extra.

### 8.1 Replace `wanted_range`

Delete `src/extensions/syevx_direct_subset.cc:76-84` and use the Phase 1 resolver.
Populate `StebzParams` from it at `:242-246`:

```cpp
StebzParams<Real> bp;
bp.range  = rr.value_range ? EigenRangeType::Value : EigenRangeType::Index;
bp.il     = rr.il;   bp.iu = rr.iu;
bp.vl     = rr.vl;   bp.vu = rr.vu;
bp.abstol = params.abstol;
bp.order  = SortOrder::Ascending;   // ALWAYS -- stein requires ascending; the
                                    // reversal happens in the finalize kernel
```

The `Ascending` pin is already there and already correct; keep the comment at `:234-236`
that explains why, and strengthen it to say that `params.order` is honoured only by
the finalize kernel.

### 8.2 Workspace sizing for value ranges

`stebz` throws if `w.size() < max_wanted`, and for a value range `max_wanted = n`
(`stebz.cc:94`). The `w_sub` allocation at `:238` is currently `k * batch`. It must
become:

```cpp
const size_t w_sub_len = rr.value_range ? size_t(n) : size_t(k);
auto w_sub_span = pool.allocate<Real>(ctx, w_sub_len * batch);
```

and the identical expression must appear in `syevx_direct_subset_buffer_size` at
`:429`. This is `n·batch` reals — negligible against the `n²·batch` copy of `A` at
`:391`.

> **BumpAllocator sizing contract.** This repo's memory records that an *exactly*
> computed workspace size is too small, in two independent ways. Every new allocation
> in this phase must be mirrored through `BumpAllocator::allocation_size<>` in the
> sizing function, in the same order, with the same arguments — not hand-computed.
> Phase 4 adds or resizes four allocations; check all four.

### 8.3 Thread `m` into `stein` and the back-transforms

* Pass `m_span` (already allocated at `:239` and filled by `stebz`) to the new `stein`
  overload from Phase 3.
* `unmqr_hb2st` (`:268-273`) and `ormqr_blocked` (`:282-306`) operate on `V_sub`, a
  fixed `n × k` slice. **Leave them at the uniform column count.** Columns beyond
  `m[b]` hold the zeros Phase 3 wrote; an orthogonal transform maps zero to zero, so
  the result stays zero and no garbage propagates. This keeps both back-transforms
  shape-uniform across the batch, which is what makes them fast.
  * The cost is real but bounded: for a value range where one item finds 3 and another
    finds 200, every item pays the 200-column back-transform. Document it. A caller
    who cares should use an index range or split the batch.
* Optional refinement, only if measurement justifies it: compute
  `max_m = max_b m[b]` on device and use it as the column count instead of the
  capacity. This needs `max_m` on the *host* to shape the `ormqr` call, i.e. one
  sync — the exact defect SYEVX_PLAN.md §7.1 catalogues in LOBPCG. **Do not do this
  by default.** Measure first; a sync per call may well cost more than the wasted
  columns.

### 8.4 Finalize kernel

`:320-347`. Two changes:

* the reversal must operate on `min(m[b], capacity)` columns, not `k`, since a
  descending value range must reverse only the valid prefix;
* copy the true `m[b]` from the internal `m_span` to the caller's `m` span.

### 8.5 What does *not* change

`kd` selection, `sy2sb`, `sb2st_hh`, the phase vector, the reflector schedule — none
of them depend on which eigenvalues are wanted. That is why interior ranges are free
here.

---

## 9. Phase 5 — routing and the public entry point

### 9.1 `syevx_select_algorithm`

`src/extensions/syevx.cc:175-239`. Add a parameter for the resolved range and enforce:

```
select != Extremal  →  Direct or DirectSubset only.
```

Concretely:

* sparse input + non-extremal range → **throw**. LOBPCG is the only sparse path and
  cannot answer (`syevx.cc:186`). Until Phase 8 there is nothing to fall back to, and
  silently returning the extremal eigenpairs would be the worst possible outcome.
* dense + explicit `method = LOBPCG` or `Filtered` + non-extremal range → **throw**,
  with a message naming the limitation. Note the existing precedent at `:188-197`:
  an unavailable algorithm *degrades* rather than failing. That precedent must not be
  followed here — degrading `Filtered` to `Direct` silently changes the performance
  characteristics a caller explicitly asked for, but degrading *the requested part of
  the spectrum* would change the answer. Different kind of substitution, different
  rule.
* `BATCHLAS_SYEVX_ALGORITHM=lobpcg` (environment, which per `:182` **wins** over the
  params) + non-extremal range → this is the one case where degrading is right, for
  the same reason `syevx_select_preconditioner` degrades an environment default
  (`extensions.hh:377-381`): the variable exists so a whole suite can be forced onto
  one algorithm for diagnosis, and aborting on the first interior call would make that
  sweep impossible. Degrade to `Direct` and, once per process, warn.

### 9.2 The thresholds carry over unchanged

The measured crossovers at `syevx.cc:63-71` and `:212-227` are functions of
`(n, batch, jobz)` and — explicitly — **not** of `k`: "the ratio is flat in k from
0.8% to 25% of the spectrum and only decays to a tie at 50%" (`:226-227`). Position
within the spectrum does not enter the cost of either path (§8.5). So:

> **No re-measurement is required for index ranges.** For value ranges, feed
> `max_count = neigs` (the capacity) as the `k` argument to the selector, which is the
> correct conservative choice — capacity is exactly what both paths will do work
> proportional to.

State this in the plan and *verify it* with one benchmark point (§11.2) rather than
assuming it. Per this repo's measurement-hygiene rule, compare only at saturation.

### 9.3 Entry point

`syevx.cc:263-310`. Resolve the range once, validate, select the algorithm, forward
`m` to whichever path was chosen. Same in `syevx_buffer_size`, which must resolve the
range too — Phase 4 made the workspace size range-dependent (§8.2).

---

## 10. Phase 6 — tests

`tests/syevx_tests.cc` already has the right shape: parameterized suites per algorithm
that check against a reference `syev` and skip when the environment forces another
algorithm (`:396-401`, `:556-557`). Extend, don't restructure.

### 10.1 Reference oracle

One helper, shared by every new test: run a full `syev` on the host, sort ascending,
then select by index or by `(vl, vu]` in plain host code. That oracle is trivially
correct, which is what makes it a useful reference for four solver paths.

### 10.2 Index ranges

Parameterize over `(n, batch, il, iu, jobz, algorithm)` with, at minimum:

* an interior block: `n = 64`, `il = 20`, `iu = 27` — the case that has never worked;
* a block touching each end, which must reproduce the `Extremal` answer exactly
  (**assert bit-identical values against the `Extremal` call**, not merely close —
  they take the same code path with the same inputs, so any difference is a bug in
  normalization);
* `il == iu` (a single interior eigenpair);
* the full range `il = 0, iu = n-1`, which must match a full `syev`;
* `Descending` order on an interior block.

### 10.3 Value ranges

* an interval straddling the middle of a known spectrum. Use
  `Matrix::TriDiagToeplitz(n, diag, sub, super)` (`include/blas/matrix.hh:401`,
  already used by the suite at `tests/syevx_tests.cc:211`): its eigenvalues are
  `diag + 2·sqrt(sub·super)·cos(jπ/(n+1))`, `j = 1..n`, in closed form, so both the
  values and the exact count in any interval are known analytically. Pick `vl`, `vu`
  in the *gaps* between consecutive eigenvalues so the count is unambiguous;
* an interval containing nothing (`m == 0` for every item) — assert `m` is zero and
  that nothing was written;
* an interval containing everything (`m == n`);
* **a batch where items genuinely disagree**: build item 0 with a spectrum inside the
  interval and item 1 with a spectrum outside it, assert `m = {n, 0}`. This is the
  test that catches §2.4, and it must be run with `jobz = EigenVectors`.
* **capacity overflow**: an interval containing 20 eigenvalues with `neigs = 5`.
  Assert `m[b] == 20`, that the 5 written are the 5 lowest in the interval, and that
  slot 5 onward was not touched (pre-fill `W` with a sentinel).

### 10.4 Cross-path consistency

For the same `(A, range)`, assert `Direct` and `DirectSubset` agree on `m` and on
every eigenvalue to tolerance. Choose boundaries in spectral gaps (§6.2) so the ±1
count ambiguity cannot fire; then add *one* test that deliberately puts a boundary on
an eigenvalue and asserts only `|m_direct − m_subset| ≤ 1`, documenting the tolerance
rather than pretending it does not exist.

### 10.5 Rejection tests

One `EXPECT_THROW` per validator rule in §5.3, plus: sparse + interior throws, explicit
`LOBPCG` + interior throws, `Value` on the `m`-less overload throws.

### 10.6 Cost control

Per this repo's selective-testing policy, do **not** grow the default test time. The
new cases are small-`n` by nature (correctness, not performance), so keep every one at
`n ≤ 128` and `batch ≤ 4`, and put any large-`n` case behind the `slow` label. Run
scoped during development:

```
ctest -L syevx -LE slow
./tests/syevx_tests --gtest_filter='*Range*'
```

Full `ctest` only before pushing. Note the baseline is not green — three known
failures, one of them (`stedc_flat`) still open — so compare against the baseline, not
against zero.

---

## 11. Phase 7 — bindings, benchmarks, docs

> **STATUS: implemented.** §11.1 (Python) and §11.3 (docs) are done; §11.2's
> benchmark exists as `BM_SYEVX_RangePosition` but **has not been run**, which is
> the one honest gap in this document. See §17.3 for what that leaves unverified
> and §17.4 for the exact command that would close it.

### 11.1 Python

* `python/batchlas/_options.py:18` — add `select: str = "extremal"`, `il`, `iu`, `vl`,
  `vu`, `abstol`, `order`.
* `python/batchlas/bindings/ops_spectral.cc:590-620` — parse them into
  `SyevxParams`, allocate `m` as an `int32` array of length `batch_size`, and return it.
  The natural Python return for a value range is `(w, v, m)` with `w` shaped
  `(batch, neigs)`; document that `w[b, m[b]:]` is undefined and that `m[b] > neigs`
  means truncation.
* A NumPy-friendly convenience wrapper in `_api.py` that, for a value range, returns a
  **list of per-item arrays** sliced to `m[b]` — that is what a Python caller actually
  wants, and it costs one host copy that a Python caller is already paying for.

> **Watch the per-call `Queue`.** This repo's memory records that a per-call `Queue` in
> the Python layer silently defeated the P2 arena. Whatever the new binding does, it
> must reuse the existing queue plumbing in `ops_spectral.cc` rather than constructing
> one.

### 11.2 Benchmarks

`benchmarks/syevx_benchmark.cc` — add one range axis to the existing crossover sweep,
purely to *verify* the §9.2 claim that position in the spectrum does not affect cost.
Three points suffice: `(il, iu)` at the bottom, middle, and top of the spectrum for
one `(n, batch)` at saturation. Expect flat; if it is not flat, §9.2 is wrong and the
routing needs a range-aware term.

Do not extend the sweep further than that. The existing crossover map
(`benchmarks/syevx_crossover_rtx4090.csv`) stays valid.

### 11.3 Docs

Update SYEVX_PLAN.md §13 with a pointer to this document and a one-line status, and
update the `syevx` doc comment blocks at `extensions.hh:245-257` and `:293-305`, whose
`@param neigs` lines (`:251`, `:299`) both read "Number of eigenvalues to compute" —
the single most misleading line once capacity semantics land. The `@brief` at `:246`
also says "of a sparse matrix", which has been wrong since dense input started routing
to `Direct`/`DirectSubset`; fix it in the same pass.

---

## 12. Phase 8 (later) — interior ranges for the iterative paths

Out of scope for the first delivery, recorded so the boundary is explicit. Both are
real algorithms, not plumbing.

### 12.1 Folded-spectrum LOBPCG — the sparse story

The only way `syevx` will ever answer an interior query on CSR input. Apply LOBPCG to
`(A − σI)²`, whose smallest eigenvalues are those of `A` closest to `σ`; recover `λ`
from the Rayleigh quotient of `A` (not of the squared operator), so the eigenvalues
come back at full accuracy.

* Cost: two matvecs per operator application instead of one.
* Real drawback: squaring the operator squares the condition number and roughly
  **halves the number of correct digits in the residual criterion**. This is inherent,
  not an implementation defect, and must be documented rather than tuned away.
* Fits the existing structure: `syevx_lobpcg` already has a matvec abstraction; the
  fold is a wrapper around it plus a Rayleigh-quotient change.
* Preconditioning interacts badly — the ILU(k) and Jacobi validity arguments at
  `syevx.cc:100-151` are all framed around approximating `A^{-1}` for the smallest
  eigenpairs, and none of them transfer to `(A − σI)²`. Start unpreconditioned.

### 12.2 Band-pass polynomial filter — the dense interior story

`syevx_filtered`'s high-pass Chebyshev construction (`syevx_filtered.cc:10-21`) is
replaced, for interior slices, by a polynomial approximation to the indicator function
of `[vl, vu]` — the EVSL approach. Two candidate constructions: a Chebyshev expansion
of a step function with Jackson damping to suppress Gibbs oscillation, or the
least-squares/Zolotarev family.

Only worth building if a user needs interior eigenpairs of a matrix too large for
`DirectSubset`. Given that `DirectSubset` handles interior ranges at *no extra cost*
once Phase 4 lands, that constituency may be empty. **Do not build it speculatively.**

---

## 13. Workspace scaling

The one place where value ranges have a genuinely unpleasant cost, and it is worth
knowing before the first `Value` call is written.

`stein_buffer_size` (`src/extensions/stein.cc:322-330`) allocates

```
5 * sizeof(T) * n * k * batch  +  1 * n * k * batch    bytes
```

— five length-`n` scratch arrays plus pivot flags **per (batch, vector)**, because each
work-item owns a private tridiagonal LU. That is `21·n·k·batch` bytes for `float`.

With an index range, `k` is what the caller wants and this is fine. With a value range,
`k` is the **capacity**, and a defensive caller who passes `neigs = n` gets

```
n = 1024, batch = 128, float:  21 * 1024 * 1024 * 128  ≈  2.8 GB
```

on a 24 GB card, for scratch alone, on top of the `n²·batch` copy of `A` (0.5 GB) and
`V` itself.

Three responses, in order of preference:

1. **Document it and make callers pass a realistic capacity.** A caller asking for a
   value range almost always has an estimate of how many eigenvalues are in the
   interval; `neigs = n` is a worst case that is rarely the real one.
2. **Re-index `stein`'s scratch by work-item rather than by `(b, j)`.** The number of
   *concurrently resident* work-items is bounded by the launch geometry, not by
   `k·batch`. Indexing the five scratch arrays by global linear id and sizing them to
   the launch bound would cut this from `O(n·k·batch)` to `O(n · resident)`. This is a
   contained change to one kernel, benefits every `stein` caller including today's
   extremal path, and is the *right* fix. Schedule it as an independent follow-up —
   not a blocker, but do not lose it.
3. Per-item packed output (§3.1 option b). Only if 1 and 2 are insufficient.

---

## 14. Risks, ranked

1. **`stein`'s cluster walk over invalid slots (§2.4).** Highest severity: it produces
   *wrong vectors for valid eigenvalues*, silently, only for value ranges, only when
   items disagree on `m`. This repo has a documented case study in mis-diagnosing
   exactly this class of silent numerical failure (the sy2sb trailing-panel bug).
   Mitigation: Phase 3 lands before Phase 4, and the §10.3 disagreeing-batch test is
   written to fail without it.
2. **Capacity semantics for `neigs` silently under-filling output.** Two prior
   incidents in this repo were option-struct traps that silently returned wrong
   numbers. Mitigation: the `m`-less overload throws for `Value` (§4.3); the sentinel
   test in §10.3 asserts untouched slots.
3. **`Direct` and `DirectSubset` disagreeing on `m` by one at a boundary.** Inherent
   to computing the count two different ways. Mitigation: match `stebz`'s `≤`
   predicate exactly (§6.2), document the tolerance, and test it explicitly rather
   than papering over it (§10.4).
4. **BumpAllocator under-sizing on the new value-range allocation.** The sizing
   function is 200 lines away from the allocation it mirrors
   (`syevx_direct_subset.cc:238` vs `:429`) and this repo's memory records that an
   exactly computed size is too small in two independent ways. Mitigation: §8.2's
   rule, plus a test that runs a value range through the real `buffer_size` path with
   an exactly-sized buffer.
5. **`Auto` routing an interior request somewhere that cannot answer it.** Mitigation:
   §9.1's throw-don't-degrade rule, and §10.5's rejection tests.
6. **Scope creep into Phase 8.** The filtered/LOBPCG interior algorithms are
   genuinely interesting and genuinely not needed to close the capability gap.
   Mitigation: §12 states the boundary; `DirectSubset` covers interior at no extra
   cost, which removes the motivation.

---

## 15. Non-goals

* MRRR. SYEVX_PLAN.md §6.1-6.2 already argued bisection over MRRR for this regime and
  nothing about range selection changes that argument.
* A `syevr` entry point. The range vocabulary here is the useful half of `syevr`'s
  API; the other half is the MRRR algorithm, which is a non-goal.
* Generalized `sygvx` ranges.
* Per-batch-item *different* ranges (item 0 wants `[0,5]`, item 1 wants `[10,20]`).
  `stebz` takes scalar `il`/`iu`, so this would be a device-side range vector
  threaded through every layer. No demand; note it and move on.
* Changing the default behaviour of any existing call.

---

## 16. Sequencing

| Phase | Deliverable | Depends on | Size |
|---|---|---|---|
| 1 | `SyevxSelect`, params, resolver, validator | — | S |
| 2 | `syevx_direct`: all ranges, all types | 1 | S |
| 3 | `stein` per-item counts | — | S |
| 4 | `syevx_direct_subset`: all ranges | 1, 3 | M |
| 5 | Routing + `m` plumbing through `syevx` | 1, 2, 4 | S |
| 6 | Tests | 2, 3, 4, 5 | M |
| 7 | Python, benchmark verification, docs | 5, 6 | S |
| 8 | Interior for iterative paths | — | L, deferred |

Phases 1–3 are independent enough to land in one PR. Phase 4 is the substantive one.
The capability is *usable* after Phase 5 and *trustworthy* after Phase 6.

A reasonable first PR is **Phases 1 + 2 + the §10.2 index-range tests against
`Direct`**: it closes the capability gap for every scalar type and every matrix size,
with the simplest possible implementation and a trivially correct oracle, and it gives
Phase 4 a reference to test against.

---

## 17. What actually landed

Written after the fact. The plan above is left as it was written so the two can be
compared; this section is the correction where they differ.

### 17.1 Phases 1–7, as designed

Everything in §5–§11 is implemented and tested. The parts worth restating because
they are the *contract* and not an implementation detail:

* `SyevxSelect { Extremal, Index, Value }` in `include/blas/enums.hh`, with
  `SyevxParams::{select, il, iu, vl, vu, abstol, order}` all defaulted so that an
  existing caller's behaviour is byte-for-byte unchanged.
* `syevx_resolve_range` is the single normalization point, shared by every solve
  and every `buffer_size`, so the two can never disagree about what was asked for
  or how big the workspace has to be.
* `neigs` means **capacity**. For `Extremal` and `Index` capacity equals count;
  for `Value` the count is data-dependent, per batch item, and comes back through
  `m`. `m[b] > neigs` is the overflow signal, and the kept eigenvalues are the
  **lowest `neigs` of the interval**.
* **`W` past `m[b]` is untouched; `V` past `m[b]` is exactly zero.** The asymmetry
  is deliberate and is documented at the declaration: the subset path's
  back-transforms run over a uniform column count and need something inert there,
  and since `Auto` routes between the two dense paths on `(n, batch)` alone, both
  paths must agree. `W` has no such consumer, so its tail stays usable as a "was
  this slot written" sentinel.

§11.3's `extensions.hh` edits were already made during phase 5, ahead of the
schedule here: the `@brief` no longer says "of a sparse matrix" (wrong since dense
input started routing to `Direct`/`DirectSubset`) and neither `@param neigs` line
still says "Number of eigenvalues to compute" (the single most misleading line
once capacity semantics landed). Nothing was left to do in that file for phase 7.

### 17.2 Where the implementation departed from the plan

* **§5.3 rule 3 (`vl >= vu` throws) was kept, but rule 1 gained a clamp as well.**
  The plan offered a host-side throw *or* a clamp in the resolver; both landed,
  because they answer different questions. The throw tells the caller they made a
  mistake; the clamp is what makes `max_count`'s "already clamped to `n`" claim
  true for any future consumer that indexes `[il, il + max_count)` without a bound
  of its own. An entirely out-of-range or inverted `Index` block resolves to the
  canonical empty block (`il = 0`, `iu = -1`, `max_count = 0`) so that
  `iu - il + 1 == max_count` holds for *every* resolved range.
* **§4.4's overload-resolution argument was wrong about the mechanism, and the
  code comment now says so.** The plan claimed the `m`-taking and `m`-less forms
  are told apart by `Span<int32_t>` vs `size_t` in position 4. They are not:
  `Span` has a non-explicit `Span(T&)` constructor, so `Span<int32_t>` *is*
  implicitly constructible from an `int32_t` lvalue. What actually discriminates
  is parameter **5** — `size_t neigs` against `Span<std::byte>` / `JobType`, which
  are mutually non-convertible. The conclusion (the two forms are unambiguous)
  survived; the reason did not.
* **§5.3 rule 5 ("`Extremal` + a contradicting `order` throws") was NOT
  implemented, deliberately.** `order` is simply *ignored* for `Extremal`:
  `syevx_resolve_range` takes `reverse` from `find_largest` there and never looks
  at `order`, and the declaration says so. Throwing would have been actively
  harmful once the Python layer landed, because `SyevxOptions` sends every field
  it has on every call — so an ordinary `bl.syevx(a, k)` would carry
  `order = "ascending"` alongside the default `find_largest = True` and would have
  started throwing for no reason. Ignoring is the behaviour that keeps "no
  existing caller's output changes" true.
* **§8.3's uniform-column back-transform was kept, and the optional `max_m`
  refinement was not built.** It needs `max_m` on the host to shape the `ormqr`
  call, i.e. one sync per call — the exact defect SYEVX_PLAN.md §7.1 catalogues in
  LOBPCG. The plan said measure first; nothing has been measured, so nothing was
  changed.
* **§11.1's Python surface grew a wrapper the plan only sketched.** `bl.syevx`
  returns `m` as an extra element *only* when a range was requested, so no
  existing caller's unpacking changes; `bl.syevx_range` is the NumPy-shaped form
  that returns per-item arrays already sliced to `min(m[b], capacity)`, plus
  `counts`, `truncated` and `capacity`. A rectangular array cannot express a
  ragged value-range answer without lying about the slots past `m[b]`, which is
  why the wrapper returns lists.
* **The `m`-less `syevx_direct` / `syevx_direct_subset` overloads were briefly
  deleted and then restored.** They were edited in place rather than added
  alongside, which broke source and ABI compatibility for two exported symbols
  that nothing in-tree calls — which is exactly why it was invisible. Restored as
  inline forwarders distinguished by arity, with a test
  (`SolverEntryPointsKeepTheirMLessOverloads`) that fails if they go missing again.

### 17.3 What is NOT verified

* **§11.2 was never measured.** `BM_SYEVX_RangePosition` exists in
  `benchmarks/syevx_benchmark.cc` and has not been run. So §9.2's claim — that
  position within the spectrum cannot enter the cost of either dense path, and
  therefore that the extremal crossover thresholds carry over to index and value
  ranges unchanged — remains a *structural argument* (§8.5) plus a routing unit
  test that pins the *decision*, not a timing. The failure mode if it is wrong is
  a mis-routed interior request: a slower correct answer, never a wrong one.
* **The Python layer is unverified by compilation.** `BATCHLAS_BUILD_PYTHON` is
  `OFF` in the working build directory, so `_options.py`, `_api.py`,
  `bindings/support.hh`, `bindings/ops_spectral.cc` and the new tests in
  `python/tests/test_batchlas.py` have never been built or run. The Python files
  are syntax-checked only.
* **The benchmark is unverified by compilation.** `BATCHLAS_BUILD_BENCHMARKS` is
  `OFF` in the same build directory.

### 17.4 The commands that would close §17.3

```sh
# Python bindings + the new range tests
cmake -S . -B build-py -DBATCHLAS_BUILD_PYTHON=ON
cmake --build build-py -j 32
python -m pytest python/tests/test_batchlas.py -k syevx

# The one benchmark point. Quiet GPU, saturation only -- this repo's
# measurement-hygiene rule makes an unsaturated comparison meaningless.
cmake -S . -B build-bench -DBATCHLAS_BUILD_BENCHMARKS=ON
cmake --build build-bench -j 32 --target syevx_benchmark
./build-bench/benchmarks/syevx_benchmark --name BM_SYEVX_RangePosition
```

Read the benchmark output as: `Direct` (algo 1) is the control and cannot depend
on position, so its spread across the three positions is the noise floor.
`DirectSubset` (algo 2) is flat if its spread sits inside that band. If it does
not, §9.2 is wrong and `syevx.cc`'s `kSyevxSubsetMinN` / `kSyevxSubsetMinWork`
gate needs a range-aware term.

### 17.5 Still open, and independent of this work

* **Phase 8 (§12) stays deferred, on purpose.** Folded-spectrum LOBPCG and a
  band-pass polynomial filter are real algorithms, not plumbing, and
  `DirectSubset` answers interior ranges at no extra cost — which removes most of
  the motivation. Do not build them speculatively.
* **`stein`'s workspace (§13).** `5 * n * k * batch` scratch, ~2.8 GB at
  `n = 1024`, `batch = 128`, float, when a defensive value-range caller passes
  `neigs = n`. Re-indexing the five scratch arrays by work-item instead of by
  `(b, j)` would cut it to `O(n * resident)` and would benefit every `stein`
  caller including today's extremal path. Not a blocker; do not lose it.

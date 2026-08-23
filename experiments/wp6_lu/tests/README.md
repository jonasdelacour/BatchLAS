# WP6 — the LU test suite: what it asserts, what was broken to prove it, and what it does not settle

Everything here concerns `tests/getrf_tests.cc` — one file covering `getrf`,
`getrs` and `getri`, wired into `tests/CMakeLists.txt` `TEST_TARGETS` and the
`blas` label. The kernel-side evidence lives in `../kernels/`; this directory is
only the correctness gates.

---

## 1. Headline

**Both builds green.** `tests/getrf_tests` is 62 passed / 0 failed in `build/`
and 54 passed / 0 failed in `build-novendor/` (the eight extra vendor-free skips
are the two drop-in tests, which have no vendor to cross to). Full vendor `ctest`
is **59 of 61** with the two pre-existing NETLIB-double failures and nothing
else; `ctest -L "blas|ortho"` is **23 of 23**.

> **UPDATED BY THE REPAIR PASS.** The counts above are this directory's own run
> and are left as written. The repair pass added a 17th typed case
> (`InfoFillIsOrderedAheadOfThePanelOnAnOutOfOrderQueue`, float only — see
> `../README.md` §6.1) and a `RouteLuPivotFormat` case in
> `tests/route_vocabulary_tests.cc`, so the suite is now **63 passed / 73 skipped
> / 0 failed** in `build/` and **55 / 81 / 0** in `build-novendor/`, out of 136.
> Everything else in this file still holds; all 15 break anchors were re-verified
> to match exactly once in both directions, and two of them (`getri_perm_t`,
> `laswp_left`) were re-run end to end and are still red.

**Vendor-free burn-down: 33 of 56 on `ctest -LE slow`** — 30 of 55 before WP6,
32 of 55 after the kernels, plus `getrf_tests` itself. `inverse_tests` and
`linalg_layer_tests` are both out of the failing set, and the predicted suite is
verified two ways: it passes, **and** a coverage capture shows it actually
reaches the native kernels rather than merely linking them.

```
BATCHLAS_COVERAGE_OUT=... ./build-novendor/tests/inverse_tests
reached,getrf,float,CUDA,1281,40,40,40,2,native,cta,...
reached,getri,float,CUDA,1281,40,40,40,2,native,blocked,...
```

**Fourteen breaks, every one rebuilt and run, all fourteen red.** Two of the
results are findings rather than confirmations, and one of them is a defect in
this file's own test matrix.

---

## 2. The oracles

Nothing here is held to vendor agreement. Four host oracles, each answering a
question the others cannot:

| oracle | what only it can see |
|---|---|
| `‖PA − LU‖_F / ‖A‖_F`, P reconstructed on the host from the returned pivots | the pivot BASE and DIRECTION — a 0-based array, a permutation vector, or a backwards walk each make this O(1) |
| the pivot sequence ELEMENTWISE against a sequence known without arithmetic | a valid-but-different pivot choice, which every residual passes |
| `cabs1(L(i,k)·U(k,k)) ≤ cabs1(U(k,k))` | that this is a PARTIAL-pivoting factorization, **in the metric the library pivots on** |
| `‖op(A)X − B‖` and `‖A·A⁻¹ − I‖` against the ORIGINAL A | that the solvers consume the factor their producer returned |

Two of these needed a design decision that is not obvious.

**The exact pivot sequence needs a matrix, not an algorithm.** A host `getf2`
cannot be the reference for the blocked tier: a blocked LU rounds its trailing
update in a different order, and a near-tie can legitimately flip. So the pivot
tests run on a strictly column-diagonally-dominant matrix (`|B(k,k)| = 4n`,
`|B(i,k)| ≤ 1`) that has then been row-permuted. Dominance is preserved by
elimination, so at step *k* the winner is the row carrying B's row *k*, ahead by
a factor of order *4n/3* — outside any rounding of any of the four types — and
the expected interchange list follows from integer bookkeeping alone.

**`max |L| ≤ 1` IS WRONG FOR COMPLEX.** LAPACK selects on `cabs1 = |Re| + |Im|`
and `cabs1(z) ≤ √2·|z|`, so a correct `zgetrf` returns `|L|` up to √2 —
measured at 1.051 on the first random `cfloat` matrix this file generated. The
exact statement survives in the factor, because `L(i,k) = a_ik / a_kk` and the
chosen pivot `a_kk` **is** `U(k,k)`:

```
cabs1( L(i,k) · U(k,k) ) ≤ cabs1( U(k,k) )
```

which is uniform over the four types and, unlike the `|L| ≤ 1` form, is
**sensitive to the metric**. That matters: see finding 3 below.

---

## 3. The break record

`break.py <name>` applies, `--revert` restores, **every anchor must match exactly
once in both directions** (WP6's kernel-side tooling patched the wrong line
because an 8-space anchor is a substring of a 12-space one, and left the tree
corrupted). `run_break.sh` rebuilds the `.so` and runs the binary **once per
scalar type** — four filtered runs, because two of these breaks abort the
process and a single run then says nothing about the three types that never
executed.

| break | property corrupted | outcome |
|---|---|---|
| `piv_base_zero` | `ipiv` written 0-based | **RED**, 12–13 of 16 per type |
| `getrs_forward` | transposed permutation walked forwards | **RED**, all 4 — *see finding 1* |
| `info_block_local` | `info` offset panel-local, not global | **RED**, `SingularColumn…`, all 4 |
| `short_final` | panel loop stops at the last FULL panel | **RED** + SIGSEGV (139), all 4 |
| `subview_ld` | sub-view built with rows, not the parent `ld` | **RED**, 9 of 16 per type |
| `getrs_perm_first` | transposed permutation moved to the INPUT | **RED**, all 4 |
| `hole_pad` | the 48 KB pad removed | **RED**, all 4 — *arithmetic half only, finding 2* |
| `pivot_metric` | `cabs1` → the modulus (cuBLAS's rule) | **RED** cfloat (5) / cdouble (4); nothing for float/double, correctly |
| `laswp_left` | interchanges not applied to columns `[0, j0)` | **RED**, 9 of 16 per type |
| `getri_forward` | getri's backward trace run forwards | **RED**, all 4 |
| `leaf_swap_right` | leaf row exchange restricted to columns ≥ k | **RED**, 12 of 16 per type |
| `info_epsilon_floor` | an epsilon floor in the singularity test | **RED**, `NearlySingularIsNotFlagged`, all 4 |
| `piv_stride_nb` | pivot stride `nb` instead of the matrix order | **RED** + SIGABRT (`OUT_OF_RESOURCES`), all 4 |
| `getri_perm_t` | F written transposed into C | **RED**, all 4 |

### FINDING 1 — the first version of this file could not test a permutation direction at all

`getrs_forward` turned **NOTHING** red: 62 passed, 0 failed.

`make_dominant_permuted` originally permuted rows by a **REVERSAL**, and a
reversal is its own inverse. The permutation the interchange list composes to
then satisfies `F = F⁻¹`, and getrs's transposed arm — whose entire content is
*"the SAME list walked BACKWARDS"* — returns the identical answer walked
forwards. Three tests of a direction (getrs `Trans`, getrs `ConjTrans`, getri's
backward trace) were **unfalsifiable on every scalar type** while reading as the
strongest tests in the file.

The fix is a **cyclic shift**, which composes to an n-cycle, plus
`interchange_is_involution()` asserted at every direction-sensitive use so the
property cannot regress silently. On the shift, `getrs_forward`,
`getrs_perm_first`, `getri_forward` and `getri_perm_t` are all red for all four
types.

This is the repository's blind-guard class again, and it is the first instance
caught in a **test file** rather than in a kernel. The general lesson is sharper
than "use a random permutation": *a test of an inverse operation is vacuous on
any self-inverse instance*, and self-inverse instances are exactly the ones a
tidy-minded author reaches for.

### FINDING 2 — the 48 KB hole does not reproduce for this kernel, and the test says which half knows it

`hole_pad` turns the **pad-arithmetic** layer red for every in-band row and every
type — `getrf_leaf_fits` admits a 49,152 B tile at a 49,152 B budget — and leaves
the **launch** layer green: the resident launch at 49,152 B succeeds on this box
without the pad.

That agrees with `getrf_cta.cc:124-129`'s own reading — WP6 attributed the hole
to `sycl::reduce_over_group` alone, and this body uses no group collective, only
`permute_group_by_xor`. So the pad here is **defensive**, and the arithmetic
assertion is the one with teeth. Both layers are kept, and they are `EXPECT`
rather than `ASSERT` precisely so that a failure in the first cannot mask the
second — without that split the break record could not have said which layer was
carrying the guard.

### FINDING 3 — the metric-aware pivot oracle sees what `|L| ≤ 1` could not

WP6's kernel-side campaign recorded `pivot_metric` (cabs1 → modulus) as *"turned
NOTHING red on the ordinary sweep"*, and had to build a dedicated probe matrix
for it. With the `cabs1(L·U) ≤ cabs1(U)` form of oracle 3, the break turns the
**ordinary complex sweeps** red — `CtaFactorises`, `BlockedFactorises`,
`BothPanelLeaves` — on plain random data, in addition to the dedicated probe. An
oracle that asks the question in the right metric does not need the adversarial
matrix; the probe is kept anyway, because it is the one that names the two
candidate rules by number.

---

## 4. What this directory does NOT settle

1. **`preferred()` is not tested for a window, because there is none.**
   `RouteTableAndTheVendorFreeFallback` asserts the opposite — that a
   vendor-present build still routes to the vendor at every shape — which is the
   assertion that would fire if a `preferred()` window landed without the
   measured grid that justifies it. When the routing step lands, that assertion
   is the one to replace, not to delete.
2. **`native_tier_preferred()` is covered synthetically, not here.**
   `tests/route_vocabulary_tests.cc` pins the tier hook (and WP6's `tier_window`
   break turns it red). This file only asserts that the REAL builder reports a
   non-zero `cta_max_n` and `blocked_available` on this device — which is the
   part a synthetic shape cannot see. A real-device assertion that `double`
   resolves to `Blocked` at n = 40 while the other three take `CTA` is visible in
   the coverage capture but is not asserted; adding it would need its own break.
3. **`Backend::NETLIB` on a GPU queue is still ungated**, and this file cannot
   change that: the fixture skips every NETLIB row (they run on a CPU queue, and
   the native LU is GPU-only by `supports()`), so the packed-int32-vs-int64
   pivot mismatch the kernel README records at §8 has no test here either.
4. **The two aborting breaks are red by crash, not by assertion.**
   `short_final` segfaults and `piv_stride_nb` aborts with `OUT_OF_RESOURCES`
   before the suite can report per-test results for the later tests. A crash is a
   legitimate red, but it means those two breaks do not demonstrate *which*
   assertions would have caught them had the process survived.
5. **The residual tolerances are `c·n·eps` with `c` of 200–800** and were not
   tightened against a measured error distribution. They are loose enough that a
   small systematic error could hide inside them; every break above was caught by
   an equality or a structural assertion, not by a tolerance, so nothing in the
   record depends on the constants being tight.
6. **No performance claim is made or checked here.** The A/B grid is
   `../kernels/`; this file times nothing and must not be used to.

---

## 5. Files

`break.py` — 14 breaks, each with the property it corrupts written at the site,
and an exactly-once anchor check in both directions.
`run_break.sh` — apply, rebuild, run once per scalar type, revert, capture.
`run_all_breaks.sh` — the whole list; writes `breaks.txt`.
`breaks.txt` — the per-type summary table.
`break_<name>.txt` — full output of each break run, so a row the break should
*not* have moved can be read (which is how WP6's kernel-side campaign caught its
own corrupted tree).

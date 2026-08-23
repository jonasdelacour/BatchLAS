# WP6 — the native LU kernels: what landed, what it measures, and what it does not settle

Everything here is produced by ONE program, `luverify.cpp`, linked twice — once
against `build/` (vendor present) and once against `build-novendor/` — so
"vendor-free" means the BUILD and not a forced route. It calls only the PUBLIC
`getrf` / `getrs` / `getri`, checks correctness IN THE SAME PROCESS against a
HOST oracle, and prints the RESOLVED ROUTE on every row so a pin that silently
became the vendor is visible rather than assumed.

---

## 1. Headline

**Three native LU ops landed, all four scalar types, both native `getrf` tiers.**
`preferred()` is still `false` everywhere: a vendor-present build keeps taking
cuBLAS for every shape, and these kernels are reachable only through
`BATCHLAS_GETRF_ROUTE` / `_GETRS_` / `_GETRI_`, through the direct entry points,
or in a vendor-free build.

**Burn-down: 30 → 32 of 55.** `inverse_tests` (the predicted one) and
`linalg_layer_tests` (the scaffolding agent's second candidate) both go green
vendor-free. The vendor-free failing set drops from 18 to 16 of 35 on
`-L "blas|ortho|util" -LE slow`; the vendor-present build stays 35/35 and the
full vendor `ctest` stays at its two pre-existing NETLIB-double failures
(`lanczos_tests`, `steqr_tests`).

**A/B geomeans against cuBLAS at the saturating batch schedule** (28 cells each,
4 types × 7 orders, no cell discarded, every residual green):

| op | geomean | wins | worst | best |
|---|---|---|---|---|
| `getrf` | **0.887×** | 9/28 | 0.265× (double n=128) | 7.29× (float n=2048) |
| `getri` | **1.463×** | 16/28 | 0.219× (cfloat n=32) | 33.9× (float n=2048) |
| `getrs` (nrhs=1) | **0.257×** | 0/28 | 0.083× (cdouble n=512) | 0.736× (double n=2048) |

The shape is the one the baseline predicted: native loses at small `n` where
cuBLAS's small-`n` batched paths are strong, and wins increasingly above
`n ≈ 512`. Every `n ≥ 512` ratio is against an **unsaturated vendor** and must
not be quoted alone.

---

## 2. What was built

| file | what |
|---|---|
| `src/extensions/getrf_cta_device.hh` | the ?GETF2 device body, one algorithm over a `Tile` concept, instantiated for a `local_accessor` and for a raw global pointer |
| `src/extensions/getrf_cta.cc` | capacity, fit predicate, the two panel launchers, `getrf_panel_factorize`, the CTA direct entry point |
| `src/extensions/getrf_blocked.cc` | the right-looking driver: panel leaf → laswp → routed trsm → routed gemm |
| `src/extensions/lu_laswp.hh` | the interchange kernel, a template on a per-TU **tag** so the three ops keep their CMake clusters |
| `src/extensions/getrs_native.cc` | permutation + two routed trsm, all three `transA` modes |
| `src/extensions/getri_blocked.cc` | write P into C + two routed trsm; zero workspace |

Capacities are a function of the **runtime** local-memory budget (97,280 B here),
never the hardcoded 49,152 in `device_limits.hh`:

```
float 155   double 109   cfloat 109   cdouble 77
```

and the blocked driver's leading panel switches from the resident leaf to the
global one at `n ≈ 380` for double/cfloat, `n ≈ 190` for cdouble, `n ≈ 760` for
float (`run_params.sh`).

---

## 3. Correctness, and the oracle that is stronger than a residual

`run_verify.sh` sweeps `getrf` / `getri` / `getrs` at orders that straddle the
`nb = 32` block boundary (31, 32, 33, 64, 96, 100, 128, 256, 800), batch > 1,
checking the **first and last** batch items. Every cell green in both builds and
under both native pins. What is asserted, beyond a residual bound:

* **the pivot sequence, elementwise, against an independent host `LAPACKE_?getrf`.**
  A residual bound is satisfied by ANY valid pivot choice; only this can see a
  wrong base, direction, or metric.
* **`ntpiv`**, the count of non-diagonal pivots on item 0. The matrix is
  diagonally dominant and then **row-permuted**, because on the dominant matrix
  alone partial pivoting picks the diagonal at every step and every pivot probe
  in the directory is vacuous (a recorded baseline finding).
* **`adiff`**, `max |A_after − A_factored|` for `getri`: cuBLAS takes
  `const T* const A[]` and this arm must not write A either. Measured 0.0 for
  all four types.
* **all three `transA` modes** for `getrs`. The scaffolding's break B8 measured
  that no test in the suite issues a `Trans` getrs at all.
* **`singular`**: exact-zero `info`, 1-based, global, per item, from a span
  pre-poisoned with `-12345`, plus a finiteness check on the failed item.

### FINDING — cuBLAS pivots on the MODULUS; LAPACK, NETLIB and this kernel pivot on `cabs1`

`run_pivmetric.sh` builds a matrix whose column 0 holds `(3+0i)` in row 0 and
`(2+2i)` in row 1, so `cabs1 = |Re|+|Im|` reads 3 vs 4 and the modulus reads
3 vs 2.828 — the two rules select **different rows**.

```
native:cta        ipiv[0] = 2   == host LAPACKE     (cfloat and cdouble)
native:blocked    ipiv[0] = 2   == host LAPACKE
cublas?getrfBatched  ipiv[0] = 1  != host LAPACKE
```

Substituting the modulus into `lu_cabs1` reproduces cuBLAS's answer exactly,
which identifies the cause rather than merely observing a difference.
Consequences: this kernel is LAPACK-faithful and the vendor is not; a test that
compares native pivots to **vendor** pivots elementwise is wrong and will go red
on complex; and mixing arms is still safe, because `getrs`/`getri` consume `ipiv`
together with the factor the same `getrf` produced.

### FINDING — the exact-zero `info` predicate is not stable across implementations

On the singular probe (item 1 has row 1 = 2 × row 0):

| type | native `|U₆₆|` / info | host LAPACKE info | cuBLAS `|U₆₆|` / info |
|---|---|---|---|
| float | 0.0 → 6 | 6 | 0.0 → 6 |
| double | 0.0 → 6 | 6 | 0.0 → 6 |
| cfloat | 0.0 → 6 | **0** | 9.78e-10 → 0 |
| cdouble | 0.0 → 6 | 6 | **2.93e-18 → 0** |

cuBLAS itself mismatches the host oracle at cdouble. So "device info == host
info" cannot be a test gate. The gate used here is structural: non-zero exactly
when `|U(i,i)|` is a true binary zero; the failed item stays finite; the
non-singular items report 0.

---

## 4. The breaks — every guarded property corrupted, the `.so` rebuilt, the outcome recorded

| break | what it corrupts | outcome |
|---|---|---|
| `laswp_left` | the blocked driver's interchange on the already-factorised columns `[0, j0)` | **RED** at n ≥ 64 (residual 1.5e-07 → 1.2e-01); n=31 and n=33 stay green, and correctly so — at n=31 there is one panel, and at n=33 the second panel's single pivot is its own row |
| `getrs_reverse` | the transposed permutation walked forwards instead of backwards | **RED** for all four types (residual → 1.6–1.9) |
| `short_final` | the panel loop stops at the last FULL panel | **RED** at n=33 and n=100, green at n=64 and n=96 — i.e. it discriminates exactly the short final panel |
| `getri_perm_t` | F written transposed into C | **RED** for all four types |
| `leaf_swap_right` | the leaf's row exchange restricted to columns ≥ k | **RED** for all four types |
| `pivot_metric` | `cabs1` → modulus | **turned NOTHING red on the ordinary sweep** — see below |
| `tier_window` | double's tier ceiling widened to infinity | **RED** on `RouteGetrf.NativeTierPreferredIsDeclaredAndPinsTheMeasuredTierChoice`, on exactly the double-resolves-to-Blocked assertion |

### The two most valuable break results

**`pivot_metric` turned nothing red, and that is a finding about the ORACLE.**
On the random test matrix the two selection rules agree at every step, so the
elementwise pivot comparison — the strongest oracle in the harness — was blind to
the metric. The fix was a new probe (`pivmetric`) on a matrix built to separate
the functionals, on which the break IS red. An oracle can be correct, necessary,
and still blind; only a break says which.

**The residual was computed and not asserted on.** The first version of this
harness gated `ok` on `isfinite()` alone. `laswp_left` drove the getrf residual to
1.2e-01 and the row still printed `ok`, `FAILS=0`. That is this repository's
blind-guard class exactly — a probe that computes the right number and does not
assert on it — caught only because the break was actually run. Every criterion now
carries a bound (`Tol<T>`).

**A third, procedural one worth recording:** the `getrs_reverse` revert patched
the WRONG line, because an 8-space anchor is a substring of the 12-space line, and
left BOTH permutation walks inverted in the tree. It was caught only because the
NEXT break's run showed `getrs` failing for float and double — types that break
could not touch. `break.py` now requires every anchor to match **exactly once**,
and break runs capture full output so rows the break should not have moved can be
read.

---

## 5. The A/B grid

`grid_vendor.csv` (build/, routes pinned `vendor`) and `grid_native.csv`
(build-novendor/, **no pin at all**, so the tier is whatever `resolve_route`
picks); `summary.txt` is `analyse.py`'s join. No cell discarded: every relative
sd < 10%, every row flagged `ok`, every route as expected (`native:cta` below the
per-type ceiling, `native:blocked` above).

`getrf`, native/vendor:

```
n(batch)     float   double  cfloat  cdouble
  32(8192)   0.545   0.311   0.595   0.414
  64(8192)   1.614   0.305   0.828   0.430
 128(4096)   0.734   0.265   0.383   0.337
 256(2048)   0.966   0.618   0.805   0.491
 512(512)    1.235   0.657   1.220   0.700
1024(128)    2.029   0.921   1.644   0.960
2048(32)     7.291   2.768   5.029   3.730
```

`getri`:

```
  32(8192)   0.564   0.229   0.219   0.232
  64(8192)   0.858   0.528   0.344   0.469
 128(4096)   1.274   0.902   1.059   0.623
 256(2048)   2.888   1.164   2.067   0.845
 512(512)    4.131   1.295   3.112   0.956
1024(128)    9.066   1.160   6.072   1.136
2048(32)    33.942   4.264  25.858   4.589
```

`getrs` at nrhs = 1 loses everywhere, 0.083×–0.736×. That is the baseline's
negative result reproduced, and it is why this arm ships route-neutral.

### The grid folds in TWO differences, and `crossbuild.txt` separates them

The grid compares "vendor build, vendor routes" against "vendor-free build,
Auto". The second changes the LU arm **and** what the routed trsm/gemm underneath
it resolve to. Pinning the LU routes to native in the **vendor-present** build
isolates them:

| cell | native in vendor build | native vendor-free | vendor |
|---|---|---|---|
| `getrf` float 512(512) | 39.76 | 40.44 | 49.89 |
| `getrf` cdouble 2048(32) | 744.0 | 694.1 | 2589 |
| `getri` float 2048(32) | 19.15 | 43.04 | 1473 |
| `getrs` cdouble 2048(32) | 10.53 | 50.64 | 11.12 |

* **`getrf` is build-independent**: its trailing gemm/trsm already take native
  arms in both builds, so the vendor-free build costs it nothing.
* **`getri` float** is 1.5–2.2× slower vendor-free — its inner trsm benefits from
  cuBLAS — and still 34× faster than `cublas?getriBatched` at n=2048.
* **`getrs` cdouble** is 3–5× slower vendor-free. In the vendor build the native
  `getrs` at cdouble n=2048 is 10.53 ms against `cublas?getrsBatched`'s 11.12 ms —
  **1.06×**, which reproduces the baseline's 1.07× at that cell exactly. So the
  grid's 0.219× there is mostly the vendor-free **trsm**, not the composition.

---

## 6. FINDING — the row interchange is 44–74% of native `getrf` for float

Priced by disabling both of the blocked driver's interchange passes (a
TIMING-ONLY break; the answers are wrong by construction — `laswp_cost.txt`),
vendor-free build:

| type | n(batch) | with laswp | without | laswp share |
|---|---|---|---|---|
| float | 512(512) | 40.44 ms | 10.53 ms | **74%** |
| float | 2048(32) | 70.86 ms | 39.71 ms | **44%** |
| cdouble | 512(512) | 232.09 ms | 181.97 ms | 22% |
| cdouble | 2048(32) | 694.05 ms | 645.02 ms | 7% |

**This is the single biggest remaining lever in WP6.** Without it those two float
cells run 4445 and 4652 GFLOP/s against cuBLAS's 918 and 354, so the 0.886×
`getrf` geomean is very largely this one kernel's number.

The mechanism decides which fixes can work. Per panel the interchange touches
`2·ib` rows across every column. The **k side is free**: rows `j0 … j0+ib-1` are
consecutive, so one 128 B line per column serves all `ib` steps. The **p side is
the whole cost**: the `ib` selected rows are scattered over `[j0, n)`, each its
own line — 4 B used of 128 B for float, a 32× inflation. The total is `O(n²·L)`
against the gemm's `O(n³/3)`, which is why it hurts most at moderate `n`.

Two candidates, neither implemented, both shape-dependent:
1. **full-range gather** over `[j0, n)`: coalesced on both sides, but moves
   `2(n−j0)` elements per column instead of `2·ib`, so it wins only while
   `n − j0 < 16·ib` (≈ n = 512 at ib = 32) and loses on the leading panels of a
   large problem. Needs an out-of-place buffer or an in-place cycle walk.
2. **row-major staging** of the trailing block — the only way to make the p side
   contiguous at all. A layout change, not a kernel change.

The same measurement from the other side prices `getrs`'s permutation at nrhs = 1
at 26% (float n=512), 11% (float n=2048), 2% and 1.4% (cdouble) — confirming the
baseline's finding that at one right-hand side the permutation is a rounding error.

---

## 6b. The tier window — `native_tier_preferred()` is declared, and measured

The scaffolding left the hook deliberately absent and said so in the test:
*"Delete this case when the tier sweep lands and the predicate is declared."*
The sweep has run (`run_tier.sh`, `tier.txt`): both arms **pinned**, in the
vendor-free build, with the resolved route read off every row — which matters,
because a `cta` pin above the per-type ceiling falls through to `automatic()`
and four rows did exactly that and are excluded.

Ratio is `blocked_ms / cta_ms`, so > 1 means CTA is ahead:

```
          n=64(8192)  n=76(8192)  n=96(8192)  n=100(4096)  n=128(4096)
float        1.74        1.48        1.49        1.68         1.13
cfloat       1.39        1.59        1.30        1.33          --
cdouble      1.37        1.09         --          --           --
double       0.98        0.85        0.77        1.00          --
```

and `double` re-run across four batches at its worst order, because a one-cell
window is exactly the over-fit this campaign keeps warning about:

```
n=76      b=2048  b=4096  b=8192  b=16384
double     0.78    0.84    0.85     0.85
float      1.19    1.44    1.48     1.48
cfloat     1.31    1.55    1.59     1.60
cdouble    1.04    1.08    1.09     1.10
```

One-directional, flat in batch, per type, every relative sd < 0.2%. **So `double`
alone prefers the blocked driver below its own CTA ceiling**, and that is the
whole window: `cta_max_order = 32` for double, unbounded for the other three.

`n ≤ 32` goes back to CTA for double too, and that is not a hedge: there the
blocked driver's `nb` is `min(32, n) = n`, so it runs ONE panel whose leaf **is**
the CTA device function. The two arms are the same code, measured identical
(1.8126 vs 1.8113 ms at n=32, batch 8192), and CTA is the cheaper spelling of it
— one launch instead of three, no pointer arrays, no workspace draw.

It moves **nothing** in a vendor-present build: the hook is consulted only inside
`route_resolve.hh`'s `!vendor_available` branch. That is the whole reason it is
the third predicate and not a `preferred()` window.

**Break:** widening double's ceiling to infinity (`break.py tier_window`) removes
the one place the hook disagrees with `kGetrfOrder`, and
`RouteGetrf.NativeTierPreferredIsDeclaredAndPinsTheMeasuredTierChoice` goes
**RED** on exactly the double-resolves-to-Blocked assertion.

---

## 7. Register residency

Gate: **zero spill on entry functions**, and `registers × work-group ≤ 65536`.

| target | entry functions | with spill | max registers among the new kernels | link |
|---|---|---|---|---|
| `batchlas_extensions_cta` | 880 (was 848) | **0** | 50 (`GetrfPanelGlobalKernel<cdouble>`) | 142.2 s (was 122.7) |
| `batchlas_extensions_factorization` | 464 (was 424) | **0** | 40 (`LuLaswpKernel<GetrsLaswpTag, cdouble>`) | 21.3 s (was 21.8) |

At the widest work-group this file launches (512), 50 × 512 = 25,600, well inside
the 65,536 per-block limit. The 16 all-function spills on the CTA target are the
pre-existing 255-register `gesvdj_cta_impl<complex<double>>`, not a regression.
Device-link cost of WP6: **+19.5 s** on the CTA cluster, none on the other.

---

## 8. What this directory does NOT settle

1. **No `preferred()` window.** All three tables still return `false`
   everywhere. The grid above is the input a routing step needs, but a window
   written from it would be a claim about crossovers at cells the grid samples
   only coarsely (7 orders), and `getrf`'s crossover moves the moment §6's laswp
   work lands.
2. **The tier window is measured but narrow.** `native_tier_preferred()` IS
   declared (see §6b) and rests on 5 orders × 1–4 batches per type. The bands
   between each type's last measured order and its capacity ceiling — float
   129…155, cfloat 101…109, cdouble 77 — are EXTRAPOLATED onto CTA. cdouble is
   the one whose advantage is visibly collapsing (1.37× at n=64 → 1.09× at
   n=76 against a ceiling of 77) and is where a re-measurement would find a
   crossover first.
4. **`nb = 32` for every type is not tuned.** It satisfies the structural
   constraints (multiple of 16; ≥ 32 so complex reaches the wide-scalar GEMM)
   and nothing more.
5. **The work-group rule is not tuned either.** `getrf_leaf_wg` reproduces the
   two measured best widths (256 at n=64, 512 at n=128) and extrapolates; the
   baseline measured an 8.3× spread across widths, so a real sweep is owed.
6. **`double` on the LU trailing path still has no register GEMM kernel** —
   `Tiled16` at all 13 shapes, structural and outside WP6 (the wide-scalar
   CTA-count relaxation is complex-only and the other door needs
   `min_dim ≥ 256`, which `k = nb` can never satisfy). It is visible in the
   grid as double's 0.265–0.921× band below n = 2048.
7. **The `getrs` gather was not built.** Measured worth +0.38 geomean at
   nrhs = 64 and nothing at nrhs = 1, at the price of an out-of-place RHS
   (67 MB at n=2048, nrhs=64, batch=32). Left to the routing step, where a
   `preferred()` window on `GetrsShape::nrhs()` is what would justify it.
8. **`Backend::NETLIB` on a GPU queue is not gated.** The native arms write
   packed int32 pivots, which is right for CUDA and ROCm and wrong for NETLIB's
   genuine int64 span. `supports()` requires `is_gpu`, and NETLIB is a host
   backend whose every path round-trips through host pointers, so the
   combination is not a supported configuration — but no gate refuses it, and
   adding one would need a `GetrfShape.backend` test the existing route tests
   (which leave `backend` at `AUTO`) cannot see.
9. **Two pre-existing vendor defects were not fixed**, as the ground brief
   recorded them: the CUDA `getrs` `batch ≤ 1` crash (`cusolverDnXgetrs` handed
   packed int32 as `const int64_t*`), and NETLIB `getri`'s `std::copy(..., n*n,
   ...)` which ignores `ld`.

---

## 9. Files

**Programs**
`luverify.cpp` — the whole harness: modes `getrf` / `getrs` / `getri` /
`singular` / `pivmetric` / `params`; `build_v.sh`, `build_nv.sh`.

**Runners**
`run_verify.sh` (correctness sweep), `run_params.sh` (capacity + leaf choice),
`run_singular.sh`, `run_pivmetric.sh`, `run_grid.sh` + `run_both_grids.sh`
(the A/B), `run_crossbuild.sh` (same routes, two builds), `run_break.sh`,
`run_ctest.sh`.

**Breaks**
`break.py` — six correctness breaks and three timing-only ones, each with the
property it corrupts written at the site, and an exactly-once anchor check.

**Data**
`grid_vendor.csv`, `grid_native.csv`, `summary.txt` (`analyse.py`),
`crossbuild.txt`, `laswp_cost.txt`, `break_*.txt`.

**Measurement hygiene actually applied**: GPU 1 pinned throughout; the two arms
run SEQUENTIALLY, never concurrently; `WARM_S` seconds of untimed warm-up per
run; medians of 5 reps with mean and relative sd on every row; > 10% relative sd
would be discarded and named (none was); nothing timed under
`BATCHLAS_KERNEL_TRACE`; every route printed on its own row; correctness checked
in process against a host reference on every timed row.

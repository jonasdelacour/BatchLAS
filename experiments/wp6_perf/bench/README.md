# WP6-PERF — the fused narrow-RHS `getrs` tier, measured, and the routing decision

This is the MEASURE step for the fused `getrs` tier (`src/extensions/getrs_fused.cc`,
`{Native, CTA}` in `route_getrs.hh`). It re-derives nothing from the implementer's
`experiments/wp6_getrs/`: every number here comes from its own runs, on
`experiments/wp6_lu/bench/lubench6.cpp` — **the same harness, the same build
scripts and the same cell format WP6 used** — so the BEFORE column here and the
BEFORE column in `wp6_lu/bench/README.md` are the same cell and not merely the
same word.

`preferred()` was **not edited** *by this directory's measure step*. THE REPAIR STEP
THEN LANDED IT, and the record of what shipped is one level up in
`experiments/wp6_perf/README.md` — read that first. Three things below were
superseded or corrected there:

* **the window shipped**, so this file's "no unpinned CUDA user reaches it" is no
  longer true: `route_diff.sh` records 27 decisions moving `vendor:auto ->
  native:cta`, and `run_default.sh` / `default_summary.txt` measure the unpinned,
  vendor-present default at **2.084x** over 63 cells with zero losses;
* **a fourth flatness pass (`flat4`) was added**, because C3's stated minimum came
  from a cell measured at ONE batch. It ladders `nrhs = 2` at `n = 128` and `1024`
  and everything at `n = 32` and `256`. Every row FLAT-WIN, zero crossings; the
  minimum is now a full ladder whose lowest rung is **1.116x**;
* **`analyse_window.py` now pools `flat4` too** (461 cells, not 354), so the numbers
  in section 4 below are the pre-`flat4` ones. The re-scored table is in the parent
  README.

Section 5's proposal was accepted as written.

---

## 0. The one-paragraph answer

**At `nrhs = 1` the fused tier turns WP6's worst op into its best.** WP6 measured
native `getrs` at **0.32×** of `cublas?getrsBatched` at `nrhs = 1`, rising to
1.36× at `nrhs = 128`; on the same harness the fused tier is **2.12× at `nrhs = 1`
with 28 wins in 28 cells**, and **7.9× faster than the composition it replaces**.
The win is one-directional and flat across every batch ladder this box holds. It
does **not** extend to the whole capability: the tier is instantiated to
`nrhs ≤ 8`, and against cuBLAS it stops being a win between `nrhs = 2` and
`nrhs = 8`, per type and per order. The window that is a WIN AT EVERY MEASURED
CELL, all 354 of them, is

```
    nrhs <= 2   for every type and every order
  + nrhs <= 4   for float only
```

— 215 cells, geomean **2.14×**, minimum **1.12×**, **zero losses**. Every wider
window that was scored carries losses. `getrf` and `getri` are **unchanged**
(geomean after/before 0.9996 and 0.9983 over 96 cells each).

---

## 1. Both arms reproduce WP6 before any ratio is quoted

`wp6_reproduction.txt`, produced by `check_wp6.py`.

| arm | shared cells with `wp6_lu/bench` | worst disagreement | cells above 5 % |
|---|---|---|---|
| cuBLAS (`vendor`) | 60 | **2.90 %** | 0 |
| the composition (`blocked`, i.e. BEFORE) | 42 | **0.76 %** | 0 |

The order axis at `nrhs = 1` is the sharpest case: WP6 published float
`0.214 / 0.515 / 0.568 / 0.647 / 0.606` at `n = 64…2048`; this directory measures
`0.213 / 0.508 / 0.566 / 0.648 / 0.604`. cdouble reproduces to three decimals at
every one of those five orders. So the disagreements below are about the FUSED
TIER, not about the machine.

At the brief's own cell — float `n = 512`, `nrhs = 1`, batch 512, one public API
call:

```
  cuBLAS            1.4427 ms   (orchestrator 1.4500, WP6 1.4490)
  composition       2.5491 ms   (orchestrator 2.5376)
  fused tier        0.6498 ms   = 2.22x cuBLAS, 3.92x the composition
```

---

## 2. The three arms, and why two of them are pins

```
  vendor   lubench6_v  + BATCHLAS_GETRS_ROUTE=vendor          -> cublas?getrsBatched
  blocked  lubench6_nv + BATCHLAS_GETRS_ROUTE=native:blocked  -> the WP6 composition  (BEFORE)
  cta      lubench6_nv + BATCHLAS_GETRS_ROUTE=native:cta      -> the fused tier       (AFTER)
```

wp6_lu's rule — *the arm is the binary, never a pin* — is about vendor-vs-native
and still holds: the vendor arm is the vendor-PRESENT binary and the native arms
are the vendor-FREE one. Choosing between two NATIVE tiers cannot be done by the
binary, so it is done by the pin — and **every pin is verified from the resolved
route printed on the row**, because an unsupported forced route falls through to
`automatic()` (`route_resolve.hh:165 → :175`) and would silently measure the other
tier.

`run_cells.sh` differs from wp6_lu's by exactly one thing: it pins the **GETRS**
variable alone rather than all three LU variables together, so a `native:cta`
value cannot also pin `getrf` and change the untimed factorisation's arm partway
up the `n` ladder.

---

## 3. Hygiene, and the discard rule, fixed before it was applied

* **GPU 1 pinned** (`CUDA_VISIBLE_DEVICES=1`) for every run; the arms never ran
  concurrently. Two RTX 4090s in this box; co-running fabricates results.
* **JIT and clocks warmed** `WARM_S` seconds per cell, untimed and discarded
  (0.8 s for every table here, 1.0 s for the three-pass re-run).
* **Medians of 5 reps** (9 for the re-run), with mean and relative sd on every row.
* **Correctness in process on every timed row** against a HOST oracle — residual
  under `Tol<T>`, plus the factorisation's own residual and a non-zero
  non-diagonal pivot count. A fast wrong answer cannot enter a table here.
* **The resolved route is printed on every row** and is checked by the reader.
* **Nothing was run under `BATCHLAS_KERNEL_TRACE`**, and there are no nsys
  captures in this directory at all.

**THE DISCARD RULE** (`analyse.py`): a cell is dropped and NAMED when any arm is
flagged `BAD`, any arm's relative sd exceeds 10 %, an arm is missing, or **the pin
did not take**. Applied across the six sweeps:

| reason | cells |
|---|---|
| `cta` pin fell through the fused tier's CAPACITY ceiling | 162 rows across the six sweeps |
| vendor arm relative sd above 10 % | 2 distinct cells (3 rows) |
| anything else | 0 |

Those 162 rows are not noise and are not failures: they are `n·nrhs` above
`fused_max_elems` or `nrhs` above `kGetrsFusedMaxRhs = 8`, i.e. the capacity gate
doing its job, and at those shapes the shipped vendor-free route IS the
composition. `analyse_shipped.py` reads them that way rather than discarding them.

**THE READER'S OWN FIRST VERSION WAS WRONG AND THE RULE CAUGHT IT.** `lubench6.cpp`
prints 16 columns for `getrf`/`getri` and 15 for `getrs` under one 16-column
header, so `csv.DictReader` returned `flag=None` for every `getrs` row and the
verdict landed in the column named `extra2`. The first run dropped all 168 cells
with "flag=None" — the good failure. The bad one would have been a reader that
treats `None` as "not BAD" and quotes a geomean over unchecked rows. `analyse.py`
now reads by position and takes the flag as the LAST field.

**THE TWO REJECTED CELLS, RE-RUN IN THREE PASSES** (`run_noisy.sh`,
`noisy_p*_{vendor,cta}.csv`), because a relative sd above 10 % on a heavy-tailed
rep distribution is not the same thing as an unstable median:

```
  float n=64  nrhs=1 b=8192   vendor 0.3092 / 0.3056 / 0.3084 ms   cta 0.1765 / 0.1781 / 0.1761
                              ratio  1.752  / 1.716  / 1.751
  float n=512 nrhs=1 b=64     vendor 0.5343 / 0.5284 / 0.5333 ms   cta 0.2545 / 0.2499 / 0.2210
                              ratio  2.099  / 2.114  / 2.413
```

The vendor median reproduces to 1.2 % across passes in both cells; the cta arm
moves 15 % at `n = 512, b = 64` between passes (in-pass relative sd 0.13–0.21 %,
so it is a between-pass state change, not rep noise). **Both cells are wins on
every pass**, so neither changes anything below; they are reported rather than
quietly reinstated.

**CROSS-PASS REPRODUCTION.** 68 cells were measured by more than one sweep. The
worst spread between sweeps is **4.7 %** (float `n = 512, nrhs = 2, b = 512`:
2.180 / 2.175 / 2.276) and the median duplicate agrees to under 2 %. See the head
of `window_summary.txt`.

---

## 4. THE GRID — WP6's own getrs table, BEFORE and AFTER

`grid_cells.txt` → `grid_{vendor,blocked,cta}.csv` → `grid_summary.txt`.
All four types, `n = 32…2048`, `nrhs = 1, 2, 4, 16, 64, 128`, at the saturating
batch schedule `32:8192  64:8192  128:4096  256:2048  512:512  1024:128  2048:32`
(the top of each rung of wp6_lu's `SAT_LADDER`; the shared orders take the same
batch wp6_lu's `getrs_cells.txt` used).

### Geomean of `cuBLAS_ms / native_ms`, by nrhs, over 28 (type, n) cells each

```
  nrhs              1        2        4       16       64      128
  BEFORE        0.256    0.331    0.326    0.457    1.094    1.478
    wins         0/28     2/28     2/28     3/28    19/28    24/28
  AFTER         2.117    2.173    1.421        -        -        -
    cells          28       28       27        0        0        0
    wins        28/28    28/28    22/27      0/0      0/0      0/0
```

`AFTER` is blank at `nrhs ≥ 16` because the tier is not instantiated that wide —
`kGetrsFusedMaxRhs = 8`. At those widths the shipped route is still the
composition, so AFTER = BEFORE there by construction.

**The AFTER row is WP6's headline claim inverted.** WP6: *0.32× at `nrhs = 1`
rising to 1.36× at 128*. Now: **2.12× at `nrhs = 1`**, and the crossover has moved
to the other end of the axis.

### `nrhs = 1`, per (type, n) — `cuBLAS_ms / native_ms`

```
  type          n=32     n=64    n=128    n=256    n=512   n=1024   n=2048
  float  BEFORE 0.201    0.213    0.354    0.508    0.566    0.648    0.604
  float  AFTER  2.453    1.720    1.830    1.905    2.220    2.466    2.275
  double BEFORE 0.190    0.206    0.233    0.256    0.321    0.675    0.733
  double AFTER  3.839    3.647    2.801    2.036    2.025    2.592    2.912
  cfloat BEFORE 0.099    0.158    0.300    0.357    0.423    0.550    0.484
  cfloat AFTER  1.602    1.277    1.588    1.394    1.611    1.866    1.991
  cdoubl BEFORE 0.092    0.110    0.099    0.086    0.083    0.150    0.219
  cdoubl AFTER  2.144    2.632    2.397    1.897    1.765    2.215    2.160
```

WP6's worst cell in the whole LU family was `getrs` cdouble `n = 512, nrhs = 1` at
**0.083×**. It is now **1.765×** — a factor of **21** on that cell.

### How much the fused tier bought over the composition

```
  nrhs 1   geomean 8.255x over 28 cells, min 3.391, max 24.093
  nrhs 2   geomean 6.560x over 28 cells, min 2.657, max 18.192
  nrhs 4   geomean 4.403x over 27 cells, min 1.420, max  8.268
  nrhs 8   geomean 2.666x over 24 cells, min 1.497, max  3.818   (w8_summary.txt)
```

### What a vendor-free build now delivers over the WHOLE grid

`analyse_shipped.py` → `shipped_summary.txt`. The shipped arm at each cell is what
the route column says — fused where the capacity gate admits it (83 cells),
composition where it does not (85 cells):

```
  168 cells, cuBLAS_ms / native_ms
    BEFORE (composition everywhere)  geomean 0.523,  50 wins
    AFTER  (shipped vendor-free)     geomean 1.291, 124 wins
    the op itself, BEFORE_ms / AFTER_ms: geomean 2.468
```

---

## 5. THE CROSSOVER IN nrhs, AND IT IS FLAT IN BATCH — OR IT IS NOT, PER CLAUSE

Three separate questions live on the `nrhs` axis and they have different answers.

### 5a. Against the COMPOSITION there is no crossover inside the capability

`w8_summary.txt`, `nrhs_summary.txt`. `blocked_ms / cta_ms` is above 1 at every
one of the 107 pooled cells measured at `nrhs ≤ 8` — geomean **5.148×**, worst
**1.419** (double `n = 2048, nrhs = 4, b = 32`), zero losses. The composition is never the better native arm where the fused tier
can run, which is what `native_tier_preferred` already says.

### 5b. Against cuBLAS the crossover is real, and it is per type and per order

`nrhs_cells.txt` and `w8_cells.txt` fill in the widths the headline grid skipped.
`cuBLAS_ms / cta_ms`, saturating batch:

```
  nrhs = 4      n=32     n=64    n=128    n=256    n=512   n=1024   n=2048
  float        1.357    1.351    1.679    1.739    1.893    1.857    1.521
  double       0.842    0.926    1.099    1.184    3.100    3.141    1.813
  cfloat       0.806    1.151    1.411    1.369    1.486    1.470    1.205
  cdouble      0.577    0.703    1.230    1.879    2.197    2.673        -

  nrhs = 8      n=32     n=64    n=128    n=256    n=512   n=1024   n=2048
  float        1.032    1.209    1.522    1.609    1.549    1.335    1.096
  double       0.354    0.414    0.559    0.545    1.230    1.801        -
  cfloat       0.596    0.854    1.013    0.813    0.799    1.040        -
  cdouble      0.303    0.370    0.610    0.701    0.717        -        -
```

Geomeans over all 28 (type, n) cells: `nrhs = 4` → **1.422**, 22 wins;
`nrhs = 8` → **0.819**, 11 wins of 24. **`nrhs = 8` is a net LOSS.**

### 5c. FLATNESS IN BATCH — three passes, and the third one refuted a window

A window read at one batch per order is the over-fit this campaign keeps paying
for, so every candidate clause was re-read across the full wp6_lu `SAT_LADDER`.
`flat_summary.txt`, `flat2_summary.txt`, `flat3_summary.txt`.

**`nrhs = 1` and `nrhs = 2`: FLAT-WIN on every ladder, every type, every order
measured** — 20 ladder rows at `nrhs = 1` and 12 at `nrhs = 2`, not one of them crossing 1.0.

```
  nrhs = 1, cuBLAS_ms / cta_ms          nrhs = 2, cuBLAS_ms / cta_ms
  cdouble n=512  b=64..1024             cdouble n=512  b=64..1024
     2.854 2.750 2.353 1.771 1.602         2.775 3.743 4.542 4.895 4.765
  cfloat  n=64   b=1024..16384          cfloat  n=64   b=1024..16384
     2.810 2.359 1.437 1.280 1.242         2.088 1.708 1.270 1.306 1.300
  float   n=2048 b=4..64                float   n=2048 b=4..64
     1.896 1.536 1.982 2.279 2.310         1.975 2.066 2.309 2.415 2.260
```

Minimum over all 111 `nrhs = 1` cells: **1.242**. Over all 76 `nrhs = 2` cells:
**1.123**.

**`nrhs = 4` for float: FLAT-WIN at all five orders swept** (`n = 64, 128, 512,
1024, 2048`), minimum **1.133** at `n = 2048, b = 4`, ladder spread 1.12–1.59.

**`nrhs = 4` for the other three types: CROSSES, and not only at the ends.**
Eight ladder rows cross 1.0:

```
  double  n=64    0.911 .. 1.121     double  n=128   0.940 at b=2048   (mid-ladder)
  double  n=512   FLAT-WIN           double  n=2048  0.953 at b=4
  cfloat  n=64    0.952 .. 1.268     cfloat  n=1024  0.976 at b=16
  cfloat  n=512   FLAT-WIN           cfloat  n=2048  0.886 at b=4
  cdouble n=64    FLAT-LOSS          cdouble n=128   0.980 at b=1024   (mid-ladder)
  cdouble n=512   FLAT-WIN           cdouble n=1024  0.987 at b=16
```

The `n = 2048, b = 4..8` dips have a mechanism — the fused tier is ONE WORK-GROUP
PER MATRIX, so the CTA count IS the batch and batch 4 occupies 4 of 128 SMs — and
a `batch ≥ 16` guard closes them. **The `n = 128` and `n = 1024` dips do not.**
They sit in the MIDDLE of their ladders (`b = 2048` at `n = 128`, `b = 1024` at
`n = 128` for cdouble), so no boundary in `n` or in batch removes them. That is
`flat3`, and it is the pass that killed candidate C8 below.

**`nrhs = 8`: crosses everywhere except float `n ≤ 512`**, and float `n = 2048`
crosses too (0.686 at `b = 4`). Not a window.

---

## 6. THE ROUTING DECISION — the proposal, and every candidate that lost

`analyse_window.py` → `window_summary.txt`. Every `getrs` cell from all six
sweeps is pooled, deduplicated on `(type, n, nrhs, batch)`, filtered by the
discard rule, and each candidate predicate is asked what is INSIDE it (losses?)
and what it leaves OUTSIDE (wins handed to the vendor). **354 cells.**

```
  candidate                                        inside  geomean   min   losses | wins left
  C1  nrhs <= 1                                       111    2.256  1.242      0  |  188
  C2  nrhs <= 2                                       187    2.238  1.123      0  |  112
  C3  C2 + (float and nrhs <= 4)                      215    2.142  1.123      0  |   84
  C4  C3 + (float and nrhs <= 8)                      234    2.039  0.686      3  |   68
  C5  C3 + (nrhs <= 4 and n >= 128)                   276    1.976  0.886      7  |   30
  C6  nrhs <= 4, every type                           294    1.881  0.577     20  |   25
  C7  nrhs <= 8, every type (the whole capability)    354    1.625  0.294     55  |    0
  C8  C5 + (that clause needs batch >= 16)            272    1.997  0.940      4  |   31
  C9  C5 + (that clause bounded at n <= 1024)         266    2.010  0.940      4  |   37
```

**C3 is the widest window with ZERO losses, and it is the recommendation.**

### The proposed predicate

```cpp
    // MEASURED, experiments/wp6_perf/bench/. 354 cells, four types,
    // n = 32..2048, nrhs = 1..128, the full wp6_lu batch ladder at five orders,
    // every route verified from the resolved-route column. Ratio is
    // cublas?getrsBatched / fused, so above 1 means the fused tier is ahead.
    static bool preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return false;
        if (r.algo != Algorithm::CTA) return false;   // the composition is a
                                                      // measured loss at every
                                                      // nrhs the fused tier serves
        if (s.nrhs() <= 2) return true;               // clause A
        if constexpr (std::is_same_v<T, float>) {     // clause B
            if (s.nrhs() <= 4) return true;
        }
        return false;
    }
```

`supports()` already refuses anything the tier cannot launch, and `resolve_route`
requires `supports() && preferred()`, so no capacity term is repeated here — and
none may be: a speed threshold in `supports()` would make a pinned `native:cta`
fall through to `automatic()`.

### Clause A — `nrhs <= 2`, every type, every order

**187 cells, geomean 2.238×, minimum 1.123×, zero losses.** 32 batch-ladder rows,
all FLAT-WIN. Worst cell `cdouble n = 32, nrhs = 2, b = 8192` at 1.123×; best
`cdouble n = 512, nrhs = 2, b = 512` at 4.919×. The `nrhs = 1` half alone is 111
cells at geomean 2.256×, minimum 1.242×.

This clause is the one that matters: `linalg::solve` and the Python binding pass
the caller's own `B.cols()` straight through, and a linear solve is a single
right-hand side unless the caller says otherwise.

### Clause B — float, `nrhs <= 4`

**28 additional cells (float at `nrhs` 3–4), geomean 1.598×, minimum 1.133×, zero
losses.** Full batch ladders at `n = 64, 128, 512, 1024, 2048`, every one
FLAT-WIN, ladder spread 1.12–1.59. It is float-only because at `nrhs = 4` the
other three types lose at `n ≤ 64` (down to 0.577× for cdouble) and cross 1.0
mid-ladder at `n = 128` and `n = 1024`.

### What C3 COSTS, stated rather than hidden

84 measured cells are wins the vendor keeps, geomean of the pooled outside band
1.060×. The largest are all `nrhs = 4`, non-float, large `n`:

```
  double  n=1024 nrhs=4 b=256   3.944      double  n=512  nrhs=4 b=512   3.097
  double  n=1024 nrhs=4 b=128   3.144      double  n=2048 nrhs=4 b=64    2.880
  cdouble n=512  nrhs=4 b=256   2.546      cdouble n=1024 nrhs=4 b=128   2.685
```

Those are real and they are large. **They were not taken because the same clause
that captures them also crosses below 1.0 at `n = 128` and `n = 1024` in the
middle of the batch ladder** (C5, C8, C9 above, and §5c). Recovering them needs
either a per-(type, order) window measured at more orders than this pass swept, or
a kernel change that removes the mid-ladder dip. Both are new work, and neither is
a reason to ship a window with losses in it.

### Rejected, with their numbers — these are RESULTS

* **C7 "prefer the whole capability"**: 55 losses of 354, worst **0.294×**
  (cdouble `n = 64, nrhs = 8`). The tier being ABLE to run a shape is not evidence
  it should.
* **C6 "`nrhs <= 4`, every type"**: 20 losses, worst 0.577×. The small-`n`
  complex cells are the whole problem.
* **C4 "float up to the full `nrhs = 8`"**: 3 losses, all float `n = 2048` at
  batch 4–16 (0.686 / 0.778 / 0.877) — CTA-count starvation.
* **C8 / C9 "`nrhs <= 4` for every type above `n = 128`, guarded"**: 4 losses each
  at 0.940–0.987. **These two were the leading proposal after the second flatness
  pass and were refuted by the third**, which is the entire reason `flat3` was
  run: the guard closes the small-batch end and cannot touch a dip at `b = 2048`.
* **C1 "`nrhs = 1` only"**: correct but needlessly narrow — it hands back 76 cells
  at `nrhs = 2` whose minimum is 1.123×.

### THE TWO TESTS THAT MUST CHANGE WITH IT, and one of them is BLIND

1. **`tests/getrf_tests.cc:2970-2974`** asserts
   `is_vendor(getrs_route(..., nrhs=1, NoTrans, vendor_available=true))` with the
   message *"preferred() moved without the measured grid that would justify it"*.
   Under C3 that route becomes `native:cta` and the assertion **goes red** — which
   is exactly what it is for. It must be REPLACED by the window, not deleted:
   assert `native:cta` inside C3 (`nrhs = 1`, and `nrhs = 4` for float) and
   `vendor` outside it (`nrhs = 8` for every type, `nrhs = 4` for double/cdouble
   at `n = 64`).

2. **`tests/route_vocabulary_tests.cc:2195` `PreferredIsFalseEverywhereAndAbsent
   DriverIsUnsupported` CANNOT SEE THIS CHANGE AT ALL, and that was checked rather
   than assumed.** `blindguard.cpp` (built and run; source in this directory)
   rebuilds the file's own `getrs_shape()` helper field for field and reports:

   ```
   route_vocabulary_tests' own getrs_shape(): 36 shapes swept,
       supports({Native,CTA}) true on 0 of them
   fused_max_elems=0 fused_max_nrhs=0 on that helper's shape
   with the capacities set: supports({Native,CTA}) = true
   VERDICT: the sweep CANNOT see a preferred() window on the fused tier.
   ```

   The helper sets `blocked_available` and leaves both fused capacity fields at 0,
   so `supports({Native, CTA})` is false, `resolve_route` never consults
   `preferred()` for CTA, and every `is_vendor(...)` assertion in that sweep holds
   no matter what window lands. **`route_vocabulary_tests` will stay 78/78 through
   this change, and that is not evidence.** The helper needs a
   `fused_max_elems` / `fused_max_nrhs` parameter and the sweep needs rows on both
   sides of the window.

3. `getrf_tests.cc`'s `RouteTableAndTheVendorFreeFallback` asserts vendor-present
   routing for `getrf` and `getri` only — it queries `getrs` with
   `vendor_available=false`. It is unaffected.

---

## 7. `getrf` and `getri` — unchanged, and measured rather than argued

The diff is `getrs`-only by inspection, but `getrf` and `getri` share the
`LuLaswp` kernel family and a tier table, so both were re-run on **wp6_lu's own
`order32` and `order1024` cells, at wp6_lu's own `WARM_S = 0.5` and `REPS = 3`**,
and diffed cell by cell against the recorded medians in
`wp6_lu/bench/order{32,1024}_{vendor,native}.csv`. `run_lu.sh`, `analyse_lu.py`,
`lu_summary.txt`.

```
  arm      cells   geomean AFTER/BEFORE   worst   best   outside +/-5 %
  vendor      96          0.9996          0.934  1.031        2
  native      96          0.9983          0.931  1.017        1
```

The three cells outside 5 % are all **float `n = 32`** — `getrf` b=1024 (0.934),
`getri` b=32 (0.935) on the VENDOR arm, and `getrf` b=32 (0.931) on the native
arm — at absolute times of 34–38 µs, the smallest cells in the table. **Two of the
three are on the vendor arm, which this change cannot touch**, so they are the
run-to-run floor of a 35 µs cell, not a regression; and all three moved in the
FAST direction. Nothing moved by more than 7 %, and nothing moved that the change
could reach.

---

## 8. Correctness — the full baseline, re-run in both builds

| gate | result |
|---|---|
| `build/tests/getrf_tests` | 200 tests ran, **95 passed, 0 FAILED**, 105 skipped |
| `build-novendor/tests/getrf_tests` | 200 tests ran, **87 passed, 0 FAILED** |
| `build/tests/route_vocabulary_tests` | **78 passed, 0 failed** |
| `build-novendor/tests/route_vocabulary_tests` | **78 passed, 0 failed** |
| `build`: `ctest -L "blas\|ortho"` | **100 % passed, 0 failed out of 23** |
| `build-novendor`: `ctest -LE slow` | **33 PASSED of 56** (23 failed) |

The 23 vendor-free failures are the recorded pre-existing set —
`options_api`, `syevx`, `lanczos`, `gemv`, `trsm`, `ortho`, `cond`, `ormqr`×3,
`orgqr`, `iluk`, `symm`, `hemm`, `herk`, `her2k`, `syrk`, `syr2k`, `syev`,
`trmm`, `sytrd_blocked`, `syev_cta`, `syev_blocked` — **no LU test among them**.

The brief quotes the `getrf_tests` baseline as 63/55 passed; the file now runs 200
tests because the implementer added cases. `95` and `87` are the pass counts of
the CURRENT file. **The number that is the gate is the FAILURE count, and it is
zero in both builds.**

In-process correctness is also on every one of the 1,820 timed rows in this
directory: the harness checks the solve residual against a host oracle before it
prints, and no row in any table here is flagged `BAD`.

---

## 9. Files and command lines

**Programs.** `lubench6_v` / `lubench6_nv` are
`experiments/wp6_lu/bench/lubench6.cpp` built by that directory's own scripts:

```bash
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
bash $W/experiments/wp6_lu/bench/build_v.sh  \
     $W/experiments/wp6_lu/bench/lubench6.cpp $W/experiments/wp6_perf/bench/lubench6_v
bash $W/experiments/wp6_lu/bench/build_nv.sh \
     $W/experiments/wp6_lu/bench/lubench6.cpp $W/experiments/wp6_perf/bench/lubench6_nv
```

**Sweeps.** Every cell list is generated by `gen_cells.py`, so each schedule is a
written-down decision:

```bash
cd $W/experiments/wp6_perf/bench
./gen_cells.py grid  > grid_cells.txt      # 168 cells: 4 types x n 32..2048 x nrhs 1,2,4,16,64,128
./gen_cells.py nrhs  > nrhs_cells.txt      #  96 cells: dense nrhs axis at n = 64, 512, 2048
./gen_cells.py w8    > w8_cells.txt        #  56 cells: nrhs 4 and 8 at all seven orders
./gen_cells.py flat  > flat_cells.txt      # 180 cells: batch ladders, nrhs 1/4/8
./gen_cells.py flat2 > flat2_cells.txt     # 115 cells: batch ladders at nrhs 2, and two more orders
./gen_cells.py flat3 > flat3_cells.txt     #  33 cells: nrhs 4 ladders, non-float, n = 128 and 1024
./gen_cells.py lu    > lu_cells.txt        #  96 cells: wp6_lu's order32 + order1024, getrf and getri

GPU=1 LIST=grid  bash run_grid.sh          # three arms: vendor, blocked, cta
GPU=1 LIST=nrhs  bash run_grid.sh
GPU=1 LIST=w8    bash run_grid.sh
GPU=1 LIST=flat  bash run_flat.sh          # two arms: vendor, cta
GPU=1 LIST=flat2 bash run_flat.sh
GPU=1 LIST=flat3 bash run_flat.sh
GPU=1           bash run_lu.sh             # getrf/getri regression, wp6_lu's WARM_S and REPS
GPU=1           bash run_noisy.sh          # the two rejected cells, 3 passes x 9 reps
```

**Readers.**

```bash
python3 analyse_nrhs.py grid  > grid_summary.txt
python3 analyse_nrhs.py nrhs  > nrhs_summary.txt
python3 analyse_nrhs.py w8    > w8_summary.txt
python3 analyse_flat.py flat  > flat_summary.txt
python3 analyse_flat.py flat2 > flat2_summary.txt
python3 analyse_flat.py flat3 > flat3_summary.txt
python3 analyse_shipped.py grid > shipped_summary.txt
python3 analyse_window.py     > window_summary.txt
python3 analyse_lu.py         > lu_summary.txt
python3 check_wp6.py          > wp6_reproduction.txt
/opt/dpcpp-cuda/bin/clang++ -std=c++20 -I$W/include -I$W/build/include \
    blindguard.cpp -o blindguard && ./blindguard
```

**Data.** `{grid,nrhs,w8}_{vendor,blocked,cta}.csv`,
`{flat,flat2,flat3}_{vendor,cta}.csv`, `lu_{vendor,native}.csv`,
`noisy_p{1,2,3}_{vendor,cta}.csv`, and the `*_err.txt` stderr streams which carry
the per-transA residual for every row. **Every cell measured is in those CSVs**;
the `*_summary.txt` files are derived and regenerate from them.

**Not committed**: the two binaries (`lubench6_v`, `lubench6_nv`, `blindguard`).
There are no `*.nsys-rep`, `*.sqlite` or trace JSON in this directory — nothing
here was captured under a profiler, and nothing was timed under one.

---

## 10. What this directory does NOT settle

1. **The 84 cells C3 leaves to the vendor** (§6). `nrhs = 4` at `n ≥ 128` for
   double and cdouble is worth up to 3.94×, and the only thing stopping it is a
   0.94–0.99 dip in the middle of two batch ladders. Whether that dip is a real
   occupancy cliff or a cuBLAS heuristic switching arms was not investigated.
2. **`nrhs = 3, 5, 6, 7` were never measured.** The window covers 3 (inside
   `nrhs ≤ 4` for float) by interpolation between two measured wins. 5–7 are
   outside the proposal, so nothing rests on them.
3. **The vendor is UNSATURATED at `n ≥ 1024`** on any ladder 24 GB holds
   (wp6_lu/bench §2). The `n = 1024` and `n = 2048` columns of §4 are therefore
   read against a latency-bound cuBLAS. §5c's ladders are the mitigation — the
   `nrhs ≤ 2` clause is a win at EVERY rung including the shallowest — but the
   headline ratios at those two orders should not be quoted without this line.
4. **Only `transA = NoTrans` was timed.** `NTRANS=1` throughout, wp6_lu's own
   setting: the transposed modes are a correctness question and cost the same two
   substitutions. Their residuals are checked by the implementer's sweeps and by
   `getrf_tests`, not here.
5. **No profile was taken.** The mechanism claims in this file are the
   implementer's nsys structure captures, cited, not re-measured.
6. **`preferred()` was not edited.** §6 is a proposal with its evidence, handed to
   the repair step, together with the two tests that have to change and the one
   that cannot see the change at all.

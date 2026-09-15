# Known defects, located and not fixed

Everything on this page is **in the tree today**. Each entry was found during the
vendor-independence campaign, located to a line, and left alone on purpose — because fixing it
was outside the work package that found it, because the fix is a route change that needs its own
measurement, or because the machine cannot observe it. None of them is a mystery: they are
liabilities with an address.

Two things this page is not. It is not a performance-debt list — those live per op under
[`../perf/`](../perf/README.md), one "Open debts" section each. And it is not history: what is
written here was re-checked against the working tree, and where a source document's claim did not
survive that check it is marked as such.

**Two entries no longer fit the title, and are kept in place rather than deleted so the numbering
stays stable.** #7 is **closed** — the tree grew the writer this page asked for and the entry went
stale. #10 is **diagnosed and fixed, pending verification** — the mechanism is located and the fix
is in the tree, but no run has confirmed it, and in this repository an unwatched guard is not a
verified one. Neither is "in the tree today" in the sense the paragraph above means.

The superseded root documents these were filed in are preserved at the git tag
`perf-evidence/vendor-independence` (`git show perf-evidence/vendor-independence:WP7_FILED_DEFECTS.md`).

## At a glance

| # | site | what is wrong | severity today |
|---|---|---|---|
| 1 | `src/extensions/ortho.cc:218-224` | the transposed arm builds a view whose extents and `ld` do not describe the memory, against a vector of the wrong length | latent — a shape check routes it to the vendor |
| 2 | `src/extra/cond.cc:46,52,127` | reaches into `dispatch::detail` and demands the **vendor** `syev` instead of resolving a route | throws in a vendor-free build |
| 3 | `src/extensions/lanczos.cc:107-111` | the level-3 call carries two right-hand-side columns and one is consumed | 2x work, right answer |
| 4 | `src/backends/rocsparse.cc:30-31,62-63` | `ConjTrans` maps to the conjugating enum for **real** scalars | inferred wrong answers on AMD; unobservable here |
| 5 | `src/backends/netlib_lapack.cc:508,520,537,549` | `trsm` reads `B` when `alpha == 0` | `NaN` from unwritten workspace |
| 6 | `src/backends/netlib_lapack.cc:1389` | `getri` copies `n*n` contiguous elements and ignores both `ld`s | wrong answer at padded `ld` |
| 7 | `src/backends/trsm_route.hh:51` | ~~the heterogeneous-batch rejection has no writer~~ | **not a defect — the field IS written; entry closed 2026-09-15** |
| 8 | `src/backends/syrk_custom_dispatch.cc:261` | a forced native `syrk` lands on a route that writes both triangles | wrong answer, forced routes only |
| 9 | `src/backends/syr2k_custom_dispatch.cc:210` | a forced native `syr2k` throws a cuBLASDx message it did not ask for | misleading diagnostic |
| 10 | grid `latrd` (`src/extensions/latrd_lower_panel.cc`, the grid kernel's column-update / sumsq pair) | a cross-sub-group read-after-write on `Ab(r, i)` with no barrier between the two loops | **fixed; armed 20/20 red on deletion under the amplified geometry; residual rate at the default geometry not bounded** |

## 1. `ortho`'s transposed arm builds a view that does not describe the memory

`src/extensions/ortho.cc:218-224`, inside the CGS lambda:

```cpp
auto A_i = transA == Transpose::NoTrans
      ? MatrixView<T, fmt>(A.data_ptr(), m, i, m, A.stride(), batch_size)
      : MatrixView<T, fmt>(A.data_ptr(), i, m, m, A.stride(), batch_size);
auto C      = VectorView(Ymem.data(), i, batch_size);
auto A_next = A(Slice(), i);
```

Under `transA = Trans` or `ConjTrans`, `is_A_trans` is true and `inv_trans` is `NoTrans`
(`:118-120`). Three things then disagree:

* `A_i` is declared `i` rows by `m` columns with `ld = m`. The leading dimension of a view onto
  the first `i` rows of a column-major `A` is `A.ld()`, not `m`.
* the call at `:227` is `gemv(A_i, A_next, C, {.transA = NoTrans})`, so `x` must have length
  `A_i.cols() == m`.
* `A_next = A(Slice(), i)` is **column** `i`, of length `A.rows()`. On the transposed arm the
  vectors being orthogonalised are the *rows* of `A`, so `A.rows()` is the vector **count**.

The lengths coincide only when `A.rows() == m`.

**Why it is not live.** `gemv_op_shape` (`src/backends/gemv_route.hh:75-78`) returns
`std::nullopt` when `X.size() != red_len` or `Y.size() != out_len`, which resolves to
`{Vendor, Auto}`. The call therefore goes to cuBLAS/OpenBLAS exactly as it did before a native
`gemv` existed, and the native kernel never sees it.

**Why it was left.** Turning today's silent misbehaviour into a host-level throw would put a
live path's failure on the work package that added the kernel, not on the caller that has been
wrong all along. The native kernel accepts exactly what the vendor accepts, deliberately.

**What fixing it needs.** A correct `A_i` for the transposed arm (`ld = A.ld()`, extents in the
stored orientation), an `A_next` that is the `i`-th *vector* rather than the `i`-th column, and a
test that actually runs it.

**Why no test caught it.** `tests/ortho_tests.cc:249` and `:293` both read
`const std::vector<Transpose> transposes = {Transpose::NoTrans};`. The fixture's
`check_orthonormality` helper handles `transQ == Trans` and forms `Q Qᴴ` for it (`:50-78`) — the
machinery is there and nothing drives it. The transposed arm of `ortho` has never been executed
by the suite, for any algorithm or type.

## 2. `cond` demands the vendor `syev` instead of resolving a route

`src/extra/cond.cc:46`, `:52` and `:127`:

```cpp
Event e = blas::dispatch::detail::syev_vendor_or_throw<B, T>(ctx, ...);
```

The buffer-size query and the call both reach past the public entry point into
`dispatch::detail` and name the vendor implementation. A vendor-free build has no vendor `syev`,
so this throws rather than falling to a native tier.

**Why it was left.** It is a routing-vocabulary defect owned by `syev`, not by any of the BLAS
work packages that found it; the fix is to call the routed `syev` and let `resolve_route` choose,
which touches `syev`'s route table and needs `syev`'s own measurement. See
[`../perf/dispatch.md`](../perf/dispatch.md) for the vocabulary.

**What fixing it needs.** Replace all three sites with the public `syev` / `syev_buffer_size`.
The workspace query has to move with the call — `syev_vendor_buffer_size_or_throw` throws in the
same build, so half a fix is no fix.

## 3. `lanczos` issues a two-column multiply and consumes one column

`src/extensions/lanczos.cc:107-111`:

```cpp
auto padded_vector = MatrixView(Vmem.data() + it*n, n, 2, n, (n+1)*n, batch_size);
...
spmm<B>(ctx, A, padded_vector, padded_output, ...);              // :110, sparse arm
gemm<B>(ctx, A, padded_vector, padded_output, GemmOptions<T>{}); // :112, dense arm
```

`padded_output` is likewise two columns wide (`:53`), and the kernel that consumes it reads one:
`local_v_next = Span(v_next_ptr + bid*2*n, n)` (`:127`) is column 0 only. Both arms — sparse and
dense — do twice the level-3 work the iteration needs. The answer is right; the second column is
computed against whatever occupies the next basis slot and then discarded.

**Why it was left.** It is a call-site defect in an extension, not in any kernel or route, and
belongs to whoever owns `lanczos`.

**Note for whoever picks it up.** `lanczos_tests` fails identically in the vendor-present and
vendor-free builds, and its failures are not attributable to any native kernel: re-run under
`BATCHLAS_SPMM_ROUTE=vendor`, the same two cases fail (`LanczosTestBase.LanczosTest`,
`LanczosTestBase.ToeplitzEigenpairs`). Do not read a green/red flip here as evidence about
routing.

## 4. rocSPARSE conjugates a real transpose

`src/backends/rocsparse.cc:30-31` and `:62-63` pass both operands' transpose modes straight
through:

```cpp
enum_convert<BackendLibrary::ROCSPARSE>(transA),
enum_convert<BackendLibrary::ROCSPARSE>(transB),
```

and that conversion (`src/linalg-impl.hh:327-329`) maps `Transpose::ConjTrans` to
`rocsparse_operation_conjugate_transpose` unconditionally, with no dependence on the scalar type.

This is the same defect that was **found and fixed** in cuSPARSE. On a real scalar `ConjTrans`
*is* `Trans` — conjugating a real number is the identity — and passing
`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE` with `CUDA_R_32F`/`CUDA_R_64F` silently produced wrong
results across the whole real `ConjTrans` family, on **both** operands (a call with
`transA = NoTrans, transB = ConjTrans` was wrong on the dense operand alone). The fix is the
type-conditional `cusparse_op<T>` helper at `src/backends/cusparse.cc:29-39`; the complex arms,
where the conjugating enum is the distinct and correct operation, were always right and stayed
untouched.

**Why it was left: it is inferred, not observed.** There is no AMD device on this machine. That
rocSPARSE mishandles the real conjugating enum the way cuSPARSE did is an inference from the
cuSPARSE finding. `scripts/rocm_syntax_check.sh` compiles the ROCm TUs (the headers are under
`/opt/rocm*/include/roc*/`) and catches signature drift, but it cannot run a kernel.

**What fixing it needs.** A `rocsparse_op<T>` mirroring `cusparse_op<T>`, applied to both
operands — and, before or after, one run of `tests/spmm_tests.cc` on real hardware, since that
suite is what exposed the cuSPARSE version.

## 5. netlib `trsm` reads `B` when `alpha == 0`

`src/backends/netlib_lapack.cc:508`, `:520`, `:537`, `:549` — all four arms of the host solve:

```cpp
T x = alpha * Bb.at(i, j, 0) - sum;
```

There is no `alpha == 0` quick return anywhere in `trsm_vendor`. Reference `xTRSM` sets `B` to
zero without reading it in that case, and the reason matters here: callers hand these ops a
`BumpAllocator` allocation that is **not zeroed**, and `0 * NaN` is `NaN`, so an operand that
should have dropped out of the arithmetic poisons the result instead.

**Why it was left.** The identical defect in `spmm` (`netlib_lapack.cc:248,272` — `A` read at
`alpha == 0`, `C` read at `beta == 0`) was fixed by the work package that owns `spmm`; `trsm`'s
belongs to `trsm` and was out of that package's scope. The native `trsm` bodies already make the
guarantee. See [`../perf/spmm.md`](../perf/spmm.md) for the fixed sibling.

**What fixing it needs.** Skip the `alpha` term and substitute `T(0)`, matching what the native
bodies do — plus an `alpha == 0` case in `tests/trsm_tests.cc` whose `B` is poisoned with `NaN`
before the call. A test that passes a merely *wrong* `B` cannot see this; the poison has to be
something that survives multiplication by zero.

## 6. netlib `getri` ignores the leading dimension

`src/backends/netlib_lapack.cc:1389`:

```cpp
std::copy(Ab.data_ptr(), Ab.data_ptr() + n * n, Cb.data_ptr());
```

Both views are copied as `n*n` contiguous elements. Neither `Ab.ld()` nor `Cb.ld()` is consulted,
so any padded leading dimension gives a wrong answer (and, if `C` is the tighter of the two, a
write past its last column). Pre-existing, recorded in [`../perf/lu.md`](../perf/lu.md), not
fixed. The correct form is the per-column `std::copy_n` already used 400 lines above at `:995`.

## 7. CLOSED — `trsm`'s heterogeneous-batch rejection *can* fire

**Closed 2026-09-15, by reading the file.** This entry claimed that `trsm_op_shape` never writes
`heterogeneous_batch`, so `route_trsm.hh:43`'s `if (s.heterogeneous_batch) return false;` could
never be reached with a true value. That is false in the working tree. `src/backends/trsm_route.hh:51`,
inside `trsm_op_shape`, reads:

```cpp
// supports() refuses a heterogeneous batch; without this the field keeps
// OpShape's default false and that correctness gate can never fire.
s.heterogeneous_batch = A.is_heterogeneous() || B.is_heterogeneous();
```

Writer and gate now agree, and the in-tree comment is a verbatim paraphrase of this defect entry —
i.e. the fix was made *in response to* this filing and the filing was never retired. The same
staleness had propagated into [`../perf/trsm.md`](../perf/trsm.md) (open debt 12 and the
`supports()` paragraph); both are corrected in the same pass.

**On what basis this is closed, and what is still not established.** Closed on source inspection
only: the field is written, `B` is checked as well as `A` (a wider check than `getrf`'s, which
sees one operand), and the default at `include/batchlas/blas/dispatch/route.hh:179` is no longer
what a heterogeneous batch would leave behind. The original entry's *second* sentence still
stands and is NOT closed: **no test in the tree constructs a heterogeneous `trsm`**, so the gate
is argued, not armed. Per this page's own checklist item 1, a gate nobody has watched go red is
not a verified gate. Anyone picking this up should build a heterogeneous `A` or `B`, assert
`resolve_trsm_route` returns the vendor arm, and — the part that actually matters — assert it
under `vendor_available == false`, where the route walk has nowhere left to go.

## 8, 9. Two forced-route defects in the level-3 dispatchers

Both are pre-existing, both were preserved deliberately rather than quietly improved, and both
are reachable only through an environment pin.

* **`BATCHLAS_SYRK_ROUTE=native` returns a wrong answer.** `{Native, Auto}` passes
  `syrk_use_cuda_custom`, then matches no arm inside `syrk_cuda_custom` (the gram arm needs
  `origin == Auto`, the tile arm needs `algo == TriangularTiles || origin == Auto`) and falls
  through to `syrk_cublasdx_fallback_gemm` at `src/backends/syrk_custom_dispatch.cc:261` —
  the `DiagFullGemm` route, which **writes both triangles**, clobbering the one the caller did
  not name. **No test in the tree sets `BATCHLAS_SYRK_ROUTE`.**
* **`BATCHLAS_SYR2K_ROUTE=native` throws a cuBLASDx message it did not ask for.** The throw at
  `src/backends/syr2k_custom_dispatch.cc:210` is not guarded by `forced`, so a non-fused named
  route reaching it gets a diagnostic about a fused kernel it never requested.

Full context in [`../perf/level3.md`](../perf/level3.md) and
[`../perf/dispatch.md`](../perf/dispatch.md).

## 10. The grid `latrd` path: an unsynchronised cross-sub-group read-after-write

**Status: diagnosed, fixed, and the fix ARMED.** Filed 2026-09-14 as a nondeterministic wrong
answer; the root cause was located on 2026-09-15, the barrier restored in
`src/extensions/latrd_lower_panel.cc`, and the restoration armed the same day by deleting it
again and observing red. The arming is under *What was run* below. Read that section before
quoting this as closed: the amplified leg is decisive, the unamplified leg is not.

### The observation, as filed

`SytrdBlockedLatrdGridCudaTest.GridMatchesLegacyTridiagonal` failed about **one run in seven** —
a flaky WRONG ANSWER, not a flaky timeout. **6 of 40** runs failed at the default routes, **4 of
40** with `BATCHLAS_GEMM_ROUTE=vendor`; indistinguishable, so routed GEMM was not the cause. The
disagreeing eigenvalue index was different on **every** failure observed (1058, 1090, 1219, 2135,
1063, 5253, 7377, 1366, 1155, 194). A representative miss at `n=1024 batch=8 nb=32 seed=456` gave
grid `-0.067254041269475984` against legacy `-0.067303749995033968`, tolerance `1.024e-08`: a
~5e-5 disagreement, four orders of magnitude outside tolerance and far too large to be an
association-order difference between two reduction trees.

### The mechanism

Verified twice by reading `src/extensions/latrd_lower_panel.cc`, and the line numbers below
re-checked against the working tree on **2026-09-15, after the barrier was inserted** (the insert
moved everything below it down by five lines). They are approximate on purpose — the file is
under concurrent edit — so **each step is quoted, and the excerpt, not the number, is the
citation**.

1. **The row partition.** For column `i` the grid kernel gives group `gg` a contiguous row block
   (~line 728):
   ```cpp
   const int chunk = (total_r + G - 1) / G;
   const int rlo = sycl::min(n, base_r + gg * chunk);     // base_r == i + 1
   const int rhi = sycl::min(n, rlo + chunk);
   const int slo = sycl::max(rlo, i + 2);                 // reflector tail start
   ```
2. **The write.** The column-update loop writes under the mapping `r = rlo + lid` (loop head
   ~line 750; the write `Ab(r, i) = val;` itself is ~14 lines further down, ~line 764, at the
   bottom of the `p < i` accumulation):
   ```cpp
   for (int r = rlo + lid; r < rhi; r += wg) { ... Ab(r, i) = val; }
   ```
3. **The read.** The sumsq loop immediately after re-reads the same column under a **different**
   mapping, `r = slo + lid` (loop head ~line 774, the read `sumsq += abs2_if_complex(Ab(r, i));`
   ~line 775):
   ```cpp
   for (int r = slo + lid; r < rhi; r += wg) { sumsq += abs2_if_complex(Ab(r, i)); }
   ```
4. **Nothing separated them.** The next synchronisation in the original grid kernel was
   `grid_barrier(it, bar, G);   // barrier 1` (~line 780) — *after* the sumsq loop, one loop too
   late.
5. **The offset is exactly one row.** For group `gg == 0`, `rlo == base_r == i + 1`, so
   `slo == max(i+1, i+2) == rlo + 1`. Work-item `lid` therefore reads row `rlo + lid + 1`, which
   is the row work-item `lid + 1` wrote in step 2. For `gg >= 1`, `rlo >= i + 2` so `slo == rlo`
   and each item reads only what it wrote — **the race is confined to group 0**.
6. **The legacy kernel in the same file has the barrier.** Its column update ends with the same
   `Ab(r, i) = val;` (~line 430) and is followed **immediately** by
   `it.barrier(sycl::access::fence_space::global_space);` (~line 432) before its own
   `x0 = i + 2` sumsq loop (~line 439). The grid rewrite dropped it; the fix restores the
   identical statement at ~line 770, between the write loop and the sumsq loop, with the
   write/read mappings recorded in a three-line comment immediately above it.

### Provenance: the grid rewrite, not the small-n campaign

The original filing said "**not caused by the small-n campaign (P0-P7)** — no `sytrd`, `latrd`,
`syr2k`, `her2k` or `steqr` source was modified by it". That sentence is **kept, but restated**,
because the diagnosis moved the defect from "somewhere, possibly routing" to a specific pair of
loops in a specific source file, and a blanket "no `latrd` source was modified" now reads as a
claim about a file that *is* modified in the working tree (by the fix above).

The precise form: the racing loop pair was introduced by **`87f6887`, 2026-08-03,
"latrd: add a multi-work-group panel path (`BATCHLAS_LATRD_IMPL=grid`)"** — that commit adds
`chunk`/`rlo`/`rhi`/`slo`, the `r = rlo + lid` write loop and the `r = slo + lid` sumsq loop with
no barrier between them, and the pre-fix tip of this branch still shows that gap.
**`5401f63`, the same day**, made the grid path the default for `n >= 768`, which is what turned a
latent opt-in race into a flake anybody could hit. The small-n campaign's own commits run
**2026-09-10 to 2026-09-14** (`30c0e8f` .. `a6f6294`) — six weeks later — and none of them touches
this file; the only edit to it in this working tree is the restored barrier. The supporting
measurement from the filing also still stands: pinning the one routed op these paths share
(`BATCHLAS_GEMM_ROUTE=vendor`) did not move the failure rate (4/40, against 6/40 at the default
routes).

**So: a pre-existing defect, six weeks older than P0-P7, found while the campaign was running.**
Do not let the campaign's dates in the revision history attach it to the campaign.

### When it can fire

Item `lid` and item `lid + 1` execute in lock step whenever they share a sub-group, so the race
only expresses itself at a sub-group boundary: `lid ≡ 31 (mod 32)`. That needs **both**

* `wg > 32`, so item `lid + 1` exists in another sub-group, and
* `chunk > 32`, so row `rlo + 32` is still inside `[slo, rhi)` and item 31 actually reads it.

`chunk = (total_r + G - 1) / G` depends on `n`, `i` and **`G` alone — not on `wg`**. That is the
part that makes the configuration counter-intuitive: *lowering* the group count raises `chunk`
and widens the race. Working through `latrd_grid_launch` (same file, ~line 1160-1215) with
`resident_cap = MAX_COMPUTE_UNITS = 128` on this box, `cap = 128 / batch`,
`G = min(cap, ceil(rows/32))` and `wg = ceil(ceil(rows/G)/32)*32` clamped to `[32, 256]`, the
condition reduces to `floor(128 / batch) < ceil((n-1) / 32)` — and when it holds, `wg >= 64`
follows automatically.

At the default `latrd_grid_min_n` of **768** (`src/util/settings.cc:176`) the grid path is the
default for `n >= 768` with **no environment variable set at all**. The filed configuration
`n=1024, batch=8` gives `cap=16`, `G=16`, `chunk=64`, `wg=64` — race live, exactly as reported.
So does `n=768` at `batch >= 6` (`cap=21 < 24`, `chunk=37`). At `batch=1..4` and `n=1024`,
`G = ceil(rows/32) = 32` and `chunk = 32`, which is **not** `> 32`: the default route is clean
there, which is part of why this took so long to see.

### What was run (2026-09-15)

R9 arming, four legs, each a full relink of the shared library (the `.so` link is the AOT device
compile, so none of these could have been a stale build):

| leg | geometry | expected | observed |
|---|---|---|---|
| barrier present | amplified, x20 | green | **green** |
| barrier present | default, x40 | green | **green** |
| **barrier deleted** | **amplified, x20** | **red** | **RED, 20 of 20** |
| barrier deleted | default, x40 | red | green |
| barrier restored | amplified, x20 | green | **green** |

The red leg fails at `tests/sytrd_blocked_tests.cc:545`, the grid-vs-legacy `ASSERT_NEAR` on the
diagonal, with differences of `4.45e-05` and `8.56e-04` against a tolerance of `1.024e-08` — four
to five orders out, the same signature as the original filing and far too large for a reduction
association-order difference.

**The fourth leg is the honest caveat and must not be filed off.** With the barrier deleted, the
*default* geometry passed 40 of 40. That is not evidence the default route is safe; it is the
expression rate. At `n=1024, batch=8` the default gives `wg=64` — two sub-groups, so exactly
**one** boundary lane crosses per column — against eight sub-groups and seven crossings per
column under the amplifier, over `chunk=512` rows instead of 64. The default window is roughly
16x narrower, which is exactly why the original flake was ~1 run in 7 rather than 1 in 1, and why
40 runs of a single filter can miss it. The amplified leg is what proves the barrier is
load-bearing; the unamplified leg proves only that 40 samples are too few to see this window.

What this therefore does and does not establish: the write/read hazard is real, the restored
barrier removes it, and the deleted-barrier configuration is reproducibly wrong under a geometry
the code reaches by supported environment variables. It does **not** establish a bound on the
residual rate at the default geometry — the barrier makes the hazard unexpressible by the memory
model, and that argument is stronger than any number of green runs, but it remains an argument.

### What is still owed

1. **An amplified repro, red before the fix and green after.** *(Done — see the table above.)*
   Force the group count **down**,
   which forces `chunk` **up**:
   ```
   BATCHLAS_LATRD_GRID_GROUPS=2 BATCHLAS_LATRD_GRID_WG=256 \
     ./build/presets/dev-tests/tests/sytrd_blocked_tests \
       --gtest_filter=SytrdBlockedLatrdGridCudaTest.GridMatchesLegacyTridiagonal
   ```
   At `n=1024` that gives `chunk = ceil(1023/2) = 512` and `wg = 256`, so eight sub-group
   boundaries per read pass instead of one — roughly a 16x wider window than the default
   configuration the flake was observed at. `GROUPS=2` only *lowers* `G` below the residency cap,
   so it does **not** need `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` and cannot deadlock the software
   grid barrier. The pre-fix leg has to be run with the barrier at ~line 770 deleted and the
   shared library relinked — the `.so` link is the AOT device compile, so a stale build proves
   nothing (checklist item 1).
2. **An unamplified loop**, the original 40-run form below, expected 0/40. Run post-fix: 0/40,
   green. But see the caveat above — the same 40-run form was *also* green with the barrier
   deleted, so on its own it does not distinguish *gone* from *narrowed*, and it did not settle
   that question here. Settling it needs either many more samples at the default geometry or
   `compute-sanitizer --tool racecheck`, which is not part of ctest and needs its own GPU slot.

```
for i in $(seq 1 40); do
  ./build/presets/dev-tests/tests/sytrd_blocked_tests \
    --gtest_filter=SytrdBlockedLatrdGridCudaTest.GridMatchesLegacyTridiagonal \
      >/dev/null 2>&1 || echo "fail $i"
done
```

### A remediation note in the original filing that is wrong

The filing suggested adding a test that **pins the work-group size high**. That is not the
remediation, and `tests/sytrd_blocked_tests.cc:597-612` is the proof:

```cpp
TEST(SytrdBlockedLatrdGridCudaTest, ForcedGroupCountsAgree) {
    for (const char* groups : {"2", "3", "7", "16", "64"}) {
        for (const char* wgs : {"32", "128"}) {
            ScopedEnvVar g("BATCHLAS_LATRD_GRID_GROUPS", groups);
            ScopedEnvVar w("BATCHLAS_LATRD_GRID_WG", wgs);
            run_latrd_grid_case<double>(129, 1, 32, "grid", 200.0);
```

`wg` is already pinned to 128 in half its cells, and the test has not been reported flaky. Two
corrections to the filing, in opposite directions:

* **Pinning `wg` high is necessary and not sufficient.** `wg > 32` is only one of the two
  conditions; `chunk > 32` is the other, and `chunk` is set by `G`. Of this test's ten
  `(groups, wg)` cells, the five at `wg = 32` cannot race at all, and of the five at `wg = 128`
  only `G = 2` and `G = 3` give `chunk > 32` on the first panel (`n_t = 129`, `total_r = 128`, so
  `chunk = 64` and `43`); `G = 7/16/64` give `chunk = 19/8/2` and are clean by construction.
* **But it is *not* "effectively clean", and this page should not say so.** Two of its ten cells
  do reach the race, and its tolerance is tight enough to see one: `eig_tol = 200 *
  tolerance<double>() = 200 * 1e-10 = 2e-8` (`tests/test_utils.hh:201-210`), against an observed
  disagreement of ~5e-5. The honest statement is that the test is an **unreliable** guard, not a
  blind one — at `n=129, batch=1` group 0 contributes exactly **one** crossing pair per column
  and only its first two panels have `chunk > 32`, so about **64** crossing pairs per run in its
  best cell, against about **2,300** for the flaky test (9 grid-path panels x 32 columns x 8
  batch items, one pair each). Roughly 40x less exposure, from a static count of the loops — not
  a failure-rate measurement. It has not been observed to fail; nobody has run it enough times to
  say it cannot.

The real remediation is a case that forces `G` **down** at a large `n` — the amplified repro
above — because that is the only knob that inflates `chunk`. `BATCHLAS_LATRD_GRID_WG` alone
cannot reach the defect no matter what it is set to.

## One filed claim that did not survive re-checking

[`../perf/lu.md`](../perf/lu.md) records "a latent vendor gate defect: `cublas.cc`'s `getrs` sits
in a TU gated on `BATCHLAS_HAS_CUBLAS`, so a cuBLAS-present / cuSOLVER-absent configure claims a
vendor it cannot link." Re-checked against the tree: `getrs_vendor` for CUDA calls
`cublas?getrsBatched` (`src/backends/cublas.cc:1491`) and nothing from cuSOLVER; `cublas.cc` is
added to `BACKEND_CUDA_SOURCES` under `BATCHLAS_HAS_CUBLAS` (`src/backends/CMakeLists.txt:69`);
and `factorization_vendor_available<Backend::CUDA>` is `BATCHLAS_HAS_CUBLAS`
(`include/batchlas/blas/dispatch/vendor_available.hh:42`). Gate and definition agree. Marked
`unverified` rather than deleted: the stated mismatch could not be reproduced, but the entry may
be describing an earlier `getrs` that did call cuSOLVER.

## The recurring failure mode: guards that cannot fail

This is the most expensive thing the campaign learned, and it is not about any one op.
Repeatedly, a suite that looked thorough was proved — by applying a deliberate break, rebuilding
and running — to be **structurally incapable** of failing for the property it named. Not "did not
happen to catch it": could not.

| the guard | what made it blind | how it was proved | the fix |
|---|---|---|---|
| the `trsm` barrier's own regression test | it drove V1 directly at n=16 and *asserted* it had cleared the work-group ladder. Clearing the ladder is necessary and not sufficient — the race needs more than one sub-group, and a final V1 block landing in the `N=16` bucket | applied with the barrier deleted and the library rebuilt: **green, twice** | drive V2 at order 48, q=976, batch=128 (`tests/trsm_tests.cc:538`); orders 48/77/80/109 fail 90-128 of 128 items deterministically, 32/33/64/65/96/155 are clean |
| all 232 `gemv` cases, on batch stride | every case used the **natural** stride (`a_stride == ld*n`, `x_stride == size*inc`), so a kernel deriving each stride instead of reading it passed the whole suite — while `ortho.cc:218-220` hands it an `A.stride()` its `ld*cols` does not equal, every CGS iteration | break `padstride`: exactly 32 cases red, all four of them new | four `stride_pad` cases, one per kernel body |
| `spmm`'s transposed `nnz` bound | the poison was a NaN at an **out-of-range** column, and the scatter's own range guard `continue`s *before* the multiply — poison and value went in the bin together, so the test was green because of a kernel guard, not the property it named | break `scatterBound` came back green over all 352 cases; two control runs then isolated it (broken bound + guard deleted → segfault; correct bound + guard deleted → 352/352 green) | poison with an **in-range** column and a large **finite** sentinel — an entry the scatter accepts and accumulates. Finite, not NaN: NaN is absorbing under atomic addition, says nothing about where it landed, and a fast-math build may fold the assertion away |
| `gemv`'s tail sub-group | out-of-range writes landed past the end of the allocation, where nothing was looking. **Three separate breaks came back green over 376 cases** | after adding 64 elements of poisoned guard band, `segTtailwrite` and `segTclampoff2` turn exactly the three partial-tail cases red | allocate guard, poison before the call, assert untouched after |
| `geqrf`'s tau/beta convention | the only test that could see it opened with `GTEST_SKIP` in a vendor-free build — a **null** in exactly the build the work exists for | break K3 shipped green vendor-free | an independent host `xGEQR2` reference (`ConventionMatchesReferenceLapackWithoutAVendor`); K3 is now red for all four types |
| `potrf`'s no-fold residual | computed over the **lower triangle only**, so writing the symmetric product into the upper triangle was invisible by construction | — | poison the opposite triangle and assert it survives bit for bit |

The running tally in the sources reaches "twelfth", but the ordinals are not consistent between
documents ([`../perf/potrf.md`](../perf/potrf.md) numbers the stale-pivot break fourth,
[`../perf/trsm.md`](../perf/trsm.md) numbers the `trsm` barrier fifth, and
[`../perf/lu.md`](../perf/lu.md) marks its own "seventh" `unverified` because no source assigns
it). Treat the count as a running campaign tally, not an index.

### The checklist this produces

Before trusting any guard in this tree:

1. **Apply the break.** A test is armed only if you have watched it go red for the specific
   defect it names. Rebuild between the break and the run — this is a device-linked library and a
   stale `.so` passes everything.
2. **Ask what the kernel does with your poison,** not whether a case exists. A defensive
   predicate sitting between the poison and the assertion makes the case vacuous. Every contract
   here has two independent implementations, and a poison tuned to one body's failure mode can be
   inert against the other's.
3. **Vary the axis you claim to cover.** Natural strides, square shapes, `ld == rows`, real data
   in a complex test, and a batch whose items all have the same `nnz` are each a whole axis that
   never moves.
4. **Look past the last element you assert on.** An out-of-bounds write into slack is silent.
5. **Check whether the suite runs at all in the build you care about.** A `GTEST_SKIP` on
   vendor-absence, or a fixture whose queue is a CPU queue, turns a suite into a null without
   ever reporting one.
6. **A coverage row is not a break.** Rows are keyed on a power-of-two `shape_class` and are
   first-writer-wins, so a row proves that *some* shape resolved to a route, never that *this*
   shape ran *that* body. Only a break red for one body proves the body ran.
7. **Make the break as narrow as the contract it denies.** A break that also falsifies its own
   named controls identifies nothing.
8. **`git diff` cannot verify the revert of an untracked file.** For a new source file, take an
   `md5sum` of the pristine copy *before* the first break and compare after the last.
9. **A residual bound is not a convention test.** Residuals catch a convention break
   *sometimes*, on some data, for some types. Assert the convention.
10. **Half the type list can be blind by construction.** For a real scalar, `ConjTrans` is
    `Trans` and a conjugate is the identity — defect 4 above is exactly this, and so was
    `zgeqr2` applying `conj(tau)`. Complex test data must have an imaginary part that is a
    *different* function of the indices than the real part, or the triangle is accidentally real,
    symmetric or Hermitian and the test proves nothing.

## Where the originals are

| filed in | now |
|---|---|
| `WP7_FILED_DEFECTS.md` | defects 1-3 above |
| `VENDOR_INDEPENDENCE_PLAN.md`, "defects found and filed" | defects 4-5, and the blind-guard tally |
| `VENDOR_FREE_BASELINE.md` | the ninth and eleventh blind guards |
| `WP1_LEVEL3_SPEC.md` | defects 8-9 (its stated fall-through destination is stale; the defect is not) |

All are retrievable at the tag: `git show perf-evidence/vendor-independence:<path>`.

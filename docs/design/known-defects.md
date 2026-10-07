# Known defects, located and not fixed

> **Covers:** defects located to a line and left in the tree, unverified candidates, and the
> recurring "guard that cannot fail" failure mode.
> **Status:** current; re-checked against the tree on 2026-09-15, 2026-09-30 and 2026-10-06
> (after flat kernel selection replaced the route layer). Referenced as
> `@ref md_docs_2design_2known-defects`; the page has no label of its own because that id is
> already cited.

Everything on this page is **in the tree today**. Each entry was found during the
vendor-independence campaign, located to a line, and left alone on purpose — because fixing it
was outside the work package that found it, because the fix is a route change that needs its own
measurement, or because the machine cannot observe it. None of them is a mystery: they are
liabilities with an address.

Two things this page is not. It is not a performance-debt list — those live per op under
[`../perf/`](../perf/README.md), one "Open debts" section each. And it is not history: what is
written here was re-checked against the working tree, and where a source document's claim did not
survive that check it is marked as such.

**Some entries no longer fit the title, and are kept in place rather than deleted so the numbering
stays stable.** #7 is **closed** — the tree grew the writer this page asked for and the entry went
stale. #8 and #9 are **closed** — the phase 5 rip deleted the route they describe. #10 is **diagnosed and fixed, pending verification** — the mechanism is located and the fix
is in the tree, but no run has confirmed it, and in this repository an unwatched guard is not a
verified one. Neither is "in the tree today" in the sense the paragraph above means.

**Citing an entry from code.** A heading that starts with a number (`## 11. ...`) gets a Doxygen
id with an `autotoc_md` prefix, so its GitHub slug is not a live anchor on the site and
`check_doc_anchors.py` rejects an `evidence:` pointer to it. An entry that code cites therefore
starts with distinctive text instead (`## Defect 11: ...`, `## Known defects: ...`). Entries 1, 3,
11, 12 and 14 are cited and carry the `Defect N:` form (renamed 2026-10-06, every pointer updated
in the same change); the other entries keep their numbered headings until something needs to cite
them. Line citations below were refreshed on 2026-10-06 where the code
still exists; citations into deleted files are marked as such.

The superseded root documents these were filed in are preserved at the git tag
`perf-evidence/vendor-independence` (`git show perf-evidence/vendor-independence:WP7_FILED_DEFECTS.md`).

## At a glance

| # | site | what is wrong | severity today |
|---|---|---|---|
| 1 | `src/extensions/ortho.cc:189-191` | the transposed arm builds a view whose extents and `ld` do not describe the memory, against a vector of the wrong length | latent — a shape check routes it to the vendor |
| 2 | `src/extra/cond.cc:48,54,131` | reaches into `dispatch::detail` and demands the **vendor** `syev` instead of calling the public one | throws in a vendor-free build |
| 3 | `src/extensions/lanczos.cc:112-117` | the level-3 call carries two right-hand-side columns and one is consumed | 2x work, right answer |
| 4 | `src/backends/rocsparse.cc:30-31,62-63` | `ConjTrans` maps to the conjugating enum for **real** scalars | inferred wrong answers on AMD; unobservable here |
| 5 | `src/backends/netlib_lapack.cc:484,496,513,525` | `trsm` reads `B` when `alpha == 0` | `NaN` from unwritten workspace |
| 6 | `src/backends/netlib_lapack.cc:1250` | `getri` copies `n*n` contiguous elements and ignores both `ld`s | wrong answer at padded `ld` |
| 7 | `src/backends/trsm_route.hh:51` (deleted in P3.3) | ~~the heterogeneous-batch rejection has no writer~~ | **not a defect — the field IS written; entry closed 2026-09-15** |
| 8 | `src/backends/syrk_custom_dispatch.cc` | ~~a forced native `syrk` lands on a route that writes both triangles~~ | **closed in the phase 5 rip: `native` is the tile kernel** |
| 9 | `src/backends/syr2k_custom_dispatch.cc` | ~~a forced native `syr2k` throws a cuBLASDx message it did not ask for~~ | **closed in the phase 5 rip** |
| 10 | grid `latrd` (`src/extensions/latrd_lower_panel.cc`, the grid kernel's column-update / sumsq pair) | a cross-sub-group read-after-write on `Ab(r, i)` with no barrier between the two loops | **fixed; armed 20/20 red on deletion under the amplified geometry; residual rate at the default geometry not bounded** |
| 11 | `src/sycl/gemm/epilogue_linear.hh`, `src/sycl/gemm_kernels.cc` (`launch_direct`) | native GEMM reads `C` at `beta == 0` | `NaN` from an unzeroed arena; worked around in `geqrf_blocked` |
| 12 | `src/ops/potrf/potrf.cc`, `src/ops/trsm/trsm.cc` (each `can_run(Vendor)`), `src/backends/cusolver.cc:72-77` | vendor `potrf` and `trsm` accept a heterogeneous batch and run at the full storage order (symm, syrk, syr2k, trmm: fixed, their vendor refuses one) | silent wrong answer on a direct heterogeneous call; posv refuses it upstream |
| 13 | `src/backends/cublas.cc` (`gemm_vendor_impl`, `gemv_vendor`), cuBLASLt; cuSPARSE spmm | complex<double> gemm/gemv with a unit dimension segfault inside cuBLASLt on one box, root cause unknown; two cuSPARSE spmm shapes misbehave | gemm worked around; gemv crashes `ortho_tests`; spmm refused in `can_run` |
| 14 | `gesvd_cta` (Upper), `gesvd_blocked` (Lower, n <= 32), `syev_cta` (Upper), `syev_blocked` (Lower, n <= 32), `syev_two_stage` (Lower) | the Hermitian drivers read the triangle the caller did not name | **wrong answer under Auto** for gesvd Hermitian Upper n <= 32 and syev cfloat n 9..32, cdouble n <= 32 with Upper |
| 15 | cuSOLVER `gesvdjBatched` | values-only, non-square input faults with `CUDA_ERROR_ILLEGAL_ADDRESS` | pinned vendor only; Auto never sends the shape there |
| 16 | `ormqr_blocked`'s sub-kernels | batch > 65535 exceeds the grid's dimension-2 limit and throws | throws under Auto at batch > 65535 |
| 17 | cuSPARSE spmm, operands off their natural alignment | silently mis-handled; not modelled in `can_run` | an explicit R3 waiver; tests skip misaligned cases unless pinned native |
| — | `sytrd_cta` / `syev_cta` Lower ([below](#known-defects-the-cta-sytrd-lower-path)) | the CTA SYTRD Lower path was reported wrong; both CTA paths run Upper and mirror Lower input | worked around; not reproduced |
| — | `linalg::qr` ([below](#linalgqr-returns-a-wrong-qr-after-an-earlier-call-in-the-process)) | the composed QR returns \f$QR \ne A\f$ after an earlier call in the same process | wrapper withheld; cause not located |

## Defect 1: `ortho`'s transposed arm builds a view that does not describe the memory

`src/extensions/ortho.cc:189-191`, inside the CGS lambda (was `:218-224` when filed):

```cpp
auto A_i = transA == Transpose::NoTrans
      ? MatrixView<T, fmt>(A.data_ptr(), m, i, m, A.stride(), batch_size)
      : MatrixView<T, fmt>(A.data_ptr(), i, m, m, A.stride(), batch_size);
auto C      = VectorView(Ymem.data(), i, batch_size);
auto A_next = A(Slice(), i);
```

Under `transA = Trans` or `ConjTrans`, `is_A_trans` is true and `inv_trans` is `NoTrans`
(`:122-123`). Three things then disagree:

* `A_i` is declared `i` rows by `m` columns with `ld = m`. The leading dimension of a view onto
  the first `i` rows of a column-major `A` is `A.ld()`, not `m`.
* the call at `:198` is `gemv(A_i, A_next, C, {.transA = inv_trans})`, i.e. `NoTrans` on this arm, so `x` must have length
  `A_i.cols() == m`.
* `A_next = A(Slice(), i)` is **column** `i`, of length `A.rows()`. On the transposed arm the
  vectors being orthogonalised are the *rows* of `A`, so `A.rows()` is the vector **count**.

The lengths coincide only when `A.rows() == m`.

**Why it is not live.** gemv's `can_run` (`src/ops/gemv/gemv.cc`; before phase 5 `gemv_op_shape`
in the deleted `src/backends/gemv_route.hh`) refuses every native family when
`X.size() != red_len` or `Y.size() != out_len`, which leaves only the vendor. The call therefore goes to cuBLAS/OpenBLAS exactly as it did before a native
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

`src/extra/cond.cc:48`, `:54` and `:131`:

```cpp
Event e = blas::dispatch::detail::syev_vendor_or_throw<B, T>(ctx, ...);
```

The buffer-size query and the call both reach past the public entry point into
`dispatch::detail` and name the vendor implementation. A vendor-free build has no vendor `syev`,
so this throws rather than falling to a native tier.

**Why it was left.** It is a routing-vocabulary defect owned by `syev`, not by any of the BLAS
work packages that found it; the fix is to call the public `syev` and let its selection
(`src/ops/syev/syev.cc`, flat selection since phase 5) choose. The shims now live in
`src/ops/syev/vendor.hh`.

**What fixing it needs.** Replace all three sites with the public `syev` / `syev_buffer_size`.
The workspace query has to move with the call — `syev_vendor_buffer_size_or_throw` throws in the
same build, so half a fix is no fix.

## Defect 3: `lanczos` issues a two-column multiply and consumes one column

`src/extensions/lanczos.cc:112-117` (was `:107-111` when filed):

```cpp
auto padded_vector = MatrixView(Vmem.data() + it*n, n, 2, n, (n+1)*n, batch_size);
...
spmm<B>(ctx, A, padded_vector, padded_output, ...);              // :115, sparse arm
gemm<B>(ctx, A, padded_vector, padded_output, GemmOptions<T>{}); // :117, dense arm
```

`padded_output` is likewise two columns wide (`:56`), and the kernel that consumes it reads one:
`local_v_next = Span(v_next_ptr + bid*2*n, n)` (`:132`) is column 0 only. Both arms — sparse and
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

`src/backends/netlib_lapack.cc:484`, `:496`, `:513`, `:525` — all four arms of the host solve:

```cpp
T x = alpha * Bb.at(i, j, 0) - sum;
```

There is no `alpha == 0` quick return anywhere in `trsm_vendor`. Reference `xTRSM` sets `B` to
zero without reading it in that case, and the reason matters here: callers hand these ops a
`BumpAllocator` allocation that is **not zeroed**, and `0 * NaN` is `NaN`, so an operand that
should have dropped out of the arithmetic poisons the result instead.

**Why it was left.** The identical defect in `spmm` (`netlib_lapack.cc:228,252` — `A` read at
`alpha == 0`, `C` read at `beta == 0`) was fixed by the work package that owns `spmm`; `trsm`'s
belongs to `trsm` and was out of that package's scope. The native `trsm` bodies already make the
guarantee. See [`../perf/spmm.md`](../perf/spmm.md) for the fixed sibling.

**What fixing it needs.** Skip the `alpha` term and substitute `T(0)`, matching what the native
bodies do — plus an `alpha == 0` case in `tests/trsm_tests.cc` whose `B` is poisoned with `NaN`
before the call. A test that passes a merely *wrong* `B` cannot see this; the poison has to be
something that survives multiplication by zero.

## 6. netlib `getri` ignores the leading dimension

`src/backends/netlib_lapack.cc:1250`:

```cpp
std::copy(Ab.data_ptr(), Ab.data_ptr() + n * n, Cb.data_ptr());
```

Both views are copied as `n*n` contiguous elements. Neither `Ab.ld()` nor `Cb.ld()` is consulted,
so any padded leading dimension gives a wrong answer (and, if `C` is the tighter of the two, a
write past its last column). Pre-existing, recorded in [`../perf/lu.md`](../perf/lu.md), not
fixed. The correct form is the per-column `std::copy_n` already used above at `:841`.

## 7. CLOSED — `trsm`'s heterogeneous-batch rejection *can* fire

**Closed 2026-09-15, by reading the file.** This entry claimed that `trsm_op_shape` never writes
`heterogeneous_batch`, so `route_trsm.hh:43`'s `if (s.heterogeneous_batch) return false;` could
never be reached with a true value. That was false in the tree of 2026-09-15.
`src/backends/trsm_route.hh:51` (line 40 after the 2026-09-30 comment pass; the file is deleted
since P3.3), inside `trsm_op_shape`, read:

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
sees one operand), and the default at `include/batchlas/blas/dispatch/route.hh:179` (deleted in
phase 5) is no longer
what a heterogeneous batch would leave behind. The original entry's *second* sentence still
stands and is NOT closed: **no test in the tree constructs a heterogeneous `trsm`**, so the gate
is argued, not armed. Per this page's own checklist item 1, a gate nobody has watched go red is
not a verified gate. Anyone picking this up should build a heterogeneous `A` or `B`, assert
`resolve_trsm_route` returns the vendor arm, and — the part that actually matters — assert it
under `vendor_available == false`, where the route walk has nowhere left to go.

**Since P3.3 (flat selection)** `route_trsm.hh` and `trsm_route.hh` are deleted. The heterogeneity
term now sits in the native families' `can_run` (`src/ops/trsm/trsm.cc`); the vendor's `can_run`
does not carry it, which is entry 12. The arming test is
`TrsmCandidates.HeterogeneousBatchHasNoNativeRoute` (`tests/trsm_candidates_tests.cc`): every native
pin must refuse a heterogeneous A and a heterogeneous B with "cannot run this shape", and Auto must
take the vendor, or throw `NoRouteError` vendor-free. Armed: dropping the term from the native
`can_run` turns exactly that test red for all four CUDA dtypes (cta, sg_left and blocked are then
"accepted") and nothing else. The vendor arm still lacks the term (entry 12).

## 8, 9. Two forced-route defects in the level-3 dispatchers

**CLOSED in the phase 5 rip (2026-10-05).** The `DiagFullGemm` route is deleted, so
`BATCHLAS_SYRK_ROUTE=native` and `BATCHLAS_SYR2K_ROUTE=native` take the tile kernel, which writes
only the named triangle (guard: `SyrkCudaCustomTest.AutoAndNativeRoutesLeaveTheOtherHalfUntouched`),
and a `cublasdx` pin that cannot run throws a message about the kernel it asked for. Since the
level-3 flat-selection wave both dispatchers are deleted and `cublasdx` is an unknown family. The
filing below is kept as it was; its line numbers describe the deleted code.

Both were pre-existing, both were preserved deliberately rather than quietly improved, and both
were reachable only through an environment pin.

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

The original filing said "<b>not caused by the small-n campaign (P0-P7)</b> — no `sytrd`, `latrd`,
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

## Defect 11: native gemm reads C at beta zero

**Status: located, worked around in `geqrf_blocked`, not fixed in `gemm`.** Found 2026-09-30
on threadripper02 (RTX PRO 6000 Blackwell, sm_120) by
`GeqrfTest.BlockedIgnoresAGarbageWorkspace` (`tests/geqrf_tests.cc`): a 64 x 64 blocked
`geqrf` whose caller workspace is filled with `0xff` bytes (every float a NaN) returns NaN for
all four scalar types when the trailing update runs through a native gemm choice (`sycl_gemm::gemm_custom` before P3.4, deleted since; `launch_direct` is behind the `direct` choice).

**The mechanism.** The blocked driver's `W1 = V^H A22` and `W2 = T^H W1` are `beta = 0` GEMMs
into scratch carved from the workspace. `LinearEpilogue::apply`
(`src/sycl/gemm/epilogue_linear.hh`) computes `alpha * acc + beta * prior` unconditionally,
and `launch_direct` (`src/sycl/gemm_kernels.cc`, the complex path here) reads
`c_ptr[...]` the same way, so `0 * NaN` poisons the output. Reference BLAS does not read `C`
at `beta == 0`. The arena behind `BumpAllocator` is not zeroed, so any op that feeds an
unwritten scratch region to a native `beta = 0` GEMM is exposed.

**The workaround.** `geqrf_blocked_dispatch` zero-fills W1 and W2 once per call
(`src/extensions/geqrf_blocked.cc`). The first panel writes the whole W extent, so later
panels read finite values.

**What fixing it needs** (owned by the `gemm` package): do not read `C` when `beta == 0` in
`LinearEpilogue` and in `launch_direct`. The resumed geqrf work package had a two-line patch
for both (the test then passes for all four types), and dropped it because `gemm_kernels.cc`
belongs to `gemm`. The epilogue branch needs a gemm timing A/B before it ships. When it lands,
delete the memset in `geqrf_blocked.cc`; the test stays as the guard.

## Defect 12: vendor potrf and trsm accept a heterogeneous batch

potrf's `can_run(Vendor)` is `d.has_vendor` with no heterogeneity term (the native families
carry `!A.is_heterogeneous()`), and `potrf_vendor` (`src/backends/cusolver.cc:72-77`) passes
`descrA.rows()`, the full storage order, to `cusolverDn?potrf[Batched]`. trsm's `can_run(Vendor)`
(`src/ops/trsm/trsm.cc`, P3.3; before it `route_trsm.hh:36`) is likewise `d.has_vendor` alone, and the
cuBLAS trsm path has no active-dims handling either. A heterogeneous call therefore factors and
solves the padded matrix with no error: the leading block of a Cholesky factor is still right, but
the backward `L^H` solve couples the active rows to the padding through `L21`.

Found during the P3.1 posv migration, whose first draft made `can_run(Blocked)` unconditional and so
turned posv's old `internal_error` on a heterogeneous batch into exactly this silent answer. posv now
refuses heterogeneous A or B before `choose()` (`throw_if_unservable` in `src/ops/posv/posv.cc`,
test `PosvCandidates.HeterogeneousBatchIsRefusedUnderEveryPin`). The potrf and trsm gaps themselves
are unfixed: the fix is a `!A.is_heterogeneous()` term on potrf's Vendor `can_run` (or a per-item
loop) and the same term on trsm's Vendor `can_run`, each a routing
change for its own phase. No test constructs a heterogeneous potrf or trsm.

**Level-3 four: CLOSED (level-3 flat-selection wave).** Every symm, syrk, syr2k and trmm vendor
loop (cuBLAS, rocBLAS, netlib) runs each item at the top-level extents from
`shape::validate_product` / `validate_rank_k` / `validate_rank_2k`, which never check
heterogeneity. Measured on `ff340fc6` for symm: with only A heterogeneous (active orders
16/14/12/10 in a 16 x 16 batch of 4) the cuBLAS loop answered off the active-order reference by 3.8,
and a batch heterogeneous in all three operands by 2.56; syrk and syr2k by reading
(`cublas.cc` `syrk_vendor_impl`, `syr2k_vendor_impl`). The first draft of the migration refused it
for trmm only, so symm's Auto went from main's throw (the old expansion's gemm rejected a
heterogeneous B or C inside the old expand window) to the loop's silent answer. Now every one of
the four refuses a heterogeneous operand in `can_run(Vendor)` as well as in its native families, so
Auto throws `runtime_error` (`NoRouteError` vendor-free) on every backend:
`{Symm,Syrk,Syr2k,Trmm}Candidates{,Cpu}.HeterogeneousBatchHasNoRoute`.
`SymmCandidates.HeterogeneousBatchHasNoRoute` also keeps the A-only wrong answer of the direct
expansion as the reason `expand` carries the term.

## 13. complex<double> cuBLAS calls with a unit dimension segfault inside cuBLASLt

Seen on threadripper02 (RTX PRO 6000 Blackwell, cuBLAS 13.4.1 from HPC SDK 26.5, 2026-10-04). Every
complex<double> `cublasGemmEx` / `cublasGemmStridedBatchedEx` with m or n == 1, and
`cublasZgemvStridedBatched`, segfaults on the host inside `cublasLtZZZMatmul` when called from a
BatchLAS process. The same calls from a standalone program (plain CUDA, or a SYCL queue's native
stream with SYCL USM, same libraries) do not crash, and neither LD_PRELOADing the netlib libraries
into it nor the cuBLASLt log (algo 13, workspace 0 in both) separated the two. Root cause unknown.

- Reached through trsm: `blocked` with one right-hand side makes every trailing update a
  complex<double> gemm with n == 1 (or m == 1 on Side::Right). The parent build crashes on
  `trsm cdouble L/R order 64-384 q 1 batch 128` under Auto; `getrf_tests`
  (`LuTest/7.BlockedFactorisesAndPivotsExactly`) and `ortho_tests`
  (`OrthoMatrixTest/7.OrthogonalizeMatrix`, through gemv) crash on the parent too.
- **gemm worked around (P3.3):** `gemm_vendor_impl` (`src/backends/cublas.cc`) calls the typed
  `cublasZgemmStridedBatched` for complex<double> when m or n is 1. Guard:
  `TrsmNativeBlocked.ComplexDoubleSingleRhsTrailingGemm`. getrf_tests passes with it.
- **gemv open:** `ortho_tests` still crashes in `gemv_vendor` for complex<double>, as on the parent.

## Defect 14: the Hermitian drivers read the unreferenced triangle

Located during the phase 5 gesvd and syev migrations; the drivers were not changed by either.
With large finite poison in the triangle the caller did not name:

* `gesvd_cta` with Upper, and `gesvd_blocked` with Lower at n <= 32 (8 and 32 tested; 48 is fine),
  return wrong singular values, all four dtypes.
* `syev_cta` with Upper (n = 5, 17, 32) reads the lower triangle; `syev_blocked` with Lower at
  n <= 32 reads the upper; `syev_two_stage` with Lower at n = 40 reads the upper.

Auto reaches some of these: gesvd routes Hermitian Upper n <= 32 to `cta`, and syev sends cfloat
n = 9..32 and cdouble n <= 32 (vectors or not) to `syev_cta`, so a caller that stores only the upper
triangle gets a wrong answer today. syev `two_stage` with Lower is the Auto choice for jobz=N above
n = 320 and float V 449..1024. `syev_blocked` with Lower at n <= 32 is reached only through a pin or
on a device without a 32-wide sub-group.

**Tests that reproduce it:** `GesvdCandidates.DISABLED_HermitianFamiliesIgnoreTheUnreferencedTriangle`
(`tests/gesvd_candidates_tests.cc`) and `SyevCandidates.OtherTriangleIsNeverRead`
(`tests/syev_candidates_tests.cc`), whose skip list names exactly these drivers; removing the skip
list turns it red for all four dtypes. gesvd's `can_run` also refuses blocked Hermitian Upper,
which the driver could run by mirroring, until this is fixed.

**What fixing it needs.** Mirror (or read only) the named triangle in each driver, then delete the
skip list and enable the gesvd test.

## 15. cuSOLVER `gesvdjBatched` faults on values-only non-square input

`BATCHLAS_GESVD_ROUTE=vendor`, float 8x4, jobs N/N, batch 3: `CUDA_ERROR_ILLEGAL_ADDRESS`.
Reproduced on the pre-phase-5 router (`424a45bc`) too. Auto never sends such a shape to the vendor
(a native family runs first), and gesvd's vendor `can_run` is `has_vendor` alone, as the old
`supports()` was; cuSOLVER's other refusals (`max(m, n) > 32`, non-packed batches, thin factors)
arrive as launch-time throws. The gesvd candidate tests' vendor envelope excludes the shape.

## 16. `ormqr` blocked throws at batch > 65535

`ormqr` float L T m=2 k=1 q=1 batch=65536 under Auto (`blocked`): "Number of work-groups exceed limit
for dimension 2". The same on `424a45bc`. `ormqr_blocked`'s sub-kernels put the batch on grid
dimension 2 and the driver does not check the limit, so `can_run` does not model it (R3) and Auto
does not fall to the vendor. Other ops that launch with the batch on dimension 2 may share the
ceiling ([`flat-kernel-selection.md`](flat-kernel-selection.md) §11; gemm's `can_run` carries it).

**What fixing it needs.** Fold the batch into dimension 0 (or loop over batch chunks) in the
sub-kernels, or, as a stopgap, a `batch <= 65535` term in the driver and `can_run`.

## 17. cuSPARSE spmm: shapes refused in `can_run`, and the alignment waiver

Measured on threadripper02 (cuSPARSE from HPC SDK 26.5 / CUDA 13.2) by calling
`backend::spmm_vendor` directly, so it is vendor behaviour:

* complex (cfloat, cdouble) with `transB == ConjTrans` and nrhs 1 returns an error status the vendor
  arm never checks and leaves C unwritten (a silent wrong answer). The old Auto routed some of these
  to the vendor; spmm's vendor `can_run` now refuses them on CUDA, so Auto runs `direct`.
* complex<double> N/N with one column segfaults on the host inside cuSPARSE (compare #13). Never on
  Auto's path; a `vendor` pin now falls back to Auto instead of crashing.
* **R3 waiver:** cuSPARSE silently mis-handles operands off their natural alignment. Alignment is a
  property of the pointers, not of the selection key, and modelling it would move routing, so
  `can_run` does not carry it; `spmm_tests` skips misaligned cases unless pinned native.

## Known defects: the CTA SYTRD Lower path

**Status: located, worked around on both CTA paths, not reproduced.** The `syev_cta` driver
(`src/extensions/syev_cta.cc`, the `uplo_eff` note in the real and complex drivers) carried this
note, verbatim: "The CTA SYTRD/SYEV pipeline currently exhibits severe correctness issues
specifically for the Uplo::Lower path. Until the lower-path kernel is fixed, we run the
(known-good) Uplo::Upper pipeline. To preserve the public API contract when callers only
initialize the lower triangle, we first explicitly symmetrize: A_upper := conj(A_lower)."

No test, commit or failure mode was recorded with it. The fused kernel (`syev_cta_fused.cc`) also
always runs the Upper reduction and symmetrises Lower input while loading its tile, so neither CTA
path reaches the Lower `sytrd_cta` kernel today. Whether that kernel is still wrong, and how, is
unverified.

**What would settle it.** Pin the Lower reduction (bypass the `uplo_eff` mirror) on graded input
with complex data and a poisoned upper triangle, compare against the Upper pipeline item by item,
and either fix the kernel and drop the mirror or record the failure mode here. Note that the
mirror itself interacts with [defect 14](#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle):
`syev_cta` with Upper reads the lower triangle.

## linalg::qr returns a wrong QR after an earlier call in the process

**Status: observed, cause not located; the wrapper is withheld.** `linalg::qr` is deliberately
absent from `include/batchlas/blas/linalg-ops.hh`. The composition `geqrf` +
`triangular_mask_into` + `orgqr` returned \f$QR \ne A\f$ once an earlier `linalg::qr` test had
run in the same process, and passed when run alone. The original header note read "repro:
tests/linalg_layer_tests.cc, 4x". docs/developer/agent-guide.md section 9 describes it as "an unexplained cross-Queue
wrong-answer defect".

Related: the `linalg::` value-returning wrappers free their scratch while kernels may still be
enqueued, and the workspace arena belongs to the `Queue` (see
@ref design_runtime_internals), so state that outlives one call is the first suspect.

**What fixing it needs.** Locate the cross-Queue state (arena reuse, a static, or a stale event)
with the two-call repro, then add the wrapper with that repro as its guard.

## One filed claim that did not survive re-checking

[`../perf/lu.md`](../perf/lu.md) records "a latent vendor gate defect: `cublas.cc`'s `getrs` sits
in a TU gated on `BATCHLAS_HAS_CUBLAS`, so a cuBLAS-present / cuSOLVER-absent configure claims a
vendor it cannot link." Re-checked against the tree: `getrs_vendor` for CUDA calls
`cublas?getrsBatched` (`src/backends/cublas.cc:1491`) and nothing from cuSOLVER; `cublas.cc` is
added to `BACKEND_CUDA_SOURCES` under `BATCHLAS_HAS_CUBLAS` (`src/backends/CMakeLists.txt:69`);
and `factorization_vendor_available<Backend::CUDA>` is `BATCHLAS_HAS_CUBLAS`
(then `include/batchlas/blas/dispatch/vendor_available.hh:42`). Gate and definition agree. Since
phase 5 the constant is `src/select/vendor.hh:28` and requires cuSOLVER as well, so the stated
configuration now claims no factorization vendor at all. Marked
`unverified` rather than deleted: the stated mismatch could not be reproduced, but the entry may
be describing an earlier `getrs` that did call cuSOLVER.

## Known defects: unverified candidates from the documentation pass

Observations made while migrating the design notes into `docs/` (2026-09-30 and 2026-10-06).
Each is located to a line but **not confirmed by a test**; none is numbered above until it is.
Whoever confirms or refutes one moves it into the table or into the section above.

- **`gesvdj_cta`'s global rescale ignores columns 32..63 on the C=64 rung** (reported by the gesvd
  migration). The `nmax`/`nmin` reductions read `Nrm_local[base_n + lane]` only for `lane < CC`
  (`src/extensions/gesvdj_cta.cc:354`), so on the 64-column rung the upper half of the columns does
  not influence `beta`. Correctness is unaffected (`beta` is a power of two, and the scaling is
  undone exactly), but the overflow/underflow headroom for graded input with 33 to 64 columns is
  narrower than the design claims. Design record:
  [global power-of-two scaling](gesvd.md#gesvdj_cta-global-power-of-two-scaling). What would
  settle it: graded 33..64-column input whose largest column norm sits in columns 32..63, near the
  overflow threshold, compared against the n <= 32 behaviour.
- **The recursive `stedc` driver may merge from unset leaf eigenvectors under `NoEigenVectors`**
  (reported by the tridiagonal migration). `stedc_impl` forwards the caller's `jobz` to the leaf
  `steqr_dispatch` (`src/extensions/stedc.cc:535`), while the merges always consume the leaf
  eigenvectors; the level-synchronous driver ignores `jobz` (`:640`). A direct
  `stedc(..., JobType::NoEigenVectors, ...)` with `StedcAlgorithm::Recursive` could therefore merge
  from vectors that were never written. `syev` does not reach this, as far as the reporter could
  see. What would settle it: that direct call, compared against the eigenvalues-only reference,
  with the eigenvector buffer poisoned beforehand. The trap is also recorded on the stedc page,
  [stedc: eigenvalues-only still builds eigenvectors](../perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors),
  and at the recursive leaf in `src/extensions/stedc.cc`.
- **`compute_optimal_wg_size`'s power-of-two rounding assumes a 32-bit `long`** (reported by the
  dispatch/device header pass). For REDUCTION and SCAN it rounds with
  `size_t(1) << (31 - __builtin_clzl(base_wg_size))` (`include/batchlas/util/kernel-heuristics.hh:103`,
  `:108`). On LP64 `__builtin_clzl` counts leading zeros of a 64-bit value, so for a base of 256 the
  shift count is 31 - 55 = -24: undefined behaviour (x86 masks it to 40, giving \f$2^{40}\f$, which
  the later `problem_size` and `MAX_WORK_GROUP_SIZE` clamps cut down). The result is
  min(problem_size, device max), not a power of two. Callers: `src/matrix.cc:387` (REDUCTION, `rows`)
  and `:487` (SCAN, `rows + 1`). Not checked: whether those kernels assume a power-of-two
  work-group. Fix: `63 - __builtin_clzl` or `std::bit_floor`. The same header's doc comments record
  unused parameters (`batch_size` and `memory_per_problem` of `compute_optimal_wg_size`,
  `max_wg_size_for_kernel` of `compute_batched_nd_range_sizes`).
- **The generic (MKL-instantiated) `trmm` recursion reads the whole square of `A`** (reported by
  the level-3 comment pass). Its base case at n <= 256 is a plain `gemm` on the diagonal block
  with `beta = 1`, so it reads the unreferenced triangle and the stored diagonal under `Diag::Unit`,
  and accumulates into `C` instead of overwriting it. Full note and what would settle it:
  [the trmm generic recursion](../perf/level3.md#trmm-the-generic-recursion-reads-the-whole-square-of-a)
  (`src/extensions/trmm.cc:22-26`).
- **Matrix and vector container defects** (reported by the matrix header pass). Five located,
  unfixed defects in `matrix.hh` / `src/matrix.cc` are listed under
  [Matrix model: open debts](matrix-model.md#matrix-model-open-debts): an undefined
  `MatrixView::transpose`, packed-only addressing and identical items in `fill_triangular_random` /
  `fill_tridiag_toeplitz`, `fill_random` writing padding, a no-op slice assert in
  `KernelMatrixView`, and a length assert in `fill_diagonal(ctx, Span, k)` that can fire for
  `k != 0`.
- **Three environment-parser defects** (reported by the core header pass), recorded under
  [environment parser defects](environment.md#environment-parser-defects-recorded-not-fixed):
  `BATCHLAS_SYEVX_SOFT_LOCK` is inverted (`=off` and an empty export read as on),
  `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` reads an unparseable value as 0 and widens the blocked-gebrd
  path to every n, and `BATCHLAS_TUNE_STEDC_MERGE_VARIANT` is cast to the enum with no range check.
- **Device group BLAS: the 3-D tile-group race in `gemm`, `symm` and `trmm`** (reported by the
  device header pass; read from the dispatch code, not reproduced). With a `sycl::nd_item<3>`
  executor, group dimensions 1 and 2 index output tiles, but only the tiled kernels read them.
  `gemm`, `syrk` and `syr2k` guard their generic fallback to tile-group (0, 0), yet `gemm`'s
  non-register sub-group path runs before that guard and covers the whole output per work-group,
  and `symm` / `trmm` have no guard at all. Every tile-group then writes all of `C`: benign at
  `beta == 0`, wrong at `beta != 0` or for an in-place `trmm`. Full write-up and what would settle
  it: [the 3-D launch generic fallback](device-group-blas.md#device-group-blas-the-3-d-launch-generic-fallback).
- **`francis_sweep` is declared and never defined** (reported by the eigen header pass).
  `include/batchlas/blas/extensions.hh` declares it `BATCHLAS_API`, but nothing in `src/` defines
  or instantiates it, so a call compiles and fails at link time. The declaration carries a
  `@warning`; the fix is to delete it or implement it.
- **`tridiagonal_solver` assumes `Q.ld() == n`** (same pass). Its rotation update addresses `Q`
  with stride `n` (`Q[k*m+l]`, `src/extensions/tridiag_solver.cc:29`) while the identity fill uses
  `Q.ld()`, so a padded `Q` gets a wrong answer; it also caps QR steps at six per eigenvalue with no
  convergence report. Nothing in `src/` calls it; the API doc states `@pre Q.ld() == n`.
- **`UnifiedVector`'s move assignment leaks the destination's allocation** (reported by the
  matrix header pass). `operator=(UnifiedVector&&)` (`include/batchlas/util/sycl-vector.hh`)
  overwrites `data_` without freeing the previous allocation, so moving into a non-empty
  `UnifiedVector`, `Matrix` or `Vector` leaks USM shared memory. The header carries a `@trap`;
  the record is under [Matrix model: open debts](matrix-model.md#matrix-model-open-debts).
- **Kernel selection throws outside the error hierarchy** (reported by the error-model pass).
  `src/select/` throws plain `std::invalid_argument` (an unparseable or unservable pin) and
  `std::runtime_error` (no runnable kernel, a bad tuned table), and `batchlas::NoRouteError`
  derives from `std::runtime_error` only, so `catch (const batchlas::exception&)` misses all of
  them. Separately, `gesv` and `posv` throw `batchlas::internal_error` for an empty or
  heterogeneous batch (`throw_if_unservable` in `src/ops/gesv/gesv.cc` and
  `src/ops/posv/posv.cc`), which by meaning is `invalid_argument` or `unsupported`. Reclassifying
  either is a behaviour change for callers that catch the current type. Full table:
  [kernel selection throws outside the hierarchy](error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy).
- **`internal/sytrd_blocked.hh` declares a second, undefined `sytrd_blocked` template**
  (reported by the factorization header pass, confirmed by the eigen review). It takes
  `Span<std::byte> ws` by value with no default `block_size`
  (`include/batchlas/internal/sytrd_blocked.hh:46-53`), while `blas/extensions.hh:1059-1066`
  declares, and `src/extensions/sytrd_blocked.cc:915` defines, the `const Span<std::byte>&`
  overload with a default. These are two distinct function templates, not a redeclaration, so a
  translation unit that sees only the internal header and calls it fails at link time. Fix: make
  the internal declaration match (or drop it). Related header hygiene from the same passes, not
  defects: `sytrd_band_reduction_single_step` and its `_buffer_size` are declared twice each in
  `extensions.hh` with identical signatures, and `OrmqCtaFactorization` (`extensions.hh:989`) is
  referenced by no entry point, test or source file.

### Known defects: fixed while documenting

- **`miniacc --help` used to terminate its host.** `miniacc::ParseCommandLine`'s `--help` arm
  called `std::exit(0)`, from a header that is installed (`cmake/BatchLASPackaging.cmake` installs
  `include/batchlas` except `minibench*.hh` and `bench_structured.hh`). `exit()` runs no
  destructors for live automatic objects in the caller's frames, so a consumer that parses an
  argv containing `--help`, or embeds the harness, could not stop it. It now sets
  `CliOptions::help_requested` and `MiniAccMain` returns 0
  (`include/batchlas/util/miniacc.hh`).
- **Superseded, not applied:** a proposed ledger item that `trsm_op_shape` never set `s.backend`
  (so every trsm coverage row read `Backend::AUTO`) described `src/backends/trsm_route.hh`, which
  P3.3 deleted. Coverage rows now come from `src/select/coverage.hh`.

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

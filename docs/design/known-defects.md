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

The superseded root documents these were filed in are preserved at the git tag
`perf-evidence/vendor-independence` (`git show perf-evidence/vendor-independence:WP7_FILED_DEFECTS.md`).

## At a glance

| # | site | what is wrong | severity today |
|---|---|---|---|
| 1 | `src/extensions/ortho.cc:218-224` | the transposed arm builds a view whose extents and `ld` do not describe the memory, against a vector of the wrong length | latent — a shape check routes it to the vendor |
| 2 | `src/extra/cond.cc:46,52,127` | reaches into `dispatch::detail` and demands the **vendor** `syev` instead of resolving a route | throws in a vendor-free build |
| 3 | `src/extensions/lanczos.cc:107-112` | the level-3 call carries two right-hand-side columns and one is consumed | 2x work, right answer |
| 4 | `src/backends/rocsparse.cc:30-31,62-63` | `ConjTrans` maps to the conjugating enum for **real** scalars | inferred wrong answers on AMD; unobservable here |
| 5 | `src/backends/netlib_lapack.cc:508,520,537,549` | `trsm` reads `B` when `alpha == 0` | `NaN` from unwritten workspace |
| 6 | `src/backends/netlib_lapack.cc:1389` | `getri` copies `n*n` contiguous elements and ignores both `ld`s | wrong answer at padded `ld` |
| 7 | `src/backends/trsm_route.hh:40-56` | the heterogeneous-batch rejection has no writer, so the gate cannot fire | a stated safety property that is not enforced |
| 8 | `src/backends/syrk_custom_dispatch.cc:261-262` | a forced native `syrk` lands on a route that writes both triangles | wrong answer, forced routes only |
| 9 | `src/backends/syr2k_custom_dispatch.cc:210` | a forced native `syr2k` throws a cuBLASDx message it did not ask for | misleading diagnostic |

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

**Why it is not live.** `gemv_op_shape` (`src/backends/gemv_route.hh:75-76`) returns
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

`src/extensions/lanczos.cc:107-112`:

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
type-conditional `cusparse_op<T>` helper at `src/backends/cusparse.cc:29-47`; the complex arms,
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

## 7. `trsm`'s heterogeneous-batch rejection can never fire

`route_trsm.hh:43` rejects a heterogeneous batch (`if (s.heterogeneous_batch) return false;`) —
correctly, since one `trsm` launch covers the whole batch with a single `(order, q, ld, stride)`
tuple. But `trsm_op_shape` (`src/backends/trsm_route.hh:40-56`) never writes the field, so it
keeps `OpShape`'s default of `false` (`include/batchlas/blas/dispatch/route.hh:162`).
`MatrixView::is_heterogeneous()` exists and `getrf`'s builder calls it
(`src/backends/getrf_route.hh:45`); `trsm`'s does not.

The gate is a documented intention, not an enforced one. No measurement either way, and no test
constructs a heterogeneous `trsm`.

## 8, 9. Two forced-route defects in the level-3 dispatchers

Both are pre-existing, both were preserved deliberately rather than quietly improved, and both
are reachable only through an environment pin.

* **`BATCHLAS_SYRK_ROUTE=native` returns a wrong answer.** `{Native, Auto}` passes
  `syrk_use_cuda_custom`, then matches no arm inside `syrk_cuda_custom` (the gram arm needs
  `origin == Auto`, the tile arm needs `algo == TriangularTiles || origin == Auto`) and falls
  through to `syrk_cublasdx_fallback_gemm` at `src/backends/syrk_custom_dispatch.cc:261-262` —
  the `DiagFullGemm` route, which **writes both triangles**, clobbering the one the caller did
  not name. **No test in the tree sets `BATCHLAS_SYRK_ROUTE`.**
* **`BATCHLAS_SYR2K_ROUTE=native` throws a cuBLASDx message it did not ask for.** The throw at
  `src/backends/syr2k_custom_dispatch.cc:210` is not guarded by `forced`, so a non-fused named
  route reaching it gets a diagnostic about a fused kernel it never requested.

Full context in [`../perf/level3.md`](../perf/level3.md) and
[`../perf/dispatch.md`](../perf/dispatch.md).

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
| all 232 `gemv` cases, on batch stride | every case used the **natural** stride (`a_stride == ld*n`, `x_stride == size*inc`), so a kernel deriving each stride instead of reading it passed the whole suite — while `ortho.cc:218-222` hands it an `A.stride()` its `ld*cols` does not equal, every CGS iteration | break `padstride`: exactly 32 cases red, all four of them new | four `stride_pad` cases, one per kernel body |
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

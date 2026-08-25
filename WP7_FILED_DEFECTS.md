# WP7 — defects found and DELIBERATELY NOT FIXED

Three pre-existing defects were located during WP7 and its repair pass. All three
are **out of scope by the lead's decisions D4, D5 and D6**, and all three are
recorded here as named, located defects rather than as prose in a report, so that
the next work package can pick them up without re-deriving them.

None of them was introduced by WP7, and **none of them changed behaviour under
WP7**: each keeps taking exactly the path it took before the native gemv landed.

---

## 1. `src/extensions/ortho.cc:216-224` — the `transA = Trans` branch builds a view whose extents do not match the vector it is handed

**Status: STRUCTURALLY WRONG TODAY. Not fixed, not thrown on. (D6)**

```cpp
auto A_i = transA == Transpose::NoTrans
      ? MatrixView<T, fmt>(A.data_ptr(), m, i, m, A.stride(), batch_size)
      : MatrixView<T, fmt>(A.data_ptr(), i, m, m, A.stride(), batch_size);
...
auto A_next = A(Slice(), i);
...
gemv<B>(ctx, A_i, A_next, C, {.transA = inv_trans});
```

Under `transA = Trans` (`is_A_trans` true, so `inv_trans = NoTrans`, ortho.cc:118-120):

* `A_i` is declared **`i` rows × `m` columns** with **`ld = m`**. Both are wrong for
  a view onto the first `i` rows of a column-major `A`: the leading dimension of
  such a view is `A.ld()`, not `m`.
* the call is `gemv(A_i, A_next, C, transA = NoTrans)`, so `x` must have length
  `A_i.cols() == m`.
* but `A_next = A(Slice(), i)` is **column `i` of `A`, of length `A.rows()`**,
  and under `transA = Trans` the vectors are the ROWS of `A`, so `A.rows()` is
  the vector COUNT, not the vector length.

The lengths therefore agree only in the accidental case `A.rows() == m`.

**Why WP7 did not fix it and did not throw on it.** D6 forbids a new host-level
validation throw: the native kernel must accept exactly what the vendor accepts,
and turning today's silent misbehaviour into a crash in a live path would make WP7
unattributable for a defect it did not create. What happens instead is that
`gemv_op_shape` (src/backends/gemv_route.hh:73-76) checks
`X.size() != red_len` and returns `std::nullopt`, which resolves to
`{Vendor, Auto}` — so this call keeps going to cuBLAS/OpenBLAS **exactly as it did
before WP7**, and the native kernel never sees it.

**What fixing it needs.** The right `A_i` for the transposed arm, a `A_next` that
is the `i`-th VECTOR rather than the `i`-th column, and a test on
`ortho(..., Transpose::Trans)` that actually checks orthogonality of the rows —
`ortho_tests` does not have one today, which is why this has survived.

---

## 2. `src/extra/cond.cc:52` — `syev_vendor_or_throw` bypasses the route table

**Status: out of scope by D4.**

```cpp
Event e = blas::dispatch::detail::syev_vendor_or_throw<B, T>(ctx, ...);
```

`cond` reaches into `dispatch::detail` and demands the VENDOR `syev` rather than
resolving a route. In a vendor-free build that throws, which is 6 of `cond_tests`'
30 vendor-free failures (the other 20 are `NETLIB`/`getri`, i.e. WP9). It is a
routing-vocabulary defect, not a gemv one: the fix is to call the routed `syev`
and let `resolve_route` decide, which is a `syev` work item.

---

## 3. `src/extensions/lanczos.cc:112` — a `gemm` whose second output column is discarded

**Status: out of scope by D5.**

```cpp
auto padded_vector = MatrixView(Vmem.data() + it*n, n, 2, n, (n+1)*n, batch_size);
...
gemm<B>(ctx, A, padded_vector, padded_output, GemmOptions<T>{});
```

The multiply is issued with **two** right-hand-side columns and only the first is
consumed. That is a 2× overcount of the level-3 work in the Lanczos inner loop.
It is a `gemm` call-site defect and belongs to whoever owns lanczos; WP7 touched
neither.

Note for whoever picks it up: `lanczos_tests` fails **identically in both builds**
and its coverage dump contains only `linked,gemv,...` rows and **zero `reached`
rows** — it never calls `gemv` at all — so that failure is unrelated to WP7 and
must not be attributed to it.

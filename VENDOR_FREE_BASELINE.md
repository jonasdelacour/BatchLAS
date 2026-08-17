# The vendor-free burn-down baseline

Recorded at WP0 S6, on `worktree-vendor-independence-plan`.

```
cmake -S . -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF
cmake --build build-novendor -j
ctest --test-dir build-novendor -LE slow
```

**This configuration now configures, compiles, links, loads and runs.** That is
WP0's deliverable. It is emphatically **not** green, and no dispatch mechanism
could make it so: the gap is missing *kernels*, not missing *routing*. Closing it
is WP1–WP8.

## Result at S6

`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` on this machine yields
`BATCHLAS_HAS_CUDA_BACKEND 1` with every CUDA math library at 0 — a CUDA device
with no CUDA math libraries, a state the pre-WP0 scheme could not express and
could not link.

    ctest -LE slow: 20 / 53 pass (38%)

Failures are `NoRouteError`, not crashes and not link errors. The message names
the op, the scalar type and the switch that would restore it:

    BatchLAS: no route for getri<float> on this backend (built without cuBLAS).
      This build has no vendor library for that op, and BatchLAS has no
      native kernel for it yet. If you configured with
      -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF, re-enabling it restores this op;
      otherwise the vendor library was not found at configure time.

## The 33 failing suites

Any change to this SET is a reviewable diff: a suite leaving it means a native
kernel now covers that op, and a suite joining it is a regression.

```
backend_dispatch_tests   bdsdc_tests            cond_tests
gemm_tests               gemv_tests             hemm_tests
her2k_tests              herk_tests             iluk_tests
inverse_tests            lanczos_tests          linalg_layer_tests
options_api_tests        orgqr_tests            ormqr_blocked_tests
ormqr_cta_tests          ormqr_tests            ortho_tests
ritz_values_tests        syev_blocked_tests     syev_cta_tests
syev_tests               syev_two_stage_tests   syevx_tests
symm_tests               syr2k_tests            syrk_tests
sytrd_blocked_tests      sytrd_cta_tests        sytrd_sy2sb_tests
transpose_tests          trmm_tests             trsm_tests
```

Two of those — `lanczos_tests` and part of `steqr_tests` — fail in the
vendor-PRESENT build too, for reasons unrelated to this work; see
`VENDOR_INDEPENDENCE_PLAN.md`.

## After WP1

The numbers below are the **pre-WP1** baseline and are kept as the historical record. WP1
moved them: the suite is now **24/53**, `gemm_tests` is 167 passing / 17 failing (was
48/136), and the `symm`/`syrk`/`syr2k`/`trmm` tile kernels are reached rather than merely
linked. Four suites recovered — `bdsdc_tests`, `ritz_values_tests`, `sytrd_cta_tests`,
`transpose_tests` — with none newly failing. See `WP1_LEVEL3_SPEC.md`.

## After WP2's correctness track

`gemm_tests` is now **184/184 vendor-free** (was 48/184 pre-WP1, 167/184 after WP1 S5), and
the suite is **25/53**. `gemm_tests` left the failing set and nothing joined it.

The gap that closed was heterogeneous batch, and it was never a kernel gap: the per-item loop
— including the `m==0`/`n==0` skips and the `k==0 → scale(beta)` substitution — lived inside
cuBLAS-gated code, so a vendor-free build simply did not have those semantics. See
`WP2_GEMM_SPEC.md`.

One named gap remains in this area:

- **Level-3 non-float.** `syrk`'s gram branch and `trmm`'s tile branch for double/complex
  are still reachable only from `cublas.cc`, and `syr2k` has no non-float tile route at all.

And one that WP2's envelope track owns: `select_kernel_variant` is **float-only**, so a
vendor-free `double` or `complex` GEMM reaches `Tiled16`, not a register kernel.

## The same gap, per op

`cmake --build build-novendor --target batchlas_coverage` writes `coverage.csv`, whose
`linked` rows answer this exactly and without running anything:

| native kernel linked | ops |
|---|---|
| yes | `gemm`, `ormqr`, `syev`, `gesvd` |
| **no** | `gemv`, `trsm`, `trmm`, `symm`, `syrk`, `syr2k`, `hemm`, `herk`, `her2k`, `geqrf`, `orgqr`, `getrf`, `getrs`, `getri`, `potrf`, `spmm` |

`miss` rows add what a run actually reached, e.g.

    miss,getri,float,CUDA,,,,,,,,1,0,0,cuBLAS

## Why so many, and what closes them

The count is dominated by transitive dependence rather than by breadth of
missing ops. `ortho` alone needs `potrf`, `trsm`, `geqrf`, `orgqr` and `gemv`,
and `ortho` sits under `syevx`, `lobpcg` and `lanczos` — so one missing native
`trsm` fails a dozen suites.

`gemm` is the interesting entry in the list, and the first version of this
paragraph got it **wrong** in a way worth keeping visible.

It claimed the native register-tiled kernel "exists and is vendor-free, so
`gemm_tests` fails only on the shapes outside `gemm_custom_problem_supported`".
The kernel does exist and is vendor-free. It is also **unreachable**: the facade
(`src/dispatch/entry_points/level3.cc:60-64`) is

```cpp
if constexpr (!dispatch::level3_vendor_available<Back>) { throw_no_vendor_route(...); }
else { return backend::gemm_vendor<Back, T>(...); }
```

and the three-way route that would pick the native kernel lives in
`backend::gemm_vendor` at `cublas.cc:153-178` — inside the cuBLAS-gated TU. So a
vendor-free `gemm` throws on *every* call. Measured, not argued:
`build-novendor/tests/gemm_tests` is 48 passed / 136 failed, and every one of the
48 is a pure route-resolution test (`GemmDispatchPolicyTest.*`,
`GemmTest/*.RouteAdapter*`, all 0 ms) that never executes a GEMM. The failures
are `NoRouteError: no route for gemm<float> on this backend`.

**`linked` is not `reachable`, and the table above reports `linked`.** That is
the distinction the whole burn-down turns on, so read the `yes` column as "the
kernel is in the build", never as "the op works". Closing the gap for `gemm` is
WP1 S5; until then `gemm`'s `yes` is a statement about the linker.

The four level-3 tile routes (`symm`/`syrk`/`syr2k`/`trmm`) are portable SYCL
already, but their dispatch still terminates in `*_vendor_cuda_raw`, so they are
compiled only with cuBLAS. WP1 frees them — but see `WP1_LEVEL3_SPEC.md`, which
corrects three claims made here and in the plan:

- `symm` carries **no tile kernel**; its only portable kernel is the mirrored
  expansion, and everything after it is a GEMM.
- The four routes are reachable **only from `cublas.cc`** (`:269`, `:763`,
  `:882`, `:1005`). Compiling the TUs everywhere makes them linked everywhere
  and called nowhere.
- Routing the terminal at the public `gemm` is the right destination but is not
  sufficient, per the `gemm` correction above: it converts a `NoRouteError`
  naming `symm` into one naming `gemm`.

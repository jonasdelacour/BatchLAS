# Device-side group BLAS {#design_device_group_blas}

> **Status:** current · checked 2026-10-06

Function templates in `namespace batchlas::device` (`#include <batchlas/blas/device.hh>`) that run
**inside** a SYCL kernel. A work-item group computes **one** operation on **one** matrix or vector.
There is no queue, event or batching here: the kernel picks the batch item (`view.batch_item(b)`) and
synchronises. The API reference is the @ref api_device_blas group.

## Device group BLAS: what is provided

- Level 1: `fill`, `copy`, `copyc`, `scal`, `axpy`, `hadamard` (n-ary functor), `dotu`, `dotc`.
- Level 2: `gemv`, `ger` / `geru` / `gerc`, `trmv`, `symv`, `hemv`.
- Level 3: `gemm`, `trmm`, `symm`, `syrk`, `herk`, `syr2k`, `her2k`.

Not provided: `hemm`, triangular solves, banded and packed formats. Two differences from reference BLAS:

* `trmv` is out of place: \f$ y := \alpha\,\mathrm{op}(A)\,x + \beta\,y \f$. `x` and `y` may alias.
* `trmm` takes `beta` and a separate output, and `B` and `C` may alias. Aliasing disables the fast paths.

## Device group BLAS: calling conventions

```cpp
template <[DeviceBlasPolicy Policy,] <compile-time mode params>, typename Group, typename T>
void op(const Group& exec, <operands>..., T alpha = 1, T beta = 0, T* workspace = nullptr);
```

- `Transpose`, `Uplo`, `Side` and `Diag` are template parameters, so each mode is its own instantiation.
  Without a `DeviceBlasPolicy` the policy is `Auto`. The operand-struct form (`make_*_operand`) calls the same function.
- **Collective:** every work-item of the executor makes the call with the same arguments, from uniform
  control flow. A work-item that skips it deadlocks or corrupts the reduction.
- **One logical problem:** each view has `batch_size() == 1`. Shape checks are `assert`s that vanish under
  `NDEBUG`, so a wrong shape computes garbage in release builds.
- **`beta == 0` still reads the output.** A NaN already in `C` survives. This differs from reference BLAS.
- **Writes are not published.** Call `sycl::group_barrier` before another work-item reads the output.
- `dotu` and `dotc` return the group-wide sum to every work-item.

## Device group BLAS: sub-group versus work-group

| executor passed | collective scope | eligible paths |
| --- | --- | --- |
| `sycl::sub_group` | the sub-group | generic only |
| `sycl::group<D>` | the work-group | generic, plus the tiled `gemv` / `symv` / `hemv` |
| `sycl::nd_item<1>` / `nd_item<3>` | its work-group | everything, including the level-3 sub-group fast paths |

The level-3 fast paths are chosen per call, and only when all of these hold:

* the scalar is `float`; the complex rank-k and rank-2k kernels take `std::complex<float>`;
* the policy admits the sub-group size (`Auto`: 16 or 32; `Subgroup16` / `Subgroup32`: exactly that; `Generic`: never);
* the work-group has the size the kernel was written for (256 for the register-tiled and aligned GEMM),
  and the extents clear the minimum tile;
* a workspace pointer was passed, for any path that stages through local memory.

## Device group BLAS: the workspace protocol

1. Describe the launch with `make_group_launch_info`, `make_nd_item_1d_launch_info` or
   `make_nd_item_3d_launch_info`. It must match the real launch, since the size query uses the same predicates.
2. Call `*_workspace_elements<T, ...>(launch, extents...)`. It returns a count of **elements of `T`**, and 0
   when no staged path applies.
3. Allocate a `sycl::local_accessor<T, 1>` of that size and pass `batchlas::util::get_raw_ptr(acc)`, or
   `nullptr` when the count is 0.

A non-null pointer asserts the buffer is large enough; the call cannot check. `nullptr` is always safe and
only forgoes the staged paths.

## Device group BLAS: the 3-D launch generic fallback

In an `nd_item<3>` launch, work-group id dimensions 1 and 2 index output tiles (128 x 64 for the
register-tiled paths). Only the tiled kernels read them.

When no tiled path applies, `gemm`, `syrk`/`herk` and `syr2k`/`her2k` run the generic path in
**tile-group (0, 0) only**. Otherwise each group would scale `C` by `beta` again. The result is correct
but slow (guard: `group_blas_gemm.hh`, `group_blas_rankk.hh`).

The guard is narrower than the hazard (read from the code on 2026-10-06, not tested):

* `gemm`'s non-register sub-group path runs before the guard, and every tile-group runs it over the whole output.
* `symm` and `trmm` have no guard at all.

With `beta == 0` the race is benign. With `beta != 0`, or an in-place `trmm`, the output is wrong. Until
this is fixed, use a 3-D launch only with a shape and workspace that admit the register-tiled path, or
with one tile-group.

## Device group BLAS: register and local-memory caveats

- **Local-memory budget:** level-3 workspaces are sized against `device_limits::subgroup_workspace_budget_bytes()`.
  A type that does not fit gets a constantly-false fast-path predicate. See
  [the subgroup workspace budget](../perf/gemm.md#the-subgroup-workspace-budget).
- **Registers:** the register-tiled paths keep a 4 x 8 accumulator per work-item (complex rank-2k: 4 x 4).
  That is why the fast paths take 32-bit scalars. A `double` tile of that shape at 256 work-items exceeds the
  register file, and NVIDIA rejects the launch. Widening the types means shrinking the thread tile first.
- **Barriers:** the tiled kernels call `group_barrier`; every work-item must stay live until the call returns,
  including those with an empty output tile.
- **Alignment:** the aligned GEMM path (NN, `float`) needs `data`, `ld` and the batch stride of `A` and `B`
  to be multiples of 4, and `aligned_a` / `aligned_b` set in the query. Its column gate is 64
  (`kRegisterMatrixTileN`), stricter than its 32-wide tile (`kOptimizedGemmTileN`). Seen in the code, not measured.

## Device group BLAS: traps for callers

- `trmm_workspace_elements` without a policy takes `T` after the four mode parameters. Every other
  `*_workspace_elements` takes `T` first.
- Pass `aliased = true` to `trmm_workspace_elements` when `B` and `C` overlap. It returns 0, since the fast paths are skipped.
- `symv` and `hemv` call `group_barrier` on the tiled path, so pass a `sycl::group` or `sycl::sub_group`, not an `nd_item`.
- `herk` and `her2k` take complex `alpha` and `beta`, and force the diagonal imaginary parts of `C` to zero.
  A non-real `alpha` in `herk` still gives a non-Hermitian off-diagonal.

## Device group BLAS: where it is used

- Library: `src/extensions/latrd_lower_panel.cc`, `larft_wy.hh`, `ormqr_cta.cc`, `ormqr_blocked.cc`,
  `sytrd_blocked.cc`, `src/math-helpers.hh`.
- Tests: `tests/device_blas_tests.cc`. Benchmarks: `benchmarks/device_blas_level{2,3}_benchmark.cc`.

The device group BLAS is outside the host-side selection layer ([flat kernel selection](flat-kernel-selection.md)).
An in-kernel call is never looked up in a table or pinned by `BATCHLAS_<OP>_ROUTE`. Its choices are fixed in the header.

## Device group BLAS: open debts

- No evidence page measures these kernels. The fast-path gates (256 work-items, minimum extents, `float` only) are design constants.
- No triangular solve and no `hemm`.
- The 3-D tile-group race in `gemm`, `symm` and `trmm` ([above](#device-group-blas-the-3-d-launch-generic-fallback)).
  A test that settles it: per op, an `nd_item<3>` launch of at least 2 x 2 tile-groups, `beta != 0`, no
  workspace, compared against the host reference.
- The tiled `symv` / `hemv` path cannot take an `nd_item`.

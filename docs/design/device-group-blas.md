# Device-side group BLAS {#design_device_group_blas}

> **Covers:** the in-kernel BLAS templates in `namespace batchlas::device`
> (`#include <batchlas/blas/device.hh>`): what they are, how a kernel calls them,
> how the executor type decides the collective scope, the local-memory workspace
> protocol, and the register and local-memory caveats of the sub-group fast paths.
> **Status:** current. Written 2026-09-30 from the headers under
> `include/batchlas/blas/device/`; no measurements are recorded here.

The device group BLAS is a set of function templates that run **inside** a SYCL
kernel. A group of work-items (a work-group, a sub-group, or the work-group of an
`nd_item`) cooperatively computes **one** BLAS operation on **one** matrix or vector.
There is no queue, no `Event` and no batching at this level: the caller's kernel owns
the launch, picks the batch item (`view.batch_item(b)`), and synchronises afterwards.
The API reference is the @ref device group; this page is the calling contract and
the reasoning behind it.

## Device group BLAS: what is provided

| level | operations | header |
| --- | --- | --- |
| 1 | `fill`, `copy`, `copyc` (conjugating copy), `scal`, `axpy`, `hadamard` (element-wise, with an arbitrary n-ary functor), `dotu`, `dotc` | `detail/group_blas_fill.hh`, `detail/group_blas_vector.hh` |
| 2 | `gemv`, `ger` / `geru` / `gerc`, `trmv`, `symv`, `hemv` | `detail/group_blas_{gemv,ger,trmv,symv}.hh` |
| 3 | `gemm`, `trmm`, `symm`, `syrk`, `herk`, `syr2k`, `her2k` | `detail/group_blas_{gemm,trmm,symm,rankk}.hh` |

Absent on purpose or not yet written: `hemm`, any triangular **solve** (`trsv`,
`trsm`), and the banded and packed formats. Every operand is a dense column-major
`KernelMatrixView<T, MatrixFormat::Dense>` or a strided `VectorView<T>`; the views
carry `ld`, `inc` and the batch stride, and every kernel reads them rather than
assuming a natural layout.

The mathematics follows reference BLAS with two deliberate differences:

* **`trmv` is out of place.** It computes
  \f$ y := \alpha\,\mathrm{op}(A)\,x + \beta\,y \f$ with `A` triangular, not BLAS's
  in-place \f$ x := \mathrm{op}(A)\,x \f$. `x` and `y` may be the same vector: the
  rows are visited in the order that never overwrites an entry a later row still reads.
* **`trmm` has a `beta` and a separate output.** It computes
  \f$ C := \alpha\,\mathrm{op}(A)\,B + \beta\,C \f$ (left) or
  \f$ C := \alpha\,B\,\mathrm{op}(A) + \beta\,C \f$ (right). `B` and `C` may alias; the
  generic path orders its traversal for that case, and aliasing switches the fast
  paths off (see [the traps](#device-group-blas-traps-for-callers)).

## Device group BLAS: calling conventions

Every operation has the same shape:

```cpp
template <[DeviceBlasPolicy Policy,] <compile-time mode params>, typename Group, typename T>
void op(const Group& exec, <operands>..., T alpha = 1, T beta = 0, T* workspace = nullptr);
```

* **Mode parameters are template parameters.** `Transpose`, `Uplo`, `Side` and `Diag`
  are compile-time (`gemm<Transpose::Trans, Transpose::NoTrans>(...)`), so each mode is
  a separate instantiation and the per-element branch on the mode is resolved at
  compile time. A leading `DeviceBlasPolicy` parameter selects the policy overload;
  without it the policy is `Auto`.
* **Two spellings per operation.** The expanded form takes the operands and scalars
  directly; the operand-struct form takes a `MatrixVectorOperand`, `MatrixMatrixOperand`,
  `RankKOperand` or `Rank1UpdateOperand` built with the `make_*_operand` helpers. They
  are the same function.
* **Collective.** Every work-item of the executor must make the call, with the same
  arguments, from uniform control flow. The implementations use
  `sycl::reduce_over_group`, `sycl::select_from_group` and `sycl::group_barrier`; a
  work-item that skips the call deadlocks or corrupts the reduction.
* **One logical problem.** Each view must describe a single matrix or vector
  (`batch_size() == 1`); pass `batch_item(b)` for batched data. This, and every shape
  check, is an `assert` and therefore disappears under `NDEBUG`: a release build does
  not diagnose a wrong shape, it computes garbage or reads out of bounds.
* **`beta == 0` still reads the output.** Every path forms `alpha * acc + beta * C`
  (or `y *= beta`), so a NaN or Inf already in `C` / `y` survives a `beta` of zero.
  Initialise the output. This differs from reference BLAS.
* **Results are not published by the call.** Several paths write each output element
  from one work-item (the group leader after a reduction, or the lane that owns a
  tile). Issue a `sycl::group_barrier` on the executor before any other work-item
  reads the output, as with any other cross-work-item write.
* **Level 1 reductions return the value to every work-item.** `dotu` and `dotc`
  return the group-wide sum on every work-item of the executor.

## Device group BLAS: sub-group versus work-group

The executor type decides both the collective scope and which implementations are
eligible.

| executor passed | collective scope | eligible paths |
| --- | --- | --- |
| `sycl::sub_group` | the sub-group; independent sub-groups can each solve their own problem | generic only |
| `sycl::group<D>` | the work-group | generic, plus the local-memory tiled `gemv` / `symv` / `hemv` |
| `sycl::nd_item<1>` / `nd_item<3>` | its work-group | everything, including the sub-group fast paths of the level-3 ops |

The level-3 **sub-group fast paths** exist only behind an `nd_item`, because they need
the item's sub-group and its position in the launch. They are chosen at run time,
per call, and only when all of these hold:

* the scalar is `float` (the tiled complex rank-k / rank-2k kernels take
  `std::complex<float>`); every other type always runs the generic path;
* the policy admits the actual sub-group size: `Auto` takes 16 or 32, `Subgroup16`
  and `Subgroup32` exactly that size, `Generic` never takes a fast path;
* the work-group has the exact size the tiled kernel was written for (256 work-items
  for the register-tiled and aligned GEMM paths), and the extents clear the kernel's
  minimum tile;
* a workspace pointer was passed, for every path that stages through local memory.

`DeviceBlasPolicy` is ignored by the level-1 operations, `ger` and `trmv`, which have a
single implementation.

**Three-dimensional launches.** With an `nd_item<3>`, dimensions 1 and 2 of the
work-group id index output tiles (128 x 64 for the register-tiled paths, 32 x 8
work-items per group), so several work-groups share one output matrix. When no fast
path applies to a call, `gemm`, `syrk`/`herk` and `syr2k`/`her2k` run the generic
path in **tile-group (0, 0) only**: every group running the generic loop over the
whole output would race. The result is correct but computed by one work-group, so a
3-D launch that misses its fast path is slow rather than wrong.

## Device group BLAS: the workspace protocol

Paths that stage operands in local memory take a `T* workspace` that the **caller**
allocates. The protocol:

1. Describe the launch with `make_group_launch_info(local_size)`,
   `make_nd_item_1d_launch_info(local_size, sg_size)` or
   `make_nd_item_3d_launch_info(local_size, sg_size)`. It must describe the launch you
   will actually make: the size query evaluates the same eligibility predicates the
   call evaluates on the real executor.
2. Call the matching `*_workspace_elements<T, ...>(launch, extents...)` with the same
   template arguments as the operation. It returns a count of **elements of `T`**, not
   bytes, and 0 when no staged path applies.
3. Allocate `sycl::local_accessor<T, 1>` of that many elements in the command group,
   and pass `batchlas::util::get_raw_ptr(acc)` to the call. When the query returned
   0, pass `nullptr`.

A non-null pointer is an assertion that the buffer is large enough for whichever
staged path the call selects; the call does not know the buffer's size. Passing
`nullptr` is always safe and always correct; it only forgoes the staged paths.

The fixed local-memory structs behind the queries are
`GemvTransposeWorkspace` (a 32 x 33 tile plus a 32-element slice of `x`),
`SymvTransposeWorkspace` (16 x 17 plus two 16-element slices), and, for the level-3
fast paths, `RegisterMatrixWorkspace`, `OptimizedGemmWorkspace` / `GemmWorkspace` and
`ComplexRank2kWorkspace`. Tile strides are padded by one element (`TileK + 1`,
`TileN + 1`) so a column walk does not hit one shared-memory bank.

## Device group BLAS: register and local-memory caveats

* **The local-memory budget is a compile-time constant.** Each level-3 workspace is
  sized against `device_limits::subgroup_workspace_budget_bytes()`; a type whose
  workspace does not fit gets a fast-path predicate that is constantly false, and the
  structs clamp their array extents to at least 1 so no zero-length array is formed.
  Where that number comes from and what moving it changes:
  [the subgroup workspace budget](../perf/gemm.md#the-subgroup-workspace-budget).
* **Registers.** The register-tiled paths keep a 4 x 8 accumulator tile per work-item
  (`RegisterMatrixAccumTile`), and the complex rank-2k path a 4 x 4 tile of
  `std::complex<float>`. That is why the fast paths are restricted to 32-bit scalars:
  a `double` or `complex<double>` tile of the same shape at 256 work-items per group
  would exceed the register file, and on NVIDIA an over-subscribed launch is rejected
  by the driver rather than slowed down. Widening the type list means shrinking the
  thread tile first.
* **Barriers inside the fast paths.** The tiled kernels synchronise with
  `sycl::group_barrier` on the work-group and, between stages, on the sub-group. A
  kernel that calls them must therefore keep every work-item of the work-group live
  until the call returns, including work-items whose output tile is empty.
* **Alignment.** The aligned GEMM path (NN, `float`) loads 4-wide packets and requires
  `data`, `ld` and the batch stride of `A` and `B` to be multiples of 4 elements, the
  extents to be multiples of the tile, and a caller who asserts the alignment through
  `aligned_a` / `aligned_b` in the workspace query. Its column-extent gate tests a
  multiple of 64 (`kRegisterMatrixTileN`) although that kernel's own tile is 32 wide
  (`kOptimizedGemmTileN`); the gate is stricter than the kernel needs. Observed while
  documenting, not measured.

## Device group BLAS: traps for callers

* `trmm_workspace_elements` (the overload without a policy) lists `T` **after** the four
  mode parameters: `trmm_workspace_elements<Side::Left, Uplo::Upper, Transpose::NoTrans,
  Diag::NonUnit, float>(...)`. Every other `*_workspace_elements`, and the policy
  overload of this one, takes `T` first.
* `trmm_workspace_elements` has an `aliased` argument. When `B` and `C` overlap, the
  call skips the fast paths, so size with `aliased = true` (which returns 0) rather than
  allocating local memory the call will not use.
* `symv` / `hemv` issue `sycl::group_barrier` on the executor on the tiled path, so pass
  a `sycl::group` or `sycl::sub_group`, not an `nd_item`.
* `herk` / `her2k` take a complex `alpha` and `beta` (`T`), unlike BLAS's real scalars.
  The kernels force the imaginary part of each diagonal element of `C` to zero; a
  non-real `alpha` in `herk` still produces a non-Hermitian off-diagonal.
* `hemv` / `herk` / `her2k` on a real `T` are exactly `symv` / `syrk` / `syr2k`: the
  Hermitian flag is `ComplexScalar<T>`.

## Device group BLAS: where it is used

In the library: `src/extensions/latrd_lower_panel.cc`, `larft_wy.hh`,
`ormqr_cta.cc`, `ormqr_blocked.cc`, `sytrd_blocked.cc`, and `src/math-helpers.hh`.
Tests: `tests/device_blas_tests.cc` (every operation against a host reference, and the
`Auto` and `Generic` policies against each other). Benchmarks:
`benchmarks/device_blas_level{1,2,3}_benchmark.cc`.

## Device group BLAS: open debts

* No evidence page measures these kernels. The fast-path gates (the 256-work-item
  shape, the minimum extents, the `float`-only type list) are design constants, not
  measured windows.
* No triangular solve and no `hemm`.
* The tiled `symv` / `hemv` path cannot take an `nd_item` executor, unlike `gemv`, which
  unwraps one to its work-group.

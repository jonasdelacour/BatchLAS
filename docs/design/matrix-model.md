# Matrix model {#design_matrix_model}

> **Status:** current · checked 2026-10-06

Entry points take a `MatrixView`: a pointer, shape, leading dimension and batch stride over memory
the caller owns. `Matrix` owns memory and produces views. `KernelMatrixView` is the struct a kernel
captures. The user contract is in [the C++ API guide](../cpp-api.md#column-major-always).

## Matrix model: column-major storage

Dense storage is column-major, as in LAPACK and the vendor libraries, so view pointers, `ld` and
stride pass to vendor kernels unchanged. The `Layout` enum only labels host CBLAS/LAPACKE calls. There
is no row-major `Matrix`. Row-major data is converted at the edge (`gemm` by operand swap, others by
`to_column_major()` / `to_row_major()` with an explicit row pitch).

## Matrix model: the element address and its 64-bit batch term

Element \f$(i, j)\f$ of item \f$b\f$ is at \f$\mathtt{data}[\,b \cdot \mathtt{stride} + j \cdot \mathtt{ld} + i\,]\f$.
`rows`, `cols`, `ld`, `stride` and `batch_size` are `int`, but every accessor (including `batch_item`)
evaluates the batch term in `int64_t`. A 512 x 512 `float` item has `stride == 262144`, so the product
wraps in `int` at \f$b = 8192\f$. The in-item offset cannot overflow for an allocatable matrix.

> **Note:** debug asserts must compare in `int64_t`. An `int` index has already wrapped, and a negative
> one converted to `size_t` passes the check (`VectorView::at`).

## Matrix model: how ld and stride resolve

`0` means packed:

| Argument | `0` resolves to | Constraint |
| --- | --- | --- |
| `ld` | `rows` | resolved `ld >= rows` |
| `stride` | resolved `ld * cols` | `Matrix`: resolved `stride >= ld * cols` |

- The stride resolves against the **resolved** `ld`. Writing `ld * cols` with `ld == 0` gives
  `stride == 0`, and every element access fails the debug bounds assert.
- `stride = 0` is not a broadcast. A `batch_size > 1` view over one matrix reads past the buffer.
- `Matrix(rows, cols, batch, ld, stride)` allocates `stride * batch` elements and pads nothing. The copying
  constructor reads the source with its own `ld` and `stride`, keeps the source `ld`, and packs the copy.

## Matrix model: what each constructor validates

- `Matrix` rejects negative arguments, resolved `ld < rows` and resolved `stride < ld * cols`.
- The copying `Matrix` and `Span` forms also reject null data, non-positive extents or batch, a
  `stride < ld * cols` when `batch > 1`, and a `Span` shorter than the layout.
- `MatrixView` rejects negative arguments and resolved `ld < rows`. It accepts null data, zero extents
  and `stride < ld * cols`.

The `ld < rows` check catches an argument-order trap. The batch is the third argument of
`Matrix(rows, cols, batch)` but the sixth of `MatrixView(data, rows, cols, ld, stride, batch)`, so
`MatrixView<float> V(p, n, n, batch)` means `ld = batch` and throws when `batch < n`.

Unchecked on purpose: null data and zero extents (shape-only views are the argument to about 37
`*_buffer_size` queries), and `stride >= ld * cols`. Two in-repo sites violate the last one, the
transposed CGS view in `src/extensions/ortho.cc` (see [known defects](known-defects.md)) and a sizing
dummy in `syevx_lobpcg`. A throw would break both.

`rows_`, `cols_` and `batch_size_` must default-initialise to zero. The USM check reads them to decide
whether a null pointer is legal.

## Matrix model: owning Matrix versus MatrixView

- `Matrix<T, F>` owns a USM-shared `UnifiedVector<T>`, copies deeply, and frees on destruction.
  Factories (`Identity`, `Random`, `Zeros`, ...) wait before returning, so the host can read elements.
- `MatrixView<T, F>` aliases caller memory and must not outlive it, including in-flight kernels. A view
  over `sycl::malloc_device` memory is not host-dereferenceable.
- `Matrix::clone()` copies in the source layout. For CSR it allocates `matrix_stride` slots per item.

## Matrix model: the pointer array and sliced views

Pointer-array vendor calls (cuBLAS `*Batched`) need one base pointer per item. A `Matrix` with
`batch_size > 1` allocates that array; `data_ptrs(ctx)` refills it from the view and waits. A sliced
view inherits its parent's array, which holds the **unsliced** addresses until `data_ptrs(ctx)` is
called on the slice. Fill it for the view immediately before the call.

## Matrix model: KernelMatrixView, the device-side view

`KernelMatrixView<T, F>` is an aggregate of raw pointers and `int` extents, built by `kernel_view()` and
passed by value, with no ownership or backend handle.

- Dense `operator()(i, j, b)` computes the address above. Its checks are debug asserts only.
- CSR `get(i, j, b)` searches row \f$i\f$ linearly and returns zero for an absent entry.
- `batch_item(b)` returns a single-item view, or 0 x 0 for an out-of-range `b`.
- Slicing does not diagnose an empty slice (see [open debts](#matrix-model-open-debts)).

## Matrix model: heterogeneous batches

Per-item **active extents** (`Matrix::set_active_dims`, `MatrixView::with_active_dims`) give item
\f$b\f$ a leading `active_rows[b] x active_cols[b]` block. `ld` and `stride` are unchanged. `rows()` and
`cols()` report capacity; `rows(b)` and `cols(b)` report the active extents.

## Matrix model: CSR storage and the nnz capacity

Per item, CSR stores `rows + 1` offsets (`offset_stride`) and values and column indices (`matrix_stride`).

`nnz()` is a per-item **capacity**. `convert_to<MatrixFormat::CSR>()` sizes the batch by its largest item,
so `for (k = 0; k < nnz(); ++k)` on a heterogeneous batch walks past smaller items. This is deliberate:
vendor SpMM descriptors take one count for a strided batch. Use `nnz(b)` for an item's count and
`nnz_capacity()` for buffer sizes. See [the C++ API guide](../cpp-api.md#the-csr-non-zero-count-has-its-own-type).

## Matrix model: strong types for positional integers

- CSR `Matrix(rows, cols, nnz)` and dense `Matrix(rows, cols, batch)` take the same third `int`, so the CSR
  count is `NonZeros{nnz}` and the bare-`int` CSR overload is deleted. The `nnz`-first CSR order is deleted too.
- `Vector(size, batch, stride, inc)` and `VectorView(data, size, batch, inc, stride)` take the same values in
  opposite order, and both fit the buffer, so a transliterated call silently reads wrong addresses. `Vector`
  takes `Stride{}` and `Inc{}`. Its bare-`int` overloads, and `Vector(size, batch, stride)`, are deleted
  (the last would make `batch` the fill value).
- Tags (`NonZeros`, `Inc`, `Stride`, `Ld`, `BatchSize`) have an `explicit` constructor and a **deleted**
  `operator int`. Without that deletion a tag decays back into the `int` ambiguity.

## Matrix model: vectors, inc and stride

Entry \f$i\f$ of item \f$b\f$ is at \f$b \cdot \mathtt{stride} + i \cdot \mathtt{inc}\f$; `stride = 0` means
`size * inc`. The span touched is \f$(\mathtt{batch}-1)\,\mathtt{stride} + (\mathtt{size}-1)\,\mathtt{inc} + 1\f$
(`required_span_length`). A matrix row is a `VectorView` with `inc = ld`; a column has `inc = 1`.
`Vector::view()` is needed where template deduction must see a `VectorView`.

## Matrix model: Span and UnifiedVector

`Span<T>` is the 1-D argument type (eigenvalues, pivots, `tau`, `info`, workspace). Its single-element
constructor `Span(T&)` is `explicit`: an implicit one let `syev(q, A, W[i * n], ...)` compile and overrun
the caller's array.

`UnifiedVector<T>` is the owning USM-shared array behind `Matrix` and `Vector`. It reallocates on growth,
which invalidates earlier spans. `resize(n, value)` is a no-op unless `n` exceeds the capacity; `resize(n)`
is not. Global-scope names are using-declarations, which `BATCHLAS_NO_GLOBAL_NAMES` removes.

## Matrix model: USM hints need the caller's Queue

Advice calls (`set_access_device`, `prefetch`, ...) take a mandatory `const Queue&`. A default `Queue` is on
`Device::default_device()`, the wrong device in a multi-GPU process. `Span`'s advice calls default the queue
and swallow errors.

## Matrix model: open debts

- `MatrixView::transpose(ctx, conjugate)` is declared but not defined. A call fails to link.
- `fill_triangular_random` and `fill_tridiag_toeplitz` address items as packed \f$n \times n\f$, ignoring `ld`
  and `stride`. `fill_triangular_random` seeds by position, so every item gets the same matrix.
- `fill_random` writes all of `data()`, padding included, so on a sub-block view it writes neighbours.
- `KernelMatrixView` slicing uses `assert("...")`, which is always true, so empty slices pass.
- `UnifiedVector`'s move assignment leaks the destination's previous allocation.
- `fill_diagonal(ctx, Span, k)` with `k != 0` and one shared diagonal builds a view over only \f$n - |k|\f$
  values, so its debug assert can fire on a valid call.

# Matrix model: storage, views and the device-side view {#design_matrix_model}

> **Covers:** how a batch of matrices is laid out in memory (column-major, `ld`, batch
> `stride`, CSR strides), the owning `Matrix` versus the non-owning `MatrixView`, the
> device-side `KernelMatrixView`, heterogeneous batches, the vector types, and the strong
> integer types that keep positional arguments apart.
> **Status:** current; checked against `include/batchlas/blas/matrix.hh` and `src/matrix.cc`
> on 2026-09-30.

Every entry point takes a `MatrixView`: a pointer, a shape, a leading dimension and a batch
stride, over memory the caller owns. `Matrix` is the owning container that produces such
views, and `KernelMatrixView` is the plain struct a kernel captures. The user-facing
statement of the layout contract is in [the C++ API guide](../cpp-api.md#column-major-always);
this page records why the model looks the way it does and what each type guarantees.

## Matrix model: why column-major

Dense storage is column-major, as in LAPACK and the vendor BLAS/LAPACK libraries
(cuBLAS, cuSOLVER, rocBLAS, rocSOLVER, CBLAS with `CblasColMajor`). Every vendor route passes
a `MatrixView`'s pointer, `ld` and stride straight through, with no transposition or copy, and
the native kernels follow the same convention so the two kinds of route are interchangeable
behind one `Route`. The `Layout` enum exists only to label host CBLAS/LAPACKE calls; there is
no row-major `Matrix`.

Row-major data is handled at the edge rather than inside the model: `gemm` by the operand
swap (\f$C^T = B^T A^T\f$), everything else by `Matrix::to_column_major()` and
`to_row_major()`, which take the row pitch as an explicit parameter and never infer it. See
[row-major source data](../cpp-api.md#row-major-source-data).

## Matrix model: the element address and its 64-bit batch term

Element \f$(i, j)\f$ of batch item \f$b\f$ of a dense matrix lives at

\f[ \mathtt{data}[\,b \cdot \mathtt{stride} + j \cdot \mathtt{ld} + i\,], \f]

counted in elements. `rows`, `cols`, `ld`, `stride` and `batch_size` are `int`, but every
accessor evaluates the batch term in `int64_t`:
`KernelMatrixView::operator()` and `::batch_item`, `Matrix::operator()`, `MatrixView::at`
and `::batch_item`, `Vector::at` and `VectorView::at`.

Only the batch term is widened, deliberately. \f$b \cdot \mathtt{stride}\f$ in `int` wraps at
batch sizes this library is built for: a 512 x 512 `float` item has `stride == 262144`, so
the product wraps at \f$b = 8192\f$. \f$j \cdot \mathtt{ld} + i\f$ is an offset inside one
item and can only overflow for a single matrix above \f$2^{31}\f$ elements (8 GB of `float`),
which cannot be allocated anyway. In `batch_item` the widening matters more than in element
access: an overflowed product there poisons a base pointer, not one element.

The debug asserts that guard these accessors compare `int64_t` against `int64_t`. An index
built in `int` has already wrapped, and converting it to `size_t` for the comparison turns a
negative index into one that passes the check (`VectorView::at`).

## Matrix model: how ld and stride resolve

`ld` is the element distance between column \f$j\f$ and \f$j+1\f$; `stride` is the element
distance between item \f$b\f$ and \f$b+1\f$. Both use `0` for "packed":

| Argument | `0` resolves to | Constraint |
| --- | --- | --- |
| `ld` | `rows` | resolved `ld >= rows` |
| `stride` | resolved `ld * cols` | `Matrix`: resolved `stride >= ld * cols` |

The stride resolves against the **resolved** `ld`, not the argument. Written as
`ld * cols`, the three-argument `KernelMatrixView(data, rows, cols)` gets `stride_ = 0`, and
every element access then fires the bounds assert in a debug build.

`stride = 0` is the packed default, **not** cuBLAS's stride-0 broadcast. BatchLAS has no
broadcast operand: a view with `batch_size > 1` over a buffer that holds one matrix reads past
the end of the buffer. To apply one matrix to many right-hand sides, fold the batch into
columns or replicate the matrix.

`Matrix(rows, cols, batch, ld, stride)` allocates exactly `stride * batch` elements; nothing
pads. The copying constructors `Matrix(data, rows, cols, ld, stride, batch)` read the
*source* with the given `ld` and `stride`, keep the source's `ld` in the copy, and pack the
items back to back (`stride() == ld() * cols()`). They copy column by column when the source
is padded, so padding the caller never allocated is never read, and zero the copy's padding
rows.

## Matrix model: what each constructor validates

| Constructor | Rejects (`batchlas::invalid_argument`) | Deliberately accepts |
| --- | --- | --- |
| `Matrix(rows, cols, batch, ld, stride)` | negative arguments; resolved `ld < rows`; resolved `stride < ld * cols` | zero extents |
| `Matrix(data, rows, cols, ld, stride, batch)` and the `Span` forms | null data; `rows` or `cols <= 0`; `batch <= 0`; negative `ld`/`stride`; resolved `ld < rows`; `stride < ld * cols` when `batch > 1`; a `Span` shorter than the layout needs | |
| `MatrixView(data, rows, cols, ld, stride, batch, ptrs)` | negative arguments; resolved `ld < rows` | null `data`; zero extents; `stride < ld * cols` |

The `ld < rows` check on `MatrixView` is aimed at one trap. The batch size is the *third*
argument of `Matrix(rows, cols, batch)` and the *sixth* of `MatrixView(data, rows, cols, ld,
stride, batch)`, so a caller who learned the order from `Matrix` writes
`MatrixView<float> V(p, n, n, batch)`, meaning `ld = batch`, `stride = batch * n`,
`batch_size = 1`. That used to be accepted and produce plausible wrong numbers. It now throws
whenever `batch < n`, and the message names the spelling that was probably meant.

What `MatrixView` does not check, and why:

- **A null pointer or a zero extent.** A shape-only view (`MatrixView<T>(nullptr, ...)`) is
  the standard argument to a `*_buffer_size` workspace query; about 37 in-repo sites build
  one, some with shape `(nullptr, 0, 0, 1, 1, batch)`.
- **`stride >= ld * cols`.** Two in-repo call sites violate it: the transposed CGS view in
  `src/extensions/ortho.cc` (see [known defects](known-defects.md)) and a workspace-sizing
  dummy in `syevx_lobpcg`. A throw would take out `ortho` and `syevx_lobpcg_buffer_size`,
  and the check adds nothing against the argument-order trap, where the resolved stride
  equals `ld * cols` exactly.

`MatrixView`'s shape members `rows_`, `cols_` and `batch_size_` must stay
default-initialised to zero: the entry points' USM check reads them to decide whether a null
data pointer is legal, and indeterminate values made a valid call on a default-constructed
view throw "The pointer is null.".

## Matrix model: owning Matrix versus MatrixView

| | `Matrix<T, F>` | `MatrixView<T, F>` |
| --- | --- | --- |
| Storage | owns a `UnifiedVector<T>` (USM shared, `Device::default_device()`) | a `Span<T>` over caller memory |
| Copy | deep copy (values, active extents, CSR arrays) | aliases the same memory |
| Lifetime | frees on destruction | must not outlive the memory, including in-flight kernels |
| Backend handle | created lazily, shared with its views | shares the viewed `Matrix`'s handle, or creates its own |
| Passed to entry points | converts implicitly to a view | directly |

`Matrix` memory is USM **shared**, so the host can read elements after waiting on the queue,
and the factories (`Identity`, `Random`, `Zeros`, ...) wait before returning because the
result owns the memory their kernel writes. A `MatrixView` makes no such promise: over
`sycl::malloc_device` memory the host cannot dereference it, which is why
`MatrixView::nnz(int)` documents a host-readable precondition and `KernelMatrixView::nnz(int)`
exists for use inside a kernel.

`Matrix::clone()` allocates in *this* matrix's layout, not a packed one: the copy is a flat
`std::copy` of `stride * batch_size` elements, so a packed destination would be overrun by any
padded `ld`. For CSR it allocates `matrix_stride` slots per item, not `nnz`, for the same
reason.

## Matrix model: the pointer array and sliced views

Pointer-array batched vendor calls (cuBLAS `*Batched`) take one base pointer per item. A
`Matrix` with `batch_size > 1` allocates that array next to its data; a `Matrix` with batch
size 1 does not, and `data_ptrs(ctx)` on it throws. `data_ptrs(ctx)` (re)fills the array from
the view's own base and stride on `ctx` and waits.

A sliced view (`M(Slice, Slice)` or `V(Slice, Slice)`) carries its parent's pointer array.
Until something calls `data_ptrs(ctx)` on the slice, that array holds the **unsliced** base
addresses, which a pointer-array backend would read for the slice. When it is refilled for
the slice, it no longer describes the parent. The array is shared state; code that needs a
pointer array for a particular view must fill it for that view immediately before the call.

## Matrix model: KernelMatrixView, the device-side view

`KernelMatrixView<T, F>` is what a kernel captures. It is an aggregate of raw pointers and
`int` extents, statically asserted trivially copyable for both formats, and specialised on
the format through `requires` clauses so format-specific code resolves at compile time. It is
built on the host by `kernel_view()` and passed by value; it holds no ownership and no backend
handle.

- Dense `operator()(i, j, b)` computes the address above; its only checks are debug asserts.
- CSR `get(i, j, b)` searches row \f$i\f$ linearly and returns zero for an absent entry.
- `batch_item(b)` returns a single-item view with that item's active extents, and a 0 x 0 view
  for an out-of-range `b` rather than a fault. For CSR it assumes column indices use the same
  per-item stride as the values (`matrix_stride`), which every constructor guarantees.
- The slicing operators mirror `MatrixView`'s pointer arithmetic but do not diagnose an empty
  slice (see [open debts](#matrix-model-open-debts)).

## Matrix model: heterogeneous batches

A dense batch may carry per-item **active extents**: `Matrix::set_active_dims(rows, cols)`
copies them into the matrix, `MatrixView::with_active_dims` attaches caller-owned spans to a
view. Item \f$b\f$ then occupies the leading `active_rows[b] x active_cols[b]` block of the
allocated `rows() x cols()`; `ld` and `stride` do not change. `rows()` and `cols()` report
the capacity, `rows(b)` and `cols(b)` the active extents, and `is_heterogeneous()` whether
any are attached. Validation requires both spans to hold `batch_size` entries, each in
`[0, capacity]`; passing two empty spans clears them.

A view built from another view with overridden extents keeps the active extents only when
every override is a no-op, because active extents larger than a narrowed capacity would be
invalid.

## Matrix model: CSR storage and the nnz capacity

A CSR batch stores, per item, `rows + 1` zero-based row offsets (items `offset_stride` apart,
default `rows + 1`) and value and column-index arrays (items `matrix_stride` apart, default
`nnz`). The non-zero count is spelled with the `NonZeros` type (next section).

`nnz()` is a per-item **capacity**, not a count. `convert_to<MatrixFormat::CSR>()` sizes a
whole batch by its largest item, so on the heterogeneous batch it usually produces,
`for (k = 0; k < nnz(); ++k)` walks past a smaller item's entries. This is deliberate: the
vendor SpMM descriptors (`cusparseCreateCsr`, `rocsparse_create_csr_descr`) take one number
for a strided batch, and the capacity is the right one there. `nnz(b)` reads item \f$b\f$'s
actual count from its offsets, and `nnz_capacity()` returns `matrix_stride`, which the
from-data constructor lets exceed the declared count and which is what sizes the buffers.
The user-facing statement is in
[the C++ API guide](../cpp-api.md#the-csr-non-zero-count-has-its-own-type).

## Matrix model: strong types for positional integers

Several constructors take runs of `int` whose meaning depends on position, and two of them
disagree:

- `Matrix(rows, cols, batch)` versus `Matrix(rows, cols, nnz)` for CSR: the third `int` means
  a different thing per format. The CSR count is therefore `NonZeros{nnz}`, and the bare-`int`
  CSR overload is `= delete`.
- `Vector(size, batch, stride, inc)` versus `VectorView(data, size, batch, inc, stride)`: the
  same two values in the opposite order, and both readings fit the same buffer, so a
  transliterated call reads every element from the wrong address and nothing reports it.
  `Vector` therefore takes `Stride{}` and `Inc{}`, and its bare-`int` overloads are deleted.
  The three-`int` `Vector(size, batch, stride)` is deleted too: `Vector<float>(n, batch, stride)`
  would otherwise pick `(size, value, batch_size)` and turn `batch` into the fill value.
  `VectorView` keeps its positional constructors (every in-repo use is correct) and adds
  tagged ones for new code.
- The pre-tag CSR from-data order (`nnz` before `rows, cols`) is a deleted overload on both
  `Matrix` and `MatrixView`, so old code fails to compile instead of being reinterpreted.

The tags (`NonZeros`, `Inc`, `Stride`, `Ld`, `BatchSize`) are trivially copyable structs with
an `explicit` constructor and a **deleted** `operator int`. The deleted conversion is
load-bearing: without it a tag decays back into the `int` ambiguity it exists to remove.

## Matrix model: vectors, inc and stride

`Vector<T>` (owning, `UnifiedVector` storage) and `VectorView<T>` (non-owning) describe
`batch_size` vectors of `size` entries: entry \f$i\f$ of item \f$b\f$ is at
\f$b \cdot \mathtt{stride} + i \cdot \mathtt{inc}\f$, with `stride = 0` meaning `size * inc`.
`required_span_length` gives the elements a layout touches,
\f$(\mathtt{batch}-1)\,\mathtt{stride} + (\mathtt{size}-1)\,\mathtt{inc} + 1\f$, and a
`VectorView` over a `Span` asserts the span is at least that long.

A row of a dense matrix is a `VectorView` with `inc = ld` and the matrix's stride; a column
has `inc = 1`. `MatrixView(VectorView, VectorOrientation)` goes the other way: `Column` gives
an \f$n \times 1\f$ view and requires `inc == 1`, `Row` a \f$1 \times n\f$ view with `ld = inc`.
`Vector::view()` exists because template argument deduction ignores the implicit conversion,
so a call that must deduce `T` cannot take a `Vector` directly.

## Matrix model: Span and UnifiedVector

`Span<T>` is the 1-D argument type (eigenvalues, pivots, `tau`, `info`, workspace). Its
single-element constructor `Span(T&)` is `explicit`. Implicitly, any scalar lvalue became a
length-1 buffer wherever a `Span` parameter's element type is not deduced from that argument,
which is the whole eigenvalue, singular-value, pivot and workspace surface:
`syev(q, A, W[i * n], ...)`, the LAPACK idiom without the `&`, compiled and overran the
caller's array on the backends that do not size-check.

`UnifiedVector<T>` is the owning USM-shared array under `Matrix` and `Vector`. It allocates in
the shared context of `Device::default_device()`, grows by reallocation (which invalidates
spans taken earlier), and deep-copies on copy. `resize(n, value)` is a no-op unless `n` exceeds
the capacity, unlike `resize(n)`.

`Span`, `is_std_array` and `UnifiedVector` used to live at global scope; using-declarations
keep the old spellings until nothing in tree needs them, and `BATCHLAS_NO_GLOBAL_NAMES`
switches them off. The free `swap(UnifiedVector&, UnifiedVector&)` is deliberately not
re-exported: ADL finds it, and a global `swap` is the kind of collision the move removed.

## Matrix model: USM hints need the caller's Queue

`set_access_device`, `prefetch` and the other advice calls on `Matrix`, `MatrixView`,
`Vector` and `VectorView` take a **mandatory** `const Queue&`. A default-constructed `Queue`
is a new queue on `Device::default_device()`, so a defaulted argument would put the hint on
the wrong device in a multi-GPU process. `Span`'s own advice calls still default the queue,
and swallow runtime errors (they are hints). For the same reason `MatrixView::fill_zeros` and
`fill_ones` have no queue-less overload, while the older `fill`, `fill_random`, `symmetrize`,
... keep theirs for compatibility.

## Matrix model: open debts

- `MatrixView::transpose(ctx, conjugate)` is declared in `matrix.hh` but defined nowhere in
  `src/`; a call fails to link. Either implement it or delete the declaration.
- `MatrixView::fill_triangular_random` and `fill_tridiag_toeplitz` address items as packed
  \f$n \times n\f$ (`b * n * n`), ignoring `ld` and `stride`, and `fill_triangular_random`
  seeds by position only, so every item of a batch receives the same matrix.
- `MatrixView::fill_random` writes every element of `data()`, the padding between columns and
  items included, which on a sub-block view means neighbouring data.
- `KernelMatrixView`'s slicing operators check the slice with `assert("...")`, which is always
  true, so an empty or inverted slice is not diagnosed.
- `fill_diagonal(ctx, Span, k)` builds a `VectorView` of length `n` over a span that, for
  `k != 0` and one shared diagonal, holds only `n - |k|` values; the view's debug length assert
  can fire on a valid call.

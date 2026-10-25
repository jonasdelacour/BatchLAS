# Symbol visibility {#design_symbol_visibility}

> **Status:** current · checked 2026-10-21

`BATCHLAS_API` is generated into `<build>/include/batchlas/export.hh` by `generate_export_header()`
against `batchlas_core_obj` (`src/CMakeLists.txt:121-125`). The entities below carry it where it
looks arbitrary. That placement is required.

## symbol visibility: build modes

- On ELF both arms expand to `__attribute__((visibility("default")))`, so one header serves split,
  monolithic and external builds. `batchlas_core_obj` is the target that exists in every mode; the
  split-mode `batchlas` facade is INTERFACE, for which `generate_export_header()` produces nothing.
- `-fvisibility=hidden -fvisibility-inlines-hidden` (`CXX_VISIBILITY_PRESET hidden`,
  `VISIBILITY_INLINES_HIDDEN`) applies only when `BATCHLAS_MONOLITHIC_LIBRARY=ON` and
  `BATCHLAS_HIDDEN_VISIBILITY=ON` (`src/CMakeLists.txt:141-146`).
- `BATCHLAS_HIDDEN_VISIBILITY` exists only in a monolithic tree and defaults to `ON` there unless
  tests are built: the in-tree tests reach about 1,000 unannotated internal symbols
  (`cmake/BatchLASOptions.cmake:280-288`). The macOS presets set it `OFF`.
- The split build keeps default visibility: four weak data objects
  (`batchlas_kernel_trace::g_records`, `batchlas::QueueImpl::device_arrays` and both guard
  variables) are defined in every component library and merge only under default visibility.
  Hidden, each `.so` would get its own kernel-trace vector and per-device array cache, with no
  undefined reference to point at it. The measurement (14 split libraries, 879 symbol references
  crossing a component boundary) predates `batchlas_blas` and `batchlas_extensions_symmetric`
  becoming INTERFACE.
- A missing annotation is invisible in the split build and in any tree that builds tests. It breaks
  a hidden-visibility monolithic build or a consumer's link. No in-tree test or CI job links a
  hidden-visibility monolithic install (the `release` preset is not built in CI).

## symbol visibility: enums used as template arguments

GCC and Clang give an instantiation the minimum visibility of its template and its arguments.
`Backend` and `MatrixFormat` are template parameters across the public surface (214
`template <Backend ...>` declarations in `include/`, 2026-10-21). An unannotated enum hides those instantiations, and `BATCHLAS_API` on the function does
nothing.

- Without the enum annotation, `batchlas::gemm<Backend::CUDA, float>` is a local `t` symbol. With it,
  an exported `W`.
- 688 of the 1,083 symbols a consumer links (63%) were affected.
- The failure is an undefined reference at the consumer's link, with no compiler diagnostic.

Annotate every enum in `include/batchlas/blas/enums.hh` that becomes a template argument
(`MatrixFormat` and `Backend` today, lines 72 and 111).

## symbol visibility: BinaryOp is a template-argument enum too

`batchlas::linalg::BinaryOp` (`include/batchlas/blas/linalg-ops.hh`) is annotated for the same reason.
Without it, all four `elementwise_into<float, BinaryOp::*>` specialisations stay hidden.

## symbol visibility: the export attribute on the Matrix class template

`BATCHLAS_API` is on the class templates `Matrix`, `MatrixView` and `VectorView`
(`include/batchlas/blas/matrix.hh`), not on the explicit instantiations in `src/matrix.cc` (313 when
counted; 205 lines start `template ` on 2026-10-21). Only 22 of them are `template class ...;`. The
rest instantiate member templates (factories, `convert_to`, `to_row_major` / `to_column_major`,
`MatrixView` members), which a whole-class instantiation does not reach. A class-level attribute
covers them all.

Do not put the attribute on an explicit instantiation. g++ 13 rejects
`template class __attribute__((visibility("default"))) F<double,1>;`; clang accepts it.

## symbol visibility: exception typeinfo must be exported

Every class in `include/batchlas/error.hh` carries `BATCHLAS_API`, including `detail::exception_bridge`,
and so does `NoRouteError` (`include/batchlas/no_route.hh`).
None has an out-of-line member, so vtables and typeinfo have vague linkage. Catch-by-type works only
because every copy has default visibility and the linker merges them. Under `-fvisibility=hidden` a
consumer's `catch (const batchlas::error&)` stops matching a throw from the library, and the exception
calls `std::terminate`. `BATCHLAS_API` forces default visibility. This is rule 1 of
@ref design_error_model reached another way. `detail::exception_bridge` is on the catch-time upcast
path, so its typeinfo must merge too.

> **Note:** A throw/catch test cannot check this on x86-64. This DPC++ emits typeinfo names without the
> leading `*`, so libstdc++ falls back to `strcmp` and duplicate typeinfos still compare equal. The test
> passes with or without the annotation. Only reading the annotations discriminates.

## symbol visibility: WorkspaceLease is exported at class level

`WorkspaceLease` is annotated on the class, not its members, because inline public members call the
private `release_(bool)`. Without the class annotation a consumer gets an unresolved reference to it.
`_ZN8batchlas14WorkspaceLease8release_Eb` is among the 1,083 symbols the in-tree tests need.

## symbol visibility: Event carries per-member exports

`Event` is `struct [[nodiscard]] Event` (`include/batchlas/util/sycl-device-queue.hh:280`), with
`BATCHLAS_API` on each of its ten out-of-line members.
g++ 13 rejects a class-key with both a C++11 and a GNU attribute, in either order (`-std=c++20`), so the
class-level form does not compile there. Clang accepts both. `Event` has no virtual member, so
per-member annotation is equivalent.

`[[nodiscard]]` is on the type, not on the roughly 279 functions returning it, because the operations are
generated forwarders. It is a warning. Dropping an `Event` is harmless on the default in-order `Queue`,
but it is a silent race on an out-of-order `Queue`, a second `Queue` sharing the context, or a hand-off to
raw SYCL. In-tree discards are written `(void)`.

## symbol visibility: exported inline Queue members

`Queue::attach_to_current_thread()` and `Queue::native_handle()` carry `BATCHLAS_API` on the member, even
though the class is annotated. Both are `[[gnu::used]] inline` in `src/queue.hh`
(`BATCHLAS_QUEUE_EXPORTED_INLINE`, plain `inline` in the device pass), as are the `sycl_queue`,
`sycl_event` and `event_from_sycl` helpers there. `used` keeps the symbol but does not set visibility, and
`-fvisibility-inlines-hidden` hides inline members whatever the class says.

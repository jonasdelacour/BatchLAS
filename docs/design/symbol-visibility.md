# Symbol visibility {#design_symbol_visibility}

> **Status:** current.

`BATCHLAS_API` is generated into `<build>/include/batchlas/export.hh` by `generate_export_header()`
against `batchlas_core_obj`. The entities below carry it where it looks arbitrary. That placement is
required.

## symbol visibility: build modes

- On ELF both arms expand to `__attribute__((visibility("default")))`, so one header serves split,
  monolithic and external builds.
- Only `-DBATCHLAS_MONOLITHIC_LIBRARY=ON` uses `-fvisibility=hidden -fvisibility-inlines-hidden`. The
  split build keeps default visibility: four weak data objects are defined in all 14 component
  libraries and merge only under default visibility.
- A missing annotation is invisible in the split build. It breaks the monolithic build or a consumer's
  link. No in-tree test links a monolithic install.

## symbol visibility: enums used as template arguments

GCC and Clang give an instantiation the minimum visibility of its template and its arguments.
`Backend` and `MatrixFormat` are template parameters across the public surface (207 `template <Backend ...>`
declarations). An unannotated enum hides those instantiations, and `BATCHLAS_API` on the function does
nothing.

- Without the enum annotation, `batchlas::gemm<Backend::CUDA, float>` is a local `t` symbol. With it,
  an exported `W`.
- 688 of the 1,083 symbols a consumer links (63%) were affected.
- The failure is an undefined reference at the consumer's link, with no compiler diagnostic.

Annotate every enum in `include/batchlas/blas/enums.hh` that becomes a template argument.

## symbol visibility: BinaryOp is a template-argument enum too

`batchlas::linalg::BinaryOp` (`include/batchlas/blas/linalg-ops.hh`) is annotated for the same reason.
Without it, all four `elementwise_into<float, BinaryOp::*>` specialisations stay hidden.

## symbol visibility: the export attribute on the Matrix class template

`BATCHLAS_API` is on the class templates `Matrix`, `MatrixView` and `VectorView`, not on the 313 explicit
instantiations in `src/matrix.cc`. Only 22 of those lines are `template class ...;`. The other 291
instantiate member templates (factories, `convert_to`, `to_row_major` / `to_column_major`, `MatrixView`
members), which a whole-class instantiation does not reach. A class-level attribute covers all 313.

Do not put the attribute on an explicit instantiation. g++ 13 rejects
`template class __attribute__((visibility("default"))) F<double,1>;`; clang accepts it.

## symbol visibility: exception typeinfo must be exported

Every class in `include/batchlas/error.hh` carries `BATCHLAS_API`, including `detail::exception_bridge`.
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

`Event` is `struct [[nodiscard]] Event`, with `BATCHLAS_API` on each of its ten out-of-line members.
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
(`BATCHLAS_QUEUE_EXPORTED_INLINE`). `used` keeps the symbol but does not set visibility, and
`-fvisibility-inlines-hidden` hides inline members whatever the class says.

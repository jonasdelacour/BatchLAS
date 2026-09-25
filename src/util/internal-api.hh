#pragma once

// BATCHLAS_INTERNAL_API -- default visibility for symbols that are NOT ABI.
//
// WHY A SECOND MACRO. Under BATCHLAS_MONOLITHIC_LIBRARY the object libraries are
// compiled -fvisibility=hidden, and BATCHLAS_API (from the generated
// <batchlas/export.hh>) is what pulls a declaration back to default visibility.
// A measurable slice of what the LIBRARY exports today is needed only by the
// in-tree tests, not by any consumer: 199 of the 1,083 symbols the 63 test
// binaries actually resolve out of libbatchlas*.so belong to 55 entities that
// are declared in no public header at all. They are reachable because eight-plus
// test TUs include a private header by relative path -- tests/gemm_tests.cc
// includes ../src/sycl/gemm_kernels.hh, tests/getrf_tests.cc includes
// ../src/extensions/getrf_native.hh, and so on -- and then link the shared
// library. Hidden visibility breaks those links.
//
// Spelling them BATCHLAS_API would fix the link and re-widen the very ABI the
// hidden-visibility work exists to bound: every one of those 55 would become a
// name the library promises. This macro expands to the same thing on ELF and
// carries the opposite promise. It lives here, in a header that is never
// installed, precisely so that the distinction survives: a symbol marked with it
// cannot appear in a consumer's translation unit, because the consumer cannot
// reach the declaration that names it.
//
// IF THE TEST SUITE STOPS INCLUDING PRIVATE HEADERS, DELETE THIS. Relinking
// those test targets against the OBJECT libraries instead of the .so is the
// other way to solve the same problem, and it is the better one -- it removes
// the symbols from the library's dynamic table entirely rather than exporting
// them under a different name.

#include <batchlas/export.hh>

#ifndef BATCHLAS_INTERNAL_API
#define BATCHLAS_INTERNAL_API BATCHLAS_API
#endif

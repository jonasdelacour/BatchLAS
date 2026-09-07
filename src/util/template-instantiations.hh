#pragma once

#define BATCHLAS_UNPAREN(...) __VA_ARGS__

// A comma that survives macro argument splitting, for signature aliases that
// take more than one template argument: sig::spmm<fp BATCHLAS_COMMA F>.
#define BATCHLAS_COMMA ,

// Explicitly instantiate a function template from a signature alias.
//
//   BATCHLAS_INSTANTIATE(sig::gemm<float>, gemm, Backend::NETLIB, float)
//     ==> template Event gemm<Backend::NETLIB, float>(Queue&, ...);
//
// SIG must be a *function type* (see the `sig` namespaces next to each public
// declaration in include/batchlas/blas/functions/). Naming the type explicitly, rather
// than deducing it, is what makes this work: every public entry point is an
// overload set -- the MatrixView primary plus an inline Matrix-taking forwarder
// with an identical template parameter list -- so the tempting
//
//   template decltype(FN<Args...>) FN<Args...>;
//
// is ill-formed ("reference to overloaded function could not be resolved").
// Supplying the type disambiguates it.
//
// Because the alias lives beside the declaration, a signature change is now a
// single header edit rather than one edit per backend: gemm's signature alone
// was restated verbatim in netlib_lapack.cc, cublas.cc, rocblas.cc and mkl.cc.
//
// Note that function types cannot carry default arguments -- write the alias
// with every parameter spelled out and no `= default` clauses.
#define BATCHLAS_INSTANTIATE(SIG, FN, ...) template SIG FN<__VA_ARGS__>;

// Name an op once instead of five times.
//
// The signature alias and the function it describes share a spelling, so a
// single op token drives both halves of BATCHLAS_INSTANTIATE:
//
//   BATCHLAS_INSTANTIATE_OP(Backend::CUDA, (float), potrf)
//     ==> template sig::potrf<float> potrf<Backend::CUDA, float>;
//
// `fp` arrives parenthesised so these compose directly with the
// BATCHLAS_FOR_EACH_*_TYPE_1 drivers below, which hand out (float), (double),
// (std::complex<float>), (std::complex<double>); BATCHLAS_UNPAREN strips the
// parentheses again on both uses. That is what replaces the old per-TU
// `#define X_INSTANTIATE` / aggregate-list-line / `#undef` triple: a backend TU
// now writes one line per instantiated op and lets the driver supply the type.
//
// _BACKEND_OP is for the `backend::`-qualified vendor entry points
// (syev_vendor, gesvd_vendor, ormqr_vendor, ...) whose aliases still live in
// `sig`, not `backend::sig`. _FORMAT_OP carries the second template argument
// that the sparse ops (spmm) take, using the comma escape above.
#define BATCHLAS_INSTANTIATE_OP(B, fp, OP) \
    BATCHLAS_INSTANTIATE(sig::OP<BATCHLAS_UNPAREN fp>, OP, B, BATCHLAS_UNPAREN fp)

#define BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, OP) \
    BATCHLAS_INSTANTIATE(sig::OP<BATCHLAS_UNPAREN fp>, backend::OP, B, BATCHLAS_UNPAREN fp)

#define BATCHLAS_INSTANTIATE_FORMAT_OP(B, fp, F, OP) \
    BATCHLAS_INSTANTIATE(sig::OP<BATCHLAS_UNPAREN fp BATCHLAS_COMMA F>, OP, B, BATCHLAS_UNPAREN fp, F)

#define BATCHLAS_FOR_EACH_REAL_TYPE(INVOKE) \
    INVOKE((float)) \
    INVOKE((double))

#define BATCHLAS_FOR_EACH_REAL_TYPE_1(INVOKE, arg1) \
    INVOKE(arg1, (float)) \
    INVOKE(arg1, (double))

#define BATCHLAS_FOR_EACH_SCALAR_TYPE(INVOKE) \
    BATCHLAS_FOR_EACH_REAL_TYPE(INVOKE) \
    INVOKE((std::complex<float>)) \
    INVOKE((std::complex<double>))

// Several ops have a narrower scalar domain than "all four": symm/syrk/syr2k are
// real-only and hemm/herk/her2k are complex-only, so each backend needs a driver
// for each of the three domains rather than one blanket loop.
#define BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(INVOKE, arg1) \
    INVOKE(arg1, (std::complex<float>)) \
    INVOKE(arg1, (std::complex<double>))

#define BATCHLAS_FOR_EACH_SCALAR_TYPE_1(INVOKE, arg1) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(INVOKE, arg1) \
    BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(INVOKE, arg1)

#define BATCHLAS_FOR_EACH_MATRIX_FORMAT_1(INVOKE, arg1) \
    INVOKE(arg1, MatrixFormat::Dense) \
    INVOKE(arg1, MatrixFormat::CSR)

#define BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(INVOKE, arg1, arg2) \
    INVOKE(arg1, arg2, MatrixFormat::Dense) \
    INVOKE(arg1, arg2, MatrixFormat::CSR)

// The backend member of the FOR_EACH family.
//
// Every backend-parameterised entry point has to be instantiated once per
// backend that this build actually compiled, so each .cc carried three
// hand-written `#if BATCHLAS_HAS_*_BACKEND` blocks -- plus, usually, a one-off
// `X_INSTANTIATE_FOR_BACKEND` binder macro whose only job was to bridge this
// loop and the scalar-type loop above. That is the same eleven lines in more
// than forty translation units.
//
// The arms are pre-guarded rather than tested inside the loop because the
// preprocessor cannot emit an `#if` from a macro expansion: a backend that was
// not compiled has to expand to nothing at all. backend_config.h already ships
// BATCHLAS_IF_CUDA / _ROCM / _MKL in exactly that shape; only the host arm was
// missing, so it is supplied here.
//
// Deliberately absent: MKL. The files that carry an MKL arm (ritz_values.cc,
// symm.cc, syrk.cc, syr2k.cc, trmm.cc) do not instantiate the same set as the
// {CUDA, ROCM, HOST} triple, and steqr_legacy.cc has no ROCM arm; folding any of
// them into a blanket loop would silently add or drop exported symbols. Those
// files keep their hand-written blocks.
#include <batchlas/backend_config.h>

#if BATCHLAS_HAS_HOST_BACKEND
  #define BATCHLAS_IF_HOST(x) x
#else
  #define BATCHLAS_IF_HOST(x)
#endif

// INVOKE is called once per compiled backend with the Backend enumerator last,
// matching the argument order of the type loops above (fixed arguments first).
#define BATCHLAS_FOR_EACH_ENABLED_BACKEND(INVOKE) \
    BATCHLAS_IF_CUDA(INVOKE(Backend::CUDA)) \
    BATCHLAS_IF_ROCM(INVOKE(Backend::ROCM)) \
    BATCHLAS_IF_HOST(INVOKE(Backend::NETLIB))

#define BATCHLAS_FOR_EACH_ENABLED_BACKEND_1(INVOKE, arg1) \
    BATCHLAS_IF_CUDA(INVOKE(arg1, Backend::CUDA)) \
    BATCHLAS_IF_ROCM(INVOKE(arg1, Backend::ROCM)) \
    BATCHLAS_IF_HOST(INVOKE(arg1, Backend::NETLIB))

// Combined backend x scalar-type drivers, so the per-file binder disappears too.
// LEAF is a two-argument `LEAF(back, fp)` instantiation macro that takes its type
// parenthesised, i.e. one that spells the type as `BATCHLAS_UNPAREN fp`.
//
//   BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEBRD_BLOCKED_INSTANTIATE)
//
// replaces the binder, the three #if/#endif pairs and the three invocations.
#define BATCHLAS_TYPES_ON_BACKEND_REAL_(LEAF, back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(LEAF, back)

#define BATCHLAS_TYPES_ON_BACKEND_SCALAR_(LEAF, back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(LEAF, back)

#define BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(LEAF) \
    BATCHLAS_FOR_EACH_ENABLED_BACKEND_1(BATCHLAS_TYPES_ON_BACKEND_REAL_, LEAF)

#define BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(LEAF) \
    BATCHLAS_FOR_EACH_ENABLED_BACKEND_1(BATCHLAS_TYPES_ON_BACKEND_SCALAR_, LEAF)

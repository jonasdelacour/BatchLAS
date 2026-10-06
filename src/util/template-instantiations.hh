#pragma once

#define BATCHLAS_UNPAREN(...) __VA_ARGS__

// A comma that survives macro splitting: sig::spmm<fp BATCHLAS_COMMA F>.
#define BATCHLAS_COMMA ,

// BATCHLAS_INSTANTIATE(sig::gemm<float>, gemm, Backend::NETLIB, float)
//   ==> template Event gemm<Backend::NETLIB, float>(Queue&, ...);
// SIG must be a function type with no default arguments (entry points are overload sets).
// NO BATCHLAS_API here, deliberately. evidence: docs/design/runtime-internals.md#runtime-internals-explicit-instantiation-macros
#define BATCHLAS_INSTANTIATE(SIG, FN, ...) template SIG FN<__VA_ARGS__>;

// `fp` arrives parenthesised, as the FOR_EACH_*_TYPE_1 drivers hand it out.
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

// Arms are pre-guarded (a macro cannot emit #if). MKL is deliberately absent: its TUs
// instantiate a different set, and a blanket loop would add or drop exported symbols.
#include <batchlas/backend_config.h>

#if BATCHLAS_HAS_HOST_BACKEND
  #define BATCHLAS_IF_HOST(x) x
#else
  #define BATCHLAS_IF_HOST(x)
#endif

// INVOKE gets the Backend enumerator LAST, matching the type loops above.
#define BATCHLAS_FOR_EACH_ENABLED_BACKEND(INVOKE) \
    BATCHLAS_IF_CUDA(INVOKE(Backend::CUDA)) \
    BATCHLAS_IF_ROCM(INVOKE(Backend::ROCM)) \
    BATCHLAS_IF_HOST(INVOKE(Backend::NETLIB))

#define BATCHLAS_FOR_EACH_ENABLED_BACKEND_1(INVOKE, arg1) \
    BATCHLAS_IF_CUDA(INVOKE(arg1, Backend::CUDA)) \
    BATCHLAS_IF_ROCM(INVOKE(arg1, Backend::ROCM)) \
    BATCHLAS_IF_HOST(INVOKE(arg1, Backend::NETLIB))

// LEAF is `LEAF(back, fp)` with fp parenthesised:
//   BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEBRD_BLOCKED_INSTANTIATE)
#define BATCHLAS_TYPES_ON_BACKEND_REAL_(LEAF, back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(LEAF, back)

#define BATCHLAS_TYPES_ON_BACKEND_SCALAR_(LEAF, back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(LEAF, back)

#define BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(LEAF) \
    BATCHLAS_FOR_EACH_ENABLED_BACKEND_1(BATCHLAS_TYPES_ON_BACKEND_REAL_, LEAF)

#define BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(LEAF) \
    BATCHLAS_FOR_EACH_ENABLED_BACKEND_1(BATCHLAS_TYPES_ON_BACKEND_SCALAR_, LEAF)

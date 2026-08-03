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
// declaration in include/blas/functions/). Naming the type explicitly, rather
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

// The class-template equivalent, for the handful of `template struct X<...>;`
// instantiations (e.g. backend_handle_instantiations.cc).
#define BATCHLAS_INSTANTIATE_CLASS(C, ...) template struct C<__VA_ARGS__>;

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

#define BATCHLAS_FOR_EACH_SCALAR_TYPE_1(INVOKE, arg1) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(INVOKE, arg1) \
    INVOKE(arg1, (std::complex<float>)) \
    INVOKE(arg1, (std::complex<double>))

#define BATCHLAS_FOR_EACH_MATRIX_FORMAT_1(INVOKE, arg1) \
    INVOKE(arg1, MatrixFormat::Dense) \
    INVOKE(arg1, MatrixFormat::CSR)

#define BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(INVOKE, arg1, arg2) \
    INVOKE(arg1, arg2, MatrixFormat::Dense) \
    INVOKE(arg1, arg2, MatrixFormat::CSR)
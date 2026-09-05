#pragma once

// Is a vendor implementation of this op COMPILED IN for this backend?
//
// This is the fact the whole work package turns on, and until S5 it could not
// be asked at all: the public entry point was defined inside the vendor TU, so
// "no vendor" and "no entry point" were the same condition and the answer was
// always trivially yes. Now they are separate, and the facade needs to know.
//
// It is a per-LIBRARY question, not a per-device-family one, and the mapping is
// not uniform: on NVIDIA, geqrf/getrf/ormqr and friends come from cublas.cc
// while potrf and syev come from cusolver.cc, whereas on AMD all of them come
// from rocsolver.cc. Hence one predicate per group rather than one per backend.
//
// WHY THIS RATHER THAN THE SPEC'S src/dispatch/absent/*.cc STUBS. The spec
// proposes seven stub TUs defining a throwing backend::<op>_vendor for every
// absent library. That works, but it restates all 26 vendor signatures a second
// time -- and S5b's bugs were precisely signature divergence between restated
// copies. An `if constexpr` in the facade is the same gate with no signature
// duplicated anywhere: the vendor call is not compiled at all when the library
// is absent, so there is no symbol to satisfy.

#include <batchlas/backend_config.h>

#include <batchlas/blas/enums.hh>

namespace batchlas::dispatch {

// The netlib pair is always tested together -- netlib_lapack.cc calls both
// LAPACKE and CBLAS and is compiled only when both were found.
inline constexpr bool kHasNetlib = BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS;

// gemm/gemv/trsm/trmm/symm/syrk/syr2k/hemm/herk/her2k
template <Backend B>
inline constexpr bool level3_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCBLAS) :
    B == Backend::NETLIB ? kHasNetlib : false;

// geqrf/orgqr/getrf/getrs/getri/ormqr -- cuBLAS on NVIDIA, rocSOLVER on AMD.
template <Backend B>
inline constexpr bool factorization_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS)    :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSOLVER) :
    B == Backend::NETLIB ? kHasNetlib : false;

// potrf/syev -- cuSOLVER on NVIDIA.
template <Backend B>
inline constexpr bool solver_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUSOLVER)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSOLVER) :
    B == Backend::NETLIB ? kHasNetlib : false;

// spmm
template <Backend B>
inline constexpr bool sparse_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUSPARSE)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSPARSE) :
    B == Backend::NETLIB ? kHasNetlib : false;

// The library name to quote in a diagnostic when the answer is no.
template <Backend B>
inline constexpr const char* kLevel3Library =
    B == Backend::CUDA ? "cuBLAS" : B == Backend::ROCM ? "rocBLAS" : "netlib CBLAS/LAPACKE";
template <Backend B>
inline constexpr const char* kFactorizationLibrary =
    B == Backend::CUDA ? "cuBLAS" : B == Backend::ROCM ? "rocSOLVER" : "netlib CBLAS/LAPACKE";
template <Backend B>
inline constexpr const char* kSolverLibrary =
    B == Backend::CUDA ? "cuSOLVER" : B == Backend::ROCM ? "rocSOLVER" : "netlib CBLAS/LAPACKE";
template <Backend B>
inline constexpr const char* kSparseLibrary =
    B == Backend::CUDA ? "cuSPARSE" : B == Backend::ROCM ? "rocSPARSE" : "netlib CBLAS/LAPACKE";

} // namespace batchlas::dispatch

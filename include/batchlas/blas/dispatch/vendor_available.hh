#pragma once

/// @file
/// @brief Is a vendor implementation of an op COMPILED IN for this backend?
///
/// A per-LIBRARY question, not a per-device-family one, and the mapping is not
/// uniform, hence one predicate per op group. The facade tests it with
/// `if constexpr`, so an absent library's vendor call is not compiled at all.
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#the-vendor-gate

#include <batchlas/backend_config.h>

#include <batchlas/blas/enums.hh>

namespace batchlas::dispatch {

/// @addtogroup dispatch
/// @{

/// @brief Host netlib LAPACKE and CBLAS are both present.
///
/// Always tested together: netlib_lapack.cc calls both and is compiled only when both were found.
inline constexpr bool kHasNetlib = BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS;

/// @brief Vendor BLAS present for gemm, gemv, trsm, trmm, symm, syrk, syr2k, hemm, herk, her2k.
template <Backend B>
inline constexpr bool level3_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCBLAS) :
    B == Backend::NETLIB ? kHasNetlib : false;

/// @brief Vendor present for geqrf, orgqr, getrf, getrs, getri, ormqr.
///
/// NOT one library on NVIDIA: getrf and getri are cuBLAS batched calls, while
/// geqrf, orgqr, ormqr and getrs's batch <= 1 arm are cuSOLVER, so the CUDA
/// answer needs both. A finer per-op split is open debt.
// evidence: docs/design/vendor-independence.md#the-vendor-gate-history-of-the-per-library-predicates
template <Backend B>
inline constexpr bool factorization_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS) && bool(BATCHLAS_HAS_CUSOLVER) :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSOLVER) :
    B == Backend::NETLIB ? kHasNetlib : false;

/// @brief Vendor solver present for potrf and syev (cuSOLVER on NVIDIA).
template <Backend B>
inline constexpr bool solver_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUSOLVER)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSOLVER) :
    B == Backend::NETLIB ? kHasNetlib : false;

/// @brief Vendor sparse library present for spmm.
template <Backend B>
inline constexpr bool sparse_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUSPARSE)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSPARSE) :
    B == Backend::NETLIB ? kHasNetlib : false;

/// @brief Library name a diagnostic quotes when level3_vendor_available is false.
template <Backend B>
inline constexpr const char* kLevel3Library =
    B == Backend::CUDA ? "cuBLAS" : B == Backend::ROCM ? "rocBLAS" : "netlib CBLAS/LAPACKE";
/// @brief Library name a diagnostic quotes when factorization_vendor_available is false.
template <Backend B>
inline constexpr const char* kFactorizationLibrary =
    B == Backend::CUDA ? "cuBLAS and cuSOLVER" : B == Backend::ROCM ? "rocSOLVER"
                                               : "netlib CBLAS/LAPACKE";
/// @brief Library name a diagnostic quotes when solver_vendor_available is false.
template <Backend B>
inline constexpr const char* kSolverLibrary =
    B == Backend::CUDA ? "cuSOLVER" : B == Backend::ROCM ? "rocSOLVER" : "netlib CBLAS/LAPACKE";
/// @brief Library name a diagnostic quotes when sparse_vendor_available is false.
template <Backend B>
inline constexpr const char* kSparseLibrary =
    B == Backend::CUDA ? "cuSPARSE" : B == Backend::ROCM ? "rocSPARSE" : "netlib CBLAS/LAPACKE";

/// @}

} // namespace batchlas::dispatch

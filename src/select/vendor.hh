#pragma once

/// @file
/// @brief Link facts for `if constexpr` gates: is an op's vendor library compiled in for a
/// backend, and is the native level-3 tile kernel linked. @ingroup selection

#include "coverage.hh"

#include <batchlas/backend_config.h>
#include <batchlas/blas/enums.hh>
#include <batchlas/no_route.hh>

#include <cstdint>
#include <string>
#include <type_traits>

namespace batchlas::select {
/// @addtogroup selection
/// @{

inline constexpr bool kHasNetlib = BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS;

template <Backend B>  // gemm gemv trsm trmm symm syrk syr2k hemm herk her2k
inline constexpr bool level3_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCBLAS) :
    B == Backend::NETLIB ? kHasNetlib : false;

/// geqrf/orgqr/getrf/getrs/getri/ormqr. On NVIDIA the group spans cuBLAS (getrf, getri)
/// and cuSOLVER (geqrf, orgqr, ormqr, getrs), so it needs both.
template <Backend B>
inline constexpr bool factorization_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS) && bool(BATCHLAS_HAS_CUSOLVER) :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSOLVER) :
    B == Backend::NETLIB ? kHasNetlib : false;

template <Backend B>  // potrf syev gesvd: cuSOLVER on NVIDIA
inline constexpr bool solver_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUSOLVER)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSOLVER) :
    B == Backend::NETLIB ? kHasNetlib : false;

template <Backend B>  // spmm
inline constexpr bool sparse_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUSPARSE)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCSPARSE) :
    B == Backend::NETLIB ? kHasNetlib : false;

/// The library name a diagnostic quotes when the answer is no.
template <Backend B>
inline constexpr const char* kLevel3Library =
    B == Backend::CUDA ? "cuBLAS" : B == Backend::ROCM ? "rocBLAS" : "netlib CBLAS/LAPACKE";
template <Backend B>
inline constexpr const char* kFactorizationLibrary =
    B == Backend::CUDA ? "cuBLAS and cuSOLVER" : B == Backend::ROCM ? "rocSOLVER"
                                               : "netlib CBLAS/LAPACKE";
template <Backend B>
inline constexpr const char* kSolverLibrary =
    B == Backend::CUDA ? "cuSOLVER" : B == Backend::ROCM ? "rocSOLVER" : "netlib CBLAS/LAPACKE";
template <Backend B>
inline constexpr const char* kSparseLibrary =
    B == Backend::CUDA ? "cuSPARSE" : B == Backend::ROCM ? "rocSPARSE" : "netlib CBLAS/LAPACKE";

/// The library group an op's Vendor family calls; an op names exactly one (OpSpec::vendor).
enum class Lib : std::uint8_t { none, level3, factorization, solver, sparse };

/// Is library group @p l compiled in for backend B? Always false for Lib::none.
template <Backend B>
constexpr bool has_library(Lib l) {
    switch (l) {
        case Lib::level3: return level3_vendor_available<B>;
        case Lib::factorization: return factorization_vendor_available<B>;
        case Lib::solver: return solver_vendor_available<B>;
        case Lib::sparse: return sparse_vendor_available<B>;
        case Lib::none: return false;
    }
    return false;
}

template <Backend B>
constexpr const char* library_name(Lib l) {
    switch (l) {
        case Lib::factorization: return kFactorizationLibrary<B>;
        case Lib::solver: return kSolverLibrary<B>;
        case Lib::sparse: return kSparseLibrary<B>;
        default: return kLevel3Library<B>;
    }
}

// Is a level-3 tile kernel linked for (B, T)? Kept at its pre-flat-selection value (float, or
// cuBLAS) for sytrd_blocked, ortho, ormqr_blocked and coverage: double syrk gram, symm expand and
// every trmm family now run vendor-free too, but widening this moves ortho's and ormqr's routes.
// evidence: docs/design/vendor-independence.md#the-vendor-gate-why-the-tile-route-predicate-is-per-backend-and-scalar
template <Backend B, typename T>
inline constexpr bool level3_tile_route_available =
    B == Backend::CUDA && (std::is_same_v<T, float> || bool(BATCHLAS_HAS_CUBLAS));

/// The single funnel for "nothing can serve this call": records the miss, then throws NoRouteError.
template <typename T>
[[noreturn]] inline void throw_no_vendor_route(Op op, Backend backend, const char* library) {
    coverage::record_miss(op, scalar_kind_of<T>, backend, library);
    throw NoRouteError(op, backend, scalar_kind_of<T>, std::string("built without ") + library);
}

/// @}
} // namespace batchlas::select

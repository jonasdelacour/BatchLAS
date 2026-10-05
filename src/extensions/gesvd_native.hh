#pragma once

// The gesvd drivers' own shape ceilings, sycl-free, read by gesvd's can_run and the drivers.

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_gesvd {

// Largest max(m, n) gesvdj_cta accepts: local memory for the V tile (gesvdj_cta.cc).
template <typename T>
constexpr std::int64_t gesvd_jacobi_max_dim(bool want_vectors) {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return want_vectors ? 32 : 64;
    } else {
        return 64;
    }
}

// Largest max(m, n) gesvd_cta accepts: one sub-group per matrix.
inline constexpr std::int64_t kGesvdCtaMaxDim = 32;

}  // namespace batchlas::sycl_gesvd

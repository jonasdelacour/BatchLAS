#pragma once

// The gesvd drivers' own shape ceilings, sycl-free, read by gesvd's can_run and the drivers.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_gesvd {

// gesvdj_cta's smallest work-group at tile capacity C: 32 / P problems (P = min(C, 32)) plus
// the pivot-pair table, as gesvdj_cta.cc allocates. evidence: docs/design/gesvd.md#gesvdj_cta-local-memory-budget-formula
template <typename T>
constexpr std::size_t gesvdj_slm_bytes(std::size_t C, bool want_vectors) {
    using Real = std::conditional_t<std::is_same_v<T, std::complex<float>>, float,
                                    std::conditional_t<std::is_same_v<T, std::complex<double>>, double, T>>;
    constexpr bool kComplex = !std::is_same_v<T, Real>;
    const std::size_t P = C < 32 ? C : 32;
    const std::size_t rot = C / 2 > 0 ? C / 2 : 1;
    const std::size_t tile = (C + 1) * C;
    const std::size_t per_prob = (want_vectors ? 2 : 1) * tile * sizeof(T) + C * sizeof(Real) +
                                 2 * rot * sizeof(Real) + (kComplex ? rot * sizeof(T) : 0) + C * sizeof(std::int16_t);
    return (32 / P) * per_prob + (C - 1) * rot * sizeof(std::int16_t);
}

// Largest max(m, n) gesvdj_cta accepts: the widest C whose smallest work-group fits the budget.
template <typename T>
constexpr std::int64_t gesvd_jacobi_max_dim(bool want_vectors, std::size_t slm_budget_bytes) {
    for (std::size_t C = 64; C >= 4; C /= 2)
        if (gesvdj_slm_bytes<T>(C, want_vectors) <= slm_budget_bytes) return static_cast<std::int64_t>(C);
    return 0;
}

// Largest max(m, n) gesvd_cta accepts: one sub-group per matrix.
inline constexpr std::int64_t kGesvdCtaMaxDim = 32;

}  // namespace batchlas::sycl_gesvd

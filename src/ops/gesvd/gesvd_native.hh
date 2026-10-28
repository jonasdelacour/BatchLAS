// SPDX-License-Identifier: MIT
#pragma once

// The gesvd drivers' own shape ceilings and mask predicate, read by gesvd's can_run and the drivers.

#include <batchlas/settings.hh>

#include <src/ops/ormqr/wy_spelling.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
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

// Largest max(m, n) of gesvd_cta (one sub-group per matrix) and of cuSOLVER's gesvdjBatched.
inline constexpr std::int64_t kGesvdCtaMaxDim = 32;
inline constexpr std::int64_t kGesvdVendorCudaMaxDim = 32;

// The Blocked path's bidiagonal solver, BATCHLAS_GESVD_BIDIAG = bdsdc (default) | normal | bdsqr.
// evidence: docs/perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver
enum class GesvdBidiagSolver { NormalEquations, Bdsdc, Bdsqr };

inline GesvdBidiagSolver gesvd_bidiag_solver() {
    const char* v = batchlas::settings().selection.gesvd_bidiag.get();
    if (v == nullptr) return GesvdBidiagSolver::Bdsdc;
    const std::string s(v);
    if (s == "bdsqr") return GesvdBidiagSolver::Bdsqr;
    if (s == "normal") return GesvdBidiagSolver::NormalEquations;
    return GesvdBidiagSolver::Bdsdc;
}

// gesvd_blocked's general path honours a mask only where every child it reaches does: not stedc
// (bdsdc, normal), not trmm (ormqr_blocked's Left update; ormqx_cta takes square n <= 32), not
// NETLIB's host gemm. Canonical jobs. mask: lift after stedc (W3a), trmm (W1b)
template <Backend B, typename T>
bool gesvd_blocked_honours_mask(std::int64_t m, std::int64_t n, bool want_u, bool want_vh) {
    const bool left_apply = m >= n ? want_u : want_vh;  // m < n solves A^H: its U is A's V^H
    const bool left_cta = m == n && m <= 32;
    return select::native_gpu_backend<B> && gesvd_bidiag_solver() == GesvdBidiagSolver::Bdsqr &&
           !(left_apply && !left_cta && ops::ormqr::wy_reaches_trmm<B, T>(Side::Left));
}

}  // namespace batchlas::sycl_gesvd

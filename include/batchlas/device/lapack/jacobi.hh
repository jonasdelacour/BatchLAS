// SPDX-License-Identifier: MIT
#pragma once
// Shared pieces of the Jacobi routines (syevj two-sided, gesvj one-sided): pivot schedule,
// rotation threshold and record, the one-sided norm recurrence. sym_schur2 and the rotation
// rounds wait in src/kernels/common/dla_v2_jacobi.hh for core inv_sqrt and the unitary rot.

#include <cstdint>
#include <limits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>

namespace batchlas::device::inline v2::jacobi_detail {

struct PivotPair {
    int p, q;
};
// Round-robin ("circle method") pairing for even m: round t in [0, m-2] has m/2 disjoint pairs
// (k in [0, m/2)), the m-1 rounds cover every pair once; index 0 is fixed, the rest rotate.
// Weakly equivalent to cyclic-by-rows (Hari & Begovic Kovac 2017). Returns p < q.
BATCHLAS_DLA_INLINE PivotPair round_robin_pair(int m, int t, int k) {
    const int ring = m - 1;
    int p, q;
    if (k == 0) {
        p = 0;
        q = (t % ring) + 1;
    } else {
        p = ((t + k) % ring) + 1;
        q = (((t - k) % ring + ring) % ring) + 1;
    }
    if (p > q) {
        const int tmp = p;
        p = q;
        q = tmp;
    }
    return {p, q};
}
// Both indices (< 128) in one 16-bit slot.
BATCHLAS_DLA_INLINE std::int16_t pack_pair(PivotPair pq) { return static_cast<std::int16_t>(pq.p | (pq.q << 8)); }
BATCHLAS_DLA_INLINE PivotPair unpack_pair(std::int16_t packed) {
    const int pq = static_cast<int>(packed);
    return {pq & 0xFF, (pq >> 8) & 0xFF};
}

// ---- the 2x2 Hermitian Jacobi rotation ------------------------------------------------------
template <class R> struct JacobiThreshold {
    R tol;      // rotate only when |a_pq| > tol * sqrt(|a_pp a_qq|) (relative test)
    R tiny;     // and |a_pq| > tiny: denormal / zero off-diagonals count as converged
    R tau_big;  // |tau| above this uses t = 1 / (2 tau): tau * tau would overflow
    // tol = tol_multiplier * n * eps, tiny = the smallest normal, tau_big = 1 / sqrt(eps).
    static JacobiThreshold relative(R tol_multiplier, int n) {
        return {tol_multiplier * static_cast<R>(n) * std::numeric_limits<R>::epsilon(), std::numeric_limits<R>::min(),
                R(1) / sycl::sqrt(std::numeric_limits<R>::epsilon())};
    }
};

// U = [c s; -d s  d c] with d a unit phase; t = s / c, g the real off-diagonal after the phase.
template <class T> struct JacobiRotation {
    real_t<T> c, s;
    T d;
    real_t<T> t, g;
    bool active;  // false: U is the identity (below threshold, or s rounded to 0)
};
template <class T> BATCHLAS_DLA_INLINE JacobiRotation<T> identity_rotation() {
    using R = real_t<T>;
    return {R(1), R(0), T(1), R(0), R(0), false};
}

// One-sided Jacobi: squared column norms after rotation r from the norms before it (LAPACK
// ?GESVJ's SVA recurrence), both clamped at 0. Meaningful only when r.active.
template <class R> struct NormPair {
    R p, q;
};
template <class T> BATCHLAS_DLA_INLINE NormPair<real_t<T>> rotated_norms2(const JacobiRotation<T>& r, real_t<T> app, real_t<T> aqq) {
    using R = real_t<T>;
    return {sycl::fmax(app - r.t * r.g, R(0)), sycl::fmax(aqq + r.t * r.g, R(0))};
}

}  // namespace batchlas::device::jacobi_detail

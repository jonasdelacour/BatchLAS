// SPDX-License-Identifier: MIT
#pragma once
// syev-driver-private device LA v2 glue (flat 2-D ids, the reflector repack, dgmm, the sign
// similarity); the syev routine's own steps are device/lapack/syev.hh's, shared ones core's.
// Strict ports: arith::Native by default, the hand code's expression shapes; Uplo stays runtime
// where the legacy kernels branch on it.

#include <cstdint>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/lapack/syev.hh>

namespace batchlas::syev_v2 {

namespace v2 = batchlas::device;

// ---- executors and flat 2-D glue ---------------------------------------------------------------
// One element (r, c) as an executor (Grid2D's row()/col() contract): core 2-D ops visit only it.
struct ElemAt {
    int r, c;
    BATCHLAS_DLA_INLINE int row() const { return r; }
    BATCHLAS_DLA_INLINE int col() const { return c; }
};

// Flat id over batch x n x n (column-major per item): the item first, so a masked item leaves early.
struct RowCol {
    std::int32_t r, c;
};
BATCHLAS_DLA_INLINE std::int32_t batched_item(std::int64_t idx, std::int32_t n) {
    return static_cast<std::int32_t>(idx / (static_cast<std::int64_t>(n) * n));
}
BATCHLAS_DLA_INLINE RowCol batched_rc(std::int64_t idx, std::int32_t b, std::int32_t n) {
    const std::int64_t rem = idx - static_cast<std::int64_t>(b) * n * n;
    return {static_cast<std::int32_t>(rem % n), static_cast<std::int32_t>(rem / n)};
}

// ---- element kernels -----------------------------------------------------------------------------
// C(i, j) := d(i) * B(i, j); a real B is lifted to C's scalar type. C may alias B.
template <class Ops = v2::arith::Native, class Dv, class Bm, class Cm>
BATCHLAS_DLA_INLINE void dgmm_left_at(const Dv& d, const Bm& B, const Cm& C, int i, int j) {
    using T = std::remove_cvref_t<decltype(C(0, 0))>;
    using AR = v2::arith_t<Ops, T>;
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(B(i, j))>, T>) C(i, j) = AR::mul(d(i), B(i, j));
    else C(i, j) = AR::mul(d(i), v2::lift<T>(B(i, j)));
}

// Element (r, c) of the QR/QL layout ormqx expects, from a SYTRD output A (n x n):
// Uplo::Lower -> QR reflectors in columns 1..n-1 (AQ(r, c) = A(r, c-1), r > c);
// Uplo::Upper -> QL reflectors in columns 0..n-2 (AQ(r, c) = A(r, c+1), r < c); 0 elsewhere.
// With a TAU / TAUQ pair (not NoTau), row 0 also writes TAUQ(c): tau shifted the same way.
struct NoTau {};
template <class Am, class AQm, class Tv = NoTau, class TQv = NoTau>
BATCHLAS_DLA_INLINE void sytrd_reflector_pack_at(const Am& A, const AQm& AQ, Uplo uplo, int n, int r, int c, const Tv& tau = {},
                                                 const TQv& tauq = {}) {
    using T = std::remove_cvref_t<decltype(AQ(0, 0))>;
    constexpr bool kTau = !std::is_same_v<Tv, NoTau>;
    AQ(r, c) = T(0);
    if (uplo == Uplo::Lower) {
        if (c >= 1) {
            const int i = c - 1;
            if (r >= (c + 1) && r < n) AQ(r, c) = A(r, i);
        }
        if constexpr (kTau) {
            if (r == 0) tauq(c) = (c >= 1 && (c - 1) < (n - 1)) ? tau(c - 1) : T(0);
        }
    } else {
        if (c <= n - 2) {
            const int k = c + 1;
            if (r < c) AQ(r, c) = A(r, k);
        }
        if constexpr (kTau) {
            if (r == 0) tauq(c) = (c < (n - 1)) ? tau(c) : T(0);
        }
    }
}
// TAUQ(j) of the same repack as a separate pass.
template <class Tv, class TQv> BATCHLAS_DLA_INLINE void sytrd_tau_pack_at(const Tv& tau, const TQv& tauq, Uplo uplo, int n, int j) {
    using T = std::remove_cvref_t<decltype(tauq(0))>;
    if (uplo == Uplo::Lower) {
        if (j == 0) tauq(j) = T(0);
        else tauq(j) = (j - 1 < (n - 1)) ? tau(j - 1) : T(0);
    } else {
        if (j < (n - 1)) tauq(j) = tau(j);
        else tauq(j) = T(0);
    }
}

// ---- tridiagonal similarities ------------------------------------------------------------------
// Real symmetric tridiagonal: e := |e| by the +-1 similarity S (S(0) = 1, S(i+1) = S(i) sign(e_i)).
template <class Ev, class Sv> BATCHLAS_DLA_INLINE void sign_similarity(const v2::Thread&, int n, const Ev& e, const Sv& s) {
    using T = std::remove_cvref_t<decltype(e(0))>;
    s(0) = T(1);
    for (int i = 0; i < n - 1; ++i) {
        const T ei = e(i);
        const T sgn = (ei >= T(0)) ? T(1) : T(-1);
        s(i + 1) = s(i) * sgn;
        e(i) = sycl::fabs(ei);
    }
}

// The Hermitian load, the phase similarity and replay, and the column-owner xORG2L / xORM2L
// are the public syev's (device/lapack/syev.hh); the host drivers name them from there.
using v2::syev_detail::chunk_barrier;
using v2::syev_detail::dgmm_lift;
using v2::syev_detail::herm_load;
using v2::syev_detail::larf_col;
using v2::syev_detail::larf_unit_col;
using v2::syev_detail::laset_col;
using v2::syev_detail::org2l;
using v2::syev_detail::orm2l;
using v2::syev_detail::phase_replay;
using v2::syev_detail::phase_similarity;

}  // namespace batchlas::syev_v2

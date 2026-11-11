// SPDX-License-Identifier: MIT
#pragma once
// Shared pieces of the Jacobi routines (syevj two-sided, gesvj one-sided): pivot schedule,
// rotation threshold and record, the one-sided norm recurrence, the phased plane rotation
// U = [c s; -d s  d c] and the rounds of disjoint rotations from the right (A U) or the left
// (U^H A). sym_schur2 waits in src/kernels/common/dla_v2_jacobi.hh for core inv_sqrt.
//
// Ops = arith::Native (default) writes every update with T's own operators, the legacy kernels'
// expressions kept whole (bit identity; -ffp-contract=on fuses within one expression; for
// std::complex the Annex-G multiply). arith::Explicit: Arith<T> with the real (c, s), complex only.

#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>

namespace batchlas::device::detail {

template <class Ops> inline constexpr bool is_native_v = std::is_same_v<Ops, arith::Native>;

struct PivotPair {
    int p, q;
};
// Round-robin ("circle method") pairing for even m: round t in [0, m-2] has m/2 disjoint pairs
// (k in [0, m/2)), the m-1 rounds cover every pair once; index 0 is fixed, the rest rotate.
// Weakly equivalent to cyclic-by-rows (Hari & Begovic Kovac 2017). Returns p < q.
BATCHLAS_DEVICE_INLINE PivotPair round_robin_pair(int m, int t, int k) {
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
BATCHLAS_DEVICE_INLINE std::int16_t pack_pair(PivotPair pq) { return static_cast<std::int16_t>(pq.p | (pq.q << 8)); }
BATCHLAS_DEVICE_INLINE PivotPair unpack_pair(std::int16_t packed) {
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
template <class T> BATCHLAS_DEVICE_INLINE JacobiRotation<T> identity_rotation() {
    using R = real_t<T>;
    return {R(1), R(0), T(1), R(0), R(0), false};
}

// One-sided Jacobi: squared column norms after rotation r from the norms before it (LAPACK
// ?GESVJ's SVA recurrence), both clamped at 0. Meaningful only when r.active.
template <class R> struct NormPair {
    R p, q;
};
template <class T> BATCHLAS_DEVICE_INLINE NormPair<real_t<T>> rotated_norms2(const JacobiRotation<T>& r, real_t<T> app, real_t<T> aqq) {
    using R = real_t<T>;
    return {sycl::fmax(app - r.t * r.g, R(0)), sycl::fmax(aqq + r.t * r.g, R(0))};
}

// ---- applying a rotation -------------------------------------------------------------------
// Not a plain rot: U = [c s; -d s  d c] scales y by the phase in both outputs, and folding d into
// y first (then a real rot) rounds differently.
template <class T> struct Rot2 {
    T u11, u12, u21, u22;
};
template <class T> struct Pair2 {
    T x, y;
};

// The entries of U (Conj: of conj(U), for rot_left).
template <bool Conj, class Ops = arith::Native, class T> BATCHLAS_DEVICE_INLINE Rot2<T> rot2(real_t<T> c, real_t<T> s, T d) {
    using A = Arith<T>;
    if constexpr (!A::is_complex) {
        (void)d;
        return {T(c), T(s), T(-s), T(c)};
    } else if constexpr (is_native_v<Ops>) {
        const T dd = Conj ? A::conj(d) : d;
        return {T(c), T(s), -(dd * T(s)), dd * T(c)};
    } else {
        const T dd = Conj ? A::conj(d) : d;
        return {T(c), T(s), A::make(-(A::re(dd) * s), -(A::im(dd) * s)), A::make(A::re(dd) * c, A::im(dd) * c)};
    }
}

// a x + b y with a real and b complex (the explicit form).
template <class T> BATCHLAS_DEVICE_INLINE T rx_plus_by(real_t<T> a, T x, T b, T y) {
    using A = Arith<T>;
    return A::make(a * A::re(x) + A::re(b) * A::re(y) - A::im(b) * A::im(y),
                   a * A::im(x) + A::re(b) * A::im(y) + A::im(b) * A::re(y));
}

// [x y] := [x y] U: row r of a column pair (A(r, p), A(r, q)) under A := A U.
template <class Ops = arith::Native, class T> BATCHLAS_DEVICE_INLINE Pair2<T> rot_right(const Rot2<T>& u, T x, T y) {
    if constexpr (is_native_v<Ops> || !Arith<T>::is_complex) {
        return {x * u.u11 + y * u.u21, x * u.u12 + y * u.u22};
    } else {
        using A = Arith<T>;
        return {rx_plus_by(A::re(u.u11), x, u.u21, y), rx_plus_by(A::re(u.u12), x, u.u22, y)};
    }
}
// [x; y] := U^H [x; y] for u = rot2<true>(...): column c of a row pair under A := U^H A.
template <class Ops = arith::Native, class T> BATCHLAS_DEVICE_INLINE Pair2<T> rot_left(const Rot2<T>& u, T x, T y) {
    if constexpr (is_native_v<Ops> || !Arith<T>::is_complex) {
        return {u.u11 * x + u.u21 * y, u.u12 * x + u.u22 * y};
    } else {
        using A = Arith<T>;
        return {rx_plus_by(A::re(u.u11), x, u.u21, y), rx_plus_by(A::re(u.u12), x, u.u22, y)};
    }
}
// In place on two elements (both read, then both written).
template <class Ops = arith::Native, class T> BATCHLAS_DEVICE_INLINE void rot_right_inplace(const Rot2<T>& u, T& x, T& y) {
    const T a = x;
    const T b = y;
    const Pair2<T> r = rot_right<Ops>(u, a, b);
    x = r.x;
    y = r.y;
}

// One round of disjoint rotations as the caller published it: slot k holds (c, s) in one 2-wide
// element, the phase d (complex T only; None for real T) and the packed pair (p, q). Each slot
// array is an operand read with (k): a Vec over the kernel's local accessor or a pointer.
template <class T, class CS, class D, class PR> struct RotRound {
    using R = real_t<T>;
    CS cs;
    D d;
    PR pairs;

    BATCHLAS_DEVICE_INLINE PivotPair pair(int k) const { return unpack_pair(pairs(k)); }
    BATCHLAS_DEVICE_INLINE sycl::vec<R, 2> coef(int k) const { return cs(k); }
    BATCHLAS_DEVICE_INLINE void store(int k, const JacobiRotation<T>& r) const {
        cs(k) = sycl::vec<R, 2>(r.c, r.s);
        if constexpr (Arith<T>::is_complex) d(k) = r.d;
    }
    template <bool Conj, class Ops = arith::Native> BATCHLAS_DEVICE_INLINE Rot2<T> rot(int k, R c, R s) const {
        if constexpr (Arith<T>::is_complex) return rot2<Conj, Ops, T>(c, s, d(k));
        else return rot2<Conj, Ops, T>(c, s, T(1));
    }
};
template <class T, class CS, class D, class PR> BATCHLAS_DEVICE_INLINE RotRound<T, CS, D, PR> rot_round(CS cs, D d, PR pairs) {
    return {cs, d, pairs};
}

// A_i := A_i U_k for every rotation k < npairs of the round (s == 0 skipped), on row `row` of
// every A_i: the calling item owns that row, and the pairs of a round touch disjoint columns,
// so no barrier is needed between rotations. One pass serves A and its accumulators (Z := Z U).
template <class Ops = arith::Native, class T, class CS, class D, class PR, class N, class... Ms>
BATCHLAS_DEVICE_INLINE void rot_round_right(const RotRound<T, CS, D, PR>& rd, N npairs, int row, const Ms&... ms) {
    using R = real_t<T>;
    for (int k = 0; k < int(npairs); ++k) {
        const sycl::vec<R, 2> cs = rd.coef(k);
        const R c = cs[0];
        const R s = cs[1];
        if (s == R(0)) continue;
        const PivotPair pq = rd.pair(k);
        const Rot2<T> u = rd.template rot<false, Ops>(k, c, s);
        (rot_right_inplace<Ops>(u, ms(row, pq.p), ms(row, pq.q)), ...);
    }
}

// A := U_k^H A for every rotation of the round, on column `col` (owned by the caller): the
// second half of a two-sided Jacobi step on a Hermitian A. The pivot entries are stored as
// their exact values (a_pq = a_qp = 0, real diagonal), not as rounding noise.
template <class Ops = arith::Native, class T, class CS, class D, class PR, class N, class M>
BATCHLAS_DEVICE_INLINE void rot_round_left_hermitian(const RotRound<T, CS, D, PR>& rd, N npairs, int col, const M& a) {
    using R = real_t<T>;
    for (int k = 0; k < int(npairs); ++k) {
        const sycl::vec<R, 2> cs = rd.coef(k);
        const R c = cs[0];
        const R s = cs[1];
        if (s == R(0)) continue;
        const PivotPair pq = rd.pair(k);
        const Rot2<T> u = rd.template rot<true, Ops>(k, c, s);
        const T x = a(pq.p, col);
        const T y = a(pq.q, col);
        Pair2<T> r = rot_left<Ops>(u, x, y);
        if (col == pq.q) {
            r.x = T(0);
            r.y = force_real(r.y);
        } else if (col == pq.p) {
            r.y = T(0);
            r.x = force_real(r.x);
        }
        a(pq.p, col) = r.x;
        a(pq.q, col) = r.y;
    }
}

}  // namespace batchlas::device::detail

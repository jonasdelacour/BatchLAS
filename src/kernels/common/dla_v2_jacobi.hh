// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, Jacobi family (syev_jacobi_cta, gesvdj_cta): the 2x2 Hermitian rotation
// (sym_schur2), rounds of disjoint plane rotations from the right (A U) or the left (U^H A), the
// round-robin pivot schedule, the reduce-scatter behind the one-sided Gram and column-norm
// passes, and per-item fragment ops on Cyclic rows of a static extent. src-side: no other family
// uses them. Shared ops (abs2, modulus, Vec / Mat over a local_accessor, allreduce) come from core.
//
// Ops = arith::Native (default) writes every update with T's own operators, the expressions of
// the kernels these ops replaced (bit identity; for std::complex the Annex-G multiply).
// arith::Explicit is Arith<T> with the rotation's real (c, s): differs only for complex T, an
// algo-mode alternative. -ffp-contract=on fuses within one expression, so the Native bodies keep
// each legacy expression whole.

#include <cstdint>
#include <limits>
#include <type_traits>

#include <batchlas/blas/device/v2/operand.hh>
#include <batchlas/blas/device/v2/scalar.hh>
#include <src/kernels/common/dla_v2_partition.hh>

namespace batchlas::device::v2 {

namespace detail {
// conj of an element reference: the load happens inside, as with v1 device::detail::conj.
template <class T> inline T conj_ref(const T& x) {
    if constexpr (Arith<T>::is_complex) return T(x.real(), -x.imag());
    else return x;
}
template <class Ops> inline constexpr bool is_native_v = std::is_same_v<Ops, arith::Native>;
template <int N> constexpr int ilog2() {
    int r = 0;
    for (int v = N; v > 1; v >>= 1) ++r;
    return r;
}
}  // namespace detail

// x with its imaginary part zeroed (a Hermitian diagonal); identity for real T.
template <class T> BATCHLAS_DLA_INLINE T force_real(T x) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(x), real_t<T>(0));
    else return x;
}

// An operand slot a kernel leaves unnamed (real T's phase, the V tile without vectors), so the
// kernel does not capture its accessor.
struct NoPhase {};

// IEEE max / min reduction operators: a NaN operand loses, as sycl::fmax / fmin.
struct FMax {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmax(a, b); }
};
struct FMin {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmin(a, b); }
};

// Value of lane (lid ^ m) on a Partition / SubGroup; complex moves as two real shuffles.
template <class Ex, class T> BATCHLAS_DLA_INLINE T shuffle_xor(const Ex& ex, T v, std::uint32_t m) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(detail::xor_shuffle(ex.part, A::re(v), m), detail::xor_shuffle(ex.part, A::im(v), m));
    else return detail::xor_shuffle(ex.part, v, m);
}

// Reduce-scatter of K values over the P = Ex::size lanes (K | P, both powers of two): lane l
// returns the sum over all lanes of x[l / (P / K)]. log2(K) halving steps over xor masks
// P/2 ... P/K, then an all-reduce over the low log2(P / K) lane bits. Consumes x.
template <int K, class Ex, class T> BATCHLAS_DLA_INLINE T reduce_scatter(const Ex& ex, Frag<T, K>& x) {
    constexpr int P = Ex::size;
    static_assert(Ex::sub_whole && detail::is_pow2(K) && K <= P, "v2::reduce_scatter: K | P on a partition executor");
    constexpr int kSteps = detail::ilog2<K>();
    constexpr int kAll = detail::ilog2<P>();
    const std::uint32_t lane = static_cast<std::uint32_t>(ex.lid());
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
        const int mask = P >> (st + 1);
        const int half = K >> (st + 1);
        const bool hi = (lane & static_cast<std::uint32_t>(mask)) != 0u;
#pragma unroll
        for (int j = 0; j < half; ++j) {
            // Copy both slots first: `hi ? x[a] : x[b]` is an lvalue, a lane-dependent slot index
            // that turns x into a stack array or compare/select chains (float 8x8 +73% instructions).
            const T lo = x[j];
            const T up = x[j + half];
            const T own = hi ? up : lo;
            const T send = hi ? lo : up;
            x[j] = Arith<T>::add(own, shuffle_xor(ex, send, static_cast<std::uint32_t>(mask)));
        }
    }
#pragma unroll
    for (int st = kSteps; st < kAll; ++st) {
        const std::uint32_t mask = static_cast<std::uint32_t>(P >> (st + 1));
        x[0] = Arith<T>::add(x[0], shuffle_xor(ex, x[0], mask));
    }
    return x[0];
}

// ---- Cyclic rows of a static extent N (N % size == 0): item l owns rows l + k*size ----------
// f(k, row) for the item's N / size rows, k a compile-time Frag slot; no bound check.
// U = unroll::Full is a full unroll pragma, unroll::Default none.
template <int N, class U = unroll::Full, class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned_frag(const Ex& ex, F&& f) {
    static_assert(N % Ex::size == 0, "v2::for_each_owned_frag: N must be a multiple of the executor size");
    constexpr int K = N / Ex::size;
    const int l = ex.lid();
    if constexpr (std::is_same_v<U, unroll::Full>) {
#pragma unroll
        for (int k = 0; k < K; ++k) f(k, l + k * Ex::size);
    } else {
        static_assert(std::is_same_v<U, unroll::Default>, "v2::for_each_owned_frag: unroll::Full or unroll::Default");
        for (int k = 0; k < K; ++k) f(k, l + k * Ex::size);
    }
}

// x[k] = v(l + k*size) for rows < n, else 0.
template <class Ex, class V, class T, int K> BATCHLAS_DLA_INLINE void load_owned_masked(const Ex& ex, const V& v, int n, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r < n) ? static_cast<T>(v(r)) : T(0); });
}
// x = e_j restricted to the item's rows.
template <class Ex, class T, int K> BATCHLAS_DLA_INLINE void set_unit_owned(const Ex& ex, int j, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r == j) ? T(1) : T(0); });
}
// sum_k conj(x[k]) y[k] over the item's fragment (a partial dotc; reduce it with allreduce).
template <class Ops = arith::Native, class T, int K> BATCHLAS_DLA_INLINE T dotc_owned(const Frag<T, K>& x, const Frag<T, K>& y) {
    T acc = T(0);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (detail::is_native_v<Ops>) acc = acc + detail::conj_ref(x[k]) * y[k];
        else arith_t<Ops, T>::mac_conj(acc, x[k], y[k]);
    }
    return acc;
}
// x[k] = u(l + k*size), y[k] = v(l + k*size) and the partial dotc sum_k conj(x[k]) y[k] in one
// pass (x, y: Frag or T[K]). Two loads plus dotc_owned gave complex T extra BRA.
template <int K, class Ex, class U, class V, class X>
BATCHLAS_DLA_INLINE auto load_dotc_owned(const Ex& ex, const U& u, const V& v, X& x, X& y) {
    using T = std::remove_cvref_t<decltype(x[0])>;
    T acc = T(0);
    const int l = ex.lid();
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int r = l + k * Ex::size;
        x[k] = u(r);
        y[k] = v(r);
        acc = acc + detail::conj_ref<T>(x[k]) * y[k];
    }
    return acc;
}
// y[k] := y[k] - x[k] a (one classical Gram-Schmidt projection step).
template <class Ops = arith::Native, class T, int K> BATCHLAS_DLA_INLINE void sub_mul(Frag<T, K>& y, const Frag<T, K>& x, T a) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (detail::is_native_v<Ops>) y[k] = y[k] - x[k] * a;
        else arith_t<Ops, T>::msub(y[k], x[k], a);
    }
}
// sum of |x[k]|^2 over the item's rows < n (each abs2 its own rounding, as the hand loop).
template <class Ex, class T, int K> BATCHLAS_DLA_INLINE real_t<T> sumsq_owned(const Ex& ex, const Frag<T, K>& x, int n) {
    real_t<T> acc = real_t<T>(0);
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) {
        if (r < n) acc += abs2(x[k]);
    });
    return acc;
}

// Lane l returns ||a(:, l)||^2 for the P = Ex::size columns of a, whose static row count (a.m,
// an Ext) is Cyclic over the executor: per-lane partial sums, then reduce_scatter<P> (log2 P
// exchanges; core sumsq over P columns would be P all-reduces).
template <class Ex, class M> BATCHLAS_DLA_INLINE auto col_sumsq(const Ex& ex, const M& a) {
    using T = typename M::value_type;
    using R = real_t<T>;
    constexpr int P = Ex::size;
    constexpr int Rows = decltype(M::m)::value;
    Frag<R, P> x;
#pragma unroll
    for (int c = 0; c < P; ++c) {
        R acc = R(0);
        for_each_owned_frag<Rows>(ex, [&](int, int r) { acc += abs2(static_cast<T>(a(r, c))); });
        x[c] = acc;
    }
    return reduce_scatter<P>(ex, x);
}

// ---- pivot schedule ------------------------------------------------------------------------
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

// Golub & Van Loan sym.schur2 generalised by a diagonal phase: for H = [app apq; conj(apq) aqq]
// (app, aqq real) U^H H U is diagonal. d = conj(apq) / |apq| makes the 2x2 real, then the
// classical rotation with tau = (aqq - app) / (2 g), t = sign(tau) / (|tau| + sqrt(1 + tau^2)).
// For real T, d = 1 and g carries the sign of apq. A rotation below the threshold or whose s
// rounds to 0 (it could never annihilate apq) is the identity with active == false.
template <class T>
BATCHLAS_DLA_INLINE JacobiRotation<T> sym_schur2(real_t<T> app, real_t<T> aqq, T apq, const JacobiThreshold<real_t<T>>& th) {
    using R = real_t<T>;
    JacobiRotation<T> r = identity_rotation<T>();
    const R g_abs = modulus(apq);
    const R thresh = th.tol * sycl::sqrt(sycl::fabs(app) * sycl::fabs(aqq));
    if (g_abs > thresh && g_abs > th.tiny) {
        R g;
        if constexpr (Arith<T>::is_complex) {
            g = g_abs;
            r.d = T(apq.real() / g_abs, -apq.imag() / g_abs);
        } else {
            g = apq;
            r.d = T(1);
        }
        r.g = g;
        const R tau = (aqq - app) / (R(2) * g);
        R tt;
        if (sycl::fabs(tau) > th.tau_big) {
            tt = R(1) / (R(2) * tau);
        } else {
            tt = sycl::copysign(R(1), tau) / (sycl::fabs(tau) + sycl::sqrt(R(1) + tau * tau));
        }
        r.t = tt;
        r.c = R(1) / sycl::sqrt(R(1) + tt * tt);
        r.s = tt * r.c;
        r.active = !(r.s == R(0));
    }
    if (!r.active) {
        r.c = R(1);
        r.s = R(0);
        r.d = T(1);
    }
    return r;
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

// ---- applying a rotation -------------------------------------------------------------------
// Not core rot: U = [c s; -d s  d c] scales y by the phase in both outputs, and folding d into y
// first (then a real rot) rounds differently.
template <class T> struct Rot2 {
    T u11, u12, u21, u22;
};
template <class T> struct Pair2 {
    T x, y;
};

// The entries of U (Conj: of conj(U), for rot_left).
template <bool Conj, class Ops = arith::Native, class T> BATCHLAS_DLA_INLINE Rot2<T> rot2(real_t<T> c, real_t<T> s, T d) {
    using A = Arith<T>;
    if constexpr (!A::is_complex) {
        (void)d;
        return {T(c), T(s), T(-s), T(c)};
    } else if constexpr (detail::is_native_v<Ops>) {
        const T dd = Conj ? A::conj(d) : d;
        return {T(c), T(s), -(dd * T(s)), dd * T(c)};
    } else {
        const T dd = Conj ? A::conj(d) : d;
        return {T(c), T(s), A::make(-(A::re(dd) * s), -(A::im(dd) * s)), A::make(A::re(dd) * c, A::im(dd) * c)};
    }
}

namespace detail {
// a x + b y with a real and b complex (the explicit form).
template <class T> BATCHLAS_DLA_INLINE T rx_plus_by(real_t<T> a, T x, T b, T y) {
    using A = Arith<T>;
    return A::make(a * A::re(x) + A::re(b) * A::re(y) - A::im(b) * A::im(y),
                   a * A::im(x) + A::re(b) * A::im(y) + A::im(b) * A::re(y));
}
}  // namespace detail

// [x y] := [x y] U: row r of a column pair (A(r, p), A(r, q)) under A := A U.
template <class Ops = arith::Native, class T> BATCHLAS_DLA_INLINE Pair2<T> rot_right(const Rot2<T>& u, T x, T y) {
    if constexpr (detail::is_native_v<Ops> || !Arith<T>::is_complex) {
        return {x * u.u11 + y * u.u21, x * u.u12 + y * u.u22};
    } else {
        using A = Arith<T>;
        return {detail::rx_plus_by(A::re(u.u11), x, u.u21, y), detail::rx_plus_by(A::re(u.u12), x, u.u22, y)};
    }
}
// [x; y] := U^H [x; y] for u = rot2<true>(...): column c of a row pair under A := U^H A.
template <class Ops = arith::Native, class T> BATCHLAS_DLA_INLINE Pair2<T> rot_left(const Rot2<T>& u, T x, T y) {
    if constexpr (detail::is_native_v<Ops> || !Arith<T>::is_complex) {
        return {u.u11 * x + u.u21 * y, u.u12 * x + u.u22 * y};
    } else {
        using A = Arith<T>;
        return {detail::rx_plus_by(A::re(u.u11), x, u.u21, y), detail::rx_plus_by(A::re(u.u12), x, u.u22, y)};
    }
}
// In place on two elements (both read, then both written).
template <class Ops = arith::Native, class T> BATCHLAS_DLA_INLINE void rot_right_inplace(const Rot2<T>& u, T& x, T& y) {
    const T a = x;
    const T b = y;
    const Pair2<T> r = rot_right<Ops>(u, a, b);
    x = r.x;
    y = r.y;
}

// One round of disjoint rotations as the caller published it: slot k holds (c, s) in one 2-wide
// element, the phase d (complex T only; NoPhase for real T) and the packed pair (p, q). Each slot
// array is an operand read with (k): a v2::Vec over the kernel's local accessor or a pointer.
template <class T, class CS, class D, class PR> struct RotRound {
    using R = real_t<T>;
    CS cs;
    D d;
    PR pairs;

    BATCHLAS_DLA_INLINE PivotPair pair(int k) const { return unpack_pair(pairs(k)); }
    BATCHLAS_DLA_INLINE sycl::vec<R, 2> coef(int k) const { return cs(k); }
    BATCHLAS_DLA_INLINE void store(int k, const JacobiRotation<T>& r) const {
        cs(k) = sycl::vec<R, 2>(r.c, r.s);
        if constexpr (Arith<T>::is_complex) d(k) = r.d;
    }
    template <bool Conj, class Ops = arith::Native> BATCHLAS_DLA_INLINE Rot2<T> rot(int k, R c, R s) const {
        if constexpr (Arith<T>::is_complex) return rot2<Conj, Ops, T>(c, s, d(k));
        else return rot2<Conj, Ops, T>(c, s, T(1));
    }
};
template <class T, class CS, class D, class PR> BATCHLAS_DLA_INLINE RotRound<T, CS, D, PR> rot_round(CS cs, D d, PR pairs) {
    return {cs, d, pairs};
}

// A_i := A_i U_k for every rotation k < npairs of the round (s == 0 skipped), on row `row` of
// every A_i: the calling item owns that row, and the pairs of a round touch disjoint columns,
// so no barrier is needed between rotations. One pass serves A and its accumulators (Z := Z U).
template <class Ops = arith::Native, class T, class CS, class D, class PR, class N, class... Ms>
BATCHLAS_DLA_INLINE void rot_round_right(const RotRound<T, CS, D, PR>& rd, N npairs, int row, const Ms&... ms) {
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
BATCHLAS_DLA_INLINE void rot_round_left_hermitian(const RotRound<T, CS, D, PR>& rd, N npairs, int col, const M& a) {
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

}  // namespace batchlas::device::v2

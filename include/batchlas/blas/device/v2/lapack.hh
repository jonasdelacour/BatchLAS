// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, LAPACK auxiliaries on an executor: 2-D copies and fills on a triangle
// (lacpy, laset, mirror), row interchanges (laswp), plane rotations (rot, lasr, lasr_rows) and
// Householder reflectors (larfg, larf). Ownership is the executor's Cyclic layout (2-D: flat
// column-major, for_each_owned_2d), so ops on the same executor compose without a barrier
// unless one reads an element another item wrote (each op says so).

#include <type_traits>

#include <batchlas/blas/device/v2/level1.hh>
#include <batchlas/blas/device/v2/scalar.hh>

namespace batchlas::device::v2 {

// The part of a matrix an op touches: Full, a triangle with its diagonal (Lower, Upper) or
// without it (StrictLower, StrictUpper).
enum class Part { Full, Lower, Upper, StrictLower, StrictUpper };
namespace detail {
template <Part P> BATCHLAS_DLA_INLINE bool in_part(int i, int j) {
    if constexpr (P == Part::Full) return true;
    else if constexpr (P == Part::Lower) return i >= j;
    else if constexpr (P == Part::Upper) return i <= j;
    else if constexpr (P == Part::StrictLower) return i > j;
    else return i < j;
}
}  // namespace detail

// B := op(A) on part P of B (LAPACK lacpy; OP = Trans / ConjTrans reads A(j, i)). B outside P
// is not written; A outside the mirrored part is not read. B must not alias A for OP != NoTrans.
template <Part P = Part::Full, Transpose OP = Transpose::NoTrans, class Alg = alg::Walk, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b) {
    using T = typename MB::value_type;
    for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int j) {
        if (!detail::in_part<P>(i, j)) return;
        if constexpr (OP == Transpose::NoTrans) b(i, j) = a(i, j);
        else if constexpr (OP == Transpose::Trans) b(i, j) = a(j, i);
        else b(i, j) = Arith<T>::conj(static_cast<T>(a(j, i)));
    });
}

// A := alpha off the diagonal and beta on it, on part P (LAPACK laset). Tags fold: laset of
// Zero, Zero is a store of zero with no select.
template <Part P = Part::Full, class Alg = alg::Walk, class Ex, class AS, class BS, class MA>
BATCHLAS_DLA_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, const MA& a) {
    using T = typename MA::value_type;
    const T off = tag_mul(as_tag(alpha_), Arith<T>::one());
    const T dia = tag_mul(as_tag(beta_), Arith<T>::one());
    for_each_owned_2d<Alg>(ex, a.rows(), a.cols(), [&](int i, int j) {
        if (!detail::in_part<P>(i, j)) return;
        if constexpr (std::is_same_v<tag_t<AS>, tag_t<BS>> && !std::is_same_v<tag_t<AS>, Scalar<T>>) a(i, j) = off;
        else a(i, j) = i == j ? dia : off;
    });
}

// The strict triangle opposite FROM, from FROM: A(i, j) := A(j, i) (Herm: conj, and the
// diagonal's imaginary part is zeroed). Each write reads only the FROM triangle, so items do
// not race; a later op reading the mirrored triangle across owners needs a barrier.
template <Uplo FROM, bool Herm = false, class Alg = alg::Walk, class Ex, class MA>
BATCHLAS_DLA_INLINE void mirror(const Ex& ex, const MA& a) {
    using T = typename MA::value_type;
    using A = Arith<T>;
    const int n = a.rows();
    for_each_owned_2d<Alg>(ex, n, n, [&](int i, int j) {
        const bool dst = FROM == Uplo::Lower ? i < j : i > j;
        if (dst) a(i, j) = Herm ? A::conj(static_cast<T>(a(j, i))) : static_cast<T>(a(j, i));
        else if constexpr (Herm && A::is_complex) {
            if (i == j) a(i, j) = A::make(A::re(static_cast<T>(a(i, j))), real_t<T>(0));
        }
    });
}

// LAPACK laswp: for k = k1 .. k2-1 (Forward; Backward: k2-1 .. k1) swap rows k and ipiv(k) - 1
// of A (pivots 1-based int32, LAPACK's convention). Columns are owned Cyclic, each item runs the
// whole interchange sequence on its columns: no barrier, no race.
namespace dir {
struct Forward {};
struct Backward {};
}  // namespace dir
template <class Dir = dir::Forward, class Ex, class MA, class P>
BATCHLAS_DLA_INLINE void laswp(const Ex& ex, const MA& a, const P& ipiv, int k1, int k2) {
    using T = typename MA::value_type;
    for_each_owned(ex, a.cols(), [&](int j) {
        auto step = [&](int k) {
            const int p = static_cast<int>(ipiv(k)) - 1;
            if (p != k) {
                const T t = a(k, j);
                a(k, j) = a(p, j);
                a(p, j) = t;
            }
        };
        if constexpr (std::is_same_v<Dir, dir::Forward>)
            for (int k = k1; k < k2; ++k) step(k);
        else
            for (int k = k2 - 1; k >= k1; --k) step(k);
    });
}

// ---- plane rotations ----------------------------------------------------------------------------
// rot::Lapack (zrot, the lartg convention): (x, y) := (c x + s y, c y - conj(s) x).
// rot::Transposed (the steqr / dlasr-pivot-V store): (x, y) := (c x - s y, s x + c y), s real.
namespace rot_conv {
struct Lapack {};
struct Transposed {};
}  // namespace rot_conv
template <class Conv = rot_conv::Lapack, class R, class S, class T> BATCHLAS_DLA_INLINE void rot(const Givens<R, S>& g, T& x, T& y) {
    using A = Arith<T>;
    const T a = x, b = y;
    if constexpr (std::is_same_v<Conv, rot_conv::Lapack>) {
        if constexpr (A::is_complex) {
            const T sb = Arith<T>::mul(static_cast<T>(g.s), b), sa = Arith<T>::mul(Arith<T>::conj(static_cast<T>(g.s)), a);
            x = A::make(g.c * A::re(a) + A::re(sb), g.c * A::im(a) + A::im(sb));
            y = A::make(g.c * A::re(b) - A::re(sa), g.c * A::im(b) - A::im(sa));
        } else {
            x = g.c * a + g.s * b;
            y = g.c * b - g.s * a;
        }
    } else {
        static_assert(std::is_same_v<Conv, rot_conv::Transposed> && std::is_same_v<R, S>, "v2::rot: rot_conv::Transposed takes a real s");
        if constexpr (A::is_complex) {
            x = A::make(g.c * A::re(a) - g.s * A::re(b), g.c * A::im(a) - g.s * A::im(b));
            y = A::make(g.s * A::re(a) + g.c * A::re(b), g.s * A::im(a) + g.c * A::im(b));
        } else {
            x = g.c * a - g.s * b;
            y = g.s * a + g.c * b;
        }
    }
}
// (x(i), y(i)) := G (x(i), y(i)) for every owned i; elementwise, no barrier.
template <class Conv = rot_conv::Lapack, class Ex, class R, class S, class X, class Y>
BATCHLAS_DLA_INLINE void rot(const Ex& ex, const Givens<R, S>& g, const X& x, const Y& y) {
    for_each_owned(ex, x.size(), [&](int i) {
        auto& xi = x(i);
        auto& yi = y(i);
        rot<Conv>(g, xi, yi);
    });
}
// x := x G(0) G(1) ... G(n-2) (dlasr SIDE = R, PIVOT = V, DIRECT = F on one row): rotation j
// acts on (x(j), x(j+1)) with Conv's convention. One vector is one dependence chain: Thread.
template <class Conv = rot_conv::Transposed, class Rots, class X> BATCHLAS_DLA_INLINE void lasr(const Thread&, int n, const Rots& rots, const X& x) {
    for (int j = 0; j < n - 1; ++j) {
        const auto g = rots(j);  // a copy: the table is not re-read after a store
        auto& p = x(j);
        auto& q = x(j + 1);
        rot<Conv>(g, p, q);
    }
}
// A := A G(0) ... G(n-2) on every owned row (dlasr SIDE = R): each row runs the serial chain.
template <class Conv = rot_conv::Transposed, class Ex, class Rots, class MA>
BATCHLAS_DLA_INLINE void lasr_rows(const Ex& ex, const Rots& rots, const MA& a) {
    for_each_owned(ex, a.rows(), [&](int r) {
        for (int j = 0; j < a.cols() - 1; ++j) {
            const auto g = rots(j);
            auto& p = a(r, j);
            auto& q = a(r, j + 1);
            rot<Conv>(g, p, q);
        }
    });
}

// ---- Householder -------------------------------------------------------------------------------
// xLARFG on (alpha, x) distributed Cyclic over ex: alpha := beta, x := v(1:), returns the
// scalars (tau, identity, ...). alpha must be uniform. conv::Lapack: every item seeds its max
// with alpha, then two reductions (max, scaled sum of squares: geqrf's two-pass norm) and the
// fma / Smith scaling of arith::Fma; conv::Phase / RealBeta: ||x|| Unscaled (one reduction, the
// legacy group larfg) and x *= 1 / (alpha - beta). No barrier: the caller owns the one before
// x is read across owners.
template <class Conv = conv::Lapack, Scope S = Scope::Auto, class Ex, class T, class X>
BATCHLAS_DLA_INLINE LarfgScalars<T> larfg(const Ex& ex, T& alpha, const X& x) {
    using A = Arith<T>;
    using R = real_t<T>;
    LarfgScalars<T> h;
    if constexpr (std::is_same_v<Conv, conv::Lapack>) {
        auto amax_ri = [](const T& v) {
            if constexpr (A::is_complex) return sycl::fmax(sycl::fabs(A::re(v)), sycl::fabs(A::im(v)));
            else return sycl::fabs(v);
        };
        R smax = amax_ri(alpha);
        for_each_owned(ex, x.size(), [&](int i) { smax = sycl::fmax(smax, amax_ri(static_cast<T>(x(i)))); });
        smax = detail::scope_reduce<S>(ex, smax, sycl::maximum<R>());
        R ssq = R(0);  // the reduction stays outside the test: every item reaches it
        if (smax > R(0)) {
            for_each_owned(ex, x.size(), [&](int i) {
                const T v = x(i);
                const R a = A::re(v) / smax, b = A::im(v) / smax;
                ssq = ssq + sycl::fma(a, a, b * b);
            });
        }
        ssq = detail::scope_reduce<S>(ex, ssq);
        h = larfg_scalars<conv::Lapack>(alpha, ScaledSsq<R>{smax, ssq});
        alpha = h.beta;
        if (!h.identity) {
            if (h.use_mul) for_each_owned(ex, x.size(), [&](int i) { x(i) = FmaArith<T>::mul(static_cast<T>(x(i)), h.scale); });
            else for_each_owned(ex, x.size(), [&](int i) { x(i) = FmaArith<T>::div(static_cast<T>(x(i)), h.scale); });
        }
    } else {
        const int n = x.size();
        const R xnorm = n > 0 ? nrm2<alg::Unscaled, S>(ex, x) : R(0);
        h = larfg_scalars<Conv>(alpha, xnorm, n + 1);
        alpha = h.beta;
        if (!h.identity && n > 0) for_each_owned(ex, n, [&](int i) { x(i) = A::mul(h.scale, static_cast<T>(x(i))); });
    }
    return h;
}

// C := H C (Side::Left) or C H (Side::Right), H = I - tau v v^H (pass conj(tau) for H^H). v is
// UnitHead<V> (v(head) = 1 implied, zero above) or any Published vector; every item reads all
// of it. Left on a team executor (alg::SgPerCol): team t owns columns t mod nteams, its lanes
// stride the rows, Scope::Sub only (no work-group barrier). Right (alg::RowOwner): each item
// owns rows, serial over the columns. Thread: both serial. C is updated in place; the caller
// owns any barrier before another owner reads C.
template <Side SD, class Ex, class T, class V, class MC> BATCHLAS_DLA_INLINE void larf(const Ex& ex, T tau, const V& vp, const MC& c) {
    using A = Arith<T>;
    const auto& v = detail::unpub(vp);
    const int m = c.rows(), n = c.cols();
    if (A::is_zero(tau)) return;
    if constexpr (SD == Side::Left) {
        const int lane = is_thread_v<Ex> ? 0 : ex.lane();
        const int tsize = detail::team_size_of(ex), nt = detail::nteams_of(ex);
        const int t0 = is_thread_v<Ex> ? 0 : ex.team();
        static_assert(Ex::sub_whole || is_work_group_v<Ex> || is_runtime_v<Ex>, "v2::larf<Left>: a team or work-group executor");
        for (int j = t0; j < n; j += nt) {
            T w = A::zero();
            for (int r = lane; r < m; r += tsize) A::mac_conj(w, static_cast<T>(v(r)), static_cast<T>(c(r, j)));
            w = ex.team_reduce(w);
            const T tw = A::mul(tau, w);
            for (int r = lane; r < m; r += tsize) {
                T cij = c(r, j);
                A::msub(cij, static_cast<T>(v(r)), tw);
                c(r, j) = cij;
            }
        }
    } else {
        for_each_owned(ex, m, [&](int r) {
            T w = A::zero();
            for (int j = 0; j < n; ++j) A::mac(w, static_cast<T>(c(r, j)), static_cast<T>(v(j)));
            const T tw = A::mul(w, tau);
            for (int j = 0; j < n; ++j) {
                T crj = c(r, j);
                A::msub(crj, tw, A::conj(static_cast<T>(v(j))));
                c(r, j) = crj;
            }
        });
    }
}

}  // namespace batchlas::device::v2

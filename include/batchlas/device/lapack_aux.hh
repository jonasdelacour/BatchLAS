// SPDX-License-Identifier: MIT
#pragma once
// Device LA, LAPACK auxiliaries on an executor: 2-D copies and fills on a triangle
// (lacpy, laset, mirror), row interchanges (laswp), plane rotations (rot, lasr, lasr_rows) and
// Householder reflectors (larfg, larf). Ownership is the executor's Cyclic layout (2-D: flat
// column-major, for_each_owned_2d), so ops on the same executor compose without a barrier
// unless one reads an element another item wrote (each op says so).

#include <type_traits>

#include <batchlas/device/level1.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/view.hh>

namespace batchlas::device {

// B := op(A) on part P of B (LAPACK lacpy; OP = Trans / ConjTrans reads A(j, i)). B outside P
// is not written; A outside the mirrored part is not read. B must not alias A for OP != NoTrans.
//
// Edge (how op(A)'s own extent meets B's): alg::EdgeNone (op(A) covers B), alg::EdgeGuard (B's
// elements outside op(A) are not written), alg::EdgeZeroPad (B is written everywhere: outside op(A)
// or outside P it gets 0; a tile load with padding). DiagFill (what B's diagonal gets inside P):
// DiagFill::Keep (copied), Real ((re, 0): a Hermitian diagonal), Unit (1, A not read).
namespace alg {
struct EdgeNone {};
struct EdgeGuard {};
struct EdgeZeroPad {};
}  // namespace alg
template <Part P = Part::Full, Transpose OP = Transpose::NoTrans, class Alg = alg::Walk, class Edge = alg::EdgeNone,
          DiagFill DF = DiagFill::Keep, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b) {
    using T = typename MB::value_type;
    if constexpr (std::is_same_v<Edge, alg::EdgeNone> && DF == DiagFill::Keep) {
        for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int j) {
            if (!detail::in_part<P>(i, j)) return;
            if constexpr (OP == Transpose::NoTrans) b(i, j) = a(i, j);
            else if constexpr (OP == Transpose::Trans) b(i, j) = a(j, i);
            else b(i, j) = Arith<T>::conj(static_cast<T>(a(j, i)));
        });
    } else {
        using A = Arith<T>;
        static_assert(std::is_same_v<Edge, alg::EdgeNone> || std::is_same_v<Edge, alg::EdgeGuard> || std::is_same_v<Edge, alg::EdgeZeroPad>,
                      "lacpy: alg::EdgeNone, Guard or ZeroPad");
        static_assert(DF == DiagFill::Keep || DF == DiagFill::Real || DF == DiagFill::Unit,
                      "lacpy: DiagFill::Keep, Real or Unit");
        const int ar = OP == Transpose::NoTrans ? a.rows() : a.cols();
        const int ac = OP == Transpose::NoTrans ? a.cols() : a.rows();
        for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int j) {
            const bool inside = std::is_same_v<Edge, alg::EdgeNone> || (i < ar && j < ac);
            if (!detail::in_part<P>(i, j) || !inside) {
                if constexpr (std::is_same_v<Edge, alg::EdgeZeroPad>) b(i, j) = A::zero();
                return;
            }
            if constexpr (DF == DiagFill::Unit) {
                if (i == j) {
                    b(i, j) = A::one();
                    return;
                }
            }
            T v;
            if constexpr (OP == Transpose::NoTrans) v = a(i, j);
            else if constexpr (OP == Transpose::Trans) v = a(j, i);
            else v = A::conj(static_cast<T>(a(j, i)));
            if constexpr (DF == DiagFill::Real && A::is_complex)
                if (i == j) v = A::make(A::re(v), real_t<T>(0));
            b(i, j) = v;
        });
    }
}

// A := alpha off the diagonal and beta on it, on part P (LAPACK laset). Tags fold: laset of
// Zero, Zero is a store of zero with no select.
template <Part P = Part::Full, class Alg = alg::Walk, class Ex, class AS, class BS, class MA>
BATCHLAS_DEVICE_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, const MA& a) {
    using T = typename MA::value_type;
    const T off = tag_mul(as_tag(alpha_), Arith<T>::one());
    const T dia = tag_mul(as_tag(beta_), Arith<T>::one());
    for_each_owned_2d<Alg>(ex, a.rows(), a.cols(), [&](int i, int j) {
        if (!detail::in_part<P>(i, j)) return;
        if constexpr (std::is_same_v<tag_t<AS>, tag_t<BS>> && !std::is_same_v<tag_t<AS>, Scalar<T>>) a(i, j) = off;
        else a(i, j) = select(i == j, dia, off);
    });
}
// A block of a larger matrix: its diagonal (and P) in the parent's coordinates, element (i, j)
// being the parent's (i + off, j) up to a common shift (off = block row origin - column origin).
template <Part P = Part::Full, class Alg = alg::Walk, class Ex, class AS, class BS, class MA>
BATCHLAS_DEVICE_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, const MA& a, int off) {
    using T = typename MA::value_type;
    const T o = tag_mul(as_tag(alpha_), Arith<T>::one());
    const T d = tag_mul(as_tag(beta_), Arith<T>::one());
    for_each_owned_2d<Alg>(ex, a.rows(), a.cols(), [&](int i, int j) {
        if (!detail::in_part<P>(i + off, j)) return;
        a(i, j) = select(i + off == j, d, o);
    });
}
// Register forms: f[r] := beta at r == j, alpha elsewhere (one column of xLASET); t(i, j) :=
// beta where i - j == off, alpha elsewhere.
template <class T, int N, class L, class AS, class BS> BATCHLAS_DEVICE_INLINE void laset(Frag<T, N, L>& f, AS alpha_, BS beta_, int j) {
    const T o = tag_mul(as_tag(alpha_), Arith<T>::one());
    const T d = tag_mul(as_tag(beta_), Arith<T>::one());
#pragma unroll
    for (int r = 0; r < N; ++r) f[r] = select(r == j, d, o);
}
template <class T, int R, int C, class AS, class BS> BATCHLAS_DEVICE_INLINE void laset(Tile<T, R, C>& t, AS alpha_, BS beta_, int off = 0) {
    const T o = tag_mul(as_tag(alpha_), Arith<T>::one());
    const T d = tag_mul(as_tag(beta_), Arith<T>::one());
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) t(i, j) = select(i - j == off, d, o);
}

// The strict triangle opposite FROM, from FROM: A(i, j) := A(j, i) (Herm: conj, and the
// diagonal's imaginary part is zeroed). Each write reads only the FROM triangle, so items do
// not race; a later op reading the mirrored triangle across owners needs a barrier.
template <Uplo FROM, bool Herm = false, class Alg = alg::Walk, class Ex, class MA>
BATCHLAS_DEVICE_INLINE void mirror(const Ex& ex, const MA& a) {
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
enum class Direct { Forward, Backward };
template <Direct Dir = Direct::Forward, class Ex, class MA, class P>
BATCHLAS_DEVICE_INLINE void laswp(const Ex& ex, const MA& a, const P& ipiv, int k1, int k2) {
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
        if constexpr (Dir == Direct::Forward)
            for (int k = k1; k < k2; ++k) step(k);
        else
            for (int k = k2 - 1; k >= k1; --k) step(k);
    });
}
// Where row i of A ends up after laswp<Dir>(ipiv, k1, k2) (the interchanges followed on one
// index: no memory touched). laswp_index<Backward> of a position is the row that lands there.
template <Direct Dir = Direct::Forward, class P> BATCHLAS_DEVICE_INLINE int laswp_index(const P& ipiv, int k1, int k2, int i) {
    int r = i;
    auto step = [&](int k) {
        const int p = static_cast<int>(ipiv(k)) - 1;
        if (r == k) r = p;
        else if (r == p) r = k;
    };
    if constexpr (Dir == Direct::Forward)
        for (int k = k1; k < k2; ++k) step(k);
    else
        for (int k = k2 - 1; k >= k1; --k) step(k);
    return r;
}
// B := laswp<Dir>(A) out of place (B must not alias A): B(i, j) = A(row landing at i, j), every
// element one gather, 2-D flat ownership (no per-column interchange chain).
template <Direct Dir = Direct::Forward, class Ex, class MA, class P, class MB>
BATCHLAS_DEVICE_INLINE void laswp_gather(const Ex& ex, const MA& a, const P& ipiv, int k1, int k2, const MB& b) {
    constexpr Direct Inv = Dir == Direct::Forward ? Direct::Backward : Direct::Forward;
    for_each_owned_2d(ex, b.rows(), b.cols(), [&](int i, int j) { b(i, j) = a(laswp_index<Inv>(ipiv, k1, k2, i), j); });
}
// LAPACK xLAPMR out of place with a 1-based permutation k: Forward B(i, :) = A(k(i) - 1, :);
// Backward B(k(i) - 1, :) = A(i, :).
template <Direct Dir = Direct::Forward, class Ex, class MA, class K, class MB>
BATCHLAS_DEVICE_INLINE void lapmr(const Ex& ex, const MA& a, const K& k, const MB& b) {
    for_each_owned_2d(ex, b.rows(), b.cols(), [&](int i, int j) {
        const int p = static_cast<int>(k(i)) - 1;
        if constexpr (Dir == Direct::Forward) b(i, j) = a(p, j);
        else b(p, j) = a(i, j);
    });
}
// The pivot sink of a panel factorization: step k's 0-based row is stored 1-based plus base
// (the panel's global row), so a panel writes global pivots with no fix-up pass.
struct IpivSink {
    int* p;
    int base;
    BATCHLAS_DEVICE_INLINE void set(int k, int row) const { p[k] = base + row + 1; }
};

// ---- plane rotations ----------------------------------------------------------------------------
// alg::RotLapack (zrot, the lartg convention): (x, y) := (c x + s y, c y - conj(s) x).
// alg::RotTransposed (the steqr / dlasr-pivot-V store): (x, y) := (c x - s y, s x + c y), s real.
// alg::RotReflector2 (a 2-element Householder reflector, g.c = v0, g.s = v1 real, unit norm):
// t = x v0 + y v1; x -= 2 v0 t; y -= 2 v1 t (T's own operators, one expression each).
namespace alg {
struct RotLapack {};
struct RotTransposed {};
struct RotReflector2 {};
}  // namespace alg
template <class C> inline constexpr bool is_rot_conv_v =
    std::is_same_v<C, alg::RotLapack> || std::is_same_v<C, alg::RotTransposed> || std::is_same_v<C, alg::RotReflector2>;
template <class Conv = alg::RotLapack, class R, class S, class T> BATCHLAS_DEVICE_INLINE void rot(const Givens<R, S>& g, T& x, T& y) {
    using A = Arith<T>;
    const T a = x, b = y;
    if constexpr (std::is_same_v<Conv, alg::RotLapack>) {
        if constexpr (A::is_complex) {
            const T sb = Arith<T>::mul(static_cast<T>(g.s), b), sa = Arith<T>::mul(Arith<T>::conj(static_cast<T>(g.s)), a);
            x = A::make(g.c * A::re(a) + A::re(sb), g.c * A::im(a) + A::im(sb));
            y = A::make(g.c * A::re(b) - A::re(sa), g.c * A::im(b) - A::im(sa));
        } else {
            x = g.c * a + g.s * b;
            y = g.c * b - g.s * a;
        }
    } else if constexpr (std::is_same_v<Conv, alg::RotReflector2>) {
        static_assert(std::is_same_v<R, S>, "rot: alg::RotReflector2 takes a real (v0, v1)");
        const T t = x * g.c + y * g.s;
        x -= R(2) * g.c * t;
        y -= R(2) * g.s * t;
    } else {
        static_assert(std::is_same_v<Conv, alg::RotTransposed> && std::is_same_v<R, S>, "rot: alg::RotTransposed takes a real s");
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
template <class Conv = alg::RotLapack, class Ex, class R, class S, class X, class Y>
BATCHLAS_DEVICE_INLINE void rot(const Ex& ex, const Givens<R, S>& g, const X& x, const Y& y) {
    for_each_owned(ex, x.size(), [&](int i) {
        auto& xi = x(i);
        auto& yi = y(i);
        rot<Conv>(g, xi, yi);
    });
}
// x := x G(0) G(1) ... G(n-2) (dlasr SIDE = R, PIVOT = V, DIRECT = F on one row): rotation j
// acts on (x(j), x(j+1)) with Conv's convention. One vector is one dependence chain: Thread.
template <class Conv = alg::RotTransposed, class Rots, class X> BATCHLAS_DEVICE_INLINE void lasr(const Thread&, int n, const Rots& rots, const X& x) {
    for (int j = 0; j < n - 1; ++j) {
        const auto g = rots(j);  // a copy: the table is not re-read after a store
        auto& p = x(j);
        auto& q = x(j + 1);
        rot<Conv>(g, p, q);
    }
}
// A := A G(0) ... G(n-2) on every owned row (dlasr SIDE = R): each row runs the serial chain.
template <class Conv = alg::RotTransposed, class Ex, class Rots, class MA>
BATCHLAS_DEVICE_INLINE void lasr_rows(const Ex& ex, const Rots& rots, const MA& a) {
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
// scalars (tau, identity, ...). alpha must be uniform. alg::LarfgLapack: every item seeds its max
// with alpha, then two reductions (max, scaled sum of squares: geqrf's two-pass norm) and the
// fma / Smith scaling of arith::Fma; alg::Phase / RealBeta: ||x|| Unscaled (one reduction, the
// legacy group larfg) and x *= 1 / (alpha - beta). No barrier: the caller owns the one before
// x is read across owners.
template <class Conv = alg::LarfgLapack, Scope S = Scope::Own, class Ex, class T, class X>
BATCHLAS_DEVICE_INLINE LarfgScalars<T> larfg(const Ex& ex, T& alpha, const X& x) {
    using A = Arith<T>;
    using R = real_t<T>;
    LarfgScalars<T> h;
    if constexpr (std::is_same_v<Conv, alg::LarfgLapack>) {
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
        h = larfg_scalars<alg::LarfgLapack>(alpha, ScaledSsq<R>{smax, ssq});
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
//
// Ops: the arithmetic; Form: alg::ScaleW (C -= v (tau w), LAPACK) or ScaleV (C -= (tau v) w,
// Left only); Loop (Left): alg::Diverge (team t walks j = t, t + nteams, .. to its own bound) or
// alg::Lockstep (every team runs the same trip count, so its team_reduce is reached by every
// lane when teams share a sub-group).
namespace alg {
struct ScaleW {};
struct ScaleV {};
}  // namespace alg
template <Side SD, class Ops = arith::Explicit, class Form = alg::ScaleW, class Loop = alg::Diverge, class Ex, class T, class V,
          class MC>
BATCHLAS_DEVICE_INLINE void larf(const Ex& ex, T tau, const V& vp, const MC& c) {
    using A = arith_t<Ops, T>;
    const auto& v = detail::unpub(vp);
    const int m = c.rows(), n = c.cols();
    if (A::is_zero(tau)) return;
    constexpr bool scale_w = std::is_same_v<Form, alg::ScaleW>;
    static_assert(scale_w || (std::is_same_v<Form, alg::ScaleV> && SD == Side::Left), "larf: ScaleV is Side::Left only");
    if constexpr (SD == Side::Left && !(scale_w && std::is_same_v<Loop, alg::Diverge>)) {
        const int lane = is_thread_v<Ex> ? 0 : ex.lane();
        const int tsize = detail::team_size_of(ex);
        for_each_team_owned<Loop>(ex, n, [&](int j, bool live) BATCHLAS_DEVICE_BODY {
            T w = A::zero();
            if (live)
                for (int r = lane; r < m; r += tsize) A::mac_conj(w, static_cast<T>(v(r)), static_cast<T>(c(r, j)));
            w = ex.team_reduce(w);
            const T tw = scale_w ? A::mul(tau, w) : w;
            if (live)
                for (int r = lane; r < m; r += tsize) {
                    T cij = c(r, j);
                    if constexpr (scale_w) A::msub(cij, static_cast<T>(v(r)), tw);
                    else A::msub(cij, A::mul(tau, static_cast<T>(v(r))), tw);
                    c(r, j) = cij;
                }
        });
    } else if constexpr (SD == Side::Left) {
        const int lane = is_thread_v<Ex> ? 0 : ex.lane();
        const int tsize = detail::team_size_of(ex), nt = detail::nteams_of(ex);
        const int t0 = is_thread_v<Ex> ? 0 : ex.team();
        static_assert(Ex::sub_whole || is_work_group_v<Ex> || is_runtime_v<Ex>, "larf<Left>: a team or work-group executor");
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


// Register column form (xORG2L / larft's column step): c := (I - tau v v^H) c for c held in a
// Frag (element r of the column in c[r], r < N), v read at every r < N (zero outside its support:
// no predicate); v is read across owners (Published, or a unit_head of one).
template <Side SD, class Ops = arith::Explicit, class T, int N, class L, class V>
BATCHLAS_DEVICE_INLINE void larf(Frag<T, N, L>& c, const V& vp, T tau) {
    static_assert(SD == Side::Left, "larf (Frag): Side::Left, a column");
    using A = arith_t<Ops, T>;
    const auto& v = detail::unpub(vp);
    T dot = A::zero();
#pragma unroll
    for (int r = 0; r < N; ++r) A::mac_conj(dot, static_cast<T>(v(r)), c[r]);
    const T gamma = A::mul(tau, dot);
#pragma unroll
    for (int r = 0; r < N; ++r) A::msub(c[r], static_cast<T>(v(r)), gamma);
}

// xLARFG on a vector held ONE element per item (n <= the executor's size): item alpha_item holds
// alpha, items with x_active hold x (others pass x_active = false). ||x|| and alpha are made
// replicated (one reduction, one bcast), so every item forms the scalars (larfg_scalars<Conv>,
// default alg::Phase); on return alpha_item holds beta, the x items hold v (implicit unit head),
// and every item gets tau. n <= 1: tau = 0. COLLECTIVE.
template <class Conv = alg::Phase, class Ops = arith::Explicit, class Ex, class T>
BATCHLAS_DEVICE_INLINE T larfg(const Ex& ex, int n, int alpha_item, T& alpha, T& x, bool x_active) {
    using Ar = arith_t<Ops, T>;
    using R = real_t<T>;
    const R xsq = x_active ? abs2(x) : R(0);
    const R xnorm = sycl::sqrt(detail::scope_reduce<Scope::Own>(ex, xsq));
    const T alpha_b = ex.bcast(alpha, alpha_item);
    const auto s = larfg_scalars<Conv>(alpha_b, xnorm, n);
    if (ex.lid() == alpha_item) alpha = s.beta;
    else if (x_active && s.tau != Ar::zero()) x = Ar::mul(x, s.scale);
    return s.tau;
}

// ---- vectors of plane rotations (LAPACK xLARGV / xLARTV / xLAR2V), elementwise over ex ----------
// The first template argument is either core's (largv: lartg<W>'s W; lartv / lar2v:
// alg::RotLapack) or a rotation kernel class Rot supplying the scalar steps with its own
// arithmetic, c real: Rot::lartg(f, g, c&, s&, r&), Rot::rot_pair(x&, y&, c, s),
// Rot::lar2v_pair(x&, y&, z&, c, s) (the band-reduction kernels' btrd_givens_ops).
namespace detail {
template <class W> inline constexpr bool is_rot_kernel_v = !std::is_void_v<W> && !std::is_floating_point_v<W> && !is_rot_conv_v<W> &&
                                                           !std::is_same_v<W, alg::LartgExact> && !std::is_same_v<W, alg::LartgFast>;
}  // namespace detail
// largv: rotation i from (x(i), y(i)) by lartg<W>: x(i) := r, y(i) := s, c(i) := c.
template <class W = void, class Ex, class X, class Y, class C> BATCHLAS_DEVICE_INLINE void largv(const Ex& ex, const X& x, const Y& y, const C& c) {
    if constexpr (detail::is_rot_kernel_v<W>) {
        using T = typename X::value_type;
        using R = real_t<T>;
        for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DEVICE_BODY {
            const T f = x(t);
            const T g = y(t);
            R ct = R(0);
            T st = T(0);
            T rt = T(0);
            W::lartg(f, g, ct, st, rt);
            c(t) = ct;
            y(t) = st;
            x(t) = rt;
        });
    } else {
        for_each_owned(ex, x.size(), [&](int i) {
            const auto g = lartg<W>(static_cast<elem_t<X>>(x(i)), static_cast<elem_t<X>>(y(i)));
            c(i) = g.c;
            y(i) = g.s;
            x(i) = g.r;
        });
    }
}
// lartv: (x(i), y(i)) := rotation i applied (alg::RotLapack: x := c x + s y, y := c y - conj(s) x).
template <class Rot = alg::RotLapack, class Ex, class X, class Y, class C, class S>
BATCHLAS_DEVICE_INLINE void lartv(const Ex& ex, const X& x, const Y& y, const C& c, const S& s) {
    if constexpr (detail::is_rot_kernel_v<Rot>) {
        for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DEVICE_BODY { Rot::rot_pair(x(t), y(t), c(t), s(t)); });
    } else {
        static_assert(std::is_same_v<Rot, alg::RotLapack>, "lartv: alg::RotLapack or a rotation kernel class");
        for_each_owned(ex, x.size(), [&](int i) {
            using T = elem_t<X>;
            T xi = x(i), yi = y(i);
            rot<alg::RotLapack>(Givens<real_t<T>, std::remove_cvref_t<decltype(s(0))>>{c(i), s(i)}, xi, yi);
            x(i) = xi;
            y(i) = yi;
        });
    }
}
namespace detail {
template <class Ex, class X, class Y, class Z, class C, class S>
BATCHLAS_DEVICE_INLINE void lar2v_lapack(const Ex& ex, const X& x, const Y& y, const Z& z, const C& c, const S& s) {
    using T = elem_t<Z>;
    using A = Arith<T>;
    using R = real_t<T>;
    for_each_owned(ex, x.size(), [&](int i) {
        const R xi = A::re(static_cast<T>(x(i))), yi = A::re(static_cast<T>(y(i)));
        const R ci = c(i);
        if constexpr (A::is_complex) {
            const T zi = z(i), si = s(i);
            const R zir = A::re(zi), zii = A::im(zi), sir = A::re(si), sii = A::im(si);
            const R t1r = sir * zir - sii * zii, t1i = sir * zii + sii * zir;
            const T t2 = A::make(ci * zir, ci * zii);
            const T t3 = A::make(A::re(t2) - sir * xi, A::im(t2) + sii * xi);
            const T t4 = A::make(A::re(t2) + sir * yi, -A::im(t2) + sii * yi);
            const R t5 = ci * xi + t1r, t6 = ci * yi - t1r;
            x(i) = A::make(ci * t5 + (sir * A::re(t4) + sii * A::im(t4)), R(0));
            y(i) = A::make(ci * t6 - (sir * A::re(t3) - sii * A::im(t3)), R(0));
            z(i) = A::make(ci * A::re(t3) + (sir * t6 + sii * t1i), ci * A::im(t3) + (sir * t1i - sii * t6));
        } else {
            const R zi = z(i), si = s(i);
            const R t1 = si * zi, t2 = ci * zi, t3 = t2 - si * xi, t4 = t2 + si * yi, t5 = ci * xi + t1, t6 = ci * yi - t1;
            x(i) = ci * t5 + si * t4;
            y(i) = ci * t6 - si * t3;
            z(i) = ci * t4 - si * t5;
        }
    });
}
}  // namespace detail
// lar2v: the 2 x 2 Hermitian [x z; conj(z) y] (x, y real diagonals) := G [x z; conj(z) y] G^H with
// G = [c conj(s); -s c] per i (LAPACK zlar2v / dlar2v expressions, in their order).
template <class Rot = alg::RotLapack, class Ex, class X, class Y, class Z, class C, class S>
BATCHLAS_DEVICE_INLINE void lar2v(const Ex& ex, const X& x, const Y& y, const Z& z, const C& c, const S& s) {
    if constexpr (detail::is_rot_kernel_v<Rot>) {
        for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DEVICE_BODY { Rot::lar2v_pair(x(t), y(t), z(t), c(t), s(t)); });
    } else {
        static_assert(std::is_same_v<Rot, alg::RotLapack>, "lar2v: alg::RotLapack or a rotation kernel class");
        detail::lar2v_lapack(ex, x, y, z, c, s);
    }
}

// ---- norms ------------------------------------------------------------------------------------
// lanst<NT>(ex, d, e): the NT norm of the symmetric tridiagonal (d, e), e(i) coupling d(i) and
// d(i + 1). Max: max |d(i)|, |e(i)| by fmax (a NaN entry is ignored: the steqr rule); One = Inf:
// max_i |e(i-1)| + |d(i)| + |e(i)| (sycl::max); Frobenius: sqrt(sum d^2 + 2 sum e^2) unscaled.
// Uniform over scope S. LaneVec form (Max): one element of d and e per item, ascending butterfly.
template <NormType NT, Scope S = Scope::Own, class Ex, class D, class E> BATCHLAS_DEVICE_INLINE auto lanst(const Ex& ex, const D& d, const E& e) {
    using R = elem_t<D>;
    static_assert(!Arith<R>::is_complex, "lanst: d and e are real");
    const int n = d.size();
    if constexpr (NT == NormType::Max) {
        R m = R(0);
        for_each_owned(ex, n, [&](int i) {
            m = sycl::fmax(m, sycl::fabs(d(i)));
            if (i < n - 1) m = sycl::fmax(m, sycl::fabs(e(i)));
        });
        return allreduce<S>(ex, m, sycl::maximum<R>());  // partials are never NaN (fmax from 0)
    } else if constexpr (NT == NormType::One || NT == NormType::Inf) {
        R m = R(0);
        for_each_owned(ex, n, [&](int i) {
            const R left = (i > 0) ? sycl::fabs(e(i - 1)) : R(0);
            const R right = (i < n - 1) ? sycl::fabs(e(i)) : R(0);
            m = sycl::max(m, sycl::fabs(d(i)) + left + right);
        });
        return allreduce<S>(ex, m, sycl::maximum<R>());
    } else {
        static_assert(NT == NormType::Frobenius, "lanst: Max, One, Inf or Frobenius");
        R ss = R(0);
        for_each_owned(ex, n, [&](int i) {
            ss += d(i) * d(i);
            if (i < n - 1) ss += R(2) * e(i) * e(i);
        });
        return static_cast<R>(sycl::sqrt(allreduce<S>(ex, ss)));
    }
}
template <NormType NT, class Ex, class T> BATCHLAS_DEVICE_INLINE T lanst(const Ex& ex, const LaneVec<T>& d, const LaneVec<T>& e) {
    static_assert(NT == NormType::Max, "lanst (LaneVec): NormType::Max");
    T acc = T(0);
    if (d.owns(ex)) acc = sycl::fabs(d.v);
    if (e.owns(ex)) acc = sycl::fmax(acc, sycl::fabs(e.v));
    return detail::butterfly_inline<1, Ex::size, alg::BflyAsc>(ex, acc, reduce_op::FMax{});
}

// LAPACK lange on a Cyclic executor (|.| is the modulus). Max / Fro walk the flat column-major
// order; One gives each item whole columns, Inf whole rows (serial sums, no SLM), then one
// Scope S max. Fro: alg::Unscaled = sqrt(sum |a|^2), one reduction (overflows past
// sqrt(max)); alg::Scaled = the two-pass lassq. Max keeps std::max(|a|, m)'s NaN behaviour.
template <NormType NT, class Alg = alg::Unscaled, Scope S = Scope::Own, class Ex, class MA>
BATCHLAS_DEVICE_INLINE auto lange(const Ex& ex, const MA& a) {
    using T = typename MA::value_type;
    using R = real_t<T>;
    const int m = a.rows(), n = a.cols();
    if constexpr (NT == NormType::Max) {
        R mx = R(0);
        for_each_owned_2d(ex, m, n, [&](int i, int j) {
            const R v = modulus(static_cast<T>(a(i, j)));
            mx = v < mx ? mx : v;
        });
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else if constexpr (NT == NormType::One || NT == NormType::Inf) {
        constexpr bool one = NT == NormType::One;
        const int outer = one ? n : m, inner = one ? m : n;
        R mx = R(0);
        for (int k = ex.lid(); k < outer; k += Ex::size) {  // caller-owned shape (core 2.2 rule)
            R s = R(0);
            for (int l = 0; l < inner; ++l) s += modulus(static_cast<T>(one ? a(l, k) : a(k, l)));
            mx = sycl::maximum<R>()(mx, s);
        }
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else {
        static_assert(NT == NormType::Frobenius, "lange: Max, One, Inf or Frobenius");
        if constexpr (std::is_same_v<Alg, alg::Unscaled>) {
            R ss = R(0);
            for_each_owned_2d(ex, m, n, [&](int i, int j) { ss += abs2(static_cast<T>(a(i, j))); });
            return static_cast<R>(sycl::sqrt(detail::scope_reduce<S>(ex, ss)));
        } else {
            static_assert(std::is_same_v<Alg, alg::Scaled>, "lange: Frobenius is alg::Unscaled or alg::Scaled");
            using A = Arith<T>;
            R sc = R(0);
            auto upd = [&](R v) { sc = (v > sc || sycl::isnan(v)) ? v : sc; };
            for_each_owned_2d(ex, m, n, [&](int i, int j) {
                const T v = a(i, j);
                upd(sycl::fabs(A::re(v)));
                if constexpr (A::is_complex) upd(sycl::fabs(A::im(v)));
            });
            const R scale = detail::scope_reduce<S>(ex, sc, sycl::maximum<R>());
            if (sycl::isinf(scale)) return scale;
            const R inv = scale > R(0) ? R(1) / scale : R(1);
            R ss = R(0);
            for_each_owned_2d(ex, m, n, [&](int i, int j) {
                const T v = a(i, j);
                const R tr = A::re(v) * inv;
                ss += tr * tr;
                if constexpr (A::is_complex) {
                    const R ti = A::im(v) * inv;
                    ss += ti * ti;
                }
            });
            ss = detail::scope_reduce<S>(ex, ss);
            return scale > R(0) ? scale * sycl::sqrt(ss) : sycl::sqrt(ss);
        }
    }
}

// ---- equilibration, scaling, conjugation, packed copies ----------------------------------------
// LAPACK xLAQGE: scale rows by r and / or columns by c when the ratios call for it (thresh 0.1,
// small = safmin / eps, large = 1 / small); returns 'N', 'R', 'C' or 'B' (uniform). A(i, j)
// := c(j) r(i) A(i, j) in that order (a real scale on a complex A scales both parts).
template <class Ex, class MA, class RV, class CV, class R>
BATCHLAS_DEVICE_INLINE char laqge(const Ex& ex, const MA& a, const RV& r, const CV& c, R rowcnd, R colcnd, R amax) {
    using T = typename MA::value_type;
    constexpr R thresh = R(0.1);
    constexpr R small = lamch<'S', R>() / lamch<'P', R>();
    constexpr R large = R(1) / small;
    const bool rows = !(rowcnd >= thresh && amax >= small && amax <= large);
    const bool cols = !(colcnd >= thresh);
    if (!rows && !cols) return 'N';
    for_each_owned_2d(ex, a.rows(), a.cols(), [&](int i, int j) {
        R s;
        if (rows && cols) s = static_cast<R>(c(j)) * static_cast<R>(r(i));
        else if (rows) s = static_cast<R>(r(i));
        else s = static_cast<R>(c(j));
        a(i, j) = Arith<T>::scale_real(s, static_cast<T>(a(i, j)));
    });
    return rows ? (cols ? 'B' : 'R') : 'C';
}

namespace detail {
// A core fn_vec is computed on read, never stored: any item may read any index unpublished.
template <class V> inline constexpr bool is_fn_vec_v = false;
template <class F> inline constexpr bool is_fn_vec_v<device::FnVec<F>> = true;
}  // namespace detail
// A := diag(r) A diag(c), r and c real, unconditionally (the ortho SVQB kernels; the laqge above
// tests LAPACK's ratios first). One row per item (launch width == a.rows(), WG = k);
// c is read by every item, so it is Published or an fn_vec.
template <class Ex, class Rv, class Cv, class MA> BATCHLAS_DEVICE_INLINE void laqge(const Ex& ex, const Rv& r, const Cv& cp, const MA& a) {
    using Ar = device::Arith<typename MA::value_type>;
    static_assert(device::is_published_v<Cv> || detail::is_fn_vec_v<Cv>, "laqge: c is read across owners; publish it");
    const auto& c = device::detail::unpub(cp);
    const int n = a.cols(), ld = int(a.ld);
    device::for_each_owned_exact(ex, a.rows(), [&](int i) {
        const auto ri = r(i);
        auto* const ai = a.p + a.off + i;  // row i moved once; a per-element i + j*ld blocks the unroll
        for (int j = 0; j < n; ++j) {
            const auto s = ri * c(j);
            auto& e = ai[j * ld];
            e = Ar::scale_real(s, e);
        }
    });
}

// LAPACK xLASCL: A := (cto / cfrom) A on part P, the factor applied in steps that never over- or
// underflow (smlnum = safmin, bignum = 1 / smlnum), every item computing the same steps.
template <Part P = Part::Full, class Ex, class R, class MA> BATCHLAS_DEVICE_INLINE void lascl(const Ex& ex, R cfrom, R cto, const MA& a) {
    using T = typename MA::value_type;
    const R smlnum = safmin<R>(), bignum = R(1) / smlnum;
    R cfromc = cfrom, ctoc = cto;
    bool done = false;
    while (!done) {
        const R cfrom1 = cfromc * smlnum;
        R mul;
        if (cfrom1 == cfromc) {  // cfromc is Inf: a NaN or the exact quotient
            mul = ctoc / cfromc;
            done = true;
        } else {
            const R cto1 = ctoc / bignum;
            if (cto1 == ctoc) {  // ctoc is 0 or Inf
                mul = ctoc;
                done = true;
                cfromc = R(1);
            } else if (sycl::fabs(cfrom1) > sycl::fabs(ctoc) && ctoc != R(0)) {
                mul = smlnum;
                cfromc = cfrom1;
            } else if (sycl::fabs(cto1) > sycl::fabs(cfromc)) {
                mul = bignum;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
            }
        }
        for_each_owned_2d(ex, a.rows(), a.cols(), [&](int i, int j) {
            if (detail::in_part<P>(i, j)) a(i, j) = Arith<T>::scale_real(mul, static_cast<T>(a(i, j)));
        });
    }
}

// x := conj(x) (LAPACK xLACGV; the identity on real x).
template <class Ex, class X> BATCHLAS_DEVICE_INLINE void lacgv(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    if constexpr (Arith<T>::is_complex) {
        map(ex, x, [](const T& v) { return Arith<T>::conj(v); }, x);
    }
}

// LAPACK xTRTTP / xTPTTR: the UL triangle of the n x n A to / from packed storage AP (by
// columns, PackedTri<UL>, view.hh). 2-D flat ownership over A; elements outside UL untouched.
template <Uplo UL, class Ex, class MA, class AP> BATCHLAS_DEVICE_INLINE void trttp(const Ex& ex, const MA& a, const AP& ap) {
    const int n = a.rows();
    constexpr Part P = UL == Uplo::Lower ? Part::Lower : Part::Upper;
    for_each_owned_2d(ex, n, n, [&](int i, int j) {
        if (detail::in_part<P>(i, j)) ap(PackedTri<UL>::index(i, j, n)) = a(i, j);
    });
}
template <Uplo UL, class Ex, class AP, class MA> BATCHLAS_DEVICE_INLINE void tpttr(const Ex& ex, const AP& ap, const MA& a) {
    const int n = a.rows();
    constexpr Part P = UL == Uplo::Lower ? Part::Lower : Part::Upper;
    for_each_owned_2d(ex, n, n, [&](int i, int j) {
        if (detail::in_part<P>(i, j)) a(i, j) = ap(PackedTri<UL>::index(i, j, n));
    });
}

}  // namespace batchlas::device

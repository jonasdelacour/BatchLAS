// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension: the ops the CTA (right-looking, SLM-resident) and LPANEL (left-looking,
// row-per-item) Cholesky kernels are written in. Every op reproduces the expression tree of the
// hand-written code it replaced (ArithFma), so a port is SASS-identical, not merely equivalent.
//
//   ArithFma<T>          complex mac = four explicit FMAs, mul = two (Arith<T> contracts otherwise)
//   DynItems             runtime-sized Cyclic owner set (a launch-time stride), no collectives
//   HermLower / SlmMat   operands: a Hermitian matrix seen through its lower triangle; local tiles
//   lacpy_herm_lower / lacpy_lower / lacpy_conj_pad / fill        elementwise
//   potf2_sg_slm / potf2_panel_rows                                LAPACK potf2 (Lower)
//   trsm_rowfrag<Right, Lower, ConjTrans>                          L21 := A21 L11^-H
//   herk_tiles<Lower>, TriTilePrefix                               C := C - A A^H, register tiles
//   frag_load_row / frag_load_row_herm_lower / frag_store_row_herm_lower / copy_row_lower /
//   gemv_n_slm_bcast                                               row-owner (RowFrag) panels

#include <complex>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/level1.hh>

// Token-identical to src/sycl/kernel_attrs.hh, so either may be included first (acpp's SSCP JIT
// needs unroll(full); on DPC++ the hand-written kernels spelled `#pragma unroll`).
#ifndef BATCHLAS_UNROLL_FULL
#include <batchlas/backend_config.h>
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_UNROLL_FULL _Pragma("unroll")
#endif
#endif
#define BATCHLAS_DLA_UNROLL_FULL BATCHLAS_UNROLL_FULL

namespace batchlas::device::v2 {

// ---- arithmetic -------------------------------------------------------------------------------
namespace detail {
template <class T> concept ReImPod = requires(T a) {
    a.re;
    a.im;
};
}  // namespace detail

template <class T> struct ArithFma {
    static_assert(std::is_floating_point_v<T>, "v2::ArithFma: real, std::complex or a {re, im} POD");
    using real_t = T;
    static constexpr bool is_complex = false;
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) { acc = sycl::fma(a, b, acc); }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return a - b; }
    BATCHLAS_DLA_INLINE static T conj(T a) { return a; }
    BATCHLAS_DLA_INLINE static T re(T a) { return a; }
    BATCHLAS_DLA_INLINE static T from_real(T r) { return r; }
    BATCHLAS_DLA_INLINE static T mul_real(T a, T s) { return a * s; }
    BATCHLAS_DLA_INLINE static T div_real(T a, T d) { return a / d; }
    // p + b c in ONE expression, so -ffp-contract=on fuses it exactly as a hand-written one.
    BATCHLAS_DLA_INLINE static T add_mul(T p, T b, T c) { return p + b * c; }
};

template <class R> struct ArithFma<std::complex<R>> {
    using T = std::complex<R>;
    using real_t = R;
    static constexpr bool is_complex = true;
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        R re = sycl::fma(a.real(), b.real(), acc.real());
        re = sycl::fma(-a.imag(), b.imag(), re);
        R im = sycl::fma(a.real(), b.imag(), acc.imag());
        im = sycl::fma(a.imag(), b.real(), im);
        acc = T(re, im);
    }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        return T(sycl::fma(a.real(), b.real(), -a.imag() * b.imag()), sycl::fma(a.real(), b.imag(), a.imag() * b.real()));
    }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return T(a.real() - b.real(), a.imag() - b.imag()); }
    BATCHLAS_DLA_INLINE static T conj(T a) { return T(a.real(), -a.imag()); }
    BATCHLAS_DLA_INLINE static R re(T a) { return a.real(); }
    BATCHLAS_DLA_INLINE static T from_real(R r) { return T(r, R(0)); }
    BATCHLAS_DLA_INLINE static T mul_real(T a, R s) { return T(a.real() * s, a.imag() * s); }
    BATCHLAS_DLA_INLINE static T div_real(T a, R d) { return T(a.real() / d, a.imag() / d); }
    BATCHLAS_DLA_INLINE static T add_mul(T p, T b, T c) {
        return T(p.real() + (b.real() * c.real() - b.imag() * c.imag()), p.imag() + (b.real() * c.imag() + b.imag() * c.real()));
    }
};

// The POD device scalar (sycl_device::Cx): the exact forms of fma_acc / dev_mul / dev_sub.
template <class T>
    requires detail::ReImPod<T>
struct ArithFma<T> {
    using R = std::remove_cvref_t<decltype(std::declval<T&>().re)>;
    using real_t = R;
    static constexpr bool is_complex = true;
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        acc.re = sycl::fma(a.re, b.re, acc.re);
        acc.re = sycl::fma(-a.im, b.im, acc.re);
        acc.im = sycl::fma(a.re, b.im, acc.im);
        acc.im = sycl::fma(a.im, b.re, acc.im);
    }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        return T{sycl::fma(a.re, b.re, -a.im * b.im), sycl::fma(a.re, b.im, a.im * b.re)};
    }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return T{a.re - b.re, a.im - b.im}; }
    BATCHLAS_DLA_INLINE static T conj(T a) { return T{a.re, -a.im}; }
    BATCHLAS_DLA_INLINE static R re(T a) { return a.re; }
    BATCHLAS_DLA_INLINE static T from_real(R r) { return T{r, R(0)}; }
    BATCHLAS_DLA_INLINE static T mul_real(T a, R s) { return T{a.re * s, a.im * s}; }
    BATCHLAS_DLA_INLINE static T div_real(T a, R d) { return T{a.re / d, a.im / d}; }
    BATCHLAS_DLA_INLINE static T add_mul(T p, T b, T c) {
        return T{p.re + (b.re * c.re - b.im * c.im), p.im + (b.re * c.im + b.im * c.re)};
    }
};

namespace detail {
template <class Ar, class T> using fma_arith_t = std::conditional_t<std::is_void_v<Ar>, ArithFma<std::remove_cv_t<T>>, Ar>;
}  // namespace detail

// ---- executor ---------------------------------------------------------------------------------
// Item `id` of `n`, n a launch-time value. Prefer a static executor; this one keeps a strict port
// of a kernel whose stride is runtime (the potrf_cta L ladder) SASS-identical.
struct DynItems {
    static constexpr int team_size = 1;
    static constexpr bool sub_whole = true;
    int id;
    int n;
    BATCHLAS_DLA_INLINE int lid() const { return id; }
    BATCHLAS_DLA_INLINE bool leader() const { return id == 0; }
};

template <class Ex> BATCHLAS_DLA_INLINE int items_of(const Ex& ex) {
    if constexpr (std::is_same_v<Ex, DynItems>) return ex.n;
    else return Ex::size;
}

// Found by ADL from the level-1 ops (map, scal, copy, ...), so they run on DynItems unchanged.
template <class F> BATCHLAS_DLA_INLINE void for_each_owned(const DynItems& ex, int n, F&& f) {
    for (int i = ex.id; i < n; i += ex.n) f(i);
}

// ---- operands ---------------------------------------------------------------------------------
// The lower triangle (i >= c) of a Hermitian matrix stored in either triangle, chosen at run time:
// p[i + c ld], or conj(p[c + i ld]) when upper. A = U^H U is the lower recurrence on this view.
template <class T> struct HermLower {
    using value_type = std::remove_cv_t<T>;
    using Ar = ArithFma<value_type>;
    T* p;
    int ld;
    bool upper;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int c) const { return upper ? Ar::conj(p[c + i * ld]) : p[i + c * ld]; }
    BATCHLAS_DLA_INLINE void store(int i, int c, value_type v) const {
        if (upper) p[c + i * ld] = Ar::conj(v);
        else p[i + c * ld] = v;
    }
};

// A column-major tile in work-group local memory, addressed through a local-space multi_ptr so
// every access stays LDS/STS even in a body that is not inlined (generic T* reads are LD.E).
template <class T, class R = Dyn, class C = Dyn, class Ld = Dyn> struct SlmMat {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = Mem::Local;
    using ptr_t = sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>;
    ptr_t p;
    R m;
    C n;
    Ld ld;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[i + j * int(ld)]; }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
};
template <class T> BATCHLAS_DLA_INLINE auto local_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <class T> BATCHLAS_DLA_INLINE SlmMat<T> slm_mat(T* p, int m, int n, int ld) {
    return {local_ptr(p), Dyn{m}, Dyn{n}, Dyn{ld}};
}
template <int M, int N, class T> BATCHLAS_DLA_INLINE SlmMat<T, Ext<M>, Ext<N>, Ext<M>> slm_mat(T* p) {
    return {local_ptr(p), {}, {}, {}};
}

namespace detail {
template <class V, class T> BATCHLAS_DLA_INLINE void put(const V& v, int i, int c, const T& x) {
    if constexpr (requires { v.store(i, c, x); }) v.store(i, c, x);
    else v(i, c) = x;
}
}  // namespace detail

// ---- elementwise ------------------------------------------------------------------------------
template <class Ex, class X, class S> BATCHLAS_DLA_INLINE void fill(const Ex& ex, const X& x, S s) {
    for_each_owned(ex, x.size(), [&](int i) { x(i) = s; });
}

// dst(i, c) := src(i, c) for c <= i < n, diagonal made real (Hermitian input), and 0 in every
// other row i < dst.rows(): the whole dst.rows() x n block (pad rows too) is written once.
template <class Ar = void, class Ex, class MS, class MD>
BATCHLAS_DLA_INLINE void lacpy_herm_lower(const Ex& ex, const MS& src, const MD& dst, int n) {
    using D = typename MD::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    for (int c = 0; c < n; ++c) {
        for_each_owned(ex, dst.rows(), [&](int i) {
            D v{};
            if (i < n && i >= c) {
                v = src(i, c);
                if (i == c) v = A::from_real(A::re(v));
            }
            dst(i, c) = v;
        });
    }
}

// dst(i, c) := src(i, c) for c <= i < n; the strict upper triangle is neither read nor written.
template <class Ex, class MS, class MD> BATCHLAS_DLA_INLINE void lacpy_lower(const Ex& ex, const MS& src, const MD& dst, int n) {
    for (int c = 0; c < n; ++c) {
        for_each_owned(ex, n, [&](int i) {
            if (i >= c) detail::put(dst, i, c, src(i, c));
        });
    }
}

// dst := conj(src) zero-padded to dst's static NB x NB (src r x c, r, c <= NB); the NB*NB
// elements are distributed flat (e -> (e mod NB, e / NB)).
template <int NB, class Ar = void, class Ex, class MS, class MD>
BATCHLAS_DLA_INLINE void lacpy_conj_pad(const Ex& ex, const MS& src, const MD& dst) {
    using D = typename MD::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    const int r = src.rows(), c = src.cols();
    for_each_owned(ex, NB * NB, [&](int e) {
        const int i = e % NB;
        const int kk = e / NB;
        D v{};
        if (i < r && kk < c) v = A::conj(src(i, kk));
        dst(i, kk) = v;
    });
}

// ---- LAPACK: potf2, Lower -----------------------------------------------------------------------
// Right-looking unblocked Cholesky of the ib x ib block `a` (ib <= NB <= SG) on ONE sub-group:
// lane r holds row r in a Frag; column k is published to `a` (local) and read back after one
// sub-group barrier. diag(k) := L(k,k). At the first pivot that is not > 0 (NaN included):
// *info := info_base + k + 1 and nothing further changes. The NB barriers are unconditional,
// so the op may run under a branch uniform over the sub-group.
template <int NB, class Ar = void, int SG, class MA, class DV>
BATCHLAS_DLA_INLINE void potf2_sg_slm(const SubGroup<SG>& ex, const MA& a, int ib, const DV& diag, int* info, int info_base) {
    using D = typename MA::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    using R = typename A::real_t;
    static_assert(NB <= SG, "v2::potf2_sg_slm: one lane per row");
    const int lane = ex.lid();
    Frag<D, NB> d;
    BATCHLAS_DLA_UNROLL_FULL
    for (int c = 0; c < NB; ++c) d[c] = (lane < ib && c < ib && c <= lane) ? a(lane, c) : D{};

    bool alive = true;
    BATCHLAS_DLA_UNROLL_FULL
    for (int k = 0; k < NB; ++k) {
        const R akk = sycl::select_from_group(ex.sg, A::re(d[k]), static_cast<std::uint32_t>(k));
        bool active = alive && (k < ib);
        if (active && !(akk > R(0))) {
            if (lane == 0) *info = info_base + k + 1;
            active = false;
            alive = false;
        }
        const R dkk = active ? sycl::sqrt(akk) : R(1);
        const R r = R(1) / dkk;  // not rsqrt: rsqrt.approx is not the reference
        if (active) {
            if (lane == k) d[k] = A::from_real(dkk);
            else if (lane > k && lane < ib) d[k] = A::mul_real(d[k], r);
            if (lane < ib && lane >= k) a(lane, k) = d[k];
            if (lane == 0) diag(k) = dkk;
        }
        sycl::group_barrier(ex.sg);
        if (active) {
            BATCHLAS_DLA_UNROLL_FULL
            for (int c = k + 1; c < NB; ++c) {
                if (c < ib && lane < ib && lane >= c) d[c] = A::sub(d[c], A::mul(d[k], A::conj(a(c, k))));
            }
        }
    }
}

// Right-looking unblocked Cholesky of the n x kb panel P whose diagonal block starts at row c0,
// one item per row (row = ex.lid(); rows outside (c0, n) idle). `bar` is the caller's barrier over
// every item reading P, called three times per column (pivot read / its overwrite, scale, rank-1
// update). `ok` is sticky; the first bad pivot sets *info := c0 + i + 1 (leader). Returns ok.
template <class Ar = void, class Ex, class PM, class Bar>
BATCHLAS_DLA_INLINE bool potf2_panel_rows(const Ex& ex, const PM& p, int c0, int kb, int n, bool ok, int* info, Bar&& bar) {
    using D = typename PM::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    using R = typename A::real_t;
    const int row = ex.lid();
    for (int i = 0; i < kb; ++i) {
        const R d = A::re(p(c0 + i, i));
        const bool bad = ok && !(d > R(0));
        if (bad) {
            ok = false;
            if (ex.leader()) *info = c0 + i + 1;
        }
        bar();  // WAR: every item read the pivot cell above, item c0 + i overwrites it below
        if (ok) {
            const R dkk = sycl::sqrt(d);
            const R rs = R(1) / dkk;  // not rsqrt: rsqrt.approx is not the reference
            if (row == c0 + i) p(row, i) = A::from_real(dkk);
            else if (row > c0 + i && row < n) p(row, i) = A::mul_real(p(row, i), rs);
        }
        bar();
        if (ok && row > c0 + i && row < n) {
            const D lri = p(row, i);
            for (int c = i + 1; c < kb; ++c) {
                const D ljc = p(c0 + c, i);
                p(row, c) = A::sub(p(row, c), A::mul(lri, A::conj(ljc)));
            }
        }
        bar();
    }
    return ok;
}

// ---- level 3 ----------------------------------------------------------------------------------
// X := X L^{-H} for the m x kb block X, one row at a time in a Frag<NB> (rows Cyclic over ex, no
// barrier); L is kb x kb lower (kb <= NB) whose diagonal is read from the real vector `ldiag`.
// Divides by the diagonal, as reference trsm does (potf2 scales by a reciprocal).
template <Side S, Uplo UL, Transpose TR, int NB, class Ar = void, class Ex, class ML, class DV, class MX>
BATCHLAS_DLA_INLINE void trsm_rowfrag(const Ex& ex, const ML& l, const DV& ldiag, const MX& x, int kb) {
    static_assert(S == Side::Right && UL == Uplo::Lower && TR == Transpose::ConjTrans,
                  "v2::trsm_rowfrag: only Right / Lower / ConjTrans is implemented");
    using D = typename MX::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    const int m = x.rows();
    for (int row = ex.lid(); row < m; row += items_of(ex)) {
        Frag<D, NB> xr;
        BATCHLAS_DLA_UNROLL_FULL
        for (int c = 0; c < NB; ++c) xr[c] = (c < kb) ? x(row, c) : D{};
        BATCHLAS_DLA_UNROLL_FULL
        for (int c = 0; c < NB; ++c) {
            if (c < kb) {
                D s = xr[c];
                BATCHLAS_DLA_UNROLL_FULL
                for (int q = 0; q < NB; ++q) {
                    if (q < c) s = A::sub(s, A::mul(xr[q], A::conj(l(c, q))));
                }
                xr[c] = A::div_real(s, ldiag(c));
            }
        }
        BATCHLAS_DLA_UNROLL_FULL
        for (int c = 0; c < NB; ++c) {
            if (c < kb) x(row, c) = xr[c];
        }
    }
}

// Column-major enumeration of the lower tiles (rt >= ct) of an Rt x Rt tile grid through a prefix
// table off[c] = c Rt0 - c(c-1)/2 built once for Rt0 >= Rt and rescaled by dR = Rt0 - Rt, so a
// shrinking trailing matrix reuses it. Decoding is a binary search over the tile columns.
struct TileRC {
    int rt;
    int ct;
};
struct TriTilePrefix {
    const int* off;
    int Rt;
    int dR;
    BATCHLAS_DLA_INLINE int count() const { return Rt * (Rt + 1) / 2; }
    BATCHLAS_DLA_INLINE TileRC operator()(int t) const {
        int lo = 0, hi = Rt - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (off[mid] - mid * dR <= t) lo = mid;
            else hi = mid - 1;
        }
        return {lo + (t - (off[lo] - lo * dR)), lo};
    }
};
template <class Ex> BATCHLAS_DLA_INLINE void tri_tile_prefix_init(const Ex& ex, int* off, int Rt0) {
    for (int c = ex.lid(); c <= Rt0; c += items_of(ex)) off[c] = c * Rt0 - ((c * (c - 1)) >> 1);
}

// C := C - A A^H on the lower triangle (A m x k, C m x m) in TS x TS register tiles distributed
// Cyclic over ex in map's order; diagonal tiles are trimmed element-wise (r >= c), so the strict
// upper triangle is never written. Complex: the diagonal is stored real (LAPACK herk). No barrier.
template <Uplo UL, int TS, class Ar = void, class Ex, class AS, class MA, class BS, class MC, class Map>
BATCHLAS_DLA_INLINE void herk_tiles(const Ex& ex, AS, const MA& a, BS, const MC& c, const Map& map) {
    static_assert(UL == Uplo::Lower, "v2::herk_tiles: only Uplo::Lower is implemented");
    static_assert(std::is_same_v<AS, NegOne> && std::is_same_v<BS, One>, "v2::herk_tiles: alpha = NegOne, beta = One");
    using D = typename MC::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    const int m = a.rows(), k = a.cols();
    const int ntiles = map.count();
    for (int t = ex.lid(); t < ntiles; t += items_of(ex)) {
        const TileRC rc = map(t);
        const int r0 = rc.rt * TS;
        const int c0 = rc.ct * TS;
        D acc[TS][TS];
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < TS; ++i) {
            BATCHLAS_DLA_UNROLL_FULL
            for (int j = 0; j < TS; ++j) acc[i][j] = D{};
        }
        for (int p = 0; p < k; ++p) {
            D va[TS], vb[TS];
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < TS; ++i) va[i] = (r0 + i < m) ? a(r0 + i, p) : D{};
            BATCHLAS_DLA_UNROLL_FULL
            for (int j = 0; j < TS; ++j) vb[j] = (c0 + j < m) ? a(c0 + j, p) : D{};
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < TS; ++i) {
                BATCHLAS_DLA_UNROLL_FULL
                for (int j = 0; j < TS; ++j) A::mac(acc[i][j], va[i], A::conj(vb[j]));
            }
        }
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < TS; ++i) {
            BATCHLAS_DLA_UNROLL_FULL
            for (int j = 0; j < TS; ++j) {
                const int ri = r0 + i;
                const int cj = c0 + j;
                if (ri < m && cj < m && ri >= cj) {
                    D v = A::sub(c(ri, cj), acc[i][j]);
                    if constexpr (A::is_complex) {
                        if (ri == cj) v = A::from_real(A::re(v));
                    }
                    c(ri, cj) = v;
                }
            }
        }
    }
}

// ---- row-owner (RowFrag) panel ops: the calling item owns row `row` --------------------------
// f(i) := a(row, c0 + i) for i < kb, 0 for kb <= i < NB.
template <int NB, class MA, class F> BATCHLAS_DLA_INLINE void frag_load_row(F& f, const MA& a, int row, int c0, int kb) {
    using D = typename MA::value_type;
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < NB; ++i) f[i] = (i < kb) ? a(row, c0 + i) : D{};
}

// f(i) := a(row, c0 + i) where live, i < kb and the element is in the lower triangle (row >= c0+i),
// diagonal made real; 0 elsewhere (the strict upper triangle is never read).
template <int NB, class Ar = void, class MA, class F>
BATCHLAS_DLA_INLINE void frag_load_row_herm_lower(F& f, const MA& a, int row, int c0, int kb, bool live) {
    using D = typename MA::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < NB; ++i) {
        D v{};
        if (live && i < kb && row >= c0 + i) {
            v = a(row, c0 + i);
            if (row == c0 + i) v = A::from_real(A::re(v));
        }
        f[i] = v;
    }
}

// p(row, i) := g(i) for i < kb where row >= c0 + i (diagonal made real), 0 above the diagonal:
// column i of p is global column c0 + i of a lower-stored Hermitian block.
template <int NB, class Ar = void, class PM, class G>
BATCHLAS_DLA_INLINE void frag_store_row_herm_lower(const PM& p, int row, int c0, int kb, G&& g) {
    using D = typename PM::value_type;
    using A = detail::fma_arith_t<Ar, D>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < NB; ++i) {
        if (i < kb) {
            D v{};
            if (row >= c0 + i) {
                v = g(i);
                if (row == c0 + i) v = A::from_real(A::re(v));  // a mac(x, conj(x)) residue
            }
            p(row, i) = v;
        }
    }
}

// dst(row, c0 + i) := src(row, i) for i < kb with row >= c0 + i.
template <class MS, class MD> BATCHLAS_DLA_INLINE void copy_row_lower(const MS& src, const MD& dst, int row, int c0, int kb) {
    for (int i = 0; i < kb; ++i) {
        if (row >= c0 + i) dst(row, c0 + i) = src(row, i);
    }
}

// y(i) += sum_kk B(i, kk) x(kk) for the NB x NB column-major tile B in local memory, read as
// 16-byte vectors (every item reads the same column: a broadcast); each y(i) sums kk ascending.
template <int NB, class Ar = void, class D, class FY, class FX>
BATCHLAS_DLA_INLINE void gemv_n_slm_bcast(FY& y, const FX& x, D* b) {
    using A = detail::fma_arith_t<Ar, D>;
    using V = sycl::vec<float, 4>;
    constexpr int VW = 16 / static_cast<int>(sizeof(D));
    static_assert((NB * sizeof(D)) % 16 == 0, "v2::gemv_n_slm_bcast: a column must be whole 16-byte vectors");
    const auto bv = local_ptr(reinterpret_cast<V*>(b));
    BATCHLAS_DLA_UNROLL_FULL
    for (int kk = 0; kk < NB; ++kk) {
        D col[NB];
        BATCHLAS_DLA_UNROLL_FULL
        for (int v = 0; v < NB / VW; ++v) {
            const V t = bv[(kk * NB) / VW + v];
            __builtin_memcpy(&col[v * VW], &t, 16);
        }
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < NB; ++i) A::mac(y[i], x[kk], col[i]);
    }
}

}  // namespace batchlas::device::v2

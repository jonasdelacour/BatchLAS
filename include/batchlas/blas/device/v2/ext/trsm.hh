// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, triangular-solve extension (in-kernel trsv / trsm). General ops:
//   FmaArith<T>          Arith with chained-FMA mac/mul plus Smith div/recip and finite()
//   DynWorkGroup         executor whose width is a launch-time value (a runtime WG ladder)
//   CanonTri             a stored triangle read as the canonical lower factor of op(A)
//   CanonRhs, FlipRows   right-hand sides in canonical row order (per-rhs / 2-D form)
//   tp_pack_lower        packed lower triangle with identity padding to a static order
//   tp_diag_recip        guarded diagonal reciprocals; any non-finite one sets a flag
//   TpDiag, GuardedRecip the guarded divide-by-diagonal (reciprocal, or division fallback)
//   trsv_lower_rows      per-item forward substitution of rows [s0, s0+STEP) into a Frag
//   lacpy_tile_in/out    global <-> padded SLM tile, one tile column per item, zero fill
//   frag_*               fragment loads / stores against those views
//   partition_bcast      a lane's value to its whole partition (select_from_group)
//   trsm_lower_rowfrag   forward substitution on a Partition, row r in lane r (RowFrag)
// Diag and the canonical flags stay run-time values: one kernel serves all 24 trsm cases.

#include <cstddef>
#include <type_traits>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/dla.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

// ---- arithmetic -----------------------------------------------------------------------------
// The solve arithmetic: complex mac is four chained FMAs (re, then im), complex mul two FMAs,
// division and reciprocal are Smith's (no overflow past sqrt(max)). Real types are one FMA.
template <class T> struct FmaArith : Arith<T> {
    using B = Arith<T>;
    using R = typename B::real_t;
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        if constexpr (B::is_complex) {
            R re = B::re(acc), im = B::im(acc);
            re = sycl::fma(B::re(a), B::re(b), re);
            re = sycl::fma(-B::im(a), B::im(b), re);
            im = sycl::fma(B::re(a), B::im(b), im);
            im = sycl::fma(B::im(a), B::re(b), im);
            acc = B::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        if constexpr (B::is_complex) {
            return B::make(sycl::fma(B::re(a), B::re(b), -B::im(a) * B::im(b)),
                           sycl::fma(B::re(a), B::im(b), B::im(a) * B::re(b)));
        } else {
            return a * b;
        }
    }
    BATCHLAS_DLA_INLINE static T recip(T d) {
        if constexpr (B::is_complex) {
            const R dr = B::re(d), di = B::im(d);
            if (sycl::fabs(dr) >= sycl::fabs(di)) {
                const R r = di / dr;
                const R den = sycl::fma(di, r, dr);
                return B::make(R(1) / den, -r / den);
            }
            const R r = dr / di;
            const R den = sycl::fma(dr, r, di);
            return B::make(r / den, R(-1) / den);
        } else {
            return T(1) / d;
        }
    }
    BATCHLAS_DLA_INLINE static T div(T a, T b) {
        if constexpr (B::is_complex) {
            const R ar = B::re(a), ai = B::im(a), br = B::re(b), bi = B::im(b);
            if (sycl::fabs(br) >= sycl::fabs(bi)) {
                const R r = bi / br;
                const R den = sycl::fma(bi, r, br);
                return B::make(sycl::fma(ai, r, ar) / den, sycl::fma(-ar, r, ai) / den);
            }
            const R r = br / bi;
            const R den = sycl::fma(br, r, bi);
            return B::make(sycl::fma(ar, r, ai) / den, sycl::fma(ai, r, -ar) / den);
        } else {
            return a / b;
        }
    }
    BATCHLAS_DLA_INLINE static bool finite(T a) {
        if constexpr (B::is_complex) return sycl::isfinite(B::re(a)) && sycl::isfinite(B::im(a));
        else return sycl::isfinite(a);
    }
};

// ---- DynWorkGroup -----------------------------------------------------------------------------
// The whole work-group, width chosen at launch. No static `size`, so core ops that fold the
// stride do not compile on it; the ops below read the stride through stride_of().
struct DynWorkGroup {
    sycl::nd_item<1> it;
    int size;
    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    // sycl::group_barrier: work-group scope over all memory, the legacy kernels' barrier.
    BATCHLAS_DLA_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
};

namespace detail {
template <class Ex> BATCHLAS_DLA_INLINE int stride_of(const Ex& ex) {
    if constexpr (std::is_same_v<Ex, DynWorkGroup>) return ex.size;
    else return Ex::size;
}
}  // namespace detail

// ---- views ----------------------------------------------------------------------------------
// Element (s, t) of the canonical lower factor Lc of a stored triangle: rho(s) = fwd ? s :
// n-1-s maps canonical to stored indices, Left keeps (row, col) and Right swaps them, trans
// reads A(c, r), conj conjugates (the diagonal too). Offsets are 64-bit; Ld is the ld's type.
template <class T, bool Left, class Ld = int> struct CanonTri {
    using value_type = std::remove_cv_t<T>;
    T* a;
    Ld ld;
    int n;
    bool fwd, trans, conj;
    BATCHLAS_DLA_INLINE value_type operator()(int s, int t) const {
        const int rs = fwd ? s : (n - 1 - s);
        const int rt = fwd ? t : (n - 1 - t);
        const int r = Left ? rs : rt;
        const int c = Left ? rt : rs;
        value_type v = trans ? a[c + static_cast<std::ptrdiff_t>(r) * ld] : a[r + static_cast<std::ptrdiff_t>(c) * ld];
        if (conj) v = Arith<value_type>::conj(v);
        return v;
    }
};

// Right-hand side u in canonical order: element s is p[b0 + s*ds + u*du]. Left: column u of
// B, rows flipped when !fwd; Right: row u of B, columns flipped.
template <class T, bool Left> struct CanonRhs {
    T* p;
    std::ptrdiff_t b0, ds, du;
    struct Col {
        T* p;
        std::ptrdiff_t b0, ds, udu;
        BATCHLAS_DLA_INLINE T& operator()(int s) const { return p[b0 + static_cast<std::ptrdiff_t>(s) * ds + udu]; }
    };
    BATCHLAS_DLA_INLINE CanonRhs(T* p_, int n, int ld, bool fwd) : p(p_) {
        const std::ptrdiff_t unit_s = Left ? 1 : static_cast<std::ptrdiff_t>(ld);
        du = Left ? static_cast<std::ptrdiff_t>(ld) : 1;
        b0 = fwd ? 0 : static_cast<std::ptrdiff_t>(n - 1) * unit_s;
        ds = fwd ? unit_s : -unit_s;
    }
    BATCHLAS_DLA_INLINE Col rhs(int u) const { return {p, b0, ds, u * du}; }
};

// An m x q column-major matrix with its rows read in canonical order: (s, c) is
// p[rho(s) + c*ld], rho(s) = fwd ? s : m-1-s.
template <class T, class Ld = int> struct FlipRows {
    T* p;
    Ld ld;
    int m, q;
    bool fwd;
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return q; }
    BATCHLAS_DLA_INLINE T& operator()(int s, int c) const {
        const int row = fwd ? s : (m - 1 - s);
        return p[static_cast<std::ptrdiff_t>(row) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

// ---- packed lower triangle ------------------------------------------------------------------
// Row-major by row: (s, t <= s) at s(s+1)/2 + t.
BATCHLAS_DLA_INLINE constexpr int tp_idx(int s, int t) { return s * (s + 1) / 2 + t; }

// ap := diag(L(0:n, 0:n), I) as an N x N packed lower triangle; no barrier.
template <int N, class Ex, class L, class T> BATCHLAS_DLA_INLINE void tp_pack_lower(const Ex& ex, const L& l, int n, T* ap) {
    using Ar = Arith<T>;
    constexpr std::size_t elems = static_cast<std::size_t>(N) * (N + 1) / 2;
    const std::size_t stride = static_cast<std::size_t>(detail::stride_of(ex));
    for (std::size_t idx = ex.lid(); idx < elems; idx += stride) {
        int s = 0;
        while (tp_idx(s + 1, 0) <= static_cast<int>(idx)) ++s;
        const int t = static_cast<int>(idx) - tp_idx(s, 0);
        T v;
        if (s >= n || t >= n) v = (s == t) ? Ar::one() : Ar::zero();
        else v = l(s, t);
        ap[idx] = v;
    }
}

// rd(s) := 1 / ap(s, s) for s < n (1 when unit or s >= n). A non-finite reciprocal is
// replaced by 1 and stores 1 into *bad (work-group-scope relaxed atomic); no barrier.
template <int N, class Ar, class Ex, class T>
BATCHLAS_DLA_INLINE void tp_diag_recip(const Ex& ex, const T* ap, int n, bool unit, T* rd, int* bad) {
    const int stride = detail::stride_of(ex);
    for (int s = ex.lid(); s < N; s += stride) {
        T r = Ar::one();
        if (s < n && !unit) {
            const T d = ap[tp_idx(s, s)];
            r = Ar::recip(d);
            if (!Ar::finite(r)) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(bad[0])
                    .store(1);
                r = Ar::one();
            }
        }
        rd[s] = r;
    }
}

// v / L(s, s) for a packed L: v * rd(s), or a true division once any reciprocal was
// non-finite (divide is group-uniform); unit leaves v.
template <class T> struct TpDiag {
    const T* ap;
    const T* rd;
    bool unit, divide;
    template <class Ar> BATCHLAS_DLA_INLINE T apply(int s, T v) const {
        if (!unit) v = divide ? Ar::div(v, ap[tp_idx(s, s)]) : Ar::mul(v, rd[s]);
        return v;
    }
};

// ---- per-item forward substitution ----------------------------------------------------------
// x(s) := (b(s) - L(s, 0:s) x(0:s)) / L(s, s) for s in [s0, s0+STEP), s < N; b(s) is read
// after the row's dot product. Every index into x is a compile-time constant once unrolled.
template <int N, int STEP, class Ar, class T, class Rhs>
BATCHLAS_DLA_INLINE void trsv_lower_rows(Frag<T, N>& x, int s0, const T* ap, const TpDiag<T>& d, Rhs&& b) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int j = 0; j < STEP; ++j) {
        const int s = s0 + j;
        if (s >= N) continue;
        T acc = Ar::zero();
        BATCHLAS_DLA_UNROLL_FULL
        for (int t = 0; t < N; ++t) {
            if (t < s) Ar::mac(acc, ap[tp_idx(s, t)], x[t]);
        }
        const T rhs = b(s);
        x[s] = d.template apply<Ar>(s, Ar::sub(rhs, acc));
    }
}

// ---- tiles ------------------------------------------------------------------------------------
// tile(r, c) := src(r0 + r, c0 + c) inside src, else 0, for r < TR and c < executor width;
// tile is column-major with leading dimension LDT. The flat index is distributed Cyclic.
template <int TR, int LDT, class Ex, class Src, class T>
BATCHLAS_DLA_INLINE void lacpy_tile_in(const Ex& ex, const Src& src, int r0, int c0, T* tile) {
    const int stride = detail::stride_of(ex);
    for (int i = ex.lid(); i < TR * stride; i += stride) {
        const int r = i % TR;
        const int c = i / TR;
        const int s = r0 + r;
        const int col = c0 + c;
        T v = Arith<T>::zero();
        if (s < src.rows() && col < src.cols()) v = src(s, col);
        tile[c * LDT + r] = v;
    }
}

// dst(r0 + r, c0 + c) := tile(r, c) where that element is inside dst.
template <int TR, int LDT, class Ex, class T, class Dst>
BATCHLAS_DLA_INLINE void lacpy_tile_out(const Ex& ex, const T* tile, const Dst& dst, int r0, int c0) {
    const int stride = detail::stride_of(ex);
    for (int i = ex.lid(); i < TR * stride; i += stride) {
        const int r = i % TR;
        const int c = i / TR;
        const int s = r0 + r;
        const int col = c0 + c;
        if (s < dst.rows() && col < dst.cols()) dst(s, col) = tile[c * LDT + r];
    }
}

// ---- fragments --------------------------------------------------------------------------------
// p[j] := x[s0 + j] for j < TR, s0 + j < N.
template <int TR, int N, class T> BATCHLAS_DLA_INLINE void frag_copy_out(const Frag<T, N>& x, int s0, T* p) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int j = 0; j < TR; ++j) {
        const int s = s0 + j;
        if (s < N) p[j] = x[s];
    }
}

// y(s) := x[s] for s < n when ok.
template <int N, class T, class Y> BATCHLAS_DLA_INLINE void frag_store(const Frag<T, N>& x, bool ok, int n, const Y& y) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int s = 0; s < N; ++s) {
        if (ok && s < n) y(s) = x[s];
    }
}

// f[t] := alpha * L(r, t) for t < r when ok, else 0: row r of the strict lower part.
template <int N, class AS, class L, class T = typename L::value_type>
BATCHLAS_DLA_INLINE Frag<T, N> frag_row_lower(AS alpha_, const L& l, int r, bool ok) {
    const auto alpha = as_tag(alpha_);
    Frag<T, N> f;
    BATCHLAS_DLA_UNROLL_FULL
    for (int t = 0; t < N; ++t) {
        T v = Arith<T>::zero();
        if (ok && t < r) v = tag_mul(alpha, l(r, t));
        f[t] = v;
    }
    return f;
}

// x[j] := alpha * src(r, c0 + j) for c0 + j < src.cols() when ok, else 0.
template <int QC, class Ar, class T, class Src>
BATCHLAS_DLA_INLINE Frag<T, QC> frag_load_row(T alpha, const Src& src, int r, int c0, bool ok) {
    Frag<T, QC> x;
    BATCHLAS_DLA_UNROLL_FULL
    for (int j = 0; j < QC; ++j) {
        T v = Ar::zero();
        if (ok && c0 + j < src.cols()) v = Ar::mul(alpha, src(r, c0 + j));
        x[j] = v;
    }
    return x;
}

// dst(r, c0 + j) := x[j] for c0 + j < dst.cols() when ok.
template <int QC, class T, class Dst>
BATCHLAS_DLA_INLINE void frag_store_row(const Frag<T, QC>& x, const Dst& dst, int r, int c0, bool ok) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int j = 0; j < QC; ++j) {
        if (ok && c0 + j < dst.cols()) dst(r, c0 + j) = x[j];
    }
}

// ---- partition (RowFrag) forward substitution -------------------------------------------------
// Value of lane src of this item's partition (lockstep sub-group partitions only).
template <int P, class T> BATCHLAS_DLA_INLINE T partition_bcast(const Partition<P>& ex, T v, int src) {
    const int lane = static_cast<int>(ex.part.get_local_linear_id());
    const int base = (lane / P) * P;
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        return A::make(sycl::select_from_group(ex.part, A::re(v), base + src),
                       sycl::select_from_group(ex.part, A::im(v), base + src));
    } else {
        return sycl::select_from_group(ex.part, v, base + src);
    }
}

// The diagonal of row lid(): its reciprocal, and divide = any reciprocal in the partition is
// non-finite (Scope::Sub OR, shuffles only).
template <class T> struct GuardedRecip {
    T d, rinv;
    bool divide;
};
template <class Ar, int P, class T> BATCHLAS_DLA_INLINE GuardedRecip<T> guarded_recip(const Partition<P>& ex, T d) {
    const T rinv = Ar::recip(d);
    int bad = Ar::finite(rinv) ? 0 : 1;
    bad = ex.team_reduce(bad, sycl::bit_or<int>());
    return {d, rinv, bad != 0};
}

// X := L^{-1} X for X (P x QC, row r in lane r) on the partition; nl holds -L(r, 0:r) (the
// AXPY multipliers), d the row's diagonal. Rows n..P-1 are identity padding. Step s scales
// row s by 1/L(s, s), then every lane does x += nl[s] * x_s (x_s broadcast from lane s). The
// division branch is sub-group-uniform. Unroll = 1 keeps the step loop rolled (nl indexed at
// run time: local memory, fewer registers).
template <int Unroll, class Ar, int P, int QC, class T>
BATCHLAS_DLA_INLINE void trsm_lower_rowfrag(const Partition<P>& ex, const Frag<T, P>& nl, const GuardedRecip<T>& d, int n,
                                            bool unit, Frag<T, QC>& x) {
    const int r = ex.lid();
    const bool any_divide = sycl::any_of_group(ex.part, d.divide);
#pragma unroll Unroll
    for (int s = 0; s < P; ++s) {
        if (s >= n) {
            if constexpr (Unroll == 1) break;
            else continue;
        }
        if (r == s && !unit) {
            if (any_divide) [[unlikely]] {
                BATCHLAS_DLA_UNROLL_FULL
                for (int j = 0; j < QC; ++j) x[j] = d.divide ? Ar::div(x[j], d.d) : Ar::mul(x[j], d.rinv);
            } else {
                BATCHLAS_DLA_UNROLL_FULL
                for (int j = 0; j < QC; ++j) x[j] = Ar::mul(x[j], d.rinv);
            }
        }
        const T l = nl[s];
        BATCHLAS_DLA_UNROLL_FULL
        for (int j = 0; j < QC; ++j) {
            const T xs = partition_bcast(ex, x[j], s);
            Ar::mac(x[j], l, xs);
        }
    }
}

}  // namespace batchlas::device::v2

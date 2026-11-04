// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to trsm (kernels/trsm_native.cc V1 CTA, kernels/trsm_sg_left.cc);
// shared ops (arith::Fma, RtGroup, Partition, Frag load_row / store_row) come from core. Diag
// and the canonical flags stay run-time values: one kernel serves all 24 trsm cases.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::sycl_trsm::dla {

namespace v2 = batchlas::device;

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
        if (conj) v = v2::Arith<value_type>::conj(v);
        return v;
    }
    // Lc(s, s), addressed as rho(s) * (ld + 1). A kernel body calls this, never operator():
    // see scripts/check_kernel_entry_calls.py (the SemaSYCL kernel-body quirk).
    BATCHLAS_DLA_INLINE value_type diag(int s) const {
        const std::ptrdiff_t rs = fwd ? s : (n - 1 - s);
        value_type v = a[rs + rs * static_cast<std::ptrdiff_t>(ld)];
        if (conj) v = v2::Arith<value_type>::conj(v);
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

// Columns c0.. of v: (r, j) is v(r, c0 + j), a reference (the core store_row target).
template <class V> struct ColsFrom {
    V v;
    int c0;
    BATCHLAS_DLA_INLINE auto& operator()(int r, int j) const { return v(r, c0 + j); }
};
template <class V> ColsFrom(V, int) -> ColsFrom<V>;

// Columns c0.. of v read through f: (r, j) is f(v(r, c0 + j)), a value (the core load_row
// source; f scales by alpha or negates, so the product is formed only for a live element).
template <class V, class F> struct MapCols {
    V v;
    int c0;
    F f;
    BATCHLAS_DLA_INLINE auto operator()(int r, int j) const { return f(v(r, c0 + j)); }
};
template <class V, class F> MapCols(V, int, F) -> MapCols<V, F>;

// ---- packed lower triangle ------------------------------------------------------------------
// Row-major by row: (s, t <= s) at s(s+1)/2 + t.
BATCHLAS_DLA_INLINE constexpr int tp_idx(int s, int t) { return s * (s + 1) / 2 + t; }

// ap := diag(L(0:n, 0:n), I) as an N x N packed lower triangle; no barrier.
template <int N, class Ex, class L, class T> BATCHLAS_DLA_INLINE void tp_pack_lower(const Ex& ex, const L& l, int n, T* ap) {
    using Ar = v2::Arith<T>;
    constexpr std::size_t elems = static_cast<std::size_t>(N) * (N + 1) / 2;
    const std::size_t stride = static_cast<std::size_t>(v2::detail::size_of(ex));
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
    const int stride = v2::detail::size_of(ex);
    for (int s = ex.lid(); s < N; s += stride) {
        T r = Ar::one();
        if (s < n && !unit) {
            const T d = ap[tp_idx(s, s)];
            r = Ar::recip(d);
            if (!Ar::isfinite(r)) {
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
// after the row's dot product. Not core trsv: the dot is accumulated then subtracted (core
// msubs into x(s)), the diagonal is TpDiag's guarded reciprocal, and b is read lazily.
template <int N, int STEP, class Ar, class T, class Rhs>
BATCHLAS_DLA_INLINE void trsv_lower_rows(v2::Frag<T, N>& x, int s0, const T* ap, const TpDiag<T>& d, Rhs&& b) {
    BATCHLAS_UNROLL_FULL
    for (int j = 0; j < STEP; ++j) {
        const int s = s0 + j;
        if (s >= N) continue;
        T acc = Ar::zero();
        BATCHLAS_UNROLL_FULL
        for (int t = 0; t < N; ++t) {
            if (t < s) Ar::mac(acc, ap[tp_idx(s, t)], x[t]);
        }
        const T rhs = b(s);
        x[s] = d.template apply<Ar>(s, Ar::sub(rhs, acc));
    }
}

// ---- Side::Left staging tile ------------------------------------------------------------------
// tile(r, c) := src(r0 + r, c0 + c) inside src, else 0, for r < TR and c < executor width;
// tile is column-major with leading dimension LDT. The flat index is distributed Cyclic.
template <int TR, int LDT, class Ex, class Src, class T>
BATCHLAS_DLA_INLINE void lacpy_tile_in(const Ex& ex, const Src& src, int r0, int c0, T* tile) {
    const int stride = v2::detail::size_of(ex);
    for (int i = ex.lid(); i < TR * stride; i += stride) {
        const int r = i % TR;
        const int c = i / TR;
        const int s = r0 + r;
        const int col = c0 + c;
        T v = v2::Arith<T>::zero();
        if (s < src.rows() && col < src.cols()) v = src(s, col);
        tile[c * LDT + r] = v;
    }
}

// dst(r0 + r, c0 + c) := tile(r, c) where that element is inside dst.
template <int TR, int LDT, class Ex, class T, class Dst>
BATCHLAS_DLA_INLINE void lacpy_tile_out(const Ex& ex, const T* tile, const Dst& dst, int r0, int c0) {
    const int stride = v2::detail::size_of(ex);
    for (int i = ex.lid(); i < TR * stride; i += stride) {
        const int r = i % TR;
        const int c = i / TR;
        const int s = r0 + r;
        const int col = c0 + c;
        if (s < dst.rows() && col < dst.cols()) dst(s, col) = tile[c * LDT + r];
    }
}

// p[j] := x[s0 + j] for j < TR, s0 + j < N (a window of the Frag; core store has no offset).
template <int TR, int N, class T> BATCHLAS_DLA_INLINE void frag_copy_out(const v2::Frag<T, N>& x, int s0, T* p) {
    BATCHLAS_UNROLL_FULL
    for (int j = 0; j < TR; ++j) {
        const int s = s0 + j;
        if (s < N) p[j] = x[s];
    }
}

// ---- partition (RowFrag) forward substitution -------------------------------------------------
// The diagonal of row lid(): its reciprocal, and divide = any reciprocal in the partition is
// non-finite (a bit_or team_reduce, shuffles only).
template <class T> struct GuardedRecip {
    T d, rinv;
    bool divide;
};
template <class Ar, int P, class T> BATCHLAS_DLA_INLINE GuardedRecip<T> guarded_recip(const v2::Partition<P>& ex, T d) {
    const T rinv = Ar::recip(d);
    int bad = Ar::isfinite(rinv) ? 0 : 1;
    bad = ex.team_reduce(bad, sycl::bit_or<int>());
    return {d, rinv, bad != 0};
}

// X := L^{-1} X for X (P x QC, row r in lane r) on the partition; nl holds -L(r, 0:r) (the
// AXPY multipliers), d the row's diagonal. Rows n..P-1 are identity padding. Step s scales
// row s by 1/L(s, s), then every lane does x += nl[s] * x_s (x_s = ex.bcast from lane s). The
// division branch is sub-group-uniform. Unroll = 1 keeps the step loop rolled (nl indexed at
// run time: local memory, fewer registers). A distributed trsv: core trsv is Thread-only.
template <int Unroll, class Ar, int P, int QC, class T>
BATCHLAS_DLA_INLINE void trsm_lower_rowfrag(const v2::Partition<P>& ex, const v2::Frag<T, P>& nl, const GuardedRecip<T>& d,
                                            int n, bool unit, v2::Frag<T, QC>& x) {
    // lane % P, not ex.lid() (lane & (P-1)): the AND form shifts ptxas' register allocation in
    // the unrolled P=16 kernels by 3-4% of the instructions.
    const int r = static_cast<int>(ex.part.get_local_linear_id()) % P;
    const bool any_divide = sycl::any_of_group(ex.part, d.divide);
#pragma unroll Unroll
    for (int s = 0; s < P; ++s) {
        if (s >= n) {
            if constexpr (Unroll == 1) break;
            else continue;
        }
        if (r == s && !unit) {
            if (any_divide) [[unlikely]] {
                BATCHLAS_UNROLL_FULL
                for (int j = 0; j < QC; ++j) x[j] = d.divide ? Ar::div(x[j], d.d) : Ar::mul(x[j], d.rinv);
            } else {
                BATCHLAS_UNROLL_FULL
                for (int j = 0; j < QC; ++j) x[j] = Ar::mul(x[j], d.rinv);
            }
        }
        const T l = nl[s];
        BATCHLAS_UNROLL_FULL
        for (int j = 0; j < QC; ++j) {
            const T xs = ex.bcast(x[j], static_cast<std::uint32_t>(s));
            Ar::mac(x[j], l, xs);
        }
    }
}

}  // namespace batchlas::sycl_trsm::dla

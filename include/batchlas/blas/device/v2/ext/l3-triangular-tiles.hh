// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, level-3 tile extension: the stager / micro-kernel / epilogue components of the
// tuned syrk, herk, syr2k, her2k and trmm tile kernels (src/ops/{syrk,syr2k,trmm}/kernels).
//   Packet4, ld4, st4             4-element packets; 128-bit forms only for a 4-byte T
//   mac_value                     acc + a b by value, accumulator first (the kernels' rounding)
//   RegTile, fill_tile, ger_acc   register accumulator tile and its rank-1 update (register ger)
//   Bands4, frag_ld4              a lane's 4-wide fragment bands and their gather from a k-row
//   NoSwizzle, Swizzle<Packs>     shared packet placement: packet q of k-row kk at q ^ (kk / 4)
//   Panel, PacketStager           one packet per item: Rows x K panels into k-major shared tiles
//   stage_swz_kc, stage_swz_cc    flat-loop swizzled staging from a k- or column-contiguous source
//   tri_tile_count, tri_tile_decode   the packed triangular tile grid
//   trmat_op_at, sym_at           op(A)(i, p) of a triangular A; A(r, c) of a one-triangle symmetric A
//   FullMask, TriMask, store_elem, epilogue_cm, epilogue_tri_packets   C := alpha acc + beta C
// These ops are the tuned kernels' code, moved here. They keep its arithmetic so that a port is
// bit-identical and SASS-identical (strict mode of the SASS-parity gate):
//  - epilogues use T's own operators and a runtime beta == 0 test that skips the read;
//  - global addresses take a 64-bit column offset (v2 Mat indexes in int32);
//  - uplo, diag and trans stay runtime flags, as in the kernels.
// Barriers belong to the caller. A stager writes shared memory and an inner loop reads it, so
// the caller puts a local barrier between them and another before the next stage.

#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/core.hh>
#include <batchlas/blas/device/v2/operand.hh>

// The kernels' BATCHLAS_UNROLL_FULL spelling (src/sycl/kernel_attrs.hh), reachable from include/.
#ifndef BATCHLAS_DLA_UNROLL_FULL
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

// ---- packets ---------------------------------------------------------------------------
template <class T> struct alignas(4 * sizeof(T)) Packet4 {
    T v[4];
};

// Four contiguous elements, by value. Only a 4-byte T is read as one 128-bit access: a wider
// reinterpret assumes an alignment local_accessor never promised.
// evidence: docs/perf/level3.md#level-3-the-complex-mac-and-return-by-value-rules
template <class T> BATCHLAS_DLA_INLINE Packet4<T> ld4(const T* p) {
    if constexpr (sizeof(T) == sizeof(float)) {
        return *reinterpret_cast<const Packet4<T>*>(p);
    } else {
        Packet4<T> out;
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < 4; ++i) out.v[i] = p[i];
        return out;
    }
}

template <class T> BATCHLAS_DLA_INLINE void st4(T* p, const Packet4<T>& in) {
    if constexpr (sizeof(T) == sizeof(float)) {
        *reinterpret_cast<Packet4<T>*>(p) = in;
    } else {
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < 4; ++i) p[i] = in.v[i];
    }
}

// ---- scalar helpers --------------------------------------------------------------------
// acc + a b with the accumulator added first, by value: a T& into the accumulator array spills
// it, and Arith::mac rounds (a b) before adding.
// evidence: docs/perf/level3.md#level-3-the-complex-mac-and-return-by-value-rules
template <class T> BATCHLAS_DLA_INLINE T mac_value(const T& acc, const T& a, const T& b) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        const R ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
        return A::make(A::re(acc) + ar * br - ai * bi, A::im(acc) + ar * bi + ai * br);
    } else {
        return acc + a * b;
    }
}

template <bool C, class T> BATCHLAS_DLA_INLINE T conj_when(const T& v) {
    if constexpr (C && Arith<std::remove_cv_t<T>>::is_complex) return Arith<std::remove_cv_t<T>>::conj(v);
    else return v;
}

template <class T> BATCHLAS_DLA_INLINE T conj_rt(const T& v, bool c) {
    if constexpr (Arith<std::remove_cv_t<T>>::is_complex) return c ? Arith<std::remove_cv_t<T>>::conj(v) : v;
    else return v;
}

// ---- register micro-kernel -------------------------------------------------------------
template <class T, int M, int N> struct RegTile {
    T v[M][N];
};

template <class T, int M, int N> BATCHLAS_DLA_INLINE void fill_tile(RegTile<T, M, N>& t, T x) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < M; ++i) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int j = 0; j < N; ++j) t.v[i][j] = x;
    }
}

// acc(i, j) += x(i) y(j): BLAS ger on a register tile (mac_value rounding).
template <class T, int M, int N, class LX, class LY>
BATCHLAS_DLA_INLINE void ger_acc(RegTile<T, M, N>& acc, const Frag<T, M, LX>& x, const Frag<T, N, LY>& y) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < M; ++i) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int j = 0; j < N; ++j) acc.v[i][j] = mac_value(acc.v[i][j], x.v[i], y.v[j]);
    }
}

// Layout of a lane's 4 NB fragment rows (or columns): band b holds b Sep + lane Stride + [0, 4).
// Bands4<2, 64, 4> is the 128x128 kernels' split pair; Bands4<NB, 4, 4 NB> is contiguous.
template <int NB, int Sep, int Stride> struct Bands4 {
    static_assert(Sep % 4 == 0 && Stride % 4 == 0, "v2: fragment bands are whole packets");
    static constexpr int bands = NB, size = 4 * NB;
    BATCHLAS_DLA_INLINE static int row(int lane, int r) { return (r / 4) * Sep + lane * Stride + r % 4; }
    BATCHLAS_DLA_INLINE static int packet(int lane, int b) { return b * (Sep / 4) + lane * (Stride / 4); }
};

struct NoSwizzle {
    BATCHLAS_DLA_INLINE static int at(int packet, int) { return packet; }
};
// Packet q of k-row kk lives at q ^ (kk / 4): an aligned, unpadded stride alone puts every
// staging lane in one bank, and a whole-packet rotation keeps 16-byte fragment loads aligned.
template <int Packs> struct Swizzle {
    static_assert(Packs > 0 && (Packs & (Packs - 1)) == 0, "v2: the swizzle needs a power-of-two packet count");
    BATCHLAS_DLA_INLINE static int at(int packet, int kk) { return packet ^ ((kk >> 2) & (Packs - 1)); }
};

// f(4 b + e) := row[4 Sw::at(L::packet(lane, b), kk) + e], conjugated when Conj: a lane's
// fragment of one k-row of a shared tile.
template <class L, bool Conj, class Sw = NoSwizzle, class T, int N, class FL>
BATCHLAS_DLA_INLINE void frag_ld4(Frag<T, N, FL>& f, const T* row, int lane, int kk = 0) {
    static_assert(N == L::size, "v2: the fragment and its band layout differ in size");
    BATCHLAS_DLA_UNROLL_FULL
    for (int b = 0; b < L::bands; ++b) {
        const Packet4<T> v = ld4(row + (Sw::at(L::packet(lane, b), kk) << 2));
        BATCHLAS_DLA_UNROLL_FULL
        for (int e = 0; e < 4; ++e) f.v[b * 4 + e] = conj_when<Conj>(v.v[e]);
    }
}

// ---- staging ---------------------------------------------------------------------------
struct AsIs {
    template <class T> BATCHLAS_DLA_INLINE T operator()(T v) const { return v; }
};
// s v when On, else v (her2k stages A pre-scaled).
template <class T, bool On> struct StageScale {
    T s;
    BATCHLAS_DLA_INLINE T operator()(T v) const {
        if constexpr (On) return s * v;
        else return v;
    }
};

// Rows [row0, row0 + Rows) of the output index of a column-major source with leading dimension
// ld, staged through f into the k-major shared tile dst (element (r, l) at dst[l Rows + r]).
template <class T, class F = AsIs> struct Panel {
    const T* src;
    int ld;
    T* dst;
    int row0;
    F f{};
};

// One packet per item: a Rows x K panel per call, at k0. Trans: the source is k x n (k
// contiguous) and is transposed on the way in; else n x k. Aligned: whole tiles with 16-byte
// packets; else bounds-checked against (n, k) and zero-filled, so the inner loop has no checks.
// Two panels in one call interleave their loads and stores (syrk's A at two row offsets).
template <int Rows, int K, int Threads, bool Trans, bool Aligned> struct PacketStager {
    static_assert(Rows * K == 4 * Threads, "v2: the stager moves one packet per item");
    int s_row;
    int s_l;

    BATCHLAS_DLA_INLINE explicit PacketStager(int tid)
        : s_row(Trans ? tid / (K / 4) : (tid % (Rows / 4)) * 4), s_l(Trans ? (tid % (K / 4)) * 4 : tid / (Rows / 4)) {}

    template <class T, class F> BATCHLAS_DLA_INLINE Packet4<T> load_t(const Panel<T, F>& p, int k0) const {
        return ld4(p.src + (k0 + s_l) + static_cast<std::ptrdiff_t>(p.row0 + s_row) * p.ld);
    }
    template <class T, class F> BATCHLAS_DLA_INLINE void put_t(const Panel<T, F>& p, int i, const Packet4<T>& v) const {
        p.dst[(s_l + i) * Rows + s_row] = p.f(v.v[i]);
    }
    template <class T, class F> BATCHLAS_DLA_INLINE void copy_n(const Panel<T, F>& p, int k0) const {
        Packet4<T> v = ld4(p.src + (p.row0 + s_row) + static_cast<std::ptrdiff_t>(k0 + s_l) * p.ld);
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < 4; ++i) v.v[i] = p.f(v.v[i]);
        st4(&p.dst[s_l * Rows + s_row], v);
    }
    template <class T, class F> BATCHLAS_DLA_INLINE void elem_t(const Panel<T, F>& p, int gk, int i, int n, int k) const {
        const int row = p.row0 + s_row;
        p.dst[(s_l + i) * Rows + s_row] =
            (gk < k && row < n) ? p.f(p.src[gk + static_cast<std::ptrdiff_t>(row) * p.ld]) : T(0);
    }
    template <class T, class F> BATCHLAS_DLA_INLINE void elem_n(const Panel<T, F>& p, int gk, int i, int n, int k) const {
        const int row = p.row0 + s_row + i;
        p.dst[s_l * Rows + s_row + i] =
            (row < n && gk < k) ? p.f(p.src[row + static_cast<std::ptrdiff_t>(gk) * p.ld]) : T(0);
    }

    template <class T, class F> BATCHLAS_DLA_INLINE void operator()(int k0, int n, int k, const Panel<T, F>& p) const {
        if constexpr (Aligned && Trans) {
            const Packet4<T> v = load_t(p, k0);
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < 4; ++i) put_t(p, i, v);
        } else if constexpr (Aligned) {
            copy_n(p, k0);
        } else if constexpr (Trans) {
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < 4; ++i) elem_t(p, k0 + s_l + i, i, n, k);
        } else {
            const int gk = k0 + s_l;
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < 4; ++i) elem_n(p, gk, i, n, k);
        }
    }
    template <class T, class F0, class F1>
    BATCHLAS_DLA_INLINE void operator()(int k0, int n, int k, const Panel<T, F0>& p, const Panel<T, F1>& q) const {
        if constexpr (Aligned && Trans) {
            const Packet4<T> vp = load_t(p, k0);
            const Packet4<T> vq = load_t(q, k0);
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < 4; ++i) {
                put_t(p, i, vp);
                put_t(q, i, vq);
            }
        } else if constexpr (Aligned) {
            copy_n(p, k0);
            copy_n(q, k0);
        } else if constexpr (Trans) {
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < 4; ++i) {
                const int gk = k0 + s_l + i;
                elem_t(p, gk, i, n, k);
                elem_t(q, gk, i, n, k);
            }
        } else {
            const int gk = k0 + s_l;
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < 4; ++i) {
                elem_n(p, gk, i, n, k);
                elem_n(q, gk, i, n, k);
            }
        }
    }
};

// Flat-loop staging of a KC x Cols k-major shared tile (row stride Cols, Swizzle<Cols / 4>)
// from a source contiguous along k: an item takes four consecutive k of one column. fn(c, gk)
// is the zero-padded source element at global column c = c0 + local and global k gk = k0 + local.
template <int Cols, int KC, int Threads, class T, class Fn>
BATCHLAS_DLA_INLINE void stage_swz_kc(T* sh, int tid, int c0, int k0, Fn&& fn) {
    using Sw = Swizzle<Cols / 4>;
    for (int flat = tid; flat < KC * (Cols / 4); flat += Threads) {
        const int c = flat / (KC / 4);
        const int kk0 = (flat % (KC / 4)) * 4;
        const int gc = c0 + c;
        const int phys = (Sw::at(c >> 2, kk0) << 2) | (c & 3);
        BATCHLAS_DLA_UNROLL_FULL
        for (int e = 0; e < 4; ++e) sh[(kk0 + e) * Cols + phys] = fn(gc, k0 + kk0 + e);
    }
}

// The same tile from a source contiguous along the column index: an item takes four adjacent
// columns of one k-row, as four scalar stores, or one packet store when PacketStore.
template <int Cols, int KC, int Threads, bool PacketStore, class T, class Fn>
BATCHLAS_DLA_INLINE void stage_swz_cc(T* sh, int tid, int c0, int k0, Fn&& fn) {
    using Sw = Swizzle<Cols / 4>;
    constexpr int Packs = Cols / 4;
    for (int flat = tid; flat < KC * Packs; flat += Threads) {
        const int kk = flat / Packs;
        const int cl0 = (flat % Packs) * 4;
        const int gk = k0 + kk;
        const int phys = Sw::at(cl0 >> 2, kk) << 2;
        if constexpr (PacketStore) {
            Packet4<T> packet;
            BATCHLAS_DLA_UNROLL_FULL
            for (int e = 0; e < 4; ++e) packet.v[e] = fn(c0 + cl0 + e, gk);
            st4(&sh[kk * Cols + phys], packet);
        } else {
            BATCHLAS_DLA_UNROLL_FULL
            for (int e = 0; e < 4; ++e) sh[kk * Cols + phys + e] = fn(c0 + cl0 + e, gk);
        }
    }
}

// ---- triangular tile grid --------------------------------------------------------------
BATCHLAS_DLA_INLINE int tri_tile_count(int tiles_per_side) { return tiles_per_side * (tiles_per_side + 1) / 2; }

// Tile t of the lower triangle, packed row-major: (bi, bj) with bj <= bi. sqrt only seeds bi;
// the two correction loops make it exact whatever the device's sqrt rounds to. Uplo::Upper is
// the same set with the pair swapped.
BATCHLAS_DLA_INLINE void tri_tile_decode(int tile, int& bi, int& bj) {
#if BATCHLAS_HAS_DEVICE_FP64
    bi = static_cast<int>((sycl::sqrt(8.0 * tile + 1.0) - 1.0) * 0.5);
#else  // Metal has no double; a float seed serves as well
    bi = static_cast<int>((sycl::sqrt(8.0f * static_cast<float>(tile) + 1.0f) - 1.0f) * 0.5f);
#endif
    while (bi > 0 && bi * (bi + 1) / 2 > tile) --bi;
    while ((bi + 1) * (bi + 2) / 2 <= tile) ++bi;
    bj = tile - bi * (bi + 1) / 2;
}

// ---- element reads -----------------------------------------------------------------------
// op(A)(i, p) of an n x n triangular A, where lower_eff says op(A) is lower: zero outside the
// triangle and outside [0, n)^2, a literal 1 on a unit diagonal. Trans reads A(p, i), and
// conjugates it when conj (ignored without Trans).
template <bool Trans, class T>
BATCHLAS_DLA_INLINE T trmat_op_at(const T* A, int lda, int n, int i, int p, bool lower_eff, bool unit, bool conj) {
    T value = T(0);
    if (i < n && p < n) {
        if (i == p) {
            if constexpr (Trans) value = unit ? T(1) : conj_rt(A[i + static_cast<std::ptrdiff_t>(i) * lda], conj);
            else value = unit ? T(1) : A[i + static_cast<std::ptrdiff_t>(i) * lda];
        } else if (lower_eff ? (p < i) : (p > i)) {
            if constexpr (Trans) value = conj_rt(A[p + static_cast<std::ptrdiff_t>(i) * lda], conj);
            else value = A[i + static_cast<std::ptrdiff_t>(p) * lda];
        }
    }
    return value;
}

// A(r, c) of a symmetric (Herm: Hermitian) matrix whose named triangle is stored (lower: r >= c);
// the other half is read from its mirror, conjugated for Herm, whose diagonal is real.
template <bool Herm = false, class T> BATCHLAS_DLA_INLINE T sym_at(const T* A, int lda, int r, int c, bool lower) {
    const bool named = lower ? r >= c : r <= c;
    if constexpr (Herm && Arith<std::remove_cv_t<T>>::is_complex) {
        using Ar = Arith<std::remove_cv_t<T>>;
        if (r == c) return Ar::make(Ar::re(A[r + static_cast<std::ptrdiff_t>(c) * lda]), real_t<T>(0));
        return named ? A[r + static_cast<std::ptrdiff_t>(c) * lda] : Ar::conj(A[c + static_cast<std::ptrdiff_t>(r) * lda]);
    } else {
        return named ? A[r + static_cast<std::ptrdiff_t>(c) * lda] : A[c + static_cast<std::ptrdiff_t>(r) * lda];
    }
}

// ---- epilogues ---------------------------------------------------------------------------
// alpha acc + beta y in T's operators; a runtime beta == 0 never reads y, a Zero beta never does.
template <class T> BATCHLAS_DLA_INLINE T axpby_value(T alpha, T acc, T beta, const T* y) {
    return beta == T(0) ? alpha * acc : alpha * acc + beta * *y;
}
template <class T> BATCHLAS_DLA_INLINE T axpby_value(T alpha, T acc, Zero, const T*) { return alpha * acc; }

// Which elements of an output tile are stored.
struct FullMask {};
// A diagonal tile stores only the named triangle; Herm also drops Im(C(i, i)), which is zero
// only up to rounding (a residue makes C not quite Hermitian, which eigensolvers notice).
template <bool Herm> struct TriMask {
    bool on_diag;
    bool lower;
};

template <class T, class B>
BATCHLAS_DLA_INLINE void store_elem(FullMask, T* C, int ldc, int row, int col, T alpha, T acc, B beta) {
    T* p = &C[row + static_cast<std::ptrdiff_t>(col) * ldc];
    *p = axpby_value(alpha, acc, beta, p);
}
template <bool Herm, class T, class B>
BATCHLAS_DLA_INLINE void store_elem(TriMask<Herm> mask, T* C, int ldc, int row, int col, T alpha, T acc, B beta) {
    if (mask.on_diag && (mask.lower ? row < col : row > col)) return;
    T* p = &C[row + static_cast<std::ptrdiff_t>(col) * ldc];
    T out = axpby_value(alpha, acc, beta, p);
    if constexpr (Herm) {
        using Ar = Arith<std::remove_cv_t<T>>;
        if (row == col) out = Ar::make(Ar::re(out), real_t<T>(0));
    }
    *p = out;
}

// C(m0 + LR::row(lr, i), n0 + LC::row(lc, j)) := alpha acc(i, j) + beta C(...), column by
// column, skipping rows >= m and columns >= n.
template <class LR, class LC, class Mask, class T, int M, int N, class B>
BATCHLAS_DLA_INLINE void epilogue_cm(Mask mask, const RegTile<T, M, N>& acc, T* C, int ldc, int m0, int n0, int lr, int lc,
                                     int m, int n, T alpha, B beta) {
    static_assert(M == LR::size && N == LC::size, "v2: the tile and its band layouts differ in size");
    BATCHLAS_DLA_UNROLL_FULL
    for (int j = 0; j < N; ++j) {
        const int col = n0 + LC::row(lc, j);
        if (col >= n) continue;
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < M; ++i) {
            const int row = m0 + LR::row(lr, i);
            if (row >= m) continue;
            store_elem(mask, C, ldc, row, col, alpha, acc.v[i][j], beta);
        }
    }
}

// The square n x n output by row band: off the diagonal (Aligned: whole 16-byte-aligned tiles)
// a band of four rows is one packet; on the diagonal, or unaligned, it goes element by element.
template <bool Aligned, class LR, class LC, bool Herm, class T, int M, int N>
BATCHLAS_DLA_INLINE void epilogue_tri_packets(TriMask<Herm> mask, const RegTile<T, M, N>& acc, T* C, int ldc, int m0, int n0,
                                              int lr, int lc, int n, T alpha, T beta) {
    static_assert(M == LR::size && N == LC::size, "v2: the tile and its band layouts differ in size");
    BATCHLAS_DLA_UNROLL_FULL
    for (int band = 0; band < LR::bands; ++band) {
        const int gm = m0 + LR::row(lr, band * 4);
        BATCHLAS_DLA_UNROLL_FULL
        for (int j = 0; j < N; ++j) {
            const int gn = n0 + LC::row(lc, j);
            if constexpr (Aligned) {
                if (!mask.on_diag) {
                    T* p = &C[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                    Packet4<T> out;
                    if (beta == T(0)) {
                        BATCHLAS_DLA_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) out.v[i] = alpha * acc.v[band * 4 + i][j];
                    } else {
                        const Packet4<T> prior = ld4(const_cast<const T*>(p));
                        BATCHLAS_DLA_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) out.v[i] = alpha * acc.v[band * 4 + i][j] + beta * prior.v[i];
                    }
                    st4(p, out);
                    continue;
                }
            }
            BATCHLAS_DLA_UNROLL_FULL
            for (int i = 0; i < 4; ++i) {
                const int row = gm + i;
                if (row >= n || gn >= n) continue;
                store_elem(mask, C, ldc, row, gn, alpha, acc.v[band * 4 + i][j], beta);
            }
        }
    }
}

}  // namespace batchlas::device::v2

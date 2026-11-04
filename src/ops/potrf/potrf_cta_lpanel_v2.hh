// SPDX-License-Identifier: MIT
#pragma once
// The potrf CTA (right-looking, SLM-resident) and LPANEL (left-looking, row-per-item) kernels'
// private device LA v2 ops: the parts of the two Cholesky recurrences no other family uses.
// Everything shared comes from v2 core (executors, Mat / Vec / Frag, load_row / store_row, scal,
// lacpy, arith::Fma). Every op reproduces the expression tree of the hand-written code it
// replaced: arith::Fma is bit-equal to the src/sycl/device_scalar.hh forms (fma_acc, dev_mul,
// dev_sub, dev_mul_real, dev_div_real), and sub(mul()) stays two roundings, as dev_sub(dev_mul())
// was under -ffp-contract=on (core msub fuses, which is why trsm_rowfrag is not core trsv).
//
//   SlotItems                 runtime-sized slot of a packed work-group (LPANEL: G matrices x L)
//   HermLower                 a Hermitian matrix read through its lower triangle, uplo at run time
//   slm_mat / slm_tile        core Mat over a local-space pointer (LDS/STS, not generic LD/ST)
//   lacpy_herm_lower          Hermitian load, zero pad and real diagonal, every element once
//   lacpy_conj_pad            conj copy zero-padded to a static NB x NB tile
//   potf2_sg_slm / potf2_panel_rows     LAPACK potf2 (Lower) on a sub-group / on a row panel
//   trsm_rowfrag              L21 := A21 L11^-H, one row per item in a Frag
//   herk_tiles, TriTilePrefix C := C - A A^H on the lower triangle in register tiles
//   frag_load_row_herm_lower / frag_store_row_herm_lower / gemv_n_slm_bcast   LPANEL row ops

#include <src/kernels/common/dla_v2_scalar.hh>
#include <src/sycl/inv_sqrt.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <batchlas/device/level1.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::potrf_native::dla {

namespace v2 = batchlas::device;

// The arithmetic of both kernels: explicit fma chains, bit-equal to device_scalar.hh.
template <class D> using Ar = v2::arith_t<v2::arith::Fma, D>;


// ---- executor ---------------------------------------------------------------------------------
// Item `id` of a slot of `n` items, n a launch value. A packed LPANEL work-group holds G matrices
// of L items each, so the slot is neither the work-group (RtGroup<Group>) nor a sub-group.
struct SlotItems {
    static constexpr bool runtime = true;
    static constexpr bool sub_whole = true;
    int id;
    int n;
    BATCHLAS_DLA_INLINE int lid() const { return id; }
    BATCHLAS_DLA_INLINE int width() const { return n; }
    BATCHLAS_DLA_INLINE bool leader() const { return id == 0; }
};

// ---- operands ---------------------------------------------------------------------------------
// The lower triangle (i >= c) of a Hermitian matrix stored in either triangle, chosen at run time:
// p[i + c ld], or conj(p[c + i ld]) when upper. A = U^H U is the lower recurrence on this view.
template <class T> struct HermLower {
    using value_type = std::remove_cv_t<T>;
    T* p;
    int ld;
    bool upper;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int c) const {
        return upper ? Ar<value_type>::conj(p[c + static_cast<std::ptrdiff_t>(i) * ld]) : p[i + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

// A local tile through a local-space pointer, so every access stays LDS/STS even where a body is
// not inlined (the slot's tile is &acc[0] + slot * size: core's accessor base cannot carry it).
template <class T> using local_ptr_t = sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>;
template <class T>
using SlmMat = v2::Mat<T, v2::Dyn, v2::Dyn, v2::Dyn, v2::Mem::Local, v2::Cyclic, local_ptr_t<T>, std::ptrdiff_t>;
template <int M, int N, class T>
using SlmTile = v2::Mat<T, v2::Ext<M>, v2::Ext<N>, v2::Ext<M>, v2::Mem::Local, v2::Cyclic, local_ptr_t<T>, int>;

template <class T> BATCHLAS_DLA_INLINE local_ptr_t<T> local_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <class T> BATCHLAS_DLA_INLINE SlmMat<T> slm_mat(T* p, int m, int n, int ld) {
    return {local_ptr(p), v2::Dyn{m}, v2::Dyn{n}, v2::Dyn{ld}};
}
template <int M, int N, class T> BATCHLAS_DLA_INLINE SlmTile<M, N, T> slm_tile(T* p) { return {local_ptr(p), {}, {}, {}}; }

// ---- elementwise ------------------------------------------------------------------------------
// dst(i, c) := src(i, c) for c <= i < n, diagonal made real (Hermitian input), and 0 in every
// other row i < dst.rows(): the whole dst.rows() x n block (pad rows too) is written once.
template <class Ex, class MS, class MD> BATCHLAS_DLA_INLINE void lacpy_herm_lower(const Ex& ex, const MS& src, const MD& dst, int n) {
    using D = typename MD::value_type;
    using A = Ar<D>;
    for (int c = 0; c < n; ++c) {
        v2::for_each_owned(ex, dst.rows(), [&](int i) BATCHLAS_DLA_BODY {
            D v{};
            if (i < n && i >= c) {
                v = src(i, c);
                if (i == c) v = A::make(A::re(v), v2::real_t<D>(0));
            }
            dst(i, c) = v;
        });
    }
}

// dst := conj(src) zero-padded to dst's static NB x NB (src r x c, r, c <= NB); the NB*NB
// elements are distributed flat (e -> (e mod NB, e / NB)).
template <int NB, class Ex, class MS, class MD> BATCHLAS_DLA_INLINE void lacpy_conj_pad(const Ex& ex, const MS& src, const MD& dst) {
    using D = typename MD::value_type;
    const int r = src.rows(), c = src.cols();
    v2::for_each_owned(ex, NB * NB, [&](int e) BATCHLAS_DLA_BODY {
        const int i = e % NB;
        const int kk = e / NB;
        D v{};
        if (i < r && kk < c) v = Ar<D>::conj(src(i, kk));
        dst(i, kk) = v;
    });
}

// ---- LAPACK: potf2, Lower -----------------------------------------------------------------------
// Right-looking unblocked Cholesky of the ib x ib block `a` (ib <= NB <= SG) on ONE sub-group:
// lane r holds row r in a Frag; column k is published to `a` (local) and read back after one
// sub-group barrier. diag(k) := L(k,k). At the first pivot that is not > 0 (NaN included):
// *info := info_base + k + 1 and nothing further changes. The NB barriers are unconditional,
// so the op may run under a branch uniform over the sub-group.
namespace detail {
template <int NB, int SG, class MA, class DV>
BATCHLAS_DLA_HELPER void potf2_sg_slm_body(v2::SubGroup<SG> ex, MA a, int ib, DV diag, int* info, int info_base) {
    using D = typename MA::value_type;
    using A = Ar<D>;
    using R = v2::real_t<D>;
    static_assert(NB <= SG, "potf2_sg_slm: one lane per row");
    const int lane = ex.lid();
    v2::Frag<D, NB> d;
    BATCHLAS_UNROLL_FULL
    for (int c = 0; c < NB; ++c) d[c] = (lane < ib && c < ib && c <= lane) ? a(lane, c) : D{};

    bool alive = true;
    BATCHLAS_UNROLL_FULL
    for (int k = 0; k < NB; ++k) {
        const R akk = sycl::select_from_group(ex.sg, A::re(d[k]), static_cast<std::uint32_t>(k));
        bool active = alive && (k < ib);
        if (active && !(akk > R(0))) {
            if (lane == 0) *info = info_base + k + 1;
            active = false;
            alive = false;
        }
        const R dkk = active ? sycl::sqrt(akk) : R(1);
        const R r = ::batchlas::sycl_device::inv_sqrt(active ? akk : R(1), dkk);
        if (active) {
            if (lane == k) d[k] = A::make(dkk, R(0));
            else if (lane > k && lane < ib) d[k] = A::scale_real(r, d[k]);
            if (lane < ib && lane >= k) a(lane, k) = d[k];
            if (lane == 0) diag(k) = dkk;
        }
        sycl::group_barrier(ex.sg);
        if (active) {
            BATCHLAS_UNROLL_FULL
            for (int c = k + 1; c < NB; ++c) {
                if (c < ib && lane < ib && lane >= c) d[c] = A::sub(d[c], A::mul(d[k], A::conj(a(c, k))));
            }
        }
    }
}
}  // namespace detail

// Real T: the body is a by-value phase op the inliner takes after simplifying it alone, as the
// hand-written potf2 helper was (its column-0..3 1/sqrt shapes depend on it). Complex T: inlined
// at the call; out of line the by-value operands spill (STACK 64).
template <int NB, int SG, class MA, class DV>
BATCHLAS_DLA_INLINE void potf2_sg_slm(const v2::SubGroup<SG>& ex, const MA& a, int ib, const DV& diag, int* info, int info_base) {
    if constexpr (Ar<typename MA::value_type>::is_complex) {
        [[clang::always_inline]] detail::potf2_sg_slm_body<NB>(ex, a, ib, diag, info, info_base);
    } else {
        detail::potf2_sg_slm_body<NB>(ex, a, ib, diag, info, info_base);
    }
}

// Right-looking unblocked Cholesky of the n x kb panel P whose diagonal block starts at row c0,
// one item per row (row = ex.lid(); rows outside (c0, n) idle). `bar` is the caller's barrier over
// every item reading P, called three times per column (pivot read / its overwrite, scale, rank-1
// update). `ok` is sticky; the first bad pivot sets *info := c0 + i + 1 (leader). Returns ok.
template <class Ex, class PM, class Bar>
BATCHLAS_DLA_INLINE bool potf2_panel_rows(const Ex& ex, const PM& p, int c0, int kb, int n, bool ok, int* info, Bar&& bar) {
    using D = typename PM::value_type;
    using A = Ar<D>;
    using R = v2::real_t<D>;
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
            const R rs = ::batchlas::sycl_device::inv_sqrt(d, dkk);  // `ok`: d > 0, not NaN
            if (row == c0 + i) p(row, i) = A::make(dkk, R(0));
            else if (row > c0 + i && row < n) p(row, i) = A::scale_real(rs, p(row, i));
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
template <Side S, Uplo UL, Transpose TR, int NB, class Ex, class ML, class DV, class MX>
BATCHLAS_DLA_HELPER void trsm_rowfrag(const Ex& ex, const ML& l, const DV& ldiag, const MX& x, int kb) {
    static_assert(S == Side::Right && UL == Uplo::Lower && TR == Transpose::ConjTrans,
                  "trsm_rowfrag: only Right / Lower / ConjTrans is implemented");
    using D = typename MX::value_type;
    using A = Ar<D>;
    const int m = x.rows();
    for (int row = ex.lid(); row < m; row += v2::detail::size_of(ex)) {
        v2::Frag<D, NB> xr;
        v2::load_row(xr, x, row, kb);
        BATCHLAS_UNROLL_FULL
        for (int c = 0; c < NB; ++c) {
            if (c < kb) {
                D s = xr[c];
                BATCHLAS_UNROLL_FULL
                for (int q = 0; q < NB; ++q) {
                    if (q < c) s = A::sub(s, A::mul(xr[q], A::conj(l(c, q))));
                }
                xr[c] = A::div_real(s, ldiag(c));
            }
        }
        v2::store_row(xr, x, row, kb);
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
    for (int c = ex.lid(); c <= Rt0; c += v2::detail::size_of(ex)) off[c] = c * Rt0 - ((c * (c - 1)) >> 1);
}

// C := C - A A^H on the lower triangle (A m x k, C m x m) in TS x TS register tiles distributed
// Cyclic over ex in map's order; diagonal tiles are trimmed element-wise (r >= c), so the strict
// upper triangle is never written. Complex: the diagonal is stored real (LAPACK herk). No barrier.
template <Uplo UL, int TS, class Ex, class AS, class MA, class BS, class MC, class Map>
BATCHLAS_DLA_HELPER void herk_tiles(const Ex& ex, AS, const MA& a, BS, const MC& c, const Map& map) {
    static_assert(UL == Uplo::Lower, "herk_tiles: only Uplo::Lower is implemented");
    static_assert(std::is_same_v<AS, v2::NegOne> && std::is_same_v<BS, v2::One>, "herk_tiles: alpha = NegOne, beta = One");
    using D = typename MC::value_type;
    using A = Ar<D>;
    const int m = a.rows(), k = a.cols();
    const int ntiles = map.count();
    for (int t = ex.lid(); t < ntiles; t += v2::detail::size_of(ex)) {
        const TileRC rc = map(t);
        const int r0 = rc.rt * TS;
        const int c0 = rc.ct * TS;
        D acc[TS][TS];
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TS; ++i) {
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TS; ++j) acc[i][j] = D{};
        }
        for (int p = 0; p < k; ++p) {
            D va[TS], vb[TS];
            BATCHLAS_UNROLL_FULL
            for (int i = 0; i < TS; ++i) va[i] = (r0 + i < m) ? a(r0 + i, p) : D{};
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TS; ++j) vb[j] = (c0 + j < m) ? a(c0 + j, p) : D{};
            BATCHLAS_UNROLL_FULL
            for (int i = 0; i < TS; ++i) {
                BATCHLAS_UNROLL_FULL
                for (int j = 0; j < TS; ++j) A::mac(acc[i][j], va[i], A::conj(vb[j]));
            }
        }
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TS; ++i) {
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TS; ++j) {
                const int ri = r0 + i;
                const int cj = c0 + j;
                if (ri < m && cj < m && ri >= cj) {
                    D v = A::sub(c(ri, cj), acc[i][j]);
                    if constexpr (A::is_complex) {
                        if (ri == cj) v = A::make(A::re(v), v2::real_t<D>(0));
                    }
                    c(ri, cj) = v;
                }
            }
        }
    }
}

// ---- row-owner (RowFrag) panel ops: the calling item owns row `row` --------------------------
// f(i) := a(row, c0 + i) where live, i < kb and the element is in the lower triangle (row >= c0+i),
// diagonal made real; 0 elsewhere (the strict upper triangle is never read).
template <int NB, class MA, class F> BATCHLAS_DLA_INLINE void frag_load_row_herm_lower(F& f, const MA& a, int row, int c0, int kb, bool live) {
    using D = typename MA::value_type;
    using A = Ar<D>;
    BATCHLAS_UNROLL_FULL
    for (int i = 0; i < NB; ++i) {
        D v{};
        if (live && i < kb && row >= c0 + i) {
            v = a(row, c0 + i);
            if (row == c0 + i) v = A::make(A::re(v), v2::real_t<D>(0));
        }
        f[i] = v;
    }
}

// p(row, i) := g(i) for i < kb where row >= c0 + i (diagonal made real), 0 above the diagonal:
// column i of p is global column c0 + i of a lower-stored Hermitian block.
template <int NB, class PM, class G> BATCHLAS_DLA_INLINE void frag_store_row_herm_lower(const PM& p, int row, int c0, int kb, G&& g) {
    using D = typename PM::value_type;
    using A = Ar<D>;
    BATCHLAS_UNROLL_FULL
    for (int i = 0; i < NB; ++i) {
        if (i < kb) {
            D v{};
            if (row >= c0 + i) {
                v = g(i);
                if (row == c0 + i) v = A::make(A::re(v), v2::real_t<D>(0));  // a mac(x, conj(x)) residue
            }
            p(row, i) = v;
        }
    }
}

// y(i) += sum_kk B(i, kk) x(kk) for the NB x NB column-major tile B in local memory, read as
// 16-byte vectors (every item reads the same column: a broadcast); each y(i) sums kk ascending.
template <int NB, class D, class FY, class FX> BATCHLAS_DLA_INLINE void gemv_n_slm_bcast(FY& y, const FX& x, D* b) {
    using V = sycl::vec<float, 4>;
    constexpr int VW = 16 / static_cast<int>(sizeof(D));
    static_assert((NB * sizeof(D)) % 16 == 0, "gemv_n_slm_bcast: a column must be whole 16-byte vectors");
    const auto bv = local_ptr(reinterpret_cast<V*>(b));
    BATCHLAS_UNROLL_FULL
    for (int kk = 0; kk < NB; ++kk) {
        D col[NB];
        BATCHLAS_UNROLL_FULL
        for (int v = 0; v < NB / VW; ++v) {
            const V t = bv[(kk * NB) / VW + v];
            __builtin_memcpy(&col[v * VW], &t, 16);
        }
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < NB; ++i) Ar<D>::mac(y[i], x[kk], col[i]);
    }
}

}  // namespace batchlas::potrf_native::dla

// SPDX-License-Identifier: MIT
#pragma once
// In-group Level 3 (v22 design B.3): ONE matrix per work-group, any m, n, k, operands in global
// memory, the work-group's local memory as l3::Smem<Cfg>. Host op names and run-time argument
// order (Queue replaced by the executor); the host's run-time modes are template parameters:
//   gemm<TA, TB, Cfg>(ex, A, B, C, alpha, beta, smem)          C := alpha op(A) op(B) + beta C
//   symm / hemm<SD, UL, Cfg>(ex, A, B, C, alpha, beta, smem)   C := alpha A B + beta C (Left), alpha B A + beta C
//   syrk<UL, TR, Cfg>(ex, A, C, alpha, beta, smem)             C := alpha op(A) op(A)^T + beta C, UL triangle
//   herk<UL, TR, Cfg>(ex, A, C, alpha, beta, smem)             C := alpha op(A) op(A)^H + beta C, real alpha, beta
//   syr2k<UL, TR, Cfg>(ex, A, B, C, alpha, beta, smem)         C := alpha (op(A) op(B)^T + op(B) op(A)^T) + beta C
//   her2k<UL, TR, Cfg>(ex, A, B, C, alpha, beta, smem)         C := alpha op(A) op(B)^H + conj(alpha) op(B) op(A)^H + beta C
//   trmm<SD, UL, TR, DG, Cfg>(ex, A, B, C, alpha, smem)        C := alpha op(A) B (Left), alpha B op(A); B unchanged
//   trsm<SD, UL, TR, DG, Cfg, NB>(ex, A, B, alpha, smem)       B := alpha op(A)^-1 B (Left), alpha B op(A)^-1
// Every op is one tile body (l3::reg_tile_mma + store_tile_staged, the host reg route's) run
// over the matrix's tiles by for_each_tile. A float gemm with a Cfg of a reg_configs row (Stages,
// UnrollK and tile equal, predicated leg) gives the host route's bits. Collective: every item of
// the executor calls with the same arguments; inputs visible on entry (the caller's barrier);
// outputs visible to the work-group on return (the op ends on a global barrier). Operands: Mat
// over T* in global memory, device::item(view, b), or l3::Global; in-item offsets are int32.
// Alpha / beta: a value or a tag (One, Zero, NegOne, Scalar, MaybeZero); beta == 0 never reads C.

#include <cstdint>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/interop.hh>
#include <batchlas/device/level3/cfg.hh>
#include <batchlas/device/level3/epilogue.hh>
#include <batchlas/device/level3/schedule.hh>
#include <batchlas/device/level3/stage.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device::inline v2 {

namespace l3 {

// One matrix in global memory: element (i, j) at p[i + j * ld], m x n.
template <class T> struct Global {
    T* p;
    int ld, m, n;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[i + j * ld]; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class T> BATCHLAS_DLA_INLINE Global<T> global_of(const Global<T>& a) { return a; }
template <class T, class R, class C, class Ld, class L, class Idx>
BATCHLAS_DLA_INLINE Global<T> global_of(const Mat<T, R, C, Ld, Mem::Global, L, T*, Idx>& a) {
    return {a.p + a.off, int(a.ld), a.rows(), a.cols()};
}
template <class T> BATCHLAS_DLA_INLINE Global<T> global_of(const ItemMat<T>& a) {
    return {a.p + static_cast<std::int64_t>(a.b) * a.stride, a.ld, a.m, a.n};
}

// The value of a scalar argument (a MaybeZero with its flag set is 0).
template <class T, class S> BATCHLAS_DLA_INLINE T scalar_of(const S& s_) {
    const auto s = as_tag(s_);
    if constexpr (::batchlas::device::detail::is_maybe_zero_v<std::remove_cv_t<decltype(s)>>) return s.zero ? Arith<T>::zero() : static_cast<T>(s.v);
    else return ::batchlas::device::detail::tag_value<T>(s);
}

// One k pass of a tile: a(row, kk) = op(A)(row, k_lo + kk), b(kk, col) = op(B)(k_lo + kk, col)
// for kk < k. AR / BR: the staging walks the slab's rows fastest (the operand's contiguous index).
template <bool AR, bool BR, class RA, class RB> struct Pass {
    int k;
    RA a;
    RB b;
};
template <bool AR, bool BR, class RA, class RB> BATCHLAS_DLA_INLINE Pass<AR, BR, RA, RB> make_pass(int k, RA a, RB b) {
    return {k, a, b};
}

template <class PS> inline constexpr bool pass_a_rows = false;
template <bool AR, bool BR, class RA, class RB> inline constexpr bool pass_a_rows<Pass<AR, BR, RA, RB>> = AR;
template <class PS> inline constexpr bool pass_b_rows = false;
template <bool AR, bool BR, class RA, class RB> inline constexpr bool pass_b_rows<Pass<AR, BR, RA, RB>> = BR;

// C(m0 :, n0 :) := alpha (sum of the passes) + beta C over one Cfg tile; c is m x n (ldc).
template <class C, class Ex, class T, class Mask, class... P>
BATCHLAS_DLA_INLINE void run_tile(const Ex& ex, const Smem<C>& sm, T* c, int ldc, int m0, int n0, const int& m, const int& n, T alpha,
                                  T beta, Mask mask, const P&... passes) {
    using G = typename C::geom;
    using Ops = typename C::ops;
    constexpr int Thr = C::threads;
    const int lid = ex.lid();
    const int local_row = lid / G::LocalCols, local_col = lid % G::LocalCols;
    T* const tile_a = sm.p;
    T* const tile_b = sm.p + G::Stages * G::StageASize;
    Tile<T, G::ThreadTileRows, G::ThreadTileCols> accum;
    fill(accum, Arith<T>::zero());
    auto one_pass = [&](const auto& ps) BATCHLAS_DLA_BODY {
        using PS = std::remove_cvref_t<decltype(ps)>;
        auto stage_a = [&](int kk0, auto& dst) BATCHLAS_DLA_BODY {
            stage_block_read<pass_a_rows<PS>, G::TileM, G::TileK, Thr, T>(lid, m0, kk0, m, ps.k, ps.a, dst);
        };
        auto stage_b = [&](int kk0, auto& dst) BATCHLAS_DLA_BODY {
            stage_block_read<pass_b_rows<PS>, G::TileK, G::TileN, Thr, T>(lid, kk0, n0, ps.k, n, ps.b, dst);
        };
        reg_tile_mma<G, false, Ops>(ex, local_row, local_col, tile_a, tile_b, ps.k, stage_a, stage_b, accum);
    };
    (one_pass(passes), ...);
    store_tile_staged<G::TileM, G::TileN, Thr, G::Stages * G::StageASize, G::Stages * G::StageBSize, 2, Ops>(
        ex, lid, local_row, local_col, accum, tile_a, tile_b, c, ldc, m0, n0, m, n, alpha, beta, mask);
}

// A(i, j) of a symmetric (Herm: Hermitian, real diagonal) A from its UL triangle.
template <Uplo UL, bool Herm, class T> BATCHLAS_DLA_INLINE std::remove_cv_t<T> sym_at(const Global<T>& a, int i, int j) {
    using V = std::remove_cv_t<T>;
    using A = Arith<V>;
    if constexpr (Herm && A::is_complex) {
        if (i == j) return A::make(A::re(a(i, i)), typename A::real_t(0));
    }
    const bool stored = UL == Uplo::Lower ? i >= j : i <= j;
    if (stored) return a(i, j);
    if constexpr (Herm) return A::conj(a(j, i));
    else return a(j, i);
}

// op(A)(i, j) of a triangular A (UL stored triangle, DG = Unit: an implied unit diagonal).
template <Uplo UL, Transpose TR, Diag DG, class T> BATCHLAS_DLA_INLINE std::remove_cv_t<T> tri_at(const Global<T>& a, int i, int j) {
    using V = std::remove_cv_t<T>;
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    if (i == j) {
        if constexpr (DG == Diag::Unit) return Arith<V>::one();
        else return OperandAccessor<V, TR>::load(a.p, a.ld, 0, i, j);
    }
    const bool inside = op_lower ? i > j : i < j;
    return inside ? OperandAccessor<V, TR>::load(a.p, a.ld, 0, i, j) : Arith<V>::zero();
}

template <class C, class Ex> BATCHLAS_DLA_INLINE void check_group_cfg() {
    static_assert(Config<C>, "device L3: Cfg is an l3::cfg preset");
    static_assert(Ex::size == C::threads, "device L3: the executor has Cfg::threads items (TileM / TR x TileN / TC)");
}

// Tiles of an m x n C, column-major over tiles: f(m0, n0).
template <class C, class Ex, class F> BATCHLAS_DLA_INLINE void for_each_c_tile(const Ex& ex, int m, int n, F&& f) {
    const int tm = (m + C::TileM - 1) / C::TileM, tn = (n + C::TileN - 1) / C::TileN;
    for_each_tile(ex, tm * tn, [&](int t) BATCHLAS_DLA_BODY { f((t % tm) * C::TileM, (t / tm) * C::TileN); });
}

// The rank-k family (syrk, herk, syr2k, her2k) over the tiles that meet C's UL triangle.
template <class C, Uplo UL, bool Herm, class Ex, class T, class... P>
BATCHLAS_DLA_INLINE void rank_update(const Ex& ex, const Smem<C>& sm, const Global<T>& c, T alpha, T beta, const P&... passes) {
    const int n = c.m;
    for_each_c_tile<C>(ex, n, n, [&](int m0, int n0) BATCHLAS_DLA_BODY {
        const bool meets = UL == Uplo::Lower ? m0 + C::TileM - 1 >= n0 : m0 <= n0 + C::TileN - 1;
        if (meets) run_tile<C>(ex, sm, c.p, c.ld, m0, n0, n, n, alpha, beta, TriMask<UL, Herm>{}, passes...);
    });
}

}  // namespace l3

// ---- gemm, symm, hemm -----------------------------------------------------------------------

template <Transpose TA, Transpose TB, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void gemm(const Ex& ex, const MA& A, const MB& B, const MC& C, AS alpha, BS beta, l3::Smem<Cfg> smem) {
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const auto c = l3::global_of(C);
    const int m = c.m, n = c.n, k = TA == Transpose::NoTrans ? a.n : a.m;
    const T al = l3::scalar_of<T>(alpha), be = l3::scalar_of<T>(beta);
    const auto pass = l3::make_pass<TA == Transpose::NoTrans, TB == Transpose::NoTrans>(
        k, [=](int r, int kk) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TA>::load(a.p, a.ld, 0, r, kk); },
        [=](int kk, int col) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TB>::load(b.p, b.ld, 0, kk, col); });
    l3::for_each_c_tile<Cfg>(ex, m, n, [&](int m0, int n0) BATCHLAS_DLA_BODY {
        l3::run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, be, l3::NoMask{}, pass);
    });
    ex.barrier(Mem::Global);
}

namespace l3 {
template <Side SD, Uplo UL, bool Herm, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
BATCHLAS_DLA_INLINE void sym_product(const Ex& ex, const MA& A, const MB& B, const MC& C, AS alpha, BS beta, Smem<Cfg> smem) {
    check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto a = global_of(A);
    const auto b = global_of(B);
    const auto c = global_of(C);
    const int m = c.m, n = c.n;
    const T al = scalar_of<T>(alpha), be = scalar_of<T>(beta);
    auto plain = [=](int i, int j) BATCHLAS_DLA_BODY { return b(i, j); };
    auto sym = [=](int i, int j) BATCHLAS_DLA_BODY { return sym_at<UL, Herm>(a, i, j); };
    for_each_c_tile<Cfg>(ex, m, n, [&](int m0, int n0) BATCHLAS_DLA_BODY {
        if constexpr (SD == Side::Left) run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, be, NoMask{}, make_pass<true, true>(m, sym, plain));
        else run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, be, NoMask{}, make_pass<true, true>(n, plain, sym));
    });
    ex.barrier(Mem::Global);
}
}  // namespace l3

template <Side SD, Uplo UL, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void symm(const Ex& ex, const MA& A, const MB& B, const MC& C, AS alpha, BS beta, l3::Smem<Cfg> smem) {
    l3::sym_product<SD, UL, false>(ex, A, B, C, alpha, beta, smem);
}
template <Side SD, Uplo UL, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void hemm(const Ex& ex, const MA& A, const MB& B, const MC& C, AS alpha, BS beta, l3::Smem<Cfg> smem) {
    l3::sym_product<SD, UL, true>(ex, A, B, C, alpha, beta, smem);
}

// ---- syrk, herk, syr2k, her2k: only the UL triangle of C is read or written --------------------

template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void syrk(const Ex& ex, const MA& A, const MC& C, AS alpha, BS beta, l3::Smem<Cfg> smem) {
    static_assert(TR != Transpose::ConjTrans, "syrk: op(A) is A or A^T (herk takes A^H)");
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto a = l3::global_of(A);
    const auto c = l3::global_of(C);
    const int k = TR == Transpose::NoTrans ? a.n : a.m;
    constexpr bool rows = TR == Transpose::NoTrans;
    const auto pass = l3::make_pass<rows, !rows>(
        k, [=](int r, int kk) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, r, kk); },
        [=](int kk, int col) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, col, kk); });
    l3::rank_update<Cfg, UL, false>(ex, smem, c, l3::scalar_of<T>(alpha), l3::scalar_of<T>(beta), pass);
    ex.barrier(Mem::Global);
}

// TR: NoTrans (C := alpha A A^H + beta C) or ConjTrans (alpha A^H A + beta C); Trans is ConjTrans
// for a real T. alpha, beta real; Im C(i, i) := 0.
template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void herk(const Ex& ex, const MA& A, const MC& C, AS alpha, BS beta, l3::Smem<Cfg> smem) {
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    using Ar = Arith<T>;
    static_assert(TR != Transpose::Trans || !Ar::is_complex, "herk: op(A) is A or A^H");
    constexpr Transpose OP = TR == Transpose::NoTrans ? Transpose::NoTrans : Transpose::ConjTrans;
    const auto a = l3::global_of(A);
    const auto c = l3::global_of(C);
    const int k = OP == Transpose::NoTrans ? a.n : a.m;
    constexpr bool rows = OP == Transpose::NoTrans;
    const auto pass = l3::make_pass<rows, !rows>(
        k, [=](int r, int kk) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, OP>::load(a.p, a.ld, 0, r, kk); },
        [=](int kk, int col) BATCHLAS_DLA_BODY { return Ar::conj(l3::OperandAccessor<T, OP>::load(a.p, a.ld, 0, col, kk)); });
    l3::rank_update<Cfg, UL, true>(ex, smem, c, l3::scalar_of<T>(alpha), l3::scalar_of<T>(beta), pass);
    ex.barrier(Mem::Global);
}

// The two products accumulate in ONE accumulator, sequentially (op(A), op(B)) then (op(B), op(A))
// (interleaving them is 1.53x slower, docs/perf/level3.md).
template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void syr2k(const Ex& ex, const MA& A, const MB& B, const MC& C, AS alpha, BS beta, l3::Smem<Cfg> smem) {
    static_assert(TR != Transpose::ConjTrans, "syr2k: op is NoTrans or Trans (her2k takes ConjTrans)");
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const auto c = l3::global_of(C);
    const int k = TR == Transpose::NoTrans ? a.n : a.m;
    constexpr bool rows = TR == Transpose::NoTrans;
    auto op_a = [=](int i, int j) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, i, j); };
    auto op_b = [=](int i, int j) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TR>::load(b.p, b.ld, 0, i, j); };
    const auto ab = l3::make_pass<rows, !rows>(k, op_a, [=](int kk, int col) BATCHLAS_DLA_BODY { return op_b(col, kk); });
    const auto ba = l3::make_pass<rows, !rows>(k, op_b, [=](int kk, int col) BATCHLAS_DLA_BODY { return op_a(col, kk); });
    l3::rank_update<Cfg, UL, false>(ex, smem, c, l3::scalar_of<T>(alpha), l3::scalar_of<T>(beta), ab, ba);
    ex.barrier(Mem::Global);
}

// alpha scales op(B)^H as it is staged and conj(alpha) op(A)^H (the epilogue's alpha is 1); beta real.
template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void her2k(const Ex& ex, const MA& A, const MB& B, const MC& C, AS alpha, BS beta, l3::Smem<Cfg> smem) {
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    using Ar = arith_t<typename Cfg::ops, T>;
    static_assert(TR != Transpose::Trans || !Arith<T>::is_complex, "her2k: op is NoTrans or ConjTrans");
    constexpr Transpose OP = TR == Transpose::NoTrans ? Transpose::NoTrans : Transpose::ConjTrans;
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const auto c = l3::global_of(C);
    const int k = OP == Transpose::NoTrans ? a.n : a.m;
    constexpr bool rows = OP == Transpose::NoTrans;
    const T al = l3::scalar_of<T>(alpha), al_c = Arith<T>::conj(al);
    auto op_a = [=](int i, int j) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, OP>::load(a.p, a.ld, 0, i, j); };
    auto op_b = [=](int i, int j) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, OP>::load(b.p, b.ld, 0, i, j); };
    const auto ab = l3::make_pass<rows, !rows>(k, op_a, [=](int kk, int col) BATCHLAS_DLA_BODY { return Ar::mul(al, Arith<T>::conj(op_b(col, kk))); });
    const auto ba = l3::make_pass<rows, !rows>(k, op_b, [=](int kk, int col) BATCHLAS_DLA_BODY { return Ar::mul(al_c, Arith<T>::conj(op_a(col, kk))); });
    l3::rank_update<Cfg, UL, true>(ex, smem, c, Arith<T>::one(), l3::scalar_of<T>(beta), ab, ba);
    ex.barrier(Mem::Global);
}

// ---- trmm (out of place, as the host op), trsm (in place) -------------------------------------

// The k range of each tile is the triangle's reach (a loop bound, not a mask): Left, op(A) lower:
// rows m0 .. read columns [0, min(m, m0 + TileM)); upper: [m0, m). Right likewise over columns.
template <Side SD, Uplo UL, Transpose TR, Diag DG, class Cfg, class Ex, class MA, class MB, class MC, class AS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void trmm(const Ex& ex, const MA& A, const MB& B, const MC& C, AS alpha, l3::Smem<Cfg> smem) {
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const auto c = l3::global_of(C);
    const int m = c.m, n = c.n;
    const T al = l3::scalar_of<T>(alpha), zero = Arith<T>::zero();
    l3::for_each_c_tile<Cfg>(ex, m, n, [&](int m0, int n0) BATCHLAS_DLA_BODY {
        if constexpr (SD == Side::Left) {
            const int k_lo = op_lower ? 0 : m0;
            const int k_hi = op_lower ? (m0 + Cfg::TileM < m ? m0 + Cfg::TileM : m) : m;
            const auto pass = l3::make_pass<TR == Transpose::NoTrans, true>(
                k_hi - k_lo, [=](int r, int kk) BATCHLAS_DLA_BODY { return l3::tri_at<UL, TR, DG>(a, r, k_lo + kk); },
                [=](int kk, int col) BATCHLAS_DLA_BODY { return b(k_lo + kk, col); });
            l3::run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, zero, l3::NoMask{}, pass);
        } else {
            const int k_lo = op_lower ? n0 : 0;
            const int k_hi = op_lower ? n : (n0 + Cfg::TileN < n ? n0 + Cfg::TileN : n);
            const auto pass = l3::make_pass<true, TR == Transpose::NoTrans>(
                k_hi - k_lo, [=](int r, int kk) BATCHLAS_DLA_BODY { return b(r, k_lo + kk); },
                [=](int kk, int col) BATCHLAS_DLA_BODY { return l3::tri_at<UL, TR, DG>(a, k_lo + kk, col); });
            l3::run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, zero, l3::NoMask{}, pass);
        }
    });
    ex.barrier(Mem::Global);
}

// Blocked over NB-wide diagonal blocks of op(A), in solve order: the block's columns (Left; rows
// for Right) are solved by substitution, one column (row) per item, then the not-yet-solved part
// of B takes B -= op(A)(rest, blk) X(blk, :) (Left) on the tile body (alpha -1, beta 1).
// A zero diagonal element of a NonUnit A gives Inf / NaN, as BLAS does.
template <Side SD, Uplo UL, Transpose TR, Diag DG, class Cfg, int NB = 32, class Ex, class MA, class MB, class AS>
    requires l3::Config<Cfg>
BATCHLAS_DLA_INLINE void trsm(const Ex& ex, const MA& A, const MB& B, AS alpha, l3::Smem<Cfg> smem) {
    static_assert(NB >= 1, "trsm: NB >= 1");
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    using Ar = arith_t<typename Cfg::ops, T>;
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    constexpr bool forward = SD == Side::Left ? op_lower : !op_lower;
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const int m = b.m, n = b.n;
    const int lid = ex.lid();
    auto op_at = [&](int i, int j) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, i, j); };
    if constexpr (!std::is_same_v<tag_t<AS>, One>) {
        const T al = l3::scalar_of<T>(alpha);
        for (int e = lid; e < m * n; e += Ex::size) b(e % m, e / m) = Ar::mul(al, b(e % m, e / m));
        ex.barrier(Mem::Global);
    }
    const int order = SD == Side::Left ? m : n;
    const int blocks = (order + NB - 1) / NB;
    const T neg_one = Ar::neg(Ar::one()), one = Ar::one();
    for (int ib = 0; ib < blocks; ++ib) {
        const int kb = forward ? ib * NB : (blocks - 1 - ib) * NB;
        const int nb = order - kb < NB ? order - kb : NB;
        if constexpr (SD == Side::Left) {
            for (int j = lid; j < n; j += Ex::size) {
                for (int t = 0; t < nb; ++t) {
                    const int i = forward ? kb + t : kb + nb - 1 - t;
                    T s = b(i, j);
                    if (forward)
                        for (int p = kb; p < i; ++p) Ar::msub(s, op_at(i, p), b(p, j));
                    else
                        for (int p = i + 1; p < kb + nb; ++p) Ar::msub(s, op_at(i, p), b(p, j));
                    if constexpr (DG == Diag::NonUnit) s = Ar::div(s, op_at(i, i));
                    b(i, j) = s;
                }
            }
        } else {
            for (int i = lid; i < m; i += Ex::size) {
                for (int t = 0; t < nb; ++t) {
                    const int j = forward ? kb + t : kb + nb - 1 - t;
                    T s = b(i, j);
                    if (forward)
                        for (int p = kb; p < j; ++p) Ar::msub(s, b(i, p), op_at(p, j));
                    else
                        for (int p = j + 1; p < kb + nb; ++p) Ar::msub(s, b(i, p), op_at(p, j));
                    if constexpr (DG == Diag::NonUnit) s = Ar::div(s, op_at(j, j));
                    b(i, j) = s;
                }
            }
        }
        ex.barrier(Mem::Global);
        // The unsolved rest: [kb + nb, order) forward, [0, kb) backward.
        const int lo = forward ? kb + nb : 0;
        const int cnt = forward ? order - kb - nb : kb;
        if (cnt <= 0) continue;
        if constexpr (SD == Side::Left) {
            const auto pass = l3::make_pass<TR == Transpose::NoTrans, true>(
                nb, [=](int r, int kk) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, lo + r, kb + kk); },
                [=](int kk, int col) BATCHLAS_DLA_BODY { return b(kb + kk, col); });
            l3::for_each_c_tile<Cfg>(ex, cnt, n, [&](int m0, int n0) BATCHLAS_DLA_BODY {
                l3::run_tile<Cfg>(ex, smem, b.p + lo, b.ld, m0, n0, cnt, n, neg_one, one, l3::NoMask{}, pass);
            });
        } else {
            const auto pass = l3::make_pass<true, TR == Transpose::NoTrans>(
                nb, [=](int r, int kk) BATCHLAS_DLA_BODY { return b(r, kb + kk); },
                [=](int kk, int col) BATCHLAS_DLA_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, kb + kk, lo + col); });
            l3::for_each_c_tile<Cfg>(ex, m, cnt, [&](int m0, int n0) BATCHLAS_DLA_BODY {
                l3::run_tile<Cfg>(ex, smem, b.p + static_cast<std::int64_t>(lo) * b.ld, b.ld, m0, n0, m, cnt, neg_one, one, l3::NoMask{}, pass);
            });
        }
        ex.barrier(Mem::Global);
    }
}

}  // namespace batchlas::device

// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief In-group BLAS level 3: one matrix per work-group, any m, n and k, operands in global
/// memory (@ref api_device_blas3).
///
/// The host's operation names and BLAS argument order (`alpha` before its product, `beta` before
/// the output), with the host's run-time modes as template parameters:
///   - `gemm<TA, TB, Cfg>(ex, alpha, A, B, beta, C, ws)`: `C := alpha op(A) op(B) + beta C`.
///   - `symm` / `hemm<SD, UL, Cfg>(ex, alpha, A, B, beta, C, ws)`: `C := alpha A B + beta C` (Left)
///     or `alpha B A + beta C` (Right), `A` symmetric / Hermitian.
///   - `syrk<UL, TR, Cfg>(ex, alpha, A, beta, C, ws)`: `C := alpha op(A) op(A)^T + beta C`.
///   - `herk<UL, TR, Cfg>(ex, alpha, A, beta, C, ws)`: `C := alpha op(A) op(A)^H + beta C`.
///   - `syr2k<UL, TR, Cfg>(ex, alpha, A, B, beta, C, ws)`:
///     `C := alpha (op(A) op(B)^T + op(B) op(A)^T) + beta C`.
///   - `her2k<UL, TR, Cfg>(ex, alpha, A, B, beta, C, ws)`:
///     `C := alpha op(A) op(B)^H + conj(alpha) op(B) op(A)^H + beta C`.
///   - `trmm_out<SD, UL, TR, DG, Cfg>(ex, alpha, A, B, C, ws)`: `C := alpha op(A) B` (Left) or
///     `alpha B op(A)`; `B` unchanged.
///   - `trsm<SD, UL, TR, DG, Cfg, NB>(ex, alpha, A, B, ws)`: `B := alpha op(A)^-1 B` (Left) or
///     `alpha B op(A)^-1`.
///
/// The contract every operation here shares:
///   - **Executor**: a `WorkGroup` of exactly `Cfg::threads` items (checked at compile time).
///   - **Workspace**: `ws.local` holds `l3::smem_bytes<Cfg>()` bytes at offset 0, the stage.
///   - **Collective**: every item of the executor calls with the same arguments. The inputs must
///     be visible on entry (the caller's barrier); the outputs are visible to the work-group on
///     return, because every operation ends on a global barrier.
///   - **Operands**: a `Mat` over `T*` in global memory, `device::item(view, b)` of a host view, or
///     an `l3::Global`; in-item offsets are 32-bit.
///   - **Scalars**: `alpha` and `beta` are values or tags (`One`, `Zero`, `NegOne`, `Scalar`,
///     `MaybeZero`). `beta == 0` never reads `C`, so `C` may hold NaN.
///
/// Every operation runs one tile body (the host's register-tiled route's) over the matrix's tiles,
/// so a float `gemm` with the configuration of a host route row gives the host's bits.
/// Not included by `<batchlas/device.hh>`; `<batchlas/device/lapack.hh>` includes it.
/// @ingroup api_device_blas3

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

namespace batchlas::device {

namespace l3 {

/// @brief One column-major m x n matrix in global memory: element `(i, j)` at `p[i + j * ld]`.
///
/// The operand form every in-group level 3 operation reads; a `Mat` in global memory and
/// `device::item(view, b)` convert to it.
/// @ingroup api_device_blas3
/// @tparam T the element type (`const` for a read-only operand).
template <class T> struct Global {
    /// The first element.
    T* p;
    /// @{
    /// The leading dimension, the row count and the column count.
    int ld, m, n;
    /// @}
    /// @brief Element `(i, j)`.
    /// @param i the row.
    /// @param j the column.
    /// @return a reference to `p[i + j * ld]`.
    BATCHLAS_DEVICE_INLINE T& operator()(int i, int j) const { return p[i + j * ld]; }
    /// @brief The row count `m`.
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    /// @brief The column count `n`.
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
};
/// @brief `a` itself.
/// @ingroup api_device_lowlevel
/// @param a the global operand.
template <class T> BATCHLAS_DEVICE_INLINE Global<T> global_of(const Global<T>& a) { return a; }
/// @brief The `Global` view of a `Mat` in global memory.
/// @ingroup api_device_lowlevel
/// @param a the matrix operand.
template <class T, class R, class C, class Ld, class L, class Idx>
BATCHLAS_DEVICE_INLINE Global<T> global_of(const Mat<T, R, C, Ld, Mem::Global, L, T*, Idx>& a) {
    return {a.p + a.off, int(a.ld), a.rows(), a.cols()};
}
/// @brief The `Global` view of one batch item of a host view (`device::item(view, b)`).
/// @ingroup api_device_lowlevel
/// @param a the batch item.
template <class T> BATCHLAS_DEVICE_INLINE Global<T> global_of(const ItemMat<T>& a) {
    return {a.p + static_cast<std::int64_t>(a.b) * a.stride, a.ld, a.m, a.n};
}

/// @brief The value of a scalar argument as `Scalar`; a `MaybeZero` with its flag set is 0.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the result type.
/// @param s_ a value or a tag.
template <class Scalar, class S> BATCHLAS_DEVICE_INLINE Scalar scalar_of(const S& s_) {
    const auto s = as_tag(s_);
    if constexpr (::batchlas::device::detail::is_maybe_zero_v<std::remove_cv_t<decltype(s)>>) return s.zero ? Arith<Scalar>::zero() : static_cast<Scalar>(s.v);
    else return ::batchlas::device::detail::tag_value<Scalar>(s);
}

/// @brief One k pass of a tile: `a(row, kk)` reads `op(A)(row, k_lo + kk)` and `b(kk, col)` reads
/// `op(B)(k_lo + kk, col)` for `kk < k`.
/// @ingroup api_device_lowlevel
/// @tparam AR the staging of `a` walks the slab's rows fastest (the operand's contiguous index).
/// @tparam BR likewise for `b`.
template <bool AR, bool BR, class RA, class RB> struct Pass {
    /// The pass's depth.
    int k;
    /// The `op(A)` reader, called as `a(row, kk)`.
    RA a;
    /// The `op(B)` reader, called as `b(kk, col)`.
    RB b;
};
/// @brief A `Pass` of depth `k` over the readers `a` and `b`.
/// @ingroup api_device_lowlevel
/// @tparam AR the staging of `a` walks rows fastest.
/// @tparam BR the staging of `b` walks rows fastest.
/// @param k the depth.
/// @param a the `op(A)` reader.
/// @param b the `op(B)` reader.
template <bool AR, bool BR, class RA, class RB> BATCHLAS_DEVICE_INLINE Pass<AR, BR, RA, RB> make_pass(int k, RA a, RB b) {
    return {k, a, b};
}

/// @brief A `Pass`'s `AR` flag (`false` for any other type).
/// @ingroup api_device_lowlevel
template <class PS> inline constexpr bool pass_a_rows = false;
/// @brief A `Pass`'s `AR` flag.
/// @ingroup api_device_lowlevel
template <bool AR, bool BR, class RA, class RB> inline constexpr bool pass_a_rows<Pass<AR, BR, RA, RB>> = AR;
/// @brief A `Pass`'s `BR` flag (`false` for any other type).
/// @ingroup api_device_lowlevel
template <class PS> inline constexpr bool pass_b_rows = false;
/// @brief A `Pass`'s `BR` flag.
/// @ingroup api_device_lowlevel
template <bool AR, bool BR, class RA, class RB> inline constexpr bool pass_b_rows<Pass<AR, BR, RA, RB>> = BR;

/// @brief `C(m0 :, n0 :) := alpha (the sum of the passes) + beta C` over one `C` tile; `c` is m x n with leading dimension `ldc`.
///
/// The tile body: each pass staged and accumulated in registers (`reg_tile_mma`), then one staged
/// epilogue (`store_tile_staged`) that writes the elements `mask` keeps. COLLECTIVE over the
/// executor.
/// @ingroup api_device_lowlevel
/// @tparam C the configuration.
/// @param ex the work-group executor of `C::threads` items.
/// @param sm the local stage.
/// @param c the first element of `C`.
/// @param ldc `C`'s leading dimension.
/// @param m0 the tile's first row.
/// @param n0 the tile's first column.
/// @param m `C`'s row count.
/// @param n `C`'s column count.
/// @param alpha the product's scale.
/// @param beta `C`'s scale; 0 never reads `C`.
/// @param mask `NoMask` or a `TriMask`.
/// @param passes the k passes, summed into one accumulator in order.
template <class C, class Ex, class Scalar, class Mask, class... P>
BATCHLAS_DEVICE_INLINE void run_tile(const Ex& ex, const device::detail::Smem<C>& sm, Scalar* c, int ldc, int m0, int n0, const int& m, const int& n, Scalar alpha,
                                  Scalar beta, Mask mask, const P&... passes) {
    using G = typename C::geom;
    using Ops = typename C::ops;
    constexpr int Thr = C::threads;
    const int lid = ex.lid();
    const int local_row = lid / G::LocalCols, local_col = lid % G::LocalCols;
    const auto tile_a = sm.p.get_decorated();
    const auto tile_b = tile_a + G::Stages * G::StageASize;
    Tile<Scalar, G::ThreadTileRows, G::ThreadTileCols> accum;
    fill(accum, Arith<Scalar>::zero());
    auto one_pass = [&](const auto& ps) BATCHLAS_DEVICE_BODY {
        using PS = std::remove_cvref_t<decltype(ps)>;
        auto stage = [&](int kk0, auto& dst_a, auto& dst_b) BATCHLAS_DEVICE_BODY {
            stage_block_read<pass_a_rows<PS>, G::TileM, G::TileK, Thr, Scalar>(lid, m0, kk0, m, ps.k, ps.a, dst_a);
            stage_block_read<pass_b_rows<PS>, G::TileK, G::TileN, Thr, Scalar>(lid, kk0, n0, ps.k, n, ps.b, dst_b);
        };
        reg_tile_mma<G, false, Ops>(ex, local_row, local_col, tile_a, tile_b, stage, ps.k, accum);
    };
    (one_pass(passes), ...);
    store_tile_staged<G::TileM, G::TileN, Thr, G::Stages * G::StageASize, G::Stages * G::StageBSize, 2, Ops>(
        ex, lid, local_row, local_col, accum, tile_a, tile_b, c, ldc, m0, n0, m, n, alpha, beta, mask);
}

/// @brief Element `(i, j)` of a symmetric `A` (`Herm`: Hermitian, real diagonal) read from its `UL` triangle.
/// @ingroup api_device_lowlevel
/// @tparam UL the stored triangle.
/// @tparam Herm `true` for Hermitian: the mirrored element is conjugated and the diagonal is real.
/// @param a the matrix.
/// @param i the row.
/// @param j the column.
template <Uplo UL, bool Herm, class Scalar> BATCHLAS_DEVICE_INLINE std::remove_cv_t<Scalar> sym_at(const Global<Scalar>& a, int i, int j) {
    using V = std::remove_cv_t<Scalar>;
    using A = Arith<V>;
    if constexpr (Herm && A::is_complex) {
        if (i == j) return A::make(A::re(a(i, i)), typename A::real_t(0));
    }
    const bool stored = UL == Uplo::Lower ? i >= j : i <= j;
    if (stored) return a(i, j);
    if constexpr (Herm) return A::conj(a(j, i));
    else return a(j, i);
}

/// @brief Element `(i, j)` of `op(A)` for a triangular `A`, zero outside the triangle.
/// @ingroup api_device_lowlevel
/// @tparam UL the stored triangle.
/// @tparam TR `op(A)`.
/// @tparam DG `Diag::Unit`: an implied unit diagonal, not read.
/// @param a the matrix.
/// @param i the row of `op(A)`.
/// @param j the column of `op(A)`.
template <Uplo UL, Transpose TR, Diag DG, class Scalar> BATCHLAS_DEVICE_INLINE std::remove_cv_t<Scalar> tri_at(const Global<Scalar>& a, int i, int j) {
    using V = std::remove_cv_t<Scalar>;
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    if (i == j) {
        if constexpr (DG == Diag::Unit) return Arith<V>::one();
        else return OperandAccessor<V, TR>::load(a.p, a.ld, 0, i, j);
    }
    const bool inside = op_lower ? i > j : i < j;
    return inside ? OperandAccessor<V, TR>::load(a.p, a.ld, 0, i, j) : Arith<V>::zero();
}

/// @brief Compile-time check that `C` is a configuration and the executor has `C::threads` items.
/// @ingroup api_device_lowlevel
/// @tparam C the configuration.
/// @tparam Ex the executor type.
template <class C, class Ex> BATCHLAS_DEVICE_INLINE void check_group_cfg() {
    static_assert(Config<C>, "device L3: Cfg is an l3::cfg preset");
    static_assert(Ex::size == C::threads, "device L3: the executor has Cfg::threads items (TileM / TR x TileN / TC)");
}

/// @brief Calls `f(m0, n0)` for every `C::TileM` x `C::TileN` tile of an m x n matrix, column-major over tiles.
///
/// Every item makes every call, with a barrier after each (`for_each_tile`). COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @tparam C the configuration.
/// @param ex the executor.
/// @param m the row count.
/// @param n the column count.
/// @param f called as `f(int m0, int n0)`.
template <class C, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_c_tile(const Ex& ex, int m, int n, F&& f) {
    const int tm = (m + C::TileM - 1) / C::TileM, tn = (n + C::TileN - 1) / C::TileN;
    for_each_tile(ex, tm * tn, [&](int t) BATCHLAS_DEVICE_BODY { f((t % tm) * C::TileM, (t / tm) * C::TileN); });
}

/// @brief The rank-k family's body (syrk, herk, syr2k, her2k): `run_tile` over the tiles of the
/// n x n `C` that meet its `UL` triangle, masked to the triangle.
/// @ingroup api_device_lowlevel
/// @tparam C the configuration.
/// @tparam UL the triangle of `C` read and written.
/// @tparam Herm `true` forces a real diagonal.
/// @param ex the executor.
/// @param ws the workspace holding the stage.
/// @param c the matrix `C`.
/// @param alpha the product's scale.
/// @param beta `C`'s scale.
/// @param passes the k passes.
template <class C, Uplo UL, bool Herm, class Ex, class Scalar, class... P>
BATCHLAS_DEVICE_INLINE void rank_update(const Ex& ex, const Workspace& ws, const Global<Scalar>& c, Scalar alpha, Scalar beta, const P&... passes) {
    const int n = c.m;
    const auto sm = device::detail::smem_of<C>(ws);
    for_each_c_tile<C>(ex, n, n, [&](int m0, int n0) BATCHLAS_DEVICE_BODY {
        const bool meets = UL == Uplo::Lower ? m0 + C::TileM - 1 >= n0 : m0 <= n0 + C::TileN - 1;
        if (meets) run_tile<C>(ex, sm, c.p, c.ld, m0, n0, n, n, alpha, beta, TriMask<UL, Herm>{}, passes...);
    });
}

}  // namespace l3

// ---- gemm, symm, hemm -----------------------------------------------------------------------

/// @brief `C := alpha op(A) op(B) + beta C` (BLAS gemm) on one matrix per work-group, any m, n and k.
///
/// `C` is m x n, `op(A)` m x k and `op(B)` k x n; k is read from `A`. The file's contract applies:
/// a `WorkGroup` of `Cfg::threads` items, the stage at the start of `ws.local`, COLLECTIVE, and
/// `C` visible to the work-group on return.
/// @ingroup api_device_blas3
/// @tparam TA `op(A)`: `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam TB `op(B)`, likewise.
/// @tparam Cfg the configuration, an `l3::cfg::RegTiled` of `C`'s element type.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the matrix `A` (global memory).
/// @param B the matrix `B`.
/// @param beta the scalar `beta`; 0 never reads `C`.
/// @param[in,out] C the matrix `C`.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Transpose TA, Transpose TB, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void gemm(const Ex& ex, AS alpha, const MA& A, const MB& B, BS beta, const MC& C, const Workspace& ws) {
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto smem = device::detail::smem_of<Cfg>(ws);
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const auto c = l3::global_of(C);
    const int m = c.m, n = c.n, k = TA == Transpose::NoTrans ? a.n : a.m;
    const T al = l3::scalar_of<T>(alpha), be = l3::scalar_of<T>(beta);
    const auto pass = l3::make_pass<TA == Transpose::NoTrans, TB == Transpose::NoTrans>(
        k, [=](int r, int kk) BATCHLAS_DEVICE_BODY { return l3::OperandAccessor<T, TA>::load(a.p, a.ld, 0, r, kk); },
        [=](int kk, int col) BATCHLAS_DEVICE_BODY { return l3::OperandAccessor<T, TB>::load(b.p, b.ld, 0, kk, col); });
    l3::for_each_c_tile<Cfg>(ex, m, n, [&](int m0, int n0) BATCHLAS_DEVICE_BODY {
        l3::run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, be, l3::NoMask{}, pass);
    });
    ex.barrier(Mem::Global);
}

namespace l3 {
/// @brief The body of `symm` and `hemm`.
/// @ingroup api_device_lowlevel
/// @tparam SD `Side::Left` (`A B`) or `Side::Right` (`B A`).
/// @tparam UL the stored triangle of `A`.
/// @tparam TS `Transpose::Trans` (symm: `A = A^T`) or `Transpose::ConjTrans` (hemm: `A = A^H`, real diagonal).
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`.
/// @param A the symmetric or Hermitian matrix.
/// @param B the general matrix.
/// @param beta the scalar `beta`.
/// @param[in,out] C the output.
/// @param ws the workspace holding the stage.
template <Side SD, Uplo UL, Transpose TS, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
BATCHLAS_DEVICE_INLINE void sym_product(const Ex& ex, AS alpha, const MA& A, const MB& B, BS beta, const MC& C, const Workspace& ws) {
    static_assert(TS != Transpose::NoTrans, "symm/hemm: TS is Trans (symmetric) or ConjTrans (Hermitian)");
    constexpr bool herm = TS == Transpose::ConjTrans;
    check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto smem = device::detail::smem_of<Cfg>(ws);
    const auto a = global_of(A);
    const auto b = global_of(B);
    const auto c = global_of(C);
    const int m = c.m, n = c.n;
    const T al = scalar_of<T>(alpha), be = scalar_of<T>(beta);
    auto plain = [=](int i, int j) BATCHLAS_DEVICE_BODY { return b(i, j); };
    auto sym = [=](int i, int j) BATCHLAS_DEVICE_BODY { return sym_at<UL, herm>(a, i, j); };
    for_each_c_tile<Cfg>(ex, m, n, [&](int m0, int n0) BATCHLAS_DEVICE_BODY {
        if constexpr (SD == Side::Left) run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, be, NoMask{}, make_pass<true, true>(m, sym, plain));
        else run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, be, NoMask{}, make_pass<true, true>(n, plain, sym));
    });
    ex.barrier(Mem::Global);
}
}  // namespace l3

/// @brief `C := alpha A B + beta C` (`Side::Left`) or `alpha B A + beta C` (`Side::Right`), `A`
/// symmetric (BLAS symm), on one matrix per work-group.
///
/// `C` and `B` are m x n; `A` is m x m (Left) or n x n (Right) and only its `UL` triangle is read.
/// The file's contract applies.
/// @ingroup api_device_blas3
/// @tparam SD `Side::Left` or `Side::Right`.
/// @tparam UL the stored triangle of `A`.
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the symmetric matrix.
/// @param B the general matrix.
/// @param beta the scalar `beta`; 0 never reads `C`.
/// @param[in,out] C the output.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Side SD, Uplo UL, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void symm(const Ex& ex, AS alpha, const MA& A, const MB& B, BS beta, const MC& C, const Workspace& ws) {
    l3::sym_product<SD, UL, Transpose::Trans, Cfg>(ex, alpha, A, B, beta, C, ws);
}
/// @brief `C := alpha A B + beta C` (`Side::Left`) or `alpha B A + beta C` (`Side::Right`), `A`
/// Hermitian (BLAS hemm), on one matrix per work-group.
///
/// As `symm`, with the conjugate of the stored element opposite the triangle and the diagonal's
/// imaginary part not read. For a real element type it is `symm`. The file's contract applies.
/// @ingroup api_device_blas3
/// @tparam SD `Side::Left` or `Side::Right`.
/// @tparam UL the stored triangle of `A`.
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the Hermitian matrix.
/// @param B the general matrix.
/// @param beta the scalar `beta`; 0 never reads `C`.
/// @param[in,out] C the output.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Side SD, Uplo UL, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void hemm(const Ex& ex, AS alpha, const MA& A, const MB& B, BS beta, const MC& C, const Workspace& ws) {
    l3::sym_product<SD, UL, is_complex_v<typename Cfg::value_type> ? Transpose::ConjTrans : Transpose::Trans, Cfg>(ex, alpha, A, B, beta, C, ws);
}

// ---- syrk, herk, syr2k, her2k: only the UL triangle of C is read or written --------------------
// One body per rank: l3::rank_k<UL, TR, TS> and l3::rank_2k<UL, TR, TS> (TS = Trans: the sy* form;
// ConjTrans: the he* form, op^H for the second factor, Im C(i, i) := 0); the four BLAS names are
// one-line wrappers. The he*
// names (and hemm) take a real T too: there they ARE the sy* body (ConjTrans read as Trans), so a
// caller writes the zhe* name once for every T.

namespace l3 {
/// @brief The body of `syrk` and `herk`: `C := alpha op(A) op(A)^T + beta C` (`TS = ConjTrans`: `^H`) on `C`'s `UL` triangle.
/// @ingroup api_device_lowlevel
/// @tparam UL the triangle of `C` read and written.
/// @tparam TR `op(A)`.
/// @tparam TS `Transpose::Trans` (syrk) or `Transpose::ConjTrans` (herk).
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`.
/// @param A the matrix `A`.
/// @param beta the scalar `beta`.
/// @param[in,out] C the n x n output.
/// @param ws the workspace holding the stage.
template <Uplo UL, Transpose TR, Transpose TS, class Cfg, class Ex, class MA, class MC, class AS, class BS>
BATCHLAS_DEVICE_INLINE void rank_k(const Ex& ex, AS alpha, const MA& A, BS beta, const MC& C, const Workspace& ws) {
    static_assert(TS != Transpose::NoTrans, "syrk/herk: TS is Trans (syrk) or ConjTrans (herk)");
    constexpr bool herm = TS == Transpose::ConjTrans;
    check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    using Ar = Arith<T>;
    if constexpr (herm) static_assert(TR != Transpose::Trans || !Ar::is_complex, "herk: op(A) is A or A^H");
    else static_assert(TR != Transpose::ConjTrans, "syrk: op(A) is A or A^T (herk takes A^H)");
    constexpr Transpose OP = herm && TR != Transpose::NoTrans ? Transpose::ConjTrans : TR;
    const auto a = global_of(A);
    const auto c = global_of(C);
    const int k = OP == Transpose::NoTrans ? a.n : a.m;
    constexpr bool rows = OP == Transpose::NoTrans;
    const auto pass = make_pass<rows, !rows>(
        k, [=](int r, int kk) BATCHLAS_DEVICE_BODY { return OperandAccessor<T, OP>::load(a.p, a.ld, 0, r, kk); },
        [=](int kk, int col) BATCHLAS_DEVICE_BODY {
            if constexpr (herm) return Ar::conj(OperandAccessor<T, OP>::load(a.p, a.ld, 0, col, kk));
            else return OperandAccessor<T, OP>::load(a.p, a.ld, 0, col, kk);
        });
    rank_update<Cfg, UL, herm>(ex, ws, c, scalar_of<T>(alpha), scalar_of<T>(beta), pass);
    ex.barrier(Mem::Global);
}

// The two products accumulate sequentially; interleaving them is 1.53x slower (docs/perf/level3.md).
/// @brief The body of `syr2k` and `her2k`: `C := alpha op(A) op(B)^T + alpha op(B) op(A)^T + beta C`
/// (`TS = ConjTrans`: `^H`, and `conj(alpha)` on the second product) on `C`'s `UL` triangle.
///
/// The two products accumulate in one accumulator, `(op(A), op(B))` then `(op(B), op(A))`. The
/// `ConjTrans` form stages `alpha op(B)^H` and `conj(alpha) op(A)^H` as they are loaded, so the
/// epilogue's `alpha` is 1.
/// @ingroup api_device_lowlevel
/// @tparam UL the triangle of `C` read and written.
/// @tparam TR `op(A)` and `op(B)`.
/// @tparam TS `Transpose::Trans` (syr2k) or `Transpose::ConjTrans` (her2k).
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`.
/// @param A the matrix `A`.
/// @param B the matrix `B`.
/// @param beta the scalar `beta`.
/// @param[in,out] C the n x n output.
/// @param ws the workspace holding the stage.
template <Uplo UL, Transpose TR, Transpose TS, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
BATCHLAS_DEVICE_INLINE void rank_2k(const Ex& ex, AS alpha, const MA& A, const MB& B, BS beta, const MC& C, const Workspace& ws) {
    static_assert(TS != Transpose::NoTrans, "syr2k/her2k: TS is Trans (syr2k) or ConjTrans (her2k)");
    constexpr bool herm = TS == Transpose::ConjTrans;
    check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    if constexpr (herm) static_assert(TR != Transpose::Trans || !Arith<T>::is_complex, "her2k: op is NoTrans or ConjTrans");
    else static_assert(TR != Transpose::ConjTrans, "syr2k: op is NoTrans or Trans (her2k takes ConjTrans)");
    constexpr Transpose OP = herm && TR != Transpose::NoTrans ? Transpose::ConjTrans : TR;
    const auto a = global_of(A);
    const auto b = global_of(B);
    const auto c = global_of(C);
    const int k = OP == Transpose::NoTrans ? a.n : a.m;
    constexpr bool rows = OP == Transpose::NoTrans;
    auto op_a = [=](int i, int j) BATCHLAS_DEVICE_BODY { return OperandAccessor<T, OP>::load(a.p, a.ld, 0, i, j); };
    auto op_b = [=](int i, int j) BATCHLAS_DEVICE_BODY { return OperandAccessor<T, OP>::load(b.p, b.ld, 0, i, j); };
    if constexpr (herm) {
        using Ar = arith_t<typename Cfg::ops, T>;
        const T al = scalar_of<T>(alpha), al_c = Arith<T>::conj(al);
        const auto ab = make_pass<rows, !rows>(k, op_a, [=](int kk, int col) BATCHLAS_DEVICE_BODY { return Ar::mul(al, Arith<T>::conj(op_b(col, kk))); });
        const auto ba = make_pass<rows, !rows>(k, op_b, [=](int kk, int col) BATCHLAS_DEVICE_BODY { return Ar::mul(al_c, Arith<T>::conj(op_a(col, kk))); });
        rank_update<Cfg, UL, true>(ex, ws, c, Arith<T>::one(), scalar_of<T>(beta), ab, ba);
    } else {
        const auto ab = make_pass<rows, !rows>(k, op_a, [=](int kk, int col) BATCHLAS_DEVICE_BODY { return op_b(col, kk); });
        const auto ba = make_pass<rows, !rows>(k, op_b, [=](int kk, int col) BATCHLAS_DEVICE_BODY { return op_a(col, kk); });
        rank_update<Cfg, UL, false>(ex, ws, c, scalar_of<T>(alpha), scalar_of<T>(beta), ab, ba);
    }
    ex.barrier(Mem::Global);
}
}  // namespace l3

/// @brief `C := alpha op(A) op(A)^T + beta C` (BLAS syrk) on the `UL` triangle of `C`, one matrix per work-group.
///
/// `C` is n x n; `op(A)` is n x k. Only the `UL` triangle of `C` is read or written. The file's
/// contract applies.
/// @ingroup api_device_blas3
/// @tparam UL the triangle of `C`.
/// @tparam TR `Transpose::NoTrans` (`A A^T`) or `Trans` (`A^T A`); `ConjTrans` is rejected (use `herk`).
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the matrix `A`.
/// @param beta the scalar `beta`; 0 never reads `C`.
/// @param[in,out] C the output.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void syrk(const Ex& ex, AS alpha, const MA& A, BS beta, const MC& C, const Workspace& ws) {
    l3::rank_k<UL, TR, Transpose::Trans, Cfg>(ex, alpha, A, beta, C, ws);
}
/// @brief `C := alpha op(A) op(A)^H + beta C` (BLAS herk) on the `UL` triangle of `C`, one matrix per work-group.
///
/// `alpha` and `beta` are real; `Im C(i, i)` is set to 0, as zherk does. For a real element type
/// this is `syrk`, so a caller can write `herk` for every type. The file's contract applies.
/// @ingroup api_device_blas3
/// @tparam UL the triangle of `C`.
/// @tparam TR `Transpose::NoTrans` (`A A^H`) or `ConjTrans` (`A^H A`); for a real type `Trans`
///   means `ConjTrans`.
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the real scalar `alpha`: a value or a tag.
/// @param A the matrix `A`.
/// @param beta the real scalar `beta`; 0 never reads `C`.
/// @param[in,out] C the n x n output.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void herk(const Ex& ex, AS alpha, const MA& A, BS beta, const MC& C, const Workspace& ws) {
    if constexpr (is_complex_v<typename Cfg::value_type>) l3::rank_k<UL, TR, Transpose::ConjTrans, Cfg>(ex, alpha, A, beta, C, ws);
    else l3::rank_k<UL, TR == Transpose::NoTrans ? TR : Transpose::Trans, Transpose::Trans, Cfg>(ex, alpha, A, beta, C, ws);
}
/// @brief `C := alpha (op(A) op(B)^T + op(B) op(A)^T) + beta C` (BLAS syr2k) on the `UL` triangle of `C`.
///
/// `C` is n x n; `op(A)` and `op(B)` are n x k. The file's contract applies.
/// @ingroup api_device_blas3
/// @tparam UL the triangle of `C`.
/// @tparam TR `Transpose::NoTrans` or `Trans`; `ConjTrans` is rejected (use `her2k`).
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the matrix `A`.
/// @param B the matrix `B`.
/// @param beta the scalar `beta`; 0 never reads `C`.
/// @param[in,out] C the output.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void syr2k(const Ex& ex, AS alpha, const MA& A, const MB& B, BS beta, const MC& C, const Workspace& ws) {
    l3::rank_2k<UL, TR, Transpose::Trans, Cfg>(ex, alpha, A, B, beta, C, ws);
}
/// @brief `C := alpha op(A) op(B)^H + conj(alpha) op(B) op(A)^H + beta C` (BLAS her2k) on the `UL` triangle of `C`.
///
/// `beta` is real and `Im C(i, i)` is set to 0. For a real element type this is `syr2k`. The
/// file's contract applies.
/// @ingroup api_device_blas3
/// @tparam UL the triangle of `C`.
/// @tparam TR `Transpose::NoTrans` or `ConjTrans`; for a real type `Trans` means `ConjTrans`.
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the matrix `A`.
/// @param B the matrix `B`.
/// @param beta the real scalar `beta`; 0 never reads `C`.
/// @param[in,out] C the n x n output.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Uplo UL, Transpose TR, class Cfg, class Ex, class MA, class MB, class MC, class AS, class BS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void her2k(const Ex& ex, AS alpha, const MA& A, const MB& B, BS beta, const MC& C, const Workspace& ws) {
    if constexpr (is_complex_v<typename Cfg::value_type>) l3::rank_2k<UL, TR, Transpose::ConjTrans, Cfg>(ex, alpha, A, B, beta, C, ws);
    else l3::rank_2k<UL, TR == Transpose::NoTrans ? TR : Transpose::Trans, Transpose::Trans, Cfg>(ex, alpha, A, B, beta, C, ws);
}

// ---- trmm_out (C := alpha op(A) B or alpha B op(A), out of place: the host op), trsm (in place) --

// The k range of each tile is the triangle's reach (a loop bound, not a mask): Left, op(A) lower:
// rows m0 .. read columns [0, min(m, m0 + TileM)); upper: [m0, m). Right likewise over columns.
/// @brief `C := alpha op(A) B` (`Side::Left`) or `alpha B op(A)` (`Side::Right`), `A` triangular,
/// out of place (the host's trmm); `B` is unchanged.
///
/// `C` and `B` are m x n; `A` is m x m (Left) or n x n (Right), only its `UL` triangle is read,
/// and `C` must not alias `B`. The file's contract applies.
/// @ingroup api_device_blas3
/// @tparam SD `Side::Left` or `Side::Right`.
/// @tparam UL the stored triangle of `A`.
/// @tparam TR `op(A)`: `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` (an implied unit diagonal, not read) or `Diag::NonUnit`.
/// @tparam Cfg the configuration.
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the triangular matrix.
/// @param B the general matrix.
/// @param[out] C the output.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Side SD, Uplo UL, Transpose TR, Diag DG, class Cfg, class Ex, class MA, class MB, class MC, class AS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void trmm_out(const Ex& ex, AS alpha, const MA& A, const MB& B, const MC& C, const Workspace& ws) {
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto smem = device::detail::smem_of<Cfg>(ws);
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const auto c = l3::global_of(C);
    const int m = c.m, n = c.n;
    const T al = l3::scalar_of<T>(alpha), zero = Arith<T>::zero();
    l3::for_each_c_tile<Cfg>(ex, m, n, [&](int m0, int n0) BATCHLAS_DEVICE_BODY {
        if constexpr (SD == Side::Left) {
            const int k_lo = op_lower ? 0 : m0;
            const int k_hi = op_lower ? (m0 + Cfg::TileM < m ? m0 + Cfg::TileM : m) : m;
            const auto pass = l3::make_pass<TR == Transpose::NoTrans, true>(
                k_hi - k_lo, [=](int r, int kk) BATCHLAS_DEVICE_BODY { return l3::tri_at<UL, TR, DG>(a, r, k_lo + kk); },
                [=](int kk, int col) BATCHLAS_DEVICE_BODY { return b(k_lo + kk, col); });
            l3::run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, zero, l3::NoMask{}, pass);
        } else {
            const int k_lo = op_lower ? n0 : 0;
            const int k_hi = op_lower ? n : (n0 + Cfg::TileN < n ? n0 + Cfg::TileN : n);
            const auto pass = l3::make_pass<true, TR == Transpose::NoTrans>(
                k_hi - k_lo, [=](int r, int kk) BATCHLAS_DEVICE_BODY { return b(r, k_lo + kk); },
                [=](int kk, int col) BATCHLAS_DEVICE_BODY { return l3::tri_at<UL, TR, DG>(a, k_lo + kk, col); });
            l3::run_tile<Cfg>(ex, smem, c.p, c.ld, m0, n0, m, n, al, zero, l3::NoMask{}, pass);
        }
    });
    ex.barrier(Mem::Global);
}

/// @brief `B := alpha op(A)^-1 B` (`Side::Left`) or `alpha B op(A)^-1` (`Side::Right`), `A`
/// triangular (BLAS trsm), in place, one matrix per work-group.
///
/// Blocked over `NB`-wide diagonal blocks of `op(A)` in solve order: the block's rows of `X` are
/// solved by substitution, one column of `B` per item (Right: one row), then the unsolved rest of
/// `B` takes `B -= op(A)(rest, blk) X(blk, :)` on the tile body. A zero diagonal element of a
/// `Diag::NonUnit` `A` gives Inf or NaN, as BLAS does. The file's contract applies.
/// @ingroup api_device_blas3
/// @tparam SD `Side::Left` or `Side::Right`.
/// @tparam UL the stored triangle of `A`.
/// @tparam TR `op(A)`: `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` or `Diag::NonUnit`.
/// @tparam Cfg the configuration.
/// @tparam NB the diagonal block size, `>= 1` (default 32).
/// @param ex the work-group executor.
/// @param alpha the scalar `alpha`: a value or a tag (`One` skips the scaling pass).
/// @param A the triangular matrix, m x m (Left) or n x n (Right).
/// @param[in,out] B the m x n right-hand side, overwritten by `X`.
/// @param ws the workspace; `ws.local` holds `l3::smem_bytes<Cfg>()` bytes.
template <Side SD, Uplo UL, Transpose TR, Diag DG, class Cfg, int NB = 32, class Ex, class MA, class MB, class AS>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void trsm(const Ex& ex, AS alpha, const MA& A, const MB& B, const Workspace& ws) {
    static_assert(NB >= 1, "trsm: NB >= 1");
    l3::check_group_cfg<Cfg, Ex>();
    using T = typename Cfg::value_type;
    const auto smem = device::detail::smem_of<Cfg>(ws);
    using Ar = arith_t<typename Cfg::ops, T>;
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    constexpr bool forward = SD == Side::Left ? op_lower : !op_lower;
    const auto a = l3::global_of(A);
    const auto b = l3::global_of(B);
    const int m = b.m, n = b.n;
    const int lid = ex.lid();
    auto op_at = [&](int i, int j) BATCHLAS_DEVICE_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, i, j); };
    if constexpr (!std::is_same_v<tag_t<AS>, One>) {
        const T al = l3::scalar_of<T>(alpha);
        for (int e = lid; e < m * n; e += Ex::size) b(e % m, e / m) = Ar::mul(al, b(e % m, e / m));
        ex.barrier(Mem::Global);
    }
    const int order = SD == Side::Left ? m : n;
    // kb itself is the induction variable: a block count (order + NB - 1) / NB lets LLVM narrow a
    // downward counter to i27, which the AdaptiveCpp Metal emitter rejects.
    const int kb_first = forward ? 0 : (order - 1) / NB * NB;
    const T neg_one = Ar::neg(Ar::one()), one = Ar::one();
    for (int kb = kb_first; forward ? kb < order : kb >= 0; kb += forward ? NB : -NB) {
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
                nb, [=](int r, int kk) BATCHLAS_DEVICE_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, lo + r, kb + kk); },
                [=](int kk, int col) BATCHLAS_DEVICE_BODY { return b(kb + kk, col); });
            l3::for_each_c_tile<Cfg>(ex, cnt, n, [&](int m0, int n0) BATCHLAS_DEVICE_BODY {
                l3::run_tile<Cfg>(ex, smem, b.p + lo, b.ld, m0, n0, cnt, n, neg_one, one, l3::NoMask{}, pass);
            });
        } else {
            const auto pass = l3::make_pass<true, TR == Transpose::NoTrans>(
                nb, [=](int r, int kk) BATCHLAS_DEVICE_BODY { return b(r, kb + kk); },
                [=](int kk, int col) BATCHLAS_DEVICE_BODY { return l3::OperandAccessor<T, TR>::load(a.p, a.ld, 0, kb + kk, lo + col); });
            l3::for_each_c_tile<Cfg>(ex, m, cnt, [&](int m0, int n0) BATCHLAS_DEVICE_BODY {
                l3::run_tile<Cfg>(ex, smem, b.p + static_cast<std::int64_t>(lo) * b.ld, b.ld, m0, n0, m, cnt, neg_one, one, l3::NoMask{}, pass);
            });
        }
        ex.barrier(Mem::Global);
    }
}

}  // namespace batchlas::device

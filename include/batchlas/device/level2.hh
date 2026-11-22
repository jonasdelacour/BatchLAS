// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief In-kernel BLAS level 2: matrix-vector products, rank updates and triangular solves and
/// multiplies on an executor (@ref api_device_blas2).
///
/// Arguments come in BLAS order: `(ex, alpha, A, x, beta, y)` for the products and
/// `(ex, alpha, x, y, A)` for the rank updates. The algorithm is a template parameter whose
/// default is computed from the executor and the transpose, so a kernel never compiles a path it
/// does not take. The defaults, and where they leave the output:
///   - `gemv`, `NoTrans`: `alg::RowOwner`, one item per output row; `y` in the cyclic layout.
///   - `gemv`, `Trans` / `ConjTrans`: `alg::SgPerCol`, one team per output column with a team
///     reduction; `y(c)` is written by lane 0 of team `c mod nteams`, so publish `y` before a
///     cyclic read.
///   - `gemv` on `Thread`: `alg::Serial`.
///   - `symv` / `hemv` (`Uplo::Lower` only): `alg::RowOwnerSplit`; `y` in the cyclic layout.
///   - `ger` / `geru` / `gerc`: `alg::Cyclic`, columns outside, owned rows inside.
///
/// An operand that an operation reads at indices other items own (`x` of `gemv`, `y` of `ger`)
/// must be `Published`: pass `ex.publish(v)` or `assume_published(v)`. The compiler rejects an
/// unpublished one. Included by `<batchlas/device.hh>`.
/// @ingroup api_device_blas2

#include <type_traits>

#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/view.hh>

namespace batchlas::device {

namespace detail {
template <class Ex, Transpose TR> constexpr auto default_gemv_alg() {
    if constexpr (is_thread_v<Ex>) return alg::Serial{};
    else if constexpr (TR == Transpose::NoTrans) return alg::RowOwner{};
    else return alg::SgPerCol{};
}
template <class Alg, class Default> using pick_t = std::conditional_t<std::is_void_v<Alg>, Default, Alg>;
// gemv's output: store_epi<Shape, Ops>(alpha, acc, beta, y); Side::Right with a Scalar alpha
// forms acc * alpha first (then beta's term as a One-alpha store).
template <class S> inline constexpr bool is_scalar_value_v = false;
template <class T> inline constexpr bool is_scalar_value_v<Scalar<T>> = true;
template <class Shape, class Ops, Side AlphaSide, class AS, class Scalar, class BS, class Ref>
BATCHLAS_DEVICE_INLINE void gemv_put(AS alpha, Scalar acc, BS beta, Ref&& y) {
    if constexpr (AlphaSide == Side::Right && is_scalar_value_v<AS>) {
        store_epi<Shape, Ops>(One{}, arith_t<Ops, Scalar>::mul(acc, static_cast<Scalar>(alpha.v)), beta, y);
    } else {
        store_epi<Shape, Ops>(alpha, acc, beta, y);
    }
}
template <class Ex> BATCHLAS_DEVICE_INLINE auto lane_group(const Ex& ex) {
    if constexpr (requires { ex.part; }) return ex.part;
    else return ex.sub();
}
}  // namespace detail

/// @brief `y := alpha op(A) x + beta y` (BLAS gemv) on an executor.
///
/// `A` is m x n; `op(A) x` has `op(A)`'s row count. A `Zero` tag for `beta` never reads `y`.
/// Each output element is written once, by the item or lane the algorithm names; no barrier is
/// taken, so publish `y` before another owner reads it.
/// @ingroup api_device_blas2
/// @tparam TR `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam Alg `void` (default) picks `alg::Serial` on `Thread`, `alg::RowOwner` for `NoTrans` and
///   `alg::SgPerCol` otherwise. The others:
///   - `alg::TeamPerRow` (`NoTrans`): team `t` owns rows `t mod nteams`, its lanes stride the
///     columns, lane 0 stores.
///   - `alg::ColDot<W>` (`Trans` / `ConjTrans`): `W` consecutive lanes per column (`W` divides the
///     sub-group), a `W`-lane butterfly; the block's first lane stores.
///   - `alg::GroupReducePerOutput` (`Trans` / `ConjTrans`): one group reduction per column, the
///     leader stores.
/// @tparam Ops the arithmetic policy of the dot products.
/// @tparam Shape the epilogue, `store_epi<Shape, Ops>` (default `epi::Sum`).
/// @tparam AlphaSide `Side::Left` (default, `alpha * acc`) or `Side::Right` (`acc * alpha`, which
///   differs for a complex product).
/// @param ex the executor.
/// @param alpha_ the scalar `alpha`: a value or a tag.
/// @param a the matrix `A`.
/// @param xp the vector `x`, `Published` unless the executor is `Thread`.
/// @param beta_ the scalar `beta`: a value or a tag.
/// @param[in,out] y the output vector.
template <Transpose TR, class Alg = void, class Ops = arith::Explicit, class Shape = epi::Sum, Side AlphaSide = Side::Left,
          class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void gemv(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = detail::pick_t<Alg, decltype(detail::default_gemv_alg<Ex, TR>())>;
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    constexpr bool conj = TR == Transpose::ConjTrans;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    if constexpr (std::is_same_v<Al, alg::RowOwner>) {
        static_assert(TR == Transpose::NoTrans, "gemv: alg::RowOwner is the NoTrans algorithm");
        const int step = detail::size_of(ex);
        for (int r = ex.lid(); r < m; r += step) {
            T acc = Ar::zero();
#pragma unroll 4
            for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
            detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(r));
        }
    } else if constexpr (std::is_same_v<Al, alg::SgPerCol>) {
        static_assert(TR != Transpose::NoTrans, "gemv: alg::SgPerCol is the (Conj)Trans algorithm");
        const int lane = ex.lane();
        const int nteams = detail::nteams_of(ex), tsize = detail::team_size_of(ex);
        for (int c = ex.team(); c < n; c += nteams) {
            T acc = Ar::zero();
            for (int r = lane; r < m; r += tsize) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.team_reduce(acc);
            if (lane == 0) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
        }
    } else if constexpr (std::is_same_v<Al, alg::Serial>) {
        static_assert(is_thread_v<Ex>, "gemv: alg::Serial needs a one-item executor (Thread)");
        if constexpr (TR == Transpose::NoTrans) {
            for (int r = 0; r < m; ++r) {
                T acc = Ar::zero();
                for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
                detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(r));
            }
        } else {
            for (int c = 0; c < n; ++c) {
                T acc = Ar::zero();
                for (int r = 0; r < m; ++r) {
                    if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                    else Ar::mac(acc, a(r, c), x(r));
                }
                detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
            }
        }
    } else if constexpr (std::is_same_v<Al, alg::GroupReducePerOutput>) {
        static_assert(TR != Transpose::NoTrans, "gemv: alg::GroupReducePerOutput is a (Conj)Trans algorithm");
        for (int c = 0; c < n; ++c) {
            T acc = Ar::zero();
            for (int r = ex.lid(); r < m; r += detail::size_of(ex)) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.group_reduce(acc);
            if (ex.leader()) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
        }
    } else if constexpr (std::is_same_v<Al, alg::TeamPerRow>) {
        static_assert(TR == Transpose::NoTrans, "gemv: alg::TeamPerRow is a NoTrans algorithm");
        const int lane = ex.lane();
        const int nteams = detail::nteams_of(ex), tsize = detail::team_size_of(ex);
        for (int r = ex.team(); r < m; r += nteams) {
            T acc = Ar::zero();
            for (int c = lane; c < n; c += tsize) Ar::mac(acc, a(r, c), x(c));
            acc = ex.team_reduce(acc);
            if (lane == 0) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(r));
        }
    } else if constexpr (requires { Al::width; }) {
        static_assert(TR != Transpose::NoTrans, "gemv: alg::ColDot<W> is a (Conj)Trans algorithm");
        constexpr int W = Al::width;
        const int w = ex.lid() & (W - 1), g = ex.lid() / W, ng = detail::size_of(ex) / W;
        const auto grp = detail::lane_group(ex);
        for (int c = g; c < n; c += ng) {
            T acc = Ar::zero();
            for (int r = w; r < m; r += W) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = butterfly<1, W>(grp, acc, [](const T& p, const T& q) { return Ar::add(p, q); });
            if (w == 0) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
        }
    } else {
        static_assert(sizeof(Al) == 0, "gemv: unknown algorithm tag");
    }
}

namespace detail {
template <bool Herm, Uplo UL, class Alg, class Ops, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void symv_impl(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = pick_t<Alg, alg::RowOwnerSplit>;
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    static_assert(UL == Uplo::Lower, "symv/hemv: only Uplo::Lower is implemented");
    static_assert(std::is_same_v<Al, alg::RowOwnerSplit> || (Herm && std::is_same_v<Al, alg::RowOwnerSplitStoredDiag>),
                  "symv/hemv: alg::RowOwnerSplit (or, hemv only, alg::RowOwnerSplitStoredDiag)");
    constexpr bool masked_diag = Herm && Ar::is_complex && std::is_same_v<Al, alg::RowOwnerSplit>;
    require_published<Ex, X>();
    const auto& x = unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    if constexpr (is_trailing_v<MA>) {
        // Parent coordinates: rows/columns [k, n), the hand kernels' loop shape.
        static_assert(is_trailing_v<std::remove_cvref_t<decltype(x)>> && is_trailing_v<Y>,
                      "symv/hemv: a Trailing A takes Trailing x and y");
        const int k = a.first(), n = a.end();
        for (int r = k + ex.lid(); r < n; r += size_of(ex)) {
            const int c_split = sycl::min(r, n - 1);
            T acc = Ar::zero();
            if constexpr (masked_diag) {
#pragma unroll 4
                for (int c = k; c < c_split; ++c) Ar::mac(acc, a(r, c), x(c));
                Ar::mac(acc, Ar::make(Ar::re(a(r, r)), real_t<T>(0)), x(r));
            } else {
#pragma unroll 4
                for (int c = k; c <= c_split; ++c) Ar::mac(acc, a(r, c), x(c));
            }
#pragma unroll 4
            for (int c = c_split + 1; c < n; ++c) {
                if constexpr (Herm) Ar::mac_conj(acc, a(c, r), x(c));
                else Ar::mac(acc, a(c, r), x(c));
            }
            store_epi(alpha, acc, beta, y(r));
        }
    } else {
    const int n = a.rows();
    // Row r of the lower triangle (c <= r) then column r below the diagonal (c > r): ascending
    // c crosses c == r once, so the split keeps the accumulation order and removes the branch.
    for (int r = ex.lid(); r < n; r += size_of(ex)) {
        T acc = Ar::zero();
        if constexpr (masked_diag) {
#pragma unroll 4
            for (int c = 0; c < r; ++c) Ar::mac(acc, a(r, c), x(c));
            Ar::mac(acc, Ar::make(Ar::re(a(r, r)), real_t<T>(0)), x(r));  // Im(A(r,r)) is not read
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac_conj(acc, a(c, r), x(c));
        } else if constexpr (Herm) {  // StoredDiag: the diagonal joins the row loop as stored
#pragma unroll 4
            for (int c = 0; c <= r; ++c) Ar::mac(acc, a(r, c), x(c));
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac_conj(acc, a(c, r), x(c));
        } else {
#pragma unroll 4
            for (int c = 0; c <= r; ++c) Ar::mac(acc, a(r, c), x(c));
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac(acc, a(c, r), x(c));
        }
        store_epi(alpha, acc, beta, y(r));
    }
    }
}

template <bool Conj, class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void ger_impl(const Ex& ex, AS alpha_, const X& x, const Y& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    require_published<Ex, Y>();
    const auto& y = unpub(yp);
    const auto alpha = as_tag(alpha_);
    const int m = a.rows(), n = a.cols();
    if constexpr (std::is_same_v<Alg, alg::Cyclic>) {
        for (int c = 0; c < n; ++c) {
            const T yc = tag_mul(alpha, Conj ? Ar::conj(static_cast<T>(y(c))) : static_cast<T>(y(c)));
            for (int r = ex.lid(); r < m; r += size_of(ex)) Ar::mac(a(r, c), x(r), yc);
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::Flat>, "ger: alg::Cyclic or alg::Flat");
        require_published<Ex, X>();
        const auto& xv = unpub(x);
        for_each_owned_2d<alg::Walk>(ex, m, n, [&](int r, int c) {
            const T yc = tag_mul(alpha, Conj ? Ar::conj(static_cast<T>(y(c))) : static_cast<T>(y(c)));
            Ar::mac(a(r, c), static_cast<T>(xv(r)), yc);
        });
    }
}
}  // namespace detail

/// @brief `y := alpha A x + beta y` (BLAS symv), `A` symmetric with its `UL` triangle stored.
///
/// One item per output row (cyclic): row `r` reads the stored triangle's row and column through
/// `r`, so the opposite triangle is never read. `y` is in the cyclic layout. A `Trailing` view of
/// `A` (with trailing `x` and `y`) works in the parent's coordinates on rows and columns
/// `[k, n)`. A `Zero` tag for `beta` never reads `y`.
/// @ingroup api_device_blas2
/// @tparam UL the stored triangle; only `Uplo::Lower` is implemented.
/// @tparam Alg `void` (default): `alg::RowOwnerSplit`, the only form for `symv`.
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param a the n x n matrix `A`.
/// @param x the vector `x`, `Published` unless the executor is `Thread`.
/// @param beta the scalar `beta`: a value or a tag.
/// @param[in,out] y the output vector.
template <Uplo UL, class Alg = void, class Ops = arith::Explicit, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void symv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<false, UL, Alg, Ops>(ex, alpha, a, x, beta, y);
}
/// @brief `y := alpha A x + beta y` (BLAS hemv), `A` Hermitian with its `UL` triangle stored.
///
/// As `symv`, with the conjugate of the stored element used opposite the triangle. The imaginary
/// part of the diagonal is not read.
/// @ingroup api_device_blas2
/// @tparam UL the stored triangle; only `Uplo::Lower` is implemented.
/// @tparam Alg `void` (default): `alg::RowOwnerSplit`. `alg::RowOwnerSplitStoredDiag` reads the
///   diagonal as stored instead; the caller guarantees its imaginary part is 0.
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param a the n x n matrix `A`.
/// @param x the vector `x`, `Published` unless the executor is `Thread`.
/// @param beta the scalar `beta`: a value or a tag.
/// @param[in,out] y the output vector.
template <Uplo UL, class Alg = void, class Ops = arith::Explicit, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void hemv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<true, UL, Alg, Ops>(ex, alpha, a, x, beta, y);
}

/// @brief `A := A + alpha x y^T` (BLAS geru), the unconjugated rank-1 update.
///
/// Every item reads all of `y`, so `y` must be `Published` (unless the executor is `Thread`).
/// Each element of `A` is updated by the item that owns it; no barrier.
/// @ingroup api_device_blas2
/// @tparam Alg `alg::Cyclic` (default): columns outside, owned rows inside, `x` read in the cyclic
///   layout. `alg::Flat`: `for_each_owned_2d` ownership, so `x` is read across owners and must be
///   `Published` too.
/// @tparam Ops the arithmetic policy of the multiply-adds.
/// @param ex the executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param x the m-vector `x`.
/// @param y the n-vector `y`.
/// @param[in,out] a the m x n matrix `A`.
template <class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void geru(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    detail::ger_impl<false, Alg, Ops>(ex, alpha, x, y, a);
}
/// @brief `A := A + alpha x y^H` (BLAS gerc), the conjugated rank-1 update.
///
/// As `geru`, with `y` conjugated.
/// @ingroup api_device_blas2
/// @tparam Alg `alg::Cyclic` (default) or `alg::Flat`, as for `geru`.
/// @tparam Ops the arithmetic policy of the multiply-adds.
/// @param ex the executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param x the m-vector `x`.
/// @param y the n-vector `y`, `Published`.
/// @param[in,out] a the m x n matrix `A`.
template <class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void gerc(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    detail::ger_impl<true, Alg, Ops>(ex, alpha, x, y, a);
}
/// @brief `A := A + alpha x y^T` (BLAS ger): `geru`.
/// @ingroup api_device_blas2
/// @tparam Alg `alg::Cyclic` (default) or `alg::Flat`, as for `geru`.
/// @tparam Ops the arithmetic policy of the multiply-adds.
/// @param ex the executor.
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param x the m-vector `x`.
/// @param y the n-vector `y`, `Published`.
/// @param[in,out] a the m x n matrix `A`.
template <class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void ger(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    geru<Alg, Ops>(ex, alpha, x, y, a);
}
/// @brief One row of a rank-1 update held in registers: `arow[j] += xr alpha op(y(j))` for `j < live`.
///
/// The item holds row `r` of `A` in a register fragment and `x(r)` as `xr`.
/// @ingroup api_device_lowlevel
/// @tparam TR `Transpose::Trans` (default, `ger`) or `Transpose::ConjTrans` (`gerc`: `y` conjugated).
/// @tparam Ops the arithmetic policy.
/// @param alpha_ the scalar `alpha`: a value or a tag.
/// @param xr this row's element of `x`.
/// @param y the vector `y`, read at `j < live`.
/// @param[in,out] arow the row, `N` slots.
/// @param live the number of valid columns (default `N`).
template <Transpose TR = Transpose::Trans, class Ops = arith::Explicit, class AS, class Scalar, class Y, int N>
BATCHLAS_DEVICE_INLINE void ger_row(AS alpha_, const Scalar& xr, const Y& y, Frag<Scalar, N, RowFrag>& arow,
                                    int live = N) {
    static_assert(TR != Transpose::NoTrans, "ger_row: TR is Trans (x y^T) or ConjTrans (x y^H)");
    constexpr bool conj = TR == Transpose::ConjTrans;
    using Ar = arith_t<Ops, Scalar>;
    const auto alpha = as_tag(alpha_);
#pragma unroll
    for (int j = 0; j < N; ++j) {
        if (j >= live) continue;
        const Scalar yj = tag_mul(alpha, conj ? Ar::conj(static_cast<Scalar>(y(j))) : static_cast<Scalar>(y(j)));
        Ar::mac(arow[j], xr, yj);
    }
}

/// @brief Register form on `Thread`: `A := x y^T + A` on a register tile, one multiply-add per entry, row by row.
/// @ingroup api_device_blas2
/// @tparam Ops the arithmetic policy of the multiply-adds.
/// @param x the `R`-element register fragment.
/// @param y the `C`-element register fragment.
/// @param[in,out] a the `R` x `C` register tile.
template <class Ops = arith::Explicit, class Scalar, int R, int C, class L1, class L2>
BATCHLAS_DEVICE_INLINE void ger(const Thread&, const Frag<Scalar, R, L1>& x, const Frag<Scalar, C, L2>& y, Tile<Scalar,
                                R, C>& a) {
    using Ar = arith_t<Ops, Scalar>;
BATCHLAS_UNROLL_FULL
    for (int i = 0; i < R; ++i) {
BATCHLAS_UNROLL_FULL
        for (int j = 0; j < C; ++j) Ar::mac(a(i, j), x[i], y[j]);
    }
}

// ---- the guarded diagonal step of trsv's RowFrag block form ----------------------------------
/// @brief The diagonal step object of `trsv`'s row-fragment block form (`alg::DiagApply`).
///
/// Built by `guarded_recip` for the diagonal element of row `lid()`.
/// @ingroup api_device_lowlevel
template <class Scalar> struct GuardedRecip {
    /// @{
    /// The diagonal element `d` and its reciprocal `rinv`.
    Scalar d, rinv;
    /// @}
    /// @{
    /// `divide`: some reciprocal in the partition is not finite, so divide by `d`. `unit`: the
    /// diagonal is not applied.
    bool divide, unit;
    /// @}
};
/// @brief The `GuardedRecip` of the diagonal element `d` of row `lid()`.
///
/// `divide` is set when any item's reciprocal is not finite (a bitwise-or team reduction,
/// shuffles only). COLLECTIVE over the partition.
/// @ingroup api_device_lowlevel
/// @tparam Ar the arithmetic used for the reciprocal.
/// @param ex the partition executor.
/// @param d this row's diagonal element.
/// @param unit `true` when the diagonal is not to be applied.
/// @return the step object.
template <class Ar, int P, class Scalar> BATCHLAS_DEVICE_INLINE GuardedRecip<Scalar> guarded_recip(const Partition<P>& ex, Scalar d, bool unit) {
    const Scalar rinv = Ar::recip(d);
    int bad = Ar::isfinite(rinv) ? 0 : 1;
    bad = ex.team_reduce(bad, sycl::bit_or<int>());
    return {d, rinv, bad != 0, unit};
}

// ---- subtract-from and multi-term gemv, triangular solve / multiply -------------------------

/// @brief `y := b - A x`, each row's terms subtracted from `b(r)` in column order (`Ops::msub`).
///
/// Row owner (cyclic). The rounding differs from `gemv` with `alpha = -1`, `beta = 1`, which forms
/// `A x` first. `y` may alias `b`.
/// @ingroup api_device_lowlevel
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param a the m x n matrix.
/// @param xp the n-vector `x`, `Published` unless the executor is `Thread`.
/// @param b the m-vector `b`.
/// @param[out] y the m-vector result.
template <class Ops = arith::Explicit, class Ex, class MA, class X, class B, class Y>
BATCHLAS_DEVICE_INLINE void gemv_sub(const Ex& ex, const MA& a, const X& xp, const B& b, const Y& y) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const int m = a.rows(), n = a.cols();
    for_each_owned(ex, m, [&](int r) {
        T acc = b(r);
        for (int c = 0; c < n; ++c) Ar::msub(acc, static_cast<T>(a(r, c)), static_cast<T>(x(c)));
        y(r) = acc;
    });
}

/// @brief One term `A_t x_t` of `gemv_acc`; build it with `term`.
/// @ingroup api_device_lowlevel
template <class MA, class X> struct GemvTerm {
    /// The matrix `A_t`.
    MA a;
    /// The vector `x_t`, `Published`.
    X x;
};
/// @brief The `gemv_acc` term `A x`.
/// @ingroup api_device_lowlevel
/// @param a the matrix.
/// @param x the vector, `Published`.
/// @return the term.
template <class MA, class X> BATCHLAS_DEVICE_INLINE GemvTerm<MA, X> term(const MA& a, const X& x) { return {a, x}; }

/// @brief `y := y0 + alpha sum_t A_t x_t` in one accumulator per row (`NoTrans`, row owner).
///
/// The terms' products are summed in argument order, then scaled once. Every `x_t` is read by
/// every item, so each is `Published`. `y` may alias `y0`.
/// @ingroup api_device_lowlevel
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param alpha_ the scalar `alpha`: a value or a tag.
/// @param y0 the vector added.
/// @param[out] y the result; its size is the row count.
/// @param terms the `GemvTerm`s, from `term(A, x)`.
template <class Ops = arith::Explicit, class Ex, class AS, class Y0, class Y, class... Terms>
BATCHLAS_DEVICE_INLINE void gemv_acc(const Ex& ex, AS alpha_, const Y0& y0, const Y& y, const Terms&... terms) {
    using T = elem_t<Y>;
    using Ar = arith_t<Ops, T>;
    (detail::require_published<Ex, decltype(terms.x)>(), ...);
    const auto alpha = as_tag(alpha_);
    for_each_owned(ex, y.size(), [&](int r) {
        T acc = Ar::zero();
        auto one = [&](const auto& t) {
            const auto& x = detail::unpub(t.x);
            for (int c = 0; c < t.a.cols(); ++c) Ar::mac(acc, static_cast<T>(t.a(r, c)), static_cast<T>(x(c)));
        };
        (one(terms), ...);
        y(r) = Arith<T>::add(static_cast<T>(y0(r)), tag_mul(alpha, acc));
    });
}

namespace detail {
template <class X> constexpr int tri_cap_of(const X*) { return 0; }
template <class T, int N, class L> constexpr int tri_cap_of(const Frag<T, N, L>*) { return N; }

// Element (i, j) of op(A) for a stored UL triangle; called only inside op(A)'s triangle.
template <Transpose TR, class Scalar, class MA> BATCHLAS_DEVICE_INLINE Scalar op_at(const MA& a, int i, int j) {
    if constexpr (TR == Transpose::NoTrans) return a(i, j);
    else if constexpr (TR == Transpose::Trans) return a(j, i);
    else return Arith<Scalar>::conj(static_cast<Scalar>(a(j, i)));
}
template <class A> inline constexpr bool is_split_dot_v = false;
template <int W> inline constexpr bool is_split_dot_v<alg::SplitDot<W>> = true;

// One aligned 16-byte load of consecutive elements (one element when T is 16 bytes).
template <class T> struct alignas(16) Pack16 {
    T e[16 / sizeof(T) > 0 ? 16 / sizeof(T) : 1];
};
template <class T> inline constexpr int pack16_elems = static_cast<int>(sizeof(Pack16<T>::e) / sizeof(T));

// alg::SplitDot<W>, trsv (Solve) and trmv: x_i combines column i of A with the other entries of
// x (Below: p < i, else p > i); Solve: x_i = (x_i - sum) / a_ii, else x_i = a_ii x_i + sum. In
// place, so i is visited before every entry whose sum still needs it.
template <bool Below, bool Solve, bool Conj, int W, class Ops, class Scalar, int N, class L, class MA>
BATCHLAS_DEVICE_INLINE void split_dot_tri(const MA& a, Frag<Scalar, N, L>& x, int n) {
    static_assert(W > 0 && (W & (W - 1)) == 0, "trsv / trmv alg::SplitDot<W>: W is a power of two");
    using A = Arith<Scalar>;
    using Ar = arith_t<Ops, Scalar>;
    constexpr int V = pack16_elems<Scalar>;
    constexpr int K = static_extent_v<std::remove_cv_t<decltype(a.n)>>;  // -1: a run-time extent
    constexpr int Trip = K > 0 ? K : N;
    static_assert(Trip <= N, "trsv / trmv alg::SplitDot<W>: the extent exceeds the fragment");
    constexpr bool ascending = Below == Solve;
    auto step = [&](Scalar& acc, const Scalar& aij, const Scalar& xj) BATCHLAS_DEVICE_BODY {
        if constexpr (Solve) Ar::msub(acc, Conj ? A::conj(aij) : aij, xj);  // acc - op(a) x
        else if constexpr (Conj) Ar::mac_conj(acc, aij, xj);
        else Ar::mac(acc, aij, xj);
    };
    BATCHLAS_UNROLL_FULL
    for (int s = 0; s < Trip; ++s) {
        const int i = ascending ? s : Trip - 1 - s;
        if (i < n) {
            Scalar acc[W];
            BATCHLAS_UNROLL_FULL
            for (int k = 0; k < W; ++k) acc[k] = A::zero();
            const Scalar* const line = &a(0, i);  // column i of A
            if constexpr (Below) {
                BATCHLAS_UNROLL_FULL
                for (int p0 = 0; p0 < i; p0 += V) {
                    const Pack16<Scalar> pk = *reinterpret_cast<const Pack16<Scalar>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u < i) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            } else {
                BATCHLAS_UNROLL_FULL
                for (int p0 = (i + 1) / V * V; p0 < Trip; p0 += V) {
                    if (p0 >= n) continue;
                    const Pack16<Scalar> pk = *reinterpret_cast<const Pack16<Scalar>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u > i && p0 + u < n) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            }
            BATCHLAS_UNROLL_FULL
            for (int w = W / 2; w >= 1; w /= 2) {
                BATCHLAS_UNROLL_FULL
                for (int k = 0; k < w; ++k) acc[k] = A::add(acc[2 * k], acc[2 * k + 1]);
            }
            const auto d = A::re(line[i]);
            if constexpr (Solve) x[i] = div_real(A::add(x[i], acc[0]), d);
            else x[i] = A::add(scale_real(d, x[i]), acc[0]);
        }
    }
}

}  // namespace detail

/// @brief `x := op(A) x` (BLAS trmv) in place on one item, `A` n x n triangular with its `UL` triangle stored.
///
/// `x` is a register fragment `Frag<T, N>` (`n <= N` live entries, every index a compile-time
/// constant) or a memory vector (run-time `n`). Entries past `n` are not touched; the opposite
/// triangle is never read. `trsv` solves with the same operands.
/// @ingroup api_device_blas2
/// @tparam UL the stored triangle.
/// @tparam TR `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` never reads the diagonal; `Diag::NonUnit` multiplies by it.
/// @tparam Ops the arithmetic policy.
/// @tparam Alg `alg::Serial` (default), or `alg::SplitDot<W>` (`Trans` / `ConjTrans` with
///   `Diag::NonUnit`, the diagonal read as real: `x_i = re(a_ii) x_i` plus the column dot split
///   over `W` accumulators, 16-byte column loads, every column of `A` 16-byte aligned).
/// @param a the matrix `A`.
/// @param[in,out] x the vector.
/// @param n the order.
template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class Alg = alg::Serial, class MA, class X>
BATCHLAS_DEVICE_INLINE void trmv(const Thread&, const MA& a, X& x, int n) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
    constexpr bool lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    if constexpr (detail::is_split_dot_v<Alg>) {
        static_assert(TR != Transpose::NoTrans && DG == Diag::NonUnit, "trmv: alg::SplitDot<W> is the (Conj)Trans form with a real diagonal");
        detail::split_dot_tri<UL == Uplo::Upper, false, TR == Transpose::ConjTrans, Alg::width, Ops>(a, x, n);
        return;
    } else {
        static_assert(std::is_same_v<Alg, alg::Serial>, "trmv: alg::Serial or alg::SplitDot<W>");
    }
    // Lower: x(i) depends on x(0..i), so go from the bottom up; upper from the top down.
    auto row = [&](int i, auto&& xs) {
        T acc = DG == Diag::Unit ? T(xs(i)) : Ar::mul(detail::op_at<TR, T>(a, i, i), T(xs(i)));
        if constexpr (lower)
            for (int j = 0; j < i; ++j) Ar::mac(acc, detail::op_at<TR, T>(a, i, j), T(xs(j)));
        else
            for (int j = i + 1; j < n; ++j) Ar::mac(acc, detail::op_at<TR, T>(a, i, j), T(xs(j)));
        return acc;
    };
    if constexpr (N > 0) {
        auto xs = [&](int j) -> T { return x[j]; };
        if constexpr (lower) {
#pragma unroll
            for (int i = N - 1; i >= 0; --i)
                if (i < n) x[i] = row(i, xs);
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i)
                if (i < n) x[i] = row(i, xs);
        }
    } else {
        auto xs = [&](int j) -> T { return x(j); };
        if constexpr (lower)
            for (int i = n - 1; i >= 0; --i) x(i) = row(i, xs);
        else
            for (int i = 0; i < n; ++i) x(i) = row(i, xs);
    }
}
/// @brief `x := op(A) x` in place on one item for a memory vector of order `x.size()`, `alg::Serial`.
///
/// `x` is taken by const view: the view is const, not the data it refers to.
/// @ingroup api_device_blas2
/// @tparam UL the stored triangle.
/// @tparam TR `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` or `Diag::NonUnit`.
/// @tparam Ops the arithmetic policy.
/// @param ex the `Thread` executor.
/// @param a the matrix `A`.
/// @param x the vector, overwritten.
template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DEVICE_INLINE void trmv(const Thread& ex, const MA& a, const X& x) {
    X xv = x;
    trmv<UL, TR, DG, Ops>(ex, a, xv, x.size());
}

// ---- level 2 (design A.2) -----------------------------------------------------------------------

/// @brief `A := alpha x y^H + conj(alpha) y x^H + A` (BLAS her2) on the `UL` triangle of `A`.
///
/// For a real element type this is `alpha (x y^T + y x^T) + A` (syr2). The other triangle is
/// neither read nor written. Each element adds `alpha x(r) conj(y(c))`, then
/// `conj(alpha) y(r) conj(x(c))`, in that order; the diagonal's imaginary part is set to zero
/// (LAPACK zher2). `x` and `y` are read across owners, so both are `Published`.
/// @ingroup api_device_blas2
/// @tparam UL the triangle updated.
/// @tparam Alg `alg::RowOwner` (default): rows cyclic over the items, each item serial over its
///   row's columns. `alg::TeamPerRow`: team `t` owns rows `t mod nteams`, its lanes stride the
///   columns.
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param alpha_ the scalar `alpha`: a value or a tag.
/// @param xp the vector `x`, `Published`.
/// @param yp the vector `y`, `Published`.
/// @param[in,out] a the n x n matrix `A`.
template <Uplo UL, class Alg = alg::RowOwner, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void her2(const Ex& ex, AS alpha_, const X& xp, const Y& yp, const MA& a) {
    constexpr Part P = UL == Uplo::Lower ? Part::Lower : Part::Upper;
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    detail::require_published<Ex, Y>();
    const auto& x = detail::unpub(xp);
    const auto& y = detail::unpub(yp);
    const auto alpha = as_tag(alpha_);
    const auto alpha_c = [&] {
        if constexpr (detail::is_scalar_value_v<std::remove_cv_t<decltype(alpha)>>) return Scalar<T>{Ar::conj(static_cast<T>(alpha.v))};
        else return alpha;
    }();
    using S = std::remove_cv_t<decltype(alpha)>;
    constexpr bool native_tag = std::is_same_v<Ops, arith::Native> && (std::is_same_v<S, One> || std::is_same_v<S, NegOne>);
    const int m = a.rows(), n = a.cols();
    // Columns of row r that the triangle can touch: the bound keeps a Lower trip count at r + 1.
    auto col_end = [&](int r) BATCHLAS_DEVICE_BODY {
        if constexpr (P == Part::Lower) return sycl::min(n, r + 1);
        else return n;
    };
    auto upd = [&](int r, int c) BATCHLAS_DEVICE_BODY {
        if (!detail::in_part<P>(r, c)) return;
        T v;
        // Native with a One / NegOne alpha: one expression, so -ffp-contract=on fuses each product
        // into the sum as T's own operators do (the strict-port rounding, D6).
        if constexpr (native_tag && std::is_same_v<S, NegOne>)
            v = static_cast<T>(a(r, c)) - static_cast<T>(x(r)) * Ar::conj(static_cast<T>(y(c))) -
                static_cast<T>(y(r)) * Ar::conj(static_cast<T>(x(c)));
        else if constexpr (native_tag)
            v = static_cast<T>(a(r, c)) + static_cast<T>(x(r)) * Ar::conj(static_cast<T>(y(c))) +
                static_cast<T>(y(r)) * Ar::conj(static_cast<T>(x(c)));
        else {
            const T t1 = Ar::mul(static_cast<T>(x(r)), Ar::conj(static_cast<T>(y(c))));
            const T t2 = Ar::mul(static_cast<T>(y(r)), Ar::conj(static_cast<T>(x(c))));
            v = Ar::add(Ar::add(static_cast<T>(a(r, c)), tag_mul(alpha, t1)), tag_mul(alpha_c, t2));
        }
        if constexpr (Ar::is_complex)
            if (r == c) v = Ar::make(Ar::re(v), real_t<T>(0));
        a(r, c) = v;
    };
    if constexpr (std::is_same_v<Alg, alg::RowOwner>) {
        for_each_owned(ex, m, [&](int r) {
            const int ce = col_end(r);
            for (int c = 0; c < ce; ++c) upd(r, c);
        });
    } else {
        static_assert(std::is_same_v<Alg, alg::TeamPerRow>, "her2: alg::RowOwner or alg::TeamPerRow");
        const int nt = detail::nteams_of(ex), ts = detail::team_size_of(ex);
        for (int r = ex.team(); r < m; r += nt) {
            const int ce = col_end(r);
            for (int c = ex.lane(); c < ce; c += ts) upd(r, c);
        }
    }
}

namespace detail {
template <Diag DG, class Div, class Ar, class Scalar, class D> BATCHLAS_DEVICE_INLINE Scalar trsv_diag_apply(Scalar xi, const D& aii) {
    if constexpr (DG == Diag::Unit) return xi;
    else if constexpr (std::is_same_v<Div, alg::Div>) return Ar::div(xi, aii);
    else if constexpr (std::is_same_v<Div, alg::Recip>) return Ar::mul(xi, Ar::recip(aii));
    else {
        static_assert(std::is_same_v<Div, alg::RealDiv>, "trsv: alg::Div, alg::Recip, alg::RealDiv or alg::DiagApply");
        if constexpr (std::is_same_v<D, real_t<Scalar>>) return Ar::div_real(xi, aii);
        else return Ar::div_real(xi, Ar::re(aii));
    }
}
template <class Upd, class Ar, class Scalar> BATCHLAS_DEVICE_INLINE void trsv_update(Scalar& x, const Scalar& a,
                                                                                     const Scalar& xj) {
    if constexpr (std::is_same_v<Upd, alg::Fused>) Ar::msub(x, a, xj);
    else if constexpr (std::is_same_v<Upd, alg::UnfusedRight>) x = Ar::sub(x, Ar::mul(xj, a));
    else {
        static_assert(std::is_same_v<Upd, alg::Unfused>, "trsv: alg::Fused, Unfused or UnfusedRight here");
        x = Ar::sub(x, Ar::mul(a, xj));
    }
}
template <class A> inline constexpr int serial_rows_v = 0;
template <int R> inline constexpr int serial_rows_v<alg::SerialRows<R>> = R;
template <class X> inline constexpr bool is_row_frag_v = false;
template <class T, int N> inline constexpr bool is_row_frag_v<Frag<T, N, RowFrag>> = true;
// alg::DiagApply's step: x_i := d.apply<Ar>(i, x_i).
template <class Ar, class D, class Scalar> BATCHLAS_DEVICE_INLINE Scalar trsv_step(const D& d, int i, Scalar v) { return d.template apply<Ar>(i, v); }
// trsv's element type: A's (an OpView has no value_type), or a row Frag's (the RowFrag form).
template <class MA> struct trsv_value {
    using type = std::remove_cvref_t<decltype(std::declval<const MA&>()(0, 0))>;
};
template <class T, int N, class L> struct trsv_value<Frag<T, N, L>> {
    using type = T;
};

// The one trsv body: d the diagonal operand (None: op(A)'s own), b the right-hand side of
// alg::SerialRows (None: x in place) and r0 its first row.
template <Uplo UL, Transpose TR, Diag DG, class Div, class Alg, class Ops, class Upd, class Ex, class MA, class DV, class X, class B>
BATCHLAS_DEVICE_INLINE void trsv_run(const Ex& ex, const MA& a, const DV& d, X& x, int n, int r0, const B& b) {
    using T = typename trsv_value<MA>::type;
    using Ar = arith_t<Ops, T>;
    constexpr bool lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);  // forward substitution
    constexpr bool step_obj = std::is_same_v<Div, alg::DiagApply>;
    static_assert(!step_obj || (DG == Diag::NonUnit && !is_none_v<DV>), "trsv: alg::DiagApply takes the step object d (Diag::NonUnit)");
    // The step lambdas are always_inline, as the expression text they replace: plain lambdas cost
    // the transposed fused getrs registers. evidence: docs/perf/lu.md#the-fused-narrow-rhs-getrs
    auto diag = [&](auto i) BATCHLAS_DEVICE_BODY -> T {
        if constexpr (DG == Diag::Unit) return Ar::one();
        else if constexpr (!is_none_v<DV>) return static_cast<T>(d(i));
        else return detail::op_at<TR, T>(a, i, i);
    };
    // x_i after its updates: the diagonal step.
    auto finish = [&](T xi, auto i) BATCHLAS_DEVICE_BODY -> T {
        if constexpr (step_obj) return trsv_step<Ar>(d, i, xi);
        else if constexpr (DG == Diag::NonUnit && !is_none_v<DV> && std::is_same_v<Div, alg::RealDiv>)
            return detail::trsv_diag_apply<DG, Div, Ar>(xi, d(i));
        else return detail::trsv_diag_apply<DG, Div, Ar>(xi, diag(i));
    };
    auto upd = [&](T& xi, int i, auto j, const T& xj) BATCHLAS_DEVICE_BODY {
        detail::trsv_update<Upd, Ar>(xi, detail::op_at<TR, T>(a, i, j), xj);
    };
    if constexpr (is_split_dot_v<Alg>) {
        static_assert(is_thread_v<Ex> && TR != Transpose::NoTrans && DG == Diag::NonUnit && std::is_same_v<Div, alg::RealDiv> &&
                          is_none_v<DV>,
                      "trsv: alg::SplitDot<W> is the Thread (Conj)Trans form with a real diagonal (alg::RealDiv)");
        split_dot_tri<UL == Uplo::Upper, true, TR == Transpose::ConjTrans, Alg::width, Ops>(a, x, n);
    } else if constexpr (serial_rows_v<Alg> > 0) {
        static_assert(is_thread_v<Ex> && UL == Uplo::Lower && TR == Transpose::NoTrans && std::is_same_v<Upd, alg::DotThenSub> && !is_none_v<B>,
                      "trsv: alg::SerialRows<R> is the Thread Lower / NoTrans solve with alg::DotThenSub and a right-hand side b");
        constexpr int R = serial_rows_v<Alg>;
        constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
        static_assert(N > 0, "trsv: alg::SerialRows<R> solves a register Frag");
        BATCHLAS_UNROLL_FULL
        for (int k = 0; k < R; ++k) {
            const int s = r0 + k;
            if (s >= N || s >= n) continue;
            T acc = Ar::zero();
            BATCHLAS_UNROLL_FULL
            for (int t = 0; t < N; ++t) {
                if (t < s) Ar::mac(acc, a(s, t), x[t]);  // by reference: a complex element is read part by part
            }
            const T rhs = b(s);
            x[s] = finish(Ar::sub(rhs, acc), s);
        }
    } else if constexpr (std::is_same_v<Alg, alg::Serial>) {
        static_assert(is_thread_v<Ex>, "trsv: alg::Serial is a Thread algorithm");
        constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
        if constexpr (N > 0) {
            if constexpr (lower) {
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i >= n) break;
                    T xi = x[i];
#pragma unroll
                    for (int j = 0; j < i; ++j) upd(xi, i, j, x[j]);
                    x[i] = finish(xi, i);
                }
            } else {
#pragma unroll
                for (int i = N - 1; i >= 0; --i) {
                    if (i >= n) continue;
                    T xi = x[i];
#pragma unroll
                    for (int j = i + 1; j < N; ++j)
                        if (j < n) upd(xi, i, j, x[j]);
                    x[i] = finish(xi, i);
                }
            }
        } else if constexpr (lower) {
            for (int i = 0; i < n; ++i) {
                T xi = x(i);
                for (int j = 0; j < i; ++j) upd(xi, i, j, static_cast<T>(x(j)));
                x(i) = finish(xi, i);
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                T xi = x(i);
                for (int j = i + 1; j < n; ++j) upd(xi, i, j, static_cast<T>(x(j)));
                x(i) = finish(xi, i);
            }
        }
    } else if constexpr (is_row_frag_v<X>) {
        static_assert(std::is_same_v<Alg, alg::LanePerRow> || std::is_same_v<Alg, alg::LanePerRowRolled>,
                      "trsv: a RowFrag x is the alg::LanePerRow(Rolled) block form");
        static_assert(lower && std::is_same_v<Upd, alg::Axpy> && step_obj,
                      "trsv: the RowFrag block form is Lower / NoTrans with alg::Axpy and alg::DiagApply");
        constexpr int P = Ex::size;
        constexpr int QC = detail::tri_cap_of(static_cast<const X*>(nullptr));
        constexpr int Unroll = std::is_same_v<Alg, alg::LanePerRowRolled> ? 1 : P;
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
            if (r == s && !d.unit) {
                if (any_divide) [[unlikely]] {
                    BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < QC; ++j) x[j] = d.divide ? Ar::div(x[j], d.d) : Ar::mul(x[j], d.rinv);
                } else {
                    BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < QC; ++j) x[j] = Ar::mul(x[j], d.rinv);
                }
            }
            const T l = a[s];
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < QC; ++j) {
                const T xs = ex.bcast(x[j], static_cast<std::uint32_t>(s));
                Ar::mac(x[j], l, xs);
            }
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::LanePerRow>, "trsv: alg::Serial, SerialRows, SplitDot, LanePerRow or LanePerRowRolled");
        const int l = ex.lid();
        auto step = [&](int j) BATCHLAS_DEVICE_BODY {
            if constexpr (detail::tri_cap_of(static_cast<const X*>(nullptr)) == 0) {  // the lane's register
                if constexpr (DG == Diag::NonUnit) {
                    if (l == j) x = finish(x, j);
                }
                const T xj = ex.bcast(x, j);
                if (lower ? (l > j && l < n) : (l < j)) upd(x, l, j, xj);
            } else {
                constexpr int R = detail::tri_cap_of(static_cast<const X*>(nullptr));
                static_assert(R > 0, "trsv<LanePerRow>: x is the lane's register or a lane-cyclic Frag");
                constexpr int P = Ex::size;
                if constexpr (DG == Diag::NonUnit) {
                    BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < R; ++r)
                        if (l + P * r == j) x[r] = finish(x[r], j);
                }
                const T xj = frag_bcast(ex, x, j);
                BATCHLAS_UNROLL_FULL
                for (int r = 0; r < R; ++r) {
                    const int t = l + P * r;
                    if (lower ? (t > j && t < n) : (t < j)) upd(x[r], t, j, xj);
                }
            }
        };
        const int last = DG == Diag::Unit ? 1 : 0;
        if constexpr (lower) {
            for (int j = 0; j < n - last; ++j) step(j);
        } else {
            for (int j = n - 1; j >= last; --j) step(j);
        }
    }
}
}  // namespace detail
/// @brief `x := op(A)^{-1} x` (BLAS trsv) in place, `A` n x n triangular with its `UL` triangle stored.
///
/// The opposite triangle is never read and entries of `x` past `n` are untouched. `Diag::Unit`
/// never reads the diagonal (nor takes the last step, which updates nothing). A zero diagonal
/// element of a `Diag::NonUnit` `A` gives Inf or NaN, as BLAS does.
/// @ingroup api_device_blas2
/// @tparam UL the stored triangle.
/// @tparam TR `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` or `Diag::NonUnit`.
/// @tparam Div how a `Diag::NonUnit` diagonal is applied: `alg::Div` (default, a division),
///   `alg::Recip` (a multiply by the reciprocal), `alg::RealDiv` (a division by the real part), or
///   `alg::DiagApply` (a step object; see the form with a diagonal operand).
/// @tparam Alg the algorithm:
///   - `alg::Serial` (default): one item (`Thread`); `x` a register fragment `Frag<T, N>` (`n <= N`
///     live entries) or a memory vector.
///   - `alg::SplitDot<W>`: `Thread`, `Trans` / `ConjTrans`, `Diag::NonUnit` with `alg::RealDiv`;
///     each step's column dot in `W` accumulators summed as a tree, with 16-byte column loads
///     (every column of `A` 16-byte aligned).
///   - `alg::LanePerRow`: row `t` on lane `t mod size`, `x` the lane's register (`n <= size`) or a
///     lane-cyclic `Frag<T, R, Cyclic>` (row `lane + size r` in `x[r]`); step `j` finishes row `j`
///     and broadcasts it. No barrier. COLLECTIVE over the sub-group or partition.
///   - `alg::LanePerRowRolled`: the row-fragment block form with its step loop rolled.
/// @tparam Ops the arithmetic policy.
/// @tparam Upd how a solved entry updates the others:
///   - `alg::Fused` (default): one `Ops::msub`.
///   - `alg::Unfused`: a multiply `a * x_j`, then a subtraction (two roundings).
///   - `alg::UnfusedRight`: the same with `x_j * a`, the operand order of `X op(A) = B` (a complex
///     product is not bit-commutative).
///   - `alg::DotThenSub`, `alg::Axpy`: the `alg::SerialRows` and row-fragment forms.
/// @param ex the executor.
/// @param a the matrix `A`.
/// @param[in,out] x the right-hand side, overwritten by the solution.
/// @param n the order.
template <Uplo UL, Transpose TR, Diag DG, class Div = alg::Div, class Alg = alg::Serial, class Ops = arith::Explicit,
          class Upd = alg::Fused, class Ex, class MA, class X>
BATCHLAS_DEVICE_INLINE void trsv(const Ex& ex, const MA& a, X& x, int n) {
    detail::trsv_run<UL, TR, DG, Div, Alg, Ops, Upd>(ex, a, None{}, x, n, 0, None{});
}
/// @brief `x := op(A)^{-1} x` with the diagonal taken from the operand `d`.
///
/// `d` is a vector (`d(i)` is used as `a_ii`, for example the real diagonal of a Cholesky
/// factor), or, under `alg::DiagApply`, a step object: `x_i := d.apply<Ar>(i, x_i)` (a `TpDiag`,
/// or a `GuardedRecip` in the row-fragment block form). The row-fragment block form (`x` a
/// `Frag<T, QC, RowFrag>` holding row `lane` of an n x QC block, `A` the lane's row of negated
/// multipliers as a fragment) is `Uplo::Lower`, `NoTrans`, `alg::LanePerRow` or
/// `alg::LanePerRowRolled`, `alg::Axpy` and `alg::DiagApply`.
/// @ingroup api_device_blas2
/// @tparam UL the stored triangle.
/// @tparam TR `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` or `Diag::NonUnit`.
/// @tparam Div `alg::Div` (default), `alg::Recip`, `alg::RealDiv` or `alg::DiagApply`.
/// @tparam Alg the algorithm, as for the form without `d`.
/// @tparam Ops the arithmetic policy.
/// @tparam Upd the update form, as for the form without `d`.
/// @param ex the executor.
/// @param a the matrix `A`.
/// @param d the diagonal operand or step object.
/// @param[in,out] x the right-hand side, overwritten by the solution.
/// @param n the order.
template <Uplo UL, Transpose TR, Diag DG, class Div = alg::Div, class Alg = alg::Serial, class Ops = arith::Explicit,
          class Upd = alg::Fused, class Ex, class MA, class DV, class X>
BATCHLAS_DEVICE_INLINE void trsv(const Ex& ex, const MA& a, const DV& d, X& x, int n) {
    detail::trsv_run<UL, TR, DG, Div, Alg, Ops, Upd>(ex, a, d, x, n, 0, None{});
}
/// @brief Rows `[r0, r0 + R)` of an n-row forward solve (`alg::SerialRows<R>`), `b(i)` their right-hand side.
///
/// `Thread`, `Uplo::Lower`, `NoTrans`, `alg::DotThenSub`: rows below `r0` are already solved in the
/// register fragment `x`; each row's dot starts from zero and `x_i := b(i) - dot`, then the
/// diagonal step. The building block of a tiled forward solve.
/// @ingroup api_device_blas2
/// @tparam UL `Uplo::Lower`.
/// @tparam TR `Transpose::NoTrans`.
/// @tparam DG `Diag::Unit` or `Diag::NonUnit`.
/// @tparam Div how the diagonal is applied, as for the other forms.
/// @tparam Alg `alg::SerialRows<R>`.
/// @tparam Ops the arithmetic policy.
/// @tparam Upd `alg::DotThenSub`.
/// @param ex the `Thread` executor.
/// @param a the matrix `A`.
/// @param d the diagonal operand, or `None{}` for `A`'s own.
/// @param[in,out] x the register fragment; rows `[r0, r0 + R)` are written.
/// @param n the order.
/// @param r0 the first row solved.
/// @param b the right-hand side, read at the solved rows.
template <Uplo UL, Transpose TR, Diag DG, class Div, class Alg, class Ops, class Upd, class Ex, class MA, class DV, class X, class B>
BATCHLAS_DEVICE_INLINE void trsv(const Ex& ex, const MA& a, const DV& d, X& x, int n, int r0, const B& b) {
    detail::trsv_run<UL, TR, DG, Div, Alg, Ops, Upd>(ex, a, d, x, n, r0, b);
}
/// @brief `x := op(A)^{-1} x` in place on one item for a memory vector of order `x.size()`, `alg::Serial`.
///
/// `x` is taken by const view: the view is const, not the data it refers to.
/// @ingroup api_device_blas2
/// @tparam UL the stored triangle.
/// @tparam TR `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` or `Diag::NonUnit`.
/// @tparam Div `alg::Div` (default), `alg::Recip` or `alg::RealDiv`.
/// @tparam Ops the arithmetic policy.
/// @param ex the `Thread` executor.
/// @param a the matrix `A`.
/// @param x the vector, overwritten by the solution.
template <Uplo UL, Transpose TR, Diag DG, class Div = alg::Div, class Ops = arith::Explicit, class MA, class X>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DEVICE_INLINE void trsv(const Thread& ex, const MA& a, const X& x) {
    X xv = x;
    trsv<UL, TR, DG, Div, alg::Serial, Ops>(ex, a, xv, x.size());
}

/// @brief `A := diag(d) A` (`Side::Left`) or `A diag(d)` (`Side::Right`) in place.
///
/// `for_each_owned_2d` ownership, no barrier. A real `d` on a complex matrix scales both parts;
/// otherwise each element is `mul(a, d)`.
/// @ingroup api_device_blas2
/// @tparam SD `Side::Left` (row `i` scaled by `d(i)`) or `Side::Right` (column `j` by `d(j)`).
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param d the diagonal, a vector of `A`'s row count (Left) or column count (Right).
/// @param[in,out] a the matrix.
template <Side SD, class Ops = arith::Explicit, class Ex, class D, class MA>
BATCHLAS_DEVICE_INLINE void dgmm(const Ex& ex, const D& d, const MA& a) {
    using T = typename MA::value_type;
    using A = arith_t<Ops, T>;
    using S = std::remove_cvref_t<decltype(d(0))>;
    for_each_owned_2d(ex, a.rows(), a.cols(), [&](int i, int j) {
        const S s = d(SD == Side::Left ? i : j);
        const T aij = a(i, j);
        if constexpr (A::is_complex && !Arith<S>::is_complex) a(i, j) = A::make(A::re(aij) * s, A::im(aij) * s);
        else a(i, j) = A::mul(aij, static_cast<T>(s));
    });
}
/// @brief `C := diag(d) B` (`Side::Left`) or `B diag(d)` (`Side::Right`); `C` may alias `B`.
///
/// Over `C`'s extents, `for_each_owned_2d` ownership, no barrier.
/// @ingroup api_device_blas2
/// @tparam SD `Side::Left` or `Side::Right`.
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param d the diagonal.
/// @param b the source matrix.
/// @param[out] c the destination matrix.
template <Side SD, class Ops = arith::Explicit, class Ex, class D, class MB, class MC>
BATCHLAS_DEVICE_INLINE void dgmm(const Ex& ex, const D& d, const MB& b, const MC& c) {
    using T = typename MC::value_type;
    using A = arith_t<Ops, T>;
    using S = std::remove_cvref_t<decltype(d(0))>;
    for_each_owned_2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        const S s = d(SD == Side::Left ? i : j);
        const T bij = b(i, j);
        if constexpr (A::is_complex && !Arith<S>::is_complex) c(i, j) = A::make(A::re(bij) * s, A::im(bij) * s);
        else c(i, j) = A::mul(bij, static_cast<T>(s));
    });
}

/// @brief `y := alpha A x + beta y` (BLAS gbmv, `NoTrans`) for a general band `A`.
///
/// `A` is m x n with `KL` sub- and `KU` super-diagonals in LAPACK band storage:
/// `A(i, j) = ab(KU + i - j, j)`. Row owner, ascending `j`. A `Zero` tag for `beta` never reads `y`.
/// @ingroup api_device_blas2
/// @tparam KL the number of sub-diagonals, `>= 0`.
/// @tparam KU the number of super-diagonals, `>= 0`.
/// @tparam Ops the arithmetic policy.
/// @param ex the executor.
/// @param alpha_ the scalar `alpha`: a value or a tag.
/// @param ab the band storage, `KL + KU + 1` rows by n columns.
/// @param xp the n-vector `x`, `Published`.
/// @param beta_ the scalar `beta`: a value or a tag.
/// @param[in,out] y the m-vector `y`.
template <int KL, int KU, class Ops = arith::Explicit, class Ex, class AS, class MB, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void gbmv(const Ex& ex, AS alpha_, const MB& ab, const X& xp, BS beta_, const Y& y) {
    static_assert(KL >= 0 && KU >= 0, "gbmv: KL, KU >= 0");
    using T = elem_t<Y>;
    using Ar = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int n = x.size();
    for_each_owned(ex, y.size(), [&](int i) {
        T acc = Ar::zero();
        const int j0 = i - KL > 0 ? i - KL : 0, j1 = i + KU < n - 1 ? i + KU : n - 1;
        for (int j = j0; j <= j1; ++j) Ar::mac(acc, static_cast<T>(ab(KU + i - j, j)), static_cast<T>(x(j)));
        store_epi<epi::Sum, Ops>(alpha, acc, beta, y(i));
    });
}

}  // namespace batchlas::device

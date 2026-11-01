// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 level 2. Argument order is BLAS's: (ex, alpha, A, x, beta, y) and
// (ex, alpha, x, y, A). The algorithm is a template parameter whose default is a type
// computed from (executor, transpose); no kernel ever compiles the path it does not take.
//
//   gemv NoTrans      alg::RowOwner  item per output row; y in Cyclic layout
//   gemv Trans/Conj   alg::SgPerCol  team per output column, Scope::Sub; y written by lane 0
//                                    of team c mod nteams (publish y before Cyclic reads)
//   gemv on Thread    alg::Serial
//   symv/hemv Lower   alg::RowOwnerSplit (the latrd split loop); y in Cyclic layout
//   ger/geru/gerc     alg::Cyclic    column loop outside, owned rows inside

#include <type_traits>

#include <batchlas/blas/device/v2/level1.hh>

namespace batchlas::device::v2 {

namespace detail {
template <class Ex, Transpose TR> constexpr auto default_gemv_alg() {
    if constexpr (is_thread_v<Ex>) return alg::Serial{};
    else if constexpr (TR == Transpose::NoTrans) return alg::RowOwner{};
    else return alg::SgPerCol{};
}
template <class Alg, class Default> using pick_t = std::conditional_t<std::is_void_v<Alg>, Default, Alg>;
}  // namespace detail

// y := alpha op(A) x + beta y; beta == Zero never reads y.
template <Transpose TR, class Alg = void, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void gemv(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = detail::pick_t<Alg, decltype(detail::default_gemv_alg<Ex, TR>())>;
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    constexpr bool conj = TR == Transpose::ConjTrans;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    if constexpr (std::is_same_v<Al, alg::RowOwner>) {
        static_assert(TR == Transpose::NoTrans, "v2::gemv: alg::RowOwner is the NoTrans algorithm");
        for (int r = ex.lid(); r < m; r += Ex::size) {
            T acc = Ar::zero();
#pragma unroll 4
            for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
            store(alpha, acc, beta, y(r));
        }
    } else if constexpr (std::is_same_v<Al, alg::SgPerCol>) {
        static_assert(TR != Transpose::NoTrans, "v2::gemv: alg::SgPerCol is the (Conj)Trans algorithm");
        const int lane = ex.lane();
        for (int c = ex.team(); c < n; c += Ex::nteams) {
            T acc = Ar::zero();
            for (int r = lane; r < m; r += Ex::team_size) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.team_reduce(acc);
            if (lane == 0) store(alpha, acc, beta, y(c));
        }
    } else if constexpr (std::is_same_v<Al, alg::Serial>) {
        static_assert(Ex::size == 1, "v2::gemv: alg::Serial needs a one-item executor (Thread)");
        if constexpr (TR == Transpose::NoTrans) {
            for (int r = 0; r < m; ++r) {
                T acc = Ar::zero();
                for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
                store(alpha, acc, beta, y(r));
            }
        } else {
            for (int c = 0; c < n; ++c) {
                T acc = Ar::zero();
                for (int r = 0; r < m; ++r) {
                    if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                    else Ar::mac(acc, a(r, c), x(r));
                }
                store(alpha, acc, beta, y(c));
            }
        }
    } else if constexpr (std::is_same_v<Al, alg::GroupReducePerOutput>) {
        static_assert(TR != Transpose::NoTrans, "v2::gemv: alg::GroupReducePerOutput is a (Conj)Trans algorithm");
        for (int c = 0; c < n; ++c) {
            T acc = Ar::zero();
            for (int r = ex.lid(); r < m; r += Ex::size) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.group_reduce(acc);
            if (ex.leader()) store(alpha, acc, beta, y(c));
        }
    } else {
        static_assert(sizeof(Al) == 0, "v2::gemv: unknown algorithm tag");
    }
}

namespace detail {
template <bool Herm, Uplo UL, class Alg, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void symv_impl(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = pick_t<Alg, alg::RowOwnerSplit>;
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    static_assert(UL == Uplo::Lower, "v2::symv/hemv: only Uplo::Lower is implemented");
    static_assert(std::is_same_v<Al, alg::RowOwnerSplit>, "v2::symv/hemv: alg::RowOwnerSplit is the only algorithm");
    require_published<Ex, X>();
    const auto& x = unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int n = a.rows();
    // Row r of the lower triangle (c <= r) then column r below the diagonal (c > r): ascending
    // c crosses c == r once, so the split keeps the accumulation order and removes the branch.
    for (int r = ex.lid(); r < n; r += Ex::size) {
        T acc = Ar::zero();
        if constexpr (Herm && Ar::is_complex) {
#pragma unroll 4
            for (int c = 0; c < r; ++c) Ar::mac(acc, a(r, c), x(c));
            Ar::mac(acc, Ar::make(Ar::re(a(r, r)), real_t<T>(0)), x(r));  // Im(A(r,r)) is not read
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac_conj(acc, a(c, r), x(c));
        } else {
#pragma unroll 4
            for (int c = 0; c <= r; ++c) Ar::mac(acc, a(r, c), x(c));
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac(acc, a(c, r), x(c));
        }
        store(alpha, acc, beta, y(r));
    }
}

template <bool Conj, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void ger_impl(const Ex& ex, AS alpha_, const X& x, const Y& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    require_published<Ex, Y>();
    const auto& y = unpub(yp);
    const auto alpha = as_tag(alpha_);
    const int m = a.rows(), n = a.cols();
    for (int c = 0; c < n; ++c) {
        const T yc = tag_mul(alpha, Conj ? Ar::conj(static_cast<T>(y(c))) : static_cast<T>(y(c)));
        for (int r = ex.lid(); r < m; r += Ex::size) Ar::mac(a(r, c), x(r), yc);
    }
}
}  // namespace detail

// y := alpha A x + beta y, A symmetric with the UL triangle stored.
template <Uplo UL, class Alg = void, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void symv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<false, UL, Alg>(ex, alpha, a, x, beta, y);
}
// y := alpha A x + beta y, A Hermitian with the UL triangle stored (Im of the diagonal ignored).
template <Uplo UL, class Alg = void, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void hemv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<true, UL, Alg>(ex, alpha, a, x, beta, y);
}

// A := A + alpha x y^T (ger, geru) or A + alpha x y^H (gerc). y is read by every item.
template <class Alg = alg::Cyclic, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void geru(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    static_assert(std::is_same_v<Alg, alg::Cyclic>, "v2::ger: alg::Cyclic is the only algorithm");
    detail::ger_impl<false>(ex, alpha, x, y, a);
}
template <class Alg = alg::Cyclic, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void gerc(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    static_assert(std::is_same_v<Alg, alg::Cyclic>, "v2::ger: alg::Cyclic is the only algorithm");
    detail::ger_impl<true>(ex, alpha, x, y, a);
}
template <class Alg = alg::Cyclic, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void ger(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    geru<Alg>(ex, alpha, x, y, a);
}

}  // namespace batchlas::device::v2

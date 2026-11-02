// SPDX-License-Identifier: MIT
#pragma once

#include <src/math-helpers.hh>

#include <type_traits>

namespace batchlas::tridiag {

// R = [[c, s], [-s, c]];  R (f, g)^T = (r, 0)^T.
template <typename T> struct Givens {
    T c, s;
    __attribute__((always_inline)) constexpr Givens transpose() const { return {c, -s}; }
};

template <typename T> struct Annihilation { Givens<T> R; T r; };

// R (f, g)^T = (r, 0)^T, r = sign(f) hypot(f, g), scaled against overflow: LAPACK lartg.
template <typename T>
__attribute__((always_inline)) inline Annihilation<T> annihilate(T f, T g) {
    const auto res = detail::lartg(f, g);
    return {{res.c, res.s}, res.r};
}

// The CTA bulge chase's annihilate: in range (float), c and s from rsqrt + one Newton step instead of
// two divisions (r still takes sqrt); out of range (NaN included), and for double, the scaled annihilate.
// evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions
template <typename T>
__attribute__((always_inline)) inline Annihilation<T> annihilate_fast(T f, T g) {
    if constexpr (std::is_same_v<T, float>) {
        // |f|, |g| in (sqrt(safmin), sqrt(safmax/2)) keeps f^2 + g^2 finite and normal. g == 0
        // gives exactly (1, 0, f) by the final select, so an identity rotation does not diverge.
        const bool gz = (g == T(0));
        const T f_abs = sycl::fabs(f);
        const T g_abs = sycl::fabs(g);

        const T rtmin = sycl::sqrt(detail::safmin<T>());
        const T rtmax = sycl::sqrt(detail::safmax<T>() / T(2));

        T c = T(1), s = T(0), r = f;
        if (f_abs > rtmin && f_abs < rtmax && (gz || (g_abs > rtmin && g_abs < rtmax))) {
            const T t = f * f + g * g;
            T inv = sycl::rsqrt(t);
            // One fused Newton step, inv += (inv/2)(1 - t inv^2), squares the estimate's relative
            // error; the rounded t*inv and the final fma leave about 1 ulp (not correctly rounded).
            inv = sycl::fma(inv * T(0.5), sycl::fma(-(t * inv), inv, T(1)), inv);
            const T d = sycl::sqrt(t);
            const T signed_inv = sycl::copysign(inv, f);
            c = f_abs * inv;
            s = g * signed_inv;
            r = sycl::copysign(d, f);
        } else if (!gz) {
            const auto res = detail::lartg(f, g);
            c = res.c;
            s = res.s;
            r = res.r;
        }
        return {{gz ? T(1) : c, gz ? T(0) : s}, gz ? f : r};
    } else {
        return annihilate(f, g);
    }
}

// A = [[a, b], [b, c]].
template <typename T> struct Sym2x2 {
    T a, b, c;

    __attribute__((always_inline)) constexpr Sym2x2 congruence(Givens<T> R) const {
        const T c1 = R.c, sigma = -R.s;
        return {c1 * (c1 * a - b * sigma) - sigma * (b * c1 - sigma * c),
                c1 * (c1 * b + sigma * a) - sigma * (c1 * c + sigma * b),
                c1 * (c1 * c + b * sigma) + sigma * (b * c1 + sigma * a)};
    }

    // R^T A R = [[a - delta, b_next], [b_next, c + delta]], rho = s(a - c) + 2cb, delta = s rho,
    // b_next = c rho - b; tests only (pg_rotation in ql_chase.hh inlines this identity).
    struct Transfer { T delta, b_next; };
    __attribute__((always_inline)) constexpr Transfer transfer(Givens<T> R) const {
        const T rho = (a - c) * R.s + T(2) * R.c * b;
        return {R.s * rho, R.c * rho - b};
    }

    // |rt1| >= |rt2| (not rt1 >= rt2) and R A R^T = diag(rt1, rt2): LAPACK laev2, as detail::laev2.
    struct Eig { T rt1, rt2; Givens<T> R; };
    __attribute__((always_inline)) inline Eig eig() const {
        const auto [rt1, rt2, cs, sn] = detail::laev2(a, b, c);
        return {rt1, rt2, {cs, sn}};
    }
};

} // namespace batchlas::tridiag

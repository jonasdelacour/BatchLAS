// SPDX-License-Identifier: MIT
#pragma once

// One rotation of the implicit QL bulge chase on the plane (lo, hi) = (k-1, k), moving up.
// Pure: pad and `act` gating, shuffles and writes stay at the call site.

#include <src/ops/syev/tridiag/plane_rotation.hh>

namespace batchlas::tridiag {

// The pair R annihilates: (pivot, bulge) -> (r, 0). Seed: (d_m - sigma, e_{m-1}).
template <typename T> struct ChaseState { T pivot, bulge; };

// EXP: [[d_hi, e_mid], [e_mid, d_lo]] <- R A R^T, e_hi = r, e_below <- c e_below, and the new
// bulge -s e_below one plane up.
template <typename T> struct ExpRotation {
    Givens<T> R;
    T d_hi, d_lo, e_mid, e_below, e_hi, bulge;
    __attribute__((always_inline)) ChaseState<T> next() const { return {e_mid, bulge}; }
};

template <typename T>
__attribute__((always_inline)) inline ExpRotation<T> exp_rotation(ChaseState<T> st, T d_hi, T e_mid, T d_lo, T e_below) {
    const auto [R, r1] = annihilate_fast(st.pivot, st.bulge);
    const T c1 = R.c, sigma = -R.s;
    const T e_hi_new = st.pivot * c1 - st.bulge * sigma;
    const Sym2x2<T> A = Sym2x2<T>{d_hi, e_mid, d_lo}.congruence(R);
    const T ej_new = c1 * e_below;
    const T bulge_new = -e_below * sigma;
    return {R, A.a, A.c, A.b, ej_new, e_hi_new, bulge_new};
}

// PG (Pal-Walker-Kahan) carry: the previous rotation and (g, p). Seed {{1, 1}, d_m - sigma, 0}.
template <typename T> struct PgCarry { Givens<T> prev; T g, p; };

// PG: R from (g, s_prev e_lo); then the transfer rho = s(d_lo - (d_hi - p)) + 2c(c_prev e_lo),
// p = s rho, d_hi <- d_hi - p_old + p, next g = c rho - c_prev e_lo, e_hi = r.
template <typename T> struct PgRotation { Givens<T> R; T d_hi, e_hi; PgCarry<T> next; };

template <typename T>
__attribute__((always_inline)) inline PgRotation<T> pg_rotation(PgCarry<T> carry, T d_lo, T e_lo, T d_hi) {
    T g = carry.g, c = carry.prev.c, s = carry.prev.s, p = carry.p;

    const T f = s * e_lo;
    const auto [R, r1] = annihilate_fast(g, f);
    const T c1 = R.c, s1 = R.s;

    const T g2 = d_hi - p;
    const T r2 = (d_lo - g2) * s1 + T(2) * c1 * (c * e_lo);
    p = s1 * r2;

    const T d_ip1_new = g2 + p;
    g = c1 * r2 - (c * e_lo);
    c = c1;
    s = s1;
    return {R, d_ip1_new, r1, {{c, s}, g, p}};
}

} // namespace batchlas::tridiag

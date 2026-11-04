// SPDX-License-Identifier: MIT
#pragma once
// geqrf-private device LA v2 ops: the parts of the ?GEQR2 leaves v2.1 core has no strict
// spelling for (two-pass norm scalars, run-time and ascending butterflies, the head-on-lane-0
// larf, team slots, RowFrag applies). The rest comes from core. Op bodies keep the legacy
// statement boundaries: fp-contract=on fuses a*b+c only within one expression.

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::geqrf_v2 {

namespace v2 = batchlas::device;

// NaN-dropping max for shuffle butterflies (one FMNMX per step).
struct Fmax {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmax(a, b); }
};

// A complex copied as two scalars, not as a struct: larfg_scalars takes alpha by const&, and
// a struct-copied Cx<float> alpha is re-read from A after the reductions (+2 loads a column).
template <class T> BATCHLAS_DLA_INLINE T by_parts(const T& a) {
    using A = v2::Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(a), A::im(a));
    else return a;
}
// max(|re|, |im|), not |a|: squaring overflows.
template <class T> BATCHLAS_DLA_INLINE v2::real_t<T> absmax_ri(T a) {
    using A = v2::Arith<T>;
    if constexpr (A::is_complex) return sycl::fmax(sycl::fabs(A::re(a)), sycl::fabs(A::im(a)));
    else return sycl::fabs(a);
}
// |a / s|^2, dividing first; s > 0.
template <class T> BATCHLAS_DLA_INLINE v2::real_t<T> abs2_scaled(T a, v2::real_t<T> s) {
    using A = v2::Arith<T>;
    using R = v2::real_t<T>;
    if constexpr (A::is_complex) {
        const R x = A::re(a) / s;
        const R y = A::im(a) / s;
        return sycl::fma(x, x, y * y);
    } else {
        const R x = a / s;
        return x * x;
    }
}

// All-reduce over the sub-group, masks range/2 .. 1 with the range read at the call (one binary
// for every launch WG). Complex: a sum, both parts in one loop.
template <class T, class Op> BATCHLAS_DLA_INLINE T xor_down_rt(const sycl::sub_group& sg, T v, Op op) {
    using A = v2::Arith<T>;
    const std::uint32_t lanes = static_cast<std::uint32_t>(sg.get_local_linear_range());
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        static_assert(std::is_same_v<Op, sycl::plus<T>>, "geqrf_v2: a complex butterfly is a sum");
        R re = A::re(v);
        R im = A::im(v);
        for (std::uint32_t off = lanes / 2; off > 0; off >>= 1) {
            re += sycl::permute_group_by_xor(sg, re, off);
            im += sycl::permute_group_by_xor(sg, im, off);
        }
        return A::make(re, im);
    } else {
        for (std::uint32_t off = lanes / 2; off > 0; off >>= 1) v = op(v, sycl::permute_group_by_xor(sg, v, off));
        return v;
    }
}

// Masks 1 .. N/2 on a Partition executor (Partition::team_reduce is bfly::Down only).
template <int N, class Ex, class T, class Op> BATCHLAS_DLA_INLINE T asc_reduce(const Ex& ex, T v, Op op) {
    static_assert(!v2::Arith<T>::is_complex, "geqrf_v2::asc_reduce: real values only");
    return v2::detail::butterfly<N, v2::bfly::Asc>(ex.part, v, op);
}

// A(j:m, j+1:n) := (I - tau v v^H) A(j:m, j+1:n), v = [1; A(j+1:m, j)] in absolute rows and
// columns. Team t owns columns j+1+t mod nteams; lane 0 seeds w with the head row, the lanes
// stride rows j+1.. by `lanes`. Sub-group shuffles only: no work-group barrier inside.
template <class Ex, class T, class MA>
BATCHLAS_DLA_INLINE void larf_left_head(const Ex& ex, int lanes, T tau, const MA& a, int j, int m, int n) {
    using A = v2::Arith<T>;
    using F = v2::FmaArith<T>;
    const int lane = ex.lane();
    const int nt = v2::detail::nteams_of(ex);
    for (int c = j + 1 + ex.team(); c < n; c += nt) {
        T acc = (lane == 0) ? static_cast<T>(a(j, c)) : A::zero();
        for (int r = j + 1 + lane; r < m; r += lanes) acc = A::add(acc, F::mul(A::conj(static_cast<T>(a(r, j))), static_cast<T>(a(r, c))));
        const T w = xor_down_rt(ex.sub(), acc, sycl::plus<T>());
        const T f = F::mul(tau, w);
        if (lane == 0) a(j, c) = A::sub(static_cast<T>(a(j, c)), f);
        for (int r = j + 1 + lane; r < m; r += lanes) a(r, c) = A::sub(static_cast<T>(a(r, c)), F::mul(f, static_cast<T>(a(r, j))));
    }
}

// ---- slots: s[off + ...] from one base, never a moved pointer (the index form decides how
// ptxas unrolls the run-time folds: +112 LD, +4 REG otherwise) --------------------------------
template <class Ex, class T> BATCHLAS_DLA_INLINE void team_slot_put(const Ex& ex, T* slots, int off, T v) {
    if (ex.lane() == 0) slots[off + ex.team()] = v;
}
template <class T> BATCHLAS_DLA_INLINE T slot_fold_max(const T* s, int off, int k) {
    T v = s[off];
    for (int i = 1; i < k; ++i) v = sycl::fmax(v, s[off + i]);
    return v;
}
template <class T> BATCHLAS_DLA_INLINE T slot_fold_sum(const T* s, int off, int k) {
    T v = v2::Arith<T>::zero();
    for (int i = 0; i < k; ++i) v = v2::Arith<T>::add(v, s[off + i]);
    return v;
}

// ---- RowFrag applies: item / lane l holds row l, F[k] = column k --------------------------
// gemv^H partials: for lo <= k < hi, team sum of cv * F[k], stored by lane 0 at
// slots[off + team * ld + k]. N, lo and hi fold after the caller's unroll.
template <int N, class Ex, class T>
BATCHLAS_DLA_INLINE void gemv_rowfrag_partials(const Ex& ex, T cv, const T* F, int lo, int hi, T* slots, int off, int ld) {
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;  // uniform, or lanes drop out of the butterfly
        const T pw = xor_down_rt(ex.sub(), v2::FmaArith<T>::mul(cv, F[k]), sycl::plus<T>());
        if (ex.lane() == 0) slots[off + ex.team() * ld + k] = pw;
    }
}
// F[k] -= (tau * sum_t slots[off + t * ld + k]) * v for lo <= k < hi.
template <int N, class T>
BATCHLAS_DLA_INLINE void ger_rowfrag_slots(T tau, T v, T* F, int lo, int hi, const T* slots, int off, int nslots, int ld) {
    using A = v2::Arith<T>;
    using Fm = v2::FmaArith<T>;
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;
        T w = A::zero();
        for (int i = 0; i < nslots; ++i) w = A::add(w, slots[off + i * ld + k]);
        F[k] = A::sub(F[k], Fm::mul(Fm::mul(tau, w), v));
    }
}

// Three steps through a local tile, the caller's barrier between each: (1) tile[off + kk] =
// cv * F[k0 + kk] for k in [lo, hi), else 0; (2) lane < C: y[lane] = tau * sum_{r >= r0}
// tile(r, lane) in row order; (3) F[k0 + kk] -= v * y[kk].
template <int C, class T> BATCHLAS_DLA_INLINE void frag_scale_to_row(T cv, const T* F, int k0, int lo, int hi, T* tile, int off) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        const bool act = (k >= lo) && (k < hi);
        tile[off + kk] = act ? v2::FmaArith<T>::mul(cv, F[k]) : T{};
    }
}
template <int C, int NR, class T> BATCHLAS_DLA_INLINE void colsum_colperlane(int lane, T tau, const T* tile, int ldt, int r0, T* y) {
    if (lane < C) {
        T acc = T{};
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < NR; ++r) {
            if (r >= r0) acc = v2::Arith<T>::add(acc, tile[r * ldt + lane]);
        }
        y[lane] = v2::FmaArith<T>::mul(tau, acc);
    }
}
template <int C, class T> BATCHLAS_DLA_INLINE void frag_rank1(T v, const T* y, T* F, int k0, int lo, int hi) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        if ((k >= lo) && (k < hi)) F[k] = v2::Arith<T>::sub(F[k], v2::FmaArith<T>::mul(v, y[kk]));
    }
}

}  // namespace batchlas::geqrf_v2

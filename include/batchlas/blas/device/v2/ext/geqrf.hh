// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension for Householder QR (?GEQR2): xLARFG scalars, the reflector apply on
// sub-group teams and on row fragments, the explicit-FMA complex arithmetic of the POD device
// scalar, xor butterflies in a named order, and executors whose size is a launch value.
//
// Strict-port vocabulary: every op reproduces the expression tree of the in-tree geqrf kernels
// (src/ops/geqrf), so a kernel moved onto it stays bit-identical. fp-contract=on fuses a*b+c
// only within one expression, so op bodies keep the legacy statement boundaries.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/level1.hh>
#include <batchlas/util/group-collectives.hh>

// Full unroll of a compile-time trip count, spelled per backend as src/sycl/kernel_attrs.hh
// spells BATCHLAS_UNROLL_FULL (the SSCP JIT unrolls a bare `unroll` only partially).
#ifndef BATCHLAS_DLA_UNROLL_FULL
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

// xLARFG conventions. Lapack: beta real (also for complex), tau = (beta - alpha) / beta,
// identity (tau = 0) for a zero tail and a real alpha.
enum class Conv { Lapack };

namespace alg {
struct XorAsc {};      // butterfly masks 1, 2, .. size/2 (compile-time trip count)
struct XorDescDyn {};  // masks range/2 .. 1 over the run-time sub-group range
}  // namespace alg

// NaN-dropping max for shuffle butterflies (one FMNMX per step).
struct Fmax {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmax(a, b); }
};

// ---- scalar arithmetic with the POD device scalar's spelling ------------------------------
// a * b; complex = two FMAs (fma(ar, br, -ai*bi), fma(ar, bi, ai*br)).
template <class T> BATCHLAS_DLA_INLINE T mul_fma(T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)), sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
    } else {
        return a * b;
    }
}

// 1 / d; complex = Smith's algorithm (no overflow of |d|^2).
template <class T> BATCHLAS_DLA_INLINE T recip_smith(T d) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (A::is_complex) {
        const R dr = A::re(d), di = A::im(d);
        if (sycl::fabs(dr) >= sycl::fabs(di)) {
            const R r = di / dr;
            const R den = sycl::fma(di, r, dr);
            return A::make(R(1) / den, -r / den);
        }
        const R r = dr / di;
        const R den = sycl::fma(dr, r, di);
        return A::make(r / den, R(-1) / den);
    } else {
        return R(1) / d;
    }
}

// a / b; complex = Smith's algorithm.
template <class T> BATCHLAS_DLA_INLINE T div_smith(T a, T b) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (A::is_complex) {
        const R ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
        if (sycl::fabs(br) >= sycl::fabs(bi)) {
            const R r = bi / br;
            const R den = sycl::fma(bi, r, br);
            return A::make(sycl::fma(ai, r, ar) / den, sycl::fma(-ar, r, ai) / den);
        }
        const R r = br / bi;
        const R den = sycl::fma(br, r, bi);
        return A::make(sycl::fma(ar, r, ai) / den, sycl::fma(ai, r, -ar) / den);
    } else {
        return a / b;
    }
}

template <class T> BATCHLAS_DLA_INLINE bool all_finite(T a) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return sycl::isfinite(A::re(a)) && sycl::isfinite(A::im(a));
    else return sycl::isfinite(a);
}
template <class T> BATCHLAS_DLA_INLINE bool is_zero(T a) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (A::is_complex) return A::re(a) == R(0) && A::im(a) == R(0);
    else return a == T(0);
}

// max(|re|, |im|): the scale of xLARFG's two-pass norm (|a| would square and overflow).
template <class T> BATCHLAS_DLA_INLINE real_t<T> absmax_ri(T a) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return sycl::fmax(sycl::fabs(A::re(a)), sycl::fabs(A::im(a)));
    else return sycl::fabs(a);
}
// |a / s|^2, dividing first; s > 0.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2_scaled(T a, real_t<T> s) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (A::is_complex) {
        const R x = A::re(a) / s;
        const R y = A::im(a) / s;
        return sycl::fma(x, x, y * y);
    } else {
        const R x = a / s;
        return x * x;
    }
}

// ---- xor butterflies in a named order --------------------------------------------------
namespace detail {
template <class G, class T, class Op> BATCHLAS_DLA_INLINE T xor_desc_dyn(const G& g, T v, Op op) {
    using A = Arith<T>;
    const std::uint32_t lanes = static_cast<std::uint32_t>(g.get_local_linear_range());
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        static_assert(std::is_same_v<Op, sycl::plus<T>>, "v2: a complex butterfly is a sum");
        R re = A::re(v);
        R im = A::im(v);
        for (std::uint32_t off = lanes / 2; off > 0; off >>= 1) {
            re += xor_shuffle(g, re, off);
            im += xor_shuffle(g, im, off);
        }
        return A::make(re, im);
    } else {
        for (std::uint32_t off = lanes / 2; off > 0; off >>= 1) v = op(v, xor_shuffle(g, v, off));
        return v;
    }
}
}  // namespace detail

// ---- executors whose size is a launch value ----------------------------------------------
// For kernels launched with a run-time local range (one binary for every WG): sizes are read
// from the nd_item, never folded. Common interface: lid lane team leader size_dyn
// team_size_dyn nteams_dyn team_reduce group_reduce; barrier() where the scope has one.

// One sub-group; Scope::Sub is its whole scope. barrier() is a sub-group barrier.
struct DynSubGroup {
    static constexpr bool dynamic = true, sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    template <class R> using max_op = Fmax;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(sub().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int lane() const { return lid(); }
    BATCHLAS_DLA_INLINE int team() const { return 0; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    BATCHLAS_DLA_INLINE int size_dyn() const { return static_cast<int>(sub().get_local_linear_range()); }
    BATCHLAS_DLA_INLINE int team_size_dyn() const { return size_dyn(); }
    BATCHLAS_DLA_INLINE int nteams_dyn() const { return 1; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::xor_desc_dyn(sub(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
    BATCHLAS_DLA_INLINE void barrier() const { sycl::group_barrier(sub()); }
};

// The whole work-group; Scope::Group reduces through portable::reduce_over_group.
struct DynWorkGroup {
    static constexpr bool dynamic = true, sub_whole = false;
    static constexpr Scope natural_scope = Scope::Group;
    template <class R> using max_op = sycl::maximum<R>;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(sub().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int team() const { return static_cast<int>(sub().get_group_linear_id()); }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    BATCHLAS_DLA_INLINE int size_dyn() const { return static_cast<int>(it.get_local_range(0)); }
    BATCHLAS_DLA_INLINE int team_size_dyn() const { return static_cast<int>(sub().get_local_linear_range()); }
    BATCHLAS_DLA_INLINE int nteams_dyn() const { return static_cast<int>(sub().get_group_linear_range()); }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::xor_desc_dyn(sub(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_group(), v, op);
    }
    BATCHLAS_DLA_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
};

// Sub-group teams of a run-time-sized launch; Whole = one team is the executor. No barrier.
template <bool Whole> struct DynTeams {
    static constexpr bool dynamic = true, sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    template <class R> using max_op = Fmax;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(sub().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int lid() const { return lane(); }
    BATCHLAS_DLA_INLINE int team() const { return Whole ? 0 : static_cast<int>(sub().get_group_linear_id()); }
    BATCHLAS_DLA_INLINE bool leader() const { return lane() == 0; }
    BATCHLAS_DLA_INLINE int size_dyn() const { return static_cast<int>(sub().get_local_linear_range()); }
    BATCHLAS_DLA_INLINE int team_size_dyn() const { return size_dyn(); }
    BATCHLAS_DLA_INLINE int nteams_dyn() const { return Whole ? 1 : static_cast<int>(sub().get_group_linear_range()); }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::xor_desc_dyn(sub(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
};

BATCHLAS_DLA_INLINE DynTeams<true> teams_of(const DynSubGroup& ex) { return {ex.it}; }
BATCHLAS_DLA_INLINE DynTeams<false> teams_of(const DynWorkGroup& ex) { return {ex.it}; }

template <class Ex> inline constexpr bool is_dynamic_v = requires { Ex::dynamic; };

// Cyclic ownership on a run-time-sized executor; more constrained than level1's, so every v2
// op written on for_each_owned runs on these executors too.
template <class Ex, class F>
    requires(Ex::dynamic)
BATCHLAS_DLA_INLINE void for_each_owned(const Ex& ex, int n, F&& f) {
    const int s = ex.size_dyn();
    for (int i = ex.lid(); i < n; i += s) f(i);
}

namespace detail {
template <class Ex> BATCHLAS_DLA_INLINE int team_size_of(const Ex& ex) {
    if constexpr (is_dynamic_v<Ex>) return ex.team_size_dyn();
    else return Ex::team_size;
}
template <class Ex> BATCHLAS_DLA_INLINE int nteams_of(const Ex& ex) {
    if constexpr (is_dynamic_v<Ex>) return ex.nteams_dyn();
    else return Ex::nteams;
}
template <class Ex, class R> struct max_op_of {
    using type = sycl::maximum<R>;
};
template <class Ex, class R>
    requires(is_dynamic_v<Ex>)
struct max_op_of<Ex, R> {
    using type = typename Ex::template max_op<R>;
};
}  // namespace detail

// Team reduction with the butterfly order named: XorAsc on a Partition (shuffles only).
template <class Alg, int P, bool M, class Impl, class T, class Op>
BATCHLAS_DLA_INLINE T team_reduce_with(const Partition<P, M, Impl>& ex, T v, Op op) {
    static_assert(std::is_same_v<Alg, alg::XorAsc>, "v2::team_reduce_with: Partition takes alg::XorAsc");
    static_assert(!Arith<T>::is_complex, "v2::team_reduce_with: real values only");
    BATCHLAS_DLA_UNROLL_FULL
    for (std::uint32_t mask = 1u; mask < static_cast<std::uint32_t>(P); mask <<= 1) v = op(v, detail::xor_shuffle(ex.part, v, mask));
    return v;
}

// ---- operand with 64-bit element offsets ---------------------------------------------------
// i + j * ld formed in ptrdiff_t, for kernels whose legacy addressing is 64-bit.
template <class T, Mem M = Mem::Global> struct WideMat {
    using value_type = std::remove_cv_t<T>;
    static constexpr Mem mem = M;
    T* p;
    int m;
    int n;
    int ld;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const {
        return p[static_cast<std::ptrdiff_t>(i) + static_cast<std::ptrdiff_t>(j) * ld];
    }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Unit, M> col(int j) const { return {p + static_cast<std::ptrdiff_t>(j) * ld, Dyn{m}}; }
    BATCHLAS_DLA_INLINE WideMat sub(int i, int j, int r, int c) const {
        return {p + static_cast<std::ptrdiff_t>(i) + static_cast<std::ptrdiff_t>(j) * ld, r, c, ld};
    }
};

// ---- xLARFG ----------------------------------------------------------------------------
// identity: tau = 0, H = I, alpha and x unchanged. Otherwise alpha := beta and
// x := x * vfactor (use_mul) or x / vfactor (the reciprocal overflowed or vanished).
template <class T> struct LarfgScalars {
    T tau;
    T beta;
    T vfactor;
    bool use_mul;
    bool identity;
};

// The scalars from alpha, s = max |component| over (alpha, x) and ssq = sum |x_i / s|^2.
template <Conv C = Conv::Lapack, class T> BATCHLAS_DLA_INLINE LarfgScalars<T> larfg_from_ssq(T alpha, real_t<T> s, real_t<T> ssq) {
    static_assert(C == Conv::Lapack, "v2::larfg: Conv::Lapack is the only convention");
    using A = Arith<T>;
    using R = real_t<T>;
    LarfgScalars<T> out;
    out.tau = A::zero();
    out.beta = alpha;
    out.vfactor = A::one();
    out.use_mul = true;
    out.identity = true;
    const R alphi = A::im(alpha);
    if (ssq == R(0) && alphi == R(0)) return out;  // a complex alpha still needs a reflector
    if (!(s > R(0))) return out;
    const R alphr = A::re(alpha);
    const R ar = alphr / s;
    const R ai = alphi / s;
    const R nrm = sycl::sqrt(sycl::fma(ar, ar, sycl::fma(ai, ai, ssq)));
    const R beta_s = (alphr >= R(0)) ? -nrm : nrm;  // -SIGN(nrm, alphr): +0 counts as positive
    if constexpr (A::is_complex) {
        out.tau = A::make((beta_s - ar) / beta_s, -ai / beta_s);
        out.beta = A::make(beta_s * s, R(0));
    } else {
        out.tau = (beta_s - ar) / beta_s;
        out.beta = beta_s * s;
    }
    const T d = A::sub(alpha, out.beta);
    const T r = recip_smith(d);
    out.use_mul = all_finite(r) && !is_zero(r);
    out.vfactor = out.use_mul ? r : d;
    out.identity = false;
    return out;
}

// xLARFG on (alpha, x) distributed Cyclic over ex: every item seeds its max with alpha, two
// executor-scope reductions (max, then the scaled sum of squares). No barrier.
template <Conv C = Conv::Lapack, class Ex, class T, class X>
BATCHLAS_DLA_INLINE LarfgScalars<T> larfg_scalars(const Ex& ex, T alpha, const X& x) {
    using R = real_t<T>;
    R smax = absmax_ri(alpha);
    for_each_owned(ex, x.size(), [&](int i) { smax = sycl::fmax(smax, absmax_ri(static_cast<T>(x(i)))); });
    smax = ex.group_reduce(smax, typename detail::max_op_of<Ex, R>::type{});
    R ssq = R(0);  // the reduction stays outside the test: every item reaches it
    if (smax > R(0)) {
        for_each_owned(ex, x.size(), [&](int i) { ssq += abs2_scaled(static_cast<T>(x(i)), smax); });
    }
    ssq = ex.group_reduce(ssq, sycl::plus<R>());
    return larfg_from_ssq<C>(alpha, smax, ssq);
}

// x := x * vfactor or x / vfactor, the arm larfg chose. Cyclic, no barrier.
template <class Ex, class T, class X> BATCHLAS_DLA_INLINE void scal_or_div(const Ex& ex, const LarfgScalars<T>& h, const X& x) {
    if (h.use_mul) {
        for_each_owned(ex, x.size(), [&](int i) { x(i) = mul_fma(static_cast<T>(x(i)), h.vfactor); });
    } else {
        for_each_owned(ex, x.size(), [&](int i) { x(i) = div_smith(static_cast<T>(x(i)), h.vfactor); });
    }
}

// ---- reflector apply -----------------------------------------------------------------------
// C := (I - tau v v^H) C, v = unit_head(x, h): rows above h are outside the reflector, v(h) = 1
// is implied and never read. SgPerCol: team t owns columns t mod nteams, its lanes stride the
// rows, Scope::Sub only, so no work-group barrier can appear (pass conj(tau) for H^H, ?geqr2).
template <Side SD, class L, class Ex, class T, class V, class MC>
BATCHLAS_DLA_INLINE void larf(const Ex& teams, T tau, const UnitHead<V>& v, const MC& c) {
    static_assert(SD == Side::Left && std::is_same_v<L, SgPerCol>, "v2::larf: only Side::Left with SgPerCol is implemented");
    static_assert(Ex::sub_whole, "v2::larf<SgPerCol>: a team executor (Teams, DynTeams, SubGroup)");
    using A = Arith<T>;
    const int lane = teams.lane();
    const int ts = detail::team_size_of(teams);
    const int nt = detail::nteams_of(teams);
    const int m = c.rows(), n = c.cols(), h = v.head;
    for (int col = teams.team(); col < n; col += nt) {
        T acc = (lane == 0) ? static_cast<T>(c(h, col)) : A::zero();
        for (int r = h + 1 + lane; r < m; r += ts) acc = A::add(acc, mul_fma(A::conj(static_cast<T>(v.v(r))), static_cast<T>(c(r, col))));
        const T w = teams.team_reduce(acc);
        const T f = mul_fma(tau, w);
        if (lane == 0) c(h, col) = A::sub(c(h, col), f);
        for (int r = h + 1 + lane; r < m; r += ts) c(r, col) = A::sub(c(r, col), mul_fma(f, static_cast<T>(v.v(r))));
    }
}

// ---- RowFrag forms (item / lane l holds row l of a panel in registers, F[k] = column k) ------
// Lane 0 of each team writes its team-uniform value to slots[team]; publish before reading.
template <class Ex, class T> BATCHLAS_DLA_INLINE void team_slot_put(const Ex& ex, T* slots, T v) {
    if (ex.lane() == 0) slots[ex.team()] = v;
}
// Folds of k slots in index order: max seeds with slot 0, sum with zero.
template <class T> BATCHLAS_DLA_INLINE T slot_fold_max(const T* s, int k) {
    T v = s[0];
    for (int i = 1; i < k; ++i) v = sycl::fmax(v, s[i]);
    return v;
}
template <class T> BATCHLAS_DLA_INLINE T slot_fold_sum(const T* s, int k, int stride = 1) {
    T v = Arith<T>::zero();
    for (int i = 0; i < k; ++i) v = Arith<T>::add(v, s[i * stride]);
    return v;
}

// gemv^H partials: for lo <= k < hi, team sum of cv * F[k] (cv = conj(v_l)), stored by lane 0
// at slots[team * ld + k]. Shuffles only; N, lo and hi fold after the caller's unroll.
template <int N, class Ex, class T>
BATCHLAS_DLA_INLINE void gemv_rowfrag_partials(const Ex& ex, T cv, const T* F, int lo, int hi, T* slots, int ld) {
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;  // uniform, or lanes drop out of the butterfly
        const T pw = ex.team_reduce(mul_fma(cv, F[k]));
        if (ex.lane() == 0) slots[ex.team() * ld + k] = pw;
    }
}
// ger from the partials: F[k] -= (tau * sum_t slots[t * ld + k]) * v for lo <= k < hi.
template <int N, class T>
BATCHLAS_DLA_INLINE void ger_rowfrag_slots(T tau, T v, T* F, int lo, int hi, const T* slots, int nslots, int ld) {
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;
        const T w = slot_fold_sum(slots + k, nslots, ld);
        F[k] = Arith<T>::sub(F[k], mul_fma(mul_fma(tau, w), v));
    }
}

// The three steps of a RowFrag reflector apply through a local tile, the caller's barrier
// between each: (1) row[kk] = cv * F[k0 + kk] for k in [lo, hi), else 0; (2) lane < C:
// y[lane] = tau * sum_{r >= r0} tile(r, lane) in row order (ColPerLane); (3) F[k0 + kk] -= v * y[kk].
template <int C, class T> BATCHLAS_DLA_INLINE void frag_scale_to_row(T cv, const T* F, int k0, int lo, int hi, T* row) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        const bool act = (k >= lo) && (k < hi);
        row[kk] = act ? mul_fma(cv, F[k]) : T{};
    }
}
template <int C, int NR, class T>
BATCHLAS_DLA_INLINE void colsum_colperlane(int lane, T tau, const T* tile, int ldt, int r0, T* y) {
    if (lane < C) {
        T acc = T{};
        BATCHLAS_DLA_UNROLL_FULL
        for (int r = 0; r < NR; ++r) {
            if (r >= r0) acc = Arith<T>::add(acc, tile[r * ldt + lane]);
        }
        y[lane] = mul_fma(tau, acc);
    }
}
template <int C, class T> BATCHLAS_DLA_INLINE void frag_rank1(T v, const T* y, T* F, int k0, int lo, int hi) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        if ((k >= lo) && (k < hi)) F[k] = Arith<T>::sub(F[k], mul_fma(v, y[kk]));
    }
}

}  // namespace batchlas::device::v2

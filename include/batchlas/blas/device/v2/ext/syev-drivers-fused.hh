// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops of the syev drivers: tridiagonal phase similarities, Hermitian completion,
// 2-D batched glue (dgmm lift, reflector repack) and column-owner larf / org2l / orm2l.
// Operands are anything with operator()(i) / (i, j), so v1 views keep their own address
// arithmetic (strict ports). Uplo stays runtime only where the legacy kernels branch on it.

#include <complex>
#include <cstdint>
#include <type_traits>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/level1.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

// NativeArith<T>: Arith<T> with the std::complex operators (Annex G multiply), so a strict port
// keeps its SASS; moving a call site to Arith<T> is an algo change (drops the isnan recovery).
template <class T> struct NativeArith : Arith<T> {};
template <class R> struct NativeArith<std::complex<R>> : Arith<std::complex<R>> {
    using T = std::complex<R>;
    // const T& as std::complex's own operators take them: an operand is read where it is used.
    BATCHLAS_DLA_INLINE static T mul(const T& a, const T& b) { return a * b; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, const T& a, const T& b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, const T& a, const T& b) { acc += std::conj(a) * b; }
    BATCHLAS_DLA_INLINE static T conj(const T& a) { return std::conj(a); }
    BATCHLAS_DLA_INLINE static T add(const T& a, const T& b) { return a + b; }
    BATCHLAS_DLA_INLINE static T sub(const T& a, const T& b) { return a - b; }
    BATCHLAS_DLA_INLINE static T neg(const T& a) { return -a; }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const R& s) { return a / s; }
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) { c -= a * b; }
};

namespace detail {
// a / s for a real s, componentwise.
template <class Ar, class T, class R> BATCHLAS_DLA_INLINE T div_real(T a, R s) {
    if constexpr (requires { Ar::div_real(a, s); }) return Ar::div_real(a, s);
    else if constexpr (Ar::is_complex) return Ar::make(Ar::re(a) / s, Ar::im(a) / s);
    else return a / s;
}
// A real value as the scalar type T of the output (complex: imaginary part 0).
template <class Ar, class T, class R> BATCHLAS_DLA_INLINE T lift(R x) {
    if constexpr (Ar::is_complex && !std::is_same_v<std::remove_cvref_t<R>, T>) return T(x, typename Ar::real_t(0));
    else return x;
}
// c -= a * b as one expression, so a real update contracts to an FMA like the legacy `-=`.
template <class Ar, class T, class A> BATCHLAS_DLA_INLINE void msub(T& c, const A& a, const T& b) {
    if constexpr (requires { Ar::msub(c, a, b); }) Ar::msub(c, a, b);
    else if constexpr (Ar::is_complex) c = Ar::sub(c, Ar::mul(a, b));
    else c -= a * b;
}
// Broadcast lane k of an executor: a Partition reads its own chunk, Thread is the identity.
template <class Ex, class T> BATCHLAS_DLA_INLINE T bcast(const Ex& ex, T v, int k) {
    if constexpr (is_thread_v<Ex>) {
        return v;
    } else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(ex.part)>, sycl::sub_group>) {
        const int base = static_cast<int>(ex.part.get_local_linear_id()) & ~(Ex::size - 1);
        return sycl::select_from_group(ex.part, v, static_cast<std::uint32_t>(base + k));
    } else {
        return select_from_group(ex.part, v, static_cast<std::uint32_t>(k));  // ADL: SubGroupPartition
    }
}
// The chunk-local barrier of a Partition (orders its local-memory traffic).
template <class Ex> BATCHLAS_DLA_INLINE void chunk_barrier(const Ex& ex) {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(ex.part)>, sycl::sub_group>) sycl::group_barrier(ex.part);
    else group_barrier(ex.part);  // ADL: SubGroupPartition
}
}  // namespace detail

// v1 batched views read as item b; the view's own operator()(i, b) / (i, j, b) addresses.
template <class View> struct ItemVec {
    View v;
    int b;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(i, b); }
};
template <class View> BATCHLAS_DLA_INLINE ItemVec<View> item_vec(const View& v, int b) { return {v, b}; }
template <class View> struct ItemMat {
    View m;
    int b;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return m(i, j, b); }
};
template <class View> BATCHLAS_DLA_INLINE ItemMat<View> item_mat(const View& m, int b) { return {m, b}; }
// Column j of any (i, j) operand, n long.
template <class M> struct ColRef {
    M m;
    int j;
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return m(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class M> BATCHLAS_DLA_INLINE ColRef<M> col_ref(const M& m, int j, int n) { return {m, j, n}; }

// Local-accessor tile as acc[base + i + j * LD] through a pointer to the kernel's accessor, as the
// legacy CTA code indexed it (a v2::Mat pointer or an accessor copy changes LDS/REG: not strict).
template <class Acc, int N, int LD> struct AccCol {
    const Acc* a;
    int base, j;
    using value_type = std::remove_cvref_t<decltype(std::declval<const Acc&>()[0])>;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return (*a)[base + i + j * LD]; }
    BATCHLAS_DLA_INLINE int size() const { return N; }
    BATCHLAS_DLA_INLINE AccCol tail(int k) const { return {a, base + k, j}; }
};
template <class Acc, int N, int LD> struct AccMat {
    const Acc* a;
    int base;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return (*a)[base + i + j * LD]; }
    BATCHLAS_DLA_INLINE AccCol<Acc, N, LD> col(int j) const { return {a, base, j}; }
};
// N x N tile with leading dimension LD at acc[base]; acc_vec is its N-long vector (acc[base + i]).
template <int N, int LD, class Acc> BATCHLAS_DLA_INLINE AccMat<Acc, N, LD> acc_mat(const Acc& a, int base) { return {&a, base}; }
template <int N, class Acc> BATCHLAS_DLA_INLINE AccCol<Acc, N, 0> acc_vec(const Acc& a, int base) { return {&a, base, 0}; }

// A QL Householder vector of a P-long frame: element `head` is an implied 1, elements below it
// are 0, elements above it are stored (LAPACK xORG2L / xORM2L convention; UnitHead is QR's).
template <class V> struct UnitHeadQL {
    V v;
    int head;
    using value_type = elem_t<V>;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        using A = Arith<value_type>;
        value_type x = A::zero();
        if (i < head) x = v(i);
        else if (i == head) x = A::one();
        return x;
    }
};
template <class V> BATCHLAS_DLA_INLINE UnitHeadQL<V> unit_head_ql(V v, int head) { return {v, head}; }

// Flat id over batch x n x n (column-major per item): the item first, so a masked item leaves early.
struct RowCol {
    std::int32_t r, c;
};
BATCHLAS_DLA_INLINE std::int32_t batched_item(std::int64_t idx, std::int32_t n) {
    return static_cast<std::int32_t>(idx / (static_cast<std::int64_t>(n) * n));
}
BATCHLAS_DLA_INLINE RowCol batched_rc(std::int64_t idx, std::int32_t b, std::int32_t n) {
    const std::int64_t rem = idx - static_cast<std::int64_t>(b) * n * n;
    return {static_cast<std::int32_t>(rem % n), static_cast<std::int32_t>(rem / n)};
}

namespace diag {
struct Keep {};  // the diagonal is left as stored
struct Real {};  // the diagonal's imaginary part is dropped (Hermitian diagonals are real)
}  // namespace diag

// Element (i, j) of the Hermitian matrix whose `upper` (else lower) triangle A stores.
// herm_load assigns it to v in each branch (no merge of its own, so a caller's guard and this
// select fold into one CFG, as when written inline).
template <class Ar = void, class T, class M> BATCHLAS_DLA_INLINE void herm_load(T& v, const M& A, bool upper, int i, int j) {
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    if (i == j) {
        v = AR::make(AR::re(A(i, j)), real_t<T>(0));
    } else if (upper) {
        v = (i < j) ? A(i, j) : AR::conj(A(j, i));
    } else {
        v = (i > j) ? A(i, j) : AR::conj(A(j, i));
    }
}
template <class Ar = void, class M> BATCHLAS_DLA_INLINE auto herm_at(const M& A, bool upper, int i, int j) {
    std::remove_cvref_t<decltype(A(0, 0))> v{};
    herm_load<Ar>(v, A, upper, i, j);
    return v;
}

// mirror (xLACPY-style completion in place): writes destination entry (i, j) of the triangle
// opposite Src as conj(A(j, i)); the diagonal follows D. Entries of Src are not touched.
template <Uplo Src, class D = diag::Keep, class Ar = void, class M> BATCHLAS_DLA_INLINE void mirror_at(const M& A, int i, int j) {
    using T = std::remove_cvref_t<decltype(A(0, 0))>;
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    constexpr bool dst_lower = (Src == Uplo::Upper);
    if (dst_lower ? (i > j) : (i < j)) {
        A(i, j) = AR::conj(A(j, i));
    } else if (i == j) {
        // T(re, 0) constructed in place, not AR::make's returned value: the store then stays an
        // integer store of 0 and is not merged with the other arm's (cf: STG count, strict).
        if constexpr (std::is_same_v<D, diag::Real> && AR::is_complex) A(i, j) = T(AR::re(A(i, j)), real_t<T>(0));
    }
}

// C(i, j) := d(i) * B(i, j); a real B is lifted to C's scalar type. C may alias B.
template <class Ar = void, class Dv, class Bm, class Cm> BATCHLAS_DLA_INLINE void dgmm_left_at(const Dv& d, const Bm& B, const Cm& C, int i, int j) {
    using T = std::remove_cvref_t<decltype(C(0, 0))>;
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    C(i, j) = AR::mul(d(i), detail::lift<AR, T>(B(i, j)));
}

// Hermitian tridiagonal (d, e) -> real symmetric (dr, er = |e|) by T' = S^H T S:
// S(0) = 1, S(i+1) = S(i) * f(e_i) / |e_i| (S(i) when e_i == 0), f = identity when e is the
// subdiagonal (Uplo::Lower) and conj when it is the superdiagonal. want_s == false writes
// dr and er only. Every step reads S(i) back from s.
template <class Ar = void, class Dv, class Ev, class Drv, class Erv, class Sv>
BATCHLAS_DLA_INLINE void phase_similarity(const Thread&, int n, const Dv& d, const Ev& e, const Drv& dr, const Erv& er, const Sv& s,
                                          Uplo uplo, bool want_s) {
    using T = std::remove_cvref_t<decltype(e(0))>;
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    using R = real_t<T>;
    if (want_s) s(0) = AR::make(R(1), R(0));
    for (int i = 0; i < n; ++i) dr(i) = AR::re(d(i));
    for (int i = 0; i < n - 1; ++i) {
        const T ei = e(i);
        const R abs_e = sycl::hypot(AR::re(ei), AR::im(ei));
        er(i) = abs_e;
        if (!want_s) continue;
        if (abs_e == R(0)) {
            s(i + 1) = s(i);
        } else {
            const T sub = (uplo == Uplo::Lower) ? ei : AR::make(AR::re(ei), -AR::im(ei));
            const T f = detail::div_real<AR>(sub, abs_e);  // before s(i) is read, as `S(i) * (sub / |e|)` does
            s(i + 1) = AR::mul(s(i), f);
        }
    }
}

// Real symmetric tridiagonal: e := |e| by the +-1 similarity S (S(0) = 1, S(i+1) = S(i) sign(e_i)).
template <class Ev, class Sv> BATCHLAS_DLA_INLINE void sign_similarity(const Thread&, int n, const Ev& e, const Sv& s) {
    using T = std::remove_cvref_t<decltype(e(0))>;
    s(0) = T(1);
    for (int i = 0; i < n - 1; ++i) {
        const T ei = e(i);
        const T sgn = (ei >= T(0)) ? T(1) : T(-1);
        s(i + 1) = s(i) * sgn;
        e(i) = sycl::fabs(ei);
    }
}

// The same phase chain replayed per lane on a Partition (no scratch): lane l returns S(l) of
// the superdiagonal form, S(i+1) = S(i) * conj(e_i) / |e_i|, from e_lane / abs_lane held by
// lane i. 2 (n - 1) broadcasts; the lane-local product order is the serial one.
template <class Ar = void, class Ex, class T, class R> BATCHLAS_DLA_INLINE T phase_replay(const Ex& ex, T e_lane, R abs_lane, int n) {
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    const int lane = ex.lid();
    T phase = AR::one();
    for (int i = 0; i < n - 1; ++i) {
        const T e_i = detail::bcast(ex, e_lane, i);
        const R abs_i = detail::bcast(ex, abs_lane, i);
        if (lane > i && abs_i != R(0)) phase = AR::mul(phase, detail::div_real<AR>(AR::conj(e_i), abs_i));
    }
    return phase;
}

// Element (r, c) of the QR/QL layout ormqx expects, from a SYTRD output A (n x n):
// Uplo::Lower -> QR reflectors in columns 1..n-1 (AQ(r, c) = A(r, c-1), r > c);
// Uplo::Upper -> QL reflectors in columns 0..n-2 (AQ(r, c) = A(r, c+1), r < c); 0 elsewhere.
// With a TAU / TAUQ pair (not NoTau), row 0 also writes TAUQ(c): tau shifted the same way.
struct NoTau {};
template <class Am, class AQm, class Tv = NoTau, class TQv = NoTau>
BATCHLAS_DLA_INLINE void sytrd_reflector_pack_at(const Am& A, const AQm& AQ, Uplo uplo, int n, int r, int c, const Tv& tau = {},
                                                 const TQv& tauq = {}) {
    using T = std::remove_cvref_t<decltype(AQ(0, 0))>;
    constexpr bool kTau = !std::is_same_v<Tv, NoTau>;
    AQ(r, c) = T(0);
    if (uplo == Uplo::Lower) {
        if (c >= 1) {
            const int i = c - 1;
            if (r >= (c + 1) && r < n) AQ(r, c) = A(r, i);
        }
        if constexpr (kTau) {
            if (r == 0) tauq(c) = (c >= 1 && (c - 1) < (n - 1)) ? tau(c - 1) : T(0);
        }
    } else {
        if (c <= n - 2) {
            const int k = c + 1;
            if (r < c) AQ(r, c) = A(r, k);
        }
        if constexpr (kTau) {
            if (r == 0) tauq(c) = (c < (n - 1)) ? tau(c) : T(0);
        }
    }
}
// TAUQ(j) of the same repack as a separate pass.
template <class Tv, class TQv> BATCHLAS_DLA_INLINE void sytrd_tau_pack_at(const Tv& tau, const TQv& tauq, Uplo uplo, int n, int j) {
    using T = std::remove_cvref_t<decltype(tauq(0))>;
    if (uplo == Uplo::Lower) {
        if (j == 0) tauq(j) = T(0);
        else tauq(j) = (j - 1 < (n - 1)) ? tau(j - 1) : T(0);
    } else {
        if (j < (n - 1)) tauq(j) = tau(j);
        else tauq(j) = T(0);
    }
}

template <class T, int N, class L> BATCHLAS_DLA_INLINE void fill(Frag<T, N, L>& c, T a) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < N; ++r) c[r] = a;
}
// c(r) = beta at r == j, alpha elsewhere (xLASET on one column).
template <class T, int N, class L> BATCHLAS_DLA_INLINE void laset(Frag<T, N, L>& c, int j, T alpha, T beta) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < N; ++r) c[r] = (r == j) ? beta : alpha;
}
// A(i, j) of the n x n identity padded by 0 (xLASET alpha / beta on the leading block), one
// element: the caller keeps the loop. A row loop inside an op is unrolled before inlining (constant
// trip) or gets an inverted latch (run-time trip) that ptxas unrolls 16x deeper at P = 32 (strict).
template <class T, class M> BATCHLAS_DLA_INLINE void laset_at(const M& A, int i, int j, int n, T alpha, T beta) {
    A(i, j) = (i == j && i < n) ? beta : alpha;
}
// y(r) = c(r) for r < N.
template <class T, int N, class L, class Y> BATCHLAS_DLA_INLINE void store(const Frag<T, N, L>& c, const Y& y) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < N; ++r) y(r) = c[r];
}
// Run-time sizes below are `const int&`: an n that lives in the kernel closure is then re-read
// where the legacy loop re-read it (a by-value n is hoisted: other unrolling, more REG; strict).

// y(r) = c(r) for r < n (runtime n; the fragment is then indexed at run time).
template <class T, int N, class L, class Y> BATCHLAS_DLA_INLINE void store_n(const Frag<T, N, L>& c, const Y& y, const int& n) {
    for (int r = 0; r < n; ++r) y(r) = c[r];
}
// c(r) = d(r) * z(r) for r < n (z real, lifted): the dgmm Left lift on a register column.
template <class Ar = void, class T, int N, class L, class Dv, class Zv>
BATCHLAS_DLA_INLINE void dgmm_lift(Frag<T, N, L>& c, const Dv& d, const Zv& z, const int& n) {
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < N; ++r) {
        if (r < n) c[r] = AR::mul(d(r), detail::lift<AR, T>(z(r)));
    }
}

// larf, Side::Left, on a register column (ColPerLane): c := (I - tau v v^H) c, with v read at
// every r < N (zero outside its support: unpredicated) from an operand all lanes may read.
template <class Ar = void, class T, int N, class L, class V> BATCHLAS_DLA_INLINE void larf(Frag<T, N, L>& c, const Published<V>& v, T tau) {
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    T dot = AR::zero();
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < N; ++r) AR::mac_conj(dot, v(r), c[r]);
    const T gamma = AR::mul(tau, dot);
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < N; ++r) detail::msub<AR>(c[r], v(r), gamma);
}
// c := H e_k = e_k - tau v for the QL vector v (v(k) = 1, v(r > k) = 0): column k of xORG2L.
template <class Ar = void, class T, int N, class L, class V>
BATCHLAS_DLA_INLINE void larf_unit(Frag<T, N, L>& c, const Published<V>& v, T tau, int k) {
    using AR = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < N; ++r) c[r] = (r < k) ? AR::mul(AR::neg(tau), v(r)) : ((r == k) ? AR::sub(AR::one(), tau) : AR::zero());
}

// n - 1 QL reflectors: k sits in column k + 1 of `refl` (rows < k, implied 1 at row k), tau(k)
// in lane k's tau_lane; lane l owns column l. vs: P-long scratch the chunk shares. Two chunk
// barriers per reflector (after staging, after applying); the caller owns those around the call.

// stage reflector k into vs, zero-padded to the whole frame: vs(lane) = unit_head_ql(...)(lane).
template <class Ex, class Vs, class Rm> BATCHLAS_DLA_INLINE void stage_ql_reflector(const Ex& ex, const Vs& vs, const Rm& refl, int k) {
    const int lane = ex.lid();
    vs(lane) = unit_head_ql(refl.col(k + 1), k)(lane);
}

// xORG2L, column-owner form (no column shift): lane l < n - 1 ends with column l of
// Q = H(n-2) ... H(0) (columns >= n - 1 are the caller's). Column k is generated at step k,
// so the reflector store may overlap the columns of Q not yet generated.
template <class Ar = void, class Ex, class T, int N, class L, class Vs, class Rm>
BATCHLAS_DLA_INLINE void org2l(const Ex& ex, Frag<T, N, L>& q, const Vs& vs, const Rm& refl, T tau_lane, const int& n) {
    const int lane = ex.lid();
    fill(q, Arith<T>::zero());
    for (int k = 0; k < n - 1; ++k) {
        const T tau_k = detail::bcast(ex, tau_lane, k);
        stage_ql_reflector(ex, vs, refl, k);
        detail::chunk_barrier(ex);
        if (lane < k) larf<Ar>(q, assume_published(vs), tau_k);
        else if (lane == k) larf_unit<Ar>(q, assume_published(vs), tau_k, k);
        detail::chunk_barrier(ex);  // every lane is done reading vs before it is rebuilt
    }
}

// xORM2L, Side::Left, NoTrans, column-owner form: c := H(n-2) ... H(0) c on the lanes < n
// (the n x n block's columns); a reflector with tau == 0 is skipped.
template <class Ar = void, class Ex, class T, int N, class L, class Vs, class Rm>
BATCHLAS_DLA_INLINE void orm2l(const Ex& ex, Frag<T, N, L>& c, const Vs& vs, const Rm& refl, T tau_lane, const int& n) {
    const int lane = ex.lid();
    for (int k = 0; k < n - 1; ++k) {
        const T tau_k = detail::bcast(ex, tau_lane, k);
        stage_ql_reflector(ex, vs, refl, k);
        detail::chunk_barrier(ex);
        if (lane < n && tau_k != Arith<T>::zero()) larf<Ar>(c, assume_published(vs), tau_k);
        detail::chunk_barrier(ex);
    }
}

}  // namespace batchlas::device::v2

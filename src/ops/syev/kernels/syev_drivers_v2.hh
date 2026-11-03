// SPDX-License-Identifier: MIT
#pragma once
// syev-driver-private device LA v2 ops (similarities, Hermitian load, glue repack, column-owner
// larf / org2l / orm2l); shared ones are core's. Strict ports: arith::Native by default, the
// hand code's expression shapes; Uplo stays runtime where the legacy kernels branch on it.

#include <cstdint>
#include <type_traits>

#include <batchlas/blas/device/v2/dla.hh>
#include <src/kernels/common/dla_v2_partition.hh>

namespace batchlas::syev_v2 {

namespace v2 = batchlas::device::v2;

// ---- executors and flat 2-D glue ---------------------------------------------------------------
// One element (r, c) as an executor (Grid2D's row()/col() contract): core 2-D ops visit only it.
struct ElemAt {
    int r, c;
    BATCHLAS_DLA_INLINE int row() const { return r; }
    BATCHLAS_DLA_INLINE int col() const { return c; }
};

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

// SubGroupPartition's own chunk barrier (core Partition::barrier is a whole-sub-group barrier).
template <class Ex> BATCHLAS_DLA_INLINE void chunk_barrier(const Ex& ex) {
    if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) sycl::group_barrier(ex.part);
    else group_barrier(ex.part);  // ADL: SubGroupPartition
}

namespace detail {
// A real value as the scalar type T of the output (complex: imaginary part 0).
template <class Ar, class T, class R> BATCHLAS_DLA_INLINE T lift(const R& x) {
    if constexpr (Ar::is_complex && !std::is_same_v<std::remove_cvref_t<R>, T>) return T(x, typename Ar::real_t(0));
    else return x;
}
}  // namespace detail

// ---- element kernels -----------------------------------------------------------------------------
// v := element (i, j) of the Hermitian matrix whose `upper` (else lower) triangle A stores,
// Im(diag) dropped; assigned per branch so a caller's guard folds into one CFG.
template <class Ops = v2::arith::Native, class T, class M> BATCHLAS_DLA_INLINE void herm_load(T& v, const M& A, bool upper, int i, int j) {
    using AR = v2::arith_t<Ops, T>;
    if (i == j) {
        v = AR::make(AR::re(A(i, j)), v2::real_t<T>(0));
    } else if (upper) {
        v = (i < j) ? A(i, j) : AR::conj(A(j, i));
    } else {
        v = (i > j) ? A(i, j) : AR::conj(A(j, i));
    }
}

// C(i, j) := d(i) * B(i, j); a real B is lifted to C's scalar type. C may alias B.
template <class Ops = v2::arith::Native, class Dv, class Bm, class Cm>
BATCHLAS_DLA_INLINE void dgmm_left_at(const Dv& d, const Bm& B, const Cm& C, int i, int j) {
    using T = std::remove_cvref_t<decltype(C(0, 0))>;
    using AR = v2::arith_t<Ops, T>;
    C(i, j) = AR::mul(d(i), detail::lift<AR, T>(B(i, j)));
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

// ---- tridiagonal similarities ------------------------------------------------------------------
// Hermitian tridiagonal (d, e) -> real symmetric (dr, er = |e|) by T' = S^H T S:
// S(0) = 1, S(i+1) = S(i) * f(e_i) / |e_i| (S(i) when e_i == 0), f = identity when e is the
// subdiagonal (Uplo::Lower) and conj when it is the superdiagonal. want_s == false writes
// dr and er only. Every step reads S(i) back from s.
template <class Ops = v2::arith::Native, class Dv, class Ev, class Drv, class Erv, class Sv>
BATCHLAS_DLA_INLINE void phase_similarity(const v2::Thread&, int n, const Dv& d, const Ev& e, const Drv& dr, const Erv& er, const Sv& s,
                                          Uplo uplo, bool want_s) {
    using T = std::remove_cvref_t<decltype(e(0))>;
    using AR = v2::arith_t<Ops, T>;
    using R = v2::real_t<T>;
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
            const T f = AR::div_real(sub, abs_e);  // before s(i) is read, as `S(i) * (sub / |e|)` does
            s(i + 1) = AR::mul(s(i), f);
        }
    }
}

// Real symmetric tridiagonal: e := |e| by the +-1 similarity S (S(0) = 1, S(i+1) = S(i) sign(e_i)).
template <class Ev, class Sv> BATCHLAS_DLA_INLINE void sign_similarity(const v2::Thread&, int n, const Ev& e, const Sv& s) {
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
template <class Ops = v2::arith::Native, class Ex, class T, class R> BATCHLAS_DLA_INLINE T phase_replay(const Ex& ex, T e_lane, R abs_lane, int n) {
    using AR = v2::arith_t<Ops, T>;
    const int lane = ex.lid();
    T phase = AR::one();
    for (int i = 0; i < n - 1; ++i) {
        const T e_i = ex.bcast(e_lane, static_cast<std::uint32_t>(i));
        const R abs_i = ex.bcast(abs_lane, static_cast<std::uint32_t>(i));
        if (lane > i && abs_i != R(0)) phase = AR::mul(phase, AR::div_real(AR::conj(e_i), abs_i));
    }
    return phase;
}

// ---- column-owner Householder (lane l holds column l in a ColPerLane Frag) -------------------
// A QL Householder vector of a P-long frame: element `head` is an implied 1, elements below it
// are 0, elements above it are stored (LAPACK xORG2L / xORM2L convention; core UnitHead is QR's).
template <class V> struct UnitHeadQL {
    V v;
    int head;
    using value_type = v2::elem_t<V>;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        using A = v2::Arith<value_type>;
        value_type x = A::zero();
        if (i < head) x = v(i);
        else if (i == head) x = A::one();
        return x;
    }
};
template <class V> BATCHLAS_DLA_INLINE UnitHeadQL<V> unit_head_ql(V v, int head) { return {v, head}; }

// c(r) = beta at r == j, alpha elsewhere (xLASET on one register column).
template <class T, int N, class L> BATCHLAS_DLA_INLINE void laset_col(v2::Frag<T, N, L>& c, int j, T alpha, T beta) {
#pragma unroll
    for (int r = 0; r < N; ++r) c[r] = (r == j) ? beta : alpha;
}

// c(r) = d(r) * z(r) for r < n (z real, lifted): the dgmm Left lift on a register column.
template <class Ops = v2::arith::Native, class T, int N, class L, class Dv, class Zv>
BATCHLAS_DLA_INLINE void dgmm_lift(v2::Frag<T, N, L>& c, const Dv& d, const Zv& z, const int& n) {
    using AR = v2::arith_t<Ops, T>;
#pragma unroll
    for (int r = 0; r < N; ++r) {
        if (r < n) c[r] = AR::mul(d(r), detail::lift<AR, T>(z(r)));
    }
}

// larf, Side::Left, on a register column: c := (I - tau v v^H) c, with v read at every r < N
// (zero outside its support: unpredicated) from an operand all lanes may read.
template <class Ops = v2::arith::Native, class T, int N, class L, class V>
BATCHLAS_DLA_INLINE void larf_col(v2::Frag<T, N, L>& c, const v2::Published<V>& v, T tau) {
    using AR = v2::arith_t<Ops, T>;
    T dot = AR::zero();
#pragma unroll
    for (int r = 0; r < N; ++r) AR::mac_conj(dot, v(r), c[r]);
    const T gamma = AR::mul(tau, dot);
#pragma unroll
    for (int r = 0; r < N; ++r) AR::msub(c[r], v(r), gamma);
}
// c := H e_k = e_k - tau v for the QL vector v (v(k) = 1, v(r > k) = 0): column k of xORG2L.
template <class Ops = v2::arith::Native, class T, int N, class L, class V>
BATCHLAS_DLA_INLINE void larf_unit_col(v2::Frag<T, N, L>& c, const v2::Published<V>& v, T tau, int k) {
    using AR = v2::arith_t<Ops, T>;
#pragma unroll
    for (int r = 0; r < N; ++r) c[r] = (r < k) ? AR::mul(AR::neg(tau), v(r)) : ((r == k) ? AR::sub(AR::one(), tau) : AR::zero());
}

// n - 1 QL reflectors: k in column k + 1 of `refl` (rows < k, unit at k), tau(k) on lane k;
// lane l owns column l; vs is the chunk's P-long scratch. Stage reflector k, zero-padded:
template <class Ex, class Vs, class Rm> BATCHLAS_DLA_INLINE void stage_ql_reflector(const Ex& ex, const Vs& vs, const Rm& refl, int k) {
    const int lane = ex.lid();
    vs(lane) = unit_head_ql(refl.col(k + 1), k)(lane);
}

// xORG2L, column-owner form (no column shift): lane l < n - 1 ends with column l of
// Q = H(n-2) ... H(0) (columns >= n - 1 are the caller's). Column k is generated at step k,
// so the reflector store may overlap the columns of Q not yet generated.
template <class Ops = v2::arith::Native, class Ex, class T, int N, class L, class Vs, class Rm>
BATCHLAS_DLA_INLINE void org2l(const Ex& ex, v2::Frag<T, N, L>& q, const Vs& vs, const Rm& refl, T tau_lane, const int& n) {
    const int lane = ex.lid();
    v2::fill(q, v2::arith_t<Ops, T>::zero());
    for (int k = 0; k < n - 1; ++k) {
        const T tau_k = ex.bcast(tau_lane, static_cast<std::uint32_t>(k));
        stage_ql_reflector(ex, vs, refl, k);
        chunk_barrier(ex);
        if (lane < k) larf_col<Ops>(q, v2::assume_published(vs), tau_k);
        else if (lane == k) larf_unit_col<Ops>(q, v2::assume_published(vs), tau_k, k);
        chunk_barrier(ex);  // every lane is done reading vs before it is rebuilt
    }
}

// xORM2L, Side::Left, NoTrans, column-owner form: c := H(n-2) ... H(0) c on the lanes < n
// (the n x n block's columns); a reflector with tau == 0 is skipped.
template <class Ops = v2::arith::Native, class Ex, class T, int N, class L, class Vs, class Rm>
BATCHLAS_DLA_INLINE void orm2l(const Ex& ex, v2::Frag<T, N, L>& c, const Vs& vs, const Rm& refl, T tau_lane, const int& n) {
    const int lane = ex.lid();
    for (int k = 0; k < n - 1; ++k) {
        const T tau_k = ex.bcast(tau_lane, static_cast<std::uint32_t>(k));
        stage_ql_reflector(ex, vs, refl, k);
        chunk_barrier(ex);
        if (lane < n && tau_k != v2::arith_t<Ops, T>::zero()) larf_col<Ops>(c, v2::assume_published(vs), tau_k);
        chunk_barrier(ex);
    }
}

}  // namespace batchlas::syev_v2

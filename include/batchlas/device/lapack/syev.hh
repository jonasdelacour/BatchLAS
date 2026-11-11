// SPDX-License-Identifier: MIT
#pragma once
// xSYEV / xHEEV in a kernel: eigenvalues W (ascending) and, for JobType::EigenVectors, the
// eigenvectors over A.
//   syev<J, UL, tier::Cta<N>>       n <= N, one problem per SgpPartition<N> from load to store
//                                   (sytd2 Upper, steqr, org2l / orm2l): the host syev_cta_fused
//                                   kernel is its thin launcher, ws.local only.
//   syev<J, UL, tier::Blocked<NB>>  any n, one problem per WorkGroup<256>: sytrd<Lower, Blocked<NB>>,
//                                   steqr<J, Blocked<1>> and ormqr<Left, NoTrans, Blocked> on the
//                                   caller's global slice (Upper input is mirrored first).
// syev_detail also holds the steps the host syev drivers share (syev_drivers_v2.hh names them).
// evidence: docs/perf/syev.md#syev-the-fused-cta-kernel-design

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <batchlas/blas/extensions.hh>  // SteqrShiftStrategy, SteqrUpdateScheme
#include <batchlas/device.hh>
#include <batchlas/device/lapack/ormqr.hh>
#include <batchlas/device/lapack/steqr.hh>
#include <batchlas/device/lapack/sytrd.hh>
#include <batchlas/device/sg_partition.hh>

namespace batchlas::device {

namespace syev_detail {

// SubGroupPartition's own chunk barrier (core Partition::barrier is a whole-sub-group barrier).
template <class Ex> BATCHLAS_DLA_INLINE void chunk_barrier(const Ex& ex) {
    if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) sycl::group_barrier(ex.part);
    else group_barrier(ex.part);  // ADL: SubGroupPartition
}

template <class U> BATCHLAS_DLA_INLINE real_t<U> real_part(const U& x) {
    if constexpr (Arith<U>::is_complex) return x.real();
    else return x;
}

// v := element (i, j) of the Hermitian matrix whose `upper` (else lower) triangle A stores,
// Im(diag) dropped; assigned per branch so a caller's guard folds into one CFG.
template <class Ops = arith::Native, class T, class M> BATCHLAS_DLA_INLINE void herm_load(T& v, const M& A, bool upper, int i, int j) {
    using AR = arith_t<Ops, T>;
    if (i == j) {
        v = AR::make(AR::re(A(i, j)), real_t<T>(0));
    } else if (upper) {
        v = (i < j) ? A(i, j) : AR::conj(A(j, i));
    } else {
        v = (i > j) ? A(i, j) : AR::conj(A(j, i));
    }
}

// Hermitian tridiagonal (d, e) -> real symmetric (dr, er = |e|) by T' = S^H T S:
// S(0) = 1, S(i+1) = S(i) * f(e_i) / |e_i| (S(i) when e_i == 0), f = identity when e is the
// subdiagonal (Uplo::Lower) and conj when it is the superdiagonal. want_s == false writes
// dr and er only. Every step reads S(i) back from s.
template <class Ops = arith::Native, class Dv, class Ev, class Drv, class Erv, class Sv>
BATCHLAS_DLA_INLINE void phase_similarity(const Thread&, int n, const Dv& d, const Ev& e, const Drv& dr, const Erv& er, const Sv& s,
                                          Uplo uplo, bool want_s) {
    using T = std::remove_cvref_t<decltype(e(0))>;
    using AR = arith_t<Ops, T>;
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
            const T f = AR::div_real(sub, abs_e);  // before s(i) is read, as `S(i) * (sub / |e|)` does
            s(i + 1) = AR::mul(s(i), f);
        }
    }
}

// The same phase chain replayed per lane on a Partition (no scratch): lane l returns S(l) of
// the superdiagonal form, S(i+1) = S(i) * conj(e_i) / |e_i|, from e_lane / abs_lane held by
// lane i. 2 (n - 1) broadcasts; the lane-local product order is the serial one.
template <class Ops = arith::Native, class Ex, class T, class R> BATCHLAS_DLA_INLINE T phase_replay(const Ex& ex, T e_lane, R abs_lane, int n) {
    using AR = arith_t<Ops, T>;
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

// c(r) = beta at r == j, alpha elsewhere (xLASET on one register column).
template <class T, int N, class L> BATCHLAS_DLA_INLINE void laset_col(Frag<T, N, L>& c, int j, T alpha, T beta) {
#pragma unroll
    for (int r = 0; r < N; ++r) c[r] = (r == j) ? beta : alpha;
}

// c(r) = d(r) * z(r) for r < n (z real, core lift): the dgmm Left lift on a register column.
template <class Ops = arith::Native, class T, int N, class L, class Dv, class Zv>
BATCHLAS_DLA_INLINE void dgmm_lift(Frag<T, N, L>& c, const Dv& d, const Zv& z, const int& n) {
    using AR = arith_t<Ops, T>;
#pragma unroll
    for (int r = 0; r < N; ++r) {
        if (r < n) c[r] = AR::mul(d(r), lift<T>(z(r)));
    }
}

// larf, Side::Left, on a register column: c := (I - tau v v^H) c, with v read at every r < N
// (zero outside its support: unpredicated) from an operand all lanes may read.
template <class Ops = arith::Native, class T, int N, class L, class V>
BATCHLAS_DLA_INLINE void larf_col(Frag<T, N, L>& c, const Published<V>& v, T tau) {
    using AR = arith_t<Ops, T>;
    T dot = AR::zero();
#pragma unroll
    for (int r = 0; r < N; ++r) AR::mac_conj(dot, v(r), c[r]);
    const T gamma = AR::mul(tau, dot);
#pragma unroll
    for (int r = 0; r < N; ++r) AR::msub(c[r], v(r), gamma);
}
// c := H e_k = e_k - tau v for the QL vector v (v(k) = 1, v(r > k) = 0): column k of xORG2L.
template <class Ops = arith::Native, class T, int N, class L, class V>
BATCHLAS_DLA_INLINE void larf_unit_col(Frag<T, N, L>& c, const Published<V>& v, T tau, int k) {
    using AR = arith_t<Ops, T>;
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
template <class Ops = arith::Native, class Ex, class T, int N, class L, class Vs, class Rm>
BATCHLAS_DLA_INLINE void org2l(const Ex& ex, Frag<T, N, L>& q, const Vs& vs, const Rm& refl, T tau_lane, const int& n) {
    const int lane = ex.lid();
    fill(q, arith_t<Ops, T>::zero());
    for (int k = 0; k < n - 1; ++k) {
        const T tau_k = ex.bcast(tau_lane, static_cast<std::uint32_t>(k));
        stage_ql_reflector(ex, vs, refl, k);
        chunk_barrier(ex);
        if (lane < k) larf_col<Ops>(q, assume_published(vs), tau_k);
        else if (lane == k) larf_unit_col<Ops>(q, assume_published(vs), tau_k, k);
        chunk_barrier(ex);  // every lane is done reading vs before it is rebuilt
    }
}

// xORM2L, Side::Left, NoTrans, column-owner form: c := H(n-2) ... H(0) c on the lanes < n
// (the n x n block's columns); a reflector with tau == 0 is skipped.
template <class Ops = arith::Native, class Ex, class T, int N, class L, class Vs, class Rm>
BATCHLAS_DLA_INLINE void orm2l(const Ex& ex, Frag<T, N, L>& c, const Vs& vs, const Rm& refl, T tau_lane, const int& n) {
    const int lane = ex.lid();
    for (int k = 0; k < n - 1; ++k) {
        const T tau_k = ex.bcast(tau_lane, static_cast<std::uint32_t>(k));
        stage_ql_reflector(ex, vs, refl, k);
        chunk_barrier(ex);
        if (lane < n && tau_k != arith_t<Ops, T>::zero()) larf_col<Ops>(c, assume_published(vs), tau_k);
        chunk_barrier(ex);
    }
}

// ---- tier::Cta<N>: one problem per partition, load to store -------------------------------------
// The tile layout: real eigenvector input reduces, accumulates and reads out on ONE tile, so it
// carries the accumulator's P + 1 padding (read by column on write-out: LD == 32 serializes 32
// ways on bank conflicts); Hermitian input with vectors adds a separate real accumulator.
template <bool V, int P, class T> struct cta_layout {
    static constexpr bool fused_orgql = V && !Arith<T>::is_complex;
    static constexpr bool separate_q = V && !fused_orgql;
    static constexpr int ldq = P + 1;
    static constexpr int lda = fused_orgql ? ldq : P;
};
// The Cta tier's workspace: the tile (lda x N), v and w (N each), the real accumulator (ldq x N
// for Hermitian input with vectors), all local.
template <bool V, int N, class T> struct cta_plan {
    using L = cta_layout<V, N, T>;
    Plan plan{};
    ws::Slot<T> a{}, v{}, w{};
    ws::Slot<real_t<T>> q{};
    constexpr cta_plan()
        : a(plan.take<T>(ws::Space::Local, static_cast<std::size_t>(L::lda) * N)), v(plan.take<T>(ws::Space::Local, N)),
          w(plan.take<T>(ws::Space::Local, N)),
          q(plan.take<real_t<T>>(ws::Space::Local, L::separate_q ? static_cast<std::size_t>(L::ldq) * N : 0)) {}
};

struct cta_opts {  // the host SteqrParams fields of the same meaning
    std::int32_t max_sweeps;
    SteqrShiftStrategy shift;
    SteqrUpdateScheme update;
    bool sort;
    bool ascending;
};

// The Cta tier's one body (the host syev_cta_fused kernel's and device::syev<J, UL, Cta<P>>'s):
// stage 0 loads the Hermitian completion of A_in's `is_upper` triangle into the tile at
// A_local[base_a]; stage 1 sytd2 Upper; stage 2 the steqr sweeps on solve_part (real vectors
// seeded with Q_house in place, DSYEV's route; Hermitian vectors on the separate accumulator at
// Q_local[base_q]); stage 3 the Hermitian back-transform. W(rank) := eigenvalue (W a Real* or a
// vector operand), column rank of A_out := its vector; on_solved(failed) runs between the solve
// and the stores. live == false (a dead chunk or a masked item) runs every stage on a zero
// matrix and stores nothing. Buf / QBuf: a sycl::local_accessor or a decorated local pointer.
template <class T, std::size_t P, bool V, std::int32_t LDA, std::int32_t LDQ, bool M, class SP, class Buf, class QBuf, class MA,
          class MO, class WV, class F>
BATCHLAS_DLA_INLINE void syev_cta_run(const SubGroupPartition<P, M>& part, const SP& solve_part, const Buf& A_local, std::int32_t base_a,
                                      const Buf& V_local, std::int32_t base_v, const Buf& W_local, std::int32_t base_w,
                                      const QBuf& Q_local, std::int32_t base_q, const MA& A_in, const MO& A_out, const WV& W,
                                      std::int32_t nn, bool is_upper, bool live, const cta_opts& o, F&& on_solved) {
    using Real = real_t<T>;
    constexpr bool kComplex = Arith<T>::is_complex;
    constexpr bool kFusedOrgql = cta_layout<V, static_cast<int>(P), T>::fused_orgql;
    constexpr int kP = static_cast<int>(P);
    const std::int32_t lane = static_cast<std::int32_t>(part.get_local_linear_id());
    using Ops = arith::Native;  // the std::complex operators (strict port)
    const auto pex = wrap(part);
    using ATile = Mat<T, Ext<kP>, Ext<kP>, Ext<LDA>, Mem::Local, Cyclic, Buf>;
    const auto store_w = [&](std::int32_t dst, Real x) BATCHLAS_DLA_BODY {
        if constexpr (std::is_pointer_v<WV>) W[dst] = x;
        else W(dst) = x;
    };

    // ---- Stage 0: load the Hermitian completion of the stored triangle. ----
    for (std::int32_t c = 0; c < kP; ++c) {
        T v = T(0);
        if (live && lane < nn && c < nn) herm_load<Ops>(v, A_in, is_upper, lane, c);
        A_local[base_a + lane + c * LDA] = v;
    }
    group_barrier(part);
    const ATile tile{A_local, {}, {}, {}, base_a};
    [[maybe_unused]] const Vec<T, Dyn, Unit, Mem::Local, Cyclic, Buf> vs{V_local, Dyn{kP}, Unit{}, base_v};

    // ---- Stage 1: tridiagonal reduction (SYTD2, upper path). ----
    const T tau_lane = sytrd_detail::sytd2_upper_tile<T, LDA>(part, sytrd_detail::decorated_ptr(A_local), base_a,
                                                              sytrd_detail::decorated_ptr(V_local), base_v,
                                                              sytrd_detail::decorated_ptr(W_local), base_w, nn);

    // The tridiagonal off the tile; above the superdiagonal is the packed reflector store.
    const T d_c = (lane < nn) ? A_local[base_a + lane + lane * LDA] : T(0);
    const T e_c = (lane < (nn - 1)) ? A_local[base_a + lane + (lane + 1) * LDA] : T(0);

    Real diag = real_part(d_c);
    Real offdiag = Real(0);
    T phase = T(1);

    if constexpr (kComplex) {
        // Hermitian -> real tridiagonal (T' = S^H T S); each lane replays S itself.
        offdiag = (lane < (nn - 1)) ? sycl::hypot(e_c.real(), e_c.imag()) : Real(0);
        phase = phase_replay<Ops>(pex, e_c, offdiag, nn);
    } else {
        offdiag = real_part(e_c);
    }

    // ---- Ordering: each lane writes its eigenvalue to its rank slot. ----
    const auto slot_of = [&](Real wj) BATCHLAS_DLA_BODY {
        if (!o.sort) return lane;
        std::int32_t rank = 0;
        for (std::int32_t k = 0; k < nn; ++k) {
            const Real wk = select_from_group(part, wj, static_cast<std::uint32_t>(k));
            // Strict total order (NaN last, ties on index): a bijection even for NaN.
            const bool nan_j = sycl::isnan(wj), nan_k = sycl::isnan(wk);
            const bool before = (nan_j != nan_k) ? nan_j : (nan_j || wk == wj) ? (k < lane) : (o.ascending ? wk < wj : wk > wj);
            if (before) ++rank;
        }
        return rank;
    };

    if constexpr (!V) {
        // ---- Stage 2: eigenvalues only, no accumulator. ----
        steqr_detail::QSharedCache<Real, P, LDQ, false, QBuf> qcache(Q_local, base_q, lane, nn);
        const bool failed = steqr_detail::steqr_cta_solve<Real, P>(solve_part, diag, offdiag, qcache, nn, o.max_sweeps, o.shift, o.update);
        on_solved(failed);

        const std::int32_t dst = slot_of(diag);
        if (live && lane < nn) store_w(dst, diag);
    } else if constexpr (kFusedOrgql) {
        // ---- Stage 2a: Q_house in place (DSYEV: DORGTR seeds DSTEQR, no back-transform).
        // org2l generates column k at step k, so the reflectors it has not reached stay intact.
        Frag<Real, kP, ColPerLane> Qc;
        org2l<Ops>(pex, Qc, vs, tile, tau_lane, nn);

        // Identity outside the leading (n-1)^2 block; pad columns defined (sweeps run on all P lanes).
        if (lane >= nn) fill(Qc, Real(0));
        else if (lane == nn - 1) laset_col(Qc, nn - 1, Real(0), Real(1));

        // Hand the tile over: from here it is the accumulator.
        group_barrier(part);
        store(Qc, tile.col(lane));
        group_barrier(part);

        // ---- Stage 2b: sweeps, accumulating onto Q_house. ----
        steqr_detail::QSharedCache<Real, P, LDQ, true, Buf> qcache(A_local, base_a, lane, nn);
        const bool failed = steqr_detail::steqr_cta_solve<Real, P>(solve_part, diag, offdiag, qcache, nn, o.max_sweeps, o.shift, o.update);
        on_solved(failed);

        // The sweeps wrote the tile by row (lane == row); the readout reads it by column.
        group_barrier(part);
        const std::int32_t dst = slot_of(diag);
        if (live && lane < nn) {
            store_w(dst, diag);
            copy(Thread{}, tile.col(lane), A_out.col(dst));
        }
    } else {
        // ---- Stage 2: sweeps on a separate real accumulator (Hermitian input; the reflectors
        // are applied afterwards, in stage 3). ----
        steqr_detail::QSharedCache<Real, P, LDQ, true, QBuf> qcache(Q_local, base_q, lane, nn);
        const Mat<Real, Ext<kP>, Ext<kP>, Ext<LDQ>, Mem::Local, Cyclic, QBuf> qtile{Q_local, {}, {}, {}, base_q};

        // The identity on the leading n x n block, zero elsewhere; this lane's row.
        const Real q_diag = lane < nn ? Real(1) : Real(0);
        for (std::int32_t c = 0; c < kP; ++c) {
            laset(ElemAt{lane, c}, Zero{}, q_diag, qtile);
        }
        group_barrier(part);

        const bool failed = steqr_detail::steqr_cta_solve<Real, P>(solve_part, diag, offdiag, qcache, nn, o.max_sweeps, o.shift, o.update);
        on_solved(failed);

        const std::int32_t dst = slot_of(diag);
        if (live && lane < nn) store_w(dst, diag);

        // ---- Stage 3: back-transform, Z := Q_house * S * Z. ----
        Frag<T, kP, ColPerLane> C_col;
        fill(C_col, T(0));

        // Zc = diag(S) Z (dgmm Left), staged before vs is reused for the reflectors.
        V_local[base_v + lane] = phase;
        group_barrier(part);
        if (lane < nn) dgmm_lift<Ops>(C_col, vs, qtile.col(lane), nn);
        group_barrier(part);

        // Upper-path SYTD2 reflectors form a QL factorization: xORM2L, Left, NoTrans.
        orm2l<Ops>(pex, C_col, vs, tile, tau_lane, nn);

        if (live && lane < nn) store(C_col, A_out.col(dst), nn);
    }
}

// ---- tier::Blocked<NB>: one problem per work-group, A in global memory --------------------------
template <class Tier> inline constexpr bool is_blocked_v = false;
template <int NB, class A> inline constexpr bool is_blocked_v<tier::Blocked<NB, A>> = true;

template <class T> constexpr int ormqr_nb = larfb_default_nb<T>();
template <class T> using ormqr_tier = tier::Blocked<ormqr_nb<T>>;

// The Blocked tier's workspace: d, e, tau (A's scalar type), the real tridiagonal and the phase
// chain S (complex), Z (n x n, vectors), the rank of each eigenvalue, then ONE slice the stages
// take in turn (sytrd, steqr, ormqr: max of their global and of their local bytes).
template <bool V, int NB, class T> struct blocked_plan {
    using R = real_t<T>;
    static constexpr bool kCplx = Arith<T>::is_complex;
    Plan plan{};
    ws::Slot<T> d{}, e{}, tau{}, s{}, z{};
    ws::Slot<R> dr{}, er{};
    ws::Slot<int> rank{};
    ws::Slot<std::byte> sub_g{}, sub_l{};
    static constexpr std::size_t mx(std::size_t a, std::size_t b) { return a > b ? a : b; }
    constexpr explicit blocked_plan(int n) {
        const std::size_t un = static_cast<std::size_t>(n), um = n > 1 ? un - 1 : 1;
        d = plan.take<T>(ws::Space::Global, un);
        e = plan.take<T>(ws::Space::Global, um);
        tau = plan.take<T>(ws::Space::Global, um);
        dr = plan.take<R>(ws::Space::Global, kCplx ? un : 0);
        er = plan.take<R>(ws::Space::Global, kCplx ? um : 0);
        s = plan.take<T>(ws::Space::Global, kCplx && V ? un : 0);
        z = plan.take<T>(ws::Space::Global, V ? un * un : 0);
        rank = plan.take<int>(ws::Space::Global, un);
        constexpr JobType J = V ? JobType::EigenVectors : JobType::NoEigenVectors;
        const WorkspaceSize s1 = sytrd_buffer_size<Uplo::Lower, tier::Blocked<NB>, T>(n);
        const WorkspaceSize s2 = steqr_buffer_size<J, tier::Blocked<1>, R>(n);
        const WorkspaceSize s3 = V && n > 1 ? ormqr_buffer_size<Side::Left, Transpose::NoTrans, ormqr_tier<T>, T>(n - 1, n) : WorkspaceSize{};
        sub_g = plan.take<std::byte>(ws::Space::Global, mx(s1.global_bytes, mx(s2.global_bytes, s3.global_bytes)));
        sub_l = plan.take<std::byte>(ws::Space::Local, mx(s1.local_bytes, mx(s2.local_bytes, s3.local_bytes)));
    }
};

// xSYEV on one n x n matrix per work-group. Upper input is mirrored into the lower triangle
// (conj), then sytrd<Lower, Blocked<NB>>; a Hermitian tridiagonal becomes real by the phase
// similarity S (item 0); steqr<J, Blocked<1>> on Z = I; Z := S Z; Z(1:, :) := Q Z by ormqr on
// sytrd's reflectors A(1:, 0:n-1). The eigenvalues leave ascending (NaN last, ties by index) in W,
// column j of Z in column rank(j) of A. Returns steqr's info (uniform).
template <bool V, Uplo UL, int NB, int WG, class T, class WV>
BATCHLAS_DLA_INLINE int syev_blocked_run(const WorkGroup<WG>& ex, const l3::Global<T>& a, const WV& W, const Workspace& ws,
                                         const steqr_opts& opts) {
    using R = real_t<T>;
    using Ar = Arith<T>;
    constexpr JobType J = V ? JobType::EigenVectors : JobType::NoEigenVectors;
    const int n = a.m, nm1 = n - 1, lda = a.ld;
    const blocked_plan<V, NB, T> plan(n);
    T* const dp = carve_global(ws, plan.d);
    T* const ep = carve_global(ws, plan.e);
    T* const tp = carve_global(ws, plan.tau);
    T* const zp = carve_global(ws, plan.z);
    int* const rk = carve_global(ws, plan.rank);
    const Workspace sub{carve_local(ws, plan.sub_l), carve_global(ws, plan.sub_g)};

    if constexpr (UL == Uplo::Upper) {
        for (int j = 0; j < n; ++j)
            for_each_owned(ex, n, [&](int i) BATCHLAS_DLA_BODY {
                if (i > j) a(i, j) = ::batchlas::device::conj(a(j, i));
            });
        ex.barrier(Mem::Global);
    }
    (void)sytrd<Uplo::Lower, tier::Blocked<NB>>(ex, mat(a.p, n, n, lda), vec(dp, n), vec(ep, nm1), vec(tp, nm1), sub);

    R* drp;
    R* erp;
    if constexpr (Ar::is_complex) {
        drp = carve_global(ws, plan.dr);
        erp = carve_global(ws, plan.er);
        if (ex.leader())
            phase_similarity(Thread{}, n, vec(dp, n), vec(ep, nm1), vec(drp, n), vec(erp, nm1), vec(carve_global(ws, plan.s), n),
                             Uplo::Lower, V);
        ex.barrier(Mem::Global);
    } else {
        drp = dp;
        erp = ep;
    }
    const auto z = mat(zp, n, n, n);
    if constexpr (V) {
        for (int j = 0; j < n; ++j)
            for_each_owned(ex, n, [&](int i) BATCHLAS_DLA_BODY { z(i, j) = i == j ? Ar::one() : Ar::zero(); });
        ex.barrier(Mem::Global);
    }
    const int info = steqr<J, tier::Blocked<1>>(ex, vec(drp, n), vec(erp, nm1), z, sub, opts);
    ex.barrier(Mem::Global);
    if constexpr (V) {
        if constexpr (Ar::is_complex) {
            const T* const sp = carve_global(ws, plan.s);
            for (int j = 0; j < n; ++j)
                for_each_owned(ex, n, [&](int i) BATCHLAS_DLA_BODY { z(i, j) = sp[i] * z(i, j); });
            ex.barrier(Mem::Global);
        }
        if (n > 1)
            (void)ormqr<Side::Left, Transpose::NoTrans, ormqr_tier<T>>(ex, mat(a.p + 1, nm1, nm1, lda), mat(zp + 1, nm1, n, n), tp, sub);
        ex.barrier(Mem::Global);
    }
    for_each_owned(ex, n, [&](int i) BATCHLAS_DLA_BODY {
        const R wi = drp[i];
        const bool nan_i = sycl::isnan(wi);
        int r = 0;
        for (int k = 0; k < n; ++k) {
            const R wk = drp[k];
            const bool nan_k = sycl::isnan(wk);
            r += ((nan_i != nan_k) ? nan_i : (nan_i || wk == wi) ? (k < i) : (wk < wi)) ? 1 : 0;
        }
        rk[i] = r;
    });
    ex.barrier(Mem::Global);
    for_each_owned(ex, n, [&](int i) BATCHLAS_DLA_BODY { W(rk[i]) = drp[i]; });
    if constexpr (V) {
        for (int j = 0; j < n; ++j) {
            const int dst = rk[j];
            for_each_owned(ex, n, [&](int i) BATCHLAS_DLA_BODY { a(i, dst) = z(i, j); });
        }
    }
    ex.barrier(Mem::Global);
    return info;
}

// The host syev_cta / syev_cta_fused defaults (their bump over SteqrParams{}).
inline constexpr steqr_opts default_opts{400, SteqrShiftStrategy::Wilkinson, SteqrUpdateScheme::EXP};

}  // namespace syev_detail

namespace routine {
// xSYEV / xHEEV, jobz J, triangle UL.
template <JobType J, Uplo UL> struct syev {
    static constexpr const char* name = "syev";
};
}  // namespace routine

template <JobType J, Uplo UL, int N, class Alg, class T> struct tier_info<routine::syev<J, UL>, tier::Cta<N, Alg>, T> {
    static_assert(N == 4 || N == 8 || N == 16 || N == 32, "syev: tier::Cta<N> has N in {4, 8, 16, 32}");
    static constexpr int wg_size = 32;  // the smallest launch: one sub-group, 32 / N problems
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int) { return syev_detail::cta_plan<J == JobType::EigenVectors, N, T>{}.plan.size(); }
};
template <JobType J, Uplo UL, int NB, class Alg, class T> struct tier_info<routine::syev<J, UL>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = sytrd_detail::blocked_cfg<T>::threads;  // 256: sytrd's, steqr's and ormqr's
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) {
        return syev_detail::blocked_plan<J == JobType::EigenVectors, NB, T>(n).plan.size();
    }
};
template <JobType J, Uplo UL, class T> struct tiers<routine::syev<J, UL>, T> {
    using type = tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>, tier::Blocked<32>>;
};

// Bytes of ONE problem's workspace (Cta: local tile, v, w and the Hermitian accumulator; Blocked:
// global d, e, tau, Z and the stages' slice, local the stages' maximum).
template <JobType J, Uplo UL, class Tier, class T> constexpr WorkspaceSize syev_buffer_size(int n) {
    return tier_info<routine::syev<J, UL>, resolve_tier_t<routine::syev<J, UL>, T, Tier>, T>::buffer_size(n);
}

// xSYEV / xHEEV on one n x n matrix A, reading only its UL triangle (Im(diag) ignored): W(0..n-1)
// the eigenvalues ascending (real), A := the eigenvectors (column j for W(j)) with EigenVectors,
// A overwritten with NoEigenVectors (Blocked) or untouched (Cta).
//   tier::Cta<N>      n <= N, one problem per SgpPartition<N>, COLLECTIVE over the whole sub-group
//                     (every lane calls with its partition's problem); ws.local. opts: steqr's.
//   tier::Blocked<NB> any n, one problem per WorkGroup<256>, COLLECTIVE; ws.global is this matrix's
//                     slice; opts.max_sweeps only (the work-group steqr's shift and update).
// Returns 0, k > 0 when steqr did not converge (Cta: 1; Blocked: the e(j) != 0 left), -2 when A is
// not square with 1 <= n (<= N for Cta); uniform.
template <JobType J, Uplo UL, class Tier, class Ex, class MA, class WV>
BATCHLAS_DLA_INLINE int syev(const Ex& ex, const MA& A, const WV& W, const Workspace& ws, const steqr_opts& opts = syev_detail::default_opts) {
    using T = std::remove_cv_t<typename MA::value_type>;
    using R = routine::syev<J, UL>;
    using Tr = resolve_tier_t<R, T, Tier>;
    static_assert(HasTier<R, Tr, T>, "syev: the routine has no such tier (tier::Cta<N>, tier::Blocked<NB>)");
    constexpr bool kVecs = J == JobType::EigenVectors;
    if constexpr (syev_detail::is_blocked_v<Tr>) {
        constexpr int WG = tier_info<R, Tr, T>::wg_size;
        static_assert(std::is_same_v<Ex, WorkGroup<WG>>, "syev: tier::Blocked<NB> runs on device::WorkGroup<256> (one matrix per work-group)");
        const auto a = l3::global_of(A);
        if (a.m < 1 || a.n != a.m) return -2;
        return syev_detail::syev_blocked_run<kVecs, UL, Tr::nb>(ex, a, W, ws, opts);
    } else {
        constexpr int N = Tr::n_max;
        constexpr std::size_t P = N;
        static_assert(std::is_same_v<Ex, SgpPartition<N, false>> || std::is_same_v<Ex, SgpPartition<N, true>>,
                      "syev: tier::Cta<N> runs on device::SgpPartition<N> (device::wrap(make_partition<N>(sg)))");
        const int n = A.rows();
        if (n < 1 || n > N || A.cols() != n) return -2;
        using Lay = syev_detail::cta_layout<kVecs, N, T>;
        const syev_detail::cta_plan<kVecs, N, T> plan{};
        const auto a = carve_local(ws, plan.a).get();
        const auto v = carve_local(ws, plan.v).get();
        const auto w = carve_local(ws, plan.w).get();
        const auto q = carve_local(ws, plan.q).get();
        // The solve runs maskless over the whole sub-group: every lane reaches it (the contract).
        const auto solve = make_partition<P, false>(ex.part.sg);
        const syev_detail::cta_opts o{opts.max_sweeps, opts.shift, opts.update, true, true};
        bool failed = false;
        syev_detail::syev_cta_run<T, P, kVecs, Lay::lda, Lay::ldq>(ex.part, solve, a, 0, v, 0, w, 0, q, 0, A, A, W, n, UL == Uplo::Upper,
                                                                   true, o, [&](bool f) BATCHLAS_DLA_BODY { failed = f; });
        return failed ? 1 : 0;
    }
}

}  // namespace batchlas::device

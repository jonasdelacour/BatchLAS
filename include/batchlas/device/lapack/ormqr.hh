// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xORMQR / xUNMQR, the small tier: C := op(Q) C or C op(Q) with Q = H(1) ... H(k) from
// geqrf's packed (A, tau), on one P-lane partition per matrix (xORM2R's reflector loop). The
// QL ordering (xORM2L, Q = H(k) ... H(1), syev's tridiagonal back-transform) is the same body
// with a policy flag. ormqr_detail: the two bodies the host CTA launcher (ormqr_cta.cc) runs.

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device::inline v2 {

namespace ormqr_detail {

// Reflector order. QR: Q = H(1) ... H(k), so Left NoTrans and Right Trans apply H(k) first;
// QL: Q = H(k) ... H(1), the reverse. Compactly descending = (Left ^ QL) ? !TrConj : TrConj.
template <bool QL, bool Left, bool TrConj> constexpr bool descending() { return (Left ^ QL) ? (!TrConj) : TrConj; }

// vec_ok's 16-byte column access. sycl::vec<std::complex> does not compile on AdaptiveCpp; complex
// never takes the path (the host clears vec_ok), so it only needs a type that compiles.
template <class T, int N> struct PlainVec {
    T v[N];
    T& operator[](int i) { return v[i]; }
    const T& operator[](int i) const { return v[i]; }
};
template <class T, int N> using ColVec = std::conditional_t<Arith<T>::is_complex, PlainVec<T, N>, sycl::vec<T, N>>;

// LEFT: lane j owns column j of the n x n C in a register Frag; reflector ii is staged per
// absolute row in V (P scalars at base_v, zero outside its support) so the rank-1 update runs
// over the compile-time row range with constant subscripts. vec_ok (real T only): C's column,
// ld and stride are 16-byte aligned and n is a multiple of 16 / sizeof(T): 16-byte accesses
// (lane j's column is ld apart from its neighbour's, so scalar rows cost 32 sectors each).
// NO EARLY RETURN: a dead partition (live = false) applies tau = 0 reflectors and stores nothing.
template <int P, bool QL, bool TrConj, class Ops, class Ex, class MA, class VT, class MC, class VB>
BATCHLAS_DLA_INLINE void orm2x_left(const Ex& ex, int32_t lane, const MA& A_prob, const VT& TAU_prob, const MC& C_prob,
                                    const VB& V_local, int32_t base_v, int32_t n, int32_t k, bool live, bool vec_ok) {
    using T = std::remove_cvref_t<decltype(C_prob(0, 0))>;
    using Ar = arith_t<Ops, T>;
    const bool active_col = live && (lane < n);
    const int32_t j = lane;

    constexpr int32_t VW = static_cast<int32_t>(16 / sizeof(T));
    using VecT = ColVec<T, VW>;
    Frag<T, P, ColPerLane> C_col;
    BATCHLAS_UNROLL_FULL
    for (int32_t r = 0; r < P; ++r) C_col[r] = Ar::zero();
    if (vec_ok && active_col) {
        auto* col_ptr = reinterpret_cast<const VecT*>(&C_prob(0, j));
        BATCHLAS_UNROLL_FULL
        for (int32_t r = 0; r < P; r += VW) {
            if (r < n) {
                const VecT v = col_ptr[r / VW];
                BATCHLAS_UNROLL_FULL
                for (int32_t t = 0; t < VW; ++t) C_col[r + t] = v[t];
            }
        }
    } else if (active_col) {
        // Every index into C_col is a compile-time constant: a run-time subscript (offset + r, or
        // a bound of n) moves the array to local memory.
        BATCHLAS_UNROLL_FULL
        for (int32_t r = 0; r < P; ++r) {
            if (r < n) {
                C_col[r] = C_prob(r, j);
            }
        }
    }

    constexpr bool desc = descending<QL, true, TrConj>();
    const int32_t i0 = desc ? (k - 1) : 0;
    const int32_t i1 = desc ? -1 : k;
    const int32_t step = desc ? -1 : 1;
    using VLocal = Vec<T, Ext<P>, Unit, Mem::Local, Cyclic, VB>;

    for (int32_t ii = i0; ii != i1; ii += step) {
        const T tau_i = (live && ii >= 0 && ii < k) ? TAU_prob(ii) : Ar::zero();
        const T tau_eff = TrConj ? Ar::conj(tau_i) : tau_i;
        if constexpr (!QL) {
            // QR: reflector ii is column ii, v(ii) = 1, v(0:ii-1) = 0, support to row n-1.
            const bool in_support = live && (lane >= ii) && (lane < n);
            V_local[base_v + lane] = !in_support ? Ar::zero() : ((lane == ii) ? Ar::one() : A_prob(lane, ii));
        } else {
            // QL: reflector ii is H(ii+1); with m == n the pivot row is (n-k)+ii, v(pivot) = 1
            // and v(pivot+1:) = 0.
            const int32_t col = (n - k) + ii;
            const int32_t pivot = (n - k) + ii;
            const bool in_support = live && (lane <= pivot) && (lane < n);
            V_local[base_v + lane] = !in_support ? Ar::zero() : ((lane == pivot) ? Ar::one() : A_prob(lane, col));
        }
        group_barrier(ex.part);
        // vh(r) == V_local[base_v + r], formed here as the hand loop's address was (hoisted above
        // the loop it holds one more register: complex P = 32 spilled 8 B more).
        const auto vh = assume_published(VLocal{V_local, {}, {}, base_v});
        if (active_col && !Ar::is_zero(tau_eff)) larf<Side::Left, Ops>(C_col, vh, tau_eff);  // C(:, j) := H C(:, j)
        group_barrier(ex.part);  // every lane has consumed v before it is overwritten
    }

    if (vec_ok && active_col) {
        auto* col_ptr = reinterpret_cast<VecT*>(&C_prob(0, j));
        BATCHLAS_UNROLL_FULL
        for (int32_t r = 0; r < P; r += VW) {
            if (r < n) {
                VecT v{};
                BATCHLAS_UNROLL_FULL
                for (int32_t t = 0; t < VW; ++t) v[t] = C_col[r + t];
                col_ptr[r / VW] = v;
            }
        }
    } else if (active_col) {
        BATCHLAS_UNROLL_FULL
        for (int32_t r = 0; r < P; ++r) {
            if (r < n) {
                C_prob(r, j) = C_col[r];
            }
        }
    }
}

// RIGHT: the P x P tile of C (column-major, at base_c of CB) and v (at base_v of VB) in local
// memory; lane r owns row r (no work vector): C(r, offset:) := C(r, offset:) H by core larf on
// the lane's own row. Explicit arithmetic. NO EARLY RETURN, as LEFT.
template <int P, bool QL, bool TrConj, class Ex, class MA, class VT, class MC, class CB, class VB>
BATCHLAS_DLA_INLINE void orm2x_right(const Ex& ex, int32_t lane, bool row_live, const MA& A_prob, const VT& TAU_prob,
                                     const MC& C_prob, const CB& C_local, int32_t base_c, const VB& V_local, int32_t base_v,
                                     int32_t n, int32_t k, bool live) {
    using T = std::remove_cvref_t<decltype(C_prob(0, 0))>;
    using Ar = Arith<T>;
    const int32_t P_i = static_cast<int32_t>(P);

    if (row_live) {
        for (int32_t col = 0; col < n; ++col) C_local[base_c + lane + col * P_i] = C_prob(lane, col);
        for (int32_t col = n; col < P_i; ++col) C_local[base_c + lane + col * P_i] = Ar::zero();
    } else {
        for (int32_t col = 0; col < P_i; ++col) C_local[base_c + lane + col * P_i] = Ar::zero();
    }
    group_barrier(ex.part);

    constexpr bool desc = descending<QL, false, TrConj>();
    const int32_t i0 = desc ? (k - 1) : 0;
    const int32_t i1 = desc ? -1 : k;
    const int32_t step = desc ? -1 : 1;

    for (int32_t ii = i0; ii != i1; ii += step) {
        const T tau_i = (live && ii >= 0 && ii < k) ? TAU_prob(ii) : Ar::zero();
        int32_t offset = 0;
        int32_t len = 0;
        if constexpr (!QL) {
            // QR: reflector ii in column ii, leading zeros of length ii.
            offset = ii;
            len = n - ii;
            if (lane < len) {
                const int32_t row = offset + lane;
                V_local[base_v + lane] = (lane == 0) ? Ar::one() : (live ? A_prob(row, ii) : Ar::zero());
            } else {
                V_local[base_v + lane] = Ar::zero();
            }
        } else {
            // QL (xGEQLF): v(pivot) = 1, v(pivot+1:m) = 0, v(0:pivot-1) in A(0:pivot-1, n-k+ii).
            const int32_t col = (n - k) + ii;
            const int32_t pivot = (n - k) + ii;  // m == n
            offset = 0;
            len = pivot + 1;
            if (lane < len) {
                V_local[base_v + lane] = (lane == pivot) ? Ar::one() : (live ? A_prob(lane, col) : Ar::zero());
            } else {
                V_local[base_v + lane] = Ar::zero();
            }
        }
        group_barrier(ex.part);

        // C(lane, offset:) H on the lane's own row (algo port); a predicate, never a return.
        const T tau_eff = TrConj ? Ar::conj(tau_i) : tau_i;
        if (lane < n && !(Ar::is_zero(tau_eff) || len <= 0))
            larf<Side::Right>(Thread{}, tau_eff, assume_published(Vec<T, Dyn, Unit, Mem::Local, Cyclic, VB>{V_local, Dyn{len}, {}, base_v}),
                              Mat<T, Dyn, Dyn, Dyn, Mem::Local, Cyclic, CB>{C_local, Dyn{1}, Dyn{len}, Dyn{P_i}, base_c + lane + offset * P_i});
        group_barrier(ex.part);
    }

    if (row_live) {
        for (int32_t col = 0; col < n; ++col) C_prob(lane, col) = C_local[base_c + lane + col * P_i];
    }
}


// Cta<P> local memory of one matrix: Right's P x P tile of C first, then v (P scalars).
template <Side SD, int P, class T> struct CtaPlan {
    ws::Slot<T> c{};
    ws::Slot<T> v{};
    WorkspaceSize size{};
    constexpr CtaPlan() {
        Plan p;
        if constexpr (SD == Side::Right) c = p.take<T>(ws::Space::Local, static_cast<std::size_t>(P) * static_cast<std::size_t>(P));
        v = p.take<T>(ws::Space::Local, static_cast<std::size_t>(P));
        size = p.size();
    }
};

}  // namespace ormqr_detail

namespace routine {
template <Side SD, Transpose TR> struct ormqr {
    static constexpr const char* name = "ormqr";
};
}  // namespace routine

/// @brief Bytes one matrix needs for ormqr() at Cta<P>: Left P scalars (v); Right a P x P tile, then v.
template <Side SD, class Tier, class T> constexpr WorkspaceSize ormqr_buffer_size(int n) {
    static_assert(HasTier<routine::ormqr<SD, Transpose::NoTrans>, Tier, T>, "ormqr: the device tiers are Cta<P>, P in {4, 8, 16, 32}");
    static_cast<void>(n);
    return ormqr_detail::CtaPlan<SD, Tier::n_max, T>{}.size;
}

/// @brief xORMQR inside a kernel: C := op(Q) C (Side::Left) or C op(Q) (Side::Right).
/// @tparam SD, TR  side and op(Q) (Trans and ConjTrans both apply Q^H; complex Trans is rejected)
/// @tparam Tier    tier::Cta<P>, P in {4, 8, 16, 32}: C and A square n x n, n <= P
/// @tparam Ops     the Left form's arithmetic: arith::Native (the host route's; T real or
///                 std::complex) or arith::Explicit (any T, device::Cx included)
/// @param ex   SgpPartition<P>, one matrix per P-lane partition; COLLECTIVE over the sub-group
/// @param A    geqrf's packed reflectors (n x n); @param C the matrix applied to, overwritten
/// @param tau  tau(0:k); @param ws this matrix's slice, ormqr_buffer_size<SD, Tier, T>(n) bytes
/// @param k    reflectors, k <= n; @param live false for a partition with no matrix
/// @return 0 (ormqr has no info)
template <Side SD, Transpose TR, class Tier, class Ops = arith::Native, class Ex, class MA, class MC, class VT>
BATCHLAS_DLA_INLINE int ormqr(const Ex& ex, const MA& A, const MC& C, const VT& tau, const Workspace& ws, int k, bool live = true) {
    using T = typename MC::value_type;
    static_assert(HasTier<routine::ormqr<SD, TR>, Tier, T>, "ormqr: the device tiers are Cta<P>, P in {4, 8, 16, 32}");
    constexpr int P = Tier::n_max;
    static_assert(P == 4 || P == 8 || P == 16 || P == 32, "ormqr Cta<P>: P in {4, 8, 16, 32}");
    static_assert(Ex::size == P && Ex::sub_whole, "ormqr Cta<P>: ex is a P-lane Partition (SgpPartition<P>)");
    static_assert(!(Arith<T>::is_complex && TR == Transpose::Trans), "ormqr: complex Q^T is not LAPACK's; pass ConjTrans");
    constexpr bool tr = TR != Transpose::NoTrans;
    constexpr ormqr_detail::CtaPlan<SD, P, T> plan{};
    const int32_t n = C.rows();
    const int32_t lane = static_cast<int32_t>(ex.lid());
    if constexpr (SD == Side::Left) {
        ormqr_detail::orm2x_left<P, false, tr, Ops>(ex, lane, A, tau, C, carve_local(ws, plan.v), 0, n, k, live, false);
    } else {
        ormqr_detail::orm2x_right<P, false, tr>(ex, lane, live && (lane < n), A, tau, C, carve_local(ws, plan.c), 0,
                                                carve_local(ws, plan.v), 0, n, k, live);
    }
    return 0;
}

template <Side SD, Transpose TR, class T> struct tiers<routine::ormqr<SD, TR>, T> {
    using type = tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>>;
};
template <Side SD, Transpose TR, int P, class Alg, class T> struct tier_info<routine::ormqr<SD, TR>, tier::Cta<P, Alg>, T> {
    static constexpr int wg_size = 32, sg_size = 32;  // 32 / P matrices per sub-group
    static constexpr bool admits(int n) { return n >= 0 && n <= P; }
    static constexpr WorkspaceSize buffer_size(int n) { return ormqr_buffer_size<SD, tier::Cta<P, Alg>, T>(n); }
};

}  // namespace batchlas::device

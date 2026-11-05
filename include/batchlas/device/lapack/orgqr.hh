// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xORGQR / xUNGQR, the small tier: Q (m x n, n <= m) in place of geqrf's packed reflectors
// (A, tau), Q = H(0) H(1) ... H(k-1), k = n, by xORG2R's backward loop on one P-lane partition per
// matrix: lane j owns column j of Q in registers; reflector i is staged in local memory, applied
// to the columns right of it (core larf, Frag form), then its own column becomes
// (0, ..., 0, 1 - tau_i, -tau_i v(i+1:m)). No host route runs this tier (the host orgqr is the
// blocked identity + ormqr pipeline), so it is checked against that route with a tolerance.

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device::inline v2 {

namespace orgqr_detail {

// The xORG2R body. V: P scalars of local memory at base_v (any base: accessor, multi_ptr, T*).
// NO EARLY RETURN: every lane of the sub-group runs the 2k barriers; a dead partition
// (live = false) applies tau = 0 reflectors and stores nothing.
template <int P, class Ops, class Ex, class MA, class VT, class VB>
BATCHLAS_DLA_INLINE void org2r_body(const Ex& ex, const MA& A, const VT& tau, const VB& V, int32_t base_v, bool live) {
    using T = std::remove_cvref_t<decltype(A(0, 0))>;
    using Ar = arith_t<Ops, T>;
    const int32_t m = A.rows(), n = A.cols();
    const int32_t k = m < n ? m : n;
    const int32_t lane = static_cast<int32_t>(ex.lid());
    const bool active_col = live && (lane < n);

    // Column j := e_j. Columns k..n-1 stay the unit matrix's; column i < k is rebuilt at step i.
    Frag<T, P, ColPerLane> q;
    BATCHLAS_UNROLL_FULL
    for (int32_t r = 0; r < P; ++r) q[r] = select(r == lane, Ar::one(), Ar::zero());

    using VLocal = Vec<T, Ext<P>, Unit, Mem::Local, Cyclic, VB>;
    for (int32_t i = k - 1; i >= 0; --i) {
        const T tau_i = live ? static_cast<T>(tau(i)) : Ar::zero();
        // v_i by absolute row: 0 above i, the implicit 1 at i, A(i+1:m, i) below, 0 past m.
        const bool in_support = live && (lane >= i) && (lane < m);
        V[base_v + lane] = !in_support ? Ar::zero() : ((lane == i) ? Ar::one() : static_cast<T>(A(lane, i)));
        group_barrier(ex.part);
        const auto vh = assume_published(VLocal{V, {}, {}, base_v});
        if (active_col && lane > i && !Ar::is_zero(tau_i)) larf<Side::Left, Ops>(q, vh, tau_i);  // Q(:, j) := H(i) Q(:, j)
        if (active_col && lane == i) {
            const T ntau = Ar::neg(tau_i);
            BATCHLAS_UNROLL_FULL
            for (int32_t r = 0; r < P; ++r) {
                const T below = Ar::mul(ntau, static_cast<T>(detail::unpub(vh)(r)));  // zero past m: v(r) = 0
                q[r] = select(r < i, Ar::zero(), select(r == i, Ar::sub(Ar::one(), tau_i), below));
            }
        }
        group_barrier(ex.part);  // every lane has read v before the next step overwrites it
    }

    if (active_col) {
        BATCHLAS_UNROLL_FULL
        for (int32_t r = 0; r < P; ++r) {
            if (r < m) A(r, lane) = q[r];
        }
    }
}


// Cta<P> local memory of one matrix: v, P scalars.
template <int P, class T> struct CtaPlan {
    ws::Slot<T> v{};
    WorkspaceSize size{};
    constexpr CtaPlan() {
        Plan p;
        v = p.take<T>(ws::Space::Local, static_cast<std::size_t>(P));
        size = p.size();
    }
};

}  // namespace orgqr_detail

namespace routine {
struct orgqr {
    static constexpr const char* name = "orgqr";
};
}  // namespace routine

template <class Tier, class T> constexpr WorkspaceSize orgqr_buffer_size(int m, int n) {
    static_assert(HasTier<routine::orgqr, Tier, T>, "orgqr: the device tiers are Cta<P>, P in {4, 8, 16, 32}");
    static_cast<void>(m);
    static_cast<void>(n);
    return orgqr_detail::CtaPlan<Tier::n_max, T>{}.size;
}
template <class Tier, class T> constexpr WorkspaceSize orgqr_buffer_size(int n) { return orgqr_buffer_size<Tier, T>(n, n); }

/// @brief xORGQR inside a kernel: A (m x n, n <= m <= P) := Q from geqrf's reflectors.
/// @tparam Tier tier::Cta<P>, P in {4, 8, 16, 32}
/// @tparam Ops  arith::Explicit (any T, device::Cx included) or arith::Native (real, std::complex)
/// @param ex    SgpPartition<P>, one matrix per P-lane partition; COLLECTIVE over the sub-group
/// @param A     geqrf's output, overwritten by Q; @param tau tau(0:n)
/// @param ws    this matrix's slice, orgqr_buffer_size<Tier, T>(m, n) bytes
/// @param live  false for a partition with no matrix
/// @return 0 (orgqr has no info)
template <class Tier, class Ops = arith::Explicit, class Ex, class MA, class VT>
BATCHLAS_DLA_INLINE int orgqr(const Ex& ex, const MA& A, const VT& tau, const Workspace& ws, bool live = true) {
    using T = typename MA::value_type;
    static_assert(HasTier<routine::orgqr, Tier, T>, "orgqr: the device tiers are Cta<P>, P in {4, 8, 16, 32}");
    constexpr int P = Tier::n_max;
    static_assert(P == 4 || P == 8 || P == 16 || P == 32, "orgqr Cta<P>: P in {4, 8, 16, 32}");
    static_assert(Ex::size == P && Ex::sub_whole, "orgqr Cta<P>: ex is a P-lane Partition (SgpPartition<P>)");
    constexpr orgqr_detail::CtaPlan<P, T> plan{};
    orgqr_detail::org2r_body<P, Ops>(ex, A, tau, carve_local(ws, plan.v), 0, live);
    return 0;
}

template <class T> struct tiers<routine::orgqr, T> {
    using type = tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>>;
};
template <int P, class Alg, class T> struct tier_info<routine::orgqr, tier::Cta<P, Alg>, T> {
    static constexpr int wg_size = 32, sg_size = 32;  // 32 / P matrices per sub-group
    static constexpr bool admits(int n) { return n >= 1 && n <= P; }
    static constexpr bool admits(int m, int n) { return n >= 1 && n <= m && m <= P; }
    static constexpr WorkspaceSize buffer_size(int n) { return orgqr_buffer_size<tier::Cta<P, Alg>, T>(n); }
    static constexpr WorkspaceSize buffer_size(int m, int n) { return orgqr_buffer_size<tier::Cta<P, Alg>, T>(m, n); }
};

}  // namespace batchlas::device

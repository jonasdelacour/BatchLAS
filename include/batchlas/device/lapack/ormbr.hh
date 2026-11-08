// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xORMBR / xUNMBR for gebrd's upper bidiagonal reduction (A m x n, m >= n): C := op(Q) C,
// C op(Q), op(P) C or C op(P), Q = H(0) .. H(n-1) with H(i) = I - tauq(i) v v^H (v(i) = 1, v below
// in A(i+1:, i)), P = G(0) .. G(n-2) with G(i) = I - taup(i) u u^H (u(i+1) = 1, u(i+2:) =
// conj(A(i, i+2:)): gebrd's rows hold conj(u)). tier::Blocked<1>: any order on a work-group, one
// core larf per reflector (Left: a team per column; Right: an item per row of C).

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/view.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device::inline v2 {

namespace routine {
template <char Vect, Side SD, Transpose TR> struct ormbr {
    static constexpr const char* name = "ormbr";
};
}  // namespace routine

template <char Vect, Side SD, Transpose TR, class T> struct tiers<routine::ormbr<Vect, SD, TR>, T> {
    using type = tier::list<tier::Blocked<1>>;
};
// No workspace (C is updated in place); any work-group width runs it.
template <char Vect, Side SD, Transpose TR, class T> struct tier_info<routine::ormbr<Vect, SD, TR>, tier::Blocked<1>, T> {
    static constexpr int wg_size = 256, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};
template <char Vect, Side SD, Transpose TR, class Tier, class T> constexpr WorkspaceSize ormbr_buffer_size(int n) {
    using Ro = routine::ormbr<Vect, SD, TR>;
    return tier_info<Ro, resolve_tier_t<Ro, T, Tier>, T>::buffer_size(n);
}

// xORMBR, host order (A, tau, C, ws), vect / side / trans as template parameters ('Q': tauq,
// 'P': taup). Q / P has order C.rows() (Left) or C.cols() (Right), which must be A.rows() ('Q') or
// A.cols() ('P'); A is m x n, m >= n. Trans is real only. COLLECTIVE over a work-group; A, tau, C
// visible on entry, C on return. Returns 0, or -1 / -3 for a mismatched A / C (nothing written).
template <char Vect, Side SD, Transpose TR, class Tier = tier::Blocked<1>, class Ex, class MA, class TV, class MC>
BATCHLAS_DLA_INLINE int ormbr(const Ex& ex, const MA& A, const TV& tau, const MC& C, const Workspace&) {
    using T = typename MC::value_type;
    using Ar = Arith<T>;
    using Ro = routine::ormbr<Vect, SD, TR>;
    static_assert(Vect == 'Q' || Vect == 'P', "ormbr: Vect is 'Q' or 'P'");
    static_assert(std::is_same_v<resolve_tier_t<Ro, T, Tier>, tier::Blocked<1>>, "ormbr: tier::Blocked<1>");
    static_assert(TR != Transpose::Trans || !Ar::is_complex, "ormbr: a complex T takes Transpose::ConjTrans");
    static_assert(is_work_group_v<Ex>, "ormbr: tier::Blocked<1> runs on a WorkGroup");
    constexpr bool q = Vect == 'Q';
    const int nq = SD == Side::Left ? C.rows() : C.cols();
    const int m = A.rows(), n = A.cols();
    if (m < n) return -1;
    if (nq != (q ? m : n)) return -3;
    const int k = q ? n : n - 1;  // reflectors; reflector i acts on indices i + shift ..
    constexpr int shift = q ? 0 : 1;
    // Q C and C Q^H apply the last reflector first (LAPACK xORM2R's i3 = -1).
    constexpr bool ascending = (SD == Side::Left) == (TR != Transpose::NoTrans);
    for (int t = 0; t < k; ++t) {
        const int i = ascending ? t : k - 1 - t;
        const int s = i + shift, len = nq - s;
        const T ti = static_cast<T>(tau(i));
        const T tau_i = TR == Transpose::NoTrans ? ti : Ar::conj(ti);
        if constexpr (q) {
            const auto v = unit_head(A.col(i).tail(s), 0);
            if constexpr (SD == Side::Left) larf<Side::Left>(ex, tau_i, v, C.sub(s, 0, len, C.cols()));
            else larf<Side::Right>(ex, tau_i, v, C.sub(0, s, C.rows(), len));
        } else {
            const auto row = A.row(i).tail(s);
            const auto u = fn_vec(len, [=](int j) { return j == 0 ? Ar::one() : Ar::conj(static_cast<T>(row(j))); });
            if constexpr (SD == Side::Left) larf<Side::Left>(ex, tau_i, u, C.sub(s, 0, len, C.cols()));
            else larf<Side::Right>(ex, tau_i, u, C.sub(0, s, C.rows(), len));
        }
        // Left: lanes of a column's team stride the rows the next reflector reduces over; Right:
        // an item owns its row of C throughout.
        if constexpr (SD == Side::Left) ex.barrier(Mem::Global);
    }
    ex.barrier(Mem::Global);
    return 0;
}

}  // namespace batchlas::device

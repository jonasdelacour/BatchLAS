// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Applying the bidiagonal reduction's transforms inside a kernel: `ormbr`, LAPACK's
/// xORMBR / xUNMBR for `gebrd`'s upper bidiagonal form.
///
/// `ormbr<VT, SD, TR>(ex, A, tau, C, ws)` multiplies C by Q or P of \f$A = Q B P^H\f$ from either
/// side, on one work-group, with `tier::Blocked<1>`. Reference: @ref api_device_lapack_qr.
/// @ingroup api_device_lapack_qr
// One core larf per reflector (Left: a team per column; Right: an item per row of C).

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

namespace batchlas::device {

namespace routine {
/// @brief The routine tag of `ormbr<VT, SD, TR>`, for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_qr
template <Vect VT, Side SD, Transpose TR> struct ormbr {
    static constexpr const char* name = "ormbr";
};
}  // namespace routine

/// @brief The tiers `with_tier` tries for `ormbr<VT, SD, TR>`: `tier::Blocked<1>`.
/// @ingroup api_device_lapack_qr
template <Vect VT, Side SD, Transpose TR, class Scalar> struct tiers<routine::ormbr<VT, SD, TR>, Scalar> {
    using type = tier::list<tier::Blocked<1>>;
};
/// @brief `ormbr<VT, SD, TR>` in `tier::Blocked<1>`: one C of any order per work-group, no
/// workspace.
///
/// C is updated in place, so `buffer_size` is zero. `with_tier` checks `WorkGroup<256, 32>`, but
/// any work-group width runs it.
/// @ingroup api_device_lapack_qr
template <Vect VT, Side SD, Transpose TR, class Scalar> struct tier_info<routine::ormbr<VT, SD, TR>, tier::Blocked<1>, Scalar> {
    static constexpr int wg_size = 256, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};

/// @brief xORMBR / xUNMBR inside a kernel: C := op(Q) C, C op(Q), op(P) C or C op(P) for the Q
/// and P of `gebrd`'s reduction \f$A = Q B P^H\f$ (A m x n, m >= n, B upper bidiagonal).
///
/// \f$Q = H_0 \cdots H_{n-1}\f$ with \f$H_i = I - \tau^q_i v_i v_i^H\f$ (\f$v_i(i) = 1\f$, the
/// rest of \f$v_i\f$ below the diagonal in column i of A), and \f$P = G_0 \cdots G_{n-2}\f$ with
/// \f$G_i = I - \tau^p_i u_i u_i^H\f$ (\f$u_i(i+1) = 1\f$, the rest of \f$u_i\f$ conjugated in row
/// i of A, right of the superdiagonal). Q or P has order C.rows() (Left) or C.cols() (Right),
/// which must be A.rows() for Q and A.cols() for P. Arguments in the host order
/// (A, tau, C, ws). Collective over a work-group executor, every item with the same arguments;
/// A, tau and C visible to every item on entry, C visible on return. The fifth argument, a
/// `Workspace`, is not used: pass `Workspace{}`.
/// @pre The executor's sub-group size `SG` is the kernel's (32 for the tier's executor).
/// @tparam VT    `Vect::Q` (pass `gebrd`'s `tauq`) or `Vect::P` (pass `taup`)
/// @tparam SD    `Side::Left` or `Side::Right`
/// @tparam TR    `Transpose::NoTrans`, `Transpose::ConjTrans`, or `Transpose::Trans` for a real
///               type
/// @tparam Tier  `tier::Blocked<1>`, the default and only tier
/// @tparam Ex    a `WorkGroup` executor
/// @param[in]     ex   the executor
/// @param[in]     A    `gebrd`'s output, m x n with m >= n
/// @param[in]     tau  `tauq` (n scalars) for Q, `taup` for P
/// @param[in,out] C    the matrix to multiply, updated in place
/// @return `info`, the same on every item
/// @retval 0   success
/// @retval -1  A has fewer rows than columns (nothing written)
/// @retval -3  C's order on side SD does not match A (nothing written)
/// @ingroup api_device_lapack_qr
template <Vect VT, Side SD, Transpose TR, class Tier = tier::Blocked<1>, class Ex, class MA, class TV, class MC>
BATCHLAS_DEVICE_INLINE int ormbr(const Ex& ex, const MA& A, const TV& tau, const MC& C, const Workspace&) {
    using T = typename MC::value_type;
    using Ar = Arith<T>;
    using Ro = routine::ormbr<VT, SD, TR>;
    static_assert(std::is_same_v<resolve_tier_t<Ro, T, Tier>, tier::Blocked<1>>, "ormbr: tier::Blocked<1>");
    static_assert(TR != Transpose::Trans || !Ar::is_complex, "ormbr: a complex T takes Transpose::ConjTrans");
    static_assert(is_work_group_v<Ex>, "ormbr: tier::Blocked<1> runs on a WorkGroup");
    constexpr bool q = VT == Vect::Q;
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

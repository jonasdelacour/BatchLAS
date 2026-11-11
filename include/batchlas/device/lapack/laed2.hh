// SPDX-License-Identifier: MIT
#pragma once
// xLAED2: the deflation of a divide-and-conquer merge diag(D) + rho z z^T (D sorted):
//   laed2                    Givens rotations merge poles closer than the tolerance (z(j) -> 0,
//                            the two eigenvector columns rotated), then every |rho z_k| below
//                            it deflates; returns the gather order, kept first
//   laed2_merge_vector       z = [last row of Q1, sign(rho) first row of Q2] / sqrt(2) (xLAED1's
//                            z of the rank-one tear)
//   detail::fold       a work-group reduction over a contiguous operand (joint_reduce)
// The host merge's StedcComputeV / StedcDeflation kernels and device::stedc run these bodies.
// Every operand an op writes is a callable (i) -> T&: the host passes its batched views'
// element expressions, device::stedc its workspace vectors.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/laed4.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device {

namespace detail {
// portable::joint_reduce over a contiguous x (core allreduce folds in another order); empty: T{}.
template <class Ex, class X, class Op> BATCHLAS_DEVICE_INLINE auto fold(const Ex& ex, const X& x, Op op) {
    static_assert(detail::is_group_v<Ex>, "laed2_detail::fold: a work-group executor");
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(x.inc)>, Unit>, "laed2_detail::fold: x must be contiguous");
    const auto first = x.p + x.off;
    return batchlas::portable::joint_reduce(ex.it.get_group(), first, first + x.size(), op);
}
}  // namespace detail

/// z(0..m) = row1 / sqrt(2), z(m..) = sign * row2 / sqrt(2): the rank-one term of the tear is
/// |rho| u u^T with u = e_{m-1} + sign(rho) e_m (the diagonal corrections are -|rho|), so the
/// Q2 half carries sign(rho); without it the merged vectors mix Q1 and Q2 with inconsistent
/// signs. COLLECTIVE over ex; no trailing barrier.
template <class Ex, class Z, class R1, class R2, class T>
BATCHLAS_DEVICE_HELPER void laed2_merge_vector(const Ex& ex, const Z& z, int m, const R1& row1, const R2& row2, T sign) {
    map(ex, z.head(m), [](T q) { return q / std::sqrt(T(2)); }, row1);
    map(ex, z.tail(m), [=](T q) { return sign * q / std::sqrt(T(2)); }, row2);
}

/// Deflation of the n-pole merge on the work-group ex. d (operand) and d_at (callable) the sorted
/// poles, z the weights (rotated in place), Qb the merge's eigenvector columns, addressed through
/// perm_map (logical -> physical column), rho_at() the tear's rho. Outputs: keep(k) = 1 for a
/// kept pole, permutation(i) the gather order (kept poles ascending, then the deflated ones in
/// reverse), set_reduced(count) on the leader. Scratch (n each, indexable): perm_local, scan_inc,
/// scan_exc and absx (an operand); keep_ptr() / scan_*_ptr() return keep's and the scans' first
/// element (contiguous), read where the scans run.
/// The tolerance is LAPACK's ABSOLUTE 8 eps max(|D|_inf, |z|_inf), taken before the Givens loop so
/// proximity deflation uses it too (a relative one under-deflates clustered small eigenvalues).
/// `evidence: docs/perf/stedc.md#stedc-the-absolute-deflation-tolerance`
/// No trailing barrier: the caller syncs before it reads the outputs.
template <class Ex, class D, class Z, class Q, class AbsX, class DAt, class RhoAt, class KeepAt, class KeepPtr, class PermAt, class MapAt,
          class PermLocal, class ScanInc, class ScanIncPtr, class ScanExc, class ScanExcPtr, class SetReduced>
BATCHLAS_DEVICE_INLINE void laed2(const Ex& ex, std::int64_t n, const D& d, const Z& z, const Q& Qb, const AbsX& absx, const DAt& d_at,
                               const RhoAt& rho_at, const KeepAt& keep, const KeepPtr& keep_ptr, const PermAt& permutation,
                               const MapAt& perm_map, const PermLocal& perm_local, const ScanInc& scan_inc, const ScanIncPtr& scan_inc_ptr,
                               const ScanExc& scan_exc, const ScanExcPtr& scan_exc_ptr, const SetReduced& set_reduced) {
    using T = elem_t<D>;
    const int nn = static_cast<int>(n);
    const auto abs_of = [](T x) { return std::abs(x); };

    for_each_owned(ex, nn, [&](int k) {
        keep(k) = 0;
        scan_exc[k] = 0;
        permutation(k) = -1;
        perm_local[k] = perm_map(k);
    });

    sync<scope::Group>(ex, Mem::Global);

    map(ex, absx, abs_of, d);
    const T eig_max = detail::fold(ex, absx, sycl::maximum<T>());
    map(ex, absx, abs_of, z);
    const T v_max_pre = detail::fold(ex, absx, sycl::maximum<T>());
    const T abs_tol = T(8.0) * std::numeric_limits<T>::epsilon() * std::max(eig_max, v_max_pre);

    for (int j = 0; j < n - 1; j++) {
        if (std::abs(d_at(j + 1) - d_at(j)) <= abs_tol) {
            const T f = z(j + 1), g = z(j);
            auto [c, s, r] = lartg(f, g);
            sync<scope::Group>(ex, Mem::Global);
            if (ex.leader()) {
                z(j) = T(0.0);
                z(j + 1) = r;
            }
            const std::int32_t pj = perm_local[j];
            const std::int32_t pj1 = perm_local[j + 1];
            if (pj >= 0 && pj < n && pj1 >= 0 && pj1 < n) {
                rot(ex, Givens<T>{c, s}, Qb.col(pj1), Qb.col(pj));
            }
            // The leader's z(j), z(j+1) must be visible before the next Givens pair reads z(j+1).
            sync<scope::Group>(ex, Mem::Global);
        }
    }

    sync<scope::Group>(ex, Mem::Global);
    // LAPACK LAED8 based tolerance (small-|z| deflation), using the same absolute tolerance.

    map(ex, absx, abs_of, d);
    const T eig_max2 = detail::fold(ex, absx, sycl::maximum<T>());

    const T v_norm = nrm2<alg::Blue>(ex, z);
    map(ex, absx, [=](T x) { return std::abs(x / v_norm); }, z);
    const T v_max = detail::fold(ex, absx, sycl::maximum<T>());
    auto tol = T(8.0) * std::numeric_limits<T>::epsilon() * std::max(eig_max2, v_max);

    for_each_owned(ex, nn, [&](int k) {
        if (std::abs(rho_at() * absx(k)) > tol) {
            keep(k) = 1;
        } else {
            scan_exc[k] = 1;
        }
    });

    sync<scope::Group>(ex, Mem::Global);

    // Exclusive scans give each kept / deflated pole its slot.
    batchlas::portable::joint_exclusive_scan(ex.it.get_group(), keep_ptr(), keep_ptr() + n, scan_inc_ptr(), 0, sycl::plus<std::int32_t>());
    batchlas::portable::joint_exclusive_scan(ex.it.get_group(), scan_exc_ptr(), scan_exc_ptr() + n, scan_exc_ptr(), 0,
                                             sycl::plus<std::int32_t>());

    for_each_owned(ex, nn, [&](int k) {
        if (keep(k) == 1) {
            permutation(scan_inc[k]) = k;
        } else {
            permutation(n - 1 - scan_exc[k]) = k;
        }
    });

    if (ex.leader()) {
        set_reduced(scan_inc[n - 1] + keep(n - 1));
    }
}

}  // namespace batchlas::device

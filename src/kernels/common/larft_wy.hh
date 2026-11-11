// SPDX-License-Identifier: MIT
#pragma once

// The WY block-reflector machinery (larft, pack_v), shared by four callers.
// evidence: docs/perf/qr.md#larft_wyhh-why-the-wy-helpers-are-shared
// Kernel names are tagged by caller: the callers sit in different device-code clusters
// (src/ops/CMakeLists.txt), so an untagged name would land in two images of one library.

#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>

#include <batchlas/device/lapack/larft.hh>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::wy {

template <typename Tag, typename T, int WG> class LarftKernelName;
template <typename Tag, typename T> class PackVKernelName;

namespace detail {

// LARFT (Forward, Columnwise): T (ib x ib, upper) from unit-lower V (m x ib); one work-group per
// matrix on device::larft (include/batchlas/device/lapack/larft.hh), the only larft.
template <typename Tag, typename T, int WG>
sycl::event larft_forward_columnwise_wg(Queue& q,
                                        T* t_data, int ld_t, int stride_t,
                                        const T* v_data, int ld_v, int stride_v,
                                        int m, int ib,
                                        const T* tau_data, int tau_stride,
                                        int tau_offset, int batch,
                                        const std::uint8_t* mask) {
    static_assert(WG % 32 == 0, "WG must be a whole number of 32-wide sub-groups");

    return q->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<LarftKernelName<Tag, T, WG>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(WG)),
                              sycl::range<1>(static_cast<size_t>(WG))),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch || (mask != nullptr && mask[b] == 0)) return;  // I1: before any barrier

                const device::WorkGroup<WG, 32> ex{it};
                device::larft(
                    ex, device::mat(v_data + static_cast<std::ptrdiff_t>(b) * stride_v, m, ib, ld_v),
                    device::mat(t_data + static_cast<std::ptrdiff_t>(b) * stride_t, ib, ib, ld_t),
                    device::vec(tau_data + static_cast<std::ptrdiff_t>(b) * tau_stride + tau_offset, ib));
            });
    });
}

}  // namespace detail

// Dispatcher over the work-group ladder. device::gemv needs 32-wide sub-groups: a launch-time
// check, never an in-kernel test.
template <typename Tag, typename T>
sycl::event larft_forward_columnwise_batched(Queue& q,
                                             T* t_data, int ld_t, int stride_t,
                                             const T* v_data, int ld_v, int stride_v,
                                             int m, int ib,
                                             const T* tau_data, int tau_stride,
                                             int tau_offset, int batch,
                                             const std::uint8_t* mask = nullptr) {
    if (!q.device().supports_sub_group_size(32))
        throw batchlas::unsupported("larft: the device has no 32-wide sub-groups");
    const auto pick = [&]<int WG>(std::integral_constant<int, WG>) {
        return detail::larft_forward_columnwise_wg<Tag, T, WG>(
            q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
            tau_data, tau_stride, tau_offset, batch, mask);
    };

    if (ib <= 8 && m <= 64)    return pick(std::integral_constant<int, 32>{});
    if (ib <= 16 && m <= 128)  return pick(std::integral_constant<int, 64>{});
    if (ib <= 32 && m <= 256)  return pick(std::integral_constant<int, 128>{});
    return pick(std::integral_constant<int, 256>{});
}

// ---------------------------------------------------------------------------
// PACK_V -- materialise the unit-lower V panel from geqrf's packed output.
//
// V(r, c) = 1 for r == c, A(i0 + r, i0 + c) for r > c, 0 for r < c. That IS the
// whole of what separates orgqr from ormqr-on-an-identity, and it is why the
// trailing update below can be three plain GEMMs rather than a trmm ladder over
// the triangular top block.
// ---------------------------------------------------------------------------
template <typename Tag, typename T>
sycl::event pack_v_panel_batched(Queue& q,
                                 T* v_out, int ld_v_out, int stride_v_out,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 int i0, int ib, int nq,
                                 const std::uint8_t* mask = nullptr) {
    const int m = nq - i0;
    const int ld_a = a.ld();
    const int stride_a = a.stride();
    const T* a_ptr = a.data_ptr();
    const int batch = a.batch_size();

    return q->submit([&](sycl::handler& h) {
        // DIM 2 IS THE ROW, NOT THE COLUMN: dim 2 is fastest-varying and both operands
        // are column-major, so the column there is uncoalesced (32 sectors per warp, not 4).
        // evidence: docs/perf/qr.md#larft_wyhh-pack_v-indexes-the-row-fastest
        h.parallel_for<PackVKernelName<Tag, T>>(
            sycl::range<3>(static_cast<size_t>(batch), static_cast<size_t>(ib),
                           static_cast<size_t>(m)),
            [=](sycl::id<3> idx) {
                const int b = static_cast<int>(idx[0]);
                if (mask != nullptr && mask[b] == 0) return;  // I3
                const int c = static_cast<int>(idx[1]);
                const int r = static_cast<int>(idx[2]);
                T val = T(0);
                if (r == c) {
                    val = T(1);
                } else if (r > c) {
                    val = a_ptr[static_cast<std::ptrdiff_t>(b) * stride_a + (i0 + r) + (i0 + c) * ld_a];
                }
                v_out[static_cast<std::ptrdiff_t>(b) * stride_v_out + r + c * ld_v_out] = val;
            });
    });
}

}  // namespace batchlas::wy

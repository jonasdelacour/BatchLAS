// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

// The WY block-reflector machinery (larft, pack_v), shared by ormqr_blocked.cc and
// geqrf_blocked.cc. sytrd_sy2sb.cc's private copy is unreachable, not a third user.
// evidence: docs/perf/qr.md#larft_wyhh-why-the-wy-helpers-are-shared
//
// KERNEL NAMES ARE TAGGED BY CALLER, and that is not decoration: the callers sit in
// DIFFERENT device-code clusters (src/ops/CMakeLists.txt), so an untagged
// closure type would be emitted into two device images of one shared library.
//
// `use_device` is a PARAMETER, never a getenv here: ormqr reads BATCHLAS_ORMQR_IMPL
// itself, and a getenv here would tie geqrf's kernel choice to ormqr's variable.

#include <batchlas/blas/device.hh>
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/larft-orgqr-ormqr.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/matrix.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::wy {

// Kernel name tags. `Tag` is supplied by the caller (an incomplete struct is
// enough); WG and Device make the four work-group ladders and the two
// implementations distinct.
template <typename Tag, typename T, int WG, bool Device> class LarftKernelName;
template <typename Tag, typename T> class PackVKernelName;

namespace detail {

// ---------------------------------------------------------------------------
// LARFT -- form T for a block of Householder vectors V (Forward, Columnwise).
// V is (m x ib) unit-lower (diag = 1, upper = 0). T is (ib x ib) upper
// triangular. ONE WORK-GROUP PER MATRIX; the inner reductions are work-group
// collectives, so this is not the batch-only-parallelism shape.
// ---------------------------------------------------------------------------

// Legacy implementation: manual group-reduction inner loops. This is the DEFAULT
// path in ormqr today and the only one geqrf uses.
template <typename Tag, typename T, int WG>
sycl::event larft_forward_columnwise_wg_legacy(Queue& q,
                                               T* t_data, int ld_t, int stride_t,
                                               const T* v_data, int ld_v, int stride_v,
                                               int m, int ib,
                                               const T* tau_data, int tau_stride,
                                               int tau_offset, int batch,
                                               const std::uint8_t* mask) {
    static_assert(WG > 0, "WG must be positive");

    return q->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<LarftKernelName<Tag, T, WG, false>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(WG)),
                              sycl::range<1>(static_cast<size_t>(WG))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch || (mask != nullptr && mask[b] == 0)) return;  // I1: before any barrier

                T* t_b = t_data + static_cast<std::ptrdiff_t>(b) * stride_t;
                const T* v_b = v_data + static_cast<std::ptrdiff_t>(b) * stride_v;
                const T* tau_b = tau_data + static_cast<std::ptrdiff_t>(b) * tau_stride + tau_offset;

                const sycl::group<1> g = it.get_group();
                const int lid = static_cast<int>(it.get_local_linear_id());

                if (lid == 0) {
                    for (int j = 0; j < ib; ++j) {
                        for (int i = 0; i < ib; ++i) {
                            t_b[i + j * ld_t] = T(0);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::global_space);

                for (int j = 0; j < ib; ++j) {
                    const T tauj = tau_b[j];
                    if (tauj == T(0)) {
                        if (lid == 0) {
                            t_b[j + j * ld_t] = T(0);
                        }
                        it.barrier(sycl::access::fence_space::global_space);
                        continue;
                    }

                    for (int col = 0; col < j; ++col) {
                        T partial = T(0);
                        for (int r = j + 1 + lid; r < m; r += WG) {
                            const T v_rc = v_b[r + col * ld_v];
                            const T v_rj = v_b[r + j * ld_v];
                            partial += batchlas::device::detail::conj(v_rc) * v_rj;
                        }
                        const T sum_r = batchlas::device::detail::reduce_sum_group(g, partial);
                        if (lid == 0) {
                            const T sum = batchlas::device::detail::conj(v_b[j + col * ld_v]) + sum_r;
                            t_b[col + j * ld_t] = -tauj * sum;
                        }
                        it.barrier(sycl::access::fence_space::global_space);
                    }

                    if (lid == 0) {
                        for (int row = 0; row < j; ++row) {
                            T acc = T(0);
                            for (int col = row; col < j; ++col) {
                                acc += t_b[row + col * ld_t] * t_b[col + j * ld_t];
                            }
                            t_b[row + j * ld_t] = acc;
                        }
                        t_b[j + j * ld_t] = tauj;
                    }
                    it.barrier(sycl::access::fence_space::global_space);
                }
            });
    });
}

// Device LA v2 (algo port of the legacy kernel): zlarft's gemv^H and trmv per column, two
// barriers in all. Y_j sits in T's strictly lower row j (scratch) so ONE barrier publishes every
// Y; trmv's row owner reads back only the row it wrote, so the recurrence needs none. Reads V's
// stored unit diagonal (pack_v writes it). Reached only with use_device = true.
template <typename Tag, typename T, int WG>
sycl::event larft_forward_columnwise_wg_device(Queue& q,
                                               T* t_data, int ld_t, int stride_t,
                                               const T* v_data, int ld_v, int stride_v,
                                               int m, int ib,
                                               const T* tau_data, int tau_stride,
                                               int tau_offset, int batch,
                                               const std::uint8_t* mask) {
    static_assert(WG % 32 == 0, "WG must be a whole number of 32-wide sub-groups");

    return q->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<LarftKernelName<Tag, T, WG, true>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(WG)),
                              sycl::range<1>(static_cast<size_t>(WG))),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch || (mask != nullptr && mask[b] == 0)) return;  // I1: before any barrier

                namespace v2 = batchlas::device::v2;
                const v2::WorkGroup<WG, 32> ex{it};
                const auto V = v2::mat(v_data + static_cast<std::ptrdiff_t>(b) * stride_v, m, ib, ld_v);
                const auto Tm = v2::mat(t_data + static_cast<std::ptrdiff_t>(b) * stride_t, ib, ib, ld_t);
                const auto tau = v2::vec(tau_data + static_cast<std::ptrdiff_t>(b) * tau_stride + tau_offset, ib);

                for (int j = 1; j < ib; ++j) {  // T(j, 0:j) := Y_j = -tau_j V(j:m, 0:j)^H V(j:m, j)
                    if (tau(j) == T(0)) continue;
                    v2::gemv<Transpose::ConjTrans>(ex, -tau(j), V.sub(j, 0, m - j, j),
                                                   v2::assume_published(V.col(j).tail(j)), v2::Zero{},
                                                   Tm.row(j).head(j));
                }
                ex.barrier(v2::Mem::Global);

                v2::copy(ex, tau, v2::vec(Tm.p, ib, ld_t + 1));  // T(j, j) := tau_j
                for (int j = 1; j < ib; ++j) {  // T(0:j, j) := T(0:j, 0:j) Y_j, or 0 when tau_j == 0
                    const auto t_col = Tm.col(j).head(j);
                    if (tau(j) == T(0)) v2::map(ex, t_col, [] { return T(0); });
                    else v2::trmv<Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(
                        ex, Tm.sub(0, 0, j, j), v2::assume_published(Tm.row(j).head(j)), t_col);
                }
                ex.barrier(v2::Mem::Global);

                v2::laset<Uplo::Lower>(ex, v2::Zero{}, Tm);  // clear the Y scratch: T is upper
            });
    });
}

}  // namespace detail

// Dispatcher over the work-group ladder. UseDevice is a TEMPLATE parameter, not a runtime
// bool, because a runtime bool device-links both implementations for every rung even
// where the caller passes a literal. evidence: docs/perf/qr.md#larft_wyhh-usedevice-is-a-template-parameter
template <typename Tag, typename T, bool UseDevice>
sycl::event larft_forward_columnwise_batched_t(Queue& q,
                                               T* t_data, int ld_t, int stride_t,
                                               const T* v_data, int ld_v, int stride_v,
                                               int m, int ib,
                                               const T* tau_data, int tau_stride,
                                               int tau_offset, int batch,
                                               const std::uint8_t* mask = nullptr) {
    const auto pick = [&]<int WG>(std::integral_constant<int, WG>) {
        // v2's gemv needs 32-wide sub-groups: a launch-time choice, never an in-kernel test.
        if constexpr (UseDevice) {
            if (q.device().supports_sub_group_size(32))
                return detail::larft_forward_columnwise_wg_device<Tag, T, WG>(
                    q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
                    tau_data, tau_stride, tau_offset, batch, mask);
        }
        return detail::larft_forward_columnwise_wg_legacy<Tag, T, WG>(
            q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
            tau_data, tau_stride, tau_offset, batch, mask);
    };

    if (ib <= 8 && m <= 64)    return pick(std::integral_constant<int, 32>{});
    if (ib <= 16 && m <= 128)  return pick(std::integral_constant<int, 64>{});
    if (ib <= 32 && m <= 256)  return pick(std::integral_constant<int, 128>{});
    return pick(std::integral_constant<int, 256>{});
}

// Runtime-selecting wrapper, ONLY for a caller whose choice is not a compile-time fact
// (today ormqr alone, via BATCHLAS_ORMQR_IMPL). A caller with a literal calls the _t form.
template <typename Tag, typename T>
sycl::event larft_forward_columnwise_batched(Queue& q,
                                             T* t_data, int ld_t, int stride_t,
                                             const T* v_data, int ld_v, int stride_v,
                                             int m, int ib,
                                             const T* tau_data, int tau_stride,
                                             int tau_offset, int batch,
                                             bool use_device,
                                             const std::uint8_t* mask = nullptr) {
    return use_device
               ? larft_forward_columnwise_batched_t<Tag, T, true>(
                     q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
                     tau_data, tau_stride, tau_offset, batch, mask)
               : larft_forward_columnwise_batched_t<Tag, T, false>(
                     q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
                     tau_data, tau_stride, tau_offset, batch, mask);
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

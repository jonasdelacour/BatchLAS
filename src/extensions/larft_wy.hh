#pragma once

#include <type_traits>

// The WY block-reflector machinery, SHARED.
//
// WHY THIS FILE EXISTS. `larft` (form the ib x ib triangular factor T of a block
// of Householder reflectors) and `pack_v` (materialise the unit-lower V panel
// from geqrf's packed output) existed in this tree in TWO private copies inside
// src/extensions/ormqr_blocked.cc, plus a THIRD in sytrd_sy2sb.cc:233 that its
// own loop no longer reaches. WP5 needs both for the blocked geqrf's trailing
// update, and the WP5 brief is explicit that the answer is to factor them out
// rather than to write a sixth private copy. This is that factoring: the bodies
// below are ormqr_blocked.cc's, moved verbatim, and ormqr_blocked.cc now calls
// them here.
//
// (sytrd_sy2sb.cc's third copy is left alone. It is unreachable from that file's
// own loop, which goes through ormqr, so deleting it is a separate change with a
// separate justification. It is named here so the next reader does not mistake
// it for a fourth reusable primitive.)
//
// KERNEL NAMES ARE TAGGED BY CALLER, and that is not decoration. These are
// inline function templates in a header, so two translation units that include
// it instantiate the same closure types; if those TUs sit in DIFFERENT
// device-code clusters (ormqr_blocked.cc is in EXTENSIONS_FACTORIZATION_SOURCES,
// geqrf_blocked.cc in EXTENSIONS_CTA_SOURCES -- see src/extensions/
// CMakeLists.txt:70-85) the same SYCL kernel name would be emitted into two
// device images of one shared library. The `Tag` parameter makes each caller's
// kernels distinct types, which removes the question rather than answering it.
//
// WHAT IS *NOT* SHARED. The `use_device` switch is a PARAMETER, not a getenv
// read: ormqr reads BATCHLAS_ORMQR_IMPL for itself and passes the answer in, and
// geqrf chooses independently. Putting the getenv here would silently tie a
// geqrf kernel selection to a variable named for ormqr.

#include <batchlas/blas/device.hh>
#include <batchlas/blas/matrix.hh>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <sycl/sycl.hpp>

#include <cstddef>

namespace batchlas::wy {

// Kernel name tags. `Tag` is supplied by the caller (an incomplete struct is
// enough); WG and Device make the four work-group ladders and the two
// implementations distinct.
template <typename Tag, typename T, int WG, bool Device> class LarftKernelName;
template <typename Tag, typename T> class PackVKernelName;

namespace detail {

template <typename U>
inline U conj_if_needed(const U& x, bool do_conj) {
    if (!do_conj) return x;
    if constexpr (batchlas::internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

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
                                               int tau_offset, int batch) {
    static_assert(WG > 0, "WG must be positive");

    auto reduce_sum = [](const sycl::group<1>& g, T x) {
        if constexpr (batchlas::internal::is_complex<T>::value) {
            using R = typename T::value_type;
            const R re = sycl::reduce_over_group(g, x.real(), sycl::plus<R>());
            const R im = sycl::reduce_over_group(g, x.imag(), sycl::plus<R>());
            return T(re, im);
        } else {
            return sycl::reduce_over_group(g, x, sycl::plus<T>());
        }
    };

    return q->submit([&](sycl::handler& h) {
        h.parallel_for<LarftKernelName<Tag, T, WG, false>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(WG)),
                              sycl::range<1>(static_cast<size_t>(WG))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                T* t_b = t_data + b * stride_t;
                const T* v_b = v_data + b * stride_v;
                const T* tau_b = tau_data + b * tau_stride + tau_offset;

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
                            partial += conj_if_needed(v_rc, /*do_conj=*/true) * v_rj;
                        }
                        const T sum_r = reduce_sum(g, partial);
                        if (lid == 0) {
                            const T sum =
                                conj_if_needed(v_b[j + col * ld_v], /*do_conj=*/true) + sum_r;
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

// Device-BLAS implementation: uses device::fill / gemv / scal / trmv. Reached
// only when the caller passes use_device = true (ormqr's BATCHLAS_ORMQR_IMPL).
template <typename Tag, typename T, int WG>
sycl::event larft_forward_columnwise_wg_device(Queue& q,
                                               T* t_data, int ld_t, int stride_t,
                                               const T* v_data, int ld_v, int stride_v,
                                               int m, int ib,
                                               const T* tau_data, int tau_stride,
                                               int tau_offset, int batch) {
    static_assert(WG > 0, "WG must be positive");

    return q->submit([&](sycl::handler& h) {
        h.parallel_for<LarftKernelName<Tag, T, WG, true>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(WG)),
                              sycl::range<1>(static_cast<size_t>(WG))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                T* t_b = t_data + b * stride_t;
                const T* v_b = v_data + b * stride_v;
                const T* tau_b = tau_data + b * tau_stride + tau_offset;

                const sycl::group<1> g = it.get_group();
                const int lid = static_cast<int>(it.get_local_linear_id());
                auto t_mat =
                    KernelMatrixView<T, MatrixFormat::Dense>(t_b, ib, ib, ld_t, ld_t * ib);
                auto v_mat = KernelMatrixView<T, MatrixFormat::Dense>(
                    const_cast<T*>(v_b), m, ib, ld_v, ld_v * ib);

                batchlas::device::fill(g, t_mat, T(0));
                sycl::group_barrier(g);

                for (int j = 0; j < ib; ++j) {
                    const T tauj = tau_b[j];
                    if (tauj == T(0)) {
                        if (lid == 0) {
                            t_b[j + j * ld_t] = T(0);
                        }
                        sycl::group_barrier(g);
                        continue;
                    }

                    if (j > 0) {
                        auto t_col = t_mat(Slice(0, j), j);
                        auto t_prev = t_mat(Slice(0, j), Slice(0, j));
                        auto v_prev = v_mat(Slice(j, m), Slice(0, j));
                        auto v_col = v_mat(Slice(j, m), j);

                        batchlas::device::gemv<Transpose::ConjTrans>(
                            g, v_prev, v_col, t_col, T(1), T(0), static_cast<T*>(nullptr));
                        sycl::group_barrier(g);
                        batchlas::device::scal(g, t_col, -tauj);
                        sycl::group_barrier(g);
                        batchlas::device::trmv<Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(
                            g, t_prev, t_col, t_col, T(1), T(0));
                        sycl::group_barrier(g);
                    }

                    if (lid == 0) {
                        t_b[j + j * ld_t] = tauj;
                    }
                    sycl::group_barrier(g);
                }
            });
    });
}

}  // namespace detail

// Dispatcher over the work-group ladder. The ladder and its thresholds are
// ormqr's, moved unchanged.
//
// UseDevice IS A TEMPLATE PARAMETER, NOT A RUNTIME BOOL, AND THAT IS A DEVICE
// LINK-TIME DECISION. As a runtime bool it instantiated BOTH implementations for
// every (Tag, T, WG) the ladder can reach. geqrf passes a literal `false`
// (geqrf_blocked.cc), so `larft_forward_columnwise_wg_device<GeqrfWyTag, ...>`
// was 32 entry functions -- 4 types x 4 work-group rungs x 2 (base and
// _with_offset) -- that were compiled, ptxas'd and device-linked into
// batchlas_extensions_cta, the slowest-linking library in the tree at ~125 s,
// and could never be launched. nsys confirmed it: no `(bool)1` variant appears
// in any WP5 run. They also included the highest-register kernel in the whole
// WP5 set (cdouble, 90 registers, 208 B stack frame), so they were not free to
// leave in. A caller that genuinely chooses at RUNTIME -- ormqr, via
// BATCHLAS_ORMQR_IMPL -- keeps both by calling the runtime wrapper below.
template <typename Tag, typename T, bool UseDevice>
sycl::event larft_forward_columnwise_batched_t(Queue& q,
                                               T* t_data, int ld_t, int stride_t,
                                               const T* v_data, int ld_v, int stride_v,
                                               int m, int ib,
                                               const T* tau_data, int tau_stride,
                                               int tau_offset, int batch) {
    const auto pick = [&]<int WG>(std::integral_constant<int, WG>) {
        if constexpr (UseDevice) {
            return detail::larft_forward_columnwise_wg_device<Tag, T, WG>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
                tau_data, tau_stride, tau_offset, batch);
        } else {
            return detail::larft_forward_columnwise_wg_legacy<Tag, T, WG>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
                tau_data, tau_stride, tau_offset, batch);
        }
    };

    if (ib <= 8 && m <= 64)    return pick(std::integral_constant<int, 32>{});
    if (ib <= 16 && m <= 128)  return pick(std::integral_constant<int, 64>{});
    if (ib <= 32 && m <= 256)  return pick(std::integral_constant<int, 128>{});
    return pick(std::integral_constant<int, 256>{});
}

// Runtime-selecting wrapper. ONLY for a caller whose choice is not a compile-time
// fact -- today that is ormqr alone (use_device_ormqr() reads BATCHLAS_ORMQR_IMPL).
// Calling this from a caller that passes a literal is what put 32 dead entry
// functions in the device link; call the _t form with an explicit `false` instead.
template <typename Tag, typename T>
sycl::event larft_forward_columnwise_batched(Queue& q,
                                             T* t_data, int ld_t, int stride_t,
                                             const T* v_data, int ld_v, int stride_v,
                                             int m, int ib,
                                             const T* tau_data, int tau_stride,
                                             int tau_offset, int batch,
                                             bool use_device) {
    return use_device
               ? larft_forward_columnwise_batched_t<Tag, T, true>(
                     q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
                     tau_data, tau_stride, tau_offset, batch)
               : larft_forward_columnwise_batched_t<Tag, T, false>(
                     q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib,
                     tau_data, tau_stride, tau_offset, batch);
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
                                 int i0, int ib, int nq) {
    const int m = nq - i0;
    const int ld_a = a.ld();
    const int stride_a = a.stride();
    const T* a_ptr = a.data_ptr();
    const int batch = a.batch_size();

    return q->submit([&](sycl::handler& h) {
        // DIM 2 IS THE ROW, NOT THE COLUMN. sycl::id<3> makes dim 2 the
        // fastest-varying index and both operands are COLUMN-MAJOR, so putting
        // the column there made a warp read a_ptr at ld_a*sizeof(T) apart and
        // write v_out at ld_v_out*sizeof(T) apart -- 32 sectors per warp instead
        // of 4, on both sides. Measured before the swap: 63.7 us median per
        // instance for a 17.3 MB job (float m=n=1024, batch=128, nb=32), 3.4x
        // the DRAM floor, and the amplification was MUTED only because an 8.7 MB
        // panel is L2-resident on a 72 MB L2 -- it degrades toward the full 8x at
        // larger m or batch. Same convention as src/matrix.cc:400.
        h.parallel_for<PackVKernelName<Tag, T>>(
            sycl::range<3>(static_cast<size_t>(batch), static_cast<size_t>(ib),
                           static_cast<size_t>(m)),
            [=](sycl::id<3> idx) {
                const int b = static_cast<int>(idx[0]);
                const int c = static_cast<int>(idx[1]);
                const int r = static_cast<int>(idx[2]);
                T val = T(0);
                if (r == c) {
                    val = T(1);
                } else if (r > c) {
                    val = a_ptr[b * stride_a + (i0 + r) + (i0 + c) * ld_a];
                }
                v_out[b * stride_v_out + r + c * ld_v_out] = val;
            });
    });
}

}  // namespace batchlas::wy

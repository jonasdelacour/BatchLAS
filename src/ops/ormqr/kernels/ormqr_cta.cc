// SPDX-License-Identifier: MIT


#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <batchlas/backend_config.h>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/ormqr.hh>
#include <src/util/template-instantiations.hh>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <src/sycl/local_mem.hh>
#include <src/util/batch_mask.hh>

#include <complex>
#include <numeric>
#include <array>
#include <type_traits>


namespace batchlas {

// Small-matrix CTA ORMQR/ORMQL/UNMQR/UNMQL: apply the implicit Q of a GEQRF/GEQLF
// factorization (A, TAU) to C. Very small sizes only (n <= 32), square C (SYEVD's
// tridiagonal eigenvector post-processing). Order/storage match LAPACK's ORMQx/UNMQx.

template <typename T, size_t P, bool QL, bool Left, bool TransposeOrConj, bool Masked>
class OrmQxCTAKernel;

namespace v2 = batchlas::device;

// Masked: the batch mask folds into `live`, a template so the unmasked instance is the kernel as
// it was (the runtime fold measured 1.4% on float n = 32 Left at batch 4096).
template <typename T, size_t P, bool QL, bool Left, bool TransposeOrConj, bool Masked>
inline void ormqx_cta_impl_m(Queue& ctx,
                             MatrixView<T, MatrixFormat::Dense>& a,
                             VectorView<T>& tau,
                             MatrixView<T, MatrixFormat::Dense>& c,
                             int32_t n,
                             int32_t k,
                             size_t cta_wg_size_multiplier,
                             const std::uint8_t* mask) {
    const auto batch_size = a.batch_size();

    if (n < 0 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n || c.rows() != n || c.cols() != n) {
        throw batchlas::invalid_argument("ormqx_cta_impl: currently requires square A and C with n <= P.");
    }
    if (k < 0 || k > n) {
        throw batchlas::invalid_argument("ormqx_cta_impl: invalid k.");
    }
    if (tau.size() < k) {
        throw batchlas::invalid_argument("ormqx_cta_impl: tau too small for k.");
    }
    if (tau.batch_size() != batch_size || c.batch_size() != batch_size) {
        throw batchlas::invalid_argument("ormqx_cta_impl: batch size mismatch.");
    }

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();
        auto TAU_view = tau;
        auto C_view = c.kernel_view();

        const auto dev = ctx->get_device();

        // CTA path assumes warp-sized sub-groups on NVIDIA.
        const int32_t sg_size = 32;

        const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), static_cast<int32_t>(sg_size));
        int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1), static_cast<int32_t>(cta_wg_size_multiplier));
        int32_t wg_size = base_wg_size * wg_size_multiplier;

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        if (wg_size > max_wg_size) {
            const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
            wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        // Clamp by local memory usage. Per problem we allocate:
        // - LEFT specialization: V_local: P
        // - non-LEFT: C_local: P*P, V_local: P
        // and probs_per_wg = wg_size / P.
        {
            const std::size_t local_mem_bytes = impl::local_mem_bytes(dev);
            const std::size_t elems_per_prob = Left ? static_cast<std::size_t>(P)
                                                     : (static_cast<std::size_t>(P) * static_cast<std::size_t>(P) + static_cast<std::size_t>(P));
            const std::size_t bytes_per_prob = elems_per_prob * sizeof(T);
            wg_size_multiplier = impl::cta_wg_multiplier("ormqr_cta", wg_size_multiplier,
                                                         base_wg_size / static_cast<int32_t>(P), bytes_per_prob,
                                                         local_mem_bytes);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
        const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;

        if constexpr (Left) {
            // LEFT specialization: lane->column and keep that column in registers.
            auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);

            // Coalescing: lane j owns column j, so consecutive lanes are `ld` apart and a scalar
            // per-row access issues one 32-sector request (8x the needed L1<->L2 traffic).
            // 16 bytes per lane per access cuts the request count by VW.
            constexpr int32_t VW = static_cast<int32_t>(16 / sizeof(T));
            const bool vec_ok =
                !detail::is_complex<T>::value && VW > 1 && (n % VW) == 0 &&
                ((static_cast<std::size_t>(c.ld()) * sizeof(T)) % 16u) == 0 &&
                ((static_cast<std::size_t>(c.stride()) * sizeof(T)) % 16u) == 0 &&
                (reinterpret_cast<std::uintptr_t>(c.data_ptr()) % 16u) == 0;

            impl::parallel_for<OrmQxCTAKernel<T, P, QL, Left, TransposeOrConj, Masked>>(cgh,
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                const auto sg = it.get_sub_group();
                const auto part = make_partition<P>(sg);

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                // NO EARLY RETURN in a packed kernel: a dead partition (tail of the
                // batch, or an inactive item: I2) applies tau = 0 reflectors to nothing and
                // stores nothing.
                // if constexpr, never a call taking mask: the unmasked kernel must not capture it.
                bool live = prob_id < static_cast<int32_t>(batch_size);
                if constexpr (Masked) live = live && mask[prob_id] != 0;
                const int32_t pid = live ? prob_id : 0;

                // First uses in the inline body's order (views, n, vec_ok, k, V_local): [=]
                // orders the kernel arguments by them.
                auto A_prob = A_view.batch_item(pid);
                auto TAU_prob = TAU_view.batch_item(pid);
                auto C_prob = C_view.batch_item(pid);
                const int32_t order = n;
                const bool vec = vec_ok;
                const int32_t nref = k;
                const int32_t base_v = part_id * static_cast<int32_t>(P);

                // The body is device::ormqr's Left form (lapack/ormqr.hh), on the hand loop's
                // arithmetic (strict port).
                v2::ormqr_detail::orm2x_left<static_cast<int>(P), QL, TransposeOrConj, v2::arith::Native>(
                    v2::wrap(part), A_prob, TAU_prob, C_prob, V_local, base_v, order, nref, live, vec);
            });
        } else {
            // Per problem: the P x P tile of C (column-major) and v. Lane r owns row r: no work vector.
            auto C_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P * P), cgh);
            auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);

            impl::parallel_for<OrmQxCTAKernel<T, P, QL, Left, TransposeOrConj, Masked>>(cgh,
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                    const auto sg = it.get_sub_group();
                    const auto part = make_partition<P>(sg);

                    const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                    const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                    const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                    const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                    const int32_t prob_id = wg_id * probs_per_wg + part_id;
                    // NO EARLY RETURN, as in the LEFT kernel: a dead partition (a tail or an
                    // inactive item) carries a zero tile and tau = 0, and stores nothing.
                    bool live = prob_id < static_cast<int32_t>(batch_size);
                    if constexpr (Masked) live = live && mask[prob_id] != 0;
                    const int32_t pid = live ? prob_id : 0;
                    // First uses in the inline body's order (n, views, C_local, k, V_local).
                    const int32_t order = n;

                    auto A_prob = A_view.batch_item(pid);
                    auto TAU_prob = TAU_view.batch_item(pid);
                    auto C_prob = C_view.batch_item(pid);
                    const auto& c_tile = C_local;
                    const int32_t nref = k;

                    const int32_t base_c = part_id * static_cast<int32_t>(P) * static_cast<int32_t>(P);
                    const int32_t base_v = part_id * static_cast<int32_t>(P);

                    // The body is device::ormqr's Right form (lapack/ormqr.hh).
                    v2::ormqr_detail::orm2x_right<static_cast<int>(P), QL, TransposeOrConj>(
                        v2::wrap(part), A_prob, TAU_prob, C_prob, c_tile, base_c, V_local, base_v, order, nref, live);
                });
        }
    });
}

template <typename T, size_t P, bool QL, bool Left, bool TransposeOrConj>
inline void ormqx_cta_impl(Queue& ctx, MatrixView<T, MatrixFormat::Dense>& a, VectorView<T>& tau,
                           MatrixView<T, MatrixFormat::Dense>& c, int32_t n, int32_t k,
                           size_t cta_wg_size_multiplier, const std::uint8_t* mask) {
    if (mask != nullptr)
        return ormqx_cta_impl_m<T, P, QL, Left, TransposeOrConj, true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
    ormqx_cta_impl_m<T, P, QL, Left, TransposeOrConj, false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
}

// Public entrypoint.
//
// CTA-only small-n routine (n <= 32) to apply Q from a QR/QL factorization.
// Semantics match LAPACK ORMQx/UNMQx via (factorization, side, trans).

template <Backend B, typename T>
Event ormqx_cta(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& a_in,
               const VectorView<T>& tau_in,
               const MatrixView<T, MatrixFormat::Dense>& c_in,
               Uplo factorization,
               Side side,
               Transpose trans,
               int32_t k,
               const Span<std::byte>& ws,
               size_t cta_wg_size_multiplier) {
    // The packed kernels fold the mask into `live` (I2): an inactive item's C is untouched.
    const std::uint8_t* const mask =
        ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("ormqx_cta", a_in, tau_in, c_in));
    (void)ws;

    const bool left = (side == Side::Left);
    const bool transpose_or_conj = (trans != Transpose::NoTrans);
    const bool ql = (factorization == Uplo::Lower);

    if (a_in.rows() != a_in.cols() || c_in.rows() != c_in.cols() || a_in.rows() != c_in.rows()) {
        throw batchlas::invalid_argument("ormqx_cta: currently requires square A and C of the same order.");
    }

    const int32_t n = static_cast<int32_t>(a_in.rows());
    if (n < 0 || n > 32) {
        throw batchlas::invalid_argument("ormqx_cta: currently supports 0 <= n <= 32.");
    }
    if (k < 0 || k > n) {
        throw batchlas::invalid_argument("ormqx_cta: invalid k.");
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto& tau = const_cast<VectorView<T>&>(tau_in);
    auto& c = const_cast<MatrixView<T, MatrixFormat::Dense>&>(c_in);

    // CTA backend: requires subgroup size 32 on NVIDIA-like devices.
    if (!ctx.device().supports_sub_group_size(32)) {
        throw batchlas::unsupported("ormqx_cta: device does not support subgroup size 32 required for CTA kernels.");
    }

    auto launch = [&](auto P_tag) {
        constexpr int32_t P = decltype(P_tag)::value;
        if (ql) {
            if (left) {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/true, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/true, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                }
            } else {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/false, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/false, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                }
            }
        } else {
            if (left) {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/true, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/true, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                }
            } else {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/false, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/false, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier, mask);
                }
            }
        }
    };

    if (n <= 4) {
        launch(std::integral_constant<int32_t, 4>{});
    } else if (n <= 8) {
        launch(std::integral_constant<int32_t, 8>{});
    } else if (n <= 16) {
        launch(std::integral_constant<int32_t, 16>{});
    } else {
        launch(std::integral_constant<int32_t, 32>{});
    }

    return ctx.get_event();
}

#define ORMQX_CTA_INSTANTIATE(back, fp) \
    template Event ormqx_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<BATCHLAS_UNPAREN fp>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(ORMQX_CTA_INSTANTIATE)

#undef ORMQX_CTA_INSTANTIATE

} // namespace batchlas
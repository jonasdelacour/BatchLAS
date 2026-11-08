// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <batchlas/backend_config.h>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/sycl/local_mem.hh>
#include <src/sort.hh>
#include <complex>
#include <numeric>
#include <array>
#include <batchlas/device/lapack/sytrd.hh>


namespace batchlas {

    template <typename T, size_t P, bool Upper, bool Masked>
    class Sytd2CTAKernel;

    // Masked: the batch mask folds into `live`, a template so the unmasked instance is the kernel
    // as it was (the runtime fold measured 1.1% on float n = 16 at batch 4096).
    template <typename T, size_t P, bool Upper, bool Masked>
    inline void sytd2_cta_impl_m(Queue& ctx,
                                 MatrixView<T, MatrixFormat::Dense>& a,
                                 VectorView<T>& d,
                                 VectorView<T>& e,
                                 VectorView<T>& tau,
                                 int32_t n,
                                 size_t cta_wg_size_multiplier,
                                 const std::uint8_t* mask) {
        const auto batch_size = a.batch_size();
        if (n < 1 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n) {
            throw batchlas::invalid_argument("sytd2_cta_impl: invalid n or matrix sizes for CTA partition.");
        }
        if (d.size() != n || e.size() != (n - 1) || tau.size() != (n - 1)) {
            throw batchlas::invalid_argument("sytd2_cta_impl: invalid d/e/tau sizes.");
        }
        if (d.batch_size() != batch_size || e.batch_size() != batch_size || tau.batch_size() != batch_size) {
            throw batchlas::invalid_argument("sytd2_cta_impl: batch size mismatch.");
        }

        ctx->submit([&](sycl::handler& cgh) {
            auto A_view = a.kernel_view();
            auto D_view = d;
            auto E_view = e;
            auto TAU_view = tau;

            const auto dev = ctx->get_device();
            (void)dev;

            // CTA path assumes warp-sized sub-groups on NVIDIA.
            const int32_t sg_size = 32;

            // Baseline work-group size is LCM(P, sg_size), so we can form fixed-size partitions of size P.
            const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), static_cast<int32_t>(sg_size));
            int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1), static_cast<int32_t>(cta_wg_size_multiplier));
            int32_t wg_size = base_wg_size * wg_size_multiplier;

            const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
            if (wg_size > max_wg_size) {
                const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
                wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
                wg_size = base_wg_size * wg_size_multiplier;
            }

            // Clamp by local memory usage: per-problem local storage is
            //   A_local: P*P, V_local: P, W_local: P
            // and probs_per_wg = wg_size / P.
            {
                const std::size_t local_mem_bytes = impl::local_mem_bytes(dev);
                const std::size_t elems_per_prob = static_cast<std::size_t>(P) * static_cast<std::size_t>(P)
                                                 + static_cast<std::size_t>(2) * static_cast<std::size_t>(P);
                const std::size_t bytes_per_prob = elems_per_prob * sizeof(T);
                wg_size_multiplier = impl::cta_wg_multiplier("sytrd_cta", wg_size_multiplier,
                                                             base_wg_size / static_cast<int32_t>(P), bytes_per_prob,
                                                             local_mem_bytes);
                wg_size = base_wg_size * wg_size_multiplier;
            }

            const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
            const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
            const int32_t global_size = num_wg * wg_size;

            // Local storage per problem:
            // - Full P x P tile for A (column-major)
            // - Two length-P vectors (v and w)
            auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P * P), cgh);
            auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
            auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);

            impl::parallel_for<Sytd2CTAKernel<T, P, Upper, Masked>>(cgh,
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                    const auto wg = it.get_group();
                    const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());

                    const auto sg = it.get_sub_group();
                    const auto part = make_partition<P>(sg);

                    // Make part_id unique across sub-groups in the work-group.
                    const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                    const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                    const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                    const int32_t prob_id = wg_id * probs_per_wg + part_id;
                    if (prob_id >= static_cast<int32_t>(batch_size)) return;
                    // I2: an inactive item of the batch mask reduces a zero tile and stores nothing.
                    // const, and no capture of the mask when unmasked: a mutable flag the nested lambdas
                    // take by reference lands in local memory.
                    const bool live = [&]() BATCHLAS_DLA_BODY {
                        if constexpr (Masked) return mask[prob_id] != 0;
                        else return true;
                    }();

                    namespace v2 = batchlas::device;
                    // D, E, TAU, the tile, then A_view: the hand kernel's capture order, so its parameter layout.
                    auto D_prob = D_view.batch_item(prob_id);
                    auto E_prob = E_view.batch_item(prob_id);
                    auto TAU_prob = TAU_view.batch_item(prob_id);
                    const int32_t base_a = part_id * static_cast<int32_t>(P) * static_cast<int32_t>(P);
                    const int32_t base_v = part_id * static_cast<int32_t>(P);
                    const int32_t base_w = part_id * static_cast<int32_t>(P);
                    const auto& A_tile = A_local;
                    // v1's address expression; extents from the kernel's n, so every guard tests one value.
                    const auto A = v2::item(A_view, prob_id, n, n);

                    // Thin launcher: the body of device::sytrd<UL, tier::Cta<P>> on the accessors (the
                    // named triangle only, defect 14; an inactive item reduces a zero tile, stores nothing).
                    v2::sytrd_detail::sytrd_cta_run<Upper ? Uplo::Upper : Uplo::Lower, T>(
                        part, A_tile, base_a, V_local, base_v, W_local, base_w, A, D_prob, E_prob, TAU_prob, n, live);
                });
        });
    }

    template <typename T, size_t P, bool Upper>
    inline void sytd2_cta_impl(Queue& ctx, MatrixView<T, MatrixFormat::Dense>& a, VectorView<T>& d, VectorView<T>& e,
                               VectorView<T>& tau, int32_t n, size_t cta_wg_size_multiplier, const std::uint8_t* mask) {
        if (mask != nullptr) return sytd2_cta_impl_m<T, P, Upper, true>(ctx, a, d, e, tau, n, cta_wg_size_multiplier, mask);
        sytd2_cta_impl_m<T, P, Upper, false>(ctx, a, d, e, tau, n, cta_wg_size_multiplier, mask);
    }

    // Public wrapper: pick a compile-time P in {4,8,16,32} and choose triangle at runtime.
    // This is unblocked (SYTD2-style), so it is meant for very small n.
    template <Backend B, typename T>
    Event sytrd_cta(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<T>& d_out,
                    const VectorView<T>& e_out,
                    const VectorView<T>& tau_out,
                    Uplo uplo,
                    const Span<std::byte>& ws,
                    size_t cta_wg_size_multiplier) {
        (void)ws;
        const std::uint8_t* mask =
            batchlas::detail::mask_ptr(batchlas::detail::call_mask("sytrd_cta", a_in, d_out, e_out, tau_out));
        if (a_in.rows() != a_in.cols()) {
            throw batchlas::invalid_argument("sytrd_cta: A must be square.");
        }

        const int64_t n64 = a_in.rows();
        const int64_t batch_size = a_in.batch_size();
        if (batch_size != d_out.batch_size() || batch_size != e_out.batch_size() || batch_size != tau_out.batch_size()) {
            throw batchlas::invalid_argument("sytrd_cta: batch size mismatch.");
        }

        const int32_t n = static_cast<int32_t>(n64);
        if (n < 1 || n > 32) {
            throw batchlas::invalid_argument("sytrd_cta currently supports 1 <= n <= 32.");
        }

        // Make mutable views (A is overwritten, D/E/TAU are outputs).
        auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
        auto& d = const_cast<VectorView<T>&>(d_out);
        auto& e = const_cast<VectorView<T>&>(e_out);
        auto& tau = const_cast<VectorView<T>&>(tau_out);

        // CTA backend: requires subgroup size 32 on NVIDIA-like devices.
        if (!ctx.device().supports_sub_group_size(32)) {
            throw batchlas::unsupported("sytrd_cta: device does not support subgroup size 32 required for CTA kernels.");
        }

        auto launch = [&](auto P_tag, bool upper) {
            constexpr int32_t P = decltype(P_tag)::value;
            if (upper) {
                sytd2_cta_impl<T, P, true>(ctx, a, d, e, tau, n, cta_wg_size_multiplier, mask);
            } else {
                sytd2_cta_impl<T, P, false>(ctx, a, d, e, tau, n, cta_wg_size_multiplier, mask);
            }
        };

        const bool upper = (uplo == Uplo::Upper);

        if (n <= 4) {
            launch(std::integral_constant<int32_t, 4>{}, upper);
        } else if (n <= 8) {
            launch(std::integral_constant<int32_t, 8>{}, upper);
        } else if (n <= 16) {
            launch(std::integral_constant<int32_t, 16>{}, upper);
        } else {
            launch(std::integral_constant<int32_t, 32>{}, upper);
        }

        return ctx.get_event();
    }

#define SYTRD_CTA_INSTANTIATE(back, fp) \
    template Event sytrd_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, Uplo, const Span<std::byte>&, size_t);

    BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYTRD_CTA_INSTANTIATE)

#undef SYTRD_CTA_INSTANTIATE

}


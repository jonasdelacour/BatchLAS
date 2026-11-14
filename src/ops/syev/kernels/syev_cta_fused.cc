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
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/kernels/common/info_span.hh>
#include <src/util/template-instantiations.hh>

#include <batchlas/device/lapack/syev.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <src/sycl/local_mem.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device.hh>

#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>


namespace batchlas {


// Kernel name tag. Must live outside the anonymous namespace so it does not
// depend on internal-linkage entities.
template <typename T, size_t P, bool ComputeVectors>
class SyevCtaFusedKernel;

// Multiplier used when the caller passes 0 (the default): two warps per work-group
// clear the one-warp per-SM block limit. It pays for real float at P == 8 and 16.
// P == 32 loses on graded input, complex float at P == 8 loses about 1% through
// syev, and double gains nothing. Tuned on sm_89 only.
// evidence: docs/perf/steqr.md#work-group-multiplier
template <typename T, size_t P, bool ComputeVectors>
inline constexpr int32_t kSyevCtaFusedAutoWgMultiplier =
    (std::is_same_v<T, float> && (P == 8 || P == 16)) ? 2 : 1;

// ---------------------------------------------------------------------------
// Monolithic (fused) CTA symmetric/Hermitian eigensolver: one problem stays resident
// in one partition from load to store (sytrd -> steqr -> back-transform), so no
// intermediate touches global memory. The body is device::syev<J, UL, tier::Cta<P>>'s
// (device/lapack/syev.hh detail::syev_cta_run), its stages the SAME code as the
// standalone kernels (lapack/sytrd.hh, lapack/steqr.hh): keep it that way, so a
// head-to-head benchmark measures fusion and nothing else. Real input seeds the
// sweeps with an in-place Q_house (DSYEV's route); Hermitian input applies the
// reflectors afterwards. The reduction always runs Uplo::Upper.
// evidence: docs/perf/syev.md#syev-the-fused-cta-kernel-design
// ---------------------------------------------------------------------------

template <typename T, size_t P, bool ComputeVectors>
inline void syev_cta_fused_impl(Queue& ctx,
                                MatrixView<T, MatrixFormat::Dense>& a,
                                typename base_type<T>::type* w_ptr,
                                int32_t n,
                                bool upper,
                                const SteqrParams<T>& params,
                                size_t cta_wg_size_multiplier,
                                int32_t* info,
                                const std::uint8_t* mask) {
    using Real = typename base_type<T>::type;

    const auto batch_size = a.batch_size();

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();

        const auto dev = ctx->get_device();

        // CTA path assumes warp-sized sub-groups on NVIDIA.
        const int32_t sg_size = 32;

        // The tile layout of the one body (device/lapack/syev.hh cta_layout: real vectors on
        // ONE P+1-padded tile, Hermitian vectors with a separate real accumulator).
        using Lay = device::detail::cta_layout<ComputeVectors, static_cast<int>(P), T>;
        constexpr int32_t LDQ = Lay::ldq;
        constexpr int32_t LDA = Lay::lda;
        constexpr bool kSeparateQTile = Lay::separate_q;
        constexpr std::size_t kATileElems = static_cast<std::size_t>(LDA) * P;
        constexpr std::size_t kQTileElems = static_cast<std::size_t>(LDQ) * P;

        const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), sg_size);
        // 0 asks for the tuned value; any explicit value is taken as given.
        int32_t wg_size_multiplier = cta_wg_size_multiplier == 0
                                         ? kSyevCtaFusedAutoWgMultiplier<T, P, ComputeVectors>
                                         : std::max<int32_t>(int32_t(1),
                                                             static_cast<int32_t>(cta_wg_size_multiplier));
        int32_t wg_size = base_wg_size * wg_size_multiplier;

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        if (wg_size > max_wg_size) {
            const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
            wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        // Clamp by local memory (real path: one tile + two length-P vectors, as sytrd_cta).
        {
            const std::size_t bytes_per_prob = (kATileElems + 2 * static_cast<std::size_t>(P)) * sizeof(T)
                                             + (kSeparateQTile ? kQTileElems * sizeof(Real) : 0);
            wg_size_multiplier = impl::cta_wg_multiplier("syev_cta_fused", wg_size_multiplier,
                                                         base_wg_size / static_cast<int32_t>(P), bytes_per_prob,
                                                         impl::local_mem_bytes(dev));
            wg_size = base_wg_size * wg_size_multiplier;
        }

        const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
        const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;

        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * kATileElems), cgh);
        auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
        auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
        auto Q_local = sycl::local_accessor<Real, 1>(
            sycl::range<1>(kSeparateQTile ? (probs_per_wg * kQTileElems) : 1), cgh);

        const int32_t nn = n;
        const int32_t nb = static_cast<int32_t>(batch_size);
        const bool is_upper = upper;
        const int32_t max_sweeps = std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.max_sweeps));
        const SteqrShiftStrategy shift_strategy = params.cta_shift_strategy;
        const bool do_sort = params.sort;
        const bool ascending = (params.sort_order == SortOrder::Ascending);

        Real* W = w_ptr;
        // A local so `[=]` copies a pointer; nullptr (status not requested) makes info_store a no-op.
        int32_t* const info_dev = info;

        impl::parallel_for<SyevCtaFusedKernel<T, P, ComputeVectors>>(cgh,
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());

                const auto sg = it.get_sub_group();
                const auto part = make_partition<P>(sg);
                // The solve takes the full-sub-group partition: every lane of the
                // warp reaches it (dead chunks are clamped below, not returned), so
                // it may run maskless where the chunks share one instruction stream.
                const auto solve_part = make_partition<P, false>(sg);

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                // Clamp, do not return: a dead chunk, or an inactive item of the batch mask, runs every
                // stage on a zero matrix and writes nothing.
                const bool live = prob_id < nb && (mask == nullptr || mask[prob_id] != 0);
                auto A_prob = A_view.batch_item(live ? prob_id : 0);

                const int32_t base_a = part_id * static_cast<int32_t>(kATileElems);
                const int32_t base_v = part_id * static_cast<int32_t>(P);
                const int32_t base_w = part_id * static_cast<int32_t>(P);
                const int32_t base_q = kSeparateQTile ? (part_id * static_cast<int32_t>(kQTileElems)) : 0;

                const device::detail::cta_opts o{max_sweeps, shift_strategy, do_sort, ascending};
                // The one body (device::syev<J, UL, tier::Cta<P>>'s too): load, sytd2, sweeps,
                // back-transform; the stores skip a dead chunk.
                device::detail::syev_cta_run<T, P, ComputeVectors, LDA, LDQ>(
                    part, solve_part, A_local, base_a, V_local, base_v, W_local, base_w, Q_local, base_q, A_prob,
                    device::mat(A_prob.data(), nn, nn, A_prob.ld()), W + static_cast<int64_t>(prob_id) * nn, nn, is_upper, live, o,
                    [&](bool failed) BATCHLAS_DEVICE_BODY {
                        // A STORE, not a raise: the kernel is the item's single writer, so it
                        // needs no separate clear (a second submission, racy out of order).
                        if (live && lane == 0) detail::info_store(info_dev, prob_id, failed ? 1 : 0);
                    });
            });
    });
}

template <Backend B, typename T>
Event syev_cta_fused(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                     Span<typename base_type<T>::type> eigenvalues,
                     JobType jobz,
                     Uplo uplo,
                     const Span<std::byte>& ws,
                     SteqrParams<T> steqr_params,
                     size_t cta_wg_size_multiplier,
                     Span<int32_t> info) {
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("syev_cta_fused", a_in));
    (void)ws;

    if (a_in.rows() != a_in.cols()) {
        throw batchlas::invalid_argument("syev_cta_fused: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw batchlas::invalid_argument("syev_cta_fused: invalid JobType.");
    }

    const int64_t n64 = a_in.rows();
    const int64_t batch64 = a_in.batch_size();

    if (n64 < 1 || n64 > 32) {
        throw batchlas::invalid_argument("syev_cta_fused currently supports 1 <= n <= 32.");
    }
    if (eigenvalues.size() < static_cast<std::size_t>(n64) * static_cast<std::size_t>(batch64)) {
        throw batchlas::invalid_argument("syev_cta_fused: eigenvalues span too small for n*batch.");
    }

    if (!ctx.device().supports_sub_group_size(32)) {
        throw batchlas::unsupported("syev_cta_fused: device does not support subgroup size 32 required for CTA kernels.");
    }

    // syev_cta's robustness bump, unless the caller tuned the params.
    {
        const SteqrParams<T> defaults{};
        if (steqr_params.max_sweeps == defaults.max_sweeps &&
            steqr_params.cta_shift_strategy == defaults.cta_shift_strategy) {
            steqr_params.max_sweeps = 400;
            steqr_params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        }
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto* w_ptr = eigenvalues.data();

    const int32_t n = static_cast<int32_t>(n64);
    const bool upper = (uplo == Uplo::Upper);
    const bool vectors = (jobz == JobType::EigenVectors);

    // No clear: the kernel STORES every item's status. `info` is caller USM, so
    // syev_cta_fused_buffer_size still returns 0.
    int32_t* info_ptr = detail::info_ptr(info, batch64);

    auto launch = [&](auto P_tag) {
        constexpr size_t P = decltype(P_tag)::value;
        if (vectors) {
            syev_cta_fused_impl<T, P, true>(ctx, a, w_ptr, n, upper, steqr_params, cta_wg_size_multiplier, info_ptr, mask);
        } else {
            syev_cta_fused_impl<T, P, false>(ctx, a, w_ptr, n, upper, steqr_params, cta_wg_size_multiplier, info_ptr, mask);
        }
    };

    if (n <= 4) {
        launch(std::integral_constant<size_t, 4>{});
    } else if (n <= 8) {
        launch(std::integral_constant<size_t, 8>{});
    } else if (n <= 16) {
        launch(std::integral_constant<size_t, 16>{});
    } else {
        launch(std::integral_constant<size_t, 32>{});
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_cta_fused_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                  JobType jobz,
                                  SteqrParams<T> steqr_params) {
    (void)ctx;
    (void)jobz;
    (void)steqr_params;

    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument("syev_cta_fused_buffer_size: A must be square.");
    }
    if (a.rows() < 1 || a.rows() > 32) {
        throw batchlas::invalid_argument("syev_cta_fused_buffer_size currently supports 1 <= n <= 32.");
    }

    // Nothing is spilled to global memory: the whole solve is partition-resident.
    return 0;
}

#define SYEV_CTA_FUSED_INSTANTIATE(back, fp) \
    template Event syev_cta_fused<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                             Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, JobType, Uplo, \
                                                             const Span<std::byte>&, SteqrParams<BATCHLAS_UNPAREN fp>, size_t, Span<int32_t>); \
    template size_t syev_cta_fused_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                                          JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

    BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_CTA_FUSED_INSTANTIATE)

#undef SYEV_CTA_FUSED_INSTANTIATE

} // namespace batchlas

// SPDX-License-Identifier: MIT
#include <batchlas/blas/device/detail/group_blas_common.hh>
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/dla_v2_jacobi.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <batchlas/backend_config.h>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/kernels/common/info_span.hh>
#include <src/util/template-instantiations.hh>
#include <src/sycl/local_mem.hh>
#include <src/util/batch_mask.hh>
#include <complex>
#include <limits>
#include <numeric>

namespace batchlas {

// Kernel name tag. Must live outside the anonymous namespace below so it does
// not depend on internal-linkage entities.
template <typename T, size_t P, bool ComputeVectors, bool Upper>
class SyevJacobiCTAKernel;

// Tier-A Jacobi eigensolver: partition-resident cyclic two-sided Jacobi. One
// SubGroupPartition<P> owns one problem; A and (optionally) Z stay in local memory
// for the whole solve. Accuracy-oriented: with the RELATIVE off-diagonal threshold
// below, graded input keeps small relative eigenvalue error (proved for SPD input).
// evidence: docs/perf/syev.md#syev-the-jacobi-kernel-design-and-its-accuracy-argument
// evidence: docs/perf/syev.md#syev-the-2026-08-03-small-n-bake-off

namespace {

namespace v2 = ::batchlas::device::v2;

template <typename T, size_t P, bool ComputeVectors, bool Upper>
inline void syev_jacobi_cta_impl(Queue& ctx,
                                 MatrixView<T, MatrixFormat::Dense>& a,
                                 typename base_type<T>::type* w_ptr,
                                 int32_t n,
                                 JacobiParams<T> params,
                                 int32_t* info,
                                 const std::uint8_t* mask) {
    using Real = typename base_type<T>::type;

    const auto batch_size = a.batch_size();
    if (n < 1 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n) {
        throw batchlas::invalid_argument("syev_jacobi_cta_impl: invalid n or matrix sizes for CTA partition.");
    }

    // Pivot index space is padded to an even size so the round-robin schedule is
    // well defined; the (single) padded index is simply never paired with a real
    // one because pairs touching index >= n are skipped.
    const int32_t m = (n % 2 == 0) ? n : (n + 1);

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();

        const auto dev = ctx->get_device();

        // CTA path assumes warp-sized sub-groups on NVIDIA.
        const int32_t sg_size = 32;

        // LD padded to P+1: the row-update phase has lane == column, and LD == 32 puts
        // every lane in one bank (32-way serialization).
        constexpr int32_t LD = static_cast<int32_t>(P) + 1;
        constexpr std::size_t kTileElems = static_cast<std::size_t>(LD) * P;

        const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), sg_size);
        int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1),
                                                       static_cast<int32_t>(params.cta_wg_size_multiplier));
        int32_t wg_size = base_wg_size * wg_size_multiplier;

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        if (wg_size > max_wg_size) {
            const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
            wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        // Clamp by local memory: per problem we hold A, optionally Z, and the
        // per-round rotation coefficients. The pivot-pair table is shared by the
        // whole work-group, so it is accounted for separately below.
        constexpr std::size_t kRotSlots = (P / 2 > 0) ? (P / 2) : 1;
        constexpr std::size_t kPairSlots = (P - 1) * kRotSlots;
        constexpr std::size_t kPairTabBytes = kPairSlots * sizeof(int16_t);
        constexpr bool kNeedPhase = detail::is_complex<T>::value;
        {
            const std::size_t local_mem_bytes = impl::local_mem_bytes(dev);
            const std::size_t avail = (local_mem_bytes > kPairTabBytes) ? (local_mem_bytes - kPairTabBytes) : 1;
            const std::size_t z_elems = ComputeVectors ? kTileElems : 0;
            const std::size_t bytes_per_prob = (kTileElems + z_elems) * sizeof(T)
                                             + (kNeedPhase ? kRotSlots * sizeof(T) : 0)
                                             + 2 * kRotSlots * sizeof(Real);
            wg_size_multiplier = impl::cta_wg_multiplier("syev_jacobi_cta", wg_size_multiplier,
                                                         base_wg_size / static_cast<int32_t>(P), bytes_per_prob,
                                                         avail);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
        const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;
        const int32_t wg_sz = wg_size;

        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * kTileElems), cgh);
        auto Z_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(ComputeVectors ? (probs_per_wg * kTileElems) : 1), cgh);
        // (c, s) packed into one vector: one LDS load instead of two in the update loops.
        auto Rcs_local = sycl::local_accessor<sycl::vec<Real, 2>, 1>(
            sycl::range<1>(probs_per_wg * kRotSlots), cgh);
        // The diagonal phase is identically 1 for real types, so it is neither
        // stored nor loaded there.
        auto Rd_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(kNeedPhase ? (probs_per_wg * kRotSlots) : 1), cgh);
        // Round-robin pivot pairs, precomputed once per work-group and shared by every
        // problem in it, both indices packed into one 16-bit slot (inline modulos
        // dominated the inner loops).
        auto Pair_local = sycl::local_accessor<int16_t, 1>(sycl::range<1>(kPairSlots), cgh);

        const int32_t nn = n;
        const int32_t mm = m;
        const int32_t nb = static_cast<int32_t>(batch_size);
        const int32_t max_sweeps = std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.max_sweeps));
        const bool do_sort = params.sort;
        const bool ascending = (params.sort_order == SortOrder::Ascending);

        // RELATIVE off-diagonal threshold (an absolute test forfeits the relative accuracy);
        // tiny and tau_big as documented on v2::JacobiThreshold.
        const auto thr = v2::JacobiThreshold<Real>::relative(params.tol_multiplier, nn);

        Real* W = w_ptr;
        // A local of the submit lambda for the same reason as W, so the kernel's
        // `[=]` copies a pointer. nullptr when status was not requested, which
        // makes info_store below a no-op.
        int32_t* const info_dev = info;

        impl::parallel_for<SyevJacobiCTAKernel<T, P, ComputeVectors, Upper>>(cgh,
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) {
                constexpr int kP = static_cast<int>(P);
                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t local_id = static_cast<int32_t>(it.get_local_linear_id());

                const int32_t pairs_per_round = mm / 2;
                const int32_t rounds = mm - 1;

                // Build the shared pivot-pair table; every work-item reaches its barrier.
                for (int32_t idx = local_id; idx < rounds * pairs_per_round; idx += wg_sz) {
                    const int32_t t = idx / pairs_per_round;
                    Pair_local[idx] = v2::pack_pair(v2::round_robin_pair(mm, t, idx - t * pairs_per_round));
                }
                sycl::group_barrier(wg);

                const auto sg = it.get_sub_group();
                const v2::SgpPartition<kP> ex{sg};
                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(ex.part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(ex.part.get_group_linear_id());

                const int32_t lane = ex.lid();
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                // NO EARLY RETURN: a dead partition, or an inactive item of the batch mask, sweeps a
                // zero tile and stores nothing. The mask is read after the pair-table barrier.
                const bool live = prob_id < nb && (mask == nullptr || mask[prob_id] != 0);
                const bool pair_lane = (lane < pairs_per_round);
                const bool row_lane = (lane < nn);

                auto A_prob = A_view.batch_item(live ? prob_id : 0);
                // Tiles and rotation slots: v2 operands over the kernel's accessors.
                using Tile = v2::Mat<T, v2::Ext<kP>, v2::Ext<kP>, v2::Ext<LD>, v2::Mem::Local, v2::Cyclic,
                                     sycl::local_accessor<T, 1>>;
                const int32_t tile_off = part_id * static_cast<int32_t>(kTileElems);
                const Tile A{A_local, {}, {}, {}, tile_off};
                // Without vectors Z_local is never named, so the kernel does not capture it.
                std::conditional_t<ComputeVectors, Tile, v2::NoPhase> Z{};
                if constexpr (ComputeVectors) Z = Tile{Z_local, {}, {}, {}, tile_off};
                const int32_t base_r = part_id * static_cast<int32_t>(kRotSlots);
                // Real T never names Rd_local, so the kernel does not capture it.
                std::conditional_t<kNeedPhase, decltype(v2::vec(Rd_local, 0)), v2::NoPhase> rd{};
                if constexpr (kNeedPhase) rd = v2::vec(Rd_local, kRotSlots, base_r);

                // ---- Load: A = the Hermitian matrix of the stored triangle, Z = I, zero pad. ----
                for (int32_t c = 0; c < kP; ++c) {
                    T v = T(0);
                    if (live && lane < nn && c < nn) {
                        if constexpr (Upper) v = (lane <= c) ? A_prob(lane, c) : v2::detail::conj_ref<T>(A_prob(c, lane));
                        else v = (lane >= c) ? A_prob(lane, c) : v2::detail::conj_ref<T>(A_prob(c, lane));
                        // The diagonal of a Hermitian matrix is real: drop imaginary noise.
                        if (lane == c) {
                            if constexpr (kNeedPhase) v = T(v.real(), Real(0));
                        }
                    }
                    A(lane, c) = v;
                    if constexpr (ComputeVectors) Z(lane, c) = (lane == c && row_lane) ? T(1) : T(0);
                }
                // The partition's own barrier (chunk mask), not v2::sync: partitions of one
                // sub-group leave the sweep loop after different barrier counts.
                group_barrier(ex.part);

                // ---- Sweeps of rounds: rotations, A := A U (rows), A := U^H A (columns). ----
                // One clean sweep terminates, so `rot_count == 0` is the whole convergence
                // predicate (unlike gesvdj_cta).
                bool converged = false;
                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                    int32_t rot_count = 0;

                    for (int32_t t = 0; t < rounds; ++t) {
                        const auto rnd = v2::rot_round<T>(v2::vec(Rcs_local, kRotSlots, base_r), rd,
                                                          v2::vec(Pair_local, pairs_per_round, t * pairs_per_round));

                        v2::PivotPair pq{0, 0};
                        bool active = false;
                        if (pair_lane) {
                            pq = rnd.pair(lane);
                            active = (pq.q < nn);
                        }
                        v2::JacobiRotation<T> jr = v2::identity_rotation<T>();
                        if (active) {
                            jr = v2::sym_schur2<T>(v2::Arith<T>::re(A(pq.p, pq.p)), v2::Arith<T>::re(A(pq.q, pq.q)),
                                                   static_cast<T>(A(pq.p, pq.q)), thr);
                        }
                        active = jr.active;
                        if (pair_lane) rnd.store(lane, jr);
                        group_barrier(ex.part);

                        // One count serves as the sweep's rotation counter and as an early-out
                        // for rounds that skipped every pair (late sweeps are almost all such).
                        const int32_t round_active = v2::allreduce(ex, active ? 1 : 0);
                        rot_count += round_active;
                        if (round_active == 0) continue;

                        // Lane owns row `lane`; the pairs touch disjoint columns.
                        if (row_lane) {
                            if constexpr (ComputeVectors) v2::rot_round_right(rnd, pairs_per_round, lane, A, Z);
                            else v2::rot_round_right(rnd, pairs_per_round, lane, A);
                        }
                        group_barrier(ex.part);

                        // Lane now owns column `lane`; the pivots are stored exactly (zeros,
                        // real diagonal) so the convergence test does not chase rounding noise.
                        if (row_lane) v2::rot_round_left_hermitian(rnd, pairs_per_round, lane, A);
                        group_barrier(ex.part);
                    }

                    // Converged when a full sweep applied no rotation, i.e. all
                    // n(n-1)/2 pivot pairs passed the relative threshold test.
                    if (rot_count == 0) { converged = true; break; }
                }
                // rot_count is partition-uniform, so lane 0 reports. A STORE, not a raise: this
                // kernel is the item's single writer (no clear kernel that could race with it).
                if (live && lane == 0) detail::info_store(info_dev, prob_id, converged ? 0 : 1);

                // ---- Sort by rank and write back. ----
                // Rank is computed in parallel (one lane per eigenvalue) rather
                // than by a serial sort on the leader; ties break on index so the
                // permutation is a bijection.
                if (live && row_lane) {
                    const Real wj = v2::Arith<T>::re(A(lane, lane));
                    int32_t dst = lane;
                    if (do_sort) {
                        int32_t rank = 0;
                        for (int32_t k = 0; k < nn; ++k) {
                            const Real wk = v2::Arith<T>::re(A(k, k));
                            // A strict total order (NaN after every number, ties and NaNs on index):
                            // the ranks are a bijection even for NaN input, so no slot is left stale.
                            const bool nan_j = sycl::isnan(wj), nan_k = sycl::isnan(wk);
                            const bool before = (nan_j != nan_k) ? nan_j
                                              : (nan_j || wk == wj) ? (k < lane)
                                              : (ascending ? wk < wj : wk > wj);
                            if (before) ++rank;
                        }
                        dst = rank;
                    }

                    W[static_cast<int64_t>(prob_id) * nn + dst] = wj;

                    if constexpr (ComputeVectors) {
                        // A hand loop through the v1 view: v2::copy over v2::mat(A_prob).col(dst)
                        // changed the unrolling (STG 19 -> 42).
                        for (int32_t r = 0; r < nn; ++r) A_prob(r, dst) = Z(r, lane);
                    }
                }
            });
    });
}

} // namespace

template <Backend B, typename T>
Event syev_jacobi_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& a_in,
                      Span<typename base_type<T>::type> eigenvalues,
                      JobType jobz,
                      Uplo uplo,
                      const Span<std::byte>& ws,
                      JacobiParams<T> params,
                      Span<int32_t> info) {
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("syev_jacobi_cta", a_in));
    (void)ws;

    if (a_in.rows() != a_in.cols()) {
        throw batchlas::invalid_argument("syev_jacobi_cta: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw batchlas::invalid_argument("syev_jacobi_cta: invalid JobType.");
    }

    const int64_t n64 = a_in.rows();
    const int64_t batch64 = a_in.batch_size();

    if (n64 < 1 || n64 > 32) {
        throw batchlas::invalid_argument("syev_jacobi_cta currently supports 1 <= n <= 32.");
    }

    const int32_t n = static_cast<int32_t>(n64);

    if (eigenvalues.size() < static_cast<std::size_t>(n64) * static_cast<std::size_t>(batch64)) {
        throw batchlas::invalid_argument("syev_jacobi_cta: eigenvalues span too small for n*batch.");
    }

    // CTA backend: requires subgroup size 32 on NVIDIA-like devices.
    if (!ctx.device().supports_sub_group_size(32)) {
        throw batchlas::unsupported("syev_jacobi_cta: device does not support subgroup size 32 required for CTA kernels.");
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto* w_ptr = eigenvalues.data();

    const bool upper = (uplo == Uplo::Upper);
    const bool vectors = (jobz == JobType::EigenVectors);

    // No clear: the kernel STORES every item's status. `info` is caller USM -- no
    // workspace, which is why syev_jacobi_cta_buffer_size still returns 0.
    int32_t* info_ptr = detail::info_ptr(info, batch64);

    auto launch = [&](auto P_tag) {
        constexpr size_t P = decltype(P_tag)::value;
        if (vectors) {
            if (upper) syev_jacobi_cta_impl<T, P, true, true>(ctx, a, w_ptr, n, params, info_ptr, mask);
            else       syev_jacobi_cta_impl<T, P, true, false>(ctx, a, w_ptr, n, params, info_ptr, mask);
        } else {
            if (upper) syev_jacobi_cta_impl<T, P, false, true>(ctx, a, w_ptr, n, params, info_ptr, mask);
            else       syev_jacobi_cta_impl<T, P, false, false>(ctx, a, w_ptr, n, params, info_ptr, mask);
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
size_t syev_jacobi_cta_buffer_size(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& a,
                                   JobType jobz,
                                   JacobiParams<T> params) {
    (void)ctx;
    (void)jobz;
    (void)params;

    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument("syev_jacobi_cta_buffer_size: A must be square.");
    }
    if (a.rows() < 1 || a.rows() > 32) {
        throw batchlas::invalid_argument("syev_jacobi_cta_buffer_size currently supports 1 <= n <= 32.");
    }

    // Everything is resident in local memory for the lifetime of the kernel.
    return 0;
}

#define SYEV_JACOBI_CTA_INSTANTIATE(back, fp) \
    template Event syev_jacobi_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                              Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, JobType, Uplo, \
                                                              const Span<std::byte>&, JacobiParams<BATCHLAS_UNPAREN fp>, Span<int32_t>); \
    template size_t syev_jacobi_cta_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                                           JobType, JacobiParams<BATCHLAS_UNPAREN fp>);

    BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_JACOBI_CTA_INSTANTIATE)

#undef SYEV_JACOBI_CTA_INSTANTIATE

} // namespace batchlas

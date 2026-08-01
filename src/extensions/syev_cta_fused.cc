#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/mempool.hh>
#include <util/group-invoke.hh>
#include "sg_compat.hh"
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include "sytrd_cta_device.hh"
#include "steqr_cta_device.hh"

#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace sycl::ext::oneapi;

namespace batchlas {

// Kernel name tag. Must live outside the anonymous namespace so it does not
// depend on internal-linkage entities.
template <typename T, size_t P, bool ComputeVectors>
class SyevCtaFusedKernel;

// ---------------------------------------------------------------------------
// Monolithic (fused) CTA symmetric/Hermitian eigensolver.
//
// syev_cta runs the classical three-stage pipeline as three kernel launches:
//
//     sytrd_cta   A -> (d, e, reflectors)      [tile loaded, reduced, stored]
//     steqr_cta   (d, e) -> (w, Z)             [Z loaded, rotated, stored]
//     ormqx_cta   Z -> Q_house * Z             [Z loaded, transformed, stored]
//
// plus, on the eigenvector path, two pack kernels and a copy. For n <= 32 each
// of those stages is far too small to amortize a launch: the whole problem fits
// in one sub-group partition, so the pipeline spends most of its time writing
// intermediates (d, e, tau, Z, the packed reflector matrix) to global memory
// only to read them straight back.
//
// This kernel keeps one problem resident in a single partition from load to
// store. The reduced tile stays in local memory and is reused directly as the
// reflector store for the back-transform, so the intermediates never
// materialize: global traffic is exactly one read of A plus one write of the
// eigenvectors and eigenvalues, against ~7 round trips for the pipeline.
//
// The three stages themselves are the *same code* as the standalone kernels
// (sytrd_cta_device.hh / steqr_cta_device.hh), so a head-to-head benchmark
// isolates the cost of the partitioning.
//
// Notes on the algorithm, all matching syev_cta:
//  - The reduction always runs the Uplo::Upper path; a Uplo::Lower input is
//    symmetrized while loading the tile, which in this design is free (the
//    pipeline needs a separate pass over global memory to do the same).
//  - For Hermitian input the complex tridiagonal is reduced to a real one by a
//    diagonal unitary similarity T' = S^H T S; the phase S is reapplied to the
//    eigenvectors before the back-transform.
//  - Eigenvalues are sorted by computing each lane's rank and writing its
//    eigenvalue/eigenvector to that slot, which needs no scratch and no
//    separate sort kernel.
// ---------------------------------------------------------------------------

namespace {

template <typename U>
inline typename base_type<U>::type real_part_f(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return x.real();
    } else {
        return x;
    }
}

} // namespace

template <typename T, size_t P, bool ComputeVectors>
inline void syev_cta_fused_impl(Queue& ctx,
                                MatrixView<T, MatrixFormat::Dense>& a,
                                typename base_type<T>::type* w_ptr,
                                int32_t n,
                                bool upper,
                                const SteqrParams<T>& params,
                                size_t cta_wg_size_multiplier) {
    using Real = typename base_type<T>::type;
    constexpr bool kComplex = internal::is_complex<T>::value;

    const auto batch_size = a.batch_size();

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();

        const auto dev = ctx->get_device();

        // CTA path assumes warp-sized sub-groups on NVIDIA.
        const int32_t sg_size = 32;

        // A is only ever indexed with lane == row, so it needs no padding.
        // Q is indexed by row during the QL/QR sweeps but by *column* during the
        // back-transform (lane j owns column j), so pad its leading dimension to
        // P+1: with LDQ == P == 32 every lane of that read hits the same bank.
        constexpr int32_t LDA = static_cast<int32_t>(P);
        constexpr int32_t LDQ = static_cast<int32_t>(P) + 1;
        constexpr std::size_t kATileElems = static_cast<std::size_t>(LDA) * P;
        constexpr std::size_t kQTileElems = static_cast<std::size_t>(LDQ) * P;

        const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), sg_size);
        int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1),
                                                       static_cast<int32_t>(cta_wg_size_multiplier));
        int32_t wg_size = base_wg_size * wg_size_multiplier;

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        if (wg_size > max_wg_size) {
            const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
            wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        // Clamp by local memory. Fusion's one real cost is that the reduced tile
        // and the rotation accumulator are live at the same time, where the
        // pipeline only ever holds one of them; on the eigenvector path that
        // roughly doubles the per-problem local-memory footprint and so halves
        // the achievable problems per work-group at P == 32.
        {
            const std::size_t local_mem_bytes = dev.get_info<sycl::info::device::local_mem_size>();
            const std::size_t bytes_per_prob = (kATileElems + 2 * static_cast<std::size_t>(P)) * sizeof(T)
                                             + (ComputeVectors ? kQTileElems * sizeof(Real) : 0);
            const int32_t max_probs = (bytes_per_prob == 0)
                                          ? int32_t(1)
                                          : std::max<int32_t>(int32_t(1),
                                                              static_cast<int32_t>(local_mem_bytes / bytes_per_prob));
            wg_size_multiplier = std::min(wg_size_multiplier, max_probs);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
        const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;

        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * kATileElems), cgh);
        auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
        auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
        auto Q_local = sycl::local_accessor<Real, 1>(
            sycl::range<1>(ComputeVectors ? (probs_per_wg * kQTileElems) : 1), cgh);

        const int32_t nn = n;
        const int32_t nb = static_cast<int32_t>(batch_size);
        const bool is_upper = upper;
        const int32_t max_sweeps = std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.max_sweeps));
        const Real zero_threshold = static_cast<Real>(std::abs(params.zero_threshold));
        const SteqrShiftStrategy shift_strategy = params.cta_shift_strategy;
        const SteqrUpdateScheme update_scheme = params.cta_update_scheme;
        const bool do_sort = params.sort;
        const bool ascending = (params.sort_order == SortOrder::Ascending);

        Real* W = w_ptr;

        cgh.parallel_for<SyevCtaFusedKernel<T, P, ComputeVectors>>(
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());

                const auto sg = it.get_sub_group();
                const auto part = make_partition<P>(sg);

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                if (prob_id >= nb) return;

                auto A_prob = A_view.batch_item(prob_id);

                const int32_t base_a = part_id * static_cast<int32_t>(kATileElems);
                const int32_t base_v = part_id * static_cast<int32_t>(P);
                const int32_t base_w = part_id * static_cast<int32_t>(P);
                const int32_t base_q = ComputeVectors ? (part_id * static_cast<int32_t>(kQTileElems)) : 0;

                // ---- Stage 0: load and symmetrize into the resident tile. ----
                //
                // The reduction below reads the full symmetric matrix, so the
                // requested triangle is mirrored on the way in. The pipeline
                // needs a separate global pass to do this for Uplo::Lower;
                // here it costs nothing.
                for (int32_t c = 0; c < static_cast<int32_t>(P); ++c) {
                    T v = T(0);
                    if (lane < nn && c < nn) {
                        if (lane == c) {
                            // Hermitian diagonals are real by definition; force
                            // it so a caller's round-off cannot leak an
                            // imaginary part into the tridiagonal.
                            v = T(real_part_f(A_prob(lane, c)));
                        } else if (is_upper) {
                            v = (lane < c) ? A_prob(lane, c) : conj_if_complex(A_prob(c, lane));
                        } else {
                            v = (lane > c) ? A_prob(lane, c) : conj_if_complex(A_prob(c, lane));
                        }
                    }
                    A_local[base_a + lane + c * LDA] = v;
                }
                group_barrier(part);

                // ---- Stage 1: tridiagonal reduction (SYTD2, upper path). ----
                const T tau_lane = sytd2_cta_upper_partition<T, LDA>(
                    part, &A_local[base_a], &V_local[base_v], &W_local[base_w], nn, lane);

                // Read the tridiagonal off the tile. Everything above the
                // superdiagonal is left alone: it is the packed reflector store
                // that stage 3 consumes.
                const T d_c = (lane < nn) ? A_local[base_a + lane + lane * LDA] : T(0);
                const T e_c = (lane < (nn - 1)) ? A_local[base_a + lane + (lane + 1) * LDA] : T(0);

                Real diag = real_part_f(d_c);
                Real offdiag = Real(0);
                T phase = T(1);

                if constexpr (kComplex) {
                    // Hermitian tridiagonal -> real symmetric tridiagonal via a
                    // diagonal unitary similarity T' = S^H T S, with
                    //   S(0) = 1,  S(i+1) = S(i) * conj(e(i)) / |e(i)|.
                    // Every lane forms its own S by replaying the recurrence,
                    // which is n broadcasts total and needs no scratch.
                    offdiag = (lane < (nn - 1)) ? sycl::hypot(e_c.real(), e_c.imag()) : Real(0);
                    for (int32_t i = 0; i < nn - 1; ++i) {
                        const T e_i = select_from_group(part, e_c, static_cast<uint32_t>(i));
                        const Real abs_i = select_from_group(part, offdiag, static_cast<uint32_t>(i));
                        if (lane > i && abs_i != Real(0)) {
                            phase = phase * (conj_if_complex(e_i) / abs_i);
                        }
                    }
                } else {
                    offdiag = real_part_f(e_c);
                }

                // ---- Stage 2: tridiagonal eigenproblem (implicit QL/QR). ----
                using QLocalAccT = decltype(Q_local);
                QSharedCache<Real, P, LDQ, ComputeVectors, QLocalAccT> qcache(Q_local, base_q, lane, nn);

                if constexpr (ComputeVectors) {
                    for (int32_t c = 0; c < static_cast<int32_t>(P); ++c) {
                        Q_local[base_q + lane + c * LDQ] = (lane == c && lane < nn) ? Real(1) : Real(0);
                    }
                    group_barrier(part);
                }

                steqr_cta_solve<Real, P>(part, diag, offdiag, qcache, nn,
                                         max_sweeps, zero_threshold,
                                         shift_strategy, update_scheme);

                // ---- Ordering. ----
                //
                // Rather than permuting anything, each lane works out the slot
                // its eigenvalue belongs in and writes there. Index order breaks
                // ties, so the permutation is well defined even with repeated
                // eigenvalues.
                int32_t dst = lane;
                if (do_sort) {
                    const Real wj = diag;
                    int32_t rank = 0;
                    for (int32_t k = 0; k < nn; ++k) {
                        const Real wk = select_from_group(part, diag, static_cast<uint32_t>(k));
                        const bool before = ascending
                            ? (wk < wj || (wk == wj && k < lane))
                            : (wk > wj || (wk == wj && k < lane));
                        if (before) ++rank;
                    }
                    dst = rank;
                }

                if (lane < nn) {
                    W[static_cast<int64_t>(prob_id) * nn + dst] = diag;
                }

                if constexpr (!ComputeVectors) {
                    return;
                } else {
                    // ---- Stage 3: back-transform, Z := Q_house * Z. ----
                    //
                    // Lane j owns column j of Z in registers for the whole
                    // stage, exactly as ormqx_cta's LEFT specialization does.
                    // The indices into C_col are compile-time constants so the
                    // array stays in registers; indexing it with the reflector
                    // support would push it to local memory and turn each of the
                    // ~n^2 accesses into a dependent round trip.
                    T C_col[P];
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                        C_col[r] = T(0);
                    }

                    if constexpr (kComplex) {
                        // Lift the real eigenvectors of T' back through the
                        // diagonal phase: Zc(r, :) = S(r) * Z(r, :). S lives in
                        // lane r's register, so stage it once before V_local is
                        // reused for reflectors.
                        V_local[base_v + lane] = phase;
                        group_barrier(part);
                    }

                    if (lane < nn) {
#pragma unroll
                        for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                            if (r < nn) {
                                const Real z = Q_local[base_q + r + lane * LDQ];
                                if constexpr (kComplex) {
                                    C_col[r] = V_local[base_v + r] * T(z, Real(0));
                                } else {
                                    C_col[r] = z;
                                }
                            }
                        }
                    }
                    group_barrier(part);

                    // Reflectors of the upper-path SYTD2 form a QL factorization:
                    // Q = H(n-1) ... H(1), and applying Q on the left in
                    // ascending order is what ormqx_cta(QL, Left, NoTrans) does.
                    // Reflector ii lives in rows 0..ii-1 of tile column ii+1 with
                    // an implicit 1 at row ii, and its tau is in lane ii.
                    for (int32_t ii = 0; ii < nn - 1; ++ii) {
                        const T tau_ii = select_from_group(part, tau_lane, static_cast<uint32_t>(ii));

                        // Stage v indexed by absolute row and explicitly zeroed
                        // outside its support, so the update below runs over the
                        // full compile-time row range with no predication.
                        T vv = T(0);
                        if (lane < ii) {
                            vv = A_local[base_a + lane + (ii + 1) * LDA];
                        } else if (lane == ii) {
                            vv = T(1);
                        }
                        V_local[base_v + lane] = vv;
                        group_barrier(part);

                        if (lane < nn && tau_ii != T(0)) {
                            T dot = T(0);
#pragma unroll
                            for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                                dot += conj_if_complex(V_local[base_v + r]) * C_col[r];
                            }
                            const T gamma = tau_ii * dot;
#pragma unroll
                            for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                                C_col[r] -= V_local[base_v + r] * gamma;
                            }
                        }

                        // All lanes must be done reading v before it is rebuilt.
                        group_barrier(part);
                    }

                    if (lane < nn) {
                        for (int32_t r = 0; r < nn; ++r) {
                            A_prob(r, dst) = C_col[r];
                        }
                    }
                }
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
                     size_t cta_wg_size_multiplier) {
    (void)ws;

    if (a_in.rows() != a_in.cols()) {
        throw std::invalid_argument("syev_cta_fused: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument("syev_cta_fused: invalid JobType.");
    }

    const int64_t n64 = a_in.rows();
    const int64_t batch64 = a_in.batch_size();

    if (n64 < 1 || n64 > 32) {
        throw std::invalid_argument("syev_cta_fused currently supports 1 <= n <= 32.");
    }
    if (eigenvalues.size() < static_cast<std::size_t>(n64) * static_cast<std::size_t>(batch64)) {
        throw std::invalid_argument("syev_cta_fused: eigenvalues span too small for n*batch.");
    }

    // CTA backend: requires subgroup size 32 on NVIDIA-like devices.
    {
        const auto dev = ctx->get_device();
        const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        bool has32 = false;
        for (auto sgs : sg_sizes) {
            if (static_cast<int32_t>(sgs) == 32) {
                has32 = true;
                break;
            }
        }
        if (!has32) {
            throw std::runtime_error("syev_cta_fused: device does not support subgroup size 32 required for CTA kernels.");
        }
    }

    // Match syev_cta's robustness bump so the two solve the tridiagonal problem
    // with identical settings unless the caller tuned them.
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

    auto launch = [&](auto P_tag) {
        constexpr size_t P = decltype(P_tag)::value;
        if (vectors) {
            syev_cta_fused_impl<T, P, true>(ctx, a, w_ptr, n, upper, steqr_params, cta_wg_size_multiplier);
        } else {
            syev_cta_fused_impl<T, P, false>(ctx, a, w_ptr, n, upper, steqr_params, cta_wg_size_multiplier);
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
        throw std::invalid_argument("syev_cta_fused_buffer_size: A must be square.");
    }
    if (a.rows() < 1 || a.rows() > 32) {
        throw std::invalid_argument("syev_cta_fused_buffer_size currently supports 1 <= n <= 32.");
    }

    // Nothing is spilled to global memory: the whole solve is partition-resident.
    return 0;
}

#define SYEV_CTA_FUSED_INSTANTIATE(back, fp) \
    template Event syev_cta_fused<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                             Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, JobType, Uplo, \
                                                             const Span<std::byte>&, SteqrParams<BATCHLAS_UNPAREN fp>, size_t); \
    template size_t syev_cta_fused_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                                          JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

#define SYEV_CTA_FUSED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEV_CTA_FUSED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEV_CTA_FUSED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    SYEV_CTA_FUSED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    SYEV_CTA_FUSED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYEV_CTA_FUSED_INSTANTIATE_FOR_BACKEND
#undef SYEV_CTA_FUSED_INSTANTIATE

} // namespace batchlas

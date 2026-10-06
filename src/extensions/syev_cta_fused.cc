#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-invoke.hh>
#include "sg_compat.hh"
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include "info_span.hh"
#include "../util/template-instantiations.hh"

#include "sytrd_cta_device.hh"
#include "steqr_cta_device.hh"

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
// intermediate touches global memory. The stages are the SAME code as the standalone
// kernels (sytrd_cta_device.hh / steqr_cta_device.hh): keep it that way, so a
// head-to-head benchmark measures fusion and nothing else. Real input seeds the
// sweeps with an in-place Q_house (DSYEV's route); Hermitian input applies the
// reflectors afterwards. The reduction always runs Uplo::Upper.
// evidence: docs/perf/syev.md#syev-the-fused-cta-kernel-design
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
                                size_t cta_wg_size_multiplier,
                                int32_t* info) {
    using Real = typename base_type<T>::type;
    constexpr bool kComplex = internal::is_complex<T>::value;

    const auto batch_size = a.batch_size();

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();

        const auto dev = ctx->get_device();

        // CTA path assumes warp-sized sub-groups on NVIDIA.
        const int32_t sg_size = 32;

        // Real eigenvector path: the reflector tile and the rotation accumulator are the
        // SAME tile (stage 2a builds Q_house in place), so it carries the accumulator's
        // P+1 padding: it is read by column on write-out, and LD == 32 would serialize
        // that 32 ways on bank conflicts. The complex path keeps two tiles (real
        // accumulator, complex reflectors).
        // evidence: docs/perf/syev.md#syev-the-fused-cta-kernel-design
        constexpr bool kFusedOrgql = ComputeVectors && !kComplex;
        constexpr int32_t LDQ = static_cast<int32_t>(P) + 1;
        constexpr int32_t LDA = kFusedOrgql ? LDQ : static_cast<int32_t>(P);
        constexpr bool kSeparateQTile = ComputeVectors && !kFusedOrgql;
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
            const std::size_t local_mem_bytes = dev.get_info<sycl::info::device::local_mem_size>();
            const std::size_t bytes_per_prob = (kATileElems + 2 * static_cast<std::size_t>(P)) * sizeof(T)
                                             + (kSeparateQTile ? kQTileElems * sizeof(Real) : 0);
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
            sycl::range<1>(kSeparateQTile ? (probs_per_wg * kQTileElems) : 1), cgh);

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
        // A local so `[=]` copies a pointer; nullptr (status not requested) makes info_store a no-op.
        int32_t* const info_dev = info;

        cgh.parallel_for<SyevCtaFusedKernel<T, P, ComputeVectors>>(
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
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
                // Clamp, do not return: a dead chunk runs every stage on a zero matrix and writes nothing.
                const bool live = prob_id < nb;
                auto A_prob = A_view.batch_item(live ? prob_id : 0);

                const int32_t base_a = part_id * static_cast<int32_t>(kATileElems);
                const int32_t base_v = part_id * static_cast<int32_t>(P);
                const int32_t base_w = part_id * static_cast<int32_t>(P);
                const int32_t base_q = kSeparateQTile ? (part_id * static_cast<int32_t>(kQTileElems)) : 0;

                // ---- Stage 0: load and symmetrize into the resident tile. ----
                // The reduction reads the full matrix, so the requested triangle is
                // mirrored on the way in.
                for (int32_t c = 0; c < static_cast<int32_t>(P); ++c) {
                    T v = T(0);
                    if (live && lane < nn && c < nn) {
                        if (lane == c) {
                            // Hermitian diagonals are real: drop the caller's imaginary round-off.
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

                // Read the tridiagonal off the tile; above the superdiagonal is the packed
                // reflector store that stage 3 consumes.
                const T d_c = (lane < nn) ? A_local[base_a + lane + lane * LDA] : T(0);
                const T e_c = (lane < (nn - 1)) ? A_local[base_a + lane + (lane + 1) * LDA] : T(0);

                Real diag = real_part_f(d_c);
                Real offdiag = Real(0);
                T phase = T(1);

                if constexpr (kComplex) {
                    // Hermitian -> real tridiagonal via T' = S^H T S; every lane replays
                    // S(0) = 1, S(i+1) = S(i) * conj(e(i)) / |e(i)| itself (no scratch).
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

                // ---- Ordering: each lane writes its eigenvalue to its rank slot; index
                // order breaks ties, so repeated eigenvalues still give a bijection. ----
                const auto slot_of = [&](Real wj) {
                    if (!do_sort) return lane;
                    int32_t rank = 0;
                    for (int32_t k = 0; k < nn; ++k) {
                        const Real wk = select_from_group(part, wj, static_cast<uint32_t>(k));
                        const bool before = ascending
                            ? (wk < wj || (wk == wj && k < lane))
                            : (wk > wj || (wk == wj && k < lane));
                        if (before) ++rank;
                    }
                    return rank;
                };

                if constexpr (!ComputeVectors) {
                    // ---- Stage 2: eigenvalues only, no accumulator. ----
                    QSharedCache<Real, P, LDQ, false, decltype(Q_local)> qcache(Q_local, base_q, lane, nn);
                    const bool failed = steqr_cta_solve<Real, P>(solve_part, diag, offdiag, qcache, nn,
                                             max_sweeps, zero_threshold,
                                             shift_strategy, update_scheme);
                    // A STORE, not a raise: exactly one of the three arms below runs per
                    // problem, so this kernel is the item's single writer and needs no
                    // separate clear (a second submission, racy out of order).
                    if (live && lane == 0) detail::info_store(info_dev, prob_id, failed ? 1 : 0);

                    const int32_t dst = slot_of(diag);
                    if (live && lane < nn) {
                        W[static_cast<int64_t>(prob_id) * nn + dst] = diag;
                    }
                } else if constexpr (kFusedOrgql) {
                    // ---- Stage 2a: form Q_house explicitly, in place. ----
                    //
                    // DSYEV's structure: DORGTR builds Q, DSTEQR is seeded with it, and
                    // the sweeps accumulate straight onto Q (no back-transform pass).
                    // Q = H(n-2)...H(0); column k is GENERATED at step k and no step
                    // touches a column above its own index, so reflector k' > k (tile
                    // column k'+1) is still intact when its turn comes (DORG2L without
                    // its column shift). Lane c owns column c in registers; every
                    // subscript into Qc must stay a compile-time constant.
                    // evidence: docs/perf/syev.md#syev-the-fused-cta-kernel-design
                    Real Qc[P];
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                        Qc[r] = Real(0);
                    }

                    for (int32_t k = 0; k < nn - 1; ++k) {
                        const Real tau_k = select_from_group(part, tau_lane, static_cast<uint32_t>(k));

                        // Stage v_k indexed by absolute row, explicitly zeroed
                        // outside its support.
                        Real vv = Real(0);
                        if (lane < k) {
                            vv = A_local[base_a + lane + (k + 1) * LDA];
                        } else if (lane == k) {
                            vv = Real(1);
                        }
                        V_local[base_v + lane] = vv;
                        group_barrier(part);

                        if (lane < k) {
                            // Existing column: apply H(k).
                            Real dot = Real(0);
#pragma unroll
                            for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                                dot += V_local[base_v + r] * Qc[r];
                            }
                            const Real gamma = tau_k * dot;
#pragma unroll
                            for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                                Qc[r] -= V_local[base_v + r] * gamma;
                            }
                        } else if (lane == k) {
                            // New column: H(k) e_k = e_k - tau_k v_k.
#pragma unroll
                            for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                                Qc[r] = (r < k) ? (-tau_k * V_local[base_v + r])
                                                : ((r == k) ? (Real(1) - tau_k) : Real(0));
                            }
                        }

                        // All lanes must be done reading v before it is rebuilt.
                        group_barrier(part);
                    }

                    // Q is the identity outside its leading (n-1)x(n-1) block, and
                    // the pad columns must hold defined values because the sweeps
                    // run unguarded on all P lanes.
                    if (lane >= nn) {
#pragma unroll
                        for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                            Qc[r] = Real(0);
                        }
                    } else if (lane == nn - 1) {
#pragma unroll
                        for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                            Qc[r] = (r == nn - 1) ? Real(1) : Real(0);
                        }
                    }

                    // Hand the tile over: from here it is the accumulator.
                    group_barrier(part);
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                        A_local[base_a + r + lane * LDA] = Qc[r];
                    }
                    group_barrier(part);

                    // ---- Stage 2b: sweeps, accumulating onto Q_house. ----
                    QSharedCache<Real, P, LDQ, true, decltype(A_local)> qcache(A_local, base_a, lane, nn);
                    const bool failed = steqr_cta_solve<Real, P>(solve_part, diag, offdiag, qcache, nn,
                                             max_sweeps, zero_threshold,
                                             shift_strategy, update_scheme);
                    if (live && lane == 0) detail::info_store(info_dev, prob_id, failed ? 1 : 0);

                    // The sweeps wrote the tile by row (lane == row); the readout
                    // below reads it by column, i.e. other lanes' writes.
                    group_barrier(part);
                    const int32_t dst = slot_of(diag);
                    if (live && lane < nn) {
                        W[static_cast<int64_t>(prob_id) * nn + dst] = diag;
                        for (int32_t r = 0; r < nn; ++r) {
                            A_prob(r, dst) = A_local[base_a + r + lane * LDA];
                        }
                    }
                } else {
                    // ---- Stage 2: sweeps on a separate real accumulator (Hermitian input;
                    // reflectors are applied afterwards, in stage 3). ----
                    QSharedCache<Real, P, LDQ, true, decltype(Q_local)> qcache(Q_local, base_q, lane, nn);

                    for (int32_t c = 0; c < static_cast<int32_t>(P); ++c) {
                        Q_local[base_q + lane + c * LDQ] = (lane == c && lane < nn) ? Real(1) : Real(0);
                    }
                    group_barrier(part);

                    const bool failed = steqr_cta_solve<Real, P>(solve_part, diag, offdiag, qcache, nn,
                                             max_sweeps, zero_threshold,
                                             shift_strategy, update_scheme);
                    if (live && lane == 0) detail::info_store(info_dev, prob_id, failed ? 1 : 0);

                    const int32_t dst = slot_of(diag);
                    if (live && lane < nn) {
                        W[static_cast<int64_t>(prob_id) * nn + dst] = diag;
                    }

                    // ---- Stage 3: back-transform, Z := Q_house * Z. ----
                    // Lane j owns column j of Z in registers (as ormqx_cta LEFT). Indices
                    // into C_col must stay compile-time constants, or the array spills
                    // to local memory.
                    T C_col[P];
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                        C_col[r] = T(0);
                    }

                    // Lift through the phase: Zc(r, :) = S(r) * Z(r, :). Stage S before
                    // V_local is reused for reflectors.
                    V_local[base_v + lane] = phase;
                    group_barrier(part);

                    if (lane < nn) {
#pragma unroll
                        for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                            if (r < nn) {
                                const Real z = Q_local[base_q + r + lane * LDQ];
                                C_col[r] = V_local[base_v + r] * T(z, Real(0));
                            }
                        }
                    }
                    group_barrier(part);

                    // Upper-path SYTD2 reflectors form a QL factorization (ormqx_cta QL,
                    // Left, NoTrans). Reflector ii: rows 0..ii-1 of tile column ii+1,
                    // implicit 1 at row ii, tau in lane ii.
                    for (int32_t ii = 0; ii < nn - 1; ++ii) {
                        const T tau_ii = select_from_group(part, tau_lane, static_cast<uint32_t>(ii));

                        // v zeroed outside its support: the update runs unpredicated over all P rows.
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

                    if (live && lane < nn) {
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
                     size_t cta_wg_size_multiplier,
                     Span<int32_t> info) {
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
            throw batchlas::unsupported("syev_cta_fused: device does not support subgroup size 32 required for CTA kernels.");
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

    // No clear: the kernel STORES every item's status. `info` is caller USM, so
    // syev_cta_fused_buffer_size still returns 0.
    int32_t* info_ptr = detail::info_ptr(info, batch64);

    auto launch = [&](auto P_tag) {
        constexpr size_t P = decltype(P_tag)::value;
        if (vectors) {
            syev_cta_fused_impl<T, P, true>(ctx, a, w_ptr, n, upper, steqr_params, cta_wg_size_multiplier, info_ptr);
        } else {
            syev_cta_fused_impl<T, P, false>(ctx, a, w_ptr, n, upper, steqr_params, cta_wg_size_multiplier, info_ptr);
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

// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/sort.hh>
#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>
#include <src/math-helpers.hh>
#include <src/util/template-instantiations.hh>
#include <src/ops/syev/steqr/steqr_internal.hh>
#include <src/ops/syev/stedc/stedc_secular.hh>
#include <src/ops/syev/stedc/stedc_merge_kernels.hh>
#include <src/ops/syev/stedc/stedc_levels_plan.hh>
#include <src/ops/syev/stedc/rank_one.hh>
#include <src/kernels/common/info_span.hh>
#include <src/ops/syev/stedc/stedc_internal.hh>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace batchlas {

namespace {

// `is_gpu`, not `B != Backend::NETLIB`: why the host path exists (no ROCm-style root
// routines inside host SYCL kernels) is a property of the DEVICE, not the backend.
// evidence: docs/perf/stedc.md#stedc-convergence-reporting-through-info
template <Backend B, typename T>
inline StedcParams<T> resolve_stedc_tuning(int64_t n, StedcParams<T> params, bool is_gpu) {
    const int32_t nn = static_cast<int32_t>(n);

    if (is_gpu) {
        if (params.recursion_threshold <= 0) {
            params.recursion_threshold = tuning::stedc_recursion_threshold_for_n(nn);
        }
        if (params.merge_variant == StedcMergeVariant::Auto) {
            params.merge_variant = static_cast<StedcMergeVariant>(tuning::stedc_merge_variant_for_n(nn));
        }
        if (params.secular_threads_per_root <= 0) {
            params.secular_threads_per_root = tuning::stedc_threads_per_root_for_n(nn);
        }
        if (params.secular_cta_wg_size_multiplier <= 0) {
            params.secular_cta_wg_size_multiplier = tuning::stedc_wg_multiplier_for_n(nn);
        }
    } else {
        // The host/native CPU backend cannot safely invoke the ROCm-style
        // secular root routines from inside SYCL kernels. Keep NETLIB on the
        // legacy merge path even if callers request the newer GPU-oriented variants.
        params.secular_solver = StedcSecularSolver::Legacy;
        params.merge_variant = StedcMergeVariant::Baseline;
    }

    const int64_t safe_n = std::max<int64_t>(1, n);
    params.recursion_threshold = std::max<int64_t>(1, std::min<int64_t>(params.recursion_threshold, safe_n));

    return params;
}

} // namespace

// SYCL kernel class tags for profiling and naming (templated to avoid ODR violations)
template <Backend B, typename T> class StedcModifyDiagonal;
template <Backend B, typename T> class StedcComputeV;
template <Backend B, typename T> class StedcDeflation;

// Smallest subproblem size at which the deflation-aware back-transform is used.
// Learning the batch-wide non-deflated width needs a host sync, which stalls the
// enqueue pipeline; below this size the saved GEMM flops do not pay for it.
// evidence: docs/perf/stedc.md#stedc-the-deflation-gemm-threshold
inline constexpr int64_t stedc_deflation_gemm_min_n = 512;

// Only take the narrow path when deflation actually removed enough columns to
// be worth the extra narrow copy. If nothing much deflated, fall through to the
// original full-width GEMM.
inline constexpr double stedc_deflation_gemm_max_kept_fraction = 0.75;
template <Backend B, typename T> class StedcSecularSolve;
template <Backend B, typename T> class StedcRescaleV;
template <Backend B, typename T> class StedcMatrixUpdate;
template <Backend B, typename T> class StedcAssignEigenvalues;

// Kernel tags for the level-synchronous ("flattened") driver.
template <Backend B, typename T> class StedcLevelPad;
template <Backend B, typename T> class StedcLevelBlockDiag;
template <Backend B, typename T> class StedcLevelSplit;
template <Backend B, typename T> class StedcLevelRho;
template <Backend B, typename T> class StedcLevelZeroOffdiag;
template <Backend B, typename T> class StedcLevelUnpad;
// Folds the per-leaf convergence status down onto the caller's per-item span.
template <Backend B, typename T> class StedcLevelLeafStatus;

// Select the even- or odd-indexed half of a batched view. Both halves stay
// affine (stride doubles), so a level can write straight into its parent's two
// diagonal sub-blocks with strided-batched GEMMs.
// Siblings 2q and 2q+1 are one batch item, so both halves take the parent level's mask.
template <typename T>
inline MatrixView<T, MatrixFormat::Dense> stedc_batch_parity(const MatrixView<T, MatrixFormat::Dense>& m, int parity,
                                                             Span<const std::uint8_t> half_mask) {
    return MatrixView<T, MatrixFormat::Dense>(m.data_ptr() + parity * m.stride(),
                                              m.rows(), m.cols(), m.ld(),
                                              m.stride() * 2, m.batch_size() / 2).with_batch_mask(half_mask);
}

// Attaches the call's mask to a view rebuilt from raw memory (the drop hazard); unmasked: unchanged.
template <typename View>
inline View stedc_masked(const View& v, Span<const std::uint8_t> mask) {
    return mask.empty() || v.batch_size() == 0 ? v : v.with_batch_mask(mask);
}

// ---------------------------------------------------------------------------
// One Cuppen merge over a *super-batch* of P size-uniform sub-problems of size n whose halves
// are solved: T = diag(T1, T2) + rho u u^T with Q = diag(Q1, Q2), so Q^T T Q = D + rho z z^T.
// The recursive driver passes P = batch_size; the level-synchronous one P = nodes * batch_size.
//
//   eigenvalues  n x P, in/out: D in, the merged eigenvalues out
//   eigvects     n x n x P, in/out: Q in, the merged eigenvectors out
//   Qprime       n x n x P scratch: U, the eigenvectors of D + rho z z^T
//   temp_Q       n x n x P scratch
//   rho          P signed cut off-diagonals beta (tridiag::Tear)
//   m            size of the left half
// evidence: docs/perf/stedc.md#stedc-the-level-synchronous-driver
// ---------------------------------------------------------------------------

// The merge's scratch. One layout serves the merge and stedc_merge_step_workspace.
template <typename T> struct MergeScratch {
    tridiag::MergePermutations<VectorView<int32_t>> perm;
    VectorView<T> z;
    VectorView<int32_t> keep;   // 1 for a surviving pole
    Span<int32_t> K;            // survivors per sub-problem
    VectorView<T> roots;        // the secular roots lambda_i, i < K
};

template <typename T>
MergeScratch<T> merge_scratch(Queue& ctx, BumpAllocator& pool, int64_t n, int64_t P, Span<const std::uint8_t> mask) {
    const auto index_vector = [&] {
        return stedc_masked(VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * P), n, P), mask);
    };
    const auto compaction = index_vector();
    const auto column_map = index_vector();
    const auto z = stedc_masked(VectorView<T>(pool.allocate<T>(ctx, n * P), n, P), mask);
    const auto keep = index_vector();
    const auto K = pool.allocate<int32_t>(ctx, P);
    const auto roots = stedc_masked(VectorView<T>(pool.allocate<T>(ctx, n * P), n, P), mask);
    return {{column_map, compaction, compaction}, z, keep, K, roots};
}

// Scratch bytes `stedc_merge_step` sub-allocates for a size-`s` merge over `P` sub-problems
// (everything except the two s x s scratch matrices, which the caller owns).
template <typename T>
size_t stedc_merge_step_workspace(Queue& ctx, size_t s, size_t P) {
    return workspace_bytes([&](BumpAllocator& pool) {
        return merge_scratch<T>(ctx, pool, static_cast<int64_t>(s), static_cast<int64_t>(P), {});
    });
}

// z = [last row of Q1, sign(beta) first row of Q2] / sqrt2, so ||z|| = 1.
template <Backend B, typename T>
void form_z(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& eigvects, const Span<T>& rho,
            const VectorView<T>& z, int64_t m) {
    const int64_t n = z.size();
    const auto batch_size = z.batch_size();
    auto E1 = eigvects(Slice{0, m}, Slice(0, m));
    auto E2 = eigvects(Slice{m, SliceEnd()}, Slice(m, SliceEnd()));
    ctx -> submit([&](sycl::handler& h) {
        auto E1view = E1.kernel_view();
        auto E2view = E2.kernel_view();
        batchlas::impl::parallel_for<StedcComputeV<B, T>>(h, sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            if (!z.active(bid)) return;  // I1, no barrier
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            // u = e_{m-1} + sign(beta) e_m, so Q2's half of z carries sign(beta); without it
            // the merged vectors mix Q1 and Q2 columns with inconsistent signs.
            const T v_sign2 = tridiag::Tear<T>{rho[bid]}.sign();
            for (int i = tid; i < m; i += bdim) {
                z(i, bid) = E1view(m - 1, i, bid) / std::sqrt(T(2));
            }
            for (int i = tid; i < n - m; i += bdim) {
                z(i + m, bid) = v_sign2 * E2view(0, i, bid) / std::sqrt(T(2));
            }
        });
    });
}

// P sorts D ascending and is applied to D and z; Q's columns stay put (the lazy column map).
template <typename T>
void order_poles(Queue& ctx, const VectorView<T>& d, const VectorView<T>& z, const VectorView<int32_t>& column_map) {
    // (void) on an Event: deliberate. This Queue is in-order, so the next submission
    // is already ordered after this one and the Event carries nothing the caller needs.
    (void)argsort(ctx, d, column_map, SortOrder::Ascending, true);
    (void)permute(ctx, d, column_map);
    (void)permute(ctx, z, column_map);
}

// tol = 8 eps max(|D|inf, |z|inf). Coincident poles: one Givens on two columns of Q zeroes a weight.
// Negligible weights deflate. The compaction puts the K survivors first and is folded into P.
template <Backend B, typename T>
void deflate(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& eigvects, const VectorView<T>& eigenvalues,
             const Span<T>& rho, const MergeScratch<T>& scratch) {
    const int64_t n = eigenvalues.size();
    const auto batch_size = eigenvalues.batch_size();
    const auto& z = scratch.z;
    const auto& keep = scratch.keep;
    const auto& compaction = scratch.perm.compaction;
    const auto& column_map = scratch.perm.column_map;
    const auto& K = scratch.K;
    ctx -> submit([&](sycl::handler& h) {
        auto Q = eigvects.kernel_view();
        auto perm_local = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_include = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_exclude = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto norm_mem = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        batchlas::impl::parallel_for<StedcDeflation<B, T>>(h, sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();
        auto cta = item.get_group();
        // I1, before any barrier. K = 0 is neutral: later kernels stop at it, dd_max ignores it.
        if (!z.active(bid)) {
            if (tid == 0) K[bid] = 0;
            return;
        }

        for (int k = tid; k < n; k += bdim){
            keep(k, bid) = 0;
            scan_mem_exclude[k] = 0;
            compaction(k, bid) = -1;
            perm_local[k] = column_map(k, bid);
        }

        sycl::group_barrier(cta);

        // LAPACK-style ABSOLUTE tolerance, also used for coincident poles. A relative
        // tolerance under-deflates clustered small eigenvalues (bad orthogonality).
        // evidence: docs/perf/stedc.md#stedc-the-absolute-deflation-tolerance
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(eigenvalues(k, bid)); }
        auto eig_max = batchlas::portable::joint_reduce(cta,
                          util::get_raw_ptr(norm_mem),
                          util::get_raw_ptr(norm_mem) + n,
                          sycl::maximum<T>());
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(z(k, bid)); }
        auto z_max_pre = batchlas::portable::joint_reduce(cta,
                          util::get_raw_ptr(norm_mem),
                          util::get_raw_ptr(norm_mem) + n,
                          sycl::maximum<T>());
        const auto pole_tol = tridiag::DeflationTolerance<T>::from(eig_max, z_max_pre);

        // Coincident poles d_j ~ d_j+1: R = annihilate(z_j+1, z_j) moves z_j's weight onto z_j+1,
        // and Q <- Q R^T on the physical columns (P(j+1), P(j)).
        // WP6f: dlaed2 tests small z first, compares against the last survivor only, uses
        // |cs*gap| <= tol and rewrites d; this pass does none of the four.
        for (int j = 0; j < n - 1; j++) {
            const T d_hi = eigenvalues(j + 1, bid);   // read first: base's load order, PTX-identical
            if (tridiag::coincident_poles(eigenvalues(j, bid), d_hi, pole_tol)) {
                auto f = z(j + 1, bid);
                auto g = z(j, bid);
                auto [c, s, r] = detail::lartg(f, g);
                sycl::group_barrier(cta);
                if (tid == 0) {
                    z(j, bid) = T(0.0);
                    z(j + 1, bid) = r;
                }
                const int32_t pj = perm_local[j];
                const int32_t pj1 = perm_local[j + 1];
                if (pj >= 0 && pj < n && pj1 >= 0 && pj1 < n) {
                    for (int k = tid; k < n; k += bdim) {
                        auto Qi = Q(k, pj, bid), Qj = Q(k, pj1, bid);
                        Q(k, pj, bid) = Qi*c - Qj*s;
                        Q(k, pj1, bid) = Qj*c + Qi*s;
                    }
                }
                // Make tid 0's writes to z(j, bid) and z(j+1, bid) visible to all threads
                // before the next iteration reads z(j+1, bid) to compute its Givens pair.
                sycl::group_barrier(cta);
            }
        }

        sycl::group_barrier(cta);

        // Negligible weights (LAPACK LAED8): |beta z_k / ||z||| <= tol, tested on z_k / ||z|| (z itself is
        // not rescaled here). The pass above never writes D, so eig_max2 == eig_max; reusing eig_max keeps
        // it live across the loop and costs 2 registers (float), hence the second reduction.
        // WP6f: eps here is 2u and the test takes |beta|, not rho = 2|beta|: 4x dlaed2's threshold.
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(eigenvalues(k, bid)); }
        auto eig_max2 = batchlas::portable::joint_reduce(cta,
                          util::get_raw_ptr(norm_mem),
                          util::get_raw_ptr(norm_mem) + n,
                          sycl::maximum<T>());

        auto z_norm = detail::nrm2<T>(cta, z);
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(z(k, bid) / z_norm); }
        auto z_max = batchlas::portable::joint_reduce(cta,
                        util::get_raw_ptr(norm_mem),
                        util::get_raw_ptr(norm_mem) + n,
                        sycl::maximum<T>());
        const auto weight_tol = tridiag::DeflationTolerance<T>::from(eig_max2, z_max);

        for (int k = tid; k < n; k += bdim) {
            if (!tridiag::negligible_weight(rho[bid], norm_mem[k], weight_tol)) {
                keep(k, bid) = 1;
            } else {
                scan_mem_exclude[k] = 1;
            }
        }

        sycl::group_barrier(cta);

        // Compaction: survivors in order at [0, K), deflated poles reversed at [K, n).
        batchlas::portable::joint_exclusive_scan(cta,
                       keep.batch_item(bid).data_ptr(),
                       keep.batch_item(bid).data_ptr() + n,
                       util::get_raw_ptr(scan_mem_include),
                       0,
                       sycl::plus<int32_t>());
        batchlas::portable::joint_exclusive_scan(cta,
                       util::get_raw_ptr(scan_mem_exclude),
                       util::get_raw_ptr(scan_mem_exclude) + n,
                       util::get_raw_ptr(scan_mem_exclude),
                       0,
                       sycl::plus<int32_t>());

        for (int k = tid; k < n; k += bdim) {
            if (keep(k, bid) == 1) {
                compaction(scan_mem_include[k], bid) = k;
            } else {
                compaction(n - 1 - scan_mem_exclude[k], bid) = k;
            }
        }

        if (tid == 0) {
            K[bid] = scan_mem_include[n - 1] + keep(n - 1, bid);
        }

        });
    });

    (void)permute(ctx, eigenvalues, compaction);
    (void)permute(ctx, z, compaction);
    // P <- P o compaction.
    (void)permute(ctx, column_map, compaction);
}

// Baseline: lambda_i = d_o + tau, i < K, one secular root per thread from a column of Qprime
// that holds a copy of D (the solver's pole array).
template <Backend B, typename T>
void secular_roots(Queue& ctx, const VectorView<T>& eigenvalues, const MatrixView<T, MatrixFormat::Dense>& Qprime,
                   const Span<T>& rho, const MergeScratch<T>& scratch, int32_t* info, int64_t info_nodes_per_item) {
    const auto batch_size = eigenvalues.batch_size();
    const auto& z = scratch.z;
    const auto& K = scratch.K;
    const auto& roots = scratch.roots;
    // Locals so the kernel's `[=]` copies a pointer and a scalar; nullptr makes
    // info_report a no-op.
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        batchlas::impl::parallel_for<StedcSecularSolve<B, T>>(h, sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto cta = item.get_group();
            auto Q_bid = Qview.batch_item(bid);
            auto n = K[bid];
            for (int k = tid; k < n * n; k += bdim) {
                auto i = k % n;
                auto j = k / n;
                Q_bid(i, j) = eigenvalues(i, bid);
            }
            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) {
                auto dview = Q_bid(Slice{}, k);
                // info_report is an atomic fetch_max: exactly the "did ANY root
                // fail" reduction, however many threads report the same item.
                bool root_converged = true;
                if (k == n - 1){
                    roots(k, bid) = sec_solve_ext_roc(n, dview, z.batch_item(bid), tridiag::Tear<T>{rho[bid]}.rho(), root_converged);
                } else {
                    roots(k, bid) = sec_solve_roc(n, dview, z.batch_item(bid), tridiag::Tear<T>{rho[bid]}.rho(), k, root_converged);
                }
                if (!root_converged) {
                    detail::info_report(info_dev,
                                        detail::info_item(static_cast<int64_t>(bid), nodes_per_item),
                                        1);
                }
            }
            sycl::group_barrier(cta);
        });
    });
}

// Löwner: z_j <- sign(z_j) sqrt|prod_i delta_ji / prod_{i!=j} (d_i - d_j)| = sqrt(rho) zhat_j (the
// 1/rho factor is dropped and cancels when U's columns are normalised), so the computed
// roots are exact for zhat (ROCm's stedc_mergeValues_Rescale_kernel, group product reduction).
template <Backend B, typename T>
void loewner(Queue& ctx, const VectorView<T>& eigenvalues, const MatrixView<T, MatrixFormat::Dense>& Qprime,
             const MergeScratch<T>& scratch) {
    const auto batch_size = eigenvalues.batch_size();
    const auto& z = scratch.z;
    const auto& K = scratch.K;
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        batchlas::impl::parallel_for<StedcRescaleV<B, T>>(h,
            sycl::nd_range<1>(batch_size * 128, 128),
            [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto g   = item.get_group();
                auto tid = item.get_local_linear_id();
                auto bdim = item.get_local_range(0);
                auto Qbid = Qview.batch_item(bid);
                auto dd = K[bid];

                // In native T (double accumulation is unnecessary with the absolute tolerance).
                // evidence: docs/perf/stedc.md#stedc-the-absolute-deflation-tolerance
                for (int eid = 0; eid < dd; ++eid)
                {
                    const T Di = eigenvalues(eid, bid);
                    T partial = T(1);
                    for(int j = tid; j < dd; j += static_cast<int>(bdim))
                    {
                        const T q_elem = Qbid(eid, j);
                        T ratio;
                        if (j == eid) {
                            ratio = q_elem;
                        } else {
                            const T denom = Di - eigenvalues(j, bid);
                            ratio = q_elem / denom;
                        }
                        partial *= ratio;
                    }

                    T valf = batchlas::portable::reduce_over_group(g, partial, sycl::multiplies<T>());
                    if(tid == 0)
                    {
                        T mag  = std::sqrt(std::fabs(valf));
                        T sign = z(eid, bid) >= T(0) ? T(1) : T(-1);
                        z(eid, bid) = sign * mag;
                    }
                }
            });
    });
}

// U(:, i) = normalize(zhat ./ delta(:, i)), i < K; the deflated columns of U stay e_j.
template <Backend B, typename T>
void eigenvectors(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& Qprime, const MergeScratch<T>& scratch) {
    const auto batch_size = scratch.z.batch_size();
    const auto& z = scratch.z;
    const auto& K = scratch.K;
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        batchlas::impl::parallel_for<StedcMatrixUpdate<B, T>>(h,
            sycl::nd_range<1>(batch_size * 128, 128),
            [=](sycl::nd_item<1> item) {
                auto bid  = item.get_group_linear_id();
                auto cta  = item.get_group();
                auto tid  = item.get_local_linear_id();
                auto bdim = item.get_local_range(0);

                const int dd = K[bid];
                auto Qbid = Qview.batch_item(bid);
                for(int eig = 0; eig < dd; ++eig)
                {
                    for(int i = tid; i < dd; i += static_cast<int>(bdim))
                    {
                        Qbid(i, eig) = z(i, bid) / Qbid(i, eig);
                    }

                    auto nrm2 = detail::nrm2(cta, Qview(Slice{0, dd}, eig));
                    for(int i = tid; i < dd; i += static_cast<int>(bdim))
                    {
                        Qbid(i, eig) /= nrm2;
                    }
                }
            });
    });
}

// lambda <- roots on [0, K); Pi sorts lambda (sorted by interlacing) together with the deflated d.
template <Backend B, typename T>
void final_order(Queue& ctx, const VectorView<T>& eigenvalues, const MergeScratch<T>& scratch) {
    const auto batch_size = eigenvalues.batch_size();
    const auto& K = scratch.K;
    const auto& roots = scratch.roots;
    ctx -> submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<StedcAssignEigenvalues<B, T>>(h, sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();

        for (int k = tid; k < K[bid]; k += bdim) {
            eigenvalues(k, bid) = roots(k, bid);
        }
        });
    });

    (void)argsort(ctx, eigenvalues, scratch.perm.final_order, SortOrder::Ascending, true);
    (void)permute(ctx, eigenvalues, scratch.perm.final_order);
}

// V = (Q P) U Pi by GEMM, with P the column map and U = [W 0; 0 I] (W is K x K). With a split
// destination, node p's V goes to out_{p&1} at batch index p/2.
template <Backend B, typename T>
void back_transform(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& eigvects,
                    const MatrixView<T, MatrixFormat::Dense>& Qp, const MatrixView<T, MatrixFormat::Dense>& tQ,
                    const MergeScratch<T>& scratch,
                    const MatrixView<T, MatrixFormat::Dense>& out_even, const MatrixView<T, MatrixFormat::Dense>& out_odd) {
    const int64_t n = scratch.z.size();
    const auto batch_size = scratch.z.batch_size();
    const auto& column_map = scratch.perm.column_map;
    const auto& pi = scratch.perm.final_order;
    if (out_even.data_ptr() != nullptr) {
        // The deflation-narrow variant below needs a host sync and an in-place
        // fold-back, neither of which fits a split destination. The saving it
        // buys is small next to not having to materialise the block-diagonal
        // input in the first place, which is what splitting achieves.
        (void)permuted_copy(ctx, Qp, tQ, pi);
        (void)permuted_copy(ctx, eigvects, Qp, column_map);
        const Span<const std::uint8_t> half_mask = out_even.batch_mask();
        for (int parity = 0; parity < 2; ++parity) {
            (void)gemm<B>(ctx, stedc_batch_parity(Qp, parity, half_mask), stedc_batch_parity(tQ, parity, half_mask),
                    parity == 0 ? out_even : out_odd,
                    T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        }
        return;
    }

    // Deflation-aware: with A = Q P, A U = [A(:, :K) W | A(:, K:)], so only the first K columns need a
    // GEMM. One batch-wide dd_max keeps a single uniform vendor call and stays exact (columns
    // K..dd_max-1 of U are identity). Reading dd_max costs a host sync, hence the size threshold.
    // evidence: docs/perf/stedc.md#stedc-deflation-aware-back-transform-gemm
    int64_t dd_max = n;
    if (n >= stedc_deflation_gemm_min_n) {
        ctx.wait();
        std::vector<int32_t> host_reduced(static_cast<size_t>(batch_size));
        ctx->memcpy(host_reduced.data(), scratch.K.data(), sizeof(int32_t) * host_reduced.size()).wait();
        const int32_t observed =
            host_reduced.empty() ? 0 : *std::max_element(host_reduced.begin(), host_reduced.end());
        dd_max = std::clamp<int64_t>(observed, 1, n);
    }

    if (dd_max <= static_cast<int64_t>(stedc_deflation_gemm_max_kept_fraction * static_cast<double>(n))) {
        // A -> temp_Q. eigvects is free afterwards and is reused as the narrow
        // GEMM output, so this needs no extra workspace.
        (void)permuted_copy(ctx, eigvects, tQ, column_map);
        auto product_head = eigvects(Slice{}, Slice{0, static_cast<int>(dd_max)});
        (void)gemm<B>(ctx,
                tQ,
                Qp(Slice{}, Slice{0, static_cast<int>(dd_max)}),
                product_head,
                GemmOptions<T>{});
        // Fold the multiplied head back over A so temp_Q holds the whole
        // product A*M; its tail columns are already correct.
        (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, tQ(Slice{}, Slice{0, static_cast<int>(dd_max)}), product_head);
        (void)permuted_copy(ctx, tQ, eigvects, pi);
    } else {
        // Avoid full-matrix copy + permute by using out-of-place permuted_copy in scratch buffers.
        (void)permuted_copy(ctx, Qp, tQ, pi);
        (void)permuted_copy(ctx, eigvects, Qp, column_map);
        (void)gemm<B>(ctx, Qp, tQ, eigvects, GemmOptions<T>{});
    }
}

template <Backend B, typename T>
void stedc_merge_step(Queue& ctx,
                      const VectorView<T>& eigenvalues,
                      const MatrixView<T, MatrixFormat::Dense>& eigvects,
                      const MatrixView<T, MatrixFormat::Dense>& Qprime,
                      const MatrixView<T, MatrixFormat::Dense>& temp_Q,
                      const Span<T>& rho,
                      const Span<std::byte>& ws,
                      int64_t m,
                      const StedcParams<T>& effective_params,
                      // Per-item status or nullptr. `info_nodes_per_item`: merge nodes per
                      // item in this launch (1 recursive, 2^l level-synchronous).
                      int32_t* info,
                      int64_t info_nodes_per_item,
                      const MatrixView<T, MatrixFormat::Dense>& out_even = MatrixView<T, MatrixFormat::Dense>(),
                      const MatrixView<T, MatrixFormat::Dense>& out_odd = MatrixView<T, MatrixFormat::Dense>())
{
    // The node mask (the call's, or the level's expanded one), carried by every view below.
    const Span<const std::uint8_t> mask = eigenvalues.batch_mask();
    const auto Qp = stedc_masked(Qprime, mask);
    const auto tQ = stedc_masked(temp_Q, mask);
    auto pool = BumpAllocator(ws);
    const auto scratch = merge_scratch<T>(ctx, pool, eigenvalues.size(), eigenvalues.batch_size(), mask);

    form_z<B, T>(ctx, eigvects, rho, scratch.z, m);
    order_poles(ctx, eigenvalues, scratch.z, scratch.perm.column_map);
    deflate<B, T>(ctx, eigvects, eigenvalues, rho, scratch);

    // U starts as I, so the deflated columns stay e_j; each solver fills the K x K block.
    (void)Qp.fill_identity(ctx);
    if (effective_params.secular_solver == StedcSecularSolver::Legacy) {
        (void)secular_solver(ctx, eigenvalues, scratch.z, Qprime, scratch.roots, scratch.K, rho, T(10.0), info, info_nodes_per_item);
    } else if (effective_params.merge_variant == StedcMergeVariant::Fused
            || effective_params.merge_variant == StedcMergeVariant::FusedCta) {
        // Roots, Löwner and U in one kernel, selected by merge_variant.
        stedc_merge_dispatch<B, T>(ctx, eigenvalues, scratch.z, rho, scratch.K, Qprime, scratch.roots, effective_params, info, info_nodes_per_item);
    } else {
        // Baseline (ROCm): one kernel per step.
        secular_roots<B, T>(ctx, eigenvalues, Qprime, rho, scratch, info, info_nodes_per_item);
        loewner<B, T>(ctx, eigenvalues, Qprime, scratch);
        eigenvectors<B, T>(ctx, Qprime, scratch);
    }

    final_order<B, T>(ctx, eigenvalues, scratch);
    back_transform<B, T>(ctx, eigvects, Qp, tQ, scratch, out_even, out_odd);
}

template <Backend B, typename T>
Event stedc_impl(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects, const MatrixView<T, MatrixFormat::Dense>& temp_Q,
            Span<int32_t> info)
{
    auto n = d.size();
    auto batch_size = d.batch_size();
    auto effective_params = resolve_stedc_tuning<B, T>(n, params, ctx.device().type == DeviceType::GPU);

    // Ensure leaf subproblems return sorted eigenvalues so higher levels can safely
    // rely on the "children sorted" invariant.
    auto steqr_params = effective_params.leaf_steqr_params;
    steqr_params.sort = true;
    steqr_params.sort_order = SortOrder::Ascending;
    if (n <= effective_params.recursion_threshold){
        // steqr_dispatch, not steqr: this leaf must RAISE the caller's `info`, not clear
        // it; the two half-solves below accumulate through info_report's fetch_max.
        // `jobz` is EigenVectors here: stedc_dispatch turns NoEigenVectors into a
        // vector solve into its own Z, as the merges consume these leaf eigenvectors.
        return steqr_dispatch<B, T>(ctx, d, e, eigenvalues, ws, jobz, steqr_params, eigvects, info);
    }

    //Split the matrix into two halves
    int64_t m = n / 2;

    //When uneven the first half has size m x m and the second (m+1) x (m+1)
    auto d1 = d(Slice(0, m));
    auto e1 = e(Slice(0, m - 1));
    auto d2 = d(Slice(m, SliceEnd()));
    auto e2 = e(Slice(m, SliceEnd()));
    auto E1 = eigvects(Slice{0, m}, Slice(0, m));
    auto E2 = eigvects(Slice{m, SliceEnd()}, Slice(m, SliceEnd()));
    auto Q1 = temp_Q(Slice{0, m}, Slice(0, m));
    auto Q2 = temp_Q(Slice{m, SliceEnd()}, Slice(m, SliceEnd()));
    auto lambda1 = eigenvalues(Slice(0, m));
    auto lambda2 = eigenvalues(Slice(m, SliceEnd()));

    auto pool = BumpAllocator(ws);
    auto rho = pool.allocate<T>(ctx, batch_size);

    ctx -> parallel_for<StedcModifyDiagonal<B, T>>(sycl::range(batch_size), [=](sycl::id<1> idx) {
        // Tear at beta = e_{m-1}: d_{m-1} -= |beta|, d_m -= |beta|.
        auto ix = idx[0];
        if (!d1.active(ix)) return;  // I3: d is the caller's, and edited in place
        rho[ix] = e(m - 1, ix);
        d1(m - 1, ix) -= std::abs(rho[ix]);
        d2(0, ix) -= std::abs(rho[ix]);
    });

    //Scope this section: after the child recursions return, their workspace memory can be reused
    {
        auto pool = BumpAllocator(ws.subspan(BumpAllocator::allocation_size<T>(ctx, batch_size)));
        auto ws1 = pool.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, m, batch_size, jobz, params));
        auto ws2 = pool.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, n - m, batch_size, jobz, params));
        // Both halves are the SAME batch items, so both accumulate into the same
        // `info` slots; info_report's fetch_max is what makes that an OR.
        (void)stedc_impl<B, T>(ctx, d1, e1, lambda1, ws1, jobz, params, E1, Q1, info);
        (void)stedc_impl<B, T>(ctx, d2, e2, lambda2, ws2, jobz, params, E2, Q2, info);
    }

    //Once the children are done their workspace can be reused for the merge.
    auto merge_ws = pool.allocate<std::byte>(ctx, stedc_merge_step_workspace<T>(ctx, n, batch_size));
    MatrixView<T> Qprime = MatrixView<T>(pool.allocate<T>(ctx, n * n * batch_size).data(), n, n, n, n * n, batch_size);
    // One merge node per batch item on this driver, hence a divisor of 1.
    stedc_merge_step<B, T>(ctx, eigenvalues, eigvects, Qprime, temp_Q, rho, merge_ws, m, effective_params,
                           detail::info_ptr(info, batch_size), 1);
    return ctx.get_event();
}

// ---------------------------------------------------------------------------
// Level-synchronous ("flattened") divide and conquer: all nodes of a level are
// merged by one launch over nodes * batch_size work-groups (O(L) launches, not
// O(2^L)). Every level must be size-uniform, so the problem is padded from n to
// N = leaf * 2^L with a diagonal tail above the input's Gershgorin bound: padded
// eigenvalues sort last and the answer is the leading n x n block.
// The plan lives in stedc_levels_plan.hh so tests can assert on it without a device.
// evidence: docs/perf/stedc.md#stedc-the-level-synchronous-driver
// ---------------------------------------------------------------------------

template <Backend B, typename T> class StedcLevelMask;

// The flattened driver's scratch up to the leaf STEQR's, which takes the pool's remainder.
// One layout serves stedc_levels_impl and stedc_levels_workspace.
template <typename T> struct LevelScratch {
    Span<T> d, e, w;         // the padded tridiagonal and eigenvalues, N x bs
    Span<T> rho;             // every level's tear coefficients
    Span<T> top;             // the root's eigenvectors, when the driver owns them
    Span<T> odd, qprime, tempq;
    Span<std::byte> merge;
    Span<int32_t> leaf_status;
    Span<std::uint8_t> level_masks;
};

// `own_top`: allocate the root N x N buffer (the problem is padded, or the caller's matrix is not
// densely packed) instead of writing straight into the caller's matrix.
template <typename T>
LevelScratch<T> level_scratch(Queue& ctx, BumpAllocator& pool, const MergeTree& tree, bool own_top) {
    const size_t N = static_cast<size_t>(tree.N);
    const size_t bs = static_cast<size_t>(tree.bs);
    LevelScratch<T> s;
    s.d = pool.allocate<T>(ctx, N * bs);
    s.e = pool.allocate<T>(ctx, N * bs);
    s.w = pool.allocate<T>(ctx, N * bs);
    s.rho = pool.allocate<T>(ctx, tree.rho_count());
    if (own_top) s.top = pool.allocate<T>(ctx, N * N * bs);
    if (tree.L > 0) {
        // Odd levels ping-pong through their own buffer; level 1 is the largest.
        s.odd = pool.allocate<T>(ctx, N * (N / 2) * bs);
        // Two s x s x P scratch matrices, largest at the root.
        s.qprime = pool.allocate<T>(ctx, N * N * bs);
        s.tempq = pool.allocate<T>(ctx, N * N * bs);
    }
    size_t merge_bytes = 0;
    for (int32_t l = 0; l < tree.L; ++l) {
        const auto lv = tree.level(l);
        merge_bytes = std::max(merge_bytes, stedc_merge_step_workspace<T>(ctx, lv.node_size, lv.P));
    }
    s.merge = pool.allocate<std::byte>(ctx, merge_bytes);
    // Per leaf, not per item. Both are UNCONDITIONAL, like potrf's: the size never follows
    // `info` or the mask.
    s.leaf_status = pool.allocate<int32_t>(ctx, tree.level(tree.L).P);
    s.level_masks = pool.allocate<std::uint8_t>(ctx, tree.mask_count());
    return s;
}

template <Backend B, typename T>
size_t stedc_levels_workspace(Queue& ctx, const StedcLevelPlan& plan, size_t batch_size,
                              JobType jobz, const StedcParams<T>& params, bool own_top) {
    const auto tree = MergeTree::of(plan, static_cast<int64_t>(batch_size));
    // Leaf STEQR over every leaf of every batch item at once.
    const size_t leaf = static_cast<size_t>(plan.leaf);
    const size_t leaf_batch = static_cast<size_t>(tree.level(tree.L).P);
    auto d_leaf = VectorView<T>(nullptr, static_cast<int>(leaf), static_cast<int>(leaf_batch), 1, static_cast<int>(leaf));
    auto e_leaf = VectorView<T>(nullptr, static_cast<int>(std::max<size_t>(leaf, 2) - 1), static_cast<int>(leaf_batch), 1, static_cast<int>(leaf));
    auto w_leaf = VectorView<T>(nullptr, static_cast<int>(leaf), static_cast<int>(leaf_batch), 1, static_cast<int>(leaf));
    auto leaf_params = params.leaf_steqr_params;
    leaf_params.sort = true;
    leaf_params.sort_order = SortOrder::Ascending;
    // The merges consume the leaves' eigenvectors whatever the caller asked
    // for, so the leaf solve always computes them.
    (void)jobz;
    auto pool = BumpAllocator::measuring();
    (void)level_scratch<T>(ctx, pool, tree, own_top);
    (void)pool.allocate<std::byte>(
        ctx, steqr_buffer_size<T>(ctx, d_leaf, e_leaf, w_leaf, JobType::EigenVectors, leaf_params));
    return pool.required_bytes();
}

// True when `eigvects` is a densely packed n x n x batch block the root level can be
// solved in place in. Deeper levels never use it: their packed arrays straddle items.
template <typename T>
inline bool stedc_top_matrix_is_packed(const MatrixView<T, MatrixFormat::Dense>& eigvects, int64_t n) {
    return eigvects.ld() == static_cast<int>(n)
        && eigvects.stride() == static_cast<int>(n * n)
        && eigvects.rows() == static_cast<int>(n)
        && eigvects.cols() == static_cast<int>(n);
}

template <Backend B, typename T>
Event stedc_levels_impl(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                        const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                        JobType jobz, StedcParams<T> params,
                        const MatrixView<T, MatrixFormat::Dense>& eigvects,
                        const StedcLevelPlan& plan,
                        Span<int32_t> info)
{
    const int64_t n = d.size();
    const int64_t bs = d.batch_size();
    const int64_t N = plan.padded_n;
    const int32_t L = plan.levels;
    const int64_t leaf = plan.leaf;
    const MergeTree tree = MergeTree::of(plan, bs);
    const bool own_top = (N != n) || !stedc_top_matrix_is_packed<T>(eigvects, n);

    auto pool = BumpAllocator(ws);
    const auto scratch = level_scratch<T>(ctx, pool, tree, own_top);
    const Span<T> top_buf = own_top ? scratch.top : Span<T>(eigvects.data_ptr(), static_cast<size_t>(N) * N * bs);
    // In the caller's matrix the blocks of deeper levels straddle items (an inactive item's output).
    const auto level_data = [&](int32_t level) -> T* {
        return tree.buffer(level, top_buf.data(), scratch.odd.data(), scratch.qprime.data());
    };
    // The call's mask (d, e, w and eigvects carry it, see stedc_dispatch), expanded per level.
    const Span<const std::uint8_t> mask = d.batch_mask();
    const auto level_mask = [&](int32_t level) -> Span<const std::uint8_t> {
        if (mask.empty() || level == 0) return mask;
        return Span<const std::uint8_t>(scratch.level_masks.data() + tree.mask_offset(level), tree.level(level).P);
    };
    if (!mask.empty() && L > 0) {
        const std::uint8_t* src = mask.data();
        std::uint8_t* dst = scratch.level_masks.data();
        ctx->parallel_for<StedcLevelMask<B, T>>(sycl::range<1>(scratch.level_masks.size()), [=](sycl::id<1> idx) {
            const int64_t g = static_cast<int64_t>(idx[0]);
            const int32_t l = tree.mask_level(g);
            const int64_t p = g - tree.mask_offset(l);   // node p of item p >> l
            dst[g] = src[p >> l];
        });
    }

    T* dp_ptr = scratch.d.data();
    T* ep_ptr = scratch.e.data();
    T* w_ptr = scratch.w.data();

    // 1. Copy the tridiagonal into the padded, level-friendly layout. When the
    //    problem needs padding, the tail diagonal is placed above the
    //    Gershgorin bound so those eigenvalues sort past every real one.
    const bool needs_pad = (N > n);
    ctx->submit([&](sycl::handler& h) {
        auto scratch = sycl::local_accessor<T, 1>(sycl::range<1>(needs_pad ? n : 1), h);
        batchlas::impl::parallel_for<StedcLevelPad<B, T>>(h, sycl::nd_range<1>(bs * 128, 128), [=](sycl::nd_item<1> item) {
            const int bid = item.get_group_linear_id();
            if (!d.active(bid)) return;  // I1, before the barrier
            const int tid = item.get_local_linear_id();
            const int bdim = item.get_local_range(0);
            auto cta = item.get_group();
            const int64_t base = static_cast<int64_t>(bid) * N;

            for (int64_t i = tid; i < n; i += bdim) {
                dp_ptr[base + i] = d(i, bid);
            }
            for (int64_t i = tid; i < N; i += bdim) {
                ep_ptr[base + i] = (i < n - 1) ? e(i, bid) : T(0);
            }
            if (needs_pad) {
                for (int64_t i = tid; i < n; i += bdim) {
                    const T lo = (i > 0) ? std::abs(e(i - 1, bid)) : T(0);
                    const T hi = (i < n - 1) ? std::abs(e(i, bid)) : T(0);
                    scratch[i] = std::abs(d(i, bid)) + lo + hi;
                }
                sycl::group_barrier(cta);
                const T gersh = batchlas::portable::joint_reduce(cta,
                                                   util::get_raw_ptr(scratch),
                                                   util::get_raw_ptr(scratch) + n,
                                                   sycl::maximum<T>());
                const T big = T(2) * gersh + T(1);
                for (int64_t i = n + tid; i < N; i += bdim) {
                    dp_ptr[base + i] = big + static_cast<T>(i - n);
                }
            }
        });
    });

    if (L > 0) {
        // 2. Apply every level's rank-1 diagonal correction in one pass. Each
        //    interior index is a split boundary for at most one level, and the
        //    corrections are plain subtractions, so they compose in any order.
        ctx->parallel_for<StedcLevelSplit<B, T>>(sycl::range<1>(N * bs), [=](sycl::id<1> idx) {
            const int64_t g = idx[0];
            if (!d.active(g / N)) return;  // I3
            const int64_t i = g % N;
            const int64_t base = g - i;
            T sub = T(0);
            for (int32_t l = 0; l < L; ++l) {
                const int64_t s = N >> l;
                const int64_t half = s >> 1;
                if (((i + 1) % s) == half) sub += std::abs(ep_ptr[base + i]);
                if ((i % s) == half) sub += std::abs(ep_ptr[base + i - 1]);
            }
            dp_ptr[g] -= sub;
        });

        // 3. Gather every level's tear coefficient beta = e at its node's cut into one array.
        T* rho_ptr = scratch.rho.data();
        ctx->parallel_for<StedcLevelRho<B, T>>(sycl::range<1>(tree.rho_count()), [=](sycl::id<1> idx) {
            const int64_t g = idx[0];
            const auto [l, p] = tree.rho_slot(g);
            // Spelled out: tree.level(l) here costs 2 registers.
            const int64_t nodes = int64_t(1) << l;
            const int64_t b = p / nodes;
            const int64_t node = p % nodes;
            const int64_t s = tree.N >> l;
            rho_ptr[g] = ep_ptr[b * tree.N + node * s + (s >> 1) - 1];
        });
    }

    // Each merge writes directly into its parent's two diagonal sub-blocks; only the
    // off-diagonal zero blocks are materialised here (deflation's Givens rotations
    // mix columns across the split, so they must be zero, not stale).
    // evidence: docs/perf/stedc.md#stedc-the-level-synchronous-driver
    const auto zero_offdiagonal = [&](int32_t level) {
        const auto [s, half, nodes, P] = tree.level(level);
        auto Z = stedc_masked(MatrixView<T>(level_data(level), s, s, s, s * s, P), level_mask(level));
        ctx->submit([&](sycl::handler& h) {
            auto Zv = Z.kernel_view();
            h.parallel_for<StedcLevelZeroOffdiag<B, T>>(sycl::range<1>(half * half * 2 * P), [=](sycl::id<1> idx) {
                const int64_t g = idx[0];
                const int64_t p = g / (half * half * 2);
                if (!Zv.active(p)) return;  // I3
                const int64_t r = g - p * half * half * 2;
                const int64_t corner = r / (half * half);   // 0: upper-right, 1: lower-left
                const int64_t e = r - corner * half * half;
                const int64_t col = e / half;
                const int64_t row = e - col * half;
                Zv(row + (corner == 0 ? 0 : half), col + (corner == 0 ? half : 0), p) = T(0);
            });
        });
    };

    // Assemble one level's blocks from the packed child array below it.
    const auto gather_blockdiag = [&](int32_t level, int64_t child_n) {
        const auto [s, half, nodes, P] = tree.level(level);
        auto Zp = stedc_masked(MatrixView<T>(level_data(level), s, s, s, s * s, P), level_mask(level));
        auto Zc = MatrixView<T>(level_data(level + 1), child_n, child_n, child_n, child_n * child_n, 2 * P);
        ctx->submit([&](sycl::handler& h) {
            auto parent = Zp.kernel_view();
            auto child = Zc.kernel_view();
            h.parallel_for<StedcLevelBlockDiag<B, T>>(sycl::range<1>(s * s * P), [=](sycl::id<1> idx) {
                const int64_t g = idx[0];
                const int64_t p = g / (s * s);
                if (!parent.active(p)) return;  // I3
                const int64_t r = g - p * s * s;
                const int64_t col = r / s;
                const int64_t row = r - col * s;
                T val = T(0);
                if (row < half) {
                    if (col < half) val = child(row, col, 2 * p);
                } else if (col >= half) {
                    val = child(row - half, col - half, 2 * p + 1);
                }
                parent(row, col, p) = val;
            });
        });
    };

    // 4. Solve every leaf of every batch item in a single STEQR call, then copy the
    //    leaves into the deepest merge level (in place is slower: leaf ld doubles).
    //    evidence: docs/perf/stedc.md#stedc-the-level-synchronous-driver
    {
        const int64_t leaf_batch = tree.level(L).P;
        auto leaf_params = params.leaf_steqr_params;
        leaf_params.sort = true;
        leaf_params.sort_order = SortOrder::Ascending;
        const auto leaf_mask = level_mask(L);
        auto d_leaf = stedc_masked(VectorView<T>(dp_ptr, leaf, leaf_batch, 1, leaf), leaf_mask);
        auto e_leaf = stedc_masked(VectorView<T>(ep_ptr, std::max<int64_t>(leaf, 2) - 1, leaf_batch, 1, leaf), leaf_mask);
        auto w_leaf = stedc_masked(VectorView<T>(w_ptr, leaf, leaf_batch, 1, leaf), leaf_mask);
        auto z_leaf = stedc_masked(MatrixView<T>(level_data(L), leaf, leaf, leaf, leaf * leaf, leaf_batch), leaf_mask);
        auto leaf_ws = pool.remaining();
        // Always compute leaf eigenvectors: the merges consume them even when
        // the caller only wants eigenvalues.
        //
        // THE LEAF AXIS IS NOT THE BATCH AXIS. This call solves 2^L * bs problems, so
        // the caller's `info` (length bs) would alias and over-run. It gets its own
        // per-leaf array, folded afterwards: leaf j belongs to item j / 2^L (node p's
        // children are 2p and 2p+1). steqr_dispatch, because steqr would clear the span.
        const bool want_leaf_status = (detail::info_ptr(info, bs) != nullptr);
        Span<int32_t> leaf_info;
        if (want_leaf_status) {
            leaf_info = Span<int32_t>(scratch.leaf_status.data(), static_cast<size_t>(leaf_batch));
            ctx->memset(leaf_info.data(), 0, sizeof(int32_t) * static_cast<size_t>(leaf_batch));
        }
        (void)steqr_dispatch<B, T>(ctx, d_leaf, e_leaf, w_leaf, leaf_ws, JobType::EigenVectors, leaf_params, z_leaf, leaf_info);
        if (want_leaf_status) {
            int32_t* const out = detail::info_ptr(info, bs);
            const int32_t* const leaves_in = leaf_info.data();
            const int64_t leaves_per_item = leaf_batch / bs;
            const std::uint8_t* const leaf_on = batchlas::detail::mask_ptr(leaf_mask);
            (void)ctx->parallel_for<StedcLevelLeafStatus<B, T>>(
                sycl::range<1>(static_cast<size_t>(leaf_batch)), [=](sycl::id<1> idx) {
                    const int64_t j = static_cast<int64_t>(idx[0]);
                    if (leaf_on && !leaf_on[j]) return;  // I3
                    detail::info_report(out, detail::info_item(j, leaves_per_item), leaves_in[j]);
                });
        }
        gather_blockdiag(L - 1, leaf);
    }

    // 5. Merge one whole level at a time, root-ward.
    for (int32_t l = L - 1; l >= 0; --l) {
        const auto [s, half, k, P] = tree.level(l);

        auto Zp = stedc_masked(MatrixView<T>(level_data(l), s, s, s, s * s, P), level_mask(l));
        auto lambda = stedc_masked(VectorView<T>(w_ptr, s, P, 1, s), level_mask(l));
        auto rho_level = Span<T>(scratch.rho.data() + tree.rho_offset(l), P);
        auto Qprime = MatrixView<T>(scratch.qprime.data(), s, s, s, s * s, P);
        auto temp_Q = MatrixView<T>(scratch.tempq.data(), s, s, s, s * s, P);
        const auto level_params = resolve_stedc_tuning<B, T>(s, params, ctx.device().type == DeviceType::GPU);

        // At level l one launch merges k = 2^l sibling nodes per batch item, so the
        // kernel's node index is divided by k before it indexes `info`.
        int32_t* const info_out = detail::info_ptr(info, bs);
        if (l == 0) {
            stedc_merge_step<B, T>(ctx, lambda, Zp, Qprime, temp_Q, rho_level, scratch.merge, half, level_params,
                                   info_out, k);
        } else {
            // Hand the result straight to the parent's diagonal sub-blocks.
            const int64_t sp = s * 2;   // parent node size
            zero_offdiagonal(l - 1);
            T* pbase = level_data(l - 1);
            auto out_even = stedc_masked(MatrixView<T>(pbase, s, s, sp, sp * sp, P / 2), level_mask(l - 1));
            auto out_odd = stedc_masked(MatrixView<T>(pbase + s * (sp + 1), s, s, sp, sp * sp, P / 2), level_mask(l - 1));
            stedc_merge_step<B, T>(ctx, lambda, Zp, Qprime, temp_Q, rho_level, scratch.merge, half, level_params,
                                   info_out, k,
                                   out_even, out_odd);
        }
    }

    // 6. Hand back the leading n eigenvalues (and, when we solved a padded or
    //    non-packed problem, the leading n x n eigenvector block).
    ctx->parallel_for<StedcLevelUnpad<B, T>>(sycl::range<1>(n * bs), [=](sycl::id<1> idx) {
        const int64_t g = idx[0];
        const int64_t b = g / n;
        if (!eigenvalues.active(b)) return;  // I3
        const int64_t i = g - b * n;
        eigenvalues(i, b) = w_ptr[b * N + i];
    });
    if (own_top && eigvects.data_ptr() != nullptr) {
        auto top = MatrixView<T>(top_buf.data(), N, N, N, N * N, bs);
        (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, eigvects, top(Slice{0, n}, Slice{0, n}));
    }
    return ctx.get_event();
}

// NETLIB stedc is a sorted host steqr on steqr_legacy, which replays only the sweeps each block performed (steqr_wg replays every slot), so its Z bits differ from steqr_wg's.
template <typename T>
SteqrParams<T> stedc_netlib_steqr_params(const StedcParams<T>& params) {
    auto steqr_params = params.leaf_steqr_params;
    steqr_params.sort = true;
    steqr_params.sort_order = SortOrder::Ascending;
    steqr_params.back_transform = false;
    return steqr_params;
}

template <Backend B, typename T>
Event stedc_netlib(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues,
                   const Span<std::byte>& ws, JobType jobz, const StedcParams<T>& params,
                   const MatrixView<T, MatrixFormat::Dense>& eigvects, Span<int32_t> info) {
    return steqr_legacy<B, T>(ctx, d, e, eigenvalues, ws, jobz, stedc_netlib_steqr_params(params), eigvects, info);
}

template <typename T>
size_t stedc_netlib_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, const StedcParams<T>& params) {
    auto evals = VectorView<T>(nullptr, n, batch_size, 1, 0);
    auto diag = VectorView<T>(nullptr, n, batch_size, 1, 0);
    auto offdiag = VectorView<T>(nullptr, n - 1, batch_size, 1, 0);
    return steqr_legacy_buffer_size<T>(ctx, diag, offdiag, evals, jobz, stedc_netlib_steqr_params(params));
}

template <Backend B, typename T>
Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects,
            Span<int32_t> info)
{
    const auto mask = batchlas::detail::call_mask("stedc", d, e, eigenvalues, eigvects);
    if (d.size() != e.size() + 1) {
        throw batchlas::invalid_argument("The size of e must be one less than the size of d.");
    }
    if (d.size() != eigenvalues.size()) {
        throw batchlas::invalid_argument("The size of eigenvalues must match the size of d.");
    }
    if (d.batch_size() != e.batch_size() || d.batch_size() != eigenvalues.batch_size()) {
        throw batchlas::invalid_argument("The batch sizes of d, e, and eigenvalues must match.");
    }
    if (jobz == JobType::EigenVectors) {
        if (eigvects.rows() != d.size() || eigvects.cols() != d.size() || eigvects.batch_size() != d.batch_size()) {
            throw batchlas::invalid_argument("The dimensions of eigvects must match the size of d and its batch size.");
        }
    }

    // The single clear for the whole call, here and nowhere below: every driver,
    // leaf and merge under this point only ever RAISES a status, which is what lets
    // the recursive driver's two half-solves and the level driver's L merges all
    // write the same slots. See src/kernels/common/info_span.hh.
    detail::info_clear(ctx, info, d.batch_size(), mask);
    return stedc_dispatch<B, T>(ctx, d, e, eigenvalues, ws, jobz, params, eigvects, info);
}

// Everything after the clear; see stedc_internal.hh for who calls this directly.
template <Backend B, typename T>
Event stedc_dispatch(Queue& ctx, const VectorView<T>& d_in, const VectorView<T>& e_in, const VectorView<T>& eigenvalues_in, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects_in,
            Span<int32_t> info)
{
    // Every operand takes the call's mask, so each driver below reads it off any view.
    const auto mask = batchlas::detail::call_mask("stedc", d_in, e_in, eigenvalues_in, eigvects_in);
    const auto d = stedc_masked(d_in, mask);
    const auto e = stedc_masked(e_in, mask);
    const auto eigenvalues = stedc_masked(eigenvalues_in, mask);
    const auto eigvects = stedc_masked(eigvects_in, mask);
    if constexpr (B == Backend::NETLIB) {
        return stedc_netlib<B, T>(ctx, d, e, eigenvalues, ws, jobz, params, eigvects, info);
    }
    const auto n = d.size();
    // Values only: every merge builds its secular vector from the children's eigenvectors, which a
    // NoEigenVectors leaf never writes, and the caller's eigvects may be empty. Solve with vectors
    // into a workspace Z instead; the caller's matrix is never touched.
    // evidence: docs/perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors
    if (jobz == JobType::NoEigenVectors && n > 0) {
        const size_t z_count = static_cast<size_t>(n) * n * d.batch_size();
        auto pool = BumpAllocator(ws);
        const auto z = stedc_masked(
            MatrixView<T>(pool.allocate<T>(ctx, z_count).data(), n, n, n, n * n, d.batch_size()), mask);
        return stedc_dispatch<B, T>(ctx, d, e, eigenvalues, ws.subspan(BumpAllocator::allocation_size<T>(ctx, z_count)),
                                    JobType::EigenVectors, params, z, info);
    }

    const auto effective_params = resolve_stedc_tuning<B, T>(n, params, ctx.device().type == DeviceType::GPU);
    if (params.algorithm != StedcAlgorithm::Recursive) {
        const auto plan = plan_stedc_levels(n, effective_params.recursion_threshold);
        // A padded problem always needs its own top-level buffer, which the
        // workspace was sized for. An unpadded one only avoids that buffer when
        // the caller's matrix is densely packed, so a non-packed view (which the
        // workspace was *not* sized for) falls back to the recursive driver.
        if (plan.levels > 0 && (plan.padded_n != n || stedc_top_matrix_is_packed<T>(eigvects, n))) {
            return stedc_levels_impl<B, T>(ctx, d, e, eigenvalues, ws, jobz, params, eigvects, plan, info);
        }
    }

    //Clean the output matrix before we begin.
    (void)eigvects.fill_zeros(ctx);
    auto pool = BumpAllocator(ws);
    auto alloc_size = BumpAllocator::allocation_size<T>(ctx, n * n * d.batch_size());
    auto temp_Q = stedc_masked(MatrixView<T>(pool.allocate<T>(ctx, n * n * d.batch_size()).data(), n, n, n, n * n, d.batch_size()), mask);
    return stedc_impl<B, T>(ctx, d, e, eigenvalues, ws.subspan(alloc_size), jobz, params, eigvects, temp_Q, info);

}

template <Backend B, typename T>
size_t stedc_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    if constexpr (B == Backend::NETLIB) {
        return stedc_netlib_buffer_size<T>(ctx, n, batch_size, jobz, params);
    }
    if (jobz == JobType::NoEigenVectors) {
        // stedc_dispatch's workspace Z, then the vector solve it runs.
        return BumpAllocator::allocation_size<T>(ctx, n * n * batch_size)
               + stedc_buffer_size<B, T>(ctx, n, batch_size, JobType::EigenVectors, params);
    }

    const size_t recursive_bytes =
        stedc_internal_workspace_size<B, T>(ctx, n, batch_size, jobz, params)
        + BumpAllocator::allocation_size<T>(ctx, n * n * batch_size);

    if (params.algorithm == StedcAlgorithm::Recursive) {
        return recursive_bytes;
    }

    const auto effective_params = resolve_stedc_tuning<B, T>(static_cast<int64_t>(n), params,
                                                             ctx.device().type == DeviceType::GPU);
    const auto plan = plan_stedc_levels(static_cast<int64_t>(n), effective_params.recursion_threshold);
    if (plan.levels == 0) {
        return recursive_bytes;
    }
    // A padded problem needs its own N x N top-level buffer; an unpadded one
    // writes straight into the caller's matrix. `stedc` keeps that decision in
    // sync -- it sends a non-packed unpadded case down the recursive path,
    // whose size is covered by the max below.
    const bool own_top = (plan.padded_n != static_cast<int64_t>(n));
    const size_t level_bytes = stedc_levels_workspace<B, T>(ctx, plan, batch_size, jobz, params, own_top);
    return std::max(level_bytes, recursive_bytes);
}


template <Backend B, typename T>
size_t stedc_internal_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    params = resolve_stedc_tuning<B, T>(static_cast<int64_t>(n), params,
                                        ctx.device().type == DeviceType::GPU);

    if (n <= static_cast<size_t>(params.recursion_threshold)) {
        auto d_leaf = VectorView<T>(nullptr, static_cast<int64_t>(n), batch_size, 1, 0);
        auto e_leaf = VectorView<T>(nullptr, static_cast<int64_t>(n - 1), batch_size, 1, 0);
        auto eigenvalues_leaf = VectorView<T>(nullptr, static_cast<int64_t>(n), batch_size, 1, 0);
        return steqr_buffer_size<T>(ctx, d_leaf, e_leaf, eigenvalues_leaf, jobz, params.leaf_steqr_params);
    }

    // stedc_impl's two phases over one pool: beta, then the halves' workspaces; once they
    // return, beta, then the merge scratch and Qprime.
    const size_t m = n / 2;
    auto children = BumpAllocator::measuring();
    (void)children.allocate<T>(ctx, batch_size);
    (void)children.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, m, batch_size, jobz, params));
    (void)children.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, n - m, batch_size, jobz, params));
    auto merge = BumpAllocator::measuring();
    (void)merge.allocate<T>(ctx, batch_size);
    (void)merge.allocate<std::byte>(ctx, stedc_merge_step_workspace<T>(ctx, n, batch_size));
    (void)merge.allocate<T>(ctx, n * n * batch_size);
    return std::max(children.required_bytes(), merge.required_bytes());
}

#define STEDC_INSTANTIATE(back, fp) \
template Event stedc_dispatch<back, BATCHLAS_UNPAREN fp>(Queue& ctx, const VectorView<BATCHLAS_UNPAREN fp>& d, const VectorView<BATCHLAS_UNPAREN fp>& e, const VectorView<BATCHLAS_UNPAREN fp>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<BATCHLAS_UNPAREN fp> params, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>& eigvects, Span<int32_t> info); \
template Event stedc<back, BATCHLAS_UNPAREN fp>(Queue& ctx, const VectorView<BATCHLAS_UNPAREN fp>& d, const VectorView<BATCHLAS_UNPAREN fp>& e, const VectorView<BATCHLAS_UNPAREN fp>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<BATCHLAS_UNPAREN fp> params, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>& eigvects, Span<int32_t> info); \
template size_t stedc_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<BATCHLAS_UNPAREN fp> params);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEDC_INSTANTIATE)


#undef STEDC_INSTANTIATE


}

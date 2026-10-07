#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include "../sort.hh"
#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>
#include "../math-helpers.hh"
#include "../util/template-instantiations.hh"
#include "steqr_internal.hh"
#include "stedc_secular.hh"
#include "stedc_merge_kernels.hh"
#include "stedc_levels_plan.hh"
#include "info_span.hh"
#include "stedc_internal.hh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#define DEBUG_STEDC 0

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

// ---------------------------------------------------------------------------
// One divide-and-conquer merge, applied to a *super-batch* of P size-uniform
// sub-problems of size s whose halves are already solved. The recursive driver
// passes P = batch_size; the level-synchronous one P = nodes_at_level * batch_size.
//
//   eigenvalues  s x P, in/out: children's eigenvalues in, merged ones out
//   eigvects     s x s x P, in/out: block-diagonal diag(Q_left, Q_right) in,
//                merged eigenvectors out
//   Qprime, temp_Q   s x s x P scratch
//   rho          P signed rank-1 coefficients (the split off-diagonal element)
//   m            size of the left half
// evidence: docs/perf/stedc.md#stedc-the-level-synchronous-driver
// ---------------------------------------------------------------------------
// Select the even- or odd-indexed half of a batched view. Both halves stay
// affine (stride doubles), so a level can write straight into its parent's two
// diagonal sub-blocks with strided-batched GEMMs.
template <typename T>
inline MatrixView<T, MatrixFormat::Dense> stedc_batch_parity(const MatrixView<T, MatrixFormat::Dense>& m, int parity) {
    return MatrixView<T, MatrixFormat::Dense>(m.data_ptr() + parity * m.stride(),
                                              m.rows(), m.cols(), m.ld(),
                                              m.stride() * 2, m.batch_size() / 2);
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
    // When a split destination is supplied, node p's merged block is written to
    // out_{p&1} at batch index p/2 instead of back into `eigvects`.
    const bool split_output = out_even.data_ptr() != nullptr;
    const int64_t n = eigenvalues.size();
    const auto batch_size = eigenvalues.batch_size();

    auto E1 = eigvects(Slice{0, m}, Slice(0, m));
    auto E2 = eigvects(Slice{m, SliceEnd()}, Slice(m, SliceEnd()));

    auto pool = BumpAllocator(ws);
    auto permutation = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    // Persistent mapping from logical (current) column order -> physical column in `eigvects`.
    // We avoid physically permuting the eigenvector matrix until right before GEMM / function exit.
    auto perm_map = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    auto v = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, batch_size);

    ctx -> submit([&](sycl::handler& h) {
        auto E1view = E1.kernel_view();
        auto E2view = E2.kernel_view();
        h.parallel_for<StedcComputeV<B, T>>(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            // When rho = e(m-1) is negative, the rank-1 perturbation has the
            // form |rho| * u * u^T with u = e_{m-1} + sign(rho) * e_m (so that
            // the diagonal corrections d1[m-1] -= |rho|, d2[0] -= |rho| match
            // the off-diagonal value rho). The corresponding secular vector
            // projected into the child-eigenvector basis therefore carries a
            // sign(rho) factor on the components coming from Q2 (the second
            // half of v). Without this sign flip the computed eigenvectors
            // combine Q1 and Q2 columns with inconsistent signs and fail
            // A*z = lambda*z even though A*z is numerically close to Z*Lambda.
            const T v_sign2 = (rho[bid] >= T(0)) ? T(1) : T(-1);
            //Normalized v through division by sqrt(2)
            for (int i = tid; i < m; i += bdim) {
                v(i, bid) = E1view(m - 1, i, bid) / std::sqrt(T(2));
            }
            for (int i = tid; i < n - m; i += bdim) {
                v(i + m, bid) = v_sign2 * E2view(0, i, bid) / std::sqrt(T(2));
            }
        });
    });
    // (void) on an Event: deliberate. This Queue is in-order, so the next submission
    // is already ordered after this one and the Event carries nothing the caller needs.
    (void)argsort(ctx, eigenvalues, perm_map, SortOrder::Ascending, true);
    (void)permute(ctx, eigenvalues, perm_map);
    (void)permute(ctx, v, perm_map);

    auto keep_indices = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    auto n_reduced = pool.allocate<int32_t>(ctx, batch_size);
    //Deflation scheme
    ctx -> submit([&](sycl::handler& h) {
        auto Q = eigvects.kernel_view();
        auto perm_local = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_include = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_exclude = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto norm_mem = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        h.parallel_for<StedcDeflation<B, T>>(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();
        auto cta = item.get_group();

        for (int k = tid; k < n; k += bdim){
            keep_indices(k, bid) = 0;
            scan_mem_exclude[k] = 0;
            permutation(k, bid) = -1;
            perm_local[k] = perm_map(k, bid);
        }

        sycl::group_barrier(cta);

        // LAPACK-style ABSOLUTE deflation tolerance tol = 8*eps*max(|D|_inf, |z|_inf),
        // computed before the Givens loop so proximity deflation uses it too. A relative
        // tolerance under-deflates clustered small eigenvalues (bad orthogonality).
        // evidence: docs/perf/stedc.md#stedc-the-absolute-deflation-tolerance
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(eigenvalues(k, bid)); }
        auto eig_max = sycl::joint_reduce(cta,
                          util::get_raw_ptr(norm_mem),
                          util::get_raw_ptr(norm_mem) + n,
                          sycl::maximum<T>());
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(v(k, bid)); }
        auto v_max_pre = sycl::joint_reduce(cta,
                          util::get_raw_ptr(norm_mem),
                          util::get_raw_ptr(norm_mem) + n,
                          sycl::maximum<T>());
        const T abs_tol = T(8.0) * std::numeric_limits<T>::epsilon() * std::max(eig_max, v_max_pre);

        for (int j = 0; j < n - 1; j++) {
            if(std::abs(eigenvalues(j + 1, bid) - eigenvalues(j, bid)) <= abs_tol) {
                auto f = v(j + 1, bid);
                auto g = v(j, bid);
                auto [c, s, r] = internal::lartg(f, g);
                sycl::group_barrier(cta);
                if (tid == 0) {
                    v(j, bid) = T(0.0);
                    v(j + 1, bid) = r;
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
                // Make tid 0's writes to v(j, bid) and v(j+1, bid) visible to all threads
                // before the next iteration reads v(j+1, bid) to compute its Givens pair.
                sycl::group_barrier(cta);
            }
        }

        sycl::group_barrier(cta);
        //LAPACK LAED8 based tolerance (small-|z| deflation), using same absolute tolerance.

        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(eigenvalues(k, bid)); }
        auto eig_max2 = sycl::joint_reduce(cta,
                          util::get_raw_ptr(norm_mem),
                          util::get_raw_ptr(norm_mem) + n,
                          sycl::maximum<T>());

        auto v_norm = internal::nrm2<T>(cta, v);
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(v(k, bid) / v_norm); }
        auto v_max = sycl::joint_reduce(cta,
                        util::get_raw_ptr(norm_mem),
                        util::get_raw_ptr(norm_mem) + n,
                        sycl::maximum<T>());
        auto tol = T(8.0) * std::numeric_limits<T>::epsilon() * std::max(eig_max2, v_max);

        for (int k = tid; k < n; k += bdim) {
            if (std::abs(rho[bid] * norm_mem[k]) > tol ) {
                keep_indices(k, bid) = 1;
            } else {
                scan_mem_exclude[k] = 1;
            }
        }

        sycl::group_barrier(cta);

        //Exclusive scan to determine the indices to keep
        sycl::joint_exclusive_scan(cta,
                       keep_indices.batch_item(bid).data_ptr(),
                       keep_indices.batch_item(bid).data_ptr() + n,
                       util::get_raw_ptr(scan_mem_include),
                       0,
                       sycl::plus<int32_t>());
        sycl::joint_exclusive_scan(cta,
                       util::get_raw_ptr(scan_mem_exclude),
                       util::get_raw_ptr(scan_mem_exclude) + n,
                       util::get_raw_ptr(scan_mem_exclude),
                       0,
                       sycl::plus<int32_t>());

        for (int k = tid; k < n; k += bdim) {
            if (keep_indices(k, bid) == 1) {
                permutation(scan_mem_include[k], bid) = k;
            } else {
                permutation(n - 1 - scan_mem_exclude[k], bid) = k;
            }
        }

        if (tid == 0) {
            n_reduced[bid] = scan_mem_include[n - 1] + keep_indices(n - 1, bid);
        }

        });
    });

    // Apply deflation permutation to contiguous vectors.
    (void)permute(ctx, eigenvalues, permutation);
    (void)permute(ctx, v, permutation);

    // Update the logical->physical column map instead of physically permuting the eigenvector matrix.
    // This composes the current column map with the deflation permutation.
    (void)permute(ctx, perm_map, permutation);

    auto temp_lambdas = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, batch_size);
    (void)Qprime.fill_identity(ctx);
    if (effective_params.secular_solver == StedcSecularSolver::Legacy) {
        (void)secular_solver(ctx, eigenvalues, v, Qprime, temp_lambdas, n_reduced, rho, T(10.0), info, info_nodes_per_item);
    } else if (effective_params.merge_variant == StedcMergeVariant::Fused
            || effective_params.merge_variant == StedcMergeVariant::FusedCta) {
        // Fused merge paths: single-kernel implementations selected by merge_variant.
        stedc_merge_dispatch<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, effective_params, info, info_nodes_per_item);
    } else {
        // Baseline ROCm path: 3 separate kernels
        //
        // Locals so the kernel's `[=]` copies a pointer and a scalar; nullptr makes
        // info_report a no-op.
        int32_t* const info_dev = info;
        const int64_t nodes_per_item = info_nodes_per_item;
        ctx -> submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for<StedcSecularSolve<B, T>>(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto bdim = item.get_local_range(0);
                auto tid = item.get_local_linear_id();
                auto cta = item.get_group();
                auto Q_bid = Qview.batch_item(bid);
                auto n = n_reduced[bid];
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
                        temp_lambdas(k, bid) = sec_solve_ext_roc(n, dview, v.batch_item(bid), std::abs(2 * rho[bid]), root_converged);
                    } else {
                        temp_lambdas(k, bid) = sec_solve_roc(n, dview, v.batch_item(bid), std::abs(2 * rho[bid]), k, root_converged);
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

        // Rescale v (secular vector) to avoid bad numerics when an eigenvalue
        // is too close to a pole. This mirrors ROCm's stedc_mergeValues_Rescale_kernel
        // but uses SYCL group collectives for the product reduction.
        ctx -> submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for<StedcRescaleV<B, T>>(
                sycl::nd_range<1>(batch_size * 128, 128),
                [=](sycl::nd_item<1> item) {
                    auto bid = item.get_group_linear_id();
                    auto g   = item.get_group();
                    auto tid = item.get_local_linear_id();
                    auto bdim = item.get_local_range(0);
                    auto Qbid = Qview.batch_item(bid);
                    auto dd = n_reduced[bid];

                    // Löwner-rescale z_tilde via the ~dd-term ratio product, in native T
                    // (double accumulation is unnecessary with the absolute tolerance).
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

                        T valf = sycl::reduce_over_group(g, partial, sycl::multiplies<T>());
                        if(tid == 0)
                        {
                            T mag  = std::sqrt(std::fabs(valf));
                            T sign = v(eid, bid) >= T(0) ? T(1) : T(-1);
                            v(eid, bid) = sign * mag;
                        }
                    }
                });
        });

        ctx -> submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for<StedcMatrixUpdate<B, T>>(
                sycl::nd_range<1>(batch_size * 128, 128),
                [=](sycl::nd_item<1> item) {
                    auto bid  = item.get_group_linear_id();
                    auto cta  = item.get_group();
                    auto tid  = item.get_local_linear_id();
                    auto bdim = item.get_local_range(0);

                    const int dd = n_reduced[bid];
                    auto Qbid = Qview.batch_item(bid);
                    for(int eig = 0; eig < dd; ++eig)
                    {
                        for(int i = tid; i < dd; i += static_cast<int>(bdim))
                        {
                            Qbid(i, eig) = v(i, bid) / Qbid(i, eig);
                        }

                        auto nrm2 = internal::nrm2(cta, Qview(Slice{0, dd}, eig));
                        for(int i = tid; i < dd; i += static_cast<int>(bdim))
                        {
                            Qbid(i, eig) /= nrm2;
                        }
                    }
                });
        });
    }

    ctx -> submit([&](sycl::handler& h) {
        h.parallel_for<StedcAssignEigenvalues<B, T>>(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();

        for (int k = tid; k < n_reduced[bid]; k += bdim) {
            eigenvalues(k, bid) = temp_lambdas(k, bid);
        }
        });
    });

    (void)argsort(ctx, eigenvalues, permutation, SortOrder::Ascending, true);
    (void)permute(ctx, eigenvalues, permutation);

    // Deflation-aware back-transform. Qprime is M = [W | I], and
    //     eigvects = A * M[:, perm] = (A * M)[:, perm],   A * M = [A*W | A(:, dd:)],
    // so only the first dd columns need a GEMM. One batch-wide dd_max keeps a single
    // uniform vendor call and stays exact (columns dd..dd_max-1 of M are identity).
    // Reading dd_max costs a host sync, hence the size threshold.
    // evidence: docs/perf/stedc.md#stedc-deflation-aware-back-transform-gemm
    if (split_output) {
        // The deflation-narrow variant below needs a host sync and an in-place
        // fold-back, neither of which fits a split destination. The saving it
        // buys is small next to not having to materialise the block-diagonal
        // input in the first place, which is what splitting achieves.
        (void)permuted_copy(ctx, Qprime, temp_Q, permutation);
        (void)permuted_copy(ctx, eigvects, Qprime, perm_map);
        for (int parity = 0; parity < 2; ++parity) {
            (void)gemm<B>(ctx, stedc_batch_parity(Qprime, parity), stedc_batch_parity(temp_Q, parity),
                    parity == 0 ? out_even : out_odd,
                    T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        }
        return;
    }

    int64_t dd_max = n;
    if (n >= stedc_deflation_gemm_min_n) {
        ctx.wait();
        int32_t observed = 0;
        for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b) {
            observed = std::max(observed, n_reduced[b]);
        }
        dd_max = std::clamp<int64_t>(observed, 1, n);
    }

    if (dd_max <= static_cast<int64_t>(stedc_deflation_gemm_max_kept_fraction * static_cast<double>(n))) {
        // A -> temp_Q. eigvects is free afterwards and is reused as the narrow
        // GEMM output, so this needs no extra workspace.
        (void)permuted_copy(ctx, eigvects, temp_Q, perm_map);
        auto product_head = eigvects(Slice{}, Slice{0, static_cast<int>(dd_max)});
        (void)gemm<B>(ctx,
                temp_Q,
                Qprime(Slice{}, Slice{0, static_cast<int>(dd_max)}),
                product_head,
                GemmOptions<T>{});
        // Fold the multiplied head back over A so temp_Q holds the whole
        // product A*M; its tail columns are already correct.
        (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, temp_Q(Slice{}, Slice{0, static_cast<int>(dd_max)}), product_head);
        (void)permuted_copy(ctx, temp_Q, eigvects, permutation);
    } else {
        // Avoid full-matrix copy + permute by using out-of-place permuted_copy in scratch buffers.
        (void)permuted_copy(ctx, Qprime, temp_Q, permutation);
        (void)permuted_copy(ctx, eigvects, Qprime, perm_map);
        (void)gemm<B>(ctx, Qprime, temp_Q, eigvects, GemmOptions<T>{});
    }
}

// Scratch bytes `stedc_merge_step` sub-allocates for a size-`s` merge over `P`
// sub-problems (everything except the two s x s scratch matrices, which the
// caller owns).
template <typename T>
size_t stedc_merge_step_workspace(Queue& ctx, size_t s, size_t P) {
    return BumpAllocator::allocation_size<int32_t>(ctx, s * P)   // permutation
         + BumpAllocator::allocation_size<int32_t>(ctx, s * P)   // perm_map
         + BumpAllocator::allocation_size<T>(ctx, s * P)         // v
         + BumpAllocator::allocation_size<int32_t>(ctx, s * P)   // keep_indices
         + BumpAllocator::allocation_size<int32_t>(ctx, P)       // n_reduced
         + BumpAllocator::allocation_size<T>(ctx, s * P);        // temp_lambdas
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
        // `jobz` is forwarded as the caller's, although the merges consume leaf
        // eigenvectors: an unverified hazard for direct NoEigenVectors calls.
        // evidence: docs/perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors
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
        //Modify the two diagonal entries adjacent to the split
        auto ix = idx[0];
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

// Total scratch the flattened driver needs, given a plan. `own_top` tells it
// whether it has to allocate the top-level N x N eigenvector buffer itself
// (true when the problem is padded, or the caller's matrix is not densely
// packed) or can write straight into the caller's matrix.
template <Backend B, typename T>
size_t stedc_levels_workspace(Queue& ctx, const StedcLevelPlan& plan, size_t batch_size,
                              JobType jobz, const StedcParams<T>& params, bool own_top) {
    const size_t N = static_cast<size_t>(plan.padded_n);
    const int32_t L = plan.levels;
    const size_t leaves = size_t(1) << L;

    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx, N * batch_size);   // padded diagonal
    bytes += BumpAllocator::allocation_size<T>(ctx, N * batch_size);   // padded off-diagonal
    bytes += BumpAllocator::allocation_size<T>(ctx, N * batch_size);   // eigenvalues
    bytes += BumpAllocator::allocation_size<T>(ctx, (leaves - 1) * batch_size); // rho per level

    if (own_top) {
        bytes += BumpAllocator::allocation_size<T>(ctx, N * N * batch_size);
    }
    if (L > 0) {
        // Odd levels ping-pong through their own buffer; level 1 is the largest.
        bytes += BumpAllocator::allocation_size<T>(ctx, N * (N / 2) * batch_size);
        // Two s x s x P scratch matrices, largest at the root.
        bytes += 2 * BumpAllocator::allocation_size<T>(ctx, N * N * batch_size);
    }

    size_t merge_bytes = 0;
    for (int32_t l = 0; l < L; ++l) {
        const size_t s = N >> l;
        const size_t P = (size_t(1) << l) * batch_size;
        merge_bytes = std::max(merge_bytes, stedc_merge_step_workspace<T>(ctx, s, P));
    }
    bytes += merge_bytes;

    // Leaf STEQR over every leaf of every batch item at once.
    const size_t leaf = static_cast<size_t>(plan.leaf);
    const size_t leaf_batch = leaves * batch_size;
    auto d_leaf = VectorView<T>(nullptr, static_cast<int>(leaf), static_cast<int>(leaf_batch), 1, static_cast<int>(leaf));
    auto e_leaf = VectorView<T>(nullptr, static_cast<int>(std::max<size_t>(leaf, 2) - 1), static_cast<int>(leaf_batch), 1, static_cast<int>(leaf));
    auto w_leaf = VectorView<T>(nullptr, static_cast<int>(leaf), static_cast<int>(leaf_batch), 1, static_cast<int>(leaf));
    auto leaf_params = params.leaf_steqr_params;
    leaf_params.sort = true;
    leaf_params.sort_order = SortOrder::Ascending;
    // The merges consume the leaves' eigenvectors whatever the caller asked
    // for, so the leaf solve always computes them.
    (void)jobz;
    // Per-leaf status array (indexed by leaf, not by item). UNCONDITIONAL, like potrf's:
    // a workspace sized without `info` must not be too small for a call made with it.
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, leaf_batch);
    bytes += BumpAllocator::allocation_size<std::byte>(
        ctx, steqr_buffer_size<T>(ctx, d_leaf, e_leaf, w_leaf, JobType::EigenVectors, leaf_params));

    return bytes;
}

// True when `eigvects` is a densely packed n x n x batch block we can also
// reinterpret as the packed per-level block arrays of the deeper even levels.
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
    const bool own_top = (N != n) || !stedc_top_matrix_is_packed<T>(eigvects, n);

    auto pool = BumpAllocator(ws);
    auto dp = pool.allocate<T>(ctx, N * bs);
    auto ep = pool.allocate<T>(ctx, N * bs);
    auto wl = pool.allocate<T>(ctx, N * bs);
    auto rho_all = pool.allocate<T>(ctx, ((size_t(1) << L) - 1) * bs);

    Span<T> level_buf[2];
    level_buf[0] = own_top ? pool.allocate<T>(ctx, static_cast<size_t>(N) * N * bs)
                           : Span<T>(eigvects.data_ptr(), static_cast<size_t>(N) * N * bs);
    if (L > 0) {
        level_buf[1] = pool.allocate<T>(ctx, static_cast<size_t>(N) * (N / 2) * bs);
    }
    Span<T> qprime_buf, tempq_buf;
    if (L > 0) {
        qprime_buf = pool.allocate<T>(ctx, static_cast<size_t>(N) * N * bs);
        tempq_buf = pool.allocate<T>(ctx, static_cast<size_t>(N) * N * bs);
    }

    size_t merge_bytes = 0;
    for (int32_t l = 0; l < L; ++l) {
        merge_bytes = std::max(merge_bytes,
                               stedc_merge_step_workspace<T>(ctx, N >> l, (size_t(1) << l) * bs));
    }
    auto merge_ws = pool.allocate<std::byte>(ctx, merge_bytes);
    // Drawn unconditionally so the workspace size does not depend on whether the
    // caller asked for status; see stedc_levels_workspace.
    auto leaf_status = pool.allocate<int32_t>(ctx, static_cast<size_t>(size_t(1) << L) * static_cast<size_t>(bs));

    T* dp_ptr = dp.data();
    T* ep_ptr = ep.data();
    T* w_ptr = wl.data();

    // 1. Copy the tridiagonal into the padded, level-friendly layout. When the
    //    problem needs padding, the tail diagonal is placed above the
    //    Gershgorin bound so those eigenvalues sort past every real one.
    const bool needs_pad = (N > n);
    ctx->submit([&](sycl::handler& h) {
        auto scratch = sycl::local_accessor<T, 1>(sycl::range<1>(needs_pad ? n : 1), h);
        h.parallel_for<StedcLevelPad<B, T>>(sycl::nd_range<1>(bs * 128, 128), [=](sycl::nd_item<1> item) {
            const int bid = item.get_group_linear_id();
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
                const T gersh = sycl::joint_reduce(cta,
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

        // 3. Gather each level's rank-1 coefficients into one contiguous array;
        //    level l occupies [(2^l - 1) * bs, (2^(l+1) - 1) * bs).
        T* rho_ptr = rho_all.data();
        const int64_t rho_total = ((int64_t(1) << L) - 1) * bs;
        ctx->parallel_for<StedcLevelRho<B, T>>(sycl::range<1>(rho_total), [=](sycl::id<1> idx) {
            const int64_t g = idx[0];
            int32_t l = 0;
            int64_t start = 0;
            while (l < L) {
                const int64_t count = (int64_t(1) << l) * bs;
                if (g < start + count) break;
                start += count;
                ++l;
            }
            const int64_t k = int64_t(1) << l;
            const int64_t p = g - start;   // super-batch index = b * k + node
            const int64_t b = p / k;
            const int64_t node = p % k;
            const int64_t s = N >> l;
            rho_ptr[g] = ep_ptr[b * N + node * s + (s >> 1) - 1];
        });
    }

    // Each merge writes directly into its parent's two diagonal sub-blocks; only the
    // off-diagonal zero blocks are materialised here (deflation's Givens rotations
    // mix columns across the split, so they must be zero, not stale).
    // evidence: docs/perf/stedc.md#stedc-the-level-synchronous-driver
    const auto zero_offdiagonal = [&](int32_t level) {
        const int64_t s = N >> level;
        const int64_t half = s >> 1;
        const int64_t P = (int64_t(1) << level) * bs;
        auto Z = MatrixView<T>(level_buf[level & 1].data(), s, s, s, s * s, P);
        ctx->submit([&](sycl::handler& h) {
            auto Zv = Z.kernel_view();
            h.parallel_for<StedcLevelZeroOffdiag<B, T>>(sycl::range<1>(half * half * 2 * P), [=](sycl::id<1> idx) {
                const int64_t g = idx[0];
                const int64_t p = g / (half * half * 2);
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
        const int64_t s = N >> level;
        const int64_t half = s >> 1;
        const int64_t P = (int64_t(1) << level) * bs;
        auto Zp = MatrixView<T>(level_buf[level & 1].data(), s, s, s, s * s, P);
        auto Zc = MatrixView<T>(level_buf[(level + 1) & 1].data(), child_n, child_n, child_n,
                                child_n * child_n, 2 * P);
        ctx->submit([&](sycl::handler& h) {
            auto parent = Zp.kernel_view();
            auto child = Zc.kernel_view();
            h.parallel_for<StedcLevelBlockDiag<B, T>>(sycl::range<1>(s * s * P), [=](sycl::id<1> idx) {
                const int64_t g = idx[0];
                const int64_t p = g / (s * s);
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
        const int64_t P = (int64_t(1) << (L - 1)) * bs;
        const int64_t leaf_batch = 2 * P;
        auto leaf_params = params.leaf_steqr_params;
        leaf_params.sort = true;
        leaf_params.sort_order = SortOrder::Ascending;
        auto d_leaf = VectorView<T>(dp_ptr, leaf, leaf_batch, 1, leaf);
        auto e_leaf = VectorView<T>(ep_ptr, std::max<int64_t>(leaf, 2) - 1, leaf_batch, 1, leaf);
        auto w_leaf = VectorView<T>(w_ptr, leaf, leaf_batch, 1, leaf);
        auto z_leaf = MatrixView<T>(level_buf[L & 1].data(), leaf, leaf, leaf, leaf * leaf, leaf_batch);
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
            leaf_info = Span<int32_t>(leaf_status.data(), static_cast<size_t>(leaf_batch));
            ctx->memset(leaf_info.data(), 0, sizeof(int32_t) * static_cast<size_t>(leaf_batch));
        }
        (void)steqr_dispatch<B, T>(ctx, d_leaf, e_leaf, w_leaf, leaf_ws, JobType::EigenVectors, leaf_params, z_leaf, leaf_info);
        if (want_leaf_status) {
            int32_t* const out = detail::info_ptr(info, bs);
            const int32_t* const leaves_in = leaf_info.data();
            const int64_t leaves_per_item = leaf_batch / bs;
            (void)ctx->parallel_for<StedcLevelLeafStatus<B, T>>(
                sycl::range<1>(static_cast<size_t>(leaf_batch)), [=](sycl::id<1> idx) {
                    const int64_t j = static_cast<int64_t>(idx[0]);
                    detail::info_report(out, detail::info_item(j, leaves_per_item), leaves_in[j]);
                });
        }
        gather_blockdiag(L - 1, leaf);
    }

    // 5. Merge one whole level at a time, root-ward.
    for (int32_t l = L - 1; l >= 0; --l) {
        const int64_t s = N >> l;
        const int64_t half = s >> 1;
        const int64_t k = int64_t(1) << l;
        const int64_t P = k * bs;

        auto Zp = MatrixView<T>(level_buf[l & 1].data(), s, s, s, s * s, P);
        auto lambda = VectorView<T>(w_ptr, s, P, 1, s);
        auto rho_level = Span<T>(rho_all.data() + (k - 1) * bs, P);
        auto Qprime = MatrixView<T>(qprime_buf.data(), s, s, s, s * s, P);
        auto temp_Q = MatrixView<T>(tempq_buf.data(), s, s, s, s * s, P);
        const auto level_params = resolve_stedc_tuning<B, T>(s, params, ctx.device().type == DeviceType::GPU);

        // At level l one launch merges k = 2^l sibling nodes per batch item, so the
        // kernel's node index is divided by k before it indexes `info`.
        int32_t* const info_out = detail::info_ptr(info, bs);
        if (l == 0) {
            stedc_merge_step<B, T>(ctx, lambda, Zp, Qprime, temp_Q, rho_level, merge_ws, half, level_params,
                                   info_out, k);
        } else {
            // Hand the result straight to the parent's diagonal sub-blocks.
            const int64_t sp = s * 2;   // parent node size
            zero_offdiagonal(l - 1);
            T* pbase = level_buf[(l - 1) & 1].data();
            auto out_even = MatrixView<T>(pbase, s, s, sp, sp * sp, P / 2);
            auto out_odd = MatrixView<T>(pbase + s * (sp + 1), s, s, sp, sp * sp, P / 2);
            stedc_merge_step<B, T>(ctx, lambda, Zp, Qprime, temp_Q, rho_level, merge_ws, half, level_params,
                                   info_out, k,
                                   out_even, out_odd);
        }
    }

    // 6. Hand back the leading n eigenvalues (and, when we solved a padded or
    //    non-packed problem, the leading n x n eigenvector block).
    ctx->parallel_for<StedcLevelUnpad<B, T>>(sycl::range<1>(n * bs), [=](sycl::id<1> idx) {
        const int64_t g = idx[0];
        const int64_t b = g / n;
        const int64_t i = g - b * n;
        eigenvalues(i, b) = w_ptr[b * N + i];
    });
    if (own_top && eigvects.data_ptr() != nullptr) {
        auto top = MatrixView<T>(level_buf[0].data(), N, N, N, N * N, bs);
        (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, eigvects, top(Slice{0, n}, Slice{0, n}));
    }
    return ctx.get_event();
}

template <Backend B, typename T>
Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects,
            Span<int32_t> info)
{
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
    // write the same slots. See src/extensions/info_span.hh.
    detail::info_clear(ctx, info, d.batch_size());
    return stedc_dispatch<B, T>(ctx, d, e, eigenvalues, ws, jobz, params, eigvects, info);
}

// Everything after the clear; see stedc_internal.hh for who calls this directly.
template <Backend B, typename T>
Event stedc_dispatch(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects,
            Span<int32_t> info)
{
    if constexpr (B == Backend::NETLIB) {
        auto steqr_params = params.leaf_steqr_params;
        steqr_params.sort = true;
        steqr_params.sort_order = SortOrder::Ascending;
        steqr_params.back_transform = false;
        return steqr_legacy<B, T>(ctx, d, e, eigenvalues, ws, jobz, steqr_params, eigvects, info);
    }

    const auto n = d.size();
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
    auto temp_Q = MatrixView<T>(pool.allocate<T>(ctx, n * n * d.batch_size()).data(), n, n, n, n * n, d.batch_size());
    return stedc_impl<B, T>(ctx, d, e, eigenvalues, ws.subspan(alloc_size), jobz, params, eigvects, temp_Q, info);

}

template <Backend B, typename T>
size_t stedc_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    if constexpr (B == Backend::NETLIB) {
        auto evals = VectorView<T>(nullptr, n, batch_size, 1, 0);
        auto diag = VectorView<T>(nullptr, n, batch_size, 1, 0);
        auto offdiag = VectorView<T>(nullptr, n - 1, batch_size, 1, 0);
        auto steqr_params = params.leaf_steqr_params;
        steqr_params.sort = true;
        steqr_params.sort_order = SortOrder::Ascending;
        steqr_params.back_transform = false;
        return steqr_legacy_buffer_size<T>(ctx, diag, offdiag, evals, jobz, steqr_params);
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

    const auto add_int32 = [&](size_t count) {
        return BumpAllocator::allocation_size<int32_t>(ctx, count);
    };
    const auto add_t = [&](size_t count) {
        return BumpAllocator::allocation_size<T>(ctx, count);
    };

    if (n <= static_cast<size_t>(params.recursion_threshold)) {
        auto d_leaf = VectorView<T>(nullptr, static_cast<int64_t>(n), batch_size, 1, 0);
        auto e_leaf = VectorView<T>(nullptr, static_cast<int64_t>(n - 1), batch_size, 1, 0);
        auto eigenvalues_leaf = VectorView<T>(nullptr, static_cast<int64_t>(n), batch_size, 1, 0);
        return steqr_buffer_size<T>(ctx, d_leaf, e_leaf, eigenvalues_leaf, jobz, params.leaf_steqr_params);
    }

    const size_t m = n / 2;
    const size_t child_bytes = add_t(batch_size)
                             + stedc_internal_workspace_size<B, T>(ctx, m, batch_size, jobz, params)
                             + stedc_internal_workspace_size<B, T>(ctx, n - m, batch_size, jobz, params);

    const size_t merge_bytes = add_t(batch_size)
                             + stedc_merge_step_workspace<T>(ctx, n, batch_size)
                             + add_t(n * n * batch_size);

    return std::max(child_bytes, merge_bytes);
}

#define STEDC_INSTANTIATE(back, fp) \
template Event stedc_dispatch<back, BATCHLAS_UNPAREN fp>(Queue& ctx, const VectorView<BATCHLAS_UNPAREN fp>& d, const VectorView<BATCHLAS_UNPAREN fp>& e, const VectorView<BATCHLAS_UNPAREN fp>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<BATCHLAS_UNPAREN fp> params, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>& eigvects, Span<int32_t> info); \
template Event stedc<back, BATCHLAS_UNPAREN fp>(Queue& ctx, const VectorView<BATCHLAS_UNPAREN fp>& d, const VectorView<BATCHLAS_UNPAREN fp>& e, const VectorView<BATCHLAS_UNPAREN fp>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<BATCHLAS_UNPAREN fp> params, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>& eigvects, Span<int32_t> info); \
template size_t stedc_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<BATCHLAS_UNPAREN fp> params);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEDC_INSTANTIATE)


#undef STEDC_INSTANTIATE


}

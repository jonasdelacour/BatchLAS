#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <util/sycl-local-accessor-helpers.hh>
#include <internal/sort.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "stedc_secular.hh"
#define DEBUG_STEDC 0

namespace batchlas {

template <Backend B, typename T>
Event stedc_impl(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects, const MatrixView<T, MatrixFormat::Dense>& temp_Q)
{
    // Ensure leaf subproblems return sorted eigenvalues so higher levels can safely
    // rely on the "children sorted" invariant.
    constexpr auto steqr_params = SteqrParams<T>{32, 10, std::numeric_limits<T>::epsilon(), false, false, true};
    auto n = d.size();
    auto batch_size = d.batch_size();
    if (n <= params.recursion_threshold){
        return steqr<B, T>(ctx, d, e, eigenvalues, ws, jobz, steqr_params, eigvects);
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

    ctx -> parallel_for(sycl::range(batch_size), [=](sycl::id<1> idx) {
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
        stedc_impl<B, T>(ctx, d1, e1, lambda1, ws1, jobz, params, E1, Q1);
        stedc_impl<B, T>(ctx, d2, e2, lambda2, ws2, jobz, params, E2, Q2);
    }
    
    //Once permutations are done we can free the memory once again
    auto permutation = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    // Persistent mapping from logical (current) column order -> physical column in `eigvects`.
    // We avoid physically permuting the eigenvector matrix until right before GEMM / function exit.
    auto perm_map = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    auto v = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, batch_size);
    ctx -> submit([&](sycl::handler& h) {
        auto E1view = E1.kernel_view();
        auto E2view = E2.kernel_view();
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto sign = (e(m - 1, bid) >= 0) ? 1 : -1;
            //Normalized v through division by sqrt(2)
            for (int i = tid; i < m; i += bdim) {
                v(i, bid) = E1view(m - 1, i, bid) / std::sqrt(T(2));
            }
            for (int i = tid; i < n - m; i += bdim) {
                v(i + m, bid) = E2view(0, i, bid) / std::sqrt(T(2));
            }
        });
    });
    argsort(ctx, eigenvalues, perm_map, SortOrder::Ascending, true);
    permute(ctx, eigenvalues, perm_map);
    permute(ctx, v, perm_map);

    auto keep_indices = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    auto n_reduced = pool.allocate<int32_t>(ctx, batch_size);
    //Deflation scheme
    T reltol = T(64.0) * std::numeric_limits<T>::epsilon();
    ctx -> submit([&](sycl::handler& h) {
        auto Q = eigvects.kernel_view();
        auto perm_local = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_include = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_exclude = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto norm_mem = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
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
        for (int j = 0; j < n - 1; j++) {
            if(std::abs(eigenvalues(j + 1, bid) - eigenvalues(j, bid)) <= reltol * std::max(T(1), std::max(std::abs(eigenvalues(j + 1, bid)), std::abs(eigenvalues(j, bid))))) {
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
                if (pj < 0 || pj >= n || pj1 < 0 || pj1 >= n) {
                    continue;
                }
                for (int k = tid; k < n; k += bdim) {
                    auto Qi = Q(k, pj, bid), Qj = Q(k, pj1, bid);
                    Q(k, pj, bid) = Qi*c - Qj*s;
                    Q(k, pj1, bid) = Qj*c + Qi*s;
                }
            }
        }

        sycl::group_barrier(cta);
        //auto v_norm = std::sqrt(sycl::joint_reduce(cta, util::get_raw_ptr(norm_mem), util::get_raw_ptr(norm_mem) + n, sycl::plus<T>()));
        //LAPACK LAED8 based tolerance:
        
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(eigenvalues(k, bid)); }
        auto eig_max = sycl::joint_reduce(cta,
                          util::get_raw_ptr(norm_mem),
                          util::get_raw_ptr(norm_mem) + n,
                          sycl::maximum<T>());

        auto v_norm = internal::nrm2<T>(cta, v);
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(v(k, bid) / v_norm); }
        auto v_max = sycl::joint_reduce(cta,
                        util::get_raw_ptr(norm_mem),
                        util::get_raw_ptr(norm_mem) + n,
                        sycl::maximum<T>());
        auto tol = T(8.0) * std::numeric_limits<T>::epsilon() * std::max(eig_max, v_max);

        //if(tid == 0) sycl::ext::oneapi::experimental::printf("Tolerance for deflation: %e\n", tol);
        for (int k = tid; k < n; k += bdim) {
            //sycl::ext::oneapi::experimental::printf("|v[%d]| * 2 * |rho| = %e\n", k, std::abs(T(2) * rho[bid]) * norm_mem[k]);
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

        //sycl::group_barrier(cta);
        for (int k = tid; k < n; k += bdim) {
            if (keep_indices(k, bid) == 1) {
                permutation(scan_mem_include[k], bid) = k;
            } else {
                permutation(n - 1 - scan_mem_exclude[k], bid) = k;
            }
        }

        for (int k = tid; k < n; k += bdim) {
            //v(k, bid) *= std::sqrt(rho[bid]);
        }

        if (tid == 0) {
            n_reduced[bid] = scan_mem_include[n - 1] + keep_indices(n - 1, bid);
        }
        
        });
    });

    // Apply deflation permutation to contiguous vectors.
    permute(ctx, eigenvalues, permutation);
    permute(ctx, v, permutation);

    // Update the logical->physical column map instead of physically permuting the eigenvector matrix.
    // This composes the current column map with the deflation permutation.
    permute(ctx, perm_map, permutation);

    auto temp_lambdas = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, batch_size);
    MatrixView<T> Qprime = MatrixView<T>(pool.allocate<T>(ctx, n * n * batch_size).data(), n, n, n, n * n, batch_size);
    Qprime.fill_identity(ctx);
    if (params.secular_solver == StedcSecularSolver::Legacy) {
        secular_solver(ctx, eigenvalues, v, Qprime, temp_lambdas, n_reduced, rho, T(10.0));
    } else {
        // Problem: We ultimately need to compute Q1 ⨂ Q2 * Qprime, however since we are deflating the columns of Q1 ⨂ Q2 we need to be careful about how we form Qprime.
        // Idea: As long as the columns of Qprime are the euclidean basis vectors, multiplying by Qprime is just a permutation of the columns of Q1 ⨂ Q2.
        ctx -> submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto bdim = item.get_local_range(0);
                auto tid = item.get_local_linear_id();
                auto cta = item.get_group();
                auto Q_bid = Qview.batch_item(bid);
                auto sign = (e(m - 1, bid) >= 0) ? 1 : -1;
                auto n = n_reduced[bid];
                for (int k = tid; k < n * n; k += bdim) {
                    auto i = k % n;
                    auto j = k / n;
                    Q_bid(i, j) = eigenvalues(i, bid);
                }
                sycl::group_barrier(cta);
                for (int k = tid; k < n; k += bdim) {
                    auto dview = Q_bid(Slice{}, k);
                    if (k == n - 1){
                        temp_lambdas(k, bid) = sec_solve_ext_roc(n, dview, v.batch_item(bid), std::abs(2 * rho[bid])) * sign;
                    } else {
                        temp_lambdas(k, bid) = sec_solve_roc(n, dview, v.batch_item(bid), std::abs(2 * rho[bid]), k) * sign;
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
            h.parallel_for(
                sycl::nd_range<1>(d.batch_size() * 128, 128),
                [=](sycl::nd_item<1> item) {
                    auto bid = item.get_group_linear_id();
                    auto g   = item.get_group();
                    auto tid = item.get_local_linear_id();
                    auto bdim = item.get_local_range(0);
                    auto Qbid = Qview.batch_item(bid);
                    auto dd = n_reduced[bid];

                    for (int eid = 0; eid < dd; ++eid)
                    {
                        auto Di = eigenvalues(eid, bid);
                        T partial = T(1);
                        for(int j = tid; j < dd; j += static_cast<int>(bdim))
                        {
                            partial *= (j == eid) ? Qbid(eid, j) : Qbid(eid, j) / (Di - eigenvalues(j, bid));
                        }

                        T valf = sycl::reduce_over_group(g, partial, sycl::multiplies<T>());
                        if(tid == 0)
                        {
                            T mag  = sycl::sqrt(sycl::fabs(valf));
                            T sign = v(eid, bid) >= T(0) ? T(1) : T(-1);
                            v(eid, bid) = sign * mag;
                        }
                    }
                });
        });

        ctx -> submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for(
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
        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();
        auto cta = item.get_group();

        for (int k = tid; k < n_reduced[bid]; k += bdim) {
            eigenvalues(k, bid) = temp_lambdas(k, bid);
        }
        });
    });

    argsort(ctx, eigenvalues, permutation, SortOrder::Ascending, true);
    permute(ctx, eigenvalues, permutation);

    // Avoid full-matrix copy + permute by using out-of-place permuted_copy in scratch buffers.
    permuted_copy(ctx, Qprime, temp_Q, permutation);
    permuted_copy(ctx, eigvects, Qprime, perm_map);
    gemm<B>(ctx, Qprime, temp_Q, eigvects, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

    return ctx.get_event();
}

template <Backend B, typename T>
Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects) 
{
    if (d.size() != e.size() + 1) {
        throw std::runtime_error("The size of e must be one less than the size of d.");
    }
    if (d.size() != eigenvalues.size()) {
        throw std::runtime_error("The size of eigenvalues must match the size of d.");
    }
    if (d.batch_size() != e.batch_size() || d.batch_size() != eigenvalues.batch_size()) {
        throw std::runtime_error("The batch sizes of d, e, and eigenvalues must match.");
    }
    if (jobz == JobType::EigenVectors) {
        if (eigvects.rows() != d.size() || eigvects.cols() != d.size() || eigvects.batch_size() != d.batch_size()) {
            throw std::runtime_error("The dimensions of eigvects must match the size of d and its batch size.");
        }
    }
    //Clean the output matrix before we begin.
    eigvects.fill_zeros(ctx);
    auto pool = BumpAllocator(ws);
    auto n = d.size();
    auto alloc_size = BumpAllocator::allocation_size<T>(ctx, n * n * d.batch_size());
    auto temp_Q = MatrixView<T>(pool.allocate<T>(ctx, n * n * d.batch_size()).data(), n, n, n, n * n, d.batch_size());
    return stedc_impl<B, T>(ctx, d, e, eigenvalues, ws.subspan(alloc_size), jobz, params, eigvects, temp_Q);

}

template <Backend B, typename T>
size_t stedc_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    size_t size = 0;
    auto d = VectorView<T>(nullptr, params.recursion_threshold, batch_size);
    auto e = VectorView<T>(nullptr, params.recursion_threshold - 1, batch_size);
    auto eigenvalues = VectorView<T>(nullptr, params.recursion_threshold, batch_size);
    // How many recursions do we need?
    auto n_rec = (n + params.recursion_threshold - 1) / params.recursion_threshold;
    auto m = (n + n_rec - 1) / n_rec; // Size of each subproblem
    
    // Compute the workspace size based on the job type
    size = params.recursion_threshold <= 32 ? steqr_cta_buffer_size<T>(ctx, d, e, eigenvalues, jobz) : steqr_buffer_size<T>(ctx, d, e, eigenvalues, jobz);
    size += 2 * BumpAllocator::allocation_size<int32_t>(ctx, 2 * (m + 1) * batch_size) + BumpAllocator::allocation_size<T>(ctx, batch_size); // For permutation array and rho storage

    // Additional persistent column-map buffer used in the merge step (perm_map).
    size += BumpAllocator::allocation_size<int32_t>(ctx, n * batch_size);

    return (size * n_rec) + 2 * BumpAllocator::allocation_size<T>(ctx, n * n * batch_size); // Multiply by number of recursions needed
}


template <Backend B, typename T>
size_t stedc_internal_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    size_t size = 0;
    auto d = VectorView<T>(nullptr, params.recursion_threshold, batch_size, 1, 0);
    auto e = VectorView<T>(nullptr, params.recursion_threshold - 1, batch_size, 1, 0);
    auto eigenvalues = VectorView<T>(nullptr, params.recursion_threshold, batch_size, 1, 0);
    // How many recursions do we need?
    auto n_rec = (n + params.recursion_threshold - 1) / params.recursion_threshold;
    auto m = (n + n_rec - 1) / n_rec; // Size of each subproblem
    
    // Compute the workspace size based on the job type
    size = params.recursion_threshold <= 32 ? steqr_cta_buffer_size<T>(ctx, d, e, eigenvalues, jobz) : steqr_buffer_size<T>(ctx, d, e, eigenvalues, jobz);
    size += 2 * BumpAllocator::allocation_size<int32_t>(ctx, 2 * (m + 1) * batch_size) + BumpAllocator::allocation_size<T>(ctx, batch_size); // For permutation array and rho storage

    // Additional persistent column-map buffer used in the merge step (perm_map).
    size += BumpAllocator::allocation_size<int32_t>(ctx, n * batch_size);

    return (size * n_rec) + BumpAllocator::allocation_size<T>(ctx, n * n * batch_size); // Multiply by number of recursions needed
}

#if BATCHLAS_HAS_HOST_BACKEND
template Event stedc<Backend::NETLIB, float>(Queue& ctx, const VectorView<float>& d, const VectorView<float>& e, const VectorView<float>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<float> params, const MatrixView<float, MatrixFormat::Dense>& eigvects);
template Event stedc<Backend::NETLIB, double>(Queue& ctx, const VectorView<double>& d, const VectorView<double>& e, const VectorView<double>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<double> params, const MatrixView<double, MatrixFormat::Dense>& eigvects);

template size_t stedc_workspace_size<Backend::NETLIB, float>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<float> params);
template size_t stedc_workspace_size<Backend::NETLIB, double>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<double> params);
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
template Event stedc<Backend::CUDA, float>(Queue& ctx, const VectorView<float>& d, const VectorView<float>& e, const VectorView<float>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<float> params, const MatrixView<float, MatrixFormat::Dense>& eigvects);
template Event stedc<Backend::CUDA, double>(Queue& ctx, const VectorView<double>& d, const VectorView<double>& e, const VectorView<double>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<double> params, const MatrixView<double, MatrixFormat::Dense>& eigvects);

template size_t stedc_workspace_size<Backend::CUDA, float>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<float> params);
template size_t stedc_workspace_size<Backend::CUDA, double>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<double> params);
#endif


}

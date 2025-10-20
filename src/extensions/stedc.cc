#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <internal/sort.hh>
#include "../math-helpers.hh"


namespace batchlas {

template <typename T>
auto givens_rotation_r(const T& a, const T& b) {
    T r = std::hypot(a, b);
    if (internal::is_numerically_zero(r)) {
        return std::array<T, 3>{T(1), T(0), T(0)};
    }
    return std::array<T, 3>{a / r, b / r, r};
}


template <typename T> 
auto psi_terms(const sycl::group<1>& cta, const VectorView<T>& v, const VectorView<T>& d, const T& lam, const T& shift, const int32_t i, const int32_t n, const sycl::local_accessor<T, 1>& psi_buffer) {
    auto tid = cta.get_local_linear_id();
    auto bdim = cta.get_local_range()[0];
    auto bid = cta.get_group_linear_id();

    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / ((d(k, bid) - (lam + shift))); }
    sycl::group_barrier(cta);
    auto psi1 = sycl::joint_reduce(cta, psi_buffer.get_pointer(), psi_buffer.get_pointer() + i + 1, sycl::plus<T>());
    auto psi2 = (i + 1) < n ? sycl::joint_reduce(cta, psi_buffer.get_pointer() + i + 1, psi_buffer.get_pointer() + n, sycl::plus<T>()) : T(0);
    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / ( (d(k, bid) - (lam + shift)) * (d(k, bid) - (lam + shift))); }
    sycl::group_barrier(cta);
    auto psi1_prime = sycl::joint_reduce(cta, psi_buffer.get_pointer(), psi_buffer.get_pointer() + i + 1, sycl::plus<T>());
    auto psi2_prime = (i + 1) < n ? sycl::joint_reduce(cta, psi_buffer.get_pointer() + i + 1, psi_buffer.get_pointer() + n, sycl::plus<T>()) : T(0);
    return std::array<T, 4>{psi1, psi1_prime, psi2, psi2_prime};
}

template <typename T>
Event secular_solver(Queue& ctx, const VectorView<T>& d, const VectorView<T>& v, const MatrixView<T, MatrixFormat::Dense>& Qprime, const VectorView<T>& lambdas, const Span<int32_t>& n_reduced, const T& tol_factor = 10.0) {
    //Solve the secular equation for each row in d and v
    ctx -> wait();
    auto N_max = d.size();
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto shared_mem = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vhat = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vsign = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto numerators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto denominators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        h.parallel_for(sycl::nd_range<1>(d.batch_size()*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto cta = item.get_group();
            auto n = n_reduced[bid];
            for (int k = tid; k < n; k += bdim) { vsign[k] = (v(k, bid) >= T(0)) ? T(1) : T(-1); }
            for (int k = tid; k < n; k += bdim) { auto v_temp = v(k, bid); v(k, bid) *= v_temp; }
            
            auto v_norm2 = sycl::joint_reduce(cta, v.batch_item(bid).data_ptr(), v.batch_item(bid).data_ptr() + n, sycl::plus<T>());
            for (int i = 0; i < n; i++) {
                auto di = d(i, bid);
                auto di1 = i < n - 1 ? d(i + 1, bid) : d(n - 1, bid) + v_norm2;
                auto lam = i < n - 1 ? T(0.5) * (di + di1) : di1;
                auto [psi1, psi1_prime, psi2, psi2_prime] = psi_terms(cta, v, d, lam, T(0.0), i, n, shared_mem);

                auto condition = T(1) + psi1 + psi2 > T(0);
                auto shift = condition ? di : di1;
                lam  -= shift;
                if (condition) {di1 -= di;} else {di -= di1;}
                auto iter = 0;
                while (true) {
                    auto [psi1, psi1_prime, psi2, psi2_prime] = psi_terms(cta, v, d, lam, shift, i, n, shared_mem);
                    auto Di = condition ? -lam : di - lam;
                    auto Di1 = condition ? di1 - lam : -lam;
                    auto a = (Di + Di1) * (T(1) + psi1 + psi2) - Di * Di1 * (psi1_prime + psi2_prime);
                    auto b = Di * Di1 * (T(1) + psi1 + psi2);
                    auto c = (T(1) + psi1 + psi2) - Di * psi1_prime - Di1 * psi2_prime;
                    auto disc = a * a - T(4) * b * c;
                    disc = (disc < T(0) && disc < std::numeric_limits<T>::epsilon()) ? T(0) : disc;
                    auto s = std::sqrt(disc);
                    auto eta = (a > T(0)) ? (T(2) * b) / (a + s) : (a - s) / (T(2) * c);
                    auto new_lam = lam + eta;
                    if (std::abs(eta) <= tol_factor * std::numeric_limits<T>::epsilon() || iter >= 100) {
                        lam = new_lam;
                        break;
                    }
                    lam = new_lam;
                    iter++;
                }
                
                if(tid == 0) lambdas(i,bid) = lam;

                for (int k = tid; k < n; k += bdim) {
                    Qview(k, i, bid) = d(k, bid) - (lam + shift);
                }

                if (i == 0) {for (int k = tid; k < n; k += bdim) numerators[k] = std::abs(Qview(k, i, bid));}
                else {for (int k = tid; k < n; k += bdim) numerators[k] *= std::abs(Qview(k, i, bid));}

                for (int k = tid; k < i; k += bdim) shared_mem[k] = d(i, bid) - d(k, bid);
                auto den2 = i > 0 ? sycl::joint_reduce(cta, shared_mem.get_pointer(), shared_mem.get_pointer() + i, sycl::multiplies<T>()) : T(1);

                for (int k = tid; k < n - i - 1; k += bdim) shared_mem[k] = d(k + i + 1, bid) - d(i, bid);
                auto den1 = (i + 1) < n ? sycl::joint_reduce(cta, shared_mem.get_pointer(), shared_mem.get_pointer() + n - i - 1, sycl::multiplies<T>()) : T(1);

                if (tid == 0) denominators[i] = den1 * den2;
                
                
                // $$v_i =  \frac{ \prod_{j=1}^n |d_j - \lambda_i| }{ \prod_{j=1}^{i-1} (d_j - d_i) \prod_{j=i+1}^{n} (d_j - d_i) } $$
                
                

            }

            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) vhat[k] = vsign[k] * std::sqrt(numerators[k] / denominators[k]);

            
            
            // $$ q_i = \frac{(\lambda_i \mathtt{I} - D)^{-1} \vec{\hat{v}}}{||(\lambda_i \mathtt{I} - D)^{-1} \vec{\hat{v}}||} $$

            
            

            for (int i = 0; i < n; i++) {
                for (int k = tid; k < n; k += bdim) {
                    Qview(k, i, bid) = vhat[k] / Qview(k, i, bid);
                }
            }


            for (int i = 0; i < n; i++) {
                for (int k = tid; k < n; k += bdim) shared_mem[k] = Qview(k, i, bid) * Qview(k, i, bid);

                auto norm = std::sqrt(sycl::joint_reduce(cta, shared_mem.get_pointer(), shared_mem.get_pointer() + n, sycl::plus<T>()));

                for (int k = tid; k < n; k += bdim) Qview(k, i, bid) = Qview(k, i, bid) / norm;
            }

            for (int k = tid; k < n; k += bdim) {
                auto d_term = lambdas(k, bid) > 0 ? d(k, bid) : (k + 1 < n ? d(k + 1, bid) : d(k, bid) + v_norm2);
                lambdas(k, bid) += d_term;
            }
        });
    });

    return ctx.get_event();
}

template <Backend B, typename T>
Event stedc_impl(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects, const MatrixView<T, MatrixFormat::Dense>& temp_Q)
{
    constexpr auto steqr_params = SteqrParams<T>{32, 5, std::numeric_limits<T>::epsilon(), false, false, false};
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

    auto pool = BumpAllocator(ws);
    auto rho = pool.allocate<T>(ctx, batch_size);

    ctx -> parallel_for(sycl::range(batch_size), [=](sycl::id<1> idx) {
        //Modify the two diagonal entries adjacent to the split
        auto ix = idx[0];
        rho[ix] = std::abs(e(m - 1, ix));
        d1(m - 1, ix) -= rho[ix];
        d2(0, ix) -= rho[ix];
    });

    //Scope this section: after the child recursions return, their workspace memory can be reused
    {
        auto pool = BumpAllocator(ws.subspan(BumpAllocator::allocation_size<T>(ctx, batch_size)));
        auto ws1 = pool.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, m, batch_size, jobz, params));
        auto ws2 = pool.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, n - m, batch_size, jobz, params));
        stedc_impl<B, T>(ctx, d1, e1, eigenvalues(Slice(0, m)), ws1, jobz, params, E1, Q1);
        stedc_impl<B, T>(ctx, d2, e2, eigenvalues(Slice(m, SliceEnd())), ws2, jobz, params, E2, Q2);
    }
    
    //Once permutations are done we can free the memory once again
    auto permutation = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, 1, n, batch_size);
    auto v = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, 1, n, batch_size);
    ctx -> submit([&](sycl::handler& h) {
        auto E1view = E1.kernel_view();
        auto E2view = E2.kernel_view();
        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto sign = (e(m - 1, bid) >= 0) ? 1 : -1;
            for (int i = tid; i < m; i += bdim) {
                v(i, bid) = E1view(m - 1, i, bid) * sign;
            }
            for (int i = tid; i < n - m; i += bdim) {
                v(i + m, bid) = E2view(0, i, bid);
            }
        });
    });
    argsort(ctx, eigenvalues, permutation, SortOrder::Ascending, true);
    permute(ctx, eigenvalues, permutation);
    permute(ctx, v, permutation);
    permuted_copy(ctx, eigvects, temp_Q, permutation);
    //std::cout << "Post divide and sort temp_Q matrix:\n" << temp_Q << std::endl;

    auto keep_indices = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, 1, n, batch_size);
    auto n_reduced = pool.allocate<int32_t>(ctx, batch_size);
    //Deflation scheme
    T reltol = T(64.0) * std::numeric_limits<T>::epsilon();
    ctx -> submit([&](sycl::handler& h) {
        auto Q = temp_Q.kernel_view();
        auto scan_mem_include = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_exclude = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();
        auto cta = item.get_group();
        
        for (int k = tid; k < n; k += bdim){
            keep_indices(k, bid) = 0;
            scan_mem_exclude[k] = 0;
            permutation(k, bid) = -1;
        }


        sycl::group_barrier(cta);
        for (int j = 0; j < n - 1; j++) {
            if(std::abs(eigenvalues(j + 1, bid) - eigenvalues(j, bid)) <= reltol * std::max(T(1), std::max(std::abs(eigenvalues(j + 1, bid)), std::abs(eigenvalues(j, bid))))) {
                auto f = v(j + 1, bid);
                auto g = v(j, bid);
                auto [c, s, r] = givens_rotation_r(f, g);
                if (tid == 0) {
                    v(j, bid) = T(0.0);
                    v(j + 1, bid) = r;
                }
                for (int k = tid; k < n; k += bdim) {
                    auto Qi = Q(k,j,bid), Qj = Q(k,j + 1,bid);
                    Q(k,j,bid) = Qi*c - Qj*s;
                    Q(k,j + 1,bid) = Qj*c + Qi*s;
                }
            }
        }

        sycl::group_barrier(cta);
        auto v_norm = sycl::joint_reduce(cta, v.batch_item(bid).data_ptr(), v.batch_item(bid).data_ptr() + n, sycl::plus<T>());
        
        for (int k = tid; k < n; k += bdim) {
            if (std::abs(v(k, bid)) > 32 * std::numeric_limits<T>::epsilon() * std::max(T(1), v_norm)) {
                keep_indices(k, bid) = 1;
            } else {
                scan_mem_exclude[k] = 1;
            }
        }

        sycl::group_barrier(cta);

        //Exclusive scan to determine the indices to keep
        sycl::joint_exclusive_scan(cta, keep_indices.batch_item(bid).data_ptr(), keep_indices.batch_item(bid).data_ptr() + n, scan_mem_include.get_pointer(), 0, sycl::plus<int32_t>());
        sycl::joint_exclusive_scan(cta, scan_mem_exclude.get_pointer(), scan_mem_exclude.get_pointer() + n, scan_mem_exclude.get_pointer(), 0, sycl::plus<int32_t>());

        //sycl::group_barrier(cta);
        for (int k = tid; k < n; k += bdim) {
            if (keep_indices(k, bid) == 1) {
                permutation(scan_mem_include[k], bid) = k;
            } else {
                permutation(n - 1 - scan_mem_exclude[k], bid) = k;
            }
        }

        for (int k = tid; k < n; k += bdim) {
            v(k, bid) *= std::sqrt(rho[bid]);
        }

        if (tid == 0) {
            n_reduced[bid] = scan_mem_include[n - 1] + keep_indices(n - 1, bid);
        }
        
        });
    });

    permute(ctx, temp_Q, eigvects, permutation);

    permute(ctx, eigenvalues, permutation);
    permute(ctx, v, permutation);
    auto temp_lambdas = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, 1, n, batch_size);
    MatrixView<T> Qprime = MatrixView<T>(pool.allocate<T>(ctx, n * n * batch_size).data(), n, n, n, n * n, batch_size);
    Qprime.fill_identity(ctx);
    //Problem: We ultimately need to compute Q1 ⨂ Q2 * Qprime, however since we are deflating the columns of Q1 ⨂ Q2 we need to be careful about how we form Qprime.
    //Idea: As long as the columns of Qprime are the euclidean basis vectors, multiplying by Qprime is just a permutation of the columns of Q1 ⨂ Q2

    secular_solver(ctx, eigenvalues, v, Qprime, temp_lambdas, n_reduced, T(10.0));

    gemm<B>(ctx, temp_Q, Qprime, eigvects, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
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
    permute(ctx, eigvects, temp_Q, permutation);

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
    auto d = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    auto e = VectorView<T>(nullptr, params.recursion_threshold - 1, 1, 0, batch_size);
    auto eigenvalues = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    // How many recursions do we need?
    auto n_rec = (n + params.recursion_threshold - 1) / params.recursion_threshold;
    auto m = (n + n_rec - 1) / n_rec; // Size of each subproblem
    
    // Compute the workspace size based on the job type
    switch (jobz) {
        case JobType::NoEigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues);
            break;
        case JobType::EigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues, jobz);
            break;
        default:
            throw std::runtime_error("Invalid job type");
    }
    size += 2 * BumpAllocator::allocation_size<int32_t>(ctx, 2 * (m + 1) * batch_size) + BumpAllocator::allocation_size<T>(ctx, batch_size); // For permutation array and rho storage

    return (size * n_rec) + 2 * BumpAllocator::allocation_size<T>(ctx, n * n * batch_size); // Multiply by number of recursions needed
}


template <Backend B, typename T>
size_t stedc_internal_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    size_t size = 0;
    auto d = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    auto e = VectorView<T>(nullptr, params.recursion_threshold - 1, 1, 0, batch_size);
    auto eigenvalues = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    // How many recursions do we need?
    auto n_rec = (n + params.recursion_threshold - 1) / params.recursion_threshold;
    auto m = (n + n_rec - 1) / n_rec; // Size of each subproblem
    
    // Compute the workspace size based on the job type
    switch (jobz) {
        case JobType::NoEigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues);
            break;
        case JobType::EigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues, jobz);
            break;
        default:
            throw std::runtime_error("Invalid job type");
    }
    size += 2 * BumpAllocator::allocation_size<int32_t>(ctx, 2 * (m + 1) * batch_size) + BumpAllocator::allocation_size<T>(ctx, batch_size); // For permutation array and rho storage

    return (size * n_rec) + BumpAllocator::allocation_size<T>(ctx, n * n * batch_size); // Multiply by number of recursions needed
}

template Event stedc<Backend::NETLIB, float>(Queue& ctx, const VectorView<float>& d, const VectorView<float>& e, const VectorView<float>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<float> params, const MatrixView<float, MatrixFormat::Dense>& eigvects);
template Event stedc<Backend::NETLIB, double>(Queue& ctx, const VectorView<double>& d, const VectorView<double>& e, const VectorView<double>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<double> params, const MatrixView<double, MatrixFormat::Dense>& eigvects);
template Event stedc<Backend::CUDA, float>(Queue& ctx, const VectorView<float>& d, const VectorView<float>& e, const VectorView<float>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<float> params, const MatrixView<float, MatrixFormat::Dense>& eigvects);
template Event stedc<Backend::CUDA, double>(Queue& ctx, const VectorView<double>& d, const VectorView<double>& e, const VectorView<double>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<double> params, const MatrixView<double, MatrixFormat::Dense>& eigvects);

template size_t stedc_workspace_size<Backend::CUDA, float>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<float> params);
template size_t stedc_workspace_size<Backend::CUDA, double>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<double> params);
template size_t stedc_workspace_size<Backend::NETLIB, float>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<float> params);
template size_t stedc_workspace_size<Backend::NETLIB, double>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<double> params);


}
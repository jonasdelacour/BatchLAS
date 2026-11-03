// SPDX-License-Identifier: MIT
//Implementation file for Lanczos algorithm
#include <src/util/batch_mask.hh>
#include <src/linalg-impl.hh>
#include <src/math-helpers.hh>
#include <src/util/template-instantiations.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/queue.hh>
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <src/util/philox.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <src/sort.hh>
#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas {
    namespace dla = device::v2;

    template <Backend B, typename T, MatrixFormat MF>
    struct InitKernel {};

    template <Backend B, typename T, MatrixFormat MF>
    struct LanczosKernel {};

    namespace {
    // Shared by lanczos and lanczos_buffer_size, which must size exactly what the run uses.
    template <typename T>
    SteqrParams<T> lanczos_steqr_params() {
        return SteqrParams<T>{32, 10, false, false, true, true, SortOrder::Descending};
    }

    // Item b's off-diagonals start at betas[b * n] (one n-slot per item, n - 1 used).
    template <typename R>
    VectorView<R> lanczos_offdiagonals(R* betas, int n, int batch_size) {
        return VectorView<R>(betas, n - 1, batch_size, Inc{1}, Stride{n});
    }
    } // namespace

    template <Backend B, typename T, MatrixFormat MF>
    Event lanczos(Queue& ctx,
                const MatrixView<T, MF>& A,
                Span<typename base_type<T>::type> W, //Output eigenvalues
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V, //Output eigenvectors for jobz == JobType::EigenVectors
                const LanczosParams<T>& params) {
        // Every child honours the mask: the scratch views, both kernels, ortho, gemm/spmm, steqr and sort.
        const auto mask = detail::call_mask("lanczos", A, V);
        const std::uint8_t* mk = detail::mask_ptr(mask);
        const bool masked = mk != nullptr;  // a captured null pointer can read as non-null on AdaptiveCpp Metal
        using real_t = typename base_type<T>::type;

        auto pool = BumpAllocator(workspace);

        auto batch_size = A.batch_size();
        auto n = A.rows();

        auto Vmem = pool.allocate<T>(ctx, (1+n)*n*batch_size);
        auto V_vectormem = pool.allocate<T>(ctx, n*2*batch_size);
        auto alphas = pool.allocate<T>(ctx, n*batch_size);
        auto betas = pool.allocate<T>(ctx, n*batch_size);

        //Batched SpMV is not supported so we have to represent our vector as a dense matrix padded with an extra column (to prevent fallback to SpMV)
        // Known defect: both columns are multiplied, one is consumed.
        // evidence: docs/design/known-defects.md#defect-3-lanczos-issues-a-two-column-multiply-and-consumes-one-column
        auto padded_output = MatrixView(V_vectormem.data(), n, 2, n, 2*n, batch_size, pool.allocate<T*>(ctx, batch_size).data()).with_batch_mask(mask);
        auto padded_vector = MatrixView(Vmem.data(), n, 2, n, (n+1)*n, batch_size).with_batch_mask(mask);

        Span<std::byte> spmm_buffer;
        if constexpr (!(MF == MatrixFormat::Dense)){
            spmm_buffer = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, padded_vector, padded_output, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans));
        }
        auto ortho_buffer = pool.allocate<std::byte>(ctx, ortho_buffer_size<B>(ctx, MatrixView(Vmem.data(), n, 1, n, (n+1)*n, batch_size).with_batch_mask(mask),
            MatrixView(Vmem.data(), n, n-1, n, (n+1)*n, batch_size).with_batch_mask(mask), Transpose::NoTrans, Transpose::NoTrans, params.ortho_algorithm, params.ortho_iterations));

        auto basis_ptr_mem = pool.allocate<T*>(ctx, batch_size);
        auto vector_view_ptr_mem = pool.allocate<T*>(ctx, batch_size);


        auto wg_init = std::min(size_t(n), get_kernel_max_wg_size<InitKernel<B, T, MF>>(ctx));
        auto wg_lanczos = std::min(size_t(n), get_kernel_max_wg_size<LanczosKernel<B, T, MF>>(ctx));

        //Initial guess
        auto init = [&](){
            ctx -> submit([&](sycl::handler& h) {
                batchlas::impl::parallel_for<InitKernel<B, T, MF>>(h, sycl::nd_range<1>(batch_size*wg_init, wg_init), [=](sycl::nd_item<1> item) {
                    auto bid = item.get_group_linear_id();
                    if (masked && !mk[bid]) return;  // I1, before the group reduction
                    const dla::RtGroup<> ex{item};
                    const auto v0 = dla::vec(Vmem.data() + bid*(n+1)*n, n);
                    // Same start vector for every item: keyed on the row only.
                    dla::for_each_owned(ex, n, [&](int i) { v0(i) = T(philox::uniform_unit<real_t>(42, 0, static_cast<std::uint64_t>(i))); });
                    const real_t norm = sycl::sqrt(dla::sumsq<dla::alg::Unfused>(ex, v0));
                    dla::map(ex, v0, [=](T v) { return dla::Arith<T>::div_real(v, norm); }, v0);  // a true division, not scal(1/norm)
                });
            });
        };

        auto lanczos_iteration = [&](const int iterations){
            for (int it = 0; it < iterations; it++) {
                // Only v_it, against v_0..v_{it-1}: the earlier vectors' alphas and betas are recorded, and
                // ortho returns a basis of the span (SVQB rotates a pair, Householder flips signs), not
                // the input vectors. A flipped v_it is absorbed by the signed beta in the kernel below.
                // evidence: docs/design/known-defects.md#lanczos-returned-a-spectrum-about-three-times-too-wide
                if (it > 0) {
                    auto basis_view = MatrixView(Vmem.data(), n, it, n, (n+1)*n, batch_size, basis_ptr_mem.data()).with_batch_mask(mask);
                    auto vector_view = MatrixView(Vmem.data() + it*n, n, 1, n, (n+1)*n, batch_size, vector_view_ptr_mem.data()).with_batch_mask(mask);
                    if((it % params.reorthogonalization_iterations == 0) || (it == iterations - 1)) {
                        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
                        // is already ordered after this one and the Event carries nothing the caller needs.
                        (void)ortho<B>(ctx, vector_view, basis_view, Transpose::NoTrans, Transpose::NoTrans, ortho_buffer, params.ortho_algorithm, params.ortho_iterations);
                    }
                }

                auto padded_vector = MatrixView(Vmem.data() + it*n, n, 2, n, (n+1)*n, batch_size).with_batch_mask(mask);
                auto spmm_start = std::chrono::steady_clock::now();
                if constexpr (!(MF == MatrixFormat::Dense)) {
                    (void)spmm<B>(ctx, A, padded_vector, padded_output, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans, spmm_buffer);
                } else {
                    (void)gemm<B>(ctx, A, padded_vector, padded_output, GemmOptions<T>{});
                }
                ctx -> submit([&](sycl::handler& h) {
                    auto v_prev_ptr = Vmem.data() + (std::max(it-1,0))*n;
                    auto v_current_ptr = Vmem.data() + it*n;
                    auto v_next_ptr = padded_output.data_ptr();
                    batchlas::impl::parallel_for<LanczosKernel<B, T, MF>>(h, sycl::nd_range<1>(batch_size*wg_lanczos, wg_lanczos), [=](sycl::nd_item<1> item) {
                        auto bid = item.get_group_linear_id();
                        if (masked && !mk[bid]) return;
                        const dla::RtGroup<> ex{item};
                        const auto v_prev = dla::vec(v_prev_ptr + bid*(n+1)*n, n);
                        const auto v_cur = dla::vec(v_current_ptr + bid*(n+1)*n, n);
                        const auto w = dla::vec(v_next_ptr + bid*2*n, n);  // A v_it on entry
                        T* const alpha_b = alphas.data() + bid*n;
                        T* const beta_b = betas.data() + bid*n;
                        // x^T y with every product rounded before the sum: the SLM-staged legacy reduction.
                        auto dot_unfused = [&](const auto& x, const auto& y) {
                            return dla::reduce(ex, n, [&](int i) { return dla::Arith<T>::mul(x(i), y(i)); });
                        };

                        const T alpha = dot_unfused(v_cur, w);
                        alpha_b[it] = alpha;
                        if (it > 0) {
                            // beta_{it-1} takes the sign of v_{it-1}^T A v_it, which reorthogonalization may flip.
                            const T beta_prev = beta_b[it - 1];
                            const T coupling = dot_unfused(v_prev, w);
                            const T signed_beta = sycl::copysign(beta_prev, coupling);
                            if (ex.leader()) beta_b[it - 1] = signed_beta;
                            dla::map(ex, w, [=](T wi, T p, T c) { return wi - (signed_beta * p + alpha * c); }, w, v_prev, v_cur);
                        } else {
                            dla::axpy(ex, -alpha, v_cur, w);
                        }
                        const T beta = sycl::sqrt(dla::sumsq<dla::alg::Unfused>(ex, w));
                        beta_b[it] = beta;
                        if (it < iterations - 1)
                            dla::map(ex, dla::vec(Vmem.data() + bid*(n+1)*n + (it+1)*n, n), [=](T v) { return dla::Arith<T>::div_real(v, beta); }, w);
                    });
                });
            }
        };

        //Allocate workspace and compute eigenvalues/eigenvectors of the tridiagonal matrix
        auto Q_eigenvectors = pool.allocate<T>(ctx, jobz == JobType::EigenVectors ? n * n * batch_size : 0);
        const auto steqr_params = lanczos_steqr_params<T>();
        auto steqr_ws = pool.allocate<std::byte>(ctx, steqr_buffer_size<T>(
                                                    ctx,
                                                    VectorView<real_t>(alphas.data(), n, batch_size).with_batch_mask(mask),
                                                    lanczos_offdiagonals<real_t>(betas.data(), n, batch_size).with_batch_mask(mask),
                                                    VectorView<real_t>(W.data(), n, batch_size).with_batch_mask(mask),
                                                    jobz,
                                                    steqr_params));

        init();
        lanczos_iteration(n);

        (void)steqr<B> (ctx,
            VectorView<real_t>(alphas.data(), n, batch_size).with_batch_mask(mask),
            lanczos_offdiagonals<real_t>(betas.data(), n, batch_size).with_batch_mask(mask),
            VectorView<real_t>(W.data(), n, batch_size).with_batch_mask(mask),
            steqr_ws,
            jobz,
            steqr_params,
            MatrixView<T, MatrixFormat::Dense>(Q_eigenvectors.data(), n, n, n, n*n, batch_size).with_batch_mask(mask)
        );

        if (jobz == JobType::EigenVectors) {
            (void)gemm<B>(ctx,
                    MatrixView(Vmem.data(), n, n, n, (n+1)*n, batch_size).with_batch_mask(mask),
                    MatrixView(Q_eigenvectors.data(), n, n, n, n*n, batch_size).with_batch_mask(mask),
                    V,
                    GemmOptions<T>{});
        }
        //Sort the eigenvalues and eigenvectors using sycl::experimental::joint_sort
        if (params.sort_enabled){
            auto sort_workspace = pool.allocate<std::byte>(ctx, sort_buffer_size(ctx, W, V, jobz));
            // The call's mask, whichever operand brought it: V may be unmasked or, without vectors, a dummy.
            const bool vecs = jobz == JobType::EigenVectors && !mask.empty();
            (void)sort(ctx, VectorView<real_t>(W.data(), n, batch_size, 1, n).with_batch_mask(mask),
                       vecs ? V.with_batch_mask(mask) : V, jobz, params.sort_order, sort_workspace);
        }

        return ctx.get_event();
    }

    template <Backend B, typename T, MatrixFormat MF>
    size_t lanczos_buffer_size(
        Queue& ctx,
        const MatrixView<T, MF>& A,
        Span<typename base_type<T>::type> W, //Output eigenvalues
        JobType jobz,
        const MatrixView<T, MatrixFormat::Dense>& V, //Output eigenvectors for jobz == JobType::EigenVectors
        const LanczosParams<T>& params
    ) {
        auto n = A.rows();
        auto batch_size = A.batch_size();
        auto padded_vector_view = MatrixView(V.data_ptr(), n, 2, n, (n+1)*n, batch_size);
        auto padded_output_vector_view = MatrixView(V.data_ptr(), n, 2, n, (2)*n, batch_size);
        auto single_vector_view = MatrixView(V.data_ptr(), n, 1, n, (n+1)*n, batch_size);
        auto subspace_view = MatrixView(V.data_ptr(), n, n - 1, n, (n+1)*n, batch_size);  // ortho's largest basis, v_0..v_{n-2}

        // Basic memory required for Lanczos iteration
        size_t basic_size = BumpAllocator::allocation_size<T>(ctx, (n+1)*n*batch_size) +
                          BumpAllocator::allocation_size<T>(ctx, n*2*batch_size) +
                          BumpAllocator::allocation_size<T>(ctx, n*batch_size) +    // alphas
                          BumpAllocator::allocation_size<T>(ctx, n*batch_size) +    // betas
                          BumpAllocator::allocation_size<T*>(ctx, batch_size) * 3 +
                          BumpAllocator::allocation_size<std::byte>(ctx, ortho_buffer_size<B>(ctx, single_vector_view, subspace_view, Transpose::NoTrans, Transpose::NoTrans, params.ortho_algorithm, params.ortho_iterations));
        if constexpr (!(MF == MatrixFormat::Dense)){
            basic_size += BumpAllocator::allocation_size<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, padded_vector_view, padded_output_vector_view, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans));
        }
        // Additional memory required for the tridiagonal solver
        basic_size += BumpAllocator::allocation_size<std::byte>(ctx,
                        steqr_buffer_size<T>(ctx,
                            VectorView<typename base_type<T>::type>(nullptr, n, batch_size),
                            lanczos_offdiagonals<typename base_type<T>::type>(nullptr, n, batch_size),
                            VectorView<typename base_type<T>::type>(W.data(), n, batch_size),
                            jobz,
                            lanczos_steqr_params<T>()));

        // Eigenvector memory (only if needed)
        size_t eigenvectors_size = (jobz == JobType::EigenVectors) ?
                                 BumpAllocator::allocation_size<T>(ctx, n*n*batch_size) :
                                 BumpAllocator::allocation_size<T>(ctx, n*batch_size);

        size_t sort_size = sort_buffer_size(ctx, W, V, jobz);
        return basic_size  + eigenvectors_size + sort_size;
    }

    #define LANCZOS_INSTANTIATE(back, fp, mf) \
    template Event lanczos<back, fp, mf>( \
        Queue&, \
        const MatrixView<fp, mf>&, \
        Span<typename base_type<fp>::type>, \
        Span<std::byte>, \
        JobType, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const LanczosParams<fp>&); \
    template size_t lanczos_buffer_size<back, fp, mf>( \
        Queue&, \
        const MatrixView<fp, mf>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const LanczosParams<fp>&); \

    #define LANCZOS_INSTANTIATE_FOR_TYPE(back, fp) \
        BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(LANCZOS_INSTANTIATE, back, BATCHLAS_UNPAREN fp)

    BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(LANCZOS_INSTANTIATE_FOR_TYPE)

    //LANCZOS_INSTANTIATE_FOR_FP(std::complex<float>)
    //LANCZOS_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef LANCZOS_INSTANTIATE
    #undef LANCZOS_INSTANTIATE_FOR_TYPE


}
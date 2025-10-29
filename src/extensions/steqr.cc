#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/mempool.hh>
#include "../math-helpers.hh"
#include "../queue.hh"
#include <internal/sort.hh>

namespace batchlas {

template <typename T>
auto eigenvalues_2x2(const T& a, const T& b, const T& c) {
    // LAPACK SLAE2/DLAE2-style stable computation of eigenvalues of a 2x2
    // symmetric matrix [[A, B], [B, C]]. If (b != c), we symmetrize.
    const auto one  = T(1);
    const auto two  = T(2);
    const auto zero = T(0);
    const auto half = T(0.5);

    const auto sm  = a + c;          // sum (trace)
    const auto df  = a - c;          // difference
    const auto adf = std::abs(df);
    const auto tb  = b + b;          // 2*B
    const auto ab  = std::abs(tb);

    // Select larger/smaller of A and C by magnitude as in LAPACK
    const auto acmx = (std::abs(a) > std::abs(c)) ? a : c;
    const auto acmn = (std::abs(a) > std::abs(c)) ? c : a;

    T rt; // sqrt(adf^2 + (2B)^2) without overflow/underflow
    if (adf > ab) {
        rt = adf * std::sqrt(one + (ab / adf) * (ab / adf));
    } else if (adf < ab) {
        rt = ab * std::sqrt(one + (adf / ab) * (adf / ab));
    } else { // includes case ab = adf = 0
        rt = ab * std::sqrt(two);
    }

    T rt1, rt2;
    if (sm < zero) {
        rt1 = half * (sm - rt);
        // Order of operations important for an accurate smaller eigenvalue
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if (sm > zero) {
        rt1 = half * (sm + rt);
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        // includes case rt1 = rt2 = 0
        rt1 = half * rt;
        rt2 = -half * rt;
    }

    return std::make_pair(rt1, rt2);
}

template <typename T>
auto eigenvalues_2x2(const KernelMatrixView<T, MatrixFormat::Dense>& A) {
    const auto a = A(0, 0);
    const auto b = A(0, 1);
    const auto c = A(1, 0);
    const auto d = A(1, 1);
    return eigenvalues_2x2(a, b, c);
}

template <typename T>
auto wilkinson_shift(const T& a, const T& b, const T& c) {
    // Compute the Wilkinson shift assuming that a, b, c represents the
    // bottom-right 2x2 block of a matrix
    // |     :  :|
    // |     :  :|
    // |.... a, b|
    // |.... b, c|
    // Returns the eigenvalue closest to c
    const auto [lambda1, lambda2] = eigenvalues_2x2(a, b, c);
    return std::abs(lambda1 - c) < std::abs(lambda2 - c) ? lambda1 : lambda2;
}

template <typename T>
auto givens_rotation(const T& a, const T& b) {
    T r = std::hypot(a, b);
    if (internal::is_numerically_zero(r)) {
        return std::array<T, 2>{T(1), T(0)};
    }
    return std::array<T, 2>{a / r,  - b / r};
}

template <typename T>
T apply_givens_rotation(const VectorView<T>& d, const VectorView<T>& e, const T& prev_bulge, size_t i, size_t j, const std::array<T, 2>& givens) {
    // Apply similarity transform to rows/cols i and j of tridiagonal matrix T
    // G^T @ T @ G
    // Returns the bulge element introduced by the rotation
    T c = givens[0]; //Gamma
    T s = givens[1]; //Sigma
    T di = d(i);
    T dj = d(j);
    T ei = e(i);
    T ej = j < e.size() ? e(j) : T(0);
    d(i) = c * (c * di - ei * s) - s * (ei * c - s * dj);
    d(j) = c * (c * dj + ei * s) + s * (ei * c + s * di);
    if (i > 0) e(i - 1) = e(i - 1) * c - prev_bulge * s;
    e(i) = c * (c * ei + s * di) - s * (c * dj + s * ei);
    if (j < e.size()) e(j) = c * ej;
    return -ej * s; // Return the bulge element
}

template <typename T>
Event francis_sweep(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, JobType jobz,
                    const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, size_t max_sweeps, T zero_threshold) {
    // Perform the Francis sweep for the i-th step
    // This function will apply a francis sweep of Givens rotations
    auto n = d.size();
    auto batch_size = d.batch_size();
    bool store_givens = jobz == JobType::EigenVectors;
    auto ncus = ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    auto bsize = batch_size < ncus ? 1 : internal::ceil_div(size_t(batch_size), ncus);
    auto event = ctx->submit([&](sycl::handler& cgh) {
        auto rotations_view = givens_rotations.kernel_view();
        cgh.parallel_for(sycl::nd_range(sycl::range(internal::ceil_div(batch_size, int(bsize)) * bsize), sycl::range(bsize)), [=](sycl::nd_item<1> item) {
            auto i = item.get_global_id(0);
            if (i >= batch_size) return;
            auto d_ = d.batch_item(i);
            auto e_ = e.batch_item(i);
            for (size_t k = 0; k < max_sweeps; ++k) {
                auto shift = wilkinson_shift(d_(n - 2), 
                                             e_(n - 2), 
                                             d_(n - 1));
                T a = d_(0);
                T b = e_(0);
                auto [c, s] = givens_rotation(a - shift, b);
                if (store_givens) { rotations_view(0, k, i) = {c, s};}
                //if (store_givens) { givens_rotations.data()[(k * (n - 1))  + max_sweeps * (n-1) *i] = {c, s};}
                auto bulge = apply_givens_rotation(d_, e_, T(0.), 0, 1, {c, s});
                for (size_t j = 1; j < n - 1; ++j) {
                    auto [c, s] = givens_rotation(e_(j - 1), bulge);
                    if (store_givens) { rotations_view(j, k, i) = {c, s};}
                    //if (store_givens) { givens_rotations.data()[(k * (n - 1) + j)  + max_sweeps * (n-1) *i] = {c, s};}
                    bulge = apply_givens_rotation(d_, e_, bulge, j, j + 1, {c, s});
                }
                if (std::abs(e_(n - 2)) < zero_threshold) {
                    // If the sub-diagonal element is zero, we can skip further sweeps
                    break;
                }
            }
        });
    });
    return ctx.get_event();
}

template <Backend B, typename T>
Event block_rot_impl(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, 
              const MatrixView<T, MatrixFormat::Dense>& eigvects, const Span<std::byte>& ws, size_t Nb) {
    //Nb is the block size of the intermediate givens matrices we form and apply
    BumpAllocator pool(ws);
    auto N = 2 * Nb;
    auto batch_size = givens_rotations.batch_size();
    auto givens_block = MatrixView<T>(pool.allocate<T>(ctx, batch_size * N * N).data(), N, N, N, N * N, batch_size);
    auto eigvects_dual = MatrixView<T>(pool.allocate<T>(ctx, batch_size * eigvects.rows() * N).data(), eigvects.rows(), N, eigvects.rows(), eigvects.rows() * N, batch_size);

    //ctx->memcpy(static_cast<void*>(eigvects_dual.data().data()), static_cast<void*>(eigvects.data_ptr()), eigvects.data().size_bytes());
    //givens_block.fill_identity(ctx);
    //Apply Prologue rotations: (Example here if Nb = 4)
    // G[0,0] , G[0,1] , G[0,2] , G[0,3]
    // G[1,0] , G[1,1] , G[1,2] 
    // G[2,0] , G[2,1]
    // G[3,0]
    
    ctx -> submit([&](sycl::handler& cgh) {
        auto rotations = givens_rotations.kernel_view();
        auto smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= sizeof(T) * N * N;
        sycl::local_accessor<T, 1> local_mem(sycl::range<1>(smem_possible ? N * N : 0), cgh);
        auto givens_view = givens_block.kernel_view();
        cgh.parallel_for(sycl::nd_range(sycl::range(batch_size*(Nb + 1)), sycl::range(Nb + 1)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            auto G = smem_possible ? KernelMatrixView<T, MatrixFormat::Dense>(local_mem.get_pointer(), N, N) :  givens_view.batch_item(bid);
            auto Gglobal = givens_view.batch_item(bid);
            for (size_t j = 0; j < N; ++j){
                G(k, j) = k == j ? T(1) : T(0); //Identity fill-in
            }
            for (size_t i = 0; i < Nb; ++i) {
                size_t j_start = (k > i + 1) ? (k - i - 1) : 0;
                for (size_t j = 0; j < Nb - i; ++j) {
                    auto [c, s] = rotations(j, i, bid);
                    if (c == T(1) && s == T(0)) continue; //Skip identity rotations
                    if (k < (2 + j + i)) {
                        T temp = c * G(k, j) - s * G(k, j + 1);
                        G(k, j + 1) = s * G(k, j) + c * G(k, j + 1);
                        G(k, j) = temp;
                    }
                }
            }
            if (smem_possible) {
                //Copy back to global memory
                for (size_t j = 0; j < N; ++j) {
                    Gglobal(k, j) = G(k, j);
                }
            }
        });
    });
    
    auto p_slice = Slice(0, int64_t(Nb + 1));
    gemm<B>(ctx, eigvects(Slice(),p_slice), givens_block(p_slice,p_slice), eigvects_dual(Slice(),p_slice), T(1), T(0),
                        Transpose::NoTrans, Transpose::NoTrans);
    MatrixView<T>::copy(ctx, eigvects(Slice(),p_slice), eigvects_dual(Slice(),p_slice));

    //Apply regular rotation sets
    auto row_remainder = givens_rotations.rows() % Nb;
    for (int64_t i = Nb; i < givens_rotations.rows() - row_remainder; i += Nb) {
        //givens_block.fill_identity(ctx).wait();
        //ctx->wait();
        //std::cout << "Post identityfill regular sweep givens_block:\n" << givens_block << std::endl;

        ctx->submit([&](sycl::handler& cgh) {
            auto rotations = givens_rotations.kernel_view();
            auto smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= sizeof(T) * N * N;
            sycl::local_accessor<T, 1> local_mem(sycl::range<1>(smem_possible ? N * N : 0), cgh);
            auto givens_view = givens_block.kernel_view();
            auto row_offset = i;
            cgh.parallel_for(sycl::nd_range(sycl::range(N * batch_size), sycl::range(N)), [=](sycl::nd_item<1> item) {
                auto k =      item.get_local_linear_id();
                auto bid =      item.get_group(0);
                auto G = smem_possible ? KernelMatrixView<T, MatrixFormat::Dense>(local_mem.get_pointer(), N, N) :  givens_view.batch_item(bid);
                auto Gglobal = givens_view.batch_item(bid);
                for (size_t j = 0; j < N; ++j){
                   G(k, j) = k == j ? T(1) : T(0); //Identity fill-in
                }
                for (size_t i = 0; i < Nb; ++i) {
                    for (size_t j = 0; j < Nb; ++j) {
                        auto [c, s] = rotations(row_offset + j - i, i, bid);
                        if (c == T(1) && s == T(0)) continue; //Skip identity rotations
                        auto j0 = j - i + Nb - 1;
                        auto j1 = j0 + 1;
                        //At sweep i -> i + Nb
                        T temp = c * G(k, j0) - s * G(k, j1);
                        G(k, j1) = s * G(k, j0) + c * G(k, j1);
                        G(k, j0) = temp;
                    }
                }
                if (smem_possible) {
                    //Copy back to global memory
                    for (size_t j = 0; j < N; ++j) {
                        Gglobal(k, j) = G(k, j);
                    }
                }
            });
        });

        //Post-multiply a slice of eigvects by the givens-block matrix
        auto col_slice = Slice(int64_t(i - Nb + 1), int64_t(i + Nb + 1));
        gemm<B>(ctx, eigvects(Slice{}, col_slice), 
                     givens_block, 
                     eigvects_dual, 
                     T(1), T(0),
                     Transpose::NoTrans, Transpose::NoTrans);

        MatrixView<T>::copy(ctx, eigvects(Slice{}, col_slice), eigvects_dual);
    }


    //Epilogue rotations (example where Nb = 4, and col_remainder = 3)
    //                                            G[0, m-3], G[0, m-2], G[0, m-1]
    //                                 G[1, m-4], G[1, m-3], G[1, m-2], G[1, m-1]
    //                      G[2, m-5], G[2, m-4], G[2, m-3], G[2, m-2], G[2, m-1]
    //           G[3, m-6], G[3, m-5], G[3, m-4], G[3, m-3], G[3, m-2], G[3, m-1]

    auto row_offset = givens_rotations.rows() - row_remainder;
    ctx -> submit([&](sycl::handler& cgh) {
        auto rotations = givens_rotations.kernel_view();
        auto smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= sizeof(T) * N * N;
        sycl::local_accessor<T, 1> local_mem(sycl::range<1>(smem_possible ? N * N : 0), cgh);
        auto givens_view = givens_block.kernel_view();
        
        cgh.parallel_for(sycl::nd_range(sycl::range(batch_size*(N)), sycl::range(N)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            auto G = smem_possible ? KernelMatrixView<T, MatrixFormat::Dense>(local_mem.get_pointer(), N, N) :  givens_view.batch_item(bid);
            auto Gglobal = givens_view.batch_item(bid);
            for (size_t j = 0; j < N; ++j){
                G(k, j) = k == j ? T(1) : T(0); //Identity fill-in
            }
            for (size_t i = 0; i < Nb; ++i) { //Apply rotations
                for (size_t j = 0; j < row_remainder + i; ++j) {
                    auto [c, s] = rotations(row_offset + j - i, i, bid);
                    if (c == T(1) && s == T(0)) continue; //Skip identity rotations
                    auto j0 = j - i + Nb - 1;
                    auto j1 = j0 + 1;
                    T temp = c * G(k, j0) - s * G(k, j1);
                    G(k, j1) = s * G(k, j0) + c * G(k, j1);
                    G(k, j0) = temp;
                }
            }
            if (smem_possible) {
                //Copy back to global memory
                for (size_t j = 0; j < N; ++j) {
                   Gglobal(k, j) = G(k, j);
                }
            }
        });
    });
    //ctx -> wait();

    auto e_slice = Slice{int64_t(row_offset - Nb + 1), SliceEnd()};
    gemm<B>(ctx, eigvects(Slice{},e_slice), givens_block({0, int64_t(Nb + row_remainder)}, {0, int64_t(Nb + row_remainder)}), eigvects_dual(Slice{}, {0, int64_t(Nb + row_remainder)}), T(1), T(0),
    Transpose::NoTrans, Transpose::NoTrans);
    MatrixView<T>::copy(ctx, eigvects(Slice{},e_slice), eigvects_dual(Slice{}, {0, int64_t(Nb + row_remainder)}));
    //ctx -> wait();
    //std::cout << "Post-GEMM eigvects:\n" << eigvects << std::endl;
    return ctx.get_event();
}

template <Backend B, typename T>
Event block_rot(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, 
              const MatrixView<T, MatrixFormat::Dense>& eigvects, const Span<std::byte>& ws, size_t max_block_size) 
{
    //std::cout << "All rotations:\n" << givens_rotations << std::endl;
    auto sweep_remainder = givens_rotations.cols() % max_block_size;
    for (int64_t i = 0; i < givens_rotations.cols() - sweep_remainder; i+=max_block_size)
    {
        block_rot_impl<B>(ctx, givens_rotations({i, int64_t(i + max_block_size)}, Slice()), eigvects, ws, max_block_size);
    }
    if (sweep_remainder == 0) return ctx.get_event();
    return block_rot_impl<B>(ctx, givens_rotations({int64_t(givens_rotations.cols() - sweep_remainder), SliceEnd()}, Slice()), eigvects, ws, sweep_remainder);
}

template <Backend B, typename T>
Event rot(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, const MatrixView<T, MatrixFormat::Dense>& Q) {
    ctx -> submit([&](sycl::handler& cgh) {
        auto Q_ = Q.kernel_view();
        auto rotations_view = givens_rotations.kernel_view();
        cgh.parallel_for(sycl::nd_range(sycl::range(Q.rows()*Q.batch_size()), sycl::range(Q.rows())), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            for (int i = 0; i < rotations_view.cols; ++i) {
                for (int j = 0; j < rotations_view.rows; ++j) {
                    auto [c, s] = rotations_view(j, i, bid);
                    if (c == T(1) && s == T(0)) continue; //Skip identity rotations
                    T temp = c * Q_(k, j, bid) - s * Q_(k, j + 1, bid);
                    Q_(k, j + 1, bid) = s * Q_(k, j, bid) + c * Q_(k, j + 1, bid);
                    Q_(k, j, bid) = temp;
                }
            }
        });
    });
    return ctx.get_event();
}

template <Backend B, typename T>
Event steqr(Queue& ctx, const VectorView<T>& d_in, const VectorView<T>& e_in, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, SteqrParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    // Ensure the matrix is square
    if (eigvects.rows() != eigvects.cols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation.");
    }
    if (!params.back_transform) {
        eigvects.fill_identity(ctx);
    }
    
    int64_t n = d_in.size();
    int64_t batch_size = d_in.batch_size();   
    auto pool = BumpAllocator(ws);
    auto d = VectorView<T>(pool.allocate<T>(ctx, batch_size * n).data(), n, 1, n, batch_size);
    auto e = VectorView<T>(pool.allocate<T>(ctx, batch_size * (n - 1)).data(), n - 1, 1, n - 1, batch_size);
    //Copy inputs to working buffers
    VectorView<T>::copy(ctx, d, d_in);
    VectorView<T>::copy(ctx, e, e_in);

    auto apply_Q_ws = pool.allocate<T>(ctx, jobz == JobType::EigenVectors ? (batch_size * params.block_size*2 * params.block_size*2 + batch_size*n*params.block_size*4) : 0);
    auto n_sweeps_to_store = (jobz == JobType::EigenVectors && params.block_rotations)? std::max(params.block_size * 2, params.max_sweeps) : params.max_sweeps;
    auto stride = (n - 1) * n_sweeps_to_store;
    auto givens_rotations = jobz == JobType::EigenVectors ?  MatrixView<std::array<T,2>>(pool.allocate<std::array<T,2>>(ctx, stride * batch_size).data(), n - 1, n_sweeps_to_store, n - 1, stride, batch_size) : MatrixView<std::array<T,2>>();
    //auto mock_eigen = Matrix<T>::Identity(n, batch_size);
    for (int64_t i = 0; i < n - 1; ++i) {
        givens_rotations.fill(ctx, std::array<T,2>{1, 0}); //Fill with identity rotations
        francis_sweep(ctx, d(Slice{0, n - i}), e(Slice{0, n - i - 1}), jobz, givens_rotations, params.max_sweeps, params.zero_threshold);
        if (jobz == JobType::EigenVectors) {
            if (params.block_rotations) {
                block_rot<B>(ctx, givens_rotations, eigvects, apply_Q_ws.template as_span<std::byte>(), params.block_size);
            } else {
                rot<B>(ctx, givens_rotations, eigvects);
            }
        }
    }

    ctx -> submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range(sycl::range(batch_size* n), sycl::range(n)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto intra = item.get_local_id(0);
            eigenvalues(intra, bid) = d(intra, bid);
        });
    });

    if (params.sort){
        auto ws_sort = pool.allocate<std::byte>(ctx, sort_buffer_size<T>(ctx, eigenvalues.data(), eigvects, jobz));
        sort(ctx, eigenvalues, eigvects, jobz, params.sort_order, ws_sort);
    }
    return ctx.get_event();
}

template <typename T>
size_t steqr_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                            const VectorView<T>& eigenvalues, JobType jobz, SteqrParams<T> params) {
    // Calculate the required buffer size for the workspace
    size_t size = BumpAllocator::allocation_size<T>(ctx, d.batch_size() * d.size()) // For d
                + BumpAllocator::allocation_size<T>(ctx, d.batch_size() * (d.size() - 1)); // For e
    if (jobz == JobType::EigenVectors) {
        size += BumpAllocator::allocation_size<std::array<T,2>>(ctx, d.batch_size() * d.size() * params.max_sweeps);
        size += BumpAllocator::allocation_size<T>(ctx, d.batch_size() * params.block_size * params.block_size * 4);
        size += BumpAllocator::allocation_size<T>(ctx, d.batch_size() * 8 * params.block_size * d.size());
    }
    size += sort_buffer_size<T>(ctx, eigenvalues.data(), MatrixView<T, MatrixFormat::Dense>(nullptr, d.size(), d.size(), d.size(), d.size() * d.size(), d.batch_size()), jobz);
    return size;
}



template Event steqr<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event steqr<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
template Event steqr<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event steqr<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
template size_t steqr_buffer_size<float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, JobType, SteqrParams<float>);
template size_t steqr_buffer_size<double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, JobType, SteqrParams<double>); 

} // namespace batchlas
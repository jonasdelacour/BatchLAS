#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/kernel-heuristics.hh>
#include <util/mempool.hh>
#include "../math-helpers.hh"
#include "../queue.hh"

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
    auto event = ctx->submit([&](sycl::handler& cgh) {
        auto givens_view = givens_rotations.kernel_view();
        cgh.parallel_for(sycl::nd_range(sycl::range(internal::ceil_div(batch_size,128) * 128), sycl::range(128)), [=](sycl::nd_item<1> item) {
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
                if (store_givens) { givens_view(k, 0, i) = {c, s};}
                auto bulge = apply_givens_rotation(d_, e_, T(0.), 0, 1, {c, s});
                for (size_t j = 1; j < n - 1; ++j) {
                    auto [c, s] = givens_rotation(e_(j - 1), bulge);
                    if (store_givens) { givens_view(k, j, i) = {c, s};}
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
Event apply_Q(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, 
              const MatrixView<T, MatrixFormat::Dense>& eigvects, const Span<std::byte>& ws, size_t Nb) {
    //Nb is the block size of the intermediate givens matrices we form and apply
    BumpAllocator pool(ws);
    auto givens_block = MatrixView<T>(pool.allocate<T>(ctx, Nb * Nb).data(), Nb, Nb);
    auto batch_size = givens_rotations.batch_size();
    for (size_t i = 0; i < givens_rotations.rows(); i += Nb) {
        givens_block.fill_identity(ctx);

        ctx->submit([&](sycl::handler& cgh) {
            auto givens_view = givens_block.kernel_view();
            auto rotations_view = givens_rotations.kernel_view();
            cgh.parallel_for(sycl::nd_range(sycl::range(Nb * batch_size), sycl::range(Nb)), [=](sycl::nd_item<1> item) {
                auto tid =      item.get_local_linear_id();
                auto i_block =  item.get_group(0);
                auto givens_block = givens_view.batch_item(i_block);
                for (size_t j = 0; j < Nb; ++j) {
                    for (size_t k = 0; k < Nb; ++k) {
                        //At sweep i -> i + Nb
                        auto [c, s] = rotations_view(j, k, i_block);
                        auto left_column =  c * givens_block(tid, k) + s * givens_block(tid, k + 1);
                        auto right_column = -s * givens_block(tid, k) + c * givens_block(tid, k + 1);
                        givens_block(tid, k) = left_column;
                        givens_block(tid, k + 1) = right_column;
                    }
                }
            });
        });
        //Post-multiply a slice of eigvects by the givens-block matrix
        gemm<B>(ctx, eigvects({0, SliceEnd()},{0, int64_t(Nb)}), givens_block, eigvects({0, SliceEnd()},{0, int64_t(Nb)}), T(1), T(0),
                            Transpose::NoTrans, Transpose::NoTrans);
    }
}

template <Backend B, typename T>
void rot(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    auto G = eigvects;
    ctx -> submit([&](sycl::handler& cgh) {
        auto G_view = G.kernel_view();
        auto rotations_view = givens_rotations.kernel_view();
        cgh.parallel_for(sycl::nd_range(sycl::range(G.rows()*G.batch_size()), sycl::range(G.rows())), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            for (int i = 0; i < rotations_view.rows; ++i) {
                for (int j = 0; j < rotations_view.cols; ++j) {
                    auto [c, s] = rotations_view(i, j, bid);
                    T temp = c * G_view(k, j, bid) - s * G_view(k, j + 1, bid);
                    G_view(k, j + 1, bid) = s * G_view(k, j, bid) + c * G_view(k, j + 1, bid);
                    G_view(k, j, bid) = temp;
                }
            }
        });
    });
}

template <Backend B, typename T>
Event steqr(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, SteqrParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    // Ensure the matrix is square
    if (eigvects.rows() != eigvects.cols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation.");
    }
    eigvects.fill_identity(ctx);

    int64_t n = d.size();
    int64_t batch_size = d.batch_size();   
    auto pool = BumpAllocator(ws);
    auto apply_Q_ws = pool.allocate<std::byte>(ctx, jobz == JobType::EigenVectors ? (2* batch_size * params.block_size * params.block_size * sizeof(T)) : 0);
    auto givens_rotations = MatrixView<std::array<T,2>, MatrixFormat::Dense>(pool.allocate<std::array<T,2>>(ctx, jobz == JobType::EigenVectors ? (batch_size * (n - 1) * params.max_sweeps) : 0).data(), params.max_sweeps, n - 1, params.max_sweeps, (n - 1) * params.max_sweeps, batch_size);
    for (int64_t i = 0; i < n - 1; ++i) {
        givens_rotations.fill(ctx, std::array<T,2>{1, 0}); //Fill with identity rotations
        francis_sweep(ctx, d(Slice{0, n - i}), e(Slice{0, n - i - 1}), jobz, givens_rotations, params.max_sweeps, params.zero_threshold);
        //if (jobz == JobType::EigenVectors) {ctx -> wait(); std::cout << givens_rotations << std::endl;}
        //apply_Q<B>(ctx, givens_rotations, eigvects, apply_Q_ws, params.block_size);
        //ctx->wait();
        rot<B>(ctx, givens_rotations, eigvects);
    }
    ctx -> submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range(sycl::range(batch_size* n), sycl::range(n)), [=](sycl::nd_item<1> item) {
            auto idx = item.get_global_id(0);
            auto intra = item.get_local_id(0);
            if (idx < n) {
                eigenvalues(intra, idx) = d(intra, idx);
            }
        });
    });
    return ctx.get_event();
}

template <typename T>
size_t steqr_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                            const VectorView<T>& eigenvalues, JobType jobz, SteqrParams<T> params) {
    // Calculate the required buffer size for the workspace
    size_t size = 0;
    if (jobz == JobType::EigenVectors) {
        size += BumpAllocator::allocation_size<std::array<T,2>>(ctx, d.batch_size() * d.size() * params.max_sweeps);
        size += BumpAllocator::allocation_size<T>(ctx, d.batch_size() * params.block_size * params.block_size * 2);
    }
    return size;
}



template Event steqr<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event steqr<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
template Event steqr<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event steqr<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
template size_t steqr_buffer_size<float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, JobType, SteqrParams<float>);
template size_t steqr_buffer_size<double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, JobType, SteqrParams<double>); 

} // namespace batchlas
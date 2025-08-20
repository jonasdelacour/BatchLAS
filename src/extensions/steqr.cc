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
auto eigenvalues_2x2(const MatrixView<T>& A) {
    if (A.rows() != 2 || A.cols() != 2) {
        throw std::invalid_argument("Matrix must be 2x2 for eigenvalue computation.");
    }
    const auto a = A(0, 0);
    const auto b = A(0, 1);
    const auto c = A(1, 0);
    const auto d = A(1, 1);
    return eigenvalues_2x2(a, b, c);
}

template <typename T>
auto wilkinson_shift(const T& a, const T& b, const T& c) {
    // Compute the Wilkinson shift for a 2x2 symmetric matrix
    // |a, b|
    // |b, c|
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
T apply_givens_rotation(const VectorView<T>& d, const VectorView<T>& e, const T& prev_bulge, size_t i, size_t j, size_t bid, const std::array<T, 2>& givens) {
    // Apply similarity transform to rows/cols i and j of tridiagonal matrix T
    // G^T @ T @ G
    // Returns the bulge element introduced by the rotation
    T c = givens[0]; //Gamma
    T s = givens[1]; //Sigma
    T di = d(i,bid);
    T dj = d(j,bid);
    T ei = e(i,bid);
    T ej = j < e.size() ? e(j,bid) : T(0);
    d(i, bid) = c * (c * di - ei * s) - s * (ei * c - s * dj);
    d(j, bid) = c * (c * dj + ei * s) + s * (ei * c + s * di);
    if (i > 0) e(i - 1, bid) = e(i - 1, bid) * c - prev_bulge * s;
    e(i, bid) = c * (c * ei + s * di) - s * (c * dj + s * ei);
    if (j < e.size()) e(j, bid) = c * ej;
    return -ej * s; // Return the bulge element
}

template <typename T>
Event francis_sweep(const Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, 
                    const MatrixView<T, MatrixFormat::Dense>& givens_rotations, size_t n_sweeps) {
    // Perform the Francis sweep for the i-th step
    // This function will apply a francis sweep of Givens rotations
    auto n = d.size();
    auto batch_size = d.batch_size();
    auto event = ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range(sycl::range(internal::ceil_div(batch_size,128) * 128), sycl::range(128)), [=](sycl::nd_item<1> item) {
            auto i = item.get_global_id(0);
            if (i >= batch_size) return;
            for (size_t k = 0; k < n_sweeps; ++k) {
                auto shift = wilkinson_shift(   d(n - 2, i), 
                                                e(n - 2, i), 
                                                d(n - 1, i));
                T a = d(0, i);
                T b = e(0, i);
                auto [c, s] = givens_rotation(a - shift, b);
                auto bulge = apply_givens_rotation(d, e, T(0.), 0, 1, i, {c, s});
                for (size_t j = 1; j < n - 1; ++j) {
                    auto [c, s] = givens_rotation(e(j - 1, i), bulge);
                    bulge = apply_givens_rotation(d, e, bulge, j, j + 1, i, {c, s});
                }
                if (internal::is_numerically_zero(e(n - 2, i))) {
                    // If the sub-diagonal element is zero, we can skip further sweeps
                    break;
                }
            }
        });
    });
    return ctx.get_event();
}

template <typename T>
Event form_Q(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& givens_rotations, 
              const MatrixView<T, MatrixFormat::Dense>& eigvects, const Span<std::byte>& ws, size_t Nb) {
    //Nb is the block size of the intermediate givens matrices we form and apply
    BumpAllocator pool(ws);
    auto givens_block = MatrixView<T>(pool.allocate<T>(ctx, Nb * Nb), Nb, Nb);    
    for (size_t i = 0; i < givens_rotations.rows(); i += Nb) {
       /*  givens_block.fill_identity(ctx);
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range(sycl::range(Nb * batch_size), sycl::range(Nb)), [=](sycl::nd_item<1> item) {
                auto tid = item.get_local_linear_id();
                auto i_block = item.get_group(0);
                for (size_t j = 0; j < Nb; ++j) {
                    for (size_t k = 0; k < Nb; ++k) {
                        //At sweep i -> i + Nb 
                    }
                }
            });
        }); */
        gemm<Backend::CUDA>(ctx, eigvects, givens_block, eigvects, T(1), T(0),
                            Transpose::NoTrans, Transpose::NoTrans);
    }
}

template <typename T>
Event steqr_impl(const Queue& ctx,const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const MatrixView<T>& eigvects) {
    int64_t n = d.size();
    int64_t batch_size = d.batch_size();   
    for (int64_t i = 0; i < n - 1; ++i) {
        francis_sweep(ctx, d(Slice{0, n - i}), e(Slice{0, n - i - 1}), eigvects, 5);
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
Event steqr(const Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const MatrixView<T>& eigvects) {
    // Ensure the matrix is square
    if (eigvects.rows() != eigvects.cols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation.");
    }

    // Check if the size of d and e matches the matrix dimensions
    if (d.size() != eigvects.rows() || e.size() != eigvects.rows() - 1) {
        throw std::invalid_argument("Size of d and e must match the matrix dimensions.");
    }

    return steqr_impl(ctx, d, e, eigenvalues, eigvects);
}

template Event francis_sweep<float>(const Queue&, const VectorView<float>&, const VectorView<float>&, const MatrixView<float, MatrixFormat::Dense>&, size_t);
template Event francis_sweep<double>(const Queue&, const VectorView<double>&, const VectorView<double>&, const MatrixView<double, MatrixFormat::Dense>&, size_t);
template Event steqr<float>(const Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const MatrixView<float>&);
template Event steqr<double>(const Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const MatrixView<double>&); 

} // namespace batchlas
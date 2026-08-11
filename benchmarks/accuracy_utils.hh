#pragma once

#include <batchlas/blas/functions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include "../src/queue.hh"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace batchlas::accuracy {

inline bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline SteqrShiftStrategy parse_shift_strategy(const std::string& value) {
    const auto key = to_lower(value);
    if (key == "lapack") return SteqrShiftStrategy::Lapack;
    if (key == "wilkinson") return SteqrShiftStrategy::Wilkinson;
    throw std::invalid_argument("Invalid --cta-shift value (use lapack or wilkinson)");
}

template <typename Real>
inline void extract_tridiagonal(Queue& q,
                                const MatrixView<Real, MatrixFormat::Dense>& dense,
                                Vector<Real>& d,
                                Vector<Real>& e) {
    const int n = dense.rows();
    const int batch = dense.batch_size();
    auto a_view = dense.kernel_view();
    auto d_ptr = d.data_ptr();
    auto e_ptr = e.data_ptr();
    const int d_inc = d.inc();
    const int e_inc = e.inc();
    const int d_stride = d.stride();
    const int e_stride = e.stride();

    q->parallel_for(sycl::range<1>(static_cast<size_t>(batch * n)), [=](sycl::id<1> idx) {
        const int linear = static_cast<int>(idx[0]);
        const int b = linear / n;
        const int i = linear - b * n;
        d_ptr[b * d_stride + i * d_inc] = a_view(i, i, b);
        if (i < n - 1) {
            e_ptr[b * e_stride + i * e_inc] = a_view(i + 1, i, b);
        }
    });
    q.wait();
}

template <Backend B, typename Real>
inline UnifiedVector<typename base_type<Real>::type> orthogonality_residuals(
    Queue& q,
    const Matrix<Real, MatrixFormat::Dense>& vectors) {
    const int dimension = vectors.cols();
    const int batch = vectors.batch_size();
    auto gram_minus_i = Matrix<Real>::Identity(dimension, batch);
    gemm<B, Real>(q,
                  vectors.view(),
                  vectors.view(),
                  gram_minus_i.view(),
                  Real(1),
                  Real(-1),
                  Transpose::Trans,
                  Transpose::NoTrans);
    q.wait();
    return norm(q, gram_minus_i.view(), NormType::Frobenius);
}

} // namespace batchlas::accuracy

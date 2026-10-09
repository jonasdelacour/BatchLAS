// SPDX-License-Identifier: MIT
// Host norms of one batch item, read through ld and stride only (docs/design/verification.md).
#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/verify/scalar.hh>

#include <cmath>
#include <stdexcept>

namespace batchlas::verify {
namespace detail {

template <class T> struct Item {
    const T* data;
    int rows, cols, ld;
};

template <class T>
Item<T> make_item(const T* base, int rows, int cols, int ld, long long stride, int batch, bool hetero, int item) {
    if (hetero) throw std::invalid_argument("batchlas::verify: heterogeneous views are not supported");
    if (item < 0 || item >= batch) throw std::invalid_argument("batchlas::verify: batch item out of range");
    return {base + static_cast<long long>(item) * stride, rows, cols, ld};
}

template <class T>
Item<T> item_of(const MatrixView<T, MatrixFormat::Dense>& A, int item) {
    return make_item<T>(A.data_ptr(), A.rows(), A.cols(), A.ld(), A.stride(), A.batch_size(), A.is_heterogeneous(), item);
}

template <class T>
Item<T> item_of(const KernelMatrixView<T, MatrixFormat::Dense>& A, int item) {
    return make_item<T>(A.data_, A.rows(), A.cols(), A.ld(), A.stride(), A.batch_size(), A.is_heterogeneous(), item);
}

}  // namespace detail

/// A dense batch over raw host-readable memory: item b starts at data + b * stride; stride 0 means
/// ld * cols. @throws std::invalid_argument for ld < max(1, rows), a negative size or stride, batch < 1.
template <class T>
KernelMatrixView<T, MatrixFormat::Dense> view(T* data, int rows, int cols, int ld, int stride = 0, int batch = 1) {
    if (rows < 0 || cols < 0 || stride < 0 || batch < 1 || ld < (rows > 1 ? rows : 1))
        throw std::invalid_argument("batchlas::verify::view: invalid rows, cols, ld, stride or batch");
    return KernelMatrixView<T, MatrixFormat::Dense>(data, rows, cols, ld, stride > 0 ? stride : ld * cols, batch);
}

/// view over read-only memory; every check only reads through it.
template <class T>
KernelMatrixView<T, MatrixFormat::Dense> view(const T* data, int rows, int cols, int ld, int stride = 0, int batch = 1) {
    return view(const_cast<T*>(data), rows, cols, ld, stride, batch);
}

/// Frobenius norm of item @p item, accumulated in double.
template <class View> double frobenius(const View& A, int item) {
    const auto m = detail::item_of(A, item);
    double sum = 0.0;
    for (int j = 0; j < m.cols; ++j)
        for (int i = 0; i < m.rows; ++i) {
            const double a = abs(up(m.data[static_cast<long long>(j) * m.ld + i]));
            sum += a * a;
        }
    return std::sqrt(sum);
}

/// Largest element modulus of item @p item.
template <class View> double max_abs(const View& A, int item) {
    const auto m = detail::item_of(A, item);
    double best = 0.0;
    for (int j = 0; j < m.cols; ++j)
        for (int i = 0; i < m.rows; ++i)
            best = nanmax(best, abs(up(m.data[static_cast<long long>(j) * m.ld + i])));
    return best;
}

/// Largest absolute column sum of item @p item.
template <class View> double one_norm(const View& A, int item) {
    const auto m = detail::item_of(A, item);
    double best = 0.0;
    for (int j = 0; j < m.cols; ++j) {
        double col = 0.0;
        for (int i = 0; i < m.rows; ++i) col += abs(up(m.data[static_cast<long long>(j) * m.ld + i]));
        best = nanmax(best, col);
    }
    return best;
}

}  // namespace batchlas::verify

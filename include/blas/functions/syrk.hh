#pragma once

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <type_traits>

namespace batchlas {

template <Backend Ba, typename T,
          typename std::enable_if<std::is_floating_point_v<T>, int>::type = 0>
Event syrk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Uplo uplo,
           Transpose transA);

template <Backend Ba, typename T,
          typename std::enable_if<std::is_floating_point_v<T>, int>::type = 0>
inline Event syrk(Queue& ctx,
                  const Matrix<T, MatrixFormat::Dense>& A,
                  const Matrix<T, MatrixFormat::Dense>& Cmat,
                  T alpha,
                  T beta,
                  Uplo uplo,
                  Transpose transA) {
    return syrk<Ba, T>(ctx,
                       MatrixView<T, MatrixFormat::Dense>(A),
                       MatrixView<T, MatrixFormat::Dense>(Cmat),
                       alpha,
                       beta,
                       uplo,
                       transA);
}

} // namespace batchlas
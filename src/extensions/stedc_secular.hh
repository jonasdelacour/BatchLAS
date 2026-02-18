#pragma once

#include <blas/matrix.hh>
#include <blas/functions.hh>

namespace batchlas {

template <typename T>
SYCL_EXTERNAL T sec_solve_ext_roc(const int32_t dd,
                                  const VectorView<T>& D,
                                  const VectorView<T>& z,
                                  const T p);

template <typename T>
SYCL_EXTERNAL T sec_solve_roc(int32_t dd,
                              const VectorView<T>& d,
                              const VectorView<T>& z,
                              const T& rho,
                              const int32_t k);

template <typename T>
Event secular_solver(Queue& ctx,
                     const VectorView<T>& d,
                     const VectorView<T>& v,
                     const MatrixView<T, MatrixFormat::Dense>& Qprime,
                     const VectorView<T>& lambdas,
                     const Span<int32_t>& n_reduced,
                     const Span<T> rho,
                     const T& tol_factor = 10.0);

} // namespace batchlas

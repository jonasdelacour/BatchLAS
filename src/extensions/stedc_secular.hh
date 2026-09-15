#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>

namespace batchlas {

// `converged_out` is the flag both of these already computed and then destroyed:
// sec_solve_ext_roc ended in `(void)converged;` and sec_solve_roc in
// `assert(converged && ...)`, which is a NO-OP in a release device build -- so a
// root that never converged produced a silently wrong eigenvalue with no
// diagnostic anywhere. It is an out-parameter rather than a change of return type
// because the root itself is the return value and both are needed.
//
// It is NOT defaulted: these are SYCL_EXTERNAL, so the flag has to be threaded
// from every call site by hand, and a default would let a new call site drop the
// status again without saying so.
template <typename T>
SYCL_EXTERNAL T sec_solve_ext_roc(const int32_t dd,
                                  const VectorView<T>& D,
                                  const VectorView<T>& z,
                                  const T p,
                                  bool& converged_out);

template <typename T>
SYCL_EXTERNAL T sec_solve_roc(int32_t dd,
                              const VectorView<T>& d,
                              const VectorView<T>& z,
                              const T& rho,
                              const int32_t k,
                              bool& converged_out);

template <typename T>
Event secular_solver(Queue& ctx,
                     const VectorView<T>& d,
                     const VectorView<T>& v,
                     const MatrixView<T, MatrixFormat::Dense>& Qprime,
                     const VectorView<T>& lambdas,
                     const Span<int32_t>& n_reduced,
                     const Span<T> rho,
                     const T& tol_factor = 10.0,
                     int32_t* info = nullptr,
                     int64_t info_nodes_per_item = 1);

} // namespace batchlas

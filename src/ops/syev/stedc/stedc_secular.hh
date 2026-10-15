#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>

namespace batchlas {

// `converged_out`: whether the root met its tolerance (a release-mode assert used to be
// the only check). NOT defaulted: these are SYCL_EXTERNAL, and a default would let a
// new call site drop the status silently.
// evidence: docs/perf/stedc.md#stedc-convergence-reporting-through-info
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

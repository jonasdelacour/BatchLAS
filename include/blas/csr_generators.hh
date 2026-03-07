#pragma once

#include <blas/matrix.hh>

#include <complex>
#include <stdexcept>

namespace batchlas {
namespace csr_generators {

// Generates a sparse symmetric/Hermitian CSR matrix with approximately the requested density.
// Density is interpreted over the full matrix [0,1] including diagonal.
// The matrix storage is allocated first and then populated through the SYCL path,
// matching the dense Matrix factory style instead of assembling CSR buffers on the host.
// The diagonal is made strictly dominant using `diagonal_boost` to keep the matrix
// well-conditioned for iterative eigensolvers.
template <typename T>
Matrix<T, MatrixFormat::CSR> random_sparse_hermitian_csr(int n,
                                                         float density,
                                                         int batch_size = 1,
                                                         unsigned seed = 42,
                                                         typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                                         bool shared_pattern = true);

}  // namespace csr_generators
}  // namespace batchlas

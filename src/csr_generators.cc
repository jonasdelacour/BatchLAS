#include <blas/csr_generators.hh>

namespace batchlas {
namespace csr_generators {

template <typename T>
Matrix<T, MatrixFormat::CSR> random_sparse_hermitian_csr(int n,
                                                         float density,
                                                         int batch_size,
                                                         unsigned seed,
                                                         typename base_type<T>::type diagonal_boost,
                                                         bool shared_pattern) {
    return Matrix<T, MatrixFormat::CSR>::RandomSparseHermitian(
        n,
        density,
        batch_size,
        seed,
        diagonal_boost,
        shared_pattern);
}

template Matrix<float, MatrixFormat::CSR> random_sparse_hermitian_csr<float>(
    int, float, int, unsigned, base_type<float>::type, bool);
template Matrix<double, MatrixFormat::CSR> random_sparse_hermitian_csr<double>(
    int, float, int, unsigned, base_type<double>::type, bool);
template Matrix<std::complex<float>, MatrixFormat::CSR> random_sparse_hermitian_csr<std::complex<float>>(
    int, float, int, unsigned, base_type<std::complex<float>>::type, bool);
template Matrix<std::complex<double>, MatrixFormat::CSR> random_sparse_hermitian_csr<std::complex<double>>(
    int, float, int, unsigned, base_type<std::complex<double>>::type, bool);

}  // namespace csr_generators
}  // namespace batchlas

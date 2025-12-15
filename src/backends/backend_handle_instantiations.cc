#include "backend_handle_impl.hh"

namespace batchlas {

template struct BackendMatrixHandle<float, MatrixFormat::Dense>;
template struct BackendMatrixHandle<float, MatrixFormat::CSR>;
template struct BackendMatrixHandle<double, MatrixFormat::Dense>;
template struct BackendMatrixHandle<double, MatrixFormat::CSR>;
template struct BackendMatrixHandle<std::complex<float>, MatrixFormat::Dense>;
template struct BackendMatrixHandle<std::complex<float>, MatrixFormat::CSR>;
template struct BackendMatrixHandle<std::complex<double>, MatrixFormat::Dense>;
template struct BackendMatrixHandle<std::complex<double>, MatrixFormat::CSR>;

template std::shared_ptr<BackendMatrixHandle<float, MatrixFormat::Dense>> createBackendHandle<float, MatrixFormat::Dense>(const Matrix<float, MatrixFormat::Dense>& matrix);
template std::shared_ptr<BackendMatrixHandle<float, MatrixFormat::CSR>> createBackendHandle<float, MatrixFormat::CSR>(const Matrix<float, MatrixFormat::CSR>& matrix);
template std::shared_ptr<BackendMatrixHandle<double, MatrixFormat::Dense>> createBackendHandle<double, MatrixFormat::Dense>(const Matrix<double, MatrixFormat::Dense>& matrix);
template std::shared_ptr<BackendMatrixHandle<double, MatrixFormat::CSR>> createBackendHandle<double, MatrixFormat::CSR>(const Matrix<double, MatrixFormat::CSR>& matrix);
template std::shared_ptr<BackendMatrixHandle<std::complex<float>, MatrixFormat::Dense>> createBackendHandle<std::complex<float>, MatrixFormat::Dense>(const Matrix<std::complex<float>, MatrixFormat::Dense>& matrix);
template std::shared_ptr<BackendMatrixHandle<std::complex<float>, MatrixFormat::CSR>> createBackendHandle<std::complex<float>, MatrixFormat::CSR>(const Matrix<std::complex<float>, MatrixFormat::CSR>& matrix);
template std::shared_ptr<BackendMatrixHandle<std::complex<double>, MatrixFormat::Dense>> createBackendHandle<std::complex<double>, MatrixFormat::Dense>(const Matrix<std::complex<double>, MatrixFormat::Dense>& matrix);
template std::shared_ptr<BackendMatrixHandle<std::complex<double>, MatrixFormat::CSR>> createBackendHandle<std::complex<double>, MatrixFormat::CSR>(const Matrix<std::complex<double>, MatrixFormat::CSR>& matrix);

} // namespace batchlas

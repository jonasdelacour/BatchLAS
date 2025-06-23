#include <blas/matrix.hh>
#include "../linalg-impl.hh"
#ifdef BATCHLAS_HAS_CUDA_BACKEND
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    #include <cublas_v2.h>
    #include <cusparse.h>
    #include <cusolverDn.h>
#endif

namespace batchlas {

template <typename T, MatrixFormat MType>
struct BackendMatrixHandle {
    
    // Constructor
    BackendMatrixHandle() = default;
    
    // Destructor
    ~BackendMatrixHandle() = default;
    
    // Copy constructor
    BackendMatrixHandle(const BackendMatrixHandle&) = default;
    
    // Move constructor
    BackendMatrixHandle(BackendMatrixHandle&&) = default;
    
    // Assignment operator
    BackendMatrixHandle& operator=(const BackendMatrixHandle&) = default;
    
    // Move assignment operator
    BackendMatrixHandle& operator=(BackendMatrixHandle&&) = default;

    BackendMatrixHandle(const MatrixView<T, MType>& matrix) {
        #ifdef BATCHLAS_HAS_CUDA_BACKEND
            if constexpr (MType == MatrixFormat::Dense) {
                cusparseCreateDnMat(&cusparse_descr_, matrix.rows(), matrix.cols(), matrix.ld(), matrix.data_ptr(), BackendScalar<T, BackendLibrary::CUSPARSE>::type, CUSPARSE_ORDER_COL);
                if (matrix.batch_size() > 1) {
                    cusparseDnMatSetStridedBatch(cusparse_descr_, matrix.batch_size(), matrix.stride());
                }
            } else if constexpr (MType == MatrixFormat::CSR) {
                cusparseCreateCsr(&cusparse_descr_sp_, matrix.rows(), matrix.cols(), matrix.nnz(), matrix.row_offsets().data(), matrix.col_indices().data(), matrix.data_ptr(),
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, BackendScalar<T, BackendLibrary::CUSPARSE>::type);
                if (matrix.batch_size() > 1) {
                    cusparseCsrSetStridedBatch(cusparse_descr_sp_, matrix.batch_size(), matrix.offset_stride(), matrix.matrix_stride());
                }
            }
        #endif
        #ifdef BATCHLAS_HAS_ROCM_BACKEND
            if constexpr (MType == MatrixFormat::Dense) {
                rocsparse_create_dnmat_descr(&rocsparse_descr_, matrix.rows(), matrix.cols(), matrix.ld(), matrix.data_ptr(),
                                            std::is_same_v<T, float> ? rocsparse_datatype_f32_r :
                                            std::is_same_v<T, double> ? rocsparse_datatype_f64_r :
                                            std::is_same_v<T, std::complex<float>> ? rocsparse_datatype_f32_c :
                                            rocsparse_datatype_f64_c,
                                            rocsparse_order_column);
                if (matrix.batch_size() > 1) {
                    rocsparse_dnmat_set_strided_batch(rocsparse_descr_, matrix.batch_size(), matrix.stride());
                }
            } else if constexpr (MType == MatrixFormat::CSR) {
                rocsparse_create_csr_descr( &rocsparse_descr_sp_, matrix.rows(), matrix.cols(), matrix.nnz(),
                                            matrix.row_offsets().data(), matrix.col_indices().data(), matrix.data_ptr(),
                                            rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero,
                                            std::is_same_v<T, float> ? rocsparse_datatype_f32_r :
                                            std::is_same_v<T, double> ? rocsparse_datatype_f64_r :
                                            std::is_same_v<T, std::complex<float>> ? rocsparse_datatype_f32_c :
                                            rocsparse_datatype_f64_c);
                if (matrix.batch_size() > 1) {
                    rocsparse_csr_set_strided_batch(rocsparse_descr_sp_, matrix.batch_size(), matrix.offset_stride(), matrix.matrix_stride());
                }
            }
        #endif
    }

    BackendMatrixHandle(const Matrix<T, MType>& matrix) : BackendMatrixHandle(matrix.view()) {}

    #ifdef BATCHLAS_HAS_CUDA_BACKEND
        cusparseDnMatDescr_t cusparse_descr_ = nullptr;
        cusparseSpMatDescr_t cusparse_descr_sp_ = nullptr;

        constexpr inline operator cusparseDnMatDescr_t() const {
            return cusparse_descr_;
        }
        constexpr inline operator cusparseSpMatDescr_t() const {
            return cusparse_descr_sp_;
        }
    #endif
    #ifdef BATCHLAS_HAS_ROCM_BACKEND
        rocsparse_dnmat_descr rocsparse_descr_ = nullptr;
        rocsparse_spmat_descr rocsparse_descr_sp_ = nullptr;


        constexpr inline operator rocsparse_dnmat_descr() const {
            return rocsparse_descr_;
        }

        constexpr inline operator rocsparse_spmat_descr() const {
            return rocsparse_descr_sp_;
        }
    #endif
};

template <typename T, MatrixFormat MType>
std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const MatrixView<T, MType>& view) {
    // Create a new BackendMatrixHandle object
    auto handle = std::make_shared<BackendMatrixHandle<T, MType>>(view);
    return handle;
}

template <typename T, MatrixFormat MType>
std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const Matrix<T, MType>& matrix) {
    return createBackendHandle<T, MType>(matrix.view());
}




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

}

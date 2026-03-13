#pragma once

#include <blas/matrix.hh>
#include "../linalg-impl.hh"

#if BATCHLAS_HAS_CUDA_BACKEND
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    #include <cublas_v2.h>
    #include <cusparse.h>
    #include <cusolverDn.h>
#endif

namespace batchlas {

template <typename T, MatrixFormat MType>
struct BackendMatrixHandle {
    BackendMatrixHandle() = default;
    ~BackendMatrixHandle() = default;
    BackendMatrixHandle(const BackendMatrixHandle&) = default;
    BackendMatrixHandle(BackendMatrixHandle&&) = default;
    BackendMatrixHandle& operator=(const BackendMatrixHandle&) = default;
    BackendMatrixHandle& operator=(BackendMatrixHandle&&) = default;

    BackendMatrixHandle(const MatrixView<T, MType>& matrix) { initialize(matrix); }

    BackendMatrixHandle(const Matrix<T, MType>& matrix) : BackendMatrixHandle(matrix.view()) {}

    #if BATCHLAS_HAS_CUDA_BACKEND
        constexpr inline operator cusparseDnMatDescr_t() const { return cusparse_descr_; }
        constexpr inline operator cusparseSpMatDescr_t() const { return cusparse_descr_sp_; }
    #endif

    #if BATCHLAS_HAS_ROCM_BACKEND
        constexpr inline operator rocsparse_dnmat_descr() const { return rocsparse_descr_; }
        constexpr inline operator rocsparse_spmat_descr() const { return rocsparse_descr_sp_; }
    #endif

private:
    void initialize(const MatrixView<T, MType>& matrix) {
        #if BATCHLAS_HAS_CUDA_BACKEND
            initialize_cuda_descriptors(matrix);
        #endif
        #if BATCHLAS_HAS_ROCM_BACKEND
            initialize_rocm_descriptors(matrix);
        #endif
    }

    #if BATCHLAS_HAS_CUDA_BACKEND
        void initialize_cuda_descriptors(const MatrixView<T, MType>& matrix) {
            if constexpr (MType == MatrixFormat::Dense && is_complex_or_floating_point<T>::value) {
                cusparseCreateDnMat(&cusparse_descr_, matrix.rows(), matrix.cols(), matrix.ld(), matrix.data_ptr(), BackendScalar<T, BackendLibrary::CUSPARSE>::type, CUSPARSE_ORDER_COL);
                if (matrix.batch_size() > 1) {
                    cusparseDnMatSetStridedBatch(cusparse_descr_, matrix.batch_size(), matrix.stride());
                }
            } else if constexpr (MType == MatrixFormat::CSR && is_complex_or_floating_point<T>::value) {
                cusparseCreateCsr(&cusparse_descr_sp_, matrix.rows(), matrix.cols(), matrix.nnz(), matrix.row_offsets().data(), matrix.col_indices().data(), matrix.data_ptr(),
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, BackendScalar<T, BackendLibrary::CUSPARSE>::type);
                if (matrix.batch_size() > 1) {
                    cusparseCsrSetStridedBatch(cusparse_descr_sp_, matrix.batch_size(), matrix.offset_stride(), matrix.matrix_stride());
                }
            }
        }

        cusparseDnMatDescr_t cusparse_descr_ = nullptr;
        cusparseSpMatDescr_t cusparse_descr_sp_ = nullptr;
    #endif

    #if BATCHLAS_HAS_ROCM_BACKEND
        static constexpr rocsparse_datatype rocsparse_value_type() {
            if constexpr (std::is_same_v<T, float>) {
                return rocsparse_datatype_f32_r;
            } else if constexpr (std::is_same_v<T, double>) {
                return rocsparse_datatype_f64_r;
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return rocsparse_datatype_f32_c;
            } else {
                return rocsparse_datatype_f64_c;
            }
        }

        void initialize_rocm_descriptors(const MatrixView<T, MType>& matrix) {
            if constexpr (MType == MatrixFormat::Dense && is_complex_or_floating_point<T>::value) {
                rocsparse_create_dnmat_descr(&rocsparse_descr_, matrix.rows(), matrix.cols(), matrix.ld(), matrix.data_ptr(),
                                             rocsparse_value_type(),
                                             rocsparse_order_column);
                if (matrix.batch_size() > 1) {
                    rocsparse_dnmat_set_strided_batch(rocsparse_descr_, matrix.batch_size(), matrix.stride());
                }
            } else if constexpr (MType == MatrixFormat::CSR && is_complex_or_floating_point<T>::value) {
                rocsparse_create_csr_descr(&rocsparse_descr_sp_, matrix.rows(), matrix.cols(), matrix.nnz(),
                                           matrix.row_offsets().data(), matrix.col_indices().data(), matrix.data_ptr(),
                                           rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero,
                                           rocsparse_value_type());
                if (matrix.batch_size() > 1) {
                    rocsparse_csr_set_strided_batch(rocsparse_descr_sp_, matrix.batch_size(), matrix.offset_stride(), matrix.matrix_stride());
                }
            }
        }

        rocsparse_dnmat_descr rocsparse_descr_ = nullptr;
        rocsparse_spmat_descr rocsparse_descr_sp_ = nullptr;
    #endif
};

template <typename T, MatrixFormat MType>
std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const MatrixView<T, MType>& view) {
    return std::make_shared<BackendMatrixHandle<T, MType>>(view);
}

template <typename T, MatrixFormat MType>
std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const Matrix<T, MType>& matrix) {
    return createBackendHandle<T, MType>(matrix.view());
}

} // namespace batchlas

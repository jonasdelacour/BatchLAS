#include "../../include/blas/matrix_handle_new.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>

namespace batchlas {

// Base implementation for BackendMatrixHandle
template <typename T, MatrixFormat MType>
BackendMatrixHandle<T, MType>::~BackendMatrixHandle() = default;

// Implementation for dense matrix backend handle
#ifdef BATCHLAS_HAS_CUDA_BACKEND
// CUDA backend implementations
template <typename T>
class CudaDenseMatrixHandle : public BackendMatrixHandle<T, MatrixFormat::Dense> {
public:
    // Constructor for a dense matrix
    CudaDenseMatrixHandle(T* data, int rows, int cols, int ld)
        : rows_(rows), cols_(cols), ld_(ld) {
        // Initialize cuSPARSE dense matrix descriptor
        cusparseCreateDnMat(&dn_mat_descr_, rows, cols, ld, data, 
                          BackendScalar<T, Backend::CUDA>::type, CUSPARSE_ORDER_COL);
    }

    // Destructor
    ~CudaDenseMatrixHandle() override {
        cusparseDestroyDnMat(dn_mat_descr_);
    }

    // Get cuSPARSE dense matrix descriptor
    cusparseSpMatDescr_t& getDnMatDescriptor() {
        return dn_mat_descr_;
    }

    // Update data pointer (e.g., when using a different batch slice)
    void updateDataPointer(T* new_data) {
        // Update the descriptor with the new data pointer
        cusparseUpdateDnMatData(dn_mat_descr_, new_data);
    }

private:
    int rows_;
    int cols_;
    int ld_;
    cusparseSpMatDescr_t dn_mat_descr_;
};

// Implementation for CSR sparse matrix backend handle
template <typename T>
class CudaSparseMatrixHandle : public BackendMatrixHandle<T, MatrixFormat::CSR> {
public:
    // Constructor for CSR matrix
    CudaSparseMatrixHandle(T* values, int* row_offsets, int* col_indices, 
                           int rows, int cols, int nnz)
        : rows_(rows), cols_(cols), nnz_(nnz) {
        // Initialize CUDA sparse matrix descriptor
        cusparseCreateCsr(&sparse_descr_, 
                          rows, 
                          cols, 
                          nnz, 
                          row_offsets, 
                          col_indices, 
                          values, 
                          CUSPARSE_INDEX_32I, 
                          CUSPARSE_INDEX_32I, 
                          CUSPARSE_INDEX_BASE_ZERO, 
                          BackendScalar<T, Backend::CUDA>::type);
    }

    // Destructor
    ~CudaSparseMatrixHandle() override {
        // Destroy the sparse matrix descriptor
        cusparseDestroySpMat(sparse_descr_);
    }

    // Access to the cuSPARSE descriptor
    cusparseSpMatDescr_t& getSpMatDescriptor() {
        return sparse_descr_;
    }

    // For implicit conversion in cusparse calls
    operator cusparseSpMatDescr_t&() {
        return sparse_descr_;
    }
    
    // Update data pointers (e.g., when using a different batch slice)
    void updateDataPointers(T* values, int* row_offsets, int* col_indices) {
        // cuSPARSE doesn't have a direct update function like for dense matrices,
        // so we need to destroy the current descriptor and create a new one
        cusparseDestroySpMat(sparse_descr_);
        
        cusparseCreateCsr(&sparse_descr_, 
                          rows_, 
                          cols_, 
                          nnz_, 
                          row_offsets, 
                          col_indices, 
                          values, 
                          CUSPARSE_INDEX_32I, 
                          CUSPARSE_INDEX_32I, 
                          CUSPARSE_INDEX_BASE_ZERO, 
                          BackendScalar<T, Backend::CUDA>::type);
    }

private:
    int rows_;
    int cols_;
    int nnz_;
    
    // cuSPARSE descriptor
    cusparseSpMatDescr_t sparse_descr_;
};
#endif // BATCHLAS_HAS_CUDA_BACKEND

// Create backend handle factory functions
template <Backend B, typename T, MatrixFormat MType>
std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(T* data, int rows, int cols, int ld, Queue& ctx) {
    if constexpr (B == Backend::CUDA) {
        #ifdef BATCHLAS_HAS_CUDA_BACKEND
        if constexpr (MType == MatrixFormat::Dense) {
            return std::make_shared<CudaDenseMatrixHandle<T>>(data, rows, cols, ld);
        } 
        #else
        throw std::runtime_error("CUDA backend not available");
        #endif
    } else if constexpr (B == Backend::NETLIB) {
        // Implement for host backend
    } else {
        throw std::runtime_error("Unsupported backend for matrix handle");
    }
    return nullptr; // Should not reach here
}

template <Backend B, typename T, MatrixFormat MType>
std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(T* values, int* row_offsets, int* col_indices, 
                                                                  int rows, int cols, int nnz,
                                                                  Queue& ctx) {
    if constexpr (B == Backend::CUDA) {
        #ifdef BATCHLAS_HAS_CUDA_BACKEND
        if constexpr (MType == MatrixFormat::CSR) {
            return std::make_shared<CudaSparseMatrixHandle<T>>(values, row_offsets, col_indices, 
                                                              rows, cols, nnz);
        }
        #else
        throw std::runtime_error("CUDA backend not available");
        #endif
    } else if constexpr (B == Backend::NETLIB) {
        // Implement for host backend
    } else {
        throw std::runtime_error("Unsupported backend for matrix handle");
    }
    return nullptr; // Should not reach here
}

// Helper functions to create backend handles from Matrix and MatrixView objects
template <Backend B, typename T>
std::shared_ptr<BackendMatrixHandle<T, MatrixFormat::Dense>> createDenseBackendHandle(const Matrix<T, MatrixFormat::Dense>& matrix, Queue& ctx) {
    return createBackendHandle<B, T, MatrixFormat::Dense>(
        matrix.data().data(), matrix.rows_, matrix.cols_, 
        matrix.ld(), ctx);
}

template <Backend B, typename T>
std::shared_ptr<BackendMatrixHandle<T, MatrixFormat::Dense>> createDenseBackendHandle(const MatrixView<T, MatrixFormat::Dense>& view, Queue& ctx) {
    return createBackendHandle<B, T, MatrixFormat::Dense>(
        view.data().data(), view.rows_, view.cols_, 
        view.ld(), ctx);
}

template <Backend B, typename T>
std::shared_ptr<BackendMatrixHandle<T, MatrixFormat::CSR>> createCSRBackendHandle(const Matrix<T, MatrixFormat::CSR>& matrix, Queue& ctx) {
    return createBackendHandle<B, T, MatrixFormat::CSR>(
        matrix.data().data(), matrix.row_offsets().data(), matrix.col_indices().data(), 
        matrix.rows_, matrix.cols_, matrix.nnz(), ctx);
}

template <Backend B, typename T>
std::shared_ptr<BackendMatrixHandle<T, MatrixFormat::CSR>> createCSRBackendHandle(const MatrixView<T, MatrixFormat::CSR>& view, Queue& ctx) {
    return createBackendHandle<B, T, MatrixFormat::CSR>(
        view.data().data(), view.row_offsets().data(), view.col_indices().data(), 
        view.rows_, view.cols_, view.nnz(), ctx);
}

// Explicit instantiations for commonly used types
#define INSTANTIATE_BACKEND_HANDLE(T, MTYPE) \
    template class BackendMatrixHandle<T, MTYPE>; \
    template std::shared_ptr<BackendMatrixHandle<T, MTYPE>> createBackendHandle<Backend::CUDA>(const Matrix<T, MTYPE>&, Queue&); \
    template std::shared_ptr<BackendMatrixHandle<T, MTYPE>> createBackendHandle<Backend::CUDA>(const MatrixView<T, MTYPE>&, Queue&);

// Instantiate for float and double with Dense and CSR formats
INSTANTIATE_BACKEND_HANDLE(float, MatrixFormat::Dense)
INSTANTIATE_BACKEND_HANDLE(double, MatrixFormat::Dense)
INSTANTIATE_BACKEND_HANDLE(std::complex<float>, MatrixFormat::Dense)
INSTANTIATE_BACKEND_HANDLE(std::complex<double>, MatrixFormat::Dense)

INSTANTIATE_BACKEND_HANDLE(float, MatrixFormat::CSR)
INSTANTIATE_BACKEND_HANDLE(double, MatrixFormat::CSR)
INSTANTIATE_BACKEND_HANDLE(std::complex<float>, MatrixFormat::CSR)
INSTANTIATE_BACKEND_HANDLE(std::complex<double>, MatrixFormat::CSR)

#undef INSTANTIATE_BACKEND_HANDLE

} // namespace batchlas

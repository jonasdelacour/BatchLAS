#include "../include/blas/matrix_handle_new.hh"
#include "linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>

namespace batchlas {

//----------------------------------------------------------------------
// Matrix class implementation - Dense format
//----------------------------------------------------------------------

// Basic constructor for dense matrix (allocates uninitialized memory)
template <typename T>
Matrix<T, MatrixFormat::Dense>::Matrix(int rows, int cols, int batch_size)
    : rows_(rows), cols_(cols), batch_size_(batch_size),
      ld_(rows), stride_(rows * cols) {
    // Allocate memory for the matrix data
    data_.resize(static_cast<size_t>(rows) * cols * batch_size);
    
    // If batched, set up the data pointers
    if (batch_size > 1) {
        data_ptrs_.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            data_ptrs_[i] = data_.data() + i * stride_;
        }
    }
}

// Constructor from existing data (copies the data)
template <typename T>
Matrix<T, MatrixFormat::Dense>::Matrix(const T* data, int rows, int cols, int ld, 
                                      int stride, int batch_size)
    : rows_(rows), cols_(cols), ld_(ld), stride_(stride > 0 ? stride : ld * cols), 
      batch_size_(batch_size) {
    // Allocate memory and copy the data
    data_.resize(static_cast<size_t>(rows) * cols * batch_size);
    
    if (stride == ld * cols) {
        // Contiguous data, we can copy all at once
        std::copy(data, data + static_cast<size_t>(stride_) * batch_size, data_.data());
    } else {
        // Non-contiguous data, copy each matrix separately
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < cols; ++j) {
                for (int i = 0; i < rows; ++i) {
                    data_[b * stride_ + j * rows + i] = data[b * stride + j * ld + i];
                }
            }
        }
    }
    
    // If batched, set up the data pointers
    if (batch_size > 1) {
        data_ptrs_.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            data_ptrs_[i] = data_.data() + i * stride_;
        }
    }
}

// Destructor
template <typename T>
Matrix<T, MatrixFormat::Dense>::~Matrix() {
    // The UnifiedVector destructors will handle memory cleanup
}

// Move constructor
template <typename T>
Matrix<T, MatrixFormat::Dense>::Matrix(Matrix<T, MatrixFormat::Dense>&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), ld_(other.ld_), stride_(other.stride_),
      batch_size_(other.batch_size_), data_(std::move(other.data_)), 
      data_ptrs_(std::move(other.data_ptrs_)),
      backend_handle_(std::move(other.backend_handle_)) {
    // Reset other's state
    other.rows_ = 0;
    other.cols_ = 0;
    other.ld_ = 0;
    other.stride_ = 0;
    other.batch_size_ = 0;
}

// Move assignment operator
template <typename T>
Matrix<T, MatrixFormat::Dense>& Matrix<T, MatrixFormat::Dense>::operator=(Matrix<T, MatrixFormat::Dense>&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        ld_ = other.ld_;
        stride_ = other.stride_;
        batch_size_ = other.batch_size_;
        data_ = std::move(other.data_);
        data_ptrs_ = std::move(other.data_ptrs_);
        backend_handle_ = std::move(other.backend_handle_);
        
        // Reset other's state
        other.rows_ = 0;
        other.cols_ = 0;
        other.ld_ = 0;
        other.stride_ = 0;
        other.batch_size_ = 0;
    }
    return *this;
}

// Create a view of the entire matrix
template <typename T>
MatrixView<T, MatrixFormat::Dense> Matrix<T, MatrixFormat::Dense>::view() const {
    return MatrixView<T, MatrixFormat::Dense>(*this);
}

// Create a view of a subset of the matrix
template <typename T>
MatrixView<T, MatrixFormat::Dense> Matrix<T, MatrixFormat::Dense>::view(int rows, int cols, int ld, int stride) const {
    return MatrixView<T, MatrixFormat::Dense>(*this, 0, 0, rows, cols);
}

// Fill the matrix with a specific value
template <typename T>
void Matrix<T, MatrixFormat::Dense>::fill(T value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Copy from another matrix view
template <typename T>
void Matrix<T, MatrixFormat::Dense>::copy_from(const MatrixView<T, MatrixFormat::Dense>& src) {
    if (rows_ != src.rows_ || cols_ != src.cols_ || batch_size_ != src.batch_size_) {
        throw std::runtime_error("Matrix dimensions or batch size mismatch in copy_from");
    }
    
    for (int b = 0; b < batch_size_; ++b) {
        for (int j = 0; j < cols_; ++j) {
            for (int i = 0; i < rows_; ++i) {
                // Index calculation for both matrices
                data_[b * stride_ + j * ld_ + i] = src.data()[b * src.stride() + j * src.ld() + i];
            }
        }
    }
}

// Initialize backend
template <typename T>
void Matrix<T, MatrixFormat::Dense>::init(Queue& ctx) const {
    // No-op if we already have a backend handle
    if (backend_handle_) return;
    
    // Create a backend handle
    auto handle = createDenseBackendHandle<Backend::CUDA, T>(*this, ctx);
    backend_handle_ = handle;
}

// Initialize backend
template <typename T>
void Matrix<T, MatrixFormat::Dense>::init_backend() {
    // This is a legacy method - redirect to init with default queue
    Queue ctx;
    init(ctx);
}

// Operator-> to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::Dense>* Matrix<T, MatrixFormat::Dense>::operator->() {
    if (!backend_handle_) {
        init_backend();
    }
    return backend_handle_.get();
}

// Operator* to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::Dense>& Matrix<T, MatrixFormat::Dense>::operator*() {
    if (!backend_handle_) {
        init_backend();
    }
    return *backend_handle_;
}

// Factory method to create identity matrix
template <typename T>
Matrix<T, MatrixFormat::Dense> Matrix<T, MatrixFormat::Dense>::Identity(int n, int batch_size) {
    Matrix<T, MatrixFormat::Dense> result(n, n, batch_size);
    
    // Set diagonal elements to 1, others to 0
    result.fill(T(0));
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < n; ++i) {
            result.data_[b * result.stride_ + i * result.ld_ + i] = T(1);
        }
    }
    
    return result;
}

// Factory method to create random matrix
template <typename T>
Matrix<T, MatrixFormat::Dense> Matrix<T, MatrixFormat::Dense>::Random(int rows, int cols, int batch_size, unsigned int seed) {
    Matrix<T, MatrixFormat::Dense> result(rows, cols, batch_size);
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<typename base_type<T>::type> dist(-1.0, 1.0);
    
    for (size_t i = 0; i < result.data_.size(); ++i) {
        if constexpr (std::is_same_v<T, std::complex<float>> || 
                      std::is_same_v<T, std::complex<double>>) {
            // For complex numbers, generate random real and imaginary parts
            result.data_[i] = T(dist(gen), dist(gen));
        } else {
            // For real numbers, just generate a random value
            result.data_[i] = T(dist(gen));
        }
    }
    
    return result;
}

// Factory method to create zeros matrix
template <typename T>
Matrix<T, MatrixFormat::Dense> Matrix<T, MatrixFormat::Dense>::Zeros(int rows, int cols, int batch_size) {
    Matrix<T, MatrixFormat::Dense> result(rows, cols, batch_size);
    result.fill(T(0));
    return result;
}

// Factory method to create ones matrix
template <typename T>
Matrix<T, MatrixFormat::Dense> Matrix<T, MatrixFormat::Dense>::Ones(int rows, int cols, int batch_size) {
    Matrix<T, MatrixFormat::Dense> result(rows, cols, batch_size);
    result.fill(T(1));
    return result;
}

// Factory method to create diagonal matrix
template <typename T>
Matrix<T, MatrixFormat::Dense> Matrix<T, MatrixFormat::Dense>::Diagonal(const Span<T>& diag_values, int batch_size) {
    int n = diag_values.size();
    Matrix<T, MatrixFormat::Dense> result(n, n, batch_size);
    
    // Set all elements to zero
    result.fill(T(0));
    
    // Set diagonal elements
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < n; ++i) {
            result.data_[b * result.stride_ + i * result.ld_ + i] = diag_values[i];
        }
    }
    
    return result;
}

//----------------------------------------------------------------------
// Matrix class implementation - CSR format
//----------------------------------------------------------------------

// Basic constructor for CSR sparse matrix (allocates uninitialized memory)
template <typename T>
Matrix<T, MatrixFormat::CSR>::Matrix(int rows, int cols, int nnz, int batch_size)
    : rows_(rows), cols_(cols), nnz_(nnz), batch_size_(batch_size),
      matrix_stride_(nnz), offset_stride_(rows + 1) {
    // Allocate memory
    data_.resize(static_cast<size_t>(nnz) * batch_size);
    row_offsets_.resize(static_cast<size_t>(rows + 1) * batch_size);
    col_indices_.resize(static_cast<size_t>(nnz) * batch_size);
}

// Constructor from existing data (copies the data)
template <typename T>
Matrix<T, MatrixFormat::CSR>::Matrix(const T* values, const int* row_offsets, const int* col_indices,
                                    int nnz, int rows, int cols, int matrix_stride, 
                                    int offset_stride, int batch_size)
    : rows_(rows), cols_(cols), nnz_(nnz), batch_size_(batch_size),
      matrix_stride_(matrix_stride > 0 ? matrix_stride : nnz),
      offset_stride_(offset_stride > 0 ? offset_stride : rows + 1) {
    // Allocate and copy data
    data_.resize(static_cast<size_t>(matrix_stride_) * batch_size);
    row_offsets_.resize(static_cast<size_t>(offset_stride_) * batch_size);
    col_indices_.resize(static_cast<size_t>(matrix_stride_) * batch_size);
    
    std::copy(values, values + static_cast<size_t>(matrix_stride_) * batch_size, data_.data());
    std::copy(row_offsets, row_offsets + static_cast<size_t>(offset_stride_) * batch_size, row_offsets_.data());
    std::copy(col_indices, col_indices + static_cast<size_t>(matrix_stride_) * batch_size, col_indices_.data());
}

// Destructor
template <typename T>
Matrix<T, MatrixFormat::CSR>::~Matrix() {
    // The UnifiedVector destructors will handle memory cleanup
}

// Move constructor
template <typename T>
Matrix<T, MatrixFormat::CSR>::Matrix(Matrix<T, MatrixFormat::CSR>&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_),
      batch_size_(other.batch_size_), matrix_stride_(other.matrix_stride_),
      offset_stride_(other.offset_stride_), data_(std::move(other.data_)),
      row_offsets_(std::move(other.row_offsets_)), col_indices_(std::move(other.col_indices_)),
      backend_handle_(std::move(other.backend_handle_)) {
    // Reset other's state
    other.rows_ = 0;
    other.cols_ = 0;
    other.nnz_ = 0;
    other.batch_size_ = 0;
    other.matrix_stride_ = 0;
    other.offset_stride_ = 0;
}

// Move assignment operator
template <typename T>
Matrix<T, MatrixFormat::CSR>& Matrix<T, MatrixFormat::CSR>::operator=(Matrix<T, MatrixFormat::CSR>&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        nnz_ = other.nnz_;
        batch_size_ = other.batch_size_;
        matrix_stride_ = other.matrix_stride_;
        offset_stride_ = other.offset_stride_;
        data_ = std::move(other.data_);
        row_offsets_ = std::move(other.row_offsets_);
        col_indices_ = std::move(other.col_indices_);
        backend_handle_ = std::move(other.backend_handle_);
        
        // Reset other's state
        other.rows_ = 0;
        other.cols_ = 0;
        other.nnz_ = 0;
        other.batch_size_ = 0;
        other.matrix_stride_ = 0;
        other.offset_stride_ = 0;
    }
    return *this;
}

// Create a view of the entire matrix
template <typename T>
MatrixView<T, MatrixFormat::CSR> Matrix<T, MatrixFormat::CSR>::view() const {
    return MatrixView<T, MatrixFormat::CSR>(*this);
}

// Create a view of a subset of the matrix
template <typename T>
MatrixView<T, MatrixFormat::CSR> Matrix<T, MatrixFormat::CSR>::view(int rows, int cols, int ld, int stride) const {
    // This implementation is simplified - in practice, extracting a submatrix from CSR format
    // requires conversion to coordinate format, extracting the submatrix, and converting back
    throw std::runtime_error("Submatrix views of CSR matrices not yet implemented");
}

// Fill the values with a specific value (keeps structure intact)
template <typename T>
void Matrix<T, MatrixFormat::CSR>::fill(T value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Initialize backend
template <typename T>
void Matrix<T, MatrixFormat::CSR>::init(Queue& ctx) const {
    // No-op if we already have a backend handle
    if (backend_handle_) return;
    
    // Create a backend handle
    auto handle = createCSRBackendHandle<Backend::CUDA, T>(*this, ctx);
    backend_handle_ = handle;
}

// Initialize backend
template <typename T>
void Matrix<T, MatrixFormat::CSR>::init_backend() {
    // Legacy method - redirect to init with default queue
    Queue ctx;
    init(ctx);
}

// Operator-> to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::CSR>* Matrix<T, MatrixFormat::CSR>::operator->() {
    if (!backend_handle_) {
        init_backend();
    }
    return backend_handle_.get();
}

// Operator* to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::CSR>& Matrix<T, MatrixFormat::CSR>::operator*() {
    if (!backend_handle_) {
        init_backend();
    }
    return *backend_handle_;
}

//----------------------------------------------------------------------
// MatrixView class implementation - Dense format
//----------------------------------------------------------------------

// Constructor for dense matrix view
template <typename T>
MatrixView<T, MatrixFormat::Dense>::MatrixView(Span<T> data, int rows, int cols, int ld,
                                             int stride, int batch_size)
    : rows_(rows), cols_(cols), batch_size_(batch_size), data_(data), ld_(ld),
      stride_(stride > 0 ? stride : ld * cols) {
}

// Constructor for a submatrix view
template <typename T>
MatrixView<T, MatrixFormat::Dense>::MatrixView(Span<T> data, int row_offset, int col_offset,
                                             int rows, int cols, int ld, int stride, int batch_size)
    : rows_(rows), cols_(cols), batch_size_(batch_size), 
      ld_(ld), stride_(stride > 0 ? stride : ld * cols) {
    // Adjust the data span to point to the start of the submatrix
    size_t offset = row_offset + col_offset * ld;
    data_ = data.subspan(offset, static_cast<size_t>(rows) * cols * batch_size);
}

// Constructor from Matrix object
template <typename T>
MatrixView<T, MatrixFormat::Dense>::MatrixView(const Matrix<T, MatrixFormat::Dense>& matrix)
    : rows_(matrix.rows_), cols_(matrix.cols_), batch_size_(matrix.batch_size_),
      data_(matrix.data()), ld_(matrix.ld()), stride_(matrix.stride()) {
    // Try to reuse the backend handle
    backend_handle_ = matrix.backend_handle();
    
    // If batch size > 1, set up data_ptrs_
    if (batch_size_ > 1) {
        data_ptrs_ = Span<T*>(matrix.data_ptrs_.data(), matrix.data_ptrs_.size());
    }
}

// Constructor for a submatrix view from a Matrix object
template <typename T>
MatrixView<T, MatrixFormat::Dense>::MatrixView(
    const Matrix<T, MatrixFormat::Dense>& matrix, int row_offset, int col_offset, int rows, int cols)
    : rows_(rows), cols_(cols), batch_size_(matrix.batch_size_), ld_(matrix.ld()) {
    
    // Calculate the stride based on the source matrix
    stride_ = matrix.stride();
    
    // Adjust the data span to point to the submatrix start
    size_t offset = row_offset + col_offset * ld_;
    data_ = matrix.data().subspan(offset);
    
    // We can't reuse the backend handle directly since we're viewing a submatrix
    // The backend handle will be created on-demand when needed
}

// Create a view of a single batch item
template <typename T>
MatrixView<T, MatrixFormat::Dense> MatrixView<T, MatrixFormat::Dense>::batch_item(int batch_index) const {
    if (batch_index >= batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    // Calculate offset to the start of the batch
    size_t offset = batch_index * stride_;
    
    // Create a new view with batch_size = 1
    return MatrixView<T, MatrixFormat::Dense>(
        data_.subspan(offset, static_cast<size_t>(rows_) * cols_),
        rows_, cols_, ld_);
}

// Element access (for dense matrices)
template <typename T>
T& MatrixView<T, MatrixFormat::Dense>::at(int row, int col, int batch) {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || batch < 0 || batch >= batch_size_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[batch * stride_ + col * ld_ + row];
}

template <typename T>
const T& MatrixView<T, MatrixFormat::Dense>::at(int row, int col, int batch) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || batch < 0 || batch >= batch_size_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[batch * stride_ + col * ld_ + row];
}

// Element access operator - returns view of a single batch item
template <typename T>
MatrixView<T, MatrixFormat::Dense> MatrixView<T, MatrixFormat::Dense>::operator[](int i) const {
    return batch_item(i);
}

// Initialize backend
template <typename T>
void MatrixView<T, MatrixFormat::Dense>::init(Queue& ctx) const {
    // If we already have a backend handle from a strong reference, use it
    if (!backend_handle_.expired()) return;
    
    // Create a new backend handle
    auto handle = createDenseBackendHandle<Backend::CUDA, T>(*this, ctx);
    backend_handle_ = handle;
}

// Initialize backend
template <typename T>
void MatrixView<T, MatrixFormat::Dense>::init_backend() {
    // Legacy method - redirect to init with default queue
    Queue ctx;
    init(ctx);
}

// Operator-> to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::Dense>* MatrixView<T, MatrixFormat::Dense>::operator->() {
    if (auto handle = backend_handle_.lock()) {
        return handle.get();
    } else {
        init_backend();
        return backend_handle_.lock().get();
    }
}

// Operator* to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::Dense>& MatrixView<T, MatrixFormat::Dense>::operator*() {
    if (auto handle = backend_handle_.lock()) {
        return *handle;
    } else {
        init_backend();
        return *backend_handle_.lock();
    }
}

//----------------------------------------------------------------------
// MatrixView class implementation - CSR format
//----------------------------------------------------------------------

// Constructor for CSR sparse matrix view
template <typename T>
MatrixView<T, MatrixFormat::CSR>::MatrixView(Span<T> data, Span<int> row_offsets, Span<int> col_indices,
                                           int nnz, int rows, int cols, int matrix_stride,
                                           int offset_stride, int batch_size)
    : rows_(rows), cols_(cols), batch_size_(batch_size), data_(data),
      row_offsets_(row_offsets), col_indices_(col_indices), nnz_(nnz),
      matrix_stride_(matrix_stride > 0 ? matrix_stride : nnz),
      offset_stride_(offset_stride > 0 ? offset_stride : rows + 1) {
}

// Constructor from Matrix object
template <typename T>
MatrixView<T, MatrixFormat::CSR>::MatrixView(const Matrix<T, MatrixFormat::CSR>& matrix)
    : rows_(matrix.rows_), cols_(matrix.cols_), batch_size_(matrix.batch_size_),
      data_(matrix.data()), row_offsets_(matrix.row_offsets()), 
      col_indices_(matrix.col_indices()), nnz_(matrix.nnz()),
      matrix_stride_(matrix.matrix_stride()), offset_stride_(matrix.offset_stride()) {
    // Try to reuse the backend handle
    backend_handle_ = matrix.backend_handle();
}

// Element access operator - returns view of a single batch item
template <typename T>
MatrixView<T, MatrixFormat::CSR> MatrixView<T, MatrixFormat::CSR>::operator[](int i) const {
    if (i >= batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    // Calculate offsets to the start of the batch
    size_t val_offset = i * matrix_stride_;
    size_t row_offset = i * offset_stride_;
    
    // Create a new view with batch_size = 1
    return MatrixView<T, MatrixFormat::CSR>(
        data_.subspan(val_offset, nnz_),
        row_offsets_.subspan(row_offset, rows_ + 1),
        col_indices_.subspan(val_offset, nnz_),
        nnz_, rows_, cols_);
}

// Initialize backend
template <typename T>
void MatrixView<T, MatrixFormat::CSR>::init(Queue& ctx) const {
    // If we already have a backend handle from a strong reference, use it
    if (!backend_handle_.expired()) return;
    
    // Create a new backend handle
    auto handle = createCSRBackendHandle<Backend::CUDA, T>(*this, ctx);
    backend_handle_ = handle;
}

// Initialize backend
template <typename T>
void MatrixView<T, MatrixFormat::CSR>::init_backend() {
    // Legacy method - redirect to init with default queue
    Queue ctx;
    init(ctx);
}

// Operator-> to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::CSR>* MatrixView<T, MatrixFormat::CSR>::operator->() {
    if (auto handle = backend_handle_.lock()) {
        return handle.get();
    } else {
        init_backend();
        return backend_handle_.lock().get();
    }
}

// Operator* to access backend handle
template <typename T>
BackendMatrixHandle<T, MatrixFormat::CSR>& MatrixView<T, MatrixFormat::CSR>::operator*() {
    if (auto handle = backend_handle_.lock()) {
        return *handle;
    } else {
        init_backend();
        return *backend_handle_.lock();
    }
}

//----------------------------------------------------------------------
// Explicit template instantiations
//----------------------------------------------------------------------

// Dense Matrix instantiations
template class Matrix<float, MatrixFormat::Dense>;
template class Matrix<double, MatrixFormat::Dense>;
template class Matrix<std::complex<float>, MatrixFormat::Dense>;
template class Matrix<std::complex<double>, MatrixFormat::Dense>;

// CSR Matrix instantiations
template class Matrix<float, MatrixFormat::CSR>;
template class Matrix<double, MatrixFormat::CSR>;
template class Matrix<std::complex<float>, MatrixFormat::CSR>;
template class Matrix<std::complex<double>, MatrixFormat::CSR>;

// Dense MatrixView instantiations
template class MatrixView<float, MatrixFormat::Dense>;
template class MatrixView<double, MatrixFormat::Dense>;
template class MatrixView<std::complex<float>, MatrixFormat::Dense>;
template class MatrixView<std::complex<double>, MatrixFormat::Dense>;

// CSR MatrixView instantiations
template class MatrixView<float, MatrixFormat::CSR>;
template class MatrixView<double, MatrixFormat::CSR>;
template class MatrixView<std::complex<float>, MatrixFormat::CSR>;
template class MatrixView<std::complex<double>, MatrixFormat::CSR>;

// Static factory method instantiations
template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Identity(int, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Identity(int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Identity(int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Identity(int, int);

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Random(int, int, int, unsigned int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Random(int, int, int, unsigned int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Random(int, int, int, unsigned int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Random(int, int, int, unsigned int);

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Zeros(int, int, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Zeros(int, int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Zeros(int, int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Zeros(int, int, int);

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Ones(int, int, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Ones(int, int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Ones(int, int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Ones(int, int, int);

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Diagonal(const Span<float>&, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Diagonal(const Span<double>&, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Diagonal(const Span<std::complex<float>>&, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Diagonal(const Span<std::complex<double>>&, int);

} // namespace batchlas

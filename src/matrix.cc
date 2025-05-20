#include "../include/blas/matrix_handle_new.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept> // Include for std::runtime_error
#include <vector>    // Include for std::vector used in scan
#include "backends/matrix_handle_impl.cc"

namespace batchlas {

//----------------------------------------------------------------------
// Matrix class implementation - Dense format
//----------------------------------------------------------------------

// Basic constructor for dense matrix (allocates uninitialized memory)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType>::Matrix(int rows, int cols, int batch_size)
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
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType>::Matrix(const T* data, int rows, int cols, int ld, 
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

//----------------------------------------------------------------------
// Matrix class implementation - CSR format
//----------------------------------------------------------------------

// Basic constructor for CSR sparse matrix (allocates uninitialized memory)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::CSR, int>::type>
Matrix<T, MType>::Matrix(int rows, int cols, int nnz, int batch_size)
    : rows_(rows), cols_(cols), nnz_(nnz), batch_size_(batch_size),
      matrix_stride_(nnz), offset_stride_(rows + 1) {
    // Allocate memory
    data_.resize(static_cast<size_t>(nnz) * batch_size);
    row_offsets_.resize(static_cast<size_t>(rows + 1) * batch_size);
    col_indices_.resize(static_cast<size_t>(nnz) * batch_size);
}

// Constructor from existing data (copies the data)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::CSR, int>::type>
Matrix<T, MType>::Matrix(const T* values, const int* row_offsets, const int* col_indices,
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

//----------------------------------------------------------------------
// Common Matrix class implementations (for all formats)
//----------------------------------------------------------------------

// Convert to a different matrix format
template <typename T, MatrixFormat MType>
template <MatrixFormat NewMType>
Matrix<T, NewMType> Matrix<T, MType>::convert_to(const T& zero_threshold) const {
    Queue q; // Use default queue

    // Handle identity conversion (same format)
    if constexpr (MType == NewMType) {
        return static_cast<const Matrix<T, NewMType>&>(*this);
    }
    // Handle Dense to CSR conversion
    else if constexpr (MType == MatrixFormat::Dense && NewMType == MatrixFormat::CSR) {
        // --- Dense to CSR using work-group primitives ---
        int rows = this->rows_;
        int cols = this->cols_;
        int batch_size = this->batch_size_;
        int dense_ld = this->ld();
        int dense_stride = this->stride();
        const T* dense_data_ptr = this->data().data();

        // 1. Count NNZ per row and total NNZ using work-group reductions
        UnifiedVector<int> row_nnz_counts_mem(batch_size * rows, 0);
        UnifiedVector<int> batch_nnz_counts_mem(batch_size, 0);

        auto row_nnz_counts = row_nnz_counts_mem.to_span();
        auto batch_nnz_counts = batch_nnz_counts_mem.to_span();
        
        // Calculate best work-group size (can be tuned based on hardware)
        const int wg_size = std::min(32, rows); 
        
        // First kernel: Count non-zeros per row
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<int, 1> local_counts(sycl::range<1>(wg_size), cgh);
                
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(batch_size, cols, wg_size),
                    sycl::range<3>(1, 1, wg_size)
                ),
                [=](sycl::nd_item<3> item) {
                    const size_t b = item.get_group(0);
                    const size_t c = item.get_group(1);  // Now working on columns
                    const size_t local_id = item.get_local_id(2);
                    const size_t group_size = item.get_local_range(2);
                    
                    // Initialize local count for this thread
                    int local_count = 0;
                    
                    // Process consecutive rows within the same column for better coalescing
                    for (size_t r = local_id; r < rows; r += group_size) {
                        // In column-major format, row elements in the same column are contiguous
                        size_t dense_idx = b * dense_stride + c * dense_ld + r;
                        
                        // Use zero_threshold instead of comparing with exact zero
                        if constexpr (std::is_same_v<T, std::complex<float>> || 
                                      std::is_same_v<T, std::complex<double>>) {
                            // For complex numbers, check both real and imaginary parts
                            auto val = dense_data_ptr[dense_idx];
                            if (std::abs(val.real()) > zero_threshold || std::abs(val.imag()) > zero_threshold) {
                                local_count++;
                            }
                        } else {
                            // For real numbers
                            if (std::abs(dense_data_ptr[dense_idx]) > zero_threshold) {
                                local_count++;
                            }
                        }
                    }
                    
                    // First thread in group distributes the counts to each row
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                row_counter(row_nnz_counts[b * rows + local_id]);
                            row_counter.fetch_add(local_count);
                }
            );
        }).wait();
        
        // Compute per-matrix (per-batch) number of non-zeros using joint_reduce
        q->submit([&](sycl::handler& cgh) {
            auto row_nnz_acc = row_nnz_counts.data();
            auto batch_nnz_acc = batch_nnz_counts.data();
            
            cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(batch_size * wg_size),
                sycl::range<1>(wg_size)
            ),
            [=](sycl::nd_item<1> item) {
                const size_t b = item.get_group(0); // batch index
                const size_t local_id = item.get_local_id(0);
                
                // Calculate the begin and end iterators for this batch's row counts
                auto begin_iter = row_nnz_acc + (b * rows);
                auto end_iter = begin_iter + rows;
                
                // Use joint_reduce directly on the range of row counts for this batch
                int total_sum = sycl::joint_reduce(
                    item.get_group(),
                    begin_iter,
                    end_iter,
                    sycl::plus<int>()
                );
                
                // First thread writes the result
                if (local_id == 0) {
                    batch_nnz_acc[b] = total_sum;
                }
            }
            );
        }).wait();

        // Calculate total NNZ across all matrices
        int total_nnz = std::reduce(batch_nnz_counts.begin(), batch_nnz_counts.end(), 0);
        
        // 2. Create row offsets using exclusive scan within each batch
        Matrix<T, MatrixFormat::CSR> result(rows, cols, total_nnz, batch_size);
        
        // Initialize batch offset counters
        UnifiedVector<int> batch_offsets(batch_size + 1, 0);
        for (int b = 1; b <= batch_size; b++) {
            batch_offsets[b] = batch_offsets[b-1] + batch_nnz_counts[b-1];
        }
        
        // Second kernel: Compute row offsets using work-group scan
        q->submit([&](sycl::handler& cgh) {
            // Create accessor for row offsets
            auto row_offsets_acc = sycl::accessor(result.row_offsets_.data(), 
                                                  result.row_offsets_.size(),
                                                  cgh,
                                                  sycl::write_only);
            
            // Local memory for scans
            sycl::local_accessor<int, 1> local_scan(rows + 1, cgh);
                
            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(batch_size * wg_size), 
                    sycl::range<1>(wg_size)
                ),
                [=](sycl::nd_item<1> item) {
                    const size_t b = item.get_group(0);
                    const size_t local_id = item.get_local_id(0);
                    const size_t group_size = item.get_local_range(0);
                    
                    // Load row counts into local memory
                    if (local_id < rows) {
                        local_scan[local_id + 1] = row_nnz_counts[b * rows + local_id];
                    }
                    if (local_id == 0) {
                        local_scan[0] = 0; // Start with 0 for exclusive scan
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    // Perform work-group exclusive scan
                    // Using a simplified inclusive scan followed by shifting
                    for (int offset = 1; offset < rows + 1; offset *= 2) {
                        if (local_id < rows + 1 && local_id >= offset) {
                            int temp = local_scan[local_id - offset];
                            item.barrier(sycl::access::fence_space::local_space);
                            local_scan[local_id] += temp;
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    
                    // Write results to global memory, adding batch offset
                    if (local_id < rows + 1) {
                        row_offsets_acc[b * (rows + 1) + local_id] = local_scan[local_id] + batch_offsets[b];
                    }
                }
            );
        }).wait();
        
        // 3. Populate CSR values and column indices
        q->submit([&](sycl::handler& cgh) {
            // Create accessors
            auto row_offsets_acc = result.row_offsets_.to_span();
            auto col_indices_acc = result.col_indices_.to_span();
            auto values_acc = result.data_.to_span();
            
            // Use atomic to track position within each row
            UnifiedVector<int> row_counters(batch_size * rows, 0);
            auto row_counters_acc = row_counters.to_span();
            
            cgh.parallel_for(
                sycl::range<3>(batch_size, rows, cols),
                [=](sycl::id<3> idx) {
                    size_t b = idx[0];
                    size_t r = idx[1];
                    size_t c = idx[2];
                    
                    size_t dense_idx = b * dense_stride + c * dense_ld + r;
                    T val = dense_data_ptr[dense_idx];
                    
                    // Use zero_threshold instead of comparing with exact zero
                    bool is_nonzero = false;
                    if constexpr (std::is_same_v<T, std::complex<float>> || 
                                  std::is_same_v<T, std::complex<double>>) {
                        // For complex numbers, check both real and imaginary parts
                        is_nonzero = std::abs(val.real()) > zero_threshold || std::abs(val.imag()) > zero_threshold;
                    } else {
                        // For real numbers
                        is_nonzero = std::abs(val) > zero_threshold;
                    }
                    
                    if (is_nonzero) {
                        // Use atomic to get position in CSR arrays
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            counter(row_counters_acc[b * rows + r]);
                        int pos = counter.fetch_add(1);
                        
                        // Get base offset for this row
                        int csr_idx = row_offsets_acc[b * (rows + 1) + r] + pos;
                        
                        // Write value and column index
                        col_indices_acc[csr_idx] = c;
                        values_acc[csr_idx] = val;
                    }
                }
            );
        }).wait();

        return result;
    }
    // Handle CSR to Dense conversion
    else if constexpr (MType == MatrixFormat::CSR && NewMType == MatrixFormat::Dense) {
        // --- CSR to Dense ---
        int rows = this->rows_;
        int cols = this->cols_;
        int batch_size = this->batch_size_;
        int nnz = this->nnz(); // Assuming nnz per batch item if matrix_stride == nnz
        int csr_mat_stride = this->matrix_stride();
        int csr_off_stride = this->offset_stride();

        const T* csr_data_ptr = this->data().data();
        const int* csr_row_offsets_ptr = this->row_offsets().data();
        const int* csr_col_indices_ptr = this->col_indices().data();

        // 1. Allocate and zero Dense matrix
        Matrix<T, MatrixFormat::Dense> result(rows, cols, batch_size);
        result.fill(T(0)); // Important: Initialize dense matrix to zero
        
        T* result_data_ptr = result.data().data();
        int dense_ld = result.ld();
        int dense_stride = result.stride();

        // 2. Scatter CSR data into Dense matrix
        q->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<2>(batch_size, rows), [=](sycl::id<2> idx) {
                size_t b = idx[0];
                size_t r = idx[1];

                int row_start_offset_idx = b * csr_off_stride + r;
                int row_end_offset_idx = b * csr_off_stride + r + 1;

                int start_nnz = csr_row_offsets_ptr[row_start_offset_idx];
                int end_nnz = csr_row_offsets_ptr[row_end_offset_idx];

                // Adjust start/end nnz based on potential batch stride in values/indices
                // Assuming matrix_stride applies to values and col_indices
                int batch_val_col_base = b * csr_mat_stride; 

                for (int csr_idx = start_nnz; csr_idx < end_nnz; ++csr_idx) {
                    int c = csr_col_indices_ptr[csr_idx];
                    T val = csr_data_ptr[csr_idx];

                    // Calculate dense index (Column-Major)
                    size_t dense_idx = b * dense_stride + c * dense_ld + r;
                    result_data_ptr[dense_idx] = val;
                }
            });
        }).wait();

        return result;
    }
    // Handle other conversions (e.g., Dense to COO, CSR to CSC, etc.)
    else {
        // Throw error for unsupported or unimplemented conversions
        throw std::runtime_error("Conversion between specified matrix formats not supported or implemented.");
    }
}


// Destructor
template <typename T, MatrixFormat MType>
Matrix<T, MType>::~Matrix() {
    // The UnifiedVector destructors will handle memory cleanup
}

// Move constructor
template <typename T, MatrixFormat MType>
Matrix<T, MType>::Matrix(Matrix<T, MType>&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), batch_size_(other.batch_size_),
      data_(std::move(other.data_)), backend_handle_(std::move(other.backend_handle_)) {
    
    // Move format-specific members
    if constexpr (MType == MatrixFormat::Dense) {
        ld_ = other.ld_;
        stride_ = other.stride_;
        data_ptrs_ = std::move(other.data_ptrs_);
        
        // Reset other's state
        other.ld_ = 0;
        other.stride_ = 0;
    } else if constexpr (MType == MatrixFormat::CSR) {
        nnz_ = other.nnz_;
        matrix_stride_ = other.matrix_stride_;
        offset_stride_ = other.offset_stride_;
        row_offsets_ = std::move(other.row_offsets_);
        col_indices_ = std::move(other.col_indices_);
        
        // Reset other's state
        other.nnz_ = 0;
        other.matrix_stride_ = 0;
        other.offset_stride_ = 0;
    }
    
    // Reset common fields
    other.rows_ = 0;
    other.cols_ = 0;
    other.batch_size_ = 0;
}

// Move assignment operator
template <typename T, MatrixFormat MType>
Matrix<T, MType>& Matrix<T, MType>::operator=(Matrix<T, MType>&& other) noexcept {
    if (this != &other) {
        // Copy common fields
        rows_ = other.rows_;
        cols_ = other.cols_;
        batch_size_ = other.batch_size_;
        data_ = std::move(other.data_);
        backend_handle_ = std::move(other.backend_handle_);
        
        // Move format-specific members
        if constexpr (MType == MatrixFormat::Dense) {
            ld_ = other.ld_;
            stride_ = other.stride_;
            data_ptrs_ = std::move(other.data_ptrs_);
            
            // Reset other's state
            other.ld_ = 0;
            other.stride_ = 0;
        } else if constexpr (MType == MatrixFormat::CSR) {
            nnz_ = other.nnz_;
            matrix_stride_ = other.matrix_stride_;
            offset_stride_ = other.offset_stride_;
            row_offsets_ = std::move(other.row_offsets_);
            col_indices_ = std::move(other.col_indices_);
            
            // Reset other's state
            other.nnz_ = 0;
            other.matrix_stride_ = 0;
            other.offset_stride_ = 0;
        }
        
        // Reset common fields
        other.rows_ = 0;
        other.cols_ = 0;
        other.batch_size_ = 0;
    }
    return *this;
}

// Create a view of the entire matrix
template <typename T, MatrixFormat MType>
MatrixView<T, MType> Matrix<T, MType>::view() const {
    return MatrixView<T, MType>(*this);
}

// Create a view of a subset of the matrix
template <typename T, MatrixFormat MType>
MatrixView<T, MType> Matrix<T, MType>::view(int rows, int cols, int ld, int stride) const {
    if constexpr (MType == MatrixFormat::Dense) {
        return MatrixView<T, MType>(*this, 0, 0, rows, cols);
    } else {
        // This implementation is simplified - in practice, extracting a submatrix from CSR format
        // requires conversion to coordinate format, extracting the submatrix, and converting back
        throw std::runtime_error("Submatrix views of CSR matrices not yet implemented");
    }
}

// Fill the matrix with a specific value
template <typename T, MatrixFormat MType>
void Matrix<T, MType>::fill(T value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Copy from another matrix view
template <typename T, MatrixFormat MType>
void Matrix<T, MType>::copy_from(const MatrixView<T, MType>& src) {
    if (rows_ != src.rows_ || cols_ != src.cols_ || batch_size_ != src.batch_size_) {
        throw std::runtime_error("Matrix dimensions or batch size mismatch in copy_from");
    }
    
    if constexpr (MType == MatrixFormat::Dense) {
        for (int b = 0; b < batch_size_; ++b) {
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) {
                    // Index calculation for both matrices
                    data_[b * stride_ + j * ld_ + i] = src.data()[b * src.stride() + j * src.ld() + i];
                }
            }
        }
    } else if constexpr (MType == MatrixFormat::CSR) {
        // For CSR format, just copy all data directly if structure matches
        std::copy(src.data().begin(), src.data().end(), data_.begin());
        std::copy(src.row_offsets().begin(), src.row_offsets().end(), row_offsets_.begin());
        std::copy(src.col_indices().begin(), src.col_indices().end(), col_indices_.begin());
    }
}

// Initialize backend - make backend_handle_ mutable so it can be modified in const methods
template <typename T, MatrixFormat MType>
void Matrix<T, MType>::init(Queue& ctx) const {
    // No-op if we already have a backend handle
    if (backend_handle_) return;
    backend_handle_ = createBackendHandle<Backend::CUDA, T>(*this, ctx);
    // Create a backend handle based on matrix format
    /* if constexpr (MType == MatrixFormat::Dense) {
        backend_handle_ = createDenseBackendHandle<Backend::CUDA, T>(*this, ctx);
    } else if constexpr (MType == MatrixFormat::CSR) {
        backend_handle_ = createCSRBackendHandle<Backend::CUDA, T>(*this, ctx);
    } */
}

// Initialize backend
template <typename T, MatrixFormat MType>
void Matrix<T, MType>::init_backend() {
    // This is a legacy method - redirect to init with default queue
    Queue ctx;
    init(ctx);
}

// Operator-> to access backend handle
template <typename T, MatrixFormat MType>
BackendMatrixHandle<T, MType>* Matrix<T, MType>::operator->() {
    if (!backend_handle_) {
        init_backend();
    }
    return backend_handle_.get();
}

// Operator* to access backend handle
template <typename T, MatrixFormat MType>
BackendMatrixHandle<T, MType>& Matrix<T, MType>::operator*() {
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

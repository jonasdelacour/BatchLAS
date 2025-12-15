#include <blas/matrix.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <util/mempool.hh>
#include <util/kernel-heuristics.hh>
#include <sycl/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/random>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept> // Include for std::runtime_error
#include <vector>    // Include for std::vector used in scan
#include "backends/backend_handle_impl.hh"

namespace batchlas {

//----------------------------------------------------------------------
// Matrix class implementation - Dense format
//----------------------------------------------------------------------

// Basic constructor for dense matrix (allocates uninitialized memory)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType>::Matrix(int rows, int cols, int batch_size, int ld, int stride)
    : rows_(rows), cols_(cols), batch_size_(batch_size),
      ld_(ld > 0 ? ld : rows), stride_(stride > 0 ? stride : ld > 0 ? ld * cols : rows * cols) {
    // Allocate memory for the matrix data
    data_ = UnifiedVector<T>(static_cast<size_t>(stride_) * batch_size);
    
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
Matrix<T, NewMType> Matrix<T, MType>::convert_to(const float_t<T>& zero_threshold) const {
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
        
        // Calculate optimal work-group size for reduction operations
        const int wg_size = compute_optimal_wg_size(q.device(), KernelType::REDUCTION, rows, batch_size); 
        
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
                    
                    
                    // Process consecutive rows within the same column for better coalescing
                    for (size_t r = local_id; r < rows; r += group_size) {
                        // In column-major format, row elements in the same column are contiguous
                        size_t dense_idx = b * dense_stride + c * dense_ld + r;
                        bool is_nonzero = false;
                        // Use zero_threshold instead of comparing with exact zero
                        if constexpr (std::is_same_v<T, std::complex<float>> || 
                                      std::is_same_v<T, std::complex<double>>) {
                            // For complex numbers, check both real and imaginary parts
                            auto val = dense_data_ptr[dense_idx];
                            if (std::abs(val.real()) > zero_threshold || std::abs(val.imag()) > zero_threshold) {
                                is_nonzero = true;
                            }
                        } else {
                            // For real numbers
                            if (std::abs(dense_data_ptr[dense_idx]) > zero_threshold) {
                                is_nonzero = true;
                            }
                        }

                        if (is_nonzero) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                row_counter(row_nnz_counts[b * rows + r]);
                            row_counter.fetch_add(1);
                        }
                    }
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
        int max_nnz = oneapi::dpl::reduce(oneapi::dpl::execution::par_unseq, batch_nnz_counts.begin(), batch_nnz_counts.end(), 0, sycl::maximum<int>());
        // 2. Create row offsets using exclusive scan within each batch
        Matrix<T, MatrixFormat::CSR> result(rows, cols, max_nnz, batch_size);

        // Initialize batch offset counters
        UnifiedVector<int> batch_offsets(batch_size + 1, 0);

        oneapi::dpl::exclusive_scan(oneapi::dpl::execution::par_unseq, batch_offsets.begin(), batch_offsets.end(), batch_offsets.begin(), 0);
        
        // Second kernel: Compute row offsets using work-group scan
        q -> submit ([&](sycl::handler& cgh) {
            auto row_offsets_acc = result.row_offsets().data();
            auto row_nnz_acc = row_nnz_counts.data();
            auto batch_offsets_acc = batch_offsets.data();
            auto wgs = compute_optimal_wg_size(q.device(), KernelType::SCAN, rows + 1, batch_size);
            cgh.parallel_for(
                sycl::nd_range<1>(batch_size * wgs, 
                                 wgs),
                [=](sycl::nd_item<1> item) {
                    size_t idx_val = item.get_global_id(0);
                    size_t bix = item.get_group(0);
                    sycl::joint_inclusive_scan(
                        item.get_group(),
                        row_nnz_acc + bix * rows,
                        row_nnz_acc + bix * rows + rows,
                        row_offsets_acc + bix * (rows + 1) + 1,
                        sycl::plus<int>(),
                        0
                    );
                }
            );
        }).wait();
        
        // 3. Populate CSR values and column indices
        UnifiedVector<int> row_counters(batch_size * rows, 0);
        q->submit([&](sycl::handler& cgh) {
            // Create accessors
            auto row_offsets_acc = result.row_offsets();
            auto col_indices_acc = result.col_indices();
            auto values_acc = result.data();
            auto dense_acc = this->data();

            // Use atomic to track position within each row
            auto row_counters_acc = row_counters.to_span();
            auto wgs = compute_optimal_wg_size(q.device(), KernelType::SPARSE, rows, batch_size);
            cgh.parallel_for(
                sycl::nd_range<1>(batch_size * wgs, wgs),
                [=](sycl::nd_item<1> item) {
                    size_t local_id = item.get_local_linear_id();
                    size_t b = item.get_group(0); // batch index
                    
                    for (size_t id = local_id; id < rows*cols; id += item.get_local_range(0)) {
                        size_t r = id / rows; // Row index
                        size_t c = id % rows; // Column index
                        
                        // Calculate dense index (Column-Major)
                        size_t dense_idx = b * dense_stride + c * dense_ld + r;
                        T val = dense_acc[dense_idx];

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
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group>
                                counter(row_counters_acc[b * rows + r]);
                            int pos = counter.fetch_add(1);
                            // Get base offset for this row
                            int csr_idx = row_offsets_acc[b * (rows + 1) + r] + pos;
                      
                            // Write value and column index
                            col_indices_acc[b*max_nnz + csr_idx] = c;
                            values_acc[b*max_nnz + csr_idx] = val;
                        }
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

// Create a view of the entire matrix
template <typename T, MatrixFormat MType>
MatrixView<T, MType> Matrix<T, MType>::view() const {
    return MatrixView<T, MType>(*this);
}

// Create a view of a subset of the matrix
template <typename T, MatrixFormat MType>
MatrixView<T, MType> Matrix<T, MType>::view(int rows, int cols, int ld, int stride) const {
    if constexpr (MType == MatrixFormat::Dense) {
        return MatrixView<T, MType>(*this, rows, cols, 
                                    ld > 0 ? ld : this->ld(), 
                                    stride > 0 ? stride : this->stride(), 
                                    batch_size_);
    } else {
        // This implementation is simplified - in practice, extracting a submatrix from CSR format
        // requires conversion to coordinate format, extracting the submatrix, and converting back
        throw std::runtime_error("Submatrix views of CSR matrices not yet implemented");
    }
}


template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
MatrixView<T, MType>::MatrixView(const VectorView<T>& vector_view, VectorOrientation orientation)
    : rows_(orientation == VectorOrientation::Column ? vector_view.size() : 1),
      cols_(orientation == VectorOrientation::Row ? vector_view.size() : 1),
      ld_(orientation == VectorOrientation::Column ? vector_view.size() : vector_view.inc()),
      stride_(vector_view.stride()),
      batch_size_(vector_view.batch_size()),
      data_(vector_view.data()) {
    if (orientation == VectorOrientation::Column && vector_view.inc() != 1) {
        throw std::runtime_error("Cannot create column matrix view from vector with inc != 1");
    }
}

template <typename T>
template <typename BinaryOperatorOp>
Event VectorView<T>::hadamard_product(const Queue& ctx, T a, T b, const VectorView<T>& x, const VectorView<T>& y, const VectorView<T>& z, BinaryOperatorOp op) {
    // Implementation of hadamard product
    // z = a*x .* b*y
    auto [global_size, local_size] = compute_nd_range_sizes(
        z.size() * z.batch_size(), ctx.device(), KernelType::ELEMENTWISE);

    return static_cast<EventImpl>(ctx -> parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t global_id = item.get_global_id(0);
        size_t total_work_items = item.get_global_range(0);
        for (size_t idx = global_id; idx < x.size() * x.batch_size(); idx += total_work_items) {
            size_t bid = idx / x.size();
            size_t i = idx % x.size();
            z(i, bid) = op(a * x(i, bid), b * y(i, bid));
        }
    }));
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
void Matrix<T, MType>::init() const {
    // No-op if we already have a backend handle
    if (backend_handle_) return;
    backend_handle_ = createBackendHandle(*this);
}

// Operator-> to access backend handle
template <typename T, MatrixFormat MType>
BackendMatrixHandle<T, MType>* Matrix<T, MType>::operator->() {
    if (!backend_handle_) {
        init();
    }
    return backend_handle_.get();
}

// Operator* to access backend handle
template <typename T, MatrixFormat MType>
BackendMatrixHandle<T, MType>& Matrix<T, MType>::operator*() {
    if (!backend_handle_) {
        init();
    }
    return *backend_handle_;
}

//----------------------------------------------------------------------
// Matrix class static method implementations - Dense format only
//----------------------------------------------------------------------

// Factory method to create identity matrix with SYCL
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::Identity(int n, int batch_size) {
    Matrix<T, MType> result(n, n, batch_size);
    
    // Create a queue to submit work to
    Queue q;
    
    // Get pointer to the matrix data
    T* data_ptr = result.data_.data();
    
    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(n) * n * batch_size;
    
    // Use batched matrix decomposition strategy
    auto [global_size, local_size, wg_per_matrix] = compute_batched_matrix_decomposition(
        batch_size, n * n, q.device(), KernelType::ELEMENTWISE);
    
    // Submit a kernel to initialize the identity matrix
    q -> parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t global_id = item.get_global_id(0);
        size_t total_work_items = item.get_global_range(0);
        
        // Grid-stride loop to handle large matrices
        for (size_t flat_idx = global_id; flat_idx < total_elements; flat_idx += total_work_items) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / (n * n);          // batch index
            size_t remainder = flat_idx % (n * n);
            size_t i = remainder / n;               // row index
            size_t j = remainder % n;               // column index
            
            // Set diagonal elements to 1, others to 0
            data_ptr[b * n * n + i * n + j] = (i == j) ? T(1) : T(0);
        }
    }).wait();
    
    return result;
}

// Factory method to create triangular matrix with specific values using SYCL
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::Triangular(int n, Uplo uplo, T diagonal_value,
                                             T non_diagonal_value, int batch_size) {
    Matrix<T, MType> result(n, n, batch_size);
    result.view().fill_triangular(uplo, diagonal_value, non_diagonal_value).wait();
    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_triangular(const Queue& ctx, Uplo uplo, T diagonal_value, 
                                             T non_diagonal_value) const {
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    auto stride = stride_; // Stride for batched matrices
    auto ld = ld_; // Leading dimension

    // Use batched matrix decomposition strategy
    auto [global_size, local_size, wg_per_matrix] = compute_batched_matrix_decomposition(
        batch_size, n * n, ctx.device(), KernelType::ELEMENTWISE);

    ctx -> parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        // Grid-stride loop to handle large matrices
        auto bid = item.get_group(0);
        auto bdim = item.get_local_range(0);
        auto n_wgs = item.get_group_range(0);
        auto grid_stride = n_wgs / wg_per_matrix;
        auto batch_id = bid / wg_per_matrix;
        auto matrix_wg_ix = bid % wg_per_matrix;
        for (batch_id; 
             batch_id < batch_size; 
             batch_id += grid_stride) {
            for (auto flat_idx = matrix_wg_ix; 
                 flat_idx < n*n; 
                 flat_idx += bdim * wg_per_matrix) {
                auto i = flat_idx % n; // row index
                auto j = flat_idx / n; // column index
                if (i == j) {
                    // Diagonal elements
                    data_ptr[batch_id * stride + i * ld + j] = diagonal_value;
                } else if ((uplo == Uplo::Lower && i > j) || 
                        (uplo == Uplo::Upper && i < j)) {
                    // Lower or upper triangular elements (Left = lower, Right = upper)
                    data_ptr[batch_id * stride + i * ld + j] = non_diagonal_value;
                } else {
                    // Zero elements
                    data_ptr[batch_id * stride + i * ld + j] = T(0);
                }
            }
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_triangular_random(const Queue& ctx, Uplo uplo, 
                                                    Diag diag,
                                                    unsigned int seed) const {
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;

    // Use batched matrix decomposition strategy
    auto [global_size, local_size, wg_per_matrix] = compute_batched_matrix_decomposition(
        batch_size, n * n, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t global_id = item.get_global_id(0);
        size_t total_work_items = item.get_global_range(0);
        
        // Grid-stride loop to handle large matrices
        for (size_t flat_idx = global_id; flat_idx < total_elements; flat_idx += total_work_items) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / (n * n);          // batch index
            size_t remainder = flat_idx % (n * n);
            size_t i = remainder % n;               // row index
            size_t j = remainder / n;               // column index
            
            oneapi::dpl::uniform_real_distribution<float_t<T>> dist(-1.0, 1.0);
            oneapi::dpl::minstd_rand engine(seed, remainder);
            T rand_value;
            if constexpr (std::is_same_v<T, std::complex<float>> || 
                          std::is_same_v<T, std::complex<double>>) {
                // For complex numbers, generate both real and imaginary parts
                auto r1 = dist(engine);
                auto r2 = dist(engine);
                rand_value = T(r1, r2);
            } else {
                // For real numbers
                rand_value = dist(engine);
            }
            if (i == j) {
                // Diagonal elements
                data_ptr[b * n * n + i * n + j] = (diag == Diag::Unit) ? T(1) : rand_value;
            } else if ((uplo == Uplo::Lower && i < j) || 
                      (uplo == Uplo::Upper && i > j)) {
                // Lower or upper triangular elements
                data_ptr[b * n * n + i * n + j] = rand_value;
            } else {
                // Zero elements
                data_ptr[b * n * n + i * n + j] = T(0);
            }
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::triangularize(const Queue& ctx, Uplo uplo, 
                                                    Diag diag) const {
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    auto stride = stride_; // Stride for batched matrices
    auto ld = ld_; // Leading dimension

    // Use batched matrix decomposition strategy
    auto [global_size, local_size, wg_per_matrix] = compute_batched_matrix_decomposition(
        batch_size, n * n, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t global_id = item.get_global_id(0);
        size_t total_work_items = item.get_global_range(0);
        
        // Grid-stride loop to handle large matrices
        for (size_t flat_idx = global_id; flat_idx < total_elements; flat_idx += total_work_items) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / (n * n);          // batch index
            size_t remainder = flat_idx % (n * n);
            size_t i = remainder / n;               // row index
            size_t j = remainder % n;               // column index
            
            if (i == j && diag == Diag::Unit) {
                data_ptr[b * stride + i * ld + j] = T(1);
            } else if ((uplo == Uplo::Lower && i < j) || (uplo == Uplo::Upper && i > j)) {
                // Zero out elements above or below the diagonal depending on Uplo
                data_ptr[b * stride + i * ld + j] = T(0);
            }
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::symmetrize(const Queue& ctx, Uplo uplo) const {
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    auto ld = ld_; // Leading dimension
    auto stride = stride_; // Stride for batched matrices

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t flat_idx = item.get_global_id(0);
        
        // Bounds check
        if (flat_idx >= total_elements) return;
        
        // Calculate 3D coordinates from flat index
        size_t b = flat_idx / (n * n);          // batch index
        size_t remainder = flat_idx % (n * n);
        size_t i = remainder / n;               // row index
        size_t j = remainder % n;               // column index

        if (uplo == Uplo::Upper) {
            // Upper triangular part
            if (i < j) {
                size_t idx1 = b * stride + i * ld + j;
                size_t idx2 = b * stride + j * ld + i;
                data_ptr[idx2] = data_ptr[idx1];
            }
        } else {
            // Lower triangular part
            if (i > j) {
                size_t idx1 = b * stride + i * ld + j;
                size_t idx2 = b * stride + j * ld + i;
                data_ptr[idx2] = data_ptr[idx1];
            }
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::hermitize(const Queue& ctx, Uplo uplo) const{
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    auto ld = ld_; // Leading dimension
    auto stride = stride_; // Stride for batched matrices

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t flat_idx = item.get_global_id(0);
        
        // Bounds check
        if (flat_idx >= total_elements) return;
        
        // Calculate 3D coordinates from flat index
        size_t b = flat_idx / (n * n);          // batch index
        size_t remainder = flat_idx % (n * n);
        size_t i = remainder / n;               // row index
        size_t j = remainder % n;               // column index

        if (uplo == Uplo::Upper) {
            // Upper triangular part
            if (i < j) {
                size_t idx1 = b * stride + i * ld + j;
                size_t idx2 = b * stride + j * ld + i;
                data_ptr[idx2] = std::conj(data_ptr[idx1]);
            }
        } else {
            // Lower triangular part
            if (i > j) {
                size_t idx1 = b * stride + i * ld + j;
                size_t idx2 = b * stride + j * ld + i;
                data_ptr[idx2] = std::conj(data_ptr[idx1]);
            }
        }
        if (i == j) {
            // Ensure diagonal elements are real
            if constexpr (std::is_same_v<T, std::complex<float>> || 
                          std::is_same_v<T, std::complex<double>>) {
                size_t diag_idx = b * stride + i * ld + i;
                data_ptr[diag_idx] = T(std::real(data_ptr[diag_idx]), 0);
            }
        }
    });
    return ctx.get_event();
}

// Factory method to create random matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::Random(int rows, int cols, bool hermitian, int batch_size, unsigned int seed) {
    Matrix<T, MType> result(rows, cols, batch_size);
    result.view().fill_random(hermitian, seed).wait();
    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_random(const Queue& ctx, bool hermitian, unsigned int seed) const {
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto rows = rows_;
    auto cols = cols_;
    auto batch_size = batch_size_;

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t idx = item.get_global_id(0);
        
        // Bounds check
        if (idx >= total_elements) return;
        
        oneapi::dpl::uniform_real_distribution<float_t<T>> dist(-1.0, 1.0);
        oneapi::dpl::minstd_rand engine(seed, idx);
        auto r1 = dist(engine);
        if constexpr (std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            auto r2 = dist(engine);
            data_ptr[idx] = T(r1, r2);
        } else {
            data_ptr[idx] = T(r1);
        }
    });
    
    // If hermitian flag is set, enforce Hermitian property using a kernel
    if (hermitian) {
        if (rows != cols) {
            throw std::runtime_error("Hermitian matrices must be square");
        }

        int ld = ld_;
        int stride = stride_;
        size_t hermitian_elements = static_cast<size_t>(batch_size) * rows * cols;
        
        // Compute optimal work-group size for the hermitian operation
        auto [herm_global_size, herm_local_size] = compute_nd_range_sizes(
            hermitian_elements, ctx.device(), KernelType::ELEMENTWISE);
        
        ctx->parallel_for(sycl::nd_range<1>(herm_global_size, herm_local_size),
                        [=](sycl::nd_item<1> item) {
            size_t flat = item.get_global_id(0);
            
            // Bounds check
            if (flat >= hermitian_elements) return;
            
            size_t b = flat / (rows * cols);
            size_t rem = flat % (rows * cols);
            size_t j = rem / rows; // column index
            size_t i = rem % rows; // row index

            if (j > i) {
                size_t idx1 = b * stride + j * ld + i;
                size_t idx2 = b * stride + i * ld + j;
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                              std::is_same_v<T, std::complex<double>>) {
                    data_ptr[idx2] = std::conj(data_ptr[idx1]);
                } else {
                    data_ptr[idx2] = data_ptr[idx1];
                }
            } else if (i == j) {
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                              std::is_same_v<T, std::complex<double>>) {
                    size_t diag_idx = b * stride + i * ld + i;
                    data_ptr[diag_idx] = T(std::real(data_ptr[diag_idx]), 0);
                }
            }
        });
    }
    return ctx.get_event();
}

template <typename T, MatrixFormat MType> struct FillContiguousKernel {};
template <typename T, MatrixFormat MType> struct FillNonContiguousKernel {};

template <typename T, MatrixFormat MType>
Event MatrixView<T, MType>::fill(const Queue& ctx, T value) const {
    auto data_ptr = data_.data();
    size_t total_elements = 0;
    bool contiguous = false;
    if constexpr (MType == MatrixFormat::CSR) {
        total_elements = static_cast<size_t>(data_.size());
        contiguous = true;
    } else {
        total_elements = static_cast<size_t>(batch_size_) * static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
        contiguous = (stride_ == rows_ * cols_) && (ld_ == rows_);
    }
    bool char_val = 0;
    bool fill_val_is_zero = value == detail::convert_to_fill_value<T>(0);
    if constexpr (sizeof(T) == sizeof(char)) {
        char_val = static_cast<char>(value);
    }
    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    if (fill_val_is_zero && contiguous) {
        ctx->memset(data_ptr, char_val, total_elements * sizeof(T));
    } else if (contiguous) {
        ctx->parallel_for<FillContiguousKernel<T, MType>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            size_t global_id = item.get_global_id(0);
            size_t total_work_items = item.get_global_range(0);
            
            // Grid-stride loop to handle large data
            for (size_t idx = global_id; idx < total_elements; idx += total_work_items) {
                data_ptr[idx] = value;
            }
        });
    } else {
        auto batch_size = batch_size_;
        auto stride = stride_; // Stride for batched matrices
        auto ld = ld_; // Leading dimension
        auto rows = rows_;
        auto cols = cols_;
        auto data_view = this->kernel_view();
        if constexpr (MType == MatrixFormat::Dense) {
        ctx->parallel_for<FillNonContiguousKernel<T, MType>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            size_t global_id = item.get_global_id(0);
            size_t total_work_items = item.get_global_range(0);
            
            // Grid-stride loop to handle large data
            for (size_t flat_idx = global_id; flat_idx < total_elements; flat_idx += total_work_items) {
                // Calculate 3D coordinates from flat index
                size_t b = flat_idx / (cols * rows);          // batch index
                size_t remainder = flat_idx % (cols * rows);
                size_t col = remainder / rows;               // column index
                size_t row = remainder % rows;               // row index

                data_view(row, col, b) = value;
            }
        });
        }
        
    }
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_identity(const Queue& ctx, T value) const {
    fill_zeros(ctx);
    fill_diagonal(ctx, value);
    return ctx.get_event();
}

// Factory method to create zeros matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::Zeros(int rows, int cols, int batch_size) {
    Matrix<T, MType> result(rows, cols, batch_size);
    result.fill(T(0));
    return result;
}

// Factory method to create ones matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::Ones(int rows, int cols, int batch_size) {
    Matrix<T, MType> result(rows, cols, batch_size);
    result.fill(T(1));
    return result;
}

// Factory method to create diagonal matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::Diagonal(const Span<T>& diag_values, int batch_size) {
    int n = diag_values.size();
    Matrix<T, MType> result(n, n, batch_size);

    // Set all elements to zero
    result.fill(T(0));
    result.view().template fill_diagonal<MType>(diag_values);

    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_diagonal(const Queue& ctx, const Span<T>& diag_values, int64_t k) const {
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    bool batch_diagonals = (diag_values.size() == (n - std::abs(k)) * batch_size);
    return fill_diagonal(ctx, VectorView<T>(diag_values, n, 1, n, batch_diagonals ? batch_size : 1), k);
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_diagonal(const Queue& ctx, const VectorView<T>& diag_values, int64_t k) const {
    int ld = ld_;
    int stride = stride_;
    int n = rows_; // Assuming square matrix
    int batch_size = batch_size_;
    bool batch_diagonals = diag_values.batch_size() == batch_size;
    size_t total_elements = static_cast<size_t>(batch_size) * (n - std::abs(k));
    
    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    auto col_offset = (k >= 0) ? k : 0; // Adjust column offset based on k
    auto row_offset = (k < 0) ? -k : 0; // Adjust row offset based on k
    ctx->submit([=](sycl::handler& cgh) {
        auto view = this->kernel_view();
        cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item){
            size_t gid = item.get_global_id(0);
            
            // Bounds check
            if (gid >= total_elements) return;
            for (int flat = gid; flat < total_elements; flat += item.get_global_range(0)) {
                size_t b = flat / (n - std::abs(k));
                size_t i = flat % (n - std::abs(k));
                view(i + row_offset, i + col_offset, b) = diag_values(i, batch_diagonals ? b : 0);
            }
        });
    });    
    return ctx.get_event();
}

template <typename T, MatrixFormat MType> struct FillDiagonalKernel {};
template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_diagonal(const Queue& ctx, const T& diag_value) const {
    T* data_ptr = data_.data();
    int ld = ld_;
    int stride = stride_;
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    
    size_t total_elements = static_cast<size_t>(batch_size) * n;
    
    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for<FillDiagonalKernel<T, MType>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item){
        size_t flat = item.get_global_id(0);
        
        // Bounds check
        if (flat >= total_elements) return;
        
        size_t b = flat / n;
        size_t i = flat % n;
        data_ptr[b * stride + i * ld + i] = diag_value;
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::TriDiagToeplitz(int n, T diag, T sub, T super, int batch_size) {
    Matrix<T, MType> result(n, n, batch_size);
    result.view().template fill_tridiag_toeplitz<MType>(diag, sub, super);
    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Event MatrixView<T, MType>::fill_tridiag_toeplitz(const Queue& ctx, T diag, T sub_diag, T super_diag) const{
    T* data_ptr = data().data();
    int n = rows_; // Assuming square matrix
    int batch_size = batch_size_;
    
    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(n) * n * batch_size;
    
    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);
    
    // Submit a kernel to initialize the tridiagonal Toeplitz matrix
    ctx -> parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t flat_idx = item.get_global_id(0);
        
        // Bounds check
        if (flat_idx >= total_elements) return;
        
        // Calculate 3D coordinates from flat index
        int64_t b = flat_idx / (n * n);          // batch index
        int64_t remainder = flat_idx % (n * n);
        int64_t i = remainder % n;               // row index
        int64_t j = remainder / n;               // column index
        
        if (i == j) {
            data_ptr[b * n * n + j * n + i] = diag; // Diagonal element
        } else if (i == j - 1) {
            data_ptr[b * n * n + j * n + i] = super_diag; // Super-diagonal element
        } else if (i == j + 1) {
            data_ptr[b * n * n + j * n + i] = sub_diag;   // Sub-diagonal element
        } else {
            data_ptr[b * n * n + j * n + i] = T(0);   // Zero elsewhere
        }
    });
    return ctx.get_event();
}

// Implement to_column_major using SYCL
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::to_column_major() const {
    Matrix<T, MType> col_major(rows_, cols_, batch_size_);
    
    // Create a queue to submit work to
    Queue q;
    
    // Get pointers to source and destination data
    const T* src_ptr = data_.data();
    T* dst_ptr = col_major.data_.data();
    
    // Capture dimensions for kernel
    int rows = rows_;
    int cols = cols_;
    int ld = ld_;
    int stride = stride_;
    
    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(rows_) * cols_ * batch_size_;
    
    // Compute optimal work-group size for memory-bound operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, q.device(), KernelType::MEMORY_BOUND);
    
    // Submit a kernel to convert from row-major to column-major
    q->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t flat_idx = item.get_global_id(0);
        
        // Bounds check
        if (flat_idx >= total_elements) return;
        
        // Calculate 3D coordinates from flat index
        size_t b = flat_idx / (rows * cols);    // batch index
        size_t remainder = flat_idx % (rows * cols);
        size_t i = remainder / cols;            // row index
        size_t j = remainder % cols;            // column index
        
        // src is row-major: [b, i, j] -> b * stride + i * cols + j
        // dst is col-major: [b, j, i] -> b * stride + j * rows + i
        dst_ptr[b * stride + j * rows + i] = src_ptr[b * stride + i * cols + j];
    }).wait();
    
    return col_major;
}

// Implement to_row_major using SYCL
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
Matrix<T, MType> Matrix<T, MType>::to_row_major() const {
    Matrix<T, MType> row_major(rows_, cols_, batch_size_);
    
    // Create a queue to submit work to
    Queue q;
    
    // Get pointers to source and destination data
    const T* src_ptr = data_.data();
    T* dst_ptr = row_major.data_.data();
    
    // Capture dimensions for kernel
    int rows = rows_;
    int cols = cols_;
    int ld = ld_;
    int stride = stride_;
    
    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(rows_) * cols_ * batch_size_;
    
    // Compute optimal work-group size for memory-bound operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, q.device(), KernelType::MEMORY_BOUND);
    
    // Submit a kernel to convert from column-major to row-major
    q->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t flat_idx = item.get_global_id(0);
        
        // Bounds check
        if (flat_idx >= total_elements) return;
        
        // Calculate 3D coordinates from flat index
        size_t b = flat_idx / (rows * cols);    // batch index
        size_t remainder = flat_idx % (rows * cols);
        size_t i = remainder / cols;            // row index
        size_t j = remainder % cols;            // column index
        
        // src is col-major: [b, j, i] -> b * stride + j * rows + i
        // dst is row-major: [b, i, j] -> b * stride + i * cols + j
        dst_ptr[b * stride + i * cols + j] = src_ptr[b * stride + j * rows + i];
    }).wait();
    
    return row_major;
}

//----------------------------------------------------------------------
// MatrixView class implementation - Dense format
//----------------------------------------------------------------------

// Constructor for dense matrix view
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
MatrixView<T, MType>::MatrixView(T* data, int rows, int cols, int ld,
                  int stride, int batch_size, T** data_ptrs) :
                data_(data, (stride > 0 ? stride : 
                            ld > 0 ? ld * cols : rows * cols) * batch_size),
                rows_(rows), cols_(cols), batch_size_(batch_size),
                ld_(ld > 0 ? ld : rows), stride_(stride > 0 ? stride : ld > 0 ? ld * cols : rows * cols), data_ptrs_(data_ptrs, (data_ptrs ? batch_size : 0)) {}


//----------------------------------------------------------------------
// MatrixView class implementation - CSR format
//----------------------------------------------------------------------

// Constructor for CSR sparse matrix view
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::CSR, int>::type>
MatrixView<T, MType>::MatrixView(T* data, int* row_offsets, int* col_indices,
            int nnz, int rows, int cols, int matrix_stride,
            int offset_stride, int batch_size, T** data_ptrs) :
            data_(data, (matrix_stride > 0 ? matrix_stride : nnz) * batch_size),
            col_indices_(col_indices, (matrix_stride > 0 ? matrix_stride : nnz) * batch_size),
            row_offsets_(row_offsets, (offset_stride > 0 ? offset_stride : rows + 1) * batch_size),
            rows_(rows), cols_(cols), batch_size_(batch_size), nnz_(nnz), matrix_stride_(matrix_stride > 0 ? matrix_stride : nnz),
            offset_stride_(offset_stride > 0 ? offset_stride : rows + 1), data_ptrs_(data_ptrs, (data_ptrs ? batch_size : 0)) {
}

//----------------------------------------------------------------------
// Common MatrixView implementations (for both Dense and CSR)
//----------------------------------------------------------------------

// Constructor from Matrix object - fix access to private members
template <typename T, MatrixFormat MType>
MatrixView<T, MType>::MatrixView(const Matrix<T, MType>& matrix)
    : rows_(matrix.rows_), cols_(matrix.cols_), batch_size_(matrix.batch_size_),
      data_(matrix.data()), data_ptrs_(matrix.data_ptrs_, matrix.batch_size_) {
      
    // Format-specific initializations
    if constexpr (MType == MatrixFormat::Dense) {
        ld_ = matrix.template ld<MType>();
        stride_ = matrix.template stride<MType>();
        
        // Try to reuse the backend handle
        backend_handle_ = matrix.backend_handle();
        
        // Don't try to access private data_ptrs_ directly
        // The data pointers will be created on-demand if needed
    } else if constexpr (MType == MatrixFormat::CSR) {
        row_offsets_ = matrix.template row_offsets<MType>();
        col_indices_ = matrix.template col_indices<MType>();
        nnz_ = matrix.template nnz<MType>();
        matrix_stride_ = matrix.template matrix_stride<MType>();
        offset_stride_ = matrix.template offset_stride<MType>();
        
        // Try to reuse the backend handle
        backend_handle_ = matrix.backend_handle();
    }
}

// Constructor for a submatrix view from a Matrix object
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
MatrixView<T, MType>::MatrixView(
    const MatrixView<T, MType>& matrix, int rows, int cols, int ld, int stride, int batch_size) : 
    MatrixView<T, MType>(matrix.data_ptr(),
                        rows > 0 ? rows : matrix.rows_, 
                        cols > 0 ? cols : matrix.cols_, 
                        ld > 0 ? ld : matrix.ld_, 
                        stride > 0 ? stride : matrix.stride_,
                        batch_size > 0 ? batch_size : matrix.batch_size_,
                        matrix.data_ptrs_.data()) {}

template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
MatrixView<T, MType>::MatrixView(
    const Matrix<T, MType>& matrix, int rows, int cols, int ld, int stride, int batch_size) : MatrixView<T, MType>(matrix.view(), rows, cols, ld, stride, batch_size) {}


// Element access (for dense matrices)
template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
T& MatrixView<T, MType>::at(int row, int col, int batch) {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || batch < 0 || batch >= batch_size_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[batch * stride_ + col * ld_ + row];
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
const T& MatrixView<T, MType>::at(int row, int col, int batch) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || batch < 0 || batch >= batch_size_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[batch * stride_ + col * ld_ + row];
}

// Create a view of a single batch item
template <typename T, MatrixFormat MType>
MatrixView<T, MType> MatrixView<T, MType>::batch_item(int batch_index) const {
    if (batch_index >= batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    if constexpr (MType == MatrixFormat::Dense) {
        // Calculate offset to the start of the batch
        size_t offset = batch_index * stride_;
        
        // Create a new view with batch_size = 1
        return MatrixView<T, MType>(
            data_.data() + offset,
            rows_, cols_, ld_);
    } else if constexpr (MType == MatrixFormat::CSR) {
        // Calculate offsets to the start of the batch
        size_t val_offset = batch_index * matrix_stride_;
        size_t row_offset = batch_index * offset_stride_;
        
        // Create a new view with batch_size = 1
        return MatrixView<T, MType>(
            data_.data() + val_offset,
            row_offsets_.data() + row_offset,
            col_indices_.data() + val_offset,
            nnz_, rows_, cols_);
    }
}

template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M, typename std::enable_if<M == MatrixFormat::Dense, int>::type>
MatrixView<T, MType> MatrixView<T, MType>::deep_copy(const MatrixView<T, MType>& other,
                                                     T* data,
                                                     T** data_ptrs) {
    MatrixView<T, MType> result(data, other.rows_, other.cols_, other.ld_, other.stride_, other.batch_size_, data_ptrs);
    Queue q(Device::default_device());
    q -> memcpy(data, other.data_.data(), other.data_.size() * sizeof(T));
    if (data_ptrs) {
        q -> memcpy(data_ptrs, other.data_ptrs_.data(), other.data_ptrs_.size() * sizeof(T*));
    }
    q -> wait();
    return result;
}

template <typename T, MatrixFormat MType> struct CopyKernel3D {};
template <typename T, MatrixFormat MType>
Event MatrixView<T, MType>::copy(Queue& ctx, const MatrixView<T, MType>& dest, const MatrixView<T, MType>& src) {
    if (src.rows() != dest.rows() || src.cols() != dest.cols() || src.batch_size() != dest.batch_size()) {
        throw std::invalid_argument("Copy called on incompatible matrices");
    }
    if constexpr(MType == MatrixFormat::Dense) {
        if (src.data_ptr() == dest.data_ptr()) return ctx.get_event(); // No-op if same buffer
        if (src.stride() == src.rows() * src.cols() && dest.stride() == dest.rows() * dest.cols()) {
            //If the stride is the same as the size of each matrix just do a straight memcpy
            auto event = static_cast<EventImpl>(ctx->memcpy(dest.data_.data(), src.data_.data(), src.data().size_bytes()));
            return event;
        } else if (src.ld() == src.rows() && dest.ld() == dest.rows()){
            //If both leading dimensions are the same as the number of rows we can utilize oneapis 2d copy kernel
            auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                    dest.stride() * sizeof(T),
                                                    static_cast<void*>(src.data_ptr()),
                                                    src.stride() * sizeof(T),
                                                    src.rows()*src.cols()*sizeof(T), src.batch_size()));
            return event;
        } else if (src.stride() == src.ld()* src.cols() && dest.stride() == dest.ld()* dest.cols()) {
            // If both the strides and leading dimensions are the same as the size of each matrix just do a straight memcpy
            auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                    dest.ld() * sizeof(T),
                                                    static_cast<void*>(src.data_ptr()),
                                                    src.ld() * sizeof(T),
                                                    src.rows()*sizeof(T), src.cols() * src.batch_size()));
            return event;
        } else {
            //If neither the ld nor the strides are unit, we need to do essentially a 3D copy
            auto [ global_size, local_size ] = compute_nd_range_sizes(
                    static_cast<size_t>(src.rows()) * src.cols() * src.batch_size(),
                    ctx.device(), KernelType::MEMORY_BOUND);
            auto event = static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
                auto dest_view = dest.kernel_view();
                auto src_view  = src.kernel_view();
                

                cgh.parallel_for<CopyKernel3D<T, MType>>(sycl::nd_range<1>(global_size, local_size),
                                [=](sycl::nd_item<1> item) {
                    size_t gid = item.get_global_id(0);
                    
                    // Calculate 3D coordinates from flat index
                    for (size_t idx = gid; idx < static_cast<size_t>(src_view.rows()) * src_view.cols() * src_view.batch_size(); idx += item.get_global_range(0)) {
                        size_t b = idx / (src_view.rows() * src_view.cols());    // batch index
                        size_t remainder = idx % (src_view.rows() * src_view.cols());
                        size_t i = remainder % src_view.cols();            // row index
                        size_t j = remainder / src_view.cols();            // column index
                        
                        dest_view(i, j, b) = src_view(i, j, b);
                    }
                });
            }));
            return event;
        }
    } else if constexpr(MType == MatrixFormat::CSR) {
        throw std::runtime_error("CSR copy not implemented yet");
    } else {
        throw std::runtime_error("Unsupported matrix format in copy");
    }
    return ctx.get_event();
}

template <typename T>
Event VectorView<T>::copy(Queue& ctx, const VectorView<T>& dest, const VectorView<T>& src) {
    if (src.size() != dest.size() || src.batch_size() != dest.batch_size()) {
        throw std::invalid_argument("Copy called on incompatible vectors");
    }
    if (src.data_ptr() == dest.data_ptr()) return ctx.get_event(); // No-op if same buffer
    if ((src.stride() == src.size() && dest.stride() == dest.size() && src.inc() == 1 && dest.inc() == 1) ||
        (src.stride() == 1 && dest.stride() == 1 && src.inc() == src.size() && dest.inc() == dest.size())) {
        //If the stride is the same as the size of each vector just do a straight memcpy
        auto event = static_cast<EventImpl>(ctx->memcpy(dest.data_.data(), src.data_.data(), src.size() * src.batch_size() * sizeof(T)));
        return event;
    } else if (src.stride() == src.inc() * src.size() && dest.stride() == dest.inc() * dest.size()) {
        // If both the strides and increments are the same as the size of each vector just do a straight 2D memcpy
        auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                dest.inc() * sizeof(T),
                                                static_cast<void*>(src.data_ptr()),
                                                src.inc() * sizeof(T),
                                                sizeof(T), src.size() * src.batch_size()));
        return event;
    } else if (src.inc() == 1 && dest.inc() == 1 && src.stride() >= src.size() && dest.stride() >= dest.size()) {
        // If both increments are 1, we can do a 2D memcpy with the strides
        auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                dest.stride() * sizeof(T),
                                                static_cast<void*>(src.data_ptr()),
                                                src.stride() * sizeof(T),
                                                src.size() * sizeof(T), src.batch_size()));
        return event;
    } else if (src.inc() == src.stride() * src.size() && dest.inc() == dest.stride() * dest.size()) {
        // If both the increments are the same as the size of each vector just do a straight 2D memcpy
        auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                dest.stride() * sizeof(T),
                                                static_cast<void*>(src.data_ptr()),
                                                src.stride() * sizeof(T),
                                                sizeof(T), src.size() * src.batch_size()));
        return event;
    } else {
        //If neither the ld nor the strides are unit, we need to do essentially a 2D copy
        auto event = static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<2>(src.size(), src.batch_size()),
                            [=](sycl::id<2> idx) {
                size_t i = idx[0]; // index
                size_t b = idx[1]; // batch
                dest(i, b) = src(i, b);
            });
        }));
        return event;
    }
}

template <typename T>
bool VectorView<T>::all_close(Queue& ctx, const VectorView<T>& a, const VectorView<T>& b, float_t<T> tol) {
    if (a.size() != b.size() || a.batch_size() != b.batch_size()) {
        throw std::invalid_argument("all_close called on incompatible vectors");
    }
    UnifiedVector<bool> batch_results(a.batch_size());

    ctx -> submit([&](sycl::handler& cgh) {
        auto results = batch_results.data();
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(a.batch_size() * a.size()), sycl::range<1>(a.size())), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto lid = item.get_local_id(0);
            bool is_close = true;
            auto compare = [=](const auto& x,const auto& y) { 
                auto max_val = std::max(std::abs(x), std::abs(y));
                if (max_val <= std::numeric_limits<float_t<T>>::epsilon()) {
                    return true; // Both values are effectively zero
                }
                auto diff = std::abs(x - y);
                return diff / (max_val > tol ? max_val : 1) <= tol;
            };
            if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                is_close = compare(a(lid, bid).real(), b(lid, bid).real()) &&
                           compare(a(lid, bid).imag(), b(lid, bid).imag());
            } else {
                is_close = compare(a(lid, bid), b(lid, bid));   
            }
            
            results[bid] = sycl::all_of_group(item.get_group(), is_close);
        });
    }).wait();

    ctx -> submit([&](sycl::handler& cgh) {
        auto results = batch_results.data();
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(std::min(1024, a.batch_size())), sycl::range<1>(std::min(1024, a.batch_size()))), [=](sycl::nd_item<1> item) {
            auto cta = item.get_group();
            bool result = sycl::joint_reduce(cta, results, results + a.batch_size(), sycl::logical_and<bool>());
            if (item.get_local_id(0) == 0) {
                results[0] = result;
            }
        });

    }).wait();
    return batch_results[0];
}

template <typename T, MatrixFormat MType>
bool MatrixView<T, MType>::all_close(Queue& ctx, const MatrixView<T, MType>& A, const MatrixView<T, MType>& B, float_t<T> tol) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.batch_size() != B.batch_size()) {
        throw std::invalid_argument("all_close called on incompatible matrices");
    }
    UnifiedVector<bool> batch_results(A.batch_size());
    if constexpr (MType == MatrixFormat::Dense) {
        
    
        ctx -> submit([&](sycl::handler& cgh) {
            auto results = batch_results.data();
            auto Aview = A.kernel_view();
            auto Bview = B.kernel_view();

            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(A.batch_size() *  A.rows()), sycl::range<1>(A.rows())), [=](sycl::nd_item<1> item) {
                auto bid = item.get_group(0);
                auto lid = item.get_local_id(0);
                bool is_close = true;
                auto compare = [=](const auto& x,const auto& y) { 
                    auto max_val = std::max(std::abs(x), std::abs(y));
                    if (max_val <= std::numeric_limits<float_t<T>>::epsilon()) {
                        return true; // Both values are effectively zero
                    }
                    auto diff = std::abs(x - y);
                    return diff / (max_val > tol ? max_val : 1) <= tol;
                };
                results[bid] = true; // Initialize to true

                for (int col = 0; col < Aview.cols(); ++col) {
                    auto a_element = Aview(lid, col, bid);
                    auto b_element = Bview(lid, col, bid);
                    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                        is_close = is_close && compare(a_element.real(), b_element.real()) &&
                                        compare(a_element.imag(), b_element.imag());
                    } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                        is_close = is_close && compare(a_element, b_element);
                    } else {
                        is_close = is_close && a_element == b_element;
                    }
                    results[bid] &= sycl::all_of_group(item.get_group(), is_close);
                    if (!results[bid]) {
                        break; // Early exit if already false
                    }
                }
            });
        }).wait();
    } else {
        throw std::runtime_error("all_close not implemented for this matrix format");
    }

    ctx -> submit([&](sycl::handler& cgh) {
        auto results = batch_results.data();
        auto Aview = A.kernel_view();
        auto batch_size = A.batch_size();
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(std::min(1024, batch_size)), sycl::range<1>(std::min(1024, batch_size))), [=](sycl::nd_item<1> item) {
            auto cta = item.get_group();
            bool result = sycl::joint_reduce(cta, results, results + batch_size, sycl::logical_and<bool>());
            if (item.get_local_id(0) == 0) {
                results[0] = result;
            }
        });
    }).wait();
    return batch_results[0];
}

// Element access operator - returns view of a single batch item
template <typename T, MatrixFormat MType>
MatrixView<T, MType> MatrixView<T, MType>::operator[](int i) const {
    return batch_item(i);
}

// Initialize backend - make backend_handle_ mutable or change method to non-const
template <typename T, MatrixFormat MType>
void MatrixView<T, MType>::init() const {
    if (backend_handle_) return;
    backend_handle_ = createBackendHandle(*this);
}

template <typename T, MatrixFormat MType>
Event scale(Queue& ctx, const T& alpha, const MatrixView<T, MType>& matrix) {
    auto [global_size, local_size] = compute_nd_range_sizes(
        static_cast<size_t>(matrix.rows()) * matrix.cols() * matrix.batch_size(),
        ctx.device(), KernelType::ELEMENTWISE);
    if constexpr (MType == MatrixFormat::CSR) {
        throw std::runtime_error("scale not implemented for CSR matrices yet");
    } else if constexpr (MType == MatrixFormat::Dense) {
        return static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
            auto view = matrix.kernel_view();
            cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
                size_t gid = item.get_global_id(0);
                if (gid < static_cast<size_t>(view.rows()) * view.cols() * view.batch_size()) {
                    size_t b = gid / (view.rows() * view.cols());
                    size_t rem = gid % (view.rows() * view.cols());
                    size_t j = rem / view.rows();
                    size_t i = rem % view.rows();
                    view(i, j, b) *= alpha;
                }
            });
        }));
    }
}

template <typename T, MatrixFormat MType>
Event scale(Queue& ctx, const T& alpha, const VectorView<T>& vector) {
    auto [global_size, local_size] = compute_nd_range_sizes(
        static_cast<size_t>(vector.size()) * vector.batch_size(),
        ctx.device(), KernelType::ELEMENTWISE);
    return static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            size_t gid = item.get_global_id(0);
            if (gid < static_cast<size_t>(vector.size()) * vector.batch_size()) {
                size_t b = gid / vector.size();
                size_t i = gid % vector.size();
                vector(i, b) *= alpha;
            }
        });
    }));
}


template <typename T, MatrixFormat MType>
void MatrixView<T, MType>::init_data_ptr_array(Queue& ctx) const {
    auto [start_ptr, stride, data_ptrs] = std::make_tuple(this->data_ptr(), stride_, data_ptrs_);
    if (!data_ptrs.data()) {
        throw std::runtime_error("data_ptrs is null");
    }
    if (!start_ptr) {
        throw std::runtime_error("start_ptr is null");
    }
    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(batch_size_), [=](sycl::id<1> idx) {
            size_t b = idx[0];
            data_ptrs[b] = start_ptr + b * stride;
        });
    }).wait();
}

template <typename T, MatrixFormat MType>
void Matrix<T, MType>::init_data_ptr_array(Queue& ctx) const {
    this->view().data_ptrs(ctx);
}

// Operator-> to access backend handle
template <typename T, MatrixFormat MType>
BackendMatrixHandle<T, MType>* MatrixView<T, MType>::operator->() const{
    if (!backend_handle_) {
        init();
    }
    return backend_handle_.get();
}

// Operator* to access backend handle
template <typename T, MatrixFormat MType>
BackendMatrixHandle<T, MType>& MatrixView<T, MType>::operator*() const {
    if (!backend_handle_) {
        init();
    }
    return *backend_handle_;
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

template class MatrixView<std::array<float, 2>, MatrixFormat::Dense>;
template class MatrixView<std::array<double, 2>, MatrixFormat::Dense>;

// CSR MatrixView instantiations
template class MatrixView<float, MatrixFormat::CSR>;
template class MatrixView<double, MatrixFormat::CSR>;
template class MatrixView<std::complex<float>, MatrixFormat::CSR>;
template class MatrixView<std::complex<double>, MatrixFormat::CSR>;

template class VectorView<float>;
template class VectorView<double>;
template class VectorView<std::complex<float>>;
template class VectorView<std::complex<double>>;

// Dense constructor instantiations
template Matrix<float, MatrixFormat::Dense>::Matrix(int, int, int, int, int);
template Matrix<double, MatrixFormat::Dense>::Matrix(int, int, int, int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense>::Matrix(int, int, int, int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense>::Matrix(int, int, int, int, int);

template Matrix<float, MatrixFormat::Dense>::Matrix(const float*, int, int, int, int, int);
template Matrix<double, MatrixFormat::Dense>::Matrix(const double*, int, int, int, int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense>::Matrix(const std::complex<float>*, int, int, int, int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense>::Matrix(const std::complex<double>*, int, int, int, int, int);

// CSR constructor instantiations
template Matrix<float, MatrixFormat::CSR>::Matrix(int, int, int, int);
template Matrix<double, MatrixFormat::CSR>::Matrix(int, int, int, int);
template Matrix<std::complex<float>, MatrixFormat::CSR>::Matrix(int, int, int, int);
template Matrix<std::complex<double>, MatrixFormat::CSR>::Matrix(int, int, int, int);

template Matrix<float, MatrixFormat::CSR>::Matrix(const float*, const int*, const int*, int, int, int, int, int, int);
template Matrix<double, MatrixFormat::CSR>::Matrix(const double*, const int*, const int*, int, int, int, int, int, int);
template Matrix<std::complex<float>, MatrixFormat::CSR>::Matrix(const std::complex<float>*, const int*, const int*, int, int, int, int, int, int);
template Matrix<std::complex<double>, MatrixFormat::CSR>::Matrix(const std::complex<double>*, const int*, const int*, int, int, int, int, int, int);

// Static factory method instantiations
template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Identity(int, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Identity(int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Identity(int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Identity(int, int);

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);

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

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::to_column_major() const;
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::to_column_major() const;
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::to_column_major() const;
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::to_column_major() const;

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::to_row_major() const;
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::to_row_major() const;
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::to_row_major() const;
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::to_row_major() const;

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Triangular(int, Uplo, float, float, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Triangular(int, Uplo, double, double, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Triangular(int, Uplo, std::complex<float>, std::complex<float>, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Triangular(int, Uplo, std::complex<double>, std::complex<double>, int);

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::TriDiagToeplitz(int, float, float, float, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::TriDiagToeplitz(int, double, double, double, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::TriDiagToeplitz(int, std::complex<float>, std::complex<float>, std::complex<float>, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::TriDiagToeplitz(int, std::complex<double>, std::complex<double>, std::complex<double>, int);
//----------------------------------------------------------------------
// Matrix conversion instantiations
template Matrix<float, MatrixFormat::CSR> Matrix<float, MatrixFormat::Dense>::convert_to<MatrixFormat::CSR>(const float&) const;
template Matrix<double, MatrixFormat::CSR> Matrix<double, MatrixFormat::Dense>::convert_to<MatrixFormat::CSR>(const double&) const;
template Matrix<std::complex<float>, MatrixFormat::CSR> Matrix<std::complex<float>, MatrixFormat::Dense>::convert_to<MatrixFormat::CSR>(const float&) const;
template Matrix<std::complex<double>, MatrixFormat::CSR> Matrix<std::complex<double>, MatrixFormat::Dense>::convert_to<MatrixFormat::CSR>(const double&) const;

//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Deep copy instantiations
template MatrixView<float, MatrixFormat::Dense> MatrixView<float, MatrixFormat::Dense>::deep_copy(const MatrixView<float, MatrixFormat::Dense>&, float*, float**);
template MatrixView<double, MatrixFormat::Dense> MatrixView<double, MatrixFormat::Dense>::deep_copy(const MatrixView<double, MatrixFormat::Dense>&, double*, double**);
template MatrixView<std::complex<float>, MatrixFormat::Dense> MatrixView<std::complex<float>, MatrixFormat::Dense>::deep_copy(const MatrixView<std::complex<float>, MatrixFormat::Dense>&, std::complex<float>*, std::complex<float>**);
template MatrixView<std::complex<double>, MatrixFormat::Dense> MatrixView<std::complex<double>, MatrixFormat::Dense>::deep_copy(const MatrixView<std::complex<double>, MatrixFormat::Dense>&, std::complex<double>*, std::complex<double>**);

template Event VectorView<float>::hadamard_product<std::plus<float>>(const Queue&, float a, float b, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, std::plus<float>) ;
template Event VectorView<double>::hadamard_product<std::plus<double>>(const Queue&, double a, double b, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, std::plus<double>) ;
template Event VectorView<std::complex<float>>::hadamard_product<std::plus<std::complex<float>>>(const Queue&, std::complex<float> a, std::complex<float> b, const VectorView<std::complex<float>>&, const VectorView<std::complex<float>>&, const VectorView<std::complex<float>>&, std::plus<std::complex<float>>) ;
template Event VectorView<std::complex<double>>::hadamard_product<std::plus<std::complex<double>>>(const Queue&, std::complex<double> a, std::complex<double> b, const VectorView<std::complex<double>>&, const VectorView<std::complex<double>>&, const VectorView<std::complex<double>>&, std::plus<std::complex<double>>) ;  

// fill instantiations
template Event MatrixView<float, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, float, float) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, double, double) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, std::complex<float>, std::complex<float>) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, std::complex<double>, std::complex<double>) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_tridiag_toeplitz(const Queue&, float, float, float) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_tridiag_toeplitz(const Queue&, double, double, double) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_tridiag_toeplitz(const Queue&, std::complex<float>, std::complex<float>, std::complex<float>) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_tridiag_toeplitz(const Queue&, std::complex<double>, std ::complex<double>, std::complex<double>) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_diagonal(const Queue&, const Span<float>&, int64_t) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_diagonal(const Queue&, const Span<double>&, int64_t) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_diagonal(const Queue&, const Span<std::complex<float>>&, int64_t) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_diagonal(const Queue&, const Span<std::complex<double>>&, int64_t) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_diagonal(const Queue&, const VectorView<float>&, int64_t) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_diagonal(const Queue&, const VectorView<double>&, int64_t) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_diagonal(const Queue&, const VectorView<std::complex<float>>&, int64_t) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_diagonal(const Queue&, const VectorView<std::complex<double>>&, int64_t) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_diagonal(const Queue&, const float&) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_diagonal(const Queue&, const double&) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_diagonal(const Queue&, const std::complex<float>&) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_diagonal(const Queue&, const std::complex<double>&) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_identity(const Queue&, float) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_identity(const Queue&, double) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_identity(const Queue&, std::complex<float>) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_identity(const Queue&, std::complex<double>) const;

template Event MatrixView<float, MatrixFormat::Dense>::triangularize(const Queue&, Uplo, Diag) const;
template Event MatrixView<double, MatrixFormat::Dense>::triangularize(const Queue&, Uplo, Diag) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::triangularize(const Queue&, Uplo, Diag) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::triangularize(const Queue&, Uplo, Diag) const;

template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::hermitize(const Queue&, Uplo) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::hermitize(const Queue&, Uplo) const;

template Event MatrixView<float, MatrixFormat::Dense>::symmetrize(const Queue&, Uplo) const;
template Event MatrixView<double, MatrixFormat::Dense>::symmetrize(const Queue&, Uplo) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::symmetrize(const Queue&, Uplo) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::symmetrize(const Queue&, Uplo) const;

template Event scale<float, MatrixFormat::Dense>(Queue&, const float&, const MatrixView<float, MatrixFormat::Dense>&);
template Event scale<double, MatrixFormat::Dense>(Queue&, const double&, const MatrixView<double, MatrixFormat::Dense>&);
template Event scale<std::complex<float>, MatrixFormat::Dense>(Queue&, const std::complex<float>&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&);
template Event scale<std::complex<double>, MatrixFormat::Dense>(Queue&, const std::complex<double>&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&);
template Event scale<float, MatrixFormat::Dense>(Queue&, const float&, const VectorView<float>&);
template Event scale<double, MatrixFormat::Dense>(Queue&, const double&, const VectorView<double>&);
template Event scale<std::complex<float>, MatrixFormat::Dense>(Queue&, const std::complex<float>&, const VectorView<std::complex<float>>&);
template Event scale<std::complex<double>, MatrixFormat::Dense>(Queue&, const std::complex<double>&, const VectorView<std::complex<double>>&);

// Dense MatrixView constructors instantiations
template MatrixView<float, MatrixFormat::Dense>::MatrixView(float*, int, int, int, int, int, float**);
template MatrixView<double, MatrixFormat::Dense>::MatrixView(double*, int, int, int, int, int, double**);
template MatrixView<std::complex<float>, MatrixFormat::Dense>::MatrixView(std::complex<float>*, int, int, int, int, int, std::complex<float>**);
template MatrixView<std::complex<double>, MatrixFormat::Dense>::MatrixView(std::complex<double>*, int, int, int, int, int, std::complex<double>**);

template MatrixView<std::array<float, 2>, MatrixFormat::Dense>::MatrixView(std::array<float, 2>*, int, int, int, int, int, std::array<float, 2>**);
template MatrixView<std::array<double, 2>, MatrixFormat::Dense>::MatrixView(std::array<double, 2>*, int, int, int, int, int, std::array<double, 2>**);

template MatrixView<float, MatrixFormat::Dense>::MatrixView(
    const VectorView<float>&, VectorOrientation);
template MatrixView<double, MatrixFormat::Dense>::MatrixView(
    const VectorView<double>&, VectorOrientation);
template MatrixView<std::complex<float>, MatrixFormat::Dense>::MatrixView(
    const VectorView<std::complex<float>>&, VectorOrientation);
template MatrixView<std::complex<double>, MatrixFormat::Dense>::MatrixView(
    const VectorView<std::complex<double>>&, VectorOrientation);


template MatrixView<float, MatrixFormat::Dense>::MatrixView(
    const Matrix<float, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<double, MatrixFormat::Dense>::MatrixView(
    const Matrix<double, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<std::complex<float>, MatrixFormat::Dense>::MatrixView(
    const Matrix<std::complex<float>, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<std::complex<double>, MatrixFormat::Dense>::MatrixView(
    const Matrix<std::complex<double>, MatrixFormat::Dense>&, int, int, int, int, int);

template MatrixView<std::array<float, 2>, MatrixFormat::Dense>::MatrixView(
    const Matrix<std::array<float, 2>, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<std::array<double, 2>, MatrixFormat::Dense>::MatrixView(
    const Matrix<std::array<double, 2>, MatrixFormat::Dense>&, int, int, int, int, int);

template MatrixView<float, MatrixFormat::Dense>::MatrixView(
    const MatrixView<float, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<double, MatrixFormat::Dense>::MatrixView(
    const MatrixView<double, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<std::complex<float>, MatrixFormat::Dense>::MatrixView(
    const MatrixView<std::complex<float>, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<std::complex<double>, MatrixFormat::Dense>::MatrixView(
    const MatrixView<std::complex<double>, MatrixFormat::Dense>&, int, int, int, int, int);

template MatrixView<std::array<float, 2>, MatrixFormat::Dense>::MatrixView(const MatrixView<std::array<float, 2>, MatrixFormat::Dense>&, int, int, int, int, int);
template MatrixView<std::array<double, 2>, MatrixFormat::Dense>::MatrixView(const MatrixView<std::array<double, 2>, MatrixFormat::Dense>&, int, int, int, int, int);

/* template MatrixView<float, MatrixFormat::Dense>::MatrixView(Span<float>, int, int, int, int, int, int, int);
template MatrixView<double, MatrixFormat::Dense>::MatrixView(Span<double>, int, int, int, int, int, int, int);
template MatrixView<std::complex<float>, MatrixFormat::Dense>::MatrixView(Span<std::complex<float>>, int, int, int, int, int, int, int);
template MatrixView<std::complex<double>, MatrixFormat::Dense>::MatrixView(Span<std::complex<double>>, int, int, int, int, int, int, int); */

// CSR MatrixView constructors instantiations
template MatrixView<float, MatrixFormat::CSR>::MatrixView(float*, int*, int*, int, int, int, int, int, int, float**);
template MatrixView<double, MatrixFormat::CSR>::MatrixView(double*, int*, int*, int, int, int, int, int, int, double**);
template MatrixView<std::complex<float>, MatrixFormat::CSR>::MatrixView(std::complex<float>*, int*, int*, int, int, int, int, int, int, std::complex<float>**);
template MatrixView<std::complex<double>, MatrixFormat::CSR>::MatrixView(std::complex<double>*, int*, int*, int, int, int, int, int, int, std::complex<double>**);

// Dense MatrixView at() method instantiations
template float& MatrixView<float, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int);
template double& MatrixView<double, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int);
template std::complex<float>& MatrixView<std::complex<float>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int);
template std::complex<double>& MatrixView<std::complex<double>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int);

template std::array<float, 2>& MatrixView<std::array<float, 2>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int);
template std::array<double, 2>& MatrixView<std::array<double, 2>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int);

template const float& MatrixView<float, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int) const;
template const double& MatrixView<double, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int) const;
template const std::complex<float>& MatrixView<std::complex<float>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int) const;
template const std::complex<double>& MatrixView<std::complex<double>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int) const;

template const std::array<float, 2>& MatrixView<std::array<float, 2>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int) const;
template const std::array<double, 2>& MatrixView<std::array<double, 2>, MatrixFormat::Dense>::at<MatrixFormat::Dense>(int, int, int) const;


} // namespace batchlas

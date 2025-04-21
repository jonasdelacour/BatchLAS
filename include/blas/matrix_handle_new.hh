#pragma once
#include <memory>
#include <complex>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include "enums.hh"

namespace batchlas {

    class Queue;

    // Forward declarations
    template <typename T, MatrixFormat MType>
    class Matrix;

    template <typename T, MatrixFormat MType>
    class MatrixView;

    // Forward declare the backend handle - implementation will be in src/ folder
    template <typename T, MatrixFormat MType>
    class BackendMatrixHandle;

    // MatrixFormat enum to replace Format enum
    enum class MatrixFormat {
        Dense,
        CSR,    // Compressed Sparse Row
        CSC,    // Compressed Sparse Column
        COO,    // Coordinate
        SELL,   // Sliced ELLPACK
        BSR,    // Blocked Sparse Row
        BLOCKED_ELL // Blocked ELLPACK
    };

    // Matrix class - owning container for matrix data
    template <typename T, MatrixFormat MType>
    class Matrix {
    public:
        // Basic constructors for dense matrix (allocate uninitialized memory)
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Matrix(int rows, int cols, int batch_size = 1);

        // Basic constructors for CSR sparse matrix (allocate uninitialized memory)
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        Matrix(int rows, int cols, int nnz, int batch_size = 1);

        // Constructor from existing data (will copy data)
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Matrix(const T* data, int rows, int cols, int ld, int stride = 0, int batch_size = 1);

        // Constructor from existing data (will copy data)
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        Matrix(const T* data, const int* row_offsets, const int* col_indices, 
               int nnz, int rows, int cols, int matrix_stride = 0, 
               int offset_stride = 0, int batch_size = 1);
               
        // Convenience factory methods for common patterns
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> Identity(int n, int batch_size = 1);
        
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> Random(int rows, int cols, int batch_size = 1, unsigned int seed = 42);
        
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> Zeros(int rows, int cols, int batch_size = 1);
        
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> Ones(int rows, int cols, int batch_size = 1);
        
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> Diagonal(const Span<T>& diag_values, int batch_size = 1);

        // Destructor
        ~Matrix();

        // Prevent copying, allow moving
        Matrix(const Matrix&) = delete;
        Matrix& operator=(const Matrix&) = delete;
        Matrix(Matrix&&) noexcept;
        Matrix& operator=(Matrix&&) noexcept;

        // Create a view
        MatrixView<T, MType> view() const;
        MatrixView<T, MType> view(int rows, int cols, int ld = -1, int stride = -1) const;

        // Methods to initialize backend
        void init(Queue& ctx) const;
        void init_backend();

        // Accessors
        BackendMatrixHandle<T, MType>* operator->();
        BackendMatrixHandle<T, MType>& operator*();

        // Common dimensions and properties
        int rows_, cols_, batch_size_;

        // Data access - provides non-owning view of the data
        Span<T> data() const { return data_.to_span(); }
        
        // Fill the matrix with a specific value
        void fill(T value);
        
        // Deep copy from another matrix or view
        void copy_from(const MatrixView<T, MType>& src);
        
        // Dense matrix specific accessors
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        int ld() const { return ld_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        int stride() const { return stride_; }

        // CSR specific accessors
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        Span<int> row_offsets() const { return row_offsets_.to_span(); }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        Span<int> col_indices() const { return col_indices_.to_span(); }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        int nnz() const { return nnz_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        int matrix_stride() const { return matrix_stride_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        int offset_stride() const { return offset_stride_; }
        
        // Backend handle access
        std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle() const { 
            return backend_handle_; 
        }

    private:
        // Data storage - owned by Matrix
        UnifiedVector<T> data_;
        
        // Dense specific data
        int ld_ = 0;       // Leading dimension (for dense matrices)
        int stride_ = 0;   // Stride between matrices in a batch
        
        // For batched operations - array of pointers to the start of each matrix
        UnifiedVector<T*> data_ptrs_;

        // CSR specific data
        UnifiedVector<int> row_offsets_;  // Row offsets for CSR format 
        UnifiedVector<int> col_indices_;  // Column indices for CSR format
        int nnz_ = 0;              // Number of non-zeros
        int matrix_stride_ = 0;    // Stride between value arrays in a batch
        int offset_stride_ = 0;    // Stride between offset arrays in a batch
        
        // Backend handle - shared with views of this matrix
        std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle_;
    };

    // MatrixView class - non-owning view of a matrix
    template <typename T, MatrixFormat MType>
    class MatrixView {
    public:
        // Constructors for dense matrix view - from raw spans
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView(Span<T> data, int rows, int cols, int ld,
                  int stride = 0, int batch_size = 1);
                  
        // View into a subset of a dense matrix data (submatrix)
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView(Span<T> data, int row_offset, int col_offset, 
                  int rows, int cols, int ld, int stride = 0, int batch_size = 1);

        // Constructors for CSR sparse matrix view
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        MatrixView(Span<T> data, Span<int> row_offsets, Span<int> col_indices, 
                  int nnz, int rows, int cols, int matrix_stride = 0,
                  int offset_stride = 0, int batch_size = 1);

        // Constructors from Matrix objects - view entire matrix
        MatrixView(const Matrix<T, MType>& matrix);
        
        // Constructors from Matrix objects - view subset of matrix
        MatrixView(const Matrix<T, MType>& matrix, int row_offset, int col_offset,
                  int rows, int cols);
        
        // Copy and move
        MatrixView(const MatrixView&) = default;
        MatrixView& operator=(const MatrixView&) = default;
        MatrixView(MatrixView&&) noexcept = default;
        MatrixView& operator=(MatrixView&&) noexcept = default;

        // Destructor
        ~MatrixView() = default;

        // Initialize backend
        void init(Queue& ctx) const;
        void init_backend();

        // Accessors
        BackendMatrixHandle<T, MType>* operator->();
        BackendMatrixHandle<T, MType>& operator*();

        // Access single matrix in batch (returns view for a single matrix)
        MatrixView<T, MType> operator[](int i) const;
        
        // Common data members
        int rows_, cols_, batch_size_;

        // Data access
        Span<T> data() const { return data_; }
        
        // Dense matrix specific
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        int ld() const { return ld_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        int stride() const { return stride_; }

        // CSR specific accessors
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        Span<int> row_offsets() const { return row_offsets_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        Span<int> col_indices() const { return col_indices_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        int nnz() const { return nnz_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        int matrix_stride() const { return matrix_stride_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        int offset_stride() const { return offset_stride_; }

        // Access to individual elements (for dense matrices)
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        T& at(int row, int col, int batch = 0);
        
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        const T& at(int row, int col, int batch = 0) const;
        
        // Backend handle access
        BackendMatrixHandle<T, MType>* get_backend_handle() const { 
            auto ptr = backend_handle_.lock();
            return ptr.get(); 
        }
        
        // Create a new view into a single batch item
        MatrixView<T, MType> batch_item(int batch_index) const;
        
    private:
        // Data storage - non-owning spans
        Span<T> data_;
        
        // Dense specific data
        int ld_ = 0;          // Leading dimension
        int stride_ = 0;      // Stride between matrices in a batch
        
        // For batched operations
        Span<T*> data_ptrs_;  // View into array of matrix pointers for batched operations

        // CSR specific data
        Span<int> row_offsets_;   // Row offsets for CSR format
        Span<int> col_indices_;   // Column indices for CSR format
        int nnz_ = 0;             // Number of non-zeros
        int matrix_stride_ = 0;   // Stride between value arrays in a batch
        int offset_stride_ = 0;   // Stride between offset arrays in a batch
        
        // Backend handle (weak pointer to avoid circular references)
        std::weak_ptr<BackendMatrixHandle<T, MType>> backend_handle_;
    };

    // BackendMatrixHandle class - backend-specific implementation
    template <typename T, MatrixFormat MType>
    class BackendMatrixHandle {
    public:
        virtual ~BackendMatrixHandle();
        // Backend-specific methods are declared here and implemented in src/backends/matrix_handle_impl.cc
    };

    // Factory functions to create backend handles (implemented in src/backends/matrix_handle_impl.cc)
    template <Backend B, typename T, MatrixFormat MType>
    std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const Matrix<T, MType>& matrix, Queue& ctx);

    template <Backend B, typename T, MatrixFormat MType>
    std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const MatrixView<T, MType>& view, Queue& ctx);

    // Utility functions

    // Get batch size
    template <typename T, MatrixFormat MType>
    int get_batch_size(const Matrix<T, MType>& matrix) {
        return matrix.batch_size_;
    }

    template <typename T, MatrixFormat MType>
    int get_batch_size(const MatrixView<T, MType>& matrix_view) {
        return matrix_view.batch_size_;
    }

    // Get data pointer
    template <typename T, MatrixFormat MType>
    T* get_data(const Matrix<T, MType>& matrix) {
        return matrix.data_;
    }

    template <typename T, MatrixFormat MType>
    T* get_data(const MatrixView<T, MType>& matrix_view) {
        return matrix_view.data_;
    }

    // Create a view from raw pointers
    template <typename T, MatrixFormat MType = MatrixFormat::Dense>
    auto create_view(T* data, int rows, int cols, int ld, int stride = 0, int batch_size = 1) {
        return MatrixView<T, MType>(data, rows, cols, ld, Layout::ColMajor, stride, batch_size);
    }

    // Create a view of a submatrix
    template <typename T, MatrixFormat MType>
    auto subview(const Matrix<T, MType>& matrix, int rows, int cols = -1, int ld = -1, int stride = -1) {
        return matrix.view(rows, cols, ld, stride);
    }

    template <typename T, MatrixFormat MType>
    auto subview(const MatrixView<T, MType>& matrix_view, int rows, int cols = -1, int ld = -1, int stride = -1) {
        // Implementation depends on MType
        if constexpr (MType == MatrixFormat::Dense) {
            return MatrixView<T, MType>(
                matrix_view.data_,
                rows, 
                cols > 0 ? cols : matrix_view.cols_, 
                ld > 0 ? ld : matrix_view.ld(),
                matrix_view.layout_,
                stride > 0 ? stride : matrix_view.stride(),
                matrix_view.batch_size_);
        } else if constexpr (MType == MatrixFormat::CSR) {
            // CSR subview implementation
            // ...
        }
    }

    // Vector type definitions - simplified versions of the matrix types
    template <typename T>
    using Vector = Matrix<T, MatrixFormat::Dense>;
    
    template <typename T>
    using VectorView = MatrixView<T, MatrixFormat::Dense>;
    
    template <typename T>
    using SparseVector = Matrix<T, MatrixFormat::CSR>;
    
    template <typename T>
    using SparseVectorView = MatrixView<T, MatrixFormat::CSR>;

    // Helper utility to get effective dimensions accounting for transpose
    template <typename T, MatrixFormat MType>
    std::pair<int, int> get_effective_dims(const MatrixView<T, MType>& mat, Transpose trans) {
        return (trans == Transpose::NoTrans) 
               ? std::make_pair(mat.rows_, mat.cols_) 
               : std::make_pair(mat.cols_, mat.rows_);
    }

} // namespace batchlas

#pragma once
#include <memory>
#include <complex>
#include <iostream> // Added for std::ostream, std::cout
#include <iomanip>  // Added for std::setw, std::scientific, etc.
#include <algorithm> // Added for std::min
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include <blas/enums.hh>

namespace batchlas {
    #ifndef BASETYPE
    #define BASETYPE
    template<typename T>
    struct base_type {
        using type = T;
    };

    template<typename T>
    struct base_type<std::complex<T>> {
        using type = T;
    };

    template<typename T>
    using float_t = typename base_type<T>::type;
    #endif

    // Forward declarations with default template parameters
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class Matrix;

    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class MatrixView;

    // Forward declare the backend handle - implementation will be in src/ folder
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class BackendMatrixHandle;

    // Matrix class - owning container for matrix data
    template <typename T, MatrixFormat MType>
    class Matrix {
    public:
        // Make MatrixView a friend class to allow access to private members
        friend class MatrixView<T, MType>;
        
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

        // Create a triangular matrix with specific values
        template <typename U = T, MatrixFormat M = MType, 
                typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> Triangular(int n, Uplo uplo, T diagonal_value = T(1), 
                                          T non_diagonal_value = T(0.5), int batch_size = 1);
        
        // Convert row-major to column-major format
        template <typename U = T, MatrixFormat M = MType, 
                typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Matrix<T, MType> to_column_major() const;
        
        // Convert to a different matrix format
        template <MatrixFormat NewMType>
        Matrix<T, NewMType> convert_to(const T& zero_threshold = 1e-7) const;

        // Create a copy with data in row-major format from column-major
        template <typename U = T, MatrixFormat M = MType, 
                typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Matrix<T, MType> to_row_major() const;
        
        // Destructor
        ~Matrix();

        // Add deep copy functionality while allowing moving
        Matrix(const Matrix& other);
        Matrix& operator=(const Matrix& other);
        Matrix(Matrix&&) noexcept;
        Matrix& operator=(Matrix&&) noexcept;
        
        // Create a deep copy
        Matrix<T, MType> clone() const {
            Matrix<T, MType> result(rows_, cols_, batch_size_);
            
            // Copy main data
            std::copy(data_.begin(), data_.end(), result.data_.begin());
            
            // Copy CSR format data if applicable
            if constexpr (MType == MatrixFormat::CSR) {
                std::copy(row_offsets_.begin(), row_offsets_.end(), result.row_offsets_.begin());
                std::copy(col_indices_.begin(), col_indices_.end(), result.col_indices_.begin());
                result.nnz_ = nnz_;
                result.matrix_stride_ = matrix_stride_;
                result.offset_stride_ = offset_stride_;
            }
            
            // Copy dense format specific data
            result.ld_ = ld_;
            result.stride_ = stride_;
            
            return result;
        }

        // Create a view
        MatrixView<T, MType> view() const;
        MatrixView<T, MType> view(int rows, int cols, int ld = -1, int stride = -1) const;


        // Methods to initialize backend
        void init() const;
        // Initialize data pointers for batched operations
        Span<T*> data_ptrs(Queue& ctx) const {
            init_data_ptr_array(ctx);
            return data_ptrs_.to_span();
        }

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
        
        // Print the matrix content
        void print(std::ostream& os = std::cout, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            this->view().print(os, max_rows_to_print, max_cols_to_print, max_elements_to_print_csr);
        }

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
        void init_data_ptr_array(Queue& ctx) const;

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
        // Make mutable so it can be modified in const methods
        mutable std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle_;
    };

    // MatrixView class - non-owning view of a matrix
    template <typename T, MatrixFormat MType>
    class MatrixView {
    public:
        // Constructors for dense matrix view - from raw spans
        // data_ptrs: Optional array of pointers to the start of each matrix in a batch
        // This enables direct use of the pointers in batched operations
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView(T* data, int rows, int cols, int ld,
                  int stride = 0, int batch_size = 1, T** data_ptrs = nullptr);

        // Constructors for CSR sparse matrix view
        // data_ptrs: Optional array of pointers to the start of each matrix's values in a batch
        // This enables direct use of the pointers in batched operations
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        MatrixView(T* data, int* row_offsets, int* col_indices, 
                  int nnz, int rows, int cols, int matrix_stride = 0,
                  int offset_stride = 0, int batch_size = 1, T** data_ptrs = nullptr);

        // Constructors from Matrix objects - view entire matrix
        MatrixView(const Matrix<T, MType>& matrix);
        
        // Constructors from Matrix objects - view subset of matrix
        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView(const Matrix<T, MType>& matrix,
                  int rows = -1, int cols = -1, int ld = -1, int stride = -1, int batch_size = -1);

        template <typename U = T, MatrixFormat M = MType,
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView(const MatrixView<T, MType>& matrix_view,
                  int rows = -1, int cols = -1, int ld = -1, int stride = -1, int batch_size = -1);
        
        MatrixView() = default;

        // Copy and move
        MatrixView(const MatrixView&) = default;
        MatrixView& operator=(const MatrixView&) = default;
        MatrixView(MatrixView&&) noexcept = default;
        MatrixView& operator=(MatrixView&&) noexcept = default;

        // Destructor
        ~MatrixView() = default;

        // Initialize backend - need to make backend_handle_ mutable
        void init() const;

        // Accessors
        BackendMatrixHandle<T, MType>* operator->() const;
        BackendMatrixHandle<T, MType>& operator*() const;

        // Access single matrix in batch (returns view for a single matrix)
        MatrixView<T, MType> operator[](int i) const;
        
        // Common data members
        int rows_, cols_, batch_size_;

        // Data access
        Span<T> data() const { return data_; }
        T* data_ptr() const { return data_.data(); }
        

        int batch_size() const { return batch_size_; }
        int rows() const { return rows_; }
        int cols() const { return cols_; }
        
        // Dense matrix specific
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        int ld() const { return ld_; }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        int stride() const { return stride_; }

        // Increment for dense vectors (assuming contiguous)
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        int inc() const { return 1; } // Assuming contiguous elements for vector views

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

        // Access pointer array for batched operations
        Span<T*> data_ptrs(Queue& ctx) const { 
            init_data_ptr_array(ctx);
            return data_ptrs_; 
        }

        // Access to individual elements (for dense matrices)
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        T& at(int row, int col, int batch = 0);
        
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        const T& at(int row, int col, int batch = 0) const;
        
        // Create a new view into a single batch item
        MatrixView<T, MType> batch_item(int batch_index) const;
        
        // Print the matrix view content to a stream
        // Returns the ostream to allow chaining and use in operator<<
        std::ostream& stream_formatted_to(std::ostream& os, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            std::ios_base::fmtflags original_flags = os.flags();
            os << std::scientific << std::setprecision(4);

            for (int b_idx = 0; b_idx < batch_size_; ++b_idx) {
                if (batch_size_ > 1) {
                    os << "Batch " << b_idx << ":\n";
                }
                // Assuming batch_item() returns a view with batch_size_ = 1 and correct data pointers
                MatrixView<T, MType> current_item_view = this->batch_item(b_idx);

                if constexpr (MType == MatrixFormat::Dense) {
                    for (int r = 0; r < std::min(current_item_view.rows_, max_rows_to_print); ++r) {
                        os << "  ";
                        for (int c = 0; c < std::min(current_item_view.cols_, max_cols_to_print); ++c) {
                            os << std::setw(13) << current_item_view.at(r, c, 0); // batch is 0 for current_item_view
                        }
                        if (current_item_view.cols_ > max_cols_to_print) {
                            os << " ...";
                        }
                        os << "\n";
                    }
                    if (current_item_view.rows_ > max_rows_to_print) {
                        os << "  ...\n";
                    }
                } else if constexpr (MType == MatrixFormat::CSR) {
                    os << "CSR Matrix (rows: " << current_item_view.rows_ 
                       << ", cols: " << current_item_view.cols_ 
                       << ", nnz: " << current_item_view.nnz_ << ")\n";

                    os << "  Values:      [";
                    for (int i = 0; i < std::min(current_item_view.nnz_, max_elements_to_print_csr); ++i) {
                        os << std::setw(13) << current_item_view.data()[i] << (i == std::min(current_item_view.nnz_, max_elements_to_print_csr) - 1 ? "" : ", ");
                    }
                    if (current_item_view.nnz_ > max_elements_to_print_csr) os << " ...";
                    os << " ]\n";

                    os << "  Row Offsets: [";
                    // CSR row_offsets has size rows + 1
                    int num_row_offsets = current_item_view.rows_ + 1;
                    for (int i = 0; i < std::min(num_row_offsets, max_elements_to_print_csr); ++i) {
                        os << std::setw(6) << current_item_view.row_offsets()[i] << (i == std::min(num_row_offsets, max_elements_to_print_csr) - 1 ? "" : ", ");
                    }
                    if (num_row_offsets > max_elements_to_print_csr) os << " ...";
                    os << " ]\n";

                    os << "  Col Indices: [";
                    for (int i = 0; i < std::min(current_item_view.nnz_, max_elements_to_print_csr); ++i) {
                        os << std::setw(6) << current_item_view.col_indices()[i] << (i == std::min(current_item_view.nnz_, max_elements_to_print_csr) - 1 ? "" : ", ");
                    }
                    if (current_item_view.nnz_ > max_elements_to_print_csr) os << " ...";
                    os << " ]\n";
                }
                if (b_idx < batch_size_ - 1) {
                    os << "\n"; // Separator between batches
                }
            }
            os.flags(original_flags); // Reset stream flags
            return os;
        }

        // Convenience print function
        void print(std::ostream& os = std::cout, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            stream_formatted_to(os, max_rows_to_print, max_cols_to_print, max_elements_to_print_csr);
        }

    private:
        void init_data_ptr_array(Queue& ctx) const;

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
        
        // Backend handle
        // Make mutable so it can be modified in const methods
        mutable std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle_;
    };

    // Factory functions to create backend handles (implemented in src/backends/matrix_handle_impl.cc)
    template <typename T, MatrixFormat MType>
    std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const Matrix<T, MType>& matrix);

    template <typename T, MatrixFormat MType>
    std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const MatrixView<T, MType>& view);

    // Vector type definitions - simplified versions of the matrix types
    template <typename T = float>
    class BackendVectorHandle; // Forward declaration with default parameter

    //Forward declare VectorView with default parameter
    template <typename T = float>
    struct VectorView;

    // Vector class with batched support and stride between vectors
    template <typename T>
    struct Vector {
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using const_reference = const T&;

        Vector() : data_(), size_(0), inc_(1), stride_(0), batch_size_(1) {}
        Vector(int size, int inc = 1, int stride = 0, int batch_size = 1)
            : data_(stride > 0 ? stride * batch_size : size * batch_size),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size), batch_size_(batch_size) {}
        Vector(int size, T value, int inc = 1, int stride = 0, int batch_size = 1)
            : data_(stride > 0 ? stride * batch_size : size * batch_size, value),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size), batch_size_(batch_size) {}

        // Convenience vectors
        static Vector<T> zeros(int size, int batch_size = 1, int stride = 0) {
            return Vector<T>(size, T(0), batch_size, stride);
        }
        static Vector<T> ones(int size, int batch_size = 1, int stride = 0) {
            return Vector<T>(size, T(1), batch_size, stride);
        }
        static Vector<T> random(int size, int batch_size = 1, int stride = 0) {
            Vector<T> vec(size, batch_size, stride);
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < size; ++i) {
                    vec.at(i, b) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
                }
            }
            return vec;
        }
        static Vector<T> standard_basis(int size, int index, int batch_size = 1, int stride = 0) {
            Vector<T> vec(size, T(0), batch_size, stride);
            for (int b = 0; b < batch_size; ++b) {
                vec.at(index, b) = T(1);
            }
            return vec;
        }

        Span<T> data() { return data_.to_span(); }
        Span<const T> data() const { return data_.to_span(); }
        T* data_ptr() { return data_.data(); }
        const T* data_ptr() const { return data_.data(); }
        int size() const { return size_; }
        int inc() const { return inc_; }
        int stride() const { return stride_; }
        int batch_size() const { return batch_size_; }

        // Access element at (i, batch)
        T& at(int i, int batch = 0) { return data_[i * inc_ + batch * stride_]; }
        const T& at(int i, int batch = 0) const { return data_[i * inc_ + batch * stride_]; }

        // Flat indexing (for backward compatibility)
        T& operator[](int i) { return data_[i]; }
        const T& operator[](int i) const { return data_[i]; }

        // Get a view of a single batch
        VectorView<T> batch_item(int batch_index) const {
            return VectorView<T>(Span<T>(data_.data() + batch_index * stride_, size_, inc_), size_, 1, inc_, stride_);
        }

        BackendVectorHandle<T>* operator->();
        BackendVectorHandle<T>& operator*();
        const BackendVectorHandle<T>* operator->() const;
        const BackendVectorHandle<T>& operator*()  const;

    private:
        UnifiedVector<T> data_;
        int size_ = 0;
        int inc_ = 1;
        int stride_ = 0;
        int batch_size_ = 1;

        std::shared_ptr<BackendVectorHandle<T>> backend_handle_;
    };

    // VectorView class - non-owning view of a (possibly batched) vector with stride
    template <typename T>
    class VectorView {
    public:
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using const_reference = const T&;

        VectorView() : data_(), size_(0), inc_(1), stride_(0), batch_size_(1) {}
        VectorView(Span<T> data, int size, int inc = 1, int stride = 0, int batch_size = 1)
            : data_(data), size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {}
        VectorView(UnifiedVector<T>& data, int size, int inc = 1, int stride = 0, int batch_size = 1)
            : data_(data.to_span()), size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {}
        VectorView(T* data, int size, int inc = 1, int stride = 0, int batch_size = 1)
            : data_(data, (stride > 0 ? stride * batch_size : size * inc * batch_size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {}

        // Construct from Vector
        VectorView(const Vector<T>& vec)
            : data_(vec.data()), size_(vec.size()), inc_(vec.inc()), stride_(vec.stride()), batch_size_(vec.batch_size()), backend_handle_(vec.backend_handle_) {}

        // Copy and move
        VectorView(const VectorView<T>&) = default;
        VectorView& operator=(const VectorView<T>&) = default;
        VectorView(VectorView<T>&&) noexcept = default;
        VectorView& operator=(VectorView<T>&&) noexcept = default;

        // Data access
        Span<T> data() const { return data_; }
        T* data_ptr() const { return data_.data(); }
        int size() const { return size_; }
        int inc() const { return inc_; }
        int stride() const { return stride_; }
        int batch_size() const { return batch_size_; }

        // Access element at (i, batch)
        T& at(int i, int batch = 0) { return data_[i * inc_ + batch * stride_]; }
        const T& at(int i, int batch = 0) const { return data_[i * inc_ + batch * stride_]; }

        // Flat indexing (for backward compatibility)
        T& operator[](int i) { return data_[i]; }
        const T& operator[](int i) const { return data_[i]; }

        // Subview (single batch)
        VectorView<T> batch_item(int batch_index) const {
            return VectorView<T>(Span<T>(data_.data() + batch_index * stride_, size_, inc_), size_, 1, inc_, stride_);
        }

        // Subview (range within a batch)
        VectorView<T> subview(int offset, int count, int batch = 0) const {
            return VectorView<T>(Span<T>(data_.data() + batch * stride_ + offset * inc_, count, inc_), count, 1, inc_, stride_);
        }

        BackendVectorHandle<T>* operator->();
        BackendVectorHandle<T>& operator*();
        const BackendVectorHandle<T>* operator->() const;
        const BackendVectorHandle<T>& operator*() const;

    private:
        Span<T> data_;
        int size_ = 0;
        int inc_ = 1;
        int stride_ = 0;
        int batch_size_ = 1;
        std::weak_ptr<BackendVectorHandle<T>> backend_handle_;
    };

    // Helper utility to get effective dimensions accounting for transpose
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    std::pair<int, int> get_effective_dims(const MatrixView<T, MType>& mat, Transpose trans) {
        return (trans == Transpose::NoTrans) 
               ? std::make_pair(mat.rows_, mat.cols_) 
               : std::make_pair(mat.cols_, mat.rows_);
    }

    // Ostream operators for Matrix and MatrixView
    template <typename T, MatrixFormat MType>
    std::ostream& operator<<(std::ostream& os, const MatrixView<T, MType>& view) {
        return view.stream_formatted_to(os); // Uses default arguments from stream_formatted_to
    }

    template <typename T, MatrixFormat MType>
    std::ostream& operator<<(std::ostream& os, const Matrix<T, MType>& matrix) {
        os << matrix.view(); // Leverages MatrixView's operator<<
        return os;
    }

} // namespace batchlas

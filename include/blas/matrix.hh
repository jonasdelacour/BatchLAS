#pragma once
#include <memory>
#include <complex>
#include <type_traits>
#include <iostream> // Added for std::ostream, std::cout
#include <iomanip>  // Added for std::setw, std::scientific, etc.
#include <algorithm> // Added for std::min
#include <tuple>
#include <array>    // Added for std::array element types
#include <sstream>  // Added for temporary string formatting of non-streamable types
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include <blas/enums.hh>

namespace batchlas {
    // Forward declarations with default template parameters
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class Matrix;

    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class MatrixView;

    // Forward declare the backend handle - implementation will be in src/ folder
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class BackendMatrixHandle;

    struct SliceEnd {};
    struct Slice { //Default Slice selects entire matrix
        int64_t start = std::numeric_limits<int64_t>::min();
        int64_t end = std::numeric_limits<int64_t>::max();
        Slice(int64_t start, SliceEnd) : start(start), end(std::numeric_limits<int64_t>::max()) {}
        Slice(int64_t start, int64_t end) : start(start), end(end) {}
        Slice(int64_t start) : Slice(start, SliceEnd()) {}
        Slice() = default;
    };

    // ------------------------------------------------------------------
    // KernelMatrixView<T, MType>: device-passing trivial view mirroring
    // a subset of MatrixView functionality. It is template-specialized by
    // MatrixFormat so we avoid a runtime tag and allow the compiler to
    // optimize format-specific paths. Slicing operators mimic those of
    // MatrixView but (like MatrixView) are only defined for dense format.
    // CSR variant intentionally omits slicing to match existing MatrixView.
    // ------------------------------------------------------------------
    template <typename T, MatrixFormat MType>
    struct KernelMatrixView {
        // Common fields
        T*   data = nullptr;
        int  rows = 0;
        int  cols = 0;
        int  batch_size = 1;

        // Dense specific
        int  ld = 0;      // leading dimension
        int  stride = 0;  // batch stride (elements)

        // CSR specific (only meaningful if MType == MatrixFormat::CSR)
        int* row_offsets = nullptr;  // length rows+1 per batch
        int* col_indices = nullptr;  // length nnz per batch
        int  nnz = 0;
        int  matrix_stride = 0;      // stride between value arrays
        int  offset_stride = 0;      // stride between offset arrays

        // Element access (Dense only)
        template <MatrixFormat MF = MType, typename std::enable_if<MF == MatrixFormat::Dense, int>::type = 0>
        inline T& operator()(int i, int j, int b = 0) const { return data[b * stride + j * ld + i]; }

        // CSR coefficient lookup (returns zero if absent)
        template <MatrixFormat MF = MType, typename std::enable_if<MF == MatrixFormat::CSR, int>::type = 0>
        inline T get(int i, int j, int b = 0) const {
            const int ro_base = b * offset_stride;
            const int val_base = b * matrix_stride;
            int rs = row_offsets[ro_base + i];
            int re = row_offsets[ro_base + i + 1];
            for (int p = rs; p < re; ++p) {
                if (col_indices[val_base + p] == j) return data[val_base + p];
            }
            return T(0);
        }

        // Batch item extraction - works for both formats
        inline KernelMatrixView batch_item(int b) const {
            KernelMatrixView out = *this;
            if (b < 0 || b >= batch_size) { out.rows = out.cols = 0; return out; }
            if constexpr (MType == MatrixFormat::Dense) {
                out.data += b * stride;
            } else if constexpr (MType == MatrixFormat::CSR) {
                out.data        += b * matrix_stride;
                out.row_offsets += b * offset_stride;
                out.col_indices += b * matrix_stride; // assumes same stride
            }
            out.batch_size = 1;
            return out;
        }

        // Dense slicing operators (host-side; mirror MatrixView logic)
        template <MatrixFormat MF = MType, typename std::enable_if<MF == MatrixFormat::Dense, int>::type = 0>
        KernelMatrixView operator()(Slice rows_slice, Slice cols_slice = {}) const;
        template <MatrixFormat MF = MType, typename std::enable_if<MF == MatrixFormat::Dense, int>::type = 0>
        KernelMatrixView operator()(Slice rows_slice) const { return (*this)(rows_slice, {}); }

        KernelMatrixView() = default;
        KernelMatrixView(const KernelMatrixView&) = default;
        KernelMatrixView& operator=(const KernelMatrixView&) = default;
        KernelMatrixView(KernelMatrixView&&) = default;
        KernelMatrixView& operator=(KernelMatrixView&&) = default;

        KernelMatrixView(T* data, int rows, int cols, int ld = 0, int stride = 0, int batch_size = 1)
            : data(data), rows(rows), cols(cols), batch_size(batch_size),
              ld(ld > 0 ? ld : rows), stride(stride > 0 ? stride : ld * cols) {}
    };

    // (Slice already defined earlier)

    // --- Slice utilities (de-duplication) ---------------------------------
    namespace detail {
        inline std::pair<int64_t,int64_t> normalize_slice_component(Slice s, int64_t dim) {
            int64_t len;
            if (s.start == std::numeric_limits<int64_t>::min() && s.end == std::numeric_limits<int64_t>::max()) { s.start = 0; len = dim; }
            else if (s.end == std::numeric_limits<int64_t>::max()) { s.start = s.start < 0 ? dim + s.start : s.start; len = dim - s.start; }
            else { if (s.start < 0) s.start = dim + s.start; if (s.end < 0) s.end = dim + s.end; len = s.end - s.start; }
            return {s.start, len};
        }

        template <typename T>
        inline void apply_dense_slice_pointer_arithmetic(T*& base, int ld, int64_t row_start, int64_t col_start) {
            base += col_start * ld + row_start; // column-major offset
        }

        
        template <typename T>
        T convert_to_fill_value(int64_t value) {
            if constexpr (std::is_floating_point_v<T>) {
                return static_cast<T>(value);
            } else if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(value);
            } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                                    std::is_same_v<T, std::complex<double>> ||
                                    std::is_same_v<T, std::complex<long double>>) {
                using Real = typename T::value_type;
                return T(static_cast<Real>(value));
            } else if constexpr (
                // Detect std::array<T, N>
                std::is_class_v<T> &&
                std::is_same_v<T, std::array<typename T::value_type,
                                                std::tuple_size<T>::value>>
            ) {
                T result{};
                for (auto &elem : result) {
                    using Elem = typename T::value_type;
                    elem = convert_to_fill_value<Elem>(value);
                }
                return result;
            } else {
                return T{};
            }
        }

        // Detection idiom for streamability
        template <typename U, typename = void>
        struct is_streamable : std::false_type {};
        template <typename U>
        struct is_streamable<U, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const U&>())>> : std::true_type {};

        // Trait to detect std::array
        template <typename U>
        struct is_std_array : std::false_type {};
        template <typename V, std::size_t N>
        struct is_std_array<std::array<V, N>> : std::true_type {};

        // Generic element printer (no width handling)
        template <typename U>
        inline void print_value(std::ostream& os, const U& value) {
            if constexpr (is_streamable<U>::value) {
                os << value;
            } else if constexpr (is_std_array<U>::value) {
                os << '[';
                for (std::size_t i = 0; i < value.size(); ++i) {
                    if (i) os << ',';
                    os << value[i];
                }
                os << ']';
            } else {
                os << "{?}"; // Fallback for unknown, non-streamable types
            }
        }

        // Width-aware printer used in formatted matrix printing
        template <typename U>
        inline void print_with_width(std::ostream& os, const U& value, int width) {
            if constexpr (is_streamable<U>::value) {
                os << std::setw(width) << value;
            } else {
                std::ostringstream tmp;
                print_value(tmp, value);
                const std::string s = tmp.str();
                if ((int)s.size() < width) {
                    os << std::setw(width) << s;
                } else {
                    // If the representation is wider than the field, print as-is to retain info
                    os << s;
                }
            }
        }
    }

    // Implement dense slicing operator outside struct for clarity using helpers
    template <typename T, MatrixFormat MType>
    template <MatrixFormat MF, typename std::enable_if<MF == MatrixFormat::Dense, int>::type>
    KernelMatrixView<T, MType> KernelMatrixView<T, MType>::operator()(Slice rows_slice, Slice cols_slice) const {
        KernelMatrixView<T, MType> out = *this;
        auto [r_start, r_len] = detail::normalize_slice_component(rows_slice, rows);
        auto [c_start, c_len] = detail::normalize_slice_component(cols_slice, cols);
        if (r_len <= 0 || c_len <= 0) {
            throw std::invalid_argument("Invalid slice dimensions in KernelMatrixView");
        }
        detail::apply_dense_slice_pointer_arithmetic(out.data, ld, r_start, c_start);
        out.rows = static_cast<int>(r_len);
        out.cols = static_cast<int>(c_len);
        return out;
    }

    // Static asserts for dense and CSR instantiations
    static_assert(std::is_trivially_copyable_v<KernelMatrixView<float, MatrixFormat::Dense>>, "KernelMatrixView Dense must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<KernelMatrixView<float, MatrixFormat::CSR>>,   "KernelMatrixView CSR must be trivially copyable");

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
        static Matrix<T, MType> Random(int rows, int cols, bool hermitian = false, int batch_size = 1, unsigned int seed = 42);

        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> RandomTriangular(int n, Uplo uplo, Diag diag = Diag::NonUnit, int batch_size = 1, unsigned int seed = 42) {
            auto result = Matrix<T, MType>(n, n, batch_size);
            result.view().fill_triangular_random(uplo, diag, seed).wait();
            return result;
        }
        
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
        
        template <typename U = T, MatrixFormat M = MType, 
                typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static Matrix<T, MType> TriDiagToeplitz(int n, T diag = T(1), 
                                                T sub_diag = T(-0.5), T super_diag = T(0.5), int batch_size = 1);
        
        // Convert row-major to column-major format
        template <typename U = T, MatrixFormat M = MType, 
                typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Matrix<T, MType> to_column_major() const;
        
        // Convert to a different matrix format
        template <MatrixFormat NewMType>
        Matrix<T, NewMType> convert_to(const float_t<T>& zero_threshold = 1e-7) const;

        // Create a copy with data in row-major format from column-major
        template <typename U = T, MatrixFormat M = MType, 
                typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Matrix<T, MType> to_row_major() const;
        
        // Destructor
        ~Matrix();

        // Add deep copy functionality while allowing moving
        Matrix(const Matrix& other) = default;
        Matrix& operator=(const Matrix& other) = default;
        Matrix(Matrix&&) noexcept = default;
        Matrix& operator=(Matrix&&) noexcept = default;

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

        // Dense-only slicing operators producing MatrixView (non-owning)
        template <MatrixFormat M = MType, typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView<T, MType> operator()(Slice rows, Slice cols) const {
            auto [r_start, r_len] = detail::normalize_slice_component(rows, rows_);
            auto [c_start, c_len] = detail::normalize_slice_component(cols, cols_);
            if (r_len <= 0 || c_len <= 0) {
                throw std::invalid_argument("Invalid slice dimensions on Matrix: " + std::to_string(r_len) + "x" + std::to_string(c_len));
            }
            auto offset = c_start * ld_ + r_start;
            return MatrixView<T, MType>(data_.data() + offset, static_cast<int>(r_len), static_cast<int>(c_len), ld_, stride_, batch_size_, nullptr);
        }
        template <MatrixFormat M = MType, typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView<T, MType> operator()(Slice rows) const { return (*this)(rows, {}); }


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
        void fill(T value) {this->view().fill(value).wait();}
        
        // Deep copy from another matrix or view
        void copy_from(const MatrixView<T, MType>& src);
        
        // Print the matrix content
        void print(std::ostream& os = std::cout, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            this->view().print(os, max_rows_to_print, max_cols_to_print, max_elements_to_print_csr);
        }

        int rows() const { return rows_; }
        int cols() const { return cols_; }
        int batch_size() const { return batch_size_; }

        // Build a KernelMatrixView (device POD) for this owning Matrix
        KernelMatrixView<T, MType> kernel_view() const noexcept {
            KernelMatrixView<T, MType> kv;
            kv.data       = const_cast<T*>(data_.data());
            kv.rows       = rows_;
            kv.cols       = cols_;
            kv.ld         = ld_;
            kv.stride     = stride_;
            kv.batch_size = batch_size_;
            if constexpr (MType == MatrixFormat::CSR) {
                kv.row_offsets   = const_cast<int*>(row_offsets_.data());
                kv.col_indices   = const_cast<int*>(col_indices_.data());
                kv.nnz           = nnz_;
                kv.matrix_stride = matrix_stride_;
                kv.offset_stride = offset_stride_;
            }
            return kv;
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
        MatrixView(T* data, int rows, int cols, int ld = 0,
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

        template <typename U = T, MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        static MatrixView<T, MType> deep_copy(const MatrixView<T, MType>& other, //Matrix view to copy from
                                                T* data, //storage for the new view
                                                T** data_ptrs = nullptr //optional array of pointers to the start of each matrix in a batch
                                                );
        
        template <typename U = T, MatrixFormat M = MType,
                    typename std::enable_if<M == MatrixFormat::CSR, int>::type = 0>
        static MatrixView<T, MType> deep_copy(const MatrixView<T, MType>& other, //Matrix view to copy from
                                                T* data, int* row_offsets, int* col_indices, //storage for the new view
                                                T** data_ptrs = nullptr //optional array of pointers to the start of each matrix in a batch
                                                );
        
        static Event copy(Queue& ctx, const MatrixView<T, MType>& dest, const MatrixView<T, MType>& src);

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

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        T& operator()(int row, int col, int batch = 0) {
            return at(row, col, batch);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        const T& operator()(int row, int col, int batch = 0) const {
            return at(row, col, batch);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView<T, MType> operator()(Slice rows, Slice cols) const {
            auto [r_start, r_len] = detail::normalize_slice_component(rows, rows_);
            auto [c_start, c_len] = detail::normalize_slice_component(cols, cols_);
            if (r_len <= 0 || c_len <= 0) {
                throw std::invalid_argument("Invalid slice dimensions: " + std::to_string(r_len) + "x" + std::to_string(c_len));
            }
            auto offset = c_start * ld_ + r_start;
            return MatrixView<T, MType>(data_ptr() + offset, static_cast<int>(r_len), static_cast<int>(c_len), ld_, stride_, batch_size_, data_ptrs_.data());
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        MatrixView<T, MType> operator()(Slice rows) const {
            return (*this)(rows, {});
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event symmetrize(const Queue& ctx, Uplo uplo) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event symmetrize(Uplo uplo) const {
            return symmetrize(Queue(), uplo);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event hermitize(const Queue& ctx, Uplo uplo) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event hermitize(Uplo uplo) const {
            return hermitize(Queue(), uplo);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event transpose(const Queue& ctx, bool conjugate = false) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event transpose(bool conjugate = false) const {
            return transpose(Queue(), conjugate);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event triangularize(const Queue& ctx, Uplo uplo, Diag diag) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event triangularize(Uplo uplo, Diag diag) const {
            return triangularize(Queue(), uplo, diag);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_random(const Queue& ctx, bool hermitian = false, unsigned int seed = 42) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_random(bool hermitian = false, unsigned int seed = 42) const {
            return fill_random(Queue(), hermitian, seed);
        }

        Event fill(const Queue& ctx, T value) const;

        Event fill(T value) const {
            return fill(Queue(), value);
        }

        Event fill_zeros(const Queue& ctx) const {
            return fill(ctx, detail::convert_to_fill_value<T>(0));
        }

        Event fill_zeros() const {
            return fill_zeros(Queue());
        }

        Event fill_ones(const Queue& ctx) const {
            return fill(ctx, detail::convert_to_fill_value<T>(1));
        }

        Event fill_ones() const {
            return fill_ones(Queue());
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_identity(const Queue& ctx, T value = T(1)) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_diagonal(const Queue& ctx, const Span<T>& diag_values, int64_t k = 0) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_diagonal(const Span<T>& diag_values, int64_t k = 0) const {
            return fill_diagonal(Queue(), diag_values, k);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_diagonal(const Queue& ctx, const T& value) const;
        
        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_diagonal(const T& value) const {
            return fill_diagonal(Queue(), value);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_triangular(const Queue& ctx, Uplo uplo, T diagonal_value = T(1), 
                              T non_diagonal_value = T(0.5)) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_triangular(Uplo uplo, T diagonal_value = T(1), 
                              T non_diagonal_value = T(0.5)) const {
            return fill_triangular(Queue(), uplo, diagonal_value, non_diagonal_value);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_tridiag(const Queue& ctx, VectorView<T> sub_diag, 
                            VectorView<T> diag, VectorView<T> super_diag) const {
                                fill_diagonal(ctx, diag);
                                fill_diagonal(ctx, sub_diag, -1);
                                return fill_diagonal(ctx, super_diag, 1);
                            }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_tridiag_toeplitz(const Queue& ctx, T diag = T(1), 
                                    T sub_diag = T(-0.5), T super_diag = T(0.5)) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_tridiag_toeplitz(T diag = T(1), 
                                    T sub_diag = T(-0.5), T super_diag = T(0.5)) const {
            return fill_tridiag_toeplitz(Queue(), diag, sub_diag, super_diag);
        }

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_triangular_random(const Queue& ctx, Uplo uplo, 
                                     Diag diag = Diag::Unit, 
                                     unsigned int seed = 42) const;

        template <MatrixFormat M = MType, 
                  typename std::enable_if<M == MatrixFormat::Dense, int>::type = 0>
        Event fill_triangular_random(Uplo uplo, 
                                     Diag diag = Diag::Unit,
                                     unsigned int seed = 42) const {
            return fill_triangular_random(Queue(), uplo, diag, seed);
        }


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
                            detail::print_with_width(os, current_item_view.at(r, c, 0), 13); // Supports non-streamable element types
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

        // Build a KernelMatrixView (device POD) from this non-owning view
        KernelMatrixView<T, MType> kernel_view() const noexcept {
            KernelMatrixView<T, MType> kv;
            kv.data       = const_cast<T*>(data_.data());
            kv.rows       = rows_;
            kv.cols       = cols_;
            kv.ld         = ld_;
            kv.stride     = stride_;
            kv.batch_size = batch_size_;
            if constexpr (MType == MatrixFormat::CSR) {
                kv.row_offsets   = const_cast<int*>(row_offsets_.data());
                kv.col_indices   = const_cast<int*>(col_indices_.data());
                kv.nnz           = nnz_;
                kv.matrix_stride = matrix_stride_;
                kv.offset_stride = offset_stride_;
            }
            return kv;
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
            : data_(std::max(std::max(stride * batch_size, size * batch_size), inc * size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size), batch_size_(batch_size) {}
        Vector(int size, T value, int inc = 1, int stride = 0, int batch_size = 1)
            : data_(std::max(std::max(stride * batch_size, size * batch_size), inc * size), value),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size), batch_size_(batch_size) {}

        // Convenience vectors
        static Vector<T> zeros(int size, int batch_size = 1, int stride = 0, int inc = 1) {
            return Vector<T>(size, T(0), inc, stride, batch_size);
        }
        static Vector<T> ones(int size, int batch_size = 1, int stride = 0, int inc = 1) {
            return Vector<T>(size, T(1), inc, stride, batch_size);
        }
        static Vector<T> random(int size, int batch_size = 1, int stride = 0, int inc = 1) {
            Vector<T> vec(size, inc, stride, batch_size);

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < size; ++i) {
                    vec(i, b) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
                }
            }
            return vec;
        }
        static Vector<T> standard_basis(int size, int index, int batch_size = 1, int stride = 0) {
            Vector<T> vec(size, T(0), batch_size, stride);
            for (int b = 0; b < batch_size; ++b) {
                vec(index, b) = T(1);
            }
            return vec;
        }

        Span<T> data() const { return data_.to_span(); }
        T* data_ptr() const { return data_.data(); }
        int size() const { return size_; }
        int inc() const { return inc_; }
        int stride() const { return stride_; }
        int batch_size() const { return batch_size_; }

        // Access element at (i, batch)
        T& at(int i, int batch = 0) { return data_[i * inc_ + batch * stride_]; }
        //const T& at(int i, int batch = 0) const { return data_[i * inc_ + batch * stride_]; }

        // Flat indexing (for backward compatibility)
        T& operator[](int i) { return data_[i]; }

        T& operator()(int i, int batch = 0) { return at(i, batch); }

        // Get a view of a single batch
        VectorView<T> batch_item(int batch_index) const {
            return VectorView<T>(Span<T>(data_.data() + batch_index * stride_, stride_*batch_size_), size_, 1, inc_, stride_);
        }

        operator VectorView<T>() const {
            return VectorView<T>(data(), size_, inc_, stride_, batch_size_);
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

        //std::shared_ptr<BackendVectorHandle<T>> backend_handle_;
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
            : data_(vec.data()), size_(vec.size()), inc_(vec.inc()), stride_(vec.stride()), batch_size_(vec.batch_size()) {}

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
        T& at(int i, int batch = 0) const { return data_[i * inc_ + batch * stride_]; }

        T& operator[](int i) const { return data_[i * inc_]; }

        T& operator()(int i, int batch = 0) const {return at(i, batch);}

        // Subview (single batch)
        VectorView<T> batch_item(int batch_index) const {
            return VectorView<T>(Span<T>(data_.data() + batch_index * stride_, size_), size_, inc_, stride_, 1);
        }

        // Subview (range within a batch)
        VectorView<T> subview(int offset, int count, int batch = 0) const {
            return VectorView<T>(Span<T>(data_.data() + batch * stride_ + offset * inc_, count, inc_), count, 1, inc_, stride_);
        }

        VectorView<T> operator()(Slice slice) const {
            // Create a new view based on the slice
            int64_t n;
            if (slice.start == std::numeric_limits<int64_t>::min() && slice.end == std::numeric_limits<int64_t>::max()) {
                n = size_;
            } else if (slice.end == std::numeric_limits<int64_t>::max()) {
                slice.start = slice.start < 0 ? size_ + slice.start : slice.start;
                n = size_ - slice.start;
            } else {
                if (slice.start < 0) slice.start = size_ + slice.start;
                if (slice.end < 0) slice.end = size_ + slice.end;
                n = slice.end - slice.start;
            }
            if (n <= 0) {
                throw std::invalid_argument("Invalid slice dimensions: " + std::to_string(n) + "\n "
                                            "Requested slice: " + std::to_string(slice.start) + ":" + std::to_string(slice.end));
            }
            return VectorView<T>(Span<T>(data_.data() + slice.start * inc_, stride_ * batch_size_), n, inc_, stride_, batch_size_);
        }

        std::ostream& stream_formatted_to(std::ostream& os, int max_elements = 10, int max_cols_to_print = 10) const {
            std::ios_base::fmtflags original_flags = os.flags();
            os << std::scientific << std::setprecision(4);

            for (int b_idx = 0; b_idx < batch_size_; ++b_idx) {
                if (batch_size_ > 1) {
                    os << "Batch " << b_idx << ":\n";
                }
                os << "  [";
                for (int i = 0; i < std::min(size_, max_elements); ++i) {
                    detail::print_with_width(os, at(i, b_idx), 13); // Supports non-streamable element types
                }
                if (size_ > max_elements) {
                    os << " ...";
                }
                os << " ]\n";
            }
            os.flags(original_flags); // Reset stream flags
            return os;
        }

        void print(std::ostream& os = std::cout, int max_elements = 10) const {
            stream_formatted_to(os, max_elements);
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
        //std::weak_ptr<BackendVectorHandle<T>> backend_handle_;
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

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const VectorView<T>& view) {
        return view.stream_formatted_to(os); // Uses default arguments from stream_formatted_to
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
        os << static_cast<const VectorView<T>&>(vec); // Leverages VectorView's operator<<
        return os;
    }

} // namespace batchlas

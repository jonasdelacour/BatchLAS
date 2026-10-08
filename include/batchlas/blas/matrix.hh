#pragma once
#include <random>
#include <memory>
#include <complex>
#include <type_traits>
#include <iostream> // Added for std::ostream, std::cout
#include <iomanip>  // Added for std::setw, std::scientific, etc.
#include <algorithm> // Added for std::min
#include <tuple>
#include <array>    // Added for std::array element types
#include <sstream>  // Added for temporary string formatting of non-streamable types
#include <batchlas/export.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/blas/enums.hh>

/// @file
/// @brief The batched containers: Matrix (owning), MatrixView (non-owning),
///        KernelMatrixView (device-side), Vector and VectorView.
///
/// Dense storage is column-major. Element (i, j) of batch item b lives at
/// `data[b * stride + j * ld + i]`, all three counted in elements.
/// @see @ref design_matrix_model

namespace batchlas {
    // Forward declarations with default template parameters
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class BATCHLAS_API Matrix;

    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class BATCHLAS_API MatrixView;

    //Forward declare VectorView with default parameter
    template <typename T = float>
    class BATCHLAS_API VectorView;

    /// @brief Backend-library descriptor for a matrix (cuSPARSE / rocSPARSE handles); defined in src/.
    /// @ingroup api_internal_helpers
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    class BackendMatrixHandle;

    /// @brief Non-zero count (per-item capacity) of a CSR matrix, as a distinct type.
    ///
    /// The dense and CSR constructors would otherwise differ only in what the third
    /// `int` means: `Matrix(rows, cols, batch)` versus `Matrix(rows, cols, nnz)`.
    /// @invariant `operator int` is deleted so the tag cannot decay back into that ambiguity.
    /// @see @ref matrix-model-strong-types-for-positional-integers
    /// @ingroup api_matrix
    struct NonZeros {
        int value;  ///< The wrapped value.
        explicit constexpr NonZeros(int v) : value(v) {}  ///< Wraps @p v.
        constexpr operator int() const = delete;   ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<NonZeros>, "NonZeros must stay trivially copyable");

    /// @brief Element distance between consecutive vector entries, as a distinct type.
    ///
    /// Vector takes (stride, inc) where VectorView's positional constructors take
    /// (inc, stride); both readings fit the same buffer, so a transliterated call
    /// reads the wrong addresses silently. The tags make the order a compile error.
    /// @invariant `operator int` is deleted: without it the tag decays back into the ambiguity.
    /// @see @ref matrix-model-strong-types-for-positional-integers
    /// @ingroup api_matrix
    struct Inc {
        int value;  ///< The wrapped value.
        explicit constexpr Inc(int v = 1) : value(v) {}  ///< Wraps @p v.
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<Inc>, "Inc must stay trivially copyable");

    /// @brief Element distance between consecutive batch items, as a distinct type (0 = packed).
    /// @see Inc
    /// @ingroup api_matrix
    struct Stride {
        int value;  ///< The wrapped value.
        explicit constexpr Stride(int v = 0) : value(v) {}  ///< Wraps @p v.
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<Stride>, "Stride must stay trivially copyable");

    /// @brief Leading dimension (column pitch), as a distinct type (0 = packed).
    /// @see Inc
    /// @ingroup api_matrix
    struct Ld {
        int value;  ///< The wrapped value.
        explicit constexpr Ld(int v = 0) : value(v) {}  ///< Wraps @p v.
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<Ld>, "Ld must stay trivially copyable");

    /// @brief Number of batch items, as a distinct type.
    /// @see Inc
    /// @ingroup api_matrix
    struct BatchSize {
        int value;  ///< The wrapped value.
        explicit constexpr BatchSize(int v = 1) : value(v) {}  ///< Wraps @p v.
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<BatchSize>, "BatchSize must stay trivially copyable");

    /// @brief Marker for "to the end" in a Slice: `Slice(k, SliceEnd{})` selects [k, dim).
    /// @ingroup api_matrix
    struct SliceEnd {};

    /// @brief A half-open index range [start, end) along one dimension.
    ///
    /// A default Slice selects the whole dimension; `Slice(k)` selects [k, dim).
    /// A negative `start` or `end` counts from the end of the dimension (`-1` is
    /// the last index). Slicing is dense-only and never copies.
    /// @ingroup api_matrix
    struct Slice { //Default Slice selects entire matrix
        int64_t start = std::numeric_limits<int64_t>::min();  ///< First index; negative counts from the end. The default (min) with a default end means the whole dimension.
        int64_t end = std::numeric_limits<int64_t>::max();    ///< One past the last index; negative counts from the end; max means the end of the dimension.
        /// @brief [@p start, dim).
        Slice(int64_t start, SliceEnd) : start(start), end(std::numeric_limits<int64_t>::max()) {}
        /// @brief [@p start, @p end); either bound may be negative (counted from the end).
        Slice(int64_t start, int64_t end) : start(start), end(end) {}
        /// @brief [@p start, dim) (implicit, so an integer can stand in for a Slice).
        Slice(int64_t start) : Slice(start, SliceEnd()) {}
        /// @brief The whole dimension.
        Slice() = default;
    };

    /// @brief Trivially copyable view of a MatrixView for use inside a SYCL kernel.
    ///
    /// Obtained from Matrix::kernel_view() or MatrixView::kernel_view() on the host
    /// and captured by value. It holds raw pointers and extents only: no ownership,
    /// no backend handle, no bounds checks beyond debug-build asserts. Specialised
    /// on the format so format-specific paths resolve at compile time; slicing is
    /// dense-only. The fields are public so kernels can read them without a call.
    /// @tparam T      scalar type
    /// @tparam MType  MatrixFormat::Dense or MatrixFormat::CSR
    /// @see @ref matrix-model-kernelmatrixview-the-device-side-view
    /// @ingroup api_matrix
    template <typename T, MatrixFormat MType>
    struct KernelMatrixView {
        T*   data_ = nullptr;        ///< Values of batch item 0.
        int  rows_ = 0;              ///< Rows (the capacity when heterogeneous).
        int  cols_ = 0;              ///< Columns (the capacity when heterogeneous).
        int  batch_size_ = 1;        ///< Number of batch items.

        int  ld_ = 0;                ///< Dense: leading dimension, in elements.
        int  stride_ = 0;            ///< Dense: batch stride, in elements.
        int* active_rows_ = nullptr; ///< Dense: per-item row counts, or null when homogeneous.
        int* active_cols_ = nullptr; ///< Dense: per-item column counts, or null when homogeneous.

        int* row_offsets_ = nullptr; ///< CSR: rows + 1 offsets per item, items offset_stride_ apart.
        int* col_indices_ = nullptr; ///< CSR: column indices, items matrix_stride_ apart.
        int  nnz_ = 0;               ///< CSR: per-item non-zero capacity.
        int  matrix_stride_ = 0;     ///< CSR: element distance between items' value (and index) arrays.
        int  offset_stride_ = 0;     ///< CSR: element distance between items' offset arrays.

        /// @brief Dense element (i, j) of batch item b: `data_[b * stride_ + j * ld_ + i]`.
        /// @pre 0 <= i < rows_, 0 <= j < cols_, 0 <= b < batch_size_ (asserted in debug builds)
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        inline constexpr T& operator()(int i, int j, int b = 0) const {
            assert(i < rows_); assert(i >= 0);
            assert(j < cols_); assert(j >= 0);
            assert(b < batch_size_); assert(b >= 0);
            // Only the batch term is widened; j * ld_ + i cannot overflow at allocatable sizes.
            // evidence: docs/design/matrix-model.md#matrix-model-the-element-address-and-its-64-bit-batch-term
            assert(static_cast<int64_t>(b) * stride_ + static_cast<int64_t>(j) * ld_ + i <
                   static_cast<int64_t>(batch_size_) *
                       (stride_ > 0 ? static_cast<int64_t>(stride_)
                                    : static_cast<int64_t>(ld_) * cols_));
            return data_[static_cast<int64_t>(b) * stride_ + j * ld_ + i]; }

        /// @brief CSR coefficient (i, j) of batch item b, or zero if not stored.
        ///
        /// A linear search over row i's entries: O(nnz in row i).
        template <MatrixFormat MF = MType>
            requires CsrMatrixFormat<MF>
        inline constexpr T get(int i, int j, int b = 0) const {
            const int ro_base = b * offset_stride_;
            const int val_base = b * matrix_stride_;
            int rs = row_offsets_[ro_base + i];
            int re = row_offsets_[ro_base + i + 1];
            for (int p = rs; p < re; ++p) {
                if (col_indices_[val_base + p] == j) return data_[val_base + p];
            }
            return T(0);
        }

        /// @brief A single-item view (batch_size_ = 1) of batch item @p b.
        ///
        /// Dense: rows and columns are the item's active extents, and the result is
        /// homogeneous. An out-of-range @p b yields a 0 x 0 view rather than a fault.
        inline constexpr KernelMatrixView batch_item(int b) const {
            KernelMatrixView out = *this;
            if (b < 0 || b >= batch_size_) { out.rows_ = out.cols_ = 0; return out; }
            if constexpr (MType == MatrixFormat::Dense) {
                // int64_t as in operator(): an overflowed product here poisons a raw pointer.
                out.data_ += static_cast<int64_t>(b) * stride_;
                out.rows_ = active_rows_ ? active_rows_[b] : rows_;
                out.cols_ = active_cols_ ? active_cols_[b] : cols_;
                out.active_rows_ = nullptr;
                out.active_cols_ = nullptr;
            } else if constexpr (MType == MatrixFormat::CSR) {
                out.data_        += static_cast<int64_t>(b) * matrix_stride_;
                out.row_offsets_ += static_cast<int64_t>(b) * offset_stride_;
                out.col_indices_ += static_cast<int64_t>(b) * matrix_stride_; // assumes same stride
            }
            out.batch_size_ = 1;
            return out;
        }

        inline auto data() const { return data_; }  ///< Base pointer of item 0.
        /// @brief Row count (the capacity on a heterogeneous batch).
        inline auto rows() const { return rows_; }
        /// @brief Active row count of batch item @p batch_index.
        inline auto rows(int batch_index) const { return active_rows_ ? active_rows_[batch_index] : rows_; }
        /// @brief Column count (the capacity on a heterogeneous batch).
        inline auto cols() const { return cols_; }
        /// @brief Active column count of batch item @p batch_index.
        inline auto cols(int batch_index) const { return active_cols_ ? active_cols_[batch_index] : cols_; }
        inline auto batch_size() const { return batch_size_; }  ///< Number of batch items.
        inline auto ld() const { return ld_; }                  ///< Dense leading dimension, in elements.
        inline auto stride() const { return stride_; }          ///< Dense batch stride, in elements.
        /// @brief CSR per-item non-zero *capacity*; over-counts every item smaller than the largest.
        inline auto nnz() const { return nnz_; }
        /// @brief CSR non-zeros actually stored by batch item @p b, read from its row offsets.
        inline auto nnz(int b) const {
            const auto base = static_cast<int64_t>(b) * offset_stride_;
            return row_offsets_[base + rows_] - row_offsets_[base];
        }
        /// @brief True when per-item active extents are attached.
        inline bool is_heterogeneous() const { return active_rows_ || active_cols_; }

        /// @brief Dense sub-block view; same pointer arithmetic as MatrixView::operator()(Slice, Slice).
        /// @note Empty or inverted slices are not diagnosed here (the check is a no-op assert).
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        KernelMatrixView operator()(Slice rows_slice, Slice cols_slice = {}) const;
        /// @brief Dense row-range view over all columns.
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        KernelMatrixView operator()(Slice rows_slice) const { return (*this)(rows_slice, {}); }

        /// @brief Part of row @p row as a batched VectorView (inc = ld_, stride = stride_).
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        VectorView<T> operator()(int32_t row, Slice cols_slice) const;

        /// @brief Part of column @p col as a batched VectorView (inc = 1, stride = stride_).
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        VectorView<T> operator()(Slice rows_slice, int32_t col) const;

        /// @brief A null 0 x 0 view with batch size 1.
        KernelMatrixView() = default;
        /// @name Copy and move (trivial: the view is captured by value)
        /// @{
        KernelMatrixView(const KernelMatrixView&) = default;
        KernelMatrixView& operator=(const KernelMatrixView&) = default;
        KernelMatrixView(KernelMatrixView&&) = default;
        KernelMatrixView& operator=(KernelMatrixView&&) = default;
        /// @}

        /// @brief Dense view over raw device-accessible memory.
        /// @param data        base of batch item 0
        /// @param rows        rows per item
        /// @param cols        columns per item
        /// @param ld          leading dimension; 0 means @p rows
        /// @param stride      batch stride; 0 means (resolved ld) * @p cols
        /// @param batch_size  number of items
        // stride_ resolves against the *resolved* ld: `ld * cols` would give stride_ = 0 for
        // the three-argument spelling and fire the element-access assert on every access.
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        KernelMatrixView(T* data, int rows, int cols, int ld = 0, int stride = 0, int batch_size = 1)
            : data_(data), rows_(rows), cols_(cols), batch_size_(batch_size),
              ld_(ld > 0 ? ld : rows), stride_(stride > 0 ? stride : (ld > 0 ? ld : rows) * cols) {}
    };

    /// @cond BATCHLAS_DETAIL
    namespace detail {
        inline std::pair<int64_t,int64_t> normalize_slice_component(Slice s, int64_t dim) {
            int64_t len;
            if (s.start == std::numeric_limits<int64_t>::min() && s.end == std::numeric_limits<int64_t>::max()) { s.start = 0; len = dim; }
            else if (s.end == std::numeric_limits<int64_t>::max()) { s.start = s.start < 0 ? dim + s.start : s.start; len = dim - s.start; }
            else { if (s.start < 0) s.start = dim + s.start; if (s.end < 0) s.end = dim + s.end; len = s.end - s.start; }
            return {s.start, len};
        }

        inline void validate_dense_active_dims(int rows_capacity,
                                              int cols_capacity,
                                              int batch_size,
                                              Span<int> active_rows,
                                              Span<int> active_cols) {
            if (active_rows.empty() && active_cols.empty()) {
                return;
            }

            if (active_rows.size() != static_cast<std::size_t>(batch_size) ||
                active_cols.size() != static_cast<std::size_t>(batch_size)) {
                throw batchlas::invalid_argument("Dense heterogeneous metadata must provide rows and cols for every batch item");
            }

            for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
                const int rows = active_rows[batch_index];
                const int cols = active_cols[batch_index];
                if (rows < 0 || cols < 0) {
                    throw batchlas::invalid_argument("Dense heterogeneous metadata cannot contain negative dimensions");
                }
                if (rows > rows_capacity || cols > cols_capacity) {
                    throw batchlas::invalid_argument("Dense heterogeneous metadata exceeds the matrix storage capacity");
                }
            }
        }

        inline int dense_active_rows(int rows_capacity, Span<int> active_rows, int batch_index) {
            return active_rows.empty() ? rows_capacity : active_rows[batch_index];
        }

        inline int dense_active_cols(int cols_capacity, Span<int> active_cols, int batch_index) {
            return active_cols.empty() ? cols_capacity : active_cols[batch_index];
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

    // Out-of-line KernelMatrixView members: documented at their declarations.
    template <typename T, MatrixFormat MType>
    template <MatrixFormat MF>
        requires DenseMatrixFormat<MF>
    KernelMatrixView<T, MType> KernelMatrixView<T, MType>::operator()(Slice rows_slice, Slice cols_slice) const {
        KernelMatrixView<T, MType> out = *this;
        auto [r_start, r_len] = detail::normalize_slice_component(rows_slice, rows_);
        auto [c_start, c_len] = detail::normalize_slice_component(cols_slice, cols_);
        if (r_len <= 0 || c_len <= 0) {
            assert("Invalid slice dimensions in KernelMatrixView");
        }
        detail::apply_dense_slice_pointer_arithmetic(out.data_, ld_, r_start, c_start);
        out.rows_ = static_cast<int>(r_len);
        out.cols_ = static_cast<int>(c_len);
        return out;
    }

    template <typename T, MatrixFormat MType>
    template <MatrixFormat MF>
        requires DenseMatrixFormat<MF>
    VectorView<T> KernelMatrixView<T, MType>::operator()(int32_t row, Slice cols_slice) const {
        auto [c_start, c_len] = detail::normalize_slice_component(cols_slice, cols_);
        if (c_len <= 0) {
            assert("Invalid slice dimensions in KernelMatrixView row extraction");
        }
        T* row_data = data_ + row + c_start * ld_;
        return VectorView<T>(row_data, static_cast<int>(c_len), batch_size_, ld_, stride_);
    }

    template <typename T, MatrixFormat MType>
    template <MatrixFormat MF>
        requires DenseMatrixFormat<MF>
    VectorView<T> KernelMatrixView<T, MType>::operator()(Slice rows_slice, int32_t col) const {
        auto [r_start, r_len] = detail::normalize_slice_component(rows_slice, rows_);
        if (r_len <= 0) {
            assert("Invalid slice dimensions in KernelMatrixView column extraction");
        }
        T* col_data = data_ + r_start + col * ld_;
        return VectorView<T>(col_data, static_cast<int>(r_len), batch_size_, 1, stride_);
    }
    /// @endcond

    static_assert(std::is_trivially_copyable_v<KernelMatrixView<float, MatrixFormat::Dense>>, "KernelMatrixView Dense must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<KernelMatrixView<float, MatrixFormat::CSR>>,   "KernelMatrixView CSR must be trivially copyable");

    /// @brief Owning batch of `batch_size()` matrices of equal (capacity) shape.
    ///
    /// **Dense** (the default): column-major, element (i, j) of item b at
    /// `data()[b * stride() + j * ld() + i]`, with `ld() >= rows()` and
    /// `stride() >= ld() * cols()`. **CSR**: per item, `rows() + 1` row offsets
    /// (items `offset_stride()` apart) and `nnz_capacity()` values and column indices
    /// (items `matrix_stride()` apart), zero-based.
    ///
    /// Storage is USM shared memory (UnifiedVector) on Device::default_device(), so it
    /// is host-accessible and device-accessible. Entry points take a MatrixView;
    /// a Matrix converts to one implicitly (view()). Copy construction and copy
    /// assignment deep-copy the data; moves transfer it. The shape members are public
    /// but are not to be written.
    ///
    /// A dense batch may be **heterogeneous**: set_active_dims() attaches per-item
    /// active extents no larger than the allocated rows() x cols().
    /// @tparam T      scalar type (`float`, `double`, `std::complex<float>`, `std::complex<double>`)
    /// @tparam MType  MatrixFormat::Dense (default) or MatrixFormat::CSR
    /// @see @ref design_matrix_model, and the user guide's data-layout section in @ref md_docs_2cpp-api
    /// @ingroup api_matrix
    // BATCHLAS_API sits on the class template, not on the explicit instantiations: only it
    // reaches the member-template instantiations, and g++ rejects it on an instantiation.
    // evidence: docs/design/symbol-visibility.md#symbol-visibility-the-export-attribute-on-the-matrix-class-template
    template <typename T, MatrixFormat MType>
    class BATCHLAS_API Matrix {
    public:
        friend class MatrixView<T, MType>;  ///< Views share the private storage and backend handle.

        /// @brief Allocates an uninitialised dense batch.
        /// @param rows        rows per item
        /// @param cols        columns per item
        /// @param batch_size  number of items
        /// @param ld          leading dimension; 0 means @p rows
        /// @param stride      batch stride in elements; 0 means (resolved ld) * @p cols
        /// @throws batchlas::invalid_argument on a negative argument, a resolved ld < rows,
        ///         or a resolved stride < ld * cols
        /// @note Exactly `stride * batch_size` elements are allocated; nothing pads.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(int rows, int cols, int batch_size = 1, int ld = 0, int stride = 0);

        /// @brief Allocates an uninitialised CSR batch with room for @p nnz entries per item.
        /// @param rows        rows per item
        /// @param cols        columns per item
        /// @param nnz         per-item capacity; `matrix_stride() == nnz`, `offset_stride() == rows + 1`
        /// @param batch_size  number of items
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(int rows, int cols, NonZeros nnz, int batch_size = 1);

        /// @brief Deleted: spell the count `NonZeros{nnz}`; for a dense Matrix the third int is the batch size.
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(int rows, int cols, int nnz, int batch_size = 1) = delete;

        /// @brief Dense batch copied from existing host-readable memory.
        ///
        /// (@p ld, @p stride) describe the *source*: element (i, j, b) is read from
        /// `data[b * stride + j * ld + i]`. The copy keeps the source's leading
        /// dimension but packs the items back to back (`stride() == ld() * cols()`);
        /// padding rows in the copy are zeroed and padding in the source is never read.
        /// @param data        source; must hold `(batch_size-1)*stride + (cols-1)*ld + rows` elements
        /// @param rows        rows per item
        /// @param cols        columns per item
        /// @param ld          source leading dimension; 0 means @p rows
        /// @param stride      source batch stride; 0 means (resolved ld) * @p cols
        /// @param batch_size  number of items
        /// @throws batchlas::invalid_argument on null @p data, rows or cols <= 0,
        ///         batch_size <= 0, a resolved ld < rows, or (batch_size > 1) a stride < ld * cols
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(const T* data, int rows, int cols, int ld, int stride = 0, int batch_size = 1);

        /// @brief Dense batch copied from a Span; as the pointer overload, plus a length check.
        /// @throws batchlas::invalid_argument also when @p data is shorter than the layout needs
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(Span<const T> data, int rows, int cols, int ld, int stride = 0, int batch_size = 1);

        /// @brief Mutable-Span (and UnifiedVector) spelling of the Span<const T> overload.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(Span<T> data, int rows, int cols, int ld, int stride = 0, int batch_size = 1)
            : Matrix(Span<const T>(data.data(), data.size()), rows, cols, ld, stride, batch_size) {}

        /// @brief CSR batch copied from existing host-readable arrays.
        /// @param values         `matrix_stride * batch_size` values
        /// @param row_offsets    `offset_stride * batch_size` zero-based offsets
        /// @param col_indices    `matrix_stride * batch_size` column indices
        /// @param rows           rows per item
        /// @param cols           columns per item
        /// @param nnz            declared per-item non-zero count
        /// @param matrix_stride  distance between items' value arrays; 0 means @p nnz. May exceed it.
        /// @param offset_stride  distance between items' offset arrays; 0 means rows + 1
        /// @param batch_size     number of items
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(const T* values, const int* row_offsets, const int* col_indices,
               int rows, int cols, NonZeros nnz, int matrix_stride = 0,
               int offset_stride = 0, int batch_size = 1);

        /// @brief Deleted so the pre-tag order (nnz before rows, cols) is a compile error.
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(const T* values, const int* row_offsets, const int* col_indices,
               int nnz, int rows, int cols, int matrix_stride = 0,
               int offset_stride = 0, int batch_size = 1) = delete;


        /// @name Factories
        /// Each returns a packed dense batch (ld == rows, stride == rows * cols) and
        /// waits for its fill kernel before returning.
        /// @{

        /// @brief A batch of n x n identity matrices.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Identity(int n, int batch_size = 1);

        /// @brief Entries uniform in [-1, 1] (real and imaginary parts independently).
        ///
        /// Deterministic: the same @p seed and shape give the same matrix.
        /// With @p hermitian, the lower triangle is overwritten with the conjugate of
        /// the upper and the diagonal made real.
        /// @throws batchlas::invalid_argument if @p hermitian and rows != cols
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Random(int rows, int cols, bool hermitian = false, int batch_size = 1, unsigned int seed = 42);

        /// @brief Random diagonally dominant sparse Hermitian batch; see csr_generators::random_sparse_hermitian_csr().
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        static Matrix<T, MType> RandomSparseHermitian(int n,
                                  float density,
                                  int batch_size = 1,
                                  unsigned int seed = 42,
                                  typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                  bool shared_pattern = true);

        /// @brief Random n x n triangular batch; see MatrixView::fill_triangular_random().
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> RandomTriangular(int n, Uplo uplo, Diag diag = Diag::NonUnit, int batch_size = 1, unsigned int seed = 42) {
            auto result = Matrix<T, MType>(n, n, batch_size);
            result.view().fill_triangular_random(uplo, diag, seed).wait();
            return result;
        }
        
        /// @brief A batch of zero matrices.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Zeros(int rows, int cols, int batch_size = 1);

        /// @brief A batch of all-ones matrices.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Ones(int rows, int cols, int batch_size = 1);

        /// @brief A batch of n x n diagonal matrices, n = `diag_values.size()`, each with @p diag_values on the diagonal.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Diagonal(const Span<T>& diag_values, int batch_size = 1);

        /// @brief n x n matrices with @p diagonal_value on the diagonal, @p non_diagonal_value in
        ///        the @p uplo strict triangle and zero elsewhere.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Triangular(int n, Uplo uplo, T diagonal_value = T(1),
                                          T non_diagonal_value = T(0.5), int batch_size = 1);

        /// @brief n x n tridiagonal Toeplitz matrices (constant diagonal, sub- and super-diagonal).
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> TriDiagToeplitz(int n, T diag = T(1),
                                                T sub_diag = T(-0.5), T super_diag = T(0.5), int batch_size = 1);
        /// @}

        /// @brief Reinterprets this matrix's storage as row-major data and returns it column-major.
        ///
        /// The row pitch is a parameter, never inferred. The result is packed and owns
        /// the memory the kernel writes, so the call waits on @p ctx before returning.
        /// @param ctx        queue for the conversion kernel
        /// @param row_pitch  source row pitch in elements; 0 means packed (cols()), accepted
        ///                   only when this matrix is itself packed
        /// @return a packed column-major copy
        /// @throws std::invalid_argument when the pitch is below cols(), runs past the
        ///         allocation, straddles the next item, or is 0 on a non-packed matrix
        /// @see @ref md_docs_2cpp-api, section "Row-major source data"
        // evidence: docs/cpp-api.md#row-major-source-data
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType> to_column_major(const Queue& ctx, int row_pitch = 0) const;

        /// @brief to_column_major() on an internal queue.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType> to_column_major(int row_pitch = 0) const;

        /// @brief Copy in another storage format, on the default queue.
        ///
        /// Dense to CSR drops entries at or below @p zero_threshold in magnitude (for
        /// complex, both parts) and sizes every item by the batch's largest non-zero
        /// count (see nnz()). CSR to Dense is supported; converting to the same format
        /// returns a copy.
        /// @tparam NewMType  target format
        /// @param zero_threshold  magnitude at or below which a dense entry is not stored
        /// @throws batchlas::unsupported for any other pair of formats
        template <MatrixFormat NewMType>
        Matrix<T, NewMType> convert_to(const float_t<T>& zero_threshold = 1e-7) const;

        /// @brief Packed row-major copy (row pitch cols(), batch stride rows() * cols()).
        ///
        /// Exactly what to_column_major() reads with its default pitch, so the two
        /// round-trip. Waits on @p ctx before returning.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType> to_row_major(const Queue& ctx) const;

        /// @brief to_row_major() on an internal queue.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType> to_row_major() const;

        /// @brief Frees the storage; views of it dangle afterwards.
        ~Matrix();

        /// @name Copy and move
        /// Copies are deep (values, active extents, CSR arrays); moves transfer the storage.
        /// @{
        Matrix(const Matrix& other) = default;
        Matrix& operator=(const Matrix& other) = default;
        Matrix(Matrix&&) noexcept = default;
        Matrix& operator=(Matrix&&) noexcept = default;
        /// @}

        /// @brief Deep copy in this matrix's own layout (ld, stride, CSR strides, active extents).
        Matrix<T, MType> clone() const {
            Matrix<T, MType> result = [&]() -> Matrix<T, MType> {
                if constexpr (MType == MatrixFormat::CSR) {
                    // matrix_stride_ (not nnz_) is the allocated slots per batch item, so
                    // it is what the copies below need room for.
                    return Matrix<T, MType>(rows_, cols_, NonZeros{matrix_stride_}, batch_size_);
                } else {
                    // Must be allocated in *this* matrix's layout: the copy below is a flat
                    // std::copy of length stride_ * batch_size_, so a packed destination is
                    // overrun -- a heap write past the end -- by any padded ld_.
                    return Matrix<T, MType>(rows_, cols_, batch_size_, ld_, stride_);
                }
            }();

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
            if constexpr (MType == MatrixFormat::Dense) {
                if (active_rows_.size() != 0 || active_cols_.size() != 0) {
                    result.set_active_dims(active_rows_.to_span(), active_cols_.to_span());
                }
            }
            
            return result;
        }

        /// @brief A non-owning view of the whole batch; shares the backend handle.
        MatrixView<T, MType> view() const;
        /// @brief A dense view of the leading @p rows x @p cols block of every item.
        /// @param ld      leading dimension of the view; <= 0 keeps ld()
        /// @param stride  batch stride of the view; <= 0 keeps stride()
        /// @throws batchlas::unsupported on a CSR matrix
        MatrixView<T, MType> view(int rows, int cols, int ld = -1, int stride = -1) const;

        /// @brief Dense copy with every element converted by `static_cast<U>`, in this layout.
        template <typename U>
        Matrix<U, MType> astype() const;

        /// @brief View of the sub-block selected by @p rows x @p cols in every item; never copies.
        /// @throws batchlas::invalid_argument if either slice is empty
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> operator()(Slice rows, Slice cols) const {
            auto [r_start, r_len] = detail::normalize_slice_component(rows, rows_);
            auto [c_start, c_len] = detail::normalize_slice_component(cols, cols_);
            if (r_len <= 0 || c_len <= 0) {
                throw batchlas::invalid_argument("Invalid slice dimensions on Matrix: " + std::to_string(r_len) + "x" + std::to_string(c_len));
            }
            auto offset = c_start * ld_ + r_start;
            return MatrixView<T, MType>(data_.data() + offset, static_cast<int>(r_len), static_cast<int>(c_len), ld_, stride_, batch_size_, data_ptrs_.data());
        }
        /// @brief View of rows @p rows over all columns of every item.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> operator()(Slice rows) const { return (*this)(rows, {}); }

        /// @brief Host reference to element (@p row, @p col) of item @p batch.
        ///
        /// No bounds check beyond UnifiedVector's debug assert. The data is USM shared,
        /// so wait for any kernel writing it first.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        U& operator()(int row, int col, int batch) {
            // The batch term in int64_t; see KernelMatrixView::operator() for why only that one.
            return data_.at(static_cast<int64_t>(batch) * stride_ + col * ld_ + row);
        }

        /// @brief Host const reference to element (@p row, @p col) of item @p batch.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        const U& operator()(int row, int col, int batch) const {
            return data_.at(static_cast<int64_t>(batch) * stride_ + col * ld_ + row);
        }

        /// @brief Creates the backend descriptor now instead of on first use; idempotent.
        void init() const;
        /// @brief Fills the per-item pointer array (item b at `data + b * stride()`) on @p ctx and waits.
        /// @return the array, one pointer per item, in USM shared memory
        /// @throws batchlas::invalid_argument when batch_size() is 1: no array is allocated then
        Span<T*> data_ptrs(Queue& ctx) const {
            init_data_ptr_array(ctx);
            return data_ptrs_.to_span();
        }

        /// @brief The backend descriptor, created on first use.
        BackendMatrixHandle<T, MType>* operator->();
        /// @brief The backend descriptor, created on first use.
        BackendMatrixHandle<T, MType>& operator*();

        /// @name Public shape fields
        /// Rows, columns and batch size (capacity extents on a heterogeneous batch). Public for
        /// historical reasons; read through rows(), cols(), batch_size() and never write.
        /// @{
        int rows_, cols_, batch_size_;
        /// @}

        /// @name USM memory hints
        /// Hints for this matrix's storage (and CSR index arrays) on the device of @p ctx.
        /// See Span for the semantics.
        /// @{
        // The Queue is mandatory: a default Queue() is a new queue on Device::default_device(),
        // so the hint would land on the wrong device.
        /// @brief Maps the storage for access by the device of @p ctx.
        Event set_access_device(const Queue& ctx) const {
            if constexpr (MType == MatrixFormat::CSR) {
                (void)row_offsets_.to_span().set_access_device(ctx);
                (void)col_indices_.to_span().set_access_device(ctx);
            }
            return data_.to_span().set_access_device(ctx);
        }

        /// @brief Migrates the storage to the device of @p ctx.
        Event prefetch(const Queue& ctx) const {
            if constexpr (MType == MatrixFormat::CSR) {
                (void)row_offsets_.to_span().prefetch(ctx);
                (void)col_indices_.to_span().prefetch(ctx);
            }
            return data_.to_span().prefetch(ctx);
        }
        /// @}
        /// @brief The whole value storage: `stride() * batch_size()` elements (dense) or
        ///        `matrix_stride() * batch_size()` (CSR).
        Span<T> data() const { return data_.to_span(); }

        /// @brief Sets every element of every item to @p value; waits.
        void fill(T value) {this->view().fill(value).wait();}

        /// @brief Host copy of @p src into this matrix's own layout.
        /// @pre For CSR, @p src has exactly this matrix's strides and capacity.
        /// @throws batchlas::invalid_argument if rows, cols or batch size differ
        void copy_from(const MatrixView<T, MType>& src);

        /// @brief Prints at most the leading block of each item; see MatrixView::print().
        void print(std::ostream& os = std::cout, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            this->view().print(os, max_rows_to_print, max_cols_to_print, max_elements_to_print_csr);
        }

        /// @brief Rows per item (the capacity on a heterogeneous batch).
        int rows() const { return rows_; }
        /// @brief Active rows of item @p batch_index.
        int rows(int batch_index) const {
            return detail::dense_active_rows(rows_, active_rows_.to_span(), batch_index);
        }
        /// @brief Columns per item (the capacity on a heterogeneous batch).
        int cols() const { return cols_; }
        /// @brief Active columns of item @p batch_index.
        int cols(int batch_index) const {
            return detail::dense_active_cols(cols_, active_cols_.to_span(), batch_index);
        }
        int batch_size() const { return batch_size_; }  ///< Number of batch items.
        /// @brief Allocated rows per item; equal to rows().
        int rows_capacity() const { return rows_; }
        /// @brief Allocated columns per item; equal to cols().
        int cols_capacity() const { return cols_; }
        /// @brief True when per-item active extents are attached.
        bool is_heterogeneous() const { return active_rows_.size() != 0 || active_cols_.size() != 0; }

        /// @brief Per-item active row counts; empty when homogeneous.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Span<int> active_rows() const { return active_rows_.to_span(); }

        /// @brief Per-item active column counts; empty when homogeneous.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Span<int> active_cols() const { return active_cols_.to_span(); }

        /// @brief Attaches per-item active extents (copied), or clears them when both spans are empty.
        ///
        /// Item b then occupies the leading `active_rows[b] x active_cols[b]` block of
        /// its allocated rows() x cols(); layout (ld, stride) is unchanged.
        /// @return `*this`
        /// @throws batchlas::invalid_argument unless both spans hold batch_size() entries,
        ///         each in [0, rows()] and [0, cols()] respectively
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType>& set_active_dims(Span<int> active_rows, Span<int> active_cols) {
            detail::validate_dense_active_dims(rows_, cols_, batch_size_, active_rows, active_cols);
            if (active_rows.empty() && active_cols.empty()) {
                active_rows_.clear();
                active_cols_.clear();
                return *this;
            }

            active_rows_.resize(active_rows.size());
            active_cols_.resize(active_cols.size());
            std::copy(active_rows.begin(), active_rows.end(), active_rows_.begin());
            std::copy(active_cols.begin(), active_cols.end(), active_cols_.begin());
            return *this;
        }

        /// @brief The trivially copyable device-side view of this matrix, for capture in a kernel.
        KernelMatrixView<T, MType> kernel_view() const noexcept {
            KernelMatrixView<T, MType> kv;
            kv.data_       = const_cast<T*>(data_.data());
            kv.rows_       = rows_;
            kv.cols_       = cols_;
            kv.ld_         = ld_;
            kv.stride_     = stride_;
            kv.active_rows_ = active_rows_.size() == 0 ? nullptr : const_cast<int*>(active_rows_.data());
            kv.active_cols_ = active_cols_.size() == 0 ? nullptr : const_cast<int*>(active_cols_.data());
            kv.batch_size_ = batch_size_;
            if constexpr (MType == MatrixFormat::CSR) {
                kv.row_offsets_   = const_cast<int*>(row_offsets_.data());
                kv.col_indices_   = const_cast<int*>(col_indices_.data());
                kv.nnz_           = nnz_;
                kv.matrix_stride_ = matrix_stride_;
                kv.offset_stride_ = offset_stride_;
            }
            return kv;
        }

        /// @brief Leading dimension: element distance between columns, >= rows().
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        int ld() const { return ld_; }

        /// @brief Batch stride: element distance between items, >= ld() * cols().
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        int stride() const { return stride_; }

        /// @brief Zero-based row offsets, `offset_stride()` per item.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Span<int> row_offsets() const { return row_offsets_.to_span(); }

        /// @brief Column indices, `matrix_stride()` per item.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Span<int> col_indices() const { return col_indices_.to_span(); }

        /// @brief The per-item non-zero **capacity**, not a count.
        ///
        /// convert_to<MatrixFormat::CSR>() sizes a batch by its largest item, so on a
        /// heterogeneous batch `for (k = 0; k < nnz(); ++k)` walks past a smaller
        /// item's entries; use nnz(int) for a count. Deliberate: the vendor SpMM
        /// descriptors want one capacity for a strided batch.
        /// @see @ref md_docs_2cpp-api, section "The CSR non-zero count has its own type"
        // evidence: docs/cpp-api.md#the-csr-non-zero-count-has-its-own-type
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int nnz() const { return nnz_; }

        /// @brief Slots allocated per item (`matrix_stride()`); may exceed nnz() for the from-data constructor.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int nnz_capacity() const { return matrix_stride_; }

        /// @brief Non-zeros actually stored by item @p batch_index, read from its row offsets on the host.
        /// @pre The kernel that filled the offsets has completed (the storage is USM shared).
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int nnz(int batch_index) const {
            const std::size_t base = static_cast<std::size_t>(batch_index) *
                                     static_cast<std::size_t>(offset_stride_);
            return row_offsets_[base + static_cast<std::size_t>(rows_)] - row_offsets_[base];
        }

        /// @brief Element distance between items' value (and column-index) arrays.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int matrix_stride() const { return matrix_stride_; }

        /// @brief Element distance between items' row-offset arrays.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int offset_stride() const { return offset_stride_; }

        /// @brief The backend descriptor shared with this matrix's views (null until created).
        std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle() const {
            return backend_handle_;
        }

    private:
        void init_data_ptr_array(Queue& ctx) const;

        UnifiedVector<T> data_;

        int ld_ = 0;
        int stride_ = 0;

        UnifiedVector<T*> data_ptrs_;  // one base pointer per item; allocated only when batch_size > 1
        UnifiedVector<int> active_rows_;
        UnifiedVector<int> active_cols_;

        UnifiedVector<int> row_offsets_;
        UnifiedVector<int> col_indices_;
        int nnz_ = 0;
        int matrix_stride_ = 0;
        int offset_stride_ = 0;

        // Shared with views of this matrix; mutable so const methods can create it lazily.
        mutable std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle_;
    };

    /// @brief Non-owning view of a batch of matrices: the argument type of every entry point.
    ///
    /// Same layout as Matrix (column-major dense with ld and stride, or strided CSR)
    /// over memory it does not own. Copying a view is cheap and aliases the data;
    /// the viewed memory must outlive every use, including kernels still in flight.
    /// The memory must be USM that the Queue's device can access; entry points check
    /// the pointers at the call. A view may be built over raw pointers, a Matrix
    /// (implicitly), a sub-block (operator()(Slice, Slice)), a single item
    /// (batch_item()) or a VectorView.
    ///
    /// Members taking a `const Queue&` enqueue a kernel and return its Event without
    /// waiting. The queue-less overloads run on a new Queue() on Device::default_device().
    /// @tparam T      scalar type
    /// @tparam MType  MatrixFormat::Dense (default) or MatrixFormat::CSR
    /// @see @ref design_matrix_model
    /// @ingroup api_matrix
    template <typename T, MatrixFormat MType>
    class BATCHLAS_API MatrixView {
    public:
        /// @brief Dense view over caller memory.
        ///
        /// Note the argument order: the batch size is the *sixth* argument here and the
        /// third in `Matrix(rows, cols, batch_size)`.
        /// @param data        base of item 0 (may be null for a shape-only view, e.g. a workspace query)
        /// @param rows        rows per item
        /// @param cols        columns per item
        /// @param ld          leading dimension; 0 means @p rows
        /// @param stride      batch stride; 0 means (resolved ld) * @p cols. Not a broadcast.
        /// @param batch_size  number of items
        /// @param data_ptrs   optional per-item pointer array for pointer-array backends
        /// @throws batchlas::invalid_argument on a negative argument or a resolved ld < rows
        ///         (the message names the intended spelling)
        /// @note `stride < ld * cols` is not rejected here, unlike the Matrix constructor.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView(T* data, int rows, int cols, int ld = 0,
                  int stride = 0, int batch_size = 1, T** data_ptrs = nullptr);

        /// @brief CSR view over caller arrays; layout as in the CSR Matrix from-data constructor.
        /// @param data_ptrs  optional per-item pointer array to the values
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        MatrixView(T* data, int* row_offsets, int* col_indices,
                  int rows, int cols, NonZeros nnz, int matrix_stride = 0,
                  int offset_stride = 0, int batch_size = 1, T** data_ptrs = nullptr);

        /// @brief Deleted so the pre-tag order (nnz before rows, cols) is a compile error.
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        MatrixView(T* data, int* row_offsets, int* col_indices,
                  int nnz, int rows, int cols, int matrix_stride = 0,
                  int offset_stride = 0, int batch_size = 1, T** data_ptrs = nullptr) = delete;

        /// @brief View of a whole Matrix (implicit): its layout, active extents, pointer array and backend handle.
        MatrixView(const Matrix<T, MType>& matrix);

        /// @brief Views a batched vector as n x 1 (Column) or 1 x n (Row) matrices; the stride carries over.
        /// @throws batchlas::invalid_argument for Column when the vector's inc != 1
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView(const VectorView<T>& vector_view, VectorOrientation orientation = VectorOrientation::Column);

        /// @brief View of @p matrix with any of its extents or strides overridden; a value <= 0 keeps the matrix's.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView(const Matrix<T, MType>& matrix,
                  int rows = -1, int cols = -1, int ld = -1, int stride = -1, int batch_size = -1);

        /// @brief View of @p matrix_view with any of its extents or strides overridden; a value <= 0 keeps the view's.
        ///
        /// Active extents carry over only when every override is a no-op.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView(const MatrixView<T, MType>& matrix_view,
                  int rows = -1, int cols = -1, int ld = -1, int stride = -1, int batch_size = -1);

        /// @brief An empty 0 x 0 view with batch size 0.
        MatrixView() = default;

        /// @brief Copies @p other's span into @p data and returns a view of the copy in @p other's layout.
        ///
        /// Runs on a new queue on Device::default_device() and waits.
        /// @param other      view to copy
        /// @param data       destination, at least as long as `other.data()`
        /// @param data_ptrs  optional destination for a copy of @p other's pointer array
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static MatrixView<T, MType> deep_copy(const MatrixView<T, MType>& other, //Matrix view to copy from
                                                T* data, //storage for the new view
                                                T** data_ptrs = nullptr //optional array of pointers to the start of each matrix in a batch
                                                );

        /// @brief CSR form of deep_copy(): values, offsets and indices each go to caller storage.
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        static MatrixView<T, MType> deep_copy(const MatrixView<T, MType>& other, //Matrix view to copy from
                                                T* data, int* row_offsets, int* col_indices, //storage for the new view
                                                T** data_ptrs = nullptr //optional array of pointers to the start of each matrix in a batch
                                                );

        /// @brief Enqueues dest := src on @p ctx, honouring each side's ld and stride.
        /// @return the event of the copy; a no-op when both views share a base pointer
        /// @throws batchlas::invalid_argument if rows, cols or batch size differ
        /// @throws batchlas::unsupported for CSR views
        static Event copy(Queue& ctx, const MatrixView<T, MType>& dest, const MatrixView<T, MType>& src);

        /// @name Copy and move
        /// Shallow: the copy aliases the same memory and shares the backend handle.
        /// @{
        MatrixView(const MatrixView&) = default;
        MatrixView& operator=(const MatrixView&) = default;
        MatrixView(MatrixView&&) noexcept = default;
        MatrixView& operator=(MatrixView&&) noexcept = default;
        /// @}

        /// @brief Releases nothing but the shared backend handle; the viewed memory is untouched.
        ~MatrixView() = default;

        /// @brief Creates the backend descriptor now instead of on first use; idempotent.
        void init() const;

        /// @brief The backend descriptor, created on first use.
        BackendMatrixHandle<T, MType>* operator->() const;
        /// @brief The backend descriptor, created on first use.
        BackendMatrixHandle<T, MType>& operator*() const;

        /// @brief batch_item(@p i).
        MatrixView<T, MType> operator[](int i) const;

        /// @name Public shape fields
        /// Rows, columns and batch size (capacity extents on a heterogeneous batch). Read
        /// through the accessors; never write.
        /// @{
        // Must stay initialised: the entry points' USM check reads these extents to decide
        // whether a null data pointer is legal on a default-constructed view.
        int rows_ = 0, cols_ = 0, batch_size_ = 0;
        /// @}

        /// @brief The viewed value storage, from item 0 to the end of the last item.
        Span<T> data() const { return data_; }
        /// @name USM memory hints
        /// As Matrix::set_access_device() and Matrix::prefetch(), for the viewed memory.
        /// @{
        /// @brief Maps the viewed memory for access by the device of @p ctx.
        Event set_access_device(const Queue& ctx) const {
            if constexpr (MType == MatrixFormat::CSR) {
                (void)row_offsets_.set_access_device(ctx);
                (void)col_indices_.set_access_device(ctx);
            }
            return data_.set_access_device(ctx);
        }

        /// @brief Migrates the viewed memory to the device of @p ctx.
        Event prefetch(const Queue& ctx) const {
            if constexpr (MType == MatrixFormat::CSR) {
                (void)row_offsets_.prefetch(ctx);
                (void)col_indices_.prefetch(ctx);
            }
            return data_.prefetch(ctx);
        }
        /// @}
        /// @brief Base pointer of item 0.
        T* data_ptr() const { return data_.data(); }


        int batch_size() const { return batch_size_; }  ///< Number of batch items.
        /// @brief Rows per item (the capacity on a heterogeneous batch).
        int rows() const { return rows_; }
        /// @brief Active rows of item @p batch_index.
        int rows(int batch_index) const {
            return detail::dense_active_rows(rows_, active_rows_, batch_index);
        }
        /// @brief Columns per item (the capacity on a heterogeneous batch).
        int cols() const { return cols_; }
        /// @brief Active columns of item @p batch_index.
        int cols(int batch_index) const {
            return detail::dense_active_cols(cols_, active_cols_, batch_index);
        }
        /// @brief Equal to rows().
        int rows_capacity() const { return rows_; }
        /// @brief Equal to cols().
        int cols_capacity() const { return cols_; }
        /// @brief True when per-item active extents are attached.
        bool is_heterogeneous() const { return !active_rows_.empty() || !active_cols_.empty(); }

        /// @brief Per-item active row counts; empty when homogeneous.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Span<int> active_rows() const { return active_rows_; }

        /// @brief Per-item active column counts; empty when homogeneous.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Span<int> active_cols() const { return active_cols_; }

        /// @brief Packed dense copy with every element converted by `static_cast<U>`; host loop, no Queue.
        template <typename U>
        Matrix<U, MType> astype() const;

        /// @brief Leading dimension: element distance between columns.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        int ld() const { return ld_; }

        /// @brief Batch stride: element distance between items.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        int stride() const { return stride_; }

        /// @brief Always 1: the element distance down a column.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        int inc() const { return 1; } // Assuming contiguous elements for vector views

        /// @brief Zero-based row offsets, `offset_stride()` per item.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Span<int> row_offsets() const { return row_offsets_; }

        /// @brief Column indices, `matrix_stride()` per item.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Span<int> col_indices() const { return col_indices_; }

        /// @brief Per-item non-zero capacity, not a count; see Matrix::nnz().
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int nnz() const { return nnz_; }

        /// @brief Slots allocated per item; see Matrix::nnz_capacity().
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int nnz_capacity() const { return matrix_stride_; }

        /// @brief Non-zeros actually stored by item @p batch_index, read on the host.
        /// @pre The row offsets are host-readable: not over `sycl::malloc_device` memory
        ///      (use KernelMatrixView::nnz(int) in a kernel instead).
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int nnz(int batch_index) const {
            const std::size_t base = static_cast<std::size_t>(batch_index) *
                                     static_cast<std::size_t>(offset_stride_);
            return row_offsets_[base + static_cast<std::size_t>(rows_)] - row_offsets_[base];
        }

        /// @brief Element distance between items' value (and column-index) arrays.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int matrix_stride() const { return matrix_stride_; }

        /// @brief Element distance between items' row-offset arrays.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        int offset_stride() const { return offset_stride_; }

        /// @brief Fills the per-item pointer array from this view's base and stride on @p ctx and waits.
        ///
        /// The array may be shared with the parent Matrix and its other views, so it
        /// holds whatever the last view to fill it wrote.
        /// @throws batchlas::invalid_argument if the view has no pointer array of batch_size() entries
        Span<T*> data_ptrs(Queue& ctx) const {
            init_data_ptr_array(ctx);
            return data_ptrs_;
        }

        /// @brief Host reference to element (@p row, @p col) of item @p batch.
        /// @throws batchlas::out_of_range outside the item's active extents
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        T& at(int row, int col, int batch = 0);

        /// @brief Host const reference to element (@p row, @p col) of item @p batch.
        /// @throws batchlas::out_of_range outside the item's active extents
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        const T& at(int row, int col, int batch = 0) const;

        /// @brief at(@p row, @p col, @p batch).
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        T& operator()(int row, int col, int batch = 0) {
            return at(row, col, batch);
        }

        /// @brief at(@p row, @p col, @p batch).
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        const T& operator()(int row, int col, int batch = 0) const {
            return at(row, col, batch);
        }

        /// @brief View of the sub-block @p rows x @p cols of every item; keeps ld and stride, never copies.
        /// @throws batchlas::invalid_argument if either slice is empty
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> operator()(Slice rows, Slice cols) const {
            auto [r_start, r_len] = detail::normalize_slice_component(rows, rows_);
            auto [c_start, c_len] = detail::normalize_slice_component(cols, cols_);
            if (r_len <= 0 || c_len <= 0) {
                throw batchlas::invalid_argument("Invalid slice dimensions: " + std::to_string(r_len) + "x" + std::to_string(c_len));
            }
            auto offset = c_start * ld_ + r_start;
            // Trap: the slice keeps the parent's pointer array, which holds *unsliced* base
            // addresses until data_ptrs(ctx) refills it for this view.
            // evidence: docs/design/matrix-model.md#matrix-model-the-pointer-array-and-sliced-views
            return MatrixView<T, MType>(data_ptr() + offset, static_cast<int>(r_len), static_cast<int>(c_len), ld_, stride_, batch_size_, data_ptrs_.data());
        }

        /// @brief View of rows @p rows over all columns of every item.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> operator()(Slice rows) const {
            return (*this)(rows, {});
        }

        /// @brief Part of row @p row of every item as a batched VectorView (inc = ld(), stride = stride()).
        /// @throws batchlas::invalid_argument if the slice is empty
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        VectorView<T> operator()(int32_t row, Slice cols) const {
            auto [c_start, c_len] = detail::normalize_slice_component(cols, cols_);
            if (c_len <= 0) {
                throw batchlas::invalid_argument("Invalid slice dimensions for row vector: " + std::to_string(c_len));
            }
            auto offset = c_start * ld_ + row;
            return VectorView<T>(data_ptr() + offset, static_cast<int>(c_len), batch_size_, ld_, stride_);
        }

        /// @brief Part of column @p col of every item as a batched VectorView (inc = 1, stride = stride()).
        /// @throws batchlas::invalid_argument if the slice is empty
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        VectorView<T> operator()(Slice rows, int32_t col) const {
            auto [r_start, r_len] = detail::normalize_slice_component(rows, rows_);
            if (r_len <= 0) {
                throw batchlas::invalid_argument("Invalid slice dimensions for column vector: " + std::to_string(r_len));
            }
            auto offset = col * ld_ + r_start;
            return VectorView<T>(data_ptr() + offset, static_cast<int>(r_len), batch_size_, 1, stride_);
        }

        /// @name In-place fills and transforms
        /// Dense-only unless stated. The `const Queue&` overloads enqueue and return the
        /// kernel's Event without waiting; the queue-less overloads run on a new Queue()
        /// on Device::default_device(). The square transforms read rows() as n.
        /// @{

        /// @brief Copies the @p uplo triangle onto the other: a_ji := a_ij for (i, j) in @p uplo.
        /// @pre rows() == cols()
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event symmetrize(const Queue& ctx, Uplo uplo) const;

        /// @brief symmetrize() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event symmetrize(Uplo uplo) const {
            return symmetrize(Queue(), uplo);
        }

        /// @brief Mirrors the @p uplo triangle conjugated (a_ji := conj(a_ij)) and makes the diagonal real.
        /// @pre rows() == cols()
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event hermitize(const Queue& ctx, Uplo uplo) const;

        /// @brief hermitize() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event hermitize(Uplo uplo) const {
            return hermitize(Queue(), uplo);
        }

        /// @brief In-place (conjugate) transpose.
        /// @note Declared but not defined by the library: a call fails to link.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event transpose(const Queue& ctx, bool conjugate = false) const;

        /// @brief transpose() on a new default queue; likewise undefined.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event transpose(bool conjugate = false) const {
            return transpose(Queue(), conjugate);
        }

        /// @brief Zeros the strict triangle opposite @p uplo, keeping the @p uplo triangle;
        ///        with Diag::Unit also sets the diagonal to 1.
        ///
        /// `Uplo::Upper` leaves an upper-triangular matrix. Honours rows() != cols().
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event triangularize(const Queue& ctx, Uplo uplo, Diag diag) const;

        /// @brief triangularize() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event triangularize(Uplo uplo, Diag diag) const {
            return triangularize(Queue(), uplo, diag);
        }

        /// @brief Fills with values uniform in [-1, 1], deterministic in @p seed; see Matrix::Random().
        /// @note Writes every element of data(), padding between columns and items included.
        /// @throws batchlas::invalid_argument if @p hermitian and rows() != cols()
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random(const Queue& ctx, bool hermitian = false, unsigned int seed = 42) const;

        /// @brief fill_random() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random(bool hermitian = false, unsigned int seed = 42) const {
            return fill_random(Queue(), hermitian, seed);
        }

        /// @brief CSR: fills this storage with a random diagonally dominant Hermitian pattern and values.
        ///
        /// See csr_generators::random_sparse_hermitian_csr() for the construction.
        /// Blocks on @p ctx while building the pattern; the returned Event is the value fill.
        /// @throws batchlas::invalid_argument unless the view is square, non-empty, and
        ///         its nnz() equals the count @p density implies
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Event fill_random_sparse_hermitian(const Queue& ctx,
                                           float density,
                                           unsigned int seed = 42,
                                           typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                           bool shared_pattern = true) const;

        /// @brief fill_random_sparse_hermitian() on a new default queue.
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Event fill_random_sparse_hermitian(float density,
                                           unsigned int seed = 42,
                                           typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                           bool shared_pattern = true) const {
            return fill_random_sparse_hermitian(Queue(), density, seed, diagonal_boost, shared_pattern);
        }

        /// @brief Sets every element of every item (dense: within rows() x cols(); CSR: every stored value).
        Event fill(const Queue& ctx, T value) const;

        /// @brief fill() on a new default queue.
        Event fill(T value) const {
            return fill(Queue(), value);
        }

        /// @brief fill() with zero. Deliberately has no queue-less form, which would run on the default device.
        Event fill_zeros(const Queue& ctx) const {
            return fill(ctx, detail::convert_to_fill_value<T>(0));
        }

        /// @brief fill() with one. Deliberately has no queue-less form.
        Event fill_ones(const Queue& ctx) const {
            return fill(ctx, detail::convert_to_fill_value<T>(1));
        }

        /// @brief Zeros every item, then sets its main diagonal to @p value.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_identity(const Queue& ctx, T value = T(1)) const;

        /// @brief Writes diagonal @p k (k > 0 above the main diagonal, k < 0 below).
        ///
        /// @p diag_values holds either n - |k| values used for every item, or
        /// (n - |k|) * batch_size() values, one run per item. Other entries are untouched.
        /// @pre rows() == cols() == n
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Queue& ctx, const Span<T>& diag_values, int64_t k = 0) const;

        /// @brief Writes diagonal @p k from a VectorView: per item when its batch size equals
        ///        batch_size(), otherwise its item 0 for every item.
        /// @pre rows() == cols(), `diag_values.size() >= n - |k|`
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Queue& ctx, const VectorView<T>& diag_values, int64_t k = 0) const;

        /// @brief fill_diagonal() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Span<T>& diag_values, int64_t k = 0) const {
            return fill_diagonal(Queue(), diag_values, k);
        }

        /// @brief Sets the main diagonal of every item to @p value.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Queue& ctx, const T& value) const;

        /// @brief fill_diagonal() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const T& value) const {
            return fill_diagonal(Queue(), value);
        }

        /// @brief Writes a full triangular matrix: @p diagonal_value on the diagonal,
        ///        @p non_diagonal_value in the @p uplo strict triangle, zero elsewhere.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular(const Queue& ctx, Uplo uplo, T diagonal_value = T(1),
                              T non_diagonal_value = T(0.5)) const;

        /// @brief fill_triangular() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular(Uplo uplo, T diagonal_value = T(1),
                              T non_diagonal_value = T(0.5)) const {
            return fill_triangular(Queue(), uplo, diagonal_value, non_diagonal_value);
        }

        /// @brief Writes the three diagonals from per-item vectors; other entries are untouched.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_tridiag(const Queue& ctx, VectorView<T> sub_diag,
                            VectorView<T> diag, VectorView<T> super_diag) const {
                                // (void) on an Event: deliberate. This Queue is in-order, so the next submission
                                // is already ordered after this one and the Event carries nothing the caller needs.
                                (void)fill_diagonal(ctx, diag);
                                (void)fill_diagonal(ctx, sub_diag, -1);
                                return fill_diagonal(ctx, super_diag, 1);
                            }

        /// @brief Writes a full tridiagonal Toeplitz matrix (zero off the three diagonals).
        /// @pre The view is packed: ld() == rows() == cols(), stride() == rows() * cols()
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_tridiag_toeplitz(const Queue& ctx, T diag = T(1),
                                    T sub_diag = T(-0.5), T super_diag = T(0.5)) const;

        /// @brief fill_tridiag_toeplitz() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_tridiag_toeplitz(T diag = T(1),
                                    T sub_diag = T(-0.5), T super_diag = T(0.5)) const {
            return fill_tridiag_toeplitz(Queue(), diag, sub_diag, super_diag);
        }

        /// @brief Writes a full random triangular matrix: uniform [-1, 1] in the @p uplo
        ///        triangle, 1 on the diagonal for Diag::Unit, zero elsewhere.
        /// @note Every item receives the same matrix (the values depend on @p seed and the position only).
        /// @pre The view is packed: ld() == rows() == cols(), stride() == rows() * cols()
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular_random(const Queue& ctx, Uplo uplo,
                                     Diag diag = Diag::Unit,
                                     unsigned int seed = 42) const;

        /// @brief fill_triangular_random() on a new default queue.
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular_random(Uplo uplo,
                                     Diag diag = Diag::Unit,
                                     unsigned int seed = 42) const {
            return fill_triangular_random(Queue(), uplo, diag, seed);
        }
        /// @}


        /// @brief A batch-size-1 view of item @p batch_index.
        ///
        /// Dense: its active extents, the parent's ld, no pointer array. CSR: the parent's strides.
        /// @throws batchlas::out_of_range if @p batch_index is outside [0, batch_size())
        MatrixView<T, MType> batch_item(int batch_index) const;

        /// @brief A copy of this view with per-item active extents attached (not copied: the spans must outlive the view).
        /// @throws batchlas::invalid_argument under the same rules as Matrix::set_active_dims()
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> with_active_dims(Span<int> active_rows, Span<int> active_cols) const {
            detail::validate_dense_active_dims(rows_, cols_, batch_size_, active_rows, active_cols);
            MatrixView<T, MType> out = *this;
            out.active_rows_ = active_rows;
            out.active_cols_ = active_cols;
            return out;
        }

        /// @brief True when every element of @p a and @p b agrees to relative tolerance @p tol; waits on @p ctx.
        ///
        /// Elements agree when |x - y| / max(|x|, |y|) <= tol (the denominator is 1 when
        /// the larger magnitude is below @p tol, and two values both below epsilon
        /// agree); complex compares the real and imaginary parts separately. Dense only.
        /// @throws batchlas::invalid_argument if the shapes or batch sizes differ
        /// @throws batchlas::unsupported for CSR
        static bool all_close(Queue& ctx, const MatrixView<T, MType>& a, const MatrixView<T, MType>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon());
        /// @brief all_close() on two matrices.
        static inline bool all_close(Queue& ctx, const Matrix<T, MType>& a, const Matrix<T, MType>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon()) {return all_close(ctx, a.view(), b.view(), tol);}
        /// @brief all_close() on a view and a matrix.
        static inline bool all_close(Queue& ctx, const MatrixView<T, MType>& a, const Matrix<T, MType>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon()) {return all_close(ctx, a, b.view(), tol);}
        /// @brief all_close() on a matrix and a view.
        static inline bool all_close(Queue& ctx, const Matrix<T, MType>& a, const MatrixView<T, MType>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon()) {return all_close(ctx, a.view(), b, tol);}

        /// @brief Prints each item's leading block (dense) or leading entries (CSR) in scientific notation.
        /// @return @p os, with its format flags restored
        std::ostream& stream_formatted_to(std::ostream& os, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            std::ios_base::fmtflags original_flags = os.flags();
            os << std::scientific << std::setprecision(4);

            for (int b_idx = 0; b_idx < batch_size_; ++b_idx) {
                if (batch_size_ > 1) {
                    os << "Batch " << b_idx << ":\n";
                }
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

        /// @brief stream_formatted_to() without the return value; reads the data on the host.
        void print(std::ostream& os = std::cout, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            stream_formatted_to(os, max_rows_to_print, max_cols_to_print, max_elements_to_print_csr);
        }

        /// @brief The trivially copyable device-side view, for capture in a kernel.
        KernelMatrixView<T, MType> kernel_view() const noexcept {
            KernelMatrixView<T, MType> kv;
            kv.data_       = const_cast<T*>(data_.data());
            kv.rows_       = rows_;
            kv.cols_       = cols_;
            kv.ld_         = ld_;
            kv.stride_     = stride_;
            kv.active_rows_ = active_rows_.empty() ? nullptr : active_rows_.data();
            kv.active_cols_ = active_cols_.empty() ? nullptr : active_cols_.data();
            kv.batch_size_ = batch_size_;
            if constexpr (MType == MatrixFormat::CSR) {
                kv.row_offsets_   = const_cast<int*>(row_offsets_.data());
                kv.col_indices_   = const_cast<int*>(col_indices_.data());
                kv.nnz_           = nnz_;
                kv.matrix_stride_ = matrix_stride_;
                kv.offset_stride_ = offset_stride_;
            }
            return kv;
        }

    private:
        void init_data_ptr_array(Queue& ctx, Span<T*> target = {}) const;

        Span<T> data_;

        int ld_ = 0;
        int stride_ = 0;
        Span<int> active_rows_;
        Span<int> active_cols_;

        Span<T*> data_ptrs_;  // may alias the parent Matrix's pointer array

        Span<int> row_offsets_;
        Span<int> col_indices_;
        int nnz_ = 0;
        int matrix_stride_ = 0;
        int offset_stride_ = 0;

        // Shared with the viewed Matrix; mutable so const methods can create it lazily.
        mutable std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle_;
    };

    /// @brief Creates the backend descriptor for a matrix (src/backends/matrix_handle_impl.cc).
    /// @ingroup api_internal_helpers
    template <typename T, MatrixFormat MType>
    std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const Matrix<T, MType>& matrix);

    /// @brief Creates the backend descriptor for a view (src/backends/matrix_handle_impl.cc).
    /// @ingroup api_internal_helpers
    template <typename T, MatrixFormat MType>
    std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const MatrixView<T, MType>& view);

    /// @brief Backend-library descriptor for a vector; defined in src/.
    /// @ingroup api_internal_helpers
    template <typename T = float>
    class BackendVectorHandle; // Forward declaration with default parameter

    /// @brief Owning batch of `batch_size()` vectors of length `size()` in USM shared memory.
    ///
    /// Element i of item b lives at `data()[b * stride() + i * inc()]`. The default
    /// stride is `size() * inc()` (items back to back). Converts implicitly to
    /// VectorView.
    /// @trap The constructor takes (size, batch_size, Stride, Inc) while VectorView's
    ///       positional constructor takes (data, size, batch_size, inc, stride); the tag types
    ///       and the deleted bare-int overloads keep the two from being transliterated.
    /// @see @ref matrix-model-vectors-inc-and-stride
    /// @ingroup api_matrix
    template <typename T>
    struct Vector {
        using value_type = T;               ///< Element type.
        using pointer = T*;                 ///< Pointer to an element.
        using reference = T&;               ///< Reference to an element.
        using const_reference = const T&;   ///< Const reference to an element.

            /// @brief Elements needed to hold the layout: `(batch_size-1)*stride + (size-1)*inc + 1`, or 0 when empty.
            static constexpr std::size_t required_span_length(int size, int inc, int stride, int batch_size) {
                if (size <= 0 || batch_size <= 0) return 0;
                const int64_t max_index = int64_t(batch_size - 1) * int64_t(stride) + int64_t(size - 1) * int64_t(inc);
                return static_cast<std::size_t>(max_index + 1);
            }

        /// @brief An empty vector with batch size 1.
        Vector() : data_(), size_(0), inc_(1), stride_(0), batch_size_(1) {}

        /// @brief Allocates an uninitialised batch.
        /// @param size        elements per item
        /// @param batch_size  number of items
        /// @param stride      element distance between items; Stride{0} means size * inc
        /// @param inc         element distance between entries
        Vector(int size, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1})
            : data_(required_span_length(size, inc.value, (stride.value > 0 ? stride.value : size * inc.value), batch_size)),
              size_(size), inc_(inc.value), stride_(stride.value > 0 ? stride.value : size * inc.value), batch_size_(batch_size) {}
        /// @brief Allocates a batch with every stored element (gaps included) set to @p value.
        Vector(int size, T value, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1})
            : data_(required_span_length(size, inc.value, (stride.value > 0 ? stride.value : size * inc.value), batch_size), value),
              size_(size), inc_(inc.value), stride_(stride.value > 0 ? stride.value : size * inc.value), batch_size_(batch_size) {}

        /// @brief Deleted: spell the layout `Stride{s}, Inc{i}`.
        Vector(int size, int batch_size, int stride, int inc) = delete;
        /// @brief Deleted: spell the layout `Stride{s}, Inc{i}`.
        Vector(int size, T value, int batch_size, int stride, int inc) = delete;
        /// @brief Deleted: `Vector<float>(n, batch, stride)` would otherwise pick the
        ///        (size, value, batch_size) overload and turn `batch` into the fill value.
        // Vector<int> loses (size, value, batch) to ambiguity; nothing in the repo uses it.
        Vector(int size, int batch_size, int stride) = delete;

        /// @brief A batch of zero vectors.
        static Vector<T> zeros(int size, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1}) {
            return Vector<T>(size, T(0), batch_size, stride, inc);
        }
        /// @brief A batch of all-ones vectors.
        static Vector<T> ones(int size, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1}) {
            return Vector<T>(size, T(1), batch_size, stride, inc);
        }
        /// @brief Entries uniform in [0, 1) (real part only), from a host RNG with a fixed seed of 42.
        static Vector<T> random(int size, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1}) {
            Vector<T> vec(size, batch_size, stride, inc);
            std::mt19937 rng(42); // Mersenne Twister engine with fixed seed
            std::uniform_real_distribution<float_t<T>> dist(0.0f, 1.0f);
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < size; ++i) {
                    vec(i, b) = static_cast<T>(dist(rng));
                }
            }
            return vec;
        }
        /// @brief Unit vectors e_@p index (zero except a one at @p index), in every item.
        static Vector<T> standard_basis(int size, int index, int batch_size = 1, Stride stride = Stride{0}) {
            Vector<T> vec(size, T(0), batch_size, stride, Inc{1});
            for (int b = 0; b < batch_size; ++b) {
                vec(index, b) = T(1);
            }
            return vec;
        }

        /// @name Deleted bare-int factory spellings
        /// Spell the layout `Stride{s}, Inc{i}`; the old argument order is a compile error.
        /// @{
        // standard_basis's third argument really is batch_size, so it stays legal.
        static Vector<T> zeros(int size, int batch_size, int stride, int inc) = delete;
        static Vector<T> ones(int size, int batch_size, int stride, int inc) = delete;
        static Vector<T> random(int size, int batch_size, int stride, int inc) = delete;
        static Vector<T> zeros(int size, int batch_size, int stride) = delete;
        static Vector<T> ones(int size, int batch_size, int stride) = delete;
        static Vector<T> random(int size, int batch_size, int stride) = delete;
        static Vector<T> standard_basis(int size, int index, int batch_size, int stride) = delete;
        /// @}

        /// @brief The whole storage, gaps included.
        Span<T> data() const { return data_.to_span(); }

        /// @brief A non-owning view; mirrors Matrix::view().
        ///
        /// Needed where T must be deduced: deduction does not consider the implicit conversion.
        VectorView<T> view() const { return VectorView<T>(*this); }

        /// @brief Maps the storage for access by the device of @p ctx (mandatory Queue; see Matrix).
        Event set_access_device(const Queue& ctx) const {
            return data_.to_span().set_access_device(ctx);
        }

        /// @brief Migrates the storage to the device of @p ctx.
        Event prefetch(const Queue& ctx) const {
            return data_.to_span().prefetch(ctx);
        }
        T* data_ptr() const { return data_.data(); }  ///< Pointer to entry 0 of item 0.
        int size() const { return size_; }             ///< Entries per item.
        /// @brief Element distance between entries.
        int inc() const { return inc_; }
        /// @brief Element distance between items.
        int stride() const { return stride_; }
        int batch_size() const { return batch_size_; }  ///< Number of batch items.

        /// @brief Copy with every element converted by `static_cast<U>`, in this layout.
        template <typename U>
        Vector<U> astype() const;

        /// @brief Host reference to entry @p i of item @p batch; not bounds-checked.
        // The batch term is int64_t: batch * stride_ in int wraps inside the target batch sizes.
        T& at(int i, int batch = 0) { return data_[i * inc_ + static_cast<int64_t>(batch) * stride_]; }
        //const T& at(int i, int batch = 0) const { return data_[i * inc_ + batch * stride_]; }

        /// @brief Raw storage index @p i, ignoring inc and stride (kept for compatibility).
        T& operator[](int i) { return data_[i]; }

        /// @brief at(@p i, @p batch).
        T& operator()(int i, int batch = 0) { return at(i, batch); }

        /// @brief A batch-size-1 view of item @p batch_index.
        VectorView<T> batch_item(int batch_index) const {
            const std::size_t single_batch_length = (size_ > 0) ? ((size_ - 1) * inc_ + 1) : 0;
            return VectorView<T>(Span<T>(data_.data() + static_cast<int64_t>(batch_index) * stride_, single_batch_length), size_, 1, inc_, 0);
        }

        /// @brief Prints the leading entries of each item; see VectorView::stream_formatted_to().
        void print(std::ostream& os = std::cout, int max_elements = 10) const {
            VectorView<T>(*this).stream_formatted_to(os, max_elements);
        }

        /// @name Backend descriptor
        /// The backend-library vector descriptor (internal_helpers), created on first use.
        /// @{
        BackendVectorHandle<T>* operator->();
        BackendVectorHandle<T>& operator*();
        const BackendVectorHandle<T>* operator->() const;
        const BackendVectorHandle<T>& operator*()  const;
        /// @}

    private:
        UnifiedVector<T> data_;
        int size_ = 0;
        int inc_ = 1;
        int stride_ = 0;
        int batch_size_ = 1;

        //std::shared_ptr<BackendVectorHandle<T>> backend_handle_;
    };

    /// @brief Non-owning view of a batch of strided vectors: the vector argument type of the entry points.
    ///
    /// Entry i of item b is at `data_ptr()[b * stride() + i * inc()]`; a stride of 0
    /// means `size * inc` (items back to back). The memory must be USM the Queue's
    /// device can access and must outlive every use.
    /// @trap The positional constructors take (data, size, batch_size, inc, stride),
    ///       the opposite order to Vector's (size, batch_size, Stride, Inc). Prefer the
    ///       tagged overloads in new code.
    /// @see @ref matrix-model-vectors-inc-and-stride
    /// @ingroup api_matrix
    template <typename T>
    class BATCHLAS_API VectorView {
    public:
        using value_type = T;               ///< Element type.
        using pointer = T*;                 ///< Pointer to an element.
        using reference = T&;               ///< Reference to an element.
        using const_reference = const T&;   ///< Const reference to an element.

        /// @brief Elements needed to hold the layout: `(batch_size-1)*stride + (size-1)*inc + 1`, or 0 when empty.
        static constexpr std::size_t required_span_length(int size, int inc, int stride, int batch_size) {
            if (size <= 0 || batch_size <= 0) return 0;
            const int64_t max_index = int64_t(batch_size - 1) * int64_t(stride) + int64_t(size - 1) * int64_t(inc);
            return static_cast<std::size_t>(max_index + 1);
        }

        /// @brief An empty view with batch size 1.
        VectorView() : data_(), size_(0), inc_(1), stride_(0), batch_size_(1) {}
        /// @brief Views @p data with the given layout.
        /// @param data        storage; must hold required_span_length() elements (debug-asserted)
        /// @param size        entries per item
        /// @param batch_size  number of items
        /// @param inc         element distance between entries
        /// @param stride      element distance between items; 0 means size * inc
        VectorView(Span<T> data, int size, int batch_size = 1, int inc = 1, int stride = 0)
            : data_(data.data(), required_span_length(size, inc, (stride > 0 ? stride : size * inc), batch_size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {
                assert(data.size() >= data_.size());
            }
        /// @brief Views a UnifiedVector's storage; parameters as the Span overload.
        VectorView(UnifiedVector<T>& data, int size, int batch_size = 1, int inc = 1, int stride = 0)
            : data_(data.data(), required_span_length(size, inc, (stride > 0 ? stride : size * inc), batch_size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {
                assert(data.size() >= data_.size());
            }
        /// @brief Views raw memory; parameters as the Span overload (no length check).
        VectorView(T* data, int size, int batch_size = 1, int inc = 1, int stride = 0)
            : data_(data, required_span_length(size, inc, (stride > 0 ? stride : size * inc), batch_size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {}

        /// @name Tagged constructors (preferred for new code)
        /// Same as the positional forms, with inc and stride spelled `Inc{}` / `Stride{}`.
        /// @{
        // Additive on purpose: the in-repo positional constructions are all correct.
        /// @brief Views a Span; see the positional Span overload.
        VectorView(Span<T> data, int size, int batch_size, Inc inc, Stride stride = Stride{0})
            : VectorView(data, size, batch_size, inc.value, stride.value) {}
        /// @brief Views a UnifiedVector's storage.
        VectorView(UnifiedVector<T>& data, int size, int batch_size, Inc inc, Stride stride = Stride{0})
            : VectorView(data, size, batch_size, inc.value, stride.value) {}
        /// @brief Views raw memory (no length check).
        VectorView(T* data, int size, int batch_size, Inc inc, Stride stride = Stride{0})
            : VectorView(data, size, batch_size, inc.value, stride.value) {}
        /// @}

        /// @brief View of a whole Vector (implicit).
        VectorView(const Vector<T>& vec)
            : data_(vec.data()), size_(vec.size()), inc_(vec.inc()), stride_(vec.stride()), batch_size_(vec.batch_size()) {}

        /// @name Copy and move
        /// Shallow: the copy aliases the same memory.
        /// @{
        VectorView(const VectorView<T>&) = default;
        VectorView& operator=(const VectorView<T>&) = default;
        VectorView(VectorView<T>&&) noexcept = default;
        VectorView& operator=(VectorView<T>&&) noexcept = default;
        /// @}

        /// @brief The viewed storage, from entry 0 of item 0 to the last entry of the last item.
        Span<T> data() const { return data_; }
        /// @name USM memory hints
        /// See Span; the Queue is mandatory, as on Matrix.
        /// @{
        /// @brief Maps the viewed memory for access by the device of @p ctx.
        Event set_access_device(const Queue& ctx) const {
            return data_.set_access_device(ctx);
        }

        /// @brief Prefers residence of the viewed memory on the device of @p ctx.
        Event set_preferred_location(const Queue& ctx) const {
            return data_.set_preferred_location(ctx);
        }

        /// @brief Migrates the viewed memory to the device of @p ctx.
        Event prefetch(const Queue& ctx) const {
            return data_.prefetch(ctx);
        }
        /// @}
        T* data_ptr() const { return data_.data(); }  ///< Pointer to entry 0 of item 0.
        int size() const { return size_; }             ///< Entries per item.
        /// @brief Element distance between entries.
        int inc() const { return inc_; }
        /// @brief Element distance between items.
        int stride() const { return stride_; }
        int batch_size() const { return batch_size_; }  ///< Number of batch items.

        /// @brief Owning copy with every element converted by `static_cast<U>`, in this layout; host loop.
        template <typename U>
        Vector<U> astype() const;

        /// @brief Enqueues dest := src on @p ctx, honouring each side's inc and stride.
        /// @throws batchlas::invalid_argument if the sizes or batch sizes differ
        static Event copy(Queue& ctx, const VectorView<T>& dest, const VectorView<T>& src);

        /// @brief True when every entry of @p a and @p b agrees to relative tolerance @p tol; waits on @p ctx.
        /// @see MatrixView::all_close() for the comparison rule.
        static bool all_close(Queue& ctx, const VectorView<T>& a, const VectorView<T>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon());
        /// @brief all_close() on two vectors.
        static bool all_close(Queue& ctx, const Vector<T>& a, const Vector<T>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon()) {
            return all_close(ctx, VectorView<T>(a), VectorView<T>(b), tol);
        }
        /// @brief all_close() on a view and a vector.
        static bool all_close(Queue& ctx, const VectorView<T>& a, const Vector<T>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon()) {
            return all_close(ctx, a, VectorView<T>(b), tol);
        }
        /// @brief all_close() on a vector and a view.
        static bool all_close(Queue& ctx, const Vector<T>& a, const VectorView<T>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon()) {
            return all_close(ctx, VectorView<T>(a), b, tol);
        }

        /// @brief Entry @p i of item @p batch; usable on the host and in a kernel.
        /// @pre 0 <= i < size(), 0 <= batch < batch_size() (asserted in debug builds)
        T& at(int i, int batch = 0) const {
            assert(i < size_); assert(i >= 0);
            assert(batch < batch_size_); assert(batch >= 0);
            // Both sides int64_t: an int index has already wrapped, and a size_t
            // conversion turns a negative index into one that passes the check.
            assert(static_cast<int64_t>(i) * inc_ + static_cast<int64_t>(batch) * stride_ <
                   static_cast<int64_t>(data_.size()));
                return data_[static_cast<int64_t>(i) * inc_ + static_cast<int64_t>(batch) * stride_]; }

        /// @brief Entry @p i of item 0.
        T& operator[](int i) const { assert(i < size_); assert(i >= 0); assert(static_cast<int64_t>(i) * inc_ < static_cast<int64_t>(data_.size())); return data_[static_cast<int64_t>(i) * inc_]; }

        /// @brief at(@p i, @p batch).
        T& operator()(int i, int batch = 0) const {return at(i, batch);}

        /// @brief A batch-size-1 view of item @p batch_index.
        VectorView<T> batch_item(int batch_index) const {
            // For a single batch, we need (size-1)*inc + 1 elements starting at batch_index * stride
            const std::size_t single_batch_length = (size_ > 0) ? ((size_ - 1) * inc_ + 1) : 0;
            return VectorView<T>(Span<T>(data_.data() + static_cast<int64_t>(batch_index) * stride_, single_batch_length), size_, 1, inc_, 0);
        }

        /// @brief Entries @p slice of every item; keeps inc, stride and batch size.
        /// @pre The slice is non-empty (asserted in debug builds)
        VectorView<T> operator()(Slice slice) const {
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
                assert("Invalid slice dimensions for VectorView encountered in operator()(Slice)" && false);
            }
            const auto required = required_span_length(static_cast<int>(n), inc_, stride_, batch_size_);
            return VectorView<T>(Span<T>(data_.data() + slice.start * inc_, required), n, batch_size_, inc_, stride_);
        }

        /// @brief Element-wise z_i := op(a * x_i, b * y_i) for every item; the default op gives z = (a x) .* (b y).
        ///
        /// z is overwritten, not accumulated. Enqueued on @p ctx; returns its Event.
        /// @pre x, y and z have the same size and batch size
        template <typename BinaryOperatorOp = std::multiplies<T>>
        static Event hadamard_product(const Queue& ctx, T a, T b, const VectorView<T>& x, const VectorView<T>& y, const VectorView<T>& z, BinaryOperatorOp op = BinaryOperatorOp());

        /// @brief Element-wise z := a * x + b * y for every item (overwrites z).
        static Event add(const Queue& ctx, T a, T b, const VectorView<T>& x, const VectorView<T>& y, const VectorView<T>& z) {
            return hadamard_product(ctx, a, b, x, y, z, std::plus<T>());
        }

        //Computes result = x' * y
        //Event dot(const Queue& ctx, const VectorView<T>& x, const VectorView<T>& y, T& result);

        /// @brief Prints the leading @p max_elements entries of each item in scientific notation.
        /// @return @p os, with its format flags restored
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

        /// @brief stream_formatted_to(@p os, @p max_elements).
        void print(std::ostream& os = std::cout, int max_elements = 10) const {
            stream_formatted_to(os, max_elements);
        }

        /// @name Backend descriptor
        /// The backend-library vector descriptor (internal_helpers), created on first use.
        /// @{
        BackendVectorHandle<T>* operator->();
        BackendVectorHandle<T>& operator*();
        const BackendVectorHandle<T>* operator->() const;
        const BackendVectorHandle<T>& operator*() const;
        /// @}

    private:
        Span<T> data_;
        int size_ = 0;
        int inc_ = 1;
        int stride_ = 0;
        int batch_size_ = 1;
        //std::weak_ptr<BackendVectorHandle<T>> backend_handle_;
    };

    // ------------------------------------------------------------------
    // Conversion helpers
    // ------------------------------------------------------------------
    template <typename T, MatrixFormat MType>
    template <typename U>
    Matrix<U, MType> Matrix<T, MType>::astype() const {
        static_assert(MType == MatrixFormat::Dense, "Matrix::astype only supports dense matrices");
        Matrix<U, MType> result(rows_, cols_, batch_size_, ld_, stride_);
        auto src = data();
        auto dst = result.data();
        for (std::size_t i = 0; i < dst.size(); ++i) {
            dst[i] = static_cast<U>(src[i]);
        }
        return result;
    }

    template <typename T, MatrixFormat MType>
    template <typename U>
    Matrix<U, MType> MatrixView<T, MType>::astype() const {
        static_assert(MType == MatrixFormat::Dense, "MatrixView::astype only supports dense matrices");
        Matrix<U, MType> result(rows_, cols_, batch_size_);
        // Deliberately a host loop, not a kernel: astype takes no Queue, and building one
        // (then waiting on it) would cost more than the copy.
        const T* src_ptr = data_.data();
        U* dst_ptr = result.data().data();
        const std::size_t src_stride = static_cast<std::size_t>(stride_);
        const std::size_t packed = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);

        if (ld_ == rows_ && src_stride == packed) {
            const std::size_t n = packed * static_cast<std::size_t>(batch_size_);
            for (std::size_t k = 0; k < n; ++k) dst_ptr[k] = static_cast<U>(src_ptr[k]);
        } else {
            for (int b = 0; b < batch_size_; ++b) {
                for (int j = 0; j < cols_; ++j) {
                    const T* src_col = src_ptr + static_cast<std::size_t>(b) * src_stride +
                                       static_cast<std::size_t>(j) * static_cast<std::size_t>(ld_);
                    U* dst_col = dst_ptr + static_cast<std::size_t>(b) * packed +
                                 static_cast<std::size_t>(j) * static_cast<std::size_t>(rows_);
                    for (int i = 0; i < rows_; ++i) dst_col[i] = static_cast<U>(src_col[i]);
                }
            }
        }
        return result;
    }

    template <typename T>
    template <typename U>
    Vector<U> Vector<T>::astype() const {
        Vector<U> result(size_, batch_size_, Stride{stride_}, Inc{inc_});
        auto src = data();
        auto dst = result.data();
        for (std::size_t i = 0; i < dst.size(); ++i) {
            dst[i] = static_cast<U>(src[i]);
        }
        return result;
    }

    template <typename T>
    template <typename U>
    Vector<U> VectorView<T>::astype() const {
        Vector<U> result(size_, batch_size_, Stride{stride_}, Inc{inc_});
        for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < size_; ++i) {
                result(i, b) = static_cast<U>((*this)(i, b));
            }
        }
        return result;
    }

    /// @brief Shape (rows, cols) of op(@p mat) for @p trans: swapped unless NoTrans.
    /// @ingroup api_matrix
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    std::pair<int, int> get_effective_dims(const MatrixView<T, MType>& mat, Transpose trans) {
        return (trans == Transpose::NoTrans)
               ? std::make_pair(mat.rows_, mat.cols_)
               : std::make_pair(mat.cols_, mat.rows_);
    }

    /// @brief Shape of op(@p mat) for item @p batch_index, using that item's active extents.
    /// @ingroup api_matrix
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    std::pair<int, int> get_effective_dims(const MatrixView<T, MType>& mat, Transpose trans, int batch_index) {
        const int rows = mat.rows(batch_index);
        const int cols = mat.cols(batch_index);
        return (trans == Transpose::NoTrans)
               ? std::make_pair(rows, cols)
               : std::make_pair(cols, rows);
    }

    /// @brief Prints with MatrixView::stream_formatted_to() defaults.
    /// @ingroup api_matrix
    template <typename T, MatrixFormat MType>
    std::ostream& operator<<(std::ostream& os, const MatrixView<T, MType>& view) {
        return view.stream_formatted_to(os); // Uses default arguments from stream_formatted_to
    }

    /// @brief Prints the matrix's view.
    /// @ingroup api_matrix
    template <typename T, MatrixFormat MType>
    std::ostream& operator<<(std::ostream& os, const Matrix<T, MType>& matrix) {
        os << matrix.view(); // Leverages MatrixView's operator<<
        return os;
    }

    /// @brief Prints with VectorView::stream_formatted_to() defaults.
    /// @ingroup api_matrix
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const VectorView<T>& view) {
        return view.stream_formatted_to(os); // Uses default arguments from stream_formatted_to
    }

    /// @brief Prints the vector's view.
    /// @ingroup api_matrix
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
        os << VectorView(vec); // Leverages VectorView's operator<<
        return os;
    }

    /// @brief In-place A := alpha * A on every element of every item (dense).
    /// @param ctx       queue the kernel is enqueued on
    /// @param alpha     scale factor
    /// @param mat_view  matrices to scale, within rows() x cols() (ld and stride honoured)
    /// @return event of the kernel
    /// @throws batchlas::unsupported for CSR
    /// @ingroup api_extra
    template <typename T, MatrixFormat MType>
    BATCHLAS_API Event scale(Queue& ctx, const T& alpha, const MatrixView<T, MType>& mat_view);

    /// @brief In-place x := alpha * x on every entry of every item.
    /// @return event of the kernel
    /// @ingroup api_extra
    template <typename T>
    BATCHLAS_API Event scale(Queue& ctx, const T& alpha, const VectorView<T>& vec_view);

} // namespace batchlas

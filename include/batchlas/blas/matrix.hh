// SPDX-License-Identifier: MIT
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
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <batchlas/export.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/blas/enums.hh>

/// @file
/// @brief The batched containers: Matrix (owning), MatrixView (non-owning),
///        KernelMatrixView (device-side), Vector and VectorView.
///
/// Dense storage is column-major. Element \f$(i, j)\f$ of batch item \f$b\f$ lives at
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

    template <typename T>
    struct Vector;

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
        /// @brief Wraps @p v.
        /// @param[in] v  per-item non-zero count
        explicit constexpr NonZeros(int v) : value(v) {}
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
        /// @brief Wraps @p v.
        /// @param[in] v  element distance between consecutive entries
        explicit constexpr Inc(int v = 1) : value(v) {}
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<Inc>, "Inc must stay trivially copyable");

    /// @brief Element distance between consecutive batch items, as a distinct type (0 = packed).
    /// @see Inc
    /// @ingroup api_matrix
    struct Stride {
        int value;  ///< The wrapped value.
        /// @brief Wraps @p v.
        /// @param[in] v  element distance between consecutive batch items; 0 means packed
        explicit constexpr Stride(int v = 0) : value(v) {}
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<Stride>, "Stride must stay trivially copyable");

    /// @brief Leading dimension (column pitch), as a distinct type (0 = packed).
    /// @see Inc
    /// @ingroup api_matrix
    struct Ld {
        int value;  ///< The wrapped value.
        /// @brief Wraps @p v.
        /// @param[in] v  leading dimension in elements; 0 means packed
        explicit constexpr Ld(int v = 0) : value(v) {}
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<Ld>, "Ld must stay trivially copyable");

    /// @brief Number of batch items, as a distinct type.
    /// @see Inc
    /// @ingroup api_matrix
    struct BatchSize {
        int value;  ///< The wrapped value.
        /// @brief Wraps @p v.
        /// @param[in] v  number of batch items
        explicit constexpr BatchSize(int v = 1) : value(v) {}
        constexpr operator int() const = delete;  ///< Deleted: no silent decay back to int.
    };
    static_assert(std::is_trivially_copyable_v<BatchSize>, "BatchSize must stay trivially copyable");

    /// @brief Marker for "to the end" in a Slice: `Slice(k, SliceEnd{})` selects \f$[k, \mathrm{dim})\f$.
    /// @ingroup api_matrix
    struct SliceEnd {};

    /// @brief A half-open index range \f$[\mathrm{start}, \mathrm{end})\f$ along one dimension.
    ///
    /// A default Slice selects the whole dimension; `Slice(k)` selects \f$[k, \mathrm{dim})\f$.
    /// A negative `start` or `end` counts from the end of the dimension (`-1` is
    /// the last index). Slicing is dense-only and never copies.
    /// @ingroup api_matrix
    struct Slice { //Default Slice selects entire matrix
        int64_t start = std::numeric_limits<int64_t>::min();  ///< First index; negative counts from the end. The default (min) with a default end means the whole dimension.
        int64_t end = std::numeric_limits<int64_t>::max();    ///< One past the last index; negative counts from the end; max means the end of the dimension.
        /// @brief Selects \f$[\mathrm{start}, \mathrm{dim})\f$.
        /// @param[in] start  first index; negative counts from the end
        Slice(int64_t start, SliceEnd) : start(start), end(std::numeric_limits<int64_t>::max()) {}
        /// @brief Selects \f$[\mathrm{start}, \mathrm{end})\f$; either bound may be negative (counted from the end).
        /// @param[in] start  first index; negative counts from the end
        /// @param[in] end    one past the last index; negative counts from the end
        Slice(int64_t start, int64_t end) : start(start), end(end) {}
        /// @brief Selects \f$[\mathrm{start}, \mathrm{dim})\f$ (implicit, so an integer can stand in for a Slice).
        /// @param[in] start  first index; negative counts from the end
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

        /// @brief One byte per item, nonzero = active; null when the view is unmasked (every item active).
        const std::uint8_t* mask_ = nullptr;

        /// @brief True when batch item @p b takes part in the call: no mask, or a nonzero mask byte.
        /// @param[in] b  batch item index
        /// @return whether item @p b is active
        /// @pre \f$0 \le b < \mathrm{batch\_size\_}\f$ when a mask is attached
        inline constexpr bool active(int64_t b) const { return mask_ == nullptr || mask_[b] != 0; }

        /// @brief Dense element \f$(i, j)\f$ of batch item \f$b\f$: `data_[b * stride_ + j * ld_ + i]`.
        /// @param[in] i  row index
        /// @param[in] j  column index
        /// @param[in] b  batch item index
        /// @return a reference to the element
        /// @pre \f$0 \le i < \mathrm{rows\_}\f$, \f$0 \le j < \mathrm{cols\_}\f$,
        ///      \f$0 \le b < \mathrm{batch\_size\_}\f$ (asserted in debug builds)
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

        /// @brief CSR coefficient \f$(i, j)\f$ of batch item \f$b\f$, or zero if not stored.
        ///
        /// A linear search over row @p i's entries: \f$O(\mathrm{nnz}_i)\f$, with \f$\mathrm{nnz}_i\f$
        /// the number of entries stored in that row.
        /// @param[in] i  row index
        /// @param[in] j  column index
        /// @param[in] b  batch item index
        /// @return the stored value, or \f$0\f$ when \f$(i, j)\f$ is not in the pattern
        template <MatrixFormat MF = MType>
            requires CsrMatrixFormat<MF>
        inline constexpr T get(int i, int j, int b = 0) const {
            const std::ptrdiff_t ro_base = static_cast<std::ptrdiff_t>(b) * offset_stride_;
            const std::ptrdiff_t val_base = static_cast<std::ptrdiff_t>(b) * matrix_stride_;
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
        /// homogeneous. A mask moves to the item (`batch_item(b).active(0) == active(b)`).
        /// An out-of-range @p b yields a \f$0 \times 0\f$ view rather than a fault.
        /// @param[in] b  batch item index
        /// @return the single-item view
        inline constexpr KernelMatrixView batch_item(int b) const {
            KernelMatrixView out = *this;
            if (b < 0 || b >= batch_size_) { out.rows_ = out.cols_ = 0; return out; }
            if (mask_) out.mask_ = mask_ + b;
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
        /// @param[in] batch_index  batch item index
        /// @return the item's active rows (rows() when homogeneous)
        inline auto rows(int batch_index) const { return active_rows_ ? active_rows_[batch_index] : rows_; }
        /// @brief Column count (the capacity on a heterogeneous batch).
        inline auto cols() const { return cols_; }
        /// @brief Active column count of batch item @p batch_index.
        /// @param[in] batch_index  batch item index
        /// @return the item's active columns (cols() when homogeneous)
        inline auto cols(int batch_index) const { return active_cols_ ? active_cols_[batch_index] : cols_; }
        inline auto batch_size() const { return batch_size_; }  ///< Number of batch items.
        inline auto ld() const { return ld_; }                  ///< Dense leading dimension, in elements.
        inline auto stride() const { return stride_; }          ///< Dense batch stride, in elements.
        /// @brief CSR per-item non-zero *capacity*; over-counts every item smaller than the largest.
        inline auto nnz() const { return nnz_; }
        /// @brief CSR non-zeros actually stored by batch item @p b, read from its row offsets.
        /// @param[in] b  batch item index
        /// @return the item's stored non-zero count
        inline auto nnz(int b) const {
            const auto base = static_cast<int64_t>(b) * offset_stride_;
            return row_offsets_[base + rows_] - row_offsets_[base];
        }
        /// @brief True when per-item active extents are attached.
        inline bool is_heterogeneous() const { return active_rows_ || active_cols_; }

        /// @brief Dense sub-block view; same pointer arithmetic as MatrixView::operator()(Slice, Slice).
        /// @note Empty or inverted slices are not diagnosed here (the check is a no-op assert).
        /// @param[in] rows_slice  rows to keep
        /// @param[in] cols_slice  columns to keep; the default keeps all
        /// @return the sub-block view (same ld_ and stride_)
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        KernelMatrixView operator()(Slice rows_slice, Slice cols_slice = {}) const;
        /// @brief Dense row-range view over all columns.
        /// @param[in] rows_slice  rows to keep
        /// @return the sub-block view
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        KernelMatrixView operator()(Slice rows_slice) const { return (*this)(rows_slice, {}); }

        /// @brief Part of row @p row as a batched VectorView (inc = ld_, stride = stride_).
        /// @param[in] row         row index
        /// @param[in] cols_slice  columns to keep
        /// @return a view of the row segment of every item
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        VectorView<T> operator()(int32_t row, Slice cols_slice) const;

        /// @brief Part of column @p col as a batched VectorView (inc = 1, stride = stride_).
        /// @param[in] rows_slice  rows to keep
        /// @param[in] col         column index
        /// @return a view of the column segment of every item
        template <MatrixFormat MF = MType>
            requires DenseMatrixFormat<MF>
        VectorView<T> operator()(Slice rows_slice, int32_t col) const;

        /// @brief A null \f$0 \times 0\f$ view with batch size 1.
        KernelMatrixView() = default;
        /// @name Copy and move (trivial: the view is captured by value)
        /// @{
        KernelMatrixView(const KernelMatrixView&) = default;
        KernelMatrixView& operator=(const KernelMatrixView&) = default;
        KernelMatrixView(KernelMatrixView&&) = default;
        KernelMatrixView& operator=(KernelMatrixView&&) = default;
        /// @}

        /// @brief Dense view over raw device-accessible memory.
        /// @param[in] data        base of batch item 0 (stored, not read here)
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] ld          leading dimension; 0 means @p rows
        /// @param[in] stride      batch stride; 0 means \f$\mathrm{ld} \cdot \mathrm{cols}\f$ with the resolved ld
        /// @param[in] batch_size  number of items
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

        // Host slicing only: KernelMatrixView shares normalize_slice_component() in device code,
        // which cannot throw. An empty in-range slice is left to the caller's invalid_argument.
        inline std::pair<int64_t,int64_t> checked_slice_component(Slice s, int64_t dim) {
            const auto [start, len] = normalize_slice_component(s, dim);
            if (start < 0 || start > dim || start + std::max<int64_t>(len, 0) > dim) {
                throw batchlas::out_of_range("Slice [" + std::to_string(start) + ", " +
                                             std::to_string(start + len) + ") is outside a dimension of " +
                                             std::to_string(dim));
            }
            return {start, len};
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

        // Length only: the bytes may live in device-only memory, so they are never read here.
        inline void check_batch_mask_length(std::size_t mask_size, int batch_size) {
            if (mask_size != 0 && mask_size != static_cast<std::size_t>(batch_size < 0 ? 0 : batch_size)) {
                throw batchlas::invalid_argument("with_batch_mask: the mask holds " + std::to_string(mask_size) +
                                                 " bytes but the view has " + std::to_string(batch_size) +
                                                 " batch items");
            }
        }

        // Host copy of mask bytes for astype (no Queue); throws for a device-only source. Defined in src/matrix.cc.
        BATCHLAS_API void copy_batch_mask(Span<std::uint8_t> dst, Span<const std::uint8_t> src);

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
            assert(r_len > 0 && c_len > 0 && "Invalid slice dimensions in KernelMatrixView");
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
            assert(c_len > 0 && "Invalid slice dimensions in KernelMatrixView row extraction");
        }
        T* row_data = data_ + row + c_start * ld_;
        VectorView<T> out(row_data, static_cast<int>(c_len), batch_size_, ld_, stride_);
        if (mask_) out.batch_mask_ = Span<const std::uint8_t>(mask_, static_cast<std::size_t>(batch_size_));
        return out;
    }

    template <typename T, MatrixFormat MType>
    template <MatrixFormat MF>
        requires DenseMatrixFormat<MF>
    VectorView<T> KernelMatrixView<T, MType>::operator()(Slice rows_slice, int32_t col) const {
        auto [r_start, r_len] = detail::normalize_slice_component(rows_slice, rows_);
        if (r_len <= 0) {
            assert(r_len > 0 && "Invalid slice dimensions in KernelMatrixView column extraction");
        }
        T* col_data = data_ + r_start + col * ld_;
        VectorView<T> out(col_data, static_cast<int>(r_len), batch_size_, 1, stride_);
        if (mask_) out.batch_mask_ = Span<const std::uint8_t>(mask_, static_cast<std::size_t>(batch_size_));
        return out;
    }
    /// @endcond

    static_assert(std::is_trivially_copyable_v<KernelMatrixView<float, MatrixFormat::Dense>>, "KernelMatrixView Dense must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<KernelMatrixView<float, MatrixFormat::CSR>>,   "KernelMatrixView CSR must be trivially copyable");

    /// @brief Non-owning view of a batch of matrices: the argument type of every entry point.
    ///
    /// Same layout as Matrix (column-major dense with ld and stride, or strided CSR)
    /// over memory it does not own. Copying a view is cheap and aliases the data;
    /// the viewed memory must outlive every use, including kernels still in flight.
    /// The memory must be USM that the Queue's device can access; entry points check
    /// the pointers at the call. A view may be built over raw pointers, a Matrix
    /// (every Matrix is a MatrixView), a sub-block (operator()(Slice, Slice)), a single
    /// item (batch_item()) or a VectorView.
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
        /// @param[in] data        base of item 0 (may be null for a shape-only view, e.g. a workspace query);
        ///                        stored, not read here
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] ld          leading dimension; 0 means @p rows
        /// @param[in] stride      batch stride; 0 means \f$\mathrm{ld} \cdot \mathrm{cols}\f$ with the resolved ld.
        ///                        Not a broadcast.
        /// @param[in] batch_size  number of items
        /// @param[in] data_ptrs   optional per-item pointer array of @p batch_size entries for
        ///                        pointer-array backends; stored, filled later by data_ptrs(Queue&)
        /// @throws batchlas::invalid_argument on a negative argument or a resolved \f$\mathrm{ld} < \mathrm{rows}\f$
        ///         (the message names the intended spelling)
        /// @note \f$\mathrm{stride} < \mathrm{ld} \cdot \mathrm{cols}\f$ is not rejected here, unlike the Matrix constructor.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView(T* data, int rows, int cols, int ld = 0,
                  int stride = 0, int batch_size = 1, T** data_ptrs = nullptr);

        /// @brief CSR view over caller arrays; layout as in the CSR Matrix from-data constructor.
        /// @param[in] data           values, \f$\mathrm{matrix\_stride} \cdot \mathrm{batch\_size}\f$ entries (stored, not read here)
        /// @param[in] row_offsets    zero-based row offsets, \f$\mathrm{offset\_stride} \cdot \mathrm{batch\_size}\f$ entries
        /// @param[in] col_indices    column indices, \f$\mathrm{matrix\_stride} \cdot \mathrm{batch\_size}\f$ entries
        /// @param[in] rows           rows per item
        /// @param[in] cols           columns per item
        /// @param[in] nnz            per-item non-zero count
        /// @param[in] matrix_stride  distance between items' value arrays; 0 means @p nnz
        /// @param[in] offset_stride  distance between items' offset arrays; 0 means \f$\mathrm{rows} + 1\f$
        /// @param[in] batch_size     number of items
        /// @param[in] data_ptrs      optional per-item pointer array to the values (stored)
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        MatrixView(T* data, int* row_offsets, int* col_indices,
                  int rows, int cols, NonZeros nnz, int matrix_stride = 0,
                  int offset_stride = 0, int batch_size = 1, T** data_ptrs = nullptr);

        /// @brief Deleted so the pre-tag order (nnz before rows, cols) is a compile error.
        /// @param[in] data           values
        /// @param[in] row_offsets    row offsets
        /// @param[in] col_indices    column indices
        /// @param[in] nnz            the count, which must be spelled `NonZeros{nnz}` after @p cols
        /// @param[in] rows           rows per item
        /// @param[in] cols           columns per item
        /// @param[in] matrix_stride  distance between items' value arrays
        /// @param[in] offset_stride  distance between items' offset arrays
        /// @param[in] batch_size     number of items
        /// @param[in] data_ptrs      per-item pointer array
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        MatrixView(T* data, int* row_offsets, int* col_indices,
                  int nnz, int rows, int cols, int matrix_stride = 0,
                  int offset_stride = 0, int batch_size = 1, T** data_ptrs = nullptr) = delete;

        /// @brief Views a batched vector as \f$n \times 1\f$ (Column) or \f$1 \times n\f$ (Row) matrices; the stride and the batch mask carry over.
        /// @param[in] vector_view  the vector of length \f$n\f$ to view
        /// @param[in] orientation  Column or Row
        /// @throws batchlas::invalid_argument for Column when the vector's \f$\mathrm{inc} \ne 1\f$
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView(const VectorView<T>& vector_view, VectorOrientation orientation = VectorOrientation::Column);

        /// @brief View of @p matrix_view with any of its extents or strides overridden; a value \f$\le 0\f$ keeps the view's.
        ///
        /// Active extents carry over only when every override is a no-op; the batch mask carries over
        /// unless the batch size changes, which throws batchlas::invalid_argument on a masked view.
        /// @param[in] matrix_view  the view to re-describe (its memory is aliased)
        /// @param[in] rows         rows per item, or \f$\le 0\f$ to keep
        /// @param[in] cols         columns per item, or \f$\le 0\f$ to keep
        /// @param[in] ld           leading dimension, or \f$\le 0\f$ to keep
        /// @param[in] stride       batch stride, or \f$\le 0\f$ to keep
        /// @param[in] batch_size   number of items, or \f$\le 0\f$ to keep
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView(const MatrixView<T, MType>& matrix_view,
                  int rows = -1, int cols = -1, int ld = -1, int stride = -1, int batch_size = -1);

        /// @brief An empty \f$0 \times 0\f$ view with batch size 0.
        MatrixView() = default;

        /// @brief Enqueues the copy @p dest \f$\leftarrow\f$ @p src on @p ctx, honouring each side's ld and stride.
        /// @param[in]  ctx   queue the copy is enqueued on
        /// @param[out] dest  destination batch; every element in its rows() \f$\times\f$ cols() block of each active
        ///                   item is written (masked-out items are left untouched)
        /// @param[in]  src   source batch, same shape and batch size as @p dest
        /// @return the event of the copy; a no-op when both views share a base pointer
        /// @throws batchlas::invalid_argument if rows, cols, batch sizes or attached batch masks differ
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
        /// @param[in] i  batch item index
        /// @return a batch-size-1 view of item @p i
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
        /// @param[in] ctx  queue whose device gets access
        /// @return the event of the hint
        Event set_access_device(const Queue& ctx) const {
            if constexpr (MType == MatrixFormat::CSR) {
                (void)row_offsets_.set_access_device(ctx);
                (void)col_indices_.set_access_device(ctx);
            }
            return data_.set_access_device(ctx);
        }

        /// @brief Migrates the viewed memory to the device of @p ctx.
        /// @param[in] ctx  queue the prefetch is enqueued on
        /// @return the event of the prefetch
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
        /// @param[in] batch_index  batch item index
        /// @return the item's active rows (rows() when homogeneous)
        int rows(int batch_index) const {
            return detail::dense_active_rows(rows_, active_rows_, batch_index);
        }
        /// @brief Columns per item (the capacity on a heterogeneous batch).
        int cols() const { return cols_; }
        /// @brief Active columns of item @p batch_index.
        /// @param[in] batch_index  batch item index
        /// @return the item's active columns (cols() when homogeneous)
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

        /// @name Batch mask
        /// One byte per item, nonzero = active. An op skips an inactive item and leaves its
        /// outputs bit-for-bit untouched. The mask is not owned and may be device-only
        /// (`sycl::malloc_device`): the library never reads it on the host.
        /// @{

        /// @brief A copy of this view with @p mask attached (not copied: it must outlive the view);
        ///        an empty span clears the mask.
        /// @param[in] mask  batch_size() bytes, nonzero = active; empty to clear
        /// @return the masked view
        /// @throws batchlas::invalid_argument unless @p mask is empty or holds batch_size() bytes
        ///         (only the length is checked; the bytes are never read)
        MatrixView<T, MType> with_batch_mask(Span<const std::uint8_t> mask) const {
            detail::check_batch_mask_length(mask.size(), batch_size_);
            MatrixView<T, MType> out = *this;
            out.batch_mask_ = mask;
            return out;
        }
        /// @brief with_batch_mask() for a mutable span (a Matrix's batch_mask(), a UnifiedVector's to_span()).
        /// @param[in] mask  batch_size() bytes, nonzero = active; empty to clear
        /// @return the masked view
        MatrixView<T, MType> with_batch_mask(Span<std::uint8_t> mask) const {
            return with_batch_mask(Span<const std::uint8_t>(mask.data(), mask.size()));
        }
        /// @brief The attached mask; empty when unmasked.
        Span<const std::uint8_t> batch_mask() const { return batch_mask_; }
        /// @brief True when a mask is attached.
        bool has_batch_mask() const { return !batch_mask_.empty(); }
        /// @}

        /// @brief Packed dense copy with every element converted by `static_cast<U>`; host loop, no Queue.
        ///        A batch mask is copied into the result's own (enable_batch_mask()) on the host.
        /// @tparam U  element type of the result
        /// @return a packed owning copy
        /// @throws batchlas::unsupported for a device-only (device USM) batch mask
        template <typename U>
        Matrix<U, MType> astype() const;

        /// @name Verification
        /// Accuracy checks of a computed result against its inputs (@ref design_verification). Each
        /// reads every operand through ld() and stride(), promotes to double / complex<double>, runs
        /// on @p ctx (an in-order sibling of it when @p ctx is out of order), waits, and returns the
        /// worst value over @p items (empty: default_items(); no items checked returns 0); a NaN
        /// anywhere is the result. Operands are read as stored: apply triangularize(), hermitize()
        /// or symmetrize() to a copy first to check against a triangle or a mirrored matrix.
        /// A masked operand is refused: every check, product() and form_q() throw batchlas::unsupported
        /// when any operand carries a batch mask.
        /// @pre every operand is host-readable (unified or host memory) and the work that wrote it
        ///      has completed: the caller waits on its queue first.
        /// @{

        /// @brief T's promoted type: double, or complex<double> for complex T.
        using promoted_type = std::conditional_t<is_std_complex_v<T>, std::complex<double>, double>;

        /// @brief @p x as promoted_type.
        /// @param[in] x  the value to promote
        /// @return @p x in double or complex<double>
        template <typename U = T>
            requires(std::is_floating_point_v<U> || is_std_complex_v<U>)
        static promoted_type up(U x) { return promoted_type(x); }

        /// @brief \f$\mathrm{re} + i\,\mathrm{im}\f$ rounded to T; the imaginary part is dropped for a real T.
        /// @param[in] re  real part
        /// @param[in] im  imaginary part (ignored for a real T)
        /// @return the value in T's precision
        template <typename U = T>
            requires(std::is_floating_point_v<U> || is_std_complex_v<U>)
        static U make(double re, double im = 0.0) {
            if constexpr (is_std_complex_v<U>) return U(float_t<U>(re), float_t<U>(im));
            else { (void)im; return U(re); }
        }

        /// @brief The complex conjugate \f$\bar{x}\f$ of @p x; @p x itself for a real T.
        /// @param[in] x  the value to conjugate
        /// @return \f$\bar{x}\f$
        template <typename U = T>
            requires(std::is_floating_point_v<U> || is_std_complex_v<U>)
        static U conj(U x) {
            if constexpr (is_std_complex_v<U>) return std::conj(x);
            else return x;
        }

        /// @brief \f$\lvert \mathrm{Re}\,x \rvert + \lvert \mathrm{Im}\,x \rvert\f$, LAPACK's partial-pivoting metric;
        ///        \f$\lvert x \rvert\f$ for a real T.
        /// @param[in] x  the value to measure
        /// @return the 1-norm of @p x's components, in double
        template <typename U = T>
            requires(std::is_floating_point_v<U> || is_std_complex_v<U>)
        static double cabs1(U x) {
            if constexpr (is_std_complex_v<U>) return std::fabs(double(x.real())) + std::fabs(double(x.imag()));
            else return std::fabs(double(x));
        }

        /// @brief True when every component of @p x is finite.
        /// @param[in] x  the value to test
        /// @return whether neither part is infinite or NaN
        template <typename U = T>
            requires(std::is_floating_point_v<U> || is_std_complex_v<U>)
        static bool finite(U x) {
            if constexpr (is_std_complex_v<U>) return std::isfinite(double(x.real())) && std::isfinite(double(x.imag()));
            else return std::isfinite(double(x));
        }

        /// @brief Every item index \f$0, 1, \dots, b - 1\f$, with \f$b\f$ = batch_size().
        /// @return the indices in ascending order
        std::vector<int> all_items() const;

        /// @brief \f$\{0, \lfloor b/2 \rfloor, b - 1\}\f$ with \f$b\f$ = batch_size(), sorted and de-duplicated
        ///        (\f$\emptyset\f$ for batch 0): what a check reads by default.
        /// @return the default item indices
        std::vector<int> default_items() const;

        /// @brief astype() of @p items only, packed in that order (item \f$g\f$ at offset
        ///        \f$g \cdot \mathrm{rows} \cdot \mathrm{cols}\f$).
        /// @tparam U  element type of the result
        /// @param[in] items  batch item indices to copy, in output order
        /// @return a packed owning copy of batch size `items.size()`
        /// @throws batchlas::out_of_range for an item outside \f$[0, \mathrm{batch\_size})\f$
        /// @throws batchlas::invalid_argument for a heterogeneous view
        /// @throws batchlas::unsupported when a batch mask is attached
        template <typename U>
        Matrix<U, MType> astype(std::span<const int> items) const;

        /// @brief \f$C := \mathrm{op}(A)\,\mathrm{op}(B)\f$ per item: one work-item per element of C, the inner sum in order.
        ///        The reference product of the checks, independent of gemm's tuned kernels.
        /// @param[in]  ctx  queue the kernel is enqueued on
        /// @param[in]  A    batch of \f$m \times k\f$ \f$\mathrm{op}(A)\f$; batch size 1 or C's (1 is shared by every item)
        /// @param[in]  ta   op applied to @p A
        /// @param[in]  B    batch of \f$k \times n\f$ \f$\mathrm{op}(B)\f$; batch size 1 or C's
        /// @param[in]  tb   op applied to @p B
        /// @param[out] C    batch of \f$m \times n\f$ products, overwritten
        /// @return the event of the kernel
        /// @throws batchlas::invalid_argument on a shape or batch-size mismatch
        /// @throws batchlas::unsupported if any operand carries a batch mask, or for double data on a device without fp64
        static Event product(const Queue& ctx, const MatrixView& A, Transpose ta, const MatrixView& B, Transpose tb,
                             const MatrixView& C);

        /// @brief \f$\lVert A_0 - L L^H \rVert_F / \lVert A_0 \rVert_F\f$ (Lower) or
        ///        \f$\lVert A_0 - U^H U \rVert_F / \lVert A_0 \rVert_F\f$ (Upper) over the factor's triangle;
        ///        the other triangle of @p F is not read.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original \f$n \times n\f$ matrices
        /// @param[in] F      potrf output: the Cholesky factor in its @p uplo triangle
        /// @param[in] uplo   triangle holding the factor
        /// @param[in] items  batch items to check; empty means A0's default_items()
        /// @return the worst relative residual over @p items
        static double potrf_residual(Queue& ctx, const MatrixView& A0, const MatrixView& F, Uplo uplo,
                                     std::span<const int> items = {});

        /// @brief \f$\lVert P A_0 - L U \rVert_F / \lVert A_0 \rVert_F\f$ for an \f$m \times n\f$ factor with packed
        ///        1-based pivots (\f$\min(m, n)\f$ per item); an out-of-range pivot makes that item NaN.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original \f$m \times n\f$ matrices
        /// @param[in] F      getrf output: unit-lower L below the diagonal, U on and above it
        /// @param[in] piv    1-based pivots, \f$\min(m, n)\f$ per item
        /// @param[in] items  batch items to check; empty means A0's default_items()
        /// @return the worst relative residual over @p items
        static double getrf_residual(Queue& ctx, const MatrixView& A0, const MatrixView& F,
                                     const VectorView<std::int32_t>& piv, std::span<const int> items = {});

        /// @brief \f$\lVert A_0 - Q R \rVert_F / \lVert A_0 \rVert_F\f$ for @p F in LAPACK geqrf storage and its
        ///        \f$\min(m, n)\f$ @p tau.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original \f$m \times n\f$ matrices
        /// @param[in] F      geqrf output: R on and above the diagonal, reflectors below
        /// @param[in] tau    reflector scalars, \f$\min(m, n)\f$ per item
        /// @param[in] items  batch items to check; empty means A0's default_items()
        /// @return the worst relative residual over @p items
        static double qr_residual(Queue& ctx, const MatrixView& A0, const MatrixView& F, const VectorView<T>& tau,
                                  std::span<const int> items = {});

        /// @brief The first @p cols columns of \f$Q = H_0 H_1 \cdots H_{k-1}\f$ of item @p item of a geqrf
        ///        factor (\f$k = \min(m, n)\f$); F's diagonal and upper triangle are not read.
        /// @param[in] ctx   queue the reconstruction runs on
        /// @param[in] F     geqrf output holding the reflectors below its diagonal
        /// @param[in] tau   reflector scalars, \f$\min(m, n)\f$ per item
        /// @param[in] item  batch item to form
        /// @param[in] cols  number of leading columns of Q to form
        /// @return an \f$m \times \mathrm{cols}\f$ matrix in promoted_type
        static Matrix<promoted_type, MType> form_q(Queue& ctx, const MatrixView& F, const VectorView<T>& tau, int item,
                                                   int cols);

        /// @brief form_q() of each of @p items (empty: default_items()), packed in that order: item \f$g\f$ of
        ///        the result is the first @p cols columns of Q of item `items[g]`.
        /// @param[in] ctx    queue the reconstruction runs on
        /// @param[in] F      geqrf output holding the reflectors below its diagonal
        /// @param[in] tau    reflector scalars, \f$\min(m, n)\f$ per item
        /// @param[in] items  batch items to form; empty means F's default_items()
        /// @param[in] cols   number of leading columns of Q to form
        /// @return one \f$m \times \mathrm{cols}\f$ matrix per entry of @p items, in promoted_type
        static Matrix<promoted_type, MType> form_q(Queue& ctx, const MatrixView& F, const VectorView<T>& tau,
                                                   std::span<const int> items, int cols);

        /// @brief \f$\lVert A_0 - Q\,\mathrm{triu}(R) \rVert_F / \lVert A_0 \rVert_F\f$ with an explicit
        ///        \f$m \times k\f$ @p Q; the first \f$k\f$ rows of @p R are read on and above the diagonal.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original \f$m \times n\f$ matrices
        /// @param[in] Q      batch of explicit \f$m \times k\f$ factors
        /// @param[in] R      batch whose leading \f$k \times n\f$ upper triangle is the R factor
        /// @param[in] items  batch items to check; empty means A0's default_items()
        /// @return the worst relative residual over @p items
        static double qr_reconstruction(Queue& ctx, const MatrixView& A0, const MatrixView& Q, const MatrixView& R,
                                        std::span<const int> items = {});

        /// @brief \f$\lVert A_0 - Q B Q^H \rVert_F / \lVert A_0 \rVert_F\f$ for square operands of equal order.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original \f$n \times n\f$ matrices
        /// @param[in] Q      batch of \f$n \times n\f$ transforms
        /// @param[in] B      batch of \f$n \times n\f$ reduced matrices, read whole
        /// @param[in] items  batch items to check; empty means A0's default_items()
        /// @return the worst relative residual over @p items
        static double similarity_residual(Queue& ctx, const MatrixView& A0, const MatrixView& Q, const MatrixView& B,
                                          std::span<const int> items = {});

        /// @brief \f$\lVert A_0 - U \operatorname{diag}(s) V^H \rVert_F / \lVert A_0 \rVert_F\f$ for an
        ///        \f$m \times n\f$ @p A0 over the first \f$k = \min(m, n)\f$ columns of @p U, values of @p s
        ///        and rows of @p Vh (thin or full factors).
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original \f$m \times n\f$ matrices
        /// @param[in] U      left singular vectors, at least \f$k\f$ columns
        /// @param[in] s      singular values, at least \f$k\f$ per item
        /// @param[in] Vh     \f$V^H\f$: right singular vectors as rows, at least \f$k\f$ rows
        /// @param[in] items  batch items to check; empty means A0's default_items()
        /// @return the worst relative residual over @p items
        /// @throws batchlas::invalid_argument on a shape mismatch
        static double svd_residual(Queue& ctx, const MatrixView& A0, const MatrixView& U, const VectorView<float_t<T>>& s,
                                   const MatrixView& Vh, std::span<const int> items = {});

        /// @brief \f$\lVert \mathrm{op}(A_0) X - B_0 \rVert_F / (\lVert A_0 \rVert_F \lVert X \rVert_F)\f$; an empty
        ///        @p B0 (\f$\mathrm{rows} = 0\f$) is the identity.
        ///        Above @p full_limit multiply-adds per item, 8 seeded probe columns of X stand in for all.
        /// @param[in] ctx         queue the check runs on
        /// @param[in] A0          batch of the original coefficient matrices
        /// @param[in] ta          op applied to @p A0
        /// @param[in] X           batch of computed solutions
        /// @param[in] B0          batch of the original right-hand sides, or an empty view for the identity
        /// @param[in] items       batch items to check; empty means A0's default_items()
        /// @param[in] full_limit  per-item multiply-add count above which X is probed instead of used whole
        /// @return the worst relative residual over @p items
        static double solve_residual(Queue& ctx, const MatrixView& A0, Transpose ta, const MatrixView& X,
                                     const MatrixView& B0, std::span<const int> items = {},
                                     double full_limit = double(1 << 28));

        /// @brief solve_residual() with \f$\mathrm{op}(A_0) = A_0\f$.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original coefficient matrices
        /// @param[in] X      batch of computed solutions
        /// @param[in] B0     batch of the original right-hand sides, or an empty view for the identity
        /// @param[in] items  batch items to check; empty means A0's default_items()
        /// @return the worst relative residual over @p items
        static double solve_residual(Queue& ctx, const MatrixView& A0, const MatrixView& X, const MatrixView& B0,
                                     std::span<const int> items = {});

        /// @brief solve_residual() with \f$A = P L U\f$ formed per item from getrf factors @p F and pivots @p piv;
        ///        @p full_limit as for solve_residual().
        /// @param[in] ctx         queue the check runs on
        /// @param[in] F           getrf output: unit-lower L below the diagonal, U on and above it
        /// @param[in] piv         1-based pivots from getrf
        /// @param[in] ta          op applied to \f$A\f$
        /// @param[in] X           batch of computed solutions
        /// @param[in] B0          batch of the original right-hand sides, or an empty view for the identity
        /// @param[in] items       batch items to check; empty means F's default_items()
        /// @param[in] full_limit  per-item multiply-add count above which X is probed instead of used whole
        /// @return the worst relative residual over @p items
        static double lu_solve_residual(Queue& ctx, const MatrixView& F, const VectorView<std::int32_t>& piv,
                                        Transpose ta, const MatrixView& X, const MatrixView& B0,
                                        std::span<const int> items = {}, double full_limit = double(1 << 28));

        /// @brief Componentwise backward error of a computed gemm result @p C.
        ///
        /// \f[
        ///   \max_{i,j} \frac{\bigl\lvert C - (\alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C_0) \bigr\rvert_{ij}}
        ///                   {\bigl(\lvert\alpha\rvert\,\lvert\mathrm{op}(A)\rvert\,\lvert\mathrm{op}(B)\rvert
        ///                    + \lvert\beta\rvert\,\lvert C_0\rvert\bigr)_{ij}}
        /// \f]
        /// with \f$\lvert\cdot\rvert\f$ taken elementwise, over the @p c_triangle of C when given;
        /// @p C0 is not read when \f$\beta = 0\f$.
        /// @param[in] ctx         queue the check runs on
        /// @param[in] A           batch of the A operand
        /// @param[in] ta          op applied to @p A
        /// @param[in] B           batch of the B operand
        /// @param[in] tb          op applied to @p B
        /// @param[in] C0          C before the update (not read when \f$\beta = 0\f$)
        /// @param[in] C           the computed result
        /// @param[in] alpha       scalar \f$\alpha\f$
        /// @param[in] beta        scalar \f$\beta\f$
        /// @param[in] items       batch items to check; empty means C's default_items()
        /// @param[in] c_triangle  restrict the check to this triangle of C; nullopt checks all of C
        /// @return the worst componentwise backward error over @p items
        static double gemm_backward_error(Queue& ctx, const MatrixView& A, Transpose ta, const MatrixView& B, Transpose tb,
                                          const MatrixView& C0, const MatrixView& C, promoted_type alpha,
                                          promoted_type beta, std::span<const int> items = {},
                                          std::optional<Uplo> c_triangle = std::nullopt);

        /// @brief The largest denominator of gemm_backward_error() over @p items (empty: A's default_items(),
        ///        while gemm_backward_error() defaults to C's: pass both the same span when A's batch differs).
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A      batch of the A operand
        /// @param[in] ta     op applied to @p A
        /// @param[in] B      batch of the B operand
        /// @param[in] tb     op applied to @p B
        /// @param[in] C0     C before the update (not read when \f$\beta = 0\f$)
        /// @param[in] alpha  scalar \f$\alpha\f$
        /// @param[in] beta   scalar \f$\beta\f$
        /// @param[in] items  batch items to scan; empty means A's default_items()
        /// @return the largest elementwise denominator over @p items
        static double gemm_max_denominator(Queue& ctx, const MatrixView& A, Transpose ta, const MatrixView& B,
                                           Transpose tb, const MatrixView& C0, promoted_type alpha, promoted_type beta,
                                           std::span<const int> items = {});

        /// @brief gemm_backward_error() of \f$y = \alpha\,\mathrm{op}(A)\,x + \beta y_0\f$.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A      batch of matrices
        /// @param[in] ta     op applied to @p A
        /// @param[in] x      batch of input vectors
        /// @param[in] y0     y before the update (not read when \f$\beta = 0\f$)
        /// @param[in] y      the computed result
        /// @param[in] alpha  scalar \f$\alpha\f$
        /// @param[in] beta   scalar \f$\beta\f$
        /// @param[in] items  batch items to check; empty means y's default_items()
        /// @return the worst componentwise backward error over @p items
        static double gemv_backward_error(Queue& ctx, const MatrixView& A, Transpose ta, const VectorView<T>& x,
                                          const VectorView<T>& y0, const VectorView<T>& y, promoted_type alpha,
                                          promoted_type beta, std::span<const int> items = {});

        /// @brief \f$\lVert \mathrm{op}(A) X - \alpha B_0 \rVert_F / (\lVert A \rVert_F \lVert X \rVert_F + \lvert\alpha\rvert \lVert B_0 \rVert_F)\f$
        ///        (Side::Right: \f$X\,\mathrm{op}(A)\f$); only the @p uplo triangle of A is read, its diagonal
        ///        only for Diag::NonUnit.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A      batch of triangular matrices
        /// @param[in] side   Left (\f$\mathrm{op}(A) X\f$) or Right (\f$X\,\mathrm{op}(A)\f$)
        /// @param[in] uplo   triangle of @p A that is read
        /// @param[in] ta     op applied to @p A
        /// @param[in] diag   Unit (diagonal taken as 1) or NonUnit
        /// @param[in] X      batch of computed solutions
        /// @param[in] B0     batch of the original right-hand sides
        /// @param[in] alpha  scalar \f$\alpha\f$
        /// @param[in] items  batch items to check; empty means X's default_items()
        /// @return the worst relative residual over @p items
        static double trsm_residual(Queue& ctx, const MatrixView& A, Side side, Uplo uplo, Transpose ta, Diag diag,
                                    const MatrixView& X, const MatrixView& B0, promoted_type alpha,
                                    std::span<const int> items = {});

        /// @brief gemm_backward_error() with a CSR @p A; an out-of-range column index makes the result NaN.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A      batch of sparse matrices
        /// @param[in] ta     op applied to @p A
        /// @param[in] B      batch of dense operands
        /// @param[in] tb     op applied to @p B
        /// @param[in] C0     C before the update (not read when \f$\beta = 0\f$)
        /// @param[in] C      the computed result
        /// @param[in] alpha  scalar \f$\alpha\f$
        /// @param[in] beta   scalar \f$\beta\f$
        /// @param[in] items  batch items to check; empty means C's default_items()
        /// @return the worst componentwise backward error over @p items
        static double spmm_backward_error(Queue& ctx, const MatrixView<T, MatrixFormat::CSR>& A, Transpose ta,
                                          const MatrixView& B, Transpose tb, const MatrixView& C0, const MatrixView& C,
                                          promoted_type alpha, promoted_type beta, std::span<const int> items = {});

        /// @brief Componentwise backward error of a rank-2k update over the @p uplo triangle of C: her2k
        ///        (@p hermitian, real \f$\beta\f$, C0's diagonal read as real) or syr2k.
        /// @param[in] ctx        queue the check runs on
        /// @param[in] A          batch of the A operand
        /// @param[in] B          batch of the B operand, same shape as @p A
        /// @param[in] trans      op applied to @p A and @p B (her2k: NoTrans or ConjTrans; syr2k: NoTrans or Trans)
        /// @param[in] C0         C before the update (not read when \f$\beta = 0\f$)
        /// @param[in] C          the computed \f$n \times n\f$ result
        /// @param[in] uplo       triangle of C that is checked
        /// @param[in] alpha      scalar \f$\alpha\f$
        /// @param[in] beta       scalar \f$\beta\f$ (real for her2k)
        /// @param[in] hermitian  true for her2k, false for syr2k
        /// @param[in] items      batch items to check; empty means C's default_items()
        /// @return the worst componentwise backward error over @p items
        static double rank2k_backward_error(Queue& ctx, const MatrixView& A, const MatrixView& B, Transpose trans,
                                            const MatrixView& C0, const MatrixView& C, Uplo uplo, promoted_type alpha,
                                            promoted_type beta, bool hermitian, std::span<const int> items = {});

        /// @brief \f$\lVert Q^H Q - I \rVert_F\f$ over Q's columns.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] Q      batch of matrices with (nominally) orthonormal columns
        /// @param[in] items  batch items to check; empty means Q's default_items()
        /// @return the worst loss of orthogonality over @p items
        static double orthogonality(Queue& ctx, const MatrixView& Q, std::span<const int> items = {});

        /// @brief \f$\lVert A V - V \operatorname{diag}(w) \rVert_F / \lVert A \rVert_F\f$; A Hermitian, read from
        ///        its lower triangle.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A      batch of the original \f$n \times n\f$ Hermitian matrices
        /// @param[in] V      batch of eigenvectors, one per column
        /// @param[in] w      eigenvalues, one per column of @p V
        /// @param[in] items  batch items to check; empty means V's default_items()
        /// @return the worst relative residual over @p items
        static double eigen_residual(Queue& ctx, const MatrixView& A, const MatrixView& V, const VectorView<float_t<T>>& w,
                                     std::span<const int> items = {});

        /// @brief The generalized eigen residual of @p itype over X's columns, with \f$W = \operatorname{diag}(w)\f$.
        ///
        /// - itype 1 (\f$A X = B X W\f$): \f$\lVert A_0 X - B_0 X W \rVert_F / (\lVert A_0 \rVert_F \lVert X \rVert_F)\f$
        /// - itype 2 (\f$A B X = X W\f$): \f$\lVert A_0 (B_0 X) - X W \rVert_F / (\lVert A_0 \rVert_F \lVert B_0 X \rVert_F)\f$
        /// - itype 3 (\f$B A X = X W\f$): \f$\lVert B_0 (A_0 X) - X W \rVert_F / (\lVert B_0 \rVert_F \lVert A_0 X \rVert_F)\f$
        ///
        /// A0 and B0 are read whole; an empty @p B0 (\f$\mathrm{rows} = 0\f$) is the identity, every itype then the
        /// standard problem.
        /// @param[in] ctx    queue the check runs on
        /// @param[in] A0     batch of the original \f$n \times n\f$ matrices A
        /// @param[in] B0     batch of the original \f$n \times n\f$ matrices B, or an empty view for the identity
        /// @param[in] X      batch of eigenvectors, one per column
        /// @param[in] w      eigenvalues, one per column of @p X
        /// @param[in] itype  problem type: 1, 2 or 3 as above
        /// @param[in] items  batch items to check; empty means X's default_items()
        /// @return the worst relative residual over @p items
        /// @throws batchlas::invalid_argument for an itype other than 1, 2 or 3, or a dimension mismatch
        static double generalized_eigen_residual(Queue& ctx, const MatrixView& A0, const MatrixView& B0, const MatrixView& X,
                                                 const VectorView<float_t<T>>& w, int itype = 1,
                                                 std::span<const int> items = {});

        /// @brief \f$\max_i \lvert w_i - \mathrm{ref}_i \rvert / \mathrm{scale}\f$ per item; `ref[item]` holds
        ///        `w.size()` values in the caller's order.
        /// @param[in] w      batch of computed values
        /// @param[in] ref    reference values, indexed by batch item, then by entry
        /// @param[in] scale  divisor of every absolute difference
        /// @param[in] items  batch items to check; empty means w's default_items()
        /// @return the worst scaled error over @p items
        static double values_error(const VectorView<float_t<T>>& w, const std::vector<std::vector<double>>& ref,
                                   double scale, std::span<const int> items = {});

        /// @brief \f$\max_{i > k} \mathrm{cabs1}(L_{ik} U_{kk}) / \mathrm{cabs1}(U_{kk})\f$ over a getrf factor: at
        ///        most 1, up to rounding, for cabs1 partial pivoting (a residual cannot see the pivot rule).
        /// @param[in] F      getrf output: unit-lower L below the diagonal, U on and above it
        /// @param[in] items  batch items to check; empty means F's default_items()
        /// @return the worst ratio over @p items (columns with \f$U_{kk} = 0\f$ are skipped)
        static double pivot_ratio(const MatrixView& F, std::span<const int> items = {});

        /// @brief True when every pivot of every item lies in \f$[1, n]\f$.
        /// @param[in] piv  1-based pivots
        /// @param[in] n    largest valid pivot (the matrix order)
        /// @return whether all pivots are in range
        static bool pivots_valid(const VectorView<std::int32_t>& piv, int n);
        /// @}

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
        /// @param[in] batch_index  batch item index
        /// @return the item's stored non-zero count
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
        /// @param[in] ctx  queue the fill runs on (waited on before returning)
        /// @return the filled pointer array, one base pointer per item
        /// @throws batchlas::invalid_argument if the view has no pointer array of batch_size() entries
        Span<T*> data_ptrs(Queue& ctx) const {
            init_data_ptr_array(ctx);
            return data_ptrs_;
        }

        /// @brief Host reference to element (@p row, @p col) of item @p batch.
        /// @param[in] row    row index
        /// @param[in] col    column index
        /// @param[in] batch  batch item index
        /// @return a reference to the element
        /// @throws batchlas::out_of_range outside the item's active extents
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        T& at(int row, int col, int batch = 0);

        /// @brief Host const reference to element (@p row, @p col) of item @p batch.
        /// @param[in] row    row index
        /// @param[in] col    column index
        /// @param[in] batch  batch item index
        /// @return a const reference to the element
        /// @throws batchlas::out_of_range outside the item's active extents
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        const T& at(int row, int col, int batch = 0) const;

        /// @brief at(@p row, @p col, @p batch).
        /// @param[in] row    row index
        /// @param[in] col    column index
        /// @param[in] batch  batch item index
        /// @return a reference to the element
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        T& operator()(int row, int col, int batch = 0) {
            return at(row, col, batch);
        }

        /// @brief at(@p row, @p col, @p batch).
        /// @param[in] row    row index
        /// @param[in] col    column index
        /// @param[in] batch  batch item index
        /// @return a const reference to the element
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        const T& operator()(int row, int col, int batch = 0) const {
            return at(row, col, batch);
        }

        /// @brief View of the sub-block selected by @p rows and @p cols of every item; keeps ld, stride and
        ///        the batch mask; never copies.
        /// @param[in] rows  rows to keep
        /// @param[in] cols  columns to keep
        /// @return the sub-block view
        /// @throws batchlas::invalid_argument if either slice is empty; batchlas::out_of_range past a dimension
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> operator()(Slice rows, Slice cols) const {
            auto [r_start, r_len] = detail::checked_slice_component(rows, rows_);
            auto [c_start, c_len] = detail::checked_slice_component(cols, cols_);
            if (r_len <= 0 || c_len <= 0) {
                throw batchlas::invalid_argument("Invalid slice dimensions: " + std::to_string(r_len) + "x" + std::to_string(c_len));
            }
            auto offset = c_start * ld_ + r_start;
            // Trap: the slice keeps the parent's pointer array, which holds *unsliced* base
            // addresses until data_ptrs(ctx) refills it for this view.
            // evidence: docs/design/matrix-model.md#matrix-model-the-pointer-array-and-sliced-views
            MatrixView<T, MType> out(data_ptr() + offset, static_cast<int>(r_len), static_cast<int>(c_len), ld_, stride_, batch_size_, data_ptrs_.data());
            out.batch_mask_ = batch_mask_;
            return out;
        }

        /// @brief View of rows @p rows over all columns of every item.
        /// @param[in] rows  rows to keep
        /// @return the sub-block view
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> operator()(Slice rows) const {
            return (*this)(rows, {});
        }

        /// @brief Part of row @p row of every item as a batched VectorView (inc = ld(), stride = stride()).
        /// @param[in] row   row index
        /// @param[in] cols  columns to keep
        /// @return a view of the row segment of every item
        /// @throws batchlas::invalid_argument if the slice is empty; batchlas::out_of_range past a dimension
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        VectorView<T> operator()(int32_t row, Slice cols) const {
            auto [c_start, c_len] = detail::checked_slice_component(cols, cols_);
            if (c_len <= 0) {
                throw batchlas::invalid_argument("Invalid slice dimensions for row vector: " + std::to_string(c_len));
            }
            auto offset = c_start * ld_ + row;
            return VectorView<T>(data_ptr() + offset, static_cast<int>(c_len), batch_size_, ld_, stride_)
                .with_batch_mask(batch_mask_);
        }

        /// @brief Part of column @p col of every item as a batched VectorView (inc = 1, stride = stride()).
        /// @param[in] rows  rows to keep
        /// @param[in] col   column index
        /// @return a view of the column segment of every item
        /// @throws batchlas::invalid_argument if the slice is empty; batchlas::out_of_range past a dimension
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        VectorView<T> operator()(Slice rows, int32_t col) const {
            auto [r_start, r_len] = detail::checked_slice_component(rows, rows_);
            if (r_len <= 0) {
                throw batchlas::invalid_argument("Invalid slice dimensions for column vector: " + std::to_string(r_len));
            }
            auto offset = col * ld_ + r_start;
            return VectorView<T>(data_ptr() + offset, static_cast<int>(r_len), batch_size_, 1, stride_)
                .with_batch_mask(batch_mask_);
        }

        /// @name In-place fills and transforms
        /// Dense-only unless stated. The `const Queue&` overloads enqueue and return the
        /// kernel's Event without waiting; the queue-less overloads run on a new Queue()
        /// on Device::default_device_for<T>(). The square transforms read rows() as n.
        /// @{

        /// @brief Copies the @p uplo triangle onto the other: \f$a_{ji} := a_{ij}\f$ for \f$(i, j)\f$ in @p uplo.
        /// @param[in] ctx   queue the kernel is enqueued on
        /// @param[in] uplo  triangle that is kept and mirrored
        /// @return the event of the kernel
        /// @pre \f$\mathrm{rows} = \mathrm{cols}\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event symmetrize(const Queue& ctx, Uplo uplo) const;

        /// @brief symmetrize() on a new default queue.
        /// @param[in] uplo  triangle that is kept and mirrored
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event symmetrize(Uplo uplo) const {
            return symmetrize(Queue(Device::default_device_for<T>()), uplo);
        }

        /// @brief Mirrors the @p uplo triangle conjugated (\f$a_{ji} := \overline{a_{ij}}\f$) and makes the diagonal real;
        ///        symmetrize() for real T.
        /// @param[in] ctx   queue the kernel is enqueued on
        /// @param[in] uplo  triangle that is kept and mirrored
        /// @return the event of the kernel
        /// @pre \f$\mathrm{rows} = \mathrm{cols}\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event hermitize(const Queue& ctx, Uplo uplo) const;

        /// @brief hermitize() on a new default queue.
        /// @param[in] uplo  triangle that is kept and mirrored
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event hermitize(Uplo uplo) const {
            return hermitize(Queue(Device::default_device_for<T>()), uplo);
        }

        /// @brief In-place (conjugate) transpose.
        /// @note Declared but not defined by the library: a call fails to link.
        /// @param[in] ctx        queue the kernel would be enqueued on
        /// @param[in] conjugate  true for \f$A^H\f$, false for \f$A^T\f$
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event transpose(const Queue& ctx, bool conjugate = false) const;

        /// @brief transpose() on a new default queue; likewise undefined.
        /// @param[in] conjugate  true for \f$A^H\f$, false for \f$A^T\f$
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event transpose(bool conjugate = false) const {
            return transpose(Queue(Device::default_device_for<T>()), conjugate);
        }

        /// @brief Zeros the strict triangle opposite @p uplo, keeping the @p uplo triangle;
        ///        with Diag::Unit also sets the diagonal to 1.
        ///
        /// `Uplo::Upper` leaves an upper-triangular matrix. Honours \f$\mathrm{rows} \ne \mathrm{cols}\f$.
        /// @param[in] ctx   queue the kernel is enqueued on
        /// @param[in] uplo  triangle that is kept
        /// @param[in] diag  Unit sets the diagonal to 1; NonUnit keeps it
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event triangularize(const Queue& ctx, Uplo uplo, Diag diag) const;

        /// @brief triangularize() on a new default queue.
        /// @param[in] uplo  triangle that is kept
        /// @param[in] diag  Unit sets the diagonal to 1; NonUnit keeps it
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event triangularize(Uplo uplo, Diag diag) const {
            return triangularize(Queue(Device::default_device_for<T>()), uplo, diag);
        }

        /// @brief Fills with values uniform in \f$[-1, 1]\f$, deterministic in @p seed; see Matrix::Random().
        /// @note Writes every element of data(), padding between columns and items included.
        /// @param[in] ctx        queue the kernel is enqueued on
        /// @param[in] hermitian  make every item Hermitian (lower triangle from the conjugated upper, real diagonal)
        /// @param[in] seed       RNG seed
        /// @return the event of the kernel
        /// @throws batchlas::invalid_argument if @p hermitian and \f$\mathrm{rows} \ne \mathrm{cols}\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random(const Queue& ctx, bool hermitian = false, unsigned int seed = 42) const;

        /// @brief fill_random() on a new default queue.
        /// @param[in] hermitian  make every item Hermitian
        /// @param[in] seed       RNG seed
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random(bool hermitian = false, unsigned int seed = 42) const {
            return fill_random(Queue(Device::default_device_for<T>()), hermitian, seed);
        }

        /// @brief CSR: fills this storage with a random diagonally dominant Hermitian pattern and values.
        ///
        /// See csr_generators::random_sparse_hermitian_csr() for the construction.
        /// Blocks on @p ctx while building the pattern; the returned Event is the value fill.
        /// @param[in] ctx             queue the pattern build and value fill run on
        /// @param[in] density         target fraction of stored entries over the full \f$n \times n\f$ matrix
        /// @param[in] seed            RNG seed
        /// @param[in] diagonal_boost  margin added to each diagonal above its off-diagonal row sum
        /// @param[in] shared_pattern  true: every item has the same sparsity pattern; false: one per item
        /// @return the event of the value fill
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
        /// @param[in] density         target fraction of stored entries over the full \f$n \times n\f$ matrix
        /// @param[in] seed            RNG seed
        /// @param[in] diagonal_boost  margin added to each diagonal above its off-diagonal row sum
        /// @param[in] shared_pattern  true: every item has the same sparsity pattern; false: one per item
        /// @return the event of the value fill
        template <MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Event fill_random_sparse_hermitian(float density,
                                           unsigned int seed = 42,
                                           typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                           bool shared_pattern = true) const {
            return fill_random_sparse_hermitian(Queue(Device::default_device_for<T>()), density, seed, diagonal_boost, shared_pattern);
        }

        /// @brief Sets every element of every item (dense: within rows() \f$\times\f$ cols(); CSR: every stored value).
        /// @param[in] ctx    queue the kernel is enqueued on
        /// @param[in] value  the value written
        /// @return the event of the kernel
        Event fill(const Queue& ctx, T value) const;

        /// @brief fill() on a new default queue.
        /// @param[in] value  the value written
        /// @return the event of the kernel
        Event fill(T value) const {
            return fill(Queue(Device::default_device_for<T>()), value);
        }

        /// @brief fill() with zero. Deliberately has no queue-less form, which would run on the default device.
        /// @param[in] ctx  queue the kernel is enqueued on
        /// @return the event of the kernel
        Event fill_zeros(const Queue& ctx) const {
            return fill(ctx, detail::convert_to_fill_value<T>(0));
        }

        /// @brief Zeros every item, then sets its main diagonal to @p value.
        /// @param[in] ctx    queue the kernel is enqueued on
        /// @param[in] value  the diagonal value
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_identity(const Queue& ctx, T value = T(1)) const;

        /// @brief Writes diagonal @p k (\f$k > 0\f$ above the main diagonal, \f$k < 0\f$ below).
        ///
        /// With \f$m\f$ = rows() and \f$n\f$ = cols(), the diagonal has
        /// \f[
        ///   d = \max\bigl(0,\; \min(m - \max(-k, 0),\; n - \max(k, 0))\bigr)
        /// \f]
        /// entries. @p diag_values holds either \f$d\f$ values used for every item, or
        /// \f$d \cdot \mathrm{batch\_size}\f$ values, one run per item. Other entries are untouched;
        /// \f$d = 0\f$ writes nothing.
        /// @param[in] ctx          queue the kernel is enqueued on
        /// @param[in] diag_values  \f$d\f$ or \f$d \cdot \mathrm{batch\_size}\f$ values
        /// @param[in] k            diagonal offset
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Queue& ctx, const Span<T>& diag_values, int64_t k = 0) const;

        /// @brief Writes diagonal @p k from a VectorView: per item when its batch size equals
        ///        batch_size(), otherwise its item 0 for every item.
        /// @param[in] ctx          queue the kernel is enqueued on
        /// @param[in] diag_values  at least \f$d\f$ entries per item, \f$d\f$ as in the Span overload
        /// @param[in] k            diagonal offset
        /// @return the event of the kernel
        /// @pre @p diag_values holds at least \f$d\f$ entries, \f$d\f$ as in the Span overload
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Queue& ctx, const VectorView<T>& diag_values, int64_t k = 0) const;

        /// @brief fill_diagonal() on a new default queue.
        /// @param[in] diag_values  \f$d\f$ or \f$d \cdot \mathrm{batch\_size}\f$ values
        /// @param[in] k            diagonal offset
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Span<T>& diag_values, int64_t k = 0) const {
            return fill_diagonal(Queue(Device::default_device_for<T>()), diag_values, k);
        }

        /// @brief Sets the main diagonal of every item to @p value.
        /// @param[in] ctx    queue the kernel is enqueued on
        /// @param[in] value  the diagonal value
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const Queue& ctx, const T& value) const;

        /// @brief fill_diagonal() on a new default queue.
        /// @param[in] value  the diagonal value
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_diagonal(const T& value) const {
            return fill_diagonal(Queue(Device::default_device_for<T>()), value);
        }

        /// @brief Writes a full triangular matrix: @p diagonal_value on the diagonal,
        ///        @p non_diagonal_value in the @p uplo strict triangle, zero elsewhere.
        /// @param[in] ctx                 queue the kernel is enqueued on
        /// @param[in] uplo                triangle that receives @p non_diagonal_value
        /// @param[in] diagonal_value      value on the diagonal
        /// @param[in] non_diagonal_value  value in the strict triangle
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular(const Queue& ctx, Uplo uplo, T diagonal_value = T(1),
                              T non_diagonal_value = T(0.5)) const;

        /// @brief fill_triangular() on a new default queue.
        /// @param[in] uplo                triangle that receives @p non_diagonal_value
        /// @param[in] diagonal_value      value on the diagonal
        /// @param[in] non_diagonal_value  value in the strict triangle
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular(Uplo uplo, T diagonal_value = T(1),
                              T non_diagonal_value = T(0.5)) const {
            return fill_triangular(Queue(Device::default_device_for<T>()), uplo, diagonal_value, non_diagonal_value);
        }

        /// @brief Writes the three diagonals from per-item vectors; other entries are untouched.
        /// @param[in] ctx         queue the three kernels are enqueued on
        /// @param[in] sub_diag    subdiagonal, \f$n - 1\f$ entries per item
        /// @param[in] diag        main diagonal, \f$n\f$ entries per item
        /// @param[in] super_diag  superdiagonal, \f$n - 1\f$ entries per item
        /// @return the event of the last kernel
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
        /// @param[in] ctx         queue the kernel is enqueued on
        /// @param[in] diag        value on the main diagonal
        /// @param[in] sub_diag    value on the subdiagonal
        /// @param[in] super_diag  value on the superdiagonal
        /// @return the event of the kernel
        /// @pre The view is packed: \f$\mathrm{ld} = \mathrm{rows} = \mathrm{cols}\f$,
        ///      \f$\mathrm{stride} = \mathrm{rows} \cdot \mathrm{cols}\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_tridiag_toeplitz(const Queue& ctx, T diag = T(1),
                                    T sub_diag = T(-0.5), T super_diag = T(0.5)) const;

        /// @brief fill_tridiag_toeplitz() on a new default queue.
        /// @param[in] diag        value on the main diagonal
        /// @param[in] sub_diag    value on the subdiagonal
        /// @param[in] super_diag  value on the superdiagonal
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_tridiag_toeplitz(T diag = T(1),
                                    T sub_diag = T(-0.5), T super_diag = T(0.5)) const {
            return fill_tridiag_toeplitz(Queue(Device::default_device_for<T>()), diag, sub_diag, super_diag);
        }

        /// @brief Writes a full random triangular matrix: uniform \f$[-1, 1]\f$ in the @p uplo
        ///        triangle, 1 on the diagonal for Diag::Unit, zero elsewhere.
        /// @note Every item receives the same matrix (the values depend on @p seed and the position only).
        /// @param[in] ctx   queue the kernel is enqueued on
        /// @param[in] uplo  triangle that receives the random values
        /// @param[in] diag  Unit writes 1 on the diagonal; NonUnit a random value
        /// @param[in] seed  RNG seed
        /// @return the event of the kernel
        /// @pre The view is packed: \f$\mathrm{ld} = \mathrm{rows} = \mathrm{cols}\f$,
        ///      \f$\mathrm{stride} = \mathrm{rows} \cdot \mathrm{cols}\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular_random(const Queue& ctx, Uplo uplo,
                                     Diag diag = Diag::Unit,
                                     unsigned int seed = 42) const;

        /// @brief fill_triangular_random() on a new default queue.
        /// @param[in] uplo  triangle that receives the random values
        /// @param[in] diag  Unit writes 1 on the diagonal; NonUnit a random value
        /// @param[in] seed  RNG seed
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_triangular_random(Uplo uplo,
                                     Diag diag = Diag::Unit,
                                     unsigned int seed = 42) const {
            return fill_triangular_random(Queue(Device::default_device_for<T>()), uplo, diag, seed);
        }

        /// @brief Random Hermitian (real T: symmetric) positive definite matrix, a different one per item.
        ///
        /// The Hermitian fill_random() values with the shift \f$n\f$ (\f$2n\f$ for complex T, whose entries reach
        /// modulus \f$\sqrt{2}\f$) added to the real diagonal: every row is strictly diagonally dominant
        /// with a positive diagonal. \f$a_{ij} = \overline{a_{ji}}\f$ holds bitwise. Deterministic in @p seed,
        /// independent of ld() and stride(); writes only the rows() \f$\times\f$ cols() entries.
        /// @param[in] ctx   queue the kernel is enqueued on
        /// @param[in] seed  RNG seed
        /// @return the event of the kernel
        /// @throws batchlas::invalid_argument if \f$\mathrm{rows} \ne \mathrm{cols}\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random_spd(const Queue& ctx, unsigned int seed = 42) const;

        /// @brief fill_random_spd() on a new default queue.
        /// @param[in] seed  RNG seed
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random_spd(unsigned int seed = 42) const {
            return fill_random_spd(Queue(Device::default_device_for<T>()), seed);
        }

        /// @brief Row-permuted diagonally dominant random matrix, so LU partial pivoting is not the identity.
        ///
        /// Item \f$b\f$ is \f$P_b D_b\f$: \f$D_b\f$ holds uniform \f$[-1, 1)\f$ entries plus the fill_random_spd()
        /// shift on the diagonal, and \f$P_b\f$ is a seeded random \f$n\f$-cycle (no row stays in place for
        /// \f$n \ge 2\f$), a different one per item. Each column's dominant entry therefore sits in a distinct row.
        /// Deterministic in @p seed; writes only the rows() \f$\times\f$ cols() entries.
        /// @param[in] ctx   queue the kernel is enqueued on
        /// @param[in] seed  RNG seed
        /// @return the event of the kernel
        /// @throws batchlas::invalid_argument if \f$\mathrm{rows} \ne \mathrm{cols}\f$, or if \f$n\f$ ints exceed the device's local memory
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random_pivoted(const Queue& ctx, unsigned int seed = 42) const;

        /// @brief fill_random_pivoted() on a new default queue.
        /// @param[in] seed  RNG seed
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_random_pivoted(unsigned int seed = 42) const {
            return fill_random_pivoted(Queue(Device::default_device_for<T>()), seed);
        }

        /// @brief Hermitian \f$Q \operatorname{diag}(\lambda) Q^H\f$ with condition number \f$10^{c}\f$,
        ///        \f$c\f$ = @p log10_cond, a different Q per item.
        ///
        /// \f$\lambda_i = 10^{-c\, i / (n - 1)}\f$ for \f$i = 0, \dots, n - 1\f$ (\f$\lambda_0 = 1\f$ when \f$n = 1\f$).
        /// Q is the product of \f$n\f$ Householder reflectors \f$I - 2 v v^H\f$ of seeded random unit vectors
        /// (real for real T), applied in T's precision. The result is exactly Hermitian with a real diagonal.
        /// Deterministic in @p seed; writes only the rows() \f$\times\f$ cols() entries.
        /// @param[in] ctx         queue the kernel is enqueued on
        /// @param[in] log10_cond  \f$c = \log_{10}\kappa\f$ of the target condition number \f$\kappa\f$; finite
        /// @param[in] seed        RNG seed
        /// @return the event of the kernel
        /// @throws batchlas::invalid_argument if \f$\mathrm{rows} \ne \mathrm{cols}\f$, if @p log10_cond is not finite,
        ///         or if \f$2n\f$ elements of T exceed the device's local memory
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_graded_hermitian(const Queue& ctx, double log10_cond, unsigned int seed = 42) const;

        /// @brief fill_graded_hermitian() on a new default queue.
        /// @param[in] log10_cond  \f$\log_{10}\f$ of the target condition number; finite
        /// @param[in] seed        RNG seed
        /// @return the event of the kernel
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_graded_hermitian(double log10_cond, unsigned int seed = 42) const {
            return fill_graded_hermitian(Queue(Device::default_device_for<T>()), log10_cond, seed);
        }

        /// @brief Clustered-spectrum Hermitian matrix: diagonal \f$\lfloor i/4 \rfloor + (i \bmod 4) \cdot 10^{-4}\f$
        ///        plus @p noise times a seeded Hermitian perturbation.
        ///
        /// With \f$\eta\f$ = @p noise, off-diagonal \f$a_{ij} = \eta\, z_{ij}\f$ (\f$i > j\f$, \f$z\f$ uniform
        /// \f$[-1, 1)\f$, both parts for complex T) and \f$a_{ji} = \overline{a_{ij}}\f$ bitwise; the diagonal adds
        /// \f$\eta\, \mathrm{Re}(z_{ii})\f$. Groups of four nearly equal eigenvalues exercise deflation and
        /// tie-breaking. Deterministic in @p seed (Philox, the same on every compiler), independent of ld()
        /// and stride(); writes only the rows() \f$\times\f$ cols() entries.
        /// @param[in] ctx    queue the kernel is enqueued on
        /// @param[in] noise  perturbation scale \f$\eta\f$
        /// @param[in] seed   RNG seed
        /// @return the event of the kernel
        /// @throws batchlas::invalid_argument if \f$\mathrm{rows} \ne \mathrm{cols}\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Event fill_near_degenerate_hermitian(const Queue& ctx, double noise, unsigned int seed = 42) const;
        /// @}


        /// @brief A batch-size-1 view of item @p batch_index.
        ///
        /// Dense: its active extents, the parent's ld, no pointer array. CSR: the parent's strides.
        /// A mask becomes the item's one byte (`batch_mask().subspan(batch_index, 1)`).
        /// @param[in] batch_index  batch item index
        /// @return the single-item view
        /// @throws batchlas::out_of_range if @p batch_index is outside \f$[0, \mathrm{batch\_size})\f$
        MatrixView<T, MType> batch_item(int batch_index) const;

        /// @brief Items \f$[\mathrm{first}, \mathrm{first} + \mathrm{count})\f$ as one view: the same extents, ld
        ///        and stride, the matching range of any active extents and of the batch mask, no pointer array.
        /// @param[in] first  index of the first item
        /// @param[in] count  number of items, at least 1
        /// @return a view of batch size @p count
        /// @throws batchlas::out_of_range if \f$\mathrm{count} < 1\f$ or the range leaves \f$[0, \mathrm{batch\_size})\f$
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        MatrixView<T, MType> batch_range(int first, int count) const {
            if (first < 0 || count < 1 || static_cast<int64_t>(first) + count > batch_size_)
                throw batchlas::out_of_range("batch_range [" + std::to_string(first) + ", " +
                                             std::to_string(static_cast<int64_t>(first) + count) +
                                             ") is outside [0, " + std::to_string(batch_size_) + ")");
            MatrixView<T, MType> out(data_ptr() + static_cast<int64_t>(first) * stride_, rows_, cols_, ld_,
                                     stride_, count);
            if (!active_rows_.empty()) out.active_rows_ = active_rows_.subspan(first, count);
            if (!active_cols_.empty()) out.active_cols_ = active_cols_.subspan(first, count);
            if (!batch_mask_.empty()) out.batch_mask_ = batch_mask_.subspan(first, count);
            return out;
        }

        /// @brief A copy of this view with per-item active extents attached (not copied: the spans must outlive the view).
        /// @param[in] active_rows  per-item active row counts, batch_size() entries each in \f$[0, \mathrm{rows}]\f$
        /// @param[in] active_cols  per-item active column counts, batch_size() entries each in \f$[0, \mathrm{cols}]\f$
        /// @return the heterogeneous view
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
        /// Elements agree when
        /// \f[
        ///   \frac{\lvert x - y \rvert}{\max(\lvert x \rvert, \lvert y \rvert)} \le \mathrm{tol}
        /// \f]
        /// (the denominator is 1 when the larger magnitude is below @p tol, and two values both below
        /// \f$\varepsilon\f$ agree); complex compares the real and imaginary parts separately. Dense only.
        /// @param[in] ctx  queue the comparison runs on (waited on)
        /// @param[in] a    first batch
        /// @param[in] b    second batch, same shape and batch size as @p a
        /// @param[in] tol  relative tolerance; defaults to the machine epsilon \f$\varepsilon\f$ of float_t<T>
        /// @return whether every element pair agrees
        /// @throws batchlas::invalid_argument if the shapes or batch sizes differ
        /// @throws batchlas::unsupported for CSR
        static bool all_close(Queue& ctx, const MatrixView<T, MType>& a, const MatrixView<T, MType>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon());

        /// @brief Prints each item's leading block (dense) or leading entries (CSR) in scientific notation.
        /// @param[in,out] os                         stream written to; its format flags are restored, but its
        ///                                           precision is left at 4
        /// @param[in]     max_rows_to_print          dense: rows printed per item
        /// @param[in]     max_cols_to_print          dense: columns printed per item
        /// @param[in]     max_elements_to_print_csr  CSR: values, offsets and indices printed per item
        /// @return @p os, with its format flags restored and its precision set to 4
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
        /// @param[in,out] os                         stream written to
        /// @param[in]     max_rows_to_print          dense: rows printed per item
        /// @param[in]     max_cols_to_print          dense: columns printed per item
        /// @param[in]     max_elements_to_print_csr  CSR: values, offsets and indices printed per item
        void print(std::ostream& os = std::cout, int max_rows_to_print = 10, int max_cols_to_print = 10, int max_elements_to_print_csr = 20) const {
            stream_formatted_to(os, max_rows_to_print, max_cols_to_print, max_elements_to_print_csr);
        }

        /// @brief The trivially copyable device-side view, for capture in a kernel.
        /// @return a KernelMatrixView over the same memory, extents, active extents and batch mask
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
            kv.mask_       = batch_mask_.empty() ? nullptr : batch_mask_.data();
            if constexpr (MType == MatrixFormat::CSR) {
                kv.row_offsets_   = const_cast<int*>(row_offsets_.data());
                kv.col_indices_   = const_cast<int*>(col_indices_.data());
                kv.nnz_           = nnz_;
                kv.matrix_stride_ = matrix_stride_;
                kv.offset_stride_ = offset_stride_;
            }
            return kv;
        }

    protected:
        /// @brief Writes `data_ptr() + b * stride()` into slot \f$b\f$ of a pointer array, for every item, and waits.
        /// @param[in]  ctx     queue the fill runs on (waited on before returning)
        /// @param[out] target  array of at least batch_size() pointers; empty means this view's own pointer array
        /// @throws batchlas::invalid_argument if the array is null or shorter than batch_size(), or the view has no data
        void init_data_ptr_array(Queue& ctx, Span<T*> target = {}) const;

        Span<T> data_;

        int ld_ = 0;
        int stride_ = 0;
        Span<int> active_rows_;
        Span<int> active_cols_;
        // Empty = unmasked. Possibly device-only: never dereferenced on the host.
        Span<const std::uint8_t> batch_mask_;

        Span<T*> data_ptrs_;  // may alias the parent Matrix's pointer array

        Span<int> row_offsets_;
        Span<int> col_indices_;
        int nnz_ = 0;
        int matrix_stride_ = 0;
        int offset_stride_ = 0;

        // Shared with every copy of the view; mutable so const methods can create it lazily.
        mutable std::shared_ptr<BackendMatrixHandle<T, MType>> backend_handle_;
    };

    /// @brief Owning batch of `batch_size()` matrices of equal (capacity) shape.
    ///
    /// **Dense** (the default): column-major, element \f$(i, j)\f$ of item \f$b\f$ at
    /// `data()[b * stride() + j * ld() + i]`, with \f$\mathrm{ld} \ge \mathrm{rows}\f$ and
    /// \f$\mathrm{stride} \ge \mathrm{ld} \cdot \mathrm{cols}\f$. **CSR**: per item, \f$\mathrm{rows} + 1\f$ row offsets
    /// (items `offset_stride()` apart) and `nnz_capacity()` values and column indices
    /// (items `matrix_stride()` apart), zero-based.
    ///
    /// Storage is USM shared memory (UnifiedVector) on Device::default_device(), so it
    /// is host-accessible and device-accessible. A Matrix is a MatrixView of its own
    /// storage, so it is accepted wherever a MatrixView is and has every view member.
    /// Copy construction and copy assignment deep-copy the data; moves transfer it.
    /// The shape members are public but are not to be written.
    ///
    /// A dense batch may be **heterogeneous**: set_active_dims() attaches per-item
    /// active extents no larger than the allocated rows() \f$\times\f$ cols().
    /// @tparam T      scalar type (`float`, `double`, `std::complex<float>`, `std::complex<double>`)
    /// @tparam MType  MatrixFormat::Dense (default) or MatrixFormat::CSR
    /// @see @ref design_matrix_model, and the user guide's data-layout section in @ref md_docs_2cpp-api
    /// @ingroup api_matrix
    // BATCHLAS_API sits on the class template, not on the explicit instantiations: only it
    // reaches the member-template instantiations, and g++ rejects it on an instantiation.
    // evidence: docs/design/symbol-visibility.md#symbol-visibility-the-export-attribute-on-the-matrix-class-template
    template <typename T, MatrixFormat MType>
    class BATCHLAS_API Matrix : public MatrixView<T, MType> {
        using Base = MatrixView<T, MType>;
        using Base::ld_, Base::stride_, Base::nnz_, Base::matrix_stride_, Base::offset_stride_;

    public:
        using Base::rows_, Base::cols_, Base::batch_size_;


        /// @brief Allocates an uninitialised dense batch.
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] batch_size  number of items
        /// @param[in] ld          leading dimension; 0 means @p rows
        /// @param[in] stride      batch stride in elements; 0 means \f$\mathrm{ld} \cdot \mathrm{cols}\f$ with the resolved ld
        /// @throws batchlas::invalid_argument on a negative argument, a resolved \f$\mathrm{ld} < \mathrm{rows}\f$,
        ///         or a resolved \f$\mathrm{stride} < \mathrm{ld} \cdot \mathrm{cols}\f$
        /// @note Exactly \f$\mathrm{stride} \cdot \mathrm{batch\_size}\f$ elements are allocated; nothing pads.
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(int rows, int cols, int batch_size = 1, int ld = 0, int stride = 0);

        /// @brief Allocates an uninitialised CSR batch with room for @p nnz entries per item.
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] nnz         per-item capacity; \f$\mathrm{matrix\_stride} = \mathrm{nnz}\f$,
        ///                        \f$\mathrm{offset\_stride} = \mathrm{rows} + 1\f$
        /// @param[in] batch_size  number of items
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(int rows, int cols, NonZeros nnz, int batch_size = 1);

        /// @brief Deleted: spell the count `NonZeros{nnz}`; for a dense Matrix the third int is the batch size.
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] nnz         the count, which must be spelled `NonZeros{nnz}`
        /// @param[in] batch_size  number of items
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(int rows, int cols, int nnz, int batch_size = 1) = delete;

        /// @brief Dense batch copied from existing host-readable memory.
        ///
        /// (@p ld, @p stride) describe the *source*: element \f$(i, j)\f$ of item \f$b\f$ is read from
        /// `data[b * stride + j * ld + i]`. The copy keeps the source's leading
        /// dimension but packs the items back to back (\f$\mathrm{stride} = \mathrm{ld} \cdot \mathrm{cols}\f$);
        /// padding rows in the copy are zeroed and padding in the source is never read.
        /// @param[in] data        source; must hold
        ///                        \f$(\mathrm{batch\_size} - 1)\,\mathrm{stride} + (\mathrm{cols} - 1)\,\mathrm{ld} + \mathrm{rows}\f$ elements
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] ld          source leading dimension; 0 means @p rows
        /// @param[in] stride      source batch stride; 0 means \f$\mathrm{ld} \cdot \mathrm{cols}\f$ with the resolved ld
        /// @param[in] batch_size  number of items
        /// @throws batchlas::invalid_argument on null @p data, \f$\mathrm{rows} \le 0\f$ or \f$\mathrm{cols} \le 0\f$,
        ///         \f$\mathrm{batch\_size} \le 0\f$, a resolved \f$\mathrm{ld} < \mathrm{rows}\f$, or
        ///         (\f$\mathrm{batch\_size} > 1\f$) a \f$\mathrm{stride} < \mathrm{ld} \cdot \mathrm{cols}\f$
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(const T* data, int rows, int cols, int ld, int stride = 0, int batch_size = 1);

        /// @brief Dense batch copied from a Span; as the pointer overload, plus a length check.
        /// @param[in] data        source elements, laid out as described by @p ld and @p stride
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] ld          source leading dimension; 0 means @p rows
        /// @param[in] stride      source batch stride; 0 means \f$\mathrm{ld} \cdot \mathrm{cols}\f$ with the resolved ld
        /// @param[in] batch_size  number of items
        /// @throws batchlas::invalid_argument also when @p data is shorter than the layout needs
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(Span<const T> data, int rows, int cols, int ld, int stride = 0, int batch_size = 1);

        /// @brief Mutable-Span (and UnifiedVector) spelling of the Span<const T> overload.
        /// @param[in] data        source elements, laid out as described by @p ld and @p stride (only read)
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] ld          source leading dimension; 0 means @p rows
        /// @param[in] stride      source batch stride; 0 means \f$\mathrm{ld} \cdot \mathrm{cols}\f$ with the resolved ld
        /// @param[in] batch_size  number of items
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix(Span<T> data, int rows, int cols, int ld, int stride = 0, int batch_size = 1)
            : Matrix(Span<const T>(data.data(), data.size()), rows, cols, ld, stride, batch_size) {}

        /// @brief CSR batch copied from existing host-readable arrays.
        /// @param[in] values         \f$\mathrm{matrix\_stride} \cdot \mathrm{batch\_size}\f$ values
        /// @param[in] row_offsets    \f$\mathrm{offset\_stride} \cdot \mathrm{batch\_size}\f$ zero-based offsets
        /// @param[in] col_indices    \f$\mathrm{matrix\_stride} \cdot \mathrm{batch\_size}\f$ column indices
        /// @param[in] rows           rows per item
        /// @param[in] cols           columns per item
        /// @param[in] nnz            declared per-item non-zero count
        /// @param[in] matrix_stride  distance between items' value arrays; 0 means @p nnz. May exceed it.
        /// @param[in] offset_stride  distance between items' offset arrays; 0 means \f$\mathrm{rows} + 1\f$
        /// @param[in] batch_size     number of items
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(const T* values, const int* row_offsets, const int* col_indices,
               int rows, int cols, NonZeros nnz, int matrix_stride = 0,
               int offset_stride = 0, int batch_size = 1);

        /// @brief Deleted so the pre-tag order (nnz before rows, cols) is a compile error.
        /// @param[in] values         values
        /// @param[in] row_offsets    row offsets
        /// @param[in] col_indices    column indices
        /// @param[in] nnz            the count, which must be spelled `NonZeros{nnz}` after @p cols
        /// @param[in] rows           rows per item
        /// @param[in] cols           columns per item
        /// @param[in] matrix_stride  distance between items' value arrays
        /// @param[in] offset_stride  distance between items' offset arrays
        /// @param[in] batch_size     number of items
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        Matrix(const T* values, const int* row_offsets, const int* col_indices,
               int nnz, int rows, int cols, int matrix_stride = 0,
               int offset_stride = 0, int batch_size = 1) = delete;


        /// @name Factories
        /// Each returns a packed dense batch (\f$\mathrm{ld} = \mathrm{rows}\f$,
        /// \f$\mathrm{stride} = \mathrm{rows} \cdot \mathrm{cols}\f$) and waits for its fill kernel before returning.
        /// @{

        /// @brief A batch of \f$n \times n\f$ identity matrices.
        /// @param[in] n           matrix order
        /// @param[in] batch_size  number of items
        /// @return the new batch
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Identity(int n, int batch_size = 1);

        /// @brief Entries uniform in \f$[-1, 1]\f$ (real and imaginary parts independently).
        ///
        /// Deterministic: the same @p seed and shape give the same matrix.
        /// With @p hermitian, the lower triangle is overwritten with the conjugate of
        /// the upper and the diagonal made real.
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] hermitian   make every item Hermitian
        /// @param[in] batch_size  number of items
        /// @param[in] seed        RNG seed
        /// @return the new batch
        /// @throws batchlas::invalid_argument if @p hermitian and \f$\mathrm{rows} \ne \mathrm{cols}\f$
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Random(int rows, int cols, bool hermitian = false, int batch_size = 1, unsigned int seed = 42);

        /// @brief Random diagonally dominant sparse Hermitian batch; see csr_generators::random_sparse_hermitian_csr().
        /// @param[in] n               matrix order
        /// @param[in] density         target fraction of stored entries over the full \f$n \times n\f$ matrix
        /// @param[in] batch_size      number of items
        /// @param[in] seed            RNG seed
        /// @param[in] diagonal_boost  margin added to each diagonal above its off-diagonal row sum
        /// @param[in] shared_pattern  true: every item has the same sparsity pattern; false: one per item
        /// @return the new CSR batch
        template <typename U = T, MatrixFormat M = MType>
            requires CsrMatrixFormat<M>
        static Matrix<T, MType> RandomSparseHermitian(int n,
                                  float density,
                                  int batch_size = 1,
                                  unsigned int seed = 42,
                                  typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                  bool shared_pattern = true);

        /// @brief Random \f$n \times n\f$ triangular batch; see MatrixView::fill_triangular_random().
        /// @param[in] n           matrix order
        /// @param[in] uplo        triangle that receives the random values
        /// @param[in] diag        Unit writes 1 on the diagonal; NonUnit a random value
        /// @param[in] batch_size  number of items
        /// @param[in] seed        RNG seed
        /// @return the new batch
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> RandomTriangular(int n, Uplo uplo, Diag diag = Diag::NonUnit, int batch_size = 1, unsigned int seed = 42) {
            auto result = Matrix<T, MType>(n, n, batch_size);
            result.fill_triangular_random(uplo, diag, seed).wait();
            return result;
        }

        /// @brief A batch of zero matrices.
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] batch_size  number of items
        /// @return the new batch
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Zeros(int rows, int cols, int batch_size = 1);

        /// @brief A batch of all-ones matrices.
        /// @param[in] rows        rows per item
        /// @param[in] cols        columns per item
        /// @param[in] batch_size  number of items
        /// @return the new batch
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Ones(int rows, int cols, int batch_size = 1);

        /// @brief A batch of \f$n \times n\f$ diagonal matrices, \f$n\f$ = `diag_values.size()`, each with
        ///        @p diag_values on the diagonal.
        /// @param[in] diag_values  the \f$n\f$ diagonal entries, shared by every item
        /// @param[in] batch_size   number of items
        /// @return the new batch
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Diagonal(const Span<T>& diag_values, int batch_size = 1);

        /// @brief \f$n \times n\f$ matrices with @p diagonal_value on the diagonal, @p non_diagonal_value in
        ///        the @p uplo strict triangle and zero elsewhere.
        /// @param[in] n                   matrix order
        /// @param[in] uplo                triangle that receives @p non_diagonal_value
        /// @param[in] diagonal_value      value on the diagonal
        /// @param[in] non_diagonal_value  value in the strict triangle
        /// @param[in] batch_size          number of items
        /// @return the new batch
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> Triangular(int n, Uplo uplo, T diagonal_value = T(1),
                                          T non_diagonal_value = T(0.5), int batch_size = 1);

        /// @brief \f$n \times n\f$ tridiagonal Toeplitz matrices (constant diagonal, sub- and super-diagonal).
        /// @param[in] n           matrix order
        /// @param[in] diag        value on the main diagonal
        /// @param[in] sub_diag    value on the subdiagonal
        /// @param[in] super_diag  value on the superdiagonal
        /// @param[in] batch_size  number of items
        /// @return the new batch
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        static Matrix<T, MType> TriDiagToeplitz(int n, T diag = T(1),
                                                T sub_diag = T(-0.5), T super_diag = T(0.5), int batch_size = 1);
        /// @}

        /// @brief Reinterprets this matrix's storage as row-major data and returns it column-major.
        ///
        /// The row pitch is a parameter, never inferred. The result is packed and owns
        /// the memory the kernel writes, so the call waits on @p ctx before returning.
        /// @param[in] ctx        queue for the conversion kernel
        /// @param[in] row_pitch  source row pitch in elements; 0 means packed (cols()), accepted
        ///                       only when this matrix is itself packed
        /// @return a packed column-major copy
        /// @throws std::invalid_argument when the pitch is below cols(), runs past the
        ///         allocation, straddles the next item, or is 0 on a non-packed matrix
        /// @see @ref md_docs_2cpp-api, section "Row-major source data"
        // evidence: docs/cpp-api.md#row-major-source-data
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType> to_column_major(const Queue& ctx, int row_pitch = 0) const;

        /// @brief to_column_major() on an internal queue.
        /// @param[in] row_pitch  source row pitch in elements; 0 means packed (cols())
        /// @return a packed column-major copy
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
        /// @param[in] zero_threshold  magnitude at or below which a dense entry is not stored
        /// @return the copy in format @p NewMType
        /// @throws batchlas::unsupported for any other pair of formats
        template <MatrixFormat NewMType>
        Matrix<T, NewMType> convert_to(const float_t<T>& zero_threshold = 1e-7) const;

        /// @brief Packed row-major copy (row pitch cols(), batch stride rows() \f$\cdot\f$ cols()).
        ///
        /// Exactly what to_column_major() reads with its default pitch, so the two
        /// round-trip. Waits on @p ctx before returning.
        /// @param[in] ctx  queue for the conversion kernel
        /// @return the row-major copy, held in a Matrix of the same shape
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType> to_row_major(const Queue& ctx) const;

        /// @brief to_row_major() on an internal queue.
        /// @return the row-major copy, held in a Matrix of the same shape
        template <typename U = T, MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType> to_row_major() const;

        /// @name Copy and move
        /// Copies are deep (values, active extents, CSR arrays) and get their own backend
        /// handle; moves transfer the storage and leave the source empty.
        /// @{
        /// @brief Deep copy of @p other.
        /// @param[in] other  the matrix to copy
        Matrix(const Matrix& other)
            : Base(other), values_(other.values_), ptrs_(other.ptrs_), active_rows_store_(other.active_rows_store_),
              active_cols_store_(other.active_cols_store_), row_offsets_store_(other.row_offsets_store_),
              col_indices_store_(other.col_indices_store_), batch_mask_store_(other.batch_mask_store_) {
            this->backend_handle_.reset();
            for (std::size_t b = 0; b < ptrs_.size(); ++b) ptrs_[b] = values_.data() + static_cast<std::ptrdiff_t>(b) * stride_;
            bind();
        }
        /// @brief Takes over the storage of @p other.
        /// @param[in,out] other  the matrix moved from; its storage is released, but its shape accessors still
        ///                       report the old extents (only destroy or assign to it)
        Matrix(Matrix&& other) noexcept : Base() { *this = std::move(other); }
        /// @brief Replaces this matrix with a deep copy of @p other.
        /// @param[in] other  the matrix to copy
        /// @return `*this`
        Matrix& operator=(const Matrix& other) { return *this = Matrix(other); }
        /// @brief Replaces this matrix with the storage of @p other.
        /// @param[in,out] other  the matrix moved from; its storage is released, but its shape accessors still
        ///                       report the old extents (only destroy or assign to it)
        /// @return `*this`
        Matrix& operator=(Matrix&& other) noexcept {
            Base::operator=(std::move(other));
            values_ = std::move(other.values_);
            ptrs_ = std::move(other.ptrs_);
            active_rows_store_ = std::move(other.active_rows_store_);
            active_cols_store_ = std::move(other.active_cols_store_);
            row_offsets_store_ = std::move(other.row_offsets_store_);
            col_indices_store_ = std::move(other.col_indices_store_);
            batch_mask_store_ = std::move(other.batch_mask_store_);
            bind();
            other.bind();
            return *this;
        }
        /// @}

        /// @brief Deep copy in this matrix's own layout (ld, stride, CSR strides, active extents, batch mask).
        /// @return the independent copy
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

            std::copy(values_.begin(), values_.end(), result.values_.begin());
            if constexpr (MType == MatrixFormat::CSR) {
                std::copy(row_offsets_store_.begin(), row_offsets_store_.end(), result.row_offsets_store_.begin());
                std::copy(col_indices_store_.begin(), col_indices_store_.end(), result.col_indices_store_.begin());
                result.nnz_ = nnz_;
                result.matrix_stride_ = matrix_stride_;
                result.offset_stride_ = offset_stride_;
            } else if (this->is_heterogeneous()) {
                result.set_active_dims(this->active_rows(), this->active_cols());
            }
            if (batch_mask_store_.size() != 0) {
                result.batch_mask_store_ = batch_mask_store_;
                result.bind();
            }
            return result;
        }

        /// @brief This matrix as a plain MatrixView (a sliced copy); shares the backend handle.
        /// @return a view of this matrix's storage
        MatrixView<T, MType> view() const { return *this; }
        /// @brief A dense view of the leading @p rows \f$\times\f$ @p cols block of every item.
        /// @param[in] rows    rows of the view
        /// @param[in] cols    columns of the view
        /// @param[in] ld      leading dimension of the view; \f$\le 0\f$ keeps ld()
        /// @param[in] stride  batch stride of the view; \f$\le 0\f$ keeps stride()
        /// @return a view of this matrix's storage
        /// @throws batchlas::unsupported on a CSR matrix
        MatrixView<T, MType> view(int rows, int cols, int ld = -1, int stride = -1) const;

        /// @brief Dense copy with every element converted by `static_cast<U>`, in this layout; a batch mask is copied as by MatrixView::astype().
        /// @tparam U  element type of the result
        /// @return an owning copy with the same ld and stride
        template <typename U>
        Matrix<U, MType> astype() const;
        using Base::astype;

        /// @brief Sets every element of every item to @p value; waits.
        /// @param[in] value  the value written
        void fill(T value) { Base::fill(value).wait(); }
        using Base::fill;

        /// @brief Host reference to element (@p row, @p col) of item @p batch anywhere in the
        ///        allocated rows() \f$\times\f$ cols(), active extents ignored (MatrixView's checks them).
        ///
        /// No bounds check beyond UnifiedVector's debug assert. The data is USM shared,
        /// so wait for any kernel writing it first.
        /// @param[in] row    row index
        /// @param[in] col    column index
        /// @param[in] batch  batch item index
        /// @return a reference to the element
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        T& operator()(int row, int col, int batch) {
            // The batch term in int64_t; see KernelMatrixView::operator() for why only that one.
            return values_.at(static_cast<int64_t>(batch) * stride_ + col * ld_ + row);
        }

        /// @brief Host const reference to element (@p row, @p col) of item @p batch.
        /// @param[in] row    row index
        /// @param[in] col    column index
        /// @param[in] batch  batch item index
        /// @return a const reference to the element
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        const T& operator()(int row, int col, int batch) const {
            return values_.at(static_cast<int64_t>(batch) * stride_ + col * ld_ + row);
        }
        using Base::operator();

        /// @brief Host copy of @p src into this matrix's own layout.
        /// @param[in] src  the batch to copy, same rows, cols and batch size; host-readable
        /// @pre For CSR, @p src has exactly this matrix's strides and capacity.
        /// @throws batchlas::invalid_argument if rows, cols or batch size differ
        void copy_from(const MatrixView<T, MType>& src);

        /// @brief Attaches per-item active extents (copied), or clears them when both spans are empty.
        ///
        /// Item \f$b\f$ then occupies the leading \f$\mathrm{active\_rows}[b] \times \mathrm{active\_cols}[b]\f$ block of
        /// its allocated rows() \f$\times\f$ cols(); layout (ld, stride) is unchanged.
        /// @param[in] active_rows  per-item active row counts, batch_size() entries each in \f$[0, \mathrm{rows}]\f$
        /// @param[in] active_cols  per-item active column counts, batch_size() entries each in \f$[0, \mathrm{cols}]\f$
        /// @return `*this`
        /// @throws batchlas::invalid_argument unless both spans hold batch_size() entries,
        ///         each in \f$[0, \mathrm{rows}]\f$ and \f$[0, \mathrm{cols}]\f$ respectively
        template <MatrixFormat M = MType>
            requires DenseMatrixFormat<M>
        Matrix<T, MType>& set_active_dims(Span<int> active_rows, Span<int> active_cols) {
            detail::validate_dense_active_dims(rows_, cols_, batch_size_, active_rows, active_cols);
            active_rows_store_.resize(active_rows.size());
            active_cols_store_.resize(active_cols.size());
            std::copy(active_rows.begin(), active_rows.end(), active_rows_store_.begin());
            std::copy(active_cols.begin(), active_cols.end(), active_cols_store_.begin());
            bind();
            return *this;
        }

        /// @brief Allocates an owned batch mask (USM shared, batch_size() bytes) if there is none, sets every
        ///        item active, and attaches it to this matrix and every view taken from it afterwards.
        /// @return the writable mask: write it on the host, or from a kernel on any queue of the device
        Span<std::uint8_t> enable_batch_mask() {
            if (batch_mask_store_.size() != static_cast<std::size_t>(batch_size_))
                batch_mask_store_ = UnifiedVector<std::uint8_t>(static_cast<std::size_t>(batch_size_));
            batch_mask_store_.fill(std::uint8_t{1});
            bind();
            return batch_mask_store_.to_span();
        }

        /// @brief The owned mask, writable; empty until enable_batch_mask().
        Span<std::uint8_t> batch_mask() const { return batch_mask_store_.to_span(); }

    private:
        // Points the view half at the owned storage; rerun after every (re)allocation.
        void bind() {
            this->data_ = values_.to_span();
            this->data_ptrs_ = ptrs_.to_span();
            this->active_rows_ = active_rows_store_.to_span();
            this->active_cols_ = active_cols_store_.to_span();
            this->row_offsets_ = row_offsets_store_.to_span();
            this->col_indices_ = col_indices_store_.to_span();
            this->batch_mask_ = Span<const std::uint8_t>(batch_mask_store_.data(), batch_mask_store_.size());
        }

        UnifiedVector<T> values_;
        UnifiedVector<T*> ptrs_;  // one base pointer per item; allocated only when batch_size > 1
        UnifiedVector<int> active_rows_store_;
        UnifiedVector<int> active_cols_store_;
        UnifiedVector<int> row_offsets_store_;
        UnifiedVector<int> col_indices_store_;
        UnifiedVector<std::uint8_t> batch_mask_store_;
    };

    /// @brief Creates the backend descriptor for a view (src/backends/matrix_handle_impl.cc).
    /// @param[in] view  the view to describe
    /// @return the new descriptor, owned by the returned pointer
    /// @ingroup api_internal_helpers
    template <typename T, MatrixFormat MType>
    std::shared_ptr<BackendMatrixHandle<T, MType>> createBackendHandle(const MatrixView<T, MType>& view);

    /// @brief Non-owning view of a batch of strided vectors: the vector argument type of the entry points.
    ///
    /// Entry \f$i\f$ of item \f$b\f$ is at `data_ptr()[b * stride() + i * inc()]`; a stride of 0
    /// means \f$\mathrm{size} \cdot \mathrm{inc}\f$ (items back to back). The memory must be USM the Queue's
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

        /// @brief Elements needed to hold the layout:
        ///        \f$(\mathrm{batch\_size} - 1)\,\mathrm{stride} + (\mathrm{size} - 1)\,\mathrm{inc} + 1\f$, or 0 when empty.
        /// @param[in] size        entries per item
        /// @param[in] inc         element distance between entries
        /// @param[in] stride      element distance between items (already resolved; 0 is not expanded)
        /// @param[in] batch_size  number of items
        /// @return the span length in elements
        static constexpr std::size_t required_span_length(int size, int inc, int stride, int batch_size) {
            if (size <= 0 || batch_size <= 0) return 0;
            const int64_t max_index = int64_t(batch_size - 1) * int64_t(stride) + int64_t(size - 1) * int64_t(inc);
            return static_cast<std::size_t>(max_index + 1);
        }

        /// @brief An empty view with batch size 1.
        VectorView() : data_(), size_(0), inc_(1), stride_(0), batch_size_(1) {}
        /// @brief Views @p data with the given layout.
        /// @param[in] data        storage; must hold required_span_length() elements (debug-asserted)
        /// @param[in] size        entries per item
        /// @param[in] batch_size  number of items
        /// @param[in] inc         element distance between entries
        /// @param[in] stride      element distance between items; 0 means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        VectorView(Span<T> data, int size, int batch_size = 1, int inc = 1, int stride = 0)
            : data_(data.data(), required_span_length(size, inc, (stride > 0 ? stride : size * inc), batch_size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {
                assert(data.size() >= data_.size());
            }
        /// @brief Views a UnifiedVector's storage; parameters as the Span overload.
        /// @param[in] data        storage (aliased, not copied); must hold required_span_length() elements
        /// @param[in] size        entries per item
        /// @param[in] batch_size  number of items
        /// @param[in] inc         element distance between entries
        /// @param[in] stride      element distance between items; 0 means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        VectorView(UnifiedVector<T>& data, int size, int batch_size = 1, int inc = 1, int stride = 0)
            : data_(data.data(), required_span_length(size, inc, (stride > 0 ? stride : size * inc), batch_size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {
                assert(data.size() >= data_.size());
            }
        /// @brief Views raw memory; parameters as the Span overload (no length check).
        /// @param[in] data        base of entry 0 of item 0
        /// @param[in] size        entries per item
        /// @param[in] batch_size  number of items
        /// @param[in] inc         element distance between entries
        /// @param[in] stride      element distance between items; 0 means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        VectorView(T* data, int size, int batch_size = 1, int inc = 1, int stride = 0)
            : data_(data, required_span_length(size, inc, (stride > 0 ? stride : size * inc), batch_size)),
              size_(size), inc_(inc), stride_(stride > 0 ? stride : size * inc), batch_size_(batch_size) {}

        /// @name Tagged constructors (preferred for new code)
        /// Same as the positional forms, with inc and stride spelled `Inc{}` / `Stride{}`.
        /// @{
        // Additive on purpose: the in-repo positional constructions are all correct.
        /// @brief Views a Span; see the positional Span overload.
        /// @param[in] data        storage; must hold required_span_length() elements
        /// @param[in] size        entries per item
        /// @param[in] batch_size  number of items
        /// @param[in] inc         element distance between entries
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        VectorView(Span<T> data, int size, int batch_size, Inc inc, Stride stride = Stride{0})
            : VectorView(data, size, batch_size, inc.value, stride.value) {}
        /// @brief Views a UnifiedVector's storage.
        /// @param[in] data        storage (aliased, not copied); must hold required_span_length() elements
        /// @param[in] size        entries per item
        /// @param[in] batch_size  number of items
        /// @param[in] inc         element distance between entries
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        VectorView(UnifiedVector<T>& data, int size, int batch_size, Inc inc, Stride stride = Stride{0})
            : VectorView(data, size, batch_size, inc.value, stride.value) {}
        /// @brief Views raw memory (no length check).
        /// @param[in] data        base of entry 0 of item 0
        /// @param[in] size        entries per item
        /// @param[in] batch_size  number of items
        /// @param[in] inc         element distance between entries
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        VectorView(T* data, int size, int batch_size, Inc inc, Stride stride = Stride{0})
            : VectorView(data, size, batch_size, inc.value, stride.value) {}
        /// @}

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
        /// @param[in] ctx  queue whose device gets access
        /// @return the event of the hint
        Event set_access_device(const Queue& ctx) const {
            return data_.set_access_device(ctx);
        }

        /// @brief Prefers residence of the viewed memory on the device of @p ctx.
        /// @param[in] ctx  queue whose device is preferred
        /// @return the event of the hint
        Event set_preferred_location(const Queue& ctx) const {
            return data_.set_preferred_location(ctx);
        }

        /// @brief Migrates the viewed memory to the device of @p ctx.
        /// @param[in] ctx  queue the prefetch is enqueued on
        /// @return the event of the prefetch
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
        /// @brief Every item index \f$0, 1, \dots, b - 1\f$, with \f$b\f$ = batch_size().
        /// @return the indices in ascending order
        std::vector<int> all_items() const {
            std::vector<int> items(static_cast<std::size_t>(batch_size_ > 0 ? batch_size_ : 0));
            for (std::size_t i = 0; i < items.size(); ++i) items[i] = static_cast<int>(i);
            return items;
        }

        /// @brief \f$\{0, \lfloor b/2 \rfloor, b - 1\}\f$ with \f$b\f$ = batch_size(), sorted and de-duplicated
        ///        (\f$\emptyset\f$ for batch 0): what a check reads by default, as MatrixView::default_items().
        /// @return the default item indices
        std::vector<int> default_items() const {
            if (batch_size_ < 1) return {};
            std::vector<int> items{0, batch_size_ / 2, batch_size_ - 1};
            items.erase(std::unique(items.begin(), items.end()), items.end());
            return items;
        }

        /// @brief Owning copy with every element converted by `static_cast<U>`, in this layout; host loop.
        /// @tparam U  element type of the result
        /// @return an owning copy with the same inc and stride
        /// @throws batchlas::unsupported when a batch mask is attached
        template <typename U>
        Vector<U> astype() const;

        /// @brief Enqueues the copy @p dest \f$\leftarrow\f$ @p src on @p ctx, honouring each side's inc and stride.
        /// @param[in]  ctx   queue the copy is enqueued on
        /// @param[out] dest  destination vectors; every entry of each active item is written (masked-out items untouched)
        /// @param[in]  src   source vectors, same size and batch size as @p dest
        /// @return the event of the copy; a no-op when both views share a base pointer
        /// @throws batchlas::invalid_argument if the sizes, batch sizes or attached batch masks differ
        static Event copy(Queue& ctx, const VectorView<T>& dest, const VectorView<T>& src);

        /// @brief True when every entry of @p a and @p b agrees to relative tolerance @p tol; waits on @p ctx.
        /// @see MatrixView::all_close() for the comparison rule.
        /// @param[in] ctx  queue the comparison runs on (waited on)
        /// @param[in] a    first batch
        /// @param[in] b    second batch, same size and batch size as @p a
        /// @param[in] tol  relative tolerance; defaults to the machine epsilon \f$\varepsilon\f$ of float_t<T>
        /// @return whether every entry pair agrees
        static bool all_close(Queue& ctx, const VectorView<T>& a, const VectorView<T>& b, float_t<T> tol = std::numeric_limits<float_t<T>>::epsilon());

        /// @brief Entry @p i of item @p batch; usable on the host and in a kernel.
        /// @param[in] i      entry index
        /// @param[in] batch  batch item index
        /// @return a reference to the entry
        /// @pre \f$0 \le i < \mathrm{size}\f$, \f$0 \le \mathrm{batch} < \mathrm{batch\_size}\f$ (asserted in debug builds)
        T& at(int i, int batch = 0) const {
            assert(i < size_); assert(i >= 0);
            assert(batch < batch_size_); assert(batch >= 0);
            // Both sides int64_t: an int index has already wrapped, and a size_t
            // conversion turns a negative index into one that passes the check.
            assert(static_cast<int64_t>(i) * inc_ + static_cast<int64_t>(batch) * stride_ <
                   static_cast<int64_t>(data_.size()));
                return data_[static_cast<int64_t>(i) * inc_ + static_cast<int64_t>(batch) * stride_]; }

        /// @brief Entry @p i of item 0.
        /// @param[in] i  entry index
        /// @return a reference to the entry
        T& operator[](int i) const { assert(i < size_); assert(i >= 0); assert(static_cast<int64_t>(i) * inc_ < static_cast<int64_t>(data_.size())); return data_[static_cast<int64_t>(i) * inc_]; }

        /// @brief at(@p i, @p batch).
        /// @param[in] i      entry index
        /// @param[in] batch  batch item index
        /// @return a reference to the entry
        T& operator()(int i, int batch = 0) const {return at(i, batch);}

        /// @brief A batch-size-1 view of item @p batch_index; a mask becomes the item's one byte.
        /// @param[in] batch_index  batch item index (not range-checked)
        /// @return the single-item view
        VectorView<T> batch_item(int batch_index) const {
            // For a single batch, we need (size-1)*inc + 1 elements starting at batch_index * stride
            const std::size_t single_batch_length = (size_ > 0) ? ((size_ - 1) * inc_ + 1) : 0;
            VectorView<T> out(Span<T>(data_.data() + static_cast<int64_t>(batch_index) * stride_, single_batch_length), size_, 1, inc_, 0);
            if (!batch_mask_.empty()) out.batch_mask_ = batch_mask_.subspan(static_cast<std::size_t>(batch_index), 1);
            return out;
        }

        /// @name Batch mask
        /// As MatrixView's: one byte per item, nonzero = active, not owned, possibly device-only.
        /// @{
        /// @brief A copy of this view with @p mask attached; an empty span clears it.
        /// @param[in] mask  batch_size() bytes, nonzero = active; empty to clear (not copied: must outlive the view)
        /// @return the masked view
        /// @throws batchlas::invalid_argument unless @p mask is empty or holds batch_size() bytes
        VectorView<T> with_batch_mask(Span<const std::uint8_t> mask) const {
            detail::check_batch_mask_length(mask.size(), batch_size_);
            VectorView<T> out = *this;
            out.batch_mask_ = mask;
            return out;
        }
        /// @brief with_batch_mask() for a mutable span.
        /// @param[in] mask  batch_size() bytes, nonzero = active; empty to clear
        /// @return the masked view
        VectorView<T> with_batch_mask(Span<std::uint8_t> mask) const {
            return with_batch_mask(Span<const std::uint8_t>(mask.data(), mask.size()));
        }
        /// @brief The attached mask; empty when unmasked.
        Span<const std::uint8_t> batch_mask() const { return batch_mask_; }
        /// @brief True when a mask is attached.
        bool has_batch_mask() const { return !batch_mask_.empty(); }
        /// @brief True when item @p b takes part in the call; usable on the host only for a host-readable mask.
        /// @param[in] b  batch item index
        /// @return whether item @p b is active
        bool active(int64_t b) const { return batch_mask_.data() == nullptr || batch_mask_.data()[b] != 0; }
        /// @}

        /// @brief Entries @p slice of every item; keeps inc, stride, batch size and the batch mask.
        /// @param[in] slice  entries to keep
        /// @return the sliced view
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
            VectorView<T> out(Span<T>(data_.data() + slice.start * inc_, required), n, batch_size_, inc_, stride_);
            out.batch_mask_ = batch_mask_;
            return out;
        }

        /// @brief Prints the leading @p max_elements entries of each item in scientific notation.
        /// @param[in,out] os                 stream written to; its format flags are restored, but its precision
        ///                                   is left at 4
        /// @param[in]     max_elements       entries printed per item
        /// @param[in]     max_cols_to_print  unused
        /// @return @p os, with its format flags restored and its precision set to 4
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
        /// @param[in,out] os            stream written to
        /// @param[in]     max_elements  entries printed per item
        void print(std::ostream& os = std::cout, int max_elements = 10) const {
            stream_formatted_to(os, max_elements);
        }

    protected:
        // KernelMatrixView's row and column views attach its mask without the host-side check.
        template <typename, MatrixFormat> friend struct KernelMatrixView;

        Span<T> data_;
        int size_ = 0;
        int inc_ = 1;
        int stride_ = 0;
        int batch_size_ = 1;
        Span<const std::uint8_t> batch_mask_;  // empty = unmasked; possibly device-only
    };

    /// @brief Owning batch of `batch_size()` vectors of length `size()` in USM shared memory.
    ///
    /// Element \f$i\f$ of item \f$b\f$ lives at `data()[b * stride() + i * inc()]`. The default
    /// stride is \f$\mathrm{size} \cdot \mathrm{inc}\f$ (items back to back). A Vector is a VectorView of its
    /// own storage, so it is accepted wherever a VectorView is.
    /// @trap The constructor takes (size, batch_size, Stride, Inc) while VectorView's
    ///       positional constructor takes (data, size, batch_size, inc, stride); the tag types
    ///       and the deleted bare-int overloads keep the two from being transliterated.
    /// @see @ref matrix-model-vectors-inc-and-stride
    /// @ingroup api_matrix
    template <typename T>
    struct Vector : public VectorView<T> {
        using Base = VectorView<T>;

        /// @brief An empty vector with batch size 1.
        Vector() = default;

        /// @brief Allocates an uninitialised batch.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        /// @param[in] inc         element distance between entries
        Vector(int size, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1})
            : Vector(size, batch_size, stride, inc, UnifiedVector<T>(Base::required_span_length(size, inc.value, resolved(size, stride, inc), batch_size))) {}
        /// @brief Allocates a batch with every stored element (gaps included) set to @p value.
        /// @param[in] size        elements per item
        /// @param[in] value       fill value
        /// @param[in] batch_size  number of items
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        /// @param[in] inc         element distance between entries
        Vector(int size, T value, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1})
            : Vector(size, batch_size, stride, inc, UnifiedVector<T>(Base::required_span_length(size, inc.value, resolved(size, stride, inc), batch_size), value)) {}

        /// @brief Deleted: spell the layout `Stride{s}, Inc{i}`.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @param[in] inc         must be spelled `Inc{i}`
        Vector(int size, int batch_size, int stride, int inc) = delete;
        /// @brief Deleted: spell the layout `Stride{s}, Inc{i}`.
        /// @param[in] size        elements per item
        /// @param[in] value       fill value
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @param[in] inc         must be spelled `Inc{i}`
        Vector(int size, T value, int batch_size, int stride, int inc) = delete;
        /// @brief Deleted: `Vector<float>(n, batch, stride)` would otherwise pick the
        ///        (size, value, batch_size) overload and turn `batch` into the fill value.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        // Vector<int> loses (size, value, batch) to ambiguity; nothing in the repo uses it.
        Vector(int size, int batch_size, int stride) = delete;

        /// @brief A batch of zero vectors.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        /// @param[in] inc         element distance between entries
        /// @return the new batch
        static Vector<T> zeros(int size, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1}) {
            return Vector<T>(size, T(0), batch_size, stride, inc);
        }
        /// @brief A batch of all-ones vectors.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        /// @param[in] inc         element distance between entries
        /// @return the new batch
        static Vector<T> ones(int size, int batch_size = 1, Stride stride = Stride{0}, Inc inc = Inc{1}) {
            return Vector<T>(size, T(1), batch_size, stride, inc);
        }
        /// @brief Entries uniform in \f$[0, 1)\f$ (real part only), from a host RNG with a fixed seed of 42.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        /// @param[in] inc         element distance between entries
        /// @return the new batch
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
        /// @brief Unit vectors \f$e_k\f$, \f$k\f$ = @p index (zero except a one at @p index), in every item.
        /// @param[in] size        elements per item
        /// @param[in] index       position of the one, in \f$[0, \mathrm{size})\f$
        /// @param[in] batch_size  number of items
        /// @param[in] stride      element distance between items; `Stride{0}` means @p size
        /// @return the new batch
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
        /// @brief Deleted: spell the layout `Stride{s}, Inc{i}`.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @param[in] inc         must be spelled `Inc{i}`
        /// @return never; the overload is deleted
        static Vector<T> zeros(int size, int batch_size, int stride, int inc) = delete;
        /// @brief Deleted: spell the layout `Stride{s}, Inc{i}`.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @param[in] inc         must be spelled `Inc{i}`
        /// @return never; the overload is deleted
        static Vector<T> ones(int size, int batch_size, int stride, int inc) = delete;
        /// @brief Deleted: spell the layout `Stride{s}, Inc{i}`.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @param[in] inc         must be spelled `Inc{i}`
        /// @return never; the overload is deleted
        static Vector<T> random(int size, int batch_size, int stride, int inc) = delete;
        /// @brief Deleted: spell the stride `Stride{s}`.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @return never; the overload is deleted
        static Vector<T> zeros(int size, int batch_size, int stride) = delete;
        /// @brief Deleted: spell the stride `Stride{s}`.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @return never; the overload is deleted
        static Vector<T> ones(int size, int batch_size, int stride) = delete;
        /// @brief Deleted: spell the stride `Stride{s}`.
        /// @param[in] size        elements per item
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @return never; the overload is deleted
        static Vector<T> random(int size, int batch_size, int stride) = delete;
        /// @brief Deleted: spell the stride `Stride{s}`.
        /// @param[in] size        elements per item
        /// @param[in] index       position of the one
        /// @param[in] batch_size  number of items
        /// @param[in] stride      must be spelled `Stride{s}`
        /// @return never; the overload is deleted
        static Vector<T> standard_basis(int size, int index, int batch_size, int stride) = delete;
        /// @}

        /// @name Copy and move
        /// Copies are deep; moves transfer the storage and leave the source empty.
        /// @{
        /// @brief Deep copy of @p other.
        /// @param[in] other  the vector to copy
        Vector(const Vector& other) : Base(other), values_(other.values_) { bind(); }
        /// @brief Takes over the storage of @p other.
        /// @param[in,out] other  the vector moved from; its storage is released, but its shape accessors still
        ///                       report the old extents (only destroy or assign to it)
        Vector(Vector&& other) noexcept : Base() { *this = std::move(other); }
        /// @brief Replaces this vector with a deep copy of @p other.
        /// @param[in] other  the vector to copy
        /// @return `*this`
        Vector& operator=(const Vector& other) { return *this = Vector(other); }
        /// @brief Replaces this vector with the storage of @p other.
        /// @param[in,out] other  the vector moved from; its storage is released, but its shape accessors still
        ///                       report the old extents (only destroy or assign to it)
        /// @return `*this`
        Vector& operator=(Vector&& other) noexcept {
            Base::operator=(std::move(other));
            values_ = std::move(other.values_);
            bind();
            other.bind();
            return *this;
        }
        /// @}

        /// @brief This vector as a plain VectorView (a sliced copy).
        /// @return a view of this vector's storage
        VectorView<T> view() const { return *this; }

        /// @brief Raw storage index @p i, ignoring inc and stride (kept for compatibility).
        /// @param[in] i  index into the underlying storage
        /// @return a reference to that storage element
        T& operator[](int i) { return values_[i]; }

    private:
        /// @brief The item stride a layout resolves to.
        /// @param[in] size    elements per item
        /// @param[in] stride  requested item stride; `Stride{0}` (or negative) means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        /// @param[in] inc     element distance between entries
        /// @return @p stride when positive, else \f$\mathrm{size} \cdot \mathrm{inc}\f$
        static int resolved(int size, Stride stride, Inc inc) { return stride.value > 0 ? stride.value : size * inc.value; }

        /// @brief Adopts @p values as the storage of a (size, batch_size, stride, inc) batch.
        /// @param[in]     size        elements per item
        /// @param[in]     batch_size  number of items
        /// @param[in]     stride      element distance between items; `Stride{0}` means \f$\mathrm{size} \cdot \mathrm{inc}\f$
        /// @param[in]     inc         element distance between entries
        /// @param[in,out] values      storage of at least the required span length; moved from
        Vector(int size, int batch_size, Stride stride, Inc inc, UnifiedVector<T>&& values)
            : Base(values.data(), size, batch_size, inc.value, resolved(size, stride, inc)), values_(std::move(values)) {
            bind();
        }

        void bind() { this->data_ = values_.to_span(); }

        UnifiedVector<T> values_;
    };

    // ------------------------------------------------------------------
    // Conversion helpers
    // ------------------------------------------------------------------
    template <typename T, MatrixFormat MType>
    template <typename U>
    Matrix<U, MType> Matrix<T, MType>::astype() const {
        static_assert(MType == MatrixFormat::Dense, "Matrix::astype only supports dense matrices");
        Matrix<U, MType> result(rows_, cols_, batch_size_, ld_, stride_);
        if (this->has_batch_mask()) detail::copy_batch_mask(result.enable_batch_mask(), this->batch_mask_);
        auto src = this->data();
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
        if (has_batch_mask()) detail::copy_batch_mask(result.enable_batch_mask(), batch_mask_);
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

    template <typename T, MatrixFormat MType>
    template <typename U>
    Matrix<U, MType> MatrixView<T, MType>::astype(std::span<const int> items) const {
        static_assert(MType == MatrixFormat::Dense, "MatrixView::astype only supports dense matrices");
        if (is_heterogeneous()) throw batchlas::invalid_argument("MatrixView::astype: heterogeneous views are not supported");
        if (has_batch_mask()) throw batchlas::unsupported("MatrixView::astype(items): batch mask not supported yet");
        Matrix<U, MType> result(rows_, cols_, static_cast<int>(items.size()));
        const std::size_t packed = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
        U* dst = result.data().data();
        for (std::size_t g = 0; g < items.size(); ++g) {
            if (items[g] < 0 || items[g] >= batch_size_) throw batchlas::out_of_range("MatrixView::astype: batch item out of range");
            const T* src = data_.data() + static_cast<std::size_t>(items[g]) * static_cast<std::size_t>(stride_);
            for (int j = 0; j < cols_; ++j)
                for (int i = 0; i < rows_; ++i)
                    dst[g * packed + static_cast<std::size_t>(j) * rows_ + i] =
                        static_cast<U>(src[static_cast<std::size_t>(j) * static_cast<std::size_t>(ld_) + i]);
        }
        return result;
    }

    template <typename T, MatrixFormat MType>
    std::vector<int> MatrixView<T, MType>::all_items() const {
        std::vector<int> items(static_cast<std::size_t>(std::max(batch_size_, 0)));
        for (std::size_t i = 0; i < items.size(); ++i) items[i] = static_cast<int>(i);
        return items;
    }

    template <typename T, MatrixFormat MType>
    std::vector<int> MatrixView<T, MType>::default_items() const {
        if (batch_size_ < 1) return {};
        std::vector<int> items{0, batch_size_ / 2, batch_size_ - 1};
        items.erase(std::unique(items.begin(), items.end()), items.end());
        return items;
    }

    template <typename T>
    template <typename U>
    Vector<U> VectorView<T>::astype() const {
        if (has_batch_mask()) throw batchlas::unsupported("VectorView::astype: batch mask not supported yet");
        Vector<U> result(size_, batch_size_, Stride{stride_}, Inc{inc_});
        for (int b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < size_; ++i) {
                result(i, b) = static_cast<U>((*this)(i, b));
            }
        }
        return result;
    }

    /// @brief Shape (rows, cols) of \f$\mathrm{op}(\mathrm{mat})\f$ for @p trans: swapped unless NoTrans.
    /// @param[in] mat    the matrix batch
    /// @param[in] trans  op applied to @p mat
    /// @return (rows, cols) of the operand as the op sees it
    /// @ingroup api_matrix_lowlevel
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    std::pair<int, int> get_effective_dims(const MatrixView<T, MType>& mat, Transpose trans) {
        return (trans == Transpose::NoTrans)
               ? std::make_pair(mat.rows_, mat.cols_)
               : std::make_pair(mat.cols_, mat.rows_);
    }

    /// @brief Shape of \f$\mathrm{op}(\mathrm{mat})\f$ for item @p batch_index, using that item's active extents.
    /// @param[in] mat          the matrix batch
    /// @param[in] trans        op applied to @p mat
    /// @param[in] batch_index  batch item index
    /// @return (rows, cols) of that item as the op sees it
    /// @ingroup api_matrix_lowlevel
    template <typename T = float, MatrixFormat MType = MatrixFormat::Dense>
    std::pair<int, int> get_effective_dims(const MatrixView<T, MType>& mat, Transpose trans, int batch_index) {
        const int rows = mat.rows(batch_index);
        const int cols = mat.cols(batch_index);
        return (trans == Transpose::NoTrans)
               ? std::make_pair(rows, cols)
               : std::make_pair(cols, rows);
    }

    /// @brief Prints with MatrixView::stream_formatted_to() defaults.
    /// @param[in,out] os    stream written to
    /// @param[in]     view  the batch to print (read on the host)
    /// @return @p os
    /// @ingroup api_matrix_lowlevel
    template <typename T, MatrixFormat MType>
    std::ostream& operator<<(std::ostream& os, const MatrixView<T, MType>& view) {
        return view.stream_formatted_to(os); // Uses default arguments from stream_formatted_to
    }

    /// @brief Prints with VectorView::stream_formatted_to() defaults.
    /// @param[in,out] os    stream written to
    /// @param[in]     view  the vectors to print (read on the host)
    /// @return @p os
    /// @ingroup api_matrix_lowlevel
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const VectorView<T>& view) {
        return view.stream_formatted_to(os); // Uses default arguments from stream_formatted_to
    }

    /// @brief In-place \f$A := \alpha A\f$ on every element of every item (dense).
    /// @param[in]     ctx       queue the kernel is enqueued on
    /// @param[in]     alpha     scale factor \f$\alpha\f$
    /// @param[in,out] mat_view  matrices to scale, within rows() \f$\times\f$ cols() (ld and stride honoured)
    /// @return event of the kernel
    /// @throws batchlas::unsupported for CSR
    /// @ingroup api_extra
    template <typename T, MatrixFormat MType>
    BATCHLAS_API Event scale(Queue& ctx, const T& alpha, const MatrixView<T, MType>& mat_view);

    /// @brief In-place \f$x := \alpha x\f$ on every entry of every item.
    /// @param[in]     ctx       queue the kernel is enqueued on
    /// @param[in]     alpha     scale factor \f$\alpha\f$
    /// @param[in,out] vec_view  vectors to scale (inc and stride honoured)
    /// @return event of the kernel
    /// @ingroup api_extra
    template <typename T>
    BATCHLAS_API Event scale(Queue& ctx, const T& alpha, const VectorView<T>& vec_view);

} // namespace batchlas

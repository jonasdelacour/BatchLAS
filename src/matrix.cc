// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/group-collectives.hh>
#include <sycl/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept> // Include for std::runtime_error
#include <string>
#include <vector>    // Include for std::vector used in scan
#include <batchlas/blas/queue-dispatch.hh>
#include <src/queue.hh>
#include <src/backends/backend_handle_impl.hh>
#include <src/util/philox.hh>
#include <src/util/batch_mask.hh>

namespace batchlas {

namespace {

template <typename T>
using csr_random_real_t = typename base_type<T>::type;

struct CsrRandomPatternParams {
    std::uint64_t multiplier = 0;
    std::uint64_t offset = 0;
};

inline std::uint64_t csr_random_splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

inline std::uint64_t csr_random_pick_coprime_stride(std::uint64_t modulus, std::uint64_t seed) {
    if (modulus <= 1) {
        return 0;
    }

    std::uint64_t candidate = csr_random_splitmix64(seed) % modulus;
    if (candidate == 0) {
        candidate = 1;
    }
    while (std::gcd(candidate, modulus) != 1) {
        ++candidate;
        if (candidate == modulus) {
            candidate = 1;
        }
    }
    return candidate;
}

inline CsrRandomPatternParams csr_random_make_pattern_params(std::uint64_t modulus, std::uint64_t seed) {
    if (modulus <= 1) {
        return {};
    }

    return CsrRandomPatternParams{
        csr_random_pick_coprime_stride(modulus, seed ^ 0x6a09e667f3bcc909ull),
        csr_random_splitmix64(seed ^ 0xbb67ae8584caa73bull) % modulus,
    };
}

inline std::uint64_t csr_random_upper_row_start(int row, int n) {
    return static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(2 * n - row - 1) / 2ull;
}

inline std::pair<int, int> csr_random_upper_pair_from_index(std::uint64_t edge_index, int n) {
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (csr_random_upper_row_start(mid, n) <= edge_index) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    const int row = lo;
    const std::uint64_t row_start = csr_random_upper_row_start(row, n);
    const int col = row + 1 + static_cast<int>(edge_index - row_start);
    return {row, col};
}

template <typename T>
inline csr_random_real_t<T> csr_random_magnitude(const T& value) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return sycl::fabs(value.real()) + sycl::fabs(value.imag());
    } else {
        return sycl::fabs(value);
    }
}

template <typename T>
inline T csr_random_conjugate_if_needed(const T& value) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return std::conj(value);
    } else {
        return value;
    }
}

template <typename T>
inline T csr_random_make_diagonal(csr_random_real_t<T> value) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return T(value, csr_random_real_t<T>(0));
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
inline T csr_random_scalar(std::uint64_t seed, std::uint64_t stream_id) {
    return philox::uniform_symmetric<T>(seed, 0, stream_id);
}

inline int csr_random_nnz_per_matrix(int n, float density) {
    density = std::clamp(density, 0.0f, 1.0f);
    const std::uint64_t total_entries = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
    int target_nnz = static_cast<int>(std::llround(density * static_cast<double>(total_entries)));
    target_nnz = std::max(target_nnz, n);

    int offdiag_nnz = std::max(0, target_nnz - n);
    if ((offdiag_nnz & 1) != 0) {
        ++offdiag_nnz;
    }
    const int max_offdiag_nnz = n * (n - 1);
    offdiag_nnz = std::min(offdiag_nnz, max_offdiag_nnz);
    return n + offdiag_nnz;
}

// Resolved layout of the caller's buffer in the dense "copy from existing data"
// constructors.
struct DenseSourceLayout {
    int ld;
    std::size_t stride;
};

// Validates the arguments of the dense "copy from existing data" constructors and
// resolves the layout of the source buffer they describe.
inline DenseSourceLayout dense_source_layout(const char* ctor, bool has_data,
                                             int rows, int cols, int ld, int stride,
                                             int batch_size) {
    const std::string prefix = std::string(ctor) + ": ";
    if (!has_data) {
        throw batchlas::invalid_argument(prefix + "data pointer is null");
    }
    if (rows <= 0 || cols <= 0) {
        throw batchlas::invalid_argument(prefix + "invalid matrix dimensions " +
                                    std::to_string(rows) + "x" + std::to_string(cols));
    }
    if (batch_size <= 0) {
        throw batchlas::invalid_argument(prefix + "invalid batch size " + std::to_string(batch_size));
    }
    if (ld < 0) {
        throw batchlas::invalid_argument(prefix + "invalid leading dimension " + std::to_string(ld));
    }
    if (stride < 0) {
        throw batchlas::invalid_argument(prefix + "invalid stride " + std::to_string(stride));
    }

    const int src_ld = ld > 0 ? ld : rows;
    if (src_ld < rows) {
        throw batchlas::invalid_argument(prefix + "leading dimension " + std::to_string(src_ld) +
                                    " is smaller than the row count " + std::to_string(rows));
    }

    const std::size_t packed = static_cast<std::size_t>(src_ld) * static_cast<std::size_t>(cols);
    const std::size_t src_stride = stride > 0 ? static_cast<std::size_t>(stride) : packed;
    if (batch_size > 1 && src_stride < packed) {
        throw batchlas::invalid_argument(prefix + "stride " + std::to_string(src_stride) +
                                    " is smaller than ld * cols (" + std::to_string(packed) + ")");
    }
    return DenseSourceLayout{src_ld, src_stride};
}

// Number of elements the source buffer must hold for the given layout.
inline std::size_t dense_source_extent(const DenseSourceLayout& src, int rows, int cols,
                                       int batch_size) {
    return static_cast<std::size_t>(batch_size - 1) * src.stride +
           static_cast<std::size_t>(cols - 1) * static_cast<std::size_t>(src.ld) +
           static_cast<std::size_t>(rows);
}

// Validates a Span source (shape plus length) and hands back its pointer.
template <typename T>
inline const T* dense_checked_source(Span<const T> data, int rows, int cols, int ld,
                                     int stride, int batch_size) {
    constexpr const char* ctor = "Matrix(Span<const T> data, rows, cols, ld, stride, batch_size)";
    const auto src = dense_source_layout(ctor, data.data() != nullptr, rows, cols, ld, stride, batch_size);
    const std::size_t required = dense_source_extent(src, rows, cols, batch_size);
    if (data.size() < required) {
        throw batchlas::invalid_argument(std::string(ctor) + ": span holds " + std::to_string(data.size()) +
                                    " elements but the requested shape needs " + std::to_string(required));
    }
    return data.data();
}

// The queue-less helpers run on the default device, or the CPU for double on an fp64-less GPU.
template <typename T>
Queue default_queue_for() {
    return Queue(Device::default_device_for<T>());
}

}  // namespace

namespace detail {
void copy_batch_mask(Span<std::uint8_t> dst, Span<const std::uint8_t> src) {
    if (dst.size() != src.size()) throw batchlas::invalid_argument("astype: batch mask length mismatch");
    if (src.empty()) return;
    // A device memcpy, not a host loop: the source may be sycl::malloc_device memory.
    Queue q;
    q->memcpy(dst.data(), src.data(), src.size());
    q.wait();
}
}  // namespace detail

//----------------------------------------------------------------------
// Matrix class implementation - Dense format
//----------------------------------------------------------------------

// Basic constructor for dense matrix (allocates uninitialized memory)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType>::Matrix(int rows, int cols, int batch_size, int ld, int stride) {
    rows_ = rows;
    cols_ = cols;
    batch_size_ = batch_size;
    ld_ = ld > 0 ? ld : rows;
    stride_ = stride > 0 ? stride : ld > 0 ? ld * cols : rows * cols;
    // The layout invariant the whole class assumes: ld >= rows, stride holds a full item.
    // evidence: docs/design/runtime-internals.md#runtime-internals-matrix-layout-checks
    {
        const std::string prefix = "Matrix(rows, cols, batch_size, ld, stride): ";
        if (rows < 0 || cols < 0) {
            throw batchlas::invalid_argument(prefix + "invalid matrix dimensions " +
                                        std::to_string(rows) + "x" + std::to_string(cols));
        }
        if (batch_size < 0) {
            throw batchlas::invalid_argument(prefix + "invalid batch size " + std::to_string(batch_size));
        }
        if (ld < 0) {
            throw batchlas::invalid_argument(prefix + "invalid leading dimension " + std::to_string(ld));
        }
        if (stride < 0) {
            throw batchlas::invalid_argument(prefix + "invalid stride " + std::to_string(stride));
        }
        if (ld_ < rows) {
            throw batchlas::invalid_argument(prefix + "leading dimension " + std::to_string(ld_) +
                                        " is smaller than the row count " + std::to_string(rows));
        }
        const std::size_t packed = static_cast<std::size_t>(ld_) * static_cast<std::size_t>(cols);
        if (static_cast<std::size_t>(stride_) < packed) {
            throw batchlas::invalid_argument(prefix + "stride " + std::to_string(stride_) +
                                        " is smaller than ld * cols (" + std::to_string(packed) + ")");
        }
    }

    // Allocate memory for the matrix data
    values_ = UnifiedVector<T>(static_cast<size_t>(stride_) * batch_size);

    // If batched, set up the data pointers
    if (batch_size > 1) {
        ptrs_.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            ptrs_[i] = values_.data() + static_cast<std::ptrdiff_t>(i) * stride_;
        }
    }
    bind();
}

// Constructor from existing data (copies the data)
// (ld, stride) describe the *source* buffer; the destination keeps the caller's leading
// dimension but packs the batch items back to back (stride_ == ld_ * cols).
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType>::Matrix(const T* data, int rows, int cols, int ld,
                        int stride, int batch_size) {
    rows_ = rows;
    cols_ = cols;
    batch_size_ = batch_size;
    ld_ = ld > 0 ? ld : rows;
    stride_ = ld_ * cols;
    const auto src = dense_source_layout("Matrix(const T* data, rows, cols, ld, stride, batch_size)",
                                         data != nullptr, rows, cols, ld, stride, batch_size);

    // Allocate memory in the destination layout and copy the data
    const std::size_t dst_stride = static_cast<std::size_t>(ld_) * static_cast<std::size_t>(cols);
    const std::size_t dst_size = dst_stride * static_cast<std::size_t>(batch_size);
    values_.resize(dst_size);

    if (src.ld == rows && src.stride == dst_stride) {
        // Source and destination are both packed, we can copy all at once
        std::copy(data, data + dst_size, values_.data());
    } else {
        // The source columns are padded and/or the batch items are separated by gaps.
        // Copy column by column so that neither the padding (which need not be part of
        // the caller's allocation) nor the gaps are read.
        if (ld_ > rows) {
            std::fill(values_.data(), values_.data() + dst_size, T{});
        }
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < cols; ++j) {
                const T* src_col = data + static_cast<std::size_t>(b) * src.stride +
                                          static_cast<std::size_t>(j) * static_cast<std::size_t>(src.ld);
                std::copy(src_col, src_col + rows,
                          values_.data() + static_cast<std::size_t>(b) * dst_stride +
                                         static_cast<std::size_t>(j) * static_cast<std::size_t>(ld_));
            }
        }
    }

    // If batched, set up the data pointers
    if (batch_size > 1) {
        ptrs_.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            ptrs_[i] = values_.data() + static_cast<std::ptrdiff_t>(i) * stride_;
        }
    }
    bind();
}

// Constructor from an existing span (copies the data, and checks the span length)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType>::Matrix(Span<const T> data, int rows, int cols, int ld,
                        int stride, int batch_size)
    : Matrix(dense_checked_source<T>(data, rows, cols, ld, stride, batch_size),
             rows, cols, ld, stride, batch_size) {}

//----------------------------------------------------------------------
// Matrix class implementation - CSR format
//----------------------------------------------------------------------

// Basic constructor for CSR sparse matrix (allocates uninitialized memory)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires CsrMatrixFormat<M>
Matrix<T, MType>::Matrix(int rows, int cols, NonZeros nnz, int batch_size) {
    rows_ = rows;
    cols_ = cols;
    batch_size_ = batch_size;
    nnz_ = matrix_stride_ = nnz.value;
    offset_stride_ = rows + 1;
    values_.resize(static_cast<size_t>(nnz.value) * batch_size);
    row_offsets_store_.resize(static_cast<size_t>(rows + 1) * batch_size);
    col_indices_store_.resize(static_cast<size_t>(nnz.value) * batch_size);
    bind();
}

// Constructor from existing data (copies the data)
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires CsrMatrixFormat<M>
Matrix<T, MType>::Matrix(const T* values, const int* row_offsets, const int* col_indices,
                        int rows, int cols, NonZeros nnz, int matrix_stride,
                        int offset_stride, int batch_size) {
    rows_ = rows;
    cols_ = cols;
    batch_size_ = batch_size;
    nnz_ = nnz.value;
    matrix_stride_ = matrix_stride > 0 ? matrix_stride : nnz.value;
    offset_stride_ = offset_stride > 0 ? offset_stride : rows + 1;
    values_.resize(static_cast<size_t>(matrix_stride_) * batch_size);
    row_offsets_store_.resize(static_cast<size_t>(offset_stride_) * batch_size);
    col_indices_store_.resize(static_cast<size_t>(matrix_stride_) * batch_size);

    std::copy(values, values + static_cast<size_t>(matrix_stride_) * batch_size, values_.data());
    std::copy(row_offsets, row_offsets + static_cast<size_t>(offset_stride_) * batch_size, row_offsets_store_.data());
    std::copy(col_indices, col_indices + static_cast<size_t>(matrix_stride_) * batch_size, col_indices_store_.data());
    bind();
}

//----------------------------------------------------------------------
// Common Matrix class implementations (for all formats)
//----------------------------------------------------------------------

// Convert to a different matrix format
template <typename T, MatrixFormat MType>
template <MatrixFormat NewMType>
Matrix<T, NewMType> Matrix<T, MType>::convert_to(const float_t<T>& zero_threshold) const {
    Queue q = default_queue_for<T>(); // Use default queue

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

            batchlas::impl::parallel_for(cgh,
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

            batchlas::impl::parallel_for(cgh,
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
        int max_nnz = 0;
        if (batch_nnz_counts.size() > 0) {
            max_nnz = *std::max_element(batch_nnz_counts.begin(), batch_nnz_counts.end());
        }
        // 2. Create row offsets using exclusive scan within each batch
        Matrix<T, MatrixFormat::CSR> result(rows, cols, NonZeros{max_nnz}, batch_size);

        // The scan never writes element 0 of each item and resize() does not zero: zero it all.
        // evidence: docs/design/runtime-internals.md#runtime-internals-matrix-layout-checks
        std::fill(result.row_offsets().begin(), result.row_offsets().end(), 0);

        // Initialize batch offset counters
        UnifiedVector<int> batch_offsets(batch_size + 1, 0);

        // Second kernel: Compute row offsets using work-group scan
        q -> submit ([&](sycl::handler& cgh) {
            auto row_offsets_acc = result.row_offsets().data();
            auto row_nnz_acc = row_nnz_counts.data();
            auto batch_offsets_acc = batch_offsets.data();
            auto wgs = compute_optimal_wg_size(q.device(), KernelType::SCAN, rows + 1, batch_size);
            batchlas::impl::parallel_for(cgh,
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
            batchlas::impl::parallel_for(cgh,
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
                    int idx_in_batch = batch_val_col_base + csr_idx;
                    int c = csr_col_indices_ptr[idx_in_batch];
                    T val = csr_data_ptr[idx_in_batch];

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
        throw batchlas::unsupported("Conversion between specified matrix formats not supported or implemented.");
    }
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
        throw batchlas::unsupported("Submatrix views of CSR matrices not yet implemented");
    }
}


template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
MatrixView<T, MType>::MatrixView(const VectorView<T>& vector_view, VectorOrientation orientation)
    : rows_(orientation == VectorOrientation::Column ? vector_view.size() : 1),
      cols_(orientation == VectorOrientation::Row ? vector_view.size() : 1),
      ld_(orientation == VectorOrientation::Column ? vector_view.size() : vector_view.inc()),
      stride_(vector_view.stride()),
      batch_size_(vector_view.batch_size()),
      data_(vector_view.data()) {
    if (orientation == VectorOrientation::Column && vector_view.inc() != 1) {
        throw batchlas::invalid_argument("Cannot create column matrix view from vector with inc != 1");
    }
    batch_mask_ = vector_view.batch_mask();
}

// Copy from another matrix view
template <typename T, MatrixFormat MType>
void Matrix<T, MType>::copy_from(const MatrixView<T, MType>& src) {
    if (rows_ != src.rows_ || cols_ != src.cols_ || batch_size_ != src.batch_size_) {
        throw batchlas::invalid_argument("Matrix dimensions or batch size mismatch in copy_from");
    }

    if constexpr (MType == MatrixFormat::Dense) {
        // Same tiering as the from-data constructor above: one bulk copy when
        // both sides are packed, one copy per column when a leading dimension or
        // a batch stride gets in the way. A dense layout is always a set of
        // contiguous column segments, so there is no case left for an
        // element-at-a-time loop.
        T* dst_ptr = values_.data();
        const T* src_ptr = src.data().data();
        const std::size_t dst_stride = static_cast<std::size_t>(stride_);
        const std::size_t src_stride = static_cast<std::size_t>(src.stride());
        const std::size_t packed = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);

        if (ld_ == rows_ && src.ld() == rows_ && dst_stride == packed && src_stride == packed) {
            std::copy(src_ptr, src_ptr + packed * static_cast<std::size_t>(batch_size_), dst_ptr);
        } else {
            for (int b = 0; b < batch_size_; ++b) {
                for (int j = 0; j < cols_; ++j) {
                    const T* src_col = src_ptr + static_cast<std::size_t>(b) * src_stride +
                                       static_cast<std::size_t>(j) * static_cast<std::size_t>(src.ld());
                    std::copy(src_col, src_col + rows_,
                              dst_ptr + static_cast<std::size_t>(b) * dst_stride +
                                        static_cast<std::size_t>(j) * static_cast<std::size_t>(ld_));
                }
            }
        }
    } else if constexpr (MType == MatrixFormat::CSR) {
        // For CSR format, just copy all data directly if structure matches
        std::copy(src.data().begin(), src.data().end(), values_.begin());
        std::copy(src.row_offsets().begin(), src.row_offsets().end(), row_offsets_store_.begin());
        std::copy(src.col_indices().begin(), src.col_indices().end(), col_indices_store_.begin());
    }
}

//----------------------------------------------------------------------
// Matrix class static method implementations - Dense format only
//----------------------------------------------------------------------

// Factory method to create identity matrix with SYCL
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::Identity(int n, int batch_size) {
    Matrix<T, MType> result(n, n, batch_size);

    // Create a queue to submit work to
    Queue q = default_queue_for<T>();

    // Get pointer to the matrix data
    T* data_ptr = result.values_.data();

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
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::Triangular(int n, Uplo uplo, T diagonal_value,
                                             T non_diagonal_value, int batch_size) {
    Matrix<T, MType> result(n, n, batch_size);
    result.fill_triangular(uplo, diagonal_value, non_diagonal_value).wait();
    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_triangular(const Queue& ctx, Uplo uplo, T diagonal_value,
                                             T non_diagonal_value) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_triangular");
    T* data_ptr = data_.data();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    auto stride = stride_; // Stride for batched matrices
    auto ld = ld_; // Leading dimension
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);

    // Use batched matrix decomposition strategy
    auto [global_size, local_size, wg_per_matrix] = compute_batched_matrix_decomposition(
        batch_size, n * n, ctx.device(), KernelType::ELEMENTWISE);
    (void)wg_per_matrix; // the grid-stride loop below covers the batch directly

    ctx -> parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        const size_t global_id = item.get_global_id(0);
        const size_t total_work_items = item.get_global_range(0);
        const size_t elements_per_matrix = static_cast<size_t>(n) * static_cast<size_t>(n);
        const size_t total_work = elements_per_matrix * static_cast<size_t>(batch_size);

        // Grid-stride loop over every element of every matrix in the batch.
        for (size_t flat_idx = global_id; flat_idx < total_work; flat_idx += total_work_items) {
            const size_t batch_id = flat_idx / elements_per_matrix;
            if (mask && !mask[batch_id]) continue;
            const size_t remainder = flat_idx % elements_per_matrix;
            const size_t i = remainder % static_cast<size_t>(n); // row index
            const size_t j = remainder / static_cast<size_t>(n); // column index
            T* element = data_ptr + batch_id * stride + j * ld + i;
            if (i == j) {
                // Diagonal elements
                *element = diagonal_value;
            } else if ((uplo == Uplo::Lower && i > j) ||
                    (uplo == Uplo::Upper && i < j)) {
                // Lower or upper triangular elements (Left = lower, Right = upper)
                *element = non_diagonal_value;
            } else {
                // Zero elements
                *element = T(0);
            }
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_triangular_random(const Queue& ctx, Uplo uplo,
                                                    Diag diag,
                                                    unsigned int seed) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_triangular_random");
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);

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
            if (mask && (b >= static_cast<size_t>(batch_size) || !mask[b])) continue;
            size_t remainder = flat_idx % (n * n);
            size_t i = remainder % n;               // row index
            size_t j = remainder / n;               // column index

            // Keyed on the in-matrix position only: every item gets the same matrix (documented).
            const T rand_value = philox::uniform_symmetric<T>(seed, 0, remainder);
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
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::triangularize(const Queue& ctx, Uplo uplo,
                                                    Diag diag) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::triangularize");
    // `uplo` names the triangle to KEEP: Uplo::Upper zeroes the strict lower triangle. The count is
    // rows*cols*batch, never data_.size() (wrong for a strided view).
    // evidence: docs/design/runtime-internals.md#runtime-internals-matrix-layout-checks
    T* data_ptr = data_.data();
    const size_t rows = static_cast<size_t>(rows_);
    const size_t cols = static_cast<size_t>(cols_);
    auto batch_size = batch_size_;
    auto stride = stride_; // Stride for batched matrices
    auto ld = ld_; // Leading dimension
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);

    const size_t elems_per_matrix = rows * cols;
    const size_t total_elements = elems_per_matrix * static_cast<size_t>(batch_size);
    if (total_elements == 0) return ctx.get_event();

    // Use batched matrix decomposition strategy
    auto [global_size, local_size, wg_per_matrix] = compute_batched_matrix_decomposition(
        batch_size, static_cast<int>(elems_per_matrix), ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        size_t global_id = item.get_global_id(0);
        size_t total_work_items = item.get_global_range(0);

        // Grid-stride loop to handle large matrices
        for (size_t flat_idx = global_id; flat_idx < total_elements; flat_idx += total_work_items) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / elems_per_matrix;   // batch index
            if (mask && !mask[b]) continue;
            size_t remainder = flat_idx % elems_per_matrix;
            // Column-major: consecutive flat indices walk DOWN a column.
            size_t col = remainder / rows;
            size_t row = remainder % rows;

            if (row == col && diag == Diag::Unit) {
                data_ptr[b * stride + col * ld + row] = T(1);
            } else if ((uplo == Uplo::Lower && row < col) || (uplo == Uplo::Upper && row > col)) {
                // Zero the triangle that `uplo` does not name.
                data_ptr[b * stride + col * ld + row] = T(0);
            }
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::symmetrize(const Queue& ctx, Uplo uplo) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::symmetrize");
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    T* data_ptr = data_.data();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    auto ld = ld_; // Leading dimension
    auto stride = stride_; // Stride for batched matrices
    const size_t total_elements = static_cast<size_t>(batch_size) * static_cast<size_t>(n) * static_cast<size_t>(n);

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        for (size_t flat_idx = item.get_global_id(0); flat_idx < total_elements; flat_idx += item.get_global_range(0)) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / (n * n);          // batch index
            if (mask && !mask[b]) continue;
            size_t remainder = flat_idx % (n * n);
            size_t i = remainder / n;               // row index
            size_t j = remainder % n;               // column index

            if (uplo == Uplo::Upper) {
                // Upper triangular part
                if (i < j) {
                    size_t idx1 = b * stride + j * ld + i;
                    size_t idx2 = b * stride + i * ld + j;
                    data_ptr[idx2] = data_ptr[idx1];
                }
            } else {
                // Lower triangular part
                if (i > j) {
                    size_t idx1 = b * stride + j * ld + i;
                    size_t idx2 = b * stride + i * ld + j;
                    data_ptr[idx2] = data_ptr[idx1];
                }
            }
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::hermitize(const Queue& ctx, Uplo uplo) const{
    detail::require_device_fp64<T>(ctx, "MatrixView::hermitize");
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    T* data_ptr = data_.data();
    auto n = rows_; // Assuming square matrix
    auto batch_size = batch_size_;
    auto ld = ld_; // Leading dimension
    auto stride = stride_; // Stride for batched matrices
    const size_t total_elements = static_cast<size_t>(batch_size) * static_cast<size_t>(n) * static_cast<size_t>(n);

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    // std::conj of a real returns a complex; a real hermitize is symmetrize.
    auto conj_if_complex = [](T v) -> T {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) return std::conj(v);
        else return v;
    };
    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        for (size_t flat_idx = item.get_global_id(0); flat_idx < total_elements; flat_idx += item.get_global_range(0)) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / (n * n);          // batch index
            if (mask && !mask[b]) continue;
            size_t remainder = flat_idx % (n * n);
            size_t i = remainder / n;               // row index
            size_t j = remainder % n;               // column index

            if (uplo == Uplo::Upper) {
                // Upper triangular part
                if (i < j) {
                    size_t idx1 = b * stride + j * ld + i;
                    size_t idx2 = b * stride + i * ld + j;
                    data_ptr[idx2] = conj_if_complex(data_ptr[idx1]);
                }
            } else {
                // Lower triangular part
                if (i > j) {
                    size_t idx1 = b * stride + j * ld + i;
                    size_t idx2 = b * stride + i * ld + j;
                    data_ptr[idx2] = conj_if_complex(data_ptr[idx1]);
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
        }
    });
    return ctx.get_event();
}

// Factory method to create random matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::Random(int rows, int cols, bool hermitian, int batch_size, unsigned int seed) {
    Matrix<T, MType> result(rows, cols, batch_size);
    result.fill_random(hermitian, seed).wait();
    return result;
}

template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires CsrMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::RandomSparseHermitian(int n,
                                                         float density,
                                                         int batch_size,
                                                         unsigned int seed,
                                                         typename base_type<T>::type diagonal_boost,
                                                         bool shared_pattern) {
    Matrix<T, MType> result(n, n, NonZeros{csr_random_nnz_per_matrix(n, density)}, batch_size);
    result.fill_random_sparse_hermitian(default_queue_for<T>(), density, seed, diagonal_boost, shared_pattern).wait();
    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_random(const Queue& ctx, bool hermitian, unsigned int seed) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_random");
    T* data_ptr = data_.data();
    size_t total_elements = data_.size();
    auto rows = rows_;
    auto cols = cols_;
    auto batch_size = batch_size_;
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    // The flat loop covers the padding too, so an element's item is idx / stride.
    const size_t item_stride = stride_ > 0 ? static_cast<size_t>(stride_) : total_elements + 1;

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        for (size_t idx = item.get_global_id(0); idx < total_elements; idx += item.get_global_range(0)) {
            if (mask && !mask[idx / item_stride]) continue;
            data_ptr[idx] = philox::uniform_symmetric<T>(seed, 0, idx);
        }
    });

    // If hermitian flag is set, enforce Hermitian property using a kernel
    if (hermitian) {
        if (rows != cols) {
            throw batchlas::invalid_argument("Hermitian matrices must be square");
        }

        int ld = ld_;
        int stride = stride_;
        size_t hermitian_elements = static_cast<size_t>(batch_size) * rows * cols;

        // Compute optimal work-group size for the hermitian operation
        auto [herm_global_size, herm_local_size] = compute_nd_range_sizes(
            hermitian_elements, ctx.device(), KernelType::ELEMENTWISE);

        ctx->parallel_for(sycl::nd_range<1>(herm_global_size, herm_local_size),
                        [=](sycl::nd_item<1> item) {
            for (size_t flat = item.get_global_id(0); flat < hermitian_elements; flat += item.get_global_range(0)) {
                size_t b = flat / (rows * cols);
                if (mask && !mask[b]) continue;
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
            }
        });
    }
    return ctx.get_event();
}

namespace {

// Uniform [-1, 1) entry (both parts for complex T), a pure function of (seed, flat).
template <typename T>
T seeded_uniform(unsigned int seed, std::uint64_t flat, std::uint64_t stream = 0) {
    return philox::uniform_symmetric<T>(seed, stream, flat);
}

template <typename T>
T conj_of(const T& x) {
    if constexpr (is_std_complex_v<T>) return std::conj(x);
    else return x;
}

template <typename T>
float_t<T> abs2_of(const T& x) {
    if constexpr (is_std_complex_v<T>) return x.real() * x.real() + x.imag() * x.imag();
    else return x * x;
}

// Off-diagonal entries reach modulus 1 (real) or sqrt(2) (complex): this shift makes every row
// strictly diagonally dominant.
template <typename T>
float_t<T> dominant_shift(int n) {
    return float_t<T>(is_std_complex_v<T> ? 2 * n : n);
}

// Philox streams for the row permutation and the reflector vectors (values use stream 0), so no
// seed of one generator reproduces another's draws.
constexpr std::uint64_t kPermutationStream = 1;
constexpr std::uint64_t kReflectorStream = 2;

std::size_t square_fill_group_size(const Queue& ctx, const char* who, std::size_t local_bytes) {
    const Device dev = ctx.device();
    if (local_bytes > dev.get_property(DeviceProperty::LOCAL_MEM_SIZE))
        throw batchlas::invalid_argument(std::string(who) + ": n too large for the device's local memory");
    return std::min<std::size_t>(256, dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
}

}  // namespace

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_random_spd(const Queue& ctx, unsigned int seed) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_random_spd");
    if (rows_ != cols_) throw batchlas::invalid_argument("fill_random_spd: the matrix must be square");
    const std::size_t n = static_cast<std::size_t>(rows_);
    const std::size_t total = static_cast<std::size_t>(batch_size_) * n * n;
    if (total == 0) return ctx.get_event();
    T* data = data_ptr();
    const std::size_t ld = static_cast<std::size_t>(ld_);
    const std::size_t stride = static_cast<std::size_t>(stride_);
    const float_t<T> shift = dominant_shift<T>(rows_);
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    auto [global_size, local_size] = compute_nd_range_sizes(total, ctx.device(), KernelType::ELEMENTWISE);
    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        for (size_t flat = item.get_global_id(0); flat < total; flat += item.get_global_range(0)) {
            const std::size_t b = flat / (n * n);
            if (mask && !mask[b]) continue;
            const std::size_t j = (flat / n) % n;
            const std::size_t i = flat % n;
            // The upper triangle conjugates the very value the lower one stores: exactly Hermitian.
            T value = i >= j ? seeded_uniform<T>(seed, (b * n + j) * n + i)
                             : conj_of(seeded_uniform<T>(seed, (b * n + i) * n + j));
            if (i == j) value = T(std::real(value) + shift);
            data[b * stride + j * ld + i] = value;
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_near_degenerate_hermitian(const Queue& ctx, double noise, unsigned int seed) const {
    if (rows_ != cols_) throw batchlas::invalid_argument("fill_near_degenerate_hermitian: the matrix must be square");
    const std::size_t n = static_cast<std::size_t>(rows_);
    const std::size_t total = static_cast<std::size_t>(batch_size_) * n * n;
    if (total == 0) return ctx.get_event();
    T* data = data_ptr();
    const std::size_t ld = static_cast<std::size_t>(ld_);
    const std::size_t stride = static_cast<std::size_t>(stride_);
    const float_t<T> eps = static_cast<float_t<T>>(noise);
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    auto [global_size, local_size] = compute_nd_range_sizes(total, ctx.device(), KernelType::ELEMENTWISE);
    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        const std::size_t flat = item.get_global_id(0);
        if (flat >= total) return;
        const std::size_t b = flat / (n * n);
        if (mask && !mask[b]) return;
        const std::size_t j = (flat / n) % n;
        const std::size_t i = flat % n;
        T value = i >= j ? eps * seeded_uniform<T>(seed, (b * n + j) * n + i)
                         : conj_of(eps * seeded_uniform<T>(seed, (b * n + i) * n + j));
        if (i == j) value = T(float_t<T>(i / 4) + float_t<T>(i % 4) * float_t<T>(1e-4) + std::real(value));
        data[b * stride + j * ld + i] = value;
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_random_pivoted(const Queue& ctx, unsigned int seed) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_random_pivoted");
    if (rows_ != cols_) throw batchlas::invalid_argument("fill_random_pivoted: the matrix must be square");
    const int n = rows_;
    const int batch = batch_size_;
    if (n == 0 || batch == 0) return ctx.get_event();
    const std::size_t wg = square_fill_group_size(ctx, "fill_random_pivoted", sizeof(int) * n);
    T* data = data_ptr();
    const std::size_t ld = static_cast<std::size_t>(ld_);
    const std::size_t stride = static_cast<std::size_t>(stride_);
    const float_t<T> shift = dominant_shift<T>(n);
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    ctx->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> perm(sycl::range<1>(static_cast<std::size_t>(n)), cgh);
        batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(wg * batch, wg), [=](sycl::nd_item<1> item) {
            const std::size_t b = item.get_group(0);
            if (mask && !mask[b]) return;  // uniform per work-group: before the barrier
            const std::size_t lid = item.get_local_id(0);
            const std::size_t N = static_cast<std::size_t>(n);
            if (lid == 0) {
                // Sattolo's shuffle: a uniformly random n-cycle, so no row stays in place.
                for (int i = 0; i < n; ++i) perm[i] = i;
                for (int i = n - 1; i > 0; --i) {
                    const float u = philox::uniform_unit<float>(seed, kPermutationStream, b * N + static_cast<std::size_t>(i));
                    const int j = sycl::min(static_cast<int>(u * static_cast<float>(i)), i - 1);
                    const int t = perm[i];
                    perm[i] = perm[j];
                    perm[j] = t;
                }
            }
            sycl::group_barrier(item.get_group());
            for (std::size_t e = lid; e < N * N; e += wg) {
                const std::size_t i = e % N;
                const std::size_t c = e / N;
                const std::size_t r = static_cast<std::size_t>(perm[i]);
                T value = seeded_uniform<T>(seed, (b * N + c) * N + r);
                if (r == c) value += T(shift);
                data[b * stride + c * ld + i] = value;
            }
        });
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_graded_hermitian(const Queue& ctx, double log10_cond, unsigned int seed) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_graded_hermitian");
    if (rows_ != cols_) throw batchlas::invalid_argument("fill_graded_hermitian: the matrix must be square");
    if (!std::isfinite(log10_cond)) throw batchlas::invalid_argument("fill_graded_hermitian: log10_cond must be finite");
    using R = float_t<T>;
    const int n = rows_;
    const int batch = batch_size_;
    if (n == 0 || batch == 0) return ctx.get_event();
    const std::size_t wg = square_fill_group_size(ctx, "fill_graded_hermitian", 2 * sizeof(T) * n);
    T* data = data_ptr();
    const std::size_t ld = static_cast<std::size_t>(ld_);
    const std::size_t stride = static_cast<std::size_t>(stride_);
    const R decades = static_cast<R>(log10_cond);
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    ctx->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> v(sycl::range<1>(static_cast<std::size_t>(n)), cgh);
        sycl::local_accessor<T, 1> s(sycl::range<1>(static_cast<std::size_t>(n)), cgh);
        batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(wg * batch, wg), [=](sycl::nd_item<1> item) {
            const auto g = item.get_group();
            const std::size_t b = item.get_group(0);
            if (mask && !mask[b]) return;  // uniform per work-group: before the first barrier
            const std::size_t lid = item.get_local_id(0);
            const std::size_t N = static_cast<std::size_t>(n);
            T* A = data + b * stride;
            for (std::size_t e = lid; e < N * N; e += wg) {
                const std::size_t r = e % N;
                const std::size_t c = e / N;
                const R lambda = n == 1 ? R(1) : sycl::pow(R(10), -decades * R(r) / R(n - 1));
                A[c * ld + r] = r == c ? T(lambda) : T(0);
            }
            sycl::group_barrier(g);
            for (std::size_t k = 0; k < N; ++k) {
                R partial = 0;
                for (std::size_t r = lid; r < N; r += wg) {
                    // Real T needs real reflectors, or the complex Q leaves a non-real result.
                    v[r] = seeded_uniform<T>(seed, (b * N + k) * N + r, kReflectorStream);
                    partial += abs2_of(v[r]);
                }
                const R vv = portable::reduce_over_group(g, partial, sycl::plus<R>());
                if (vv == R(0)) continue;
                const R inv = R(1) / sycl::sqrt(vv);
                for (std::size_t r = lid; r < N; r += wg) v[r] *= inv;
                sycl::group_barrier(g);
                // M <- H M H, H = I - 2 v v^H: first the columns (H M), then the rows (M H).
                for (std::size_t c = lid; c < N; c += wg) {
                    T acc = 0;
                    for (std::size_t r = 0; r < N; ++r) acc += conj_of(v[r]) * A[c * ld + r];
                    s[c] = acc;
                }
                sycl::group_barrier(g);
                for (std::size_t e = lid; e < N * N; e += wg) {
                    const std::size_t r = e % N;
                    const std::size_t c = e / N;
                    A[c * ld + r] -= R(2) * v[r] * s[c];
                }
                sycl::group_barrier(g);
                for (std::size_t r = lid; r < N; r += wg) {
                    T acc = 0;
                    for (std::size_t c = 0; c < N; ++c) acc += A[c * ld + r] * v[c];
                    s[r] = acc;
                }
                sycl::group_barrier(g);
                for (std::size_t e = lid; e < N * N; e += wg) {
                    const std::size_t r = e % N;
                    const std::size_t c = e / N;
                    A[c * ld + r] -= R(2) * s[r] * conj_of(v[c]);
                }
                sycl::group_barrier(g);
            }
            for (std::size_t e = lid; e < N * N; e += wg) {
                const std::size_t r = e % N;
                const std::size_t c = e / N;
                if (r > c) A[r * ld + c] = conj_of(A[c * ld + r]);
                else if (r == c) A[c * ld + c] = T(std::real(A[c * ld + c]));
            }
        });
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires CsrMatrixFormat<M>
Event MatrixView<T, MType>::fill_random_sparse_hermitian(const Queue& ctx,
                                                         float density,
                                                         unsigned int seed,
                                                         typename base_type<T>::type diagonal_boost,
                                                         bool shared_pattern) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_random_sparse_hermitian");
    if (rows_ <= 0 || cols_ <= 0) {
        throw batchlas::invalid_argument("fill_random_sparse_hermitian: matrix dimensions must be > 0");
    }
    if (rows_ != cols_) {
        throw batchlas::invalid_argument("fill_random_sparse_hermitian: Hermitian matrices must be square");
    }
    if (batch_size_ <= 0) {
        throw batchlas::invalid_argument("fill_random_sparse_hermitian: batch_size must be > 0");
    }
    // The pattern (row offsets) is built on the host for every item, which a device-only mask cannot steer.
    if (has_batch_mask()) {
        throw batchlas::unsupported("fill_random_sparse_hermitian: a masked view is not supported");
    }

    density = std::clamp(density, 0.0f, 1.0f);
    const int n = rows_;
    const int nnz_per_matrix = csr_random_nnz_per_matrix(n, density);
    const int offdiag_edges = (nnz_per_matrix - n) / 2;

    if (nnz_ != nnz_per_matrix || matrix_stride_ < nnz_per_matrix || offset_stride_ < (n + 1)) {
        throw batchlas::invalid_argument("fill_random_sparse_hermitian: CSR storage shape does not match requested density");
    }

    auto row_offsets_ptr = row_offsets_.data();
    auto col_indices_ptr = col_indices_.data();
    auto values_ptr = data_.data();

    const std::uint64_t max_edges = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n - 1) / 2ull;
    const int pattern_batches = shared_pattern ? 1 : batch_size_;

    UnifiedVector<std::uint64_t> edge_indices(static_cast<std::size_t>(pattern_batches) * static_cast<std::size_t>(offdiag_edges));
    UnifiedVector<std::uint64_t> multipliers(static_cast<std::size_t>(pattern_batches), 0);
    UnifiedVector<std::uint64_t> offsets(static_cast<std::size_t>(pattern_batches), 0);

    for (int p = 0; p < pattern_batches; ++p) {
        const auto params = csr_random_make_pattern_params(max_edges, static_cast<std::uint64_t>(seed) + 1315423911ull * static_cast<std::uint64_t>(p + 1));
        multipliers[static_cast<std::size_t>(p)] = params.multiplier;
        offsets[static_cast<std::size_t>(p)] = params.offset;
    }

    if (offdiag_edges > 0) {
        ctx->parallel_for(static_cast<std::size_t>(pattern_batches) * static_cast<std::size_t>(offdiag_edges),
                          [edge_indices_data = edge_indices.data(),
                           multipliers_data = multipliers.data(),
                           offsets_data = offsets.data(),
                           max_edges,
                           offdiag_edges](std::size_t flat) {
            const int pattern = static_cast<int>(flat / static_cast<std::size_t>(offdiag_edges));
            const std::size_t local_edge = flat % static_cast<std::size_t>(offdiag_edges);
            edge_indices_data[flat] = (offsets_data[pattern] + multipliers_data[pattern] * static_cast<std::uint64_t>(local_edge)) % max_edges;
        });
        ctx.wait_and_throw();

        for (int p = 0; p < pattern_batches; ++p) {
            auto* begin = edge_indices.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(offdiag_edges);
            auto* end = begin + offdiag_edges;
            std::sort(begin, end);
        }
        ctx.wait_and_throw();
    }

    UnifiedVector<int> row_counts(static_cast<std::size_t>(batch_size_) * static_cast<std::size_t>(n), 1);
    UnifiedVector<int> lower_counts(static_cast<std::size_t>(batch_size_) * static_cast<std::size_t>(n), 0);
    UnifiedVector<int> lower_cursors(static_cast<std::size_t>(batch_size_) * static_cast<std::size_t>(n), 0);
    UnifiedVector<int> upper_cursors(static_cast<std::size_t>(batch_size_) * static_cast<std::size_t>(n), 0);
    UnifiedVector<csr_random_real_t<T>> offdiag_sums(static_cast<std::size_t>(batch_size_) * static_cast<std::size_t>(n), csr_random_real_t<T>(0));

    if (offdiag_edges > 0) {
        ctx->parallel_for(static_cast<std::size_t>(batch_size_) * static_cast<std::size_t>(offdiag_edges),
                          [edge_indices_data = edge_indices.data(),
                           row_counts_data = row_counts.data(),
                           lower_counts_data = lower_counts.data(),
                           n,
                           offdiag_edges,
                           shared_pattern](std::size_t flat) {
            const int batch = static_cast<int>(flat / static_cast<std::size_t>(offdiag_edges));
            const int local_edge = static_cast<int>(flat % static_cast<std::size_t>(offdiag_edges));
            const int pattern = shared_pattern ? 0 : batch;
            const std::uint64_t edge_index = edge_indices_data[static_cast<std::size_t>(pattern) * static_cast<std::size_t>(offdiag_edges) + static_cast<std::size_t>(local_edge)];
            const auto [row, col] = csr_random_upper_pair_from_index(edge_index, n);
            const int base = batch * n;

            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> row_count_upper(row_counts_data[base + row]);
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> row_count_lower(row_counts_data[base + col]);
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> lower_count(lower_counts_data[base + col]);
            row_count_upper.fetch_add(1);
            row_count_lower.fetch_add(1);
            lower_count.fetch_add(1);
        });
        ctx.wait_and_throw();
    }

    for (int b = 0; b < batch_size_; ++b) {
        row_offsets_ptr[static_cast<std::size_t>(b) * static_cast<std::size_t>(offset_stride_)] = 0;
        // row_offsets[0] = 0 and row_offsets[i+1] = sum(counts[0..i]), i.e. an
        // inclusive scan of the per-row counts written one slot in. Using an
        // exclusive scan here shifts every offset down by one row, which leaves
        // row 0 empty and makes rows 0 and 1 share a start index.
        std::inclusive_scan(
            row_counts.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(n),
            row_counts.data() + static_cast<std::size_t>(b + 1) * static_cast<std::size_t>(n),
            row_offsets_ptr + static_cast<std::size_t>(b) * static_cast<std::size_t>(offset_stride_) + 1);
    }
    ctx.wait_and_throw();

    ctx->parallel_for(static_cast<std::size_t>(batch_size_),
                      [edge_indices_data = edge_indices.data(),
                       row_offsets_ptr,
                       col_indices_ptr,
                       values_ptr,
                       lower_counts_data = lower_counts.data(),
                       lower_cursors_data = lower_cursors.data(),
                       upper_cursors_data = upper_cursors.data(),
                       offdiag_sums_data = offdiag_sums.data(),
                       n,
                       offdiag_edges,
                       matrix_stride = matrix_stride_,
                       offset_stride = offset_stride_,
                       seed,
                       diagonal_boost,
                       shared_pattern](std::size_t batch_index) {
        const int batch = static_cast<int>(batch_index);
        const int row_base = batch * n;
        const int offset_base = batch * offset_stride;
        const int value_base = batch * matrix_stride;
        const int pattern = shared_pattern ? 0 : batch;
        const auto* batch_edges = edge_indices_data + static_cast<std::size_t>(pattern) * static_cast<std::size_t>(offdiag_edges);
        const std::uint64_t batch_seed = static_cast<std::uint64_t>(seed) + 31ull * static_cast<std::uint64_t>(batch + 1);

        for (int row = 0; row < n; ++row) {
            const int row_start = row_offsets_ptr[offset_base + row];
            lower_cursors_data[row_base + row] = row_start;
            upper_cursors_data[row_base + row] = row_start + lower_counts_data[row_base + row] + 1;
            offdiag_sums_data[row_base + row] = csr_random_real_t<T>(0);
        }

        for (int e = 0; e < offdiag_edges; ++e) {
            const std::uint64_t edge_index = batch_edges[e];
            const auto [row, col] = csr_random_upper_pair_from_index(edge_index, n);
            const T upper_value = csr_random_scalar<T>(batch_seed, edge_index + (static_cast<std::uint64_t>(e) << 32u));
            const T lower_value = csr_random_conjugate_if_needed(upper_value);
            const auto magnitude = csr_random_magnitude(upper_value);

            const int upper_pos = upper_cursors_data[row_base + row]++;
            col_indices_ptr[value_base + upper_pos] = col;
            values_ptr[value_base + upper_pos] = upper_value;

            const int lower_pos = lower_cursors_data[row_base + col]++;
            col_indices_ptr[value_base + lower_pos] = row;
            values_ptr[value_base + lower_pos] = lower_value;

            offdiag_sums_data[row_base + row] += magnitude;
            offdiag_sums_data[row_base + col] += magnitude;
        }

        for (int row = 0; row < n; ++row) {
            const int diag_pos = row_offsets_ptr[offset_base + row] + lower_counts_data[row_base + row];
            col_indices_ptr[value_base + diag_pos] = row;
            values_ptr[value_base + diag_pos] = csr_random_make_diagonal<T>(offdiag_sums_data[row_base + row] + diagonal_boost);
        }
    });

    return ctx.get_event();
}

template <typename T, MatrixFormat MType> struct FillContiguousKernel {};
template <typename T, MatrixFormat MType> struct FillNonContiguousKernel {};

template <typename T, MatrixFormat MType>
Event MatrixView<T, MType>::fill(const Queue& ctx, T value) const {
    auto data_ptr = data_.data();
    size_t total_elements = 0;
    bool contiguous = false;
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);
    // Masked: the memset fast path would write the inactive items, so a kernel skips them.
    // CSR stays on the flat kernel (item = idx / matrix_stride); dense takes the per-element one.
    size_t item_span = 1;
    if constexpr (MType == MatrixFormat::CSR) {
        total_elements = static_cast<size_t>(data_.size());
        contiguous = true;
        item_span = matrix_stride_ > 0 ? static_cast<size_t>(matrix_stride_) : total_elements + 1;
    } else {
        total_elements = static_cast<size_t>(batch_size_) * static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
        contiguous = !mask && (stride_ == rows_ * cols_) && (ld_ == rows_);
    }
    bool char_val = 0;
    bool fill_val_is_zero = value == detail::convert_to_fill_value<T>(0);
    if constexpr (sizeof(T) == sizeof(char)) {
        char_val = static_cast<char>(value);
    }
    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    const bool use_memset = fill_val_is_zero && contiguous && !mask;
    if (!use_memset) detail::require_device_fp64<T>(ctx, "MatrixView::fill");
    if (use_memset) {
        ctx->memset(data_ptr, char_val, total_elements * sizeof(T));
    } else if (contiguous) {
        ctx->parallel_for<FillContiguousKernel<T, MType>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            size_t global_id = item.get_global_id(0);
            size_t total_work_items = item.get_global_range(0);

            // Grid-stride loop to handle large data
            for (size_t idx = global_id; idx < total_elements; idx += total_work_items) {
                if (mask && !mask[idx / item_span]) continue;
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
                if (!data_view.active(b)) continue;
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
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_identity(const Queue& ctx, T value) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_identity");
    // (void) on an Event: deliberate. This Queue is in-order, so the next submission
    // is already ordered after this one and the Event carries nothing the caller needs.
    (void)fill_zeros(ctx);
    (void)fill_diagonal(ctx, value);
    return ctx.get_event();
}

// Factory method to create zeros matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::Zeros(int rows, int cols, int batch_size) {
    Matrix<T, MType> result(rows, cols, batch_size);
    result.fill(T(0));
    return result;
}

// Factory method to create ones matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::Ones(int rows, int cols, int batch_size) {
    Matrix<T, MType> result(rows, cols, batch_size);
    result.fill(T(1));
    return result;
}

// Factory method to create diagonal matrix
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::Diagonal(const Span<T>& diag_values, int batch_size) {
    int n = diag_values.size();
    Matrix<T, MType> result(n, n, batch_size);

    // Set all elements to zero
    result.fill(T(0));
    // Ensure the returned matrix is fully initialized before returning.
    // The fill_diagonal(Span<...>) convenience overload uses an internal Queue;
    // without waiting here, callers can race when they immediately use a different Queue.
    result.template fill_diagonal<MType>(diag_values).wait();

    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_diagonal(const Queue& ctx, const Span<T>& diag_values, int64_t k) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_diagonal");
    const int d = std::max<int64_t>(0, std::min<int64_t>(rows_ - (k < 0 ? -k : 0), cols_ - (k > 0 ? k : 0)));
    if (d == 0 || batch_size_ == 0) return ctx.get_event();
    const bool batch_diagonals = diag_values.size() == static_cast<std::size_t>(d) * batch_size_;
    return fill_diagonal(ctx, VectorView<T>(diag_values, d, batch_diagonals ? batch_size_ : 1, 1, d), k);
}

template <typename T, MatrixFormat MType> struct FillDiagonalVectorKernel;

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_diagonal(const Queue& ctx, const VectorView<T>& diag_values, int64_t k) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_diagonal");
    int ld = ld_;
    int stride = stride_;
    auto col_offset = (k >= 0) ? k : 0;
    auto row_offset = (k < 0) ? -k : 0;
    // A wide or tall view, or |k| at or past an edge: min(...) entries, possibly none.
    const int n = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(rows_ - row_offset, cols_ - col_offset)));
    int batch_size = batch_size_;
    bool batch_diagonals = diag_values.batch_size() == batch_size;
    size_t total_elements = static_cast<size_t>(batch_size) * n;
    if (total_elements == 0) return ctx.get_event();
    const std::uint8_t* mask = detail::mask_ptr(detail::call_mask("fill_diagonal", *this, diag_values));

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->submit([=, this](sycl::handler& cgh) {
        auto view = this->kernel_view();
        auto diag = diag_values;
        batchlas::impl::parallel_for<FillDiagonalVectorKernel<T, MType>>(cgh,
            sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
                size_t gid = item.get_global_id(0);

                // Bounds check
                if (gid >= total_elements) return;
                for (int flat = gid; flat < total_elements; flat += item.get_global_range(0)) {
                    size_t b = flat / n;
                    if (mask && !mask[b]) continue;
                    size_t i = flat % n;
                    view(i + row_offset, i + col_offset, b) = diag(i, batch_diagonals ? b : 0);
                }
            });
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType> struct FillDiagonalVectorKernel {};

template <typename T, MatrixFormat MType> struct FillDiagonalKernel {};
template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_diagonal(const Queue& ctx, const T& diag_value) const {
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_diagonal");
    T* data_ptr = data_.data();
    int ld = ld_;
    int stride = stride_;
    auto n = std::min(rows_, cols_);  // a tall or wide view has min(rows, cols) diagonal entries
    auto batch_size = batch_size_;
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);

    size_t total_elements = static_cast<size_t>(batch_size) * n;
    if (total_elements == 0) return ctx.get_event();

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for<FillDiagonalKernel<T, MType>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item){
        for (size_t flat = item.get_global_id(0); flat < total_elements; flat += item.get_global_range(0)) {
            size_t b = flat / n;
            if (mask && !mask[b]) continue;
            size_t i = flat % n;
            data_ptr[b * stride + i * ld + i] = diag_value;
        }
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::TriDiagToeplitz(int n, T diag, T sub, T super, int batch_size) {
    Matrix<T, MType> result(n, n, batch_size);
    // Ensure the returned matrix is fully initialized before returning.
    result.template fill_tridiag_toeplitz<MType>(diag, sub, super).wait();
    return result;
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
Event MatrixView<T, MType>::fill_tridiag_toeplitz(const Queue& ctx, T diag, T sub_diag, T super_diag) const{
    detail::require_device_fp64<T>(ctx, "MatrixView::fill_tridiag_toeplitz");
    T* data_ptr = data().data();
    int n = rows_; // Assuming square matrix
    int batch_size = batch_size_;
    const std::uint8_t* mask = detail::mask_ptr(batch_mask_);

    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(n) * n * batch_size;

    // Compute optimal work-group size for element-wise operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    // Submit a kernel to initialize the tridiagonal Toeplitz matrix
    ctx -> parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        for (size_t flat_idx = item.get_global_id(0); flat_idx < total_elements; flat_idx += item.get_global_range(0)) {
            // Calculate 3D coordinates from flat index
            int64_t b = flat_idx / (n * n);          // batch index
            if (mask && !mask[b]) continue;
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
        }
    });
    return ctx.get_event();
}

// Row pitch of row-major data: a PARAMETER, never inferred from ld (ld is a column distance). The
// packed default is accepted only when ld == rows and stride == rows*cols, where no padded layout fits;
// elsewhere it would read wrong elements IN BOUNDS, so it throws. Also bounds-checks the read.
// evidence: docs/design/runtime-internals.md#runtime-internals-matrix-layout-checks
namespace {
inline std::size_t checked_row_major_pitch(int row_pitch, int rows, int cols, std::size_t ld,
                                           std::size_t stride, int batch_size,
                                           std::size_t buffer_elems) {
    const std::string prefix = "Matrix::to_column_major: ";
    if (row_pitch < 0) {
        throw batchlas::invalid_argument(prefix + "invalid row pitch " + std::to_string(row_pitch));
    }
    const std::size_t pitch = row_pitch > 0 ? static_cast<std::size_t>(row_pitch)
                                            : static_cast<std::size_t>(cols);
    if (rows <= 0 || cols <= 0 || batch_size <= 0) {
        return pitch;  // nothing is read
    }
    if (row_pitch == 0 && rows > 1) {
        // rows == 1 is exempt: a single row is read the same way at every pitch,
        // so there is nothing to guess and a throw would be pure noise.
        const std::size_t packed_item = static_cast<std::size_t>(rows) *
                                        static_cast<std::size_t>(cols);
        if (ld != static_cast<std::size_t>(rows) || stride != packed_item) {
            throw batchlas::invalid_argument(
                prefix + "the default row pitch means packed (pitch cols = " + std::to_string(cols) +
                "), but this matrix is not packed: rows=" + std::to_string(rows) +
                " cols=" + std::to_string(cols) + " ld=" + std::to_string(ld) +
                " stride=" + std::to_string(stride) + ", where packed is ld=" +
                std::to_string(rows) + " stride=" + std::to_string(packed_item) +
                ". A row-major buffer with a padded row pitch fits in this item too, and the "
                "two cannot be told apart from the layout. Pass the buffer's row pitch, or pass "
                "cols (" + std::to_string(cols) + ") if it really is packed");
        }
    }
    if (pitch < static_cast<std::size_t>(cols)) {
        throw batchlas::invalid_argument(prefix + "row pitch " + std::to_string(pitch) +
                                    " is smaller than the column count " + std::to_string(cols));
    }
    const std::size_t item = static_cast<std::size_t>(rows - 1) * pitch +
                             static_cast<std::size_t>(cols);
    if (batch_size > 1 && item > stride) {
        throw batchlas::invalid_argument(prefix + "a row-major read at pitch " + std::to_string(pitch) +
                                    " spans " + std::to_string(item) +
                                    " elements per batch item, but the batch items are only " +
                                    std::to_string(stride) + " apart (stride)");
    }
    const std::size_t needed = static_cast<std::size_t>(batch_size - 1) * stride + item;
    if (needed > buffer_elems) {
        throw batchlas::invalid_argument(prefix + "a row-major read at pitch " + std::to_string(pitch) +
                                    " needs " + std::to_string(needed) +
                                    " elements but the matrix owns " + std::to_string(buffer_elems) +
                                    "; pass the pitch the buffer really has, or compact it first");
    }
    return pitch;
}

// The same bounds question for the column-major read to_row_major performs, which
// uses the matrix' own ld and stride.
inline void check_column_major_extent(int rows, int cols, std::size_t ld, std::size_t stride,
                                      int batch_size, std::size_t buffer_elems) {
    if (rows <= 0 || cols <= 0 || batch_size <= 0) return;
    const std::size_t needed = static_cast<std::size_t>(batch_size - 1) * stride +
                               static_cast<std::size_t>(cols - 1) * ld +
                               static_cast<std::size_t>(rows);
    if (needed > buffer_elems) {
        throw batchlas::invalid_argument("Matrix::to_row_major: a column-major read at ld " +
                                    std::to_string(ld) + " needs " + std::to_string(needed) +
                                    " elements but the matrix owns " + std::to_string(buffer_elems));
    }
}
}  // namespace

// Implement to_column_major using SYCL
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::to_column_major(const Queue& ctx, int row_pitch) const {
    detail::require_device_fp64<T>(ctx, "Matrix::to_column_major");
    // The source holds row-major data. Its row pitch is what the caller says it is
    // (default: packed, pitch cols -- allowed only where the layout leaves no room
    // for a padded pitch); its batch items are stride_ apart, which need not be
    // ld_ * cols. Validate before allocating anything.
    const size_t src_pitch = checked_row_major_pitch(row_pitch, rows_, cols_,
                                                     static_cast<size_t>(ld_),
                                                     static_cast<size_t>(stride_), batch_size_,
                                                     values_.size());

    // The destination is packed in the target major: ld = rows, stride = rows * cols.
    Matrix<T, MType> col_major(rows_, cols_, batch_size_);

    // Get pointers to source and destination data
    const T* src_ptr = values_.data();
    T* dst_ptr = col_major.values_.data();

    // Capture dimensions for kernel
    int rows = rows_;
    int cols = cols_;
    size_t src_stride = static_cast<size_t>(stride_);
    size_t dst_stride = static_cast<size_t>(col_major.stride_);
    size_t dst_ld = static_cast<size_t>(col_major.ld_);

    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(rows_) * cols_ * batch_size_;

    // Compute optimal work-group size for memory-bound operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::MEMORY_BOUND);

    // Submit a kernel to convert from row-major to column-major
    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        for (size_t flat_idx = item.get_global_id(0); flat_idx < total_elements; flat_idx += item.get_global_range(0)) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / (static_cast<size_t>(rows) * cols);  // batch index
            size_t remainder = flat_idx % (static_cast<size_t>(rows) * cols);
            size_t i = remainder / cols;            // row index
            size_t j = remainder % cols;            // column index

            // src is row-major: [b, i, j] -> b * src_stride + i * src_pitch + j
            // dst is col-major: [b, j, i] -> b * dst_stride + j * dst_ld + i
            dst_ptr[b * dst_stride + j * dst_ld + i] = src_ptr[b * src_stride + i * src_pitch + j];
        }
    }).wait();

    return col_major;
}

template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::to_column_major(int row_pitch) const {
    // Builds a queue of its own, and waits on it before returning.
    Queue q = default_queue_for<T>();
    return to_column_major(q, row_pitch);
}

// Implement to_row_major using SYCL
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::to_row_major(const Queue& ctx) const {
    detail::require_device_fp64<T>(ctx, "Matrix::to_row_major");
    // The source is read column-major with this matrix' own ld_/stride_.
    check_column_major_extent(rows_, cols_, static_cast<size_t>(ld_),
                              static_cast<size_t>(stride_), batch_size_, values_.size());

    // The destination holds packed row-major data (row pitch cols, batch stride
    // rows * cols). The Matrix it is handed back in keeps the class' usual
    // column-major-shaped metadata (ld = rows); the pitch is not recoverable from
    // that metadata, which is exactly why to_column_major takes the pitch as an
    // argument and defaults it to "packed" -- the layout produced here.
    Matrix<T, MType> row_major(rows_, cols_, batch_size_);

    // Get pointers to source and destination data
    const T* src_ptr = values_.data();
    T* dst_ptr = row_major.values_.data();

    // Capture dimensions for kernel
    int rows = rows_;
    int cols = cols_;
    // The source holds column-major data, so ld_ is its column pitch directly.
    size_t src_ld = static_cast<size_t>(ld_);
    size_t src_stride = static_cast<size_t>(stride_);
    size_t dst_pitch = static_cast<size_t>(cols_);
    size_t dst_stride = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);

    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(rows_) * cols_ * batch_size_;

    // Compute optimal work-group size for memory-bound operations
    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::MEMORY_BOUND);

    // Submit a kernel to convert from column-major to row-major
    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        for (size_t flat_idx = item.get_global_id(0); flat_idx < total_elements; flat_idx += item.get_global_range(0)) {
            // Calculate 3D coordinates from flat index
            size_t b = flat_idx / (static_cast<size_t>(rows) * cols);  // batch index
            size_t remainder = flat_idx % (static_cast<size_t>(rows) * cols);
            size_t i = remainder / cols;            // row index
            size_t j = remainder % cols;            // column index

            // src is col-major: [b, j, i] -> b * src_stride + j * src_ld + i
            // dst is row-major: [b, i, j] -> b * dst_stride + i * dst_pitch + j
            dst_ptr[b * dst_stride + i * dst_pitch + j] = src_ptr[b * src_stride + j * src_ld + i];
        }
    }).wait();

    return row_major;
}

template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
Matrix<T, MType> Matrix<T, MType>::to_row_major() const {
    // Builds a queue of its own, and waits on it before returning.
    Queue q = default_queue_for<T>();
    return to_row_major(q);
}

//----------------------------------------------------------------------
// MatrixView class implementation - Dense format
//----------------------------------------------------------------------

// Constructor for dense matrix view
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
MatrixView<T, MType>::MatrixView(T* data, int rows, int cols, int ld,
                  int stride, int batch_size, T** data_ptrs) :
                data_(data, static_cast<size_t>(stride > 0 ? stride :
                            ld > 0 ? ld * cols : rows * cols) * static_cast<size_t>(batch_size)),
                rows_(rows), cols_(cols), batch_size_(batch_size),
                ld_(ld > 0 ? ld : rows), stride_(stride > 0 ? stride : ld > 0 ? ld * cols : rows * cols), data_ptrs_(data_ptrs, (data_ptrs ? batch_size : 0)) {
    // TRAP: MatrixView(p, n, n, batch) (Matrix's argument order) gives ld = batch; `ld_ < rows`
    // catches it. Deliberately NOT checked: null data / zero extents (shape-only sizing views) and
    // stride >= ld*cols (the recorded violator left is a syevx_lobpcg sizing view; ortho's was fixed).
    // evidence: docs/design/runtime-internals.md#runtime-internals-matrix-layout-checks
    const std::string prefix = "MatrixView(data, rows, cols, ld, stride, batch_size): ";
    if (rows < 0 || cols < 0) {
        throw batchlas::invalid_argument(prefix + "invalid matrix dimensions " +
                                    std::to_string(rows) + "x" + std::to_string(cols));
    }
    if (batch_size < 0) {
        throw batchlas::invalid_argument(prefix + "invalid batch size " + std::to_string(batch_size));
    }
    if (ld < 0) {
        throw batchlas::invalid_argument(prefix + "invalid leading dimension " + std::to_string(ld));
    }
    if (stride < 0) {
        throw batchlas::invalid_argument(prefix + "invalid stride " + std::to_string(stride));
    }
    if (ld_ < rows) {
        throw batchlas::invalid_argument(prefix + "leading dimension " + std::to_string(ld_) +
                                    " is smaller than the row count " + std::to_string(rows) +
                                    " (did you mean MatrixView(data, rows, cols, /*ld=*/" +
                                    std::to_string(rows) + ", /*stride=*/0, /*batch_size=*/" +
                                    std::to_string(ld_) + ")?)");
    }
}


//----------------------------------------------------------------------
// MatrixView class implementation - CSR format
//----------------------------------------------------------------------

// Constructor for CSR sparse matrix view
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires CsrMatrixFormat<M>
MatrixView<T, MType>::MatrixView(T* data, int* row_offsets, int* col_indices,
            int rows, int cols, NonZeros nnz, int matrix_stride,
            int offset_stride, int batch_size, T** data_ptrs) :
            data_(data, (matrix_stride > 0 ? matrix_stride : nnz.value) * batch_size),
            col_indices_(col_indices, (matrix_stride > 0 ? matrix_stride : nnz.value) * batch_size),
            row_offsets_(row_offsets, (offset_stride > 0 ? offset_stride : rows + 1) * batch_size),
            rows_(rows), cols_(cols), batch_size_(batch_size), nnz_(nnz.value), matrix_stride_(matrix_stride > 0 ? matrix_stride : nnz.value),
            offset_stride_(offset_stride > 0 ? offset_stride : rows + 1), data_ptrs_(data_ptrs, (data_ptrs ? batch_size : 0)) {
}

//----------------------------------------------------------------------
// Common MatrixView implementations (for both Dense and CSR)
//----------------------------------------------------------------------

// Constructor for a submatrix view from a Matrix object
template <typename T, MatrixFormat MType>
template <typename U, MatrixFormat M>
    requires DenseMatrixFormat<M>
MatrixView<T, MType>::MatrixView(
    const MatrixView<T, MType>& matrix, int rows, int cols, int ld, int stride, int batch_size) :
    MatrixView<T, MType>(matrix.data_ptr(),
                        rows > 0 ? rows : matrix.rows_,
                        cols > 0 ? cols : matrix.cols_,
                        ld > 0 ? ld : matrix.ld_,
                        stride > 0 ? stride : matrix.stride_,
                        batch_size > 0 ? batch_size : matrix.batch_size_,
                        matrix.data_ptrs_.data()) {
    if constexpr (MType == MatrixFormat::Dense) {
        const bool preserves_shape = (rows <= 0 || rows == matrix.rows_) &&
                                     (cols <= 0 || cols == matrix.cols_) &&
                                     (ld <= 0 || ld == matrix.ld_) &&
                                     (stride <= 0 || stride == matrix.stride_) &&
                                     (batch_size <= 0 || batch_size == matrix.batch_size_);
        if (preserves_shape) {
            active_rows_ = matrix.active_rows_;
            active_cols_ = matrix.active_cols_;
        }
    }
    // A reshaped batch cannot keep a per-item mask, and dropping it would run the inactive items.
    if (matrix.has_batch_mask() && batch_size_ != matrix.batch_size_) {
        throw batchlas::invalid_argument("MatrixView: a masked view cannot change its batch size (" +
                                         std::to_string(matrix.batch_size_) + " -> " + std::to_string(batch_size_) + ")");
    }
    batch_mask_ = matrix.batch_mask_;
}



// Element access (for dense matrices)
template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
T& MatrixView<T, MType>::at(int row, int col, int batch) {
    if (batch < 0 || batch >= batch_size_) {
        throw batchlas::out_of_range("Matrix indices out of range");
    }
    const int batch_rows = rows(batch);
    const int batch_cols = cols(batch);
    if (row < 0 || row >= batch_rows || col < 0 || col >= batch_cols) {
        throw batchlas::out_of_range("Matrix indices out of range");
    }
    // The batch term in int64_t: batch * stride_ in int wraps at batch sizes this
    // library targets (a 512x512 float item has stride 262144, so it wraps at b = 8192).
    return data_[static_cast<int64_t>(batch) * stride_ + col * ld_ + row];
}

template <typename T, MatrixFormat MType>
template <MatrixFormat M>
    requires DenseMatrixFormat<M>
const T& MatrixView<T, MType>::at(int row, int col, int batch) const {
    if (batch < 0 || batch >= batch_size_) {
        throw batchlas::out_of_range("Matrix indices out of range");
    }
    const int batch_rows = rows(batch);
    const int batch_cols = cols(batch);
    if (row < 0 || row >= batch_rows || col < 0 || col >= batch_cols) {
        throw batchlas::out_of_range("Matrix indices out of range");
    }
    // The batch term in int64_t: batch * stride_ in int wraps at batch sizes this
    // library targets (a 512x512 float item has stride 262144, so it wraps at b = 8192).
    return data_[static_cast<int64_t>(batch) * stride_ + col * ld_ + row];
}

// Create a view of a single batch item
template <typename T, MatrixFormat MType>
MatrixView<T, MType> MatrixView<T, MType>::batch_item(int batch_index) const {
    if (batch_index < 0 || batch_index >= batch_size_) {
        throw batchlas::out_of_range("Batch index out of range");
    }

    MatrixView<T, MType> out;
    if constexpr (MType == MatrixFormat::Dense) {
        // Calculate offset to the start of the batch. In int64_t, not int: the product is
        // being widened to size_t anyway, and computing it in int wraps before the widening
        // at batch sizes this library targets.
        int64_t offset = static_cast<int64_t>(batch_index) * stride_;

        // Create a new view with batch_size = 1
        out = MatrixView<T, MType>(
            data_.data() + offset,
            rows(batch_index), cols(batch_index), ld_);
    } else if constexpr (MType == MatrixFormat::CSR) {
        // Calculate offsets to the start of the batch (int64_t for the same reason).
        int64_t val_offset = static_cast<int64_t>(batch_index) * matrix_stride_;
        int64_t row_offset = static_cast<int64_t>(batch_index) * offset_stride_;

        // Create a new view with batch_size = 1, keeping the parent's strides so the
        // spans cover the slots this item actually owns.
        out = MatrixView<T, MType>(
            data_.data() + val_offset,
            row_offsets_.data() + row_offset,
            col_indices_.data() + val_offset,
            rows_, cols_, NonZeros{nnz_}, matrix_stride_, offset_stride_, 1);
    }
    if (!batch_mask_.empty()) out.batch_mask_ = batch_mask_.subspan(static_cast<std::size_t>(batch_index), 1);
    return out;
}

template <typename T, MatrixFormat MType> struct CopyKernel3D {};
template <typename T, MatrixFormat MType>
Event MatrixView<T, MType>::copy(Queue& ctx, const MatrixView<T, MType>& dest, const MatrixView<T, MType>& src) {
    if (src.rows() != dest.rows() || src.cols() != dest.cols() || src.batch_size() != dest.batch_size()) {
        throw batchlas::invalid_argument("Copy called on incompatible matrices");
    }
    if constexpr(MType == MatrixFormat::Dense) {
        if (src.data_ptr() == dest.data_ptr()) return ctx.get_event(); // No-op if same buffer
        // Masked: the memcpy fast paths would write the inactive items, so only the kernel runs.
        const std::uint8_t* mask = detail::mask_ptr(detail::call_mask("copy", dest, src));
        if (!mask && src.stride() == src.rows() * src.cols() && dest.stride() == dest.rows() * dest.cols()) {
            //If the stride is the same as the size of each matrix just do a straight memcpy
            auto event = static_cast<EventImpl>(ctx->memcpy(dest.data_.data(), src.data_.data(), src.data().size_bytes()));
            return event;
#if defined(SYCL_EXT_ONEAPI_MEMCPY2D)  // sycl_ext_oneapi_memcpy2d; without it every strided case takes the copy kernel
        } else if (!mask && src.ld() == src.rows() && dest.ld() == dest.rows()){
            //If both leading dimensions are the same as the number of rows we can utilize oneapis 2d copy kernel
            auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                    dest.stride() * sizeof(T),
                                                    static_cast<void*>(src.data_ptr()),
                                                    src.stride() * sizeof(T),
                                                    src.rows()*src.cols()*sizeof(T), src.batch_size()));
            return event;
        } else if (!mask && src.stride() == src.ld()* src.cols() && dest.stride() == dest.ld()* dest.cols()) {
            // If both the strides and leading dimensions are the same as the size of each matrix just do a straight memcpy
            auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                    dest.ld() * sizeof(T),
                                                    static_cast<void*>(src.data_ptr()),
                                                    src.ld() * sizeof(T),
                                                    src.rows()*sizeof(T), src.cols() * src.batch_size()));
            return event;
#endif
        } else {
            //If neither the ld nor the strides are unit, we need to do essentially a 3D copy
            detail::require_device_fp64<T>(ctx, "MatrixView::copy");
            auto [ global_size, local_size ] = compute_nd_range_sizes(
                    static_cast<size_t>(src.rows()) * src.cols() * src.batch_size(),
                    ctx.device(), KernelType::MEMORY_BOUND);
            auto event = static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
                auto dest_view = dest.kernel_view();
                auto src_view  = src.kernel_view();


                batchlas::impl::parallel_for<CopyKernel3D<T, MType>>(cgh, sycl::nd_range<1>(global_size, local_size),
                                [=](sycl::nd_item<1> item) {
                    size_t gid = item.get_global_id(0);

                    // Calculate 3D coordinates from flat index
                    for (size_t idx = gid; idx < static_cast<size_t>(src_view.rows()) * src_view.cols() * src_view.batch_size(); idx += item.get_global_range(0)) {
                        size_t b = idx / (static_cast<size_t>(src_view.rows()) * src_view.cols());    // batch index
                        if (mask && !mask[b]) continue;
                        size_t remainder = idx % (static_cast<size_t>(src_view.rows()) * src_view.cols());
                        size_t i = remainder % src_view.rows();            // row index
                        size_t j = remainder / src_view.rows();            // column index

                        dest_view(i, j, b) = src_view(i, j, b);
                    }
                });
            }));
            return event;
        }
    } else if constexpr(MType == MatrixFormat::CSR) {
        throw batchlas::unsupported("CSR copy not implemented yet");
    } else {
        throw batchlas::unsupported("Unsupported matrix format in copy");
    }
    return ctx.get_event();
}

template <typename T>
Event VectorView<T>::copy(Queue& ctx, const VectorView<T>& dest, const VectorView<T>& src) {
    if (src.size() != dest.size() || src.batch_size() != dest.batch_size()) {
        throw batchlas::invalid_argument("Copy called on incompatible vectors");
    }
    if (src.data_ptr() == dest.data_ptr()) return ctx.get_event(); // No-op if same buffer
    // Masked: as MatrixView::copy, only the kernel runs.
    const std::uint8_t* mask = detail::mask_ptr(detail::call_mask("copy", dest, src));
    if (!mask && ((src.stride() == src.size() && dest.stride() == dest.size() && src.inc() == 1 && dest.inc() == 1) ||
        (src.stride() == 1 && dest.stride() == 1 && src.inc() == src.size() && dest.inc() == dest.size()))) {
        //If the stride is the same as the size of each vector just do a straight memcpy
        auto event = static_cast<EventImpl>(ctx->memcpy(dest.data_.data(), src.data_.data(), src.size() * src.batch_size() * sizeof(T)));
        return event;
#if defined(SYCL_EXT_ONEAPI_MEMCPY2D)  // sycl_ext_oneapi_memcpy2d; without it every strided case takes the copy kernel
    } else if (!mask && src.stride() == src.inc() * src.size() && dest.stride() == dest.inc() * dest.size()) {
        // If both the strides and increments are the same as the size of each vector just do a straight 2D memcpy
        auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                dest.inc() * sizeof(T),
                                                static_cast<void*>(src.data_ptr()),
                                                src.inc() * sizeof(T),
                                                sizeof(T), src.size() * src.batch_size()));
        return event;
    } else if (!mask && src.inc() == 1 && dest.inc() == 1 && src.stride() >= src.size() && dest.stride() >= dest.size()) {
        // If both increments are 1, we can do a 2D memcpy with the strides
        auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                dest.stride() * sizeof(T),
                                                static_cast<void*>(src.data_ptr()),
                                                src.stride() * sizeof(T),
                                                src.size() * sizeof(T), src.batch_size()));
        return event;
    } else if (!mask && src.inc() == src.stride() * src.size() && dest.inc() == dest.stride() * dest.size()) {
        // If both the increments are the same as the size of each vector just do a straight 2D memcpy
        auto event = static_cast<EventImpl>(ctx->ext_oneapi_memcpy2d(  static_cast<void*>(dest.data_ptr()),
                                                dest.stride() * sizeof(T),
                                                static_cast<void*>(src.data_ptr()),
                                                src.stride() * sizeof(T),
                                                sizeof(T), src.size() * src.batch_size()));
        return event;
#endif
    } else {
        //If neither the ld nor the strides are unit, we need to do essentially a 2D copy
        detail::require_device_fp64<T>(ctx, "VectorView::copy");
        auto event = static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<2>(src.size(), src.batch_size()),
                            [=](sycl::id<2> idx) {
                size_t i = idx[0]; // index
                size_t b = idx[1]; // batch
                if (mask && !mask[b]) return;
                dest(i, b) = src(i, b);
            });
        }));
        return event;
    }
}

template <typename T>
bool VectorView<T>::all_close(Queue& ctx, const VectorView<T>& a, const VectorView<T>& b, float_t<T> tol) {
    detail::require_device_fp64<T>(ctx, "VectorView::all_close");
    if (a.size() != b.size() || a.batch_size() != b.batch_size()) {
        throw batchlas::invalid_argument("all_close called on incompatible vectors");
    }
    UnifiedVector<bool> batch_results(a.batch_size());
    // An inactive item compares true without being read.
    const std::uint8_t* mask = detail::mask_ptr(detail::call_mask("all_close", a, b));

    ctx -> submit([&](sycl::handler& cgh) {
        auto results = batch_results.data();
        auto group_matches = sycl::local_accessor<int32_t, 1>(sycl::range<1>(a.size()), cgh);
        batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(sycl::range<1>(a.batch_size() * a.size()), sycl::range<1>(a.size())), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto lid = item.get_local_id(0);
            if (mask && !mask[bid]) {  // uniform per work-group: before the barrier
                if (lid == 0) results[bid] = true;
                return;
            }
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

            group_matches[lid] = is_close ? 1 : 0;
            sycl::group_barrier(item.get_group());
            if (lid == 0) {
                bool group_all_close = true;
                for (int64_t idx = 0; idx < a.size(); ++idx) {
                    if (group_matches[idx] == 0) {
                        group_all_close = false;
                        break;
                    }
                }
                results[bid] = group_all_close;
            }
        });
    }).wait();

    ctx -> submit([&](sycl::handler& cgh) {
        auto results = batch_results.data();
        batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(sycl::range<1>(std::min(1024, a.batch_size())), sycl::range<1>(std::min(1024, a.batch_size()))), [=](sycl::nd_item<1> item) {
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
    detail::require_device_fp64<T>(ctx, "MatrixView::all_close");
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.batch_size() != B.batch_size()) {
        throw batchlas::invalid_argument("all_close called on incompatible matrices");
    }
    UnifiedVector<bool> batch_results(A.batch_size());
    if constexpr (MType == MatrixFormat::Dense) {
        // An inactive item compares true without being read.
        const std::uint8_t* mask = detail::mask_ptr(detail::call_mask("all_close", A, B));


        ctx -> submit([&](sycl::handler& cgh) {
            auto results = batch_results.data();
            auto Aview = A.kernel_view();
            auto Bview = B.kernel_view();
            auto group_matches = sycl::local_accessor<int32_t, 1>(sycl::range<1>(A.rows()), cgh);

            batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(sycl::range<1>(A.batch_size() *  A.rows()), sycl::range<1>(A.rows())), [=](sycl::nd_item<1> item) {
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
                if (mask && !mask[bid]) return;  // uniform per work-group: before the barriers

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
                    group_matches[lid] = is_close ? 1 : 0;
                    sycl::group_barrier(item.get_group());
                    if (lid == 0) {
                        bool group_all_close = true;
                        for (int64_t row = 0; row < Aview.rows(); ++row) {
                            if (group_matches[row] == 0) {
                                group_all_close = false;
                                break;
                            }
                        }
                        results[bid] = results[bid] && group_all_close;
                    }
                    sycl::group_barrier(item.get_group());
                    if (!results[bid]) {
                        break; // Early exit if already false
                    }
                }
            });
        }).wait();
    } else {
        throw batchlas::unsupported("all_close not implemented for this matrix format");
    }

    ctx -> submit([&](sycl::handler& cgh) {
        auto results = batch_results.data();
        auto Aview = A.kernel_view();
        auto batch_size = A.batch_size();
        batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(sycl::range<1>(std::min(1024, batch_size)), sycl::range<1>(std::min(1024, batch_size))), [=](sycl::nd_item<1> item) {
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
    detail::require_device_fp64<T>(ctx, "scale");
    auto [global_size, local_size] = compute_nd_range_sizes(
        static_cast<size_t>(matrix.rows()) * matrix.cols() * matrix.batch_size(),
        ctx.device(), KernelType::ELEMENTWISE);
    if constexpr (MType == MatrixFormat::CSR) {
        throw batchlas::unsupported("scale not implemented for CSR matrices yet");
    } else if constexpr (MType == MatrixFormat::Dense) {
        return static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
            auto view = matrix.kernel_view();
            batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
                for (size_t gid = item.get_global_id(0); gid < static_cast<size_t>(view.rows()) * view.cols() * view.batch_size(); gid += item.get_global_range(0)) {
                    size_t b = gid / (static_cast<size_t>(view.rows()) * view.cols());
                    if (!view.active(b)) continue;
                    size_t rem = gid % (static_cast<size_t>(view.rows()) * view.cols());
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
    detail::require_device_fp64<T>(ctx, "scale");
    auto [global_size, local_size] = compute_nd_range_sizes(
        static_cast<size_t>(vector.size()) * vector.batch_size(),
        ctx.device(), KernelType::ELEMENTWISE);
    return static_cast<EventImpl>(ctx->submit([&](sycl::handler& cgh) {
        batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            for (size_t gid = item.get_global_id(0); gid < static_cast<size_t>(vector.size()) * vector.batch_size(); gid += item.get_global_range(0)) {
                size_t b = gid / vector.size();
                if (!vector.active(b)) continue;
                size_t i = gid % vector.size();
                vector(i, b) *= alpha;
            }
        });
    }));
}


template <typename T, MatrixFormat MType>
void MatrixView<T, MType>::init_data_ptr_array(Queue& ctx, Span<T*> target) const {
    if (target.data() == nullptr) target = data_ptrs_;
    auto [start_ptr, stride, data_ptrs] = std::make_tuple(this->data_ptr(), stride_, target);
    if (!data_ptrs.data()) {
        throw batchlas::invalid_argument("data_ptrs target is null");
    }
    if (data_ptrs.size() < static_cast<std::size_t>(batch_size_)) {
        throw batchlas::invalid_argument("data_ptrs target is smaller than batch_size");
    }
    if (!start_ptr) {
        throw batchlas::invalid_argument("start_ptr is null");
    }
    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(batch_size_), [=](sycl::id<1> idx) {
            size_t b = idx[0];
            data_ptrs[b] = start_ptr + b * stride;
        });
    }).wait();
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

template Matrix<float, MatrixFormat::Dense>::Matrix(Span<const float>, int, int, int, int, int);
template Matrix<double, MatrixFormat::Dense>::Matrix(Span<const double>, int, int, int, int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense>::Matrix(Span<const std::complex<float>>, int, int, int, int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense>::Matrix(Span<const std::complex<double>>, int, int, int, int, int);

// CSR constructor instantiations
template Matrix<float, MatrixFormat::CSR>::Matrix(int, int, NonZeros, int);
template Matrix<double, MatrixFormat::CSR>::Matrix(int, int, NonZeros, int);
template Matrix<std::complex<float>, MatrixFormat::CSR>::Matrix(int, int, NonZeros, int);
template Matrix<std::complex<double>, MatrixFormat::CSR>::Matrix(int, int, NonZeros, int);

template Matrix<float, MatrixFormat::CSR>::Matrix(const float*, const int*, const int*, int, int, NonZeros, int, int, int);
template Matrix<double, MatrixFormat::CSR>::Matrix(const double*, const int*, const int*, int, int, NonZeros, int, int, int);
template Matrix<std::complex<float>, MatrixFormat::CSR>::Matrix(const std::complex<float>*, const int*, const int*, int, int, NonZeros, int, int, int);
template Matrix<std::complex<double>, MatrixFormat::CSR>::Matrix(const std::complex<double>*, const int*, const int*, int, int, NonZeros, int, int, int);

// Static factory method instantiations
template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Identity(int, int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Identity(int, int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Identity(int, int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Identity(int, int);

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::Random(int, int, bool, int, unsigned int);

template Matrix<float, MatrixFormat::CSR> Matrix<float, MatrixFormat::CSR>::RandomSparseHermitian(int, float, int, unsigned int, base_type<float>::type, bool);
template Matrix<double, MatrixFormat::CSR> Matrix<double, MatrixFormat::CSR>::RandomSparseHermitian(int, float, int, unsigned int, base_type<double>::type, bool);
template Matrix<std::complex<float>, MatrixFormat::CSR> Matrix<std::complex<float>, MatrixFormat::CSR>::RandomSparseHermitian(int, float, int, unsigned int, base_type<std::complex<float>>::type, bool);
template Matrix<std::complex<double>, MatrixFormat::CSR> Matrix<std::complex<double>, MatrixFormat::CSR>::RandomSparseHermitian(int, float, int, unsigned int, base_type<std::complex<double>>::type, bool);

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

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::to_column_major(int) const;
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::to_column_major(int) const;
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::to_column_major(int) const;
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::to_column_major(int) const;

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::to_column_major(const Queue&, int) const;
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::to_column_major(const Queue&, int) const;
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::to_column_major(const Queue&, int) const;
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::to_column_major(const Queue&, int) const;

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::to_row_major() const;
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::to_row_major() const;
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::to_row_major() const;
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::to_row_major() const;

template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::Dense>::to_row_major(const Queue&) const;
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::Dense>::to_row_major(const Queue&) const;
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::Dense>::to_row_major(const Queue&) const;
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::Dense>::to_row_major(const Queue&) const;

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
template Matrix<float, MatrixFormat::Dense> Matrix<float, MatrixFormat::CSR>::convert_to<MatrixFormat::Dense>(const float&) const;
template Matrix<double, MatrixFormat::Dense> Matrix<double, MatrixFormat::CSR>::convert_to<MatrixFormat::Dense>(const double&) const;
template Matrix<std::complex<float>, MatrixFormat::Dense> Matrix<std::complex<float>, MatrixFormat::CSR>::convert_to<MatrixFormat::Dense>(const float&) const;
template Matrix<std::complex<double>, MatrixFormat::Dense> Matrix<std::complex<double>, MatrixFormat::CSR>::convert_to<MatrixFormat::Dense>(const double&) const;

//----------------------------------------------------------------------


// fill instantiations
template Event MatrixView<float, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_random(const Queue&, bool, unsigned int) const;

template Event MatrixView<float, MatrixFormat::CSR>::fill_random_sparse_hermitian(const Queue&, float, unsigned int, base_type<float>::type, bool) const;
template Event MatrixView<double, MatrixFormat::CSR>::fill_random_sparse_hermitian(const Queue&, float, unsigned int, base_type<double>::type, bool) const;
template Event MatrixView<std::complex<float>, MatrixFormat::CSR>::fill_random_sparse_hermitian(const Queue&, float, unsigned int, base_type<std::complex<float>>::type, bool) const;
template Event MatrixView<std::complex<double>, MatrixFormat::CSR>::fill_random_sparse_hermitian(const Queue&, float, unsigned int, base_type<std::complex<double>>::type, bool) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, float, float) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, double, double) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, std::complex<float>, std::complex<float>) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_triangular(const Queue&, Uplo, std::complex<double>, std::complex<double>) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_triangular_random(const Queue&, Uplo, Diag, unsigned int) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_random_spd(const Queue&, unsigned int) const;
template Event MatrixView<float, MatrixFormat::Dense>::fill_near_degenerate_hermitian(const Queue&, double, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_random_spd(const Queue&, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_near_degenerate_hermitian(const Queue&, double, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_random_spd(const Queue&, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_near_degenerate_hermitian(const Queue&, double, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_random_spd(const Queue&, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_near_degenerate_hermitian(const Queue&, double, unsigned int) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_random_pivoted(const Queue&, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_random_pivoted(const Queue&, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_random_pivoted(const Queue&, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_random_pivoted(const Queue&, unsigned int) const;

template Event MatrixView<float, MatrixFormat::Dense>::fill_graded_hermitian(const Queue&, double, unsigned int) const;
template Event MatrixView<double, MatrixFormat::Dense>::fill_graded_hermitian(const Queue&, double, unsigned int) const;
template Event MatrixView<std::complex<float>, MatrixFormat::Dense>::fill_graded_hermitian(const Queue&, double, unsigned int) const;
template Event MatrixView<std::complex<double>, MatrixFormat::Dense>::fill_graded_hermitian(const Queue&, double, unsigned int) const;

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

template Event MatrixView<float, MatrixFormat::Dense>::hermitize(const Queue&, Uplo) const;
template Event MatrixView<double, MatrixFormat::Dense>::hermitize(const Queue&, Uplo) const;
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
template MatrixView<float, MatrixFormat::CSR>::MatrixView(float*, int*, int*, int, int, NonZeros, int, int, int, float**);
template MatrixView<double, MatrixFormat::CSR>::MatrixView(double*, int*, int*, int, int, NonZeros, int, int, int, double**);
template MatrixView<std::complex<float>, MatrixFormat::CSR>::MatrixView(std::complex<float>*, int*, int*, int, int, NonZeros, int, int, int, std::complex<float>**);
template MatrixView<std::complex<double>, MatrixFormat::CSR>::MatrixView(std::complex<double>*, int*, int*, int, int, NonZeros, int, int, int, std::complex<double>**);

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

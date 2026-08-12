#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <batchlas.hh>
#include <batchlas/backend_config.h>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>

#include <algorithm>
#include <array>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace batchlas::python {

template <typename ReturnType = void>
[[noreturn]] inline ReturnType not_implemented(const char* message) {
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw py::error_already_set();
}

[[noreturn]] inline void throw_not_implemented(const char* message) {
    not_implemented(message);
}

template <typename T>
struct type_tag {
    using type = T;
};

enum class DTypeCode {
    Float32,
    Float64,
    Complex64,
    Complex128,
};

template <typename T>
struct dtype_code_of;

template <>
struct dtype_code_of<float> {
    static constexpr DTypeCode value = DTypeCode::Float32;
};

template <>
struct dtype_code_of<double> {
    static constexpr DTypeCode value = DTypeCode::Float64;
};

template <>
struct dtype_code_of<std::complex<float>> {
    static constexpr DTypeCode value = DTypeCode::Complex64;
};

template <>
struct dtype_code_of<std::complex<double>> {
    static constexpr DTypeCode value = DTypeCode::Complex128;
};

inline std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline std::string dtype_name(DTypeCode dtype) {
    switch (dtype) {
        case DTypeCode::Float32:
            return "float32";
        case DTypeCode::Float64:
            return "float64";
        case DTypeCode::Complex64:
            return "complex64";
        case DTypeCode::Complex128:
            return "complex128";
    }
    throw py::value_error("unknown dtype");
}

inline DTypeCode dtype_from_format(const std::string& format) {
    if (format == py::format_descriptor<float>::format()) {
        return DTypeCode::Float32;
    }
    if (format == py::format_descriptor<double>::format()) {
        return DTypeCode::Float64;
    }
    if (format == py::format_descriptor<std::complex<float>>::format()) {
        return DTypeCode::Complex64;
    }
    if (format == py::format_descriptor<std::complex<double>>::format()) {
        return DTypeCode::Complex128;
    }
    throw py::value_error("unsupported dtype; expected float32, float64, complex64, or complex128");
}

template <typename T>
using DenseMatrixT = Matrix<T, MatrixFormat::Dense>;

template <typename T>
using SparseMatrixT = Matrix<T, MatrixFormat::CSR>;

template <typename T>
using GivensMatrixT = Matrix<std::array<T, 2>, MatrixFormat::Dense>;

using DenseMatrixVariant = std::variant<
    DenseMatrixT<float>,
    DenseMatrixT<double>,
    DenseMatrixT<std::complex<float>>,
    DenseMatrixT<std::complex<double>>>;

using DenseVectorVariant = std::variant<
    Vector<float>,
    Vector<double>,
    Vector<std::complex<float>>,
    Vector<std::complex<double>>>;

using SparseMatrixVariant = std::variant<
    SparseMatrixT<float>,
    SparseMatrixT<double>,
    SparseMatrixT<std::complex<float>>,
    SparseMatrixT<std::complex<double>>>;

using ILUKVariant = std::variant<
    ILUKPreconditioner<float>,
    ILUKPreconditioner<double>,
    ILUKPreconditioner<std::complex<float>>,
    ILUKPreconditioner<std::complex<double>>>;

template <typename T>
struct wrapped_scalar;

template <typename T, MatrixFormat MF>
struct wrapped_scalar<Matrix<T, MF>> {
    using type = T;
};

template <typename T>
struct wrapped_scalar<Vector<T>> {
    using type = T;
};

template <typename T>
struct wrapped_scalar<ILUKPreconditioner<T>> {
    using type = T;
};

struct DenseMatrix {
    DenseMatrixVariant storage;

    DTypeCode dtype() const {
        return std::visit([](const auto& matrix) {
            using scalar_type = typename wrapped_scalar<std::decay_t<decltype(matrix)>>::type;
            return dtype_code_of<scalar_type>::value;
        }, storage);
    }

    int rows() const {
        return std::visit([](const auto& matrix) { return matrix.rows(); }, storage);
    }

    int cols() const {
        return std::visit([](const auto& matrix) { return matrix.cols(); }, storage);
    }

    int batch_size() const {
        return std::visit([](const auto& matrix) { return matrix.batch_size(); }, storage);
    }

    bool is_heterogeneous() const {
        return std::visit([](const auto& matrix) { return matrix.is_heterogeneous(); }, storage);
    }
};

struct DenseVector {
    DenseVectorVariant storage;

    DTypeCode dtype() const {
        return std::visit([](const auto& vector) {
            using scalar_type = typename wrapped_scalar<std::decay_t<decltype(vector)>>::type;
            return dtype_code_of<scalar_type>::value;
        }, storage);
    }

    int size() const {
        return std::visit([](const auto& vector) { return vector.size(); }, storage);
    }

    int batch_size() const {
        return std::visit([](const auto& vector) { return vector.batch_size(); }, storage);
    }
};

struct SparseMatrix {
    SparseMatrixVariant storage;

    DTypeCode dtype() const {
        return std::visit([](const auto& matrix) {
            using scalar_type = typename wrapped_scalar<std::decay_t<decltype(matrix)>>::type;
            return dtype_code_of<scalar_type>::value;
        }, storage);
    }

    int rows() const {
        return std::visit([](const auto& matrix) { return matrix.rows(); }, storage);
    }

    int cols() const {
        return std::visit([](const auto& matrix) { return matrix.cols(); }, storage);
    }

    int batch_size() const {
        return std::visit([](const auto& matrix) { return matrix.batch_size(); }, storage);
    }
};

struct ILUKHandle {
    ILUKVariant storage;

    DTypeCode dtype() const {
        return std::visit([](const auto& preconditioner) {
            using scalar_type = typename wrapped_scalar<std::decay_t<decltype(preconditioner)>>::type;
            return dtype_code_of<scalar_type>::value;
        }, storage);
    }

    int n() const {
        return std::visit([](const auto& preconditioner) { return preconditioner.n; }, storage);
    }

    int batch_size() const {
        return std::visit([](const auto& preconditioner) { return preconditioner.batch_size; }, storage);
    }
};

template <typename T>
inline DenseMatrix wrap_dense(DenseMatrixT<T>&& matrix) {
    return DenseMatrix{DenseMatrixVariant(std::move(matrix))};
}

template <typename T>
inline DenseVector wrap_vector(Vector<T>&& vector) {
    return DenseVector{DenseVectorVariant(std::move(vector))};
}

template <typename T>
inline SparseMatrix wrap_sparse(SparseMatrixT<T>&& matrix) {
    return SparseMatrix{SparseMatrixVariant(std::move(matrix))};
}

template <typename T>
inline ILUKHandle wrap_iluk(ILUKPreconditioner<T>&& preconditioner) {
    return ILUKHandle{ILUKVariant(std::move(preconditioner))};
}

template <typename F>
decltype(auto) visit_dtype(DTypeCode dtype, F&& fn) {
    switch (dtype) {
        case DTypeCode::Float32:
            return fn(type_tag<float>{});
        case DTypeCode::Float64:
            return fn(type_tag<double>{});
        case DTypeCode::Complex64:
            return fn(type_tag<std::complex<float>>{});
        case DTypeCode::Complex128:
            return fn(type_tag<std::complex<double>>{});
    }
    throw py::value_error("unknown dtype");
}

template <typename F>
decltype(auto) visit_dense(DenseMatrix& matrix, F&& fn) {
    return visit_dtype(matrix.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<DenseMatrixT<scalar_type>>(matrix.storage));
    });
}

template <typename F>
decltype(auto) visit_dense(const DenseMatrix& matrix, F&& fn) {
    return visit_dtype(matrix.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<DenseMatrixT<scalar_type>>(matrix.storage));
    });
}

template <typename F>
decltype(auto) visit_vector(DenseVector& vector, F&& fn) {
    return visit_dtype(vector.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<Vector<scalar_type>>(vector.storage));
    });
}

template <typename F>
decltype(auto) visit_vector(const DenseVector& vector, F&& fn) {
    return visit_dtype(vector.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<Vector<scalar_type>>(vector.storage));
    });
}

template <typename F>
decltype(auto) visit_sparse(SparseMatrix& matrix, F&& fn) {
    return visit_dtype(matrix.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<SparseMatrixT<scalar_type>>(matrix.storage));
    });
}

template <typename F>
decltype(auto) visit_sparse(const SparseMatrix& matrix, F&& fn) {
    return visit_dtype(matrix.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<SparseMatrixT<scalar_type>>(matrix.storage));
    });
}

template <typename F>
decltype(auto) visit_iluk(ILUKHandle& handle, F&& fn) {
    return visit_dtype(handle.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<ILUKPreconditioner<scalar_type>>(handle.storage));
    });
}

template <typename F>
decltype(auto) visit_iluk(const ILUKHandle& handle, F&& fn) {
    return visit_dtype(handle.dtype(), [&](auto tag) -> decltype(auto) {
        using scalar_type = typename decltype(tag)::type;
        return fn(tag, std::get<ILUKPreconditioner<scalar_type>>(handle.storage));
    });
}

// Turn the queue's backend into a compile-time tag.
//
// This used to be a switch over a `Backend` argument, duplicating the table in
// blas/queue-dispatch.hh. The backend is a property of the Queue now, so the
// table lives in exactly one place and this just forwards to it.
//
// It does not translate exceptions: with_backend only throws when no compiled
// backend matches, and acquire_queue has already rejected that case. Catching
// here would also swallow genuine runtime errors raised from inside `fn` and
// report them as "backend unavailable", which is far worse than the raw error.
template <typename F>
decltype(auto) visit_backend(Queue& queue, F&& fn) {
    return batchlas::with_backend(queue, std::forward<F>(fn));
}

// Size a workspace and run the call with it.
//
// The bytes come from the queue's arena (util/workspace.hh) rather than a fresh
// UnifiedVector per call, so a repeated operation reuses the same allocation
// instead of malloc/free-ing device memory on every invocation. The lease is
// released when this returns; the queue is in-order and every binding waits
// before handing results back to Python, so the next borrower is safely ordered
// behind this call.
//
// Note this now resolves the backend once. The old version called the dispatch
// switch twice -- once to size, once to invoke.
template <typename SizeFn, typename InvokeFn>
void run_backend_with_workspace(Queue& queue, SizeFn&& size_fn, InvokeFn&& invoke_fn) {
    visit_backend(queue, [&](auto backend_tag) {
        auto lease = queue.workspace(static_cast<std::size_t>(size_fn(backend_tag)));
        invoke_fn(backend_tag, lease.span());
    });
}

inline Backend parse_backend(std::string value) {
    value = lower_copy(std::move(value));
    if (value.empty() || value == "auto") {
        return Backend::AUTO;
    }
    if (value == "cuda") {
        return Backend::CUDA;
    }
    if (value == "rocm") {
        return Backend::ROCM;
    }
    if (value == "mkl") {
        return Backend::MKL;
    }
    if (value == "netlib" || value == "host" || value == "cpu") {
        return Backend::NETLIB;
    }
    if (value == "sycl") {
        return Backend::SYCL;
    }
    if (value == "magma") {
        return Backend::MAGMA;
    }
    throw py::value_error("invalid backend");
}

inline ComputePrecision parse_compute_precision(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "default") {
        return ComputePrecision::Default;
    }
    if (value == "f32" || value == "float32") {
        return ComputePrecision::F32;
    }
    if (value == "f64" || value == "float64") {
        return ComputePrecision::F64;
    }
    if (value == "f16" || value == "float16" || value == "half") {
        return ComputePrecision::F16;
    }
    if (value == "bf16" || value == "bfloat16") {
        return ComputePrecision::BF16;
    }
    if (value == "tf32") {
        return ComputePrecision::TF32;
    }
    throw py::value_error("invalid compute_precision");
}

inline Transpose parse_transpose(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "n" || value == "no_trans" || value == "notrans" || value == "no" || value == "none") {
        return Transpose::NoTrans;
    }
    if (value == "t" || value == "trans" || value == "transpose") {
        return Transpose::Trans;
    }
    if (value == "c" || value == "ct" || value == "conj_trans" || value == "conjtrans" || value == "hermitian") {
        return Transpose::ConjTrans;
    }
    throw py::value_error("invalid transpose");
}

inline Side parse_side(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "left" || value == "l") {
        return Side::Left;
    }
    if (value == "right" || value == "r") {
        return Side::Right;
    }
    throw py::value_error("invalid side");
}

inline Uplo parse_uplo(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "upper" || value == "u") {
        return Uplo::Upper;
    }
    if (value == "lower" || value == "l") {
        return Uplo::Lower;
    }
    throw py::value_error("invalid uplo");
}

inline Diag parse_diag(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "non_unit" || value == "nonunit" || value == "n") {
        return Diag::NonUnit;
    }
    if (value == "unit" || value == "u") {
        return Diag::Unit;
    }
    throw py::value_error("invalid diag");
}

inline JobType parse_job_type(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "vectors" || value == "eigenvectors" || value == "vector" || value == "v") {
        return JobType::EigenVectors;
    }
    if (value == "values" || value == "novectors" || value == "no_vectors" || value == "n") {
        return JobType::NoEigenVectors;
    }
    throw py::value_error("invalid job type");
}

inline SortOrder parse_sort_order(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "ascending" || value == "asc") {
        return SortOrder::Ascending;
    }
    if (value == "descending" || value == "desc") {
        return SortOrder::Descending;
    }
    throw py::value_error("invalid sort_order");
}

// LAPACK ?syevx's RANGE argument, spelled the way SyevxOptions.select spells it.
// Deliberately NOT parse_eigen_range_type: EigenRangeType is the tridiagonal
// layer's vocabulary and its `All` member has no SyevxSelect counterpart (syevx's
// default is "the neigs extremal ones", not "all of them").
inline SyevxSelect parse_syevx_select(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "extremal") {
        return SyevxSelect::Extremal;
    }
    if (value == "index" || value == "i") {
        return SyevxSelect::Index;
    }
    if (value == "value" || value == "v") {
        return SyevxSelect::Value;
    }
    throw py::value_error("invalid syevx select: expected 'extremal', 'index' or 'value'");
}

inline OrthoAlgorithm parse_ortho_algorithm(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "chol2") {
        return OrthoAlgorithm::Chol2;
    }
    if (value == "cholesky") {
        return OrthoAlgorithm::Cholesky;
    }
    if (value == "shiftchol3" || value == "shift_chol3") {
        return OrthoAlgorithm::ShiftChol3;
    }
    if (value == "householder") {
        return OrthoAlgorithm::Householder;
    }
    if (value == "cgs2") {
        return OrthoAlgorithm::CGS2;
    }
    if (value == "svqb") {
        return OrthoAlgorithm::SVQB;
    }
    if (value == "svqb2") {
        return OrthoAlgorithm::SVQB2;
    }
    throw py::value_error("invalid orthogonalization algorithm");
}

inline NormType parse_norm_type(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "fro" || value == "frobenius") {
        return NormType::Frobenius;
    }
    if (value == "one" || value == "1") {
        return NormType::One;
    }
    if (value == "inf" || value == "infinity") {
        return NormType::Inf;
    }
    if (value == "max") {
        return NormType::Max;
    }
    if (value == "spectral" || value == "2") {
        return NormType::Spectral;
    }
    throw py::value_error("invalid norm type");
}

inline SvdVectors parse_svd_vectors(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "none" || value == "n") {
        return SvdVectors::None;
    }
    if (value == "all" || value == "a") {
        return SvdVectors::All;
    }
    throw py::value_error("invalid svd vector policy");
}

inline SteqrShiftStrategy parse_steqr_shift_strategy(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "lapack") {
        return SteqrShiftStrategy::Lapack;
    }
    if (value == "wilkinson") {
        return SteqrShiftStrategy::Wilkinson;
    }
    throw py::value_error("invalid steqr shift strategy");
}

inline SteqrUpdateScheme parse_steqr_update_scheme(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "pg") {
        return SteqrUpdateScheme::PG;
    }
    if (value == "exp" || value == "explicit") {
        return SteqrUpdateScheme::EXP;
    }
    throw py::value_error("invalid steqr update scheme");
}

inline StedcSecularSolver parse_stedc_secular_solver(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "rocm") {
        return StedcSecularSolver::Rocm;
    }
    if (value == "legacy") {
        return StedcSecularSolver::Legacy;
    }
    throw py::value_error("invalid stedc secular_solver");
}

inline StedcMergeVariant parse_stedc_merge_variant(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "auto") {
        return StedcMergeVariant::Auto;
    }
    if (value == "baseline") {
        return StedcMergeVariant::Baseline;
    }
    if (value == "fused") {
        return StedcMergeVariant::Fused;
    }
    if (value == "fused_cta" || value == "fusedcta") {
        return StedcMergeVariant::FusedCta;
    }
    throw py::value_error("invalid stedc merge_variant");
}

inline OrmqCtaFactorization parse_ormqx_factorization(std::string value) {
    value = lower_copy(std::move(value));
    if (value == "qr") {
        return OrmqCtaFactorization::QR;
    }
    if (value == "ql") {
        return OrmqCtaFactorization::QL;
    }
    throw py::value_error("invalid ormqx factorization");
}

inline Device resolve_device(const std::optional<std::string>& device_name) {
    if (!device_name.has_value()) {
        return Device::default_device();
    }

    const std::string value = lower_copy(*device_name);
    if (value.empty() || value == "default") {
        return Device::default_device();
    }
    if (value == "cpu" || value == "gpu" || value == "accelerator") {
        return Device(value);
    }
    throw py::value_error("device must be one of: default, cpu, gpu, accelerator");
}

inline std::string device_cache_key(const std::optional<std::string>& device_name) {
    if (!device_name.has_value()) {
        return "default";
    }
    const std::string value = lower_copy(*device_name);
    return value.empty() ? "default" : value;
}

// One Queue per (device, backend), created on first use and reused thereafter.
//
// This is a cache rather than a fresh Queue per call for three reasons:
//
//   * The Queue owns the workspace arena. A Queue that lives for one call gives
//     an arena that lives for one call, so every call re-allocates the workspace
//     it needs and frees it again -- the arena can only amortise across calls if
//     the queue outlives them. This is the change that makes P2 pay off here.
//   * Constructing a SYCL queue is not free, and the bindings were doing it on
//     every single operation.
//   * The backend is a property of the Queue, so it belongs in the cache key
//     rather than being re-selected at each call site.
//
// Deliberately never destroyed. ~QueueImpl waits on the queue and frees the
// arena against the SYCL context; at static-destruction time that context may
// already be gone, so running these destructors at exit risks a hang or crash
// for no benefit -- the process is ending regardless. The map is heap-allocated
// and leaked so neither it nor the Queues it owns are destroyed.
//
// Thread safety: these bindings never release the GIL, so calls into BatchLAS
// are serialised and the shared arena only ever has one user at a time. Adding
// a py::gil_scoped_release around any call below would break that assumption and
// require per-thread queues.
inline Queue& acquire_queue(const std::optional<std::string>& device_name, Backend backend) {
    const Device device = resolve_device(device_name);  // validates the name

    // Checked here rather than inside with_backend so the error stays a Python
    // NotImplementedError, which is what callers have always seen.
    if (backend != Backend::AUTO && !Queue::backend_available(backend)) {
        throw_not_implemented("requested backend is not available in this build");
    }

    using Key = std::pair<std::string, Backend>;
    static auto& cache = *new std::map<Key, std::unique_ptr<Queue>>();

    const Key key{device_cache_key(device_name), backend};
    auto it = cache.find(key);
    if (it == cache.end()) {
        // Backend::AUTO is passed through rather than resolved here: the Queue
        // resolves it from the device vendor on first use. That is a behaviour
        // change -- the old code picked the first *compiled* backend, so
        // backend='auto' with device='cpu' selected CUDA on a CUDA build.
        it = cache.emplace(key, std::make_unique<Queue>(device, backend)).first;
    }
    return *it->second;
}

template <typename T>
DenseMatrixT<T> dense_matrix_from_array_t(const py::array_t<T, py::array::forcecast>& array) {
    const py::buffer_info info = array.request();
    if (info.ndim != 2 && info.ndim != 3) {
        throw py::value_error("dense arrays must have shape (m, n) or (batch, m, n)");
    }

    const int batch_size = info.ndim == 2 ? 1 : static_cast<int>(info.shape[0]);
    const int rows = static_cast<int>(info.shape[info.ndim - 2]);
    const int cols = static_cast<int>(info.shape[info.ndim - 1]);
    DenseMatrixT<T> matrix(rows, cols, batch_size);

    if (info.ndim == 2) {
        auto view = array.template unchecked<2>();
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                matrix(row, col, 0) = view(row, col);
            }
        }
    } else {
        auto view = array.template unchecked<3>();
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    matrix(row, col, batch) = view(batch, row, col);
                }
            }
        }
    }

    return matrix;
}

template <typename T>
Vector<T> dense_vector_from_array_t(const py::array_t<T, py::array::forcecast>& array) {
    const py::buffer_info info = array.request();
    if (info.ndim != 1 && info.ndim != 2) {
        throw py::value_error("vector arrays must have shape (n) or (batch, n)");
    }

    const int batch_size = info.ndim == 1 ? 1 : static_cast<int>(info.shape[0]);
    const int size = static_cast<int>(info.shape[info.ndim - 1]);
    Vector<T> vector(size, batch_size);

    if (info.ndim == 1) {
        auto view = array.template unchecked<1>();
        for (int index = 0; index < size; ++index) {
            vector(index, 0) = view(index);
        }
    } else {
        auto view = array.template unchecked<2>();
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int index = 0; index < size; ++index) {
                vector(index, batch) = view(batch, index);
            }
        }
    }

    return vector;
}

template <typename T>
DenseMatrixT<T> heterogeneous_dense_matrix_from_sequence_t(const py::sequence& items) {
    const py::ssize_t batch_size = py::len(items);
    if (batch_size <= 0) {
        throw py::value_error("heterogeneous batches must not be empty");
    }

    int max_rows = 0;
    int max_cols = 0;
    std::vector<py::array_t<T, py::array::forcecast>> arrays;
    arrays.reserve(static_cast<std::size_t>(batch_size));
    std::vector<int> active_rows(static_cast<std::size_t>(batch_size));
    std::vector<int> active_cols(static_cast<std::size_t>(batch_size));

    for (py::ssize_t batch = 0; batch < batch_size; ++batch) {
        py::array_t<T, py::array::forcecast> array(items[batch]);
        const py::buffer_info info = array.request();
        if (info.ndim != 2) {
            throw py::value_error("heterogeneous dense batches require 2D arrays");
        }

        const int rows = static_cast<int>(info.shape[0]);
        const int cols = static_cast<int>(info.shape[1]);
        max_rows = std::max(max_rows, rows);
        max_cols = std::max(max_cols, cols);
        active_rows[static_cast<std::size_t>(batch)] = rows;
        active_cols[static_cast<std::size_t>(batch)] = cols;
        arrays.push_back(std::move(array));
    }

    DenseMatrixT<T> matrix(max_rows, max_cols, static_cast<int>(batch_size));
    matrix.fill(T(0));

    for (py::ssize_t batch = 0; batch < batch_size; ++batch) {
        auto view = arrays[static_cast<std::size_t>(batch)].template unchecked<2>();
        const int rows = active_rows[static_cast<std::size_t>(batch)];
        const int cols = active_cols[static_cast<std::size_t>(batch)];
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                matrix(row, col, static_cast<int>(batch)) = view(row, col);
            }
        }
    }

    UnifiedVector<int> rows_meta(static_cast<std::size_t>(batch_size));
    UnifiedVector<int> cols_meta(static_cast<std::size_t>(batch_size));
    for (py::ssize_t batch = 0; batch < batch_size; ++batch) {
        rows_meta[static_cast<std::size_t>(batch)] = active_rows[static_cast<std::size_t>(batch)];
        cols_meta[static_cast<std::size_t>(batch)] = active_cols[static_cast<std::size_t>(batch)];
    }
    matrix.set_active_dims(rows_meta.to_span(), cols_meta.to_span());
    return matrix;
}

template <typename T>
SparseMatrixT<T> sparse_matrix_from_csr_objects_t(const std::vector<py::object>& items) {
    if (items.empty()) {
        throw py::value_error("sparse batches must not be empty");
    }

    int rows = -1;
    int cols = -1;
    int max_nnz = 0;
    struct csr_payload {
        py::array_t<T, py::array::forcecast> data;
        py::array_t<int, py::array::forcecast> indices;
        py::array_t<int, py::array::forcecast> indptr;
    };
    std::vector<csr_payload> payloads;
    payloads.reserve(items.size());

    for (const py::object& item : items) {
        py::tuple shape = item.attr("shape");
        const int item_rows = shape[0].cast<int>();
        const int item_cols = shape[1].cast<int>();
        if (rows < 0) {
            rows = item_rows;
            cols = item_cols;
        } else if (rows != item_rows || cols != item_cols) {
            throw py::value_error("all sparse matrices in a batch must have the same shape");
        }

        csr_payload payload{
            py::array_t<T, py::array::forcecast>(item.attr("data")),
            py::array_t<int, py::array::forcecast>(item.attr("indices")),
            py::array_t<int, py::array::forcecast>(item.attr("indptr")),
        };
        max_nnz = std::max(max_nnz, static_cast<int>(payload.data.request().shape[0]));
        payloads.push_back(std::move(payload));
    }

    SparseMatrixT<T> matrix(rows, cols, NonZeros{max_nnz}, static_cast<int>(items.size()));
    std::fill(matrix.data().begin(), matrix.data().end(), T(0));
    std::fill(matrix.col_indices().begin(), matrix.col_indices().end(), 0);
    std::fill(matrix.row_offsets().begin(), matrix.row_offsets().end(), 0);

    for (std::size_t batch = 0; batch < payloads.size(); ++batch) {
        const auto data = payloads[batch].data.template unchecked<1>();
        const auto indices = payloads[batch].indices.template unchecked<1>();
        const auto indptr = payloads[batch].indptr.template unchecked<1>();
        const int batch_offset = static_cast<int>(batch) * matrix.matrix_stride();
        const int row_offset = static_cast<int>(batch) * matrix.offset_stride();
        for (ssize_t index = 0; index < data.shape(0); ++index) {
            matrix.data()[static_cast<std::size_t>(batch_offset + static_cast<int>(index))] = data(index);
            matrix.col_indices()[static_cast<std::size_t>(batch_offset + static_cast<int>(index))] = indices(index);
        }
        for (ssize_t index = 0; index < indptr.shape(0); ++index) {
            matrix.row_offsets()[static_cast<std::size_t>(row_offset + static_cast<int>(index))] = indptr(index);
        }
    }

    return matrix;
}

inline DenseMatrix dense_matrix_from_numpy(const py::array& array) {
    const DTypeCode dtype = dtype_from_format(array.request().format);
    return visit_dtype(dtype, [&](auto tag) {
        using scalar_type = typename decltype(tag)::type;
        return wrap_dense(dense_matrix_from_array_t<scalar_type>(py::array_t<scalar_type, py::array::forcecast>(array)));
    });
}

inline DenseVector dense_vector_from_numpy(const py::array& array) {
    const DTypeCode dtype = dtype_from_format(array.request().format);
    return visit_dtype(dtype, [&](auto tag) {
        using scalar_type = typename decltype(tag)::type;
        return wrap_vector(dense_vector_from_array_t<scalar_type>(py::array_t<scalar_type, py::array::forcecast>(array)));
    });
}

inline DenseMatrix heterogeneous_dense_matrix_from_sequence(const py::sequence& items) {
    if (py::len(items) == 0) {
        throw py::value_error("heterogeneous batches must not be empty");
    }
    py::array first = py::array::ensure(items[0]);
    if (!first) {
        throw py::value_error("heterogeneous batches must contain numpy arrays");
    }
    const DTypeCode dtype = dtype_from_format(first.request().format);
    return visit_dtype(dtype, [&](auto tag) {
        using scalar_type = typename decltype(tag)::type;
        return wrap_dense(heterogeneous_dense_matrix_from_sequence_t<scalar_type>(items));
    });
}

inline SparseMatrix sparse_matrix_from_python(const py::object& object) {
    py::object csr = object;
    if (py::hasattr(object, "tocsr")) {
        csr = object.attr("tocsr")();
    }
    py::array data = py::array::ensure(csr.attr("data"));
    if (!data) {
        throw py::value_error("expected a SciPy CSR-like object");
    }
    const DTypeCode dtype = dtype_from_format(data.request().format);
    return visit_dtype(dtype, [&](auto tag) {
        using scalar_type = typename decltype(tag)::type;
        std::vector<py::object> items{csr};
        return wrap_sparse(sparse_matrix_from_csr_objects_t<scalar_type>(items));
    });
}

inline SparseMatrix sparse_matrix_batch_from_sequence(const py::sequence& items) {
    if (py::len(items) == 0) {
        throw py::value_error("sparse batches must not be empty");
    }
    py::object first = items[0];
    py::object first_csr = py::hasattr(first, "tocsr") ? first.attr("tocsr")() : first;
    py::array data = py::array::ensure(first_csr.attr("data"));
    if (!data) {
        throw py::value_error("expected SciPy CSR-like objects");
    }
    const DTypeCode dtype = dtype_from_format(data.request().format);
    return visit_dtype(dtype, [&](auto tag) {
        using scalar_type = typename decltype(tag)::type;
        std::vector<py::object> converted;
        converted.reserve(static_cast<std::size_t>(py::len(items)));
        for (py::handle item : items) {
            py::object matrix = py::reinterpret_borrow<py::object>(item);
            converted.push_back(py::hasattr(matrix, "tocsr") ? matrix.attr("tocsr")() : matrix);
        }
        return wrap_sparse(sparse_matrix_from_csr_objects_t<scalar_type>(converted));
    });
}

template <typename T>
py::array_t<T> dense_matrix_to_numpy_t(const DenseMatrixT<T>& matrix) {
    if (matrix.batch_size() == 1) {
        py::array_t<T> out({matrix.rows(), matrix.cols()});
        auto view = out.template mutable_unchecked<2>();
        for (int row = 0; row < matrix.rows(); ++row) {
            for (int col = 0; col < matrix.cols(); ++col) {
                view(row, col) = matrix(row, col, 0);
            }
        }
        return out;
    }

    py::array_t<T> out({matrix.batch_size(), matrix.rows(), matrix.cols()});
    auto view = out.template mutable_unchecked<3>();
    for (int batch = 0; batch < matrix.batch_size(); ++batch) {
        for (int row = 0; row < matrix.rows(); ++row) {
            for (int col = 0; col < matrix.cols(); ++col) {
                view(batch, row, col) = matrix(row, col, batch);
            }
        }
    }
    return out;
}

template <typename T>
py::object heterogeneous_dense_matrix_to_python_t(const DenseMatrixT<T>& matrix) {
    if (!matrix.is_heterogeneous()) {
        return dense_matrix_to_numpy_t(matrix);
    }

    py::list result;
    for (int batch = 0; batch < matrix.batch_size(); ++batch) {
        py::array_t<T> item({matrix.rows(batch), matrix.cols(batch)});
        auto view = item.template mutable_unchecked<2>();
        for (int row = 0; row < matrix.rows(batch); ++row) {
            for (int col = 0; col < matrix.cols(batch); ++col) {
                view(row, col) = matrix(row, col, batch);
            }
        }
        result.append(std::move(item));
    }
    return std::move(result);
}

inline py::object dense_matrix_to_python(const DenseMatrix& matrix) {
    return visit_dense(matrix, [&](auto, const auto& typed_matrix) -> py::object {
        return heterogeneous_dense_matrix_to_python_t(typed_matrix);
    });
}

template <typename T>
py::array_t<T> dense_vector_to_numpy_t(const Vector<T>& vector) {
    const auto data = vector.data();
    if (vector.batch_size() == 1) {
        py::array_t<T> out({vector.size()});
        auto view = out.template mutable_unchecked<1>();
        for (int index = 0; index < vector.size(); ++index) {
            view(index) = data[static_cast<std::size_t>(index * vector.inc())];
        }
        return out;
    }

    py::array_t<T> out({vector.batch_size(), vector.size()});
    auto view = out.template mutable_unchecked<2>();
    for (int batch = 0; batch < vector.batch_size(); ++batch) {
        for (int index = 0; index < vector.size(); ++index) {
            view(batch, index) =
                data[static_cast<std::size_t>(batch * vector.stride() + index * vector.inc())];
        }
    }
    return out;
}

inline py::object dense_vector_to_python(const DenseVector& vector) {
    return visit_vector(vector, [&](auto, const auto& typed_vector) -> py::object {
        return dense_vector_to_numpy_t(typed_vector);
    });
}

template <typename T>
py::object sparse_matrix_to_python_t(const SparseMatrixT<T>& matrix) {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    auto make_one = [&](int batch) -> py::object {
        // Not matrix.nnz(): that is the per-item capacity, sized by the batch maximum,
        // and this batch is heterogeneous by construction. nnz(batch) is the same row
        // offset read this line used to spell by hand.
        const int nnz = matrix.nnz(batch);
        py::array_t<T> data({nnz});
        py::array_t<int> indices({nnz});
        py::array_t<int> indptr({matrix.rows() + 1});
        auto data_view = data.template mutable_unchecked<1>();
        auto index_view = indices.template mutable_unchecked<1>();
        auto indptr_view = indptr.template mutable_unchecked<1>();
        const int data_offset = batch * matrix.matrix_stride();
        const int row_offset = batch * matrix.offset_stride();
        for (int index = 0; index < nnz; ++index) {
            data_view(index) = matrix.data()[static_cast<std::size_t>(data_offset + index)];
            index_view(index) = matrix.col_indices()[static_cast<std::size_t>(data_offset + index)];
        }
        for (int index = 0; index <= matrix.rows(); ++index) {
            indptr_view(index) = matrix.row_offsets()[static_cast<std::size_t>(row_offset + index)];
        }
        return scipy_sparse.attr("csr_matrix")(py::make_tuple(data, indices, indptr),
                                               py::make_tuple(matrix.rows(), matrix.cols()));
    };

    if (matrix.batch_size() == 1) {
        return make_one(0);
    }

    py::list result;
    for (int batch = 0; batch < matrix.batch_size(); ++batch) {
        result.append(make_one(batch));
    }
    return std::move(result);
}

inline py::object sparse_matrix_to_python(const SparseMatrix& matrix) {
    return visit_sparse(matrix, [&](auto, const auto& typed_matrix) -> py::object {
        return sparse_matrix_to_python_t(typed_matrix);
    });
}

template <typename T>
py::dict iluk_metadata_t(const ILUKPreconditioner<T>& preconditioner) {
    py::dict result;
    result["n"] = preconditioner.n;
    result["batch_size"] = preconditioner.batch_size;
    result["levels_of_fill"] = preconditioner.levels_of_fill;
    result["dtype"] = dtype_name(dtype_code_of<T>::value);
    return result;
}

inline py::dict iluk_metadata(const ILUKHandle& handle) {
    return visit_iluk(handle, [&](auto, const auto& typed_handle) {
        return iluk_metadata_t(typed_handle);
    });
}

template <typename T>
py::dict device_to_dict(const Device& device, const std::string& type_name) {
    py::dict result;
    result["name"] = device.get_name();
    result["vendor"] = static_cast<int>(device.get_vendor());
    result["type"] = type_name;
    result["index"] = device.idx;
    return result;
}

inline py::list available_devices() {
    py::list devices;
    for (const Device& device : Device::get_devices(DeviceType::CPU)) {
        devices.append(device_to_dict<float>(device, "cpu"));
    }
    for (const Device& device : Device::get_devices(DeviceType::GPU)) {
        devices.append(device_to_dict<float>(device, "gpu"));
    }
    for (const Device& device : Device::get_devices(DeviceType::ACCELERATOR)) {
        devices.append(device_to_dict<float>(device, "accelerator"));
    }
    return devices;
}

inline py::list available_backends() {
    py::list backends;
    backends.append("auto");
#if BATCHLAS_HAS_CUDA_BACKEND
    backends.append("cuda");
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    backends.append("rocm");
#endif
#if BATCHLAS_HAS_MKL_BACKEND
    backends.append("mkl");
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    backends.append("netlib");
#endif
    return backends;
}

inline py::dict compiled_features() {
    py::dict features;
    features["has_host_backend"] = static_cast<bool>(BATCHLAS_HAS_HOST_BACKEND);
    features["has_cuda_backend"] = static_cast<bool>(BATCHLAS_HAS_CUDA_BACKEND);
    features["has_rocm_backend"] = static_cast<bool>(BATCHLAS_HAS_ROCM_BACKEND);
    features["has_mkl_backend"] = static_cast<bool>(BATCHLAS_HAS_MKL_BACKEND);
    features["has_cpu_target"] = static_cast<bool>(BATCHLAS_HAS_CPU_TARGET);
    features["has_gpu_backend"] = static_cast<bool>(BATCHLAS_HAS_GPU_BACKEND);
    features["backends"] = available_backends();
    return features;
}

template <typename T>
T cast_scalar(const py::object& value) {
    return value.cast<T>();
}

template <typename T>
T scalar_from_object(const py::object& value) {
    return value.cast<T>();
}

template <typename T>
T py_scalar_or_default(const py::dict& options, const char* key, T default_value) {
    if (!options.contains(py::str(key))) {
        return default_value;
    }
    return options[py::str(key)].cast<T>();
}

inline std::optional<std::string> optional_string_from_obj(const py::object& object) {
    if (object.is_none()) {
        return std::nullopt;
    }
    return object.cast<std::string>();
}

inline py::dict ensure_options_dict(const py::object& object) {
    if (object.is_none()) {
        return py::dict();
    }
    return object.cast<py::dict>();
}

template <typename Left, typename Right>
inline void ensure_same_dtype(const Left& lhs, const Right& rhs, std::string_view message) {
    if (lhs.dtype() != rhs.dtype()) {
        throw py::value_error(std::string(message));
    }
}

inline void ensure_dense_dtype(const DenseMatrix& matrix, DTypeCode dtype, std::string_view message) {
    if (matrix.dtype() != dtype) {
        throw py::value_error(std::string(message));
    }
}

inline void ensure_vector_dtype(const DenseVector& vector, DTypeCode dtype, std::string_view message) {
    if (vector.dtype() != dtype) {
        throw py::value_error(std::string(message));
    }
}

template <typename T>
inline DenseVector make_real_vector(int size, int batch_size) {
    using real_type = typename base_type<T>::type;
    return wrap_vector(Vector<real_type>(size, batch_size));
}

template <typename T>
inline DenseVector make_vector(int size, int batch_size) {
    return wrap_vector(Vector<T>(size, batch_size));
}

template <typename T>
inline DenseVector wrap_vector_copy(const UnifiedVector<T>& values, int size, int batch_size) {
    Vector<T> vector(size, batch_size);
    for (std::size_t index = 0; index < values.size(); ++index) {
        vector[index] = values[index];
    }
    return wrap_vector(std::move(vector));
}

template <typename T>
SyevxParams<T> parse_syevx_params(const py::dict& options,
                                  const ILUKPreconditioner<T>* preconditioner = nullptr,
                                  SyevxInstrumentation<T>* instrumentation = nullptr) {
    SyevxParams<T> params;
    if (options.contains("algorithm")) {
        params.algorithm = parse_ortho_algorithm(py::cast<std::string>(options["algorithm"]));
    }
    params.ortho_iterations = py_scalar_or_default<std::size_t>(options, "ortho_iterations", params.ortho_iterations);
    params.iterations = py_scalar_or_default<std::size_t>(options, "iterations", params.iterations);
    params.extra_directions = py_scalar_or_default<std::size_t>(options, "extra_directions", params.extra_directions);
    params.find_largest = py_scalar_or_default<bool>(options, "find_largest", params.find_largest);
    params.absolute_tolerance = py_scalar_or_default<T>(options, "absolute_tolerance", params.absolute_tolerance);
    params.relative_tolerance = py_scalar_or_default<T>(options, "relative_tolerance", params.relative_tolerance);
    params.preconditioner = preconditioner;
    // Opt-in in-solver ILU(k): syevx forms the factor from its own workspace, so
    // an end-to-end timing from Python covers formation as well as application.
    params.build_preconditioner =
        py_scalar_or_default<bool>(options, "build_preconditioner", params.build_preconditioner);
    params.iluk_params.levels_of_fill =
        py_scalar_or_default<int>(options, "iluk_levels_of_fill", params.iluk_params.levels_of_fill);
    params.iluk_params.drop_tolerance = py_scalar_or_default<typename base_type<T>::type>(
        options, "iluk_drop_tolerance", params.iluk_params.drop_tolerance);
    params.iluk_params.fill_factor = py_scalar_or_default<typename base_type<T>::type>(
        options, "iluk_fill_factor", params.iluk_params.fill_factor);
    // Range selection. vl/vu/abstol are base_type<T>, not T: the eigenvalues of a
    // Hermitian matrix are real, so a complex caller must not have to spell
    // complex(vl) for a real quantity. See SyevxParams's comment on the same point.
    if (options.contains(py::str("select"))) {
        params.select = parse_syevx_select(py::cast<std::string>(options["select"]));
    }
    params.il = py_scalar_or_default<std::int64_t>(options, "il", params.il);
    params.iu = py_scalar_or_default<std::int64_t>(options, "iu", params.iu);
    params.vl = py_scalar_or_default<typename base_type<T>::type>(options, "vl", params.vl);
    params.vu = py_scalar_or_default<typename base_type<T>::type>(options, "vu", params.vu);
    params.abstol = py_scalar_or_default<typename base_type<T>::type>(options, "abstol", params.abstol);
    if (options.contains(py::str("order"))) {
        params.order = parse_sort_order(py::cast<std::string>(options["order"]));
    }
    params.instrumentation = instrumentation;
    return params;
}

template <typename T>
LanczosParams<T> parse_lanczos_params(const py::dict& options) {
    LanczosParams<T> params;
    if (options.contains("ortho_algorithm")) {
        params.ortho_algorithm = parse_ortho_algorithm(py::cast<std::string>(options["ortho_algorithm"]));
    }
    params.ortho_iterations = py_scalar_or_default<std::size_t>(options, "ortho_iterations", params.ortho_iterations);
    params.reorthogonalization_iterations =
        py_scalar_or_default<std::size_t>(options, "reorthogonalization_iterations", params.reorthogonalization_iterations);
    params.sort_enabled = py_scalar_or_default<bool>(options, "sort_enabled", params.sort_enabled);
    if (options.contains("sort_order")) {
        params.sort_order = parse_sort_order(py::cast<std::string>(options["sort_order"]));
    }
    return params;
}

template <typename T>
SteqrParams<T> parse_steqr_params(const py::dict& options) {
    SteqrParams<T> params;
    params.block_size = py_scalar_or_default<std::size_t>(options, "block_size", params.block_size);
    params.max_sweeps = py_scalar_or_default<std::size_t>(options, "max_sweeps", params.max_sweeps);
    params.zero_threshold = py_scalar_or_default<T>(options, "zero_threshold", params.zero_threshold);
    params.back_transform = py_scalar_or_default<bool>(options, "back_transform", params.back_transform);
    params.block_rotations = py_scalar_or_default<bool>(options, "block_rotations", params.block_rotations);
    params.sort = py_scalar_or_default<bool>(options, "sort", params.sort);
    params.transpose_working_vectors =
        py_scalar_or_default<bool>(options, "transpose_working_vectors", params.transpose_working_vectors);
    params.cta_wg_size_multiplier =
        py_scalar_or_default<std::size_t>(options, "cta_wg_size_multiplier", params.cta_wg_size_multiplier);
    if (options.contains("sort_order")) {
        params.sort_order = parse_sort_order(py::cast<std::string>(options["sort_order"]));
    }
    if (options.contains("cta_shift_strategy")) {
        params.cta_shift_strategy =
            parse_steqr_shift_strategy(py::cast<std::string>(options["cta_shift_strategy"]));
    }
    if (options.contains("cta_update_scheme")) {
        params.cta_update_scheme =
            parse_steqr_update_scheme(py::cast<std::string>(options["cta_update_scheme"]));
    }
    return params;
}

template <typename T>
StedcParams<T> parse_stedc_params(const py::dict& options) {
    StedcParams<T> params;
    params.recursion_threshold =
        py_scalar_or_default<std::int64_t>(options, "recursion_threshold", params.recursion_threshold);
    params.merge_threads = py_scalar_or_default<int>(options, "merge_threads", params.merge_threads);
    params.max_sec_iter = py_scalar_or_default<int>(options, "max_sec_iter", params.max_sec_iter);
    params.enable_rescale = py_scalar_or_default<bool>(options, "enable_rescale", params.enable_rescale);
    params.secular_threads_per_root =
        py_scalar_or_default<int>(options, "secular_threads_per_root", params.secular_threads_per_root);
    params.secular_cta_wg_size_multiplier =
        py_scalar_or_default<int>(options, "secular_cta_wg_size_multiplier", params.secular_cta_wg_size_multiplier);
    if (options.contains("secular_solver")) {
        params.secular_solver = parse_stedc_secular_solver(py::cast<std::string>(options["secular_solver"]));
    }
    if (options.contains("merge_variant")) {
        params.merge_variant = parse_stedc_merge_variant(py::cast<std::string>(options["merge_variant"]));
    }
    if (options.contains("leaf_steqr_params")) {
        params.leaf_steqr_params = parse_steqr_params<T>(options["leaf_steqr_params"].cast<py::dict>());
    }
    return params;
}

template <typename T>
ILUKParams<T> parse_iluk_params(const py::dict& options) {
    ILUKParams<T> params;
    params.levels_of_fill = py_scalar_or_default<int>(options, "levels_of_fill", params.levels_of_fill);
    params.diagonal_shift = py_scalar_or_default<T>(options, "diagonal_shift", params.diagonal_shift);
    params.drop_tolerance = py_scalar_or_default<typename base_type<T>::type>(
        options, "drop_tolerance", params.drop_tolerance);
    params.fill_factor = py_scalar_or_default<typename base_type<T>::type>(
        options, "fill_factor", params.fill_factor);
    params.diag_pivot_threshold = py_scalar_or_default<typename base_type<T>::type>(
        options, "diag_pivot_threshold", params.diag_pivot_threshold);
    params.modified_ilu = py_scalar_or_default<bool>(options, "modified_ilu", params.modified_ilu);
    params.validate_batch_sparsity =
        py_scalar_or_default<bool>(options, "validate_batch_sparsity", params.validate_batch_sparsity);
    return params;
}

inline SytrdBandReductionParams parse_sytrd_band_reduction_params(const py::dict& options) {
    SytrdBandReductionParams params;
    if (options.contains("d_seq")) {
        params.d_seq = options["d_seq"].cast<std::vector<int32_t>>();
    }
    if (options.contains("block_size_seq")) {
        params.block_size_seq = options["block_size_seq"].cast<std::vector<int32_t>>();
    }
    params.max_sweeps = py_scalar_or_default<int32_t>(options, "max_sweeps", params.max_sweeps);
    params.max_steps = py_scalar_or_default<int32_t>(options, "max_steps", params.max_steps);
    params.kd_work = py_scalar_or_default<int32_t>(options, "kd_work", params.kd_work);
    return params;
}

template <typename T>
JacobiParams<T> parse_jacobi_params(const py::dict& options) {
    using real_type = typename base_type<T>::type;
    JacobiParams<T> params;
    params.tol_multiplier = py_scalar_or_default<real_type>(options, "tol_multiplier", params.tol_multiplier);
    params.max_sweeps = py_scalar_or_default<std::size_t>(options, "max_sweeps", params.max_sweeps);
    params.sort = py_scalar_or_default<bool>(options, "sort", params.sort);
    params.cta_wg_size_multiplier =
        py_scalar_or_default<std::size_t>(options, "cta_wg_size_multiplier", params.cta_wg_size_multiplier);
    if (options.contains("sort_order")) {
        params.sort_order = parse_sort_order(py::cast<std::string>(options["sort_order"]));
    }
    return params;
}

}  // namespace batchlas::python

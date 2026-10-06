#pragma once

/// @file
/// @brief The BatchLAS exception hierarchy.
///
/// Every failure BatchLAS diagnoses is thrown as one of the classes below. Each derives from the
/// `std::` exception the site threw historically (so existing handlers and the pybind11 type
/// mapping are unchanged) and, virtually, from the empty tag batchlas::exception.
///
/// Three catches answer three questions: `catch (const batchlas::workspace_error&)` (a specific
/// reason), `catch (const batchlas::exception&)` (anything BatchLAS diagnosed), and
/// `catch (const std::exception&)` (that, plus `std::bad_alloc` and `sycl::exception`, which are
/// deliberately not wrapped).
/// @see @ref design_error_model
/// @ingroup errors
// Every class here must carry BATCHLAS_API even with no out-of-line member, or catch-by-type
// breaks under -fvisibility=hidden; a throw/catch test on x86-64 cannot detect it.
// evidence: docs/design/symbol-visibility.md#symbol-visibility-exception-typeinfo-must-be-exported

#include <stdexcept>
#include <string>

#include <batchlas/export.hh>

namespace batchlas {

/// @brief Empty tag base of every BatchLAS exception.
///
/// `catch (const batchlas::exception& e)` matches every class in this header and nothing else.
/// Read the text with message(); the tag deliberately has no what().
/// @ingroup errors
// Tag rules, each fails SILENTLY if broken: (1) never derive from std::exception, (2) every leaf
// inherits it virtually (only via detail::exception_bridge), (3) never declare what().
// evidence: docs/design/error-model.md#error-model-the-three-tag-base-rules
class BATCHLAS_API exception {
public:
    virtual ~exception() = default;

    /// @brief The exception's message; the same string `what()` returns on the `std::` base.
    virtual const char* message() const noexcept = 0;

    // Public so no access check stands between a leaf's implicit copy ctor and this virtual base.
    exception() = default;
    exception(const exception&) = default;
    exception& operator=(const exception&) = default;
};

namespace detail {

/// @brief Derives from `StdBase` and, virtually, from batchlas::exception.
///
/// The single spelling of that inheritance, so rule 2 of the tag cannot be broken per class.
/// @tparam StdBase the `std::` exception type the leaf extends
/// @ingroup errors
template <typename StdBase>
class BATCHLAS_API exception_bridge : public StdBase, public virtual exception {
public:
    using StdBase::StdBase;
    const char* message() const noexcept override { return StdBase::what(); }
};

}  // namespace detail

/// @brief The call violates the API contract.
///
/// Thrown for a non-square view where a square one is required, mismatched batch sizes, a span
/// too short for the batch, a negative dimension, a null or non-USM pointer, an out-of-order
/// Queue where an in-order one is required, or an enum value with no meaning here.
///
/// Not retryable: fix the call. Derives from `std::invalid_argument`, which pybind11 maps to
/// Python's `ValueError`.
/// @ingroup errors
class BATCHLAS_API invalid_argument : public detail::exception_bridge<std::invalid_argument> {
public:
    explicit invalid_argument(const std::string& what_arg)
        : detail::exception_bridge<std::invalid_argument>(what_arg) {}
    explicit invalid_argument(const char* what_arg)
        : detail::exception_bridge<std::invalid_argument>(what_arg) {}
};

/// @brief An index is outside its container.
///
/// Thrown only by `MatrixView::at(row, col, batch)` and `batch_item(i)`. Not retryable. Kept
/// separate from invalid_argument because `std::out_of_range` is what pybind11 maps to Python's
/// `IndexError`.
/// @ingroup errors
class BATCHLAS_API out_of_range : public detail::exception_bridge<std::out_of_range> {
public:
    explicit out_of_range(const std::string& what_arg)
        : detail::exception_bridge<std::out_of_range>(what_arg) {}
    explicit out_of_range(const char* what_arg)
        : detail::exception_bridge<std::out_of_range>(what_arg) {}
};

/// @brief Base of the runtime arm: the call failed for a reason that is not a bad argument.
///
/// Catch a child to say which reason. Derives from `std::runtime_error` (Python `RuntimeError`),
/// and every runtime class stays inside it. Nothing throws a bare `error` today.
/// @ingroup errors
class BATCHLAS_API error : public detail::exception_bridge<std::runtime_error> {
public:
    explicit error(const std::string& what_arg)
        : detail::exception_bridge<std::runtime_error>(what_arg) {}
    explicit error(const char* what_arg)
        : detail::exception_bridge<std::runtime_error>(what_arg) {}
};

/// @brief No route, kernel, backend or vendor entry point in this build on this device serves
/// the request.
///
/// Examples: a complex type on a real-only native path, `Uplo::Upper` where only `Lower` is
/// implemented, a device with no sub-group size 32 under a CTA kernel, a backend that was not
/// compiled in, an order above a kernel's register capacity.
///
/// Not retryable as asked, but a different route, backend, scalar type or shape may succeed:
/// this is the class to catch to fall back to another algorithm.
/// @ingroup errors
class BATCHLAS_API unsupported : public error {
public:
    explicit unsupported(const std::string& what_arg) : error(what_arg) {}
    explicit unsupported(const char* what_arg) : error(what_arg) {}
};

/// @brief The device or its vendor runtime failed.
///
/// A cuBLAS/cuSOLVER/cuSPARSE/rocBLAS/rocSPARSE status code, a CUDA launch failure, a handle that
/// would not initialise, a `sycl::malloc_device`/`malloc_host` that returned null, no device of
/// the requested type, or a host BLAS this machine ships broken.
///
/// Sometimes retryable, and the only class where a retry can be right: a transient launch failure
/// or an allocation lost to another process may clear. A status code that repeats is a real
/// fault; do not loop.
/// @ingroup errors
class BATCHLAS_API device_error : public error {
public:
    explicit device_error(const std::string& what_arg) : error(what_arg) {}
    explicit device_error(const char* what_arg) : error(what_arg) {}
};

/// @brief The scratch the operation was handed is too small, or the allocator ran out of it.
///
/// Every routine's `*_buffer_size()` query is the contract; this is thrown when the buffer
/// actually passed does not honour it.
///
/// Retry smaller: re-query `*_buffer_size()` and pass a buffer that size, or cut the batch,
/// since a batched solve's workspace scales with the batch.
/// @see @ref design_workspace
/// @ingroup errors
class BATCHLAS_API workspace_error : public error {
public:
    explicit workspace_error(const std::string& what_arg) : error(what_arg) {}
    explicit workspace_error(const char* what_arg) : error(what_arg) {}
};

/// @brief An iterative kernel did not converge, or a factorisation broke down on the data.
///
/// An eigen/SVD sweep budget exhausted, a bidiagonal QR that never deflated, an ILU(k) pivot that
/// was zero with no usable shift: the LAPACK `info > 0` case.
///
/// Retryable with different parameters (a looser tolerance, a higher sweep cap, a different
/// algorithm, a rescaled input), not with the same ones.
/// @note The exception is batch-wide: it says some item failed, not which. Pass the optional
///       per-item `info` span (`syev`, `syevx`, `gesvd`, `steqr`, `stedc`) to learn which.
/// @see @ref design_error_model
/// @ingroup errors
class BATCHLAS_API convergence_error : public error {
public:
    explicit convergence_error(const std::string& what_arg) : error(what_arg) {}
    explicit convergence_error(const char* what_arg) : error(what_arg) {}
};

/// @brief BatchLAS is internally inconsistent.
///
/// A route resolver picked a native arm no linked kernel serves, a capability query and the
/// facade that reads it disagree, an internal seam was not injected, a branch documented
/// "unreachable" was reached. Never the caller's fault, never retryable and never fixable from
/// the call site: report it with the message, which names the route and the two things that
/// disagreed.
/// @ingroup errors
class BATCHLAS_API internal_error : public error {
public:
    explicit internal_error(const std::string& what_arg) : error(what_arg) {}
    explicit internal_error(const char* what_arg) : error(what_arg) {}
};

/// @brief A well-formed call arrived in a state, order or thread where it is not valid.
///
/// A Queue used from a thread other than its owner, `attach_to_current_thread()` with a workspace
/// lease outstanding, `configure()` after a Queue already exists, a sizing-mode-only BumpAllocator
/// query asked of a real pool.
///
/// Not retryable as-is: reorder the calls or confine the object to one thread. Distinct from
/// invalid_argument because no argument is wrong.
/// @ingroup errors
class BATCHLAS_API api_misuse : public error {
public:
    explicit api_misuse(const std::string& what_arg) : error(what_arg) {}
    explicit api_misuse(const char* what_arg) : error(what_arg) {}
};

}  // namespace batchlas

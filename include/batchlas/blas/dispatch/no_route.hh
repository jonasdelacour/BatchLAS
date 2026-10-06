#pragma once

/// @file
/// @brief What a call does when nothing can serve it: NoRouteError.
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#the-entry-point-facade

#include <stdexcept>
#include <string>

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

/// @brief Thrown when no route, native or vendor, exists for an (op, backend, scalar) in this build.
///
/// The common cause is a deliberate `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` build
/// meeting an op or shape with no native kernel. `what()` names the op, the
/// scalar type and the caller's detail (usually "built without <library>"), and
/// says which build switch restores the op. The backend is carried but
/// deliberately not printed.
/// @ingroup dispatch
class NoRouteError : public std::runtime_error {
public:
    /// @param op       the op that has no route
    /// @param backend  the device family of the call
    /// @param scalar   the scalar type of the call
    /// @param detail   appended in parentheses; empty for none
    NoRouteError(Op op, Backend backend, ScalarKind scalar, std::string detail)
        : std::runtime_error(build_message(op, backend, scalar, detail)),
          op_(op), backend_(backend), scalar_(scalar) {}

    Op op() const { return op_; }                    ///< the op that has no route
    Backend backend() const { return backend_; }     ///< the device family of the call
    ScalarKind scalar() const { return scalar_; }    ///< the scalar type of the call

private:
    static std::string build_message(Op op, Backend backend, ScalarKind scalar,
                                     const std::string& detail) {
        std::string m = "BatchLAS: no route for ";
        m += op_name(op);
        m += "<";
        m += to_string(scalar);
        m += "> on this backend";
        if (!detail.empty()) {
            m += " (" + detail + ")";
        }
        m += ".\n";
        m += "  This build has no vendor library for that op, and BatchLAS has no\n"
             "  native kernel for it yet. If you configured with\n"
             "  -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF, re-enabling it restores this op;\n"
             "  otherwise the vendor library was not found at configure time.";
        static_cast<void>(backend);
        return m;
    }

    Op op_;
    Backend backend_;
    ScalarKind scalar_;
};

/// @brief The single funnel for "nothing can serve this call": records a coverage miss and throws.
///
/// Raised by the facade's availability gate (vendor_available.hh) and by the
/// `*_or_throw` shims. The miss is recorded unconditionally, unlike the per-call
/// route counters: it is rare by construction and is the row that matters most.
/// @tparam T        scalar type of the call
/// @param op        the op
/// @param backend   the device family
/// @param library   the absent library, e.g. kFactorizationLibrary<B>
/// @throws NoRouteError always
/// @ingroup dispatch
template <typename T>
[[noreturn]] inline void throw_no_vendor_route(Op op, Backend backend,
                                               const char* library) {
    coverage::record_miss(op, scalar_kind_of<T>, backend, library);
    throw NoRouteError(op, backend, scalar_kind_of<T>,
                       std::string("built without ") + library);
}

} // namespace batchlas::dispatch

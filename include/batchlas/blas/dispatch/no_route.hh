#pragma once

// What a call does when nothing can serve it.
//
// Before WP0 S5 this question had no runtime form: an op with no vendor library
// simply failed to LINK, because the public entry point was defined inside the
// vendor TU. Now the entry point always exists, so "there is no implementation
// for this (op, backend, scalar) in this build" has to be a value the program
// can carry and a message a user can act on.
//
// NoRouteError is that message. It names the op, the backend, the scalar type
// and the shape, and it says which build switch would bring an implementation
// back -- because the overwhelmingly common cause is a deliberate
// -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF, and the user needs to know whether they
// hit a gap in BatchLAS's native coverage or simply turned the vendor off.

#include <stdexcept>
#include <string>

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

class NoRouteError : public std::runtime_error {
public:
    NoRouteError(Op op, Backend backend, ScalarKind scalar, std::string detail)
        : std::runtime_error(build_message(op, backend, scalar, detail)),
          op_(op), backend_(backend), scalar_(scalar) {}

    Op op() const { return op_; }
    Backend backend() const { return backend_; }
    ScalarKind scalar() const { return scalar_; }

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

// The single funnel for "nothing can serve this call". Raised by the facade's
// availability gate (dispatch/vendor_available.hh) and by the *_or_throw shims.
template <typename T>
[[noreturn]] inline void throw_no_vendor_route(Op op, Backend backend,
                                               const char* library) {
    // Every no-route path funnels through here, so this is the one place the
    // coverage table has to be told about a gap. Recorded unconditionally --
    // unlike the per-call route counters, a miss is rare by construction and
    // is the row that matters most for the burn-down.
    coverage::record_miss(op, scalar_kind_of<T>, backend, library);
    throw NoRouteError(op, backend, scalar_kind_of<T>,
                       std::string("built without ") + library);
}

} // namespace batchlas::dispatch

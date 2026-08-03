#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <batchlas/backend_config.h>
#include <blas/enums.hh>
#include <util/sycl-device-queue.hh>

namespace batchlas {

// Turn a queue's runtime Backend back into a compile-time one and hand it to
// `f` as an integral_constant, so the body stays a template:
//
//     with_backend(ctx, [&](auto B) { return gemm<B.value, T>(ctx, ...); });
//
// Every entry point is templated on Backend and explicitly instantiated per
// backend, which is the right thing for code generation but forces the choice
// on the caller at compile time. Binding it to the Queue instead means user code
// -- and the convenience layer -- can be written once, while the generated code
// stays exactly as specialised as before: this is a switch over instantiations
// that already exist, not a virtual call or a runtime-parameterised kernel.
//
// Only backends compiled into this build get a case. The rest fall through to
// the throw, which is reachable for Backend::MAGMA and Backend::SYCL -- they are
// declared in the enum but have no implementations behind them.
template <typename F>
inline auto with_backend(Queue& ctx, F&& f) {
    static_assert(BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND ||
                      BATCHLAS_HAS_MKL_BACKEND || BATCHLAS_HAS_HOST_BACKEND,
                  "BatchLAS was built with no backends; nothing can be dispatched.");

    switch (ctx.backend()) {
#if BATCHLAS_HAS_CUDA_BACKEND
        case Backend::CUDA:
            return f(std::integral_constant<Backend, Backend::CUDA>{});
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        case Backend::ROCM:
            return f(std::integral_constant<Backend, Backend::ROCM>{});
#endif
#if BATCHLAS_HAS_MKL_BACKEND
        case Backend::MKL:
            return f(std::integral_constant<Backend, Backend::MKL>{});
#endif
#if BATCHLAS_HAS_HOST_BACKEND
        case Backend::NETLIB:
            return f(std::integral_constant<Backend, Backend::NETLIB>{});
#endif
        default:
            break;
    }
    throw std::runtime_error(
        "BatchLAS: this Queue's backend has no implementation in this build. "
        "Check Queue::backend_available() before pinning a backend.");
}

}  // namespace batchlas

// Define the backend-deducing overload of an entry point already declared as
//
//     template <Backend Back, typename T, ...> R NAME(Queue&, ...);
//
// so that callers can write `NAME(ctx, ...)` and get the queue's backend.
//
// The parameters are forwarded as a pack rather than restated. That is not
// laziness: it means this macro carries no copy of the signature to drift from
// the declaration, and -- because the inner call names the primary -- the
// primary's *default arguments* still apply to arguments the caller omitted.
// Restating the signature here would have required duplicating every default.
//
// Overload resolution stays unambiguous in both directions. Called as
// `NAME(ctx, args...)` the Backend-first overloads cannot deduce Backend and
// drop out, leaving only this one. Called as `NAME<Backend::CUDA>(ctx, args...)`
// this one drops out, because Backend::CUDA is a value and Args are types. The
// inner call always supplies Backend explicitly, so it can never re-enter here.
#define BATCHLAS_DISPATCH_ON_QUEUE(NAME)                                        \
    template <typename... Args>                                                 \
    inline auto NAME(Queue& ctx, Args&&... args) {                              \
        return ::batchlas::with_backend(ctx, [&](auto Back) {                   \
            return NAME<Back.value>(ctx, std::forward<Args>(args)...);          \
        });                                                                     \
    }

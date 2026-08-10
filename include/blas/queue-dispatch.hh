#pragma once

#include <stdexcept>
#include <string>
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
        std::string("BatchLAS: backend ") + std::string(to_string(ctx.backend())) +
        " has no implementation in this build. "
        "Check Queue::backend_available() before pinning a backend.");
}

namespace detail {
// A backend that is definitely compiled in, used only to ask "would the
// positional call be well-formed?". Any compiled backend answers that question
// identically -- the entry points are declared once and instantiated per
// backend, so they all share a signature.
inline constexpr Backend kProbeBackend =
#if BATCHLAS_HAS_CUDA_BACKEND
    Backend::CUDA;
#elif BATCHLAS_HAS_ROCM_BACKEND
    Backend::ROCM;
#elif BATCHLAS_HAS_MKL_BACKEND
    Backend::MKL;
#else
    Backend::NETLIB;
#endif
}  // namespace detail

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
//
// The requires-clause is what keeps it honest. Without it this overload accepts
// *any* argument list, so it beats a more specific overload -- an option-struct
// spelling, say, or one relying on a default argument -- and only then fails,
// deep inside its own body, on a call it should never have claimed. Constraining
// it to argument lists the positional entry point would actually accept makes it
// drop out of resolution instead, which is the whole difference between "this
// overload does not apply" and "this overload applies and is broken".
#define BATCHLAS_DISPATCH_ON_QUEUE(NAME)                                        \
    template <typename... Args>                                                 \
        requires requires(Queue& probe_ctx, Args&&... probe_args) {             \
            NAME<::batchlas::detail::kProbeBackend>(probe_ctx,                  \
                                                    std::forward<Args>(probe_args)...); \
        }                                                                       \
    inline auto NAME(Queue& ctx, Args&&... args) {                              \
        return ::batchlas::with_backend(ctx, [&](auto Back) {                   \
            return NAME<Back.value>(ctx, std::forward<Args>(args)...);          \
        });                                                                     \
    }

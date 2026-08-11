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

// ---- the USM contract, enforced -------------------------------------------
//
// A MatrixView/Span takes a bare pointer and cannot check where the memory came
// from. Handing ordinary host memory (std::vector, new, malloc) to a GPU queue
// used to reach the device as a wild address: CUDA_ERROR_ILLEGAL_ADDRESS, and
// then SIGABRT from inside the runtime during teardown, which no catch block can
// stop -- while the identical code was correct on the host backend, so a CPU
// prototype passed and the GPU run died. These helpers turn that into a thrown
// std::invalid_argument that names the offending argument.
//
// One USM query per pointer argument (~70ns measured), which is noise against a
// kernel launch. BATCHLAS_SKIP_POINTER_CHECKS=1 bypasses it.

inline bool pointer_checks_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("BATCHLAS_SKIP_POINTER_CHECKS");
        return !(v && *v && *v != '0');
    }();
    return enabled;
}

// Matrix/MatrixView spell it data_ptr(); Span/UnifiedVector spell it data().
template <typename A>
concept HasDataPtr = requires(const A& a) { { a.data_ptr() } -> std::convertible_to<const void*>; };
template <typename A>
concept HasData = requires(const A& a) {
    { a.data() } -> std::convertible_to<const void*>;
    { a.size() } -> std::convertible_to<size_t>;
};

template <typename A>
inline void require_arg_accessible(const Queue& ctx, const A& arg, const std::string& what) {
    if constexpr (HasDataPtr<std::remove_cvref_t<A>>) {
        ctx.require_device_accessible(static_cast<const void*>(arg.data_ptr()), what.c_str());
    } else if constexpr (HasData<std::remove_cvref_t<A>>) {
        // An empty span is legitimate: BumpAllocator sizing passes hand out empty
        // spans by design, so only a non-empty one carries a pointer to check.
        if (arg.size() != 0) {
            ctx.require_device_accessible(static_cast<const void*>(arg.data()), what.c_str());
        }
    }
    // Anything else (option structs, scalars, enums) carries no pointer.
}

// Positional labelling, for the dispatch macro: it forwards an unnamed pack, so
// the best it can say is which argument position was wrong.
template <typename... Args>
inline void require_pack_accessible(const Queue& ctx, const char* fn, const Args&... args) {
    if (!pointer_checks_enabled()) return;
    int pos = 0;
    (void)std::initializer_list<int>{
        (++pos,
         require_arg_accessible(ctx, args,
                                std::string(fn) + ": argument " + std::to_string(pos)),
         0)...};
}

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
        ::batchlas::detail::require_pack_accessible(ctx, #NAME, args...);       \
        return ::batchlas::with_backend(ctx, [&](auto Back) {                   \
            return NAME<Back.value>(ctx, std::forward<Args>(args)...);          \
        });                                                                     \
    }

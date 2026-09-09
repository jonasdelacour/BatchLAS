#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <batchlas/backend_config.h>
#include <batchlas/blas/enums.hh>
#include <batchlas/settings.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

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
// kernel launch. BATCHLAS_SKIP_POINTER_CHECKS=1 bypasses it, and it is one of
// the knobs the BATCHLAS_ALLOW_UNSAFE_ENV build option gates: with that option
// OFF the field below is false whatever the environment says, so an embedding
// application cannot have its argument validation switched off by ambient
// process state.
//
// The odd acceptance set is preserved verbatim in settings.cc -- ANY non-empty
// value whose first character is not '0' skips the checks, so "=false", "=off"
// and "=no" all DISABLE them. That is not env_truthy and must not become it:
// tightening it here would silently re-enable checking for anyone who wrote one
// of those spellings, which is a behaviour change at the one site where
// behaviour changes are most expensive.
//
// Still latched in a function-local static, exactly as before: this is on the
// argument-checking path of every dispatched call, and the latch is what keeps
// it off the per-argument cost. A reload therefore does not reach it, which is
// also today's behaviour.
inline bool pointer_checks_enabled() {
    static const bool enabled = !batchlas::settings().unsafe.skip_pointer_checks;
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

// Extents, so an argument that addresses no elements can be recognised.
// MatrixView spells its extent rows()/cols()/batch_size(); VectorView spells it
// size()/batch_size().
template <typename A>
concept HasMatrixExtent = requires(const A& a) {
    { a.rows() } -> std::convertible_to<long long>;
    { a.cols() } -> std::convertible_to<long long>;
    { a.batch_size() } -> std::convertible_to<long long>;
};
template <typename A>
concept HasVectorExtent = requires(const A& a) {
    { a.size() } -> std::convertible_to<long long>;
    { a.batch_size() } -> std::convertible_to<long long>;
};

// True when the argument addresses zero elements, so no kernel can dereference
// it whatever its pointer happens to be.
template <typename A>
inline bool addresses_no_elements(const A& arg) {
    using U = std::remove_cvref_t<A>;
    if constexpr (HasMatrixExtent<U>) {
        return arg.rows() == 0 || arg.cols() == 0 || arg.batch_size() == 0;
    } else if constexpr (HasVectorExtent<U>) {
        return arg.size() == 0 || arg.batch_size() == 0;
    } else {
        return false;
    }
}

template <typename A>
inline void require_arg_accessible(const Queue& ctx, const A& arg, const std::string& what) {
    if constexpr (HasDataPtr<std::remove_cvref_t<A>>) {
        // A default-constructed view is the API's spelling for an optional
        // matrix that is not in use -- `syevx(..., JobType::NoEigenVectors,
        // MatrixView<T, MatrixFormat::Dense>(), params)` is the documented call,
        // and ~50 call sites in this repo write it. It owns no memory, so there
        // is nothing to reach and nothing to reject; checking it turned every
        // such call into a throw ("the pointer is null"), which iluk_tests
        // caught. This is the same rule the empty-span branch below already
        // applies, for the same reason: zero elements, nothing dereferenced.
        if (addresses_no_elements(arg)) return;
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

// ---- owning arguments, accepted once ---------------------------------------
//
// `Matrix` converts implicitly to `MatrixView` and `Vector` to `VectorView`
// (blas/matrix.hh), which is enough wherever the parameter type is already
// concrete. It is NOT enough when the scalar has to be deduced from it:
// template argument deduction does not consider user-defined conversions, so
// `gemm<Backend::CUDA>(ctx, A, B, C, ...)` with owning matrices deduces nothing
// from A, the primary drops out, and the caller gets "no matching function".
//
// Every entry point used to carry a hand-written twin whose entire body was one
// cast per matrix argument -- one per *overload* rather than one per name, so a
// name with four positional spellings paid for four of them, and each one
// restated every defaulted argument the primary already had.
//
// `view_of` names the view a parameter should become. Anything that is not an
// owning container passes through as itself, which is what lets the forwarder
// below be variadic and still touch only the arguments that need converting --
// and is also what makes a mixed call like
// `stein(ctx, Vector d, VectorView e, ...)` work, which no hand-written twin
// covered.
template <class A>
struct view_of {
    using type = const A&;
};
template <class T, MatrixFormat F>
struct view_of<Matrix<T, F>> {
    using type = MatrixView<T, F>;
};
template <class T>
struct view_of<Vector<T>> {
    using type = VectorView<T>;
};
template <class A>
using view_t = typename view_of<std::remove_cvref_t<A>>::type;

// True for exactly the argument types view_of rewrites.
template <class A>
inline constexpr bool is_owning_arg_v =
    !std::is_same_v<view_t<A>, const std::remove_cvref_t<A>&>;

// The gate on the forwarder. Without it the forwarder would be an unconstrained
// variadic that claims every call and only then fails inside its own body --
// the mistake BATCHLAS_DISPATCH_ON_QUEUE's requires-clause already exists to
// avoid. With it, a call whose arguments are all views never considers the
// forwarder at all and binds to the primary exactly as before.
template <class... A>
concept AnyOwning = (is_owning_arg_v<A> || ...);

template <class A>
inline view_t<A> as_view(const A& a) {
    return static_cast<view_t<A>>(a);
}

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

// Accept owning `Matrix` / `Vector` arguments on an entry point whose primary
// takes `MatrixView` / `VectorView`, for every overload of that name at once.
//
// One line beside BATCHLAS_DISPATCH_ON_QUEUE replaces the family of
// hand-written twins described at detail::view_of above. The pack is converted
// argument by argument and the INNER call does the deducing, so this forwarder
// carries no copy of any signature: a new overload, a new defaulted argument or
// a reordered parameter needs no change here, which is the whole point.
//
// It is a last resort in overload resolution, by construction. detail::AnyOwning
// keeps it out of every all-view call, and the requires-clause keeps it out of
// argument lists the view spelling would not accept -- so it applies only where
// the alternative today is a compile error. Where a more specific overload is
// also viable (the option-struct spellings in blas/options.hh, or an
// arity-changing forwarder such as getrf's four-argument form), partial ordering
// prefers that one, because a fixed parameter list is more specialised than a
// trailing pack.
//
// The pack is `const Args&...`, NEVER `Args&&...`. A forwarding-reference pack
// binds a prvalue BETTER than `const MatrixView&` does, so it would beat the
// checked convenience overloads in blas/options.hh for a call like
// `getrf(ctx, A.view(), pivots)` -- which is exactly how an 8x4 matrix once
// sailed through a squareness check (see the note above detail::require_square
// in blas/options.hh). With `const Args&...` the two rank equally on conversion
// and partial ordering picks the more specialised overload, the one carrying the
// checks. Nothing is moved through here, so nothing is lost by not forwarding:
// every parameter downstream is a view, a span, an enum or a small option
// struct.
//
// Two argument lists it deliberately does NOT reach, both because a pack cannot
// deduce them, and in both cases the alternative is a diagnostic rather than a
// wrong answer:
//
//   - a bare `{}` or other braced-init-list in any position. A parameter pack
//     deduces nothing from one, so the forwarder drops out and the call has to
//     name the type it means (`stein_all_counts`, `OrthoOptions{}`). This is the
//     same property that keeps BATCHLAS_DISPATCH_ON_QUEUE out of potrf's
//     option-struct calls, and it is why the deleted bare-`{}` guards in
//     blas/options.hh and blas/extensions.hh still fire.
//   - a call that supplies SOME template arguments explicitly and leaves the
//     rest to deduction, e.g. `spmm<Back, T>(ctx, Matrix, ...)` where the
//     MatrixFormat is still deduced: `T` lands in the pack and has to match the
//     first argument. Write `spmm<Back>(ctx, ...)` and let both deduce. The one
//     in-tree call that needed the old spelling keeps a hand-written twin; see
//     ritz_values in blas/extensions.hh.
#define BATCHLAS_ACCEPT_OWNING(NAME)                                            \
    template <Backend Back, typename... Args>                                   \
        requires ::batchlas::detail::AnyOwning<Args...> &&                      \
                 requires(Queue& probe_ctx, const Args&... probe_args) {        \
                     NAME<Back>(probe_ctx,                                      \
                                ::batchlas::detail::as_view(probe_args)...);    \
                 }                                                              \
    inline auto NAME(Queue& ctx, const Args&... args) {                         \
        return NAME<Back>(ctx, ::batchlas::detail::as_view(args)...);           \
    }

// The same thing for an entry point that is NOT templated on Backend -- `norm`
// and `transpose` in blas/extra.hh, `francis_sweep` and the `steqr_*_buffer_size`
// pair in blas/extensions.hh. Those have no BATCHLAS_DISPATCH_ON_QUEUE overload
// either, for the same reason: there is no backend to deduce.
#define BATCHLAS_ACCEPT_OWNING_NB(NAME)                                         \
    template <typename... Args>                                                 \
        requires ::batchlas::detail::AnyOwning<Args...> &&                      \
                 requires(Queue& probe_ctx, const Args&... probe_args) {        \
                     NAME(probe_ctx,                                            \
                          ::batchlas::detail::as_view(probe_args)...);          \
                 }                                                              \
    inline auto NAME(Queue& ctx, const Args&... args) {                         \
        return NAME(ctx, ::batchlas::detail::as_view(args)...);                 \
    }

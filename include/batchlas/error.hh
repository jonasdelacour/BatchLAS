#pragma once

// The BatchLAS exception hierarchy.
//
// WHY IT EXISTS. Before this header the library threw 572 raw std:: exceptions
// and nothing else. A consumer could not write `catch (const batchlas::error&)`,
// and -- more importantly -- could not tell a shape mismatch from an unsupported
// route from a workspace shortfall from a device failure, so it could not
// recover from one class while letting another propagate. In a batched solver
// that is the difference between "retry this batch smaller" and "abort the run".
//
// WHAT IT DOES NOT COVER. Two failure classes deliberately stay outside:
//
//   * std::bad_alloc, thrown from three sites where a sycl::malloc_* returned
//     null (src/util/sycl-util-impl.cc, src/queue.hh, src/extensions/sytrd_sy2sb.cc).
//     std::bad_alloc is the standard type for allocation failure, it carries no
//     message worth preserving, and it is what pybind11 maps to MemoryError.
//     Wrapping it would lose MemoryError and gain nothing.
//   * sycl::exception, raised by the SYCL runtime itself. It is not ours to
//     reclassify.
//
// So `catch (const batchlas::exception&)` catches every failure BatchLAS
// diagnoses, not every failure a BatchLAS call can produce. A consumer that
// must not let anything escape still needs a `catch (const std::exception&)`
// behind it.

#include <stdexcept>
#include <string>

#include <batchlas/export.hh>

namespace batchlas {

// ---------------------------------------------------------------------------
// The tag base
// ---------------------------------------------------------------------------
//
// `exception` is an EMPTY tag whose only job is to make
//
//     catch (const batchlas::exception& e) { ... }
//
// match everything below and nothing else. Three rules govern it, and each one
// of them fails SILENTLY -- with no compiler diagnostic at any point -- if it is
// broken. All three were verified by compiling a model of this hierarchy.
//
//  1. IT MUST NOT DERIVE FROM std::exception. Every leaf already carries one
//     std::exception subobject through its std:: base. A second one, brought in
//     here, makes `catch (const std::exception&)` face two ambiguous base
//     subobjects: the handler stops matching and the exception runs to
//     std::terminate. No diagnostic is issued -- the ambiguity is only ever
//     diagnosed at a cast, and a catch clause is not one.
//
//  2. EVERY LEAF MUST INHERIT IT VIRTUALLY. Under non-virtual inheritance a
//     future class deriving from two arms (say
//     `bad_shape : public invalid_argument, public unsupported`) would get two
//     tag subobjects, and `catch (const batchlas::exception&)` would stop
//     matching that class -- again silently, again terminating.
//     detail::exception_bridge is the ONLY place the inheritance is spelled, so
//     no future class can get it wrong. The cost is one vptr, paid on the error
//     path only.
//
//  3. IT MUST NOT DECLARE what(). std::exception::what() lives in a different
//     base subobject and does not override a pure virtual declared here, so
//     adding one makes every leaf abstract ("cannot declare variable to be of
//     abstract type"). This is the one rule that fails loudly. message() below
//     is the differently-named accessor that reaches the text through a tag
//     handler; `catch (const std::exception&)` is the other way to get it.
//
//  4. EVERY CLASS IN THIS HEADER MUST CARRY BATCHLAS_API. This is rule 1's
//     failure mode again, reached by a different road, and it is the reason the
//     macro appears on ten classes that have no out-of-line member between
//     them. NOTHING here has a key function -- `~exception() = default` is
//     inline and every leaf's members are inline -- so the vtable and the
//     typeinfo are emitted with VAGUE LINKAGE in every translation unit that
//     names the type. Measured on the current build: `typeinfo for
//     batchlas::error` is a weak object (nm class V) in ALL FOURTEEN component
//     .so AND in the test executables. Catch-by-type works today only because
//     every one of those copies has default visibility, so the dynamic linker
//     collapses them to a single address and the unwinder's pointer comparison
//     succeeds. Under -fvisibility=hidden (monolithic mode) an unannotated copy
//     stops being collapsible: the library's typeinfo and the consumer's
//     typeinfo are different objects, `catch (const batchlas::error&)` in the
//     consumer stops matching a throw from inside the library, and the
//     exception runs to std::terminate. There is no diagnostic at any point.
//     BATCHLAS_API forces vtable and typeinfo back to default visibility, which
//     restores the single address the unwinder needs.
//
//     ON THIS x86-64 BUILD the failure is currently masked, which is exactly
//     why it must not be tested for: the typeinfo NAME strings this DPC++
//     emits carry no leading '*' (the bytes at `typeinfo name for
//     batchlas::error` are `N8batchlas5errorE\0`), so libstdc++'s
//     `operator==` -- `__name == __arg.__name || (__name[0] != '*' &&
//     strcmp(...) == 0)` -- takes its strcmp fallback and duplicated typeinfos
//     still compare equal. The '*' prefix is decided by the compiler's RTTI
//     uniqueness classification; the ARM64 Itanium variant and libc++'s
//     non-unique-RTTI path compare by POINTER ONLY. A throw/catch test on this
//     machine therefore passes whether or not the annotation is present, and
//     reading the annotations is the only check that discriminates.
//
//     detail::exception_bridge is annotated for the same reason and is the
//     easiest one to forget: it is a base subobject on the catch-time upcast
//     path (rule 2's virtual inheritance means that walk goes through
//     __vmi_class_type_info and reads the vtable for the virtual-base offset),
//     so its typeinfo has to unify too.
class BATCHLAS_API exception {
public:
    virtual ~exception() = default;

    // The same string what() returns. See rule 3 for why it cannot be called
    // what(): this is what a `catch (const batchlas::exception&)` handler reads.
    virtual const char* message() const noexcept = 0;

    // Public, not protected, purely so no access check can ever stand between a
    // leaf's implicitly-defined copy constructor and this virtual base. The
    // pure virtual message() already makes the tag abstract, so nothing can
    // construct a bare `batchlas::exception` regardless.
    exception() = default;
    exception(const exception&) = default;
    exception& operator=(const exception&) = default;
};

namespace detail {

// The single spelling of "derive from a std:: exception type AND, virtually,
// from the tag". Every class below goes through it, which is what makes rule 2
// above unbreakable per class rather than a convention each new class must
// remember.
template <typename StdBase>
class BATCHLAS_API exception_bridge : public StdBase, public virtual exception {
public:
    using StdBase::StdBase;
    const char* message() const noexcept override { return StdBase::what(); }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// The caller-fault arm: std::logic_error family
// ---------------------------------------------------------------------------

// The caller passed something the API contract forbids: a non-square view where
// a square one is required, mismatched batch sizes, a span too short for the
// batch, a negative dimension, a null pointer, an out-of-order Queue where an
// in-order one is required, an enum value with no meaning here.
//
// FOR A CALLER: not retryable. The arguments are wrong; nothing about the
// machine, the workspace, or the data will change that. Fix the call.
//
// Derives from std::invalid_argument, so existing `catch (const
// std::invalid_argument&)` handlers keep working and pybind11's default
// translator keeps mapping it to Python's ValueError.
class BATCHLAS_API invalid_argument : public detail::exception_bridge<std::invalid_argument> {
public:
    explicit invalid_argument(const std::string& what_arg)
        : detail::exception_bridge<std::invalid_argument>(what_arg) {}
    explicit invalid_argument(const char* what_arg)
        : detail::exception_bridge<std::invalid_argument>(what_arg) {}
};

// An index is outside its container: MatrixView::at(row, col, batch) and
// batch_item(i) are the only throwers.
//
// FOR A CALLER: not retryable, same as invalid_argument -- it is kept separate
// only because indexing protocols expect it. std::out_of_range is what
// pybind11 maps to Python's IndexError, and element access that raised
// ValueError instead would be a Python-visible regression.
class BATCHLAS_API out_of_range : public detail::exception_bridge<std::out_of_range> {
public:
    explicit out_of_range(const std::string& what_arg)
        : detail::exception_bridge<std::out_of_range>(what_arg) {}
    explicit out_of_range(const char* what_arg)
        : detail::exception_bridge<std::out_of_range>(what_arg) {}
};

// ---------------------------------------------------------------------------
// The runtime arm: std::runtime_error family
// ---------------------------------------------------------------------------

// Base of everything below. Catch it to mean "the call failed for a reason that
// is not a bad argument", and catch one of its children to say which reason.
//
// Deriving from std::runtime_error is load-bearing, not incidental: several
// in-tree call sites and tests catch std::runtime_error around calls whose
// throws land in this arm, and pybind11 maps std::runtime_error to Python's
// RuntimeError. Every class below therefore stays inside it.
//
// Nothing throws a bare `error` today; every site adjudicated during the
// migration fitted one of the five children. It is kept as a catchable base and
// as the honest home for a future failure that fits none of them.
class BATCHLAS_API error : public detail::exception_bridge<std::runtime_error> {
public:
    explicit error(const std::string& what_arg)
        : detail::exception_bridge<std::runtime_error>(what_arg) {}
    explicit error(const char* what_arg)
        : detail::exception_bridge<std::runtime_error>(what_arg) {}
};

// No route, kernel, backend or vendor entry point in THIS BUILD on THIS DEVICE
// serves the requested combination of shape, scalar type, layout and options:
// a complex type on a real-only native path, Uplo::Upper where only Lower is
// implemented, a device with no sub-group size 32 under a CTA kernel, a backend
// that was not compiled in, an order above a kernel's register capacity.
//
// FOR A CALLER: not retryable AS ASKED. Retrying the same call will fail
// identically -- but a different route, backend, scalar type or shape may
// succeed, which is what separates this from invalid_argument. This is the
// class to catch when you want to fall back to another algorithm.
class BATCHLAS_API unsupported : public error {
public:
    explicit unsupported(const std::string& what_arg) : error(what_arg) {}
    explicit unsupported(const char* what_arg) : error(what_arg) {}
};

// The device or its vendor runtime failed: a cuBLAS/cuSOLVER/cuSPARSE/rocBLAS/
// rocSPARSE status code, a CUDA launch failure, a handle that would not
// initialise, a sycl::malloc_device/malloc_host that returned null, no device of
// the requested type present, or a host BLAS this machine ships broken.
//
// FOR A CALLER: sometimes retryable, and it is the one class where retrying can
// be right -- a transient launch failure or an allocation lost to another
// process may clear. A status code that repeats is a real fault; do not loop.
class BATCHLAS_API device_error : public error {
public:
    explicit device_error(const std::string& what_arg) : error(what_arg) {}
    explicit device_error(const char* what_arg) : error(what_arg) {}
};

// The scratch the operation was handed is too small, or the allocator ran out
// of it. Every routine's *_buffer_size() query is the contract; this is what
// gets thrown when the buffer actually passed does not honour it.
//
// FOR A CALLER: RETRY SMALLER. This is the recoverable class. Re-query
// *_buffer_size() and pass a buffer that size, or cut the batch and call again
// -- the workspace a batched solve needs scales with the batch, so halving the
// batch is the usual fix when the arena itself is the limit.
class BATCHLAS_API workspace_error : public error {
public:
    explicit workspace_error(const std::string& what_arg) : error(what_arg) {}
    explicit workspace_error(const char* what_arg) : error(what_arg) {}
};

// An iterative kernel did not converge, or a factorisation broke down on the
// data: an eigen/SVD sweep budget exhausted, a bidiagonal QR that never
// deflated, an ILU(k) pivot that was zero with no usable shift. The LAPACK
// info > 0 case.
//
// FOR A CALLER: retryable with different parameters, not with the same ones.
// A looser tolerance, a higher sweep cap, a different algorithm or a rescaled
// input may converge; repeating the identical call will not.
//
// NOTE the shape of this diagnostic today: it is BATCH-WIDE. A throw says some
// item in the batch failed, not which. Per-item status spans are work package
// A-1; until they land, an exception is the only signal, and several tiers
// (every stedc merge arm, steqr_wg, syev_jacobi_cta, gesvdj_cta) do not even
// raise that -- they return a wrong answer silently.
class BATCHLAS_API convergence_error : public error {
public:
    explicit convergence_error(const std::string& what_arg) : error(what_arg) {}
    explicit convergence_error(const char* what_arg) : error(what_arg) {}
};

// BatchLAS is internally inconsistent: a route resolver picked a native arm no
// linked kernel serves, a capability query and the facade that reads it
// disagree, an internal seam was not injected, a branch documented "Unreachable"
// was reached. Never the caller's fault -- these are the sites that used to
// throw std::logic_error, and every one of them is an invariant of this
// library, not a statement about the arguments.
//
// FOR A CALLER: never retryable, and never fixable from the call site. It is a
// bug in BatchLAS. Report it with the message, which names the route and the
// two things that disagreed.
class BATCHLAS_API internal_error : public error {
public:
    explicit internal_error(const std::string& what_arg) : error(what_arg) {}
    explicit internal_error(const char* what_arg) : error(what_arg) {}
};

// The call itself is well-formed but arrives in a state or sequence where it is
// not valid: a Queue used from a thread other than the one that owns it,
// attach_to_current_thread() with a workspace lease still outstanding,
// configure() after a Queue already exists, a BumpAllocator query that only a
// sizing-mode allocator answers asked of a real pool.
//
// FOR A CALLER: not retryable as-is; the fix is to reorder the calls or to
// confine the object to one thread. Distinct from invalid_argument because no
// argument is wrong -- only when and from where the call was made.
class BATCHLAS_API api_misuse : public error {
public:
    explicit api_misuse(const std::string& what_arg) : error(what_arg) {}
    explicit api_misuse(const char* what_arg) : error(what_arg) {}
};

}  // namespace batchlas

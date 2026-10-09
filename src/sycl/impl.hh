#pragma once
// The only header above kernel level that names a SYCL implementation (DPC++ or AdaptiveCpp).
// Every helper expands on DPC++ to the exact calls it replaced.
// evidence: docs/design/sycl-implementations.md#sycl-impl-runtime-interop-seam
#include <batchlas/backend_config.h>
#include <sycl/sycl.hpp>

#include "../util/internal-api.hh"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if BATCHLAS_SYCL_IMPL_DPCPP == BATCHLAS_SYCL_IMPL_ACPP
#error "backend_config.h must select exactly one of BATCHLAS_SYCL_IMPL_DPCPP / _ACPP"
#endif

namespace batchlas::impl {

enum class Native { Cuda, Hip };

#if BATCHLAS_SYCL_IMPL_ACPP
inline constexpr sycl::backend kCudaBackend = sycl::backend::cuda;
inline constexpr sycl::backend kHipBackend = sycl::backend::hip;
#else
inline constexpr sycl::backend kCudaBackend = sycl::backend::ext_oneapi_cuda;
inline constexpr sycl::backend kHipBackend = sycl::backend::ext_oneapi_hip;
#endif

template <Native N>
inline constexpr sycl::backend backend_v = N == Native::Cuda ? kCudaBackend : kHipBackend;

inline sycl::backend backend_of(const sycl::device& d) { return d.get_backend(); }

inline sycl::backend backend_of(const sycl::queue& q) {
#if BATCHLAS_SYCL_IMPL_ACPP
    return q.get_device().get_backend();  // acpp's queue has no get_backend()
#else
    return q.get_backend();
#endif
}

inline bool is_cuda(sycl::backend b) { return b == kCudaBackend; }
inline bool is_hip(sycl::backend b) { return b == kHipBackend; }

// Compute capability as major*10+minor; 0 when the version string has neither form.
// DPC++ reports "8.9", acpp "sm_89" / "sm_120".
inline int cuda_cc(const sycl::device& d) {
    const std::string v = d.get_info<sycl::info::device::version>();
    int major = 0, minor = 0, sm = 0;
    if (std::sscanf(v.c_str(), "sm_%d", &sm) == 1) return sm;
    if (std::sscanf(v.c_str(), "%d.%d", &major, &minor) == 2) return major * 10 + minor;
    return 0;
}

// The queue's native stream as void*, or nullptr when none exists outside a command group.
// acpp has one only for an in-order queue (its in-order executor); see run_native for the rest.
inline void* native_stream(const sycl::queue& q) {
#if BATCHLAS_SYCL_IMPL_ACPP
    const sycl::backend b = backend_of(q);
    if (!is_cuda(b) && !is_hip(b)) return nullptr;
    auto* exec = q.AdaptiveCpp_inorder_executor();
    if (exec == nullptr || exec->get_queue() == nullptr) return nullptr;
    return exec->get_queue()->get_native_type();
#else
    switch (q.get_backend()) {
#if SYCL_EXT_ONEAPI_BACKEND_CUDA
        case sycl::backend::ext_oneapi_cuda:
            return static_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_cuda>(q));
#endif
#if SYCL_EXT_ONEAPI_BACKEND_HIP
        case sycl::backend::ext_oneapi_hip:
            return static_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_hip>(q));
#endif
        default:
            return nullptr;
    }
#endif
}

// Stream for host-only vendor calls (workspace queries) that must not be enqueued.
// DPC++: get_native (throws on a backend mismatch). acpp: native_stream(), possibly nullptr.
template <Native N, class Q>
auto query_stream(Q& q) {
#if BATCHLAS_SYCL_IMPL_ACPP
    return native_stream(q);
#else
    return sycl::get_native<backend_v<N>>(q);
#endif
}

// Runs f(native stream) ordered in q. DPC++: inline on get_native's stream, exactly as the call
// sites did. acpp: inside a custom operation, valid for in-order and out-of-order queues, and
// finished (host side) when this returns. f is copied, so it captures by value; it must not
// submit to q. Q is QueueImpl: submit_untraced keeps vendor work out of the kernel trace, as on
// DPC++, while still recording the queue's last event.
template <Native N, class Q, class F>
void run_native(Q& q, F f) {
#if BATCHLAS_SYCL_IMPL_ACPP
    struct State {
        std::atomic<bool> done{false};
        std::exception_ptr error;
    };
    auto state = std::make_shared<State>();
    // Out-of-order: after all prior work, as a host call between submissions would be.
    std::vector<sycl::event> deps;
    if (!q.is_in_order()) deps = q.get_wait_list();
    sycl::event e = q.submit_untraced([&](sycl::handler& h) {
        h.depends_on(deps);
        h.AdaptiveCpp_enqueue_custom_operation([f, state](sycl::interop_handle& ih) {
            try {
                F call = f;  // f may be a mutable lambda; the operation itself is invoked const
                call(ih.get_native_queue<backend_v<N>>());
            } catch (...) {
                state->error = std::current_exception();
            }
            state->done.store(true, std::memory_order_release);
        });
    });
    // acpp 25.10 runs the operation inside submit (in-order and out-of-order alike). Should a
    // scheduler defer it, wait: the error must reach the caller, and the caller's static vendor
    // handle must never be used by two threads at once.
    if (!state->done.load(std::memory_order_acquire)) e.wait();
    if (state->done.load(std::memory_order_acquire) && state->error) std::rethrow_exception(state->error);
#else
    f(sycl::get_native<backend_v<N>>(q));
#endif
}

// Barrier after all prior work of q, or after `wait_list` only (out-of-order queue).
inline sycl::event submit_barrier(sycl::queue& q) {
#if BATCHLAS_SYCL_IMPL_ACPP
    std::vector<sycl::event> deps;
    if (!q.is_in_order()) deps = q.get_wait_list();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.AdaptiveCpp_enqueue_custom_operation([](sycl::interop_handle&) {});
    });
#else
    return q.ext_oneapi_submit_barrier();
#endif
}

inline sycl::event submit_barrier(sycl::queue& q, const std::vector<sycl::event>& wait_list) {
#if BATCHLAS_SYCL_IMPL_ACPP
    return q.submit([&](sycl::handler& h) {
        h.depends_on(wait_list);
        h.AdaptiveCpp_enqueue_custom_operation([](sycl::interop_handle&) {});
    });
#else
    return q.ext_oneapi_submit_barrier(wait_list);
#endif
}

// Largest work-group kernel K can launch on dev; throws sycl::exception when the runtime cannot
// answer. acpp has no kernel_id query, so it reports the device limit.
template <class K>
std::size_t kernel_max_wg_size(const sycl::context& ctx, const sycl::device& dev) {
#if BATCHLAS_SYCL_IMPL_ACPP
    (void)ctx;
    return dev.get_info<sycl::info::device::max_work_group_size>();
#else
    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {dev}, {sycl::get_kernel_id<K>()});
    const auto kern = bundle.template get_kernel<K>();
    return kern.template get_info<sycl::info::kernel_device_specific::work_group_size>(dev);
#endif
}

// The same, with the DPC++ bundle built for every device of ctx (get_kernel_max_wg_size's form).
template <class K>
std::size_t kernel_max_wg_size_all_devices(const sycl::context& ctx, const sycl::device& dev) {
#if BATCHLAS_SYCL_IMPL_ACPP
    return kernel_max_wg_size<K>(ctx, dev);
#else
    auto kernel_id = sycl::get_kernel_id<K>();
    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {kernel_id});
    auto kernel = kernel_bundle.get_kernel(kernel_id);
    return kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(dev);
#endif
}

#if BATCHLAS_SYCL_IMPL_ACPP
// acpp keeps asynchronous errors in ONE process-wide list (launch failures included), and every
// ~queue reports it through that queue's handler: a throwing handler is std::terminate there.
// So the handler parks them here and throw_async_errors() throws the first at a call site that
// may throw. Leaked: a ~queue may run during static destruction.
// evidence: docs/design/sycl-implementations.md#sycl-impl-asynchronous-errors
struct AsyncErrorSink {
    std::mutex m;
    std::vector<std::exception_ptr> errors;
};
inline BATCHLAS_INTERNAL_API AsyncErrorSink& async_error_sink() {
    static auto* sink = new AsyncErrorSink;
    return *sink;
}

inline void on_async_errors(sycl::exception_list errors) {
    AsyncErrorSink& s = async_error_sink();
    std::lock_guard<std::mutex> lock(s.m);
    for (const std::exception_ptr& e : errors) s.errors.push_back(e);
}

// Collects the pending errors and rethrows the first; the rest are dropped, as DPC++'s
// rethrowing handler drops them.
inline void throw_async_errors(sycl::queue& q) {
    q.throw_asynchronous();
    std::exception_ptr first;
    {
        AsyncErrorSink& s = async_error_sink();
        std::lock_guard<std::mutex> lock(s.m);
        if (s.errors.empty()) return;
        first = s.errors.front();
        s.errors.clear();
    }
    std::rethrow_exception(first);
}
#else
// Rethrows the first asynchronous error at queue::wait_and_throw / throw_asynchronous; the
// default calls std::terminate.
inline void on_async_errors(sycl::exception_list errors) {
    for (const std::exception_ptr& e : errors) std::rethrow_exception(e);
}
#endif

// After every submission: acpp registers a failed launch (CU:1 for an over-limit grid or an
// unservable SLM request) asynchronously; DPC++ throws it from the submit itself.
inline void throw_launch_errors([[maybe_unused]] sycl::queue& q) {
#if BATCHLAS_SYCL_IMPL_ACPP
    throw_async_errors(q);
#endif
}

// Before an nd_range launch: the CUDA grid limit (x 2^31-1, y and z 65535; SYCL dimension d is
// CUDA axis D-1-d). DPC++ checks it itself and throws errc::nd_range with this message; acpp
// launches and gets CU:1.
template <int D>
inline void check_group_count([[maybe_unused]] const sycl::queue& q, [[maybe_unused]] const sycl::nd_range<D>& r) {
#if BATCHLAS_SYCL_IMPL_ACPP
    if (!is_cuda(backend_of(q))) return;
    for (int d = 0; d < D; ++d) {
        const std::size_t local = r.get_local_range()[d];
        const std::size_t groups = local == 0 ? 0 : r.get_global_range()[d] / local;
        const std::size_t limit = d == D - 1 ? std::size_t{2147483647} : std::size_t{65535};
        if (groups > limit) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                                  "Number of work-groups exceed limit for dimension " + std::to_string(d) + " : " +
                                      std::to_string(groups) + " > " + std::to_string(limit));
        }
    }
#endif
}

// event::wait_and_throw. acpp's event drops its handler and calls an empty std::function.
inline void wait_and_throw(sycl::event& e, sycl::queue& q) {
#if BATCHLAS_SYCL_IMPL_ACPP
    e.wait();
    throw_async_errors(q);
#else
    (void)q;
    e.wait_and_throw();
#endif
}

}  // namespace batchlas::impl

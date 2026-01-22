#pragma once
#include <util/sycl-device-queue.hh>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdlib>

#include <mutex>
#include <unordered_map>

#include "util/kernel-trace.hh"

inline bool batchlas_queue_profiling_enabled() {
    // Keep profiling opt-in to avoid overhead in non-benchmark runs.
    // Kernel trace implies profiling; benchmarks can enable profiling without tracing.
    auto env_truthy = [](const char* v) {
        if (!v) return false;
        return (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE" ||
                std::string(v) == "on" || std::string(v) == "ON");
    };
    return batchlas_kernel_trace::enabled() ||
           env_truthy(std::getenv("BATCHLAS_QUEUE_PROFILING")) ||
           env_truthy(std::getenv("BATCHLAS_BENCH_PROFILING"));
}

struct QueueImpl : public sycl::queue{
    using sycl::queue::queue;

    // Tracks the last event submitted to this queue via the wrappers below.
    // Used to implement a cheap get_event() for in-order queues.
    mutable std::optional<sycl::event> last_event_;

    static const sycl::context& shared_context(Device dev) {
        static std::mutex m;
        static std::unordered_map<std::uint64_t, sycl::context> contexts;
        const std::uint64_t key = (static_cast<std::uint64_t>(dev.idx) & 0xffffffffull) |
                                  (static_cast<std::uint64_t>(static_cast<int>(dev.type)) << 32);
        std::lock_guard<std::mutex> lock(m);
        auto it = contexts.find(key);
        if (it != contexts.end()) return it->second;

        const sycl::device sycl_dev = device_arrays.at((int)dev.type).at(dev.idx);
        auto [new_it, _] = contexts.emplace(key, sycl::context(sycl_dev));
        return new_it->second;
    }

    inline static const auto device_arrays = std::array{ 
                sycl::device::get_devices(sycl::info::device_type::cpu), 
                sycl::device::get_devices(sycl::info::device_type::gpu), 
                sycl::device::get_devices(sycl::info::device_type::accelerator),
                sycl::device::get_devices(sycl::info::device_type::host)};

    static_assert(device_arrays.size() == (int)DeviceType::NUM_DEV_TYPES && "DeviceType enum does not match device_arrays size");
    
    QueueImpl(Device dev, bool in_order)
        : sycl::queue(shared_context(dev),
                      device_arrays.at((int)dev.type).at(dev.idx),
                      [&] {
                          const bool prof = batchlas_queue_profiling_enabled();
                          if (in_order && prof)
                              return sycl::property_list{sycl::property::queue::in_order{},
                                                        sycl::property::queue::enable_profiling{}};
                          if (in_order && !prof)
                              return sycl::property_list{sycl::property::queue::in_order{}};
                          if (!in_order && prof)
                              return sycl::property_list{sycl::property::queue::enable_profiling{}};
                          return sycl::property_list{};
                      }()),
          device_(dev),
          trace_tid_(batchlas_kernel_trace::enabled() ? ++trace_tid_counter_ : 0) {}

    QueueImpl(const sycl::context& ctx, const sycl::device& dev, Device logical_dev, bool in_order)
        : sycl::queue(ctx,
                      dev,
                      [&] {
                          const bool prof = batchlas_queue_profiling_enabled();
                          if (in_order && prof)
                              return sycl::property_list{sycl::property::queue::in_order{},
                                                        sycl::property::queue::enable_profiling{}};
                          if (in_order && !prof)
                              return sycl::property_list{sycl::property::queue::in_order{}};
                          if (!in_order && prof)
                              return sycl::property_list{sycl::property::queue::enable_profiling{}};
                          return sycl::property_list{};
                      }()),
          device_(logical_dev),
          trace_tid_(batchlas_kernel_trace::enabled() ? ++trace_tid_counter_ : 0) {}

    QueueImpl()
        : sycl::queue(shared_context(Device{0, DeviceType::CPU}),
                      device_arrays.at((int)DeviceType::CPU).at(0),
                      batchlas_queue_profiling_enabled()
                          ? sycl::property_list{sycl::property::queue::enable_profiling{}}
                          : sycl::property_list{}),
          device_(Device{0, DeviceType::CPU}),
          trace_tid_(batchlas_kernel_trace::enabled() ? ++trace_tid_counter_ : 0) {}

    template <typename SubmitFunc>
    sycl::event submit(SubmitFunc&& f) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_submit";
        sycl::event e = sycl::queue::submit(std::forward<SubmitFunc>(f));
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::range<Dimensions>& num_work_items, KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_parallel_for";
        sycl::event e = sycl::queue::parallel_for(num_work_items, std::forward<KernelFunc>(kernel_func));
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <typename KernelFunc>
    sycl::event parallel_for(std::size_t num_work_items, KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_parallel_for";
        auto kfunc = std::forward<KernelFunc>(kernel_func);
        sycl::event e = sycl::queue::parallel_for(sycl::range<1>(num_work_items), [=](sycl::id<1> idx) {
            kfunc(static_cast<std::size_t>(idx[0]));
        });
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::nd_range<Dimensions>& exec_range, KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_parallel_for";
        sycl::event e = sycl::queue::parallel_for(exec_range, std::forward<KernelFunc>(kernel_func));
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <typename KernelName, int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::range<Dimensions>& num_work_items, KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_parallel_for";
        sycl::event e = sycl::queue::parallel_for<KernelName>(num_work_items, std::forward<KernelFunc>(kernel_func));
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <typename KernelName, typename KernelFunc>
    sycl::event parallel_for(std::size_t num_work_items, KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_parallel_for";
        auto kfunc = std::forward<KernelFunc>(kernel_func);
        sycl::event e = sycl::queue::parallel_for<KernelName>(sycl::range<1>(num_work_items), [=](sycl::id<1> idx) {
            kfunc(static_cast<std::size_t>(idx[0]));
        });
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <typename KernelName, int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::nd_range<Dimensions>& exec_range, KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_parallel_for";
        sycl::event e = sycl::queue::parallel_for<KernelName>(exec_range, std::forward<KernelFunc>(kernel_func));
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <typename KernelFunc>
    sycl::event single_task(KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_single_task";
        sycl::event e = sycl::queue::single_task(std::forward<KernelFunc>(kernel_func));
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    template <typename KernelName, typename KernelFunc>
    sycl::event single_task(KernelFunc&& kernel_func) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        const char* label = scope ? scope : "sycl_single_task";
        sycl::event e = sycl::queue::single_task<KernelName>(std::forward<KernelFunc>(kernel_func));
        last_event_ = e;
        batchlas_kernel_trace::record_event(*this, e, label, trace_tid_);
        return e;
    }

    const Device device_;
    const std::uint32_t trace_tid_;

    inline static std::atomic<std::uint32_t> trace_tid_counter_{0};
};

struct EventImpl : public sycl::event{
    using sycl::event::event;

    EventImpl(sycl::event&& event) : sycl::event(event) {}
};
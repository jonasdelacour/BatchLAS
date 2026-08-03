#pragma once
#include <util/sycl-device-queue.hh>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdlib>

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "util/kernel-trace.hh"
#include <util/env.hh>

inline bool batchlas_queue_profiling_enabled() {
    // Keep profiling opt-in to avoid overhead in non-benchmark runs.
    // Kernel trace implies profiling; benchmarks can enable profiling without tracing.
    return batchlas_kernel_trace::enabled() ||
           batchlas::env_truthy(std::getenv("BATCHLAS_QUEUE_PROFILING")) ||
           batchlas::env_truthy(std::getenv("BATCHLAS_BENCH_PROFILING"));
}

// Per-queue scratch memory. See util/workspace.hh for the caller-facing rules.
//
// Blocks are never reallocated or moved, only appended to, because a lease that
// is still live must keep its pointer: an inner borrow that does not fit in the
// current block opens a new one rather than growing the old one. Released bytes
// are rewound, not freed, so the steady state is one allocation per distinct
// high-water mark rather than one per call.
struct WorkspaceArena {
    struct Block {
        std::byte* ptr = nullptr;
        size_t size = 0;
    };

    std::vector<Block> blocks_;
    size_t cur_block_ = 0;   // block currently being carved from
    size_t cur_offset_ = 0;  // bytes used within it

    // Matches BumpAllocator's alignment rule so that a lease can be handed
    // straight to one without the pool having to realign it first.
    static size_t alignment_for(const sycl::device& dev) {
        size_t bits = 16 * 8;
        try {
            bits = dev.get_info<sycl::info::device::mem_base_addr_align>();
        } catch (...) {
        }
        return std::max<size_t>(16, bits / 8);
    }

    struct Loan {
        std::byte* ptr;
        size_t bytes;
        size_t block;   // rewind target
        size_t offset;
    };

    Loan acquire(sycl::queue& q, size_t bytes) {
        const size_t align = alignment_for(q.get_device());
        const size_t want = (bytes + align - 1) & ~(align - 1);

        // Position at the first block from here on that can serve the request.
        // Anything before cur_block_ is spoken for by an outstanding lease.
        while (cur_block_ < blocks_.size()) {
            const size_t off = (cur_offset_ + align - 1) & ~(align - 1);
            if (off + want <= blocks_[cur_block_].size) {
                const size_t start = off;
                cur_offset_ = off + want;
                return Loan{blocks_[cur_block_].ptr + start, bytes, cur_block_, start};
            }
            ++cur_block_;
            cur_offset_ = 0;
        }

        // Nothing fits: open a new block. Grow geometrically so a caller that
        // ratchets its request upward does not allocate once per step.
        const size_t last = blocks_.empty() ? 0 : blocks_.back().size;
        const size_t block_size = std::max({want, last * 2, static_cast<size_t>(64 * 1024)});
        auto* p = sycl::aligned_alloc_shared<std::byte>(align, block_size, q.get_device(), q.get_context());
        if (!p) throw std::bad_alloc();
        blocks_.push_back(Block{p, block_size});
        cur_block_ = blocks_.size() - 1;
        cur_offset_ = want;
        return Loan{p, bytes, cur_block_, size_t{0}};
    }

    void release(size_t block, size_t offset) {
        cur_block_ = block;
        cur_offset_ = offset;
    }

    size_t capacity() const {
        size_t total = 0;
        for (const auto& b : blocks_) total += b.size;
        return total;
    }

    // Caller must have drained the queue first -- see ~QueueImpl.
    void free_all(const sycl::context& ctx) {
        for (auto& b : blocks_) {
            try {
                sycl::free(b.ptr, ctx);
            } catch (...) {
                // Destructors must not throw; the runtime may surface prior
                // async device failures here.
            }
        }
        blocks_.clear();
        cur_block_ = 0;
        cur_offset_ = 0;
    }
};

struct QueueImpl : public sycl::queue{
    using sycl::queue::queue;

    ~QueueImpl() {
        // The arena's blocks may still be referenced by enqueued-but-unfinished
        // kernels. Freeing shared USM out from under them is a use-after-free
        // that usually only shows up under load, so drain first.
        if (!arena_.blocks_.empty()) {
            try {
                wait();
            } catch (...) {
            }
            arena_.free_all(get_context());
        }
    }

    // Tracks the last event submitted to this queue via the wrappers below.
    // Used to implement a cheap get_event() for in-order queues.
    mutable std::optional<sycl::event> last_event_;

    WorkspaceArena arena_;

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

    static sycl::property_list make_queue_properties(bool in_order) {
        const bool profiling_enabled = batchlas_queue_profiling_enabled();
        if (in_order && profiling_enabled) {
            return sycl::property_list{sycl::property::queue::in_order{},
                                       sycl::property::queue::enable_profiling{}};
        }
        if (in_order) {
            return sycl::property_list{sycl::property::queue::in_order{}};
        }
        if (profiling_enabled) {
            return sycl::property_list{sycl::property::queue::enable_profiling{}};
        }
        return sycl::property_list{};
    }

    static std::uint32_t allocate_trace_tid() {
        return batchlas_kernel_trace::enabled() ? ++trace_tid_counter_ : 0;
    }

    static const char* trace_label_or_default(const char* default_label) {
        const char* scope = batchlas_kernel_trace::current_scope_name();
        return scope ? scope : default_label;
    }

    template <typename SubmitOp>
    sycl::event submit_and_record(const char* default_label, SubmitOp&& submit_op) {
        sycl::event event = std::forward<SubmitOp>(submit_op)();
        last_event_ = event;
        batchlas_kernel_trace::record_event(*this, event, trace_label_or_default(default_label), trace_tid_);
        return event;
    }
    
    QueueImpl(Device dev, bool in_order)
        : sycl::queue(shared_context(dev),
                      device_arrays.at((int)dev.type).at(dev.idx),
                      make_queue_properties(in_order)),
          device_(dev),
          trace_tid_(allocate_trace_tid()) {}

    QueueImpl(const sycl::context& ctx, const sycl::device& dev, Device logical_dev, bool in_order)
        : sycl::queue(ctx,
                      dev,
                      make_queue_properties(in_order)),
          device_(logical_dev),
          trace_tid_(allocate_trace_tid()) {}

    QueueImpl()
        : sycl::queue(shared_context(Device{0, DeviceType::CPU}),
                      device_arrays.at((int)DeviceType::CPU).at(0),
                      make_queue_properties(false)),
          device_(Device{0, DeviceType::CPU}),
          trace_tid_(allocate_trace_tid()) {}

    template <typename SubmitFunc>
    sycl::event submit(SubmitFunc&& f) {
        return submit_and_record("sycl_submit", [&] {
            return sycl::queue::submit(std::forward<SubmitFunc>(f));
        });
    }

    template <int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::range<Dimensions>& num_work_items, KernelFunc&& kernel_func) {
        return submit_and_record("sycl_parallel_for", [&] {
            return sycl::queue::parallel_for(num_work_items, std::forward<KernelFunc>(kernel_func));
        });
    }

    template <typename KernelFunc>
    sycl::event parallel_for(std::size_t num_work_items, KernelFunc&& kernel_func) {
        auto kfunc = std::forward<KernelFunc>(kernel_func);
        return submit_and_record("sycl_parallel_for", [&] {
            return sycl::queue::parallel_for(sycl::range<1>(num_work_items), [=](sycl::id<1> idx) {
                kfunc(static_cast<std::size_t>(idx[0]));
            });
        });
    }

    template <int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::nd_range<Dimensions>& exec_range, KernelFunc&& kernel_func) {
        return submit_and_record("sycl_parallel_for", [&] {
            return sycl::queue::parallel_for(exec_range, std::forward<KernelFunc>(kernel_func));
        });
    }

    template <typename KernelName, int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::range<Dimensions>& num_work_items, KernelFunc&& kernel_func) {
        return submit_and_record("sycl_parallel_for", [&] {
            return sycl::queue::parallel_for<KernelName>(num_work_items, std::forward<KernelFunc>(kernel_func));
        });
    }

    template <typename KernelName, typename KernelFunc>
    sycl::event parallel_for(std::size_t num_work_items, KernelFunc&& kernel_func) {
        auto kfunc = std::forward<KernelFunc>(kernel_func);
        return submit_and_record("sycl_parallel_for", [&] {
            return sycl::queue::parallel_for<KernelName>(sycl::range<1>(num_work_items), [=](sycl::id<1> idx) {
                kfunc(static_cast<std::size_t>(idx[0]));
            });
        });
    }

    template <typename KernelName, int Dimensions, typename KernelFunc>
    sycl::event parallel_for(const sycl::nd_range<Dimensions>& exec_range, KernelFunc&& kernel_func) {
        return submit_and_record("sycl_parallel_for", [&] {
            return sycl::queue::parallel_for<KernelName>(exec_range, std::forward<KernelFunc>(kernel_func));
        });
    }

    template <typename KernelFunc>
    sycl::event single_task(KernelFunc&& kernel_func) {
        return submit_and_record("sycl_single_task", [&] {
            return sycl::queue::single_task(std::forward<KernelFunc>(kernel_func));
        });
    }

    template <typename KernelName, typename KernelFunc>
    sycl::event single_task(KernelFunc&& kernel_func) {
        return submit_and_record("sycl_single_task", [&] {
            return sycl::queue::single_task<KernelName>(std::forward<KernelFunc>(kernel_func));
        });
    }

    const Device device_;
    const std::uint32_t trace_tid_;

    inline static std::atomic<std::uint32_t> trace_tid_counter_{0};
};

struct EventImpl : public sycl::event{
    using sycl::event::event;

    EventImpl(sycl::event&& event) : sycl::event(event) {}
};
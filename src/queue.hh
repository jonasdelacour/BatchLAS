#pragma once
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/sycl_interop.hh>
#include <sycl/sycl.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Quoted on purpose: src/util/ is private and on no -I line. Never convert to <> or add -I src.
// evidence: docs/design/runtime-internals.md#runtime-internals-symbol-visibility-for-private-headers
#include "util/internal-api.hh"
#include "util/kernel-trace.hh"
#include "sycl/impl.hh"
#include <batchlas/util/env.hh>
#include <batchlas/settings.hh>

// `used` so these inline definitions are emitted for consumers; plain `inline` drops them (verified).
#ifdef __SYCL_DEVICE_ONLY__
#define BATCHLAS_QUEUE_EXPORTED_INLINE inline
#else
#define BATCHLAS_QUEUE_EXPORTED_INLINE [[gnu::used]] inline
#endif

// The whole single-thread enforcement: owner thread id vs caller. Deliberately NOT a mutex:
// serialised calls would still interleave two threads' leases in the arena.
// evidence: docs/design/runtime-internals.md#runtime-internals-queue-thread-ownership-and-the-last-event-holder
[[noreturn]] inline void batchlas_throw_queue_wrong_thread(const char* what) {
    throw batchlas::api_misuse(
        std::string("BatchLAS: ") + what +
        " was called from a thread other than the one that owns this Queue. A Queue is "
        "single-threaded: its workspace arena and its cached last event are unsynchronised, and "
        "sharing one across threads corrupts them. Use one Queue per thread (queues for the same "
        "device share a SYCL context, so they still see each other's memory), or transfer this one "
        "with Queue::attach_to_current_thread() while no other thread is using it.");
}

// Must be namespace batchlas to the end of file (defines types declared there); the throw helper
// above stays global on purpose. evidence: docs/design/runtime-internals.md#runtime-internals-namespace-placement-of-out-of-line-definitions
namespace batchlas {

struct QueueThreadOwner {
    std::thread::id owner_ = std::this_thread::get_id();

    void check(const char* what) const {
        if (std::this_thread::get_id() != owner_) batchlas_throw_queue_wrong_thread(what);
    }

    void rebind() { owner_ = std::this_thread::get_id(); }
};

inline bool batchlas_queue_profiling_enabled() {
    // Opt-in; kernel trace implies it. Read from settings(), never getenv, so configure() applies.
    return batchlas_kernel_trace::enabled() ||
           batchlas::settings().diagnostics.profiling;
}

// Per-queue scratch. Blocks are append-only (a live lease keeps its pointer); released bytes are
// rewound, so release ORDER matters and is enforced in release().
// evidence: docs/design/runtime-internals.md#runtime-internals-the-per-queue-workspace-arena
struct WorkspaceArena {
    struct Block {
        std::byte* ptr = nullptr;
        size_t size = 0;
    };

    std::vector<Block> blocks_;
    size_t cur_block_ = 0;   // block currently being carved from
    size_t cur_offset_ = 0;  // bytes used within it

    // Checked in acquire() and trim(), NOT in release(): ~WorkspaceLease is noexcept.
    QueueThreadOwner owner_;

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
        size_t block;        // rewind target
        size_t offset;
        std::uint64_t seq;   // identifies this loan among the outstanding ones
    };

    // Outstanding loans, innermost last: a seq alone cannot tell "innermost" from "has live leases above".
    struct LiveLoan {
        std::uint64_t seq;
        size_t block;
        size_t offset;
        bool returned;  // released out of order; reclaimed once it reaches the top
    };
    std::vector<LiveLoan> live_;
    std::uint64_t next_seq_ = 0;

    // Returned entries are popped on reaching the top, so non-empty means a lease is truly live.
    bool has_outstanding_loans() const { return !live_.empty(); }

    // Would releasing `seq` make bytes re-servable? Only then must an out-of-order queue drain.
    bool release_reclaims(std::uint64_t seq) const {
        return !live_.empty() && live_.back().seq == seq;
    }

    Loan record_loan(std::byte* ptr, size_t bytes, size_t block, size_t offset) {
        const std::uint64_t seq = ++next_seq_;  // 0 is left as "no loan"
        live_.push_back(LiveLoan{seq, block, offset, false});
        return Loan{ptr, bytes, block, offset, seq};
    }

    Loan acquire(sycl::queue& q, size_t bytes) {
        owner_.check("Queue::workspace()");
        const size_t align = alignment_for(q.get_device());
        const size_t want = (bytes + align - 1) & ~(align - 1);

        // Position at the first block from here on that can serve the request.
        // Anything before cur_block_ is spoken for by an outstanding lease.
        while (cur_block_ < blocks_.size()) {
            const size_t off = (cur_offset_ + align - 1) & ~(align - 1);
            if (off + want <= blocks_[cur_block_].size) {
                const size_t start = off;
                cur_offset_ = off + want;
                return record_loan(blocks_[cur_block_].ptr + start, bytes, cur_block_, start);
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
        return record_loan(p, bytes, cur_block_, size_t{0});
    }

    // Only the INNERMOST loan may move the cursor; rewinding for another would let the next borrow
    // alias a live lease. Out-of-order returns are marked and reclaimed later. `diagnose_out_of_order`
    // is false only for WorkspaceLease move-assignment (`ws = q.workspace(n);` is legal).
    void release(size_t block, size_t offset, std::uint64_t seq, bool diagnose_out_of_order = true) {
        if (!live_.empty() && live_.back().seq == seq) {
            live_.pop_back();
            cur_block_ = block;
            cur_offset_ = offset;
            // Whatever was returned out of order underneath is now innermost, so
            // this is the point at which its bytes become reclaimable.
            while (!live_.empty() && live_.back().returned) {
                cur_block_ = live_.back().block;
                cur_offset_ = live_.back().offset;
                live_.pop_back();
            }
            return;
        }

        assert(!diagnose_out_of_order &&
               "WorkspaceArena: workspace lease released out of order; its bytes stay "
               "reserved until the leases taken after it are released");
        (void)diagnose_out_of_order;  // unused once NDEBUG drops the assert
        // Linear, but only on a path that has already been declared a mistake,
        // and over a stack whose depth is the nesting depth of the call graph.
        for (auto it = live_.rbegin(); it != live_.rend(); ++it) {
            if (it->seq == seq) {
                it->returned = true;
                return;
            }
        }
    }

    // Refuses (no partial trim) while any lease is live. Drains first: a released lease does not
    // mean its kernels have finished reading the bytes.
    bool trim(sycl::queue& q) {
        owner_.check("Queue::trim_workspace()");
        if (has_outstanding_loans()) return false;
        if (blocks_.empty()) return true;
        q.wait();
        free_all(q.get_context());
        return true;
    }

    size_t capacity() const {
        size_t total = 0;
        for (const auto& b : blocks_) total += b.size;
        return total;
    }

    // Caller must have drained the queue first -- see ~QueueImpl -- and must
    // have checked has_outstanding_loans(); ~QueueImpl runs after every lease is
    // gone by construction, trim() checks.
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
        live_.clear();
        // next_seq_ deliberately keeps counting: a sequence number must never be
        // handed out twice, or a stale lease could match a later loan's entry.
    }
};

struct QueueImpl : public sycl::queue{
    using sycl::queue::queue;

    ~QueueImpl() {
        // A live lease here dangles (Queue move-assignment also runs this). Assert, do not defend.
        assert(!arena_.has_outstanding_loans() &&
               "QueueImpl destroyed (or its Queue move-assigned) while a workspace lease is live");

        // Drain first: freeing USM under in-flight kernels is a use-after-free.
        if (!arena_.blocks_.empty()) {
            try {
                wait();
            } catch (...) {
            }
            arena_.free_all(get_context());
        }
    }

    // Last submitted event (cheap in-order get_event()). Thread-guarded: a bare optional tears
    // under two submitting threads even with no arena use. evidence: docs/design/runtime-internals.md#runtime-internals-queue-thread-ownership-and-the-last-event-holder
    class LastEvent {
    public:
        LastEvent& operator=(sycl::event e) {
            owner_.check("A submission to this Queue");
            value_ = std::move(e);
            return *this;
        }

        bool has_value() const {
            owner_.check("Queue::get_event()");
            return value_.has_value();
        }

        const sycl::event& operator*() const {
            owner_.check("Queue::get_event()");
            return *value_;
        }

        void rebind_owner() { owner_.rebind(); }

    private:
        std::optional<sycl::event> value_;
        QueueThreadOwner owner_;
    };

    mutable LastEvent last_event_;

    WorkspaceArena arena_;

    // See Queue::attach_to_current_thread. The two guarded members each carry
    // their own recorded owner, so a transfer has to move both.
    void rebind_thread_owner() {
        arena_.owner_.rebind();
        last_event_.rebind_owner();
    }

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

    // Exported so the vague-linkage fold yields ONE device cache per process, not one per library.
    inline static BATCHLAS_INTERNAL_API const auto device_arrays = std::array{ 
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
    
    // impl::rethrow_async_errors: wait_and_throw throws instead of the default std::terminate.
    QueueImpl(Device dev, bool in_order)
        : sycl::queue(shared_context(dev),
                      device_arrays.at((int)dev.type).at(dev.idx),
                      impl::rethrow_async_errors,
                      make_queue_properties(in_order)),
          device_(dev),
          trace_tid_(allocate_trace_tid()) {}

    QueueImpl(const sycl::context& ctx, const sycl::device& dev, Device logical_dev, bool in_order)
        : sycl::queue(ctx,
                      dev,
                      impl::rethrow_async_errors,
                      make_queue_properties(in_order)),
          device_(logical_dev),
          trace_tid_(allocate_trace_tid()) {}

    QueueImpl()
        : sycl::queue(shared_context(Device{0, DeviceType::CPU}),
                      device_arrays.at((int)DeviceType::CPU).at(0),
                      impl::rethrow_async_errors,
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

// Public members that need QueueImpl/EventImpl complete; hence BATCHLAS_QUEUE_EXPORTED_INLINE.

BATCHLAS_QUEUE_EXPORTED_INLINE void Queue::attach_to_current_thread() {
    // A lease released on the new thread would rewind an arena the old thread is
    // still carving from, which is the corruption this guard exists to stop.
    if (impl_->arena_.has_outstanding_loans()) {
        throw batchlas::api_misuse(
            "Queue::attach_to_current_thread: a workspace lease is still outstanding. Release every "
            "lease before transferring the queue to another thread.");
    }
    impl_->rebind_thread_owner();
}

// CUstream / hipStream_t, else nullptr (also acpp's out-of-order queues: no stream outside a CG).
BATCHLAS_QUEUE_EXPORTED_INLINE void* Queue::native_handle() const {
    return impl::native_stream(*impl_);
}

BATCHLAS_QUEUE_EXPORTED_INLINE sycl::queue& sycl_queue(const Queue& ctx) { return *ctx.impl_; }

BATCHLAS_QUEUE_EXPORTED_INLINE sycl::event sycl_event(const Event& event) {
    // A default-constructed Event has no EventImpl; a default sycl::event is
    // already complete, so ordering against it is a no-op rather than a crash.
    if (!event.impl_) return sycl::event{};
    return static_cast<const sycl::event&>(*event.impl_);
}

BATCHLAS_QUEUE_EXPORTED_INLINE Event event_from_sycl(sycl::event event) {
    return Event(EventImpl(std::move(event)));
}

}  // namespace batchlas

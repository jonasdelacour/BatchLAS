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

// Quoted, and it has to stay quoted. src/util/ is a PRIVATE directory that is
// never installed and is not on any -I line; it holds the only headers left in
// the tree spelled `util/...`, reached exclusively by quoted relative includes.
// The public tree moved to include/batchlas/util/ (spelled <batchlas/util/...>)
// precisely so that no angle-form <util/...> exists anywhere. Do not add
// -I${PROJECT_SOURCE_DIR}/src to a target and do not convert these to <>.
#include "util/kernel-trace.hh"
#include <batchlas/util/env.hh>

// Inline definitions in this private header still have to reach consumers that
// only ever see the declaration in the installed public header, so they must be
// emitted into libbatchlas rather than dropped as unreferenced. `used` does that;
// plain `inline` does not (verified: nothing in-tree calls them, so without this
// the symbol is absent from every object file). Not applied in the SYCL device
// pass, which has no business emitting host-only queue plumbing.
#ifdef __SYCL_DEVICE_ONLY__
#define BATCHLAS_QUEUE_EXPORTED_INLINE inline
#else
#define BATCHLAS_QUEUE_EXPORTED_INLINE [[gnu::used]] inline
#endif

// A Queue is single-threaded by contract; the reasoning is on struct Queue in
// util/sycl-device-queue.hh. This is the whole enforcement: record the thread that
// built the queue, and compare against it on the paths that mutate state shared
// across a queue's calls. The check is a TLS load and a word compare, on paths
// that are about to talk to a device driver.
//
// Deliberately not a mutex. The arena below is a bump allocator whose cursor
// rewinds on release, so serialising the calls would still interleave two threads'
// leases within one block and still corrupt them -- a lock would hide the design
// constraint rather than satisfy it.
[[noreturn]] inline void batchlas_throw_queue_wrong_thread(const char* what) {
    throw std::runtime_error(
        std::string("BatchLAS: ") + what +
        " was called from a thread other than the one that owns this Queue. A Queue is "
        "single-threaded: its workspace arena and its cached last event are unsynchronised, and "
        "sharing one across threads corrupts them. Use one Queue per thread (queues for the same "
        "device share a SYCL context, so they still see each other's memory), or transfer this one "
        "with Queue::attach_to_current_thread() while no other thread is using it.");
}

struct QueueThreadOwner {
    std::thread::id owner_ = std::this_thread::get_id();

    void check(const char* what) const {
        if (std::this_thread::get_id() != owner_) batchlas_throw_queue_wrong_thread(what);
    }

    void rebind() { owner_ = std::this_thread::get_id(); }
};

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
//
// The rewind is what makes release order matter, so the order is enforced here
// rather than left to the caller's discipline: see release(). Documenting the
// invariant was not enough, because WorkspaceLease::release() exists precisely
// so that a caller can hand bytes back early, and doing that to anything but the
// innermost lease used to re-serve memory a live lease was still pointing at.
struct WorkspaceArena {
    struct Block {
        std::byte* ptr = nullptr;
        size_t size = 0;
    };

    std::vector<Block> blocks_;
    size_t cur_block_ = 0;   // block currently being carved from
    size_t cur_offset_ = 0;  // bytes used within it

    // Records the thread that constructed the owning QueueImpl, since the arena is
    // a member of it. Checked where bytes are handed out and where blocks are
    // freed -- see acquire() and trim(). Not checked in release(): every release
    // comes from a lease, every lease comes from an acquire that was already
    // checked, and ~WorkspaceLease is noexcept, so a throw there would terminate
    // instead of diagnosing.
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

    // One entry per outstanding loan, innermost last. It exists so release() can
    // tell "this is the innermost loan" from "this loan has live leases stacked
    // on top of it"; a sequence number alone cannot answer the second question
    // once more than one loan has been returned out of order.
    //
    // std::vector rather than a fixed array because nesting depth is a property
    // of the call graph, not of this file. It keeps its capacity between calls,
    // so after the first few leases the LIFO path is a push_back/pop_back into a
    // warm buffer and allocates nothing.
    struct LiveLoan {
        std::uint64_t seq;
        size_t block;
        size_t offset;
        bool returned;  // released out of order; reclaimed once it reaches the top
    };
    std::vector<LiveLoan> live_;
    std::uint64_t next_seq_ = 0;

    // An entry that has been returned is only kept while something above it is
    // still live -- release() pops it the moment it reaches the top -- so a
    // non-empty stack always means at least one lease is genuinely outstanding.
    bool has_outstanding_loans() const { return !live_.empty(); }

    // Would releasing `seq` right now move the cursor, i.e. make bytes
    // re-servable to the next borrow? WorkspaceLease::release asks before it
    // releases, because that -- and only that -- is when an out-of-order queue
    // has to be drained: bytes that merely change state from live to `returned`
    // are not handed to anyone until the loans above them come back, and that
    // later release does its own drain.
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

    // Hand back the bytes of the loan identified by `seq`.
    //
    // Only the innermost outstanding loan may move the cursor. Rewinding for any
    // other one would re-serve memory that a lease above it still points at, and
    // the next borrow would silently alias it -- a wrong answer rather than a
    // diagnosable failure. An out-of-order return is therefore recorded in place
    // and its bytes are reclaimed later, when the loans stacked on top of it come
    // back. That leaves the arena holding more than it needs to for a while,
    // which is a cost, not a correctness problem.
    //
    // `diagnose_out_of_order` is false for the one caller that knows it is about
    // to return out of order and has no way not to: WorkspaceLease's
    // move-assignment, where the right-hand lease is necessarily acquired before
    // the left-hand one is released. Asserting there would abort perfectly legal
    // code (`ws = q.workspace(n);`) in every non-NDEBUG build. Everywhere else
    // the assert is the point -- an out-of-order return from a scope-bound lease
    // means the scopes are not nested the way the caller thinks they are.
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

    // Return the arena's memory to the runtime. Refuses while any lease is
    // outstanding -- the blocks are what those leases point at -- and reports
    // that back rather than trimming partially.
    //
    // Drains the queue first, for the same reason ~QueueImpl does: a released
    // lease only says the *caller* is finished with the bytes, not that the
    // kernels it enqueued over them have finished reading them. Freeing shared
    // USM out from under work still in flight is a use-after-free that usually
    // only shows up under load.
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
        // A live lease at this point is a dangling one: WorkspaceLease holds a
        // Queue*, so releasing it after the arena is gone would hand a stale
        // block/offset to whatever arena the Queue has next. Scope-bound leases
        // make this impossible for ~Queue, but Queue's move-assignment also runs
        // ~QueueImpl (it replaces impl_), and nothing about that spelling forces
        // the leases to be gone first. Assert rather than defend: a Queue moved
        // out from under a live lease is a bug in the caller, and silently
        // draining and freeing here would only hide it.
        assert(!arena_.has_outstanding_loans() &&
               "QueueImpl destroyed (or its Queue move-assigned) while a workspace lease is live");

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
    //
    // A guarded holder rather than a bare std::optional<sycl::event> because the
    // unsynchronised optional is a race in its own right, independent of the
    // arena: two threads submitting on one Queue tear it and abort inside the SYCL
    // runtime with UR_RESULT_ERROR_INVALID_EVENT, even when both callers supply
    // their own workspace and the arena is never touched. Putting the check on the
    // member means every path that reads or writes it is covered -- the submit
    // wrappers here, and Queue::enqueue/get_event/create_event_after_external_work
    // in util/queue-impl.cc -- without each one having to remember to ask.
    //
    // The interface is the subset of std::optional those callers use, so their
    // spellings are unchanged.
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

// ---------------------------------------------------------------------------
// The public entry points that need QueueImpl/EventImpl to be complete types.
// Declared in util/sycl-device-queue.hh and batchlas/sycl_interop.hh, which
// consumers get; defined here, because this is the only place those types are
// defined. See BATCHLAS_QUEUE_EXPORTED_INLINE at the top for why they carry that
// spelling instead of plain `inline`.
// ---------------------------------------------------------------------------

BATCHLAS_QUEUE_EXPORTED_INLINE void Queue::attach_to_current_thread() {
    // A lease released on the new thread would rewind an arena the old thread is
    // still carving from, which is the corruption this guard exists to stop.
    if (impl_->arena_.has_outstanding_loans()) {
        throw std::runtime_error(
            "Queue::attach_to_current_thread: a workspace lease is still outstanding. Release every "
            "lease before transferring the queue to another thread.");
    }
    impl_->rebind_thread_owner();
}

BATCHLAS_QUEUE_EXPORTED_INLINE void* Queue::native_handle() const {
    switch (impl_->get_backend()) {
#if SYCL_EXT_ONEAPI_BACKEND_CUDA
        case sycl::backend::ext_oneapi_cuda:
            // CUstream, i.e. cudaStream_t.
            return static_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_cuda>(*impl_));
#endif
#if SYCL_EXT_ONEAPI_BACKEND_HIP
        case sycl::backend::ext_oneapi_hip:
            // HIPstream, i.e. hipStream_t.
            return static_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_hip>(*impl_));
#endif
        default:
            // Every other backend deliberately returns nullptr rather than
            // something the caller cannot treat as an unowned stream pointer:
            // Level Zero's queue interop is a variant of two handle types, OpenCL's
            // retains the handle and makes the caller release it, and the host
            // backend has no stream at all. See the declaration.
            return nullptr;
    }
}

namespace batchlas {

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

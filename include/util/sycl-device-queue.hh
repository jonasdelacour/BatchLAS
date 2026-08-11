#pragma once
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <optional>
#include <utility>
#include <iosfwd>
#include <string_view>
#include <type_traits>

#include <util/workspace.hh>
#include <blas/enums.hh>


enum class Policy
{
    SYNC,
    ASYNC
};

enum class DeviceType
{
    CPU,
    GPU,
    ACCELERATOR,
    HOST,
    NUM_DEV_TYPES
};

enum class Vendor
{
    AMD,
    ARM,
    INTEL,
    NVIDIA,
    OTHER
};

// Printers for the three enums above. These enums live in the GLOBAL namespace,
// so their to_string() overloads must too or ADL will not find them, and the
// templated operator<< in namespace batchlas cannot pick them up. See the same
// pattern (and the same rationale) in <blas/enums.hh>.
inline constexpr std::string_view to_string(Policy v) {
    switch (v) {
        case Policy::SYNC: return "SYNC";
        case Policy::ASYNC: return "ASYNC";
    }
    return "Policy(?)";
}

inline constexpr std::string_view to_string(DeviceType v) {
    switch (v) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::GPU: return "GPU";
        case DeviceType::ACCELERATOR: return "ACCELERATOR";
        case DeviceType::HOST: return "HOST";
        case DeviceType::NUM_DEV_TYPES: return "NUM_DEV_TYPES";
    }
    return "DeviceType(?)";
}

inline constexpr std::string_view to_string(Vendor v) {
    switch (v) {
        case Vendor::AMD: return "AMD";
        case Vendor::ARM: return "ARM";
        case Vendor::INTEL: return "INTEL";
        case Vendor::NVIDIA: return "NVIDIA";
        case Vendor::OTHER: return "OTHER";
    }
    return "Vendor(?)";
}

// One overload per enum, with the enum spelled out in the parameter list rather
// than deduced.
//
// The obvious spelling - one template constrained on
// `std::is_enum_v<E> && requires(E e) { to_string(e); }` - is exactly the
// constraint the templated operator<< in namespace batchlas carries, so every
// `os << Backend::CUDA` in the tree would become ambiguous. Constraining that
// template to these three types instead does not help either: batchlas'
// constraint is *satisfied* for Vendor/DeviceType/Policy, because the
// `to_string(e)` in it resolves by ADL to the overloads above, so any TU that
// says `using namespace batchlas;` (or streams from inside the namespace) sees
// two viable candidates in no subsumption relation and fails to compile.
//
// Naming the type makes these strictly more specialised than the batchlas
// template under function-template partial ordering, which settles the tie in
// both directions without either header having to know about the other's enums.
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, Policy value) {
    return os << to_string(value);
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, DeviceType value) {
    return os << to_string(value);
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, Vendor value) {
    return os << to_string(value);
}

inline Vendor str_to_vendor(std::string&& v) {
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v.find("amd") != std::string::npos || v.find("advanced micro devices") != std::string::npos) {
        return Vendor::AMD;
    } else if (v.find("arm") != std::string::npos) {
        return Vendor::ARM;
    } else if (v.find("intel") != std::string::npos) {
        return Vendor::INTEL;
    } else if (v.find("nvidia") != std::string::npos) {
        return Vendor::NVIDIA;
    } else {
        return Vendor::OTHER;
    }
}

enum class DeviceProperty
{
    MAX_WORK_GROUP_SIZE,
    MAX_CLOCK_FREQUENCY,
    MAX_COMPUTE_UNITS,
    MAX_MEM_ALLOC_SIZE,
    GLOBAL_MEM_SIZE, 
    LOCAL_MEM_SIZE,
    MAX_NUM_SUB_GROUPS,
    MAX_SUB_GROUP_SIZE,
    MEM_BASE_ADDR_ALIGN,
    GLOBAL_MEM_CACHE_LINE_SIZE,
    GLOBAL_MEM_CACHE_SIZE,
    NUMBER_OF_PROPERTIES
};

struct Device{
    static std::vector<Device> get_devices(DeviceType type);

    Device() = default;

    Device(size_t idx, DeviceType type) : idx(idx), type(type) {}

    Device(std::string type) {
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        auto pick = [](std::vector<Device> devs, const std::string& name) -> Device {
            if (devs.empty()) throw std::runtime_error("No " + name + " device available");
            return devs.at(0);
        };
        if(type == "cpu") {
            *this = pick(get_devices(DeviceType::CPU), "cpu");
        } else if(type == "gpu") {
            *this = pick(get_devices(DeviceType::GPU), "gpu");
        } else if(type == "accelerator") {
            *this = pick(get_devices(DeviceType::ACCELERATOR), "accelerator");
        } else {
            throw std::runtime_error("Invalid device type: " + type);
        }
    }

    Device(const char* type) : Device(std::string(type)) {}

    inline static Device default_device() {
        if(!get_devices(DeviceType::GPU).empty()) {
            return get_devices(DeviceType::GPU).at(0);
        } else if(!get_devices(DeviceType::CPU).empty()) {
            return get_devices(DeviceType::CPU).at(0);
        } else {
            return get_devices(DeviceType::HOST).at(0);
        }
    }

    std::string get_name() const;
    Vendor get_vendor() const;
    size_t get_property(DeviceProperty property) const;
    

    size_t     idx  = 0;
    DeviceType type = DeviceType::HOST;
};

struct EventImpl;

struct Event {
    std::unique_ptr<EventImpl> impl_;

    Event();
    ~Event();
    Event& operator=(EventImpl&& impl);
    Event(EventImpl&& impl);
    Event(Event&& other);
    Event& operator=(Event&& other);
    void wait() const;
    EventImpl* operator->() const;
    EventImpl& operator*() const;

    // Returns {command_start, command_end} timestamps in nanoseconds when
    // profiling is enabled on the underlying SYCL queue.
    // Returns std::nullopt if profiling timestamps are unavailable.
    std::optional<std::pair<std::uint64_t, std::uint64_t>> profiling_command_start_end_ns() const;
};

struct QueueImpl;

// A Queue is SINGLE-THREADED. It owns a bump-allocated workspace arena and a
// cached "last event", neither of which is synchronised, so handing one Queue to
// a thread pool corrupts both: the arena double-frees or blows its capacity up by
// orders of magnitude, and the cached event aborts inside the SYCL runtime even
// when the caller supplies its own workspace and the arena is never touched.
//
// The rule is one Queue per thread. Queues built for the same Device share a SYCL
// context (see the Queue(const Queue&, bool) constructor), so per-thread queues
// still see each other's USM allocations -- what is per-thread is the arena and
// the event bookkeeping, not the memory.
//
// This is enforced, not merely documented: the operations that mutate that state
// -- workspace(), trim_workspace(), and everything that touches the event chain
// (submissions, enqueue(), get_event(), create_event_after_external_work()) --
// compare std::this_thread::get_id() against the thread that constructed the
// Queue and throw std::runtime_error if they differ. The check is a thread-id
// compare on paths that are already submitting work to a device; it is not
// measurable. It is deliberately NOT a mutex: the arena rewinds a cursor on
// release, so serialising the calls would still hand two threads interleaved
// leases out of the same block. A lock would buy silence, not safety.
//
// Moving a Queue to another thread and using it exclusively there is fine, but
// say so with attach_to_current_thread(); otherwise the guard fires on the first
// call from the new thread.
struct Queue{

    /*  
        Default constructor and destructor must be declared ONLY here and defined in the implementation file. 
        This is necessary because QueueImpl is an incomplete type in this header file.
    */
    Queue(); 
    ~Queue();

    Queue(Device device, bool in_order = true);
    Queue(Device device, batchlas::Backend backend, bool in_order = true);
    // Create a new queue sharing the same SYCL context/device as an existing queue.
    // Useful when using USM pointers/workspaces created for the base queue.
    Queue(const Queue& base, bool in_order);
    Queue(Queue&& other); //= default;
    Queue& operator=(Queue&& other);// = default;
    Queue(const Queue& other) = delete;
    Queue& operator=(const Queue& other) = delete;
    // Copy constructor and assignment operator are deleted because the unique_ptr is non-copyable


    QueueImpl* operator->() const;
    QueueImpl& operator*() const;
    
    void enqueue(Event& event);
    Event get_event() const;
    // Force creation of a barrier event for tracking external library calls (cuBLAS, rocBLAS, etc.)
    // that execute on the queue's underlying stream but don't go through SYCL submission.
    Event create_event_after_external_work();

    template <typename EventContainer>
    void enqueue(EventContainer& events){
        for(auto& event : events){
            enqueue(event);
        }
    }

    void wait() const;
    void wait_and_throw() const;

    // Hand this Queue's single-thread ownership to the calling thread, so that
    // the guard described above accepts calls from here instead of from the
    // thread that constructed it. This is the supported way to build queues on
    // one thread and run them on another; it is a transfer, not a share.
    //
    // Call it from the new owner, once, before that thread's first use, and only
    // while no other thread is using the Queue -- there is no way for this to
    // check that for you. Throws if a workspace lease is still outstanding, since
    // a lease released on the new thread would rewind an arena the old thread is
    // still carving from.
    void attach_to_current_thread();

    // Borrow `bytes` of device scratch backed by this queue's workspace arena.
    // See util/workspace.hh for the lifetime rules; in short, the memory is
    // owned by the queue, so it stays valid until the lease is released rather
    // than until the calling function returns.
    [[nodiscard]] batchlas::WorkspaceLease workspace(size_t bytes);

    // Total bytes the arena currently holds. Diagnostics and tests only.
    size_t workspace_capacity() const;

    // Return the workspace arena's memory to the runtime, dropping the queue's
    // high-water mark back to nothing.
    //
    // The arena otherwise only frees in ~Queue, so a single large call keeps its
    // peak reserved for as long as the queue lives. That is the intended trade
    // for a scoped queue -- it is what makes repeated calls stop allocating --
    // but not for a long-lived one. The motivating case is the Python bindings,
    // which hold one queue for the lifetime of the interpreter; nothing under
    // python/batchlas/bindings/ exposes this yet, so today the only callers are
    // C++ ones and the tests.
    //
    // Returns false and does nothing while any lease is outstanding, since the
    // leases point into the blocks this would free -- the result is the only way
    // to tell "freed the peak" from "refused", so it is [[nodiscard]] rather
    // than advisory. Blocks until the queue is idle before freeing, for the same
    // reason ~Queue does: enqueued-but-unfinished kernels may still be reading
    // the blocks even though every lease has been handed back. That drain can
    // therefore throw, surfacing an async device failure from earlier work; this
    // is not a noexcept cleanup call.
    [[nodiscard]] bool trim_workspace();

    std::unique_ptr<QueueImpl> impl_;

    Device device() const { return device_; }
    bool in_order() const { return in_order_; }

    // The backend this queue dispatches to. Never returns Backend::AUTO: an
    // AUTO queue resolves once, on first query, to whatever suits its device.
    // Callers that want the unresolved setting can ask requested_backend().
    batchlas::Backend backend() const;
    batchlas::Backend requested_backend() const { return backend_; }

    // Pin this queue to a backend, or hand it back to AUTO. Throws if the
    // backend was not compiled in, because the alternative is a call that
    // type-checks and then fails at dispatch time with no useful context.
    void set_backend(batchlas::Backend backend);

    // Whether a backend was compiled into this build. Runtime-queryable
    // counterpart to the BATCHLAS_HAS_*_BACKEND macros.
    static bool backend_available(batchlas::Backend backend);

    // Whether `ptr` is reachable from the kernels this queue submits.
    //
    // A MatrixView/Span takes a bare pointer and cannot check where it came
    // from, so passing ordinary host memory (std::vector, new, malloc) to a GPU
    // queue used to reach the device as a wild address: CUDA_ERROR_ILLEGAL_ADDRESS,
    // and then SIGABRT during runtime teardown that no catch block can stop.
    // This answers the question up front instead.
    //
    // True for any USM allocation reachable from this queue's context -- device,
    // shared and host -- and, on a host/CPU device, for ordinary host memory too,
    // since there the two are the same thing. False for a null pointer only when
    // the device cannot reach it.
    //
    // Costs one USM query (~70ns measured), so it is affordable per call but not
    // per element. Defined out of line: this header deliberately does not include
    // <sycl/sycl.hpp>.
    bool is_device_accessible(const void* ptr) const;

    // is_device_accessible, but throws std::invalid_argument naming the argument
    // and what to do about it. `what` should name the parameter at the call site
    // ("gemm: A"), because the whole point is to say which one is wrong.
    void require_device_accessible(const void* ptr, const char* what) const;

    // The backend-native stream this queue submits on, as an opaque pointer:
    // a CUstream (the same thing as cudaStream_t) when the queue runs on the
    // CUDA backend, a hipStream_t on HIP. static_cast it back to that type.
    //
    // Returns nullptr on every other backend -- including the host/CPU one, which
    // has no such handle, and Level Zero and OpenCL, whose SYCL interop returns a
    // variant and a retained (caller-must-release) handle respectively; neither
    // fits "an unowned pointer you may keep using". Check for nullptr rather than
    // assuming; a Queue built with Device("cpu") is the common surprise.
    //
    // The stream belongs to the Queue. Use it to run your own work in the same
    // stream -- cublasSetStream, cudaMemcpyAsync, your own kernels -- but do not
    // destroy it, and do not use it after the Queue is gone. Work you push onto it
    // is ordered by the stream itself, so on the default in-order Queue it runs
    // after everything BatchLAS has already submitted. The reverse direction is
    // not automatic: SYCL does not know about your work, so to make the next
    // BatchLAS call wait for it, take create_event_after_external_work() after
    // enqueueing it and pass that event on. That is exactly what that function is
    // for.
    //
    // No SYCL types appear in this signature on purpose, so it stays usable
    // without <sycl/sycl.hpp>. For the SYCL-typed conversions -- the sycl::queue
    // itself, and sycl::event in both directions -- include
    // <batchlas/sycl_interop.hh>, which is not pulled in by the umbrella headers.
    void* native_handle() const;

    private:
        Device device_;
        bool in_order_;
        batchlas::Backend backend_ = batchlas::Backend::AUTO;
        mutable batchlas::Backend resolved_backend_ = batchlas::Backend::AUTO;
};


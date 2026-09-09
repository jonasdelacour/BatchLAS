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

// First, so every public header that reaches a Queue also sees the exception
// hierarchy: <batchlas/error.hh> is dependency-free (only <stdexcept> and
// <string>), so this costs nothing and cannot form a cycle.
#include <batchlas/error.hh>
#include <batchlas/util/workspace.hh>
#include <batchlas/blas/enums.hh>

namespace batchlas {

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

// These enums live in namespace batchlas, so their to_string() overloads must
// too, or ADL will not find them. Exactly the pattern of <batchlas/blas/enums.hh>,
// whose namespace these now share.
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

// One overload per enum, not a constrained template. These now share a namespace
// with the constrained enum-streaming template in <batchlas/blas/enums.hh>; a
// concrete second parameter wins partial ordering against its deduced `E`, so the
// specific form is what keeps `os << policy` unambiguous. A constrained template
// here would tie with that one instead.
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
            if (devs.empty()) throw batchlas::device_error("No " + name + " device available");
            return devs.at(0);
        };
        if(type == "cpu") {
            *this = pick(get_devices(DeviceType::CPU), "cpu");
        } else if(type == "gpu") {
            *this = pick(get_devices(DeviceType::GPU), "gpu");
        } else if(type == "accelerator") {
            *this = pick(get_devices(DeviceType::ACCELERATOR), "accelerator");
        } else {
            throw batchlas::invalid_argument("Invalid device type: " + type);
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

    // ENUMERATED from sycl::info::device::sub_group_sizes, not
    // get_property(MAX_SUB_GROUP_SIZE): a false accept aborts a
    // [[sycl::reqd_sub_group_size]] launch. evidence: docs/perf/gemv.md#the-sub-route-gates
    bool supports_sub_group_size(size_t size) const;



    size_t     idx  = 0;
    DeviceType type = DeviceType::HOST;
};

struct EventImpl;

// [[nodiscard]] on the TYPE, not on each of the ~279 functions that return one:
// the operations live in include/batchlas/blas/functions/*.hh behind generated
// forwarders and dispatch macros, and marking the class is the only way to reach
// every one of them from a single place.
//
// WHY IT MATTERS. An Event is how a caller orders work that the library cannot
// order for it: an out-of-order Queue, a second Queue sharing the context, or a
// hand-off to raw SYCL. Dropping it there is a silent race, and it used to be
// invisible. It is a WARNING, not an error -- the default Queue is in-order, so
// the overwhelmingly common case of chaining calls on one Queue is correct
// without ever touching the Event. That is exactly why the in-tree discards
// below are spelled `(void)`: each one is a claim that the queue's own ordering
// is enough, and the cast is what makes the claim deliberate and greppable
// instead of accidental.
struct [[nodiscard]] Event {
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

    // {command_start, command_end} in nanoseconds when profiling is enabled on
    // the underlying SYCL queue; std::nullopt when it is not.
    std::optional<std::pair<std::uint64_t, std::uint64_t>> profiling_command_start_end_ns() const;
};

struct QueueImpl;

// A Queue is SINGLE-THREADED: it owns an unsynchronised workspace arena and a
// cached "last event", and the operations that mutate either throw if called
// from another thread. docs/cpp-api.md#synchronisation-and-threading
struct Queue{

    /* Declared here, defined in the .cc: QueueImpl is incomplete here. */
    Queue(); 
    ~Queue();

    Queue(Device device, bool in_order = true);
    Queue(Device device, batchlas::Backend backend, bool in_order = true);
    // Shares `base`'s context/device: USM pointers and workspaces for `base` stay usable.
    Queue(const Queue& base, bool in_order);
    Queue(Queue&& other); //= default;
    Queue& operator=(Queue&& other);// = default;
    Queue(const Queue& other) = delete;
    Queue& operator=(const Queue& other) = delete;


    QueueImpl* operator->() const;
    QueueImpl& operator*() const;
    
    void enqueue(Event& event);
    Event get_event() const;
    // Barrier event for external library work (cuBLAS, rocBLAS) that runs on the
    // queue's stream without going through SYCL submission.
    Event create_event_after_external_work();

    template <typename EventContainer>
    void enqueue(EventContainer& events){
        for(auto& event : events){
            enqueue(event);
        }
    }

    void wait() const;
    void wait_and_throw() const;

    // Transfers single-thread ownership (not a share). Call once from the new owner
    // while no other thread uses the Queue; unchecked. Throws if a lease is outstanding.
    void attach_to_current_thread();

    // Borrow `bytes` of device scratch. The queue owns it: it stays valid until
    // the lease is released, not until the caller returns. See workspace.hh.
    [[nodiscard]] batchlas::WorkspaceLease workspace(size_t bytes);

    size_t workspace_capacity() const;

    // Returns the arena's memory to the runtime (otherwise it frees only in ~Queue).
    // Returns false while any lease is outstanding; waits for idle first, so it can throw.
    [[nodiscard]] bool trim_workspace();

    std::unique_ptr<QueueImpl> impl_;

    Device device() const { return device_; }
    bool in_order() const { return in_order_; }

    // Never returns Backend::AUTO: an AUTO queue resolves once, on first query.
    // requested_backend() returns the unresolved setting.
    batchlas::Backend backend() const;
    batchlas::Backend requested_backend() const { return backend_; }

    // Pin this queue to a backend, or hand it back to AUTO. Throws if not compiled in.
    void set_backend(batchlas::Backend backend);

    static bool backend_available(batchlas::Backend backend);

    // True for any USM allocation in this queue's context, and for host memory on a
    // host/CPU device. docs/cpp-api.md#where-the-memory-has-to-live-the-usm-contract
    bool is_device_accessible(const void* ptr) const;

    // Same, but throws std::invalid_argument; `what` names the call-site parameter ("gemm: A").
    void require_device_accessible(const void* ptr, const char* what) const;

    // Native stream as an opaque pointer: CUstream on CUDA, hipStream_t on HIP,
    // nullptr elsewhere; owned by the Queue. Your work on it is ordered after
    // BatchLAS's, not the reverse -- for that see create_event_after_external_work().
    void* native_handle() const;

    private:
        Device device_;
        bool in_order_;
        batchlas::Backend backend_ = batchlas::Backend::AUTO;
        mutable batchlas::Backend resolved_backend_ = batchlas::Backend::AUTO;
};

}  // namespace batchlas

// Transitional compatibility shim. These names used to be declared at global
// scope; they now live in namespace batchlas, and these using-declarations keep
// the old unqualified spellings working for existing code. A consumer that owns
// a name of its own here defines BATCHLAS_NO_GLOBAL_NAMES to switch the block
// off; the block goes away entirely once nothing in tree depends on it.
#ifndef BATCHLAS_NO_GLOBAL_NAMES
using batchlas::Device;
using batchlas::DeviceProperty;
using batchlas::DeviceType;
using batchlas::Event;
using batchlas::EventImpl;
using batchlas::Policy;
using batchlas::Queue;
using batchlas::QueueImpl;
using batchlas::Vendor;
// str_to_vendor takes a std::string, so ADL associates only namespace std and
// never reaches batchlas; its one caller (src/util/queue-impl.cc) calls it
// unqualified. to_string and operator<< are deliberately NOT shimmed: naming
// either would drag the whole batchlas overload set -- every to_string in
// blas/enums.hh, and that header's constrained enum-streaming template -- into
// the consumer's global namespace, i.e. a larger global footprint than the one
// this move removes. ADL covers both for their enums.
using batchlas::str_to_vendor;
#endif

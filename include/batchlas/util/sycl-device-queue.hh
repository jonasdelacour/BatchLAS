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

#include <batchlas/util/workspace.hh>
#include <batchlas/blas/enums.hh>


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

// These enums live in the GLOBAL namespace, so their to_string() overloads must
// too, or ADL will not find them. Same pattern as <batchlas/blas/enums.hh>.
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

// One overload per enum with the type named, not a constrained template: a
// template would tie with batchlas' and make every enum stream ambiguous.
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

// ENUMERATED from sycl::info::device::sub_group_sizes. Not
// get_property(MAX_SUB_GROUP_SIZE), which returns sub_group_sizes()[0] and is
// wrong in both directions -- the false-accept aborts a
// [[sycl::reqd_sub_group_size(32)]] launch. evidence: docs/perf/gemv.md#the-sub-route-gates
    bool supports_sub_group_size(size_t size) const;



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

    // {command_start, command_end} in nanoseconds when profiling is enabled on
    // the underlying SYCL queue; std::nullopt when it is not.
    std::optional<std::pair<std::uint64_t, std::uint64_t>> profiling_command_start_end_ns() const;
};

struct QueueImpl;

// A Queue is SINGLE-THREADED: it owns an unsynchronised workspace arena and a
// cached "last event", and the operations that mutate either throw if called
// from a thread other than the constructing one. Not a mutex, deliberately: the
// arena rewinds a cursor on release, so serialising would still interleave
// leases out of one block. docs/cpp-api.md#synchronisation-and-threading
struct Queue{

    /* Declared here, defined in the .cc: QueueImpl is incomplete here. */
    Queue(); 
    ~Queue();

    Queue(Device device, bool in_order = true);
    Queue(Device device, batchlas::Backend backend, bool in_order = true);
    // Shares `base`'s SYCL context/device, so USM pointers and workspaces made
    // for `base` stay usable.
    Queue(const Queue& base, bool in_order);
    Queue(Queue&& other); //= default;
    Queue& operator=(Queue&& other);// = default;
    Queue(const Queue& other) = delete;
    Queue& operator=(const Queue& other) = delete;


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

    // Transfer single-thread ownership to the caller -- a transfer, not a share.
    // Call once from the new owner before its first use and while no other thread
    // is using the Queue; that is unchecked. Throws if a lease is outstanding.
    void attach_to_current_thread();

    // Borrow `bytes` of device scratch. The queue owns it: it stays valid until
    // the lease is released, not until the caller returns. See workspace.hh.
    [[nodiscard]] batchlas::WorkspaceLease workspace(size_t bytes);

    // Total bytes the arena currently holds. Diagnostics and tests only.
    size_t workspace_capacity() const;

    // Return the arena's memory to the runtime; otherwise it only frees in ~Queue.
    // Returns false and does nothing while any lease is outstanding. Blocks until
    // the queue is idle first, so it can throw -- not a noexcept cleanup call.
    [[nodiscard]] bool trim_workspace();

    std::unique_ptr<QueueImpl> impl_;

    Device device() const { return device_; }
    bool in_order() const { return in_order_; }

    // Never returns Backend::AUTO: an AUTO queue resolves once, on first query.
    // requested_backend() returns the unresolved setting.
    batchlas::Backend backend() const;
    batchlas::Backend requested_backend() const { return backend_; }

    // Pin this queue to a backend, or hand it back to AUTO. Throws if the
    // backend was not compiled in.
    void set_backend(batchlas::Backend backend);

    static bool backend_available(batchlas::Backend backend);

    // True for any USM allocation in this queue's context, and on a host/CPU
    // device for ordinary host memory too. Costs one USM query: affordable per
    // call, not per element. docs/cpp-api.md#where-the-memory-has-to-live-the-usm-contract
    bool is_device_accessible(const void* ptr) const;

    // is_device_accessible, but throws std::invalid_argument. `what` should name
    // the parameter at the call site ("gemm: A"), since that is the point.
    void require_device_accessible(const void* ptr, const char* what) const;

    // The native stream, as an opaque pointer: CUstream on CUDA, hipStream_t on
    // HIP, nullptr on every other backend (host/CPU included). Owned by the Queue.
    // Your work on it is ordered after BatchLAS's, but not the reverse -- for that
    // take create_event_after_external_work(). docs/cpp-api.md#interop-with-cuda-and-with-your-own-sycl
    void* native_handle() const;

    private:
        Device device_;
        bool in_order_;
        batchlas::Backend backend_ = batchlas::Backend::AUTO;
        mutable batchlas::Backend resolved_backend_ = batchlas::Backend::AUTO;
};


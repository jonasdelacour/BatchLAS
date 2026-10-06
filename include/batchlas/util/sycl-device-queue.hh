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

/// @file
/// @brief batchlas::Device, batchlas::Event and batchlas::Queue: where work runs and how it is
/// ordered.
///
/// Every entry point enqueues on a Queue and returns an Event immediately; results are readable
/// only after waiting on the Event or the Queue. See "Devices and queues" and "Synchronisation and
/// threading" in @ref md_docs_2cpp-api.
/// @ingroup core

// error.hh first, so every header that reaches a Queue sees the exception hierarchy; it is
// dependency-free, so this cannot form a cycle.
#include <batchlas/error.hh>
#include <batchlas/export.hh>
#include <batchlas/util/workspace.hh>
#include <batchlas/blas/enums.hh>

namespace batchlas {

/// @brief Synchronous or asynchronous execution policy.
/// @ingroup core
enum class Policy
{
    SYNC,   ///< Wait for completion.
    ASYNC   ///< Return after enqueueing.
};

/// @brief The kind of SYCL device.
/// @ingroup core
enum class DeviceType
{
    CPU,            ///< A CPU device.
    GPU,            ///< A GPU device.
    ACCELERATOR,    ///< An accelerator device.
    HOST,           ///< The SYCL host device.
    NUM_DEV_TYPES   ///< Number of device types (not a device type).
};

/// @brief Device vendor, parsed from the SYCL vendor string by str_to_vendor().
/// @ingroup core
enum class Vendor
{
    AMD,     ///< AMD.
    ARM,     ///< Arm.
    INTEL,   ///< Intel.
    NVIDIA,  ///< NVIDIA.
    OTHER    ///< Any other vendor.
};

// to_string() must share the enums' namespace or ADL will not find it.
/// @brief Name of a Policy value.
/// @ingroup core
inline constexpr std::string_view to_string(Policy v) {
    switch (v) {
        case Policy::SYNC: return "SYNC";
        case Policy::ASYNC: return "ASYNC";
    }
    return "Policy(?)";
}

/// @brief Name of a DeviceType value.
/// @ingroup core
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

/// @brief Name of a Vendor value.
/// @ingroup core
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

// One overload per enum, NOT a constrained template: the concrete parameter is what wins partial
// ordering against the enum-streaming template in blas/enums.hh; a template here would tie.
/// @brief Stream a Policy by name.
/// @ingroup core
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, Policy value) {
    return os << to_string(value);
}

/// @brief Stream a DeviceType by name.
/// @ingroup core
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, DeviceType value) {
    return os << to_string(value);
}

/// @brief Stream a Vendor by name.
/// @ingroup core
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, Vendor value) {
    return os << to_string(value);
}

/// @brief Classify a SYCL vendor string (case-insensitive substring match).
/// @param v the vendor string, e.g. from `sycl::info::device::vendor`
/// @return the matching Vendor, or Vendor::OTHER
/// @ingroup core
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

/// @brief A device property queryable with Device::get_property(); each maps to the SYCL
/// `sycl::info::device` descriptor of the same name.
/// @ingroup core
enum class DeviceProperty
{
    MAX_WORK_GROUP_SIZE,         ///< Maximum work-group size.
    MAX_CLOCK_FREQUENCY,         ///< Maximum clock frequency (MHz).
    MAX_COMPUTE_UNITS,           ///< Number of compute units.
    MAX_MEM_ALLOC_SIZE,          ///< Largest single allocation (bytes).
    GLOBAL_MEM_SIZE,             ///< Global memory size (bytes).
    LOCAL_MEM_SIZE,              ///< Local (shared) memory per work-group (bytes).
    MAX_NUM_SUB_GROUPS,          ///< Maximum sub-groups per work-group.
    MAX_SUB_GROUP_SIZE,          ///< Largest supported sub-group size; see Device::supports_sub_group_size().
    MEM_BASE_ADDR_ALIGN,         ///< Base address alignment (bits).
    GLOBAL_MEM_CACHE_LINE_SIZE,  ///< Global memory cache line (bytes).
    GLOBAL_MEM_CACHE_SIZE,       ///< Global memory cache size (bytes).
    NUMBER_OF_PROPERTIES         ///< Number of properties (not a property).
};

/// @brief True for consumer Blackwell (RTX 50xx / RTX PRO 6000, sm_120/121).
/// @param cuda_cc a value of Device::cuda_compute_capability()
/// @ingroup core
// evidence: docs/perf/blackwell.md
constexpr bool is_sm120_family(int cuda_cc) { return cuda_cc >= 120 && cuda_cc < 130; }

/// @brief A handle to one SYCL device: its type and its index among devices of that type.
///
/// @code
/// auto gpus = Device::get_devices(DeviceType::GPU);
/// Queue ctx(gpus.at(1));          // the second GPU
/// Queue cpu("cpu");               // the first CPU device
/// @endcode
/// @ingroup core
struct BATCHLAS_API Device{
    /// @brief Every device of `type`, in the SYCL runtime's order.
    static std::vector<Device> get_devices(DeviceType type);

    /// @brief The host device (index 0).
    Device() = default;

    /// @brief The `idx`-th device of `type`, as numbered by get_devices().
    Device(size_t idx, DeviceType type) : idx(idx), type(type) {}

    /// @brief The first device of the named type.
    /// @param type `"cpu"`, `"gpu"` or `"accelerator"` (case-insensitive)
    /// @throws batchlas::device_error if no device of that type exists
    /// @throws batchlas::invalid_argument for any other string
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

    /// @brief As Device(std::string).
    Device(const char* type) : Device(std::string(type)) {}

    /// @brief The first GPU, else the first CPU, else the host device.
    inline static Device default_device() {
        if(!get_devices(DeviceType::GPU).empty()) {
            return get_devices(DeviceType::GPU).at(0);
        } else if(!get_devices(DeviceType::CPU).empty()) {
            return get_devices(DeviceType::CPU).at(0);
        } else {
            return get_devices(DeviceType::HOST).at(0);
        }
    }

    /// @brief The device's name as reported by SYCL.
    std::string get_name() const;
    /// @brief The device's vendor.
    Vendor get_vendor() const;
    /// @brief The value of one device property (units as documented on DeviceProperty).
    size_t get_property(DeviceProperty property) const;

    /// @brief True when `size` is one of the device's supported sub-group sizes.
    // ENUMERATED from sycl::info::device::sub_group_sizes, not
    // get_property(MAX_SUB_GROUP_SIZE): a false accept aborts a
    // [[sycl::reqd_sub_group_size]] launch. evidence: docs/perf/gemv.md#the-sub-route-gates
    bool supports_sub_group_size(size_t size) const;

    /// @brief CUDA compute capability as major*10+minor (89 = sm_89, 120 = sm_120), or 0 for a
    /// non-CUDA device. Memoized per device; for per-architecture routing windows.
    // Parsed from info::device::version, because ext_oneapi_architecture reports
    // "unknown" for sm_100/sm_120 on current DPC++.
    int cuda_compute_capability() const;



    size_t     idx  = 0;                 ///< Index among devices of `type`.
    DeviceType type = DeviceType::HOST;  ///< The device's type.
};

struct EventImpl;

/// @brief Completion handle of enqueued work, returned by every entry point.
///
/// Waiting on it (or on the Queue) is what makes results readable. On the default in-order Queue,
/// later calls on the same Queue are already ordered after it, so it may be ignored there; it is
/// needed to order work across an out-of-order Queue, a second Queue sharing the context, or raw
/// SYCL (see `<batchlas/sycl_interop.hh>`). The type is `[[nodiscard]]`: discard one deliberately
/// with `(void)`. Movable, not copyable.
/// @ingroup core
// Per-member BATCHLAS_API is forced (GCC rejects [[nodiscard]] plus a GNU attribute on the
// class-key). evidence: docs/design/symbol-visibility.md#symbol-visibility-event-carries-per-member-exports
struct [[nodiscard]] Event {
    std::unique_ptr<EventImpl> impl_;  ///< Implementation; null for a default-constructed or moved-from Event.

    /// @brief An empty Event, already complete.
    BATCHLAS_API Event();
    BATCHLAS_API ~Event();
    BATCHLAS_API Event& operator=(EventImpl&& impl);
    BATCHLAS_API Event(EventImpl&& impl);
    BATCHLAS_API Event(Event&& other);
    BATCHLAS_API Event& operator=(Event&& other);
    /// @brief Block until the work this Event tracks has completed.
    BATCHLAS_API void wait() const;
    BATCHLAS_API EventImpl* operator->() const;
    BATCHLAS_API EventImpl& operator*() const;

    /// @brief `{command_start, command_end}` in nanoseconds, or `std::nullopt` when profiling is
    /// not enabled on the underlying SYCL queue (see DiagnosticsSettings::profiling).
    BATCHLAS_API std::optional<std::pair<std::uint64_t, std::uint64_t>> profiling_command_start_end_ns() const;
};

struct QueueImpl;

/// @brief Where BatchLAS work runs: a device, a backend, an ordered SYCL queue and a workspace
/// arena.
///
/// @code
/// Queue ctx;                                   // default device, in-order, backend AUTO
/// Queue ooo(device, /*in_order=*/false);       // out-of-order
/// Queue host(device, Backend::NETLIB);         // backend pinned
/// Queue sibling(ctx, /*in_order=*/true);       // shares ctx's context and device
/// @endcode
/// Every entry point takes its backend from the Queue and leases its default workspace from the
/// Queue's arena (see WorkspaceLease). Movable, not copyable.
/// @warning A Queue is single-threaded: it owns an unsynchronised arena and a cached last event,
///          and workspace(), trim_workspace(), submissions, enqueue(), get_event() and
///          create_event_after_external_work() throw batchlas::api_misuse when called from a
///          thread other than the owner. Use one Queue per thread, or hand one over with
///          attach_to_current_thread().
/// @see "Synchronisation and threading" in @ref md_docs_2cpp-api
/// @ingroup core
struct BATCHLAS_API Queue{

    /// @brief A Queue on Device::default_device(), in-order, backend AUTO.
    Queue();
    ~Queue();

    /// @brief A Queue on `device` with backend AUTO.
    /// @param device   the device
    /// @param in_order true (default) for an in-order queue; out-of-order queues make arena-backed
    ///                 workspace releases drain the queue
    Queue(Device device, bool in_order = true);
    /// @brief A Queue on `device` pinned to `backend`.
    /// @throws batchlas::unsupported if `backend` is not compiled in (see set_backend())
    Queue(Device device, batchlas::Backend backend, bool in_order = true);
    /// @brief A Queue sharing `base`'s SYCL context and device, so USM pointers and workspaces
    /// allocated for `base` stay usable. It has its own arena and event chain.
    Queue(const Queue& base, bool in_order);
    Queue(Queue&& other); //= default;
    Queue& operator=(Queue&& other);// = default;
    Queue(const Queue& other) = delete;
    Queue& operator=(const Queue& other) = delete;


    QueueImpl* operator->() const;
    QueueImpl& operator*() const;

    /// @brief Order all later work on this Queue after `event`, without a host wait.
    /// A default-constructed Event is ignored.
    void enqueue(Event& event);
    /// @brief An Event that completes after everything enqueued on this Queue so far.
    Event get_event() const;
    /// @brief Like get_event(), but also ordered after work submitted directly to the native
    /// stream (cuBLAS, rocBLAS, your own kernels) that bypassed SYCL submission.
    Event create_event_after_external_work();

    /// @brief enqueue() each Event in `events`.
    template <typename EventContainer>
    void enqueue(EventContainer& events){
        for(auto& event : events){
            enqueue(event);
        }
    }

    /// @brief Block until all work enqueued on this Queue has completed.
    void wait() const;
    /// @brief As wait(), then rethrow any asynchronous error the device reported.
    void wait_and_throw() const;

    /// @brief Transfer single-thread ownership to the calling thread (a move, not a share).
    /// @pre No other thread uses the Queue during or after the call (unchecked).
    /// @throws batchlas::api_misuse if a workspace lease is outstanding
    // BATCHLAS_API on the member is required (inline member defined in src/queue.hh).
    // evidence: docs/design/symbol-visibility.md#symbol-visibility-exported-inline-queue-members
    BATCHLAS_API void attach_to_current_thread();

    /// @brief Borrow `bytes` of device scratch from this Queue's arena.
    ///
    /// The Queue owns the memory: it stays valid until the lease is released, not until the
    /// caller returns. Keep the lease in a named variable (`auto ws = ctx.workspace(n);`).
    [[nodiscard]] batchlas::WorkspaceLease workspace(size_t bytes);

    /// @brief Bytes the arena currently holds. It grows to the peak requested and keeps it.
    size_t workspace_capacity() const;

    /// @brief Return the arena's memory to the runtime (otherwise it is freed only in ~Queue).
    /// @return false, doing nothing, while any lease is outstanding
    /// @note Waits for the queue to be idle first, so it can throw an asynchronous error.
    [[nodiscard]] bool trim_workspace();

    std::unique_ptr<QueueImpl> impl_;  ///< Implementation (SYCL queue, arena, event chain).

    /// @brief The device this Queue runs on.
    Device device() const { return device_; }
    /// @brief True for an in-order Queue.
    bool in_order() const { return in_order_; }

    /// @brief The backend calls dispatch to; never Backend::AUTO (an AUTO Queue resolves once, on
    /// first query, from the device vendor and the compiled-in backends).
    batchlas::Backend backend() const;
    /// @brief The backend as requested, possibly Backend::AUTO.
    batchlas::Backend requested_backend() const { return backend_; }

    /// @brief Pin this Queue to `backend`, or pass Backend::AUTO to hand it back to automatic
    /// selection.
    /// @throws batchlas::unsupported if `backend` is not compiled into this build
    void set_backend(batchlas::Backend backend);

    /// @brief True when `backend` is compiled into this build.
    static bool backend_available(batchlas::Backend backend);

    /// @brief True for any USM allocation in this Queue's context, and for any non-null pointer on
    /// a host/CPU device.
    // docs/cpp-api.md#where-the-memory-has-to-live-the-usm-contract
    bool is_device_accessible(const void* ptr) const;

    /// @brief As is_device_accessible(), but throws.
    /// @param ptr  the pointer to check
    /// @param what names the call-site parameter in the message, e.g. `"gemm: A"`
    /// @throws batchlas::invalid_argument if `ptr` is null or not reachable from the device
    void require_device_accessible(const void* ptr, const char* what) const;

    /// @brief The native stream as an opaque pointer: a `CUstream` when the device runs on the
    /// CUDA SYCL backend, a `hipStream_t` on HIP, `nullptr` elsewhere.
    ///
    /// Keyed off the device, not backend(). Owned by the Queue: do not destroy it or use it past
    /// the Queue's lifetime. Work pushed to it runs after BatchLAS's; to make BatchLAS wait for
    /// that work, call create_event_after_external_work().
    BATCHLAS_API void* native_handle() const;

    private:
        Device device_;
        bool in_order_;
        batchlas::Backend backend_ = batchlas::Backend::AUTO;
        mutable batchlas::Backend resolved_backend_ = batchlas::Backend::AUTO;
};

}  // namespace batchlas

// Transitional global-scope shim; define BATCHLAS_NO_GLOBAL_NAMES to switch it off.
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
// str_to_vendor needs the shim (a std::string argument never reaches batchlas by ADL).
// to_string/operator<< are deliberately NOT shimmed: that would drag the whole enums.hh overload
// set into the global namespace, and ADL already finds them.
using batchlas::str_to_vendor;
#endif

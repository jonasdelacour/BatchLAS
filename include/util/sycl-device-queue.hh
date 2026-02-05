#pragma once
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <optional>
#include <utility>


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
        if(type == "cpu") {
            *this = get_devices(DeviceType::CPU).at(0);
        } else if(type == "gpu") {
            *this = get_devices(DeviceType::GPU).at(0);
        } else if(type == "accelerator") {
            *this = get_devices(DeviceType::ACCELERATOR).at(0);
        } else {
            throw std::runtime_error("Invalid device type");
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

    inline static std::vector<Device> cpus_ =         get_devices(DeviceType::CPU);
    inline static std::vector<Device> gpus_ =         get_devices(DeviceType::GPU);
    inline static std::vector<Device> accelerators_ = get_devices(DeviceType::ACCELERATOR);

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

struct Queue{

    /*  
        Default constructor and destructor must be declared ONLY here and defined in the implementation file. 
        This is necessary because QueueImpl is an incomplete type in this header file.
    */
    Queue(); 
    ~Queue();

    Queue(Device device, bool in_order = true);
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
    
    std::unique_ptr<QueueImpl> impl_;

    Device device() const { return device_; }
    bool in_order() const { return in_order_; }
    
    private:
        Device device_;
        bool in_order_;
};


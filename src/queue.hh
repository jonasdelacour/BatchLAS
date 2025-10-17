#pragma once
#include <util/sycl-device-queue.hh>
#include <sycl/sycl.hpp>

struct QueueImpl : public sycl::queue{
    using sycl::queue::queue;

    inline static const auto device_arrays = std::array{ 
                sycl::device::get_devices(sycl::info::device_type::cpu), 
                sycl::device::get_devices(sycl::info::device_type::gpu), 
                sycl::device::get_devices(sycl::info::device_type::accelerator),
                sycl::device::get_devices(sycl::info::device_type::host)};

    static_assert(device_arrays.size() == (int)DeviceType::NUM_DEV_TYPES && "DeviceType enum does not match device_arrays size");
    
    QueueImpl(Device dev, bool in_order) : sycl::queue( device_arrays.at((int)dev.type).at(dev.idx), in_order ? sycl::property::queue::in_order{} : sycl::property_list{}), device_(dev) {}
    QueueImpl() : sycl::queue( device_arrays.at((int)DeviceType::CPU).at(0)), device_(Device{0, DeviceType::CPU}) {}
    const Device device_;
};

struct EventImpl : public sycl::event{
    using sycl::event::event;

    EventImpl(sycl::event&& event) : sycl::event(event) {}
};
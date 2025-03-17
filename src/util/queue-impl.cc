#pragma once
#include "../queue.hh"
#ifndef DEVICE_CAST
    #define DEVICE_CAST(x,ix) (reinterpret_cast<const sycl::device*>(x)[ix])
#endif

struct EventImpl : public sycl::event{
    using sycl::event::event;

    EventImpl(sycl::event&& event) : sycl::event(event) {}
};

Event::Event() : impl_(std::make_unique<EventImpl>(sycl::event())) {}
Event::Event(EventImpl&& impl) : impl_(std::make_unique<EventImpl>(std::move(impl))) {}
Event& Event::operator=(Event&& other) {
    impl_ = std::move(other.impl_);
    return *this;
}
Event& Event::operator=(EventImpl&& impl) {
    impl_ = std::make_unique<EventImpl>(std::move(impl));
    return *this;
}

Event::Event(Event&& other) = default;


Event::~Event() = default;

void Event::wait() const {impl_->wait();}

EventImpl* Event::operator ->() const {return impl_.get();}
EventImpl& Event::operator *() const {return *impl_;}


Queue::Queue() : device_({0, DeviceType::CPU}), in_order_(true) {
    impl_ = std::make_unique<QueueImpl>(device_, in_order_);
}

Queue::Queue(Device dev, bool in_order) : device_(dev), in_order_(in_order) {
    impl_ = std::make_unique<QueueImpl>(dev, in_order);
}

Queue::~Queue() = default;
Queue::Queue(Queue&& other) = default;
Queue& Queue::operator=(Queue&& other) = default;

void Queue::wait() const {impl_->wait();}
void Queue::wait_and_throw() const {impl_->wait_and_throw();}


QueueImpl* Queue::operator->() const {
    return impl_.get();
}

QueueImpl& Queue::operator*() const {
    return *impl_;
}

void Queue::enqueue(Event& event) {
    if(event.impl_.get() == nullptr) return;
    impl_->submit([&](sycl::handler& h) { 
        h.depends_on(static_cast<sycl::event>(*event));
    });
}

Event Queue::get_event() {
    EventImpl event = impl_->submit([](sycl::handler& h){h.single_task([](){});});
    return event;
}

std::vector<Device> Device::get_devices(DeviceType type){
    std::vector<Device> devices(QueueImpl::device_arrays.at(static_cast<int>(type)).size());
    std::generate(devices.begin(), devices.end(), 
        [i = 0, type]() mutable { return Device(i,type); });
    return devices;
}


std::string Device::get_name() const  {
    return QueueImpl::device_arrays.at(static_cast<int>(type)).at(idx).get_info<sycl::info::device::name>();
}

std::string Device::get_vendor() const {
    return QueueImpl::device_arrays.at(static_cast<int>(type)).at(idx).get_info<sycl::info::device::vendor>();
}

size_t Device::get_property(DeviceProperty property) const {
    auto d = QueueImpl::device_arrays.at(static_cast<int>(type)).at(idx);
    
    switch(int(property)){
        case 0: return d.get_info<sycl::info::device::max_work_group_size>();
        case 1: return d.get_info<sycl::info::device::max_clock_frequency>();
        case 2: return d.get_info<sycl::info::device::max_compute_units>();
        case 3: return d.get_info<sycl::info::device::max_mem_alloc_size>();
        case 4: return d.get_info<sycl::info::device::global_mem_size>();
        case 5: return d.get_info<sycl::info::device::local_mem_size>();
        case 7: return d.get_info<sycl::info::device::max_constant_args>();
        case 8: return d.get_info<sycl::info::device::max_num_sub_groups>();
        case 9: return d.get_info<sycl::info::device::sub_group_sizes>()[0];
        case 10: return d.get_info<sycl::info::device::mem_base_addr_align>();
        default: std::cerr << "Unknown property" << std::endl; return 0;
    }
}

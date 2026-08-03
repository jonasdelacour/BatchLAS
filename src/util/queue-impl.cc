#include "../queue.hh"
#include <batchlas/backend_config.h>
#include <util/sycl-span.hh>
#ifndef DEVICE_CAST
    #define DEVICE_CAST(x,ix) (reinterpret_cast<const sycl::device*>(x)[ix])
#endif

namespace {
class QueueGetEventNoopKernel;
class QueueEnqueueNoopKernel;
class QueueExternalWorkBarrierKernel;
}

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

std::optional<std::pair<std::uint64_t, std::uint64_t>> Event::profiling_command_start_end_ns() const {
    try {
        const auto start = impl_->get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end = impl_->get_profiling_info<sycl::info::event_profiling::command_end>();
        return std::make_pair(static_cast<std::uint64_t>(start), static_cast<std::uint64_t>(end));
    } catch (...) {
        return std::nullopt;
    }
}

EventImpl* Event::operator ->() const {return impl_.get();}
EventImpl& Event::operator *() const {return *impl_;}


Queue::Queue() : device_(Device::default_device()), in_order_(true) {
    impl_ = std::make_unique<QueueImpl>(device_, in_order_);
}

Queue::Queue(Device dev, bool in_order) : device_(dev), in_order_(in_order) {
    impl_ = std::make_unique<QueueImpl>(dev, in_order);
}

Queue::Queue(Device dev, batchlas::Backend backend, bool in_order) : Queue(dev, in_order) {
    set_backend(backend);
}

Queue::Queue(const Queue& base, bool in_order)
    : device_(base.device_), in_order_(in_order), backend_(base.backend_), resolved_backend_(base.resolved_backend_) {
    impl_ = std::make_unique<QueueImpl>(base.impl_->get_context(), base.impl_->get_device(), device_, in_order_);
}

bool Queue::backend_available(batchlas::Backend backend) {
    using batchlas::Backend;
    switch (backend) {
        case Backend::CUDA:   return BATCHLAS_HAS_CUDA_BACKEND;
        case Backend::ROCM:   return BATCHLAS_HAS_ROCM_BACKEND;
        case Backend::MKL:    return BATCHLAS_HAS_MKL_BACKEND;
        case Backend::NETLIB: return BATCHLAS_HAS_HOST_BACKEND;
        // AUTO is a request, not a backend; MAGMA and SYCL have no dispatch
        // targets, so claiming they are available would be a lie that only
        // surfaces at the call.
        default: return false;
    }
}

void Queue::set_backend(batchlas::Backend backend) {
    if (backend != batchlas::Backend::AUTO && !backend_available(backend)) {
        throw std::runtime_error("Queue::set_backend: backend is not compiled into this build of BatchLAS.");
    }
    backend_ = backend;
    resolved_backend_ = batchlas::Backend::AUTO;  // re-resolve on next query
}

batchlas::Backend Queue::backend() const {
    using batchlas::Backend;
    if (backend_ != Backend::AUTO) return backend_;
    if (resolved_backend_ != Backend::AUTO) return resolved_backend_;

    // Resolve AUTO from the device: prefer the vendor's own stack, fall back to
    // the host implementation, which is the only one guaranteed to run anywhere.
    Backend choice = Backend::AUTO;
    if (device_.type == DeviceType::GPU) {
        const Vendor vendor = device_.get_vendor();
        if (vendor == Vendor::NVIDIA && backend_available(Backend::CUDA)) {
            choice = Backend::CUDA;
        } else if (vendor == Vendor::AMD && backend_available(Backend::ROCM)) {
            choice = Backend::ROCM;
        } else if (vendor == Vendor::INTEL && backend_available(Backend::MKL)) {
            choice = Backend::MKL;
        }
    }
    if (choice == Backend::AUTO && backend_available(Backend::NETLIB)) {
        choice = Backend::NETLIB;
    }
    if (choice == Backend::AUTO) {
        throw std::runtime_error("Queue::backend: no backend compiled into this build can serve this device.");
    }
    resolved_backend_ = choice;
    return choice;
}

Queue::~Queue() = default;
Queue::Queue(Queue&& other) = default;

// Written out rather than `= default` on purpose. Move-assignment destroys the
// destination's QueueImpl without running ~Queue, so per-queue state has to be
// torn down on this path too. It is: the workspace arena is a member of
// QueueImpl, so overwriting impl_ below runs ~QueueImpl, which drains the queue
// and frees the arena's blocks. Keeping this written out documents that the
// requirement exists and gives it somewhere to live if state is ever added
// outside QueueImpl -- storing it in a side table keyed on impl_ would be a bug,
// since a later heap reuse of the same address would inherit the entry.
Queue& Queue::operator=(Queue&& other) {
    if (this == &other) return *this;
    device_ = other.device_;
    in_order_ = other.in_order_;
    backend_ = other.backend_;
    resolved_backend_ = other.resolved_backend_;
    impl_ = std::move(other.impl_);
    return *this;
}

void Queue::wait() const {impl_->wait();}
void Queue::wait_and_throw() const {impl_->wait_and_throw();}

batchlas::WorkspaceLease Queue::workspace(size_t bytes) {
    auto loan = impl_->arena_.acquire(*impl_, bytes);
    return batchlas::WorkspaceLease(this, loan.ptr, loan.bytes, loan.block, loan.offset);
}

size_t Queue::workspace_capacity() const { return impl_->arena_.capacity(); }

namespace batchlas {

Span<std::byte> WorkspaceLease::span() const { return Span<std::byte>(ptr_, size_); }
WorkspaceLease::operator Span<std::byte>() const { return span(); }

void WorkspaceLease::release() {
    if (!queue_) return;
    queue_->impl_->arena_.release(block_, offset_);
    queue_ = nullptr;
    ptr_ = nullptr;
    size_ = 0;
}

}  // namespace batchlas


QueueImpl* Queue::operator->() const {
    return impl_.get();
}

QueueImpl& Queue::operator*() const {
    return *impl_;
}

void Queue::enqueue(Event& event) {
    if(event.impl_.get() == nullptr) return;
    // Ensure the queue is ordered after `event`.
    // A command group with only depends_on() is not guaranteed to create an actual
    // scheduling node on all backends. Use a barrier when available, else fall
    // back to a no-op kernel.
    try {
        sycl::event e = impl_->ext_oneapi_submit_barrier({static_cast<sycl::event>(*event)});
        impl_->last_event_ = e;
    } catch (const sycl::exception&) {
        impl_->submit([&](sycl::handler& h) {
            h.depends_on(static_cast<sycl::event>(*event));
            h.single_task<QueueEnqueueNoopKernel>([]() {});
        });
    }
}

Event Queue::get_event() const {
    // For in-order queues, the last submitted event is ordered after all
    // previously enqueued work. Returning it avoids submitting an extra
    // barrier/no-op kernel, which can be very expensive on some backends.
    if (in_order_ && impl_->last_event_.has_value()) {
        EventImpl event = sycl::event(*impl_->last_event_);
        return event;
    }

    // For out-of-order queues (or if no work has been submitted yet), return an
    // event that is ordered after all previously enqueued work.
    // Submitting an unnamed `single_task` can fail under AOT/kernel-bundle
    // builds ("No kernel named ... was found"), especially on CUDA backends.
    try {
        sycl::event e = impl_->ext_oneapi_submit_barrier();
        impl_->last_event_ = e;
        EventImpl event = std::move(e);
        return event;
    } catch (const sycl::exception&) {
        // Some backends (notably certain CUDA/UR stacks) don't support
        // ext_oneapi_submit_barrier and may throw unsupported-feature errors.
        EventImpl event = impl_->submit([&](sycl::handler& h) {
            h.single_task<QueueGetEventNoopKernel>([]() {});
        });
        return event;
    }
}

Event Queue::create_event_after_external_work() {
    // Always create a new barrier event, never use the cached last_event_.
    // This ensures the returned event properly depends on external library calls
    // (cuBLAS, rocBLAS, etc.) that execute on the stream but don't update last_event_.
    try {
        sycl::event e = impl_->ext_oneapi_submit_barrier();
        impl_->last_event_ = e;
        EventImpl event = std::move(e);
        return event;
    } catch (const sycl::exception&) {
        EventImpl event = impl_->submit([&](sycl::handler& h) {
            h.single_task<QueueExternalWorkBarrierKernel>([]() {});
        });
        return event;
    }
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

Vendor Device::get_vendor() const {
    return str_to_vendor(QueueImpl::device_arrays.at(static_cast<int>(type)).at(idx).get_info<sycl::info::device::vendor>());
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
        case 6: return d.get_info<sycl::info::device::max_num_sub_groups>();
        case 7: return d.get_info<sycl::info::device::sub_group_sizes>()[0];
        case 8: return d.get_info<sycl::info::device::mem_base_addr_align>();
        case 9: return d.get_info<sycl::info::device::global_mem_cache_line_size>();
        case 10: return d.get_info<sycl::info::device::global_mem_cache_size>();
        default: std::cerr << "Unknown property" << std::endl; return 0;
    }
}

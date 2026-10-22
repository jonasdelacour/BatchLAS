// SPDX-License-Identifier: MIT
#include <sycl/sycl.hpp>
#include <numeric>
#include <unordered_map>
#include <exception>
#include <limits>
#include <cstdint>
#include <complex>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <src/queue.hh>
#include <batchlas/backend_config.h>

// SYCL 2020 leaves mem_advise's advice values implementation-defined; the
// UR_USM_ADVICE_FLAG_* ones are DPC++'s Unified Runtime. Advice is only a hint
// and these wrappers already return an empty Event on failure, so elsewhere
// they are no-ops.
#if BATCHLAS_SYCL_IMPL_DPCPP
    #define BATCHLAS_MEM_ADVISE(q, ptr, bytes, flag) static_cast<EventImpl>((q)->mem_advise((ptr), (bytes), (flag)))
#else
    #define BATCHLAS_MEM_ADVISE(q, ptr, bytes, flag) ((void)(q), (void)(ptr), (void)(bytes), Event{})
#endif

using namespace sycl;

// MUST be namespace batchlas to the end of file: at global scope the friend operator<< would
// SILENTLY define unrelated functions (consumer link error only). The std::array operator<< is
// file-local (no ADL). evidence: docs/design/runtime-internals.md#runtime-internals-namespace-placement-of-out-of-line-definitions
namespace batchlas {

template <typename U, size_t N>
constexpr std::ostream& operator<<(std::ostream& os, const std::array<U,N>& arr) {
    os << "[";
    for (size_t i = 0; i < N; i++) {
        os << arr[i];
        if(i < N - 1) os << ", ";
    }
    os << "]";
    return os;
}

template <typename T>
T* UnifiedVector<T>::alloc_shared(size_t n) {
    Device dev = Device::default_device();
    const auto& ctx = QueueImpl::shared_context(dev);
    const sycl::device sycl_dev = QueueImpl::device_arrays.at((int)dev.type).at(dev.idx);
    T* p = sycl::malloc_shared<T>(n, sycl_dev, ctx);
    if (!p && n > 0) throw std::bad_alloc();
    return p;
}

template <typename T>
UnifiedVector<T>::UnifiedVector(size_t size) : size_(size), capacity_(size), data_(alloc_shared(size)) {}

template <typename T>
UnifiedVector<T>::UnifiedVector(size_t size, T value) : UnifiedVector<T>(size) {
    for(size_t i = 0; i < size; i++) data_[i] = value;
}

template <typename T>
UnifiedVector<T>::UnifiedVector(const UnifiedVector<T>& other) : UnifiedVector<T>(other.size_) {
    for(size_t i = 0; i < size_; i++) data_[i] = other.data_[i];
}

template <typename T>
UnifiedVector<T>::~UnifiedVector() {
    if (!data_) return;
    Device dev = Device::default_device();
    const auto& ctx = QueueImpl::shared_context(dev);
    try {
        sycl::free(data_, ctx);
    } catch (...) {
        // Destructors must not throw; runtime may report prior async device failures here.
    }
}

template <typename T>
void UnifiedVector<T>::resize(size_t new_size) {
    if(new_size > capacity_){
        T* new_data = alloc_shared(new_size);
        const auto& ctx = QueueImpl::shared_context(Device::default_device());
        memcpy(new_data, data_, size_*sizeof(T));
        if(data_) sycl::free(data_, ctx);
        data_ = new_data;
        capacity_ = new_size;
    }
    size_ = new_size;
}

template <typename T>
Event Span<T>::set_preferred_location(const Queue &ctx) const {
    if (!data_ || this->size_bytes() == 0) return Event{};
    try {
        return BATCHLAS_MEM_ADVISE(ctx, data_, this->size_bytes(), UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION);
    } catch (const sycl::exception&) {
        return Event{};
    } catch (...) {
        return Event{};
    }
}

template <typename T>
Event Span<T>::set_access_device(const Queue &ctx) const {
    if (!data_ || this->size_bytes() == 0) return Event{};
    try {
        return BATCHLAS_MEM_ADVISE(ctx, data_, this->size_bytes(), UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE);
    } catch (const sycl::exception&) {
        return Event{};
    } catch (...) {
        return Event{};
    }
}

template <typename T>
Event Span<T>::prefetch(const Queue &ctx) const {
    if (!data_ || this->size_bytes() == 0) return Event{};
    try {
        return static_cast<EventImpl>(ctx -> prefetch(data_, this->size_bytes()));
    } catch (const sycl::exception&) {
        return Event{};
    } catch (...) {
        return Event{};
    }
}

template <typename T>
void UnifiedVector<T>::resize(size_t new_size, T val){
    if(new_size > capacity_){
        T* new_data = alloc_shared(new_size);
        const auto& ctx = QueueImpl::shared_context(Device::default_device());
        memcpy(new_data, data_, size_*sizeof(T));
        if(data_) sycl::free(data_, ctx);
        data_ = new_data;
        capacity_ = new_size;
        std::generate(data_ + size_ , data_ + new_size, [val](){return val;});
        size_ = new_size;
    }
}

template <typename T>
void UnifiedVector<T>::resize(size_t new_size, size_t front, size_t back, size_t seg_size){
    if(new_size > capacity_){
        T* new_data = alloc_shared(new_size);
        const auto& ctx = QueueImpl::shared_context(Device::default_device());
        std::fill_n(new_data, new_size, T{});
        if (capacity_ > 0){
            auto n_first_segment = back < front ? capacity_ - front : (back - front + seg_size);
            assert(n_first_segment <= new_size);
            assert((n_first_segment + front) <= new_size);
            if(back < front) assert(back + seg_size <= front);
            memcpy(new_data, data_ + front, n_first_segment*sizeof(T));
            if( back < front) memcpy(new_data + n_first_segment, data_, (back + seg_size)*sizeof(T));
        }
        if(data_) sycl::free(data_, ctx);
        data_ = new_data;
        size_ = new_size;
        capacity_ = new_size;
    }
}

template <typename T>
void UnifiedVector<T>::reserve(size_t new_capacity) {
    if(new_capacity > capacity_){
        T* new_data = alloc_shared(new_capacity);
        const auto& ctx = QueueImpl::shared_context(Device::default_device());
        memcpy(new_data, data_, size_*sizeof(T));
        if(data_) sycl::free(data_, ctx);
        data_ = new_data;
        capacity_ = new_capacity;
    }
}

template <typename T>
UnifiedVector<T>& UnifiedVector<T>::operator=(const UnifiedVector<T>& other) {
    // Reallocate only when the capacity is smaller; allocate before freeing, so a failed
    // allocation leaves this vector unchanged.
    if (capacity_ < other.capacity_) {
        T* new_data = alloc_shared(other.capacity_);
        const auto& ctx = QueueImpl::shared_context(Device::default_device());
        if(data_) sycl::free(data_, ctx);
        data_ = new_data;
        capacity_ = other.capacity_;
    }
    size_ = other.size_;
    //If the type is trivially copyable, use the sycl memcpy function
    if constexpr(std::is_trivially_copyable_v<T>){
        Device dev = Device::default_device();
        const auto& ctx = QueueImpl::shared_context(dev);
        const sycl::device sycl_dev = QueueImpl::device_arrays.at((int)dev.type).at(dev.idx);
        sycl::queue q{ctx, sycl_dev};
        q.memcpy(data_, other.data_, size_*sizeof(T)).wait();
    } else {
        for(size_t i = 0; i < size_; i++) data_[i] = other.data_[i];
    }
    return *this;
}

template <typename U>
std::ostream& operator<<(std::ostream& os, const UnifiedVector<U>& vec) {
    os << (Span<U>)vec;
    return os;
}

template <typename U>
std::ostream& operator<<(std::ostream& os, const Span<U>& vec) {
    os << "[" ;
     for (size_t i = 0; i < vec.size(); i++) {
        os << vec[i];
        if(i < vec.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

template <typename T>
bool Span<T>::operator==(const Span<T> other) const {
    if(size_ != other.size_) return false;
    if(data() == other.data()) return true;
    if constexpr (std::is_floating_point_v<std::decay_t<T>>){
        return std::equal(begin(), end(), other.begin(), [](auto& a,auto& b){
            T eps = std::numeric_limits<T>::epsilon() * 20;
            T max_v = std::max(std::abs(a), std::abs(b));
            return std::abs(a - b) / (max_v > eps ? max_v : 1) < eps;});
    } else{
        return std::equal(begin(), end(), other.begin());
    }
    return true;
}



template struct UnifiedVector<int8_t>;
template struct UnifiedVector<int16_t>;
template struct UnifiedVector<int32_t>;
template struct UnifiedVector<int64_t>;
template struct UnifiedVector<uint8_t>;
template struct UnifiedVector<uint16_t>;
template struct UnifiedVector<uint32_t>;
template struct UnifiedVector<uint64_t>;
template struct UnifiedVector<float>;
template struct UnifiedVector<double>;
template struct UnifiedVector<std::complex<float>>;
template struct UnifiedVector<std::complex<double>>;
// BATCHLAS_API repeated HERE only: std::byte is hidden, and an instantiation takes the MINIMUM
// visibility of template and arguments. Removing it silently un-exports this specialisation.
// evidence: docs/design/runtime-internals.md#runtime-internals-symbol-visibility-for-private-headers
template struct BATCHLAS_API UnifiedVector<std::byte>;
template struct UnifiedVector<bool>;
template struct UnifiedVector<std::array<double,2>>;
template struct UnifiedVector<std::array<double,3>>;
template struct UnifiedVector<std::array<float,2>>;
template struct UnifiedVector<std::array<float,3>>;
template struct UnifiedVector<std::array<int32_t,2>>;
template struct UnifiedVector<std::array<int32_t,3>>;
template struct UnifiedVector<std::array<uint16_t,3>>;
template struct UnifiedVector<std::array<uint32_t,3>>;
template struct UnifiedVector<std::array<uint16_t,6>>;
template struct UnifiedVector<std::array<uint32_t,6>>;
template struct UnifiedVector<std::bitset<21>>;
template struct UnifiedVector<std::bitset<3>>;

template struct UnifiedVector<float*>;
template struct UnifiedVector<double*>;
template struct UnifiedVector<std::complex<float>*>;
template struct UnifiedVector<std::complex<double>*>;
template struct UnifiedVector<int*>;
template struct UnifiedVector<size_t*>;
template struct UnifiedVector<std::array<double,2>*>;
template struct UnifiedVector<std::array<double,3>*>;
template struct UnifiedVector<std::array<float,2>*>;
template struct UnifiedVector<std::array<float,3>*>;

template struct Span<int8_t>;
template struct Span<int16_t>;
template struct Span<int32_t>;
template struct Span<int64_t>;
template struct Span<uint8_t>;
template struct Span<uint16_t>;
template struct Span<uint32_t>;
template struct Span<uint64_t>;
template struct Span<float>;
template struct Span<double>;
template struct Span<std::complex<float>>;
template struct Span<std::complex<double>>;
template struct Span<std::byte>;
template struct Span<bool>;
template struct Span<std::array<double,2>>;
template struct Span<std::array<double,3>>;
template struct Span<std::array<float,2>>;
template struct Span<std::array<float,3>>;
template struct Span<std::array<int32_t,2>>;
template struct Span<std::array<int32_t,3>>;
template struct Span<std::array<uint16_t,3>>;
template struct Span<std::array<uint32_t,3>>;
template struct Span<std::array<uint16_t,6>>;
template struct Span<std::array<uint32_t,6>>;
template struct Span<std::bitset<21>>;
template struct Span<std::bitset<3>>;

template struct Span<float*>;
template struct Span<double*>;

template std::ostream& operator<<(std::ostream& os, const UnifiedVector<float>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<double>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::complex<float>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::complex<double>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<uint16_t>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<uint32_t>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<int>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<size_t>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<bool>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<double,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<double,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<float,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<float,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<int32_t,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<int32_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<uint16_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<uint32_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<uint16_t,6>>& vec);
template std::ostream& operator<<(std::ostream& os, const UnifiedVector<std::array<uint32_t,6>>& vec);


template std::ostream& operator<<(std::ostream& os, const Span<float>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<double>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::complex<float>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::complex<double>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<uint16_t>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<uint32_t>& vec);

template std::ostream& operator<<(std::ostream& os, const Span<int>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<size_t>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<bool>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<double,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<double,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<float,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<float,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<int32_t,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<int32_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint16_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint32_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint16_t,6>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint32_t,6>>& vec);


}  // namespace batchlas

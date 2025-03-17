#include <sycl/sycl.hpp>
#include <numeric>
#include <unordered_map>
#include <exception>
#include <limits>
#include <cstdint>
#include <complex>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include <util/reference-wrapper.hh>

#ifndef DEVICE_CAST
    #define DEVICE_CAST(x,ix) (reinterpret_cast<const sycl::device*>(x)[ix])
#endif

using namespace sycl;

template <typename U>
constexpr std::ostream& operator<<(std::ostream& os, const ReferenceWrapper<U>& ref) {
    os << ref.get();
    return os;
}

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
SyclVector<T>::SyclVector(size_t size) : size_(size), capacity_(size) {
    data_ = sycl::malloc_shared<T>(size, sycl::device(default_selector_v), sycl::context(device(default_selector_v)));
    if (!data_ && size > 0) {
        std::cout << "Could not allocate " + std::to_string(size) + " elements of type " + typeid(T).name() << std::endl;
        throw std::bad_alloc();
    } 
}

template <typename T>
SyclVector<T>::SyclVector(size_t size, T value) : SyclVector<T>(size) {
    for(size_t i = 0; i < size; i++) data_[i] = value;
}

template <typename T>
SyclVector<T>::SyclVector(const SyclVector<T>& other) : SyclVector<T>(other.size_) {
    for(size_t i = 0; i < size_; i++) data_[i] = other.data_[i];
}

template <typename T>
SyclVector<T>::~SyclVector() {if(data_) sycl::free(data_,  sycl::context(device(default_selector_v)));}

template <typename T>
void SyclVector<T>::resize(size_t new_size) {
    if(new_size > capacity_){
        T* new_data = sycl::malloc_shared<T>(new_size, sycl::device(default_selector_v), sycl::context(device(default_selector_v)));
        memcpy(new_data, data_, size_*sizeof(T));
        if(data_) sycl::free(data_,  sycl::context(device(default_selector_v)));
        data_ = new_data;
        capacity_ = new_size;
    }
    size_ = new_size;
}

template <typename T>
void SyclVector<T>::resize(size_t new_size, T val){
    if(new_size > capacity_){
        T* new_data = sycl::malloc_shared<T>(new_size, sycl::device(default_selector_v), sycl::context(device(default_selector_v)));
        memcpy(new_data, data_, size_*sizeof(T));
        if(data_) sycl::free(data_,  sycl::context(device(default_selector_v)));
        data_ = new_data;
        capacity_ = new_size;
        std::generate(data_ + size_ , data_ + new_size, [val](){return val;});
        size_ = new_size;
    }
}

template <typename T>
void SyclVector<T>::resize(size_t new_size, size_t front, size_t back, size_t seg_size){
    if(new_size > capacity_){
        T* new_data = sycl::malloc_shared<T>(new_size, sycl::device(default_selector_v), sycl::context(device(default_selector_v)));
        std::fill_n(new_data, new_size, T{});
        if (capacity_ > 0){
            auto n_first_segment = back < front ? capacity_ - front : (back - front + seg_size);
            assert(n_first_segment <= new_size);
            assert((n_first_segment + front) <= new_size);
            if(back < front) assert(back + seg_size <= front);
            memcpy(new_data, data_ + front, n_first_segment*sizeof(T));
            if( back < front) memcpy(new_data + n_first_segment, data_, (back + seg_size)*sizeof(T));
        }
        if(data_) sycl::free(data_,  sycl::context(device(default_selector_v)));
        data_ = new_data;
        size_ = new_size;
        capacity_ = new_size;
    }
}

template <typename T>
void SyclVector<T>::reserve(size_t new_capacity) {
    if(new_capacity > capacity_){
        T* new_data = sycl::malloc_shared<T>(new_capacity, sycl::device(default_selector_v), sycl::context(device(default_selector_v)));
        memcpy(new_data, data_, size_*sizeof(T));
        if(data_) sycl::free(data_,  sycl::context(device(default_selector_v)));
        data_ = new_data;
        capacity_ = new_capacity;
    }
}

template <typename T>
SyclVector<T>& SyclVector<T>::operator=(const SyclVector<T>& other) {
    size_ = other.size_;
    //Only perform memory allocation if the new size is greater than the current capacity
    if (capacity_ < other.capacity_) {
        capacity_ = other.capacity_;
        if(data_) sycl::free(data_,  sycl::context(device(default_selector_v)));
        data_ = sycl::malloc_shared<T>(capacity_, sycl::device(default_selector_v), sycl::context(device(default_selector_v)));
    }
    //If the type is trivially copyable, use the sycl memcpy function
    if constexpr(std::is_trivially_copyable_v<T>){
        sycl::queue q{default_selector_v};
        q.memcpy(data_, other.data_, size_*sizeof(T)).wait();
    } else {
        for(size_t i = 0; i < size_; i++) data_[i] = other.data_[i];
    }
    return *this;
}

template <typename U>
std::ostream& operator<<(std::ostream& os, const SyclVector<U>& vec) {
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


template <size_t N, typename... Args, std::size_t... I>
void resize_all(std::array<int,N>&& sizes, std::tuple<Args...>&& args, std::index_sequence<I...>){
    (std::get<I>(args).resize(sizes[I], typename std::decay_t<decltype(std::get<I>(args))>::value_type{}), ...);
}


template struct SyclVector<int8_t>;
template struct SyclVector<int16_t>;
template struct SyclVector<int32_t>;
template struct SyclVector<int64_t>;
template struct SyclVector<uint8_t>;
template struct SyclVector<uint16_t>;
template struct SyclVector<uint32_t>;
template struct SyclVector<uint64_t>;
template struct SyclVector<float>;
template struct SyclVector<double>;
template struct SyclVector<std::complex<float>>;
template struct SyclVector<std::complex<double>>;
template struct SyclVector<std::byte>;
template struct SyclVector<bool>;
template struct SyclVector<std::array<double,2>>;
template struct SyclVector<std::array<double,3>>;
template struct SyclVector<std::array<float,2>>;
template struct SyclVector<std::array<float,3>>;
template struct SyclVector<std::array<uint16_t,3>>;
template struct SyclVector<std::array<uint32_t,3>>;
template struct SyclVector<std::array<uint16_t,6>>;
template struct SyclVector<std::array<uint32_t,6>>;
template struct SyclVector<std::bitset<21>>;
template struct SyclVector<std::bitset<3>>;

template struct SyclVector<float*>;
template struct SyclVector<double*>;
template struct SyclVector<std::complex<float>*>;
template struct SyclVector<std::complex<double>*>;
template struct SyclVector<int*>;
template struct SyclVector<size_t*>;
//template struct SyclVector<NodeNeighbours<uint16_t>>;
//template struct SyclVector<NodeNeighbours<uint32_t>>;
//template struct SyclVector<Constants<float,uint16_t>>;

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
template struct Span<std::byte>;
template struct Span<bool>;
template struct Span<std::array<double,2>>;
template struct Span<std::array<double,3>>;
template struct Span<std::array<float,2>>;
template struct Span<std::array<float,3>>;
template struct Span<std::array<uint16_t,3>>;
template struct Span<std::array<uint32_t,3>>;
template struct Span<std::array<uint16_t,6>>;
template struct Span<std::array<uint32_t,6>>;
template struct Span<std::bitset<21>>;
template struct Span<std::bitset<3>>;

template struct Span<float*>;
template struct Span<double*>;

template std::ostream& operator<<(std::ostream& os, const SyclVector<float>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<double>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<uint16_t>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<uint32_t>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<int>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<size_t>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<bool>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<double,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<double,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<float,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<float,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<uint16_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<uint32_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<uint16_t,6>>& vec);
template std::ostream& operator<<(std::ostream& os, const SyclVector<std::array<uint32_t,6>>& vec);


template std::ostream& operator<<(std::ostream& os, const Span<float>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<double>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<uint16_t>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<uint32_t>& vec);

template std::ostream& operator<<(std::ostream& os, const Span<int>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<size_t>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<bool>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<double,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<double,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<float,2>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<float,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint16_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint32_t,3>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint16_t,6>>& vec);
template std::ostream& operator<<(std::ostream& os, const Span<std::array<uint32_t,6>>& vec);


template std::ostream& operator<<(std::ostream& os, const ReferenceWrapper<int>& ref);
template std::ostream& operator<<(std::ostream& os, const ReferenceWrapper<size_t>& ref);
template std::ostream& operator<<(std::ostream& os, const ReferenceWrapper<float>& ref);
template std::ostream& operator<<(std::ostream& os, const ReferenceWrapper<double>& ref);
template std::ostream& operator<<(std::ostream& os, const ReferenceWrapper<uint16_t>& ref);
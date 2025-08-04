#pragma once
#include <cassert>
#include <util/sycl-span.hh>
#include <util/sycl-device-queue.hh>
template <typename T>
struct UnifiedVector
{   
    using value_type = T;
    using pointer = T*;
    using size_t = std::size_t;
    //Constructors implementations depend on sycl, so they are not defined here
    UnifiedVector(size_t size);
    UnifiedVector(size_t size, T value);
    UnifiedVector(const UnifiedVector<T> &other);
    UnifiedVector<T> &operator=(const UnifiedVector<T> &other);
    ~UnifiedVector();

    void resize(size_t new_size);
    void resize(size_t new_size, T value);
    void resize(size_t new_size, size_t front, size_t back, size_t seg_size);

    void reserve(size_t new_capacity);

    //Movement semantics can be defined here
    UnifiedVector() : size_(0), capacity_(0), data_(nullptr) {}
    UnifiedVector(UnifiedVector<T> &&other) : size_(other.size_), capacity_(other.capacity_), data_(other.data_) {
        other.size_ = 0;
        other.capacity_ = 0;
        other.data_ = nullptr;
    }
    UnifiedVector<T> &operator=(UnifiedVector<T> &&other) {
        if (this == &other) return *this;
        this->data_ = other.data_;
        this->size_ = other.size_;
        this->capacity_ = other.capacity_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        return *this;
    }

    inline constexpr operator Span<T>() const { return Span<T>(data_, size_); }
    inline constexpr Span<T> to_span() const { return Span<T>(data_, size_); }
    inline constexpr Span<T> subspan(size_t offset, size_t count) const { return Span<T>(data_ + offset, count); }
    inline constexpr Span<T> subspan(size_t offset) const { return Span<T>(data_ + offset, size_ - offset); }
    inline constexpr void fill(T data) { std::fill(begin(), end(), data); }
    inline constexpr T *data() const { return data_; }
    inline constexpr size_t size() const { return size_; }
    inline constexpr size_t capacity() const { return capacity_; }
    
    
    inline constexpr void clear() { size_ = 0; }

    inline constexpr T &operator[](size_t index) { if(index >= size_) printf("Index: %zu, Size: %zu\n", index, size_); assert (index < size_); return data_[index]; }
    inline constexpr const T &operator[](size_t index) const { if (index >= size_) printf("Index: %zu, Size: %zu\n", index, size_); assert (index < size_); return data_[index]; }

    inline constexpr T &at(size_t index) { assert(index < size_); return data_[index]; }
    inline constexpr const T &at(size_t index) const { assert(index < size_); return data_[index]; }

    inline constexpr bool operator==(const UnifiedVector<T> &other) const {
        return Span<T>(*this) == Span<T>(other);
    }

    inline constexpr void push_back(const T &value) { 
        if(size_ == capacity_){
            size_t new_capacity = capacity_ == 0 ? 1 : 2*capacity_;
            reserve(new_capacity);
        }
        data_[size_++] = value;
    }

    inline constexpr void push_back(T &&value) { 
        if(size_ == capacity_){
            size_t new_capacity = capacity_ == 0 ? 1 : 2*capacity_;
            reserve(new_capacity);
        }
        data_[size_++] = std::move(value);
    }


    inline constexpr T pop_back() { assert(size_ > 0); return data_[--size_]; }

    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const UnifiedVector<U> &vec);

    inline constexpr T *begin() const { return data_; }
    inline constexpr T *end() const { return data_ + size_; }

    inline constexpr T &back() const { return data_[size_ - 1]; }
    inline constexpr T &front() const { return data_[0]; }

    inline constexpr void swap(UnifiedVector<T> &other) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }
private:
    size_t size_;
    size_t capacity_;
    pointer data_;
};

template <typename T>
inline constexpr void swap(UnifiedVector<T> &lhs, UnifiedVector<T> &rhs) {
    lhs.swap(rhs);
}
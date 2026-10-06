#pragma once
#include <cassert>
#include <batchlas/export.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-device-queue.hh>

/// @file
/// @brief batchlas::UnifiedVector, an owning growable array in USM shared memory.

namespace batchlas {

/// @brief Owning, growable array of `T` in USM shared memory (`sycl::malloc_shared`).
///
/// The storage is allocated in the shared context of Device::default_device(),
/// so it is readable and writable from the host and from that device. Copying
/// deep-copies on the host; moving transfers ownership. Converts implicitly to
/// Span<T>, which is how it is passed to entry points. Growth reallocates, so
/// any Span or pointer taken earlier is invalidated.
/// @tparam T  element type (trivially copyable: growth uses memcpy)
/// @ingroup matrix
template <typename T>
struct BATCHLAS_API UnifiedVector
{
    using value_type = T;        ///< Element type.
    using pointer = T*;          ///< Pointer to an element.
    using size_t = std::size_t;  ///< Size and index type.
    /// @brief Allocates @p size uninitialised elements.
    /// @throws std::bad_alloc if the allocation fails
    UnifiedVector(size_t size);
    /// @brief Allocates @p size elements set to @p value (filled on the host).
    UnifiedVector(size_t size, T value);
    /// @brief Deep copy of @p other's first `size()` elements into a new allocation.
    UnifiedVector(const UnifiedVector<T> &other);
    /// @brief Deep copy of @p other's elements; reallocates only when this capacity is smaller.
    ///
    /// Trivially copyable `T` is copied with a device memcpy that is waited on.
    UnifiedVector<T> &operator=(const UnifiedVector<T> &other);
    /// @brief Frees the allocation; never throws, even if the runtime reports an earlier async error.
    ~UnifiedVector();

    /// @brief Sets the size; reallocates (keeping the contents) only when @p new_size exceeds the capacity.
    void resize(size_t new_size);
    /// @brief Grows to @p new_size, setting the new elements to @p value.
    /// @note A no-op unless @p new_size exceeds the capacity: it neither shrinks nor
    ///       grows within the existing capacity (unlike the one-argument resize()).
    void resize(size_t new_size, T value);
    /// @brief Grows a ring buffer to @p new_size, unrolling its live segment to the front.
    ///
    /// The live data starts at @p front and ends with the @p seg_size elements at
    /// @p back, wrapping when back < front. After the call it is contiguous from
    /// index 0 and the rest is value-initialised. No-op unless @p new_size exceeds
    /// the capacity.
    void resize(size_t new_size, size_t front, size_t back, size_t seg_size);

    /// @brief Ensures a capacity of at least @p new_capacity, keeping the contents and the size.
    void reserve(size_t new_capacity);

    /// @brief An empty vector with no allocation.
    UnifiedVector() : size_(0), capacity_(0), data_(nullptr) {}
    /// @brief Takes @p other's allocation and leaves @p other empty.
    UnifiedVector(UnifiedVector<T> &&other) : size_(other.size_), capacity_(other.capacity_), data_(other.data_) {
        other.size_ = 0;
        other.capacity_ = 0;
        other.data_ = nullptr;
    }
    /// @brief Takes @p other's allocation and leaves @p other empty.
    /// @trap This vector's previous allocation is not freed (a leak; see @ref matrix-model-open-debts).
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

    /// @brief A Span over the first `size()` elements.
    inline constexpr operator Span<T>() const { return Span<T>(data_, size_); }
    /// @brief A Span over the first `size()` elements.
    inline constexpr Span<T> to_span() const { return Span<T>(data_, size_); }
    /// @brief A Span over elements [@p offset, @p offset + @p count); not bounds-checked.
    inline constexpr Span<T> subspan(size_t offset, size_t count) const { return Span<T>(data_ + offset, count); }
    /// @brief A Span over elements [@p offset, size()).
    inline constexpr Span<T> subspan(size_t offset) const { return Span<T>(data_ + offset, size_ - offset); }
    /// @brief Sets every element to @p data, on the host.
    inline constexpr void fill(T data) { std::fill(begin(), end(), data); }
    inline constexpr T *data() const { return data_; }               ///< Pointer to the first element (null when never allocated).
    inline constexpr size_t size() const { return size_; }           ///< Number of elements.
    inline constexpr size_t capacity() const { return capacity_; }   ///< Allocated elements.


    /// @brief Sets the size to 0 and keeps the allocation.
    inline constexpr void clear() { size_ = 0; }

    /// @brief Element @p index; prints and asserts (debug build) when out of range.
    inline constexpr T &operator[](size_t index) { if(index >= size_) printf("Index: %zu, Size: %zu\n", index, size_); assert (index < size_); return data_[index]; }
    /// @brief Element @p index (const); same checks as the non-const overload.
    inline constexpr const T &operator[](size_t index) const { if (index >= size_) printf("Index: %zu, Size: %zu\n", index, size_); assert (index < size_); return data_[index]; }

    /// @brief Element @p index. @pre index < size() (asserted in debug builds, not thrown)
    inline constexpr T &at(size_t index) { assert(index < size_); return data_[index]; }
    /// @brief Element @p index (const). @pre index < size()
    inline constexpr const T &at(size_t index) const { assert(index < size_); return data_[index]; }

    /// @brief Element-wise comparison with Span::operator== semantics.
    inline constexpr bool operator==(const UnifiedVector<T> &other) const {
        return Span<T>(*this) == Span<T>(other);
    }

    /// @brief Appends @p value, doubling the capacity when full (which invalidates earlier Spans).
    inline constexpr void push_back(const T &value) {
        if(size_ == capacity_){
            size_t new_capacity = capacity_ == 0 ? 1 : 2*capacity_;
            reserve(new_capacity);
        }
        data_[size_++] = value;
    }

    /// @brief Appends @p value by move; growth as for the copying overload.
    inline constexpr void push_back(T &&value) {
        if(size_ == capacity_){
            size_t new_capacity = capacity_ == 0 ? 1 : 2*capacity_;
            reserve(new_capacity);
        }
        data_[size_++] = std::move(value);
    }


    /// @brief Removes and returns the last element; keeps the capacity. @pre size() > 0
    inline constexpr T pop_back() { assert(size_ > 0); return data_[--size_]; }

    /// @brief Prints the elements on the host as `[a, b, c]`, like Span's operator<<.
    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const UnifiedVector<U> &vec);

    inline constexpr T *begin() const { return data_; }            ///< Iterator to the first element.
    inline constexpr T *end() const { return data_ + size_; }      ///< Iterator one past the last element.

    inline constexpr T &back() const { return data_[size_ - 1]; }  ///< Last element. @pre size() > 0
    inline constexpr T &front() const { return data_[0]; }         ///< First element. @pre size() > 0

    /// @brief Exchanges storage, size and capacity with @p other; no element is copied.
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

/// @brief Swaps the storage of two vectors without copying; found by ADL.
/// @ingroup matrix
template <typename T>
inline constexpr void swap(UnifiedVector<T> &lhs, UnifiedVector<T> &rhs) {
    lhs.swap(rhs);
}

}  // namespace batchlas

// Transitional shim: UnifiedVector used to live at global scope; define
// BATCHLAS_NO_GLOBAL_NAMES to drop it. swap() is deliberately not shimmed: ADL
// finds it, and a global `swap` is exactly the collision the move removed.
#ifndef BATCHLAS_NO_GLOBAL_NAMES
using batchlas::UnifiedVector;
#endif

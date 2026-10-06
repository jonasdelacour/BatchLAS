#pragma once
#include <cassert>
#include <iterator>
#include <array>
#include <type_traits>
#include <batchlas/export.hh>
#include <batchlas/util/sycl-device-queue.hh>

/// @file
/// @brief batchlas::Span, the non-owning pointer + length every entry point takes for 1-D data.

namespace batchlas {

/// @brief True for `std::array<T, N>`, false otherwise.
/// @ingroup matrix
template <typename T>
struct is_std_array : std::false_type {};

/// @brief Specialisation for `std::array<T, N>`.
/// @ingroup matrix
template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

/// @brief Non-owning view of `size()` contiguous elements of type `T`.
///
/// The 1-D argument type of the entry points: eigenvalues, singular values,
/// pivots, tau vectors, info arrays and caller-supplied workspace. A Span does
/// not own or free its memory, and copying one copies the pointer. The memory
/// must be USM that the device of the Queue it is used with can access; the
/// entry points check this at the call.
///
/// Element access asserts the bounds in a debug build only.
/// @tparam T  element type
/// @ingroup matrix
// The out-of-line members (USM advice calls, operator==) are explicitly instantiated in
// src/util/sycl-util-impl.cc, so Span crosses the shared-library boundary like Matrix.
template <typename T>
struct BATCHLAS_API Span
{
    using value_type = T;
    using pointer = T*;
    using size_t = std::size_t;
    /// @brief An empty span (null data, size 0).
    inline constexpr Span() : data_(nullptr), size_(0) {}
    /// @brief Views @p size elements starting at @p data.
    inline constexpr Span(T *data, size_t size) : data_(data), size_(size) {}
    /// @brief Views the range [@p begin, @p end).
    inline constexpr Span(T *begin, T *end) : data_(begin), size_(std::distance(begin, end)) {}
    inline constexpr Span(const Span<T> &other) = default;
    inline constexpr Span(Span<T> &&other) = default;
    /// @brief Views @p other with its first @p offset elements dropped.
    /// @pre offset <= other.size()
    inline constexpr Span(const Span<T>& other, size_t offset) : data_(other.data_ + offset), size_(other.size_ - offset) {}
    /// @brief Views @p other with its first @p offset elements dropped.
    /// @pre offset <= other.size()
    inline constexpr Span(Span<T>&& other, size_t offset) : data_(other.data_ + offset), size_(other.size_ - offset) {}
    /// @brief A one-element span over @p value.
    ///
    /// Explicit on purpose: implicitly, a scalar lvalue became a length-1 buffer
    /// wherever a Span parameter's element type is not deduced from the argument,
    /// so `syev(q, A, W[i * n], ...)` (the LAPACK idiom minus the `&`) compiled
    /// and overran the caller's array. Write `Span(x)` when one element is meant.
    inline constexpr explicit Span(T& value) : data_(&value), size_(1) {}

    /// @name USM memory hints
    /// Advice for USM shared memory on the device of @p ctx. Each returns the event of
    /// the advice, or an empty Event when the span is empty or the runtime rejects
    /// the advice (errors are swallowed: these are hints). Pass the Queue the data
    /// will be used on; the default Queue() is a new queue on Device::default_device().
    /// @{
    Event set_read_mostly(const Queue &ctx = Queue()) const;           ///< Mark read-mostly (replicated on read).
    Event unset_read_mostly(const Queue &ctx = Queue()) const;         ///< Clear read-mostly.
    Event set_preferred_location(const Queue &ctx = Queue()) const;    ///< Prefer residence on the device of @p ctx.
    Event clear_preferred_location(const Queue &ctx = Queue()) const;  ///< Clear the preferred location.
    Event set_access_device(const Queue &ctx = Queue()) const;         ///< Map for access by the device of @p ctx.
    Event clear_access_device(const Queue &ctx = Queue()) const;       ///< Clear the access mapping.
    Event prefetch(const Queue &ctx = Queue()) const;                  ///< Migrate to the device of @p ctx.
    /// @}

    /// @brief Reinterprets the bytes as elements of type @p U; the size is `size_bytes() / sizeof(U)`.
    template <typename U>
    inline constexpr Span<U> as_span() const {
        return Span<U>(reinterpret_cast<U*>(data_), (sizeof(T) * size_ / sizeof(U)) );
    }

    /// @brief Elements [@p offset, size()).
    inline constexpr Span<T> subspan(size_t offset) const { assert(size_ - offset >= 0); return Span<T>(data_ + offset, size_ - offset); }
    /// @brief Elements [@p offset, @p offset + @p count). @pre offset + count <= size()
    inline constexpr Span<T> subspan(size_t offset, size_t count) const { assert(offset + count <= size_);  return Span<T>(data_ + offset, count); }
    inline constexpr Span<T>& operator= (const Span<T> &other) { data_ = other.data_; size_ = other.size_; return *this; }
    inline constexpr Span<T>& operator= (Span<T> &&other) { return *this = other; }
    /// @brief Element-wise comparison on the host.
    ///
    /// Equal sizes are required. Floating-point elements compare with a relative
    /// tolerance of 20 epsilon; other types compare exactly. Two spans over the same
    /// pointer are equal without reading the data.
    bool operator==(const Span<T> other) const;
    /// @brief Element @p index. @pre index < size()
    inline constexpr T &operator[](size_t index) const {assert(index < size_); assert(data_); return data_[index];}
    /// @brief Element @p index. @pre index < size() (asserted, not thrown)
    inline constexpr T &at(size_t index) const{assert(index < size_); assert(data_); return data_[index];}
    inline constexpr T *data() const { return data_; }
    inline constexpr size_t size() const { return size_; }
    inline constexpr bool empty() const { return size_ == 0; }
    /// @brief `size() * sizeof(T)`.
    inline constexpr size_t size_bytes() const { return size_ * sizeof(T); }
    inline constexpr T *begin() const { return data_; }
    inline constexpr T *end() const { return data_ + size_; }
    inline constexpr T &front() const { return data_[0]; }
    inline constexpr T &back() const { return data_[size_ - 1]; }
    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const Span<U> &vec);

private:
    T *data_;
    size_t size_;
};

template <typename T>
Span(T*, typename Span<T>::size_t) -> Span<T>;

template <typename T>
Span(T*, std::size_t) -> Span<T>;

template <typename T>
Span(T*, T*) -> Span<T>;

template <typename T>
Span(T&) -> Span<T>;

}  // namespace batchlas

// Transitional shim: Span and is_std_array used to live at global scope. Define
// BATCHLAS_NO_GLOBAL_NAMES to drop these; CTAD still works because the deduction
// guides are found in Span's own namespace.
#ifndef BATCHLAS_NO_GLOBAL_NAMES
using batchlas::Span;
using batchlas::is_std_array;
#endif


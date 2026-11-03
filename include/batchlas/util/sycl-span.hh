// SPDX-License-Identifier: MIT
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
/// @ingroup api_matrix
template <typename T>
struct is_std_array : std::false_type {};

/// @brief Specialisation for `std::array<T, N>`.
/// @ingroup api_matrix
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
/// @ingroup api_matrix
// The out-of-line members (USM advice calls, operator==) are explicitly instantiated in
// src/util/sycl-util-impl.cc, so Span crosses the shared-library boundary like Matrix.
template <typename T>
struct BATCHLAS_API Span
{
    using value_type = T;        ///< Element type.
    using pointer = T*;          ///< Pointer to an element.
    using size_t = std::size_t;  ///< Size and index type.
    /// @brief An empty span (null data, size 0).
    inline constexpr Span() : data_(nullptr), size_(0) {}
    /// @brief Views @p size elements starting at @p data.
    /// @param[in] data  pointer to the first element (USM the device can access)
    /// @param[in] size  number of elements viewed
    inline constexpr Span(T *data, size_t size) : data_(data), size_(size) {}
    /// @brief Views the range \f$[\mathit{begin}, \mathit{end})\f$ (@p begin to @p end, end excluded).
    /// @param[in] begin  pointer to the first element
    /// @param[in] end    pointer one past the last element; \f$\mathit{end} \ge \mathit{begin}\f$
    inline constexpr Span(T *begin, T *end) : data_(begin), size_(std::distance(begin, end)) {}
    /// @brief Copies the pointer and size; the elements are not copied.
    /// @param[in] other  span whose pointer and size are copied
    inline constexpr Span(const Span<T> &other) = default;
    /// @brief Same as the copy constructor.
    /// @param[in] other  span whose pointer and size are copied
    inline constexpr Span(Span<T> &&other) = default;
    /// @brief Views @p other with its first @p offset elements dropped.
    /// @param[in] other   span to view a suffix of
    /// @param[in] offset  number of leading elements dropped
    /// @pre `offset <= other.size()`
    inline constexpr Span(const Span<T>& other, size_t offset) : data_(other.data_ + offset), size_(other.size_ - offset) { assert(offset <= other.size_); }
    /// @brief Views @p other with its first @p offset elements dropped.
    /// @param[in] other   span to view a suffix of
    /// @param[in] offset  number of leading elements dropped
    /// @pre `offset <= other.size()`
    inline constexpr Span(Span<T>&& other, size_t offset) : data_(other.data_ + offset), size_(other.size_ - offset) { assert(offset <= other.size_); }
    /// @brief A one-element span over @p value.
    ///
    /// Explicit on purpose: implicitly, a scalar lvalue became a length-1 buffer
    /// wherever a Span parameter's element type is not deduced from the argument,
    /// so `syev(q, A, W[i * n], ...)` (the LAPACK idiom minus the `&`) compiled
    /// and overran the caller's array. Write `Span(x)` when one element is meant.
    /// @see @ref matrix-model-span-and-unifiedvector
    /// @param[in] value  the element viewed (by reference: the span aliases it, nothing is copied)
    inline constexpr explicit Span(T& value) : data_(&value), size_(1) {}

    /// @name USM memory hints
    /// Advice for USM shared memory on the device of @p ctx. Each returns the event of
    /// the advice, or an empty Event when the span is empty or the runtime rejects
    /// the advice (errors are swallowed: these are hints). Pass the Queue the data
    /// will be used on; the default Queue() is a new queue on Device::default_device().
    /// @{

    /// @brief Prefer residence on the device of @p ctx.
    /// @param[in] ctx  queue whose device the data will be used on
    /// @return event of the advice, or an empty Event
    Event set_preferred_location(const Queue &ctx = Queue()) const;
    /// @brief Map for access by the device of @p ctx.
    /// @param[in] ctx  queue whose device the data will be used on
    /// @return event of the advice, or an empty Event
    Event set_access_device(const Queue &ctx = Queue()) const;
    /// @brief Migrate to the device of @p ctx.
    /// @param[in] ctx  queue whose device the data will be used on
    /// @return event of the prefetch, or an empty Event
    Event prefetch(const Queue &ctx = Queue()) const;
    /// @}

    /// @brief Reinterprets the bytes as elements of type @p U; the size is `size_bytes() / sizeof(U)`.
    /// @tparam U  element type of the returned span
    /// @return a span of `U` over the same memory
    template <typename U>
    inline constexpr Span<U> as_span() const {
        return Span<U>(reinterpret_cast<U*>(data_), (sizeof(T) * size_ / sizeof(U)) );
    }

    /// @brief Elements \f$[\mathit{offset}, \mathit{size})\f$, i.e. from @p offset to size().
    /// @param[in] offset  index of the first element viewed; `offset <= size()`
    /// @return a span over the suffix
    inline constexpr Span<T> subspan(size_t offset) const { assert(offset <= size_); return Span<T>(data_ + offset, size_ - offset); }
    /// @brief Elements \f$[\mathit{offset}, \mathit{offset} + \mathit{count})\f$.
    /// @param[in] offset  index of the first element viewed
    /// @param[in] count   number of elements viewed
    /// @return a span over the \f$\mathit{count}\f$ elements
    /// @pre `offset + count <= size()`
    inline constexpr Span<T> subspan(size_t offset, size_t count) const { assert(offset + count <= size_);  return Span<T>(data_ + offset, count); }
    /// @brief Rebinds this span to @p other's pointer and size; no elements are copied.
    /// @param[in] other  span whose pointer and size are taken
    /// @return `*this`
    inline constexpr Span<T>& operator= (const Span<T> &other) { data_ = other.data_; size_ = other.size_; return *this; }
    /// @brief Same as the copy assignment.
    /// @param[in] other  span whose pointer and size are taken
    /// @return `*this`
    inline constexpr Span<T>& operator= (Span<T> &&other) { return *this = other; }
    /// @brief Element-wise comparison on the host.
    ///
    /// Spans of different sizes compare unequal. Real floating-point elements
    /// \f$a\f$, \f$b\f$ compare with a relative tolerance of \f$20\,\varepsilon\f$:
    /// \f$\lvert a - b \rvert < 20\,\varepsilon \cdot \max(\lvert a \rvert, \lvert b \rvert)\f$, or
    /// \f$\lvert a - b \rvert < 20\,\varepsilon\f$ when that maximum is at most \f$20\,\varepsilon\f$.
    /// Every other type, complex included, compares exactly. Two spans over the same
    /// pointer are equal without reading the data.
    /// @param[in] other  span to compare with
    /// @return true when the sizes match and every element pair compares equal
    /// @pre the memory is host-readable and no kernel is writing it
    bool operator==(const Span<T> other) const;
    /// @brief Element @p index.
    /// @param[in] index  element index; `index < size()`
    /// @return reference to the element
    inline constexpr T &operator[](size_t index) const {assert(index < size_); assert(data_); return data_[index];}
    /// @brief Element @p index.
    /// @param[in] index  element index; `index < size()` (asserted, not thrown)
    /// @return reference to the element
    inline constexpr T &at(size_t index) const{assert(index < size_); assert(data_); return data_[index];}
    inline constexpr T *data() const { return data_; }            ///< Pointer to the first element (may be null).
    inline constexpr size_t size() const { return size_; }        ///< Number of elements.
    inline constexpr bool empty() const { return size_ == 0; }    ///< True when `size() == 0`.
    /// @brief `size() * sizeof(T)`.
    inline constexpr size_t size_bytes() const { return size_ * sizeof(T); }
    inline constexpr T *begin() const { return data_; }           ///< Iterator to the first element.
    inline constexpr T *end() const { return data_ + size_; }     ///< Iterator one past the last element.
    inline constexpr T &front() const { return data_[0]; }        ///< First element. @pre `!empty()`
    inline constexpr T &back() const { return data_[size_ - 1]; } ///< Last element. @pre `!empty()`
    /// @brief Prints the elements on the host as `[a, b, c]`.
    /// @param[in,out] os   stream written to
    /// @param[in]     vec  span whose elements are printed
    /// @return @p os
    /// @pre the memory is host-readable
    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const Span<U> &vec);

private:
    T *data_;
    size_t size_;
};

/// @name Span deduction guides
/// `Span(p, n)`, `Span(first, last)` and the explicit `Span(x)` deduce `T`.
/// @{
template <typename T>
Span(T*, typename Span<T>::size_t) -> Span<T>;

template <typename T>
Span(T*, std::size_t) -> Span<T>;

template <typename T>
Span(T*, T*) -> Span<T>;

template <typename T>
Span(T&) -> Span<T>;
/// @}

}  // namespace batchlas

// Transitional shim: Span and is_std_array used to live at global scope. Define
// BATCHLAS_NO_GLOBAL_NAMES to drop these; CTAD still works because the deduction
// guides are found in Span's own namespace.
#ifndef BATCHLAS_NO_GLOBAL_NAMES
using batchlas::Span;
using batchlas::is_std_array;
#endif


#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

/// @file
/// @brief BumpAllocator, the linear sub-allocator over a workspace span, and workspace_bytes().
/// @see @ref design_workspace
/// @ingroup workspace

namespace batchlas {

/// @brief Linear (bump) sub-allocator over a caller-supplied workspace buffer.
///
/// Each allocate() carves the next suitably aligned slice off the front of the buffer; nothing
/// is ever freed individually. Every alignment is `max(16, MEM_BASE_ADDR_ALIGN / 8, alignof(T))`.
///
/// A pool made by measuring() is a *sizing* pool: it runs the same arithmetic over a fictitious
/// region and reports through required_bytes() the buffer size the same call sequence needs.
/// That is how every `*_buffer_size` query is computed (see workspace_bytes()).
/// @warning Size a workspace with measuring() and required_bytes(), never by re-summing
///          allocation_size() by hand: the capacity check uses the rounded size while the cursor
///          advances by the raw size, so an exactly simulated total is too small.
/// @ingroup workspace
// evidence: docs/design/workspace.md#workspace-the-bumpallocator-alignment-trap
struct BumpAllocator {
    /// @brief A pool over `byte_size` bytes starting at `data`.
    template <typename T>
    BumpAllocator(T* data, size_t byte_size): data(data), byte_size(byte_size){}

    /// @brief A pool over the bytes of `span`.
    template <typename T>
    BumpAllocator(Span<T> span): data(span.data()), byte_size(span.size()*sizeof(T)){}

    /// @brief A sizing pool: an effectively unbounded, maximally aligned fictitious region.
    ///
    /// Give a real pool required_bytes() and the same call sequence succeeds, with at most one
    /// alignment quantum to spare.
    /// @warning The pointers it hands out are non-null and aligned (views can be built over them)
    ///          but unbacked: sizing code must never touch their contents, launch a kernel, or
    ///          test `data() != nullptr` to mean "a workspace was passed".
    // Exact because every alignment is one uniform value (device_align >= 16 >= alignof(T)), so
    // offsets from the base match a real pool's. evidence: docs/design/workspace.md#workspace-bumpallocator-sizing-mode
    static BumpAllocator measuring() { return BumpAllocator(measure_tag{}); }

    /// @brief True for a pool made by measuring().
    inline bool is_measuring() const { return measuring_; }

    /// @brief Bytes a real pool must be given for the call sequence so far to succeed.
    ///
    /// Rounded up to the coarsest alignment the sequence asked for.
    /// @throws batchlas::api_misuse on a pool not made by measuring()
    // Rounding is load-bearing: callers add a callee's size into their total and re-serve it
    // with allocate<std::byte>(), which rounds again.
    inline size_t required_bytes() const {
        if (!measuring_) {
            throw batchlas::api_misuse("BumpAllocator::required_bytes() on a real pool; use BumpAllocator::measuring().");
        }
        if (align_quantum_ == 0) return high_water_;
        return (high_water_ + align_quantum_ - 1) & ~(align_quantum_ - 1);
    }

    /// @brief Alignment in bytes of an allocation of `T` on `device`.
    template<typename T>
    constexpr inline static auto alignment(const Device& device){
        // 16-byte floor: vendors commonly require it and SYCL cannot query it.
        auto device_align_bytes = std::max((size_t)16, (size_t)device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN)/8);
        return std::max(device_align_bytes, static_cast<std::uintptr_t>(alignof(T)));
    }

    /// @brief Bytes `size` elements of `T` occupy, rounded up to alignment<T>(); 0 for `size == 0`.
    template<typename T>
    constexpr inline static size_t allocation_size(const Device& device, size_t size){
        if (size == 0) return 0; // Handle zero size allocation gracefully
        std::uintptr_t total_size = size * sizeof(T);
        return (total_size + alignment<T>(device) - 1) & ~(alignment<T>(device) - 1);
    }

    /// @brief allocation_size() for the Queue's device.
    template<typename T>
    constexpr inline static size_t allocation_size(Queue& ctx, size_t size)   {return allocation_size<T>(ctx.device(), size);}

    /// @brief Carve `size` elements of `T`, aligned to alignment<T>(), off the front of the pool.
    /// @tparam T element type
    /// @param device device whose alignment rule applies
    /// @param size   number of elements; 0 returns an empty Span and consumes nothing
    /// @return the allocated elements (uninitialised; unbacked in a sizing pool)
    /// @throws batchlas::workspace_error if the pool has fewer than allocation_size<T>() bytes left
    template<typename T>
    constexpr inline Span<T> allocate(const Device& device, size_t size){
        if (size == 0) return {};
        size_t alloc_size = allocation_size<T>(device,size);
        if (alloc_size > byte_size){
            throw batchlas::workspace_error("Attempted to allocate " + std::to_string(alloc_size) + " bytes from a BumpAllocator with only " + std::to_string(byte_size) + " bytes remaining.");
        }

        void* aligned = data;
        size_t remaining = byte_size;
        if (std::align(alignment<T>(device), size * sizeof(T), aligned, remaining) == nullptr) {
            throw batchlas::workspace_error("Failed to align BumpAllocator storage for requested allocation.");
        }

        if (measuring_) {
            // Record what a real pool must be GIVEN, not what it consumes: the check above uses
            // the rounded size from the unaligned cursor, the advance below only the raw extent.
            const auto base = static_cast<std::byte*>(measure_base_);
            const size_t need_for_check = static_cast<size_t>(static_cast<std::byte*>(data) - base) + alloc_size;
            const size_t need_for_data  = static_cast<size_t>(static_cast<std::byte*>(aligned) - base) + size * sizeof(T);
            high_water_ = std::max(high_water_, std::max(need_for_check, need_for_data));
            align_quantum_ = std::max(align_quantum_, static_cast<size_t>(alignment<T>(device)));
        }

        auto* next = static_cast<std::byte*>(aligned) + size * sizeof(T);
        T* ptr = static_cast<T*>(aligned);
        byte_size -= static_cast<size_t>(next - static_cast<std::byte*>(data));
        data = next;

        return Span(ptr, size);
    }

    /// @brief allocate() with the Queue's device.
    template<typename T>
    constexpr inline Span<T> allocate(Queue& ctx, size_t size) {return allocate<T>(ctx.device(), size);}

    /// @brief The still-unclaimed tail of the pool.
    ///
    /// Lets a callee sub-allocate without the caller knowing its size up front; pair with
    /// consume() to hand back the bytes it actually took.
    /// @throws batchlas::api_misuse on a sizing pool, whose extent is fictitious
    inline Span<std::byte> remaining() const {
        if (measuring_) {
            // Callees sized against remaining().size() must be converted deliberately
            // (see iluk / syevx_lobpcg), never implicitly.
            throw batchlas::api_misuse("BumpAllocator::remaining() is not available in sizing mode.");
        }
        return Span<std::byte>(static_cast<std::byte*>(data), byte_size);
    }

    /// @brief Advance the pool by `bytes` (after a callee used remaining()).
    /// @throws batchlas::workspace_error if fewer than `bytes` remain
    inline void consume(size_t bytes) {
        if (bytes > byte_size) {
            throw batchlas::workspace_error("BumpAllocator::consume called with more bytes than remain.");
        }
        data = static_cast<std::byte*>(data) + bytes;
        byte_size -= bytes;
    }
    
    private:

        struct measure_tag {};

        // 4 GiB-aligned (past any MEM_BASE_ADDR_ALIGN) and unmapped: a stray dereference faults.
        static constexpr std::uintptr_t kMeasureBase = std::uintptr_t(1) << 32;

        explicit BumpAllocator(measure_tag)
            : data(reinterpret_cast<void*>(kMeasureBase)),
              // No real request trips the check, and base + byte_size cannot wrap.
              byte_size(std::numeric_limits<size_t>::max() / 4),
              measuring_(true),
              measure_base_(reinterpret_cast<void*>(kMeasureBase)) {}

        void* data;
        size_t byte_size;
        bool measuring_ = false;
        void* measure_base_ = nullptr;
        size_t high_water_ = 0;
        size_t align_quantum_ = 0;
};

/// @brief Bytes a workspace layout needs, obtained by replaying it against a sizing pool.
///
/// An algorithm describes its workspace once, in a `*_layout` function, and both its
/// `*_buffer_size` entry point and its implementation go through it, so they cannot drift:
/// @code
/// size_t foo_buffer_size(Queue& ctx, ...) {
///     return workspace_bytes([&](BumpAllocator& p) { return foo_layout<B, T>(ctx, p, ...); });
/// }
/// @endcode
/// @param layout callable taking a `BumpAllocator&`; its result is discarded
/// @return BumpAllocator::required_bytes() of the sizing pool after `layout` ran
/// @pre `layout` is pure with respect to the workspace: it may read the caller's views and build
///      views over what it allocates, but never reads or writes workspace memory, never launches
///      a kernel, and asks nested size queries about the caller's views only.
/// @see @ref design_workspace
/// @ingroup workspace
template <typename Fn>
inline size_t workspace_bytes(Fn&& layout) {
    auto sizer = BumpAllocator::measuring();
    (void)layout(sizer);
    return sizer.required_bytes();
}

}  // namespace batchlas

// Transitional global-scope shim; define BATCHLAS_NO_GLOBAL_NAMES to switch it off.
// workspace_bytes MUST be shimmed: its only argument is a lambda, so ADL cannot find it.
#ifndef BATCHLAS_NO_GLOBAL_NAMES
using batchlas::BumpAllocator;
using batchlas::workspace_bytes;
#endif

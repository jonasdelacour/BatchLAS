#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

struct BumpAllocator {
    template <typename T>
    BumpAllocator(T* data, size_t byte_size): data(data), byte_size(byte_size){}

    template <typename T>
    BumpAllocator(Span<T> span): data(span.data()), byte_size(span.size()*sizeof(T)){}

    // ---- sizing mode -------------------------------------------------------
    //
    // A pool over a fictitious, maximally-aligned, effectively unbounded region.
    // It runs the same bump arithmetic as a real pool and reports the smallest
    // buffer that will satisfy the same call sequence: give a real pool
    // required_bytes() and every allocation succeeds, with at most one alignment
    // quantum to spare.
    //
    // That holds because every alignment this allocator uses is
    // max(device_align, alignof(T)) and device_align is at least 16 while every
    // T we allocate has alignof(T) <= 16: the alignment is therefore a single
    // uniform value, and the layout depends only on offsets relative to the
    // start of the pool. The fake base is aligned far beyond any device
    // requirement, and a real pool's base is device-aligned, so both produce the
    // same offsets.
    //
    // The pointers handed out are non-null and correctly aligned so that views
    // can be constructed over them, but they address nothing: dereferencing one
    // faults immediately rather than corrupting memory. Sizing code must
    // therefore build views over workspace only, never touch their contents,
    // and never launch a kernel.
    static BumpAllocator measuring() { return BumpAllocator(measure_tag{}); }

    inline bool is_measuring() const { return measuring_; }

    // Bytes a real pool must be given for this call sequence to succeed. Only
    // meaningful in sizing mode.
    //
    // Rounded up to the coarsest alignment the sequence asked for. Sizing
    // results have always been alignment multiples -- they were sums of
    // allocation_size terms, each of which is rounded -- and callers depend on
    // that: several of them add a callee's size straight into their own total
    // and then re-serve it via allocate<std::byte>(), which rounds again. Handing
    // back an unrounded figure silently under-provisions every such caller by up
    // to one quantum.
    inline size_t required_bytes() const {
        if (!measuring_) {
            throw std::runtime_error("BumpAllocator::required_bytes() on a real pool; use BumpAllocator::measuring().");
        }
        if (align_quantum_ == 0) return high_water_;
        return (high_water_ + align_quantum_ - 1) & ~(align_quantum_ - 1);
    }

    template<typename T>
    constexpr inline static auto alignment(const Device& device){
        //It is common for GPU vendors to require 16 byte alignment of pointers (equal to 4 floats).
        //It seems however that this property can't be immediately queried through the sycl runtime, 
        //hence the hardcoded value of 16.
        auto device_align_bytes = std::max((size_t)16, (size_t)device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN)/8);
        return std::max(device_align_bytes, static_cast<std::uintptr_t>(alignof(T)));
    }

    template<typename T>
    constexpr inline static size_t allocation_size(const Device& device, size_t size){
        if (size == 0) return 0; // Handle zero size allocation gracefully
        std::uintptr_t total_size = size * sizeof(T);
        return (total_size + alignment<T>(device) - 1) & ~(alignment<T>(device) - 1);
    }

    template<typename T>
    constexpr inline static size_t allocation_size(Queue& ctx, size_t size)   {return allocation_size<T>(ctx.device(), size);}

    template<typename T>
    constexpr inline Span<T> allocate(const Device& device, size_t size){
        if (size == 0) return {};
        size_t alloc_size = allocation_size<T>(device,size);
        if (alloc_size > byte_size){
            throw std::runtime_error("Attempted to allocate " + std::to_string(alloc_size) + " bytes from a BumpAllocator with only " + std::to_string(byte_size) + " bytes remaining.");
        }

        void* aligned = data;
        size_t remaining = byte_size;
        if (std::align(alignment<T>(device), size * sizeof(T), aligned, remaining) == nullptr) {
            throw std::runtime_error("Failed to align BumpAllocator storage for requested allocation.");
        }

        if (measuring_) {
            // What a real pool must be *given* for this call to succeed, which is
            // more than what it goes on to consume. The capacity check above tests
            // the alignment-rounded alloc_size against the bytes left measured from
            // the unaligned cursor, while the cursor only advances by the raw
            // extent -- so a pool sized by the advance alone fails its own check on
            // any allocation whose extent is not a multiple of the alignment.
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

    template<typename T>
    constexpr inline Span<T> allocate(Queue& ctx, size_t size) {return allocate<T>(ctx.device(), size);}

    // The still-unclaimed tail of the pool. Lets a callee sub-allocate without the
    // caller having to know its size up front; pair with consume() to hand the
    // bytes it actually took back to this allocator.
    inline Span<std::byte> remaining() const {
        if (measuring_) {
            // A sizing pool has no tail to hand out: its extent is fictitious, so
            // any callee that sizes itself against remaining().size() would size
            // against a number that means nothing. Such call sites have to be
            // converted deliberately (see iluk / syevx_lobpcg), not implicitly.
            throw std::runtime_error("BumpAllocator::remaining() is not available in sizing mode.");
        }
        return Span<std::byte>(static_cast<std::byte*>(data), byte_size);
    }

    inline void consume(size_t bytes) {
        if (bytes > byte_size) {
            throw std::runtime_error("BumpAllocator::consume called with more bytes than remain.");
        }
        data = static_cast<std::byte*>(data) + bytes;
        byte_size -= bytes;
    }
    
    private:

        struct measure_tag {};

        // Aligned to 4 GiB -- past any conceivable MEM_BASE_ADDR_ALIGN -- and far
        // outside any mapping, so a stray dereference faults instead of corrupting.
        static constexpr std::uintptr_t kMeasureBase = std::uintptr_t(1) << 32;

        explicit BumpAllocator(measure_tag)
            : data(reinterpret_cast<void*>(kMeasureBase)),
              // Large enough that no real sizing request trips the capacity check,
              // small enough that base + byte_size cannot wrap.
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

// Bytes a workspace layout needs, obtained by replaying the layout against a
// sizing pool. This is the whole point of sizing mode: an algorithm describes
// its workspace exactly once, in a `*_layout` function, and both its
// `*_buffer_size` entry point and its implementation go through that one
// description. Neither can drift from the other because there is only one.
//
//   template <Backend B, typename T>
//   FooWorkspace<T> foo_layout(Queue& ctx, BumpAllocator& pool, <shape args>) {
//       return { pool.allocate<T>(ctx, n * batch), ... };
//   }
//
//   size_t foo_buffer_size(Queue& ctx, ...) {
//       return workspace_bytes([&](BumpAllocator& p) { return foo_layout<B,T>(ctx, p, ...); });
//   }
//
// A layout function must be pure with respect to the workspace: it may read the
// *caller's* views (shapes, and their contents -- those are real), and it may
// build views over what it allocates, but it must never read or write workspace
// memory and never launch a kernel. In sizing mode the workspace addresses it
// hands out are unbacked. Nested size queries must be asked about the caller's
// views, not about workspace-derived ones, for the same reason.
template <typename Fn>
inline size_t workspace_bytes(Fn&& layout) {
    auto sizer = BumpAllocator::measuring();
    (void)layout(sizer);
    return sizer.required_bytes();
}
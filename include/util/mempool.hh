#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

struct BumpAllocator {
    template <typename T>
    BumpAllocator(T* data, size_t byte_size): data(data), byte_size(byte_size){}

    template <typename T>
    BumpAllocator(Span<T> span): data(span.data()), byte_size(span.size()*sizeof(T)){}

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

        void* data;
        size_t byte_size;
};
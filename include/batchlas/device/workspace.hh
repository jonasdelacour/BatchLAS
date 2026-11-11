// SPDX-License-Identifier: MIT
#pragma once
// Device LA workspace protocol (v22 design C.5). An op with a tier states its memory as a
// constexpr plan: the host sizes the launch with <op>_buffer_size<..., Tier, T>(n) == plan.total,
// and the op carves its pointers with the SAME plan, so size and use cannot drift apart.
//   WorkspaceSize  local and global bytes of ONE matrix, and the alignment of both
//   Workspace      the matrix's slice: a decorated local pointer and a global pointer (the
//                  caller offsets the global by matrix index * global_bytes)
//   Plan           a constexpr bump allocator over the two spaces

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>

namespace batchlas::device {

struct WorkspaceSize {
    std::size_t local_bytes = 0;
    std::size_t global_bytes = 0;
    std::size_t align = 16;
    friend constexpr bool operator==(const WorkspaceSize&, const WorkspaceSize&) = default;
};

template <class T> using local_mptr = sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::yes>;

struct Workspace {
    local_mptr<std::byte> local;  // decorated: accesses stay in the shared address space
    std::byte* global = nullptr;
};

enum class WsSpace { Local, Global };
template <class T> struct WsSlot {
    WsSpace space;
    std::size_t offset;
    std::size_t count;
};

// The bump allocator both sides run: take<T>(space, count) -> a slot aligned to max(alignof(T), align).
class Plan {
  public:
    constexpr explicit Plan(std::size_t align = 16) : align_(align) {}
    template <class T> constexpr WsSlot<T> take(WsSpace s, std::size_t count) {
        std::size_t& top = s == WsSpace::Local ? local_ : global_;
        const std::size_t a = alignof(T) > align_ ? alignof(T) : align_;
        top = (top + a - 1) / a * a;
        const WsSlot<T> slot{s, top, count};
        top += count * sizeof(T);
        return slot;
    }
    constexpr WorkspaceSize size() const {
        auto round = [&](std::size_t b) { return (b + align_ - 1) / align_ * align_; };
        return {round(local_), round(global_), align_};
    }

  private:
    std::size_t align_;
    std::size_t local_ = 0;
    std::size_t global_ = 0;
};

// The typed pointer of a slot inside this matrix's workspace: a decorated local pointer for a
// Local slot (a generic T* would lose the shared address space: generic LD/ST), a T* for Global.
template <class T> BATCHLAS_DEVICE_INLINE local_mptr<T> carve_local(const Workspace& w, const WsSlot<T>& s) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        reinterpret_cast<T*>(w.local.get_raw() + s.offset));
}
template <class T> BATCHLAS_DEVICE_INLINE T* carve_global(const Workspace& w, const WsSlot<T>& s) {
    return reinterpret_cast<T*>(w.global + s.offset);
}

// Matrix b's slice of the caller's global buffer of batch * global_bytes.
BATCHLAS_DEVICE_INLINE std::byte* global_slice(std::byte* base, const WorkspaceSize& sz, std::int64_t b) {
    return base + static_cast<std::ptrdiff_t>(b) * static_cast<std::ptrdiff_t>(sz.global_bytes);
}

}  // namespace batchlas::device

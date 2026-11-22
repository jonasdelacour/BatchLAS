// SPDX-License-Identifier: MIT
#pragma once
/** @file
 * @brief The device workspace protocol: one constexpr Plan sizes a matrix's workspace on the
 * host and carves it in the kernel.
 * @ingroup api_device_launch
 *
 * An op with a tier states its memory as a constexpr Plan. The host sizes a launch with
 * `launch<R, Tier, T>::buffer_size(extents...)` (the plan's size()) and the op carves its
 * pointers with the same plan, so size and use cannot drift apart. WorkspaceSize is one matrix's
 * local and global bytes, Workspace is that matrix's slice of both spaces, and Plan is the bump
 * allocator over them.
 */

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>

namespace batchlas::device {

/// @brief The workspace of ONE matrix: its local and global bytes and the alignment of both.
/// @ingroup api_device_launch
///
/// A tier_info's `buffer_size(extents...)` returns it, and so do launch::buffer_size and
/// device::buffer_size. Two sizes compare equal member by member.
struct WorkspaceSize {
    std::size_t local_bytes = 0;  ///< Bytes of local memory one matrix needs.
    /// @brief Bytes of global memory one matrix needs; the caller's buffer holds batch times this.
    std::size_t global_bytes = 0;
    std::size_t align = 16;  ///< The alignment of both slices, in bytes (local_workspace admits at most 16).
    friend constexpr bool operator==(const WorkspaceSize&, const WorkspaceSize&) = default;  ///< Member-wise equality.
};

/// @brief A decorated pointer to T in local memory, the type of a Workspace's local slice.
/// @ingroup api_device_launch
/// @tparam T the element type
template <class T> using local_mptr = sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::yes>;

/// @brief One matrix's slice of the workspace: a decorated local pointer and a global pointer.
/// @ingroup api_device_launch
///
/// launch::workspace builds it for the calling item's matrix; carve_local and carve_global turn
/// a Plan's slot into a typed pointer inside it.
struct Workspace {
    local_mptr<std::byte> local;  ///< The local slice; decorated, so accesses stay in the shared address space.
    std::byte* global = nullptr;  ///< This matrix's slice of the caller's global buffer (see global_slice), or nullptr.
};

/// @brief The space a Plan slot lives in: `Local` (the work-group's local memory) or `Global` (the
/// caller's global buffer).
/// @ingroup api_device_launch
enum class WsSpace { Local, Global };
/// @brief A typed region of one matrix's workspace, as Plan::take returns it.
/// @ingroup api_device_launch
/// @tparam T the element type of the region
template <class T> struct WsSlot {
    WsSpace space;  ///< The space the region lives in.
    std::size_t offset;  ///< Byte offset of the region from the start of the matrix's slice of that space.
    std::size_t count;  ///< Number of T elements in the region.
};

/// @brief The constexpr bump allocator the host and the kernel both run over the two spaces.
/// @ingroup api_device_launch
///
/// `take<T>(space, count)` returns a slot aligned to max(alignof(T), align), and size() is the
/// WorkspaceSize of everything taken. Sizing and carving run the same sequence of take calls, so
/// the host allocates exactly what the kernel uses.
class Plan {
public:
    /// @brief An empty plan.
    /// @param align the minimum alignment of every slot and of both totals, in bytes
    constexpr explicit Plan(std::size_t align = 16) : align_(align) {}
    /// @brief Reserves count elements of T in space s.
    /// @tparam T the element type
    /// @param s the space to take from
    /// @param count the number of elements
    /// @return the slot, at an offset aligned to max(alignof(T), align)
    template <class T> constexpr WsSlot<T> take(WsSpace s, std::size_t count) {
        std::size_t& top = s == WsSpace::Local ? local_ : global_;
        const std::size_t a = alignof(T) > align_ ? alignof(T) : align_;
        top = (top + a - 1) / a * a;
        const WsSlot<T> slot{s, top, count};
        top += count * sizeof(T);
        return slot;
    }
    /// @brief The local and global totals so far, each rounded up to align.
    /// @return the WorkspaceSize to allocate for one matrix
    constexpr WorkspaceSize size() const {
        auto round = [&](std::size_t b) { return (b + align_ - 1) / align_ * align_; };
        return {round(local_), round(global_), align_};
    }

private:
    std::size_t align_;
    std::size_t local_ = 0;
    std::size_t global_ = 0;
};

/// @brief The typed pointer of a Local slot inside this matrix's workspace.
/// @ingroup api_device_launch
///
/// The pointer stays decorated: a generic T* would lose the shared address space and compile to
/// generic loads and stores.
/// @tparam T the element type of the slot
/// @param w this matrix's workspace
/// @param s a Local slot of the plan that sized w
/// @return a decorated local pointer to the slot's first element
template <class T> BATCHLAS_DEVICE_INLINE local_mptr<T> carve_local(const Workspace& w, const WsSlot<T>& s) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        reinterpret_cast<T*>(w.local.get_raw() + s.offset));
}
/// @brief The typed pointer of a Global slot inside this matrix's workspace.
/// @ingroup api_device_launch
/// @tparam T the element type of the slot
/// @param w this matrix's workspace
/// @param s a Global slot of the plan that sized w
/// @return a pointer to the slot's first element inside `w.global`
template <class T> BATCHLAS_DEVICE_INLINE T* carve_global(const Workspace& w, const WsSlot<T>& s) {
    return reinterpret_cast<T*>(w.global + s.offset);
}

/// @brief The workspace of an op called inside another op, from a Local slot on.
/// @ingroup api_device_launch
///
/// The local pointer stays decorated and the global pointer is w's own.
/// @tparam L the element type of the slot
/// @param w the calling op's workspace
/// @param local the Local slot where the inner op's local workspace starts
/// @return w with its local pointer advanced to the slot
template <class L> BATCHLAS_DEVICE_INLINE Workspace sub_workspace(const Workspace& w, const WsSlot<L>& local) {
    return {w.local + local.offset, w.global};
}
/// @brief The workspace of an op called inside another op, from a Local and a Global slot on.
/// @ingroup api_device_launch
///
/// The local pointer stays decorated.
/// @tparam L the element type of the local slot
/// @tparam G the element type of the global slot
/// @param w the calling op's workspace
/// @param local the Local slot where the inner op's local workspace starts
/// @param global the Global slot where the inner op's global workspace starts
/// @return w with both pointers advanced to their slots
template <class L, class G> BATCHLAS_DEVICE_INLINE Workspace sub_workspace(const Workspace& w, const WsSlot<L>& local, const WsSlot<G>& global) {
    return {w.local + local.offset, w.global + global.offset};
}

/// @brief Matrix b's slice of the caller's global buffer of batch * global_bytes.
/// @ingroup api_device_launch
/// @param base the caller's global workspace, batch * `sz.global_bytes` bytes
/// @param sz the workspace of one matrix
/// @param b the matrix index
/// @return `base + b * sz.global_bytes`
BATCHLAS_DEVICE_INLINE std::byte* global_slice(std::byte* base, const WorkspaceSize& sz, std::int64_t b) {
    return base + static_cast<std::ptrdiff_t>(b) * static_cast<std::ptrdiff_t>(sz.global_bytes);
}

}  // namespace batchlas::device

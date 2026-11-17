// SPDX-License-Identifier: MIT
#pragma once
// launch<Routine, Tier, T>: one tier of a routine as a kernel launch, from tier_info alone:
// buffer_size(extents...) per matrix, nd_range(batch), exec(it) (executor{it}), slot(it, batch)
// ({b, live}), workspace(it, slm, gws, sz) (this matrix's slices) and sg_size. A packed tier
// (per_group > 1) has dead slots in its last work-group: they run the op with live = false.

#include <cstddef>
#include <cstdint>
#include <string>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>
#include <batchlas/error.hh>

namespace batchlas::device {

struct MatrixSlot {
    int b;
    bool live;
};

// The in-kernel matrix index of every packed launch, the library's included: work-group wg
// holds per_group matrices and its team-th executor owns matrix wg * per_group + team.
BATCHLAS_DEVICE_INLINE MatrixSlot matrix_slot(int wg, int per_group, int team, int batch) {
    const int b = wg * per_group + team;
    return {b, b < batch};
}

namespace detail {
struct alignas(16) local_slab {
    std::byte b[16];
};
}  // namespace detail

using LocalWorkspace = sycl::local_accessor<detail::local_slab, 1>;

template <class Routine, class Tier, class T> struct launch {
    using tier_type = resolve_tier_t<Routine, T, Tier>;
    using info = tier_info<Routine, tier_type, T>;
    using executor = typename info::executor;
    static constexpr int wg_size = info::wg_size;
    static constexpr int sg_size = info::sg_size;
    static constexpr int per_group = info::per_group;
    static constexpr int team_items = wg_size / per_group;  // the items of one executor
    static_assert(per_group >= 1 && wg_size % per_group == 0, "launch: per_group must divide tier_info::wg_size");

    template <class... E> static constexpr WorkspaceSize buffer_size(E... extents) { return info::buffer_size(extents...); }

    static sycl::nd_range<1> nd_range(int batch) {
        const std::size_t groups = batch > 0 ? (static_cast<std::size_t>(batch) + per_group - 1) / per_group : 0;
        return {sycl::range<1>(groups * wg_size), sycl::range<1>(wg_size)};
    }
    static BATCHLAS_DEVICE_INLINE executor exec(const sycl::nd_item<1>& it) { return executor{it}; }
    // This item's executor within its work-group: 0 .. per_group - 1.
    static BATCHLAS_DEVICE_INLINE int team(const sycl::nd_item<1>& it) {
        if constexpr (per_group == 1) return 0;
        else return static_cast<int>(it.get_local_linear_id()) / team_items;
    }
    static BATCHLAS_DEVICE_INLINE MatrixSlot slot(const sycl::nd_item<1>& it, int batch) {
        return matrix_slot(static_cast<int>(it.get_group_linear_id()), per_group, team(it), batch);
    }
    // gws: batch * sz.global_bytes (or nullptr); a dead slot's global slice lies past it.
    static BATCHLAS_DEVICE_INLINE Workspace workspace(const sycl::nd_item<1>& it, const LocalWorkspace& slm, std::byte* gws,
                                                      const WorkspaceSize& sz) {
        auto* const base = reinterpret_cast<std::byte*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get());
        const std::size_t off = static_cast<std::size_t>(team(it)) * sz.local_bytes;
        return {sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(base + off),
                global_slice(gws, sz, static_cast<int>(it.get_group_linear_id()) * per_group + team(it))};
    }
};

// Host: the local buffer of one work-group of L (per_group slices of sz.local_bytes; one slab
// when the tier needs none, so the kernel's capture list does not depend on the tier).
template <class L> LocalWorkspace local_workspace(sycl::handler& h, const WorkspaceSize& sz) {
    if (sz.align > alignof(detail::local_slab))
        throw batchlas::invalid_argument("local_workspace: alignment " + std::to_string(sz.align) + " exceeds 16 bytes");
    const std::size_t bytes = sz.local_bytes * static_cast<std::size_t>(L::per_group);
    const std::size_t slabs = (bytes + sizeof(detail::local_slab) - 1) / sizeof(detail::local_slab);
    return LocalWorkspace(sycl::range<1>(slabs > 0 ? slabs : 1), h);
}

// One matrix's workspace for tier Tier of Routine at the given extents (host and device).
template <class Routine, class Tier, class T, class... E> constexpr WorkspaceSize buffer_size(E... extents) {
    return launch<Routine, Tier, T>::buffer_size(extents...);
}

}  // namespace batchlas::device

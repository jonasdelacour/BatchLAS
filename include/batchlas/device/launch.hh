// SPDX-License-Identifier: MIT
#pragma once
/** @file
 * @brief launch: one tier of a routine as a kernel launch, from tier_info alone.
 * @ingroup api_device_launch
 *
 * `launch<Routine, Tier, T>` gives the workspace of one matrix (buffer_size), the kernel's
 * nd_range, each item's executor, its matrix slot and its workspace slices. A packed tier
 * (`per_group > 1`) has dead slots in its last work-group: they run the op with `live = false`.
 */

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

/// @brief The batch item a work-item's executor owns, and whether it exists.
/// @ingroup api_device_launch
struct MatrixSlot {
    int b;  ///< The matrix index in the batch.
    /// @brief False for a dead slot of a packed launch's last work-group (b >= batch): pass it to
    /// the op as its `live` argument and do not touch matrix b.
    bool live;
};

/// @brief The in-kernel matrix index of every packed launch, the library's own included.
/// @ingroup api_device_launch
///
/// Work-group wg holds per_group matrices and its team-th executor owns matrix
/// wg * per_group + team.
/// @param wg the work-group's linear id
/// @param per_group the matrices per work-group
/// @param team the executor's index within its work-group, 0 to per_group - 1
/// @param batch the number of matrices in the batch
/// @return the slot `{b, b < batch}`
BATCHLAS_DEVICE_INLINE MatrixSlot matrix_slot(int wg, int per_group, int team, int batch) {
    const int b = wg * per_group + team;
    return {b, b < batch};
}

namespace detail {
struct alignas(16) local_slab {
    std::byte b[16];
};
}  // namespace detail

/// @brief The local buffer of one work-group, in 16-byte aligned slabs; local_workspace makes it.
/// @ingroup api_device_launch
using LocalWorkspace = sycl::local_accessor<detail::local_slab, 1>;

/// @brief The device function a routine tag names, called in one tier: what launch::run calls.
/// @ingroup api_device_launch
///
/// A specialization's `run<Tier>(args...)` calls the routine with the tag's modes and `Tier`:
/// `routine_call<routine::potrf<Uplo::Lower>>::run<Tier>(ex, A, ws)` is
/// `potrf<Uplo::Lower, Tier>(ex, A, ws)`. The LAPACK routines' specializations are in
/// `<batchlas/device/lapack_run.hh>`, which `<batchlas/device/lapack.hh>` includes; the sparse
/// routines' are in `<batchlas/device/sparse_run.hh>`.
/// @tparam Routine a routine tag from device::routine
template <class Routine> struct routine_call {
    static_assert(sizeof(Routine) == 0, "launch::run: no routine_call for this routine tag; include <batchlas/device/lapack.hh> "
                                        "(LAPACK) or <batchlas/device/sparse_run.hh> (sparse)");
};

/// @brief One tier of a routine as a kernel launch: sizes, nd-range, executor, slot and workspace.
/// @ingroup api_device_launch
///
/// Everything comes from `tier_info<Routine, tier_type, T>`. Host: buffer_size and nd_range size
/// and shape the launch, local_workspace (a free function) allocates its local buffer. Kernel:
/// exec, slot and workspace give the calling item its executor, matrix and slices, and run calls
/// the routine in this tier. A tier with `per_group > 1` packs that many executors (and matrices)
/// into one work-group.
/// @tparam Routine a routine tag from device::routine
/// @tparam Tier a concrete tier, or a tier::upto that resolve_tier_t resolves
/// @tparam T the scalar type
template <class Routine, class Tier, class T> struct launch {
    using tier_type = resolve_tier_t<Routine, T, Tier>;  ///< The concrete tier (Tier with a tier::upto resolved).
    using info = tier_info<Routine, tier_type, T>;  ///< The tier_info of the concrete tier.
    using executor = typename info::executor;  ///< The executor type of one matrix.
    static constexpr int wg_size = info::wg_size;  ///< Work-items per work-group.
    static constexpr int sg_size = info::sg_size;  ///< The sub-group size the kernel must run at.
    static constexpr int per_group = info::per_group;  ///< Matrices (executors) per work-group.
    static constexpr int team_items = wg_size / per_group;  ///< The work-items of one executor.
    static_assert(per_group >= 1 && wg_size % per_group == 0, "launch: per_group must divide tier_info::wg_size");

    /// @brief One matrix's workspace at the given extents.
    /// @tparam E the extent types, as the routine's tier_info takes them
    /// @param extents the problem's extents (n, or n and nrhs, ...)
    /// @return the WorkspaceSize of one matrix
    template <class... E> static constexpr WorkspaceSize buffer_size(E... extents) { return info::buffer_size(extents...); }

    /// @brief The kernel's nd-range for batch matrices: ceil(batch / per_group) work-groups of
    /// wg_size items (no work-group for batch <= 0).
    /// @param batch the number of matrices
    /// @return the 1-D nd_range
    static sycl::nd_range<1> nd_range(int batch) {
        const std::size_t groups = batch > 0 ? (static_cast<std::size_t>(batch) + per_group - 1) / per_group : 0;
        return {sycl::range<1>(groups * wg_size), sycl::range<1>(wg_size)};
    }
    /// @brief This item's executor.
    /// @param it the kernel's nd_item
    /// @return `executor{it}`
    static BATCHLAS_DEVICE_INLINE executor exec(const sycl::nd_item<1>& it) { return executor{it}; }
    /// @brief This item's executor within its work-group: 0 to per_group - 1.
    /// @param it the kernel's nd_item
    /// @return the team index (always 0 when per_group is 1)
    static BATCHLAS_DEVICE_INLINE int team(const sycl::nd_item<1>& it) {
        if constexpr (per_group == 1) return 0;
        else return static_cast<int>(it.get_local_linear_id()) / team_items;
    }
    /// @brief The matrix this item's executor owns.
    /// @param it the kernel's nd_item
    /// @param batch the number of matrices
    /// @return the MatrixSlot; `live` is false on a dead slot
    static BATCHLAS_DEVICE_INLINE MatrixSlot slot(const sycl::nd_item<1>& it, int batch) {
        return matrix_slot(static_cast<int>(it.get_group_linear_id()), per_group, team(it), batch);
    }
    /// @brief This item's matrix's slices of the work-group's local buffer and of the caller's
    /// global buffer.
    /// @param it the kernel's nd_item
    /// @param slm the work-group's local buffer, from local_workspace
    /// @param gws the caller's global buffer of batch * `sz.global_bytes` bytes, or nullptr; a dead
    ///        slot's global slice lies past its end
    /// @param sz the workspace of one matrix (buffer_size)
    /// @return the Workspace of this item's matrix
    static BATCHLAS_DEVICE_INLINE Workspace workspace(const sycl::nd_item<1>& it, const LocalWorkspace& slm, std::byte* gws,
                                                      const WorkspaceSize& sz) {
        auto* const base = reinterpret_cast<std::byte*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get());
        const std::size_t off = static_cast<std::size_t>(team(it)) * sz.local_bytes;
        return {sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(base + off),
                global_slice(gws, sz, static_cast<int>(it.get_group_linear_id()) * per_group + team(it))};
    }
    /// @brief Calls the routine in this tier: its device function with the tag's modes and
    /// tier_type. With `using Potrf = launch<routine::potrf<Uplo::Lower>, Tier, T>`,
    /// `Potrf::run(ex, A, ws, s.live)` is `potrf<Uplo::Lower, Tier>(ex, A, ws, s.live)`.
    /// @tparam A the argument types
    /// @param args the routine's runtime arguments, from the executor on (C3)
    /// @return what the routine returns: `info` for a LAPACK-level routine
    template <class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... args) {
        return routine_call<Routine>::template run<tier_type>(static_cast<A&&>(args)...);
    }
};

/// @brief Host: the local buffer of one work-group of a launch.
/// @ingroup api_device_launch
///
/// It holds per_group slices of `sz.local_bytes`; a tier that needs none still gets one slab, so
/// the kernel's capture list does not depend on the tier.
/// @tparam Launch a launch type
/// @param h the command group handler
/// @param sz the workspace of one matrix (`Launch::buffer_size(...)`)
/// @return the local accessor to capture in the kernel
/// @throws batchlas::invalid_argument when `sz.align` exceeds 16 bytes
template <class Launch> LocalWorkspace local_workspace(sycl::handler& h, const WorkspaceSize& sz) {
    if (sz.align > alignof(detail::local_slab))
        throw batchlas::invalid_argument("local_workspace: alignment " + std::to_string(sz.align) + " exceeds 16 bytes");
    const std::size_t bytes = sz.local_bytes * static_cast<std::size_t>(Launch::per_group);
    const std::size_t slabs = (bytes + sizeof(detail::local_slab) - 1) / sizeof(detail::local_slab);
    return LocalWorkspace(sycl::range<1>(slabs > 0 ? slabs : 1), h);
}

/// @brief One matrix's workspace for tier Tier of Routine at the given extents (host and device).
/// @ingroup api_device_launch
///
/// The same as `launch<Routine, Tier, T>::buffer_size(extents...)`; there is no per-op sizer.
/// @tparam Routine a routine tag from device::routine
/// @tparam Tier a concrete tier or a tier::upto
/// @tparam T the scalar type
/// @tparam E the extent types
/// @param extents the problem's extents
/// @return the WorkspaceSize of one matrix
template <class Routine, class Tier, class T, class... E> constexpr WorkspaceSize buffer_size(E... extents) {
    return launch<Routine, Tier, T>::buffer_size(extents...);
}

}  // namespace batchlas::device

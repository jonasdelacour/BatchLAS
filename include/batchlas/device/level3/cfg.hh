// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Configurations of the in-group level 3 operations: the tile geometry, the local stage it
/// needs and the executor size it requires (@ref api_device_blas3).
///
/// A configuration is a compile-time choice, never derived from a run-time m, n or k:
///   - `l3::RegGeom<TM, TN, TK, TR, TC, UnrollK, Stages>`: the register-tiled tile geometry.
///   - `l3::cfg::RegTiled<T, TM, TN, TK, TR, TC, Stages, UnrollK, Ops>`: a geometry for scalar `T`
///     and an arithmetic policy; the `Cfg` every in-group operation takes.
///   - `l3::smem_elems<Cfg>()`, `l3::smem_bytes<Cfg>()`: the local stage an operation takes at the
///     start of its workspace's local memory.
///
/// Included by `<batchlas/device/level3/group.hh>`.
/// @ingroup api_device_blas3
// A run-time choice inside a kernel pays the max REG over its paths.

#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device::l3 {

/// @brief The register-tiled tile geometry: a `TM` x `TN` output tile in `TK`-deep k slabs, each
/// item holding a `TR` x `TC` register tile.
///
/// The work-group is `(TM / TR) x (TN / TC)` items. `op(A)`'s slab slot `(r, c)` is at
/// `c * TileAStride + r` and `op(B)`'s slab slot `(r, c)` at `b_slot(c, r)`.
/// @ingroup api_device_blas3
/// @tparam TM output tile rows.
/// @tparam TN output tile columns.
/// @tparam TK k-slab depth.
/// @tparam TR register tile rows per item; divides `TM`.
/// @tparam TC register tile columns per item (default `TR`); divides `TN`.
/// @tparam UnrollK_ k steps per inner iteration (default 1).
/// @tparam Stages_ local stages: 1 (default) or 2 (double-buffered).
template <int TM, int TN, int TK, int TR, int TC = TR, int UnrollK_ = 1, int Stages_ = 1> struct RegGeom {
    /// Output tile rows.
    static constexpr int TileM = TM;
    /// Output tile columns.
    static constexpr int TileN = TN;
    /// k-slab depth.
    static constexpr int TileK = TK;
    /// Register tile rows per item.
    static constexpr int ThreadTileRows = TR;
    /// Register tile columns per item.
    static constexpr int ThreadTileCols = TC;
    /// k steps per inner iteration.
    static constexpr int UnrollK = UnrollK_;
    /// Local stages (1 or 2).
    static constexpr int Stages = Stages_;
    /// Items along the tile's rows.
    static constexpr int LocalRows = TileM / ThreadTileRows;
    /// Items along the tile's columns.
    static constexpr int LocalCols = TileN / ThreadTileCols;
    /// Work-group size the geometry needs.
    static constexpr int ThreadsPerGroup = LocalRows * LocalCols;
    /// Column stride of the `op(A)` slab in local memory (padded by one).
    static constexpr int TileAStride = TileM + 1;
    /// Column stride of the `op(B)` slab in local memory (padded by one).
    static constexpr int TileBStride = TileK + 1;
    /// Elements of one `op(A)` stage.
    static constexpr int StageASize = TileAStride * TileK;
    /// Elements of one `op(B)` stage.
    static constexpr int StageBSize = TileBStride * TileN;
    /// @brief The local slot of `op(B)`'s slab element `(row, col)`, swizzled against bank conflicts.
    /// @param col the column in the tile.
    /// @param row the row in the slab.
    /// @return the slot index in one `op(B)` stage.
    // B slot of (k row, n col). With ThreadTileCols 4 and an odd stride, local_col and
    // local_col + 8 read the same bank at every t (64-wide tiles: 2-way; 32x128: 4-way), and
    // no stride pad can fix that (4 * stride * 8 == 0 mod 32). XOR-ing the row with the col's
    // 32-column group stays inside the row's aligned 4-block, so it needs no extra memory.
    // evidence: docs/perf/gemm.md#gemm-wp-g-transpose-aware-scalar-staging
    static constexpr int b_slot(int col, int row) {
        if constexpr (TileN <= 32) return col * TileBStride + row;  // one group: the compiler cannot see q == 0
        else return col * TileBStride + (row ^ ((col >> 5) & 3));
    }
};

/// @brief The arithmetic policy a configuration defaults to: `arith::Native` for a real `Scalar`,
/// `arith::Explicit` for a complex one.
///
/// With `arith::Native` a float in-group gemm is bit-identical to the host's register-tiled route.
/// @ingroup api_device_blas3
/// @tparam Scalar the element type.
template <class Scalar> using default_ops_t = std::conditional_t<Arith<Scalar>::is_complex, arith::Explicit, arith::Native>;

namespace cfg {
/// @brief The register-tiled configuration: a `RegGeom` for element type `Scalar` and an
/// arithmetic policy, the `Cfg` of every in-group level 3 operation.
///
/// The operation's executor must have exactly `threads` items (a `WorkGroup<threads, SG>`), and
/// its workspace must hold `smem_bytes<RegTiled>()` bytes of local memory at offset 0. The
/// constraints below are checked at compile time.
/// @ingroup api_device_blas3
/// @tparam Scalar the element type.
/// @tparam TM output tile rows.
/// @tparam TN output tile columns.
/// @tparam TK k-slab depth, a multiple of 4; `TM * TK` and `TN * TK` are at least `threads`.
/// @tparam TR register tile rows per item; divides `TM`.
/// @tparam TC register tile columns per item (default `TR`); divides `TN` and 32.
/// @tparam Stages_ local stages: 1 (default) or 2.
/// @tparam UnrollK_ k steps per inner iteration, `>= 1` (default 1).
/// @tparam Ops_ the arithmetic policy (default `default_ops_t<Scalar>`).
template <class Scalar, int TM, int TN, int TK, int TR, int TC = TR, int Stages_ = 1, int UnrollK_ = 1, class Ops_ = default_ops_t<Scalar>>
struct RegTiled : RegGeom<TM, TN, TK, TR, TC, UnrollK_, Stages_> {
    /// The geometry.
    using G = RegGeom<TM, TN, TK, TR, TC, UnrollK_, Stages_>;
    /// The element type.
    using value_type = Scalar;
    /// The arithmetic policy.
    using ops = Ops_;
    /// The geometry (as `G`).
    using geom = G;
    /// The executor size the configuration requires.
    static constexpr int threads = G::ThreadsPerGroup;
    /// Elements of `value_type` in the local stage.
    static constexpr int smem_elems = G::Stages * (G::StageASize + G::StageBSize);
    static_assert(TM % TR == 0 && TN % TC == 0, "l3::cfg::RegTiled: TR divides TM and TC divides TN");
    static_assert(TM * TK >= threads && TN * TK >= threads, "l3::cfg::RegTiled: each slab covers every item");
    static_assert(TK % 4 == 0 && 32 % TC == 0, "l3::cfg::RegTiled: b_slot's swizzle needs TK % 4 == 0 and 32 % TC == 0");
    static_assert(Stages_ == 1 || Stages_ == 2, "l3::cfg::RegTiled: one or two local stages");
    static_assert(UnrollK_ >= 1, "l3::cfg::RegTiled: UnrollK >= 1");
};

/// @brief `true` when `C` is a `RegTiled` configuration.
/// @ingroup api_device_lowlevel
template <class C> inline constexpr bool is_reg_tiled_v = false;
/// @brief `true` when `C` is a `RegTiled` configuration.
/// @ingroup api_device_lowlevel
template <class Scalar, int TM, int TN, int TK, int TR, int TC, int S, int U, class O>
inline constexpr bool is_reg_tiled_v<RegTiled<Scalar, TM, TN, TK, TR, TC, S, U, O>> = true;
}  // namespace cfg

/// @brief Satisfied by an in-group level 3 configuration (a `cfg::RegTiled`).
/// @ingroup api_device_blas3
template <class C> concept Config = cfg::is_reg_tiled_v<C>;

/// @brief Elements of `C::value_type` in configuration `C`'s local stage.
/// @ingroup api_device_blas3
/// @tparam C a configuration.
template <class C> inline constexpr int smem_elems() { return C::smem_elems; }
/// @brief Bytes of local memory configuration `C`'s stage needs at the start of the workspace's local memory.
///
/// Size the kernel's local memory (for example a `local_accessor` or `local_workspace`) with it.
/// @ingroup api_device_blas3
/// @tparam C a configuration.
template <class C> inline constexpr std::size_t smem_bytes() { return sizeof(typename C::value_type) * std::size_t(C::smem_elems); }

}  // namespace batchlas::device::l3

namespace batchlas::device::detail {
// An L3 op's stage: Cfg::smem_elems elements of Cfg::value_type at the start of ws.local, kept
// decorated (a generic T* would leave the address space to inference).
template <class C> struct Smem {
    local_mptr<typename C::value_type> p;
};
template <class C> BATCHLAS_DEVICE_INLINE Smem<C> smem_of(const Workspace& ws) {
    using T = typename C::value_type;
    return {carve_local(ws, WsSlot<T>{WsSpace::Local, 0, static_cast<std::size_t>(C::smem_elems)})};
}
}  // namespace batchlas::device::detail

// SPDX-License-Identifier: MIT
#pragma once
// Level 3 configurations (v22 design B.2 "Cfg presets"): one compile-time choice per component,
// the local memory it needs and the executor size it requires. Never derived from a run-time
// m, n or k (a run-time choice inside a kernel pays the max REG over its paths).
//   RegGeom<TM, TN, TK, TR, TC, UnrollK, Stages>  the register-tiled tile (GemmRegisterTiledKernel)
//   cfg::RegTiled<T, TM, TN, TK, TR, TC, Stages, UnrollK, Ops>  RegGeom for scalar T and an Arith policy
//   Smem<Cfg>, smem_elems<Cfg>(), smem_bytes<Cfg>()  the caller's local memory for one work-group

#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>

namespace batchlas::device::inline v2::l3 {

/// TM x TN output tile, TK-deep k slabs, each item a TR x TC register tile; Stages local stages
/// (1, or 2 double-buffered), UnrollK k steps per inner iteration. op(A) slab slot (r, c) at
/// c * TileAStride + r, op(B) slab slot (r, c) at b_slot(c, r).
template <int TM, int TN, int TK, int TR, int TC = TR, int UnrollK_ = 1, int Stages_ = 1> struct RegGeom {
    static constexpr int TileM = TM;
    static constexpr int TileN = TN;
    static constexpr int TileK = TK;
    static constexpr int ThreadTileRows = TR;
    static constexpr int ThreadTileCols = TC;
    static constexpr int UnrollK = UnrollK_;
    static constexpr int Stages = Stages_;
    static constexpr int LocalRows = TileM / ThreadTileRows;
    static constexpr int LocalCols = TileN / ThreadTileCols;
    static constexpr int ThreadsPerGroup = LocalRows * LocalCols;
    static constexpr int TileAStride = TileM + 1;
    static constexpr int TileBStride = TileK + 1;
    static constexpr int StageASize = TileAStride * TileK;
    static constexpr int StageBSize = TileBStride * TileN;
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

/// The Arith policy a configuration defaults to: Native for a real T (the tuned kernels' form: an
/// in-group gemm is bit-identical to the host reg route), Explicit for a complex T (no Annex-G).
template <class T> using default_ops_t = std::conditional_t<Arith<T>::is_complex, arith::Explicit, arith::Native>;

namespace cfg {
/// The register-tiled preset (design C.9 RegTiled): RegGeom for T, executor size threads.
template <class T, int TM, int TN, int TK, int TR, int TC = TR, int Stages_ = 1, int UnrollK_ = 1, class Ops_ = default_ops_t<T>>
struct RegTiled : RegGeom<TM, TN, TK, TR, TC, UnrollK_, Stages_> {
    using G = RegGeom<TM, TN, TK, TR, TC, UnrollK_, Stages_>;
    using value_type = T;
    using ops = Ops_;
    using geom = G;
    static constexpr int threads = G::ThreadsPerGroup;
    static constexpr int smem_elems = G::Stages * (G::StageASize + G::StageBSize);
    static_assert(TM % TR == 0 && TN % TC == 0, "l3::cfg::RegTiled: TR divides TM and TC divides TN");
    static_assert(TM * TK >= threads && TN * TK >= threads, "l3::cfg::RegTiled: each slab covers every item");
    static_assert(TK % 4 == 0 && 32 % TC == 0, "l3::cfg::RegTiled: b_slot's swizzle needs TK % 4 == 0 and 32 % TC == 0");
    static_assert(Stages_ == 1 || Stages_ == 2, "l3::cfg::RegTiled: one or two local stages");
    static_assert(UnrollK_ >= 1, "l3::cfg::RegTiled: UnrollK >= 1");
};

template <class C> inline constexpr bool is_reg_tiled_v = false;
template <class T, int TM, int TN, int TK, int TR, int TC, int S, int U, class O>
inline constexpr bool is_reg_tiled_v<RegTiled<T, TM, TN, TK, TR, TC, S, U, O>> = true;
}  // namespace cfg

template <class C> concept Config = cfg::is_reg_tiled_v<C>;

/// The caller's local memory for one work-group: Cfg::smem_elems elements of Cfg::value_type
/// (a raw pointer into a local_accessor or a Workspace's local slice).
template <class C> struct Smem {
    typename C::value_type* p;
};
template <class C> inline constexpr int smem_elems() { return C::smem_elems; }
template <class C> inline constexpr std::size_t smem_bytes() { return sizeof(typename C::value_type) * std::size_t(C::smem_elems); }
/// Smem over a local_accessor of Cfg::value_type (at least smem_elems<Cfg>() long).
template <class C, class Acc> BATCHLAS_DLA_INLINE Smem<C> smem_of(const Acc& acc) {
    return {acc.template get_multi_ptr<sycl::access::decorated::no>().get()};
}

}  // namespace batchlas::device::inline v2::l3

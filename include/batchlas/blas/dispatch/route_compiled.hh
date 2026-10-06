#pragma once

/// @file
/// @brief Is the kernel for a native route actually LINKED into this build?
///
/// A different question from the device family: the level-3 tile kernels are
/// portable SYCL, but their translation units are compiled only when cuBLAS is
/// present, so `B == Backend::CUDA` claims a kernel the vendor-free build lacks.
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#the-vendor-gate

#include <batchlas/backend_config.h>

#include <batchlas/blas/enums.hh>

#include <type_traits>

namespace batchlas::dispatch {

/// @brief True when the tile-masked / expand-then-gemm level-3 routes are linked for `(B, T)`.
///
/// Covers syrk's GramTiles and TriangularTiles, syr2k's and trmm's
/// TriangularTiles, and symm's ExpandGemm. The float routes are wired on CUDA in
/// every build; the other scalar types only when cuBLAS is compiled in. No other
/// backend has them.
/// @invariant Per (backend, scalar), never a bare `true` or a backend test alone:
///            either would claim a kernel where none is linked, and the
///            vendor-free call would throw.
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#the-vendor-gate-why-the-tile-route-predicate-is-per-backend-and-scalar
template <Backend B, typename T>
inline constexpr bool level3_tile_route_available =
    B == Backend::CUDA && (std::is_same_v<T, float> || bool(BATCHLAS_HAS_CUBLAS));

} // namespace batchlas::dispatch

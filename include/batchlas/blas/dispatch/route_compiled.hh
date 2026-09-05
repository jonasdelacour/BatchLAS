#pragma once

// Is the kernel for a native route actually LINKED into this build?
//
// This is the question four sites in src/extensions/ were asking with
// `B == Backend::CUDA`, and one of them says so in its own comment:
//
//     // The routes whose trmm reaches the batched triangular tile kernel. Not a
//     // statement about CUDA the vendor -- it is where the kernel is wired.
//     constexpr bool route_has_tile_kernel = (B == Backend::CUDA);
//
// The comment is right and the expression is wrong, in a way that matters. The
// tile kernels are portable SYCL -- verified by compiling triangular_expand.hh
// and the *_tiles.hh family standalone at -fsycl-targets=spir64_x86_64 -- so
// nothing about them is CUDA. What is true is that they live in
// {symm,syrk,syr2k,trmm}_custom_dispatch.cc, which src/backends/CMakeLists.txt
// compiles only when cuBLAS is present, because their dispatch still terminates
// in *_vendor_cuda_raw. So the real condition is "that TU is in this build".
//
// Spelling it that way is not cosmetic. `B == Backend::CUDA` is wrong the moment
// the kernels move (WP1) or a second backend gains them, and it is wrong TODAY
// in the vendor-free build: with -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF the backend
// is still Backend::CUDA, the tile TUs are NOT compiled, and every one of these
// sites would claim a kernel that is not linked.

#include <batchlas/backend_config.h>

#include <batchlas/blas/enums.hh>

#include <type_traits>

namespace batchlas::dispatch {

// The tile-masked / expand-then-gemm level-3 routes: syrk's GramTiles and
// TriangularTiles, syr2k's and trmm's TriangularTiles, symm's ExpandGemm.
//
// WP1 S7. THE PREDICTION IN THIS FILE WAS WRONG, and the way it was wrong is
// worth keeping. It said that once WP1 freed the four TUs, "this becomes true
// for every backend -- and that is the only edit needed here". Flipping it to a
// bare `true` would have been a bug in two directions at once:
//
//   * TOO WIDE IN TYPE. The float routes are now reachable everywhere, because
//     S6 put their gate in the facade. The double and complex routes are NOT:
//     syrk's non-float gram branch and trmm's non-float tile branch stayed in
//     cublas.cc, reachable only when cuBLAS is compiled. And syr2k has no
//     non-float tile route at ALL -- syr2k_triangular_tiles has exactly one
//     call site in the whole tree, in the float-only dispatcher. A bare `true`
//     tells ortho.cc's gram_via_syrk (which admits double) that a kernel exists
//     where it does not, and vendor-free that call throws.
//
//   * TOO WIDE IN BACKEND. The facade gate is guarded on Backend::CUDA, so
//     nothing is wired for ROCM or NETLIB. Claiming otherwise re-introduces
//     exactly the defect WP0 S8 removed -- the Backend enum standing in for a
//     question it does not answer.
//
// So the question grows a second parameter instead. It is now per (backend,
// scalar), because that is the granularity at which the answer actually varies.
//
// Vendor-present behaviour is unchanged by construction: with BATCHLAS_HAS_CUBLAS
// true this is true for every T on CUDA, exactly as before. Only the vendor-free
// float case moves, which is the WP1 gain.
template <Backend B, typename T>
inline constexpr bool level3_tile_route_available =
    B == Backend::CUDA && (std::is_same_v<T, float> || bool(BATCHLAS_HAS_CUBLAS));

} // namespace batchlas::dispatch

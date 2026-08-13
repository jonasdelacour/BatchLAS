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

namespace batchlas::dispatch {

// The tile-masked / expand-then-gemm level-3 routes: syrk's GramTiles and
// TriangularTiles, syr2k's and trmm's TriangularTiles, symm's ExpandGemm.
//
// Keyed on BATCHLAS_HAS_CUBLAS because that is the gate on the TUs that carry
// them (src/backends/CMakeLists.txt), NOT because the kernels are CUDA. When
// WP1 routes their terminal *_vendor_cuda_raw call through the public gemm
// entry point, those TUs stop depending on cuBLAS and this becomes true for
// every backend -- and that is the only edit needed here.
template <Backend B>
inline constexpr bool level3_tile_kernels_compiled =
    B == Backend::CUDA ? bool(BATCHLAS_HAS_CUBLAS) : false;

} // namespace batchlas::dispatch

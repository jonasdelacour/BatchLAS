set(BACKEND_COMMON_SOURCES
    backend_handle_instantiations.cc
    # WP1 S1. The seam where a level-3 tile route gives up and asks for the
    # vendor. Compiled in EVERY configuration -- including the one with no CUDA
    # object library at all -- because it is what lets the four *_custom_dispatch
    # TUs stop naming cuBLAS symbols. Its own #if picks vendor-or-throw; see
    # level3_vendor_fallback.hh for why this cannot just call the public entry
    # point (it would recurse: every call site is reached AFTER a gate that
    # already returned true).
    level3_vendor_fallback.cc
)

set(BACKEND_HOST_SOURCES)
set(BACKEND_CUDA_SOURCES)
set(BACKEND_ROCM_SOURCES)

# WP0 S3: gate each vendor translation unit on ITS OWN library, not on the
# device family. netlib_lapack.cc calls LAPACKE and CBLAS, so it belongs to
# those libraries; cublas.cc calls cuBLAS; cusolver.cc calls cuSOLVER; and so
# on. Under the old family gate, "there is a CUDA device" compiled all four
# NVIDIA TUs whether or not their libraries existed -- which is why cuSOLVER and
# cuSPARSE were linked without ever being probed (S1).
#
# On a full-vendor box every condition below is true and the source lists are
# unchanged, so this step is bit-identical here. Turning a library off does not
# yet produce a LINKABLE build: the public op definitions still live inside
# these vendor TUs (e.g. gemm<Back,T> at cublas.cc:1568-1580), so dropping a TU
# drops those definitions too. Moving them into a vendor-independent facade is
# S5, and the no-vendor build links at S6.
if(BATCHLAS_HAS_LAPACKE AND BATCHLAS_HAS_CBLAS)
    list(APPEND BACKEND_HOST_SOURCES
        netlib_lapack.cc
    )
endif()

# mkl.cc is DELETED as of S5, along with this block.
#
# It had never been compiled -- the list(APPEND) here was commented out for its
# whole history -- yet it carried its own copies of the public gemm/gemv/trsm
# definitions. S5 moves those definitions out of every vendor TU, and keeping a
# fourth, uncompiled, silently diverging copy would have made each remaining
# op's move harder for no benefit. It is also one of the four TUs named in the
# spec's Risk 1, so deleting it removes a quarter of that risk outright.
#
# Backend::MKL the ENUM VALUE is untouched: it is the GPU-backend alias in
# tests/test_utils.hh and across the benchmarks, so retiring it is a separate,
# wider change (the spec's Backend-narrowing), not part of the facade move.

if(BATCHLAS_HAS_CUBLAS)
    list(APPEND BACKEND_CUDA_SOURCES cublas.cc)
endif()
if(BATCHLAS_HAS_CUSOLVER)
    list(APPEND BACKEND_CUDA_SOURCES cusolver.cc)
endif()
if(BATCHLAS_HAS_CUSPARSE)
    list(APPEND BACKEND_CUDA_SOURCES cusparse.cc)
endif()

# The cuBLASDx/cuSolverDx wrapper TUs are gated on cuBLAS/cuSOLVER, NOT on
# BATCHLAS_HAS_CUBLASDX.
#
# That looks wrong and is deliberate. These files are compiled TODAY on this
# machine even though MathDx is absent and BATCHLAS_ENABLE_CUBLASDX_WRAPPER is
# OFF: each one carries its own internal #if and, with MathDx missing, still
# emits the fallback definitions that gemm_cublasdx_dispatch.cc and the four
# *_custom_dispatch.cc TUs call. Gating them on BATCHLAS_HAS_CUBLASDX (which is
# 0 here) therefore does not remove dead code -- it removes symbols that live
# callers reference, and the link fails. Verified against the baseline build
# log, which shows all five .cu TUs compiling with MathDx absent.
#
# Making these genuinely optional means first routing their callers' terminal
# GEMM through the public entry point; that is WP1, not S3.
if(BATCHLAS_HAS_CUBLAS)
    list(APPEND BACKEND_CUDA_SOURCES
        gemm_cublasdx.cu
        symm_cublasdx_fused.cu
        syrk_cublasdx_fused.cu
        syr2k_cublasdx_fused.cu
        trmm_cublasdx_fused.cu
    )
endif()
if(BATCHLAS_HAS_CUSOLVER)
    list(APPEND BACKEND_CUDA_SOURCES
        cusolverdx.cc
        cusolverdx_kernels.cu
    )
endif()

# These four are the portable expand-then-gemm / tile-masked level-3 routes, and
# the dispatch shims that pick between them and the vendor. They are SYCL, not
# CUDA -- see the standalone-compile check on triangular_expand.hh and the
# *_tiles.hh family -- but they still call *_vendor_cuda_raw and the cuBLASDx
# entry points, so they cannot leave the CUDA object library until that terminal
# call is routed through the public gemm entry point (WP1). Keep them with
# cuBLAS for now, which is the library they actually depend on.
if(BATCHLAS_HAS_CUBLAS)
    list(APPEND BACKEND_CUDA_SOURCES
        gemm_cublasdx_dispatch.cc
        symm_custom_dispatch.cc
        syrk_custom_dispatch.cc
        syr2k_custom_dispatch.cc
        trmm_custom_dispatch.cc
    )
endif()

if(BATCHLAS_HAS_ROCBLAS)
    list(APPEND BACKEND_ROCM_SOURCES rocblas.cc)
endif()
if(BATCHLAS_HAS_ROCSOLVER)
    list(APPEND BACKEND_ROCM_SOURCES rocsolver.cc)
endif()
if(BATCHLAS_HAS_ROCSPARSE)
    list(APPEND BACKEND_ROCM_SOURCES rocsparse.cc)
endif()

target_sources(batchlas_backends_obj PRIVATE ${BACKEND_COMMON_SOURCES} ${BACKEND_HOST_SOURCES})

if(TARGET batchlas_backends_cuda_obj)
    target_sources(batchlas_backends_cuda_obj PRIVATE ${BACKEND_CUDA_SOURCES})
endif()

if(TARGET batchlas_backends_rocm_obj)
    target_sources(batchlas_backends_rocm_obj PRIVATE ${BACKEND_ROCM_SOURCES})
endif()
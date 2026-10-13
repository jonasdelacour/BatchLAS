set(BACKEND_COMMON_SOURCES
    backend_handle_instantiations.cc
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

# The three CUDA TUs are one group: cublas.cc makes cusolverDn* calls through the
# LinalgHandle<Backend::CUDA> that needs all three libraries (linalg-impl.hh), and
# cusparse.cc borrows cuBLAS's compute-type enum. BatchLASDependencies.cmake rejects a
# partial set, so BATCHLAS_HAS_CUDA_VENDOR_TUS is the one gate.
if(BATCHLAS_HAS_CUDA_VENDOR_TUS)
    list(APPEND BACKEND_CUDA_SOURCES cublas.cc cusolver.cc cusparse.cc)
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
# The vendor-independent facade: the public entry points themselves.
#
# Deliberately NOT part of batchlas_backends_obj, and deliberately not gated on
# any vendor library. These TUs define `batchlas::gemm<Back, T>` and friends,
# which used to be defined inside cublas.cc / rocblas.cc / netlib_lapack.cc --
# so turning a vendor library off removed the public entry point along with the
# vendor path. This directory is what makes the two separable.
#
# It compiles unconditionally. Which BACKENDS get instantiated is decided inside
# each TU by the per-library BATCHLAS_HAS_* macros.
target_sources(batchlas_dispatch_obj PRIVATE
    entry_points/level3.cc
    entry_points/factorization.cc
)

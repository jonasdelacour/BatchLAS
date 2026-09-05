# One device-link unit. Keeping it small matters: this is the .so whose device
# link ptxas runs on, and it already measures ~43.4 s with gemm_kernels.cc alone
# (scripts/register_probe.sh). Measure the delta when adding to it.
set(SYCL_SOURCES
    gemm_kernels.cc
    trsm_native.cc
)

target_sources(batchlas_sycl_obj PRIVATE ${SYCL_SOURCES})

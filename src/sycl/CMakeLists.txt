# One device-link unit. Keeping it small matters: this is the .so whose device
# link ptxas runs on, and it already measures ~43.4 s with gemm_kernels.cc alone
# (scripts/register_probe.sh). Measure the delta when adding to it.
#
# gemv_native.cc goes HERE, in the ordinary list, and not into a
# NO_CPU_TARGETS object library, however tempting the link time makes that.
# batchlas_sycl_obj is built with batchlas_apply_object_options() with no
# NO_CPU_TARGETS flag (src/CMakeLists.txt:68-74), i.e. for
# -fsycl-targets=nvidia_gpu_sm_89,native_cpu. The native_cpu image is the whole
# WP7 deliverable: 20 of the 40 vendor-free gemv_tests failures are
# GemvMatrixViewTest/0..3 on a native_cpu Device("cpu") queue, and a GPU-only
# native gemv closes the other 20 while leaving the suite red and the burn-down
# unmoved. Measure the device-link delta and report it; do not pay for it by
# dropping the CPU target.
set(SYCL_SOURCES
    gemm_kernels.cc
    trsm_native.cc
    gemv_native.cc
)

target_sources(batchlas_sycl_obj PRIVATE ${SYCL_SOURCES})

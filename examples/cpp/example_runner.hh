#pragma once

// Backend/device selection shared by examples 02-12.
//
// `01_getting_started.cc` spells this logic out inline, because choosing a
// backend is one of the things you have to understand to use the C++ API. The
// rest of the examples use this header so they can get on with their subject.
//
// This header is not part of the BatchLAS API.

#include <string>

#include <batchlas/backend_config.h>
#include <blas/enums.hh>
#include <util/sycl-device-queue.hh>

#include "example_common.hh"

namespace examples {

// The GPU backend this build was compiled with, if any.
#if BATCHLAS_HAS_CUDA_BACKEND
inline constexpr batchlas::Backend gpu_backend = batchlas::Backend::CUDA;
inline constexpr const char* gpu_backend_name = "CUDA";
#elif BATCHLAS_HAS_ROCM_BACKEND
inline constexpr batchlas::Backend gpu_backend = batchlas::Backend::ROCM;
inline constexpr const char* gpu_backend_name = "ROCM";
#elif BATCHLAS_HAS_MKL_BACKEND
inline constexpr batchlas::Backend gpu_backend = batchlas::Backend::MKL;
inline constexpr const char* gpu_backend_name = "MKL";
#endif

inline bool on_gpu(const Queue& ctx) { return ctx.device().type == DeviceType::GPU; }

// The CTA routines are instantiated for GPU backends only — calling one with
// Backend::NETLIB is a link error, not a runtime one, so guard with
// `if constexpr` before the runtime device check.
template <batchlas::Backend B>
inline constexpr bool has_cta_variants = (B != batchlas::Backend::NETLIB);

// Several routines map one work-group onto one matrix and need a sub-group
// width of 32, which no CPU device provides. Examples use this to skip them
// rather than fail.
inline bool supports_cta(const Queue& ctx) {
    return on_gpu(ctx) && ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE) >= 32;
}

// Selects a backend at compile time and a device to match, then calls
// Body<B>::run(ctx). Pass "cpu" on the command line to force the host path.
template <template <batchlas::Backend> class Body>
int run_example(int argc, char** argv, const std::string& title) {
    header(title);

    const bool force_cpu = (argc > 1 && std::string(argv[1]) == "cpu");

#if BATCHLAS_HAS_GPU_BACKEND
    if (!force_cpu && !Device::get_devices(DeviceType::GPU).empty()) {
        Queue ctx("gpu", /*in_order=*/true);
        report("backend", std::string(gpu_backend_name) + " on " + ctx.device().get_name());
        Body<gpu_backend>::run(ctx);
        return exit_code();
    }
#else
    (void)force_cpu;
#endif

#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    Queue ctx("cpu", /*in_order=*/true);
    report("backend", std::string("NETLIB on ") + ctx.device().get_name());
    Body<batchlas::Backend::NETLIB>::run(ctx);
    return exit_code();
#else
    std::cout << "\nNo usable backend/device combination in this build.\n";
    return 0;
#endif
}

}  // namespace examples

// Boilerplate for an example whose body is `template <Backend B> struct Example`.
#define BATCHLAS_EXAMPLE_MAIN(TITLE)                                  \
    int main(int argc, char** argv) {                                 \
        return examples::run_example<Example>(argc, argv, TITLE);     \
    }

#pragma once

// Small shared helpers: section headings, and the backend/device selection the
// examples all need. Nothing here is part of the BatchLAS API.
//
// Example 01 spells the backend selection out inline instead of using this
// header, because picking a backend is one of the things you have to
// understand to use the C++ interface.

#include <iostream>
#include <string>

#include <batchlas/backend_config.h>
#include <blas/enums.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

namespace examples {

inline void header(const std::string& title) {
    std::cout << "\n" << title << "\n" << std::string(title.size(), '=') << "\n";
}

inline void section(const std::string& title) { std::cout << "\n--- " << title << " ---\n"; }

// "name: value"
template <typename T>
void print(const std::string& name, const T& value) {
    std::cout << name << ": " << value << "\n";
}

inline void print(const std::string& name, bool value) {
    std::cout << name << ": " << (value ? "true" : "false") << "\n";
}

// The first `count` entries of a span, on one line.
template <typename T>
void print_values(const std::string& name, Span<T> values, int count = 6) {
    std::cout << name << ":";
    for (int i = 0; i < count && i < static_cast<int>(values.size()); ++i) std::cout << " " << values[i];
    if (static_cast<int>(values.size()) > count) std::cout << " ...";
    std::cout << "\n";
}

// Something this device or backend cannot do.
inline void skip(const std::string& what, const std::string& why) {
    std::cout << "(skipped) " << what << ": " << why << "\n";
}

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

// The *_cta routines put one work-group on one matrix and need a sub-group
// width of 32, so they are GPU-only and capped at n <= 32.
inline bool supports_cta(const Queue& ctx) {
    return on_gpu(ctx) && ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE) >= 32;
}

// They are also instantiated for GPU backends only, so calling one with
// Backend::NETLIB is a *link* error — guard with `if constexpr`, not `if`.
template <batchlas::Backend B>
inline constexpr bool has_cta_variants = (B != batchlas::Backend::NETLIB);

// Picks a backend at compile time and a device to match, then calls
// Body<B>::run(ctx). Pass "cpu" on the command line to force the host path.
template <template <batchlas::Backend> class Body>
int run_example(int argc, char** argv, const std::string& title) {
    header(title);
    const bool force_cpu = (argc > 1 && std::string(argv[1]) == "cpu");

#if BATCHLAS_HAS_GPU_BACKEND
    if (!force_cpu && !Device::get_devices(DeviceType::GPU).empty()) {
        Queue ctx("gpu", /*in_order=*/true);
        print("backend", std::string(gpu_backend_name) + " on " + ctx.device().get_name());
        Body<gpu_backend>::run(ctx);
        return 0;
    }
#else
    (void)force_cpu;
#endif

#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    Queue ctx("cpu", /*in_order=*/true);
    print("backend", std::string("NETLIB on ") + ctx.device().get_name());
    Body<batchlas::Backend::NETLIB>::run(ctx);
    return 0;
#else
    std::cout << "\nNo usable backend/device combination in this build.\n";
    return 0;
#endif
}

}  // namespace examples

// For an example whose body is `template <Backend B> struct Example`.
#define BATCHLAS_EXAMPLE_MAIN(TITLE)                              \
    int main(int argc, char** argv) {                             \
        return examples::run_example<Example>(argc, argv, TITLE); \
    }

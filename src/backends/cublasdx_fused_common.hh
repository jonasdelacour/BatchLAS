#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <initializer_list>

#if defined(BATCHLAS_ENABLE_CUBLASDX_WRAPPER) && __has_include(<cublasdx.hpp>)
    #define CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION
    #define CUBLASDX_MAKE_TMA_ATOM cute::make_tma_atom
    #define CUBLASDX_PIPELINE_EXECUTION __device__ __forceinline__
    #define CUBLASDX_DEVICE_PIPELINE_EXECUTION __host__ __device__ __forceinline__
    #include <cublasdx.hpp>
    #undef CUBLASDX_DEVICE_PIPELINE_EXECUTION
    #undef CUBLASDX_PIPELINE_EXECUTION
    #undef CUBLASDX_MAKE_TMA_ATOM
    #undef CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION
    #define BATCHLAS_HAS_CUBLASDX_HEADER 1
#else
    #define BATCHLAS_HAS_CUBLASDX_HEADER 0
#endif

namespace batchlas::backend::detail {

constexpr int kCublasDxBlockSize = 256;
constexpr int kCublasDxSupportedSM = 890;

constexpr unsigned int round_up_unsigned(unsigned int value, unsigned int alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

inline bool is_16b_aligned(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % 16u) == 0u;
}

inline bool pointers_16b_aligned(std::initializer_list<const void*> pointers) {
    for (const void* ptr : pointers) {
        if (!is_16b_aligned(ptr)) {
            return false;
        }
    }
    return true;
}

inline bool multiples_of_four(std::initializer_list<int> values) {
    for (const int value : values) {
        if ((value % 4) != 0) {
            return false;
        }
    }
    return true;
}

inline int current_device_sm() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 0;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return 0;
    }

    return prop.major * 100 + prop.minor * 10;
}

inline bool current_device_sm_supported() {
    return current_device_sm() == kCublasDxSupportedSM;
}

inline int tile_extent_32_64(cublasdx_gemm::CuBLASDxGemmVariant variant) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
        return 32;
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TN:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NT:
        return 64;
    default:
        return 0;
    }
}

#if BATCHLAS_HAS_CUBLASDX_HEADER
template <class Kernel>
cudaError_t validate_launch_resources(Kernel kernel, unsigned int shared_memory_size) {
    cudaFuncAttributes attr{};
    const cudaError_t attr_status = cudaFuncGetAttributes(&attr, kernel);
    if (attr_status != cudaSuccess) {
        return attr_status;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return cudaErrorNotSupported;
    }

    int optin_limit = 0;
    const cudaError_t optin_status = cudaDeviceGetAttribute(&optin_limit,
                                                            cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                            device);
    if (optin_status != cudaSuccess) {
        return optin_status;
    }

    int default_limit = 0;
    const cudaError_t default_status = cudaDeviceGetAttribute(&default_limit,
                                                              cudaDevAttrMaxSharedMemoryPerBlock,
                                                              device);
    if (default_status != cudaSuccess) {
        return default_status;
    }

    int kernel_limit = attr.maxDynamicSharedSizeBytes;
    if (kernel_limit <= 0) {
        kernel_limit = optin_limit > 0 ? optin_limit : default_limit;
    }

    int supported_limit = kernel_limit;
    if (optin_limit > 0) {
        supported_limit = std::min(supported_limit, optin_limit);
    }
    if (supported_limit <= 0) {
        supported_limit = default_limit;
    }

    if (shared_memory_size > static_cast<unsigned int>(supported_limit)) {
        return cudaErrorNotSupported;
    }

    return cudaSuccess;
}
#endif

} // namespace batchlas::backend::detail
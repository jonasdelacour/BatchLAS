#include "symm_cublasdx_fused.hh"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

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

namespace batchlas::backend::symm_cublasdx {

namespace {

constexpr int kBlockSize = 256;
constexpr int kSupportedSM = 890;

#if BATCHLAS_HAS_CUBLASDX_HEADER
template <int Sm, int TileM, int TileN, int TileK>
using GemmBlock = decltype(
    cublasdx::Size<TileM, TileN, TileK>() +
    cublasdx::Precision<float>() +
    cublasdx::Type<cublasdx::type::real>() +
    cublasdx::MaxAlignment() +
    cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
    cublasdx::Function<cublasdx::function::MM>() +
    cublasdx::Block() +
    cublasdx::BlockDim<kBlockSize>() +
    cublasdx::SM<Sm>());
#endif

inline bool is_16b_aligned(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % 16u) == 0u;
}

inline bool has_max_alignment(const SymmLaunchDescriptor& desc) {
    return is_16b_aligned(desc.a_ptr) && is_16b_aligned(desc.b_ptr) && is_16b_aligned(desc.c_ptr) &&
        (desc.lda % 4) == 0 && (desc.ldb % 4) == 0 && (desc.ldc % 4) == 0 &&
        (desc.stride_a % 4) == 0 && (desc.stride_b % 4) == 0 && (desc.stride_c % 4) == 0;
}

inline bool current_device_sm_supported() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return false;
    }

    return prop.major * 100 + prop.minor * 10 == kSupportedSM;
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

template <Uplo FillMode>
__device__ float symmetric_value(const float* ptr, int ld, int row, int col) {
    int src_row = row;
    int src_col = col;
    if constexpr (FillMode == Uplo::Lower) {
        if (row < col) {
            src_row = col;
            src_col = row;
        }
    } else {
        if (row > col) {
            src_row = col;
            src_col = row;
        }
    }
    return ptr[src_col * ld + src_row];
}

template <typename Tensor>
__device__ void load_dense_tile(Tensor tensor,
                                const float* ptr,
                                int ld,
                                int row_start,
                                int col_start,
                                int rows,
                                int cols) {
    const int elems = rows * cols;
    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % rows;
        const int col = linear / rows;
        tensor(row, col) = ptr[(col_start + col) * ld + (row_start + row)];
    }
}

template <Uplo FillMode, typename Tensor>
__device__ void load_symmetric_tile(Tensor tensor,
                                    const float* ptr,
                                    int ld,
                                    int row_start,
                                    int col_start,
                                    int rows,
                                    int cols) {
    const int elems = rows * cols;
    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % rows;
        const int col = linear / rows;
        tensor(row, col) = symmetric_value<FillMode>(ptr, ld, row_start + row, col_start + col);
    }
}

template <class BLAS, int TileM, int TileN, int TileK, Side SideMode, Uplo FillMode>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void symm_cublasdx_kernel(SymmLaunchDescriptor desc) {
    extern __shared__ __align__(16) char smem[];

    const int batch_idx = static_cast<int>(blockIdx.z);
    const float* a_batch = desc.a_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_a);
    const float* b_batch = desc.b_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_b);
    float* c_batch = desc.c_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_c);

    auto global_c = cublasdx::make_gmem_tensor<cublasdx::col_major>(c_batch, desc.m, desc.n, desc.ldc);
    auto tile_c_gmem = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);

    const auto a_smem_layout = BLAS::suggest_layout_smem_a();
    const auto b_smem_layout = BLAS::suggest_layout_smem_b();
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<BLAS>(smem, a_smem_layout, b_smem_layout);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, a_smem_layout);
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, b_smem_layout);
    auto accumulator = BLAS::suggest_accumulator();

    const int tile_row = static_cast<int>(blockIdx.x) * TileM;
    const int tile_col = static_cast<int>(blockIdx.y) * TileN;
    const int stages = desc.k / TileK;

    for (int stage = 0; stage < stages; ++stage) {
        const int k_start = stage * TileK;
        if constexpr (SideMode == Side::Left) {
            load_symmetric_tile<FillMode>(a_shared_tensor, a_batch, desc.lda, tile_row, k_start, TileM, TileK);
            load_dense_tile(b_shared_tensor, b_batch, desc.ldb, k_start, tile_col, TileK, TileN);
        } else {
            load_dense_tile(a_shared_tensor, b_batch, desc.ldb, tile_row, k_start, TileM, TileK);
            load_symmetric_tile<FillMode>(b_shared_tensor, a_batch, desc.lda, k_start, tile_col, TileK, TileN);
        }

        __syncthreads();
        BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
        __syncthreads();
    }

    auto c_fragment = accumulator.make_partition_and_copy(tile_c_gmem);
    cublasdx::axpby(desc.alpha, accumulator.get_results(), desc.beta, c_fragment);
    accumulator.partition_and_copy(c_fragment, tile_c_gmem);
}

template <int Sm, int TileM, int TileN, int TileK, Side SideMode, Uplo FillMode>
cudaError_t launch_for_sm(const SymmLaunchDescriptor& desc, cudaStream_t stream) {
    using BLAS = GemmBlock<Sm, TileM, TileN, TileK>;
    auto kernel = symm_cublasdx_kernel<BLAS, TileM, TileN, TileK, SideMode, FillMode>;
    const auto a_smem_layout = BLAS::suggest_layout_smem_a();
    const auto b_smem_layout = BLAS::suggest_layout_smem_b();
    const unsigned int shared_memory_size = cublasdx::get_shared_storage_size_ab<BLAS>(a_smem_layout, b_smem_layout);

    const cudaError_t resource_status = validate_launch_resources(kernel, shared_memory_size);
    if (resource_status != cudaSuccess) {
        return resource_status;
    }

    const cudaError_t attr_status = cudaFuncSetAttribute(kernel,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         static_cast<int>(shared_memory_size));
    if (attr_status != cudaSuccess) {
        return attr_status;
    }

    const dim3 grid(static_cast<unsigned int>(desc.m / TileM),
                    static_cast<unsigned int>(desc.n / TileN),
                    static_cast<unsigned int>(desc.batch));
    kernel<<<grid, BLAS::block_dim, shared_memory_size, stream>>>(desc);
    return cudaGetLastError();
}

template <int TileM, int TileN, int TileK, Side SideMode, Uplo FillMode>
cudaError_t dispatch_family_for_current_sm(const SymmLaunchDescriptor& desc, cudaStream_t stream) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return cudaErrorNotSupported;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return cudaErrorNotSupported;
    }

    switch (prop.major * 100 + prop.minor * 10) {
    case kSupportedSM:
        return launch_for_sm<kSupportedSM, TileM, TileN, TileK, SideMode, FillMode>(desc, stream);
    default:
        return cudaErrorNotSupported;
    }
}

template <Side SideMode, Uplo FillMode>
cudaError_t dispatch_for_variant(cublasdx_gemm::CuBLASDxGemmVariant variant,
                                 const SymmLaunchDescriptor& desc,
                                 cudaStream_t stream) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
        return dispatch_family_for_current_sm<32, 32, 32, SideMode, FillMode>(desc, stream);
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
        return dispatch_family_for_current_sm<64, 64, 32, SideMode, FillMode>(desc, stream);
    default:
        return cudaErrorNotSupported;
    }
}
#endif

} // namespace

bool available() {
#if BATCHLAS_HAS_CUBLASDX_HEADER
    return true;
#else
    return false;
#endif
}

cudaError_t launch_float(cublasdx_gemm::CuBLASDxGemmVariant variant,
                         const SymmLaunchDescriptor& desc,
                         Side side,
                         Uplo uplo,
                         cudaStream_t stream) {
    if (!available() || !current_device_sm_supported()) {
        return cudaErrorNotSupported;
    }

    if (variant != cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN &&
        variant != cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN) {
        return cudaErrorNotSupported;
    }

    if (desc.batch <= 0 || desc.m <= 0 || desc.n <= 0 || desc.k <= 0) {
        return cudaErrorNotSupported;
    }

    const int tile_m = variant == cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN ? 64 : 32;
    const int tile_n = tile_m;
    const int tile_k = 32;
    if ((desc.m % tile_m) != 0 || (desc.n % tile_n) != 0 || (desc.k % tile_k) != 0) {
        return cudaErrorNotSupported;
    }

    if (!has_max_alignment(desc)) {
        return cudaErrorNotSupported;
    }

#if BATCHLAS_HAS_CUBLASDX_HEADER
    if (side == Side::Left && uplo == Uplo::Lower) {
        return dispatch_for_variant<Side::Left, Uplo::Lower>(variant, desc, stream);
    }
    if (side == Side::Left && uplo == Uplo::Upper) {
        return dispatch_for_variant<Side::Left, Uplo::Upper>(variant, desc, stream);
    }
    if (side == Side::Right && uplo == Uplo::Lower) {
        return dispatch_for_variant<Side::Right, Uplo::Lower>(variant, desc, stream);
    }
    if (side == Side::Right && uplo == Uplo::Upper) {
        return dispatch_for_variant<Side::Right, Uplo::Upper>(variant, desc, stream);
    }
#endif

    return cudaErrorNotSupported;
}

} // namespace batchlas::backend::symm_cublasdx
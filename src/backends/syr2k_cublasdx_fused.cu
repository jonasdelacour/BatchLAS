#include "syr2k_cublasdx_fused.hh"

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

namespace batchlas::backend::syr2k_cublasdx {

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

inline bool has_max_alignment(const Syr2kLaunchDescriptor& desc) {
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

inline int tile_extent(cublasdx_gemm::CuBLASDxGemmVariant variant) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
        return 32;
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

template <typename Tensor>
__device__ void load_swapped_source_tile(Tensor tensor,
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
        tensor(row, col) = ptr[(row_start + row) * ld + (col_start + col)];
    }
}

template <typename Tensor>
__device__ void load_transposed_lhs_tile(Tensor tensor,
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
        tensor(row, col) = ptr[(row_start + row) * ld + (col_start + col)];
    }
}

template <Uplo FillMode>
__device__ void mirror_symmetric_tile(float* ptr,
                                      int ld,
                                      int tile_row,
                                      int tile_col,
                                      int rows,
                                      int cols) {
    const bool diagonal_tile = tile_row == tile_col;
    const int elems = rows * cols;
    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % rows;
        const int col = linear / rows;

        const int global_row = tile_row + row;
        const int global_col = tile_col + col;
        bool source_entry = !diagonal_tile;
        if (diagonal_tile) {
            if constexpr (FillMode == Uplo::Lower) {
                source_entry = global_row > global_col;
            } else {
                source_entry = global_row < global_col;
            }
        }
        if (source_entry && global_row != global_col) {
            ptr[global_row * ld + global_col] = ptr[global_col * ld + global_row];
        }
    }
}

template <class BLAS, int TileM, int TileN, int TileK, Uplo FillMode, Transpose TransMode>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void syr2k_cublasdx_kernel(Syr2kLaunchDescriptor desc) {
    if constexpr (FillMode == Uplo::Lower) {
        if (blockIdx.x < blockIdx.y) {
            return;
        }
    } else {
        if (blockIdx.x > blockIdx.y) {
            return;
        }
    }

    extern __shared__ __align__(16) char smem[];

    const int batch_idx = static_cast<int>(blockIdx.z);
    const float* a_batch = desc.a_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_a);
    const float* b_batch = desc.b_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_b);
    float* c_batch = desc.c_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_c);

    auto global_c = cublasdx::make_gmem_tensor<cublasdx::col_major>(c_batch, desc.n, desc.n, desc.ldc);
    auto tile_c_gmem = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);

    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<BLAS>(smem);
    const auto a_smem_layout = BLAS::get_layout_smem_a();
    const auto b_smem_layout = BLAS::get_layout_smem_b();
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, a_smem_layout);
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, b_smem_layout);
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());
    auto accumulator = BLAS::suggest_accumulator();

    const int tile_row = static_cast<int>(blockIdx.x) * TileM;
    const int tile_col = static_cast<int>(blockIdx.y) * TileN;
    const int stages = desc.k / TileK;

    for (int stage = 0; stage < stages; ++stage) {
        const int k_start = stage * TileK;
        if constexpr (TransMode == Transpose::NoTrans) {
            load_dense_tile(a_shared_tensor, a_batch, desc.lda, tile_row, k_start, TileM, TileK);
            load_swapped_source_tile(b_shared_tensor, b_batch, desc.ldb, k_start, tile_col, TileK, TileN);
            __syncthreads();
            BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
            __syncthreads();

            load_dense_tile(a_shared_tensor, b_batch, desc.ldb, tile_row, k_start, TileM, TileK);
            load_swapped_source_tile(b_shared_tensor, a_batch, desc.lda, k_start, tile_col, TileK, TileN);
            __syncthreads();
            BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
            __syncthreads();
        } else {
            load_transposed_lhs_tile(a_shared_tensor, a_batch, desc.lda, tile_row, k_start, TileM, TileK);
            load_dense_tile(b_shared_tensor, b_batch, desc.ldb, k_start, tile_col, TileK, TileN);
            __syncthreads();
            BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
            __syncthreads();

            load_transposed_lhs_tile(a_shared_tensor, b_batch, desc.ldb, tile_row, k_start, TileM, TileK);
            load_dense_tile(b_shared_tensor, a_batch, desc.lda, k_start, tile_col, TileK, TileN);
            __syncthreads();
            BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
            __syncthreads();
        }
    }

    auto c_fragment = accumulator.make_partition_and_copy(tile_c_gmem);
    cublasdx::axpby(desc.alpha, accumulator.get_results(), desc.beta, c_fragment);
    accumulator.partition_and_copy(c_fragment, c_shared_tensor);
    __syncthreads();

    const bool diagonal_tile = tile_row == tile_col;
    const int elems = TileM * TileN;
    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        const int global_row = tile_row + row;
        const int global_col = tile_col + col;
        const float value = c_shared_tensor(row, col);
        c_batch[global_col * desc.ldc + global_row] = value;

        bool source_entry = !diagonal_tile;
        if (diagonal_tile) {
            if constexpr (FillMode == Uplo::Lower) {
                source_entry = global_row > global_col;
            } else {
                source_entry = global_row < global_col;
            }
        }
        if (source_entry && global_row != global_col) {
            c_batch[global_row * desc.ldc + global_col] = value;
        }
    }
}

template <int Sm, int TileM, int TileN, int TileK, Uplo FillMode, Transpose TransMode>
cudaError_t launch_for_sm(const Syr2kLaunchDescriptor& desc, cudaStream_t stream) {
    using BLAS = GemmBlock<Sm, TileM, TileN, TileK>;
    auto kernel = syr2k_cublasdx_kernel<BLAS, TileM, TileN, TileK, FillMode, TransMode>;
    const unsigned int shared_memory_size = cublasdx::get_shared_storage_size<BLAS>();

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

    const dim3 grid(static_cast<unsigned int>(desc.n / TileM),
                    static_cast<unsigned int>(desc.n / TileN),
                    static_cast<unsigned int>(desc.batch));
    kernel<<<grid, BLAS::block_dim, shared_memory_size, stream>>>(desc);
    return cudaGetLastError();
}

template <int TileM, int TileN, int TileK, Uplo FillMode, Transpose TransMode>
cudaError_t dispatch_family_for_current_sm(const Syr2kLaunchDescriptor& desc, cudaStream_t stream) {
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
        return launch_for_sm<kSupportedSM, TileM, TileN, TileK, FillMode, TransMode>(desc, stream);
    default:
        return cudaErrorNotSupported;
    }
}

template <Uplo FillMode, Transpose TransMode>
cudaError_t dispatch_for_variant(cublasdx_gemm::CuBLASDxGemmVariant variant,
                                 const Syr2kLaunchDescriptor& desc,
                                 cudaStream_t stream) {
    switch (tile_extent(variant)) {
    case 32:
        return dispatch_family_for_current_sm<32, 32, 32, FillMode, TransMode>(desc, stream);
    case 64:
        return dispatch_family_for_current_sm<64, 64, 32, FillMode, TransMode>(desc, stream);
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
                         const Syr2kLaunchDescriptor& desc,
                         Uplo uplo,
                         Transpose transA,
                         cudaStream_t stream) {
    if (!available() || !current_device_sm_supported()) {
        return cudaErrorNotSupported;
    }

    if (transA == Transpose::ConjTrans) {
        return cudaErrorNotSupported;
    }

    const int tile = tile_extent(variant);
    if (tile == 0 || desc.batch <= 0 || desc.n <= 0 || desc.k <= 0) {
        return cudaErrorNotSupported;
    }
    if ((desc.n % tile) != 0 || (desc.k % 32) != 0) {
        return cudaErrorNotSupported;
    }
    if (!has_max_alignment(desc)) {
        return cudaErrorNotSupported;
    }

#if BATCHLAS_HAS_CUBLASDX_HEADER
    if (uplo == Uplo::Lower && transA == Transpose::NoTrans) {
        return dispatch_for_variant<Uplo::Lower, Transpose::NoTrans>(variant, desc, stream);
    }
    if (uplo == Uplo::Upper && transA == Transpose::NoTrans) {
        return dispatch_for_variant<Uplo::Upper, Transpose::NoTrans>(variant, desc, stream);
    }
    if (uplo == Uplo::Lower && transA == Transpose::Trans) {
        return dispatch_for_variant<Uplo::Lower, Transpose::Trans>(variant, desc, stream);
    }
    if (uplo == Uplo::Upper && transA == Transpose::Trans) {
        return dispatch_for_variant<Uplo::Upper, Transpose::Trans>(variant, desc, stream);
    }
#else
    static_cast<void>(variant);
    static_cast<void>(desc);
    static_cast<void>(uplo);
    static_cast<void>(transA);
    static_cast<void>(stream);
#endif

    return cudaErrorNotSupported;
}

} // namespace batchlas::backend::syr2k_cublasdx
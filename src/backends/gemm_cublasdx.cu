#include "gemm_cublasdx.hh"

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

namespace batchlas::backend::cublasdx_gemm {

namespace {

constexpr int kBlockSize = 256;
constexpr int kSupportedSM = 890;

constexpr int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline int batch_dim(const int* batch_dims, int fallback, int batch_idx) {
    return batch_dims ? batch_dims[batch_idx] : fallback;
}

template <Transpose Op>
struct ArrangementTag;

template <>
struct ArrangementTag<Transpose::NoTrans> {
#if BATCHLAS_HAS_CUBLASDX_HEADER
    static constexpr auto value = cublasdx::col_major;
#endif
};

template <>
struct ArrangementTag<Transpose::Trans> {
#if BATCHLAS_HAS_CUBLASDX_HEADER
    static constexpr auto value = cublasdx::row_major;
#endif
};

#if BATCHLAS_HAS_CUBLASDX_HEADER
template <int Sm, int TileM, int TileN, int TileK, Transpose OpA, Transpose OpB>
using GemmBlock = decltype(
    cublasdx::Size<TileM, TileN, TileK>() +
    cublasdx::Precision<float>() +
    cublasdx::Type<cublasdx::type::real>() +
    cublasdx::MaxAlignment() +
    cublasdx::Arrangement<ArrangementTag<OpA>::value, ArrangementTag<OpB>::value, cublasdx::col_major>() +
    cublasdx::Function<cublasdx::function::MM>() +
    cublasdx::Block() +
    cublasdx::BlockDim<kBlockSize>() +
    cublasdx::SM<Sm>());

template <int Sm, int TileM, int TileN, int TileK, Transpose OpA, Transpose OpB>
using PipelinedGemmBlock = decltype(
    GemmBlock<Sm, TileM, TileN, TileK, OpA, OpB>() +
    cublasdx::WithPipeline());
#endif

struct CuBLASDxTileConfig {
    int tile_m;
    int tile_n;
    int tile_k;
};

inline CuBLASDxTileConfig tile_config(CuBLASDxGemmVariant variant) {
    switch (variant) {
    case CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
    case CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
    case CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
    case CuBLASDxGemmVariant::CuBLASDx32x32x32TT:
        return {32, 32, 32};
    case CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32TN:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32NT:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32TT:
        return {64, 64, 32};
    default:
        return {0, 0, 0};
    }
}

inline bool is_16b_aligned(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % 16u) == 0u;
}

constexpr unsigned int round_up_unsigned(unsigned int value, unsigned int alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

inline bool has_max_alignment(const GemmLaunchDescriptor& desc) {
    return is_16b_aligned(desc.a_ptr) && is_16b_aligned(desc.b_ptr) && is_16b_aligned(desc.c_ptr) &&
        (desc.lda % 4) == 0 && (desc.ldb % 4) == 0 && (desc.ldc % 4) == 0 &&
        (desc.stride_a % 4) == 0 && (desc.stride_b % 4) == 0 && (desc.stride_c % 4) == 0;
}

inline bool is_cublasdx_variant(CuBLASDxGemmVariant variant) {
    switch (variant) {
    case CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
    case CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
    case CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
    case CuBLASDxGemmVariant::CuBLASDx32x32x32TT:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32TN:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32NT:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32TT:
        return true;
    default:
        return false;
    }
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

    const int sm = prop.major * 100 + prop.minor * 10;
    switch (sm) {
    case kSupportedSM:
        return true;
    default:
        return false;
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

template <class BLAS>
constexpr int pipelined_depth() {
    const auto a_smem_layout = BLAS::suggest_layout_smem_a();
    const auto b_smem_layout = BLAS::suggest_layout_smem_b();
    return cublasdx::suggest_max_pipeline_depth<BLAS, typename BLAS::a_value_type, typename BLAS::b_value_type>(
        a_smem_layout, b_smem_layout);
}

template <class BLAS>
constexpr unsigned int pipelined_shared_memory_size() {
    constexpr int depth = pipelined_depth<BLAS>();
    static_assert(depth > 0, "Pipeline depth must be positive for pipelined cuBLASDx kernels");

    using cute_layout_a = decltype(cublasdx::detail::get_cute_layout(BLAS::suggest_layout_smem_a()));
    using cute_layout_b = decltype(cublasdx::detail::get_cute_layout(BLAS::suggest_layout_smem_b()));

    constexpr unsigned int shared_bytes_ab =
        round_up_unsigned(depth * cute::cosize(cute_layout_a {}) * sizeof(typename BLAS::a_value_type),
                          cublasdx::alignment_of_v_b<BLAS>) +
        depth * cute::cosize(cute_layout_b {}) * sizeof(typename BLAS::b_value_type);
    constexpr unsigned int barrier_bytes = depth * 2u * sizeof(cublasdx::detail::barrier_storage_t);

    return round_up_unsigned(shared_bytes_ab, cublasdx::detail::barrier_buffer_alignment_bytes) + barrier_bytes;
}

template <class BLAS, int TileM, int TileN, Transpose OpA, Transpose OpB>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void gemm_cublasdx_kernel(GemmLaunchDescriptor desc) {
    extern __shared__ __align__(16) char smem[];

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int actual_m = batch_dim(desc.m_batch, desc.m, batch_idx);
    const int actual_n = batch_dim(desc.n_batch, desc.n, batch_idx);
    const int actual_k = batch_dim(desc.k_batch, desc.k, batch_idx);

    if (blockIdx.x * TileM >= actual_m || blockIdx.y * TileN >= actual_n || actual_m == 0 || actual_n == 0 || actual_k == 0) {
        return;
    }

    const float* a_batch = desc.a_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_a);
    const float* b_batch = desc.b_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_b);
    float* c_batch = desc.c_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_c);

    auto global_a = cublasdx::make_gmem_tensor<ArrangementTag<OpA>::value>(a_batch, actual_m, actual_k, desc.lda);
    auto global_b = cublasdx::make_gmem_tensor<ArrangementTag<OpB>::value>(b_batch, actual_k, actual_n, desc.ldb);
    auto global_c = cublasdx::make_gmem_tensor<cublasdx::col_major>(c_batch, actual_m, actual_n, desc.ldc);

    auto tile_slice_a_gmem = cublasdx::get_tile_row(global_a, BLAS::a_shape, blockIdx.x);
    auto tile_slice_b_gmem = cublasdx::get_tile_col(global_b, BLAS::b_shape, blockIdx.y);
    auto tile_c_gmem = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);

    const auto a_smem_layout = BLAS::suggest_layout_smem_a();
    const auto b_smem_layout = BLAS::suggest_layout_smem_b();
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<BLAS>(smem, a_smem_layout, b_smem_layout);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, a_smem_layout);
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, b_smem_layout);
    auto accumulator = BLAS::suggest_accumulator();
    auto get_tile_from_slice = [](auto& slice, auto index) {
        return slice(cublasdx::slice, cublasdx::slice, index);
    };
    const int k_stages = static_cast<int>(cute::get<2>(cute::shape(tile_slice_a_gmem.layout())));

    using alignment = cublasdx::alignment_of<BLAS>;

    for (int stage = 0; stage < k_stages; ++stage) {
        cublasdx::copy<BLAS, alignment::a>(get_tile_from_slice(tile_slice_a_gmem, stage), a_shared_tensor);
        cublasdx::copy<BLAS, alignment::b>(get_tile_from_slice(tile_slice_b_gmem, stage), b_shared_tensor);
        cublasdx::copy_wait();

        BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
        __syncthreads();
    }

    auto c_fragment = accumulator.make_partition_and_copy(tile_c_gmem);
    cublasdx::axpby(desc.alpha, accumulator.get_results(), desc.beta, c_fragment);
    accumulator.partition_and_copy(c_fragment, tile_c_gmem);
}

template <class BLAS, int TileM, int TileN, Transpose OpA, Transpose OpB>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void gemm_cublasdx_pipelined_kernel(GemmLaunchDescriptor desc) {
    extern __shared__ __align__(16) char smem[];

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int actual_m = batch_dim(desc.m_batch, desc.m, batch_idx);
    const int actual_n = batch_dim(desc.n_batch, desc.n, batch_idx);
    const int actual_k = batch_dim(desc.k_batch, desc.k, batch_idx);

    if (blockIdx.x * TileM >= actual_m || blockIdx.y * TileN >= actual_n || actual_m == 0 || actual_n == 0 || actual_k == 0) {
        return;
    }

    const float* a_batch = desc.a_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_a);
    const float* b_batch = desc.b_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_b);
    float* c_batch = desc.c_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_c);

    auto global_a = cublasdx::make_gmem_tensor<ArrangementTag<OpA>::value>(a_batch, actual_m, actual_k, desc.lda);
    auto global_b = cublasdx::make_gmem_tensor<ArrangementTag<OpB>::value>(b_batch, actual_k, actual_n, desc.ldb);
    auto global_c = cublasdx::make_gmem_tensor<cublasdx::col_major>(c_batch, actual_m, actual_n, desc.ldc);

    auto tile_c_gmem = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);

    constexpr int depth = pipelined_depth<BLAS>();
    static_assert(depth > 0, "Pipeline depth must be positive for pipelined cuBLASDx kernels");

    auto device_pipeline = cublasdx::make_device_pipeline<depth, BLAS, cublasdx::external_accumulation>(
        global_a,
        cublasdx::detail::get_cute_layout(BLAS::suggest_layout_smem_a()),
        global_b,
        cublasdx::detail::get_cute_layout(BLAS::suggest_layout_smem_b()));

    auto pipeline = device_pipeline.get_tile(smem, cute::make_coord(blockIdx.x), cute::make_coord(blockIdx.y));
    auto accumulator = pipeline.get_accumulator();
    pipeline.execute(accumulator);
    pipeline.epilogue(accumulator, [&](auto& epilogue_accumulator) {
        auto c_fragment = epilogue_accumulator.make_partition_and_copy(tile_c_gmem);
        cublasdx::axpby(desc.alpha, epilogue_accumulator.get_results(), desc.beta, c_fragment);
        epilogue_accumulator.partition_and_copy(c_fragment, tile_c_gmem);
    });
    pipeline.epilogue_sync();
}

template <int Sm, int TileM, int TileN, int TileK, Transpose OpA, Transpose OpB>
cudaError_t launch_for_sm(const GemmLaunchDescriptor& desc, cudaStream_t stream) {
    using BLAS = GemmBlock<Sm, TileM, TileN, TileK, OpA, OpB>;
    auto kernel = gemm_cublasdx_kernel<BLAS, TileM, TileN, OpA, OpB>;
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

    const dim3 grid(static_cast<unsigned int>(ceil_div(desc.m, TileM)),
                    static_cast<unsigned int>(ceil_div(desc.n, TileN)),
                    static_cast<unsigned int>(desc.batch));
    kernel<<<grid, BLAS::block_dim, shared_memory_size, stream>>>(desc);
    return cudaGetLastError();
}

template <int Sm, int TileM, int TileN, int TileK, Transpose OpA, Transpose OpB>
cudaError_t launch_pipelined_for_sm(const GemmLaunchDescriptor& desc, cudaStream_t stream) {
    using BLAS = PipelinedGemmBlock<Sm, TileM, TileN, TileK, OpA, OpB>;
    constexpr int depth = pipelined_depth<BLAS>();
    if constexpr (depth <= 0) {
        return cudaErrorNotSupported;
    }

    if ((desc.k / TileK) < depth) {
        return cudaErrorNotSupported;
    }

    auto kernel = gemm_cublasdx_pipelined_kernel<BLAS, TileM, TileN, OpA, OpB>;
    constexpr unsigned int shared_memory_size = pipelined_shared_memory_size<BLAS>();

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

    const dim3 grid(static_cast<unsigned int>(ceil_div(desc.m, TileM)),
                    static_cast<unsigned int>(ceil_div(desc.n, TileN)),
                    static_cast<unsigned int>(desc.batch));
    kernel<<<grid, BLAS::block_dim, shared_memory_size, stream>>>(desc);
    return cudaGetLastError();
}

template <int TileM, int TileN, int TileK, Transpose OpA, Transpose OpB>
cudaError_t dispatch_family_for_current_sm(const GemmLaunchDescriptor& desc, cudaStream_t stream) {
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
        return launch_for_sm<kSupportedSM, TileM, TileN, TileK, OpA, OpB>(desc, stream);
    default:
        return cudaErrorNotSupported;
    }
}

template <int TileM, int TileN, int TileK, Transpose OpA, Transpose OpB>
cudaError_t dispatch_pipelined_family_for_current_sm(const GemmLaunchDescriptor& desc,
                                                     cudaStream_t stream) {
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
        return launch_pipelined_for_sm<kSupportedSM, TileM, TileN, TileK, OpA, OpB>(desc, stream);
    default:
        return cudaErrorNotSupported;
    }
}

template <Transpose OpA, Transpose OpB>
cudaError_t dispatch_for_variant(CuBLASDxGemmVariant variant,
                                 const GemmLaunchDescriptor& desc,
                                 cudaStream_t stream) {
    switch (tile_config(variant).tile_m) {
    case 32:
        return dispatch_family_for_current_sm<32, 32, 32, OpA, OpB>(desc, stream);
    case 64:
        return dispatch_pipelined_family_for_current_sm<64, 64, 32, OpA, OpB>(desc, stream);
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

bool variant_supported(CuBLASDxGemmVariant variant,
                       const GemmLaunchDescriptor& desc) {
    if (!is_cublasdx_variant(variant) || !available()) {
        return false;
    }

    const auto tile = tile_config(variant);
    if (tile.tile_m <= 0 || tile.tile_n <= 0 || tile.tile_k <= 0) {
        return false;
    }

    if (desc.batch <= 0 || desc.m <= 0 || desc.n <= 0 || desc.k < 0) {
        return false;
    }

    if (desc.heterogeneous && tile.tile_m != 32) {
        return false;
    }

    for (int batch_idx = 0; batch_idx < desc.batch; ++batch_idx) {
        const int batch_m = batch_dim(desc.m_batch, desc.m, batch_idx);
        const int batch_n = batch_dim(desc.n_batch, desc.n, batch_idx);
        const int batch_k = batch_dim(desc.k_batch, desc.k, batch_idx);

        if (batch_m < 0 || batch_n < 0 || batch_k < 0) {
            return false;
        }

        if (batch_m == 0 || batch_n == 0) {
            continue;
        }

        if ((batch_m % tile.tile_m) != 0 || (batch_n % tile.tile_n) != 0) {
            return false;
        }

        if (batch_k == 0) {
            continue;
        }

        if ((batch_k % tile.tile_k) != 0) {
            return false;
        }
    }

    if (!has_max_alignment(desc)) {
        return false;
    }

    return current_device_sm_supported();
}

cudaError_t launch_float(CuBLASDxGemmVariant variant,
                         const GemmLaunchDescriptor& desc,
                         cudaStream_t stream) {
    if (!variant_supported(variant, desc)) {
        return cudaErrorNotSupported;
    }

#if BATCHLAS_HAS_CUBLASDX_HEADER
    switch (variant) {
    case CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
        return dispatch_for_variant<Transpose::NoTrans, Transpose::NoTrans>(variant, desc, stream);
    case CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32TN:
        return dispatch_for_variant<Transpose::Trans, Transpose::NoTrans>(variant, desc, stream);
    case CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32NT:
        return dispatch_for_variant<Transpose::NoTrans, Transpose::Trans>(variant, desc, stream);
    case CuBLASDxGemmVariant::CuBLASDx32x32x32TT:
    case CuBLASDxGemmVariant::CuBLASDx64x64x32TT:
        return dispatch_for_variant<Transpose::Trans, Transpose::Trans>(variant, desc, stream);
    default:
        return cudaErrorNotSupported;
    }
#else
    static_cast<void>(variant);
    static_cast<void>(desc);
    static_cast<void>(stream);
    return cudaErrorNotSupported;
#endif
}

} // namespace batchlas::backend::cublasdx_gemm
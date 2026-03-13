#include "trmm_cublasdx_fused.hh"
#include "cublasdx_fused_common.hh"

namespace batchlas::backend::trmm_cublasdx {

namespace {

template <int TileM, int TileN>
constexpr unsigned int diagonal_shared_memory_size() {
    return detail::round_up_unsigned(static_cast<unsigned int>((TileM * TileM + TileM * TileN) * sizeof(float)), 16u);
}

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
    cublasdx::BlockDim<detail::kCublasDxBlockSize>() +
    cublasdx::SM<Sm>());

template <int Sm, int TileM, int TileN, int TileK>
using PipelinedGemmBlock = decltype(
    GemmBlock<Sm, TileM, TileN, TileK>() +
    cublasdx::WithPipeline());
#endif

inline bool has_max_alignment(const TrmmLaunchDescriptor& desc) {
    return detail::pointers_16b_aligned({desc.a_ptr, desc.b_ptr, desc.c_ptr}) &&
        detail::multiples_of_four({desc.lda, desc.ldb, desc.ldc, desc.stride_a, desc.stride_b, desc.stride_c});
}

#if BATCHLAS_HAS_CUBLASDX_HEADER
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
        detail::round_up_unsigned(depth * cute::cosize(cute_layout_a {}) * sizeof(typename BLAS::a_value_type),
                                  cublasdx::alignment_of_v_b<BLAS>) +
        depth * cute::cosize(cute_layout_b {}) * sizeof(typename BLAS::b_value_type);
    constexpr unsigned int barrier_bytes = depth * 2u * sizeof(cublasdx::detail::barrier_storage_t);

    return detail::round_up_unsigned(shared_bytes_ab, cublasdx::detail::barrier_buffer_alignment_bytes) + barrier_bytes;
}

template <class BLAS, int TileM, int TileN, int TileK, Diag DiagMode>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void trmm_cublasdx_kernel(TrmmLaunchDescriptor desc) {
    extern __shared__ __align__(16) char smem[];

    const int batch_idx = static_cast<int>(blockIdx.z);
    const float* a_batch = desc.a_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_a);
    const float* b_batch = desc.b_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_b);
    float* c_batch = desc.c_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_c);

    auto global_c = cublasdx::make_gmem_tensor<cublasdx::col_major>(c_batch, desc.m, desc.n, desc.ldc);
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
    const int elems = TileM * TileN;

    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        c_shared_tensor(row, col) = 0.0f;
    }
    __syncthreads();

    const int offdiag_stages = tile_row / TileK;
    for (int stage = 0; stage < offdiag_stages; ++stage) {
        const int k_start = stage * TileK;
        load_dense_tile(a_shared_tensor, a_batch, desc.lda, tile_row, k_start, TileM, TileK);
        load_dense_tile(b_shared_tensor, b_batch, desc.ldb, k_start, tile_col, TileK, TileN);
        __syncthreads();
        BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
        __syncthreads();
    }

    if (offdiag_stages > 0) {
        auto c_fragment = accumulator.make_empty_fragment();
        cublasdx::axpby(desc.alpha, accumulator.get_results(), 0.0f, c_fragment);
        accumulator.partition_and_copy(c_fragment, c_shared_tensor);
        __syncthreads();
    }

    float* diag_a_shared = reinterpret_cast<float*>(smem_a);

    for (int linear = static_cast<int>(threadIdx.x); linear < TileM * TileM; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        float a_value = 0.0f;
        if (row >= col) {
            a_value = a_batch[(tile_row + col) * desc.lda + (tile_row + row)];
            if constexpr (DiagMode == Diag::Unit) {
                if (row == col) {
                    a_value = 1.0f;
                }
            }
        }
        diag_a_shared[linear] = a_value;
    }
    __syncthreads();

    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        float diagonal_sum = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < TileM; ++kk) {
            if (kk <= row) {
                diagonal_sum += diag_a_shared[row + kk * TileM] * b_batch[(tile_col + col) * desc.ldb + (tile_row + kk)];
            }
        }
        c_shared_tensor(row, col) += desc.alpha * diagonal_sum;
    }
    __syncthreads();

    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        c_batch[(tile_col + col) * desc.ldc + (tile_row + row)] = c_shared_tensor(row, col);
    }
}

template <class BLAS, int TileM, int TileN, int TileK, Diag DiagMode>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void trmm_cublasdx_pipelined_kernel(TrmmLaunchDescriptor desc) {
    extern __shared__ __align__(16) char smem[];

    const int batch_idx = static_cast<int>(blockIdx.z);
    const float* a_batch = desc.a_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_a);
    const float* b_batch = desc.b_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_b);
    float* c_batch = desc.c_ptr + static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(desc.stride_c);

    auto global_c = cublasdx::make_gmem_tensor<cublasdx::col_major>(c_batch, desc.m, desc.n, desc.ldc);
    auto tile_c_gmem = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);

    const int tile_row = static_cast<int>(blockIdx.x) * TileM;
    const int tile_col = static_cast<int>(blockIdx.y) * TileN;
    const int elems = TileM * TileN;

    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        c_batch[(tile_col + col) * desc.ldc + (tile_row + row)] = 0.0f;
    }
    __syncthreads();

    const int offdiag_stages = tile_row / TileK;
    auto accumulator = BLAS::suggest_accumulator();

    constexpr int depth = pipelined_depth<BLAS>();
    static_assert(depth > 0, "Pipeline depth must be positive for pipelined cuBLASDx kernels");

    if (offdiag_stages >= depth) {
        const float* offdiag_a_ptr = a_batch + tile_row;
        const float* offdiag_b_ptr = b_batch + tile_col * desc.ldb;
        auto global_a = cublasdx::make_gmem_tensor<cublasdx::col_major>(offdiag_a_ptr, TileM, tile_row, desc.lda);
        auto global_b = cublasdx::make_gmem_tensor<cublasdx::col_major>(offdiag_b_ptr, tile_row, TileN, desc.ldb);

        auto device_pipeline = cublasdx::make_device_pipeline<depth, BLAS, cublasdx::external_accumulation>(
            global_a,
            cublasdx::detail::get_cute_layout(BLAS::suggest_layout_smem_a()),
            global_b,
            cublasdx::detail::get_cute_layout(BLAS::suggest_layout_smem_b()));

        auto pipeline = device_pipeline.get_tile(smem, cute::make_coord(0), cute::make_coord(0));
        pipeline.execute(accumulator);
    } else {
        const auto a_smem_layout = BLAS::suggest_layout_smem_a();
        const auto b_smem_layout = BLAS::suggest_layout_smem_b();
        auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<BLAS>(smem, a_smem_layout, b_smem_layout);
        auto a_shared_tensor = cublasdx::make_tensor(smem_a, a_smem_layout);
        auto b_shared_tensor = cublasdx::make_tensor(smem_b, b_smem_layout);

        for (int stage = 0; stage < offdiag_stages; ++stage) {
            const int k_start = stage * TileK;
            load_dense_tile(a_shared_tensor, a_batch, desc.lda, tile_row, k_start, TileM, TileK);
            load_dense_tile(b_shared_tensor, b_batch, desc.ldb, k_start, tile_col, TileK, TileN);
            __syncthreads();
            BLAS().execute(a_shared_tensor, b_shared_tensor, accumulator);
            __syncthreads();
        }
    }

    if (offdiag_stages > 0) {
        auto c_fragment = accumulator.make_partition_and_copy(tile_c_gmem);
        cublasdx::axpby(desc.alpha, accumulator.get_results(), 1.0f, c_fragment);
        accumulator.partition_and_copy(c_fragment, tile_c_gmem);
        __syncthreads();
    }

    float* diag_a_shared = reinterpret_cast<float*>(smem);
    float* diag_b_shared = diag_a_shared + TileM * TileM;

    for (int linear = static_cast<int>(threadIdx.x); linear < TileM * TileM; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        float a_value = 0.0f;
        if (row >= col) {
            a_value = a_batch[(tile_row + col) * desc.lda + (tile_row + row)];
            if constexpr (DiagMode == Diag::Unit) {
                if (row == col) {
                    a_value = 1.0f;
                }
            }
        }
        diag_a_shared[linear] = a_value;
    }

    for (int linear = static_cast<int>(threadIdx.x); linear < TileM * TileN; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        diag_b_shared[linear] = b_batch[(tile_col + col) * desc.ldb + (tile_row + row)];
    }
    __syncthreads();

    for (int linear = static_cast<int>(threadIdx.x); linear < elems; linear += static_cast<int>(blockDim.x)) {
        const int row = linear % TileM;
        const int col = linear / TileM;
        float diagonal_sum = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < TileM; ++kk) {
            if (kk <= row) {
                diagonal_sum += diag_a_shared[row + kk * TileM] * diag_b_shared[kk + col * TileM];
            }
        }
        c_batch[(tile_col + col) * desc.ldc + (tile_row + row)] += desc.alpha * diagonal_sum;
    }
}

template <int Sm, int TileM, int TileN, int TileK, Diag DiagMode>
cudaError_t launch_for_sm(const TrmmLaunchDescriptor& desc, cudaStream_t stream) {
    using BLAS = GemmBlock<Sm, TileM, TileN, TileK>;
    auto kernel = trmm_cublasdx_kernel<BLAS, TileM, TileN, TileK, DiagMode>;
    const unsigned int shared_memory_size = cublasdx::get_shared_storage_size<BLAS>();

    const cudaError_t resource_status = detail::validate_launch_resources(kernel, shared_memory_size);
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

template <int Sm, int TileM, int TileN, int TileK, Diag DiagMode>
cudaError_t launch_pipelined_for_sm(const TrmmLaunchDescriptor& desc, cudaStream_t stream) {
    using BLAS = PipelinedGemmBlock<Sm, TileM, TileN, TileK>;
    constexpr int depth = pipelined_depth<BLAS>();
    if constexpr (depth <= 0) {
        return cudaErrorNotSupported;
    }

    auto kernel = trmm_cublasdx_pipelined_kernel<BLAS, TileM, TileN, TileK, DiagMode>;
    constexpr unsigned int mm_shared_memory_size = pipelined_shared_memory_size<BLAS>();
    constexpr unsigned int shared_memory_size = mm_shared_memory_size > diagonal_shared_memory_size<TileM, TileN>()
        ? mm_shared_memory_size
        : diagonal_shared_memory_size<TileM, TileN>();

    const cudaError_t resource_status = detail::validate_launch_resources(kernel, shared_memory_size);
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

template <int TileM, int TileN, int TileK, Diag DiagMode>
cudaError_t dispatch_family_for_current_sm(const TrmmLaunchDescriptor& desc, cudaStream_t stream) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return cudaErrorNotSupported;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return cudaErrorNotSupported;
    }

    switch (prop.major * 100 + prop.minor * 10) {
    case detail::kCublasDxSupportedSM:
        return launch_for_sm<detail::kCublasDxSupportedSM, TileM, TileN, TileK, DiagMode>(desc, stream);
    default:
        return cudaErrorNotSupported;
    }
}

template <int TileM, int TileN, int TileK, Diag DiagMode>
cudaError_t dispatch_pipelined_family_for_current_sm(const TrmmLaunchDescriptor& desc, cudaStream_t stream) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return cudaErrorNotSupported;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return cudaErrorNotSupported;
    }

    switch (prop.major * 100 + prop.minor * 10) {
    case detail::kCublasDxSupportedSM:
        return launch_pipelined_for_sm<detail::kCublasDxSupportedSM, TileM, TileN, TileK, DiagMode>(desc, stream);
    default:
        return cudaErrorNotSupported;
    }
}

template <Diag DiagMode>
cudaError_t dispatch_for_variant(cublasdx_gemm::CuBLASDxGemmVariant variant,
                                 const TrmmLaunchDescriptor& desc,
                                 cudaStream_t stream) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
        return dispatch_family_for_current_sm<32, 32, 32, DiagMode>(desc, stream);
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
        if (desc.m >= 256 || desc.n >= 256) {
            return dispatch_family_for_current_sm<32, 32, 32, DiagMode>(desc, stream);
        }
        if (desc.m <= 128 && desc.n <= 128) {
            return dispatch_pipelined_family_for_current_sm<64, 64, 32, DiagMode>(desc, stream);
        }
        return dispatch_family_for_current_sm<64, 64, 32, DiagMode>(desc, stream);
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
                         const TrmmLaunchDescriptor& desc,
                         Diag diag,
                         cudaStream_t stream) {
    if (!available() || !detail::current_device_sm_supported()) {
        return cudaErrorNotSupported;
    }

    if (variant != cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN &&
        variant != cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN) {
        return cudaErrorNotSupported;
    }
    if (desc.batch <= 0 || desc.m <= 0 || desc.n <= 0) {
        return cudaErrorNotSupported;
    }

    const int tile_m = variant == cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN ? 64 : 32;
    const int tile_n = tile_m;
    if ((desc.m % tile_m) != 0 || (desc.n % tile_n) != 0) {
        return cudaErrorNotSupported;
    }
    if (!has_max_alignment(desc)) {
        return cudaErrorNotSupported;
    }

#if BATCHLAS_HAS_CUBLASDX_HEADER
    if (diag == Diag::Unit) {
        return dispatch_for_variant<Diag::Unit>(variant, desc, stream);
    }
    if (diag == Diag::NonUnit) {
        return dispatch_for_variant<Diag::NonUnit>(variant, desc, stream);
    }
#else
    static_cast<void>(variant);
    static_cast<void>(desc);
    static_cast<void>(diag);
    static_cast<void>(stream);
#endif

    return cudaErrorNotSupported;
}

} // namespace batchlas::backend::trmm_cublasdx
#include <cuda_runtime_api.h>
#include <cstdint>
#include <type_traits>

#if defined(BATCHLAS_ENABLE_CUSOLVERDX_WRAPPER) && __has_include(<cusolverdx.hpp>)
    #include <cusolverdx.hpp>
    #define BATCHLAS_HAS_CUSOLVERDX_HEADER 1
#else
    #define BATCHLAS_HAS_CUSOLVERDX_HEADER 0
#endif

namespace batchlas::backend::cusolverdx::cuda_kernels {

constexpr int kMinHeevN = 1;
constexpr int kMinHtevN = 2;
constexpr int kMaxSupportedN = 512;

#if BATCHLAS_HAS_CUSOLVERDX_HEADER
namespace {

constexpr int kSupportedSM = 890;

template <int Sm, int N, typename T, bool ComputeVectors, bool Lower>
struct HeevSolverFactory;

template <int Sm, int N, bool ComputeVectors, bool Lower>
struct HeevSolverFactory<Sm, N, float, ComputeVectors, Lower> {
    using type = decltype(
    ::cusolverdx::Size<N>() +
    ::cusolverdx::Precision<float>() +
    ::cusolverdx::Type<::cusolverdx::type::real>() +
    ::cusolverdx::Function<::cusolverdx::heev>() +
    ::cusolverdx::FillMode<Lower ? ::cusolverdx::fill_mode::lower : ::cusolverdx::fill_mode::upper>() +
    ::cusolverdx::Arrangement<::cusolverdx::arrangement::col_major>() +
    ::cusolverdx::Job<ComputeVectors ? ::cusolverdx::job::overwrite_vectors : ::cusolverdx::job::no_vectors>() +
    ::cusolverdx::SM<Sm>() +
    ::cusolverdx::Block() +
    ::cusolverdx::BatchesPerBlock<1>());
};

template <int Sm, int N, bool ComputeVectors, bool Lower>
struct HeevSolverFactory<Sm, N, double, ComputeVectors, Lower> {
    using type = decltype(
    ::cusolverdx::Size<N>() +
    ::cusolverdx::Precision<double>() +
    ::cusolverdx::Type<::cusolverdx::type::real>() +
    ::cusolverdx::Function<::cusolverdx::heev>() +
    ::cusolverdx::FillMode<Lower ? ::cusolverdx::fill_mode::lower : ::cusolverdx::fill_mode::upper>() +
    ::cusolverdx::Arrangement<::cusolverdx::arrangement::col_major>() +
    ::cusolverdx::Job<ComputeVectors ? ::cusolverdx::job::overwrite_vectors : ::cusolverdx::job::no_vectors>() +
    ::cusolverdx::SM<Sm>() +
    ::cusolverdx::Block() +
    ::cusolverdx::BatchesPerBlock<1>());
};

template <int Sm, int N, typename T, bool ComputeVectors>
struct HtevSolverFactory;

template <int Sm, int N, bool ComputeVectors>
struct HtevSolverFactory<Sm, N, float, ComputeVectors> {
    using type = decltype(
    ::cusolverdx::Size<N>() +
    ::cusolverdx::Precision<float>() +
    ::cusolverdx::Type<::cusolverdx::type::real>() +
    ::cusolverdx::Function<::cusolverdx::htev>() +
    ::cusolverdx::Arrangement<::cusolverdx::arrangement::col_major>() +
    ::cusolverdx::Job<ComputeVectors ? ::cusolverdx::job::all_vectors : ::cusolverdx::job::no_vectors>() +
    ::cusolverdx::SM<Sm>() +
    ::cusolverdx::Block() +
    ::cusolverdx::BatchesPerBlock<1>());
};

template <int Sm, int N, bool ComputeVectors>
struct HtevSolverFactory<Sm, N, double, ComputeVectors> {
    using type = decltype(
    ::cusolverdx::Size<N>() +
    ::cusolverdx::Precision<double>() +
    ::cusolverdx::Type<::cusolverdx::type::real>() +
    ::cusolverdx::Function<::cusolverdx::htev>() +
    ::cusolverdx::Arrangement<::cusolverdx::arrangement::col_major>() +
    ::cusolverdx::Job<ComputeVectors ? ::cusolverdx::job::all_vectors : ::cusolverdx::job::no_vectors>() +
    ::cusolverdx::SM<Sm>() +
    ::cusolverdx::Block() +
    ::cusolverdx::BatchesPerBlock<1>());
};

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

template <class Solver>
__global__ __launch_bounds__(Solver::max_threads_per_block)
void heev_kernel(typename Solver::a_data_type* A,
                 int lda,
                 typename Solver::a_precision* lambda,
                 typename Solver::status_type* info,
                 int batches) {
    constexpr int m = Solver::m_size;
    const int b = static_cast<int>(blockIdx.x);
    if (b >= batches) return;

    using Data = typename Solver::a_data_type;
    using Prec = typename Solver::a_precision;

    extern __shared__ __align__(16) unsigned char smem[];
    auto* A_s = reinterpret_cast<Data*>(smem);
    auto* lambda_s = reinterpret_cast<Prec*>(A_s + static_cast<std::size_t>(Solver::lda) * m);
    auto* work_s = reinterpret_cast<Data*>(lambda_s + m);

    const int tid = static_cast<int>(threadIdx.x);
    Data* A_b = A + static_cast<std::size_t>(b) * static_cast<std::size_t>(lda) * static_cast<std::size_t>(m);
    Prec* lambda_b = lambda + static_cast<std::size_t>(b) * static_cast<std::size_t>(m);

    for (int idx = tid; idx < Solver::lda * m; idx += static_cast<int>(blockDim.x)) {
        const int r = idx % Solver::lda;
        const int c = idx / Solver::lda;
        if (r < m) {
            A_s[idx] = A_b[r + c * lda];
        } else {
            A_s[idx] = Data(0);
        }
    }
    __syncthreads();

    Solver().execute(A_s, Solver::lda, lambda_s, work_s, &info[b]);
    __syncthreads();

    for (int i = tid; i < m; i += static_cast<int>(blockDim.x)) {
        lambda_b[i] = lambda_s[i];
    }

    if constexpr (Solver::job == ::cusolverdx::job::overwrite_vectors) {
        for (int idx = tid; idx < m * m; idx += static_cast<int>(blockDim.x)) {
            const int r = idx % m;
            const int c = idx / m;
            A_b[r + c * lda] = A_s[r + c * Solver::lda];
        }
    }
}

template <class Solver>
__global__ __launch_bounds__(Solver::max_threads_per_block)
void htev_kernel(typename Solver::a_data_type* d,
                 typename Solver::a_data_type* e,
                 typename Solver::a_data_type* V,
                 int ldv,
                 typename Solver::status_type* info,
                 int batches) {
    constexpr int m = Solver::m_size;
    const int b = static_cast<int>(blockIdx.x);
    if (b >= batches) return;

    using Data = typename Solver::a_data_type;

    extern __shared__ __align__(16) unsigned char smem[];
    auto* d_s = reinterpret_cast<Data*>(smem);
    auto* e_s = reinterpret_cast<Data*>(d_s + m);
    auto* V_s = reinterpret_cast<Data*>(e_s + (m - 1));

    const int tid = static_cast<int>(threadIdx.x);
    Data* d_b = d + static_cast<std::size_t>(b) * static_cast<std::size_t>(m);
    Data* e_b = e + static_cast<std::size_t>(b) * static_cast<std::size_t>(m - 1);

    for (int i = tid; i < m; i += static_cast<int>(blockDim.x)) d_s[i] = d_b[i];
    for (int i = tid; i < m - 1; i += static_cast<int>(blockDim.x)) e_s[i] = e_b[i];

    if constexpr (Solver::job == ::cusolverdx::job::all_vectors) {
        Data* V_b = V + static_cast<std::size_t>(b) * static_cast<std::size_t>(ldv) * static_cast<std::size_t>(m);
        for (int idx = tid; idx < m * m; idx += static_cast<int>(blockDim.x)) {
            const int r = idx % m;
            const int c = idx / m;
            V_s[r + c * Solver::lda] = V_b[r + c * ldv];
        }
    }

    __syncthreads();

    if constexpr (Solver::job == ::cusolverdx::job::no_vectors) {
        Solver().execute(d_s, e_s, &info[b]);
    } else {
        Solver().execute(d_s, e_s, V_s, Solver::lda, &info[b]);
    }

    __syncthreads();

    for (int i = tid; i < m; i += static_cast<int>(blockDim.x)) d_b[i] = d_s[i];
    for (int i = tid; i < m - 1; i += static_cast<int>(blockDim.x)) e_b[i] = e_s[i];

    if constexpr (Solver::job == ::cusolverdx::job::all_vectors) {
        Data* V_b = V + static_cast<std::size_t>(b) * static_cast<std::size_t>(ldv) * static_cast<std::size_t>(m);
        for (int idx = tid; idx < m * m; idx += static_cast<int>(blockDim.x)) {
            const int r = idx % m;
            const int c = idx / m;
            V_b[r + c * ldv] = V_s[r + c * Solver::lda];
        }
    }
}

template <int Sm, int N, typename T, bool ComputeVectors, bool Lower>
cudaError_t heev_launch_n(T* A,
                          int lda,
                          T* lambda,
                          int* info,
                          int batches,
                          cudaStream_t stream) {
    using Solver = typename HeevSolverFactory<Sm, N, T, ComputeVectors, Lower>::type;
    constexpr int shmem = Solver::shared_memory_size;
    cudaFuncSetAttribute(heev_kernel<Solver>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    heev_kernel<Solver><<<batches, Solver::block_dim, shmem, stream>>>(A, lda, lambda, info, batches);
    return cudaGetLastError();
}

template <int Sm, int N, typename T, bool ComputeVectors>
cudaError_t htev_launch_n(T* d,
                          T* e,
                          T* V,
                          int ldv,
                          int* info,
                          int batches,
                          cudaStream_t stream) {
    using Solver = typename HtevSolverFactory<Sm, N, T, ComputeVectors>::type;
    constexpr int shmem = Solver::shared_memory_size;
    cudaFuncSetAttribute(htev_kernel<Solver>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    htev_kernel<Solver><<<batches, Solver::block_dim, shmem, stream>>>(d, e, V, ldv, info, batches);
    return cudaGetLastError();
}

template <int Sm, typename T, bool ComputeVectors, bool Lower, int N, int MaxN>
cudaError_t heev_dispatch_n_range(T* A,
                                  int n,
                                  int lda,
                                  T* lambda,
                                  int* info,
                                  int batches,
                                  cudaStream_t stream) {
    if (n == N) {
        return heev_launch_n<Sm, N, T, ComputeVectors, Lower>(A, lda, lambda, info, batches, stream);
    }
    if constexpr (N < MaxN) {
        return heev_dispatch_n_range<Sm, T, ComputeVectors, Lower, N + 1, MaxN>(A, n, lda, lambda, info, batches, stream);
    }
    return cudaErrorNotSupported;
}

template <int Sm, typename T, bool ComputeVectors, int N, int MaxN>
cudaError_t htev_dispatch_n_range(T* d,
                                  int n,
                                  T* e,
                                  T* V,
                                  int ldv,
                                  int* info,
                                  int batches,
                                  cudaStream_t stream) {
    if (n == N) {
        return htev_launch_n<Sm, N, T, ComputeVectors>(d, e, V, ldv, info, batches, stream);
    }
    if constexpr (N < MaxN) {
        return htev_dispatch_n_range<Sm, T, ComputeVectors, N + 1, MaxN>(d, n, e, V, ldv, info, batches, stream);
    }
    return cudaErrorNotSupported;
}

template <int Sm, typename T>
cudaError_t heev_dispatch_for_sm(T* A,
                                 int n,
                                 int lda,
                                 T* lambda,
                                 int* info,
                                 int batches,
                                 bool compute_vectors,
                                 bool lower,
                                 cudaStream_t stream) {
    if (n < kMinHeevN || n > kMaxSupportedN) {
        return cudaErrorNotSupported;
    }

    if (compute_vectors) {
        if (lower) {
            return heev_dispatch_n_range<Sm, T, true, true, kMinHeevN, kMaxSupportedN>(A, n, lda, lambda, info, batches, stream);
        }
        return heev_dispatch_n_range<Sm, T, true, false, kMinHeevN, kMaxSupportedN>(A, n, lda, lambda, info, batches, stream);
    }
    if (lower) {
        return heev_dispatch_n_range<Sm, T, false, true, kMinHeevN, kMaxSupportedN>(A, n, lda, lambda, info, batches, stream);
    }
    return heev_dispatch_n_range<Sm, T, false, false, kMinHeevN, kMaxSupportedN>(A, n, lda, lambda, info, batches, stream);
}

template <int Sm, typename T>
cudaError_t htev_dispatch_for_sm(T* d,
                                 int n,
                                 T* e,
                                 T* V,
                                 int ldv,
                                 int* info,
                                 int batches,
                                 bool compute_vectors,
                                 cudaStream_t stream) {
    if (n < kMinHtevN || n > kMaxSupportedN) {
        return cudaErrorNotSupported;
    }

    if (compute_vectors) {
        return htev_dispatch_n_range<Sm, T, true, kMinHtevN, kMaxSupportedN>(d, n, e, V, ldv, info, batches, stream);
    }
    return htev_dispatch_n_range<Sm, T, false, kMinHtevN, kMaxSupportedN>(d, n, e, V, ldv, info, batches, stream);
}

template <typename T>
cudaError_t heev_dispatch(T* A,
                          int n,
                          int lda,
                          T* lambda,
                          int* info,
                          int batches,
                          bool compute_vectors,
                          bool lower,
                          cudaStream_t stream) {
    if (n <= 0 || lda < n || batches <= 0) return cudaErrorInvalidValue;

    switch (current_device_sm()) {
        case kSupportedSM:
            return heev_dispatch_for_sm<kSupportedSM>(A, n, lda, lambda, info, batches, compute_vectors, lower, stream);
        default:
            return cudaErrorNotSupported;
    }
}

template <typename T>
cudaError_t htev_dispatch(T* d,
                          int n,
                          T* e,
                          T* V,
                          int ldv,
                          int* info,
                          int batches,
                          bool compute_vectors,
                          cudaStream_t stream) {
    if (n <= 1 || ldv < n || batches <= 0) return cudaErrorInvalidValue;

    switch (current_device_sm()) {
        case kSupportedSM:
            return htev_dispatch_for_sm<kSupportedSM>(d, n, e, V, ldv, info, batches, compute_vectors, stream);
        default:
            return cudaErrorNotSupported;
    }
}

} // namespace
#endif

bool available() {
#if BATCHLAS_HAS_CUSOLVERDX_HEADER
    return current_device_sm() == kSupportedSM;
#else
    return false;
#endif
}

bool heev_supported_n(int n) {
    return n >= kMinHeevN && n <= kMaxSupportedN;
}

bool htev_supported_n(int n) {
    return n >= kMinHtevN && n <= kMaxSupportedN;
}

cudaError_t heev_launch_float(float* A,
                              int n,
                              int lda,
                              float* lambda,
                              int* info,
                              int batches,
                              bool compute_vectors,
                              bool lower,
                              cudaStream_t stream) {
#if BATCHLAS_HAS_CUSOLVERDX_HEADER
    return heev_dispatch<float>(A, n, lda, lambda, info, batches, compute_vectors, lower, stream);
#else
    static_cast<void>(A);
    static_cast<void>(n);
    static_cast<void>(lda);
    static_cast<void>(lambda);
    static_cast<void>(info);
    static_cast<void>(batches);
    static_cast<void>(compute_vectors);
    static_cast<void>(lower);
    static_cast<void>(stream);
    return cudaErrorNotSupported;
#endif
}

cudaError_t heev_launch_double(double* A,
                               int n,
                               int lda,
                               double* lambda,
                               int* info,
                               int batches,
                               bool compute_vectors,
                               bool lower,
                               cudaStream_t stream) {
#if BATCHLAS_HAS_CUSOLVERDX_HEADER
    return heev_dispatch<double>(A, n, lda, lambda, info, batches, compute_vectors, lower, stream);
#else
    static_cast<void>(A);
    static_cast<void>(n);
    static_cast<void>(lda);
    static_cast<void>(lambda);
    static_cast<void>(info);
    static_cast<void>(batches);
    static_cast<void>(compute_vectors);
    static_cast<void>(lower);
    static_cast<void>(stream);
    return cudaErrorNotSupported;
#endif
}

cudaError_t htev_launch_float(float* d,
                              int n,
                              float* e,
                              float* V,
                              int ldv,
                              int* info,
                              int batches,
                              bool compute_vectors,
                              cudaStream_t stream) {
#if BATCHLAS_HAS_CUSOLVERDX_HEADER
    return htev_dispatch<float>(d, n, e, V, ldv, info, batches, compute_vectors, stream);
#else
    static_cast<void>(d);
    static_cast<void>(n);
    static_cast<void>(e);
    static_cast<void>(V);
    static_cast<void>(ldv);
    static_cast<void>(info);
    static_cast<void>(batches);
    static_cast<void>(compute_vectors);
    static_cast<void>(stream);
    return cudaErrorNotSupported;
#endif
}

cudaError_t htev_launch_double(double* d,
                               int n,
                               double* e,
                               double* V,
                               int ldv,
                               int* info,
                               int batches,
                               bool compute_vectors,
                               cudaStream_t stream) {
#if BATCHLAS_HAS_CUSOLVERDX_HEADER
    return htev_dispatch<double>(d, n, e, V, ldv, info, batches, compute_vectors, stream);
#else
    static_cast<void>(d);
    static_cast<void>(n);
    static_cast<void>(e);
    static_cast<void>(V);
    static_cast<void>(ldv);
    static_cast<void>(info);
    static_cast<void>(batches);
    static_cast<void>(compute_vectors);
    static_cast<void>(stream);
    return cudaErrorNotSupported;
#endif
}

} // namespace batchlas::backend::cusolverdx::cuda_kernels

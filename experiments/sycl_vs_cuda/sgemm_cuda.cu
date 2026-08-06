// CUDA half of the head-to-head. The kernel body comes verbatim from
// sgemm_body.h; only the coordinate/shared/barrier bindings are CUDA-specific.
//
// Also runs cuBLAS at three math modes so we can tell whether the vendor
// number everyone has been chasing is true FP32 or TF32 tensor-core.

#include "sgemm_body.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(x)                                                        \
    do {                                                                       \
        cublasStatus_t s_ = (x);                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_,        \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define SG_TX          ((int)threadIdx.x)
#define SG_TY          ((int)threadIdx.y)
#define SG_BM_ID       ((int)blockIdx.x)
#define SG_BN_ID       ((int)blockIdx.y)
#define SG_BATCH_ID    ((int)blockIdx.z)
#define SG_SA          smem_a
#define SG_SB          smem_b
#define SG_BARRIER()   __syncthreads()

__global__ __launch_bounds__(SG_THREADS) void sgemm_batched_kernel(
    int M, int N, int K,
    const float* __restrict__ Ag, int lda, long long strideA,
    const float* __restrict__ Bg, int ldb, long long strideB,
    float* __restrict__ Cg, int ldc, long long strideC,
    float alpha, float beta)
{
    __shared__ float smem_a[SG_BK * SG_BM];
    __shared__ float smem_b[SG_BK * SG_BN];

    SGEMM_BODY(M, N, K, Ag, lda, strideA, Bg, ldb, strideB,
               Cg, ldc, strideC, alpha, beta)
}

// ---------------------------------------------------------------- harness

struct Shape { int m, n, k, batch; };

static double gflop_count(const Shape& s) {
    return 2.0 * s.m * s.n * s.k * (double)s.batch;
}

int main(int argc, char** argv) {
    Shape shape{512, 512, 512, 512};
    int iters = 30, warmup = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() { return std::atoi(argv[++i]); };
        if (a == "--m") shape.m = val();
        else if (a == "--n") shape.n = val();
        else if (a == "--k") shape.k = val();
        else if (a == "--batch") shape.batch = val();
        else if (a == "--iters") iters = val();
        else if (a == "--warmup") warmup = val();
    }
    if (shape.m % SG_BM || shape.n % SG_BN || shape.k % SG_BK) {
        std::fprintf(stderr,
                     "shape must be a multiple of %dx%dx%d for this kernel\n",
                     SG_BM, SG_BN, SG_BK);
        return 1;
    }

    const size_t elemsA = (size_t)shape.m * shape.k * shape.batch;
    const size_t elemsB = (size_t)shape.k * shape.n * shape.batch;
    const size_t elemsC = (size_t)shape.m * shape.n * shape.batch;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, elemsA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, elemsB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, elemsC * sizeof(float)));

    // Deterministic host fill, so the CUDA and SYCL runs verify against the
    // same reference values.
    std::vector<float> hA(elemsA), hB(elemsB);
    for (size_t i = 0; i < elemsA; ++i) hA[i] = (float)((i * 1103515245u + 12345u) % 1000) / 1000.0f - 0.5f;
    for (size_t i = 0; i < elemsB; ++i) hB[i] = (float)((i * 22695477u + 1u) % 1000) / 1000.0f - 0.5f;
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), elemsA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), elemsB * sizeof(float), cudaMemcpyHostToDevice));

    const int lda = shape.m, ldb = shape.k, ldc = shape.m;
    const long long strideA = (long long)shape.m * shape.k;
    const long long strideB = (long long)shape.k * shape.n;
    const long long strideC = (long long)shape.m * shape.n;
    const float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    auto report = [&](const char* label, float ms_total, int n_iters) {
        const double ms = ms_total / n_iters;
        const double tflops = gflop_count(shape) / (ms * 1e-3) / 1e12;
        std::printf("%-28s %9.3f ms   %8.2f TFLOP/s\n", label, ms, tflops);
    };

    // ---- our kernel
    dim3 block(16, 16);
    dim3 grid(shape.m / SG_BM, shape.n / SG_BN, shape.batch);
    auto launch_ours = [&]() {
        sgemm_batched_kernel<<<grid, block>>>(
            shape.m, shape.n, shape.k, dA, lda, strideA, dB, ldb, strideB,
            dC, ldc, strideC, alpha, beta);
    };
    for (int i = 0; i < warmup; ++i) launch_ours();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch_ours();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    report("cuda handwritten", ms, iters);

    // Keep our result for the correctness check against cuBLAS, and print the
    // same checksum the SYCL program prints so the two can be compared.
    std::vector<float> ours(elemsC);
    CUDA_CHECK(cudaMemcpy(ours.data(), dC, elemsC * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (size_t i = 0; i < elemsC; ++i) checksum += (double)ours[i];
    std::printf("checksum: %.6f\n", checksum);

    // ---- cuBLAS at three math modes
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublasMath_t default_mode;
    CUBLAS_CHECK(cublasGetMathMode(handle, &default_mode));
    std::printf("cublas default math mode = %d "
                "(DEFAULT=%d, PEDANTIC=%d, TF32=%d)\n",
                (int)default_mode, (int)CUBLAS_DEFAULT_MATH,
                (int)CUBLAS_PEDANTIC_MATH, (int)CUBLAS_TF32_TENSOR_OP_MATH);

    struct ModeCase { const char* name; cublasMath_t mode; };
    const ModeCase modes[] = {
        {"cublas default", CUBLAS_DEFAULT_MATH},
        {"cublas pedantic (strict fp32)", CUBLAS_PEDANTIC_MATH},
        {"cublas tf32 tensor op", CUBLAS_TF32_TENSOR_OP_MATH},
    };

    std::vector<float> ref(elemsC);
    for (const auto& mc : modes) {
        CUBLAS_CHECK(cublasSetMathMode(handle, mc.mode));
        auto launch_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, shape.m, shape.n, shape.k,
                &alpha, dA, lda, strideA, dB, ldb, strideB, &beta, dC, ldc,
                strideC, shape.batch));
        };
        for (int i = 0; i < warmup; ++i) launch_cublas();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) launch_cublas();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        report(mc.name, ms, iters);
        if (mc.mode == CUBLAS_PEDANTIC_MATH) {
            CUDA_CHECK(cudaMemcpy(ref.data(), dC, elemsC * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    }

    // ---- correctness of our kernel against strict-fp32 cuBLAS
    double max_rel = 0.0;
    for (size_t i = 0; i < elemsC; ++i) {
        const double d = std::fabs((double)ours[i] - (double)ref[i]);
        const double s = std::fabs((double)ref[i]) + 1e-6;
        max_rel = std::max(max_rel, d / s);
    }
    std::printf("max relative error vs cublas pedantic: %.3e  -> %s\n",
                max_rel, max_rel < 1e-4 ? "PASS" : "FAIL");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return max_rel < 1e-4 ? 0 : 2;
}

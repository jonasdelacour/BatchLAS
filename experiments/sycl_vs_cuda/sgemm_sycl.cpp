// SYCL half of the head-to-head. The kernel body comes verbatim from
// sgemm_body.h -- byte-for-byte the same source text the CUDA build compiles.
// Only the thread-coordinate, shared-memory and barrier bindings differ.
//
// Both programs print a checksum of C computed the same way, so a matching
// checksum proves the two kernels compute the same thing and the timing
// comparison is honest.

#include "sgemm_body.h"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define SG_TX          ((int)it.get_local_id(2))
#define SG_TY          ((int)it.get_local_id(1))
#define SG_BM_ID       ((int)it.get_group(2))
#define SG_BN_ID       ((int)it.get_group(1))
#define SG_BATCH_ID    ((int)it.get_group(0))
#define SG_SA          smem_a
#define SG_SB          smem_b
#define SG_BARRIER()   it.barrier(sycl::access::fence_space::local_space)

struct Shape { int m, n, k, batch; };

class SgemmBatchedKernel;

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

    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property::queue::enable_profiling{}};
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    const size_t elemsA = (size_t)shape.m * shape.k * shape.batch;
    const size_t elemsB = (size_t)shape.k * shape.n * shape.batch;
    const size_t elemsC = (size_t)shape.m * shape.n * shape.batch;

    float* dA = sycl::malloc_device<float>(elemsA, q);
    float* dB = sycl::malloc_device<float>(elemsB, q);
    float* dC = sycl::malloc_device<float>(elemsC, q);

    // Identical deterministic fill to the CUDA program.
    std::vector<float> hA(elemsA), hB(elemsB);
    for (size_t i = 0; i < elemsA; ++i) hA[i] = (float)((i * 1103515245u + 12345u) % 1000) / 1000.0f - 0.5f;
    for (size_t i = 0; i < elemsB; ++i) hB[i] = (float)((i * 22695477u + 1u) % 1000) / 1000.0f - 0.5f;
    q.memcpy(dA, hA.data(), elemsA * sizeof(float)).wait();
    q.memcpy(dB, hB.data(), elemsB * sizeof(float)).wait();

    const int lda = shape.m, ldb = shape.k, ldc = shape.m;
    const long long strideA = (long long)shape.m * shape.k;
    const long long strideB = (long long)shape.k * shape.n;
    const long long strideC = (long long)shape.m * shape.n;
    const float alpha = 1.0f, beta = 0.0f;

    const int M = shape.m, N = shape.n, K = shape.k;
    const sycl::range<3> local(1, 16, 16);
    const sycl::range<3> global((size_t)shape.batch,
                                (size_t)(shape.n / SG_BN) * 16,
                                (size_t)(shape.m / SG_BM) * 16);

    auto launch = [&]() {
        return q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> smem_a_acc{sycl::range<1>(SG_BK * SG_BM), h};
            sycl::local_accessor<float, 1> smem_b_acc{sycl::range<1>(SG_BK * SG_BN), h};
            h.parallel_for<SgemmBatchedKernel>(
                sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
                    float* smem_a = smem_a_acc.get_multi_ptr<
                        sycl::access::decorated::no>().get();
                    float* smem_b = smem_b_acc.get_multi_ptr<
                        sycl::access::decorated::no>().get();
                    SGEMM_BODY(M, N, K, dA, lda, strideA, dB, ldb, strideB,
                               dC, ldc, strideC, alpha, beta)
                });
        });
    };

    for (int i = 0; i < warmup; ++i) launch();
    q.wait();

    // Wall clock over the whole batch of launches, plus the summed device-side
    // profiling interval, so we can see whether submission overhead matters.
    const auto w0 = std::chrono::steady_clock::now();
    std::vector<sycl::event> evs;
    evs.reserve(iters);
    for (int i = 0; i < iters; ++i) evs.push_back(launch());
    q.wait();
    const auto w1 = std::chrono::steady_clock::now();

    double wall_ms = std::chrono::duration<double, std::milli>(w1 - w0).count() / iters;
    double dev_ns = 0.0;
    for (auto& e : evs) {
        dev_ns += (double)(e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                           e.get_profiling_info<sycl::info::event_profiling::command_start>());
    }
    const double dev_ms = dev_ns / 1e6 / iters;

    auto report = [&](const char* label, double ms) {
        std::printf("%-28s %9.3f ms   %8.2f TFLOP/s\n", label, ms,
                    gflop_count(shape) / (ms * 1e-3) / 1e12);
    };
    report("sycl handwritten (wall)", wall_ms);
    report("sycl handwritten (device)", dev_ms);

    // Checksum, identical formula to the CUDA program.
    std::vector<float> hC(elemsC);
    q.memcpy(hC.data(), dC, elemsC * sizeof(float)).wait();
    double checksum = 0.0;
    for (size_t i = 0; i < elemsC; ++i) checksum += (double)hC[i];
    std::printf("checksum: %.6f\n", checksum);

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}

// Smoke test: does the DPC++ CUDA backend actually emit tensor-core mma for
// tf32 on sm_89, from portable SYCL joint_matrix?
//
// This is deliberately a correctness/codegen probe, not a tuned GEMM. It does
// no shared-memory staging and no reuse, so its TFLOP/s figure is meaningless.
// What matters is (a) that it compiles for sm_89, (b) that the SASS contains
// HMMA/OMMA tensor-core instructions, and (c) that the result is right.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace sycl::ext::oneapi::experimental::matrix;

class Tf32SmokeKernel;

int main() {
    constexpr int M = 256, N = 256, K = 256;
    constexpr int TM = 16, TN = 16, TK = 8;
    constexpr int SG = 32;

    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = (float)(i % 7) * 0.125f - 0.5f;
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (float)(i % 5) * 0.25f - 0.5f;

    float* dA = sycl::malloc_device<float>(hA.size(), q);
    float* dB = sycl::malloc_device<float>(hB.size(), q);
    float* dC = sycl::malloc_device<float>(hC.size(), q);
    q.memcpy(dA, hA.data(), hA.size() * sizeof(float)).wait();
    q.memcpy(dB, hB.data(), hB.size() * sizeof(float)).wait();

    // One sub-group per 16x16 output tile. Row-major throughout, so a naive
    // host reference is easy to write.
    const sycl::range<2> global((size_t)(M / TM) * 1, (size_t)(N / TN) * SG);
    const sycl::range<2> local(1, SG);

    q.submit([&](sycl::handler& h) {
         h.parallel_for<Tf32SmokeKernel>(
             sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it)
                 [[sycl::reqd_sub_group_size(SG)]] {
                 const auto sg = it.get_sub_group();
                 const int tile_m = (int)it.get_group(0);
                 const int tile_n = (int)it.get_group(1);

                 joint_matrix<sycl::sub_group, precision::tf32, use::a, TM, TK,
                              layout::row_major> a_frag;
                 joint_matrix<sycl::sub_group, precision::tf32, use::b, TK, TN,
                              layout::row_major> b_frag;
                 joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN>
                     acc;
                 joint_matrix_fill(sg, acc, 0.0f);

                 for (int k0 = 0; k0 < K; k0 += TK) {
                     joint_matrix_load(
                         sg, a_frag,
                         sycl::address_space_cast<
                             sycl::access::address_space::global_space,
                             sycl::access::decorated::no>(
                             dA + (size_t)tile_m * TM * K + k0),
                         (size_t)K);
                     joint_matrix_load(
                         sg, b_frag,
                         sycl::address_space_cast<
                             sycl::access::address_space::global_space,
                             sycl::access::decorated::no>(
                             dB + (size_t)k0 * N + tile_n * TN),
                         (size_t)N);
                     joint_matrix_mad(sg, acc, a_frag, b_frag, acc);
                 }

                 joint_matrix_store(
                     sg, acc,
                     sycl::address_space_cast<
                         sycl::access::address_space::global_space,
                         sycl::access::decorated::no>(
                         dC + (size_t)tile_m * TM * N + tile_n * TN),
                     (size_t)N, layout::row_major);
             });
     }).wait();

    q.memcpy(hC.data(), dC, hC.size() * sizeof(float)).wait();

    // tf32 keeps 10 explicit mantissa bits, so tolerance is loose by design.
    double max_rel = 0.0;
    for (int i = 0; i < M; i += 37) {
        for (int j = 0; j < N; j += 41) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += (double)hA[(size_t)i * K + k] * (double)hB[(size_t)k * N + j];
            const double d = std::fabs(ref - (double)hC[(size_t)i * N + j]);
            max_rel = std::max(max_rel, d / (std::fabs(ref) + 1e-6));
        }
    }
    std::printf("max relative error: %.3e -> %s\n", max_rel,
                max_rel < 5e-2 ? "PASS" : "FAIL");

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return max_rel < 5e-2 ? 0 : 2;
}

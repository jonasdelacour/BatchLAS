// SPDX-License-Identifier: MIT
// docs/guide/device-api.md#device-guide-example-syev: a user kernel that computes the
// eigenvalues (ascending) and eigenvectors of a batch of n x n symmetric matrices with
// device::syev, any n: Cta<N> runs one matrix per SgpPartition<N> (32 / N per work-group),
// Blocked<32> one per WorkGroup<256>. Exit 0 when every max |A v - w v| / (n max|A|) and
// max |V^T V - I| / n are small, w ascends and every info is 0, else 1.
#include <batchlas/device/lapack.hh>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace batchlas;
using T = float;

int run(sycl::queue& q, int n, int batch) {
    const int ld = n + 2;
    const long stride = long(ld) * n;
    std::vector<T> ha(std::size_t(stride) * batch, T(0));
    for (int b = 0; b < batch; ++b)  // graded diagonal plus a smooth symmetric coupling
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ha[b * stride + j * ld + i] = (i == j ? T(i % 4 - 2 * (b % 3)) : T(0)) + T(1) / T(2 + i + j);
    T* a = sycl::malloc_device<T>(ha.size(), q);
    T* w = sycl::malloc_device<T>(std::size_t(n) * batch, q);
    int* info_out = sycl::malloc_device<int>(batch, q);
    q.copy(ha.data(), a, ha.size()).wait();

    const char* tier = "";
    using R = device::routine::syev<JobType::EigenVectors, Uplo::Lower>;
    device::with_tier<R, T>(q.get_device(), n, [&]<class Tier>(Tier) {
        tier = device::tier::is_cta_v<Tier> ? "Cta" : "Blocked";
        using L = device::launch<R, Tier, T>;
        const auto sz = L::buffer_size(n);
        std::byte* gws = sz.global_bytes ? sycl::malloc_device<std::byte>(sz.global_bytes * batch, q) : nullptr;
        q.submit([&](sycl::handler& h) {
             auto slm = device::local_workspace<L>(h, sz);
             h.parallel_for(L::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
                 const auto ex = L::exec(it);
                 const auto s = L::slot(it, batch);
                 const int info = L::run(  // device::syev<JobType::EigenVectors, Uplo::Lower, Tier>(...)
                     ex, device::mat(a + s.b * stride, n, n, ld), device::vec(w + s.b * n, n), L::workspace(it, slm, gws, sz),
                     device::SyevOptions<T>{}, s.live);
                 if (s.live && ex.leader()) info_out[s.b] = info;
             });
         }).wait_and_throw();
        if (gws) sycl::free(gws, q);
    });

    std::vector<T> hv(ha.size()), hw(std::size_t(n) * batch);
    std::vector<int> hinfo(batch);
    q.copy(a, hv.data(), hv.size()).wait();
    q.copy(w, hw.data(), hw.size()).wait();
    q.copy(info_out, hinfo.data(), batch).wait();
    sycl::free(a, q);
    sycl::free(w, q);
    sycl::free(info_out, q);
    double res = 0, orth = 0;
    for (int b = 0; b < batch; ++b) {
        if (hinfo[b] != 0) return std::printf("syev_any_n: n=%d matrix %d info %d\n", n, b, hinfo[b]), 1;
        const T* A = &ha[b * stride];
        const T* V = &hv[b * stride];
        const T* W = &hw[std::size_t(b) * n];
        double amax = 0;
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) amax = std::fmax(amax, std::fabs(A[j * ld + i]));
        for (int j = 0; j < n; ++j) {
            if (j > 0 && W[j] < W[j - 1]) return std::printf("syev_any_n: n=%d matrix %d w not ascending\n", n, b), 1;
            for (int i = 0; i < n; ++i) {
                double av = 0;
                for (int k = 0; k < n; ++k) av += double(A[k >= i ? i * ld + k : k * ld + i]) * double(V[j * ld + k]);
                res = std::fmax(res, std::fabs(av - double(W[j]) * double(V[j * ld + i])) / (n * amax));
            }
            for (int c = 0; c <= j; ++c) {
                double d = 0;
                for (int k = 0; k < n; ++k) d += double(V[c * ld + k]) * double(V[j * ld + k]);
                orth = std::fmax(orth, std::fabs(d - (c == j ? 1.0 : 0.0)) / n);
            }
        }
    }
    std::printf("syev_any_n: n=%-4d batch %-4d tier %-7s residual %.2e orthogonality %.2e\n", n, batch, tier, res, orth);
    return res < 64 * 1.2e-7 && orth < 64 * 1.2e-7 ? 0 : 1;
}

int main() {
    sycl::queue q;
    int bad = 0;
    // Cta<4>, Cta<16>, Cta<32>, Blocked<32> on a 4090; batch 253 leaves dead partitions in the
    // last work-group of the packed Cta<4> launch (8 matrices per work-group).
    for (int n : {3, 13, 32, 70}) bad += run(q, n, 256);
    bad += run(q, 3, 253);
    return bad == 0 ? 0 : 1;
}

// SPDX-License-Identifier: MIT
// The device API guide's worked example (docs/guide/device-api.md#device-guide-example-potrf):
// a user kernel that factors a batch of n x n SPD matrices with device::potrf, for any n.
// with_tier picks ONE tier on the host from n and the device's limits; each tier is a separate
// kernel, so no kernel carries the registers of a path it never takes. launch<R, Tier, T> gives
// the tier's launch: nd_range, executor (Reg<N>: a Partition<N>, 32 / N matrices per work-group;
// Cta / Blocked: a WorkGroup per matrix), the matrix slot and its workspace slices.
// Exit 0 when every factor's residual max |L L^T - A| / (n max|A|) is small, else 1.
#include <batchlas/device/lapack.hh>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace batchlas;  // device::, Uplo
using T = float;

int run(sycl::queue& q, int n, int batch) {
    const int ld = n + 3;  // a padded leading dimension
    const long stride = long(ld) * n;
    std::vector<T> host(std::size_t(stride) * batch, T(0));
    for (int b = 0; b < batch; ++b)  // A = I n + a small symmetric perturbation: SPD
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                host[b * stride + j * ld + i] = (i == j ? T(n) : T(0)) + T(1) / T(1 + i + j + b % 7);
    T* a = sycl::malloc_device<T>(host.size(), q);
    int* info_out = sycl::malloc_device<int>(batch, q);
    q.copy(host.data(), a, host.size()).wait();

    const char* tier = "";
    // [device-guide-frame] begin: the launch frame docs/guide/device-api.md quotes
    using R = device::routine::potrf<Uplo::Lower>;
    device::with_tier<R, T>(q.get_device(), n, [&]<class Tier>(Tier) {
        tier = device::tier::is_reg_v<Tier>   ? "Reg"
               : device::tier::is_cta_v<Tier> ? "Cta"
                                              : "Blocked";
        using L = device::launch<R, Tier, T>;
        const auto sz = L::buffer_size(n);  // potrf: no global workspace
        const std::size_t gbytes = sz.global_bytes * batch;
        auto* gws = gbytes ? sycl::malloc_device<std::byte>(gbytes, q) : nullptr;
        q.submit([&](sycl::handler& h) {
             auto slm = device::local_workspace<L>(h, sz);
             h.parallel_for(L::nd_range(batch),
                 [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
                 const auto ex = L::exec(it);
                 const auto s = L::slot(it, batch);  // {b, live}
                 const auto A = device::mat(a + s.b * stride, n, n, ld);
                 const auto ws = L::workspace(it, slm, gws, sz);
                 const int info =
                     device::potrf<Uplo::Lower, Tier>(ex, A, ws, s.live);
                 if (s.live && ex.leader()) info_out[s.b] = info;
             });
         }).wait_and_throw();
        if (gws) sycl::free(gws, q);
    });
    // [device-guide-frame] end

    std::vector<T> l(host.size());
    std::vector<int> hinfo(batch);
    q.copy(a, l.data(), l.size()).wait();
    q.copy(info_out, hinfo.data(), batch).wait();
    sycl::free(a, q);
    sycl::free(info_out, q);
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        if (hinfo[b] != 0) return std::printf("potrf_any_n: n=%d matrix %d info %d\n", n, b, hinfo[b]), 1;
        const T* Lb = &l[b * stride];
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                double s = 0;
                for (int k = 0; k <= j; ++k) s += double(Lb[k * ld + i]) * double(Lb[k * ld + j]);
                worst = std::fmax(worst, std::fabs(s - double(host[b * stride + j * ld + i])) / (n * (n + 1.0)));
            }
    }
    std::printf("potrf_any_n: n=%-4d batch %-4d tier %-7s residual %.2e\n", n, batch, tier, worst);
    return worst < 64 * 1.2e-7 ? 0 : 1;
}

int main() {
    sycl::queue q;
    int bad = 0;
    // Reg<8>, Reg<32>, Cta<128>, Blocked<32> on a 4090; batch 253 leaves dead slots in the last
    // work-group of the packed Reg<8> launch (4 matrices per work-group).
    for (int n : {5, 24, 100, 300}) bad += run(q, n, 256);
    bad += run(q, 5, 253);
    return bad == 0 ? 0 : 1;
}

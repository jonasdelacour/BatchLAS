// SPDX-License-Identifier: MIT
// docs/guide/device-api.md#device-guide-example-getrs: a user kernel that solves
// A X = B for a batch of n x n matrices, any n, by device::getrf then device::getrs in one kernel
// (getrs's tiers share getrf's executor and local plan). Exit 0 when every residual
// max |A X - B| / (n max|A| max|X|) is small and every info is 0, else 1.
#include <batchlas/device/lapack.hh>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace batchlas;
using T = float;

int run(sycl::queue& q, int n, int nrhs, int batch) {
    const int lda = n + 3, ldb = n + 1;
    const long sa = long(lda) * n, sb = long(ldb) * nrhs;
    std::vector<T> ha(std::size_t(sa) * batch, T(0)), hb(std::size_t(sb) * batch, T(0));
    for (int b = 0; b < batch; ++b) {  // a weighted cyclic shift (pivots away from the diagonal) plus a smooth part
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ha[b * sa + j * lda + i] = (i == (j + 1) % n ? T(4) : T(0)) + T(1) / T(1 + i + 2 * j + b % 5);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) hb[b * sb + j * ldb + i] = T(1 + (i * 7 + j * 3 + b) % 11) / T(11);
    }
    T* a = sycl::malloc_device<T>(ha.size(), q);
    T* x = sycl::malloc_device<T>(hb.size(), q);
    int* ipiv = sycl::malloc_device<int>(std::size_t(n) * batch, q);
    int* info_out = sycl::malloc_device<int>(batch, q);
    q.copy(ha.data(), a, ha.size()).wait();
    q.copy(hb.data(), x, hb.size()).wait();

    // [device-guide-getrs]
    using R = device::routine::getrs<Transpose::NoTrans>;
    device::with_tier<R, T>(q.get_device(), n, [&]<class Tier>(Tier) {
        using L = device::launch<R, Tier, T>;
        const auto sz = L::buffer_size(n);
        std::byte* gws = sz.global_bytes ? sycl::malloc_device<std::byte>(sz.global_bytes * batch, q) : nullptr;
        q.submit([&](sycl::handler& h) {
             auto slm = device::local_workspace<L>(h, sz);
             h.parallel_for(L::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
                 const auto ex = L::exec(it);
                 const auto s = L::slot(it, batch);
                 const auto ws = L::workspace(it, slm, gws, sz);
                 const auto A = device::mat(a + s.b * sa, n, n, lda);
                 const auto P = device::vec(ipiv + s.b * n, n);
                 const auto B = device::mat(x + s.b * sb, n, nrhs, ldb);
                 const int info = device::getrf<Tier>(ex, A, P, ws, s.live);
                 device::getrs<Transpose::NoTrans, Tier>(ex, A, P, B, ws, s.live);
                 if (s.live && ex.leader()) info_out[s.b] = info;
             });
         }).wait_and_throw();
        if (gws) sycl::free(gws, q);
    });
    // [device-guide-getrs]

    std::vector<T> hx(hb.size());
    std::vector<int> hinfo(batch);
    q.copy(x, hx.data(), hx.size()).wait();
    q.copy(info_out, hinfo.data(), batch).wait();
    for (void* p : {static_cast<void*>(a), static_cast<void*>(x), static_cast<void*>(ipiv), static_cast<void*>(info_out)})
        sycl::free(p, q);
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        if (hinfo[b] != 0) return std::printf("getrs_any_n: n=%d matrix %d info %d\n", n, b, hinfo[b]), 1;
        double amax = 0, xmax = 0, rmax = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(ha[b * sa + j * lda + i]));
        for (int c = 0; c < nrhs; ++c)
            for (int i = 0; i < n; ++i) {
                double r = -double(hb[b * sb + c * ldb + i]);
                for (int k = 0; k < n; ++k) r += double(ha[b * sa + k * lda + i]) * double(hx[b * sb + c * ldb + k]);
                rmax = std::fmax(rmax, std::fabs(r));
                xmax = std::fmax(xmax, std::fabs(hx[b * sb + c * ldb + i]));
            }
        worst = std::fmax(worst, rmax / (n * amax * xmax));
    }
    std::printf("getrs_any_n: n=%-4d nrhs %d batch %-4d residual %.2e\n", n, nrhs, batch, worst);
    return worst < 64 * 1.2e-7 ? 0 : 1;
}

int main() {
    sycl::queue q;
    int bad = 0;
    for (int n : {5, 40, 100}) bad += run(q, n, 3, 128);
    return bad == 0 ? 0 : 1;
}

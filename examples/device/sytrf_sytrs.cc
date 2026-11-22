// SPDX-License-Identifier: MIT
// docs/guide/device-api.md, sytrf / sytrs: a user kernel that solves A X = B for a batch of n x n
// symmetric indefinite matrices, any n, by the Bunch-Kaufman factor A = P L D L^T P^T
// (device::sytrf, 1 x 1 and 2 x 2 pivots, ipiv 1-based) then device::sytrs, in one kernel.
// with_tier picks sytrf's tier (Cta<32>: one sub-group, n <= 32; Blocked<1>: a WorkGroup<256>);
// sytrs offers both on the same executor, so it takes that tier too, sharing one workspace sized
// for the larger of the two. float, and double where the device has fp64. Exit 0 when every
// max |A X - B| / (n max|A| max|X|) is small and every info is 0, else 1.
#include <batchlas/device/lapack.hh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace batchlas;

// Two calls in sequence on one workspace: each carves its slices from offset 0.
device::WorkspaceSize larger(const device::WorkspaceSize& x, const device::WorkspaceSize& y) {
    return {std::max(x.local_bytes, y.local_bytes), std::max(x.global_bytes, y.global_bytes), std::max(x.align, y.align)};
}

template <class T> int run(sycl::queue& q, const char* type, int n, int nrhs, int batch) {
    const int lda = n + 3, ldb = n + 1;  // padded leading dimensions
    const long sa = long(lda) * n, sb = long(ldb) * nrhs;
    std::vector<T> ha(std::size_t(sa) * batch, T(0)), hb(std::size_t(sb) * batch, T(0));
    for (int b = 0; b < batch; ++b) {  // zeros on every third diagonal entry force 2 x 2 pivots
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ha[b * sa + j * lda + i] = i == j ? T(i % 3 == 0 ? 0 : (i % 2 ? 4 : -4))
                                                  : T(int(((i + j) * 5 + std::abs(i - j) * 3 + b) % 11) - 5) / T(5);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) hb[b * sb + j * ldb + i] = T(1 + (i * 7 + j * 3 + b) % 11) / T(11);
    }
    T* a = sycl::malloc_device<T>(ha.size(), q);
    T* x = sycl::malloc_device<T>(hb.size(), q);
    int* ipiv = sycl::malloc_device<int>(std::size_t(n) * batch, q);
    int* info_out = sycl::malloc_device<int>(batch, q);
    q.copy(ha.data(), a, ha.size()).wait();
    q.copy(hb.data(), x, hb.size()).wait();

    const char* tier = "";
    //! [launch]
    using R = device::routine::sytrf<Uplo::Lower>;
    device::with_tier<R, T>(q.get_device(), n, [&]<class Tier>(Tier) {
        tier = device::tier::is_cta_v<Tier> ? "Cta<32>" : "Blocked<1>";
        using L = device::launch<R, Tier, T>;
        const auto sz = larger(L::buffer_size(n), device::buffer_size<device::routine::sytrs<Uplo::Lower>, Tier, T>(n, nrhs));
        std::byte* gws = sz.global_bytes ? sycl::malloc_device<std::byte>(sz.global_bytes * batch, q) : nullptr;
        q.submit([&](sycl::handler& h) {
             auto slm = device::local_workspace<L>(h, sz);
             h.parallel_for(L::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
                 const auto ex = L::exec(it);
                 const auto s = L::slot(it, batch);
                 const auto ws = L::workspace(it, slm, gws, sz);
                 //! [call]
                 const auto A = device::mat(a + s.b * sa, n, n, lda);  // the lower triangle is read
                 const auto P = device::vec(ipiv + s.b * n, n);
                 const auto B = device::mat(x + s.b * sb, n, nrhs, ldb);
                 int info = device::sytrf<Uplo::Lower, Tier>(ex, A, P, ws);
                 if (info == 0) info = device::sytrs<Uplo::Lower, Tier>(ex, A, P, B, ws);  // info is uniform
                 //! [call]
                 if (s.live && ex.leader()) info_out[s.b] = info;
             });
         }).wait_and_throw();
        if (gws) sycl::free(gws, q);
    });
    //! [launch]

    std::vector<T> hx(hb.size());
    std::vector<int> hinfo(batch);
    q.copy(x, hx.data(), hx.size()).wait();
    q.copy(info_out, hinfo.data(), batch).wait();
    for (void* p : std::initializer_list<void*>{a, x, ipiv, info_out}) sycl::free(p, q);
    //! [check]
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        if (hinfo[b] != 0) return std::printf("sytrf_sytrs: %s n=%d matrix %d info %d\n", type, n, b, hinfo[b]), 1;
        const T* A = &ha[b * sa];
        const T* X = &hx[b * sb];
        double amax = 0, xmax = 0, rmax = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(A[j * lda + i]));
        for (int c = 0; c < nrhs; ++c)
            for (int i = 0; i < n; ++i) {
                double r = -double(hb[b * sb + c * ldb + i]);
                for (int k = 0; k < n; ++k) r += double(A[k * lda + i]) * double(X[c * ldb + k]);
                rmax = std::fmax(rmax, std::fabs(r));
                xmax = std::fmax(xmax, std::fabs(X[c * ldb + i]));
            }
        worst = std::fmax(worst, rmax / (n * amax * xmax));
    }
    std::printf("sytrf_sytrs: %-6s n=%-4d nrhs %d batch %-4d tier %-10s residual %.2e\n", type, n, nrhs, batch, tier, worst);
    return worst < 16 * double(std::numeric_limits<T>::epsilon()) ? 0 : 1;
    //! [check]
}

template <class T> int run_all(sycl::queue& q, const char* type) {
    int bad = 0;
    for (int n : {7, 32, 33, 100}) bad += run<T>(q, type, n, 3, 128);  // Cta<32> up to n = 32, then Blocked<1>
    return bad;
}

int main() {
    sycl::queue q;
    int bad = run_all<float>(q, "float");
    if (q.get_device().has(sycl::aspect::fp64)) bad += run_all<double>(q, "double");
    else std::printf("sytrf_sytrs: double not run: the device has no fp64\n");
    return bad == 0 ? 0 : 1;
}

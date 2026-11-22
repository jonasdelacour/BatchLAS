// SPDX-License-Identifier: MIT
// docs/guide/device-api.md#device-guide-example-gemm: a user kernel that runs
// C := alpha A B^T + beta C on a batch of matrices with the in-group device::gemm, one matrix per
// work-group, any m, n, k. No tier: the l3::cfg preset fixes the executor (a WorkGroup of
// Cfg::threads items) and the local stage (l3::smem_bytes<Cfg>()). Exit 0 when every
// max |C - C_ref| / (k |alpha| max|A| max|B| + |beta| max|C0|) is small, else 1.
#include <batchlas/device.hh>
#include <batchlas/device/level3/group.hh>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace batchlas;
using T = float;
using Cfg = device::l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>;  // 256 items, one local stage
using Ex = device::WorkGroup<Cfg::threads>;

int run(sycl::queue& q, int m, int n, int k, int batch) {
    const int lda = m + 1, ldb = n + 3, ldc = m + 2;
    const long sa = long(lda) * k, sb = long(ldb) * k, sc = long(ldc) * n;
    std::vector<T> ha(std::size_t(sa) * batch, T(0)), hb(std::size_t(sb) * batch, T(0)), hc(std::size_t(sc) * batch, T(0));
    for (std::size_t e = 0; e < ha.size(); ++e) ha[e] = T(int(e * 37 % 17) - 8) / T(8);
    for (std::size_t e = 0; e < hb.size(); ++e) hb[e] = T(int(e * 53 % 13) - 6) / T(6);
    for (std::size_t e = 0; e < hc.size(); ++e) hc[e] = T(int(e * 29 % 11) - 5) / T(5);
    T* a = sycl::malloc_device<T>(ha.size(), q);
    T* b = sycl::malloc_device<T>(hb.size(), q);
    T* c = sycl::malloc_device<T>(hc.size(), q);
    q.copy(ha.data(), a, ha.size()).wait();
    q.copy(hb.data(), b, hb.size()).wait();
    q.copy(hc.data(), c, hc.size()).wait();
    const T alpha = T(1.5), beta = T(-0.5);

    // [device-guide-gemm]
    constexpr std::size_t stage = device::l3::smem_bytes<Cfg>();
    q.submit([&](sycl::handler& h) {
         device::LocalWorkspace slm(sycl::range<1>((stage + 15) / 16), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * Cfg::threads, Cfg::threads),
                        [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                            const Ex ex{it};
                            const long g = long(it.get_group_linear_id());
                            auto* base = reinterpret_cast<std::byte*>(slm.get_multi_ptr<sycl::access::decorated::no>().get());
                            const device::Workspace ws{
                                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(base),
                                nullptr};
                            device::gemm<Transpose::NoTrans, Transpose::Trans, Cfg>(
                                ex, alpha, device::mat(a + g * sa, m, k, lda), device::mat(b + g * sb, n, k, ldb), beta,
                                device::mat(c + g * sc, m, n, ldc), ws);
                        });
     }).wait_and_throw();
    // [device-guide-gemm]

    std::vector<T> got(hc.size());
    q.copy(c, got.data(), got.size()).wait();
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    double worst = 0;
    for (int g = 0; g < batch; ++g)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double ref = double(beta) * double(hc[g * sc + j * ldc + i]);
                for (int p = 0; p < k; ++p) ref += double(alpha) * double(ha[g * sa + p * lda + i]) * double(hb[g * sb + p * ldb + j]);
                worst = std::fmax(worst, std::fabs(double(got[g * sc + j * ldc + i]) - ref) / (k * 1.5 + 0.5));
            }
    std::printf("gemm_group: m=%-3d n=%-3d k=%-3d batch %-4d error %.2e\n", m, n, k, batch, worst);
    return worst < 16 * 1.2e-7 ? 0 : 1;
}

int main() {
    sycl::queue q;
    int bad = 0;
    bad += run(q, 70, 45, 33, 128);  // edge tiles in m, n and k (32 x 32 tiles, k slabs of 8)
    bad += run(q, 32, 32, 8, 128);   // one full tile
    return bad == 0 ? 0 : 1;
}

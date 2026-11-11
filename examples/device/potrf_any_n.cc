// SPDX-License-Identifier: MIT
// The device API guide's worked example (docs/design/device-la.md#device-la-worked-examples):
// a user kernel that factors a batch of n x n SPD matrices with device::potrf, for any n.
// with_tier picks ONE tier on the host from n and the device's limits; each tier is a
// separate kernel, so no kernel carries the registers of a path it never takes.
//   tier::Reg<N>       one matrix per N lanes of a 32-wide sub-group, no workspace
//   tier::Cta<N>       one matrix per work-group, in local memory (potrf_buffer_size bytes)
//   tier::Blocked<NB>  one matrix per WorkGroup<256, 32>, any n, A in global memory
// Exit 0 when every factor's residual max |L L^T - A| / (n max|A|) is small, else 1.
#include <batchlas/device/lapack/potrf.hh>

#include <cmath>
#include <cstdio>
#include <vector>

using batchlas::Uplo;
using T = float;
using Routine = batchlas::device::routine::potrf<Uplo::Lower>;

struct alignas(16) Slab {
    unsigned char b[16];
};

template <class Tier> void factor(sycl::queue& q, T* a, int n, int ld, int batch, int* info) {
    constexpr int wg = batchlas::device::tier_info<Routine, Tier, T>::wg_size;
    const long stride = long(ld) * n;
    if constexpr (batchlas::device::tier::is_reg_v<Tier>) {
        constexpr int N = Tier::n_max, per = wg / N;  // matrices per work-group
        const int groups = batch / per;  // batch is a multiple of 32: no partition idles
        q.parallel_for(sycl::nd_range<1>(std::size_t(groups) * wg, wg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const batchlas::device::Partition<N> ex{it.get_sub_group()};
             const int b = int(it.get_group_linear_id()) * per + int(it.get_local_linear_id()) / N;
             const int r = batchlas::device::potrf<Uplo::Lower, Tier>(ex, batchlas::device::mat(a + b * stride, n, n, ld));
             if (ex.leader()) info[b] = r;
         }).wait_and_throw();
    } else {
        const batchlas::device::WorkspaceSize ws = batchlas::device::potrf_buffer_size<Uplo::Lower, Tier, T>(n);
        q.submit([&](sycl::handler& h) {
             sycl::local_accessor<Slab, 1> slm(sycl::range<1>((ws.local_bytes + 15) / 16 + 1), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * wg, wg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                 const batchlas::device::WorkGroup<wg, 32> ex{it};
                 const int b = int(it.get_group_linear_id());
                 auto* raw = reinterpret_cast<std::byte*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get());
                 const batchlas::device::Workspace w{
                     sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(raw), nullptr};
                 const int r = batchlas::device::potrf<Uplo::Lower, Tier>(ex, batchlas::device::mat(a + b * stride, n, n, ld), w);
                 if (ex.leader()) info[b] = r;
             });
         }).wait_and_throw();
    }
}

int run(sycl::queue& q, int n, int batch) {
    const int ld = n + 3;  // a padded leading dimension
    const long stride = long(ld) * n;
    std::vector<T> host(std::size_t(stride) * batch, T(0));
    for (int b = 0; b < batch; ++b)  // A = I n + a small symmetric perturbation: SPD
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                host[b * stride + j * ld + i] = (i == j ? T(n) : T(0)) + T(1) / T(1 + i + j + b % 7);
    T* a = sycl::malloc_device<T>(host.size(), q);
    int* info = sycl::malloc_device<int>(batch, q);
    q.copy(host.data(), a, host.size()).wait();

    const char* tier = "";
    batchlas::device::with_tier<Routine, T>(q.get_device(), n, [&]<class Tier>(Tier) {
        tier = batchlas::device::tier::is_reg_v<Tier> ? "Reg" : batchlas::device::tier::is_cta_v<Tier> ? "Cta" : "Blocked";
        factor<Tier>(q, a, n, ld, batch, info);
    });

    std::vector<T> l(host.size());
    std::vector<int> hinfo(batch);
    q.copy(a, l.data(), l.size()).wait();
    q.copy(info, hinfo.data(), batch).wait();
    sycl::free(a, q);
    sycl::free(info, q);
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        if (hinfo[b] != 0) return std::printf("potrf_any_n: n=%d matrix %d info %d\n", n, b, hinfo[b]), 1;
        const T* L = &l[b * stride];
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                double s = 0;
                for (int k = 0; k <= j; ++k) s += double(L[k * ld + i]) * double(L[k * ld + j]);
                worst = std::fmax(worst, std::fabs(s - double(host[b * stride + j * ld + i])) / (n * (n + 1.0)));
            }
    }
    std::printf("potrf_any_n: n=%-4d tier %-7s residual %.2e\n", n, tier, worst);
    return worst < 64 * 1.2e-7 ? 0 : 1;
}

int main() {
    sycl::queue q;
    int bad = 0;
    for (int n : {5, 24, 100, 300}) bad += run(q, n, 256);  // Reg<8>, Reg<32>, Cta<128>, Blocked<32> on a 4090
    return bad == 0 ? 0 : 1;
}

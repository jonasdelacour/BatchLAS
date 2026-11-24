// SPDX-License-Identifier: MIT
// docs/guide/device-api.md, geqrf / orgqr: a user kernel that factors a batch of m x n (m >= n)
// matrices A = Q R with device::geqrf, copies the factor into Q's storage and forms Q's n columns
// there with device::orgqr, in one kernel. Both routines offer one tier, Blocked<NB>
// (NB = larfb_default_nb<T>(): 32 for float, 16 for double; 32 for both complex types), on a
// WorkGroup per matrix: with_tier picks it, both calls take it
// and share one workspace sized for the larger of the two. float, and double where the device has
// fp64. Exit 0 when max |A - Q R| / (m max|A|) and max |Q^T Q - I| / m are small and every info
// is 0, else 1.
#include <batchlas/device/lapack.hh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

using namespace batchlas;

// Two calls in sequence on one workspace: each carves its slices from offset 0.
device::WorkspaceSize larger(const device::WorkspaceSize& x, const device::WorkspaceSize& y) {
    return {std::max(x.local_bytes, y.local_bytes), std::max(x.global_bytes, y.global_bytes), std::max(x.align, y.align)};
}

template <class T> int run(sycl::queue& q, const char* type, int m, int n, int batch) {
    const int lda = m + 3, ldq = m + 1;  // padded leading dimensions
    const long sa = long(lda) * n, sq = long(ldq) * n;
    std::vector<T> ha(std::size_t(sa) * batch, T(0));
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                ha[b * sa + j * lda + i] = T(int((i * 7 + j * 13 + b * 3) % 17) - 8) / T(8) + (i == j ? T(1) : T(0));
    T* a = sycl::malloc_device<T>(ha.size(), q);
    T* qm = sycl::malloc_device<T>(std::size_t(sq) * batch, q);
    T* tau = sycl::malloc_device<T>(std::size_t(n) * batch, q);
    int* info_out = sycl::malloc_device<int>(batch, q);
    q.copy(ha.data(), a, ha.size()).wait();

    int nb = 0;
    //! [launch]
    using GeqrfRoutine = device::routine::geqrf;
    device::with_tier<GeqrfRoutine, T>(q.get_device(), n, [&]<class Tier>(Tier) {
        nb = Tier::nb;
        using Geqrf = device::launch<GeqrfRoutine, Tier, T>;
        using Orgqr = device::launch<device::routine::orgqr, Tier, T>;
        const auto sz = larger(Geqrf::buffer_size(m, n), Orgqr::buffer_size(m, n));
        std::byte* gws = sz.global_bytes ? sycl::malloc_device<std::byte>(sz.global_bytes * batch, q) : nullptr;
        q.submit([&](sycl::handler& h) {
             auto slm = device::local_workspace<Geqrf>(h, sz);
             h.parallel_for(Geqrf::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(Geqrf::sg_size) {
                 const auto ex = Geqrf::exec(it);
                 const auto s = Geqrf::slot(it, batch);
                 const auto ws = Geqrf::workspace(it, slm, gws, sz);
                 //! [call]
                 const auto A = device::mat(a + s.b * sa, m, n, lda);
                 const auto Q = device::mat(qm + s.b * sq, m, n, ldq);
                 const auto t = device::vec(tau + s.b * n, n);
                 int info = Geqrf::run(ex, A, t, ws);  // R above A's diagonal, the reflectors below
                 device::lacpy(ex, A, Q);
                 ex.barrier(device::Mem::Global);  // the copy complete before orgqr reads it
                 if (info == 0) info = Orgqr::run(ex, Q, t, ws);  // info is uniform
                 //! [call]
                 if (s.live && ex.leader()) info_out[s.b] = info;
             });
         }).wait_and_throw();
        if (gws) sycl::free(gws, q);
    });
    //! [launch]

    std::vector<T> hr(ha.size()), hq(std::size_t(sq) * batch);
    std::vector<int> hinfo(batch);
    q.copy(a, hr.data(), hr.size()).wait();
    q.copy(qm, hq.data(), hq.size()).wait();
    q.copy(info_out, hinfo.data(), batch).wait();
    for (void* p : std::initializer_list<void*>{a, qm, tau, info_out}) sycl::free(p, q);
    //! [check]
    double res = 0, orth = 0;
    for (int b = 0; b < batch; ++b) {
        if (hinfo[b] != 0) return std::printf("geqrf_orgqr: %s m=%d n=%d matrix %d info %d\n", type, m, n, b, hinfo[b]), 1;
        const T* A = &ha[b * sa];
        const T* Rb = &hr[b * sa];
        const T* Q = &hq[b * sq];
        double amax = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) amax = std::fmax(amax, std::fabs(A[j * lda + i]));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                double qr = 0;
                for (int k = 0; k <= j; ++k) qr += double(Q[k * ldq + i]) * double(Rb[j * lda + k]);
                res = std::fmax(res, std::fabs(qr - double(A[j * lda + i])) / (m * amax));
            }
            for (int c = 0; c <= j; ++c) {
                double d = 0;
                for (int k = 0; k < m; ++k) d += double(Q[c * ldq + k]) * double(Q[j * ldq + k]);
                orth = std::fmax(orth, std::fabs(d - (c == j ? 1.0 : 0.0)) / m);
            }
        }
    }
    const double tol = 16 * double(std::numeric_limits<T>::epsilon());
    std::printf("geqrf_orgqr: %-6s m=%-4d n=%-4d batch %-4d tier Blocked<%d> residual %.2e orthogonality %.2e\n", type, m, n,
                batch, nb, res, orth);
    return res < tol && orth < tol ? 0 : 1;
    //! [check]
}

template <class T> int run_all(sycl::queue& q, const char* type) {
    int bad = 0;
    // One panel; several panels with a partial last one (NB 32 for float, 16 for double); square.
    for (auto [m, n] : {std::pair{12, 8}, std::pair{75, 40}, std::pair{70, 70}}) bad += run<T>(q, type, m, n, 128);
    return bad;
}

int main() {
    sycl::queue q;
    int bad = run_all<float>(q, "float");
    if (q.get_device().has(sycl::aspect::fp64)) bad += run_all<double>(q, "double");
    else std::printf("geqrf_orgqr: double not run: the device has no fp64\n");
    return bad == 0 ? 0 : 1;
}

// SPDX-License-Identifier: MIT
// docs/guide/device-api.md, gesvd: a user kernel that computes A = U S V^T for a batch of m x n
// matrices, any m and n (tall, square or wide), with device::gesvd<SvdVectors::All,
// SvdVectors::All, Tier>: S the k = min(m, n) singular values descending, U m x m, V^T n x n.
// gesvd offers one tier, Blocked<NB> (gebrd's panel width) on a WorkGroup<256> per matrix:
// with_tier picks it. float, and double where the device has fp64. Exit 0 when
// max |A - U S V^T| / (max(m, n) max|A|), max |U^T U - I| / m and max |V^T V - I| / n are small,
// S descends and is non-negative and every info is 0, else 1.
#include <batchlas/device/lapack.hh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

using namespace batchlas;

// max |M^T M - I| / r over the r x r matrix M (column-major, leading dimension ld).
template <class T> double orth_defect(const T* M, int r, int ld) {
    double worst = 0;
    for (int j = 0; j < r; ++j)
        for (int c = 0; c <= j; ++c) {
            double d = 0;
            for (int k = 0; k < r; ++k) d += double(M[c * ld + k]) * double(M[j * ld + k]);
            worst = std::fmax(worst, std::fabs(d - (c == j ? 1.0 : 0.0)) / r);
        }
    return worst;
}

template <class T> int run(sycl::queue& q, const char* type, int m, int n, int batch) {
    const int k = std::min(m, n);
    const int lda = m + 3, ldu = m + 2, ldv = n + 1;  // padded leading dimensions
    const long sa = long(lda) * n, su = long(ldu) * m, sv = long(ldv) * n;
    std::vector<T> ha(std::size_t(sa) * batch, T(0));
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                ha[b * sa + j * lda + i] = T(int((i * 11 + j * 5 + b * 7) % 19) - 9) / T(9) + (i == j ? T(2) : T(0));
    T* a = sycl::malloc_device<T>(ha.size(), q);
    T* s_out = sycl::malloc_device<T>(std::size_t(k) * batch, q);
    T* u = sycl::malloc_device<T>(std::size_t(su) * batch, q);
    T* vt = sycl::malloc_device<T>(std::size_t(sv) * batch, q);
    int* info_out = sycl::malloc_device<int>(batch, q);
    q.copy(ha.data(), a, ha.size()).wait();

    int nb = 0;
    //! [launch]
    using GesvdRoutine = device::routine::gesvd<SvdVectors::All, SvdVectors::All>;
    device::with_tier<GesvdRoutine, T>(q.get_device(), std::max(m, n), [&]<class Tier>(Tier) {
        nb = Tier::nb;
        using Gesvd = device::launch<GesvdRoutine, Tier, T>;
        const auto sz = Gesvd::buffer_size(m, n);
        std::byte* gws = sz.global_bytes ? sycl::malloc_device<std::byte>(sz.global_bytes * batch, q) : nullptr;
        q.submit([&](sycl::handler& h) {
             auto slm = device::local_workspace<Gesvd>(h, sz);
             h.parallel_for(Gesvd::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(Gesvd::sg_size) {
                 const auto ex = Gesvd::exec(it);
                 const auto s = Gesvd::slot(it, batch);
                 //! [call]
                 const int info = Gesvd::run(  // device::gesvd<SvdVectors::All, SvdVectors::All, Tier>(...)
                     ex, device::mat(a + s.b * sa, m, n, lda), device::vec(s_out + s.b * k, k), device::mat(u + s.b * su, m, m, ldu),
                     device::mat(vt + s.b * sv, n, n, ldv), Gesvd::workspace(it, slm, gws, sz));  // A is overwritten
                 //! [call]
                 if (s.live && ex.leader()) info_out[s.b] = info;
             });
         }).wait_and_throw();
        if (gws) sycl::free(gws, q);
    });
    //! [launch]

    std::vector<T> hs(std::size_t(k) * batch), hu(std::size_t(su) * batch), hv(std::size_t(sv) * batch);
    std::vector<int> hinfo(batch);
    q.copy(s_out, hs.data(), hs.size()).wait();
    q.copy(u, hu.data(), hu.size()).wait();
    q.copy(vt, hv.data(), hv.size()).wait();
    q.copy(info_out, hinfo.data(), batch).wait();
    for (void* p : std::initializer_list<void*>{a, s_out, u, vt, info_out}) sycl::free(p, q);
    //! [check]
    double res = 0, orth = 0;
    for (int b = 0; b < batch; ++b) {
        if (hinfo[b] != 0) return std::printf("gesvd_any_n: %s m=%d n=%d matrix %d info %d\n", type, m, n, b, hinfo[b]), 1;
        const T* A = &ha[b * sa];
        const T* S = &hs[std::size_t(b) * k];
        const T* U = &hu[b * su];
        const T* Vt = &hv[b * sv];
        for (int p = 0; p < k; ++p)
            if (S[p] < T(0) || (p > 0 && S[p] > S[p - 1]))
                return std::printf("gesvd_any_n: %s m=%d n=%d matrix %d S not descending and >= 0\n", type, m, n, b), 1;
        double amax = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) amax = std::fmax(amax, std::fabs(A[j * lda + i]));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double usv = 0;
                for (int p = 0; p < k; ++p) usv += double(U[p * ldu + i]) * double(S[p]) * double(Vt[j * ldv + p]);
                res = std::fmax(res, std::fabs(usv - double(A[j * lda + i])) / (std::max(m, n) * amax));
            }
        std::vector<T> v(std::size_t(n) * n);  // V = (V^T)^T, so V^T V - I is the rows' defect
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) v[j * n + i] = Vt[i * ldv + j];
        orth = std::fmax(orth, std::fmax(orth_defect(U, m, ldu), orth_defect(v.data(), n, n)));
    }
    const double tol = 16 * double(std::numeric_limits<T>::epsilon());
    std::printf("gesvd_any_n: %-6s m=%-4d n=%-4d batch %-4d tier Blocked<%d> residual %.2e orthogonality %.2e\n", type, m, n, batch, nb, res,
                orth);
    return res < tol && orth < tol ? 0 : 1;
    //! [check]
}

template <class T> int run_all(sycl::queue& q, const char* type) {
    int bad = 0;
    // Tall, wide (gesvd solves A^T and returns the transposes) and square, across gebrd's panels.
    for (auto [m, n] : {std::pair{9, 6}, std::pair{40, 33}, std::pair{33, 40}, std::pair{64, 64}}) bad += run<T>(q, type, m, n, 128);
    return bad;
}

int main() {
    sycl::queue q;
    int bad = run_all<float>(q, "float");
    if (q.get_device().has(sycl::aspect::fp64)) bad += run_all<double>(q, "double");
    else std::printf("gesvd_any_n: double not run: the device has no fp64\n");
    return bad == 0 ? 0 : 1;
}

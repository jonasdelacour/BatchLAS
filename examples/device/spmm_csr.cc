// SPDX-License-Identifier: MIT
// docs/guide/device-api.md, spmm: a user kernel that runs C := alpha A B + beta C on a batch of
// CSR matrices A (m x k, each with its own sparsity pattern) and dense B (k x n), C (m x n), with
// device::spmm<Transpose::NoTrans, Transpose::NoTrans, Tier>. A batch item is a device::CsrItem:
// values and column indices at nz_base = b * nz_stride, item-local row offsets at
// ro_base = b * ro_stride. spmm offers one tier, Blocked<NC> (NC = 64 / sizeof(T) columns per
// item), needs no workspace and runs on any executor; launch gives the tested WorkGroup<128>.
// float, and double where the device has fp64. Exit 0 when every
// max |C - C_ref| / (|alpha| nnz_row max|A| max|B| + |beta| max|C0|) is small, else 1.
#include <batchlas/device.hh>
#include <batchlas/device/sparse.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace batchlas;

template <class T> int run(sycl::queue& q, const char* type, int m, int k, int n, int batch) {
    // Item b keeps A(r, c) where (3 r + 5 c + b) % 7 < 2, plus the "diagonal" c = r % k.
    std::vector<std::vector<int>> cols(batch);
    std::vector<std::vector<int>> offs(batch, std::vector<int>(1, 0));
    std::size_t nz_stride = 0;
    int row_max = 0;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < k; ++c)
                if ((3 * r + 5 * c + b) % 7 < 2 || c == r % k) cols[b].push_back(c);
            row_max = std::max(row_max, int(cols[b].size()) - offs[b].back());
            offs[b].push_back(int(cols[b].size()));
        }
        nz_stride = std::max(nz_stride, cols[b].size() + 3);  // padded, as the strides of a real batch
    }
    const int ro_stride = m + 2, ldb = k + 3, ldc = m + 2;
    const long sb = long(ldb) * n, sc = long(ldc) * n;
    std::vector<T> hval(nz_stride * batch, T(0)), hb(std::size_t(sb) * batch, T(0)), hc(std::size_t(sc) * batch, T(0));
    std::vector<int> hcol(nz_stride * batch, 0), hro(std::size_t(ro_stride) * batch, 0);
    for (int b = 0; b < batch; ++b) {
        for (std::size_t p = 0; p < cols[b].size(); ++p) {
            hval[b * nz_stride + p] = T(int((p * 37 + b * 11) % 17) - 8) / T(8);
            hcol[b * nz_stride + p] = cols[b][p];
        }
        for (int r = 0; r <= m; ++r) hro[std::size_t(b) * ro_stride + r] = offs[b][r];
    }
    for (std::size_t e = 0; e < hb.size(); ++e) hb[e] = T(int(e * 53 % 13) - 6) / T(6);
    for (std::size_t e = 0; e < hc.size(); ++e) hc[e] = T(int(e * 29 % 11) - 5) / T(5);
    T* val = sycl::malloc_device<T>(hval.size(), q);
    int* col = sycl::malloc_device<int>(hcol.size(), q);
    int* ro = sycl::malloc_device<int>(hro.size(), q);
    T* bm = sycl::malloc_device<T>(hb.size(), q);
    T* cm = sycl::malloc_device<T>(hc.size(), q);
    q.copy(hval.data(), val, hval.size()).wait();
    q.copy(hcol.data(), col, hcol.size()).wait();
    q.copy(hro.data(), ro, hro.size()).wait();
    q.copy(hb.data(), bm, hb.size()).wait();
    q.copy(hc.data(), cm, hc.size()).wait();
    const T alpha = T(1.5), beta = T(-0.5);

    int nc = 0;
    //! [launch]
    using R = device::routine::spmm<Transpose::NoTrans, Transpose::NoTrans>;
    device::with_tier<R, T>(q.get_device(), n, [&]<class Tier>(Tier) {
        nc = Tier::nb;
        using L = device::launch<R, Tier, T>;
        static_assert(L::buffer_size(1) == device::WorkspaceSize{}, "spmm needs no workspace");
        q.submit([&](sycl::handler& h) {
             h.parallel_for(L::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
                 const auto ex = L::exec(it);
                 const auto s = L::slot(it, batch);
                 //! [call]
                 const device::CsrItem<const T> A{val, col, ro, std::int64_t(s.b) * std::int64_t(nz_stride),
                                                  std::int64_t(s.b) * ro_stride, m, k};
                 (void)device::spmm<Transpose::NoTrans, Transpose::NoTrans, Tier>(
                     ex, alpha, A, device::mat(bm + s.b * sb, k, n, ldb), beta, device::mat(cm + s.b * sc, m, n, ldc));
                 //! [call]
             });
         }).wait_and_throw();
    });
    //! [launch]

    std::vector<T> got(hc.size());
    q.copy(cm, got.data(), got.size()).wait();
    for (void* p : std::initializer_list<void*>{val, col, ro, bm, cm}) sycl::free(p, q);
    //! [check]
    double worst = 0;
    const double scale = 1.5 * row_max + 0.5;  // |alpha| nnz_row + |beta|: max|A|, max|B|, max|C0| <= 1
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double ref = double(beta) * double(hc[b * sc + j * ldc + i]);
                for (int p = offs[b][i]; p < offs[b][i + 1]; ++p)
                    ref += double(alpha) * double(hval[b * nz_stride + p]) * double(hb[b * sb + j * ldb + cols[b][p]]);
                worst = std::fmax(worst, std::fabs(double(got[b * sc + j * ldc + i]) - ref) / scale);
            }
    std::printf("spmm_csr: %-6s m=%-4d k=%-4d n=%-3d batch %-4d tier Blocked<%d> error %.2e\n", type, m, k, n, batch, nc, worst);
    return worst < 16 * double(std::numeric_limits<T>::epsilon()) ? 0 : 1;
    //! [check]
}

template <class T> int run_all(sycl::queue& q, const char* type) {
    int bad = 0;
    bad += run<T>(q, type, 5, 7, 3, 128);     // fewer columns than one block
    bad += run<T>(q, type, 100, 80, 37, 128);  // full column blocks and an edge (NC 16 for float, 8 for double)
    return bad;
}

int main() {
    sycl::queue q;
    int bad = run_all<float>(q, "float");
    if (q.get_device().has(sycl::aspect::fp64)) bad += run_all<double>(q, "double");
    else std::printf("spmm_csr: double not run: the device has no fp64\n");
    return bad == 0 ? 0 : 1;
}

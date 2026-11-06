// SPDX-License-Identifier: MIT
// device_api_sparse_tests: the public in-kernel spmm and iluk_apply (include/batchlas/device/sparse.hh)
// called from user-style kernels, against the host routes and a long-double scalar reference.
// spmm with A NoTrans and iluk_apply are thin launchers of the same bodies: BIT-IDENTICAL to the
// host (every element of the output buffer, pads included) at work-group sizes 32, 128 and 1024.
// spmm with A (Conj)Trans scatters by atomics in an unspecified order: tolerance vs host and reference.
// Guards (agent-guide section 8): host spmm pinned to `direct` and its `reached` row read back;
// batch 1024 of three representatives, every item bit-identical to its representative; per-item
// CSR strides and row-offset pads with in-range poison (value 1e3 at column 0), ld and stride pads
// with large finite poison that must stay untouched; rows != cols, nrhs = 19 straddling every column
// block; complex data with Im != 0 and ConjTrans on both operands; beta == 0 over NaN C and
// alpha == 0 over NaN A; ILU n = 150 > the work-group (one level split over items, several levels);
// WG 1024 is the launch at the ceiling; with_tier, upto and buffer_size per routine.
// iluk_factorize runs the host route's row elimination (iluk_detail::eliminate_row): its factor in
// the plan's symbolic pattern is BIT-IDENTICAL to the host route's (device path, batch 1024) slot
// by slot, +0 where every item dropped; drop tolerance and fill quota straddled (both kept and
// dropped fill), padded item strides with poison, info = the first unusable pivot row (the host
// throws), uniform over the work-group; vs the host CPU path (another implementation) by
// tolerance; factorize + apply in one user kernel vs a long-double solve with the host factor.
// Deliberate breaks, run in-process so the comparators are seen to fail: spmm with alpha negated,
// iluk_apply with the L solve skipped (l_levels = 0), iluk_factorize with no levels eliminated.
#include <tests/device_core_common.hh>

#include <batchlas/blas/functions/iluk.hh>
#include <batchlas/blas/functions/spmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device/sparse.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/ops/iluk/iluk_plan.hh>
#include <src/ops/spmm/choice.hh>
#include <src/select/select.hh>
#include <tests/test_support/coverage_rows.hh>

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using namespace device_core_test;
using batchlas::verify::Check;

namespace {

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;
constexpr Transpose N = Transpose::NoTrans, Tr = Transpose::Trans, Cj = Transpose::ConjTrans;
constexpr int kTerms = 10;  // products per C element: at most 4 per stored row (NoTrans), about 7 per column (Trans), + beta

template <class T> T poison_v() { return T(real_of<T>(1e3)); }
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> LD<T> cj_if(LD<T> v, bool on) {
    if constexpr (is_cplx<T>) return on ? std::conj(v) : v;
    else return v;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(wide(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}
template <class T> int bit_mismatches(const std::vector<T>& a, const std::vector<T>& b) {
    int bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) bad += !same_bits(a[i], b[i]);
    return bad;
}

// groups work-groups of WG items; the ops under test use no sub-group collectives.
template <int WG, class F> void launch(sycl::queue& q, std::size_t groups, F f) {
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) { f(it); }); })
        .wait_and_throw();
}
bool fits_wg(sycl::queue& q, int wg) { return q.get_device().get_info<sycl::info::device::max_work_group_size>() >= std::size_t(wg); }

// ---- spmm problem: stored A ar x ac, op(A) rows_out x kk, op(B) kk x nrhs, C rows_out x nrhs --
template <class T> struct Spmm {
    int ar = 37, ac = 29, nrhs = 19;
    int kk = 0, rows_out = 0, brows = 0, bcols = 0, ldb = 0, strb = 0, ldc = 0, strc = 0, ms = 0, os = 0, max_nnz = 0;
    Transpose ta = N, tb = N;
    std::vector<T> a_val, b, c0;
    std::vector<int> a_ro, a_ci;
    int row_nnz(int r, int q) const { return (r * 7 + q) % 5; }  // 0..4: empty rows included
    int col_of(int r, int t, int q) const { return (r * 3 + t * 5 + q) % ac; }
    T B(int q, int r, int c) const { return b[std::size_t(q) * strb + std::size_t(c) * ldb + r]; }
    LD<T> opB(int q, int j, int c) const {
        if (tb == N) return wide(B(q, j, c));
        return cj_if<T>(wide(B(q, c, j)), tb == Cj);
    }
};

template <class T> Spmm<T> make_spmm(Transpose ta, Transpose tb, bool nan_a, bool nan_c) {
    Spmm<T> d;
    d.ta = ta;
    d.tb = tb;
    d.kk = ta == N ? d.ac : d.ar;
    d.rows_out = ta == N ? d.ar : d.ac;
    d.brows = tb == N ? d.kk : d.nrhs;
    d.bcols = tb == N ? d.nrhs : d.kk;
    for (int q = 0; q < kPeriod; ++q) {
        int nnz = 0;
        for (int r = 0; r < d.ar; ++r) nnz += d.row_nnz(r, q);
        d.max_nnz = std::max(d.max_nnz, nnz);
    }
    d.ms = d.max_nnz + 4;
    d.os = d.ar + 4;
    d.ldb = d.brows + 3;
    d.strb = d.ldb * d.bcols + 7;
    d.ldc = d.rows_out + 2;
    d.strc = d.ldc * d.nrhs + 5;
    d.a_val.assign(std::size_t(d.ms) * kBatch, poison_v<T>());  // in-range poison past each item's nnz
    d.a_ci.assign(std::size_t(d.ms) * kBatch, 0);
    d.a_ro.assign(std::size_t(d.os) * kBatch, 0);
    d.b.assign(std::size_t(d.strb) * kBatch, poison_v<T>());
    d.c0.assign(std::size_t(d.strc) * kBatch, poison_v<T>());
    for (int bi = 0; bi < kBatch; ++bi) {
        const int q = bi % kPeriod;
        int p = 0;
        for (int r = 0; r < d.ar; ++r) {
            d.a_ro[std::size_t(bi) * d.os + r] = p;
            for (int t = 0; t < d.row_nnz(r, q); ++t, ++p) {
                d.a_ci[std::size_t(bi) * d.ms + p] = d.col_of(r, t, q);
                d.a_val[std::size_t(bi) * d.ms + p] = nan_a ? nan_v<T>() : val<T>(1000003u * q + 31u * r + t);
            }
        }
        d.a_ro[std::size_t(bi) * d.os + d.ar] = p;
        for (int c = 0; c < d.bcols; ++c)
            for (int r = 0; r < d.brows; ++r) d.b[std::size_t(bi) * d.strb + std::size_t(c) * d.ldb + r] = val<T>(7000003u * (q + 1) + 977u * c + r);
        for (int c = 0; c < d.nrhs; ++c)
            for (int r = 0; r < d.rows_out; ++r)
                d.c0[std::size_t(bi) * d.strc + std::size_t(c) * d.ldc + r] = nan_c ? nan_v<T>() : val<T>(9000011u * (q + 1) + 613u * c + r);
    }
    return d;
}

// The worst relative error of C over the representatives against the long-double reference (and,
// on the same scale, against `other` when given); counts pad elements that are no longer poison.
template <class T>
double spmm_worst(const Spmm<T>& d, const std::vector<T>& got, T alpha, T beta, int* pads_moved, const std::vector<T>* other = nullptr,
                  double* vs_other = nullptr) {
    const bool az = dv::Arith<T>::is_zero(alpha), bz = dv::Arith<T>::is_zero(beta);
    double worst = 0;
    for (int q = 0; q < kPeriod; ++q) {
        std::vector<LD<T>> ref(std::size_t(d.rows_out) * d.nrhs, LD<T>(0));
        std::vector<long double> sc(ref.size(), 0.0L);
        for (int i = 0; i < d.ar && !az; ++i)
            for (int p = d.a_ro[std::size_t(q) * d.os + i]; p < d.a_ro[std::size_t(q) * d.os + i + 1]; ++p) {
                const int j = d.a_ci[std::size_t(q) * d.ms + p];
                const LD<T> a = cj_if<T>(wide(d.a_val[std::size_t(q) * d.ms + p]), d.ta == Cj);
                const int r = d.ta == N ? i : j, k = d.ta == N ? j : i;
                for (int c = 0; c < d.nrhs; ++c) {
                    ref[std::size_t(c) * d.rows_out + r] += a * d.opB(q, k, c);
                    sc[std::size_t(c) * d.rows_out + r] += std::abs(a) * std::abs(d.opB(q, k, c));
                }
            }
        for (int c = 0; c < d.nrhs; ++c)
            for (int r = 0; r < d.rows_out; ++r) {
                const std::size_t e = std::size_t(c) * d.rows_out + r, at = std::size_t(q) * d.strc + std::size_t(c) * d.ldc + r;
                const LD<T> cold = bz ? LD<T>(0) : wide(d.c0[at]);
                const LD<T> want = wide(alpha) * ref[e] + wide(beta) * cold;
                const long double s = std::abs(wide(alpha)) * sc[e] + std::abs(wide(beta)) * std::abs(cold);
                worst = std::max(worst, rel(got[at], want, s));
                if (other) *vs_other = std::max(*vs_other, rel(got[at], wide((*other)[at]), s));
            }
    }
    int moved = 0;
    for (int bi = 0; bi < kBatch; ++bi)
        for (int e = 0; e < d.strc; ++e) {
            const int c = e / d.ldc, r = e % d.ldc;
            const bool live = c < d.nrhs && r < d.rows_out;
            if (!live) moved += !same_bits(got[std::size_t(bi) * d.strc + e], poison_v<T>());
        }
    *pads_moved = moved;
    return worst;
}

template <Transpose TA, Transpose TB, int WG, class Tier = dv::tier::upto<1>, class T>
void device_spmm(sycl::queue& q, const Spmm<T>& d, KernelMatrixView<T, MatrixFormat::CSR> akv, const T* bp, T* cp, T alpha, T beta) {
    const int brows = d.brows, bcols = d.bcols, ldb = d.ldb, strb = d.strb, crows = d.rows_out, nrhs = d.nrhs, ldc = d.ldc,
              strc = d.strc;
    launch<WG>(q, kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group(0));
        const dv::WorkGroup<WG> ex{it};
        const dv::ItemMat<const T> B{bp, b, strb, ldb, brows, bcols};
        const dv::ItemMat<T> C{cp, b, strc, ldc, crows, nrhs};
        (void)dv::spmm<TA, TB, Tier>(ex, dv::csr_item(akv, b), B, C, alpha, beta);
    });
}

template <class T> class DeviceApiSparse : public DeviceCoreTest<T> {
  protected:
    struct Run {
        std::vector<T> host, device;
        double worst_device = 0, worst_host = 0, device_vs_host = 0;
        int pads_moved = 0, item_vs_rep = 0, host_route_rows = 0;
    };

    // The host spmm pinned to `direct` (its reached row counted) and the device spmm, on copies of
    // one C; alpha_dev lets a break call the device op with other arguments than the host.
    template <Transpose TA, Transpose TB, int WG, class Tier = dv::tier::upto<1>>
    Run run_spmm(T alpha, T beta, bool nan_a, bool nan_c, T alpha_dev) {
        Dev& dev = this->dev;
        const Spmm<T> d = make_spmm<T>(TA, TB, nan_a, nan_c);
        T* av = dev.up(d.a_val);
        int* aro = dev.up(d.a_ro);
        int* aci = dev.up(d.a_ci);
        T* bp = dev.up(d.b);
        T* ch = dev.up(d.c0);
        T* cd = dev.up(d.c0);
        MatrixView<T, MatrixFormat::CSR> A(av, aro, aci, d.ar, d.ac, NonZeros{d.max_nnz}, d.ms, d.os, kBatch);
        MatrixView<T, MatrixFormat::Dense> Bv(bp, d.brows, d.bcols, d.ldb, d.strb, kBatch);
        MatrixView<T, MatrixFormat::Dense> Cv(ch, d.rows_out, d.nrhs, d.ldc, d.strc, kBatch);
        Run out;
        {
            const select::ScopedPin<ops::spmm::SpmmChoice> pin("spmm", ops::spmm::SpmmChoice{ops::spmm::Direct{}});
            for (const auto& row : test_support::reached_rows([&] {
                     (void)spmm(*dev.ctx, A, Bv, Cv, alpha, beta, TA, TB, Span<std::byte>());
                     dev.ctx->wait();
                 }))
                out.host_route_rows += row.op == "spmm" && row.spelling == "direct";
        }
        device_spmm<TA, TB, WG, Tier>(dev.q(), d, A.kernel_view(), bp, cd, alpha_dev, beta);
        out.host = dev.down(ch, d.c0.size());
        out.device = dev.down(cd, d.c0.size());
        int unused = 0;
        out.worst_host = spmm_worst(d, out.host, alpha, beta, &unused);
        out.worst_device = spmm_worst(d, out.device, alpha, beta, &out.pads_moved, &out.host, &out.device_vs_host);
        for (int bi = kPeriod; bi < kBatch; ++bi)
            for (int e = 0; e < d.strc; ++e)
                out.item_vs_rep += !same_bits(out.device[std::size_t(bi) * d.strc + e], out.device[std::size_t(bi % kPeriod) * d.strc + e]);
        return out;
    }

    template <Transpose TA, Transpose TB, int WG = 128, class Tier = dv::tier::upto<1>>
    void expect_thin_launcher(T alpha, T beta, bool nan_a, bool nan_c, const char* what) {
        const Run r = run_spmm<TA, TB, WG, Tier>(alpha, beta, nan_a, nan_c, alpha);
        EXPECT_GT(r.host_route_rows, 0) << what << ": the host spmm did not reach `direct`";
        EXPECT_EQ(bit_mismatches(r.device, r.host), 0) << what << ": device spmm != host route";
        EXPECT_EQ(r.item_vs_rep, 0) << what << ": items differ from their representative";
        EXPECT_EQ(r.pads_moved, 0) << what << ": wrote outside C(0:m, 0:n)";
        EXPECT_VERIFY(T, Check::blas, kTerms, r.worst_device) << what;
    }

    template <Transpose TA, Transpose TB> void expect_scatter(T alpha, T beta, bool nan_c, const char* what) {
        const Run r = run_spmm<TA, TB, 128>(alpha, beta, false, nan_c, alpha);
        EXPECT_GT(r.host_route_rows, 0) << what << ": the host spmm did not reach `direct`";
        EXPECT_EQ(r.pads_moved, 0) << what << ": wrote outside C(0:m, 0:n)";
        EXPECT_VERIFY(T, Check::blas, kTerms, r.worst_device) << what << " (device vs reference)";
        EXPECT_VERIFY(T, Check::blas, kTerms, r.worst_host) << what << " (host vs reference)";
        // Two results each within one bound of the reference: twice the terms.
        EXPECT_VERIFY(T, Check::blas, 2 * kTerms, r.device_vs_host) << what << " (device vs host)";
    }
};
TYPED_TEST_SUITE(DeviceApiSparse, AllTypes);

template <class T> T cval(double re, double im) {
    if constexpr (is_cplx<T>) return T(real_of<T>(re), real_of<T>(im));
    else return T(real_of<T>(re));
}

TYPED_TEST(DeviceApiSparse, SpmmNoTransIsTheHostGatherBitForBit) {
    using T = TypeParam;
    const T alpha = cval<T>(0.7, -0.3), beta = cval<T>(-0.4, 0.2);
    this->template expect_thin_launcher<N, N>(alpha, beta, false, false, "NN");
    this->template expect_thin_launcher<N, Tr>(alpha, beta, false, false, "NT");
    this->template expect_thin_launcher<N, Cj>(alpha, beta, false, false, "NC");
    this->template expect_thin_launcher<N, N>(alpha, T(0), false, true, "NN beta 0 over NaN C");
    this->template expect_thin_launcher<N, Cj>(T(0), beta, true, false, "NC alpha 0 over NaN A");
    // The column block does not reorder a sum: Blocked<2> (the host's nrhs <= 2 block) is the same bits.
    this->template expect_thin_launcher<N, N, 128, dv::tier::Blocked<2>>(alpha, beta, false, false, "NN Blocked<2>");
}

TYPED_TEST(DeviceApiSparse, SpmmGatherIsIndependentOfTheWorkGroup) {
    using T = TypeParam;
    const T alpha = cval<T>(1.1, 0.5), beta = cval<T>(0.3, -0.6);
    this->template expect_thin_launcher<N, Tr, 32>(alpha, beta, false, false, "NT WG 32");
    if (!fits_wg(this->dev.q(), 1024)) GTEST_SKIP() << "no 1024-item work-group";
    this->template expect_thin_launcher<N, Tr, 1024>(alpha, beta, false, false, "NT WG 1024 (the ceiling)");
}

TYPED_TEST(DeviceApiSparse, SpmmTransScatterMatchesHostAndReference) {
    using T = TypeParam;
    const T alpha = cval<T>(-0.8, 0.45), beta = cval<T>(0.55, 0.25);
    this->template expect_scatter<Tr, N>(alpha, beta, false, "TN");
    this->template expect_scatter<Cj, Cj>(alpha, beta, false, "CC");
    this->template expect_scatter<Cj, Tr>(alpha, T(0), true, "CT beta 0 over NaN C");
}

TYPED_TEST(DeviceApiSparse, SpmmBreakIsSeenByBothComparators) {
    using T = TypeParam;
    const T alpha = cval<T>(0.7, -0.3), beta = cval<T>(-0.4, 0.2);
    const auto r = this->template run_spmm<N, N, 128>(alpha, beta, false, false, -alpha);
    EXPECT_GT(bit_mismatches(r.device, r.host), 0) << "a negated alpha passed the bit comparison";
    EXPECT_FALSE(batchlas::verify::within<T>(Check::blas, kTerms, r.worst_device)) << "a negated alpha passed the reference";
    EXPECT_VERIFY(T, Check::blas, kTerms, r.worst_host);
}

// ---- iluk_apply -----------------------------------------------------------------------------
// A batch of n x n diagonally dominant CSR matrices sharing one sorted pattern, values per
// representative: rows < 100 couple only upward (one L level of 100 rows: split over the items of
// a 32-item work-group), rows >= 100 couple to rows i - 100, i - 50 and i - 1 (a chain of levels).
template <class T> struct Iluk {
    static constexpr int n = 150, nrhs = 2;
    int nnz = 0, ld = n + 3, stride = (n + 3) * nrhs + 5;
    UnifiedVector<T> val, rhs, out_host, out_dev;
    UnifiedVector<int> ro, ci;
    std::optional<ILUKPreconditioner<T>> M;
};

template <class T> void make_iluk(Iluk<T>& d, Queue& ctx) {
    constexpr int n = Iluk<T>::n;
    std::vector<int> ro1{0}, ci1;
    for (int i = 0; i < n; ++i) {
        if (i < 100) {
            ci1.push_back(i);
            ci1.push_back(100 + i % 50);
        } else {
            for (int j : {i - 100, i - 50, i - 1, i})
                if (j != i - 1 || i > 100) ci1.push_back(j);
        }
        ro1.push_back(static_cast<int>(ci1.size()));
    }
    d.nnz = static_cast<int>(ci1.size());
    d.val = UnifiedVector<T>(std::size_t(d.nnz) * kBatch);
    d.ci = UnifiedVector<int>(std::size_t(d.nnz) * kBatch);
    d.ro = UnifiedVector<int>(std::size_t(n + 1) * kBatch);
    for (int b = 0; b < kBatch; ++b) {
        const int q = b % kPeriod;
        for (int i = 0; i <= n; ++i) d.ro[std::size_t(b) * (n + 1) + i] = ro1[i];
        for (int i = 0; i < n; ++i)
            for (int p = ro1[i]; p < ro1[i + 1]; ++p) {
                d.ci[std::size_t(b) * d.nnz + p] = ci1[p];
                const T v = val<T>(500009u * (q + 1) + 7u * p);
                d.val[std::size_t(b) * d.nnz + p] = ci1[p] == i ? v + T(real_of<T>(6)) : v;
            }
    }
    MatrixView<T, MatrixFormat::CSR> A(d.val.data(), d.ro.data(), d.ci.data(), n, n, NonZeros{d.nnz}, d.nnz, n + 1, kBatch);
    ILUKParams<T> params;
    params.levels_of_fill = 1;
    d.M.emplace(iluk_factorize(ctx, A, params));
    const std::size_t len = std::size_t(d.stride) * kBatch;
    d.rhs = UnifiedVector<T>(len);
    d.out_host = UnifiedVector<T>(len);
    d.out_dev = UnifiedVector<T>(len);
    for (std::size_t e = 0; e < len; ++e) d.rhs[e] = d.out_host[e] = d.out_dev[e] = poison_v<T>();
    for (int b = 0; b < kBatch; ++b)
        for (int c = 0; c < Iluk<T>::nrhs; ++c)
            for (int i = 0; i < n; ++i) d.rhs[std::size_t(b) * d.stride + std::size_t(c) * d.ld + i] = val<T>(300007u * (b % kPeriod) + 211u * c + i);
}

// The device iluk_apply over every (item, column) system; skip_l breaks it (no L solve).
template <int WG, class T> void device_iluk(sycl::queue& q, Iluk<T>& d, bool skip_l) {
    const auto batch = dv::iluk_batch(d.M->view());
    const T* rp = d.rhs.data();
    T* yp = d.out_dev.data();
    const int nrhs = Iluk<T>::nrhs, ld = d.ld, stride = d.stride;
    launch<WG>(q, std::size_t(kBatch) * nrhs, [=](sycl::nd_item<1> it) {
        const int g = static_cast<int>(it.get_group(0));
        const int b = g / nrhs, col = g % nrhs;
        const dv::WorkGroup<WG> ex{it};
        auto F = batch.item(b);
        if (skip_l) F.l_levels = 0;
        const dv::BasedVec<const T> r{rp, b * stride + col * ld};
        const dv::BasedVec<T> y{yp, b * stride + col * ld};
        (void)dv::iluk_apply<dv::tier::Blocked<1>>(ex, F, r, y);
    });
}

// Normwise forward error per column against L U x = rhs solved in long double from the factor.
template <class T> double iluk_worst(const Iluk<T>& d, const UnifiedVector<T>& got) {
    constexpr int n = Iluk<T>::n;
    const auto lu = d.M->lu.view();
    const T* v = lu.data_ptr();
    const auto ro = lu.row_offsets();
    const auto ci = lu.col_indices();
    double worst = 0;
    for (int q = 0; q < kPeriod; ++q)
        for (int c = 0; c < Iluk<T>::nrhs; ++c) {
            std::vector<LD<T>> x(n);
            const std::size_t vb = std::size_t(q) * lu.matrix_stride(), rb = std::size_t(q) * lu.offset_stride();
            for (int i = 0; i < n; ++i) {
                LD<T> s = wide(d.rhs[std::size_t(q) * d.stride + std::size_t(c) * d.ld + i]);
                for (int p = ro[rb + i]; p < ro[rb + i + 1]; ++p)
                    if (ci[vb + p] < i) s -= wide(v[vb + p]) * x[ci[vb + p]];
                x[i] = s;
            }
            for (int i = n - 1; i >= 0; --i) {
                LD<T> s = x[i], dg = LD<T>(1);
                for (int p = ro[rb + i]; p < ro[rb + i + 1]; ++p) {
                    if (ci[vb + p] > i) s -= wide(v[vb + p]) * x[ci[vb + p]];
                    if (ci[vb + p] == i) dg = wide(v[vb + p]);
                }
                x[i] = s / dg;
            }
            long double num = 0, den = 0;
            for (int i = 0; i < n; ++i) {
                num = std::max(num, std::abs(wide(got[std::size_t(q) * d.stride + std::size_t(c) * d.ld + i]) - x[i]));
                den = std::max(den, std::abs(x[i]));
            }
            worst = std::max(worst, std::isnan(num) ? std::numeric_limits<double>::infinity() : static_cast<double>(num / den));
        }
    return worst;
}

template <class T> int iluk_bits(const UnifiedVector<T>& a, const UnifiedVector<T>& b) {
    int bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) bad += !same_bits(a[i], b[i]);
    return bad;
}

TYPED_TEST(DeviceApiSparse, IlukApplyIsTheHostSolveBitForBit) {
    using T = TypeParam;
    Iluk<T> d;
    make_iluk(d, *this->dev.ctx);
    ASSERT_GT(d.M->l_levels, 2);
    ASSERT_GE(d.M->u_levels, 2);
    int widest = 0;
    for (int lv = 0; lv < d.M->l_levels; ++lv) widest = std::max(widest, d.M->l_level_ptr[lv + 1] - d.M->l_level_ptr[lv]);
    ASSERT_GT(widest, 32) << "no L level wider than the 32-item work-group";
    MatrixView<T, MatrixFormat::Dense> rv(d.rhs.data(), Iluk<T>::n, Iluk<T>::nrhs, d.ld, d.stride, kBatch);
    MatrixView<T, MatrixFormat::Dense> ov(d.out_host.data(), Iluk<T>::n, Iluk<T>::nrhs, d.ld, d.stride, kBatch);
    (void)iluk_apply(*this->dev.ctx, *d.M, rv, ov);
    this->dev.ctx->wait();
    EXPECT_VERIFY(T, Check::solve, Iluk<T>::n, iluk_worst(d, d.out_host)) << "host route vs reference";
    auto check = [&](const char* what) {
        EXPECT_EQ(iluk_bits(d.out_dev, d.out_host), 0) << what << ": device iluk_apply != host route (pads included)";
        int rep = 0;
        for (int b = kPeriod; b < kBatch; ++b)
            for (int e = 0; e < d.stride; ++e)
                rep += !same_bits(d.out_dev[std::size_t(b) * d.stride + e], d.out_dev[std::size_t(b % kPeriod) * d.stride + e]);
        EXPECT_EQ(rep, 0) << what << ": items differ from their representative";
        EXPECT_VERIFY(T, Check::solve, Iluk<T>::n, iluk_worst(d, d.out_dev)) << what;
        for (std::size_t e = 0; e < d.out_dev.size(); ++e) d.out_dev[e] = poison_v<T>();
    };
    device_iluk<128>(this->dev.q(), d, false);
    check("WG 128 (the host's)");
    device_iluk<32>(this->dev.q(), d, false);
    check("WG 32 (n > the work-group: every level split over items)");
    if (!fits_wg(this->dev.q(), 1024)) GTEST_SKIP() << "no 1024-item work-group";
    device_iluk<1024>(this->dev.q(), d, false);
    check("WG 1024 (the ceiling)");
}

TYPED_TEST(DeviceApiSparse, IlukApplyBreakIsSeenByBothComparators) {
    using T = TypeParam;
    Iluk<T> d;
    make_iluk(d, *this->dev.ctx);
    MatrixView<T, MatrixFormat::Dense> rv(d.rhs.data(), Iluk<T>::n, Iluk<T>::nrhs, d.ld, d.stride, kBatch);
    MatrixView<T, MatrixFormat::Dense> ov(d.out_host.data(), Iluk<T>::n, Iluk<T>::nrhs, d.ld, d.stride, kBatch);
    (void)iluk_apply(*this->dev.ctx, *d.M, rv, ov);
    this->dev.ctx->wait();
    device_iluk<128>(this->dev.q(), d, true);
    EXPECT_GT(iluk_bits(d.out_dev, d.out_host), 0) << "skipping the L solve passed the bit comparison";
    EXPECT_FALSE(batchlas::verify::within<T>(Check::solve, Iluk<T>::n, iluk_worst(d, d.out_dev))) << "skipping the L solve passed the reference";
}

// ---- iluk_factorize -------------------------------------------------------------------------
// A batch of n x n diagonally dominant CSR matrices on one sorted pattern with ILU(1) fill: rows
// < 60 couple only upward (one elimination level of 60 rows, split over a 32-item work-group);
// row i >= 60 also couples to i - 7, whose U entries fill (i, i - 4) and (i, i + 4) (about 23
// levels). Item strides padded by `pad`: values poison, indices and offsets in range (0).
template <class T> struct IlukA {
    static constexpr int n = 150;
    int nnz = 0, ms = 0, os = 0;
    UnifiedVector<T> val;
    UnifiedVector<int> ro, ci;
    MatrixView<T, MatrixFormat::CSR> view(int batch) const {
        return MatrixView<T, MatrixFormat::CSR>(val.data(), ro.data(), ci.data(), n, n, NonZeros{nnz}, ms, os, batch);
    }
};

// zero_pivot_row >= 0: that row's diagonal is exactly 0 in the items b % kPeriod == 1.
template <class T> IlukA<T> make_iluk_a(int pad, int zero_pivot_row = -1) {
    constexpr int n = IlukA<T>::n;
    std::vector<int> ro1{0}, ci1;
    for (int i = 0; i < n; ++i) {
        for (int j : {i - 7, i, i + 3, i + 11})
            if (j >= 0 && j < n && (j != i - 7 || i >= 60)) ci1.push_back(j);
        ro1.push_back(static_cast<int>(ci1.size()));
    }
    IlukA<T> d;
    d.nnz = static_cast<int>(ci1.size());
    d.ms = d.nnz + pad;
    d.os = n + 1 + pad;
    d.val = UnifiedVector<T>(std::size_t(d.ms) * kBatch);
    d.ci = UnifiedVector<int>(std::size_t(d.ms) * kBatch);
    d.ro = UnifiedVector<int>(std::size_t(d.os) * kBatch);
    for (int b = 0; b < kBatch; ++b) {
        const int q = b % kPeriod;
        for (int e = 0; e < d.os; ++e) d.ro[std::size_t(b) * d.os + e] = e <= n ? ro1[e] : 0;
        for (int p = d.nnz; p < d.ms; ++p) {
            d.val[std::size_t(b) * d.ms + p] = poison_v<T>();
            d.ci[std::size_t(b) * d.ms + p] = 0;
        }
        for (int i = 0; i < n; ++i)
            for (int p = ro1[i]; p < ro1[i + 1]; ++p) {
                const std::size_t at = std::size_t(b) * d.ms + p;
                const T v = val<T>(700001u * (q + 1) + 13u * p);
                d.ci[at] = ci1[p];
                d.val[at] = ci1[p] != i ? v : (i == zero_pivot_row && q == 1 ? T(0) : v + T(real_of<T>(6)));
            }
    }
    return d;
}

// BATCHLAS_ILUK_DEVICE pins the host route's numeric path: '0' the CPU path, '1' the device kernels.
char iluk_pin() {
    const char* v = std::getenv("BATCHLAS_ILUK_DEVICE");
    return v != nullptr ? v[0] : '\0';
}

// device iluk_factorize of every item in a user-style kernel: M at item stride mst, the value
// every work-item returned at info[b * WG + lid].
template <int WG, class T>
void device_iluk_factorize(sycl::queue& q, const IlukA<T>& d, const dv::IlukPlan& plan, const dv::IlukOptions<T>& opts, T* mv, int mst,
                           int* info) {
    const T* av = d.val.data();
    const int ms = d.ms;
    launch<WG>(q, kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group(0));
        const dv::WorkGroup<WG> ex{it};
        const int r = dv::iluk_factorize<dv::tier::Blocked<1>>(ex, dv::BasedVec<const T>{av, b * ms}, plan, dv::BasedVec<T>{mv, b * mst}, opts);
        info[std::size_t(b) * WG + ex.lid()] = r;
    });
}

// The device factor (plan's slots, item stride mst) against a host factor M (its own pattern)
// over items [0, batch): `want` is M's value where M stores the slot, else +0.
struct FactorCmp {
    int bits = 0, unmatched = 0, pads = 0;
    double worst = 0;  // max over items of max |dev - want| / max |want|
};
template <class T> FactorCmp compare_factor(const dv::IlukPlan& plan, const T* dev, int mst, const ILUKPreconditioner<T>& M, int batch) {
    FactorCmp r;
    const auto lu = M.lu.view();
    const T* hv = lu.data_ptr();
    const auto hro = lu.row_offsets();
    const auto hci = lu.col_indices();
    for (int b = 0; b < batch; ++b) {
        const std::size_t vb = std::size_t(b) * lu.matrix_stride(), rb = std::size_t(b) * lu.offset_stride(), db = std::size_t(b) * mst;
        long double num = 0, den = 0;
        for (int i = 0; i < plan.n; ++i) {
            int hp = hro[rb + i];
            const int he = hro[rb + i + 1];
            int matched = 0;
            for (int s = plan.ro[i]; s < plan.ro[i + 1]; ++s) {
                while (hp < he && hci[vb + hp] < plan.col[s]) ++hp;
                const bool stored = hp < he && hci[vb + hp] == plan.col[s];
                const T want = stored ? hv[vb + hp] : T(0);
                matched += stored;
                r.bits += !same_bits(dev[db + s], want);
                num = std::max<long double>(num, std::abs(wide(dev[db + s]) - wide(want)));
                den = std::max<long double>(den, std::abs(wide(want)));
            }
            r.unmatched += (he - hro[rb + i]) - matched;
        }
        for (int e = plan.nnz; e < mst; ++e) r.pads += !same_bits(dev[db + e], poison_v<T>());
        r.worst = std::max(r.worst, std::isnan(num) ? std::numeric_limits<double>::infinity() : static_cast<double>(num / den));
    }
    return r;
}

// Over the representatives: {fill slots kept, fill slots dropped} (fill = no entry of A lands there).
template <class T> std::array<int, 2> fill_split(const dv::IlukPlan& plan, const T* dev, int mst) {
    std::vector<char> from_a(std::size_t(plan.nnz), 0);
    for (int p = 0; p < plan.a_nnz; ++p)
        if (plan.a_to_slot[p] >= 0) from_a[std::size_t(plan.a_to_slot[p])] = 1;
    std::array<int, 2> out{0, 0};
    for (int q = 0; q < kPeriod; ++q)
        for (int s = 0; s < plan.nnz; ++s)
            if (!from_a[std::size_t(s)]) ++out[same_bits(dev[std::size_t(q) * mst + s], T(0)) ? 1 : 0];
    return out;
}

template <class T> void expect_iluk_factorize_thin(Dev& dev, const char* what, const ILUKParams<T>& params) {
    const IlukA<T> d = make_iluk_a<T>(/*pad=*/5);
    const auto A = d.view(kBatch);
    const auto host = iluk_factorize(*dev.ctx, A, params);
    const auto storage = sycl_iluk::iluk_plan(A, params);
    const dv::IlukPlan plan = storage.plan();
    ASSERT_GT(plan.nnz, d.nnz) << what << ": the symbolic pattern has no fill";
    ASSERT_GT(plan.levels, 2) << what;
    const int mst = plan.nnz + 3;
    UnifiedVector<T> m(std::size_t(mst) * kBatch);
    UnifiedVector<int> info(std::size_t(kBatch) * 1024);
    const auto opts = dv::iluk_options(params);
    auto check = [&](int wg, const char* how) {
        const FactorCmp c = compare_factor(plan, m.data(), mst, host, kBatch);
        EXPECT_EQ(c.bits, 0) << what << ", " << how << ": device iluk_factorize != host route";
        EXPECT_EQ(c.unmatched, 0) << what << ", " << how << ": a host entry outside the symbolic pattern";
        EXPECT_EQ(c.pads, 0) << what << ", " << how << ": wrote past plan.nnz";
        int nonzero_info = 0;
        for (std::size_t e = 0; e < std::size_t(kBatch) * wg; ++e) nonzero_info += info[e] != 0;
        EXPECT_EQ(nonzero_info, 0) << what << ", " << how << ": info on a usable factorization";
        const auto split = fill_split(plan, m.data(), mst);
        EXPECT_GT(split[0], 0) << what << ": no fill survived (threshold not straddled)";
        EXPECT_GT(split[1], 0) << what << ": no fill dropped (threshold not straddled)";
    };
    auto reset = [&] {
        for (std::size_t e = 0; e < m.size(); ++e) m[e] = poison_v<T>();
        for (std::size_t e = 0; e < info.size(); ++e) info[e] = -7;
    };
    reset();
    device_iluk_factorize<128>(dev.q(), d, plan, opts, m.data(), mst, info.data());
    check(128, "WG 128");
    reset();
    device_iluk_factorize<32>(dev.q(), d, plan, opts, m.data(), mst, info.data());
    check(32, "WG 32 (a level wider than the work-group)");
    if (!fits_wg(dev.q(), 1024)) return;
    reset();
    device_iluk_factorize<1024>(dev.q(), d, plan, opts, m.data(), mst, info.data());
    check(1024, "WG 1024 (the ceiling)");
}

TYPED_TEST(DeviceApiSparse, IlukFactorizeIsTheHostFactorBitForBit) {
    using T = TypeParam;
    using R = real_of<T>;
    if (iluk_pin() == '0') GTEST_SKIP() << "BATCHLAS_ILUK_DEVICE=0 pins the host route to its CPU path";
    ILUKParams<T> drop;
    drop.levels_of_fill = 1;
    drop.drop_tolerance = R(0.005);  // ~0.03 against the row scale ~6: fill and l = a / u are ~0.04
    expect_iluk_factorize_thin(this->dev, "drop + MILU", drop);
    ILUKParams<T> quota = drop;
    quota.drop_tolerance = R(0);
    quota.fill_factor = R(1);  // A's off-diagonal count survives: every filled row drops its two smallest
    quota.modified_ilu = false;
    expect_iluk_factorize_thin(this->dev, "fill quota", quota);
}

// x = U^{-1} L^{-1} rhs in long double from item b's host factor (the stored, stabilised diagonal).
template <class T> std::vector<LD<T>> ilu_solve_ref(const ILUKPreconditioner<T>& M, int b, const T* rhs) {
    const auto lu = M.lu.view();
    const T* v = lu.data_ptr();
    const auto ro = lu.row_offsets();
    const auto ci = lu.col_indices();
    const std::size_t vb = std::size_t(b) * lu.matrix_stride(), rb = std::size_t(b) * lu.offset_stride();
    const int n = M.n;
    std::vector<LD<T>> x(std::size_t(n));
    for (int i = 0; i < n; ++i) {
        LD<T> s = wide(rhs[i]);
        for (int p = ro[rb + i]; p < ro[rb + i + 1]; ++p)
            if (ci[vb + p] < i) s -= wide(v[vb + p]) * x[std::size_t(ci[vb + p])];
        x[std::size_t(i)] = s;
    }
    for (int i = n - 1; i >= 0; --i) {
        LD<T> s = x[std::size_t(i)], dg = LD<T>(1);
        for (int p = ro[rb + i]; p < ro[rb + i + 1]; ++p) {
            if (ci[vb + p] > i) s -= wide(v[vb + p]) * x[std::size_t(ci[vb + p])];
            if (ci[vb + p] == i) dg = wide(v[vb + p]);
        }
        x[std::size_t(i)] = s / dg;
    }
    return x;
}

TYPED_TEST(DeviceApiSparse, IlukFactorizeMatchesTheCpuPathAndSolvesInOneKernel) {
    using T = TypeParam;
    constexpr int n = IlukA<T>::n, WG = 128;
    ILUKParams<T> params;
    params.levels_of_fill = 1;
    const IlukA<T> d = make_iluk_a<T>(/*pad=*/0);
    const auto A = d.view(kBatch);
    const auto storage = sycl_iluk::iluk_plan(A, params);
    const dv::IlukPlan plan = storage.plan();
    const auto opts = dv::iluk_options(params);
    const int mst = plan.nnz + 3;
    UnifiedVector<T> m(std::size_t(mst) * kBatch);
    UnifiedVector<int> info(std::size_t(kBatch) * WG);
    if (iluk_pin() != '1') {
        // batch kPeriod < 32 takes the host CPU path: row by row with a sorted quota, another implementation.
        const auto cpu = iluk_factorize(*this->dev.ctx, d.view(kPeriod), params);
        for (std::size_t e = 0; e < m.size(); ++e) m[e] = poison_v<T>();
        device_iluk_factorize<WG>(this->dev.q(), d, plan, opts, m.data(), mst, info.data());
        const FactorCmp c = compare_factor(plan, m.data(), mst, cpu, kPeriod);
        EXPECT_EQ(c.unmatched, 0) << "a CPU-path entry outside the symbolic pattern";
        EXPECT_VERIFY(T, Check::factorization, n, c.worst) << "device factor vs the host CPU path";
    }
    // factorize + apply in ONE user kernel; rhs / out at item stride n + 3 (pads poison).
    const auto host = iluk_factorize(*this->dev.ctx, A, params);
    const int ld = n + 3;
    UnifiedVector<T> rhs(std::size_t(ld) * kBatch), out(std::size_t(ld) * kBatch);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < ld; ++i) {
            rhs[std::size_t(b) * ld + i] = i < n ? val<T>(90001u * (b % kPeriod) + i) : poison_v<T>();
            out[std::size_t(b) * ld + i] = poison_v<T>();
        }
    const T* av = d.val.data();
    const T* rp = rhs.data();
    T* mp = m.data();
    T* yp = out.data();
    int* ip = info.data();
    const int ms = d.ms;
    launch<WG>(this->dev.q(), kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group(0));
        const dv::WorkGroup<WG> ex{it};
        int r = dv::iluk_factorize<dv::tier::Blocked<1>>(ex, dv::BasedVec<const T>{av, b * ms}, plan, dv::BasedVec<T>{mp, b * mst}, opts);
        r |= dv::iluk_apply<dv::tier::Blocked<1>>(ex, plan.factor<T>(mp + std::size_t(b) * mst, opts.diagonal_shift),
                                                  dv::BasedVec<const T>{rp, b * ld}, dv::BasedVec<T>{yp, b * ld});
        if (ex.leader()) ip[b] = r;
    });
    double worst = 0;
    int pads = 0, rep = 0, bad_info = 0;
    for (int q = 0; q < kPeriod; ++q) {
        const auto x = ilu_solve_ref(host, q, rhs.data() + std::size_t(q) * ld);
        long double num = 0, den = 0;
        for (int i = 0; i < n; ++i) {
            num = std::max<long double>(num, std::abs(wide(out[std::size_t(q) * ld + i]) - x[std::size_t(i)]));
            den = std::max<long double>(den, std::abs(x[std::size_t(i)]));
        }
        worst = std::max(worst, std::isnan(num) ? std::numeric_limits<double>::infinity() : static_cast<double>(num / den));
    }
    for (int b = 0; b < kBatch; ++b) {
        bad_info += info[std::size_t(b)] != 0;
        for (int i = 0; i < ld; ++i) {
            const T got = out[std::size_t(b) * ld + i];
            if (i >= n) pads += !same_bits(got, poison_v<T>());
            else rep += !same_bits(got, out[std::size_t(b % kPeriod) * ld + i]);
        }
    }
    EXPECT_VERIFY(T, Check::solve, n, worst) << "in-kernel factorize + apply vs the long-double solve with the host factor";
    EXPECT_EQ(pads, 0) << "wrote past n";
    EXPECT_EQ(rep, 0) << "items differ from their representative";
    EXPECT_EQ(bad_info, 0);
}

TYPED_TEST(DeviceApiSparse, IlukFactorizeInfoIsTheFirstUnusablePivot) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int bad_row = 5;  // < 60: no L entries, its pivot is A's stored zero
    ILUKParams<T> params;
    params.levels_of_fill = 1;
    params.diagonal_shift = T(0);
    params.diag_pivot_threshold = R(0);
    params.drop_tolerance = R(0);
    params.modified_ilu = false;
    const IlukA<T> d = make_iluk_a<T>(/*pad=*/5, bad_row);
    const auto A = d.view(kBatch);
    EXPECT_THROW((void)iluk_factorize(*this->dev.ctx, A, params), batchlas::convergence_error);
    const auto storage = sycl_iluk::iluk_plan(A, params);
    const dv::IlukPlan plan = storage.plan();
    const int mst = plan.nnz;
    UnifiedVector<T> m(std::size_t(mst) * kBatch);
    UnifiedVector<int> info(std::size_t(kBatch) * 128);
    auto expect_info = [&](int wg, const char* how) {
        int wrong = 0;
        for (int b = 0; b < kBatch; ++b)
            for (int l = 0; l < wg; ++l) wrong += info[std::size_t(b) * wg + l] != (b % kPeriod == 1 ? bad_row + 1 : 0);
        EXPECT_EQ(wrong, 0) << how << ": info is not row " << bad_row + 1 << " on the failing items (0 elsewhere) on every item";
    };
    device_iluk_factorize<128>(this->dev.q(), d, plan, dv::iluk_options(params), m.data(), mst, info.data());
    expect_info(128, "WG 128");
    device_iluk_factorize<32>(this->dev.q(), d, plan, dv::iluk_options(params), m.data(), mst, info.data());
    expect_info(32, "WG 32");
}

TYPED_TEST(DeviceApiSparse, IlukFactorizeBreakIsSeenByBothComparators) {
    using T = TypeParam;
    if (iluk_pin() == '0') GTEST_SKIP() << "BATCHLAS_ILUK_DEVICE=0 pins the host route to its CPU path";
    ILUKParams<T> params;
    params.levels_of_fill = 1;
    const IlukA<T> d = make_iluk_a<T>(/*pad=*/5);
    const auto A = d.view(kBatch);
    const auto host = iluk_factorize(*this->dev.ctx, A, params);
    const auto storage = sycl_iluk::iluk_plan(A, params);
    dv::IlukPlan plan = storage.plan();
    plan.levels = 0;  // the break: A scattered into the pattern, nothing eliminated
    const int mst = plan.nnz + 3;
    UnifiedVector<T> m(std::size_t(mst) * kBatch);
    UnifiedVector<int> info(std::size_t(kBatch) * 128);
    for (std::size_t e = 0; e < m.size(); ++e) m[e] = poison_v<T>();
    device_iluk_factorize<128>(this->dev.q(), d, plan, dv::iluk_options(params), m.data(), mst, info.data());
    const FactorCmp c = compare_factor(plan, m.data(), mst, host, kBatch);
    EXPECT_GT(c.bits, 0) << "no elimination passed the bit comparison";
    EXPECT_FALSE(batchlas::verify::within<T>(Check::factorization, IlukA<T>::n, c.worst)) << "no elimination passed the tolerance";
}

// ---- tiers and workspace --------------------------------------------------------------------
TYPED_TEST(DeviceApiSparse, WithTierPicksTheOneTierAndNoWorkspace) {
    using T = TypeParam;
    using S = dv::routine::spmm<N, Cj>;
    using Blk = dv::tier::Blocked<dv::spmm_block_v<T>>;
    static_assert(std::is_same_v<dv::resolve_tier_t<S, T, dv::tier::upto<1>>, Blk>);
    static_assert(std::is_same_v<dv::resolve_tier_t<S, T, dv::tier::upto<100000>>, Blk>);
    static_assert(dv::spmm_block_v<T> * sizeof(T) == 64);
    static_assert(dv::spmm_buffer_size<N, Cj, dv::tier::upto<1>, T>(4096) == dv::WorkspaceSize{});
    static_assert(dv::iluk_apply_buffer_size<dv::tier::Blocked<1>, T>(4096) == dv::WorkspaceSize{});
    static_assert(dv::iluk_factorize_buffer_size<dv::tier::upto<100000>, T>(100000) == dv::WorkspaceSize{});
    static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::iluk_factorize, T, dv::tier::upto<1>>, dv::tier::Blocked<1>>);
    for (int n : {0, 1, 150, 100000}) {
        std::string spmm_tier = "none", iluk_tier = "none", ilu_fac_tier = "none";
        dv::with_tier<S, T>(this->dev.q().get_device(), n, [&]<class Tier>(Tier) { spmm_tier = std::is_same_v<Tier, Blk> ? "blocked" : "other"; });
        dv::with_tier<dv::routine::iluk_apply, T>(this->dev.q().get_device(), n, [&]<class Tier>(Tier) {
            iluk_tier = std::is_same_v<Tier, dv::tier::Blocked<1>> ? "blocked" : "other";
        });
        dv::with_tier<dv::routine::iluk_factorize, T>(this->dev.q().get_device(), n, [&]<class Tier>(Tier) {
            ilu_fac_tier = std::is_same_v<Tier, dv::tier::Blocked<1>> ? "blocked" : "other";
        });
        EXPECT_EQ(spmm_tier, "blocked") << "n=" << n;
        EXPECT_EQ(iluk_tier, "blocked") << "n=" << n;
        EXPECT_EQ(ilu_fac_tier, "blocked") << "n=" << n;
    }
    dv::DeviceLimits none = dv::DeviceLimits::of(this->dev.q().get_device());
    none.sub_group_sizes = {8};
    EXPECT_THROW((dv::with_tier<S, T>(none, 10, [](auto) {})), batchlas::invalid_argument);
    EXPECT_THROW((dv::with_tier<dv::routine::iluk_factorize, T>(none, 10, [](auto) {})), batchlas::invalid_argument);
}

}  // namespace

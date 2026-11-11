// SPDX-License-Identifier: MIT
// device_api_gemv_tests: the public core gemv (include/batchlas/device/level2.hh) called from a
// user-style kernel, one matrix per work-group, against the host gemv pinned to `direct` (bodies 1
// and 2: one item per output, a serial fma dot, the MacChain epilogue). The device gemv with
// alg::RowOwner (NoTrans) or alg::ColDot<1> (Trans, ConjTrans), arith::Fma and epi::MacChain is
// the same per-output expression, so the results are BIT-IDENTICAL (every element of y, pads
// included), and stay so at WG 32 and 1024: no output is shared between items.
// Guards (agent-guide section 8): the `direct` pin's `reached` row read back; m = 37 > 16 so the
// host takes body 1, not the sub-group body 4; rows != cols, ld, stride and inc pads (x inc 2, y
// inc 3) with large finite poison that must stay untouched; complex data with Im != 0, ConjTrans;
// beta == 0 over NaN y; batch 1024 of three representatives, items bit-identical to theirs; a
// long-double reference (EXPECT_VERIFY). Deliberate break, run in-process: the device gemv with
// ConjTrans where the host ran Trans (complex) or with beta negated (real) is seen by both
// comparators. Not covered: alpha == 0 over NaN A (the device gemv reads A and forms 0 * acc).
#include <tests/device_core_common.hh>

#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/ops/gemv/choice.hh>
#include <src/select/select.hh>
#include <tests/test_support/coverage_rows.hh>

#include <cmath>
#include <limits>
#include <vector>

using namespace device_core_test;
using batchlas::verify::Check;

namespace {

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;
constexpr int kM = 37, kN = 23;
constexpr Transpose N = Transpose::NoTrans, Tr = Transpose::Trans, Cj = Transpose::ConjTrans;

template <class T> T poison_v() { return T(real_of<T>(1e3)); }
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> T cval(double re, double im) {
    if constexpr (is_cplx<T>) return T(real_of<T>(re), real_of<T>(im));
    else return T(real_of<T>(re));
}

// Stored A kM x kN; op(A) out x red; x of length red at inc 2, y of length out at inc 3.
template <class T> struct Gemv {
    Transpose tr = N;
    int out = 0, red = 0, lda = kM + 3, sa = (kM + 3) * kN + 5, incx = 2, sx = 0, incy = 3, sy = 0;
    std::vector<T> a, x, y0;
};

template <class T> Gemv<T> make_gemv(Transpose tr, bool nan_y) {
    Gemv<T> d;
    d.tr = tr;
    d.out = tr == N ? kM : kN;
    d.red = tr == N ? kN : kM;
    d.sx = d.red * d.incx + 3;
    d.sy = d.out * d.incy + 4;
    d.a.assign(std::size_t(d.sa) * kBatch, poison_v<T>());
    d.x.assign(std::size_t(d.sx) * kBatch, poison_v<T>());
    d.y0.assign(std::size_t(d.sy) * kBatch, poison_v<T>());
    for (int b = 0; b < kBatch; ++b) {
        const std::uint64_t q = b % kPeriod;
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) d.a[std::size_t(b) * d.sa + std::size_t(j) * d.lda + i] = val<T>(100003u * (q + 1) + 59u * j + i);
        for (int i = 0; i < d.red; ++i) d.x[std::size_t(b) * d.sx + std::size_t(i) * d.incx] = val<T>(200003u * (q + 1) + i);
        for (int i = 0; i < d.out; ++i) d.y0[std::size_t(b) * d.sy + std::size_t(i) * d.incy] = nan_y ? nan_v<T>() : val<T>(300007u * (q + 1) + i);
    }
    return d;
}

template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double e = std::abs(wide(got) - ref);
    if (std::isnan(e)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(e / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// Worst relative error over the representatives; pads (and the elements between y's strides)
// that are no longer poison are counted.
template <class T> double gemv_worst(const Gemv<T>& d, const std::vector<T>& got, T alpha, T beta, int* pads_moved) {
    const bool bz = batchlas::device::Arith<T>::is_zero(beta);
    double worst = 0;
    for (int q = 0; q < kPeriod; ++q)
        for (int o = 0; o < d.out; ++o) {
            LD<T> acc = 0;
            long double s = 0;
            for (int r = 0; r < d.red; ++r) {
                const int i = d.tr == N ? o : r, j = d.tr == N ? r : o;
                LD<T> a = wide(d.a[std::size_t(q) * d.sa + std::size_t(j) * d.lda + i]);
                if constexpr (is_cplx<T>)
                    if (d.tr == Cj) a = std::conj(a);
                const LD<T> xv = wide(d.x[std::size_t(q) * d.sx + std::size_t(r) * d.incx]);
                acc += a * xv;
                s += std::abs(a) * std::abs(xv);
            }
            const std::size_t at = std::size_t(q) * d.sy + std::size_t(o) * d.incy;
            const LD<T> cold = bz ? LD<T>(0) : wide(d.y0[at]);
            worst = std::max(worst, rel(got[at], wide(alpha) * acc + wide(beta) * cold,
                                        std::abs(wide(alpha)) * s + std::abs(wide(beta)) * std::abs(cold)));
        }
    int moved = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int e = 0; e < d.sy; ++e)
            if (e % d.incy != 0 || e / d.incy >= d.out) moved += !same_bits(got[std::size_t(b) * d.sy + e], poison_v<T>());
    *pads_moved = moved;
    return worst;
}

template <Transpose TR, int WG, class T> void device_gemv(sycl::queue& q, const Gemv<T>& d, const T* ap, const T* xp, T* yp, T alpha, T beta) {
    using Alg = std::conditional_t<TR == N, batchlas::device::alg::RowOwner, batchlas::device::alg::ColDot<1>>;
    const int lda = d.lda, sa = d.sa, incx = d.incx, sx = d.sx, incy = d.incy, sy = d.sy, out = d.out, red = d.red;
    const bool az = batchlas::device::Arith<T>::is_zero(alpha), bz = batchlas::device::Arith<T>::is_zero(beta);
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * WG, WG), [=](sycl::nd_item<1> it) {
             const int b = static_cast<int>(it.get_group(0));
             const batchlas::device::WorkGroup<WG> ex{it};
             const batchlas::device::ItemMat<const T> A{ap, b, sa, lda, kM, kN};
             const batchlas::device::ItemVec<const T> x{xp, b, sx, incx, red};
             const batchlas::device::ItemVec<T> y{yp, b, sy, incy, out};
             batchlas::device::gemv<TR, Alg, batchlas::device::arith::Fma, batchlas::device::epi::MacChain>(ex, batchlas::device::MaybeZero<T>{alpha, az}, A, batchlas::device::assume_published(x),
                                                                   batchlas::device::MaybeZero<T>{beta, bz}, y);
         });
     }).wait_and_throw();
}

template <class T> class DeviceApiGemv : public DeviceCoreTest<T> {
  protected:
    struct Run {
        std::vector<T> host, device;
        double worst_host = 0, worst_device = 0;
        int pads_moved = 0, item_vs_rep = 0, mismatches = 0, host_route_rows = 0;
    };
    // Host gemv pinned to `direct` with (TR, alpha, beta); the device gemv with (TRD, alpha, beta_dev).
    template <Transpose TR, Transpose TRD = TR, int WG = 128> Run run(T alpha, T beta, bool nan_y, T beta_dev) {
        Dev& dev = this->dev;
        const Gemv<T> d = make_gemv<T>(TR, nan_y);
        T* ap = dev.up(d.a);
        T* xp = dev.up(d.x);
        T* yh = dev.up(d.y0);
        T* yd = dev.up(d.y0);
        MatrixView<T, MatrixFormat::Dense> A(ap, kM, kN, d.lda, d.sa, kBatch);
        VectorView<T> xv(xp, d.red, kBatch, d.incx, d.sx);
        VectorView<T> yv(yh, d.out, kBatch, d.incy, d.sy);
        Run r;
        {
            const select::ScopedPin<ops::gemv::GemvChoice> pin("gemv", ops::gemv::GemvChoice{ops::gemv::Direct{}});
            for (const auto& row : test_support::reached_rows([&] {
                     (void)gemv(*dev.ctx, A, xv, yv, alpha, beta, TR);
                     dev.ctx->wait();
                 }))
                r.host_route_rows += row.op == "gemv" && row.spelling == "direct";
        }
        device_gemv<TRD, WG>(dev.q(), d, ap, xp, yd, alpha, beta_dev);
        r.host = dev.down(yh, d.y0.size());
        r.device = dev.down(yd, d.y0.size());
        int unused = 0;
        r.worst_host = gemv_worst(d, r.host, alpha, beta, &unused);
        r.worst_device = gemv_worst(d, r.device, alpha, beta, &r.pads_moved);
        for (std::size_t e = 0; e < r.host.size(); ++e) r.mismatches += !same_bits(r.device[e], r.host[e]);
        for (int b = kPeriod; b < kBatch; ++b)
            for (int e = 0; e < d.sy; ++e)
                r.item_vs_rep += !same_bits(r.device[std::size_t(b) * d.sy + e], r.device[std::size_t(b % kPeriod) * d.sy + e]);
        return r;
    }
    template <Transpose TR, int WG = 128> void expect_same(T alpha, T beta, bool nan_y, const char* what) {
        const Run r = run<TR, TR, WG>(alpha, beta, nan_y, beta);
        EXPECT_GT(r.host_route_rows, 0) << what << ": the host gemv did not reach `direct`";
        EXPECT_EQ(r.mismatches, 0) << what << ": device gemv != host route";
        EXPECT_EQ(r.item_vs_rep, 0) << what << ": items differ from their representative";
        EXPECT_EQ(r.pads_moved, 0) << what << ": wrote outside y";
        EXPECT_VERIFY(T, Check::blas, kM, r.worst_device) << what;
        EXPECT_VERIFY(T, Check::blas, kM, r.worst_host) << what << " (host)";
    }
};
TYPED_TEST_SUITE(DeviceApiGemv, AllTypes);

TYPED_TEST(DeviceApiGemv, DeviceGemvIsTheHostDirectRouteBitForBit) {
    using T = TypeParam;
    const T alpha = cval<T>(0.9, -0.35), beta = cval<T>(-0.6, 0.15);
    this->template expect_same<N>(alpha, beta, false, "N");
    this->template expect_same<Tr>(alpha, beta, false, "T");
    this->template expect_same<Cj>(alpha, beta, false, "C");
    this->template expect_same<N>(alpha, T(0), true, "N beta 0 over NaN y");
    this->template expect_same<Cj>(alpha, T(0), true, "C beta 0 over NaN y");
}

TYPED_TEST(DeviceApiGemv, DeviceGemvIsIndependentOfTheWorkGroup) {
    using T = TypeParam;
    const T alpha = cval<T>(-1.2, 0.4), beta = cval<T>(0.25, 0.5);
    this->template expect_same<N, 32>(alpha, beta, false, "N WG 32");
    this->template expect_same<Cj, 32>(alpha, beta, false, "C WG 32");
    const sycl::device device = this->dev.q().get_device();
    if (device.get_info<sycl::info::device::max_work_group_size>() < 1024) GTEST_SKIP() << "no 1024-item work-group";
    this->template expect_same<N, 1024>(alpha, beta, false, "N WG 1024 (the ceiling)");
    this->template expect_same<Tr, 1024>(alpha, beta, false, "T WG 1024 (the ceiling)");
}

TYPED_TEST(DeviceApiGemv, BreakIsSeenByBothComparators) {
    using T = TypeParam;
    const T alpha = cval<T>(0.9, -0.35), beta = cval<T>(-0.6, 0.15);
    const auto r = is_cplx<T> ? this->template run<Tr, Cj>(alpha, beta, false, beta) : this->template run<Tr, Tr>(alpha, beta, false, -beta);
    EXPECT_GT(r.mismatches, 0) << "the break passed the bit comparison";
    EXPECT_FALSE(batchlas::verify::within<T>(Check::blas, kM, r.worst_device)) << "the break passed the reference";
    EXPECT_VERIFY(T, Check::blas, kM, r.worst_host);
}

}  // namespace

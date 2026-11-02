// SPDX-License-Identifier: MIT
// Device LA v2 stedc extension ops (include/batchlas/blas/device/v2/ext/stedc.hh) against host
// references, on every executor: DynWorkGroup at a non-power-of-two width (96) and at the 1024
// ceiling, WorkGroup<128>, SubGroup<32>, Partition<8> (lockstep), SgpPartition<8, Masked>, Thread.
// Guards (docs/developer/agent-guide.md section 8): inc 2 / 3 and ld = m + 3 with finite poison in
// the pad rows and between strided elements, complex data with nonzero imaginary parts, batch
// 1024 at period 3 with every item bit-identical to its representative, nrm2 straddling Blue's
// thresholds and the overflow of the unscaled sum, scaled products leaving the floating range
// (where the naive product must fail) on narrow and wide executors, an all-negative max fold
// (the identity), owner maps checked index by index.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/stedc.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/dla_v2_partition.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kBatch = 1024;  // saturating batch: items repeat with period kPeriod
constexpr int kPeriod = 3;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class T> T val(std::uint64_t i) {
    if constexpr (is_cplx<T>) return T(uval<real_of<T>>(2 * i), uval<real_of<T>>(2 * i + 1));
    else return uval<T>(i);
}

// kPeriod representatives copied to every item b from representative b % kPeriod.
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed, int items = kBatch) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }

// |got - ref| / scale, the blas relative error that EXPECT_VERIFY bounds.
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    template <class T> T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_device<T>(h.size(), q());
        q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    // Run-time local range: the DynWorkGroup launches are not templated on it.
    template <class F> void run(int wg, int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * wg, std::size_t(wg)),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                                f(it, batchlas::util::get_raw_ptr(lm));
                            });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

// Executors under test: item b is served by `lanes` consecutive work-items, `per_group` items
// per work-group of `wg`.
struct DynWG96 {
    static constexpr const char* name = "DynWorkGroup(96)";
    static constexpr int wg = 96, per_group = 1, lanes = 96;
    static auto make(sycl::nd_item<1> it) { return v2::DynWorkGroup<>{it}; }
};
struct DynWG1024 {
    static constexpr const char* name = "DynWorkGroup(1024)";
    static constexpr int wg = 1024, per_group = 1, lanes = 1024;
    static auto make(sycl::nd_item<1> it) { return v2::DynWorkGroup<>{it}; }
};
struct WG128 {
    static constexpr const char* name = "WorkGroup<128>";
    static constexpr int wg = 128, per_group = 1, lanes = 128;
    static auto make(sycl::nd_item<1> it) { return v2::WorkGroup<128>{it}; }
};
struct Sub32 {
    static constexpr const char* name = "SubGroup<32>";
    static constexpr int wg = 32, per_group = 1, lanes = 32;
    static auto make(sycl::nd_item<1> it) { return v2::SubGroup<32>{it.get_sub_group()}; }
};
struct Part8 {
    static constexpr const char* name = "Partition<8>";
    static constexpr int wg = 32, per_group = 4, lanes = 8;
    static auto make(sycl::nd_item<1> it) { return v2::Partition<8>{it.get_sub_group()}; }
};
struct SgpPart8 {
    static constexpr const char* name = "SgpPartition<8, Masked>";
    static constexpr int wg = 32, per_group = 4, lanes = 8;
    static auto make(sycl::nd_item<1> it) { return v2::SgpPartition<8, true>{it.get_sub_group()}; }
};
struct Thread1 {
    static constexpr const char* name = "Thread";
    static constexpr int wg = 32, per_group = 32, lanes = 1;
    static auto make(sycl::nd_item<1>) { return v2::Thread{}; }
};

template <class Spec, class Body> void launch(Dev& dev, int items, std::size_t local_bytes, Body body) {
    dev.run(Spec::wg, items / Spec::per_group, local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const auto ex = Spec::make(it);
        const int b = static_cast<int>(it.get_group_linear_id()) * Spec::per_group +
                      static_cast<int>(it.get_local_linear_id()) / Spec::lanes;
        body(ex, b, lm);
    });
}

// Every executor; the 1024-wide launch only where the body is small enough for its 64 registers.
template <bool Ceiling, class F> void for_each_spec(F f) {
    f(DynWG96{});
    if constexpr (Ceiling) f(DynWG1024{});
    f(WG128{});
    f(Sub32{});
    f(Part8{});
    f(SgpPart8{});
    f(Thread1{});
}

template <class T> class DeviceBlasV2StedcTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};
template <class T> class DeviceBlasV2StedcRealTest : public DeviceBlasV2StedcTest<T> {};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DeviceBlasV2StedcTest, Types);
TYPED_TEST_SUITE(DeviceBlasV2StedcRealTest, RealTypes);

// ---- level 1: rot, scal_inv, fill_cols (Flat and Cyclic), nrm2_blue, allreduce, core dot ---

enum L1Slot { kNrmDefault, kNrmBlue, kSumX, kSumY, kSumXY, kDot, kL1Slots };

TYPED_TEST(DeviceBlasV2StedcTest, Level1OnEveryExecutor) {
    using T = TypeParam;
    using R = real_of<T>;
    using A = v2::Arith<T>;
    constexpr int n = 77, incx = 2, incy = 3, ncol = 4, ld = n + 3;
    const R c = R(std::cos(0.7)), s = R(std::sin(0.7)), a = R(3.3);
    const T poison = T(R(1.0e6));
    const auto hx = batched<T>(n * incx, 1), hy = batched<T>(n * incy, 2);
    const std::vector<T> hA(std::size_t(kBatch) * ld * ncol, poison);
    for_each_spec<false>([&](auto spec) {
        using Spec = decltype(spec);
        Dev& dev = this->dev;
        const T* xin = dev.up(hx);
        const T* yin = dev.up(hy);
        T* X = dev.up(hx);
        T* Y = dev.up(hy);
        T* S = dev.up(hx);
        T* A1 = dev.up(hA);
        T* A2 = dev.up(hA);
        T* res = dev.up(std::vector<T>(std::size_t(kBatch) * kL1Slots, T(R(-7))));
        launch<Spec>(dev, kBatch, 0, [=](const auto& ex, int b, unsigned char*) {
            const auto xc = v2::vec(xin + std::ptrdiff_t(b) * n * incx, n, incx);
            const auto yc = v2::vec(yin + std::ptrdiff_t(b) * n * incy, n, incy);
            v2::rot(ex, v2::vec(X + std::ptrdiff_t(b) * n * incx, n, incx), v2::vec(Y + std::ptrdiff_t(b) * n * incy, n, incy), c, s);
            v2::scal_inv(ex, a, v2::vec(S + std::ptrdiff_t(b) * n * incx, n, incx));
            v2::fill_cols<v2::alg::Flat>(ex, xc, v2::mat(A1 + std::ptrdiff_t(b) * ld * ncol, n, ncol, ld));
            v2::fill_cols(ex, xc, v2::mat(A2 + std::ptrdiff_t(b) * ld * ncol, n, ncol, ld));
            const R n0 = v2::nrm2_blue(ex, xc);
            const R n1 = v2::nrm2_blue<v2::alg::Blue>(ex, xc);
            T sx = A::zero(), sy = A::zero(), sxy = A::zero();
            v2::for_each_owned(ex, n, [&](int i) {
                sx = A::add(sx, xc(i));
                sy = A::add(sy, yc(i));
                A::mac(sxy, xc(i), yc(i));
            });
            const auto sums = v2::allreduce(ex, std::array<T, 3>{sx, sy, sxy});
            const T d = v2::dot(ex, xc, yc);
            if (ex.leader()) {
                T* r = res + std::ptrdiff_t(b) * kL1Slots;
                r[kNrmDefault] = T(n0), r[kNrmBlue] = T(n1), r[kSumX] = sums[0], r[kSumY] = sums[1], r[kSumXY] = sums[2], r[kDot] = d;
            }
        });
        const auto gX = dev.down(X, hx.size()), gY = dev.down(Y, hy.size()), gS = dev.down(S, hx.size());
        const auto gA1 = dev.down(A1, hA.size()), gA2 = dev.down(A2, hA.size()), gr = dev.down(res, std::size_t(kBatch) * kL1Slots);

        int bitdiff = 0, untouched_bad = 0;
        auto twin = [&](const std::vector<T>& g, int per) {
            for (int b = kPeriod; b < kBatch; ++b)
                for (int i = 0; i < per; ++i) bitdiff += !same_bits(g[std::size_t(b) * per + i], g[std::size_t(b % kPeriod) * per + i]);
        };
        twin(gX, n * incx), twin(gY, n * incy), twin(gS, n * incx), twin(gA1, ld * ncol), twin(gA2, ld * ncol), twin(gr, kL1Slots);
        EXPECT_EQ(bitdiff, 0) << Spec::name << ": items differ from their representative";

        double worst = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const T* x = hx.data() + std::size_t(b) * n * incx;
            const T* y = hy.data() + std::size_t(b) * n * incy;
            const std::size_t ox = std::size_t(b) * n * incx, oy = std::size_t(b) * n * incy, oa = std::size_t(b) * ld * ncol;
            LD<T> sx = 0, sy = 0, sxy = 0;
            long double ssq = 0, sabs = 0, s1x = 0, s1y = 0;
            for (int i = 0; i < n; ++i) {
                const LD<T> xi = wide(x[i * incx]), yi = wide(y[i * incy]);
                const long double C = c, Sn = s;
                const long double sc = std::abs(xi) + std::abs(yi);
                worst = std::max({worst, rel(gX[ox + i * incx], C * xi + Sn * yi, sc), rel(gY[oy + i * incy], C * yi - Sn * xi, sc),
                                  rel(gS[ox + i * incx], xi / static_cast<long double>(a), std::abs(xi) / a)});
                for (int j = 0; j < ncol; ++j) {
                    untouched_bad += !same_bits(gA1[oa + j * ld + i], x[i * incx]);
                    untouched_bad += !same_bits(gA2[oa + j * ld + i], x[i * incx]);
                }
                sx += xi, sy += yi, sxy += xi * yi;
                ssq += std::norm(xi), sabs += std::abs(xi) * std::abs(yi), s1x += std::abs(xi), s1y += std::abs(yi);
            }
            // Elements between the strided ones and the pad rows of A are never written.
            for (int i = 0; i < n; ++i) {
                for (int k = 1; k < incx; ++k) untouched_bad += !same_bits(gX[ox + i * incx + k], x[i * incx + k]) + !same_bits(gS[ox + i * incx + k], x[i * incx + k]);
                for (int k = 1; k < incy; ++k) untouched_bad += !same_bits(gY[oy + i * incy + k], y[i * incy + k]);
            }
            for (int j = 0; j < ncol; ++j)
                for (int i = n; i < ld; ++i) untouched_bad += !same_bits(gA1[oa + j * ld + i], poison) + !same_bits(gA2[oa + j * ld + i], poison);
            const T* r = gr.data() + std::size_t(b) * kL1Slots;
            const long double nrm = std::sqrt(ssq);
            worst = std::max({worst, rel(r[kNrmDefault], LD<T>(nrm), nrm), rel(r[kNrmBlue], LD<T>(nrm), nrm), rel(r[kSumX], sx, s1x),
                              rel(r[kSumY], sy, s1y), rel(r[kSumXY], sxy, sabs), rel(r[kDot], sxy, sabs)});
        }
        EXPECT_EQ(untouched_bad, 0) << Spec::name << ": wrong fill_cols value, or a stride gap / pad row was written";
        EXPECT_VERIFY(T, Check::blas, n, worst) << Spec::name;
    });
}

// ---- nrm2_blue across Blue's thresholds ------------------------------------------------------

TYPED_TEST(DeviceBlasV2StedcTest, Nrm2BlueStraddlesThresholds) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 300, items = 32, ncase = 4;
    const int p = std::numeric_limits<R>::digits, emin = std::numeric_limits<R>::min_exponent, emax = std::numeric_limits<R>::max_exponent;
    auto fdiv = [](int x, int y) { return static_cast<int>(std::floor(double(x) / y)); };
    const R tbig = std::ldexp(R(1), fdiv(emax - p + 1, 2));
    const R tsml = std::ldexp(R(1), -fdiv(-(emin - 1), 2));  // ceil((emin - 1) / 2)
    std::vector<T> hx(std::size_t(items) * n);
    for (int b = 0; b < items; ++b)
        for (int i = 0; i < n; ++i) {
            const T u = val<T>(std::uint64_t(b % ncase) * 7919u + i);
            R scale = R(1);
            switch (b % ncase) {
                case 0: scale = i % 3 == 0 ? tbig * R(8) : (i % 3 == 1 ? R(1) : tsml / R(8)); break;  // all three accumulators
                case 1: scale = i % 2 == 0 ? R(1) : tsml / R(8); break;                               // medium + small
                case 2: scale = tsml / R(8); break;                                                    // small only
                default: scale = std::ldexp(R(1), emax - 8); break;                                    // squares overflow
            }
            hx[std::size_t(b) * n + i] = u * scale;
        }
    for_each_spec<false>([&](auto spec) {
        using Spec = decltype(spec);
        Dev& dev = this->dev;
        const T* x = dev.up(hx);
        T* out = dev.up(std::vector<T>(items * 2, T(R(-7))));
        launch<Spec>(dev, items, 0, [=](const auto& ex, int b, unsigned char*) {
            const auto xv = v2::vec(x + std::ptrdiff_t(b) * n, n);
            const R r0 = v2::nrm2_blue(ex, xv);
            const R r1 = v2::nrm2_blue<v2::alg::Blue>(ex, xv);
            if (ex.leader()) out[2 * b] = T(r0), out[2 * b + 1] = T(r1);
        });
        const auto got = dev.down(out, items * 2);
        for (int b = 0; b < ncase; ++b) {
            long double ssq = 0;
            for (int i = 0; i < n; ++i) ssq += std::norm(wide(hx[std::size_t(b) * n + i]));
            const long double nrm = std::sqrt(ssq);
            ASSERT_TRUE(std::isfinite(static_cast<double>(nrm)));
            EXPECT_VERIFY(T, Check::blas, n, std::max(rel(got[2 * b], LD<T>(nrm), nrm), rel(got[2 * b + 1], LD<T>(nrm), nrm)))
                << Spec::name << " case " << b;
        }
    });
}

// ---- prod_scaled / sqrt_scaled ----------------------------------------------------------------

// Factors per case: 0 in range with one negative factor; 1 overflows R (the square root does
// not); 2 underflows R in float (0.6^1024); 3 holds an exact zero.
TYPED_TEST(DeviceBlasV2StedcRealTest, ScaledProductLeavesTheFloatingRange) {
    using T = TypeParam;
    constexpr int n = 1024, items = 32, ncase = 4, nslot = 4;
    const int ebig = std::is_same_v<T, float> ? 3 : 20;  // 64 factors of ~2^(ebig + 0.5) leave range
    std::vector<T> hf(std::size_t(items) * n);
    std::vector<int> hn(items);
    for (int b = 0; b < items; ++b) {
        const int cs = b % ncase;
        hn[b] = cs == 0 ? 200 : (cs == 1 ? 64 : (cs == 2 ? n : 100));
        for (int i = 0; i < n; ++i) {
            const T u = uval<T>(std::uint64_t(cs) * 104729u + i);
            T f = T(1) + T(0.1) * u;
            if (cs == 0 && i == 17) f = -f;
            if (cs == 1) f = std::ldexp(T(1.5) + T(0.4) * u, ebig);
            if (cs == 2) f = T(0.6);
            if (cs == 3 && i == 50) f = T(0);
            hf[std::size_t(b) * n + i] = f;
        }
    }
    for_each_spec<true>([&](auto spec) {
        using Spec = decltype(spec);
        Dev& dev = this->dev;
        const T* f = dev.up(hf);
        const int* nn = dev.up(hn);
        T* out = dev.up(std::vector<T>(std::size_t(items) * nslot, T(-7)));
        int* expo = dev.up(std::vector<int>(items, -7777));
        launch<Spec>(dev, items, 0, [=](const auto& ex, int b, unsigned char*) {
            const T* fb = f + std::ptrdiff_t(b) * n;
            const auto q = v2::prod_scaled(ex, nn[b], [&](int i) { return fb[i]; });
            const T naive = v2::reduce_op(ex, nn[b], sycl::multiplies<T>(), T(1), [&](int i) { return fb[i]; });
            if (ex.leader()) {
                T* o = out + std::ptrdiff_t(b) * nslot;
                o[0] = q.mant, o[1] = v2::sqrt_scaled(q), o[2] = naive, o[3] = T(v2::exec_width(ex));
                expo[b] = q.expo;
            }
        });
        const auto got = dev.down(out, std::size_t(items) * nslot);
        const auto ge = dev.down(expo, items);
        for (int b = 0; b < ncase; ++b) {
            const T* o = got.data() + std::size_t(b) * nslot;
            long double lg = 0;  // log2 |prod|
            int neg = 0, zero = 0;
            for (int i = 0; i < hn[b]; ++i) {
                const T v = hf[std::size_t(b) * n + i];
                zero += v == T(0), neg += v < T(0);
                if (v != T(0)) lg += std::log2(std::fabs(static_cast<long double>(v)));
            }
            if (zero) {
                EXPECT_EQ(o[0], T(0)) << Spec::name << " zero factor";
                EXPECT_EQ(o[1], T(0)) << Spec::name << " zero factor, sqrt";
                continue;
            }
            EXPECT_TRUE(std::fabs(o[0]) >= T(0.5) && std::fabs(o[0]) < T(1)) << Spec::name << " case " << b << " mantissa " << o[0];
            EXPECT_EQ(o[0] < T(0), neg % 2 == 1) << Spec::name << " case " << b << " sign";
            const long double got_lg = std::log2(std::fabs(static_cast<long double>(o[0]))) + ge[b];
            // log2 error ~ relative error / ln 2: verify it as a relative error of the product.
            EXPECT_VERIFY(T, Check::blas, hn[b], static_cast<double>(std::fabs(got_lg - lg) * std::log(2.0L)))
                << Spec::name << " case " << b << " log2 " << got_lg << " vs " << lg;
            const long double half = std::exp2(lg / 2);
            EXPECT_VERIFY(T, Check::blas, hn[b], rel(o[1], LD<T>(half), half)) << Spec::name << " case " << b << " sqrt_scaled";
            const bool in_range = std::fabs(lg) < std::numeric_limits<T>::max_exponent - 2 && lg > std::numeric_limits<T>::min_exponent + 2;
            if (!in_range) {
                // The case must leave range: the naive product overflows or underflows.
                EXPECT_TRUE(o[2] == T(0) || std::isinf(o[2])) << Spec::name << " case " << b << " naive " << o[2];
            } else if (o[3] <= T(32)) {
                // One round of at most 32 items: the scaled product IS the naive one, bit for bit.
                EXPECT_TRUE(same_bits(std::ldexp(o[0], ge[b]), o[2])) << Spec::name << " case " << b;
            }
        }
    });
}

// ---- fold / reduce_op ---------------------------------------------------------------------------

TYPED_TEST(DeviceBlasV2StedcRealTest, FoldAndReduceOpMatchHost) {
    using T = TypeParam;
    constexpr int n = 333, nslot = 6;
    // All negative: a fold that starts from 0 instead of the max identity returns 0.
    std::vector<T> hx = batched<T>(n, 21);
    for (auto& v : hx) v = -T(1) - std::fabs(v);
    for_each_spec<true>([&](auto spec) {
        using Spec = decltype(spec);
        Dev& dev = this->dev;
        const T* x = dev.up(hx);
        T* out = dev.up(std::vector<T>(std::size_t(kBatch) * nslot, T(-7)));
        launch<Spec>(dev, kBatch, 0, [=](const auto& ex, int b, unsigned char*) {
            const auto xv = v2::vec(x + std::ptrdiff_t(b) * n, n);
            const T mx = v2::fold(ex, xv, sycl::maximum<T>());
            const T sm = v2::fold(ex, xv, sycl::plus<T>());
            const T em = v2::fold(ex, xv.head(0), sycl::maximum<T>());
            const T rmx = v2::reduce_op(ex, n, sycl::maximum<T>(), -std::numeric_limits<T>::infinity(), [&](int i) { return xv(i); });
            const T rmn = v2::reduce_op(ex, n, sycl::minimum<T>(), std::numeric_limits<T>::infinity(), [&](int i) { return xv(i); });
            const T tail = v2::fold(ex, xv.tail(5), sycl::plus<T>());  // elements [5, n): a fold not at the start
            if (ex.leader()) {
                T* o = out + std::ptrdiff_t(b) * nslot;
                o[0] = mx, o[1] = sm, o[2] = em, o[3] = rmx, o[4] = rmn, o[5] = tail;
            }
        });
        const auto got = dev.down(out, std::size_t(kBatch) * nslot);
        int bitdiff = 0;
        for (int b = kPeriod; b < kBatch; ++b)
            for (int k = 0; k < nslot; ++k) bitdiff += !same_bits(got[std::size_t(b) * nslot + k], got[std::size_t(b % kPeriod) * nslot + k]);
        EXPECT_EQ(bitdiff, 0) << Spec::name << ": items differ from their representative";
        double worst = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const T* xb = hx.data() + std::size_t(b) * n;
            const T* o = got.data() + std::size_t(b) * nslot;
            const T mx = *std::max_element(xb, xb + n), mn = *std::min_element(xb, xb + n);
            long double sm = 0, sa = 0, st = 0, sta = 0;
            for (int i = 0; i < n; ++i) sm += xb[i], sa += std::fabs(xb[i]);
            for (int i = 5; i < n; ++i) st += xb[i], sta += std::fabs(xb[i]);
            EXPECT_EQ(o[0], mx) << Spec::name << " fold max";
            EXPECT_EQ(o[2], T(0)) << Spec::name << " empty fold is T{}";
            EXPECT_EQ(o[3], mx) << Spec::name << " reduce_op max";
            EXPECT_EQ(o[4], mn) << Spec::name << " reduce_op min";
            worst = std::max({worst, rel(o[1], LD<T>(sm), sa), rel(o[5], LD<T>(st), sta)});
        }
        EXPECT_VERIFY(T, Check::blas, n, worst) << Spec::name << " fold sum";
    });
}

// ---- ownership, uniform, sync_group, exec_width -----------------------------------------------

TEST(DeviceBlasV2StedcOwnership, OwnersUniformAndSync) {
    Dev dev;
    if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    constexpr int items = 32, N = 211, lo = 5, hi = 203;
    for_each_spec<true>([&](auto spec) {
        using Spec = decltype(spec);
        constexpr int L = Spec::lanes;
        int* own_in = dev.up(std::vector<int>(std::size_t(items) * N, -7));
        int* own = dev.up(std::vector<int>(std::size_t(items) * N, -7));
        float* uni = dev.up(std::vector<float>(std::size_t(items) * L, -7.f));
        int* syn = dev.up(std::vector<int>(std::size_t(items) * L, -7));
        int* width = dev.up(std::vector<int>(items, -7));
        launch<Spec>(dev, items, sizeof(int) * Spec::wg, [=](const auto& ex, int b, unsigned char* lm) {
            using Ex = std::remove_cvref_t<decltype(ex)>;
            const int lid = ex.lid();
            v2::for_each_owned_in(ex, lo, hi, [&](int i) { own_in[std::ptrdiff_t(b) * N + i] = lid; });
            v2::for_each_owned(ex, N, [&](int i) { own[std::ptrdiff_t(b) * N + i] = lid; });
            uni[std::ptrdiff_t(b) * L + lid] = v2::uniform(ex, static_cast<float>(lid * 7 + 3 + b));
            if (ex.leader()) width[b] = v2::exec_width(ex);
            if constexpr (v2::is_group_exec_v<Ex>) {
                int* buf = reinterpret_cast<int*>(lm);
                buf[lid] = lid + 1 + b;
                v2::sync_group(ex);
                const int src = (lid + 33) % L;
                syn[std::ptrdiff_t(b) * L + lid] = buf[src];
            }
        });
        const auto gin = dev.down(own_in, std::size_t(items) * N), gall = dev.down(own, std::size_t(items) * N);
        const auto gu = dev.down(uni, std::size_t(items) * L);
        const auto gs = dev.down(syn, std::size_t(items) * L);
        const auto gw = dev.down(width, items);
        int bad_in = 0, bad_all = 0, bad_u = 0, bad_s = 0, bad_w = 0;
        for (int b = 0; b < items; ++b) {
            bad_w += gw[b] != L;
            for (int i = 0; i < N; ++i) {
                const int e_in = (i >= lo && i < hi) ? (i - lo) % L : -7;
                bad_in += gin[std::size_t(b) * N + i] != e_in;
                bad_all += gall[std::size_t(b) * N + i] != i % L;
            }
            for (int l = 0; l < L; ++l) {
                bad_u += gu[std::size_t(b) * L + l] != static_cast<float>(3 + b);
                if constexpr (v2::is_group_exec_v<decltype(Spec::make(std::declval<sycl::nd_item<1>>()))>)
                    bad_s += gs[std::size_t(b) * L + l] != ((l + 33) % L) + 1 + b;
            }
        }
        EXPECT_EQ(bad_in, 0) << Spec::name << " for_each_owned_in owner map";
        EXPECT_EQ(bad_all, 0) << Spec::name << " for_each_owned owner map";
        EXPECT_EQ(bad_u, 0) << Spec::name << " uniform is the leader's value";
        EXPECT_EQ(bad_s, 0) << Spec::name << " sync_group orders the group's local memory";
        EXPECT_EQ(bad_w, 0) << Spec::name << " exec_width";
    });
}

// Compile-time facts: DynWorkGroup has no static size (level-2 ops and packed exchanges do not
// compile on it); the executor classes the ops dispatch on.
template <class E> concept has_static_size = requires { E::size; };
static_assert(!has_static_size<v2::DynWorkGroup<>> && has_static_size<v2::Partition<8>> && has_static_size<v2::WorkGroup<64>>);
static_assert(v2::is_group_exec_v<v2::DynWorkGroup<>> && v2::is_group_exec_v<v2::WorkGroup<64>> && !v2::is_group_exec_v<v2::Partition<8>>);
static_assert(v2::is_partition_v<v2::SgpPartition<8, true>> && v2::is_sub_group_exec_v<v2::SubGroup<32>> && !v2::is_sub_group_exec_v<v2::Thread>);

}  // namespace

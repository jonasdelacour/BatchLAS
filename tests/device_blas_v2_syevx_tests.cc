// SPDX-License-Identifier: MIT
// syevx-private device LA v2 ops (src/ops/syevx/syevx_v2.hh) against host references; the core
// ops they feed (copy, lacpy) are covered by device_blas_v2_tests. Guards (agent guide section
// 8): run-time widths 40 (a partial sub-group), 96 and the 1024 ceiling; the flip direction read
// per item at run time; non-natural inc and ld; a view at an offset; complex data; NaN-poisoned
// outputs whose padding must stay poisoned; batch 1024; count_le ties (x == v counts), queries one
// ulp below a tie, NaN, +-Inf and an empty array; sub_scaled with an Inf imaginary part that a
// cross-term product would turn into NaN. Deliberate breaks: Flipped reading v(n - i);
// ColsFlipped ignoring rev; count_le with `<`; sub_scaled scaling conj(x).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <src/ops/syevx/syevx_v2.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
namespace sx = batchlas::device::syevx;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kBatch = 1024;

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
template <class T> std::vector<T> filled(std::size_t n, std::uint64_t seed) {
    std::vector<T> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = val<T>(seed * 1000003u + i);
    return h;
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> bool is_nan(const T& v) { return std::isnan(std::abs(v)); }

struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible()
                                   ? std::optional<Queue>(std::in_place, Device::default_device())
                                   : std::nullopt;
    sycl::queue& q() { return sycl_queue(*ctx); }
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
    // A run-time work-group size, as the syevx kernels launch RtGroup.
    template <class F> void run(int batch, int wg, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * wg, std::size_t(wg)),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2SyevxTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2SyevxTest, Types);

// Item b reverses iff b is odd. Columns: lacpy(cols_flipped(A, rev)) into an ld-padded block;
// vector: copy(flipped(x.tail(off).head(len), rev)) from a stride-inc source, as DirectSelect.
template <class T> void run_flips(Dev& dev, int wg) {
    constexpr int m = 37, n = 9, lda = m + 3, ldo = m + 5, nx = 23, inc = 2, off = 3, len = nx - 5;
    const auto ha = filled<T>(std::size_t(kBatch) * n * lda, 11), hx = filled<T>(std::size_t(kBatch) * nx * inc, 12);
    const T nan = T(std::numeric_limits<real_of<T>>::quiet_NaN());
    std::vector<T> ho(std::size_t(kBatch) * n * ldo, nan), hv(std::size_t(kBatch) * nx, nan);
    const T* a = dev.up(ha);
    const T* x = dev.up(hx);
    T* o = dev.up(ho);
    T* v = dev.up(hv);
    dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
        const v2::RtGroup<> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const bool rev = (b & 1) != 0;
        const auto A = v2::mat(a + std::ptrdiff_t(b) * n * lda, m, n, lda);
        v2::lacpy<v2::Part::Full, Transpose::NoTrans, v2::alg::Flat>(ex, sx::cols_flipped(A, rev),
                                                                     v2::mat(o + std::ptrdiff_t(b) * n * ldo, m, n, ldo));
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * nx * inc, nx, inc);
        v2::copy(ex, sx::flipped(xv.tail(off).head(len), rev), v2::vec(v + std::ptrdiff_t(b) * nx, len));
    });
    const auto go = dev.down(o, ho.size()), gv = dev.down(v, hv.size());
    int bad = 0, poison = 0;
    for (int b = 0; b < kBatch; ++b) {
        const bool rev = (b & 1) != 0;
        const T* ab = ha.data() + std::size_t(b) * n * lda;
        const T* ob = go.data() + std::size_t(b) * n * ldo;
        for (int c = 0; c < n; ++c) {
            const int src = rev ? n - 1 - c : c;
            for (int r = 0; r < m; ++r) bad += !same_bits(ob[r + std::size_t(c) * ldo], ab[r + std::size_t(src) * lda]);
            for (int r = m; r < ldo; ++r) poison += !is_nan(ob[r + std::size_t(c) * ldo]);
        }
        const T* xb = hx.data() + std::size_t(b) * nx * inc;
        const T* vb = gv.data() + std::size_t(b) * nx;
        for (int i = 0; i < len; ++i) bad += !same_bits(vb[i], xb[(off + (rev ? len - 1 - i : i)) * inc]);
        for (int i = len; i < nx; ++i) poison += !is_nan(vb[i]);
    }
    EXPECT_EQ(bad, 0) << "wg " << wg;
    EXPECT_EQ(poison, 0) << "written past the block, wg " << wg;
}

TYPED_TEST(DeviceBlasV2SyevxTest, FlippedOperandsOnRunTimeWidths) {
    for (int wg : {40, 96, 1024}) run_flips<TypeParam>(this->dev, wg);
}

// D := sub_scaled(A, s_b, X) elementwise over a batch, s_b real per item; long-double reference.
TYPED_TEST(DeviceBlasV2SyevxTest, SubScaledBatch) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int m = 19, n = 5, ld = 23, ldd = 21;
    const auto ha = filled<T>(std::size_t(kBatch) * n * ld, 31), hx = filled<T>(std::size_t(kBatch) * n * ld, 32);
    const auto hs = filled<R>(kBatch, 33);
    std::vector<T> hd(std::size_t(kBatch) * n * ldd, T(std::numeric_limits<R>::quiet_NaN()));
    const T* a = this->dev.up(ha);
    const T* x = this->dev.up(hx);
    const R* s = this->dev.up(hs);
    T* d = this->dev.up(hd);
    this->dev.run(kBatch, 96, [=](sycl::nd_item<1> it) {
        const v2::RtGroup<> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto A = v2::mat(a + std::ptrdiff_t(b) * n * ld, m, n, ld);
        const auto X = v2::mat(x + std::ptrdiff_t(b) * n * ld, m, n, ld);
        const auto D = v2::mat(d + std::ptrdiff_t(b) * n * ldd, m, n, ldd);
        v2::for_each_owned_2d(ex, m, n, [&](int r, int c) { D(r, c) = sx::sub_scaled(A(r, c), s[b], X(r, c)); });
    });
    const auto gd = this->dev.down(d, hd.size());
    double worst = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < m; ++r) {
                const std::size_t ia = std::size_t(b) * n * ld + r + std::size_t(c) * ld;
                const long double sb = hs[b];
                const std::complex<long double> av(std::real(ha[ia]), std::imag(ha[ia])), xv(std::real(hx[ia]), std::imag(hx[ia]));
                const std::complex<long double> want(av.real() - sb * xv.real(), av.imag() - sb * xv.imag());
                const T got = gd[std::size_t(b) * n * ldd + r + std::size_t(c) * ldd];
                const std::complex<long double> g(std::real(got), std::imag(got));
                const long double scale = std::max<long double>(std::abs(av) + std::abs(sb) * std::abs(xv), std::numeric_limits<R>::min());
                worst = std::max(worst, std::isnan(std::abs(g - want)) ? std::numeric_limits<double>::infinity()
                                                                       : static_cast<double>(std::abs(g - want) / scale));
            }
    EXPECT_VERIFY(R, Check::blas, 2, worst);
}

TYPED_TEST(DeviceBlasV2SyevxTest, CountLeAndSubScaledSpecials) {
    using T = TypeParam;
    using R = real_of<T>;
    const R inf = std::numeric_limits<R>::infinity(), qnan = std::numeric_limits<R>::quiet_NaN();
    const R below_m1 = std::nextafter(R(-1), -inf), below_2 = std::nextafter(R(2), -inf);
    const std::vector<R> hx = {R(-3), R(-1), R(-1), R(0), R(2), R(2), R(2), R(5)};
    const std::vector<R> hq = {R(-4), R(-1), below_m1, R(2), below_2, R(5), R(1), qnan, inf, -inf};
    const std::vector<int> want = {0, 3, 1, 7, 4, 8, 4, 0, 8, 0};
    const R* x = this->dev.up(hx);
    const R* qv = this->dev.up(hq);
    int* cnt = this->dev.up(std::vector<int>(hq.size() + 1, -7));
    T* sc = this->dev.up(std::vector<T>(2, T(0)));
    const int nq = static_cast<int>(hq.size()), nx = static_cast<int>(hx.size());
    T z, hinf;
    if constexpr (is_cplx<T>) z = T(R(3), R(4)), hinf = T(R(1), inf);
    else z = T(3), hinf = T(inf);
    this->dev.run(1, 32, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const auto xv = v2::vec(x, nx);
        for (int q = 0; q < nq; ++q) cnt[q] = sx::count_le(xv, qv[q]);
        cnt[nq] = sx::count_le(v2::vec(x, 0), R(0));
        sc[0] = sx::sub_scaled(z, R(2), hinf);
        sc[1] = sx::sub_scaled(z, R(-0.5), z);
    });
    const auto gc = this->dev.down(cnt, hq.size() + 1);
    for (int q = 0; q < nq; ++q) EXPECT_EQ(gc[q], want[q]) << "query " << q;
    EXPECT_EQ(gc[nq], 0) << "empty array";
    const auto gs = this->dev.down(sc, 2);
    if constexpr (is_cplx<T>) {
        // No cross terms: (2 + 0i)(1 + inf i) by the full product has 0 * inf = NaN in Re.
        EXPECT_EQ(gs[0].real(), R(1));
        EXPECT_EQ(gs[0].imag(), -inf);
        EXPECT_TRUE(same_bits(gs[1], T(R(4.5), R(6))));
    } else {
        EXPECT_EQ(gs[0], -inf);
        EXPECT_TRUE(same_bits(gs[1], T(R(4.5))));
    }
}

}  // namespace

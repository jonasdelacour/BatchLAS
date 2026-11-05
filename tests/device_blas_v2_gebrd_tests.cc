// SPDX-License-Identifier: MIT
// The core views and ops the complex labrd panel (lapack/gebrd.hh labrd_panel) composes, against
// long-double host references, used the way the panel uses them: op_view<ConjTrans> and conj_view
// as gemv_acc operands, a conj_view y0 that aliases y, a -0 fn_vec as y0, lacgv on a WorkGroup and
// a Thread. Guards (agent guide section 8): non-natural ld and inc, complex data with nonzero
// imaginary parts, a zero row whose result must keep the sign of zero (the -0 y0 adds nothing),
// outputs poisoned with NaN that the -0 y0 must never read, strided gaps compared bit for bit,
// batch 1024 with period-3 data.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/device.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
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
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i);
    return h;
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> T nan_of() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

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
    // No collective in these ops: no sub-group size is required.
    template <int WG, class F> void run(int groups, F f) {
        q().submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), f); }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GebrdTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GebrdTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class V> V cj(V v) {
    if constexpr (requires { v.imag(); }) return std::conj(v);
    else return v;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// y := conj(y) + alpha (B^H conj(x1) + A x2), y0 = conj_view(y) aliasing y, x1 with inc 2,
// B stored k1 x m (read transposed). Then z := beta (A x2) with a -0 y0 over a NaN-poisoned
// z; A's row 0 is zero, so z(0) = beta * (+0) must keep its sign: re(z(0)) is -0 for beta < 0.
TYPED_TEST(DeviceBlasV2GebrdTest, ConjAndHermViewsInGemvAcc) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 128, m = 37, k1 = 5, k2 = 6, ld = m + 3, ldb = k1 + 5, inc = 2;
    constexpr int oA = 0, oB = ld * k2, oX1 = oB + ldb * m, oX2 = oX1 + k1 * inc, sz = oX2 + k2;
    const T alpha = val<T>(7);
    const T beta = T(R(-1.5));
    auto hd = batched<T>(sz, 11), hy = batched<T>(m, 12);
    for (int b = 0; b < kBatch; ++b)
        for (int k = 0; k < k2; ++k) hd[std::size_t(b) * sz + oA + k * ld] = T(0);
    std::vector<T> hz(std::size_t(kBatch) * m, nan_of<T>());
    T* d = this->dev.up(hd);
    T* y = this->dev.up(hy);
    T* z = this->dev.up(hz);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        T* p = d + std::ptrdiff_t(b) * sz;
        const auto A = v2::mat(p + oA, m, k2, ld);
        const auto B = v2::mat(p + oB, k1, m, ldb);
        const auto x1 = v2::assume_published(v2::conj_view(v2::vec(p + oX1, k1, inc)));
        const auto x2 = v2::assume_published(v2::vec(p + oX2, k2));
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * m, m), zv = v2::vec(z + std::ptrdiff_t(b) * m, m);
        v2::gemv_acc(ex, alpha, v2::conj_view(yv), yv, v2::term(v2::op_view<Transpose::ConjTrans>(B), x1), v2::term(A, x2));
        const auto neg_zero = v2::fn_vec(m, [](int) { return v2::Arith<T>::make(-R(0), -R(0)); });
        v2::gemv_acc(ex, beta, neg_zero, zv, v2::term(A, x2));
    });
    const auto gy = this->dev.down(y, hy.size()), gz = this->dev.down(z, hz.size());
    int bitdiff = 0, zero_sign_bad = 0;
    for (std::size_t i = 0; i < gy.size(); ++i)
        bitdiff += !same_bits(gy[i], gy[i % (std::size_t(kPeriod) * m)]) + !same_bits(gz[i], gz[i % (std::size_t(kPeriod) * m)]);
    for (int b = 0; b < kBatch; ++b) {
        const R z0 = v2::Arith<T>::re(gz[std::size_t(b) * m]);
        zero_sign_bad += !(z0 == R(0) && std::signbit(z0));
    }
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_EQ(zero_sign_bad, 0) << "a -0 y0 must add nothing: beta * (+0) keeps its -0";
    double worst = 0, worst_z = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = hd.data() + std::size_t(b) * sz;
        for (int r = 0; r < m; ++r) {
            LD<T> acc = 0, az = 0;
            long double s = 0, sz2 = 0;
            for (int k = 0; k < k1; ++k) {
                const LD<T> e = cj(wide(p[oB + k + r * ldb])), x = cj(wide(p[oX1 + k * inc]));
                acc += e * x, s += std::abs(e) * std::abs(x);
            }
            for (int k = 0; k < k2; ++k) {
                const LD<T> e = wide(p[oA + r + k * ld]), x = wide(p[oX2 + k]);
                acc += e * x, az += e * x, s += std::abs(e) * std::abs(x), sz2 += std::abs(e) * std::abs(x);
            }
            const LD<T> y0 = cj(wide(hy[std::size_t(b) * m + r]));
            worst = std::max(worst, rel(gy[std::size_t(b) * m + r], y0 + wide(alpha) * acc, std::abs(y0) + std::abs(wide(alpha)) * s));
            worst_z = std::max(worst_z, rel(gz[std::size_t(b) * m + r], wide(beta) * az, std::abs(wide(beta)) * sz2));
        }
    }
    EXPECT_VERIFY(T, Check::blas, k1 + k2 + 1, worst);
    EXPECT_VERIFY(T, Check::blas, k2, worst_z) << "-0 y0 over a NaN-poisoned z";
}

// lacgv on a WorkGroup (n = 150 > WG: items own several elements) and on a Thread (item 0), both
// with inc 3: exactly the strided elements are conjugated (bit for bit), the gaps are untouched;
// a real T is left bit-identical.
TYPED_TEST(DeviceBlasV2GebrdTest, LacgvStridedOnWorkGroupAndThread) {
    using T = TypeParam;
    constexpr int WG = 64, n = 150, inc = 3, per = 2 * n * inc;
    auto hx = batched<T>(per, 81);
    T* x = this->dev.up(hx);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        T* p = x + std::ptrdiff_t(b) * per;
        v2::lacgv(ex, v2::vec(p, n, inc));
        if (ex.leader()) v2::lacgv(v2::Thread{}, v2::vec(p + n * inc, n, inc));
    });
    const auto gx = this->dev.down(x, hx.size());
    int conj_bad = 0, gap_bad = 0;
    for (std::size_t e = 0; e < gx.size(); ++e) {
        const T h = hx[e];
        if ((e % per) % inc == 0) conj_bad += !same_bits(gx[e], T(cj(h)));
        else gap_bad += !same_bits(gx[e], h);
    }
    EXPECT_EQ(conj_bad, 0) << "lacgv must conjugate every strided element exactly";
    EXPECT_EQ(gap_bad, 0) << "lacgv wrote outside its strided elements";
}

}  // namespace

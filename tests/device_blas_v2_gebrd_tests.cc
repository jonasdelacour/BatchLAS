// SPDX-License-Identifier: MIT
// Device LA v2 extension ops (include/batchlas/blas/device/v2/ext/gebrd.hh) against long-double
// host references. Guards (docs/developer/agent-guide.md section 8): non-natural ld and inc,
// complex data with nonzero imaginary parts, Trans and ConjTrans terms, an operand that is also
// the output (gemv_acc's y0 = conj(y)), beta == Zero over NaN-poisoned outputs, a NaN stored at
// the implied 1 of a UnitHead (never read), columns / rows outside larf's window and dead
// partitions compared bit for bit, tau == 0 items, batch 1024 with period-3 data where every item
// must be bit-identical to its representative (the SLM and shuffle paths).
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
#include <batchlas/blas/device/v2/ext/gebrd.hh>
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
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GebrdTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
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

// Householder scalars with a real beta (zlarfg as gebrd needs it), explicit real arithmetic only.
struct RealBetaGen {
    template <class T> v2::Reflector<T> operator()(T alpha, real_of<T> xnorm, int len) const {
        using A = v2::Arith<T>;
        using R = real_of<T>;
        const R ar = A::re(alpha), ai = A::im(alpha);
        if (len < 1 || (xnorm == R(0) && ai == R(0))) return {alpha, A::zero(), A::zero()};
        const R nrm = sycl::hypot(sycl::hypot(ar, ai), xnorm);
        const R beta = sycl::signbit(ar) ? nrm : -nrm;
        const R dr = ar - beta, d2 = dr * dr + ai * ai;  // scale = 1 / (alpha - beta)
        return {A::make(beta, R(0)), A::make((beta - ar) / beta, -ai / beta), A::make(dr / d2, -ai / d2)};
    }
};
struct RealBetaGenArr {
    template <class T> std::array<T, 3> operator()(T alpha, real_of<T> xnorm, int len) const {
        const auto s = RealBetaGen{}(alpha, xnorm, len);
        return {s.beta, s.tau, s.scale};
    }
};

// || H^H [alpha; x] - [beta; 0] || / || [alpha; x] || with H = I - tau v v^H, v = [1; vx].
template <class T>
double reflector_error(T alpha, const std::vector<LD<T>>& x, T beta, T tau, const std::vector<T>& vx) {
    std::vector<LD<T>> v{LD<T>(1)}, y{wide(alpha)};
    for (std::size_t i = 0; i < x.size(); ++i) v.push_back(wide(vx[i])), y.push_back(x[i]);
    LD<T> w = 0;
    long double nrm = 0;
    for (std::size_t i = 0; i < v.size(); ++i) w += cj(v[i]) * y[i], nrm += std::norm(y[i]);
    const LD<T> ct = cj(wide(tau));
    double worst = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const LD<T> r = y[i] - ct * v[i] * w;
        worst = std::max(worst, rel(T(i == 0 ? beta : T(0)), r, std::sqrt(nrm)));
    }
    return worst;
}

// ---- fused RowOwner gemv sequences -------------------------------------------------------------

// y := alpha (conj(y) - A1 conj(x1) + [A2 x2, -B^H x3 interleaved] + B2^T x4): every term kind,
// y0 aliasing y, x1 with inc 2. A second launch starts from Zero over a NaN-poisoned y.
TYPED_TEST(DeviceBlasV2GebrdTest, GemvAccChainsEveryTermKind) {
    using T = TypeParam;
    constexpr int WG = 128, m = 37, k1 = 5, k2 = 4, k3 = 6, ld = m + 3, ldb = k2 + 5, ldb2 = k3 + 3, inc = 2;
    constexpr int sz = ld * k1 + ld * k2 + ldb * m + ldb2 * m + k1 * inc + 2 * k2 + k3;
    constexpr int oA2 = ld * k1, oB = oA2 + ld * k2, oB2 = oB + ldb * m, oX1 = oB2 + ldb2 * m, oX2 = oX1 + k1 * inc,
                  oX3 = oX2 + k2, oX4 = oX3 + k2;
    const T alpha = val<T>(7);
    auto hd = batched<T>(sz, 11), hy = batched<T>(m, 12);
    std::vector<T> hz(std::size_t(kBatch) * m, nan_of<T>());
    T* d = this->dev.up(hd);
    T* y = this->dev.up(hy);
    T* z = this->dev.up(hz);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        T* p = d + std::ptrdiff_t(b) * sz;
        constexpr auto N = Transpose::NoTrans, Tt = Transpose::Trans, C = Transpose::ConjTrans;
        const auto A1 = v2::mat(p, m, k1, ld), A2 = v2::mat(p + oA2, m, k2, ld);
        const auto B = v2::mat(p + oB, k2, m, ldb), B2 = v2::mat(p + oB2, k3, m, ldb2);
        const auto x1 = v2::assume_published(v2::conj_view(v2::vec(p + oX1, k1, inc)));
        const auto x2 = v2::assume_published(v2::vec(p + oX2, k2)), x3 = v2::assume_published(v2::vec(p + oX3, k2));
        const auto x4 = v2::assume_published(v2::vec(p + oX4, k3));
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * m, m), zv = v2::vec(z + std::ptrdiff_t(b) * m, m);
        v2::gemv_acc(ex, alpha, v2::conj_view(yv), yv, v2::gemv_term<N>(v2::NegOne{}, A1, x1),
                     v2::interleave(v2::gemv_term<N>(v2::One{}, A2, x2), v2::gemv_term<C>(v2::NegOne{}, B, x3)),
                     v2::gemv_term<Tt>(v2::One{}, B2, x4));
        v2::gemv_acc(ex, v2::One{}, v2::Zero{}, zv, v2::gemv_term<N>(v2::One{}, A2, x2));
    });
    const auto gy = this->dev.down(y, hy.size()), gz = this->dev.down(z, hz.size());
    int bitdiff = 0;
    for (std::size_t i = 0; i < gy.size(); ++i)
        bitdiff += !same_bits(gy[i], gy[i % (std::size_t(kPeriod) * m)]) + !same_bits(gz[i], gz[i % (std::size_t(kPeriod) * m)]);
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    double worst = 0, worst_z = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = hd.data() + std::size_t(b) * sz;
        for (int r = 0; r < m; ++r) {
            LD<T> acc = cj(wide(hy[std::size_t(b) * m + r])), az = 0;
            long double s = std::abs(acc), sz2 = 0;
            auto add = [&](LD<T> e, LD<T> x, long double sign) { acc += sign * e * x, s += std::abs(e) * std::abs(x); };
            for (int k = 0; k < k1; ++k) add(wide(p[r + k * ld]), cj(wide(p[oX1 + k * inc])), -1);
            for (int k = 0; k < k2; ++k) {
                add(wide(p[oA2 + r + k * ld]), wide(p[oX2 + k]), 1);
                add(cj(wide(p[oB + k + r * ldb])), wide(p[oX3 + k]), -1);
                az += wide(p[oA2 + r + k * ld]) * wide(p[oX2 + k]), sz2 += std::abs(wide(p[oA2 + r + k * ld])) * std::abs(wide(p[oX2 + k]));
            }
            for (int k = 0; k < k3; ++k) add(wide(p[oB2 + k + r * ldb2]), wide(p[oX4 + k]), 1);
            worst = std::max(worst, rel(gy[std::size_t(b) * m + r], wide(alpha) * acc, std::abs(wide(alpha)) * s));
            worst_z = std::max(worst_z, rel(gz[std::size_t(b) * m + r], az, sz2));
        }
    }
    EXPECT_VERIFY(T, Check::blas, k1 + 2 * k2 + k3 + 1, worst);
    EXPECT_VERIFY(T, Check::blas, k2, worst_z) << "Zero init over a NaN-poisoned y";
}

// gemv_pair (Trans, ConjTrans) into local memory pre-filled with NaN (beta == Zero), and into
// global memory with beta != 0; gemv Trans on transposed(M) is M x.
template <Transpose TR, class T> void gemv_pair_case(Dev& dev) {
    constexpr int WG = 128, m = 41, k = 7, ld = m + 2, inc = 3, ldt = k + 4;
    constexpr int sz = 2 * ld * k + m * inc + ldt * m;
    const T alpha = val<T>(31), beta = val<T>(32);
    auto hd = batched<T>(sz, 33), hy = batched<T>(2 * k, 34);
    std::vector<T> ho(std::size_t(kBatch) * 3 * k, nan_of<T>());
    T* d = dev.up(hd);
    T* y = dev.up(hy);
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch, sizeof(T) * 2 * k, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        T* p = d + std::ptrdiff_t(b) * sz;
        T* l = reinterpret_cast<T*>(lm);
        const auto M1 = v2::mat(p, m, k, ld), M2 = v2::mat(p + ld * k, m, k, ld), Mt = v2::mat(p + 2 * ld * k + m * inc, k, m, ldt);
        const auto x = v2::assume_published(v2::vec(p + 2 * ld * k, m, inc));
        const auto l1 = v2::vec<v2::Mem::Local>(l, k), l2 = v2::vec<v2::Mem::Local>(l + k, k);
        v2::for_each_owned(ex, 2 * k, [&](int i) { l[i] = nan_of<T>(); });
        ex.barrier();
        v2::gemv_pair<TR>(ex, alpha, M1, M2, x, v2::Zero{}, l1, l2);
        T* ob = o + std::ptrdiff_t(b) * 3 * k;
        v2::gemv<Transpose::Trans, v2::alg::GroupReducePerOutput>(ex, v2::One{}, v2::transposed(Mt), x, v2::Zero{}, v2::vec(ob + 2 * k, k));
        ex.barrier();
        v2::copy(ex, l1, v2::vec(ob, k));
        v2::copy(ex, l2, v2::vec(ob + k, k));
        T* yb = y + std::ptrdiff_t(b) * 2 * k;
        v2::gemv_pair<TR>(ex, alpha, M1, M2, x, beta, v2::vec(yb, k), v2::vec(yb + k, k));
    });
    const auto go = dev.down(o, ho.size()), gy = dev.down(y, hy.size());
    int bitdiff = 0;
    for (std::size_t i = 0; i < go.size(); ++i) bitdiff += !same_bits(go[i], go[i % (std::size_t(kPeriod) * 3 * k)]);
    for (std::size_t i = 0; i < gy.size(); ++i) bitdiff += !same_bits(gy[i], gy[i % (std::size_t(kPeriod) * 2 * k)]);
    EXPECT_EQ(bitdiff, 0);
    double worst = 0, worst_t = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = hd.data() + std::size_t(b) * sz;
        for (int c = 0; c < k; ++c) {
            for (int w = 0; w < 2; ++w) {
                LD<T> acc = 0;
                long double s = 0;
                for (int r = 0; r < m; ++r) {
                    LD<T> e = wide(p[w * ld * k + r + c * ld]);
                    if (TR == Transpose::ConjTrans) e = cj(e);
                    const LD<T> xr = wide(p[2 * ld * k + r * inc]);
                    acc += e * xr, s += std::abs(e) * std::abs(xr);
                }
                const LD<T> y0 = wide(hy[std::size_t(b) * 2 * k + w * k + c]);
                worst = std::max(worst, rel(go[std::size_t(b) * 3 * k + w * k + c], wide(alpha) * acc, std::abs(wide(alpha)) * s));
                worst = std::max(worst, rel(gy[std::size_t(b) * 2 * k + w * k + c], wide(alpha) * acc + wide(beta) * y0,
                                            std::abs(wide(alpha)) * s + std::abs(wide(beta)) * std::abs(y0)));
            }
            LD<T> acc = 0;
            long double s = 0;
            for (int r = 0; r < m; ++r) {
                const LD<T> e = wide(p[2 * ld * k + m * inc + c + r * ldt]), xr = wide(p[2 * ld * k + r * inc]);
                acc += e * xr, s += std::abs(e) * std::abs(xr);
            }
            worst_t = std::max(worst_t, rel(go[std::size_t(b) * 3 * k + 2 * k + c], acc, s));
        }
    }
    EXPECT_VERIFY(T, Check::blas, m, worst);
    EXPECT_VERIFY(T, Check::blas, m, worst_t) << "transposed()";
}

TYPED_TEST(DeviceBlasV2GebrdTest, GemvPairSharesXAndTransposedView) {
    gemv_pair_case<Transpose::Trans, TypeParam>(this->dev);
    gemv_pair_case<Transpose::ConjTrans, TypeParam>(this->dev);
}

// ---- Householder reflectors --------------------------------------------------------------------

// larfg_bcast on a WorkGroup: every item must see the leader's scalars (uniform), and the scaled
// x must reflect [alpha; x] onto [beta; 0]. Item 2 of each period has x = 0, alpha real: tau = 0.
TYPED_TEST(DeviceBlasV2GebrdTest, LarfgBcastUniformAndReflects) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 64, n = 45, inc = 2, sz = 1 + n * inc;
    auto hd = batched<T>(sz, 41);
    for (int b = 2; b < kBatch; b += kPeriod) {
        hd[std::size_t(b) * sz] = T(R(0.75));
        for (int i = 0; i < n; ++i) hd[std::size_t(b) * sz + 1 + i * inc] = T(0);
    }
    std::vector<T> ho(std::size_t(kBatch) * WG * 3, nan_of<T>());
    T* d = this->dev.up(hd);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        T* p = d + std::ptrdiff_t(b) * sz;
        const auto x = v2::vec(p + 1, n, inc);
        const R sigma = v2::reduce(ex, n, [&](int i) { return v2::abs2(x(i)); });
        const auto h = v2::larfg_bcast(ex, p[0], sigma, n + 1, RealBetaGen{});
        T* ob = o + (std::ptrdiff_t(b) * WG + ex.lid()) * 3;
        ob[0] = h.beta, ob[1] = h.tau, ob[2] = h.scale;
        if (h.tau != v2::Arith<T>::zero()) v2::scal(ex, h.scale, x);
    });
    const auto go = this->dev.down(o, ho.size()), gd = this->dev.down(d, hd.size());
    int nonuniform = 0, bitdiff = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int l = 0; l < WG * 3; ++l) nonuniform += !same_bits(go[std::size_t(b) * WG * 3 + l], go[std::size_t(b) * WG * 3 + l % 3]);
    for (std::size_t i = 0; i < gd.size(); ++i) bitdiff += !same_bits(gd[i], gd[i % (std::size_t(kPeriod) * sz)]);
    EXPECT_EQ(nonuniform, 0) << "larfg_bcast scalars differ between items";
    EXPECT_EQ(bitdiff, 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<LD<T>> x0;
        std::vector<T> vx;
        for (int i = 0; i < n; ++i) x0.push_back(wide(hd[std::size_t(b) * sz + 1 + i * inc])), vx.push_back(gd[std::size_t(b) * sz + 1 + i * inc]);
        const T beta = go[std::size_t(b) * WG * 3], tau = go[std::size_t(b) * WG * 3 + 1];
        worst = std::max(worst, reflector_error<T>(hd[std::size_t(b) * sz], x0, beta, tau, vx));
        if (b == 2) EXPECT_TRUE(tau == T(0)) << "x = 0 and real alpha must give tau = 0";
    }
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// larfg_lanes on SgpPartition<16> (two problems per sub-group): lane 2 holds alpha, lanes 3..14
// hold x, lanes 0, 1, 15 are outside. Outside lanes must keep their values bit for bit.
TYPED_TEST(DeviceBlasV2GebrdTest, LarfgLanesOnAPartition) {
    using T = TypeParam;
    constexpr int P = 16, head = 2, len = 13, WG = 32;
    auto hd = batched<T>(P, 51);
    std::vector<T> ht(std::size_t(kBatch) * P, nan_of<T>());
    T* d = this->dev.up(hd);
    T* t = this->dev.up(ht);
    this->dev.template run<WG>(kBatch / 2, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::SgpPartition<P> ex{it.get_sub_group()};
        const int b = static_cast<int>(it.get_group_linear_id()) * 2 + static_cast<int>(ex.part.get_group_linear_id());
        const int lane = ex.lid();
        T* p = d + std::ptrdiff_t(b) * P;
        T alpha = lane == head ? p[lane] : T(0);
        const bool on = lane > head && lane < head + len;
        T x = on ? p[lane] : T(0);
        const T tau = v2::larfg_lanes(ex, len, head, alpha, x, on, RealBetaGenArr{});
        if (lane == head) p[lane] = alpha;
        else if (on) p[lane] = x;
        t[std::ptrdiff_t(b) * P + lane] = tau;
    });
    const auto gd = this->dev.down(d, hd.size()), gt = this->dev.down(t, ht.size());
    int bitdiff = 0, nonuniform = 0, outside = 0;
    for (std::size_t i = 0; i < gd.size(); ++i) bitdiff += !same_bits(gd[i], gd[i % (std::size_t(kPeriod) * P)]);
    for (std::size_t i = 0; i < gt.size(); ++i) nonuniform += !same_bits(gt[i], gt[i - i % P]);
    for (int b = 0; b < kBatch; ++b)
        for (int l : {0, 1, 15}) outside += !same_bits(gd[std::size_t(b) * P + l], hd[std::size_t(b) * P + l]);
    EXPECT_EQ(bitdiff, 0);
    EXPECT_EQ(nonuniform, 0) << "tau differs between lanes";
    EXPECT_EQ(outside, 0) << "a lane outside [alpha; x] was written";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<LD<T>> x0;
        std::vector<T> vx;
        for (int l = head + 1; l < head + len; ++l) x0.push_back(wide(hd[std::size_t(b) * P + l])), vx.push_back(gd[std::size_t(b) * P + l]);
        worst = std::max(worst, reflector_error<T>(hd[std::size_t(b) * P + head], x0, gd[std::size_t(b) * P + head], gt[std::size_t(b) * P], vx));
    }
    EXPECT_VERIFY(T, Check::blas, len, worst);
}

// Host H C (Left) or C H (Right), H = I - tau v v^H, on the window [lo, ...) of C's columns
// (Left) or rows (Right); v(0) = 1.
template <class T>
std::vector<LD<T>> larf_ref(Side sd, const T* c, int ldc, int m, int n, const std::vector<LD<T>>& v, T tau, int lo, long double& scale) {
    std::vector<LD<T>> out(std::size_t(m) * n);
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < m; ++r) out[r + std::size_t(j) * m] = wide(c[r + std::size_t(j) * ldc]);
    const LD<T> t = wide(tau);
    scale = 0;
    if (sd == Side::Left) {
        for (int j = lo; j < n; ++j) {
            LD<T> w = 0;
            for (int r = 0; r < m; ++r) w += cj(v[r]) * out[r + std::size_t(j) * m], scale = std::max(scale, std::abs(out[r + std::size_t(j) * m]));
            for (int r = 0; r < m; ++r) out[r + std::size_t(j) * m] -= t * v[r] * w;
        }
    } else {
        for (int r = lo; r < m; ++r) {
            LD<T> w = 0;
            for (int j = 0; j < n; ++j) w += out[r + std::size_t(j) * m] * v[j], scale = std::max(scale, std::abs(out[r + std::size_t(j) * m]));
            for (int j = 0; j < n; ++j) out[r + std::size_t(j) * m] -= t * w * cj(v[j]);
        }
    }
    return out;
}

// larf alg::Serial on a Thread: Left with a UnitHead whose stored v(0) is NaN (never read), Right
// with a plain v whose v(0) = 1 is stored. C has ld = m + 2, v inc 2.
TYPED_TEST(DeviceBlasV2GebrdTest, LarfSerialUnitHeadAndStoredHead) {
    using T = TypeParam;
    constexpr int m = 9, n = 6, ldc = m + 2, inc = 2, sz = ldc * n + m * inc + n * inc;
    const T tau = val<T>(61);
    auto hd = batched<T>(sz, 62);
    for (int b = 0; b < kBatch; ++b) hd[std::size_t(b) * sz + ldc * n] = nan_of<T>(), hd[std::size_t(b) * sz + ldc * n + m * inc] = T(1);
    const auto h0 = hd;
    T* d = this->dev.up(hd);
    this->dev.template run<32>(kBatch / 32, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const v2::Thread th{};
        T* p = d + std::ptrdiff_t(b) * sz;
        const auto C = v2::mat(p, m, n, ldc);
        v2::larf<Side::Left>(th, tau, v2::unit_head(v2::vec(p + ldc * n, m, inc), 0), C);
        v2::larf<Side::Right>(th, tau, v2::vec(p + ldc * n + m * inc, n, inc), C);
    });
    const auto gd = this->dev.down(d, hd.size());
    int bitdiff = 0;
    for (std::size_t i = 0; i < gd.size(); ++i) bitdiff += !same_bits(gd[i], gd[i % (std::size_t(kPeriod) * sz)]);
    EXPECT_EQ(bitdiff, 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = h0.data() + std::size_t(b) * sz;
        std::vector<LD<T>> vl{LD<T>(1)}, vr;
        for (int r = 1; r < m; ++r) vl.push_back(wide(p[ldc * n + r * inc]));
        for (int j = 0; j < n; ++j) vr.push_back(wide(p[ldc * n + m * inc + j * inc]));
        long double s1 = 0, s2 = 0;
        const auto left = larf_ref<T>(Side::Left, p, ldc, m, n, vl, tau, 0, s1);
        std::vector<T> mid(std::size_t(ldc) * n);
        for (int j = 0; j < n; ++j)
            for (int r = 0; r < m; ++r) mid[r + std::size_t(j) * ldc] = T(left[r + std::size_t(j) * m]);
        const auto both = larf_ref<T>(Side::Right, mid.data(), ldc, m, n, vr, tau, 0, s2);
        const long double sc = std::max(s1, s2) * (1 + std::abs(wide(tau))) * (m + n);
        for (int j = 0; j < n; ++j)
            for (int r = 0; r < m; ++r) worst = std::max(worst, rel(gd[std::size_t(b) * sz + r + j * ldc], both[r + std::size_t(j) * m], sc));
    }
    EXPECT_VERIFY(T, Check::blas, m + n, worst);
}

// larf ColPerLane (Left) then RowPerLane (Right) on a 16 x 16 local tile of an n = 13 item:
// lacpy in through masked_view, apply on the window lo = 4, lacpy out. Columns / rows below lo
// stay bit-identical; a dead partition (odd items with b % 6 == 5) loads zeros and stores nothing.
TYPED_TEST(DeviceBlasV2GebrdTest, LarfLanePerColumnAndRowOnALocalTile) {
    using T = TypeParam;
    constexpr int P = 16, n = 13, ld = n + 3, lo = 4, i0 = 3, WG = 32, sz = ld * n;
    const T tau = val<T>(71);
    auto hd = batched<T>(sz, 72);
    const auto h0 = hd;
    T* d = this->dev.up(hd);
    this->dev.template run<WG>(kBatch / 2, sizeof(T) * 2 * P * P, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const v2::SgpPartition<P> ex{it.get_sub_group()};
        const int part = static_cast<int>(ex.part.get_group_linear_id());
        const int b = static_cast<int>(it.get_group_linear_id()) * 2 + part;
        const bool live = b % 6 != 5;
        const auto Ag = v2::mat(d + std::ptrdiff_t(b) * sz, n, n, ld);
        const auto tile = v2::Mat<T, v2::Dyn, v2::Dyn, v2::Ext<P>, v2::Mem::Local>{reinterpret_cast<T*>(lm) + part * P * P, {n}, {n}, {}};
        v2::lacpy<v2::RowPerLane>(ex, v2::masked_view(Ag, live), tile);
        it.barrier(sycl::access::fence_space::local_space);
        // Left: v = tile(i0:n, 0) (head i0), C = tile(i0:n, :); Right: v = tile(1, i0:n), C = tile(:, i0:n).
        v2::larf<Side::Left, v2::ColPerLane>(ex, tau, v2::unit_head(tile.col(0).tail(i0), 0), tile.sub(i0, 0, n - i0, n), lo);
        it.barrier(sycl::access::fence_space::local_space);
        v2::larf<Side::Right, v2::RowPerLane>(ex, tau, v2::unit_head(tile.row(1).tail(i0), 0), tile.sub(0, i0, n, n - i0), lo);
        it.barrier(sycl::access::fence_space::local_space);
        v2::lacpy<v2::RowPerLane>(ex, tile, Ag, live);
    });
    const auto gd = this->dev.down(d, hd.size());
    int bitdiff = 0, window = 0, dead = 0;
    for (int b = 0; b < kBatch; ++b) {
        const bool live = b % 6 != 5;
        for (int i = 0; i < sz; ++i) {
            if (!live) dead += !same_bits(gd[std::size_t(b) * sz + i], h0[std::size_t(b) * sz + i]);
            else bitdiff += !same_bits(gd[std::size_t(b) * sz + i], gd[std::size_t(b % kPeriod) * sz + i]);
        }
    }
    EXPECT_EQ(dead, 0) << "a dead partition stored";
    EXPECT_EQ(bitdiff, 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = h0.data() + std::size_t(b) * sz;
        std::vector<LD<T>> vl{LD<T>(1)};
        for (int r = i0 + 1; r < n; ++r) vl.push_back(wide(p[r]));
        long double s1 = 0, s2 = 0;
        const auto left = larf_ref<T>(Side::Left, p + i0, ld, n - i0, n, vl, tau, lo, s1);
        std::vector<T> mid(h0.begin() + std::ptrdiff_t(b) * sz, h0.begin() + std::ptrdiff_t(b + 1) * sz);
        for (int j = 0; j < n; ++j)
            for (int r = i0; r < n; ++r) mid[r + std::size_t(j) * ld] = T(left[(r - i0) + std::size_t(j) * (n - i0)]);
        std::vector<LD<T>> vr{LD<T>(1)};
        for (int j = i0 + 1; j < n; ++j) vr.push_back(wide(mid[1 + std::size_t(j) * ld]));
        const auto right = larf_ref<T>(Side::Right, mid.data() + std::size_t(i0) * ld, ld, n, n - i0, vr, tau, lo, s2);
        const long double sc = std::max(s1, s2) * (1 + std::abs(wide(tau))) * 2 * n;
        for (int j = 0; j < n; ++j)
            for (int r = 0; r < n; ++r) {
                const T got = gd[std::size_t(b) * sz + r + std::size_t(j) * ld];
                if (j >= i0 && r >= lo) worst = std::max(worst, rel(got, right[r + std::size_t(j - i0) * n], sc));
                else if (r >= i0 && j >= lo) worst = std::max(worst, rel(got, left[(r - i0) + std::size_t(j) * (n - i0)], sc));
                else window += !same_bits(got, p[r + std::size_t(j) * ld]);  // outside both lane windows
            }
    }
    EXPECT_EQ(window, 0) << "larf wrote outside its lane window";
    EXPECT_VERIFY(T, Check::blas, 2 * n, worst);
}

// lacgv, abs2, bcast / invoke_bcast / select_lane on every executor kind they serve.
TYPED_TEST(DeviceBlasV2GebrdTest, ElementwiseAndBroadcasts) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 64, n = 50, inc = 3;
    auto hx = batched<T>(n * inc, 81);
    std::vector<T> ho(std::size_t(kBatch) * 4, nan_of<T>());
    T* x = this->dev.up(hx);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const v2::SgpPartition<8> pt{it.get_sub_group()};
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
        const R s = v2::reduce(ex, n, [&](int i) { return v2::abs2(xv(i)); });
        ex.barrier(v2::Mem::Global);
        v2::lacgv(ex, xv);
        ex.barrier(v2::Mem::Global);  // xv(0), xv(1) and lane 5's element are read by other items
        const T lead = v2::bcast(ex, ex.leader() ? xv(0) : nan_of<T>());
        const T inv = v2::invoke_bcast(ex, [&]() { return xv(1); });
        const T sel = v2::select_lane(pt, ex.lid() < n ? xv(ex.lid()) : T(0), 5);
        // Every item checks that it saw the same value; item 0 stores the results.
        const int ok = v2::reduce(ex, WG, [&](int) { return int(same_bits(lead, xv(0)) && same_bits(inv, xv(1))); });
        const int ok_sel = v2::reduce(ex, WG, [&](int i) {
            const int src = (i & ~7) + 5;
            return int(same_bits(sel, src < n ? xv(src) : T(0)));
        });
        if (ex.leader()) o[std::ptrdiff_t(b) * 4] = T(s), o[std::ptrdiff_t(b) * 4 + 1] = T(R(ok)), o[std::ptrdiff_t(b) * 4 + 2] = T(R(ok_sel));
    });
    const auto go = this->dev.down(o, ho.size()), gx = this->dev.down(x, hx.size());
    int conj_bad = 0, bc_bad = 0;
    double worst = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* g = go.data() + std::size_t(b) * 4;
        bc_bad += !(v2::Arith<T>::re(g[1]) == R(WG)) + !(v2::Arith<T>::re(g[2]) == R(WG));
        for (int i = 0; i < n * inc; ++i) {
            const T h = hx[std::size_t(b) * n * inc + i];
            conj_bad += !same_bits(gx[std::size_t(b) * n * inc + i], i % inc == 0 ? T(cj(h)) : h);
        }
        if (b >= kPeriod) continue;
        long double ref = 0;
        for (int i = 0; i < n; ++i) ref += std::norm(wide(hx[std::size_t(b) * n * inc + i * inc]));
        worst = std::max(worst, rel(T(g[0]), LD<T>(ref), ref));
    }
    EXPECT_EQ(conj_bad, 0) << "lacgv must conjugate exactly the strided elements";
    EXPECT_EQ(bc_bad, 0) << "a broadcast differed between items";
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

}  // namespace

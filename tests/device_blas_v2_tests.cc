// SPDX-License-Identifier: MIT
// Device LA v2 (include/batchlas/blas/device/v2) against host references.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld / inc and rectangular
// shapes, complex data with nonzero imaginary parts, ConjTrans, beta != 0 and beta == Zero over
// NaN-poisoned outputs, a poisoned unread triangle, batch 1024 with period-3 data where every
// item must be bit-identical to its representative (SLM paths), a WG = 1024 launch, ties and
// NaN for iamax, nrm2 straddling the overflow / underflow thresholds.
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

#include <batchlas/blas/device.hh>
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/dla_v2_partition.hh>
#include <src/kernels/common/dla_v2_potf2.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
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

// Host values per item: kPeriod representatives (optionally edited by fix(item pointer)) copied
// to every item b from representative b % kPeriod, so every item has a known twin.
template <class T, class Fix> std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix, int items = kBatch) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item);
    }
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    return batched<T>(per_item, seed, [](T*) {});
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

// The test's queue and device buffers.
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
    template <int WG, class F> void run(int batch, std::size_t local_elems, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_elems, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * WG, WG),
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

template <class T> class DeviceBlasV2Test : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2Test, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}

// |got - ref| / scale, the blas relative error that EXPECT_VERIFY bounds by 8 (k + 2) eps.
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();  // std::max would drop a NaN
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// ---- level 1 -------------------------------------------------------------------------------

// copy / axpby / axpy / scal / swap / map chained on one Cyclic executor with no barrier;
// x has inc 3, beta == Zero writes over NaN.
TYPED_TEST(DeviceBlasV2Test, Level1ChainComposesWithoutBarriers) {
    using T = TypeParam;
    constexpr int WG = 64, n = 133, inc = 3, ldo = n + 7, nout = 6;
    const T a = val<T>(91), bb = val<T>(92);
    auto hx = batched<T>(n * inc, 1), hy = batched<T>(n, 2);
    std::vector<T> ho(std::size_t(kBatch) * nout * ldo, T(std::numeric_limits<real_of<T>>::quiet_NaN()));
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * n, n);
        const auto om = v2::mat(o + std::ptrdiff_t(b) * nout * ldo, n, nout, ldo);
        v2::copy(ex, yv, om.col(0));
        v2::axpby(ex, v2::Scalar{a}, xv, v2::Scalar{bb}, om.col(0));  // a x + b y
        v2::axpby(ex, a, xv, v2::Zero{}, om.col(1));                    // a x over NaN
        v2::copy(ex, yv, om.col(2));
        v2::axpy(ex, v2::NegOne{}, xv, om.col(2));                      // y - x
        v2::copy(ex, xv, om.col(3));
        v2::scal(ex, a, om.col(3));                                     // a x
        v2::copy(ex, xv, om.col(4));
        v2::copy(ex, yv, om.col(5));
        v2::swap(ex, om.col(4), om.col(5));
        v2::map(ex, om.col(5), [=](T u, T w) { return v2::Arith<T>::add(u, v2::Arith<T>::mul(a, w)); }, om.col(5), yv);
    });
    const auto got = this->dev.down(o, ho.size());
    double worst = 0;
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* g = got.data() + std::size_t(b) * nout * ldo;
        const T* g0 = got.data() + std::size_t(b % kPeriod) * nout * ldo;
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < nout; ++k) bitdiff += !same_bits(g[k * ldo + i], g0[k * ldo + i]);
        if (b >= kPeriod) continue;
        for (int i = 0; i < n; ++i) {
            const LD<T> xi = wide(hx[std::size_t(b) * n * inc + i * inc]), yi = wide(hy[std::size_t(b) * n + i]);
            const LD<T> A = wide(a), B = wide(bb);
            const long double s = std::abs(A) * std::abs(xi) + std::abs(B) * std::abs(yi) + std::abs(yi) + std::abs(xi);
            worst = std::max({worst, rel(g[i], A * xi + B * yi, s), rel(g[ldo + i], A * xi, s), rel(g[2 * ldo + i], yi - xi, s),
                              rel(g[3 * ldo + i], A * xi, s), rel(g[4 * ldo + i], yi, s), rel(g[5 * ldo + i], xi + A * yi, s)});
        }
    }
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

// ---- reductions ----------------------------------------------------------------------------

enum RedSlot { kDot, kDotc, kAsum, kAmax, kNrm2, kNrm2U, kDotN0, kDotN1, kDotN2, kDotNs0, kDotNs1, kDotNs2, kReduce, kIamax, kSlots };

// Every reduction of item b; Ex is built from the nd_item. Split launches three kernels, each
// compiling only its part (a WG = 1024 launch has 64 registers per item).
template <class T, int WG, bool Split = false, class MakeEx>
void run_reductions(Dev& dev, int n, int inc, std::vector<T>& hx, std::vector<T>& hy, std::vector<T>& out, MakeEx make_ex,
                    int items_per_wg = 1, int item_stride_lanes = WG) {
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    std::vector<T> ho(std::size_t(kBatch) * kSlots, T(std::numeric_limits<real_of<T>>::quiet_NaN()));
    T* o = dev.up(ho);
    const int groups = kBatch / items_per_wg;
    auto launch = [&](auto part_c) {
        constexpr int P = decltype(part_c)::value;
        dev.template run<WG>(groups, sizeof(T) * 3 * 64, [=](sycl::nd_item<1> it, unsigned char* lm) {
            const int b = static_cast<int>(it.get_group_linear_id()) * items_per_wg +
                          static_cast<int>(it.get_local_linear_id()) / item_stride_lanes;
            const auto ex = make_ex(it);
            const auto xv = v2::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
            const auto yv = v2::vec(y + std::ptrdiff_t(b) * n, n);
            T* ob = o + std::ptrdiff_t(b) * kSlots;
            if constexpr (P < 0 || P == 0) {
                const T r0 = v2::dot(ex, xv, yv), r1 = v2::dotc(ex, xv, yv), r2 = T(v2::asum(ex, xv)), r3 = T(v2::amax(ex, xv));
                const T r4 = v2::reduce(ex, n, [&](int i) { return v2::Arith<T>::mul(xv(i), xv(i)); });
                if (ex.leader()) ob[kDot] = r0, ob[kDotc] = r1, ob[kAsum] = r2, ob[kAmax] = r3, ob[kReduce] = r4;
            }
            if constexpr (P < 0 || P == 1) {
                const T r0 = T(v2::nrm2(ex, xv)), r1 = T(v2::nrm2<v2::alg::Unscaled>(ex, xv)), r2 = T(real_of<T>(v2::iamax(ex, xv)));
                if (ex.leader()) ob[kNrm2] = r0, ob[kNrm2U] = r1, ob[kIamax] = r2;
            }
            if constexpr (P < 0 || P == 2) {
                const auto xs = std::array<decltype(yv), 3>{yv, v2::vec(x + std::ptrdiff_t(b) * n * inc, n), yv.tail(0)};
                const auto d3 = v2::dotN<3, true>(ex, xs, yv);
                const auto s3 = v2::dotN<3, true>(ex, v2::Scratch<T, 3 * 32>{reinterpret_cast<T*>(lm)}, xs, yv);
                if (ex.leader())
                    for (int k = 0; k < 3; ++k) ob[kDotN0 + k] = d3[k], ob[kDotNs0 + k] = s3[k];
            }
        });
    };
    if constexpr (Split) {
        launch(std::integral_constant<int, 0>{});
        launch(std::integral_constant<int, 1>{});
        launch(std::integral_constant<int, 2>{});
    } else {
        launch(std::integral_constant<int, -1>{});
    }
    out = dev.down(o, ho.size());
}

template <class T> void check_reductions(int n, int inc, const std::vector<T>& hx, const std::vector<T>& hy, const std::vector<T>& out,
                                         const char* where) {
    using R = real_of<T>;
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int k = 0; k < kSlots; ++k) bitdiff += !same_bits(out[std::size_t(b) * kSlots + k], out[std::size_t(b % kPeriod) * kSlots + k]);
    EXPECT_EQ(bitdiff, 0) << where << ": items differ from their representative";
    for (int b = 0; b < kPeriod; ++b) {
        const T* r = out.data() + std::size_t(b) * kSlots;
        LD<T> dot = 0, dotc = 0, d1 = 0, sq = 0;
        long double sabs = 0, asum = 0, amax = -1, ssq = 0, s1 = 0;
        int imax = -1;
        for (int i = 0; i < n; ++i) {
            const LD<T> xi = wide(hx[std::size_t(b) * n * inc + i * inc]), yi = wide(hy[std::size_t(b) * n + i]);
            const LD<T> x1 = wide(hx[std::size_t(b) * n * inc + i]);
            dot += xi * yi, dotc += conj_h(xi) * yi, d1 += conj_h(x1) * yi, sq += xi * xi;
            sabs += std::abs(xi) * std::abs(yi), s1 += std::abs(x1) * std::abs(yi), ssq += std::norm(xi);
            const long double a1 = std::abs(std::real(xi)) + std::abs(std::imag(xi));
            asum += a1;
            if (a1 > amax) amax = a1, imax = i;
        }
        const long double nrm = std::sqrt(ssq);
        long double yy = 0;
        for (int i = 0; i < n; ++i) yy += std::norm(wide(hy[std::size_t(b) * n + i]));
        const double e = std::max({rel(r[kDot], dot, sabs), rel(r[kDotc], dotc, sabs), rel(r[kReduce], sq, ssq),
                                   rel(r[kDotN0], LD<T>(yy), yy), rel(r[kDotN1], d1, s1), rel(r[kDotN2], LD<T>(yy), yy),
                                   rel(r[kDotNs0], LD<T>(yy), yy), rel(r[kDotNs1], d1, s1), rel(r[kDotNs2], LD<T>(yy), yy)});
        EXPECT_VERIFY(T, Check::blas, n, e) << where << " item " << b << " (dot, dotc, reduce, dotN, packed dotN)";
        EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kAsum]), LD<T>(asum), asum)) << where << " asum";
        EXPECT_EQ(std::real(r[kAmax]), R(amax)) << where << " amax is exact";
        EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kNrm2]), LD<T>(nrm), nrm)) << where << " nrm2";
        EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kNrm2U]), LD<T>(nrm), nrm)) << where << " nrm2 unscaled";
        EXPECT_EQ(int(std::real(r[kIamax])), imax) << where << " iamax";
    }
}

TYPED_TEST(DeviceBlasV2Test, ReductionsMatchHostOnEveryExecutor) {
    using T = TypeParam;
    const int n = 777, inc = 2;
    auto hx = batched<T>(n * inc, 11), hy = batched<T>(n, 12);
    std::vector<T> out;
    run_reductions<T, 256>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return v2::WorkGroup<256>{it}; });
    check_reductions(n, inc, hx, hy, out, "WorkGroup<256>");
    run_reductions<T, 64>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return v2::WorkGroup<64>{it}; });
    check_reductions(n, inc, hx, hy, out, "WorkGroup<64>");
    // The ceiling launch: 1024 items per work-group (the 4090's and the SYCL limit we target).
    run_reductions<T, 1024, true>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return v2::WorkGroup<1024>{it}; });
    check_reductions(n, inc, hx, hy, out, "WorkGroup<1024>");
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return v2::SubGroup<32>{it.get_sub_group()}; });
    check_reductions(n, inc, hx, hy, out, "SubGroup<32>");
    // 4 partitions of 8 lanes per sub-group, one item each; then the masked src-side binding.
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return v2::Partition<8>{it.get_sub_group()}; }, 4, 8);
    check_reductions(n, inc, hx, hy, out, "Partition<8>");
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out,
                          [](sycl::nd_item<1> it) { return v2::SgpPartition<8, true>{it.get_sub_group()}; }, 4, 8);
    check_reductions(n, inc, hx, hy, out, "SgpPartition<8, Masked>");
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1>) { return v2::Thread{}; }, 32, 1);
    check_reductions(n, inc, hx, hy, out, "Thread");
    // Runtime-width executors: a 96-item work-group (not a power of two), and one per sub-group.
    run_reductions<T, 96>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return v2::RtGroup<>(it); });
    check_reductions(n, inc, hx, hy, out, "RtGroup<Group> @96");
    run_reductions<T, 64>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return v2::RtGroup<v2::Scope::Sub>(it); }, 2, 32);
    check_reductions(n, inc, hx, hy, out, "RtGroup<Sub>");
}

// iamax: lowest index on ties (the tie straddles two owners), NaN wins at its lowest index.
// amax: NaN propagates. nrm2: straddles overflow and underflow of the unscaled sum.
TYPED_TEST(DeviceBlasV2Test, IamaxTiesNanAndNrm2Thresholds) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 128, n = 300, nitem = 4;
    const R big = std::sqrt(std::numeric_limits<R>::max()) * R(4), tiny = std::ldexp(std::sqrt(std::numeric_limits<R>::min()), -12);
    std::vector<T> hx(std::size_t(nitem) * n);
    for (int i = 0; i < n; ++i) {
        hx[i] = val<T>(i) * R(0.5);
        hx[n + i] = val<T>(i) * R(0.5);
        hx[2 * n + i] = val<T>(i) * big;
        hx[3 * n + i] = val<T>(i) * tiny;
    }
    hx[37] = hx[37 + WG] = hx[37 + WG + 3] = T(R(2));  // tie on one owner (37, 165) and across owners (168)
    hx[n + 250] = hx[n + 20] = T(std::numeric_limits<R>::quiet_NaN());
    hx[n + 7] = T(R(9));
    T* x = this->dev.up(hx);
    std::vector<int> hi(nitem * 2, -7);
    std::vector<R> hr(nitem * 2, R(-1));
    int* io = this->dev.up(hi);
    R* ro = this->dev.up(hr);
    this->dev.template run<WG>(nitem, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * n, n);
        const int im = v2::iamax(ex, xv);
        const int im_sub = v2::iamax(v2::SubGroup<32>{it.get_sub_group()}, xv);
        const R am = v2::amax(ex, xv), nr = v2::nrm2(ex, xv), nu = v2::nrm2<v2::alg::Unscaled>(ex, xv);
        if (ex.leader()) io[2 * b] = im, io[2 * b + 1] = im_sub, ro[2 * b] = am, ro[2 * b + 1] = nr;
        if (ex.leader() && b >= 2) ro[2 * b] = nu;
    });
    const auto gi = this->dev.down(io, hi.size());
    const auto gr = this->dev.down(ro, hr.size());
    EXPECT_EQ(gi[0], 37) << "tie: lowest index";
    EXPECT_EQ(gi[1], 37) << "tie: lowest index (Scope::Sub)";
    EXPECT_EQ(gi[2], 20) << "NaN: lowest NaN index";
    EXPECT_EQ(gi[3], 20);
    EXPECT_TRUE(std::isnan(gr[2])) << "amax propagates NaN";
    for (int b = 2; b < 4; ++b) {
        long double s = 0;
        const long double scale = b == 2 ? big : tiny;
        for (int i = 0; i < n; ++i) s += std::norm(LD<T>(hx[std::size_t(b) * n + i]) / scale);
        const long double ref = std::sqrt(s) * scale;
        EXPECT_VERIFY(T, Check::blas, n, double(std::abs(gr[2 * b + 1] - ref) / ref)) << "scaled nrm2, item " << b;  // NaN fails
        // The unscaled form must fail here, or the item does not straddle the threshold.
        EXPECT_FALSE(batchlas::verify::within<T>(Check::blas, n, double(std::abs(gr[2 * b] - ref) / ref)))
            << "unscaled nrm2 survived item " << b << ": the threshold is not straddled";
    }
}

// ---- level 2 -------------------------------------------------------------------------------

template <class T> LD<T> op_elem(const std::vector<T>& a, std::size_t base, int ld, int r, int c, Transpose tr) {
    if (tr == Transpose::NoTrans) return wide(a[base + r + std::size_t(c) * ld]);
    const LD<T> v = wide(a[base + c + std::size_t(r) * ld]);
    return tr == Transpose::ConjTrans ? conj_h(v) : v;
}

// y := alpha op(A) x + beta y for each executor and algorithm; A is m x n with ld = m + 3, x
// has inc 2, and BetaZero runs over a NaN-poisoned y.
template <Transpose TR, class Alg, int WG, bool BetaZero, class T, class MakeEx>
void gemv_case(Dev& dev, MakeEx make_ex, int items_per_wg, int lanes_per_item, const char* where) {
    constexpr int m = 37, n = 29, ld = m + 3, inc = 2;
    const int ylen = TR == Transpose::NoTrans ? m : n, xlen = TR == Transpose::NoTrans ? n : m;
    const T alpha = val<T>(5), beta = val<T>(6);
    auto ha = batched<T>(ld * n, 21), hx = batched<T>(xlen * inc, 22), hy = batched<T>(ylen, 23);
    if (BetaZero) std::fill(hy.begin(), hy.end(), T(std::numeric_limits<real_of<T>>::quiet_NaN()));
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    dev.template run<WG>(kBatch / items_per_wg, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id()) * items_per_wg +
                      static_cast<int>(it.get_local_linear_id()) / lanes_per_item;
        const auto ex = make_ex(it);
        const auto av = v2::mat(a + std::ptrdiff_t(b) * ld * n, m, n, ld);
        const auto xv = v2::assume_published(v2::vec(x + std::ptrdiff_t(b) * xlen * inc, xlen, inc));
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * ylen, ylen);
        if constexpr (BetaZero) v2::gemv<TR, Alg>(ex, v2::Scalar{alpha}, av, xv, v2::Zero{}, yv);
        else v2::gemv<TR, Alg>(ex, alpha, av, xv, beta, yv);
    });
    const auto got = dev.down(y, hy.size());
    int bitdiff = 0;
    for (std::size_t i = 0; i < got.size(); ++i) bitdiff += !same_bits(got[i], got[i % (std::size_t(kPeriod) * ylen)]);
    EXPECT_EQ(bitdiff, 0) << where;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < ylen; ++r) {
            LD<T> acc = 0;
            long double s = 0;
            for (int c = 0; c < xlen; ++c) {
                const LD<T> e = op_elem(ha, std::size_t(b) * ld * n, ld, r, c, TR);
                const LD<T> xc = wide(hx[std::size_t(b) * xlen * inc + c * inc]);
                acc += e * xc, s += std::abs(e) * std::abs(xc);
            }
            const LD<T> y0 = wide(hy[std::size_t(b) * ylen + r]);
            const LD<T> ref = wide(alpha) * acc + (BetaZero ? LD<T>(0) : wide(beta) * y0);
            const long double sc = std::abs(wide(alpha)) * s + (BetaZero ? 0 : std::abs(wide(beta)) * std::abs(y0));
            worst = std::max(worst, rel(got[std::size_t(b) * ylen + r], ref, sc));
        }
    EXPECT_VERIFY(T, Check::blas, xlen, worst) << where;
}

TYPED_TEST(DeviceBlasV2Test, GemvEveryTransposeExecutorAndAlgorithm) {
    using T = TypeParam;
    auto wg = [](sycl::nd_item<1> it) { return v2::WorkGroup<128>{it}; };
    auto sg = [](sycl::nd_item<1> it) { return v2::SubGroup<32>{it.get_sub_group()}; };
    auto pt = [](sycl::nd_item<1> it) { return v2::Partition<16>{it.get_sub_group()}; };
    auto th = [](sycl::nd_item<1>) { return v2::Thread{}; };
    auto tm = [](sycl::nd_item<1> it) { return v2::Teams<128>{it}; };
    constexpr auto N = Transpose::NoTrans, Tt = Transpose::Trans, C = Transpose::ConjTrans;
    gemv_case<N, void, 128, false, T>(this->dev, wg, 1, 128, "N WorkGroup RowOwner");
    gemv_case<N, void, 128, true, T>(this->dev, wg, 1, 128, "N WorkGroup RowOwner beta=Zero");
    gemv_case<Tt, void, 128, false, T>(this->dev, wg, 1, 128, "T WorkGroup SgPerCol");
    gemv_case<C, void, 128, false, T>(this->dev, wg, 1, 128, "C WorkGroup SgPerCol");
    gemv_case<C, void, 128, true, T>(this->dev, wg, 1, 128, "C WorkGroup SgPerCol beta=Zero");
    gemv_case<C, void, 128, false, T>(this->dev, tm, 1, 128, "C Teams SgPerCol");
    gemv_case<C, v2::alg::GroupReducePerOutput, 128, false, T>(this->dev, wg, 1, 128, "C WorkGroup GroupReducePerOutput");
    gemv_case<N, void, 32, false, T>(this->dev, sg, 1, 32, "N SubGroup");
    gemv_case<C, void, 32, false, T>(this->dev, sg, 1, 32, "C SubGroup");
    gemv_case<N, void, 32, false, T>(this->dev, pt, 2, 16, "N Partition<16>");
    gemv_case<C, void, 32, true, T>(this->dev, pt, 2, 16, "C Partition<16> beta=Zero");
    gemv_case<N, void, 32, false, T>(this->dev, th, 32, 1, "N Thread Serial");
    gemv_case<C, void, 32, false, T>(this->dev, th, 32, 1, "C Thread Serial");
    auto rt = [](sycl::nd_item<1> it) { return v2::RtGroup<>(it); };
    auto tm16 = [](sycl::nd_item<1> it) { return v2::Teams<128, 16>{it}; };
    gemv_case<N, void, 96, false, T>(this->dev, rt, 1, 96, "N RtGroup @96 RowOwner");
    gemv_case<C, void, 96, false, T>(this->dev, rt, 1, 96, "C RtGroup @96 SgPerCol");
    gemv_case<Tt, void, 128, true, T>(this->dev, tm16, 1, 128, "T Teams<128, 16> SgPerCol beta=Zero");
}

// symv/hemv Lower with x staged in local memory and published; the strict upper triangle is
// poisoned with a large finite value and, for hemv, Im(diagonal) too: neither may be read.
template <bool Herm, int WG, class T> void symv_case(Dev& dev, int n) {
    using R = real_of<T>;
    const int ld = n + 5;
    const T alpha = val<T>(7), beta = val<T>(8);
    // A is read-only: the 1024 work-groups share the kPeriod representatives (item b reads b % kPeriod).
    auto ha = batched<T>(ld * n, 31, [&](T* item) {
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < c; ++r) item[r + std::size_t(c) * ld] = T(R(1e6));
            if constexpr (is_cplx<T>)
                if (Herm) item[c + std::size_t(c) * ld].imag(R(1e6));
        }
    }, kPeriod);
    auto hx = batched<T>(n, 32), hy = batched<T>(n, 33);
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    dev.template run<WG>(kBatch, sizeof(T) * n, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const auto xl = v2::vec<v2::Mem::Local>(reinterpret_cast<T*>(lm), n);
        v2::copy(ex, v2::vec(x + std::ptrdiff_t(b) * n, n), xl);
        const auto av = v2::mat(a + std::ptrdiff_t(b % kPeriod) * ld * n, n, n, ld);
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * n, n);
        if constexpr (Herm) v2::hemv<Uplo::Lower>(ex, alpha, av, ex.publish(xl), beta, yv);
        else v2::symv<Uplo::Lower>(ex, alpha, av, ex.publish(xl), beta, yv);
    });
    const auto got = dev.down(y, hy.size());
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int r = 0; r < n; ++r) bitdiff += !same_bits(got[std::size_t(b) * n + r], got[std::size_t(b % kPeriod) * n + r]);
    EXPECT_EQ(bitdiff, 0) << (Herm ? "hemv" : "symv") << " WG=" << WG << " n=" << n;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < n; ++r) {
            LD<T> acc = 0;
            long double s = 0;
            for (int c = 0; c < n; ++c) {
                LD<T> e = r >= c ? wide(ha[std::size_t(b) * ld * n + r + std::size_t(c) * ld]) : wide(ha[std::size_t(b) * ld * n + c + std::size_t(r) * ld]);
                if (Herm && r < c) e = conj_h(e);
                if (Herm && r == c) e = LD<T>(std::real(e));
                const LD<T> xc = wide(hx[std::size_t(b) * n + c]);
                acc += e * xc, s += std::abs(e) * std::abs(xc);
            }
            const LD<T> y0 = wide(hy[std::size_t(b) * n + r]);
            worst = std::max(worst, rel(got[std::size_t(b) * n + r], wide(alpha) * acc + wide(beta) * y0,
                                        std::abs(wide(alpha)) * s + std::abs(wide(beta)) * std::abs(y0)));
        }
    EXPECT_VERIFY(T, Check::blas, n, worst) << (Herm ? "hemv" : "symv") << " WG=" << WG << " n=" << n;
}

TYPED_TEST(DeviceBlasV2Test, SymvHemvLowerReadOnlyTheStoredTriangle) {
    using T = TypeParam;
    symv_case<false, 64, T>(this->dev, 45);
    symv_case<true, 64, T>(this->dev, 45);
    symv_case<false, 256, T>(this->dev, 300);  // n > WG: rows r and r + WG on one item
    symv_case<true, 256, T>(this->dev, 300);
}

// A += alpha x y^T (geru) / alpha x y^H (gerc); A is m x n with ld = m + 3, y has inc 3.
template <bool Conj, class T> void ger_case(Dev& dev) {
    constexpr int WG = 64, m = 77, n = 23, ld = m + 3, inc = 3;
    const T alpha = val<T>(9);
    auto ha = batched<T>(ld * n, 41), hx = batched<T>(m, 42), hy = batched<T>(n * inc, 43);
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const auto av = v2::mat(a + std::ptrdiff_t(b) * ld * n, m, n, ld);
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * m, m);
        const auto yv = v2::assume_published(v2::vec(y + std::ptrdiff_t(b) * n * inc, n, inc));
        if constexpr (Conj) v2::gerc(ex, alpha, xv, yv, av);
        else v2::geru(ex, alpha, xv, yv, av);
    });
    const auto got = dev.down(a, ha.size());
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < ld; ++r) {
                const std::size_t k = std::size_t(b) * ld * n + r + std::size_t(c) * ld;
                if (r >= m) {
                    EXPECT_TRUE(same_bits(got[k], ha[k])) << "ger wrote the ld gap";
                    continue;
                }
                const LD<T> yc = wide(hy[std::size_t(b) * n * inc + c * inc]), xr = wide(hx[std::size_t(b) * m + r]);
                const LD<T> ref = wide(ha[k]) + wide(alpha) * xr * (Conj ? conj_h(yc) : yc);
                worst = std::max(worst, rel(got[k], ref, std::abs(wide(ha[k])) + std::abs(wide(alpha)) * std::abs(xr) * std::abs(yc)));
            }
    EXPECT_VERIFY(T, Check::blas, 1, worst) << (Conj ? "gerc" : "geru");
}

TYPED_TEST(DeviceBlasV2Test, GeruGercCyclic) {
    ger_case<false, TypeParam>(this->dev);
    ger_case<true, TypeParam>(this->dev);
}

// The worked latrd step (W1) on v2, complex-aware: w := A v (hemv), then for p < K
// w -= U_p (W_p^H v) + W_p (U_p^H v) with U_p = unit_head(V_p, p) taken from ONE packed dotN<2>
// exchange per p (dot_many); w *= tau; w += (-tau/2 v^H w) v. Ops chain on one Cyclic layout: the only
// barriers are publish(v) and the dotN exchange.
TYPED_TEST(DeviceBlasV2Test, LatrdStepComposesOnOneLayout) {
    using T = TypeParam;
    using A = v2::Arith<T>;
    constexpr int WG = 128, n = 150, ld = n + 2, K = 4;
    const T tau = val<T>(77);
    auto ha = batched<T>(ld * n, 51), hv = batched<T>(n, 52), hu = batched<T>(ld * K, 53), hw = batched<T>(ld * K, 54);
    std::vector<T> ho(std::size_t(kBatch) * n, T(0));
    T* a = this->dev.up(ha);
    T* v = this->dev.up(hv);
    T* u = this->dev.up(hu);
    T* w = this->dev.up(hw);
    T* o = this->dev.up(ho);
    constexpr int kScr = v2::scratch_elements<v2::op::ReduceN<2>, v2::WorkGroup<WG>, T>();
    this->dev.template run<WG>(kBatch, sizeof(T) * (2 * n + kScr), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        T* l = reinterpret_cast<T*>(lm);
        const auto vl = v2::vec<v2::Mem::Local>(l, n), wl = v2::vec<v2::Mem::Local>(l + n, n);
        const v2::Scratch<T, kScr> scr{l + 2 * n};
        const auto Am = v2::mat(a + std::ptrdiff_t(b) * ld * n, n, n, ld);
        const auto Um = v2::mat(u + std::ptrdiff_t(b) * ld * K, n, K, ld), Wm = v2::mat(w + std::ptrdiff_t(b) * ld * K, n, K, ld);
        v2::copy(ex, v2::vec(v + std::ptrdiff_t(b) * n, n), vl);
        v2::hemv<Uplo::Lower>(ex, v2::One{}, Am, ex.publish(vl), v2::Zero{}, wl);
        for (int p = 0; p < K; ++p) {
            const auto up = v2::unit_head(Um.col(p), p);
            const auto gd = v2::dot_many<true>(ex, scr, vl, Wm.col(p), up);  // one exchange: W_p^H v, U_p^H v
            const T gam = gd[0], del = gd[1];
            v2::map(ex, wl, [=](T wr, T ur, T wpr) { return A::sub(wr, A::add(A::mul(ur, gam), A::mul(wpr, del))); }, wl, up, Wm.col(p));
        }
        v2::scal(ex, tau, wl);
        const T d = v2::dotc(ex, vl, wl);
        v2::axpy(ex, A::mul(T(real_of<T>(-0.5)), A::mul(tau, d)), vl, wl);
        v2::copy(ex, wl, v2::vec(o + std::ptrdiff_t(b) * n, n));
    });
    const auto got = this->dev.down(o, ho.size());
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int r = 0; r < n; ++r) bitdiff += !same_bits(got[std::size_t(b) * n + r], got[std::size_t(b % kPeriod) * n + r]);
    EXPECT_EQ(bitdiff, 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        auto at = [&](const std::vector<T>& h, int ldh, int stride, int r, int c) { return wide(h[std::size_t(b) * stride + r + std::size_t(c) * ldh]); };
        std::vector<LD<T>> vv(n), ww(n, 0);
        long double norm = 0;
        for (int r = 0; r < n; ++r) vv[r] = wide(hv[std::size_t(b) * n + r]);
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                LD<T> e = r > c ? at(ha, ld, ld * n, r, c) : conj_h(at(ha, ld, ld * n, c, r));
                if (r == c) e = LD<T>(std::real(at(ha, ld, ld * n, r, r)));
                ww[r] += e * vv[c];
                norm += std::abs(e) * std::abs(vv[c]);
            }
        for (int p = 0; p < K; ++p) {
            auto upr = [&](int r) { return r == p ? LD<T>(1) : (r > p ? at(hu, ld, ld * K, r, p) : LD<T>(0)); };
            LD<T> gam = 0, del = 0;
            for (int r = 0; r < n; ++r) gam += conj_h(at(hw, ld, ld * K, r, p)) * vv[r], del += conj_h(upr(r)) * vv[r];
            for (int r = 0; r < n; ++r) ww[r] -= upr(r) * gam + at(hw, ld, ld * K, r, p) * del;
        }
        for (auto& e : ww) e *= wide(tau);
        LD<T> d = 0;
        for (int r = 0; r < n; ++r) d += conj_h(vv[r]) * ww[r];
        for (int r = 0; r < n; ++r) ww[r] += LD<T>(-0.5) * wide(tau) * d * vv[r];
        for (int r = 0; r < n; ++r) worst = std::max(worst, rel(got[std::size_t(b) * n + r], ww[r], norm));
    }
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// The strict-port ops (latrd P3) give the hand kernel's bits: hemv<RowOwnerSplitStoredDiag,
// ExplicitInline> on Trailing operands (a v1 KernelMatrixView and a v2 Mat), scratch-less
// dot_many<Scope::Group, Native>, Native scal / dotc / axpy, offset views (col / tail) and the
// ranged for_each_owned, each against the legacy latrd expression computed in the same kernel.
// Im(A(r,r)) != 0 is read as stored, so the default hemv (which masks it) must differ: the
// control that the bit compare can fail. A WG = 1024 launch covers the ceiling.
template <int WG, class T> void strict_port_case(Dev& dev) {
    using R = real_of<T>;
    using v2::Mem;
    using v2::Scope;
    using v2::arith::Native;
    constexpr int n = 150, ld = n + 3, k0 = 3, P = 3, nout = 6 * n + 4 * P + 2;
    auto ha = batched<T>(ld * n, 81, [&](T* item) {
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < c; ++r) item[r + std::size_t(c) * ld] = T(R(1e6));  // unread upper triangle
    }, kPeriod);
    auto hx = batched<T>(n, 82), hw = batched<T>(ld * P, 83);
    std::vector<T> ho(std::size_t(kBatch) * nout, T(0));
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* w = dev.up(hw);
    T* o = dev.up(ho);
    const KernelMatrixView<T, MatrixFormat::Dense> kv(a, n, n, ld, ld * n, kPeriod);
    const T tau = val<T>(84);
    dev.template run<WG>(kBatch, sizeof(T) * 2 * n, [=](sycl::nd_item<1> it, unsigned char* lm) {
        namespace dd = batchlas::device::detail;
        const int b = static_cast<int>(it.get_group_linear_id());
        const int lid = static_cast<int>(it.get_local_linear_id());
        const v2::WorkGroup<WG> ex{it};
        T* const lv = reinterpret_cast<T*>(lm);
        T* const lw = lv + n;
        T* const ob = o + std::ptrdiff_t(b) * nout;
        const auto Ab = kv.batch_item(b % kPeriod);
        const auto A = v2::mat(Ab.data(), n, n, ld);
        const auto W = v2::mat(w + std::ptrdiff_t(b) * ld * P, n, P, ld);
        const auto vfull = v2::vec<Mem::Local>(lv, n);
        // Ranged for_each_owned fills v(k0:) with the latrd unit-head form; v2::copy then reads
        // it back on tail(k0) with no barrier (same owners).
        v2::for_each_owned(ex, k0, n, [&](int r) { vfull(r) = (r == k0) ? T(1) : x[std::ptrdiff_t(b) * n + r]; });
        v2::copy(ex, vfull.tail(k0), v2::vec(ob + 5 * n + k0, n - k0));
        // hemv: hand (legacy latrd loop), v2 on KernelMatrixView, v2 on a v2 Mat, default hemv.
        ex.barrier(Mem::Local);
        for (int r = k0 + lid; r < n; r += WG) {
            const int c_split = sycl::min(r, n - 1);
            T acc = T(0);
            for (int c = k0; c <= c_split; ++c) v2::ArithInline<T>::mac(acc, Ab(r, c), lv[c]);
            for (int c = c_split + 1; c < n; ++c) v2::ArithInline<T>::mac_conj(acc, Ab(c, r), lv[c]);
            ob[r] = acc;
        }
        const auto vp = v2::assume_published(v2::trailing(vfull, k0));
        v2::hemv<Uplo::Lower, v2::alg::RowOwnerSplitStoredDiag, v2::arith::ExplicitInline>(
            ex, v2::One{}, v2::trailing(Ab, k0, n), vp, v2::Zero{}, v2::trailing(v2::vec(ob + n, n), k0));
        v2::hemv<Uplo::Lower, v2::alg::RowOwnerSplitStoredDiag, v2::arith::ExplicitInline>(
            ex, v2::One{}, v2::trailing(A, k0), vp, v2::Zero{}, v2::trailing(v2::vec(ob + 2 * n, n), k0));
        v2::hemv<Uplo::Lower>(ex, v2::One{}, A.tail(k0), v2::assume_published(vfull.tail(k0)), v2::Zero{}, v2::vec(ob + 3 * n + k0, n - k0));
        // gamma/delta: hand two-accumulator loop + two group reductions vs dot_many.
        const auto v = vfull.tail(k0);
        for (int p = 0; p < P; ++p) {
            T gp = T(0), dp = T(0);
            for (int c = k0 + lid; c < n; c += WG) {
                const T vc = lv[c];
                gp += dd::conj(W(c, p)) * vc;
                const T vcp = (c == p + 1) ? T(1) : ((c > p + 1) ? Ab(c, p) : T(0));
                dp += dd::conj(vcp) * vc;
            }
            const T g0 = dd::reduce_sum_group(it.get_group(), gp), d0 = dd::reduce_sum_group(it.get_group(), dp);
            const auto gd = v2::dot_many<true, Scope::Group, Native>(ex, v, W.col(p).tail(k0), v2::unit_head(A.col(p), p + 1).tail(k0));
            if (ex.leader()) ob[6 * n + 4 * p] = g0, ob[6 * n + 4 * p + 1] = gd[0], ob[6 * n + 4 * p + 2] = d0, ob[6 * n + 4 * p + 3] = gd[1];
        }
        // scal / dotc / axpy (Native) vs the hand std:: expressions, on w = hand hemv result.
        for (int r = k0 + lid; r < n; r += WG) lw[r] = ob[r];
        const auto wv = v2::vec<Mem::Local>(lw, n).tail(k0);
        T hand_dot_p = T(0);
        for (int r = k0 + lid; r < n; r += WG) hand_dot_p += dd::conj(lv[r]) * (ob[r] * tau);
        const T hand_dot = dd::reduce_sum_group(it.get_group(), hand_dot_p);
        v2::scal<Native>(ex, tau, wv);
        const T dot = v2::dotc<Scope::Group, Native>(ex, v, wv);
        v2::axpy<Native>(ex, T(-0.5) * tau * dot, v, wv);
        for (int r = k0 + lid; r < n; r += WG) {
            T h = ob[r];
            h *= tau;
            h += (T(-0.5) * tau * hand_dot) * lv[r];
            ob[4 * n + r] = h;  // hand
        }
        v2::copy(ex, wv, v2::vec(lw, n).tail(k0));
        if (ex.leader()) ob[6 * n + 4 * P] = hand_dot, ob[6 * n + 4 * P + 1] = dot;
        // Slot 5: the ranged fill read back through tail(k0) must equal v; then the v2 L1 result.
        for (int r = k0 + lid; r < n; r += WG) ob[5 * n + r] = (ob[5 * n + r] == lv[r]) ? lw[r] : T(R(NAN));
    });
    const auto got = dev.down(o, ho.size());
    int bitdiff = 0, hemv_v1 = 0, hemv_mat = 0, hemv_masked_same = 0, gd = 0, l1 = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* g = got.data() + std::size_t(b) * nout;
        const T* g0 = got.data() + std::size_t(b % kPeriod) * nout;
        for (int i = 0; i < nout; ++i) bitdiff += !same_bits(g[i], g0[i]);
        for (int r = k0; r < n; ++r) {
            hemv_v1 += !same_bits(g[n + r], g[r]);
            hemv_mat += !same_bits(g[2 * n + r], g[r]);
            hemv_masked_same += same_bits(g[3 * n + r], g[r]);
            l1 += !same_bits(g[5 * n + r], g[4 * n + r]);
        }
        for (int p = 0; p < P; ++p) gd += !same_bits(g[6 * n + 4 * p], g[6 * n + 4 * p + 1]) + !same_bits(g[6 * n + 4 * p + 2], g[6 * n + 4 * p + 3]);
        gd += !same_bits(g[6 * n + 4 * P], g[6 * n + 4 * P + 1]);
    }
    EXPECT_EQ(bitdiff, 0) << "WG=" << WG;
    EXPECT_EQ(hemv_v1, 0) << "hemv StoredDiag on KernelMatrixView vs hand, WG=" << WG;
    EXPECT_EQ(hemv_mat, 0) << "hemv StoredDiag on v2 Mat vs hand, WG=" << WG;
    if constexpr (is_cplx<T>) EXPECT_LT(hemv_masked_same, kBatch * (n - k0)) << "default hemv must mask Im(A(r,r))";
    EXPECT_EQ(gd, 0) << "dot_many / dotc (Native, Group) vs reduce_sum_group, WG=" << WG;
    EXPECT_EQ(l1, 0) << "scal / axpy Native vs the hand expressions, WG=" << WG;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = k0; r < n; ++r) {
            LD<T> acc = 0;
            long double s = 0;
            for (int c = k0; c < n; ++c) {
                const LD<T> e = r >= c ? wide(ha[std::size_t(b) * ld * n + r + std::size_t(c) * ld])
                                       : conj_h(wide(ha[std::size_t(b) * ld * n + c + std::size_t(r) * ld]));
                const LD<T> xc = c == k0 ? LD<T>(1) : wide(hx[std::size_t(b) * n + c]);
                acc += e * xc, s += std::abs(e) * std::abs(xc);
            }
            worst = std::max(worst, rel(got[std::size_t(b) * nout + r], acc, s));
        }
    EXPECT_VERIFY(T, Check::blas, n, worst) << "hemv StoredDiag (Im(A(r,r)) as stored), WG=" << WG;
}

TYPED_TEST(DeviceBlasV2Test, StrictPortOpsGiveTheHandKernelBits) {
    strict_port_case<128, TypeParam>(this->dev);
    strict_port_case<1024, TypeParam>(this->dev);
}

// v1 interop: vec(VectorView) / mat(KernelMatrixView) after batch_item(b), non-natural ld,
// inc and batch strides; and the POD Cx arithmetic gives the std::complex result.
TYPED_TEST(DeviceBlasV2Test, InteropWithV1ViewsAndPodComplex) {
    using T = TypeParam;
    constexpr int WG = 64, m = 41, n = 19, ld = m + 4, stride = ld * n + 9, inc = 2, vstride = n * inc + 5;
    auto ha = batched<T>(stride, 61), hx = batched<T>(vstride, 62);
    std::vector<T> hy(std::size_t(kBatch) * m, T(0)), hz(std::size_t(kBatch) * m, T(0));
    T* a = this->dev.up(ha);
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    T* z = this->dev.up(hz);
    const KernelMatrixView<T, MatrixFormat::Dense> kv(a, m, n, ld, stride, kBatch);
    const VectorView<T> xv(x, n, kBatch, inc, vstride);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * m, m);
        v2::gemv<Transpose::NoTrans>(ex, v2::One{}, v2::mat(kv.batch_item(b)), v2::assume_published(v2::vec(xv.batch_item(b))), v2::Zero{}, yv);
        if constexpr (is_cplx<T>) {
            using Cx = batchlas::sycl_device::Cx<real_of<T>>;
            const auto ac = v2::mat(reinterpret_cast<Cx*>(a + std::ptrdiff_t(b) * stride), m, n, ld);
            const auto xc = v2::assume_published(v2::vec(reinterpret_cast<Cx*>(x + std::ptrdiff_t(b) * vstride), n, inc));
            v2::gemv<Transpose::NoTrans>(ex, v2::One{}, ac, xc, v2::Zero{}, v2::vec(reinterpret_cast<Cx*>(z + std::ptrdiff_t(b) * m), m));
        }
    });
    const auto got = this->dev.down(y, hy.size());
    const auto gz = this->dev.down(z, hz.size());
    double worst = 0;
    int cx_diff = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < m; ++r) {
            LD<T> acc = 0;
            long double s = 0;
            for (int c = 0; c < n; ++c) {
                const LD<T> e = wide(ha[std::size_t(b) * stride + r + std::size_t(c) * ld]), xc = wide(hx[std::size_t(b) * vstride + c * inc]);
                acc += e * xc, s += std::abs(e) * std::abs(xc);
            }
            worst = std::max(worst, rel(got[std::size_t(b) * m + r], acc, s));
            if constexpr (is_cplx<T>) cx_diff += !same_bits(gz[std::size_t(b) * m + r], got[std::size_t(b) * m + r]);
        }
    EXPECT_VERIFY(T, Check::blas, n, worst);
    EXPECT_EQ(cx_diff, 0) << "Arith<Cx> and Arith<std::complex> must give the same bits";
}

// Teams: each sub-group is its own executor on its own item; Scope::Sub results per team.
TYPED_TEST(DeviceBlasV2Test, TeamsAreIndependentExecutors) {
    using T = TypeParam;
    constexpr int WG = 256, n = 70, teams = WG / 32;
    auto hx = batched<T>(n, 71), hy = batched<T>(n, 72);
    std::vector<T> ho(kBatch, T(0));
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch / teams, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::Teams<WG> tm{it};
        const int b = static_cast<int>(it.get_group_linear_id()) * teams + tm.team();
        const T r = v2::dotc(tm, v2::vec(x + std::ptrdiff_t(b) * n, n), v2::vec(y + std::ptrdiff_t(b) * n, n));
        if (tm.leader()) o[b] = r;
    });
    const auto got = this->dev.down(o, ho.size());
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b) bitdiff += !same_bits(got[b], got[b % kPeriod]);
    EXPECT_EQ(bitdiff, 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        LD<T> d = 0;
        long double s = 0;
        for (int i = 0; i < n; ++i) {
            const LD<T> xi = wide(hx[std::size_t(b) * n + i]), yi = wide(hy[std::size_t(b) * n + i]);
            d += conj_h(xi) * yi, s += std::abs(xi) * std::abs(yi);
        }
        worst = std::max(worst, rel(got[b], d, s));
    }
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// ---- v2.1 executors and loops ---------------------------------------------------------------

// Ownership record: every element of the Range [lo, hi) and of the m x n grid is visited once,
// by the item the Cyclic rule names, for every unroll policy and 2-D index form. One region
// per executor instance (a team, a partition, an item for Thread); o holds owner + 1.
template <int WG, class MakeEx> void ownership_case(Dev& dev, MakeEx make_ex, int regions, const char* where) {
    constexpr int n = 77, lo = 5, hi = 74, m2 = 7, n2 = 13, kU = 4, kA = 4;
    constexpr int per = n * kU + m2 * n2 * kA;
    constexpr int groups = 64;
    std::vector<int> ho(std::size_t(groups) * regions * per, 0);
    int* o = dev.up(ho);
    dev.template run<WG>(groups, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto ex = make_ex(it);
        const int region = int(it.get_group_linear_id()) * regions + ex.region;
        int* ob = o + std::ptrdiff_t(region) * per;
        const int me = ex.ex.lid() + 1;
        auto mark = [&](int* p) { sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(*p).fetch_add(me); };
        v2::for_each_owned(ex.ex, v2::Range{lo, hi}, [&](int i) { mark(ob + i); });
        v2::for_each_owned<v2::unroll::None>(ex.ex, v2::Range{lo, hi}, [&](int i) { mark(ob + n + i); });
        v2::for_each_owned<v2::unroll::Count<3>>(ex.ex, v2::Range{lo, hi}, [&](int i) { mark(ob + 2 * n + i); });
        v2::for_each_owned<v2::unroll::Count<2>>(ex.ex, hi, [&](int i) { mark(ob + 3 * n + i); });  // 0-based
        int* o2 = ob + n * kU;
        v2::for_each_owned_2d<v2::alg::Flat, std::ptrdiff_t>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + r + c * m2); });
        v2::for_each_owned_2d<v2::alg::DivMod>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + m2 * n2 + r + c * m2); });
        v2::for_each_owned_2d<v2::alg::Stepped>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + 2 * m2 * n2 + r + c * m2); });
        v2::for_each_owned_2d<v2::alg::Walk>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + 3 * m2 * n2 + r + c * m2); });
    });
    const auto got = dev.down(o, ho.size());
    int bad = 0;
    for (int g = 0; g < groups * regions; ++g) {
        const int* ob = got.data() + std::size_t(g) * per;
        const int w = make_ex.width;
        for (int u = 0; u < kU; ++u)
            for (int i = 0; i < n; ++i) {
                const int first = u == 3 ? 0 : lo;
                const int want = (i >= first && i < hi) ? (i - first) % w + 1 : 0;
                bad += ob[u * n + i] != want;
            }
        for (int a = 0; a < kA; ++a)
            for (int e = 0; e < m2 * n2; ++e) bad += ob[n * kU + a * m2 * n2 + e] != e % w + 1;
    }
    EXPECT_EQ(bad, 0) << where;
}

template <class Ex> struct Region {
    Ex ex;
    int region;
};

TEST(DeviceBlasV21, ForEachOwnedRangesUnrollPoliciesAnd2dFormsVisitOnceByTheOwner) {
    Dev dev;
    if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP();
    struct WgM { int width = 64; auto operator()(sycl::nd_item<1> it) const { return Region<v2::WorkGroup<64>>{{it}, 0}; } };
    struct RtM { int width = 96; auto operator()(sycl::nd_item<1> it) const { return Region<v2::RtGroup<>>{v2::RtGroup<>(it), 0}; } };
    struct RtSubM {
        int width = 32;
        auto operator()(sycl::nd_item<1> it) const {
            const v2::RtGroup<v2::Scope::Sub> ex(it);
            return Region<v2::RtGroup<v2::Scope::Sub>>{ex, int(it.get_local_linear_id()) / 32};
        }
    };
    struct TeamsM {
        int width = 8;
        auto operator()(sycl::nd_item<1> it) const {
            const v2::Teams<128, 8> ex{it};
            return Region<v2::Teams<128, 8>>{ex, ex.team()};
        }
    };
    struct ThreadM { int width = 1; auto operator()(sycl::nd_item<1> it) const { return Region<v2::Thread>{{}, int(it.get_local_linear_id())}; } };
    ownership_case<64>(dev, WgM{}, 1, "WorkGroup<64>");
    ownership_case<96>(dev, RtM{}, 1, "RtGroup<Group> @96");
    ownership_case<64>(dev, RtSubM{}, 2, "RtGroup<Sub>");
    ownership_case<128>(dev, TeamsM{}, 16, "Teams<128, 8>");
    ownership_case<32>(dev, ThreadM{}, 32, "Thread");
}

// Teams<WG, P, O>: P-lane teams reduce with the named butterfly order, bit for bit against a
// host emulation of that order (Down and Asc sum in different orders on random data).
template <class T, int P, class O> void teams_order_case(Dev& dev) {
    constexpr int WG = 128, teams = WG / P;
    const auto hx = batched<T>(WG, 81, [](T*) {}, 64);
    T* x = dev.up(hx);
    std::vector<T> ho(std::size_t(64) * teams, T(0));
    T* o = dev.up(ho);
    dev.template run<WG>(64, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::Teams<WG, P, O> tm{it};
        const int g = int(it.get_group_linear_id());
        const T r = tm.team_reduce(x[std::ptrdiff_t(g) * WG + tm.team() * P + tm.lane()]);
        if (tm.leader()) o[g * teams + tm.team()] = r;
    });
    const auto got = dev.down(o, ho.size());
    int bad = 0;
    for (int g = 0; g < 64; ++g)
        for (int t = 0; t < teams; ++t) {
            std::vector<real_of<T>> re(P), im(P);
            for (int l = 0; l < P; ++l) {
                const T v = hx[std::size_t(g) * WG + t * P + l];
                re[l] = std::real(v), im[l] = std::imag(v);
            }
            auto step = [&](int msk) {
                auto r2 = re, i2 = im;
                for (int l = 0; l < P; ++l) r2[l] = re[l] + re[l ^ msk], i2[l] = im[l] + im[l ^ msk];
                re = r2, im = i2;
            };
            if constexpr (std::is_same_v<O, v2::bfly::Down>)
                for (int msk = P / 2; msk >= 1; msk >>= 1) step(msk);
            else
                for (int msk = 1; msk < P; msk <<= 1) step(msk);
            T want;
            if constexpr (is_cplx<T>) want = T(re[0], im[0]);
            else want = re[0];
            bad += !same_bits(got[std::size_t(g) * teams + t], want);
        }
    EXPECT_EQ(bad, 0) << "P = " << P << (std::is_same_v<O, v2::bfly::Down> ? " Down" : " Asc");
}

TYPED_TEST(DeviceBlasV2Test, TeamsOfPLanesReduceInTheNamedButterflyOrder) {
    using T = TypeParam;
    teams_order_case<T, 8, v2::bfly::Down>(this->dev);
    teams_order_case<T, 8, v2::bfly::Asc>(this->dev);
    teams_order_case<T, 16, v2::bfly::Asc>(this->dev);
    teams_order_case<T, 32, v2::bfly::Asc>(this->dev);
}

// RtGroup: bcast from a runtime source over the work-group and over the sub-group; sync orders
// local memory (each item reads its neighbour's slot after the barrier). WG 96, not a power of 2.
TYPED_TEST(DeviceBlasV2Test, RtGroupBcastAndSyncAtARuntimeWidth) {
    using T = TypeParam;
    constexpr int WG = 96;
    const auto hx = batched<T>(WG, 91);
    T* x = this->dev.up(hx);
    std::vector<T> ho(std::size_t(kBatch) * WG * 3, T(0));
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch, sizeof(T) * WG, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const v2::RtGroup<> ex(it);
        const int b = int(it.get_group_linear_id()), l = ex.lid();
        const T mine = x[std::ptrdiff_t(b) * WG + l];
        T* slm = reinterpret_cast<T*>(lm);
        const T g = ex.bcast(mine, (b * 7 + 37) % ex.width());
        const T s = ex.team_exec().bcast(mine, (b + 5) % 32);
        // The reader's slot belongs to another sub-group that writes late (a dependent chain
        // first), so a missing barrier reads the slot before it is written.
        T late = mine;
        const T z = T(v2::Arith<T>::re(mine) * real_of<T>(0));  // a zero the compiler cannot fold
        if (ex.team() != 0)
            for (int k = 0; k < 4000; ++k) late = v2::Arith<T>::add(late, z);
        slm[l] = late;
        v2::sync(ex);
        const T nb = slm[(l + 32) % ex.width()];
        T* ob = o + std::ptrdiff_t(b) * WG * 3;
        ob[l] = g, ob[WG + l] = s, ob[2 * WG + l] = nb;
    });
    const auto got = this->dev.down(o, ho.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int l = 0; l < WG; ++l) {
            const T* xb = hx.data() + std::size_t(b) * WG;
            const T* ob = got.data() + std::size_t(b) * WG * 3;
            bad += !same_bits(ob[l], xb[(b * 7 + 37) % WG]);
            bad += !same_bits(ob[WG + l], xb[(l / 32) * 32 + (b + 5) % 32]);
            bad += !same_bits(ob[2 * WG + l], xb[(l + 32) % WG]);
        }
    EXPECT_EQ(bad, 0);
}

// Grid2D through the batched2d launcher: B := A + b on a 5 x 9 grid, ld 7, batch 1031 over at
// most 7 groups in dimension 0, so every item walks several batch items (for_each_batch).
TYPED_TEST(DeviceBlasV2Test, Batched2dLauncherCoversEveryItemPastTheGroupLimit) {
    using T = TypeParam;
    constexpr int m = 5, n = 9, ld = 7, batch = 1031;
    const auto ha = batched<T>(ld * n, 101, [](T*) {}, batch);
    T* a = this->dev.up(ha);
    std::vector<T> hb(ha.size(), T(-7));
    T* bb = this->dev.up(hb);
    const auto shape = v2::batched2d_shape(m, n, 64);
    v2::batched2d(this->dev.q(), batch, m, n,
                  [=](const v2::Grid2D<>& g, int b) {
                      v2::for_each_owned_2d(g, m, n, [&](int r, int c) {
                          const std::ptrdiff_t k = std::ptrdiff_t(b) * ld * n + c * ld + r;
                          bb[k] = v2::Arith<T>::add(a[k], T(real_of<T>(b)));
                      });
                  },
                  shape, 7)
        .wait_and_throw();
    const auto got = this->dev.down(bb, hb.size());
    int bad = 0;
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < ld; ++r) {
                const std::size_t k = std::size_t(b) * ld * n + c * ld + r;
                bad += !same_bits(got[k], r < m ? T(ha[k] + T(real_of<T>(b))) : T(-7));
            }
    EXPECT_EQ(bad, 0);
}

// ---- v2.1 operands ---------------------------------------------------------------------------

// The same elements through every base and index width: A goes to local memory through an
// accessor view at an int32 offset, then back out through an aligned view (column 0), the row
// view of the wide (ptrdiff_t) view (row 0) and its sub-block (1:, 1:). Rows past m are padding
// that must keep their sentinel. Every item bit for bit.
TYPED_TEST(DeviceBlasV2Test, OperandBasesAndIndexWidthsAddressTheSameElements) {
    using T = TypeParam;
    constexpr int WG = 64, m = 13, n = 11, ld = m + 3, stride = ld * n + 5, slm_off = 7;
    const auto ha = batched<T>(stride, 111);
    T* a = this->dev.up(ha);
    std::vector<T> ho(std::size_t(kBatch) * stride, T(-3));
    T* o = this->dev.up(ho);
    this->dev.q()
        .submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 1> acc(sycl::range<1>(slm_off + ld * n), h);
            h.parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const v2::WorkGroup<WG> ex{it};
                const int b = static_cast<int>(it.get_group_linear_id());
                T* ob = o + std::ptrdiff_t(b) * stride;
                const auto src = v2::mat(a + std::ptrdiff_t(b) * stride, m, n, ld);
                const auto slm = v2::mat(acc, m, n, ld, slm_off);
                v2::for_each_owned_2d(ex, m, n, [&](int r, int c) { slm(r, c) = src(r, c); });
                ex.barrier();
                const auto wide = v2::mat_wide(ob, m, n, ld);
                const auto al = v2::mat_aligned<alignof(T)>(ob, m, n, ld);
                v2::copy(ex, slm.col(0), al.col(0));
                v2::copy(ex, slm.row(0).tail(1), wide.row(0).tail(1));
                const auto ws = wide.sub(1, 1, m - 1, n - 1);
                const auto ss = slm.sub(1, 1, m - 1, n - 1);
                v2::for_each_owned_2d(ex, m - 1, n - 1, [&](int r, int c) { ws(r, c) = ss(r, c); });
            });
        })
        .wait_and_throw();
    const auto got = this->dev.down(o, ho.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int k = 0; k < stride; ++k) {
            const int r = k % ld, c = k / ld;
            const T want = (r < m && c < n) ? ha[std::size_t(b) * stride + k] : T(-3);
            bad += !same_bits(got[std::size_t(b) * stride + k], want);
        }
    EXPECT_EQ(bad, 0);
}

// item(view, b) addresses exactly what v1's operator()(i, j, b) / (i, b) addresses.
TYPED_TEST(DeviceBlasV2Test, ItemAddressesExactlyAsTheV1View) {
    using T = TypeParam;
    constexpr int WG = 64, m = 9, n = 7, ld = m + 2, stride = ld * n + 3, inc = 3, vstride = n * inc + 4;
    std::vector<T> ha(std::size_t(kBatch) * stride, T(0)), hx(std::size_t(kBatch) * vstride, T(0));
    std::vector<int> ho(kBatch, -1);
    T* a = this->dev.up(ha);
    T* x = this->dev.up(hx);
    int* o = this->dev.up(ho);
    const KernelMatrixView<T, MatrixFormat::Dense> kv(a, m, n, ld, stride, kBatch);
    const VectorView<T> xv(x, n, kBatch, inc, vstride);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto im = v2::item(kv, b), ie = v2::item(kv, b, m, n);
        const auto iv = v2::item(xv, b);
        int bad = (im.rows() != m) + (im.cols() != n) + (iv.size() != n);
        v2::for_each_owned_2d(ex, m, n, [&](int r, int c) { bad += (&im(r, c) != &kv(r, c, b)) + (&ie(r, c) != &kv(r, c, b)); });
        v2::for_each_owned(ex, n, [&](int i) { bad += &iv(i) != &xv(i, b); });
        bad = ex.group_reduce(bad);
        if (ex.leader()) o[b] = bad;
    });
    const auto got = this->dev.down(o, ho.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) bad += got[b] != 0;
    EXPECT_EQ(bad, 0);
}

// Frag and Tile moves: loads zero-fill past the live bound, stores write only inside it (the
// poison sentinel past it survives). One Thread executor per item.
TYPED_TEST(DeviceBlasV2Test, FragAndTileMovesHonourTheLiveBound) {
    using T = TypeParam;
    constexpr int WG = 32, N = 8, ld = 9, per = ld * N;
    const auto hx = batched<T>(per, 121);
    T* x = this->dev.up(hx);
    const T poison = T(-5);
    std::vector<T> ho(std::size_t(kBatch) * 3 * per, poison);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch / WG, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        using A = v2::Arith<T>;
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto xa = v2::mat(x + std::ptrdiff_t(b) * per, N, N, ld);
        T* ob = o + std::ptrdiff_t(b) * 3 * per;
        v2::Frag<T, N, v2::Cyclic> f;
        v2::load(f, xa.col(1), 5);
#pragma unroll
        for (int k = 0; k < N; ++k) f[k] = A::add(f[k], A::one());
        v2::store(f, v2::vec(ob, N), 6);
        v2::Frag<T, N> g;
        v2::load_row(g, xa, 2, 6);
        v2::store_row(g, v2::mat(ob + per, N, N, ld), 1, 7);
        v2::Tile<T, 4, 3> t;
        v2::load(t, xa.sub(1, 1, 4, 3), 3, 2);
        v2::store(t, v2::mat(ob + 2 * per, 4, 3, ld));
    });
    const auto got = this->dev.down(o, ho.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* xb = hx.data() + std::size_t(b) * per;
        const T* ob = got.data() + std::size_t(b) * 3 * per;
        for (int k = 0; k < per; ++k) {
            T w0 = poison, w1 = poison, w2 = poison;
            const int r = k % ld, c = k / ld;
            if (k < 6) w0 = k < 5 ? T(xb[ld + k] + T(1)) : T(1);
            if (r == 1 && c < 7) w1 = c < 6 ? xb[2 + c * ld] : T(0);
            if (r < 4 && c < 3) w2 = (r < 3 && c < 2) ? xb[(1 + r) + (1 + c) * ld] : T(0);
            bad += !same_bits(ob[k], w0) + !same_bits(ob[per + k], w1) + !same_bits(ob[2 * per + k], w2);
        }
    }
    EXPECT_EQ(bad, 0);
}

// ---- register fragments on a Partition ----------------------------------------------------

// bcast from a partition-uniform lane that differs per partition, then select: the built-in
// sub_group form must add the chunk base itself, the SubGroupPartition form must not.
template <class T, class MakeEx> void bcast_case(Dev& dev, MakeEx make_ex, const char* where) {
    constexpr int P = 8, WG = 32, per = WG / P;
    const auto hx = batched<T>(P, 71);
    T* x = dev.up(hx);
    std::vector<T> ho(std::size_t(kBatch) * P * P, T(std::numeric_limits<real_of<T>>::quiet_NaN()));
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch / per, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto ex = make_ex(it);
        const int b = static_cast<int>(it.get_global_linear_id()) / P, l = ex.lid();
        const T mine = x[std::ptrdiff_t(b) * P + l];
        for (int k = 0; k < P; ++k) {
            const T v = ex.bcast(mine, static_cast<std::uint32_t>((b * 3 + k) % P));
            o[(std::ptrdiff_t(b) * P + k) * P + l] = v2::select(l % 2 == 0, v, mine);
        }
    });
    const auto got = dev.down(o, ho.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int k = 0; k < P; ++k)
            for (int l = 0; l < P; ++l) {
                const T want = hx[std::size_t(b) * P + (l % 2 == 0 ? (b * 3 + k) % P : l)];
                bad += !same_bits(got[(std::size_t(b) * P + k) * P + l], want);
            }
    EXPECT_EQ(bad, 0) << where;
}

TYPED_TEST(DeviceBlasV2Test, PartitionBcastAndSelectAreExact) {
    using T = TypeParam;
    bcast_case<T>(this->dev, [](sycl::nd_item<1> it) { return v2::Partition<8>{it.get_sub_group()}; }, "Partition<8>");
    bcast_case<T>(this->dev, [](sycl::nd_item<1> it) { return v2::wrap(make_partition<8, true>(it.get_sub_group())); },
                  "SgpPartition<8, Masked>");
}

// potf2_lower on a padded RowFrag: n = 13 in a 16-lane partition, ld = 19, batch 1024,
// representative 1 indefinite at column 5 (info 5, the rows above it still factored).
template <class T, bool UseSlm> void potf2_case(Dev& dev) {
    using D = typename sycl_device::DevMap<T>::type;
    using R = real_of<T>;
    using R4 = tiny_native::tiny_r4<R>;
    constexpr int N = 16, n = 13, ld = 19, WG = 32, kMpw = WG / N, kInfoCol = 5;
    constexpr int kNV = (N + tiny_native::tiny_r4_width<D>() - 1) / tiny_native::tiny_r4_width<D>();
    auto ha = batched<T>(n * ld, 91, [](T* a) {
        std::vector<T> m(a, a + n * ld);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                T s = T(i == j ? 2 * n : 0);
                for (int k = 0; k < n; ++k) s += m[i + k * ld] * conj_h(m[j + k * ld]);
                a[i + j * ld] = s;
            }
    });
    for (int b = 1; b < kBatch; b += kPeriod) ha[std::size_t(b) * n * ld + (kInfoCol - 1) * (ld + 1)] = T(-1000);
    T* a = dev.up(ha);
    std::vector<std::int32_t> hinfo(kBatch, -7);
    std::int32_t* info = dev.up(hinfo);
    dev.template run<WG>(kBatch / kMpw, sizeof(R4) * 2 * kMpw * kNV, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const auto ex = v2::wrap(make_partition<N>(it.get_sub_group()));
        const int pidx = tiny_native::tiny_partition_id(it.get_sub_group(), ex.part);
        const int b = static_cast<int>(it.get_group_linear_id()) * kMpw + pidx, lane = ex.lid();
        D* Ab = reinterpret_cast<D*>(a) + std::ptrdiff_t(b) * n * ld;
        v2::Frag<D, N> rA;
        tiny_native::tiny_load_lower<D, N>(rA.v, Ab, ld, lane, lane < n, is_cplx<T>);
        std::int32_t r;
        if constexpr (UseSlm) {
            r = v2::potf2_lower<D, N, N, v2::OrderGuard::Skip>(ex, rA, n, true,
                                                                v2::Slm<kMpw, R4*>{reinterpret_cast<R4*>(lm), pidx});
        } else {
            r = v2::potf2_lower<D, N, N>(ex, rA, n, true, v2::Shuffle{});
        }
        tiny_native::tiny_store_lower<D, N>(rA.v, Ab, ld, lane, lane < n);
        if (ex.leader()) info[b] = r;
    });
    const auto got = dev.down(a, ha.size());
    const auto ginfo = dev.down(info, hinfo.size());
    int bitdiff = 0, infobad = 0;
    for (int b = 0; b < kBatch; ++b) {
        infobad += ginfo[b] != (b % kPeriod == 1 ? kInfoCol : 0);
        for (int i = 0; i < n * ld; ++i) bitdiff += !same_bits(got[std::size_t(b) * n * ld + i], got[std::size_t(b % kPeriod) * n * ld + i]);
    }
    EXPECT_EQ(infobad, 0);
    EXPECT_EQ(bitdiff, 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const int cols = b == 1 ? kInfoCol - 1 : n;  // a failed factorization: the leading minor
        auto L = [&](int i, int j) { return j > i ? LD<T>(0) : wide(got[std::size_t(b) * n * ld + i + j * ld]); };
        for (int i = 0; i < cols; ++i)
            for (int j = 0; j <= i; ++j) {
                LD<T> s = 0;
                long double sc = 0;
                for (int k = 0; k <= j; ++k) s += L(i, k) * conj_h(L(j, k)), sc += std::abs(L(i, k)) * std::abs(L(j, k));
                worst = std::max(worst, rel(T(s), wide(ha[std::size_t(b) * n * ld + i + j * ld]), sc));
            }
        for (int i = 0; i < n; ++i)  // the strict upper triangle is neither read nor written
            for (int j = i + 1; j < n; ++j) worst = std::max(worst, same_bits(got[std::size_t(b) * n * ld + i + j * ld], ha[std::size_t(b) * n * ld + i + j * ld]) ? 0.0 : 1e300);
    }
    EXPECT_VERIFY(T, Check::factorization, n, worst) << (UseSlm ? "Slm" : "Shuffle");
}

TYPED_TEST(DeviceBlasV2Test, Potf2LowerShuffleAndSlmBroadcasts) {
    using T = TypeParam;
    potf2_case<T, false>(this->dev);
    if constexpr (sizeof(real_of<T>) == 4) potf2_case<T, true>(this->dev);
}

// Compile-time facts: scratch sizes are constexpr; executors carry their sizes as types.
static_assert(v2::scratch_elements<v2::op::ReduceN<3>, v2::WorkGroup<256>, double>() == 24);
static_assert(v2::scratch_elements<v2::op::Reduce, v2::SubGroup<32>, float>() == 0);
static_assert(v2::WorkGroup<256>::nteams == 8 && v2::Teams<256>::size == 32 && v2::Partition<8>::size == 8);
static_assert(std::is_same_v<v2::tag_t<float>, v2::Scalar<float>> && std::is_same_v<v2::tag_t<v2::Zero>, v2::Zero>);

}  // namespace

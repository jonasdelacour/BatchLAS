// SPDX-License-Identifier: MIT
// Device LA (include/batchlas/device) against host references.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld / inc and rectangular
// shapes, complex data with nonzero imaginary parts, ConjTrans, beta != 0 and beta == Zero over
// NaN-poisoned outputs, a poisoned unread triangle, batch 1024 with period-3 data where every
// item must be bit-identical to its representative (SLM paths), a WG = 1024 launch, ties and
// NaN for iamax, nrm2 straddling the overflow / underflow thresholds.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>
#include <src/sycl/device_scalar.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <src/kernels/common/tiny_device.hh>
#include <batchlas/device/arith.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;

// LAPACKE's ?larfg is the reflector oracle (linked through batchlas::verify; skipped when
// BATCHLAS_VERIFY_HAVE_LAPACKE is 0). Declared here with std::complex (layout-compatible).
extern "C" {
int LAPACKE_slarfg(int, float*, float*, int, float*);
int LAPACKE_dlarfg(int, double*, double*, int, double*);
int LAPACKE_clarfg(int, std::complex<float>*, std::complex<float>*, int, std::complex<float>*);
int LAPACKE_zlarfg(int, std::complex<double>*, std::complex<double>*, int, std::complex<double>*);
}


using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;

// Hand-reference conj and work-group sum (the legacy latrd forms), independent of device LA.
template <class T> T hand_conj(const T& a) {
    if constexpr (is_cplx<T>) return std::conj(a);
    else return a;
}
template <class G, class T> T hand_group_sum(const G& g, const T& v) {
    if constexpr (is_cplx<T>) {
        using R = real_of<T>;
        return T(batchlas::portable::reduce_over_group(g, v.real(), sycl::plus<R>()),
                 batchlas::portable::reduce_over_group(g, v.imag(), sycl::plus<R>()));
    } else {
        return batchlas::portable::reduce_over_group(g, v, sycl::plus<T>());
    }
}

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

// The test's queue and device buffers; no queue when no SYCL device is visible (the fixture skips).
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
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
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
        const device::WorkGroup<WG> ex{it};
        const auto xv = device::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
        const auto yv = device::vec(y + std::ptrdiff_t(b) * n, n);
        const auto om = device::mat(o + std::ptrdiff_t(b) * nout * ldo, n, nout, ldo);
        device::copy(ex, yv, om.col(0));
        device::axpby(ex, device::Scalar{a}, xv, device::Scalar{bb}, om.col(0));  // a x + b y
        device::axpby(ex, a, xv, device::Zero{}, om.col(1));                    // a x over NaN
        device::copy(ex, yv, om.col(2));
        device::axpy(ex, device::NegOne{}, xv, om.col(2));                      // y - x
        device::copy(ex, xv, om.col(3));
        device::scal(ex, a, om.col(3));                                     // a x
        device::copy(ex, xv, om.col(4));
        device::copy(ex, yv, om.col(5));
        device::swap(ex, om.col(4), om.col(5));
        device::map(ex, om.col(5), [=](T u, T w) { return device::Arith<T>::add(u, device::Arith<T>::mul(a, w)); }, om.col(5), yv);
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
            const auto xv = device::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
            const auto yv = device::vec(y + std::ptrdiff_t(b) * n, n);
            T* ob = o + std::ptrdiff_t(b) * kSlots;
            if constexpr (P < 0 || P == 0) {
                const T r0 = device::dot(ex, xv, yv), r1 = device::dotc(ex, xv, yv), r2 = T(device::asum(ex, xv)), r3 = T(device::amax(ex, xv));
                const T r4 = device::reduce(ex, n, [&](int i) { return device::Arith<T>::mul(xv(i), xv(i)); });
                if (ex.leader()) ob[kDot] = r0, ob[kDotc] = r1, ob[kAsum] = r2, ob[kAmax] = r3, ob[kReduce] = r4;
            }
            if constexpr (P < 0 || P == 1) {
                const T r0 = T(device::nrm2<device::alg::Scaled>(ex, xv)), r1 = T(device::nrm2<device::alg::Unscaled>(ex, xv)), r2 = T(real_of<T>(device::iamax(ex, xv)));
                if (ex.leader()) ob[kNrm2] = r0, ob[kNrm2U] = r1, ob[kIamax] = r2;
            }
            if constexpr (P < 0 || P == 2) {
                const auto xs = std::array<decltype(yv), 3>{yv, device::vec(x + std::ptrdiff_t(b) * n * inc, n), yv.tail(0)};
                const auto d3 = device::dotN<3, true>(ex, xs, yv);
                const auto s3 = device::dotN<3, true>(ex, device::Scratch<T, 3 * 32>{reinterpret_cast<T*>(lm)}, xs, yv);
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
    run_reductions<T, 256>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return device::WorkGroup<256>{it}; });
    check_reductions(n, inc, hx, hy, out, "WorkGroup<256>");
    run_reductions<T, 64>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return device::WorkGroup<64>{it}; });
    check_reductions(n, inc, hx, hy, out, "WorkGroup<64>");
    // The ceiling launch: 1024 items per work-group (the 4090's and the SYCL limit we target).
    run_reductions<T, 1024, true>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return device::WorkGroup<1024>{it}; });
    check_reductions(n, inc, hx, hy, out, "WorkGroup<1024>");
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return device::SubGroup<32>{it.get_sub_group()}; });
    check_reductions(n, inc, hx, hy, out, "SubGroup<32>");
    // 4 partitions of 8 lanes per sub-group, one item each; then the masked src-side binding.
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return device::Partition<8>{it.get_sub_group()}; }, 4, 8);
    check_reductions(n, inc, hx, hy, out, "Partition<8>");
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out,
                          [](sycl::nd_item<1> it) { return device::SgpPartition<8, true>{it.get_sub_group()}; }, 4, 8);
    check_reductions(n, inc, hx, hy, out, "SgpPartition<8, Masked>");
    run_reductions<T, 32>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1>) { return device::Thread{}; }, 32, 1);
    check_reductions(n, inc, hx, hy, out, "Thread");
    // Runtime-width executors: a 96-item work-group (not a power of two), and one per sub-group.
    run_reductions<T, 96>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return device::RtGroup<>(it); });
    check_reductions(n, inc, hx, hy, out, "RtGroup<Group> @96");
    run_reductions<T, 64>(this->dev, n, inc, hx, hy, out, [](sycl::nd_item<1> it) { return device::RtGroup<device::Scope::Sub>(it); }, 2, 32);
    check_reductions(n, inc, hx, hy, out, "RtGroup<Sub>");
}

// iamax: lowest index on ties (the tie straddles two owners), NaN wins at its lowest index.
// amax: NaN propagates. nrm2: straddles overflow and underflow of the unscaled sum.
TYPED_TEST(DeviceBlasV2Test, IamaxTiesNanAndNrm2Thresholds) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 128, n = 300, nitem = 5;
    const R big = std::sqrt(std::numeric_limits<R>::max()) * R(4), tiny = std::ldexp(std::sqrt(std::numeric_limits<R>::min()), -12);
    std::vector<T> hx(std::size_t(nitem) * n);
    for (int i = 0; i < n; ++i) {
        hx[i] = val<T>(i) * R(0.5);
        hx[n + i] = val<T>(i) * R(0.5);
        hx[2 * n + i] = val<T>(i) * big;
        hx[3 * n + i] = val<T>(i) * tiny;
        // Item 4 straddles Blue's tbig: even entries above it, odd ones below, both sums count.
        const R tbig = std::ldexp(R(1), (std::numeric_limits<R>::max_exponent - std::numeric_limits<R>::digits + 1) / 2);
        hx[4 * n + i] = val<T>(i) * (i % 2 == 0 ? R(8) * tbig : R(0.5) * tbig);
    }
    hx[37] = hx[37 + WG] = hx[37 + WG + 3] = T(R(2));  // tie on one owner (37, 165) and across owners (168)
    hx[n + 250] = hx[n + 20] = T(std::numeric_limits<R>::quiet_NaN());
    hx[n + 7] = T(R(9));
    T* x = this->dev.up(hx);
    std::vector<int> hi(nitem * 2, -7);
    std::vector<R> hr(nitem * 3, R(-1));
    int* io = this->dev.up(hi);
    R* ro = this->dev.up(hr);
    this->dev.template run<WG>(nitem, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<WG> ex{it};
        const auto xv = device::vec(x + std::ptrdiff_t(b) * n, n);
        const int im = device::iamax(ex, xv);
        const int im_sub = device::iamax(device::SubGroup<32>{it.get_sub_group()}, xv);
        const R am = device::amax(ex, xv), nr = device::nrm2<device::alg::Scaled>(ex, xv), nu = device::nrm2<device::alg::Unscaled>(ex, xv);
        const R nb = device::nrm2<device::alg::Blue>(ex, xv);
        if (ex.leader()) io[2 * b] = im, io[2 * b + 1] = im_sub, ro[3 * b] = am, ro[3 * b + 1] = nr, ro[3 * b + 2] = nb;
        if (ex.leader() && b >= 2) ro[3 * b] = nu;
    });
    const auto gi = this->dev.down(io, hi.size());
    const auto gr = this->dev.down(ro, hr.size());
    EXPECT_EQ(gi[0], 37) << "tie: lowest index";
    EXPECT_EQ(gi[1], 37) << "tie: lowest index (Scope::Sub)";
    EXPECT_EQ(gi[2], 20) << "NaN: lowest NaN index";
    EXPECT_EQ(gi[3], 20);
    EXPECT_TRUE(std::isnan(gr[3])) << "amax propagates NaN";
    for (int b = 2; b < 4; ++b) {
        long double s = 0;
        const long double scale = b == 2 ? big : tiny;
        for (int i = 0; i < n; ++i) s += std::norm(LD<T>(hx[std::size_t(b) * n + i]) / scale);
        const long double ref = std::sqrt(s) * scale;
        EXPECT_VERIFY(T, Check::blas, n, double(std::abs(gr[3 * b + 1] - ref) / ref)) << "scaled nrm2, item " << b;  // NaN fails
        EXPECT_VERIFY(T, Check::blas, n, double(std::abs(gr[3 * b + 2] - ref) / ref)) << "Blue nrm2, item " << b;
        // The unscaled form must fail here, or the item does not straddle the threshold.
        EXPECT_FALSE(batchlas::verify::within<T>(Check::blas, n, double(std::abs(gr[3 * b] - ref) / ref)))
            << "unscaled nrm2 survived item " << b << ": the threshold is not straddled";
    }
    {
        long double s = 0;
        for (int i = 0; i < n; ++i) s += std::norm(LD<T>(hx[std::size_t(4) * n + i]));
        const long double ref = std::sqrt(s);
        EXPECT_VERIFY(T, Check::blas, n, double(std::abs(gr[3 * 4 + 2] - ref) / ref)) << "Blue nrm2, big and medium sums both nonzero";
        EXPECT_VERIFY(T, Check::blas, n, double(std::abs(gr[3 * 4 + 1] - ref) / ref)) << "scaled nrm2, item 4";
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
        const auto av = device::mat(a + std::ptrdiff_t(b) * ld * n, m, n, ld);
        const auto xv = device::assume_published(device::vec(x + std::ptrdiff_t(b) * xlen * inc, xlen, inc));
        const auto yv = device::vec(y + std::ptrdiff_t(b) * ylen, ylen);
        if constexpr (BetaZero) device::gemv<TR, Alg>(ex, device::Scalar{alpha}, av, xv, device::Zero{}, yv);
        else device::gemv<TR, Alg>(ex, alpha, av, xv, beta, yv);
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
    auto wg = [](sycl::nd_item<1> it) { return device::WorkGroup<128>{it}; };
    auto sg = [](sycl::nd_item<1> it) { return device::SubGroup<32>{it.get_sub_group()}; };
    auto pt = [](sycl::nd_item<1> it) { return device::Partition<16>{it.get_sub_group()}; };
    auto th = [](sycl::nd_item<1>) { return device::Thread{}; };
    auto tm = [](sycl::nd_item<1> it) { return device::Teams<128>{it}; };
    constexpr auto N = Transpose::NoTrans, Tt = Transpose::Trans, C = Transpose::ConjTrans;
    gemv_case<N, void, 128, false, T>(this->dev, wg, 1, 128, "N WorkGroup RowOwner");
    gemv_case<N, void, 128, true, T>(this->dev, wg, 1, 128, "N WorkGroup RowOwner beta=Zero");
    gemv_case<Tt, void, 128, false, T>(this->dev, wg, 1, 128, "T WorkGroup SgPerCol");
    gemv_case<C, void, 128, false, T>(this->dev, wg, 1, 128, "C WorkGroup SgPerCol");
    gemv_case<C, void, 128, true, T>(this->dev, wg, 1, 128, "C WorkGroup SgPerCol beta=Zero");
    gemv_case<C, void, 128, false, T>(this->dev, tm, 1, 128, "C Teams SgPerCol");
    gemv_case<C, device::alg::GroupReducePerOutput, 128, false, T>(this->dev, wg, 1, 128, "C WorkGroup GroupReducePerOutput");
    gemv_case<N, void, 32, false, T>(this->dev, sg, 1, 32, "N SubGroup");
    gemv_case<C, void, 32, false, T>(this->dev, sg, 1, 32, "C SubGroup");
    gemv_case<N, void, 32, false, T>(this->dev, pt, 2, 16, "N Partition<16>");
    gemv_case<C, void, 32, true, T>(this->dev, pt, 2, 16, "C Partition<16> beta=Zero");
    gemv_case<N, void, 32, false, T>(this->dev, th, 32, 1, "N Thread Serial");
    gemv_case<C, void, 32, false, T>(this->dev, th, 32, 1, "C Thread Serial");
    auto rt = [](sycl::nd_item<1> it) { return device::RtGroup<>(it); };
    auto tm16 = [](sycl::nd_item<1> it) { return device::Teams<128, 16>{it}; };
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
        const device::WorkGroup<WG> ex{it};
        const auto xl = device::vec<device::Mem::Local>(reinterpret_cast<T*>(lm), n);
        device::copy(ex, device::vec(x + std::ptrdiff_t(b) * n, n), xl);
        const auto av = device::mat(a + std::ptrdiff_t(b % kPeriod) * ld * n, n, n, ld);
        const auto yv = device::vec(y + std::ptrdiff_t(b) * n, n);
        if constexpr (Herm) device::hemv<Uplo::Lower>(ex, alpha, av, ex.publish(xl), beta, yv);
        else device::symv<Uplo::Lower>(ex, alpha, av, ex.publish(xl), beta, yv);
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
        const device::WorkGroup<WG> ex{it};
        const auto av = device::mat(a + std::ptrdiff_t(b) * ld * n, m, n, ld);
        const auto xv = device::vec(x + std::ptrdiff_t(b) * m, m);
        const auto yv = device::assume_published(device::vec(y + std::ptrdiff_t(b) * n * inc, n, inc));
        if constexpr (Conj) device::gerc(ex, alpha, xv, yv, av);
        else device::geru(ex, alpha, xv, yv, av);
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

// The worked latrd step (W1) on device LA, complex-aware: w := A v (hemv), then for p < K
// w -= U_p (W_p^H v) + W_p (U_p^H v) with U_p = unit_head(V_p, p) taken from ONE packed dotN<2>
// exchange per p (dot_many); w *= tau; w += (-tau/2 v^H w) v. Ops chain on one Cyclic layout: the only
// barriers are publish(v) and the dotN exchange.
TYPED_TEST(DeviceBlasV2Test, LatrdStepComposesOnOneLayout) {
    using T = TypeParam;
    using A = device::Arith<T>;
    constexpr int WG = 128, n = 150, ld = n + 2, K = 4;
    const T tau = val<T>(77);
    auto ha = batched<T>(ld * n, 51), hv = batched<T>(n, 52), hu = batched<T>(ld * K, 53), hw = batched<T>(ld * K, 54);
    std::vector<T> ho(std::size_t(kBatch) * n, T(0));
    T* a = this->dev.up(ha);
    T* v = this->dev.up(hv);
    T* u = this->dev.up(hu);
    T* w = this->dev.up(hw);
    T* o = this->dev.up(ho);
    constexpr int kScr = device::scratch_elements<device::op::ReduceN<2>, device::WorkGroup<WG>, T>();
    this->dev.template run<WG>(kBatch, sizeof(T) * (2 * n + kScr), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<WG> ex{it};
        T* l = reinterpret_cast<T*>(lm);
        const auto vl = device::vec<device::Mem::Local>(l, n), wl = device::vec<device::Mem::Local>(l + n, n);
        const device::Scratch<T, kScr> scr{l + 2 * n};
        const auto Am = device::mat(a + std::ptrdiff_t(b) * ld * n, n, n, ld);
        const auto Um = device::mat(u + std::ptrdiff_t(b) * ld * K, n, K, ld), Wm = device::mat(w + std::ptrdiff_t(b) * ld * K, n, K, ld);
        device::copy(ex, device::vec(v + std::ptrdiff_t(b) * n, n), vl);
        device::hemv<Uplo::Lower>(ex, device::One{}, Am, ex.publish(vl), device::Zero{}, wl);
        for (int p = 0; p < K; ++p) {
            const auto up = device::unit_head(Um.col(p), p);
            const auto gd = device::dot_many<true>(ex, scr, vl, Wm.col(p), up);  // one exchange: W_p^H v, U_p^H v
            const T gam = gd[0], del = gd[1];
            device::map(ex, wl, [=](T wr, T ur, T wpr) { return A::sub(wr, A::add(A::mul(ur, gam), A::mul(wpr, del))); }, wl, up, Wm.col(p));
        }
        device::scal(ex, tau, wl);
        const T d = device::dotc(ex, vl, wl);
        device::axpy(ex, A::mul(T(real_of<T>(-0.5)), A::mul(tau, d)), vl, wl);
        device::copy(ex, wl, device::vec(o + std::ptrdiff_t(b) * n, n));
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
// ExplicitInline> on Trailing operands (a host KernelMatrixView and a device Mat), scratch-less
// dot_many<Scope::Group, Native>, Native scal / dotc / axpy, offset views (col / tail) and the
// ranged for_each_owned, each against the legacy latrd expression computed in the same kernel.
// Im(A(r,r)) != 0 is read as stored, so the default hemv (which masks it) must differ: the
// control that the bit compare can fail. A WG = 1024 launch covers the ceiling.
template <int WG, class T> void strict_port_case(Dev& dev) {
    using R = real_of<T>;
    using device::Mem;
    using device::Scope;
    using device::arith::Native;
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
        const int b = static_cast<int>(it.get_group_linear_id());
        const int lid = static_cast<int>(it.get_local_linear_id());
        const device::WorkGroup<WG> ex{it};
        T* const lv = reinterpret_cast<T*>(lm);
        T* const lw = lv + n;
        T* const ob = o + std::ptrdiff_t(b) * nout;
        const auto Ab = kv.batch_item(b % kPeriod);
        const auto A = device::mat(Ab.data(), n, n, ld);
        const auto W = device::mat(w + std::ptrdiff_t(b) * ld * P, n, P, ld);
        const auto vfull = device::vec<Mem::Local>(lv, n);
        // Ranged for_each_owned fills v(k0:) with the latrd unit-head form; device::copy then reads
        // it back on tail(k0) with no barrier (same owners).
        device::for_each_owned(ex, k0, n, [&](int r) { vfull(r) = (r == k0) ? T(1) : x[std::ptrdiff_t(b) * n + r]; });
        device::copy(ex, vfull.tail(k0), device::vec(ob + 5 * n + k0, n - k0));
        // hemv: hand (legacy latrd loop), device on KernelMatrixView, device on a device Mat, default hemv.
        ex.barrier(Mem::Local);
        for (int r = k0 + lid; r < n; r += WG) {
            const int c_split = sycl::min(r, n - 1);
            T acc = T(0);
            for (int c = k0; c <= c_split; ++c) device::ArithInline<T>::mac(acc, Ab(r, c), lv[c]);
            for (int c = c_split + 1; c < n; ++c) device::ArithInline<T>::mac_conj(acc, Ab(c, r), lv[c]);
            ob[r] = acc;
        }
        const auto vp = device::assume_published(device::trailing(vfull, k0));
        device::hemv<Uplo::Lower, device::alg::RowOwnerSplitStoredDiag, device::arith::ExplicitInline>(
            ex, device::One{}, device::trailing(Ab, k0, n), vp, device::Zero{}, device::trailing(device::vec(ob + n, n), k0));
        device::hemv<Uplo::Lower, device::alg::RowOwnerSplitStoredDiag, device::arith::ExplicitInline>(
            ex, device::One{}, device::trailing(A, k0), vp, device::Zero{}, device::trailing(device::vec(ob + 2 * n, n), k0));
        device::hemv<Uplo::Lower>(ex, device::One{}, A.tail(k0), device::assume_published(vfull.tail(k0)), device::Zero{}, device::vec(ob + 3 * n + k0, n - k0));
        // gamma/delta: hand two-accumulator loop + two group reductions vs dot_many.
        const auto v = vfull.tail(k0);
        for (int p = 0; p < P; ++p) {
            T gp = T(0), dp = T(0);
            for (int c = k0 + lid; c < n; c += WG) {
                const T vc = lv[c];
                gp += hand_conj(W(c, p)) * vc;
                const T vcp = (c == p + 1) ? T(1) : ((c > p + 1) ? Ab(c, p) : T(0));
                dp += hand_conj(vcp) * vc;
            }
            const T g0 = hand_group_sum(it.get_group(), gp), d0 = hand_group_sum(it.get_group(), dp);
            const auto gd = device::dot_many<true, Scope::Group, Native>(ex, v, W.col(p).tail(k0), device::unit_head(A.col(p), p + 1).tail(k0));
            if (ex.leader()) ob[6 * n + 4 * p] = g0, ob[6 * n + 4 * p + 1] = gd[0], ob[6 * n + 4 * p + 2] = d0, ob[6 * n + 4 * p + 3] = gd[1];
        }
        // scal / dotc / axpy (Native) vs the hand std:: expressions, on w = hand hemv result.
        for (int r = k0 + lid; r < n; r += WG) lw[r] = ob[r];
        const auto wv = device::vec<Mem::Local>(lw, n).tail(k0);
        T hand_dot_p = T(0);
        for (int r = k0 + lid; r < n; r += WG) hand_dot_p += hand_conj(lv[r]) * (ob[r] * tau);
        const T hand_dot = hand_group_sum(it.get_group(), hand_dot_p);
        device::scal<Native>(ex, tau, wv);
        const T dot = device::dotc<Scope::Group, Native>(ex, v, wv);
        device::axpy<Native>(ex, T(-0.5) * tau * dot, v, wv);
        for (int r = k0 + lid; r < n; r += WG) {
            T h = ob[r];
            h *= tau;
            h += (T(-0.5) * tau * hand_dot) * lv[r];
            ob[4 * n + r] = h;  // hand
        }
        device::copy(ex, wv, device::vec(lw, n).tail(k0));
        if (ex.leader()) ob[6 * n + 4 * P] = hand_dot, ob[6 * n + 4 * P + 1] = dot;
        // Slot 5: the ranged fill read back through tail(k0) must equal v; then the device L1 result.
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
    EXPECT_EQ(hemv_mat, 0) << "hemv StoredDiag on a device Mat vs hand, WG=" << WG;
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

// host-view interop: vec(VectorView) / mat(KernelMatrixView) after batch_item(b), non-natural ld,
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
        const device::WorkGroup<WG> ex{it};
        const auto yv = device::vec(y + std::ptrdiff_t(b) * m, m);
        device::gemv<Transpose::NoTrans>(ex, device::One{}, device::mat(kv.batch_item(b)), device::assume_published(device::vec(xv.batch_item(b))), device::Zero{}, yv);
        if constexpr (is_cplx<T>) {
            using Cx = batchlas::sycl_device::Cx<real_of<T>>;
            const auto ac = device::mat(reinterpret_cast<Cx*>(a + std::ptrdiff_t(b) * stride), m, n, ld);
            const auto xc = device::assume_published(device::vec(reinterpret_cast<Cx*>(x + std::ptrdiff_t(b) * vstride), n, inc));
            device::gemv<Transpose::NoTrans>(ex, device::One{}, ac, xc, device::Zero{}, device::vec(reinterpret_cast<Cx*>(z + std::ptrdiff_t(b) * m), m));
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
        const device::Teams<WG> tm{it};
        const int b = static_cast<int>(it.get_group_linear_id()) * teams + tm.team();
        const T r = device::dotc(tm, device::vec(x + std::ptrdiff_t(b) * n, n), device::vec(y + std::ptrdiff_t(b) * n, n));
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

// ---- executors and loops --------------------------------------------------------------------

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
        device::for_each_owned(ex.ex, device::Range{lo, hi}, [&](int i) { mark(ob + i); });
        device::for_each_owned<device::unroll::None>(ex.ex, device::Range{lo, hi}, [&](int i) { mark(ob + n + i); });
        device::for_each_owned<device::unroll::Count<3>>(ex.ex, device::Range{lo, hi}, [&](int i) { mark(ob + 2 * n + i); });
        device::for_each_owned<device::unroll::Count<2>>(ex.ex, hi, [&](int i) { mark(ob + 3 * n + i); });  // 0-based
        int* o2 = ob + n * kU;
        device::for_each_owned_2d<device::alg::Flat, std::ptrdiff_t>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + r + c * m2); });
        device::for_each_owned_2d<device::alg::DivMod>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + m2 * n2 + r + c * m2); });
        device::for_each_owned_2d<device::alg::Stepped>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + 2 * m2 * n2 + r + c * m2); });
        device::for_each_owned_2d<device::alg::Walk>(ex.ex, m2, n2, [&](int r, int c) { mark(o2 + 3 * m2 * n2 + r + c * m2); });
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
    if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
    if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP();
    struct WgM { int width = 64; auto operator()(sycl::nd_item<1> it) const { return Region<device::WorkGroup<64>>{{it}, 0}; } };
    struct RtM { int width = 96; auto operator()(sycl::nd_item<1> it) const { return Region<device::RtGroup<>>{device::RtGroup<>(it), 0}; } };
    struct RtSubM {
        int width = 32;
        auto operator()(sycl::nd_item<1> it) const {
            const device::RtGroup<device::Scope::Sub> ex(it);
            return Region<device::RtGroup<device::Scope::Sub>>{ex, int(it.get_local_linear_id()) / 32};
        }
    };
    struct TeamsM {
        int width = 8;
        auto operator()(sycl::nd_item<1> it) const {
            const device::Teams<128, 8> ex{it};
            return Region<device::Teams<128, 8>>{ex, ex.team()};
        }
    };
    struct ThreadM { int width = 1; auto operator()(sycl::nd_item<1> it) const { return Region<device::Thread>{{}, int(it.get_local_linear_id())}; } };
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
        const device::Teams<WG, P, O> tm{it};
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
            if constexpr (std::is_same_v<O, device::bfly::Down>)
                for (int msk = P / 2; msk >= 1; msk >>= 1) step(msk);
            else
                for (int msk = 1; msk < P; msk <<= 1) step(msk);
            T want;
            if constexpr (is_cplx<T>) want = T(re[0], im[0]);
            else want = re[0];
            bad += !same_bits(got[std::size_t(g) * teams + t], want);
        }
    EXPECT_EQ(bad, 0) << "P = " << P << (std::is_same_v<O, device::bfly::Down> ? " Down" : " Asc");
}

TYPED_TEST(DeviceBlasV2Test, TeamsOfPLanesReduceInTheNamedButterflyOrder) {
    using T = TypeParam;
    teams_order_case<T, 8, device::bfly::Down>(this->dev);
    teams_order_case<T, 8, device::bfly::Asc>(this->dev);
    teams_order_case<T, 16, device::bfly::Asc>(this->dev);
    teams_order_case<T, 32, device::bfly::Asc>(this->dev);
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
    std::vector<int> hsg(kBatch, 0);
    int* sgs = this->dev.up(hsg);
    this->dev.template run<WG>(kBatch, sizeof(T) * WG, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const device::RtGroup<> ex(it);
        const int b = int(it.get_group_linear_id()), l = ex.lid();
        const T mine = x[std::ptrdiff_t(b) * WG + l];
        T* slm = reinterpret_cast<T*>(lm);
        const T g = ex.bcast(mine, (b * 7 + 37) % ex.width());
        const int sg = ex.team_size();  // the device's sub-group size (the CPU device may not honour 32)
        const T s = ex.team_exec().bcast(mine, (b + 5) % sg);
        if (l == 0) sgs[b] = sg;
        // The reader's slot belongs to another sub-group that writes late (a dependent chain
        // first), so a missing barrier reads the slot before it is written.
        T late = mine;
        const T z = T(device::Arith<T>::re(mine) * real_of<T>(0));  // a zero the compiler cannot fold
        if (ex.team() != 0)
            for (int k = 0; k < 4000; ++k) late = device::Arith<T>::add(late, z);
        slm[l] = late;
        device::sync(ex);
        const T nb = slm[(l + 32) % ex.width()];
        T* ob = o + std::ptrdiff_t(b) * WG * 3;
        ob[l] = g, ob[WG + l] = s, ob[2 * WG + l] = nb;
    });
    const auto got = this->dev.down(o, ho.size());
    const auto gsg = this->dev.down(sgs, hsg.size());
    int bad_g = 0, bad_s = 0, bad_sync = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int l = 0; l < WG; ++l) {
            const T* xb = hx.data() + std::size_t(b) * WG;
            const T* ob = got.data() + std::size_t(b) * WG * 3;
            bad_g += !same_bits(ob[l], xb[(b * 7 + 37) % WG]);
            const int sg = gsg[b];
            bad_s += !same_bits(ob[WG + l], xb[(l / sg) * sg + (b + 5) % sg]);
            bad_sync += !same_bits(ob[2 * WG + l], xb[(l + 32) % WG]);
        }
    EXPECT_EQ(bad_g, 0) << "group bcast";
    EXPECT_EQ(bad_s, 0) << "sub-group bcast (sub-group size " << gsg[0] << ")";
    EXPECT_EQ(bad_sync, 0) << "sync";
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
    const auto shape = device::batched2d_shape(m, n, 64);
    device::batched2d(this->dev.q(), batch, m, n,
                  [=](const device::Grid2D<>& g, int b) {
                      device::for_each_owned_2d(g, m, n, [&](int r, int c) {
                          const std::ptrdiff_t k = std::ptrdiff_t(b) * ld * n + c * ld + r;
                          bb[k] = device::Arith<T>::add(a[k], T(real_of<T>(b)));
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

// ---- operands --------------------------------------------------------------------------------

// The same elements through every base and index width: A goes to local memory through an
// accessor view at an int32 offset, then back out through an aligned view (column 0), the row
// view of the wide (ptrdiff_t) view (row 0) and its sub-block (1:, 1:). Rows past m are padding
// that must keep their sentinel. Every item bit for bit.
TYPED_TEST(DeviceBlasV2Test, OperandBasesAndIndexWidthsAddressTheSameElements) {
    using T = TypeParam;
    constexpr int WG = 64, m = 13, n = 11, ld = m + 3, stride = ld * n + 5, slm_off = 7;
    const auto ha = batched<T>(stride, 111);
    T* a = this->dev.up(ha);
    std::vector<T> ho(std::size_t(kBatch) * stride, T(-3)), ho2(std::size_t(kBatch) * (n + m), T(-3));
    T* o = this->dev.up(ho);
    T* o2 = this->dev.up(ho2);
    this->dev.q()
        .submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 1> acc(sycl::range<1>(slm_off + ld * n), h);
            h.parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const device::WorkGroup<WG> ex{it};
                const int b = static_cast<int>(it.get_group_linear_id());
                T* ob = o + std::ptrdiff_t(b) * stride;
                const auto src = device::mat(a + std::ptrdiff_t(b) * stride, m, n, ld);
                const auto slm = device::mat(acc, m, n, ld, slm_off);
                device::for_each_owned_2d(ex, m, n, [&](int r, int c) { slm(r, c) = src(r, c); });
                ex.barrier();
                const auto wide = device::mat_wide(ob, m, n, ld);
                const auto al = device::mat_aligned<alignof(T)>(ob, m, n, ld);
                device::copy(ex, slm.col(0), al.col(0));
                device::copy(ex, slm.row(0).tail(1), wide.row(0).tail(1));
                const auto ws = wide.sub(1, 1, m - 1, n - 1);
                const auto ss = slm.sub(1, 1, m - 1, n - 1);
                device::for_each_owned_2d(ex, m - 1, n - 1, [&](int r, int c) { ws(r, c) = ss(r, c); });
                // Row 2 through the accessor's row view; column 1 read back through a raw pointer
                // at the offset, so the accessor view must have honoured it.
                T* ob2 = o2 + std::ptrdiff_t(b) * (n + m);
                device::copy(ex, slm.row(2), device::vec(ob2, n));
                const auto raw = device::mat(batchlas::util::get_raw_ptr(acc) + slm_off, m, n, ld);
                device::copy(ex, raw.col(1), device::vec(ob2 + n, m));
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
    const auto got2 = this->dev.down(o2, ho2.size());
    for (int b = 0; b < kBatch; ++b) {
        for (int j = 0; j < n; ++j) bad += !same_bits(got2[std::size_t(b) * (n + m) + j], ha[std::size_t(b) * stride + 2 + j * ld]);
        for (int i = 0; i < m; ++i) bad += !same_bits(got2[std::size_t(b) * (n + m) + n + i], ha[std::size_t(b) * stride + i + ld]);
    }
    EXPECT_EQ(bad, 0);
}

// item(view, b) addresses exactly what the host view's operator()(i, j, b) / (i, b) addresses.
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
        const device::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto im = device::item(kv, b), ie = device::item(kv, b, m, n);
        const auto iv = device::item(xv, b);
        int bad = (im.rows() != m) + (im.cols() != n) + (iv.size() != n);
        device::for_each_owned_2d(ex, m, n, [&](int r, int c) { bad += (&im(r, c) != &kv(r, c, b)) + (&ie(r, c) != &kv(r, c, b)); });
        device::for_each_owned(ex, n, [&](int i) { bad += &iv(i) != &xv(i, b); });
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
        using A = device::Arith<T>;
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto xa = device::mat(x + std::ptrdiff_t(b) * per, N, N, ld);
        T* ob = o + std::ptrdiff_t(b) * 3 * per;
        device::Frag<T, N, device::Cyclic> f;
        device::load(f, xa.col(1), 5);
#pragma unroll
        for (int k = 0; k < N; ++k) f[k] = A::add(f[k], A::one());
        device::store(f, device::vec(ob, N), 6);
        device::Frag<T, N> g;
        device::load_row(g, xa, 2, 6);
        device::store_row(g, device::mat(ob + per, N, N, ld), 1, 7);
        device::Tile<T, 4, 3> t;
        device::load(t, xa.sub(1, 1, 4, 3), 3, 2);
        device::store(t, device::mat(ob + 2 * per, 4, 3, ld));
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

// ---- arithmetic policies and scalar LAPACK --------------------------------------------------

namespace lapack_ref {
// false when the build has no LAPACKE: the caller skips its reflector checks.
template <class T> bool larfg([[maybe_unused]] int n, [[maybe_unused]] T& alpha, [[maybe_unused]] T* x, [[maybe_unused]] T& tau) {
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if constexpr (std::is_same_v<T, float>) LAPACKE_slarfg(n, &alpha, x, 1, &tau);
    else if constexpr (std::is_same_v<T, double>) LAPACKE_dlarfg(n, &alpha, x, 1, &tau);
    else if constexpr (std::is_same_v<T, std::complex<float>>) LAPACKE_clarfg(n, &alpha, x, 1, &tau);
    else LAPACKE_zlarfg(n, &alpha, x, 1, &tau);
    return true;
#else
    return false;
#endif
}
}  // namespace lapack_ref

// Every policy against a long-double reference, and arith::Fma bit for bit against
// src/sycl/device_scalar.hh (dev_mul, fma_acc, dev_recip, dev_div) in the same kernel. Inputs
// take both Smith branches and a quotient whose naive form overflows.
enum ArSlot { kMul, kMac, kMacC, kMsub, kDiv, kRecip, kDivR, kScaleR, kArSlots };
TYPED_TEST(DeviceBlasV2Test, ArithPoliciesAgreeAndFmaMatchesDeviceScalar) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 256, P = 4;  // P policies: Explicit, ExplicitInline, Native, Fma
    auto h = batched<T>(3 * n, 131, [](T*) {}, kPeriod);  // only representative 0 is used
    const R big = std::is_same_v<R, float> ? R(1e30) : R(1e300);
    if constexpr (is_cplx<T>) {
        h[0] = T(big, big), h[n] = T(big, -big);           // (big + big i) / (big - big i)
        h[1] = T(R(3), R(-2)), h[n + 1] = T(R(0.5), R(7));  // |im| > |re|: Smith's second branch
    } else {
        h[0] = big, h[n] = big;
    }
    T* x = this->dev.up(h);
    std::vector<T> ho(std::size_t(n) * (P + 1) * kArSlots, T(0));
    T* o = this->dev.up(ho);
    this->dev.template run<32>(n / 32, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int i = static_cast<int>(it.get_global_linear_id());
        const T a = x[i], b = x[n + i], c = x[2 * n + i];
        const R sr = device::Arith<T>::re(b);
        auto one = [&](auto ops, int slot) {
            using Ar = device::arith_t<decltype(ops), T>;
            T* ob = o + (std::ptrdiff_t(i) * (P + 1) + slot) * kArSlots;
            T m1 = c, m2 = c, m3 = c;
            Ar::mac(m1, a, b);
            Ar::mac_conj(m2, a, b);
            Ar::msub(m3, a, b);
            ob[kMul] = Ar::mul(a, b), ob[kMac] = m1, ob[kMacC] = m2, ob[kMsub] = m3;
            ob[kDiv] = Ar::div(a, b), ob[kRecip] = Ar::recip(b), ob[kDivR] = Ar::div_real(a, sr), ob[kScaleR] = Ar::scale_real(sr, a);
        };
        one(device::arith::Explicit{}, 0);
        one(device::arith::ExplicitInline{}, 1);
        one(device::arith::Native{}, 2);
        one(device::arith::Fma{}, 3);
        T* ob = o + (std::ptrdiff_t(i) * (P + 1) + P) * kArSlots;
        if constexpr (is_cplx<T>) {
            using Cx = sycl_device::Cx<R>;
            const Cx ca{a.real(), a.imag()}, cb{b.real(), b.imag()};
            Cx m1{c.real(), c.imag()}, m2 = m1;
            sycl_device::fma_acc(m1, ca, cb);
            sycl_device::fma_acc(m2, sycl_device::dev_conj(ca), cb);
            auto put = [&](int k, Cx v) { ob[k] = T(v.re, v.im); };
            put(kMul, sycl_device::dev_mul(ca, cb)), put(kMac, m1), put(kMacC, m2), put(kDiv, sycl_device::dev_div(ca, cb)), put(kRecip, sycl_device::dev_recip(cb));
        } else {
            T m1 = c;
            sycl_device::fma_acc(m1, a, b);
            ob[kMul] = sycl_device::dev_mul(a, b), ob[kMac] = m1, ob[kMacC] = m1, ob[kDiv] = sycl_device::dev_div(a, b), ob[kRecip] = sycl_device::dev_recip(b);
        }
    });
    const auto got = this->dev.down(o, ho.size());
    int fma_bits = 0, special = 0;
    double worst = 0;
    for (int i = 0; i < n; ++i) {
        const LD<T> a = wide(h[i]), b = wide(h[n + i]), c = wide(h[2 * n + i]);
        const long double sr = std::real(b);
        const LD<T> want[kArSlots] = {a * b, c + a * b, c + conj_h(a) * b, c - a * b, a / b, LD<T>(1) / b, a / sr, sr * a};
        const long double scale[kArSlots] = {std::abs(a) * std::abs(b), std::abs(c) + std::abs(a) * std::abs(b),
                                             std::abs(c) + std::abs(a) * std::abs(b), std::abs(c) + std::abs(a) * std::abs(b),
                                             std::abs(a / b), std::abs(LD<T>(1) / b), std::abs(a / sr), std::abs(sr * a)};
        for (int pol = 0; pol < P; ++pol)
            for (int k = 0; k < kArSlots; ++k) {
                if (i == 0 && k != kDiv && k != kRecip && k != kDivR) continue;  // big * big overflows by design
                const T g = got[(std::size_t(i) * (P + 1) + pol) * kArSlots + k];
                worst = std::max(worst, rel(g, want[k], scale[k]));
            }
        for (int k : {kMul, kMac, kMacC, kDiv, kRecip})
            fma_bits += !same_bits(got[(std::size_t(i) * (P + 1) + 3) * kArSlots + k], got[(std::size_t(i) * (P + 1) + P) * kArSlots + k]);
    }
    for (int pol : {0, 1, 3}) special += !(std::abs(got[std::size_t(pol) * kArSlots + kDiv]) < R(2));  // Smith, not naive
    EXPECT_VERIFY(T, Check::blas, 4, worst);
    EXPECT_EQ(fma_bits, 0) << "arith::Fma must reproduce sycl_device::dev_mul / fma_acc / dev_div / dev_recip";
    EXPECT_EQ(special, 0) << "a quotient of huge operands must not overflow";
}

// Arith<sycl_device::Cx> and FmaArith<Cx> give the std::complex specialisations' bits on every
// op, div_real / scale_real the dev_* bits, is_zero / isfinite the dev_* answers. Inputs hold
// signed zeros, Inf and NaN parts and both Smith branches; all forms run in one kernel.
enum CxSlot { kCxMsub, kCxDiv, kCxRecip, kCxDivR, kCxScaleR, kCxSlots };
TYPED_TEST(DeviceBlasV2Test, PodComplexArithMatchesStdComplexAndDeviceScalar) {
    using T = TypeParam;
    if constexpr (is_cplx<T>) {
        using R = real_of<T>;
        constexpr int n = 64, S = 5;  // sets: Arith<Cx>, Arith<T>, FmaArith<Cx>, FmaArith<T>, dev_*
        auto h = batched<T>(3 * n, 131, [](T*) {}, kPeriod);
        const R inf = std::numeric_limits<R>::infinity(), nan = std::numeric_limits<R>::quiet_NaN();
        const T special[] = {T(0, 0), T(R(-0.0), 0), T(0, R(-0.0)), T(inf, 1), T(1, -inf), T(nan, 0), T(0, nan), T(R(0.5), R(7))};
        constexpr int ns = int(std::size(special));
        for (int k = 0; k < ns; ++k) h[k] = special[k], h[n + ns + k] = special[k];
        T* x = this->dev.up(h);
        std::vector<T> ho(std::size_t(n) * S * kCxSlots, T(0));
        std::vector<int> hf(std::size_t(n) * 6, -1);
        T* o = this->dev.up(ho);
        int* f = this->dev.up(hf);
        this->dev.template run<32>(n / 32, 0, [=](sycl::nd_item<1> it, unsigned char*) {
            using Cx = sycl_device::Cx<R>;
            const int i = static_cast<int>(it.get_global_linear_id());
            const T a = x[i], b = x[n + i], c = x[2 * n + i];
            const Cx ca{a.real(), a.imag()}, cb{b.real(), b.imag()}, cc{c.real(), c.imag()};
            const R sr = b.real();
            auto slots = [&](auto ar, auto va, auto vb, auto vc, int set) {
                using Ar = decltype(ar);
                auto m = vc;
                Ar::msub(m, va, vb);
                const decltype(va) r[kCxSlots] = {m, Ar::div(va, vb), Ar::recip(vb), Ar::div_real(va, sr), Ar::scale_real(sr, va)};
                for (int k = 0; k < kCxSlots; ++k) o[(std::ptrdiff_t(i) * S + set) * kCxSlots + k] = T(Ar::re(r[k]), Ar::im(r[k]));
            };
            slots(device::Arith<Cx>{}, ca, cb, cc, 0);
            slots(device::Arith<T>{}, a, b, c, 1);
            slots(device::FmaArith<Cx>{}, ca, cb, cc, 2);
            slots(device::FmaArith<T>{}, a, b, c, 3);
            const Cx dr = sycl_device::dev_div_real(ca, sr), ds = sycl_device::dev_mul_real(ca, sr);
            o[(std::ptrdiff_t(i) * S + 4) * kCxSlots + kCxDivR] = T(dr.re, dr.im);
            o[(std::ptrdiff_t(i) * S + 4) * kCxSlots + kCxScaleR] = T(ds.re, ds.im);
            int* fi = f + 6 * i;
            fi[0] = device::Arith<Cx>::is_zero(ca), fi[1] = device::Arith<T>::is_zero(a), fi[2] = sycl_device::dev_is_zero(ca);
            fi[3] = device::Arith<Cx>::isfinite(ca), fi[4] = device::Arith<T>::isfinite(a), fi[5] = sycl_device::dev_isfinite(ca);
        });
        const auto got = this->dev.down(o, ho.size());
        const auto flags = this->dev.down(f, hf.size());
        auto at = [&](int i, int set, int k) { return got[(std::size_t(i) * S + set) * kCxSlots + k]; };
        int explicit_bits = 0, fma_bits = 0, dev_bits = 0, flag_bits = 0;
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < kCxSlots; ++k) {
                explicit_bits += !same_bits(at(i, 0, k), at(i, 1, k));
                fma_bits += !same_bits(at(i, 2, k), at(i, 3, k));
            }
            for (int k : {kCxDivR, kCxScaleR}) dev_bits += !same_bits(at(i, 0, k), at(i, 4, k));
            const T a = h[i];
            const int z = a.real() == R(0) && a.imag() == R(0), fin = std::isfinite(a.real()) && std::isfinite(a.imag());
            const int* fi = flags.data() + 6 * i;
            flag_bits += fi[0] != z || fi[1] != z || fi[2] != z || fi[3] != fin || fi[4] != fin || fi[5] != fin;
        }
        EXPECT_EQ(explicit_bits, 0) << "Arith<Cx> msub/div/recip/div_real/scale_real must give Arith<std::complex>'s bits";
        EXPECT_EQ(fma_bits, 0) << "FmaArith<Cx> must give FmaArith<std::complex>'s bits";
        EXPECT_EQ(dev_bits, 0) << "div_real / scale_real must give dev_div_real / dev_mul_real's bits";
        EXPECT_EQ(flag_bits, 0) << "is_zero / isfinite must agree with dev_is_zero / dev_isfinite and std::complex";
    } else {
        GTEST_SKIP() << "Arith<sycl_device::Cx> is the complex POD scalar";
    }
}

// lartg (all types), larfg_scalars (three conventions), lae2 / laev2 / las2 / wilkinson_shift
// (real) on device, against reference LAPACK on the host; inputs straddle the scaling
// thresholds (1e-30 .. 1e30 float, 1e-300 .. 1e300 double), zeros and both signs.
template <class R> std::vector<R> scalar_inputs() {
    const R t = std::is_same_v<R, float> ? R(1e-30) : R(1e-300), u = std::is_same_v<R, float> ? R(1e30) : R(1e300);
    return {R(0), R(1), R(-2.5), R(3), R(0.75), t, -t, u, -u, R(1e-3), R(-7e5), R(1) / R(3)};
}

TYPED_TEST(DeviceBlasV2Test, ScalarLapackMatchesReferenceLapack) {
    using T = TypeParam;
    using R = real_of<T>;
    const auto rv = scalar_inputs<R>();
    const int nr = int(rv.size());
    std::vector<T> in;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nr; ++j) {
            if constexpr (is_cplx<T>) in.push_back(T(rv[i], rv[(i + 3) % nr])), in.push_back(T(rv[j], -rv[(j + 5) % nr]));
            else in.push_back(rv[i]), in.push_back(rv[j]);
        }
    const int np = int(in.size()) / 2;
    constexpr int kx = 5;  // larfg: alpha = f, x = {g, f, 2g, -f, g/3}
    T* d_in = this->dev.up(in);
    std::vector<T> ho(std::size_t(np) * 32, T(0));
    T* o = this->dev.up(ho);
    this->dev.template run<32>((np + 31) / 32, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int i = static_cast<int>(it.get_global_linear_id());
        if (i >= np) return;
        using A = device::Arith<T>;
        const T f = d_in[2 * i], g = d_in[2 * i + 1];
        T* ob = o + std::ptrdiff_t(i) * 32;
        const auto lr = device::lartg(f, g);
        ob[0] = T(lr.c), ob[1] = lr.s, ob[2] = lr.r;
        const T xs[kx] = {g, f, A::add(g, g), A::neg(f), A::div_real(g, R(3))};
        R s = device::modulus(f) > R(0) ? sycl::fmax(sycl::fabs(A::re(f)), sycl::fabs(A::im(f))) : R(0);
        for (int k = 0; k < kx; ++k) s = sycl::fmax(s, sycl::fmax(sycl::fabs(A::re(xs[k])), sycl::fabs(A::im(xs[k]))));
        R ssq = 0, nrm2 = 0;
        for (int k = 0; k < kx; ++k) {
            if (s > R(0)) ssq += (A::re(xs[k]) / s) * (A::re(xs[k]) / s) + (A::im(xs[k]) / s) * (A::im(xs[k]) / s);
            nrm2 += A::re(xs[k]) * A::re(xs[k]) + A::im(xs[k]) * A::im(xs[k]);
        }
        const auto hl = device::larfg_scalars<device::conv::Lapack>(f, device::ScaledSsq<R>{s, ssq});
        ob[3] = hl.tau, ob[4] = hl.beta;
        for (int k = 0; k < kx; ++k) ob[5 + k] = hl.identity ? xs[k] : (hl.use_mul ? device::FmaArith<T>::mul(xs[k], hl.scale) : device::FmaArith<T>::div(xs[k], hl.scale));
        const R xn = s * sycl::sqrt(ssq);
        const auto hr = device::larfg_scalars<device::conv::RealBeta>(f, xn, kx + 1);
        const auto hp = device::larfg_scalars<device::conv::Phase>(f, xn, kx + 1);
        ob[10] = hr.tau, ob[11] = hr.beta, ob[12] = hr.scale, ob[13] = hp.tau, ob[14] = hp.beta, ob[15] = hp.scale;
        if constexpr (!is_cplx<T>) {
            const T h3 = A::add(f, g);
            const auto e2 = device::lae2(f, g, h3);
            const auto ev = device::laev2(f, g, h3);
            const auto s2 = device::las2(f, g, h3);
            ob[16] = e2[0], ob[17] = e2[1], ob[18] = ev[0], ob[19] = ev[1], ob[20] = ev[2], ob[21] = ev[3], ob[22] = s2[0], ob[23] = s2[1];
            if (g != T(0)) ob[24] = device::wilkinson_shift(f, g, h3);
        }
        (void)nrm2;
    });
    const auto got = this->dev.down(o, ho.size());
    const long double eps = std::numeric_limits<R>::epsilon();
    auto close = [&](T a, T b, long double scale, double ulps) {
        if (!std::isfinite(std::abs(LD<T>(b)))) return std::abs(LD<T>(a)) == std::abs(LD<T>(b));
        return std::abs(wide(a) - wide(b)) <= ulps * eps * std::max(scale, (long double)std::numeric_limits<R>::min());
    };
    int bad_lartg = 0, bad_lapack = 0, bad_realbeta = 0, bad_phase = 0, bad_2x2 = 0;
    std::string first_bad, first_2x2;
    for (int i = 0; i < np; ++i) {
        const T f = in[2 * i], g = in[2 * i + 1];
        const T* ob = got.data() + std::size_t(i) * 32;
        {  // lartg: a unit rotation that zeroes g, with LAPACK 3.10's sign of r (sgn(f) r >= 0).
            const long double c = std::real(wide(ob[0]));
            const LD<T> s = wide(ob[1]), r = wide(ob[2]), F = wide(f), G = wide(g);
            const long double fg = std::abs(F) + std::abs(G);
            bool ok = c >= 0 && c <= 1 && std::abs(c * c + std::norm(s) - 1) <= 8 * eps;
            ok = ok && std::abs(c * F + s * G - r) <= 8 * eps * fg && std::abs(-conj_h(s) * F + c * G) <= 8 * eps * fg;
            if (std::abs(G) == 0) ok = ok && c == 1 && std::abs(s) == 0 && same_bits(ob[2], f);  // G = 0: identity, r = f
            else if (std::abs(F) > 0) ok = ok && std::abs(r / std::abs(r) - F / std::abs(F)) <= 8 * eps;
            else ok = ok && c == 0 && std::imag(r) == 0 && std::real(r) >= 0;
            bad_lartg += !ok;
            if (!ok && first_bad.empty()) first_bad = "lartg(" + std::to_string(std::real(f)) + ", " + std::to_string(std::real(g)) + ")";
        }
        // xLARFG with the same (alpha, x).
        T alpha = f, tau;
        T xs[kx] = {g, f, T(g + g), T(-f), T(g / R(3))};
        long double xs_abs = std::abs(wide(f));
        for (const T& e : xs) xs_abs += std::abs(wide(e));
        if (lapack_ref::larfg(kx + 1, alpha, xs, tau)) {
            bool ok = close(ob[3], tau, 2, 16) && close(ob[4], alpha, xs_abs, 16);
            for (int k = 0; k < kx; ++k) ok = ok && close(ob[5 + k], xs[k], std::abs(wide(xs[k])) + 1, 64);
            bad_lapack += !ok;
            bad_realbeta += !close(ob[10], tau, 2, 16) || !close(ob[11], alpha, xs_abs, 16);
        }
        // Phase: (I - tau v v^H)^H [f; x] = [beta; 0] with v = [1; scale * x] and real tau.
        {
            const LD<T> tp = wide(ob[13]), bp = wide(ob[14]), sp = wide(ob[15]);
            const T x0[kx] = {g, f, T(g + g), T(-f), T(g / R(3))};
            LD<T> vhw = wide(f);
            for (int k = 0; k < kx; ++k) vhw += conj_h(sp * wide(x0[k])) * wide(x0[k]);
            const LD<T> head = wide(f) - conj_h(tp) * vhw;
            long double tail = 0;
            for (int k = 0; k < kx; ++k) tail = std::max(tail, std::abs(wide(x0[k]) - conj_h(tp) * sp * wide(x0[k]) * vhw));
            const bool fin = std::isfinite(std::abs(tp)) && std::isfinite(std::abs(bp));
            bad_phase += fin && (std::abs(head - bp) > 64 * eps * xs_abs || tail > 64 * eps * xs_abs || std::abs(std::imag(tp)) > 8 * eps);
        }
        if constexpr (!is_cplx<T>) {  // the 2x2 routines against long-double closed forms
            const long double A = f, B = g, C = T(f + g);
            const long double mid = (A + C) / 2, rad = std::hypot((A - C) / 2, B);
            const long double e1 = mid + rad, e2 = mid - rad;
            const long double big = std::abs(e1) >= std::abs(e2) ? e1 : e2, small = std::abs(e1) >= std::abs(e2) ? e2 : e1;
            const long double sc = std::abs(A) + std::abs(B) + std::abs(C);
            bool ok2 = std::abs(ob[16] - big) <= 8 * eps * sc && std::abs(ob[17] - small) <= 8 * eps * sc;
            ok2 = ok2 && std::abs(ob[18] - big) <= 8 * eps * sc && std::abs(ob[19] - small) <= 8 * eps * sc;
            const long double cs = ob[20], sn = ob[21];  // (cs, sn): the unit eigenvector of rt1
            ok2 = ok2 && std::abs(cs * cs + sn * sn - 1) <= 8 * eps;
            ok2 = ok2 && std::abs(A * cs + B * sn - ob[18] * cs) <= 16 * eps * sc && std::abs(B * cs + C * sn - ob[18] * sn) <= 16 * eps * sc;
            const long double p = A * A + B * B + C * C, q = std::abs(A * C);  // [[f, g], [0, h]] with h = f + g
            // (p - 2q)(p + 2q) without cancellation: p -+ 2q = (|f| -+ |h|)^2 + g^2.
            const long double dm = (std::abs(A) - std::abs(C)) * (std::abs(A) - std::abs(C)) + B * B;
            const long double dp = (std::abs(A) + std::abs(C)) * (std::abs(A) + std::abs(C)) + B * B;
            const long double smax = std::sqrt((p + std::sqrt(dm * dp)) / 2), smin = smax > 0 ? q / smax : 0;
            ok2 = ok2 && std::abs(ob[22] - smin) <= 8 * eps * smax && std::abs(ob[23] - smax) <= 8 * eps * smax;
            if (B != 0 && std::isfinite(e1) && std::isfinite(e2)) {
                const long double near = std::abs(e1 - A) < std::abs(e2 - A) ? e1 : e2;
                ok2 = ok2 && std::abs(ob[24] - near) <= 16 * eps * sc;
            }
            bad_2x2 += !ok2;
            if (!ok2 && first_2x2.empty())
                first_2x2 = "2x2(" + std::to_string(double(f)) + ", " + std::to_string(double(g)) + "): " + std::to_string(double(ob[16])) + " " +
                            std::to_string(double(ob[17])) + " " + std::to_string(double(ob[20])) + " " + std::to_string(double(ob[21])) + " " +
                            std::to_string(double(ob[22])) + " " + std::to_string(double(ob[23])) + " " + std::to_string(double(ob[24]));
        }
    }
    EXPECT_EQ(bad_lartg, 0) << "lartg: unit rotation zeroing g, LAPACK sign of r; first: " << first_bad;
    EXPECT_EQ(bad_lapack, 0) << "larfg_scalars<Lapack> vs ?larfg";
    EXPECT_EQ(bad_realbeta, 0) << "larfg_scalars<RealBeta> vs ?larfg";
    EXPECT_EQ(bad_phase, 0) << "larfg_scalars<Phase> reflects [alpha; x] onto [beta; 0]";
    EXPECT_EQ(bad_2x2, 0) << "lae2 / laev2 / las2 / wilkinson_shift vs long-double closed forms; first: " << first_2x2;
}

// ---- reductions --------------------------------------------------------------------------------

// Per item: sumsq (Fused, and Unfused over two operands in one loop), nrm2 Blue / Unscaled /
// Scaled, find_first, argmax under each tie / NaN / key policy, amax_by<Skip>, allreduce of an
// array, on WorkGroup, RtGroup @96, Teams<128, 16> (one item per team) and Thread. Items
// b % 3 == 1 carry a NaN at index 40 and a tie of the maximum at 100 / 500; items b % 3 == 2
// are scaled to 1e30 (float) / 1e300 (double), where the unscaled sum overflows.
enum R2 { kSsqF, kSsqU0, kSsqU1, kBlue, kUnsc, kScal, kFirst, kArgLo, kArgHi, kArgSkip, kArgAbs, kArgRe, kAmaxSkip, kAll0, kAll1, kR2 };
template <class T, int WG, class MakeEx>
void reductions21_case(Dev& dev, MakeEx make_ex, int items_per_wg, int lanes_per_item, const char* where) {
    using R = real_of<T>;
    constexpr int n = 777;
    const R big = std::is_same_v<R, float> ? R(1e30) : R(1e300);
    auto hx = batched<T>(n, 141, [&](T* x) {
        static int rep = 0;
        if (rep % 3 == 1) {
            x[40] = T(std::numeric_limits<R>::quiet_NaN());
            x[100] = T(R(5)), x[500] = T(R(-5));  // the abs1 maximum, tied
        } else if (rep % 3 == 2) {
            for (int i = 0; i < n; ++i) x[i] = T(x[i] * big);
        }
        ++rep;
    });
    auto hy = batched<T>(n, 142);
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    std::vector<T> ho(std::size_t(kBatch) * kR2, T(-9));
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch / items_per_wg, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id()) * items_per_wg + static_cast<int>(it.get_local_linear_id()) / lanes_per_item;
        const auto ex = make_ex(it);
        const auto xv = device::vec(x + std::ptrdiff_t(b) * n, n), yv = device::vec(y + std::ptrdiff_t(b) * n, n);
        const R sf = device::sumsq(ex, yv);
        const auto su = device::sumsq<device::alg::Unfused>(ex, yv, xv);
        const R bl = device::nrm2<device::alg::Blue>(ex, xv), un = device::nrm2<device::alg::Unscaled>(ex, yv), sc = device::nrm2<device::alg::Scaled>(ex, xv);
        const int ff = device::find_first(ex, xv, [](const T& v) { return device::abs1(v) > real_of<T>(0.99); });
        const auto alo = device::argmax(ex, xv);
        const auto ahi = device::argmax<device::tie::Highest, device::nan::Skip>(ex, xv);
        const auto ask = device::argmax<device::tie::Lowest, device::nan::Skip>(ex, xv);
        const auto aab = device::argmax<device::tie::Lowest, device::nan::Skip, device::key::Abs>(ex, yv);
        const auto are = device::argmax<device::tie::Lowest, device::nan::Skip, device::key::Real>(ex, yv);
        const R ams = device::amax_by<device::nan::Skip>(ex, xv);
        const auto all = device::allreduce(ex, std::array<R, 2>{R(ex.lid() + 1), R(1)});
        if (ex.leader()) {
            T* ob = o + std::ptrdiff_t(b) * kR2;
            ob[kSsqF] = T(sf), ob[kSsqU0] = T(su[0]), ob[kSsqU1] = T(su[1]), ob[kBlue] = T(bl), ob[kUnsc] = T(un), ob[kScal] = T(sc);
            ob[kFirst] = T(R(ff)), ob[kArgLo] = T(R(alo.index)), ob[kArgHi] = T(R(ahi.index)), ob[kArgSkip] = T(R(ask.index));
            ob[kArgAbs] = T(R(aab.index)), ob[kArgRe] = T(R(are.index)), ob[kAmaxSkip] = T(ams), ob[kAll0] = T(all[0]), ob[kAll1] = T(all[1]);
        }
    });
    const auto got = dev.down(o, ho.size());
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int k = 0; k < kR2; ++k) bitdiff += !same_bits(got[std::size_t(b) * kR2 + k], got[std::size_t(b % kPeriod) * kR2 + k]);
    EXPECT_EQ(bitdiff, 0) << where;
    for (int b = 0; b < kPeriod; ++b) {
        const T* xb = hx.data() + std::size_t(b) * n;
        const T* yb = hy.data() + std::size_t(b) * n;
        const T* r = got.data() + std::size_t(b) * kR2;
        long double ssy = 0, ssx = 0;
        int ff = -1, alo = -1, ahi = -1, ask = -1, aab = -1, are = -1, nan_at = -1;
        long double blo = -1, bsk = -1, bab = -1, bre = -std::numeric_limits<long double>::infinity(), amax = 0;
        for (int i = 0; i < n; ++i) {
            const LD<T> xi = wide(xb[i]), yi = wide(yb[i]);
            ssy += std::norm(yi);
            const long double a1 = std::abs(std::real(xi)) + std::abs(std::imag(xi));
            if (std::isnan(a1)) {
                if (nan_at < 0) nan_at = i;
                continue;
            }
            ssx += std::norm(xi);
            if (ff < 0 && a1 > 0.99L) ff = i;
            if (a1 > blo) blo = a1, alo = i;
            if (a1 >= blo) ahi = i;
            if (a1 > bsk) bsk = a1, ask = i;
            amax = std::max(amax, a1);
            if (std::abs(yi) > bab) bab = std::abs(yi), aab = i;
            if (std::real(yi) > bre) bre = std::real(yi), are = i;
        }
        if (nan_at >= 0) alo = nan_at;
        const long double nx = std::sqrt(ssx), ny = std::sqrt(ssy);
        EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kSsqF]), LD<T>(ssy), ssy)) << where << " sumsq Fused";
        EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kSsqU0]), LD<T>(ssy), ssy)) << where << " sumsq Unfused";
        if (nan_at < 0) {
            if (b == 0)  // sumsq is unscaled: the 1e30 / 1e300 item overflows it by design
                EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kSsqU1]), LD<T>(ssx), ssx)) << where << " sumsq Unfused x";
            else
                EXPECT_TRUE(std::isinf(std::real(r[kSsqU1]))) << where << " sumsq of the scaled item overflows (the threshold is straddled)";
            EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kBlue]), LD<T>(nx), nx)) << where << " nrm2 Blue (item " << b << ")";
            EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kScal]), LD<T>(nx), nx)) << where << " nrm2 Scaled";
        } else {
            EXPECT_TRUE(std::isnan(std::real(r[kBlue]))) << where << " nrm2 Blue propagates NaN";
        }
        EXPECT_VERIFY(T, Check::blas, n, rel(T(r[kUnsc]), LD<T>(ny), ny)) << where << " nrm2 Unscaled";
        EXPECT_EQ(int(std::real(r[kFirst])), ff) << where << " find_first";
        EXPECT_EQ(int(std::real(r[kArgLo])), alo) << where << " argmax Lowest/First";
        EXPECT_EQ(int(std::real(r[kArgHi])), ahi) << where << " argmax Highest/Skip (the tie of item 1)";
        EXPECT_EQ(int(std::real(r[kArgSkip])), ask) << where << " argmax Skip";
        EXPECT_EQ(int(std::real(r[kArgAbs])), aab) << where << " argmax key::Abs";
        EXPECT_EQ(int(std::real(r[kArgRe])), are) << where << " argmax key::Real";
        EXPECT_EQ(std::real(r[kAmaxSkip]), R(amax)) << where << " amax_by<Skip> is exact";
        const int w = make_ex.width;
        EXPECT_EQ(std::real(r[kAll0]), R(w * (w + 1) / 2)) << where << " allreduce[0]";
        EXPECT_EQ(std::real(r[kAll1]), R(w)) << where << " allreduce[1]";
    }
}

TYPED_TEST(DeviceBlasV2Test, ReductionsV21OnEveryExecutor) {
    using T = TypeParam;
    struct Wg { int width = 128; auto operator()(sycl::nd_item<1> it) const { return device::WorkGroup<128>{it}; } };
    struct Rt { int width = 96; auto operator()(sycl::nd_item<1> it) const { return device::RtGroup<>(it); } };
    struct Tm { int width = 16; auto operator()(sycl::nd_item<1> it) const { return device::Teams<128, 16>{it}; } };
    struct Th { int width = 1; auto operator()(sycl::nd_item<1>) const { return device::Thread{}; } };
    reductions21_case<T, 128>(this->dev, Wg{}, 1, 128, "WorkGroup<128>");
    reductions21_case<T, 96>(this->dev, Rt{}, 1, 96, "RtGroup @96");
    reductions21_case<T, 128>(this->dev, Tm{}, 8, 16, "Teams<128, 16>");
    reductions21_case<T, 32>(this->dev, Th{}, 32, 1, "Thread");
}

// The packed exchange in a kernel WITHOUT reqd_sub_group_size: on a device whose sub-groups
// are not 32 lanes (the CPU device, AdaptiveCpp) the contract check must take the portable
// path; on NVPTX the 32-lane form is a compile-time fact. dotN<3> and allreduce with Scratch.
TYPED_TEST(DeviceBlasV2Test, PackedExchangeHonoursTheSubGroupContract) {
    using T = TypeParam;
    constexpr int WG = 128, n = 333;
    auto hx = batched<T>(n, 151), hy = batched<T>(n, 152);
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    std::vector<T> ho(std::size_t(kBatch) * 5, T(0));
    T* o = this->dev.up(ho);
    constexpr int slots = device::scratch_elements<device::op::ReduceN<3>, device::WorkGroup<WG>, T>();
    this->dev.q()
        .submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 1> lm(sycl::range<1>(slots), h);
            h.parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * WG, WG), [=](sycl::nd_item<1> it) {
                const device::WorkGroup<WG> ex{it};
                const int b = static_cast<int>(it.get_group_linear_id());
                const auto xv = device::vec(x + std::ptrdiff_t(b) * n, n), yv = device::vec(y + std::ptrdiff_t(b) * n, n);
                const device::Scratch<T, slots> scr{batchlas::util::get_raw_ptr(lm)};
                const auto d3 = device::dotN<3>(ex, scr, std::array<decltype(xv), 3>{xv, yv, xv}, yv);
                const auto a2 = device::allreduce(ex, scr, std::array<T, 2>{T(real_of<T>(1)), xv(ex.lid())});
                if (ex.leader()) {
                    T* ob = o + std::ptrdiff_t(b) * 5;
                    ob[0] = d3[0], ob[1] = d3[1], ob[2] = d3[2], ob[3] = a2[0], ob[4] = a2[1];
                }
            });
        })
        .wait_and_throw();
    const auto got = this->dev.down(o, ho.size());
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* xb = hx.data() + std::size_t(b) * n;
        const T* yb = hy.data() + std::size_t(b) * n;
        LD<T> xy = 0, yy = 0, x128 = 0;
        long double s = 0, syy = 0, s128 = 0;
        for (int i = 0; i < n; ++i) {
            xy += wide(xb[i]) * wide(yb[i]), yy += wide(yb[i]) * wide(yb[i]);
            s += std::abs(wide(xb[i])) * std::abs(wide(yb[i])), syy += std::norm(wide(yb[i]));
            if (i < WG) x128 += wide(xb[i]), s128 += std::abs(wide(xb[i]));
        }
        const T* r = got.data() + std::size_t(b) * 5;
        worst = std::max({worst, rel(r[0], xy, s), rel(r[1], yy, syy), rel(r[2], xy, s), rel(r[4], x128, s128)});
        bad += std::real(r[3]) != real_of<T>(WG);
    }
    EXPECT_VERIFY(T, Check::blas, n, worst);
    EXPECT_EQ(bad, 0);
}

// ---- LAPACK auxiliaries --------------------------------------------------------------------------

// lacpy (Lower; ConjTrans of a square block), laset (StrictUpper), mirror (Lower -> Upper,
// Hermitian), laswp forward and backward with repeated and identity pivots: every item against
// a host replay, bit for bit (these ops move and conjugate, they do not round).
TYPED_TEST(DeviceBlasV2Test, LacpyLasetMirrorLaswpMatchAHostReplay) {
    using T = TypeParam;
    constexpr int WG = 64, m = 13, n = 11, ld = m + 3, per = ld * n;
    const auto ha = batched<T>(per, 161);
    T* a = this->dev.up(ha);
    std::vector<T> hb(std::size_t(kBatch) * per * 5, T(-1));
    T* bb = this->dev.up(hb);
    std::vector<int> hp = {3, 2, 4, 5, 9, 6, 6};  // 1-based pivots for rows 0..6 (k2 = 7); 2-3-4-8 chain
    int* piv = this->dev.up(hp);
    const T alpha = T(real_of<T>(0.25)), beta = T(real_of<T>(-4));
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const device::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto av = device::mat(a + std::ptrdiff_t(b) * per, m, n, ld);
        T* ob = bb + std::ptrdiff_t(b) * per * 5;
        const auto b0 = device::mat(ob, m, n, ld), b1 = device::mat(ob + per, n, n, ld), b2 = device::mat(ob + 2 * per, m, n, ld);
        const auto b3 = device::mat(ob + 3 * per, n, n, ld), b4 = device::mat(ob + 4 * per, m, n, ld);
        device::lacpy<device::Part::Lower>(ex, av, b0);
        device::lacpy<device::Part::Full, Transpose::ConjTrans>(ex, av.sub(0, 0, n, n), b1);
        device::lacpy(ex, av, b2);
        device::lacpy(ex, av.sub(0, 0, n, n), b3);
        device::lacpy(ex, av, b4);
        ex.barrier(device::Mem::Global);
        device::laset<device::Part::Upper>(ex, alpha, beta, b2);
        device::mirror<Uplo::Lower, true>(ex, b3);
        const auto pv = device::vec(piv, 7);
        device::laswp(ex, b4, pv, 0, 7);
        device::laswp<device::dir::Backward>(ex, b4.sub(0, 0, m, 5), pv, 2, 6);
    });
    const auto got = this->dev.down(bb, hb.size());
    int bad[5] = {0, 0, 0, 0, 0};
    for (int b = 0; b < kBatch; ++b) {
        const T* A = ha.data() + std::size_t(b) * per;
        std::vector<T> want(std::size_t(per) * 5, T(-1));
        auto W = [&](int k, int i, int j) -> T& { return want[std::size_t(k) * per + i + j * ld]; };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                if (i >= j) W(0, i, j) = A[i + j * ld];
                W(2, i, j) = A[i + j * ld];
                if (i < j) W(2, i, j) = alpha;
                if (i == j) W(2, i, j) = beta;
                W(4, i, j) = A[i + j * ld];
            }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                W(1, i, j) = conj_h(A[j + i * ld]);
                W(3, i, j) = i < j ? conj_h(A[j + i * ld]) : A[i + j * ld];
                if constexpr (is_cplx<T>)
                    if (i == j) W(3, i, j) = T(std::real(A[i + j * ld]), 0);
            }
        auto swaprows = [&](int k, int p, int cols) {
            for (int j = 0; j < cols; ++j) std::swap(W(4, k, j), W(4, p, j));
        };
        for (int k = 0; k < 7; ++k) swaprows(k, hp[k] - 1, n);
        for (int k = 5; k >= 2; --k) swaprows(k, hp[k] - 1, 5);
        for (int k = 0; k < 5; ++k)
            for (int e = 0; e < per; ++e) bad[k] += !same_bits(got[std::size_t(b) * per * 5 + std::size_t(k) * per + e], want[std::size_t(k) * per + e]);
    }
    EXPECT_EQ(bad[0], 0) << "lacpy<Lower>";
    EXPECT_EQ(bad[1], 0) << "lacpy<Full, ConjTrans>";
    EXPECT_EQ(bad[2], 0) << "laset<Upper>";
    EXPECT_EQ(bad[3], 0) << "mirror<Lower, Herm>";
    EXPECT_EQ(bad[4], 0) << "laswp forward + backward";
}

// rot (Lapack convention, from lartg) zeroes y(0) of the pair; lasr_rows (Transposed) against
// a host replay; larfg<Lapack> on a WorkGroup against reference ?larfg; larf Left (team
// executor) and Right (row owner) against H C formed on the host in long double.
TYPED_TEST(DeviceBlasV2Test, RotLasrLarfgLarfMatchReferences) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 128, m = 38, nc = 9, ld = m + 2;  // reflector length m = 1 + 37
    const auto hc = batched<T>(ld * nc, 171);
    const auto hv = batched<T>(m, 172);
    T* c = this->dev.up(hc);
    T* v = this->dev.up(hv);
    std::vector<T> hcl(hc), hcr(hc), hv2(hv), hrot(std::size_t(kBatch) * 2 * nc, T(0)), htau(kBatch * 2, T(0));
    T* cl = this->dev.up(hcl);
    T* cr = this->dev.up(hcr);
    T* v2p = this->dev.up(hv2);
    T* ro = this->dev.up(hrot);
    T* tau = this->dev.up(htau);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const device::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        // rot from lartg on the first two columns' leading entries, applied along the rows.
        const auto cv = device::mat(c + std::ptrdiff_t(b) * ld * nc, m, nc, ld);
        const auto g = device::lartg(T(cv(0, 0)), T(cv(1, 0)));
        T* rb = ro + std::ptrdiff_t(b) * 2 * nc;
        device::for_each_owned(ex, nc, [&](int j) { rb[j] = cv(0, j), rb[nc + j] = cv(1, j); });
        const auto r0 = device::vec(rb, nc), r1 = device::vec(rb + nc, nc);
        device::rot(ex, device::Givens<R, T>{g.c, g.s}, r0, r1);
        // larfg on (v(0), v(1:)), then larf with H = I - tau u u^H, u = unit_head(v, 0).
        T* vb = v2p + std::ptrdiff_t(b) * m;
        T alpha = vb[0];
        const auto h = device::larfg(ex, alpha, device::vec(vb + 1, m - 1));
        ex.barrier(device::Mem::Global);
        if (ex.leader()) vb[0] = alpha, tau[2 * b] = h.tau, tau[2 * b + 1] = T(R(h.identity));
        const auto u = device::unit_head(device::vec(vb, m), 0);
        device::larf<Side::Left>(ex, h.tau, u, device::mat(cl + std::ptrdiff_t(b) * ld * nc, m, nc, ld));
        device::larf<Side::Right>(ex, h.tau, device::assume_published(device::unit_head(device::vec(vb, nc), 0)), device::mat(cr + std::ptrdiff_t(b) * ld * nc, m, nc, ld));
    });
    const auto grot = this->dev.down(ro, hrot.size()), gv = this->dev.down(v2p, hv2.size()), gt = this->dev.down(tau, htau.size());
    const auto gl = this->dev.down(cl, hcl.size()), gr = this->dev.down(cr, hcr.size());
    const long double eps = std::numeric_limits<R>::epsilon();
    int bad_rot = 0, bad_larfg = 0;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* C = hc.data() + std::size_t(b) * ld * nc;
        // rot: y'(0) = 0 within rounding; the pair's norm is kept.
        long double nn = std::abs(wide(C[0])) + std::abs(wide(C[1]));
        bad_rot += std::abs(wide(grot[std::size_t(b) * 2 * nc + nc])) > 8 * eps * nn;
        for (int j = 0; j < nc; ++j) {
            const long double before = std::norm(wide(C[j * ld])) + std::norm(wide(C[1 + j * ld]));
            const long double after = std::norm(wide(grot[std::size_t(b) * 2 * nc + j])) + std::norm(wide(grot[std::size_t(b) * 2 * nc + nc + j]));
            bad_rot += std::abs(after - before) > 16 * eps * before;
        }
        // larfg vs reference.
        T alpha = hv[std::size_t(b) * m], rtau;
        std::vector<T> x(hv.begin() + std::size_t(b) * m + 1, hv.begin() + std::size_t(b + 1) * m);
        if (lapack_ref::larfg(m, alpha, x.data(), rtau)) {
            long double sc = 0;
            for (int i = 0; i < m; ++i) sc += std::abs(wide(hv[std::size_t(b) * m + i]));
            bad_larfg += std::abs(wide(gt[2 * b]) - wide(rtau)) > 32 * eps * 2;
            bad_larfg += std::abs(wide(gv[std::size_t(b) * m]) - wide(alpha)) > 32 * eps * sc;
            for (int i = 1; i < m; ++i) bad_larfg += std::abs(wide(gv[std::size_t(b) * m + i]) - wide(x[i - 1])) > 64 * eps * (1 + std::abs(wide(x[i - 1])));
        }
        // larf against H C / C H in long double, with the device's own (v, tau).
        std::vector<LD<T>> u(m);
        for (int i = 0; i < m; ++i) u[i] = i == 0 ? LD<T>(1) : wide(gv[std::size_t(b) * m + i]);
        const LD<T> t = wide(gt[2 * b]);
        for (int j = 0; j < nc; ++j) {
            LD<T> w = 0;
            long double sw = 0;
            for (int i = 0; i < m; ++i) w += conj_h(u[i]) * wide(C[i + j * ld]), sw += std::abs(u[i]) * std::abs(wide(C[i + j * ld]));
            for (int i = 0; i < m; ++i) {
                const LD<T> want = wide(C[i + j * ld]) - t * u[i] * w;
                worst = std::max(worst, rel(gl[std::size_t(b) * ld * nc + i + j * ld], want, std::abs(wide(C[i + j * ld])) + std::abs(t) * std::abs(u[i]) * sw));
            }
        }
        for (int i = 0; i < m; ++i) {
            LD<T> w = 0;
            long double sw = 0;
            for (int j = 0; j < nc; ++j) w += wide(C[i + j * ld]) * u[j], sw += std::abs(wide(C[i + j * ld])) * std::abs(u[j]);
            for (int j = 0; j < nc; ++j) {
                const LD<T> want = wide(C[i + j * ld]) - t * w * conj_h(u[j]);
                worst = std::max(worst, rel(gr[std::size_t(b) * ld * nc + i + j * ld], want, std::abs(wide(C[i + j * ld])) + std::abs(t) * sw));
            }
        }
    }
    EXPECT_EQ(bad_rot, 0) << "rot<Lapack> with lartg's (c, s)";
    EXPECT_EQ(bad_larfg, 0) << "larfg<Lapack> vs ?larfg";
    EXPECT_VERIFY(T, Check::blas, m, worst) << "larf Left / Right";
}

// lasr_rows (Transposed convention) against a host replay in long double.
TYPED_TEST(DeviceBlasV2Test, LasrRowsAppliesTheRotationChain) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 32, m = 21, n = 8, ld = m + 1;
    const auto ha = batched<T>(ld * n, 181);
    std::vector<R> hg(2 * (n - 1));
    for (int j = 0; j < n - 1; ++j) hg[2 * j] = std::cos(R(0.3) * (j + 1)), hg[2 * j + 1] = std::sin(R(0.3) * (j + 1));
    T* a = this->dev.up(ha);
    R* g = this->dev.up(hg);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const device::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        auto rots = [&](int j) { return device::Givens<R>{g[2 * j], g[2 * j + 1]}; };
        device::lasr_rows(ex, rots, device::mat(a + std::ptrdiff_t(b) * ld * n, m, n, ld));
    });
    const auto got = this->dev.down(a, ha.size());
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < m; ++i) {
            std::vector<LD<T>> row(n);
            long double s = 0;
            for (int j = 0; j < n; ++j) row[j] = wide(ha[std::size_t(b) * ld * n + i + j * ld]), s += std::abs(row[j]);
            for (int j = 0; j < n - 1; ++j) {
                const long double cc = hg[2 * j], ss = hg[2 * j + 1];
                const LD<T> x = row[j], y = row[j + 1];
                row[j] = cc * x - ss * y, row[j + 1] = ss * x + cc * y;
            }
            for (int j = 0; j < n; ++j) worst = std::max(worst, rel(got[std::size_t(b) * ld * n + i + j * ld], row[j], s));
        }
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// trsv / trmv on one item for every (Uplo, Trans, Diag), on a register Frag<T, 8> with 6 live
// entries and on a memory vector of 13; gemv_sub and gemv_acc (two terms) on a WorkGroup. All
// against long-double host references; the opposite triangle and (Unit) the diagonal hold a
// poison that must not be read.
template <Uplo UL, Transpose TR, Diag DG, class T> void tri_case(Dev& dev, int& bad_frag, double& worst) {
    using R = real_of<T>;
    constexpr int WG = 32, N = 8, nf = 6, nm = 13, ld = nm + 2, per = ld * nm;
    auto ha = batched<T>(per, 191, [](T* a) {
        for (int j = 0; j < nm; ++j)
            for (int i = 0; i < nm; ++i) {
                const bool stored = UL == Uplo::Lower ? i >= j : i <= j;
                if (!stored || (DG == Diag::Unit && i == j)) a[i + j * ld] = T(R(1e6));
                else if (i == j) a[i + j * ld] = T(a[i + j * ld] + T(R(4)));  // well conditioned
            }
    });
    const auto hx = batched<T>(2 * nm, 192);
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    std::vector<T> ho(std::size_t(kBatch) * 4 * nm, T(0));
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch / WG, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto av = device::mat(a + std::ptrdiff_t(b) * per, nm, nm, ld);
        const T* xb = x + std::ptrdiff_t(b) * 2 * nm;
        T* ob = o + std::ptrdiff_t(b) * 4 * nm;
        device::Frag<T, N, device::Cyclic> f;
        const T tail_poison = T(real_of<T>(1e6));  // entries past n: never read, never written
        device::load(f, device::vec(const_cast<T*>(xb), nf), nf);
#pragma unroll
        for (int k = nf; k < N; ++k) f[k] = tail_poison;
        device::trsv<UL, TR, DG>(device::Thread{}, av, f, nf);
        device::store(f, device::vec(ob, N));
        device::load(f, device::vec(const_cast<T*>(xb), nf), nf);
#pragma unroll
        for (int k = nf; k < N; ++k) f[k] = tail_poison;
        device::trmv<UL, TR, DG>(device::Thread{}, av, f, nf);
        device::store(f, device::vec(ob + nm, N));
        for (int i = 0; i < nm; ++i) ob[2 * nm + i] = xb[nm + i], ob[3 * nm + i] = xb[nm + i];
        device::trsv<UL, TR, DG>(device::Thread{}, av, device::vec(ob + 2 * nm, nm));
        device::trmv<UL, TR, DG>(device::Thread{}, av, device::vec(ob + 3 * nm, nm));
    });
    const auto got = dev.down(o, ho.size());
    for (int b = 0; b < kPeriod; ++b) {
        const T* A = ha.data() + std::size_t(b) * per;
        auto op = [&](int i, int j) -> LD<T> {
            if (i == j && DG == Diag::Unit) return LD<T>(1);
            const int r = TR == Transpose::NoTrans ? i : j, c = TR == Transpose::NoTrans ? j : i;
            const bool stored = UL == Uplo::Lower ? r >= c : r <= c;
            if (!stored) return LD<T>(0);
            const LD<T> e = wide(A[r + c * ld]);
            return TR == Transpose::ConjTrans ? conj_h(e) : e;
        };
        for (int which = 0; which < 2; ++which) {
            const int n = which == 0 ? nf : nm;
            const T* x0 = hx.data() + std::size_t(b) * 2 * nm + (which == 0 ? 0 : nm);
            const T* gs = got.data() + std::size_t(b) * 4 * nm + (which == 0 ? 0 : 2 * nm);
            const T* gm = got.data() + std::size_t(b) * 4 * nm + (which == 0 ? nm : 3 * nm);
            for (int i = 0; i < n; ++i) {  // trmv: (op(A) x)(i); trsv: the residual (op(A) s - x)(i)
                LD<T> ax = 0, as = 0;
                long double sx = 0, ss = 0;
                for (int j = 0; j < n; ++j) {
                    ax += op(i, j) * wide(x0[j]), sx += std::abs(op(i, j)) * std::abs(wide(x0[j]));
                    as += op(i, j) * wide(gs[j]), ss += std::abs(op(i, j)) * std::abs(wide(gs[j]));
                }
                worst = std::max({worst, rel(gm[i], ax, sx), rel(T(0), as - wide(x0[i]), ss)});
            }
            if (which == 0)
                for (int i = nf; i < N; ++i) bad_frag += !same_bits(gs[i], T(real_of<T>(1e6))) + !same_bits(gm[i], T(real_of<T>(1e6)));
        }
    }
}

TYPED_TEST(DeviceBlasV2Test, TrsvTrmvEveryUploTransDiagOnFragAndVector) {
    using T = TypeParam;
    int bad = 0;
    double worst = 0;
    constexpr auto L = Uplo::Lower, U = Uplo::Upper;
    constexpr auto N = Transpose::NoTrans, Tt = Transpose::Trans, C = Transpose::ConjTrans;
    constexpr auto NU = Diag::NonUnit, UN = Diag::Unit;
    tri_case<L, N, NU, T>(this->dev, bad, worst), tri_case<L, N, UN, T>(this->dev, bad, worst);
    tri_case<U, N, NU, T>(this->dev, bad, worst), tri_case<U, N, UN, T>(this->dev, bad, worst);
    tri_case<L, Tt, NU, T>(this->dev, bad, worst), tri_case<U, Tt, UN, T>(this->dev, bad, worst);
    tri_case<L, C, UN, T>(this->dev, bad, worst), tri_case<U, C, NU, T>(this->dev, bad, worst);
    EXPECT_VERIFY(T, Check::blas, 13, worst);
    EXPECT_EQ(bad, 0) << "Frag entries past n must stay as they were (and not feed the solve)";
}

TYPED_TEST(DeviceBlasV2Test, GemvSubAndMultiTermGemvAcc) {
    using T = TypeParam;
    constexpr int WG = 64, m = 41, n1 = 7, n2 = 5, ld = m + 3;
    const auto ha = batched<T>(ld * (n1 + n2), 201), hx = batched<T>(n1 + n2, 202), hb = batched<T>(m, 203);
    T* a = this->dev.up(ha);
    T* x = this->dev.up(hx);
    T* bv = this->dev.up(hb);
    std::vector<T> ho(std::size_t(kBatch) * 2 * m, T(0));
    T* o = this->dev.up(ho);
    const T alpha = T(real_of<T>(-0.5));
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const device::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto av = device::mat(a + std::ptrdiff_t(b) * ld * (n1 + n2), m, n1 + n2, ld);
        const auto xv = device::assume_published(device::vec(x + std::ptrdiff_t(b) * (n1 + n2), n1 + n2));
        const auto x1 = device::assume_published(device::vec(x + std::ptrdiff_t(b) * (n1 + n2), n1));
        const auto x2 = device::assume_published(device::vec(x + std::ptrdiff_t(b) * (n1 + n2) + n1, n2));
        const auto b0 = device::vec(bv + std::ptrdiff_t(b) * m, m);
        T* ob = o + std::ptrdiff_t(b) * 2 * m;
        device::gemv_sub(ex, av, xv, b0, device::vec(ob, m));
        device::gemv_acc(ex, alpha, b0, device::vec(ob + m, m), device::term(av.sub(0, 0, m, n1), x1), device::term(av.sub(0, n1, m, n2), x2));
    });
    const auto got = this->dev.down(o, ho.size());
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < m; ++r) {
            LD<T> ax = 0;
            long double s = 0;
            for (int c = 0; c < n1 + n2; ++c) {
                const LD<T> e = wide(ha[std::size_t(b) * ld * (n1 + n2) + r + c * ld]), xc = wide(hx[std::size_t(b) * (n1 + n2) + c]);
                ax += e * xc, s += std::abs(e) * std::abs(xc);
            }
            const LD<T> b0 = wide(hb[std::size_t(b) * m + r]);
            worst = std::max({worst, rel(got[std::size_t(b) * 2 * m + r], b0 - ax, s + std::abs(b0)),
                              rel(got[std::size_t(b) * 2 * m + m + r], b0 + wide(alpha) * ax, s + std::abs(b0))});
        }
    EXPECT_VERIFY(T, Check::blas, 12, worst);
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
            o[(std::ptrdiff_t(b) * P + k) * P + l] = device::select(l % 2 == 0, v, mine);
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
    bcast_case<T>(this->dev, [](sycl::nd_item<1> it) { return device::Partition<8>{it.get_sub_group()}; }, "Partition<8>");
    bcast_case<T>(this->dev, [](sycl::nd_item<1> it) { return device::wrap(make_partition<8, true>(it.get_sub_group())); },
                  "SgpPartition<8, Masked>");
}

// detail::potf2_reg on a padded RowFrag: n = 13 in a 16-lane partition, ld = 19, batch 1024,
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
        const auto ex = device::wrap(make_partition<N>(it.get_sub_group()));
        const int pidx = tiny_native::tiny_partition_id(it.get_sub_group(), ex.part);
        const int b = static_cast<int>(it.get_group_linear_id()) * kMpw + pidx, lane = ex.lid();
        D* Ab = reinterpret_cast<D*>(a) + std::ptrdiff_t(b) * n * ld;
        device::Frag<D, N> rA;
        device::detail::reg_load_lower<D, N>(rA.v, Ab, ld, lane, lane < n, is_cplx<T>);
        std::int32_t r;
        if constexpr (UseSlm) {
            r = device::detail::potf2_reg<D, N, N, device::detail::OrderGuard::Skip>(
                ex, rA, n, true, device::bcast_policy::Slm<kMpw, R4*>{reinterpret_cast<R4*>(lm), pidx});
        } else {
            r = device::detail::potf2_reg<D, N, N>(ex, rA, n, true, device::bcast_policy::Shuffle{});
        }
        device::detail::reg_store_lower<D, N>(rA.v, Ab, ld, lane, lane < n);
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
static_assert(device::scratch_elements<device::op::ReduceN<3>, device::WorkGroup<256>, double>() == 24);
static_assert(device::scratch_elements<device::op::Reduce, device::SubGroup<32>, float>() == 0);
static_assert(device::WorkGroup<256>::nteams == 8 && device::Teams<256>::size == 32 && device::Partition<8>::size == 8);
static_assert(std::is_same_v<device::tag_t<float>, device::Scalar<float>> && std::is_same_v<device::tag_t<device::Zero>, device::Zero>);

}  // namespace

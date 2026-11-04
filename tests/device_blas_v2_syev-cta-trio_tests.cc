// SPDX-License-Identifier: MIT
// Family-private device LA v2 ops of the syev CTA trio against host references:
//   src/ops/syev/kernels/syev_drivers_v2.hh  (herm_load, ElemAt, dgmm, similarities, repack,
//                                             column-owner larf / org2l / orm2l)
//   src/ops/syev/sytrd/sytrd_v2.hh           (Fit ops, larfg_rowfrag, lacpy_herm, sytd2)
//   src/ops/syev/steqr/steqr_v2.hh           (lanst_max, tridiag_split, find_first_lane,
//                                             scal_lane, reverse, tridiag_rot)
// Shared ops (operands, item(), mirror, laset, lacpy, rot, lasr, Frag moves) are core's and are
// covered by device_blas_v2_tests. Guards (docs/developer/agent-guide.md section 8): non-natural
// ld / inc, complex data with nonzero imaginary parts, NaN poison in every entry an op must not
// read or write, a zero off-diagonal and a tau == 0 reflector, larfg's identity thresholds, the
// split test straddled (0.5x and 2x), batch 1024 with period-3 data where SLM paths must be
// bit-identical to their representative, tiles at a non-zero accessor offset (the kernels'
// form), masked and lockstep partitions of 8, 16 and 32 (n at the ceiling).
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

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/sg_partition.hh>
#include <src/ops/syev/kernels/syev_drivers_v2.hh>
#include <src/ops/syev/steqr/steqr_v2.hh>
#include <src/ops/syev/sytrd/sytrd_v2.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
namespace dv = batchlas::syev_v2;
namespace tv = batchlas::sytrd_v2;
namespace qv = batchlas::steqr_v2;
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
// Element i of item b: item b repeats representative b % kPeriod.
template <class T> T item_val(std::uint64_t seed, int b, int i) { return val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i); }
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class C> C conj_w(C v) {
    if constexpr (std::is_same_v<C, long double>) return v;
    else return std::conj(v);
}
template <class T> long double nanabs(LD<T> v) {
    const long double a = std::abs(v);
    return std::isnan(a) ? std::numeric_limits<long double>::infinity() : a;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// Per-item array: representative b % kPeriod, edited by fix(item pointer, representative).
template <class T, class Fix> std::vector<T> batched(int per, std::uint64_t seed, Fix fix) {
    std::vector<T> h(std::size_t(kBatch) * per);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per; ++i) h[std::size_t(b) * per + i] = item_val<T>(seed, b, i);
        fix(h.data() + std::size_t(b) * per, b);
    }
    for (int b = kPeriod; b < kBatch; ++b) std::copy_n(h.data() + std::size_t(b % kPeriod) * per, per, h.data() + std::size_t(b) * per);
    return h;
}
template <class T> std::vector<T> batched(int per, std::uint64_t seed) {
    return batched<T>(per, seed, [](T*, int) {});
}
template <class T> int bitdiff(const std::vector<T>& g, int per) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per; ++i) d += !same_bits(g[std::size_t(b) * per + i], g[std::size_t(b % kPeriod) * per + i]);
    return d;
}

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
    template <int WG, class F> void run(int groups, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
         }).wait_and_throw();
    }
    // A typed local accessor of `elems` T, passed itself (the kernels' accessor-based views).
    template <int WG, class T, class F> void run_acc(int groups, std::size_t elems, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<T, 1> lm(sycl::range<1>(std::max<std::size_t>(elems, 1)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, lm); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2SyevTrioTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};
template <class T> class DeviceBlasV2SyevTrioRealTest : public DeviceBlasV2SyevTrioTest<T> {};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DeviceBlasV2SyevTrioTest, Types);
TYPED_TEST_SUITE(DeviceBlasV2SyevTrioRealTest, RealTypes);

// =============================== syev drivers (syev_drivers_v2.hh) ===============================

// herm_load from both stored triangles; ElemAt driving core mirror (uplo_mirror's guarded Upper,
// syev_cta's Lower with Im(diag) dropped) and core laset (the fused kernel's Q identity on the
// leading nn block). Pure data movement: every entry bit for bit, the ld padding included.
TYPED_TEST(DeviceBlasV2SyevTrioTest, HermLoadAndElemAt) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 128, n = 13, ld = n + 4, ldo = n + 2, nn = 9;
    const int per = n * ld, pero = n * ldo;
    auto ha = batched<T>(per, 1, [&](T* h, int) {  // the diagonal keeps an imaginary part: herm_load drops it
        for (int j = 0; j < n; ++j)
            for (int i = n; i < ld; ++i) h[i + j * ld] = nan_v<T>();
    });
    std::vector<T> hout(std::size_t(kBatch) * 2 * pero, nan_v<T>());
    std::vector<R> hq(std::size_t(kBatch) * per, std::numeric_limits<R>::quiet_NaN());
    T* a = this->dev.up(ha);
    T* m1 = this->dev.up(ha);
    T* m2 = this->dev.up(ha);
    T* o = this->dev.up(hout);
    R* qg = this->dev.up(hq);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto A = v2::mat(a + std::ptrdiff_t(b) * per, n, n, ld);
        const auto M1 = v2::mat(m1 + std::ptrdiff_t(b) * per, n, n, ld);
        const auto M2 = v2::mat(m2 + std::ptrdiff_t(b) * per, n, n, ld);
        const auto Q = v2::mat(qg + std::ptrdiff_t(b) * per, n, n, ld);
        const auto Hu = v2::mat(o + std::ptrdiff_t(b) * 2 * pero, n, n, ldo);
        const auto Hl = v2::mat(o + std::ptrdiff_t(b) * 2 * pero + pero, n, n, ldo);
        for (int t = static_cast<int>(it.get_local_linear_id()); t < n * n; t += WG) {
            const int i = t % n, j = t / n;
            T hu, hl;
            dv::herm_load(hu, A, true, i, j);
            dv::herm_load<v2::arith::Explicit>(hl, A, false, i, j);
            Hu(i, j) = hu;
            Hl(i, j) = hl;
            if (i > j) v2::mirror<Uplo::Upper, true>(dv::ElemAt{i, j}, M1);  // uplo_mirror: strictly upper sources only
            v2::mirror<Uplo::Lower, true>(dv::ElemAt{i, j}, M2);
            v2::laset(dv::ElemAt{i, j}, v2::Zero{}, i < nn ? R(1) : R(0), Q);
        }
    });
    const auto go = this->dev.down(o, hout.size());
    const auto g1 = this->dev.down(m1, ha.size());
    const auto g2 = this->dev.down(m2, ha.size());
    const auto gq = this->dev.down(qg, hq.size());
    int bad_h = 0, bad_m = 0, bad_q = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* A = ha.data() + std::size_t(b) * per;
        auto realdiag = [&](int i) { return is_cplx<T> ? T(std::real(A[i + i * ld])) : A[i + i * ld]; };
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const T hu = i == j ? realdiag(i) : (i < j ? A[i + j * ld] : conj_h(A[j + i * ld]));
                const T hl = i == j ? realdiag(i) : (i > j ? A[i + j * ld] : conj_h(A[j + i * ld]));
                bad_h += !same_bits(go[std::size_t(b) * 2 * pero + i + j * ldo], hu);
                bad_h += !same_bits(go[std::size_t(b) * 2 * pero + pero + i + j * ldo], hl);
                const T m1r = i <= j ? A[i + j * ld] : conj_h(A[j + i * ld]);
                const T m2r = i == j ? realdiag(i) : (i > j ? A[i + j * ld] : conj_h(A[j + i * ld]));
                bad_m += !same_bits(g1[std::size_t(b) * per + i + j * ld], m1r);
                bad_m += !same_bits(g2[std::size_t(b) * per + i + j * ld], m2r);
                bad_q += !same_bits(gq[std::size_t(b) * per + i + j * ld], (i == j && i < nn) ? R(1) : R(0));
            }
            for (int i = n; i < ld; ++i) {
                bad_m += !same_bits(g1[std::size_t(b) * per + i + j * ld], A[i + j * ld]);
                bad_m += !same_bits(g2[std::size_t(b) * per + i + j * ld], A[i + j * ld]);
                bad_q += !std::isnan(gq[std::size_t(b) * per + i + j * ld]);
            }
        }
    }
    EXPECT_EQ(bad_h, 0) << "herm_load";
    EXPECT_EQ(bad_m, 0) << "ElemAt + core mirror";
    EXPECT_EQ(bad_q, 0) << "ElemAt + core laset";
}

// C := diag(d) Z with Z real lifted (out of place), and Y := diag(d) Y in place.
TYPED_TEST(DeviceBlasV2SyevTrioTest, DgmmLeftAt) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 64, n = 19, ldz = n + 3, ldc = n + 5;
    auto hd = batched<T>(n, 11);
    auto hz = batched<R>(n * ldz, 12);
    auto hy = batched<T>(n * ldc, 13);
    std::vector<T> hc(std::size_t(kBatch) * n * ldc, nan_v<T>());
    T* d = this->dev.up(hd);
    R* z = this->dev.up(hz);
    T* y = this->dev.up(hy);
    T* c = this->dev.up(hc);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto D = v2::vec(d + std::ptrdiff_t(b) * n, n);
        const auto Z = v2::mat(z + std::ptrdiff_t(b) * n * ldz, n, n, ldz);
        const auto C = v2::mat(c + std::ptrdiff_t(b) * n * ldc, n, n, ldc);
        const auto Y = v2::mat(y + std::ptrdiff_t(b) * n * ldc, n, n, ldc);
        for (int t = static_cast<int>(it.get_local_linear_id()); t < n * n; t += WG) {
            const int i = t % n, j = t / n;
            dv::dgmm_left_at(D, Z, C, i, j);
            dv::dgmm_left_at<v2::arith::Explicit>(D, Y, Y, i, j);
        }
    });
    const auto gc = this->dev.down(c, hc.size());
    const auto gy = this->dev.down(y, hy.size());
    double worst = 0;
    int pad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const LD<T> di = wide(hd[std::size_t(b) * n + i]);
                const LD<T> zr = LD<T>(static_cast<long double>(hz[std::size_t(b) * n * ldz + i + j * ldz]));
                const LD<T> yr = wide(hy[std::size_t(b) * n * ldc + i + j * ldc]);
                worst = std::max({worst, rel(gc[std::size_t(b) * n * ldc + i + j * ldc], di * zr, std::abs(di) * std::abs(zr)),
                                  rel(gy[std::size_t(b) * n * ldc + i + j * ldc], di * yr, std::abs(di) * std::abs(yr))});
            }
            for (int i = n; i < ldc; ++i) pad += !std::isnan(std::real(LD<T>(gc[std::size_t(b) * n * ldc + i + j * ldc])));
        }
    }
    EXPECT_EQ(pad, 0) << "ld padding written";
    EXPECT_VERIFY(T, Check::blas, 1, worst);
}

// phase_similarity, both Uplo: T' = S^H T S has the real off-diagonal |e|, |S| = 1, a zero e_i
// carries S over bit for bit; want_s == false leaves NaN-poisoned S untouched. Real types:
// sign_similarity gives e := |e| and S = +-1 exactly.
TYPED_TEST(DeviceBlasV2SyevTrioTest, TridiagonalSimilarities) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 32, n = 17, kZero = 5;
    auto hd = batched<T>(n, 21);
    auto he = batched<T>(n - 1, 22, [&](T* h, int) { h[kZero] = T(0); });
    std::vector<R> hdr(std::size_t(kBatch) * n * 3, std::numeric_limits<R>::quiet_NaN());
    std::vector<R> her(std::size_t(kBatch) * (n - 1) * 3, std::numeric_limits<R>::quiet_NaN());
    std::vector<T> hs(std::size_t(kBatch) * n * 3, nan_v<T>());
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    T* e2 = this->dev.up(he);
    R* dr = this->dev.up(hdr);
    R* er = this->dev.up(her);
    T* s = this->dev.up(hs);
    T* s2 = this->dev.up(std::vector<T>(std::size_t(kBatch) * n, nan_v<T>()));
    this->dev.template run<WG>(kBatch / WG, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto dvv = v2::vec(d + std::ptrdiff_t(b) * n, n);
        const auto ev = v2::vec(e + std::ptrdiff_t(b) * (n - 1), n - 1);
        for (int k = 0; k < 3; ++k) {
            const Uplo u = k == 1 ? Uplo::Upper : Uplo::Lower;
            const auto drv = v2::vec(dr + (std::ptrdiff_t(b) * 3 + k) * n, n);
            const auto erv = v2::vec(er + (std::ptrdiff_t(b) * 3 + k) * (n - 1), n - 1);
            const auto sv = v2::vec(s + (std::ptrdiff_t(b) * 3 + k) * n, n);
            dv::phase_similarity(v2::Thread{}, n, dvv, ev, drv, erv, sv, u, /*want_s=*/k < 2);
        }
        if constexpr (!is_cplx<T>) {
            dv::sign_similarity(v2::Thread{}, n, v2::vec(e2 + std::ptrdiff_t(b) * (n - 1), n - 1), v2::vec(s2 + std::ptrdiff_t(b) * n, n));
        }
    });
    const auto gdr = this->dev.down(dr, hdr.size());
    const auto ger = this->dev.down(er, her.size());
    const auto gs = this->dev.down(s, hs.size());
    const auto ge2 = this->dev.down(e2, he.size());
    const auto gs2 = this->dev.down(s2, std::size_t(kBatch) * n);
    double worst = 0;
    int exact = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int k = 0; k < 3; ++k) {
            const R* gdrk = gdr.data() + (std::size_t(b) * 3 + k) * n;
            const R* gerk = ger.data() + (std::size_t(b) * 3 + k) * (n - 1);
            const T* gsk = gs.data() + (std::size_t(b) * 3 + k) * n;
            for (int i = 0; i < n; ++i) exact += !same_bits(gdrk[i], static_cast<R>(std::real(hd[std::size_t(b) * n + i])));
            for (int i = 0; i < n - 1; ++i) {
                const LD<T> ei = wide(he[std::size_t(b) * (n - 1) + i]);
                worst = std::max(worst, rel<R>(gerk[i], std::abs(ei), std::abs(ei)));
            }
            if (k == 2) {
                for (int i = 0; i < n; ++i) exact += !std::isnan(std::real(LD<T>(gsk[i])));
                continue;
            }
            exact += !same_bits(gsk[0], T(1));
            for (int i = 0; i < n - 1; ++i) {
                const LD<T> ei = wide(he[std::size_t(b) * (n - 1) + i]);
                const LD<T> si = wide(gsk[i]), sn = wide(gsk[i + 1]);
                worst = std::max(worst, rel<T>(T(std::abs(sn)), LD<T>(1), 1.0L));
                if (i == kZero) {
                    exact += !same_bits(gsk[i + 1], gsk[i]);
                    continue;
                }
                const LD<T> off = (k == 0) ? conj_w(sn) * ei * si : conj_w(si) * ei * sn;  // T'(i+1,i) / T'(i,i+1)
                worst = std::max(worst, rel<T>(static_cast<T>(off), LD<T>(std::abs(ei)), std::abs(ei)));
            }
        }
        if constexpr (!is_cplx<T>) {
            T sg = T(1);
            exact += !same_bits(gs2[std::size_t(b) * n], T(1));
            for (int i = 0; i < n - 1; ++i) {
                const T ei = he[std::size_t(b) * (n - 1) + i];
                sg = sg * (ei >= T(0) ? T(1) : T(-1));
                exact += !same_bits(ge2[std::size_t(b) * (n - 1) + i], std::fabs(ei));
                exact += !same_bits(gs2[std::size_t(b) * n + i + 1], sg);
            }
        }
    }
    EXPECT_EQ(exact, 0) << "dr / S(0) / zero-e carry / unwritten S / sign_similarity";
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// phase_replay on SgpPartition<P>: lane l returns S(l) of the superdiagonal chain, the values
// phase_similarity<Upper> writes serially on lane 0 (same product order), so the two agree to
// rounding; a lane reading another chunk's e would not.
template <class T, int P> void run_phase_replay(Dev& dev, int n) {
    using R = real_of<T>;
    constexpr int WG = 64, chunks = WG / P;
    auto he = batched<T>(P, 31, [&](T* h, int) { h[2] = T(0); });
    std::vector<T> ho(std::size_t(kBatch) * P * 2, nan_v<T>());
    T* e = dev.up(he);
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch / chunks, [=](sycl::nd_item<1> it) {
        const v2::SgpPartition<P> pex{it.get_sub_group()};
        const int lane = pex.lid();
        const int b = static_cast<int>(it.get_group_linear_id()) * chunks + static_cast<int>(it.get_local_linear_id()) / P;
        const T el = lane < n - 1 ? e[std::ptrdiff_t(b) * P + lane] : T(0);
        const R al = lane < n - 1 ? sycl::hypot(v2::Arith<T>::re(el), v2::Arith<T>::im(el)) : R(0);
        o[std::ptrdiff_t(b) * 2 * P + lane] = dv::phase_replay(pex, el, al, n);
        if (lane == 0) {  // serial reference on lane 0 (Thread): S written to the second half
            T* sr = o + std::ptrdiff_t(b) * 2 * P + P;
            const auto ev = v2::vec(e + std::ptrdiff_t(b) * P, n - 1);
            R scratch[P];
            const auto rv = v2::Vec<R, v2::Ext<P>>{scratch, {}};
            dv::phase_similarity(v2::Thread{}, n, ev, ev, rv, rv, v2::vec(sr, n), Uplo::Upper, true);
        }
    });
    const auto g = dev.down(o, ho.size());
    double worst = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int l = 0; l < n; ++l) worst = std::max(worst, rel(g[std::size_t(b) * 2 * P + l], wide(g[std::size_t(b) * 2 * P + P + l]), 1.0L));
    SCOPED_TRACE(::testing::Message() << "phase_replay<" << P << "> n=" << n);
    EXPECT_VERIFY(T, Check::blas, n, worst);
}
TYPED_TEST(DeviceBlasV2SyevTrioTest, PhaseReplayPartition) {
    run_phase_replay<TypeParam, 8>(this->dev, 7);
    run_phase_replay<TypeParam, 32>(this->dev, 32);
    run_phase_replay<TypeParam, 32>(this->dev, 20);
}

// sytrd_reflector_pack_at with and without the folded TAUQ, and sytrd_tau_pack_at, both Uplo;
// exact. TAUQ starts NaN, so the no-tau form must leave it NaN.
TYPED_TEST(DeviceBlasV2SyevTrioTest, SytrdReflectorPack) {
    using T = TypeParam;
    constexpr int WG = 128, n = 11, lda = n + 2, ldq = n + 3;
    auto ha = batched<T>(n * lda, 41);
    auto ht = batched<T>(n - 1, 42);
    std::vector<T> hq(std::size_t(kBatch) * 4 * n * ldq, nan_v<T>());
    std::vector<T> htq(std::size_t(kBatch) * 6 * n, nan_v<T>());
    T* a = this->dev.up(ha);
    T* tau = this->dev.up(ht);
    T* aq = this->dev.up(hq);
    T* tq = this->dev.up(htq);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int lid = static_cast<int>(it.get_local_linear_id());
        const auto A = v2::mat(a + std::ptrdiff_t(b) * n * lda, n, n, lda);
        const auto tvv = v2::vec(tau + std::ptrdiff_t(b) * (n - 1), n - 1);
        for (int k = 0; k < 4; ++k) {
            const Uplo u = (k & 1) ? Uplo::Upper : Uplo::Lower;
            const auto AQ = v2::mat(aq + (std::ptrdiff_t(b) * 4 + k) * n * ldq, n, n, ldq);
            const auto tqv = v2::vec(tq + (std::ptrdiff_t(b) * 6 + k) * n, n);
            for (int t = lid; t < n * n; t += WG) {
                if (k < 2) dv::sytrd_reflector_pack_at(A, AQ, u, n, t % n, t / n, tvv, tqv);
                else dv::sytrd_reflector_pack_at(A, AQ, u, n, t % n, t / n);
            }
            if (k >= 2 && lid < n) dv::sytrd_tau_pack_at(tvv, v2::vec(tq + (std::ptrdiff_t(b) * 6 + k + 2) * n, n), u, n, lid);
        }
    });
    const auto gq = this->dev.down(aq, hq.size());
    const auto gt = this->dev.down(tq, htq.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* A = ha.data() + std::size_t(b) * n * lda;
        const T* tvh = ht.data() + std::size_t(b) * (n - 1);
        for (int k = 0; k < 4; ++k) {
            const bool upper = k & 1;
            const T* Q = gq.data() + (std::size_t(b) * 4 + k) * n * ldq;
            for (int c = 0; c < n; ++c) {
                for (int r = 0; r < n; ++r) {
                    T ref = T(0);
                    if (!upper && c >= 1 && r > c) ref = A[r + (c - 1) * lda];
                    if (upper && c <= n - 2 && r < c) ref = A[r + (c + 1) * lda];
                    bad += !same_bits(Q[r + c * ldq], ref);
                }
                for (int r = n; r < ldq; ++r) bad += !std::isnan(std::real(LD<T>(Q[r + c * ldq])));
                const T tref = upper ? (c < n - 1 ? tvh[c] : T(0)) : (c >= 1 ? tvh[c - 1] : T(0));
                const T* tq_fold = gt.data() + (std::size_t(b) * 6 + k) * n;
                if (k < 2) bad += !same_bits(tq_fold[c], tref);
                else bad += !std::isnan(std::real(LD<T>(tq_fold[c])));
                if (k >= 2) bad += !same_bits(gt[(std::size_t(b) * 6 + k + 2) * n + c], tref);
            }
        }
    }
    EXPECT_EQ(bad, 0);
}

// QL reflectors of a P-frame: reflector k (k < n-1) in column k+1 of R (rows < k), unitary
// H(k) = I - tau_k v v^H with tau_k = (1 - e^{i theta}) / (v^H v) (real: theta = pi), tau_2 = 0.
// Every other entry of R is NaN, so a read outside the support poisons the result.
template <class T> void ql_reflectors(int P, int ldr, int n, std::uint64_t seed, T* R, T* tau) {
    using RT = real_of<T>;
    for (int i = 0; i < P * ldr; ++i) R[i] = nan_v<T>();
    for (int k = 0; k < P; ++k) tau[k] = T(0);
    for (int k = 0; k < n - 1; ++k) {
        long double s = 1;
        for (int r = 0; r < k; ++r) {
            const T v = val<T>(seed + std::uint64_t(k) * 97 + r);
            R[r + (k + 1) * ldr] = v;
            s += std::norm(LD<T>(v));
        }
        if (k == 2) continue;
        if constexpr (is_cplx<T>) {
            const long double th = 0.7L + 0.3L * k;
            tau[k] = T(static_cast<RT>((1 - std::cos(th)) / s), static_cast<RT>(-std::sin(th) / s));
        } else {
            tau[k] = static_cast<T>(2 / s);
        }
    }
}
// M = H(n-2) ... H(0) (P x P, column-major), long double.
template <class T> std::vector<LD<T>> ql_product(int P, int ldr, int n, const T* R, const T* tau) {
    std::vector<LD<T>> M(std::size_t(P) * P, LD<T>(0));
    for (int i = 0; i < P; ++i) M[i + i * P] = LD<T>(1);
    for (int k = 0; k < n - 1; ++k) {
        std::vector<LD<T>> v(P, LD<T>(0));
        for (int r = 0; r < k; ++r) v[r] = wide(R[r + (k + 1) * ldr]);
        v[k] = LD<T>(1);
        const LD<T> t = wide(tau[k]);
        for (int j = 0; j < P; ++j) {
            LD<T> dot = 0;
            for (int r = 0; r < P; ++r) dot += conj_w(v[r]) * M[r + j * P];
            for (int r = 0; r < P; ++r) M[r + j * P] -= v[r] * (t * dot);
        }
    }
    return M;
}

// org2l (arith::Native) and orm2l (arith::Explicit) on SgpPartition<P> over accessor-based views
// at a non-zero offset (the fused kernel's form): lane l < n-1 holds column l of
// M = H(n-2)...H(0); orm2l gives M c on lanes l < n and leaves the pad lanes' columns alone.
template <class T, int P> void run_org2l_orm2l(Dev& dev, int n) {
    using Local = sycl::local_accessor<T, 1>;
    constexpr int WG = 64, chunks = WG / P, LDT = P + 1, LDR = P + 3, slm = P * LDT + P;
    std::vector<T> hr(std::size_t(kBatch) * P * LDR), ht(std::size_t(kBatch) * P);
    for (int b = 0; b < kBatch; ++b) {
        if (b < kPeriod) ql_reflectors<T>(P, LDR, n, 500 + b * 7919, hr.data() + std::size_t(b) * P * LDR, ht.data() + std::size_t(b) * P);
        else {
            std::copy_n(hr.data() + std::size_t(b % kPeriod) * P * LDR, P * LDR, hr.data() + std::size_t(b) * P * LDR);
            std::copy_n(ht.data() + std::size_t(b % kPeriod) * P, P, ht.data() + std::size_t(b) * P);
        }
    }
    auto hc = batched<T>(P * P, 51);
    std::vector<T> hq(std::size_t(kBatch) * P * P, nan_v<T>()), ho(hq);
    T* rg = dev.up(hr);
    T* tg = dev.up(ht);
    T* cg = dev.up(hc);
    T* qg = dev.up(hq);
    T* og = dev.up(ho);
    dev.template run_acc<WG, T>(kBatch / chunks, std::size_t(chunks) * slm, [=](sycl::nd_item<1> it, const Local& lm) {
        const auto pex = v2::wrap(make_partition<P>(it.get_sub_group()));
        const int lane = pex.lid();
        const int chunk = static_cast<int>(it.get_local_linear_id()) / P;
        const int b = static_cast<int>(it.get_group_linear_id()) * chunks + chunk;
        const int base = chunk * slm;
        const v2::Mat<T, v2::Ext<P>, v2::Ext<P>, v2::Ext<LDT>, v2::Mem::Local, v2::Cyclic, Local> tm{lm, {}, {}, {}, base};
        const auto vs = v2::vec(lm, P, base + P * LDT);
        for (int c = 0; c < P; ++c) tm(lane, c) = rg[std::ptrdiff_t(b) * P * LDR + lane + c * LDR];
        group_barrier(pex.part);
        const T tau_lane = tg[std::ptrdiff_t(b) * P + lane];

        v2::Frag<T, P, v2::ColPerLane> q;
        dv::org2l(pex, q, vs, tm, tau_lane, n);
        v2::store(q, v2::vec(qg + std::ptrdiff_t(b) * P * P + lane * P, P));

        v2::Frag<T, P, v2::ColPerLane> c;
        v2::load(c, v2::vec(cg + std::ptrdiff_t(b) * P * P + lane * P, P));
        dv::orm2l<v2::arith::Explicit>(pex, c, vs, tm, tau_lane, n);
        v2::store(c, v2::vec(og + std::ptrdiff_t(b) * P * P + lane * P, P));
    });
    const auto gq = dev.down(qg, hq.size());
    const auto go = dev.down(og, ho.size());
    EXPECT_EQ(bitdiff(gq, P * P), 0) << "org2l<" << P << "> items differ from their representative";
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const auto M = ql_product<T>(P, LDR, n, hr.data() + std::size_t(b) * P * LDR, ht.data() + std::size_t(b) * P);
        const T* Q = gq.data() + std::size_t(b) * P * P;
        const T* O = go.data() + std::size_t(b) * P * P;
        const T* C = hc.data() + std::size_t(b) * P * P;
        for (int l = 0; l < P; ++l) {
            for (int r = 0; r < P; ++r) {
                if (l < n - 1) worst = std::max(worst, rel(Q[r + l * P], M[r + l * P], 1.0L));
                if (l < n) {
                    LD<T> ref = 0;
                    long double sc = 0;
                    for (int k = 0; k < P; ++k) {
                        ref += M[r + k * P] * wide(C[k + l * P]);
                        sc += std::abs(M[r + k * P]) * std::abs(wide(C[k + l * P]));
                    }
                    worst = std::max(worst, rel(O[r + l * P], ref, sc));
                } else {
                    bad += !same_bits(O[r + l * P], C[r + l * P]);
                }
            }
        }
    }
    EXPECT_EQ(bad, 0) << "orm2l<" << P << "> touched a pad lane (l >= n)";
    EXPECT_VERIFY(T, Check::blas, n, worst);
}
TYPED_TEST(DeviceBlasV2SyevTrioTest, Org2lOrm2lPartition) {
    run_org2l_orm2l<TypeParam, 8>(this->dev, 6);
    run_org2l_orm2l<TypeParam, 32>(this->dev, 29);
    run_org2l_orm2l<TypeParam, 32>(this->dev, 32);
}

// Register-column ops on one work-item each: larf_col with both arithmetic policies,
// larf_unit_col, laset_col, dgmm_lift (r < n only; the rest keeps the fill).
TYPED_TEST(DeviceBlasV2SyevTrioTest, RegisterColumnOps) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 32, N = 16, n = 11, k = 9, slots = 5;
    auto hv = batched<T>(N, 61);
    auto hc = batched<T>(N, 62);
    auto hz = batched<R>(N, 63);
    const T tau = val<T>(64);
    std::vector<T> ho(std::size_t(kBatch) * slots * N, nan_v<T>());
    T* v = this->dev.up(hv);
    T* c = this->dev.up(hc);
    R* z = this->dev.up(hz);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch / WG, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto vv = v2::assume_published(v2::Vec<T, v2::Ext<N>>{v + std::ptrdiff_t(b) * N, {}});
        const auto out = [&](int s) { return v2::Vec<T, v2::Ext<N>>{o + (std::ptrdiff_t(b) * slots + s) * N, {}}; };
        const auto cv = v2::Vec<T, v2::Ext<N>>{c + std::ptrdiff_t(b) * N, {}};
        v2::Frag<T, N, v2::ColPerLane> f, g;
        v2::load(f, cv);
        v2::load(g, cv);
        dv::larf_col(f, vv, tau);
        v2::store(f, out(0));
        dv::larf_col<v2::arith::Explicit>(g, vv, tau);
        v2::store(g, out(1));
        dv::larf_unit_col(f, vv, tau, k);
        v2::store(f, out(2));
        dv::laset_col(f, k, T(2), T(-3));
        v2::store(f, out(3));
        v2::fill(f, T(5));
        dv::dgmm_lift(f, cv, v2::Vec<R, v2::Ext<N>>{z + std::ptrdiff_t(b) * N, {}}, n);
        v2::store(f, out(4));
    });
    const auto g = this->dev.down(o, ho.size());
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* V = hv.data() + std::size_t(b) * N;
        const T* C = hc.data() + std::size_t(b) * N;
        const R* Z = hz.data() + std::size_t(b) * N;
        const T* G = g.data() + std::size_t(b) * slots * N;
        LD<T> dot = 0;
        long double sd = 0;
        for (int r = 0; r < N; ++r) {
            dot += conj_w(wide(V[r])) * wide(C[r]);
            sd += std::abs(wide(V[r])) * std::abs(wide(C[r]));
        }
        const LD<T> t = wide(tau);
        for (int r = 0; r < N; ++r) {
            const LD<T> l1 = wide(C[r]) - wide(V[r]) * (t * dot);
            const long double sc = std::abs(wide(C[r])) + std::abs(wide(V[r])) * std::abs(t) * sd;
            worst = std::max({worst, rel(G[r], l1, sc), rel(G[N + r], l1, sc)});
            // larf_unit_col ignores c: e_k - tau v with v(k) = 1, v(r > k) = 0.
            const LD<T> lu = r < k ? -t * wide(V[r]) : (r == k ? LD<T>(1) - t : LD<T>(0));
            worst = std::max(worst, rel(G[2 * N + r], lu, std::abs(t) * std::abs(wide(V[r])) + 1));
            bad += !same_bits(G[3 * N + r], r == k ? T(-3) : T(2));
            if (r < n) {
                const LD<T> ref = wide(C[r]) * LD<T>(static_cast<long double>(Z[r]));
                worst = std::max(worst, rel(G[4 * N + r], ref, std::abs(ref)));
            } else {
                bad += !same_bits(G[4 * N + r], T(5));
            }
        }
    }
    EXPECT_EQ(bad, 0) << "laset_col / dgmm_lift beyond n";
    EXPECT_VERIFY(T, Check::blas, N, worst);
}

// ================================== sytrd (sytrd_v2.hh) ==================================

// The Hermitian matrix whose UL triangle is stored in h (column-major, ld), real diagonal.
template <class T> std::vector<LD<T>> herm_from(const T* h, int n, int ld, Uplo ul) {
    std::vector<LD<T>> a(std::size_t(n) * n);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            const bool stored = ul == Uplo::Lower ? r >= c : r <= c;
            LD<T> e = stored ? wide(h[r + std::size_t(c) * ld]) : conj_w(wide(h[c + std::size_t(r) * ld]));
            if (r == c) e = LD<T>(std::real(e));
            a[r + std::size_t(c) * n] = e;
        }
    return a;
}
// Poison everything the op must not read: the other triangle, Im(diag), rows past n.
template <class T> void poison_herm(T* h, int n, int ld, Uplo ul) {
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < ld; ++r) {
            const bool stored = r < n && (ul == Uplo::Lower ? r >= c : r <= c);
            if (!stored) h[r + std::size_t(c) * ld] = nan_v<T>();
            if (r == c) {
                if constexpr (is_cplx<T>)
                    h[r + std::size_t(c) * ld] = T(std::real(h[r + std::size_t(c) * ld]), std::numeric_limits<real_of<T>>::quiet_NaN());
            }
        }
}

// ||A - Q T Q^H||_F / ||A||_F from LAPACK's sytd2 packing in o (n x n, ld ldo), d, e, tau.
template <class T>
double sytd2_residual(const std::vector<LD<T>>& a, const T* o, int ldo, const T* d, const T* e, const T* tau, int n, Uplo ul) {
    using L = LD<T>;
    std::vector<L> q(std::size_t(n) * n, L(0)), qv(n), v(n);
    for (int i = 0; i < n; ++i) q[i + std::size_t(i) * n] = L(1);
    auto apply = [&](int i) {  // Q := Q (I - tau_i v_i v_i^H)
        for (int r = 0; r < n; ++r) v[r] = L(0);
        if (ul == Uplo::Lower) {
            v[i + 1] = L(1);
            for (int r = i + 2; r < n; ++r) v[r] = wide(o[r + std::size_t(i) * ldo]);
        } else {
            v[i] = L(1);
            for (int r = 0; r < i; ++r) v[r] = wide(o[r + std::size_t(i + 1) * ldo]);
        }
        for (int r = 0; r < n; ++r) {
            qv[r] = L(0);
            for (int c = 0; c < n; ++c) qv[r] += q[r + std::size_t(c) * n] * v[c];
        }
        const L t = wide(tau[i]);
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) q[r + std::size_t(c) * n] -= t * qv[r] * conj_w(v[c]);
    };
    if (ul == Uplo::Lower)
        for (int i = 0; i <= n - 2; ++i) apply(i);
    else
        for (int i = n - 2; i >= 0; --i) apply(i);
    std::vector<L> t(std::size_t(n) * n, L(0));
    for (int i = 0; i < n; ++i) t[i + std::size_t(i) * n] = wide(d[i]);
    for (int i = 0; i + 1 < n; ++i) {
        const L ei = wide(e[i]);
        if (ul == Uplo::Lower) t[i + 1 + std::size_t(i) * n] = ei, t[i + std::size_t(i + 1) * n] = conj_w(ei);
        else t[i + std::size_t(i + 1) * n] = ei, t[i + 1 + std::size_t(i) * n] = conj_w(ei);
    }
    long double num = 0, den = 0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            L s = 0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) s += q[r + std::size_t(k) * n] * t[k + std::size_t(l) * n] * conj_w(q[c + std::size_t(l) * n]);
            const long double x = nanabs<T>(a[r + std::size_t(c) * n] - s);
            num += x * x;
            den += std::norm(a[r + std::size_t(c) * n]);
        }
    return static_cast<double>(std::sqrt(num / std::max<long double>(den, 1e-300L)));
}

// sytd2 on a tile in a typed local accessor at a non-zero offset, as sytrd_cta / the fused syev
// kernel hold it: lacpy_herm from the stored triangle, sytd2, LAPACK's packing back out.
template <class T, Uplo UL, int P, bool Lockstep> void sytd2_case(Dev& dev, int n) {
    using Local = sycl::local_accessor<T, 1>;
    constexpr int WG = 64, LDA = P + 3, per_wg = WG / P, slot = LDA * P + 2 * P;
    const int lda = n + 2, ldo = n + 1, per_a = lda * n;
    auto ha = batched<T>(per_a, 7 + 31 * int(UL) + P, [&](T* h, int) { poison_herm(h, n, lda, UL); });
    std::vector<T> ho(std::size_t(kBatch) * ldo * n, T(0)), hd(std::size_t(kBatch) * n, T(0)), he(hd), ht(hd);
    T* a = dev.up(ha);
    T* o = dev.up(ho);
    T* d = dev.up(hd);
    T* e = dev.up(he);
    T* tu = dev.up(ht);
    dev.template run_acc<WG, T>(kBatch / per_wg, std::size_t(per_wg) * slot, [=](sycl::nd_item<1> it, const Local& lm) {
        const auto sg = it.get_sub_group();
        auto body = [&](const auto& ex, int s) {
            const int b = static_cast<int>(it.get_group_linear_id()) * per_wg + s;
            const int base = s * slot;
            const v2::Mat<T, v2::Ext<P>, v2::Ext<P>, v2::Ext<LDA>, v2::Mem::Local, v2::Cyclic, Local> tile{lm, {}, {}, {}, base};
            const auto v = v2::vec(lm, n, base + LDA * P), w = v2::vec(lm, n, base + LDA * P + P);
            T* eb = e + std::ptrdiff_t(b) * n;
            T* tb = tu + std::ptrdiff_t(b) * n;
            tv::lacpy_herm<UL, v2::arith::Native>(tv::fit(ex), v2::mat(a + std::ptrdiff_t(b) * per_a, n, n, lda), tile, true);
            tv::part_barrier(ex);
            const T tl = tv::sytd2<UL>(ex, tile, v, w, n, [&](int i, const T& beta, const T& t) {
                if constexpr (UL == Uplo::Lower) {
                    eb[i] = beta;
                    tb[i] = t;
                }
            });
            tv::part_barrier(ex);
            const int lane = ex.lid();
            if (lane < n) d[std::ptrdiff_t(b) * n + lane] = tile(lane, lane);
            if constexpr (UL == Uplo::Upper) {
                if (lane < n - 1) {
                    eb[lane] = tile(lane, lane + 1);
                    tb[lane] = tl;
                }
            }
            v2::lacpy(ex, tile.sub(0, 0, n, n), v2::mat(o + std::ptrdiff_t(b) * ldo * n, n, n, ldo));
        };
        const int s = static_cast<int>(sg.get_group_linear_id()) * (32 / P) + static_cast<int>(sg.get_local_linear_id()) / P;
        if constexpr (Lockstep) body(v2::Partition<P>{sg}, s);
        else body(v2::wrap(make_partition<P>(sg)), s);
    });
    const auto go = dev.down(o, ho.size()), gd = dev.down(d, hd.size()), ge = dev.down(e, he.size()), gt = dev.down(tu, ht.size());
    EXPECT_EQ(bitdiff(go, ldo * n) + bitdiff(gd, n) + bitdiff(ge, n) + bitdiff(gt, n), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const auto aref = herm_from(ha.data() + std::size_t(b) * per_a, n, lda, UL);
        const double r = sytd2_residual<T>(aref, go.data() + std::size_t(b) * ldo * n, ldo, gd.data() + std::size_t(b) * n,
                                           ge.data() + std::size_t(b) * n, gt.data() + std::size_t(b) * n, n, UL);
        worst = std::isnan(r) ? std::numeric_limits<double>::infinity() : std::max(worst, r);
    }
    EXPECT_VERIFY(T, Check::factorization, std::max(n, 1), worst)
        << (UL == Uplo::Lower ? "lower" : "upper") << " P=" << P << " n=" << n << " lockstep=" << Lockstep;
}

TYPED_TEST(DeviceBlasV2SyevTrioTest, Sytd2OnResidentTile) {
    using T = TypeParam;
    sytd2_case<T, Uplo::Upper, 16, false>(this->dev, 13);
    sytd2_case<T, Uplo::Lower, 16, false>(this->dev, 13);
    sytd2_case<T, Uplo::Upper, 32, false>(this->dev, 32);  // n at the partition ceiling
    sytd2_case<T, Uplo::Lower, 32, false>(this->dev, 32);
    sytd2_case<T, Uplo::Upper, 8, false>(this->dev, 2);
    sytd2_case<T, Uplo::Lower, 8, false>(this->dev, 1);  // no step: d only
    sytd2_case<T, Uplo::Upper, 16, true>(this->dev, 11);
    sytd2_case<T, Uplo::Lower, 16, true>(this->dev, 11);
}

// Representative 0: random; 1: x = 0 with a real alpha (tau = 0, beta = alpha bit for bit);
// 2: x = 0 with a complex alpha (a reflector for n > 1, tau = 0 at n = 1).
template <class T> void larfg_case(Dev& dev, int n, int ai) {
    constexpr int P = 8, WG = 32;
    auto hx = batched<T>(P, 300 + 17 * n + ai, [&](T* h, int rep) {
        if (rep == 0) return;
        for (int i = 0; i < P; ++i)
            if (i != ai) h[i] = T(0);
        if (rep == 1) h[ai] = T(std::real(h[ai]));
    });
    std::vector<T> hy(std::size_t(kBatch) * P, nan_v<T>()), ht(kBatch, nan_v<T>());
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    T* t = dev.up(ht);
    dev.template run<WG>(kBatch / (WG / P), [=](sycl::nd_item<1> it) {
        const auto sg = it.get_sub_group();
        const auto ex = v2::wrap(make_partition<P>(sg));
        const int b = static_cast<int>(it.get_group_linear_id()) * (WG / P) + static_cast<int>(sg.get_local_linear_id()) / P;
        const int lane = ex.lid();
        const T* xb = x + std::ptrdiff_t(b) * P;
        const bool act = lane < n && lane != ai;
        T alpha = lane == ai ? xb[lane] : T(0);
        T xv = act ? xb[lane] : T(0);
        const T tau = tv::larfg_rowfrag<v2::arith::Native>(ex, n, ai, alpha, xv, act);
        if (lane == ai) y[std::ptrdiff_t(b) * P + lane] = alpha;
        else if (act) y[std::ptrdiff_t(b) * P + lane] = xv;
        if (lane == 0) t[b] = tau;
    });
    const auto gy = dev.down(y, hy.size()), gt = dev.down(t, ht.size());
    EXPECT_EQ(bitdiff(gt, 1) + bitdiff(gy, P), 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        using L = LD<T>;
        const T* xin = hx.data() + std::size_t(b) * P;
        const T* yout = gy.data() + std::size_t(b) * P;
        const L tau = wide(gt[b]);
        if (n <= 1 || b == 1) {  // identity: tau = 0 and alpha unchanged, exactly
            EXPECT_TRUE(gt[b] == T(0)) << "n=" << n << " rep " << b;
            EXPECT_TRUE(same_bits(yout[ai], xin[ai])) << "n=" << n << " rep " << b;
            continue;
        }
        // H^H y = (beta at ai, 0 elsewhere), H = I - tau v v^H, v(ai) = 1.
        std::vector<L> v(n), z(n);
        long double ny = 0;
        for (int i = 0; i < n; ++i) v[i] = i == ai ? L(1) : wide(yout[i]), ny += std::norm(wide(xin[i]));
        L vy = 0;
        for (int i = 0; i < n; ++i) vy += conj_w(v[i]) * wide(xin[i]);
        for (int i = 0; i < n; ++i) {
            z[i] = wide(xin[i]) - conj_w(tau) * v[i] * vy;
            const L want = i == ai ? wide(yout[ai]) : L(0);
            worst = std::max(worst, static_cast<double>(nanabs<T>(z[i] - want) / std::sqrt(ny)));
        }
        if (b == 2 && is_cplx<T>) EXPECT_FALSE(gt[b] == T(0)) << "a complex alpha with x = 0 still reflects (n > 1)";
    }
    EXPECT_VERIFY(T, Check::factorization, std::max(n, 1), worst) << "larfg n=" << n << " alpha item " << ai;
}

TYPED_TEST(DeviceBlasV2SyevTrioTest, LarfgRowFrag) {
    using T = TypeParam;
    for (int n : {1, 2, 7, 8}) {
        larfg_case<T>(this->dev, n, 0);
        larfg_case<T>(this->dev, n, n - 1);
    }
}

// Fit ops on the block B = tile(o:N, o:N) (m = N - o; item r owns row r of B, i.e. tile row o + r)
// of a 32 x 32 tile (ld 35) loaded by lacpy_herm on Fit<SgpPartition<32>>:
//   y := (B x) alpha (y NaN-poisoned first);  s := x^H y;  y += -s x (core axpy through Fit);
//   B -= x y^H + y x^H (her2_full_sub). Every entry outside the block must keep its bits.
template <class T, Uplo UL> void fit_case(Dev& dev) {
    using L = LD<T>;
    using Local = sycl::local_accessor<T, 1>;
    constexpr int WG = 32, NT = 32, LDT = NT + 3, N = 24, o = 5, m = N - o, lda = N + 5;
    const T alpha = val<T>(901);
    auto ha = batched<T>(lda * N, 500 + int(UL), [&](T* h, int) { poison_herm(h, N, lda, UL); });
    auto hx = batched<T>(m, 600);
    const int per_out = NT * NT + 2 * m + 1;
    std::vector<T> ho(std::size_t(kBatch) * per_out, T(0));
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* out = dev.up(ho);
    dev.template run_acc<WG, T>(kBatch, LDT * NT + 2 * m + 7, [=](sycl::nd_item<1> it, const Local& lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto ex = v2::wrap(make_partition<NT>(it.get_sub_group()));
        const auto fx = tv::fit(ex);
        const int base = 7;  // a non-zero accessor offset
        const v2::Mat<T, v2::Ext<NT>, v2::Ext<NT>, v2::Ext<LDT>, v2::Mem::Local, v2::Cyclic, Local> t1{lm, {}, {}, {}, base};
        const auto xl = v2::vec(lm, m, base + LDT * NT), yl = v2::vec(lm, m, base + LDT * NT + m);
        T* ob = out + std::ptrdiff_t(b) * per_out;
        tv::lacpy_herm<UL, v2::arith::Native>(fx, v2::mat(a + std::ptrdiff_t(b) * lda * N, N, N, lda), t1, true);
        v2::copy(fx, v2::vec(x + std::ptrdiff_t(b) * m, m), xl);
        v2::map(fx, yl, [](T) { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }, yl);
        tv::part_barrier(ex);
        const auto B1 = t1.sub(o, o, m, m);
        tv::gemv_rows<v2::arith::Native>(fx, alpha, B1, v2::assume_published(xl), yl);  // reads t1 rows of others
        const T s = tv::dotc<v2::arith::Native>(fx, xl, yl);
        v2::axpy<v2::arith::Native>(fx, v2::Scalar<T>{v2::Arith<T>::neg(s)}, xl, yl);
        tv::part_barrier(ex);
        if (ex.lid() == 0) ob[NT * NT + 2 * m] = s;
        v2::copy(fx, yl, v2::vec(ob + NT * NT + m, m));
        v2::copy(fx, xl, v2::vec(ob + NT * NT, m));
        tv::her2_full_sub<v2::arith::Native>(fx, v2::assume_published(xl), v2::assume_published(yl), B1);
        tv::part_barrier(ex);
        v2::lacpy(ex, t1, v2::mat(ob, NT, NT, NT));
    });
    const auto g = dev.down(out, ho.size());
    EXPECT_EQ(bitdiff(g, per_out), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* gb = g.data() + std::size_t(b) * per_out;
        const auto A = herm_from(ha.data() + std::size_t(b) * lda * N, N, lda, UL);
        auto at = [&](int r, int c) { return r < N && c < N ? A[r + std::size_t(c) * N] : L(0); };
        std::vector<L> xv(m), yv(m, L(0));
        for (int i = 0; i < m; ++i) xv[i] = wide(hx[std::size_t(b) * m + i]);
        long double gscale = 0;
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < m; ++c) yv[r] += at(o + r, o + c) * xv[c] * wide(alpha), gscale += std::abs(at(o + r, o + c) * xv[c]);
        gscale *= std::abs(wide(alpha));
        L s = 0;
        for (int i = 0; i < m; ++i) s += conj_w(xv[i]) * yv[i];
        for (int i = 0; i < m; ++i) yv[i] -= s * xv[i];
        const long double scale = std::max<long double>(gscale * (1 + std::abs(s)), 1e-30L);
        worst = std::max(worst, double(nanabs<T>(wide(gb[NT * NT + 2 * m]) - s) / scale));
        for (int i = 0; i < m; ++i) worst = std::max(worst, double(nanabs<T>(wide(gb[NT * NT + m + i]) - yv[i]) / scale));
        // The device's y in the rank-2 reference: it then tests only the update itself.
        std::vector<L> yd(m);
        for (int i = 0; i < m; ++i) yd[i] = wide(gb[NT * NT + m + i]);
        for (int c = 0; c < NT; ++c)
            for (int r = 0; r < NT; ++r) {
                const bool in = r >= o && r < N && c >= o && c < N;
                const L base = at(r, c);
                L want = base;
                if (in) {
                    const int i = r - o, j = c - o;
                    want = base - (xv[i] * conj_w(yd[j]) + yd[i] * conj_w(xv[j]));
                }
                const long double sc = 1 + std::abs(base) + 2 * std::abs(xv[std::min(m - 1, std::max(0, r - o))]) *
                                                                std::abs(yd[std::min(m - 1, std::max(0, c - o))]);
                worst = std::max(worst, double(nanabs<T>(wide(gb[r + std::size_t(c) * NT]) - want) / sc));
                if (!in && !same_bits(gb[r + std::size_t(c) * NT], T(base))) worst = std::numeric_limits<double>::infinity();
            }
    }
    EXPECT_VERIFY(T, Check::blas, m, worst) << (UL == Uplo::Lower ? "lower" : "upper");
}

TYPED_TEST(DeviceBlasV2SyevTrioTest, FitOpsOnBlock) {
    fit_case<TypeParam, Uplo::Lower>(this->dev);
    fit_case<TypeParam, Uplo::Upper>(this->dev);
}

// ================================== steqr (steqr_v2.hh) ==================================

// lanst_max ignores a NaN and must read d(n-1); tridiag_split zeroes exactly the entries under
// the threshold and reports whether it did; d and e have inc 2 with a poisoned gap.
TYPED_TEST(DeviceBlasV2SyevTrioRealTest, LanstMaxAndSplitOnThread) {
    using T = TypeParam;
    constexpr int n = 9, inc = 2;
    const qv::SplitTol<T> tol{T(std::numeric_limits<T>::epsilon() / 2) * T(std::numeric_limits<T>::epsilon() / 2),
                              std::numeric_limits<T>::min()};
    std::vector<T> hd(std::size_t(kBatch) * n * inc, T(1e3)), he(std::size_t(kBatch) * n * inc, T(1e3));
    for (int b = 0; b < kBatch; ++b) {
        T* d = hd.data() + std::size_t(b) * n * inc;
        T* e = he.data() + std::size_t(b) * n * inc;
        for (int i = 0; i < n; ++i) d[i * inc] = item_val<T>(7, b, i);
        for (int i = 0; i < n - 1; ++i) e[i * inc] = item_val<T>(8, b, i);
        const int rep = b % kPeriod;
        if (rep == 0) d[(n - 1) * inc] = T(9);  // the max is the element read before the loop
        if (rep == 1) e[0] = T(-9);
        if (rep == 2) d[3 * inc] = T(-9);
        e[6 * inc] = std::numeric_limits<T>::quiet_NaN();  // ignored by lanst_max, kept by the split
        const auto thr = [&](int i) { return std::sqrt(tol.eps2 * std::abs(d[i * inc]) * std::abs(d[(i + 1) * inc]) + tol.safmin); };
        e[2 * inc] = T(2) * thr(2);                                 // kept: 4x the threshold squared
        if (rep != 0) e[(3 + rep) * inc] = T(0.5) * thr(3 + rep);  // split: 1/4 of it
    }
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    T* out = this->dev.up(std::vector<T>(2 * kBatch));
    this->dev.template run<32>(kBatch, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::Thread th{};
        const auto dvv = v2::vec(d + std::ptrdiff_t(b) * n * inc, n, inc);
        const auto ev = v2::vec(e + std::ptrdiff_t(b) * n * inc, n - 1, inc);
        out[2 * b] = qv::lanst_max(th, dvv, ev);
        out[2 * b + 1] = qv::tridiag_split(th, dvv, ev, tol) ? T(1) : T(0);
    });
    const auto go = this->dev.down(out, 2 * kBatch), ge = this->dev.down(e, he.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const int r = b % kPeriod;
        bad += !same_bits(go[2 * b], go[2 * r]) + !same_bits(go[2 * b + 1], go[2 * r + 1]);
        if (b >= kPeriod) continue;
        const T* hdb = hd.data() + std::size_t(b) * n * inc;
        const T* heb = he.data() + std::size_t(b) * n * inc;
        T mx = 0;
        for (int i = 0; i < n; ++i) mx = std::max(mx, std::abs(hdb[i * inc]));
        for (int i = 0; i < n - 1; ++i)
            if (!std::isnan(heb[i * inc])) mx = std::max(mx, std::abs(heb[i * inc]));
        bad += go[2 * b] != mx;
        bad += go[2 * b + 1] != (b != 0 ? T(1) : T(0));
        for (int i = 0; i < n - 1; ++i) {
            const T got = ge[std::size_t(b) * n * inc + i * inc];
            if (i == 6) bad += !std::isnan(got);
            else if (b != 0 && i == 3 + b) bad += got != T(0);
            else bad += !same_bits(got, heb[i * inc]);
        }
        for (int i = 1; i < n * inc; i += 2) bad += ge[std::size_t(b) * n * inc + i] != T(1e3);
    }
    EXPECT_EQ(bad, 0);
}

// tridiag_rot against the dense similarity A := G A G^T on rows/cols (i, i+1), with the bulge
// at (i-1, i+1); forward == false is checked on the reversed tridiagonal.
TYPED_TEST(DeviceBlasV2SyevTrioRealTest, TridiagRotMatchesDenseSimilarity) {
    using T = TypeParam;
    constexpr int n = 7, nprob = 2 * (n - 1) * 8;
    std::vector<T> hd(std::size_t(nprob) * n), he(std::size_t(nprob) * (n - 1)), hb(nprob);
    std::vector<v2::Givens<T>> hg(nprob);
    for (int q = 0; q < nprob; ++q) {
        for (int i = 0; i < n; ++i) hd[std::size_t(q) * n + i] = val<T>(100000 + q * 31 + i);
        for (int i = 0; i < n - 1; ++i) he[std::size_t(q) * (n - 1) + i] = val<T>(200000 + q * 31 + i);
        hb[q] = val<T>(300000 + q);
        const T t = static_cast<T>(3.14159265358979 * double(val<T>(400000 + q)));
        hg[q] = {std::cos(t), std::sin(t)};
    }
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    T* bo = this->dev.up(hb);
    const v2::Givens<T>* g = this->dev.up(hg);
    this->dev.template run<32>(nprob, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const int q = static_cast<int>(it.get_group_linear_id());
        const int i = q % (n - 1);
        const bool fwd = (q / (n - 1)) % 2 == 0;
        bo[q] = qv::tridiag_rot(v2::Thread{}, v2::vec(d + std::ptrdiff_t(q) * n, n), v2::vec(e + std::ptrdiff_t(q) * (n - 1), n - 1),
                                bo[q], i, i + 1, g[q], fwd);
    });
    const auto gd = this->dev.down(d, hd.size()), ge = this->dev.down(e, he.size()), gb = this->dev.down(bo, hb.size());
    double worst = 0;
    for (int q = 0; q < nprob; ++q) {
        const int i = q % (n - 1), j = i + 1;
        const bool fwd = (q / (n - 1)) % 2 == 0;
        const auto P = [&](int k) { return fwd ? k : n - 1 - k; };   // virtual -> physical d
        const auto Pe = [&](int k) { return fwd ? k : n - 2 - k; };  // virtual -> physical e
        std::vector<long double> A(n * n, 0.0L);
        for (int k = 0; k < n; ++k) A[k + k * n] = hd[std::size_t(q) * n + P(k)];
        for (int k = 0; k + 1 < n; ++k) A[k + (k + 1) * n] = A[k + 1 + k * n] = he[std::size_t(q) * (n - 1) + Pe(k)];
        if (i > 0) A[(i - 1) + j * n] = A[j + (i - 1) * n] = hb[q];
        const long double c = hg[q].c, s = hg[q].s;
        std::vector<long double> B = A;  // B = G A: rows i, j
        for (int k = 0; k < n; ++k) {
            B[i + k * n] = c * A[i + k * n] - s * A[j + k * n];
            B[j + k * n] = s * A[i + k * n] + c * A[j + k * n];
        }
        std::vector<long double> C = B;  // C = B G^T: columns i, j
        for (int k = 0; k < n; ++k) {
            C[k + i * n] = c * B[k + i * n] - s * B[k + j * n];
            C[k + j * n] = s * B[k + i * n] + c * B[k + j * n];
        }
        long double sc = 0;
        for (long double v : A) sc = std::max(sc, std::abs(v));
        for (int k = 0; k < n; ++k) worst = std::max(worst, rel(gd[std::size_t(q) * n + P(k)], (long double)C[k + k * n], sc));
        for (int k = 0; k + 1 < n; ++k) worst = std::max(worst, rel(ge[std::size_t(q) * (n - 1) + Pe(k)], (long double)C[k + (k + 1) * n], sc));
        const long double bulge = j + 1 < n ? C[i + (j + 1) * n] : 0.0L;
        worst = std::max(worst, rel(gb[q], bulge, sc));
    }
    EXPECT_VERIFY(T, Check::blas, 4, worst) << "tridiag_rot";
}

// One problem of n = P - 1 per chunk (the last lane is padding): lanst_max on windows that exclude
// a planted extreme on each side, the split over the whole e, find_first_lane past a leading
// zero, scal_lane on a window, then reverse of d (paired with an empty e window), and the paired
// reverse of d[2, n-1) with e (e only on even problems).
template <class T, int P, bool Masked> void lane_case(Dev& dev) {
    constexpr int WG = 32, chunks = WG / P, n = P - 1, nprob = kBatch;
    const qv::SplitTol<T> tol{T(std::numeric_limits<T>::epsilon() / 2) * T(std::numeric_limits<T>::epsilon() / 2),
                              std::numeric_limits<T>::min()};
    std::vector<T> hd(std::size_t(nprob) * n), he(std::size_t(nprob) * n, T(0));
    for (int q = 0; q < nprob; ++q) {
        T* d = hd.data() + std::size_t(q) * n;
        T* e = he.data() + std::size_t(q) * n;
        for (int i = 0; i < n; ++i) d[i] = item_val<T>(11, q, i);
        for (int i = 0; i < n - 1; ++i) e[i] = item_val<T>(12, q, i);
        d[0] = T(100);     // outside the d window [1, n-1)
        d[n - 2] = T(-5);  // inside it: the expected norm
        e[n - 2] = T(50);  // outside the e window [1, n-2)
        e[0] = T(0);       // outside the find_first window [1, n-1)
        const auto thr = [&](int i) { return std::sqrt(tol.eps2 * std::abs(d[i]) * std::abs(d[i + 1]) + tol.safmin); };
        e[1] = T(2) * thr(1);                                // kept
        e[2 + q % kPeriod] = T(0.5) * thr(2 + q % kPeriod);  // split
    }
    T* d = dev.up(hd);
    T* e = dev.up(he);
    T* nrm = dev.up(std::vector<T>(nprob));
    int* idx = dev.up(std::vector<int>(nprob));
    dev.template run<WG>(nprob / chunks, [=](sycl::nd_item<1> it) {
        const auto ex = v2::wrap(make_partition<P, Masked>(it.get_sub_group()));
        const int q = static_cast<int>(it.get_group_linear_id()) * chunks + static_cast<int>(it.get_sub_group().get_local_linear_id()) / P;
        const int l = ex.lid();
        T dl = l < n ? d[std::ptrdiff_t(q) * n + l] : T(0);
        T el = l < n - 1 ? e[std::ptrdiff_t(q) * n + l] : T(0);
        const T an = qv::lanst_max(ex, qv::lane_vec(dl, 1, n - 1), qv::lane_vec(el, 1, n - 2));
        qv::tridiag_split(ex, dl, qv::lane_vec(el, 0, n - 1), tol);
        const int first = qv::find_first_lane(ex, qv::lane_vec(el, 1, n - 1), n - 1, [](T x) { return x == T(0); });
        qv::scal_lane(ex, T(2), qv::lane_vec(dl, n - 1, n));  // the window [n-1, n): only d(n-1)
        qv::reverse(ex, qv::lane_vec(dl, 1, n - 1), qv::lane_vec(el, 0, 0), true);
        qv::reverse(ex, qv::lane_vec(dl, 2, n - 1), qv::lane_vec(el, 1, q % 2 == 0 ? n - 2 : 1), true);
        if (l < n) d[std::ptrdiff_t(q) * n + l] = dl;
        if (l < n - 1) e[std::ptrdiff_t(q) * n + l] = el;
        if (l == 0) {
            nrm[q] = an;
            idx[q] = first;
        }
    });
    const auto gd = dev.down(d, hd.size()), ge = dev.down(e, he.size()), gn = dev.down(nrm, nprob);
    const auto gi = dev.down(idx, nprob);
    int bad = 0;
    for (int q = 0; q < nprob; ++q) {
        const T* d0 = hd.data() + std::size_t(q) * n;
        std::vector<T> es(he.begin() + std::ptrdiff_t(q) * n, he.begin() + std::ptrdiff_t(q + 1) * n);
        T mx = 0;
        for (int i = 1; i < n - 1; ++i) mx = std::max(mx, std::abs(d0[i]));
        for (int i = 1; i < n - 2; ++i) mx = std::max(mx, std::abs(es[i]));
        int ff = n - 1;
        for (int i = 0; i < n - 1; ++i)
            if (es[i] != T(0) && es[i] * es[i] <= tol.eps2 * std::abs(d0[i]) * std::abs(d0[i + 1]) + tol.safmin) es[i] = T(0);
        for (int i = n - 2; i >= 1; --i)
            if (es[i] == T(0)) ff = i;
        bad += gn[q] != mx;
        bad += gi[q] != ff;
        // d reversed on [1, n-1), then [2, n-1) reversed again: a rotation of d[1, n-1); d(n-1) doubled.
        for (int i = 0; i < n; ++i) {
            const T want = i == 1 ? d0[n - 2] : (i >= 2 && i < n - 1) ? d0[i - 1] : (i == n - 1 ? T(2) * d0[i] : d0[i]);
            bad += !same_bits(gd[std::size_t(q) * n + i], want);
        }
        for (int i = 0; i < n - 1; ++i) bad += !same_bits(ge[std::size_t(q) * n + i], (q % 2 == 0 && i >= 1 && i < n - 2) ? es[n - 2 - i] : es[i]);
    }
    EXPECT_EQ(bad, 0) << "SgpPartition<" << P << ", Masked=" << Masked << ">";
}

TYPED_TEST(DeviceBlasV2SyevTrioRealTest, LaneOpsOnPartitions) {
    using T = TypeParam;
    lane_case<T, 8, true>(this->dev);
    lane_case<T, 16, false>(this->dev);
    lane_case<T, 32, true>(this->dev);  // n = 31: one pad lane at the ceiling
}

}  // namespace

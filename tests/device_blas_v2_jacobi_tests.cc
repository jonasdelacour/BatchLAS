// SPDX-License-Identifier: MIT
// Device LA Jacobi family ops (src/kernels/common/dla_v2_jacobi.hh, private to syev_jacobi_cta
// and gesvdj_cta) against host references; the core ops they compose with are covered by
// device_blas_v2_tests. Guards (docs/developer/agent-guide.md section 8): tiles with ld = P + 1
// and Rows + 1, a global matrix with ld = n + 3, complex data with nonzero imaginary parts, the
// unread triangle NaN-poisoned and Im(diag) poisoned, batch 1024 with period-3 items bit-identical
// to their representative (all SLM paths), both sides of the sym_schur2 threshold plus its
// tau_big, s == 0 and below-tiny branches, a NaN lane under FMax / FMin, odd n (a padded pivot
// index), the masked SgpPartition, arith::Native against arith::Explicit.
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
#include <src/kernels/common/dla_v2_jacobi.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;

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
template <class T> T mk(real_of<T> re, real_of<T> im) {
    if constexpr (is_cplx<T>) return T(re, im);
    else return re;
}

// kPeriod representatives (edited by fix) copied to every item b from representative b % kPeriod.
template <class T, class Fix> std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix) {
    std::vector<T> h(static_cast<std::size_t>(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item);
    }
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    return batched<T>(per_item, seed, [](T*) {});
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> int bitdiff_vs_rep(const std::vector<T>& v, int per_item) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i)
            d += !same_bits(v[std::size_t(b) * per_item + i], v[std::size_t(b % kPeriod) * per_item + i]);
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
    template <class T> T* zeros(std::size_t n) { return up(std::vector<T>(n)); }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG),
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

template <class T> class DeviceBlasV2JacobiTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2JacobiTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> LD<T> conj_h(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// ---- pivot schedule ------------------------------------------------------------------------

// Every round is a perfect matching with p < q, the m - 1 rounds cover every pair once, packing
// round-trips; the device builds the same table bit for bit.
TEST(DeviceBlasV2JacobiSchedule, RoundRobinCoversEveryPairOnce) {
    for (int m = 2; m <= 64; m += 2) {
        std::vector<int> seen(std::size_t(m) * m, 0);
        for (int t = 0; t < m - 1; ++t) {
            std::vector<int> used(m, 0);
            for (int k = 0; k < m / 2; ++k) {
                const device::jacobi_detail::PivotPair pq = device::jacobi_detail::round_robin_pair(m, t, k);
                ASSERT_TRUE(0 <= pq.p && pq.p < pq.q && pq.q < m) << "m " << m << " t " << t << " k " << k;
                EXPECT_EQ(++used[pq.p], 1);
                EXPECT_EQ(++used[pq.q], 1);
                ++seen[std::size_t(pq.p) * m + pq.q];
                const device::jacobi_detail::PivotPair back = device::jacobi_detail::unpack_pair(device::jacobi_detail::pack_pair(pq));
                EXPECT_TRUE(back.p == pq.p && back.q == pq.q);
            }
        }
        for (int p = 0; p < m; ++p)
            for (int q = p + 1; q < m; ++q) EXPECT_EQ(seen[std::size_t(p) * m + q], 1) << "m " << m << " pair " << p << "," << q;
    }
    Dev dev;
    if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
    if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "launches sub-group size 32";
    constexpr int m = 64, slots = (m - 1) * (m / 2);
    std::int16_t* d = dev.zeros<std::int16_t>(slots);
    dev.run<128>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        for (int idx = static_cast<int>(it.get_local_linear_id()); idx < slots; idx += 128) {
            const int t = idx / (m / 2);
            d[idx] = device::jacobi_detail::pack_pair(device::jacobi_detail::round_robin_pair(m, t, idx - t * (m / 2)));
        }
    });
    const auto got = dev.down(d, slots);
    for (int idx = 0; idx < slots; ++idx)
        EXPECT_EQ(got[idx], device::jacobi_detail::pack_pair(device::jacobi_detail::round_robin_pair(m, idx / (m / 2), idx % (m / 2)))) << "slot " << idx;
}

// ---- sym_schur2 ----------------------------------------------------------------------------

template <class T> struct Case2 {
    real_of<T> app, aqq;
    T apq;
    int expect;  // 1 active, 0 identity
};

// Active: U = [c s; -d s  d c] makes U^H H U diagonal, c^2 + s^2 = 1, |d| = 1, t = s / c and g the
// phase-free off-diagonal (an independent long-double check of the definition). Identity:
// below the threshold, below tiny, a_pq = 0, or s rounded to 0, returned exactly.
TYPED_TEST(DeviceBlasV2JacobiTest, SymSchur2DiagonalisesOrReturnsIdentity) {
    using T = TypeParam;
    using R = real_of<T>;
    const auto th = device::jacobi_detail::JacobiThreshold<R>::relative(R(1), 8);
    const R thresh2 = th.tol * std::sqrt(R(2));  // the threshold at app = 1, aqq = 2
    const R mn = std::numeric_limits<R>::min();
    std::vector<Case2<T>> cs = {
        {R(1), R(2), mk<T>(thresh2 * R(1.01), R(0)), 1},  // just above the relative threshold
        {R(1), R(2), mk<T>(thresh2 * R(0.99), R(0)), 0},  // just below it
        {R(0.5), R(-0.25), mk<T>(R(0), R(0)), 0},         // a_pq = 0
        {R(0), R(1), mk<T>(R(1e-12), R(1e-12)), 1},       // |tau| > tau_big: t = 1 / (2 tau)
        {R(0), R(1e30), mk<T>(R(4) * mn, R(0)), 0},       // tau overflows, s rounds to 0
        {R(1), R(1), mk<T>(mn / R(4), R(0)), 0},          // a_pq below tiny
        {R(1e-6), R(1e6), mk<T>(R(1e-3), R(-2e-3)), 1},   // graded diagonal
        {R(3), R(-1), mk<T>(R(-0.5), R(0.25)), 1},        // indefinite, tau < 0
        {R(2), R(2), mk<T>(R(0.75), R(0)), 1},            // tau = 0
        {R(-1), R(4), mk<T>(R(0), R(-0.5)), 1},           // purely imaginary a_pq (real T: 0, identity)
    };
    if constexpr (!is_cplx<T>) cs.back().expect = 0;
    for (std::uint64_t i = 0; cs.size() < 64; ++i) {
        T apq = val<T>(500 + i);
        if (std::abs(apq) < R(0.1)) apq = apq + mk<T>(R(0.5), R(0));
        cs.push_back({uval<R>(900 + i), uval<R>(1900 + i), apq, 1});
    }
    const int nc = static_cast<int>(cs.size());
    std::vector<R> happ(nc), haqq(nc);
    std::vector<T> hapq(nc);
    for (int i = 0; i < nc; ++i) happ[i] = cs[i].app, haqq[i] = cs[i].aqq, hapq[i] = cs[i].apq;
    R* app = this->dev.up(happ);
    R* aqq = this->dev.up(haqq);
    T* apq = this->dev.up(hapq);
    R* oc = this->dev.template zeros<R>(nc);
    R* os = this->dev.template zeros<R>(nc);
    R* ot = this->dev.template zeros<R>(nc);
    R* og = this->dev.template zeros<R>(nc);
    T* od = this->dev.template zeros<T>(nc);
    int* oa = this->dev.template zeros<int>(nc);
    this->dev.template run<64>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int i = static_cast<int>(it.get_local_linear_id());
        if (i >= nc) return;
        const device::jacobi_detail::JacobiRotation<T> r = device::sym_schur2<T>(app[i], aqq[i], apq[i], th);
        oc[i] = r.c, os[i] = r.s, ot[i] = r.t, og[i] = r.g, od[i] = r.d, oa[i] = r.active ? 1 : 0;
    });
    const auto c = this->dev.down(oc, nc), s = this->dev.down(os, nc), t = this->dev.down(ot, nc), g = this->dev.down(og, nc);
    const auto d = this->dev.down(od, nc);
    const auto a = this->dev.down(oa, nc);
    for (int i = 0; i < nc; ++i) {
        EXPECT_EQ(a[i], cs[i].expect) << "case " << i;
        if (!a[i]) {
            EXPECT_TRUE(c[i] == R(1) && s[i] == R(0) && same_bits(d[i], T(1))) << "case " << i << ": not the exact identity";
            continue;
        }
        using L = LD<T>;
        const long double lc = c[i], ls = s[i];
        const L ld = wide(d[i]), hpq = wide(cs[i].apq);
        // U^H H U, entry (0, 1)
        const L u11 = lc, u12 = ls, u21 = -ld * ls, u22 = ld * lc;
        const L h11 = cs[i].app, h12 = hpq, h21 = conj_h<T>(hpq), h22 = cs[i].aqq;
        const L hu12 = h11 * u12 + h12 * u22, hu22 = h21 * u12 + h22 * u22;
        const L b12 = conj_h<T>(u11) * hu12 + conj_h<T>(u21) * hu22;
        const long double scale = std::abs(cs[i].app) + std::abs(cs[i].aqq) + std::abs(hpq);
        EXPECT_VERIFY(T, Check::eigen_residual, 2, static_cast<double>(std::abs(b12) / scale)) << "case " << i << " off-diagonal";
        EXPECT_VERIFY(T, Check::blas, 2, static_cast<double>(std::abs(lc * lc + ls * ls - 1.0L))) << "case " << i << " c^2 + s^2";
        EXPECT_VERIFY(T, Check::blas, 2, static_cast<double>(std::abs(std::abs(ld) - 1.0L))) << "case " << i << " |d|";
        EXPECT_VERIFY(T, Check::blas, 2, static_cast<double>(std::abs(t[i] - ls / lc) / std::abs(ls / lc))) << "case " << i << " t";
        const long double gref = is_cplx<T> ? std::abs(hpq) : std::real(hpq);
        EXPECT_VERIFY(T, Check::blas, 2, static_cast<double>(std::abs(g[i] - gref) / std::abs(gref))) << "case " << i << " g";
    }
}

// ---- rotation rounds: a two-sided Jacobi eigensolver on one partition ------------------------

template <class T> struct SweepOut {
    std::vector<T> a, z;
    std::vector<int> sweeps;
};

// syev_jacobi_cta's composition on SgpPartition<P> (ld = P + 1 tiles): Lower triangle of an
// ld = n + 3 matrix whose upper triangle is NaN and Im(diag) 1e6. Returns A_final and Z.
template <class T, int P, class Ops> SweepOut<T> jacobi_sweeps(Dev& dev, int n) {
    using R = real_of<T>;
    using Vec2 = sycl::vec<R, 2>;
    constexpr int LDT = P + 1, kTile = P * LDT, kSlots = P / 2, kPairs = (P - 1) * kSlots, parts = 32 / P;
    const int lda = n + 3, m = (n % 2 == 0) ? n : n + 1, groups = kBatch / parts;
    const auto th = device::jacobi_detail::JacobiThreshold<R>::relative(R(1), n);
    auto hA = batched<T>(lda * n, 11, [&](T* a) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < j; ++i) a[i + j * lda] = T(std::numeric_limits<R>::quiet_NaN());
        if constexpr (is_cplx<T>)
            for (int i = 0; i < n; ++i) a[i + i * lda] = T(a[i + i * lda].real(), R(1e6));
    });
    T* gA = dev.up(hA);
    T* oA = dev.template zeros<T>(std::size_t(kBatch) * n * n);
    T* oZ = dev.template zeros<T>(std::size_t(kBatch) * n * n);
    int* oS = dev.template zeros<int>(kBatch);
    const std::size_t off_rcs = std::size_t(parts) * 2 * kTile * sizeof(T);
    const std::size_t off_rd = off_rcs + std::size_t(parts) * kSlots * sizeof(Vec2);
    const std::size_t off_pairs = off_rd + std::size_t(parts) * kSlots * sizeof(T);
    const std::size_t bytes = off_pairs + kPairs * sizeof(std::int16_t);
    dev.template run<32>(groups, bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        using Tile = device::Mat<T, device::Ext<P>, device::Ext<P>, device::Ext<LDT>, device::Mem::Local>;
        const auto sg = it.get_sub_group();
        const device::SgpPartition<P> ex{sg};
        const int part = static_cast<int>(ex.part.get_group_linear_id());
        const int b = static_cast<int>(it.get_group_linear_id()) * parts + part;
        const int lane = ex.lid();
        Vec2* rcs = reinterpret_cast<Vec2*>(lm + off_rcs) + part * kSlots;
        T* rd = reinterpret_cast<T*>(lm + off_rd) + part * kSlots;
        std::int16_t* pairs = reinterpret_cast<std::int16_t*>(lm + off_pairs);
        for (int idx = static_cast<int>(it.get_local_linear_id()); idx < (m - 1) * (m / 2); idx += 32) {
            const int t = idx / (m / 2);
            pairs[idx] = device::jacobi_detail::pack_pair(device::jacobi_detail::round_robin_pair(m, t, idx - t * (m / 2)));
        }
        sycl::group_barrier(it.get_group());
        T* tile = reinterpret_cast<T*>(lm) + part * 2 * kTile;
        const Tile A{tile, {}, {}, {}};
        const Tile Z{tile + kTile, {}, {}, {}};
        const auto Ag = device::mat(gA + std::ptrdiff_t(b) * lda * n, n, n, lda);
        for (int c = 0; c < P; ++c) {
            T v = T(0);
            if (lane < n && c < n) v = (lane >= c) ? Ag(lane, c) : device::conj(Ag(c, lane));
            A(lane, c) = (lane == c) ? device::force_real(v) : v;
            Z(lane, c) = (lane == c && lane < n) ? T(1) : T(0);
        }
        group_barrier(ex.part);
        int sweeps = 0;
        bool conv = false;
        for (; sweeps < 30 && !conv; ++sweeps) {
            int rot = 0;
            for (int t = 0; t < m - 1; ++t) {
                const auto rnd = device::jacobi_detail::rot_round<T>(device::vec<device::Mem::Local>(rcs, kSlots), device::vec<device::Mem::Local>(rd, kSlots),
                                                  device::vec<device::Mem::Local>(pairs + t * (m / 2), m / 2));
                device::jacobi_detail::PivotPair pq{0, 0};
                bool active = false;
                if (lane < m / 2) {
                    pq = rnd.pair(lane);
                    active = pq.q < n;
                }
                device::jacobi_detail::JacobiRotation<T> jr = device::jacobi_detail::identity_rotation<T>();
                if (active) jr = device::sym_schur2<T>(device::Arith<T>::re(A(pq.p, pq.p)), device::Arith<T>::re(A(pq.q, pq.q)), static_cast<T>(A(pq.p, pq.q)), th);
                if (lane < m / 2) rnd.store(lane, jr);
                group_barrier(ex.part);
                const int cnt = device::allreduce(ex, jr.active ? 1 : 0);
                rot += cnt;
                if (cnt == 0) continue;
                if (lane < n) device::jacobi_detail::rot_round_right<Ops>(rnd, m / 2, lane, A, Z);
                group_barrier(ex.part);
                if (lane < n) device::jacobi_detail::rot_round_left_hermitian<Ops>(rnd, m / 2, lane, A);
                group_barrier(ex.part);
            }
            conv = (rot == 0);
        }
        if (lane < n) {
            for (int c = 0; c < n; ++c) {
                oA[std::ptrdiff_t(b) * n * n + lane + c * n] = A(lane, c);
                oZ[std::ptrdiff_t(b) * n * n + lane + c * n] = Z(lane, c);
            }
        }
        if (lane == 0) oS[b] = conv ? sweeps : -sweeps;
    });
    SweepOut<T> out{dev.down(oA, std::size_t(kBatch) * n * n), dev.down(oZ, std::size_t(kBatch) * n * n), dev.down(oS, kBatch)};

    // Host checks on the representatives: convergence, Z unitary, Z A Z^H = A0, a real diagonal,
    // every off-diagonal under the relative threshold, and every item equal to its representative.
    for (int b = 0; b < kPeriod; ++b) {
        EXPECT_GT(out.sweeps[b], 0) << "item " << b << " did not converge";
        const T* a0 = hA.data() + std::size_t(b) * lda * n;
        const T* af = out.a.data() + std::size_t(b) * n * n;
        const T* z = out.z.data() + std::size_t(b) * n * n;
        auto A0 = [&](int i, int j) -> LD<T> {
            if (i == j) return LD<T>(std::real(wide(a0[i + i * lda])));
            return i > j ? wide(a0[i + j * lda]) : conj_h<T>(wide(a0[j + i * lda]));
        };
        long double anorm = 0, orth = 0, res = 0, off = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) anorm = std::max(anorm, static_cast<long double>(std::abs(A0(i, j))));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                LD<T> zz = 0, zaz = 0;
                for (int k = 0; k < n; ++k) zz += conj_h<T>(wide(z[k + i * n])) * wide(z[k + j * n]);
                for (int k = 0; k < n; ++k)
                    for (int l = 0; l < n; ++l) zaz += wide(z[i + k * n]) * wide(af[k + l * n]) * conj_h<T>(wide(z[j + l * n]));
                orth = std::max(orth, static_cast<long double>(std::abs(zz - LD<T>(i == j ? 1 : 0))));
                res = std::max(res, static_cast<long double>(std::abs(zaz - A0(i, j))));
                if (i != j) {
                    const long double bound = 1.01L * th.tol * std::sqrt(std::abs(std::real(wide(af[i + i * n]))) * std::abs(std::real(wide(af[j + j * n])))) + th.tiny;
                    off = std::max(off, static_cast<long double>(std::abs(wide(af[i + j * n]))) / bound);
                }
            }
            if constexpr (is_cplx<T>) EXPECT_EQ(std::imag(af[i + i * n]), real_of<T>(0)) << "item " << b << " Im(diag) " << i;
        }
        EXPECT_VERIFY(T, Check::orthogonality_rotations, n, static_cast<double>(orth)) << "item " << b << " Z^H Z";
        EXPECT_VERIFY(T, Check::eigen_residual, n, static_cast<double>(res / anorm)) << "item " << b << " Z A Z^H - A0";
        EXPECT_LE(static_cast<double>(off), 1.0) << "item " << b << " off-diagonal above the threshold";
    }
    EXPECT_EQ(bitdiff_vs_rep(out.a, n * n) + bitdiff_vs_rep(out.z, n * n), 0) << "items differ from their representative";
    return out;
}

template <class T, int P> void jacobi_case(Dev& dev, int n) {
    SCOPED_TRACE(::testing::Message() << "P " << P << " n " << n);
    const auto nat = jacobi_sweeps<T, P, device::arith::Native>(dev, n);
    const auto expl = jacobi_sweeps<T, P, device::arith::Explicit>(dev, n);
    if constexpr (!is_cplx<T>) {
        int d = 0;
        for (std::size_t i = 0; i < nat.a.size(); ++i) d += !same_bits(nat.a[i], expl.a[i]) + !same_bits(nat.z[i], expl.z[i]);
        EXPECT_EQ(d, 0) << "arith::Explicit must equal arith::Native for real T";
    }
}

TYPED_TEST(DeviceBlasV2JacobiTest, RotationRoundsSolveTheEigenproblem) {
    using T = TypeParam;
    jacobi_case<T, 8>(this->dev, 7);    // odd n: index 7 is padded, never pivoted
    jacobi_case<T, 16>(this->dev, 16);  // even n filling the partition
    jacobi_case<T, 32>(this->dev, 31);  // the ceiling partition
}

// ---- sub-group collectives ---------------------------------------------------------------------

enum ScatterSlot { kScatter, kSlotFMax, kSlotFMin, kScatterSlots };

// reduce_scatter<K> on SgpPartition<P>: lane l holds the sum over lanes of x[l / (P / K)]. In the
// same launch: FMax / FMin of re(x[0]) with lane 3 NaN.
template <class T, int P, int K> void scatter_case(Dev& dev) {
    SCOPED_TRACE(::testing::Message() << "P " << P << " K " << K);
    using R = real_of<T>;
    constexpr int parts = 32 / P, groups = kBatch / parts;
    auto hx = batched<T>(P * K, 77 + P * 10 + K);
    T* x = dev.up(hx);
    T* o = dev.template zeros<T>(std::size_t(kBatch) * P * kScatterSlots);
    dev.template run<32>(groups, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto sg = it.get_sub_group();
        const device::SgpPartition<P> ex{sg};
        const int b = static_cast<int>(it.get_group_linear_id()) * parts + static_cast<int>(ex.part.get_group_linear_id());
        const int l = ex.lid();
        device::Frag<T, K> f;
        for (int k = 0; k < K; ++k) f[k] = x[(std::ptrdiff_t(b) * P + l) * K + k];
        const R r0 = device::Arith<T>::re(f[0]);
        const R lead = (l == 3) ? std::numeric_limits<R>::quiet_NaN() : r0;
        const R mx = device::allreduce(ex, lead, device::op::FMax{});
        const R mn = device::allreduce(ex, lead, device::op::FMin{});
        const T sc = device::reduce_scatter<K>(ex, f);
        T* ob = o + (std::ptrdiff_t(b) * P + l) * kScatterSlots;
        ob[kScatter] = sc, ob[kSlotFMax] = T(mx), ob[kSlotFMin] = T(mn);
    });
    const auto got = dev.down(o, std::size_t(kBatch) * P * kScatterSlots);
    EXPECT_EQ(bitdiff_vs_rep(got, P * kScatterSlots), 0) << "items differ from their representative";
    for (int b = 0; b < kPeriod; ++b) {
        const T* xb = hx.data() + std::size_t(b) * P * K;
        long double mx = -INFINITY, mn = INFINITY;
        for (int j = 0; j < P; ++j) {
            const long double r0 = std::real(wide(xb[j * K]));
            if (j != 3) mx = std::max(mx, r0), mn = std::min(mn, r0);
        }
        for (int l = 0; l < P; ++l) {
            const T* g = got.data() + (std::size_t(b) * P + l) * kScatterSlots;
            LD<T> s = 0;
            long double sa = 0;
            for (int j = 0; j < P; ++j) s += wide(xb[j * K + l / (P / K)]), sa += std::abs(wide(xb[j * K + l / (P / K)]));
            EXPECT_VERIFY(T, Check::blas, P, rel(g[kScatter], s, sa)) << "item " << b << " lane " << l;
            EXPECT_EQ(std::real(g[kSlotFMax]), R(mx)) << "FMax drops the NaN lane";
            EXPECT_EQ(std::real(g[kSlotFMin]), R(mn)) << "FMin drops the NaN lane";
        }
    }
}

// col_sumsq over a Rows x P tile (ld = Rows + 1) whose rows are Cyclic over SgpPartition<P>.
template <class T, int P, int Rows> void col_sumsq_case(Dev& dev) {
    SCOPED_TRACE(::testing::Message() << "col_sumsq P " << P << " Rows " << Rows);
    using R = real_of<T>;
    constexpr int LDT = Rows + 1, parts = 32 / P, groups = kBatch / parts;
    auto ha = batched<T>(Rows * P, 333 + Rows);
    T* a = dev.up(ha);
    R* o = dev.template zeros<R>(std::size_t(kBatch) * P);
    dev.template run<32>(groups, std::size_t(parts) * LDT * P * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        using Tile = device::Mat<T, device::Ext<Rows>, device::Ext<P>, device::Ext<LDT>, device::Mem::Local>;
        const auto sg = it.get_sub_group();
        const device::SgpPartition<P> ex{sg};
        const int part = static_cast<int>(ex.part.get_group_linear_id());
        const int b = static_cast<int>(it.get_group_linear_id()) * parts + part;
        const Tile A{reinterpret_cast<T*>(lm) + part * LDT * P, {}, {}, {}};
        device::for_each_owned_frag<Rows, device::unroll::Default>(ex, [&](int, int r) {
            for (int c = 0; c < P; ++c) A(r, c) = a[std::ptrdiff_t(b) * Rows * P + r + c * Rows];
        });
        group_barrier(ex.part);
        o[std::ptrdiff_t(b) * P + ex.lid()] = device::col_sumsq(ex, A);
    });
    const auto got = dev.down(o, std::size_t(kBatch) * P);
    EXPECT_EQ(bitdiff_vs_rep(got, P), 0) << "items differ from their representative";
    for (int b = 0; b < kPeriod; ++b)
        for (int c = 0; c < P; ++c) {
            long double s = 0;
            for (int r = 0; r < Rows; ++r) s += std::norm(wide(ha[std::size_t(b) * Rows * P + r + c * Rows]));
            EXPECT_VERIFY(T, Check::blas, Rows, rel(T(got[std::size_t(b) * P + c]), LD<T>(s), s)) << "item " << b << " column " << c;
        }
}

TYPED_TEST(DeviceBlasV2JacobiTest, ReduceScatterFMinMaxAndColumnNorms) {
    using T = TypeParam;
    scatter_case<T, 4, 4>(this->dev);    // exact_norms shape: K = P, no all-reduce
    scatter_case<T, 4, 2>(this->dev);    // Gram shape: K = P / 2, one all-reduce step
    scatter_case<T, 8, 2>(this->dev);    // K < P / 2: two all-reduce steps
    scatter_case<T, 32, 32>(this->dev);
    scatter_case<T, 32, 16>(this->dev);  // gesvdj_cta's 4 scatter + 1 all-reduce
    col_sumsq_case<T, 8, 16>(this->dev);   // two rows per lane
    col_sumsq_case<T, 32, 32>(this->dev);  // one row per lane
}

// ---- per-item fragment ops and 2x2 applications -------------------------------------------------

enum FragRow { kX, kY, kY2, kE, kRx, kRy, kRxE, kRyE, kLx, kLy, kLxE, kLyE, kRowSlots };
enum FragScalar { kDotN, kDotE, kSsq, kScalarSlots };

// SgpPartition<32> over 64 rows (two per lane), n = 50 masked rows, e_37; rot_right / rot_left of
// (x_r, y_r) for both arithmetic policies against the long-double U = [c s; -d s  d c].
TYPED_TEST(DeviceBlasV2JacobiTest, FragmentOpsAndRotationsMatchHost) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int Rows = 64, n = 50, j = 37, per = Rows * kRowSlots + kScalarSlots;
    auto hx = batched<T>(Rows, 5), hy = batched<T>(Rows, 6);
    // Per item: a, the unit phase d (1 for real T), and (c, s) of an angle, all fixed on the host
    // so the reference uses the device's exact inputs.
    std::vector<T> hp(std::size_t(kBatch) * 2);
    std::vector<R> hcs(std::size_t(kBatch) * 2);
    for (int b = 0; b < kBatch; ++b) {
        const int rep = b % kPeriod;
        const long double ang = 0.3L + 0.7L * rep;
        hp[2 * b] = val<T>(70 + rep);
        if constexpr (is_cplx<T>) hp[2 * b + 1] = T(R(std::cos(1.1L + rep)), R(std::sin(1.1L + rep)));
        else hp[2 * b + 1] = T(1);
        hcs[2 * b] = R(std::cos(ang)), hcs[2 * b + 1] = R(std::sin(ang));
    }
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    T* pr = this->dev.up(hp);
    R* pcs = this->dev.up(hcs);
    T* o = this->dev.template zeros<T>(std::size_t(kBatch) * per);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        using Rows2 = device::Frag<T, 2>;
        using EOps = device::arith::Explicit;
        const device::SgpPartition<32> ex{it.get_sub_group()};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto xv = device::vec(x + std::ptrdiff_t(b) * Rows, Rows);
        const auto yv = device::vec(y + std::ptrdiff_t(b) * Rows, Rows);
        const T a = pr[2 * b];
        const T d = pr[2 * b + 1];
        const R c = pcs[2 * b], s = pcs[2 * b + 1];
        Rows2 X, Y, Y2, E;
        device::load_owned_masked(ex, xv, n, X);
        device::load_owned_masked(ex, yv, Rows, Y);
        Y2 = Y;
        device::set_unit_owned(ex, j, E);
        const T dn = device::allreduce(ex, device::dotc_owned(X, Y));
        const T de = device::allreduce(ex, device::dotc_owned<EOps>(X, Y));
        const device::jacobi_detail::Rot2<T> ur = device::jacobi_detail::rot2<false>(c, s, d), ul = device::jacobi_detail::rot2<true>(c, s, d);
        const device::jacobi_detail::Rot2<T> uer = device::jacobi_detail::rot2<false, EOps>(c, s, d), uel = device::jacobi_detail::rot2<true, EOps>(c, s, d);
        T* ob = o + std::ptrdiff_t(b) * per;
        device::for_each_owned_frag<Rows>(ex, [&](int k, int r) {
            const device::jacobi_detail::Pair2<T> r1 = device::jacobi_detail::rot_right(ur, X[k], Y[k]), r2 = device::jacobi_detail::rot_right<EOps>(uer, X[k], Y[k]);
            const device::jacobi_detail::Pair2<T> l1 = device::jacobi_detail::rot_left(ul, X[k], Y[k]), l2 = device::jacobi_detail::rot_left<EOps>(uel, X[k], Y[k]);
            T* row = ob + r * kRowSlots;
            row[kRx] = r1.x, row[kRy] = r1.y, row[kRxE] = r2.x, row[kRyE] = r2.y;
            row[kLx] = l1.x, row[kLy] = l1.y, row[kLxE] = l2.x, row[kLyE] = l2.y;
        });
        device::sub_mul(Y, X, a);
        device::sub_mul<EOps>(Y2, X, a);
        const R ss = device::allreduce(ex, device::sumsq_owned(ex, Y, n));
        device::for_each_owned_frag<Rows>(ex, [&](int k, int r) {
            T* row = ob + r * kRowSlots;
            row[kX] = X[k], row[kY] = Y[k], row[kY2] = Y2[k], row[kE] = E[k];
        });
        if (ex.leader()) ob[Rows * kRowSlots + kDotN] = dn, ob[Rows * kRowSlots + kDotE] = de, ob[Rows * kRowSlots + kSsq] = T(ss);
    });
    const auto got = this->dev.down(o, std::size_t(kBatch) * per);
    EXPECT_EQ(bitdiff_vs_rep(got, per), 0) << "items differ from their representative";
    for (int b = 0; b < kPeriod; ++b) {
        const T* g = got.data() + std::size_t(b) * per;
        const T* xb = hx.data() + std::size_t(b) * Rows;
        const T* yb = hy.data() + std::size_t(b) * Rows;
        const LD<T> a = wide(hp[2 * b]), d = wide(hp[2 * b + 1]);
        const long double c = hcs[2 * b], s = hcs[2 * b + 1];
        LD<T> dot = 0;
        long double sdot = 0, ssq = 0;
        double eY = 0, eR = 0, eL = 0;
        int exact = 0, eExp = 0;
        for (int r = 0; r < Rows; ++r) {
            const T* row = g + r * kRowSlots;
            const LD<T> xr = r < n ? wide(xb[r]) : LD<T>(0), yr = wide(yb[r]);
            dot += conj_h<T>(xr) * yr, sdot += std::abs(xr) * std::abs(yr);
            const LD<T> ynew = yr - xr * a;
            if (r < n) ssq += std::norm(ynew);
            const long double sc = std::abs(xr) + std::abs(yr);
            const long double sy = std::abs(yr) + std::abs(xr) * std::abs(a);
            exact += !same_bits(row[kX], T(xr)) + !same_bits(row[kE], T(r == j ? 1 : 0));
            eY = std::max({eY, rel(row[kY], ynew, sy), rel(row[kY2], ynew, sy)});
            // [x y] U and U^H [x; y] for U = [c s; -d s  d c]
            const LD<T> rx = xr * c + yr * (-d * s), ry = xr * s + yr * (d * c);
            const LD<T> lx = c * xr + (-conj_h<T>(d) * s) * yr, ly = s * xr + (conj_h<T>(d) * c) * yr;
            eR = std::max({eR, rel(row[kRx], rx, sc), rel(row[kRy], ry, sc), rel(row[kRxE], rx, sc), rel(row[kRyE], ry, sc)});
            eL = std::max({eL, rel(row[kLx], lx, sc), rel(row[kLy], ly, sc), rel(row[kLxE], lx, sc), rel(row[kLyE], ly, sc)});
            if constexpr (!is_cplx<T>) {
                eExp += !same_bits(row[kRx], row[kRxE]) + !same_bits(row[kRy], row[kRyE]) + !same_bits(row[kLx], row[kLxE]) +
                        !same_bits(row[kLy], row[kLyE]) + !same_bits(row[kY], row[kY2]);
            }
        }
        EXPECT_EQ(exact, 0) << "load_owned_masked / set_unit_owned copy exactly";
        EXPECT_VERIFY(T, Check::blas, 2, eY) << "item " << b << " sub_mul";
        EXPECT_VERIFY(T, Check::blas, 4, eR) << "item " << b << " rot_right";
        EXPECT_VERIFY(T, Check::blas, 4, eL) << "item " << b << " rot_left";
        EXPECT_EQ(eExp, 0) << "arith::Explicit must equal arith::Native for real T";
        const T* sc = g + Rows * kRowSlots;
        EXPECT_VERIFY(T, Check::blas, n, rel(sc[kDotN], dot, sdot)) << "item " << b << " dotc_owned";
        EXPECT_VERIFY(T, Check::blas, n, rel(sc[kDotE], dot, sdot)) << "item " << b << " dotc_owned arith::Explicit";
        EXPECT_VERIFY(T, Check::blas, n + 4, rel(T(std::real(sc[kSsq])), LD<T>(ssq), ssq)) << "item " << b << " sumsq_owned";
    }
}

}  // namespace

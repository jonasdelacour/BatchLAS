// SPDX-License-Identifier: MIT
// The family-private device LA v2 ops of larft / ormqr_cta (src/kernels/common/dla_v2_larft.hh:
// larf_frag, trmv_upper_row_owner, larft_forward_colwise) and the larft kernels behind them
// (src/kernels/common/larft_wy.hh), against long-double host references. Shared ops (gemv, copy,
// map, laset, core larf) are covered by device_blas_v2_tests.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld / inc / stride / offset,
// complex data with nonzero imaginary parts, the unread triangle and the rows past the panel
// poisoned with a large finite value, batch 1024 with period-3 data where every item must be
// bit-identical to its representative, masked items left untouched, a WG = 1024 launch, a zero
// tau, a trmv recurrence whose rows wrap the executor (n > WG), every larft WG rung and both sides
// of the m <= 64 rung boundary.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/dla_v2_larft.hh>
#include <src/kernels/common/larft_wy.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

struct V2LarftTestTag;  // kernel-name tag for the larft instantiations made here

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kBatch = 1024;
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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
// Accepted poison: finite, so a read of it survives the arithmetic and wrecks the result.
template <class T> T poison() {
    if constexpr (is_cplx<T>) return T(real_of<T>(1e30), real_of<T>(-1e30));
    else return T(1e30);
}

// kPeriod representatives (edited by fix) copied to every item b from b % kPeriod.
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

// Items whose per_item block differs bitwise from its representative's (skip: items left out).
template <class T> int items_differing(const std::vector<T>& got, int per_item, const std::vector<int>& skip = {}) {
    int bad = 0;
    for (int b = kPeriod; b < kBatch; ++b) {
        if (std::find(skip.begin(), skip.end(), b) != skip.end()) continue;
        for (int i = 0; i < per_item; ++i)
            if (!same_bits(got[std::size_t(b) * per_item + i], got[std::size_t(b % kPeriod) * per_item + i])) {
                ++bad;
                break;
            }
    }
    return bad;
}

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

template <class T> class DeviceBlasV2LarftTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2LarftTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}

template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// ---- trmv_upper_row_owner ------------------------------------------------------------------

// y := A x, A n x n upper with ld = n + 5 and the strictly lower part poisoned, x with inc 2.
template <int WG, class T, class MakeEx>
void trmv_case(Dev& dev, MakeEx make_ex, int items_per_wg, int lanes_per_item, const char* where) {
    constexpr int n = 23, ld = n + 5, inc = 2;
    auto ha = batched<T>(ld * n, 31, [](T* a) {
        for (int c = 0; c < n; ++c)
            for (int r = c + 1; r < ld; ++r) a[r + c * ld] = poison<T>();
    });
    auto hx = batched<T>(n * inc, 32);
    std::vector<T> hy(std::size_t(kBatch) * n, nan_v<T>());
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    dev.template run<WG>(kBatch / items_per_wg, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id()) * items_per_wg +
                      static_cast<int>(it.get_local_linear_id()) / lanes_per_item;
        const auto ex = make_ex(it);
        const auto av = v2::mat(a + std::ptrdiff_t(b) * ld * n, n, n, ld);
        const auto xv = v2::assume_published(v2::vec(x + std::ptrdiff_t(b) * n * inc, n, inc));
        v2::trmv_upper_row_owner(ex, av, xv, v2::vec(y + std::ptrdiff_t(b) * n, n));
    });
    const auto got = dev.down(y, hy.size());
    EXPECT_EQ(items_differing(got, n), 0) << where;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < n; ++r) {
            LD<T> acc = 0;
            long double s = 0;
            for (int c = r; c < n; ++c) {
                const LD<T> e = wide(ha[std::size_t(b) * ld * n + r + std::size_t(c) * ld]);
                const LD<T> xc = wide(hx[std::size_t(b) * n * inc + c * inc]);
                acc += e * xc, s += std::abs(e) * std::abs(xc);
            }
            worst = std::max(worst, rel(got[std::size_t(b) * n + r], acc, s));
        }
    EXPECT_VERIFY(T, Check::blas, n, worst) << where;
}

TYPED_TEST(DeviceBlasV2LarftTest, TrmvUpperRowOwnerOnEveryExecutor) {
    using T = TypeParam;
    auto& d = this->dev;
    trmv_case<64, T>(d, [](sycl::nd_item<1> it) { return v2::WorkGroup<64>{it}; }, 1, 64, "WorkGroup<64>");
    // n = 23 rows over 16 lanes (rows wrap) and over one item; WG = 1024 is the launch ceiling.
    trmv_case<32, T>(d, [](sycl::nd_item<1> it) { return v2::Partition<16>{it.get_sub_group()}; }, 2, 16, "Partition<16>");
    trmv_case<32, T>(d, [](sycl::nd_item<1>) { return v2::Thread{}; }, 32, 1, "Thread");
    trmv_case<1024, T>(d, [](sycl::nd_item<1> it) { return v2::WorkGroup<1024>{it}; }, 1, 1024, "WorkGroup<1024>");
}

// larft's T recurrence on one executor with NO barrier: column j of T is trmv(T(0:j, 0:j), y_j),
// whose row r is read back by the item that wrote it. n = 45 rows on 32 items, so rows wrap.
TYPED_TEST(DeviceBlasV2LarftTest, TrmvColumnRecurrenceNeedsNoBarrier) {
    using T = TypeParam;
    constexpr int WG = 32, n = 45, ld = n + 2;
    auto hy = batched<T>(ld * n, 41);  // y_j in the strictly lower row j (T(j, 0:j)), as larft stores it
    auto hd = batched<T>(n, 42);       // the diagonal
    T* t = this->dev.up(hy);
    T* dg = this->dev.up(hd);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const auto Tm = v2::mat(t + std::ptrdiff_t(b) * ld * n, n, n, ld);
        v2::copy(ex, v2::vec(dg + std::ptrdiff_t(b) * n, n), v2::vec(Tm.p, n, ld + 1));
        for (int j = 1; j < n; ++j)
            v2::trmv_upper_row_owner(ex, Tm.sub(0, 0, j, j), v2::assume_published(Tm.row(j).head(j)), Tm.col(j).head(j));
    });
    const auto got = this->dev.down(t, hy.size());
    EXPECT_EQ(items_differing(got, ld * n), 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const std::size_t o = std::size_t(b) * ld * n;
        std::vector<LD<T>> R(std::size_t(n) * n, 0);
        std::vector<long double> S(std::size_t(n) * n, 0);
        for (int j = 0; j < n; ++j) {
            R[j + j * n] = wide(hd[std::size_t(b) * n + j]);
            S[j + j * n] = std::abs(R[j + j * n]);
            for (int r = 0; r < j; ++r)
                for (int c = r; c < j; ++c) {
                    const LD<T> yc = wide(hy[o + j + std::size_t(c) * ld]);
                    R[r + j * n] += R[r + c * n] * yc, S[r + j * n] += S[r + c * n] * std::abs(yc);
                }
        }
        for (int j = 0; j < n; ++j)
            for (int r = 0; r <= j; ++r) worst = std::max(worst, rel(got[o + r + std::size_t(j) * ld], R[r + j * n], S[r + j * n]));
    }
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// ---- larf_frag: H applied to a register column ----------------------------------------------

// Each of 13 lanes of a Partition<16> owns column l of C (ld = n + 3) in a Frag<16, ColPerLane>;
// the reflector has head 3, is staged in local memory under a sub-group barrier and is zero
// outside its support: C := H C, H = I - tau v v^H, in the hand loop's (Native) and explicit
// arithmetic.
template <class Ops, class T> void larf_frag_case(Dev& dev, const char* where) {
    constexpr int P = 16, n = 13, ld = n + 3, head = 3, items_per_wg = 2;
    const T tau = val<T>(61);
    auto hc = batched<T>(ld * n, 62, [](T* c) {
        for (int k = 0; k < n; ++k)
            for (int r = n; r < ld; ++r) c[r + k * ld] = poison<T>();  // the ld slack: never read
    });
    auto hv = batched<T>(P, 63, [](T* v) {
        for (int r = 0; r < P; ++r) v[r] = r < head || r >= n ? T(0) : (r == head ? T(1) : v[r]);
    });
    T* c = dev.up(hc);
    T* vg = dev.up(hv);
    dev.template run<32>(kBatch / items_per_wg, sizeof(T) * items_per_wg * P, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const auto sg = it.get_sub_group();
        const v2::Partition<P> part{sg};
        const int lane = part.lid(), pi = static_cast<int>(sg.get_local_linear_id()) / P;
        const int b = static_cast<int>(it.get_group_linear_id()) * items_per_wg + pi;
        T* vl = reinterpret_cast<T*>(lm) + pi * P;
        const auto Cm = v2::mat(c + std::ptrdiff_t(b) * ld * n, n, n, ld);
        vl[lane] = vg[std::ptrdiff_t(b) * P + lane];
        v2::Frag<T, P, v2::ColPerLane> col;
#pragma unroll
        for (int r = 0; r < P; ++r) col[r] = lane < n && r < n ? Cm(r, lane) : T(0);
        sycl::group_barrier(sg);
        const auto vh = v2::assume_published(v2::Vec<T, v2::Ext<P>, v2::Unit, v2::Mem::Local>{vl, {}});
        if (lane < n) v2::larf_frag<Ops>(v2::Thread{}, tau, vh, col);
#pragma unroll
        for (int r = 0; r < P; ++r)
            if (lane < n && r < n) Cm(r, lane) = col[r];
    });
    const auto got = dev.down(c, hc.size());
    EXPECT_EQ(items_differing(got, ld * n), 0) << where;
    double worst = 0;
    int slack_written = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const std::size_t o = std::size_t(b) * ld * n;
        auto C0 = [&](int r, int k) { return wide(hc[o + r + std::size_t(k) * ld]); };
        auto vv = [&](int r) { return wide(hv[std::size_t(b) * P + r]); };
        const LD<T> t = wide(tau);
        for (int k = 0; k < n; ++k) {
            LD<T> w = 0;
            long double sw = 0;
            for (int i = 0; i < n; ++i) w += conj_h(vv(i)) * C0(i, k), sw += std::abs(vv(i)) * std::abs(C0(i, k));
            for (int r = 0; r < n; ++r) {
                const LD<T> ref = C0(r, k) - vv(r) * (t * w);
                const long double s = std::abs(C0(r, k)) + std::abs(vv(r)) * std::abs(t) * sw;
                worst = std::max(worst, rel(got[o + r + std::size_t(k) * ld], ref, s));
            }
            for (int r = n; r < ld; ++r) slack_written += !same_bits(got[o + r + std::size_t(k) * ld], poison<T>());
        }
    }
    EXPECT_EQ(slack_written, 0) << where;
    EXPECT_VERIFY(T, Check::blas, n, worst) << where;
}

TYPED_TEST(DeviceBlasV2LarftTest, LarfFragLeftOnRegisterColumn) {
    using T = TypeParam;
    larf_frag_case<v2::arith::Native, T>(this->dev, "Native");
    larf_frag_case<v2::arith::Explicit, T>(this->dev, "Explicit");
}

// ---- the larft kernel: v2 (use_device = true) and legacy against LAPACK's recurrence ---------

// V is materialised unit-lower (as pack_v writes it) with ld_v = m + 3 and poisoned rows past
// the panel; T has ld_t = ib + 2 and a stride with slack, all NaN on entry; tau has stride ib + 5
// and offset 2, with tau_3 == 0. Items 5 and 1000 are masked: their T stays NaN.
template <bool UseDevice, class T> void larft_case(Dev& dev, int m, int ib) {
    SCOPED_TRACE(::testing::Message() << (UseDevice ? "v2" : "legacy") << " m " << m << " ib " << ib);
    const int ld_v = m + 3, sv = ld_v * ib, ld_t = ib + 2, st = ld_t * ib + 3, ts = ib + 5, toff = 2;
    auto hv = batched<T>(sv, 71, [&](T* v) {
        for (int c = 0; c < ib; ++c)
            for (int r = 0; r < ld_v; ++r) {
                if (r == c) v[r + c * ld_v] = T(1);
                else if (r < c) v[r + c * ld_v] = T(0);
                else if (r >= m) v[r + c * ld_v] = poison<T>();  // past the panel: never read
            }
    });
    auto ht = batched<T>(ts, 72, [&](T* t) { t[toff + 3] = T(0); });
    std::vector<T> hT(std::size_t(kBatch) * st, nan_v<T>());
    std::vector<std::uint8_t> hm(kBatch, 1);
    hm[5] = hm[1000] = 0;
    const T* v = dev.up(hv);
    const T* tau = dev.up(ht);
    T* t = dev.up(hT);
    const std::uint8_t* mask = dev.up(hm);
    (void)wy::larft_forward_columnwise_batched_t<V2LarftTestTag, T, UseDevice>(*dev.ctx, t, ld_t, st, v, ld_v, sv, m, ib,
                                                                                tau, ts, toff, kBatch, mask);
    dev.ctx->wait();
    const auto got = dev.down(t, hT.size());
    EXPECT_EQ(items_differing(got, st, {5, 1000}), 0);
    int bad_masked = 0;
    for (const int b : {5, 1000})
        for (int i = 0; i < st; ++i) bad_masked += !same_bits(got[std::size_t(b) * st + i], nan_v<T>());
    EXPECT_EQ(bad_masked, 0) << "a masked item was written";

    double worst = 0;
    int bad_struct = 0;
    for (int b = 0; b < kPeriod; ++b) {
        auto V = [&](int r, int c) { return wide(hv[std::size_t(b) * sv + r + std::size_t(c) * ld_v]); };
        auto tj = [&](int j) { return wide(ht[std::size_t(b) * ts + toff + j]); };
        std::vector<LD<T>> R(std::size_t(ib) * ib, 0);
        std::vector<long double> S(std::size_t(ib) * ib, 0);
        for (int j = 0; j < ib; ++j) {
            R[j + j * ib] = tj(j), S[j + j * ib] = std::abs(tj(j));
            if (tj(j) == LD<T>(0)) continue;
            std::vector<LD<T>> y(j);
            std::vector<long double> ya(j);
            for (int c = 0; c < j; ++c) {
                LD<T> s = 0;
                long double sa = 0;
                for (int r = j; r < m; ++r) s += conj_h(V(r, c)) * V(r, j), sa += std::abs(V(r, c)) * std::abs(V(r, j));
                y[c] = -tj(j) * s, ya[c] = std::abs(tj(j)) * sa;
            }
            for (int r = 0; r < j; ++r)
                for (int c = r; c < j; ++c) R[r + j * ib] += R[r + c * ib] * y[c], S[r + j * ib] += S[r + c * ib] * ya[c];
        }
        const std::size_t o = std::size_t(b) * st;
        for (int j = 0; j < ib; ++j)
            for (int r = 0; r < ld_t; ++r) {
                const T g = got[o + r + std::size_t(j) * ld_t];
                if (r >= ib) bad_struct += !same_bits(g, nan_v<T>());        // outside the ib x ib block
                else if (r > j) bad_struct += !same_bits(g, T(0));           // T is upper: the scratch is cleared
                else if (r == j) bad_struct += !same_bits(g, ht[std::size_t(b) * ts + toff + j]);
                else worst = std::max(worst, rel(g, R[r + j * ib], S[r + j * ib]));
            }
        for (int i = ld_t * ib; i < st; ++i) bad_struct += !same_bits(got[o + i], nan_v<T>());
    }
    EXPECT_EQ(bad_struct, 0) << "diagonal != tau, lower != 0, or a write outside T";
    EXPECT_VERIFY(T, Check::blas, m + ib, worst);
}

// One shape per WG rung (32, 64, 128, 256), plus m = 64 / 65 at ib = 8: both sides of the first
// rung's m bound.
TYPED_TEST(DeviceBlasV2LarftTest, LarftV2AndLegacyMatchLapackOnEveryRung) {
    using T = TypeParam;
    for (const auto [m, ib] : {std::pair{40, 8}, std::pair{64, 8}, std::pair{65, 8}, std::pair{100, 16}, std::pair{200, 29},
                               std::pair{300, 40}}) {
        larft_case<true, T>(this->dev, m, ib);
        larft_case<false, T>(this->dev, m, ib);
    }
}

}  // namespace

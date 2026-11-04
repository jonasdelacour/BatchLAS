// SPDX-License-Identifier: MIT
// potri-private device LA v2 ops (src/ops/potri/potri_v2.hh) against host references; the shared
// ops they sit on (RtGroup, for_each_owned, find_first, Tile, Frag, load, fill, arith::Fma) are
// covered by device_blas_v2_tests. Guards (docs/developer/agent-guide.md section 8): non-natural
// ld / inc, rectangular tiles and grids (R != C, a work-group stride that is no multiple of rows),
// complex data with nonzero imaginary parts, NaN in every entry an op must not read (the unit
// trsm's diagonal and upper triangle, a gemm operand's masked half, packed slots outside the copied
// columns), sentinels where an op must not write, a launch-sized work-group of 96 (three sub-groups)
// and the 1024 ceiling, batch 1024 of period-3 data with every item bit-identical to its twin, and
// the tile ops bit-compared with the hand loops of the legacy potri_cta (dev_sub / dev_mul /
// fma_acc) besides a long-double reference.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/arith.hh>
#include <src/ops/potri/potri_v2.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
namespace pv = batchlas::sycl_potri;
using batchlas::verify::Check;

namespace {

template <class T>
using real_of = typename v2::Arith<T>::real_t;
template <class T>
constexpr bool is_cplx = v2::Arith<T>::is_complex;
template <class T>
using Dv = typename sycl_device::DevMap<T>::type;

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R>
R uval(std::uint64_t i) {
    return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0);
}
template <class T>
T val(std::uint64_t i) {
    if constexpr (is_cplx<T>)
        return T(uval<real_of<T>>(2 * i), uval<real_of<T>>(2 * i + 1));
    else
        return uval<T>(i);
}

// kPeriod representatives (edited by fix(item, rep)) repeated over the batch: item b is a copy of b % kPeriod.
template <class T, class Fix>
std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per_item; ++i)
            h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item, b);
    }
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
template <class T>
std::vector<T> batched(int per_item, std::uint64_t seed) {
    return batched<T>(per_item, seed, [](T*, int) {});
}

template <class T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template <class T>
int twin_diffs(const std::vector<T>& h, std::size_t per_item) {
    int d = 0;
    for (std::size_t b = kPeriod; b < kBatch; ++b)
        for (std::size_t i = 0; i < per_item; ++i)
            d += !same_bits(h[b * per_item + i], h[(b % kPeriod) * per_item + i]);
    return d;
}

template <class T>
using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T>
LD<T> wide(T v) {
    return LD<T>(v);
}
template <class T>
T mk(real_of<T> re, real_of<T> im) {
    if constexpr (is_cplx<T>)
        return T(re, im);
    else
        return re;
}
template <class T>
T conj_h(T v) {
    if constexpr (is_cplx<T>)
        return std::conj(v);
    else
        return v;
}
template <class T>
double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    template <class T>
    T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_device<T>(h.size(), q());
        q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T>
    void put(T* p, const std::vector<T>& h) {
        q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    }
    template <class T>
    std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    // A launch-time work-group size: the executor under test (RtGroup) never sees a constant.
    template <class F>
    void run(int groups, int wg, F f) {
        q().submit([&](sycl::handler& h) {
               h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * wg, wg),
                              [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
           })
            .wait_and_throw();
    }
    int max_wg() { return static_cast<int>(q().get_device().get_info<sycl::info::device::max_work_group_size>()); }
    ~Dev() {
        for (void* p : ptrs)
            sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T>
class DeviceBlasV2PotriTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2PotriTest, Types);

bool tri_ok(int t) {
    const pv::RowCol rc = pv::tri_index(t);
    return rc.c >= 0 && rc.c <= rc.r && static_cast<long long>(rc.r) * (rc.r + 1) / 2 + rc.c == t;
}

// ---- register tiles on the Thread executor -----------------------------------------------------

// One item per work-item: laset at an off-diagonal origin, the unit lower trsm (diagonal and upper
// triangle NaN), gemm_sub with a masked B operand (masked half NaN), K rank-1 ger steps on conj(x),
// and for_each at a global (i0, j0) into a sentinel-filled matrix. R != C everywhere. Slots 4..6
// hold the legacy hand loops (dev_sub(c, dev_mul(a, b)), fma_acc) the strict port must reproduce.
TYPED_TEST(DeviceBlasV2PotriTest, RegisterTileOps) {
    using T = TypeParam;
    using D = Dv<T>;
    using R_ = real_of<T>;
    constexpr int R = 4, C = 3, K = 3, K2 = 5;
    constexpr int lda = R + 3, ldb = R + 2, ldg = R + 1, ldh = C + 1, ldx = R + 1, ldy = C + 2;
    constexpr int OR = R + 2, OC = C + 1, I0 = 2, J0 = 1;
    constexpr int szA = lda * R, szB = ldb * C, szG = ldg * K, szH = ldh * K, szX = ldx * K2, szY = ldy * K2;
    constexpr int RC = R * C, nout = 7 * RC + OR * OC;
    const T nan = T(std::numeric_limits<R_>::quiet_NaN());
    const T sentinel = T(R_(1e6));
    auto hA = batched<T>(szA, 21, [&](T* p, int) {
        for (int j = 0; j < R; ++j)
            for (int i = 0; i <= j; ++i)
                p[i + j * lda] = nan;
    });
    auto hB = batched<T>(szB, 22), hG = batched<T>(szG, 23), hX = batched<T>(szX, 25), hY = batched<T>(szY, 26);
    auto hH = batched<T>(szH, 24, [&](T* p, int) {
        for (int k = 0; k < K; ++k)
            for (int y = k + 1; y < C; ++y)
                p[y + k * ldh] = nan;
    });
    T* dA = this->dev.up(hA);
    T* dB = this->dev.up(hB);
    T* dG = this->dev.up(hG);
    T* dH = this->dev.up(hH);
    T* dX = this->dev.up(hX);
    T* dY = this->dev.up(hY);
    T* dO = this->dev.up(std::vector<T>(std::size_t(kBatch) * nout, sentinel));
    this->dev.run(kBatch / 64, 64, [=](sycl::nd_item<1> it) {
        using namespace sycl_device;
        const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_global_linear_id());
        const v2::Thread th{};
        const D* a = reinterpret_cast<const D*>(dA) + b * szA;
        const D* bm = reinterpret_cast<const D*>(dB) + b * szB;
        const D* g = reinterpret_cast<const D*>(dG) + b * szG;
        const D* hh = reinterpret_cast<const D*>(dH) + b * szH;
        const D* xs = reinterpret_cast<const D*>(dX) + b * szX;
        const D* ys = reinterpret_cast<const D*>(dY) + b * szY;
        D* o = reinterpret_cast<D*>(dO) + b * nout;
        v2::Tile<D, R, C> t;
        D u[R][C];
        auto load_b = [&] {
            BATCHLAS_UNROLL_FULL
            for (int x = 0; x < R; ++x) {
                BATCHLAS_UNROLL_FULL
                for (int y = 0; y < C; ++y)
                    t(x, y) = u[x][y] = bm[x + y * ldb];
            }
        };
        auto store = [&](int slot, bool hand) {
            BATCHLAS_UNROLL_FULL
            for (int x = 0; x < R; ++x) {
                BATCHLAS_UNROLL_FULL
                for (int y = 0; y < C; ++y)
                    o[slot * RC + x + y * R] = hand ? u[x][y] : t(x, y);
            }
        };
        auto hb = [&](int k, int y) { return y <= k ? hh[y + k * ldh] : D{}; };
        pv::reg::laset(th, v2::Zero{}, v2::One{}, t, 1, 0);
        store(0, false);

        load_b();
        pv::reg::trsm_llnu(th, [&](int i, int j) { return a[i + j * lda]; }, t);
        BATCHLAS_UNROLL_FULL
        for (int x = 1; x < R; ++x)
            BATCHLAS_UNROLL_FULL
            for (int c = 0; c < x; ++c) {
                const D m = a[x + c * lda];
                BATCHLAS_UNROLL_FULL
                for (int y = 0; y < C; ++y) u[x][y] = dev_sub(u[x][y], dev_mul(m, u[c][y]));
            }
        store(1, false);
        store(4, true);

        load_b();
        pv::reg::gemm_sub<K>(th, [&](int x, int k) { return g[x + k * ldg]; }, hb, t);
        BATCHLAS_UNROLL_FULL
        for (int k = 0; k < K; ++k)
            BATCHLAS_UNROLL_FULL
            for (int x = 0; x < R; ++x)
                BATCHLAS_UNROLL_FULL
                for (int y = 0; y < C; ++y) u[x][y] = dev_sub(u[x][y], dev_mul(g[x + k * ldg], hb(k, y)));
        store(2, false);
        store(5, true);

        load_b();
        for (int k = 0; k < K2; ++k) {
            v2::Frag<D, R> xi;
            v2::Frag<D, C> yj;
            v2::load(xi, [&](int x) { return dev_conj(xs[x + k * ldx]); });
            v2::load(yj, [&](int y) { return ys[y + k * ldy]; });
            pv::reg::ger(th, xi, yj, t);
            BATCHLAS_UNROLL_FULL
            for (int x = 0; x < R; ++x)
                BATCHLAS_UNROLL_FULL
                for (int y = 0; y < C; ++y) fma_acc(u[x][y], dev_conj(xs[x + k * ldx]), ys[y + k * ldy]);
        }
        store(3, false);
        store(6, true);
        pv::reg::for_each(th, t, I0, J0, [&](int i, int j, const D& v) { o[7 * RC + i + j * OR] = v; });
    });
    const auto got = this->dev.down(dO, std::size_t(kBatch) * nout);
    EXPECT_EQ(twin_diffs(got, nout), 0) << "items differ from their representative";
    double worst = 0;
    int exact = 0, hand = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* o = got.data() + std::size_t(b) * nout;
        auto at = [&](const std::vector<T>& h, int sz, int i) { return wide(h[std::size_t(b) * sz + i]); };
        for (int x = 0; x < R; ++x)
            for (int y = 0; y < C; ++y)
                exact += !same_bits(o[x + y * R], T(R_(1 + x == y ? 1 : 0)));
        for (int s = 1; s <= 3; ++s)
            for (int e = 0; e < RC; ++e)
                hand += !same_bits(o[s * RC + e], o[(s + 3) * RC + e]);
        LD<T> Y[R][C];
        long double S[R][C];
        for (int x = 0; x < R; ++x)
            for (int y = 0; y < C; ++y) {
                Y[x][y] = at(hB, szB, x + y * ldb);
                S[x][y] = std::abs(Y[x][y]);
                for (int c = 0; c < x; ++c) {
                    Y[x][y] -= at(hA, szA, x + c * lda) * Y[c][y];
                    S[x][y] += std::abs(at(hA, szA, x + c * lda)) * S[c][y];
                }
                worst = std::max(worst, rel(o[RC + x + y * R], Y[x][y], S[x][y]));
                LD<T> gm = at(hB, szB, x + y * ldb), gr = gm;
                long double sg = std::abs(gm), sr = std::abs(gm);
                for (int k = 0; k < K; ++k) {
                    const LD<T> hv = y <= k ? at(hH, szH, y + k * ldh) : LD<T>(0);
                    gm -= at(hG, szG, x + k * ldg) * hv;
                    sg += std::abs(at(hG, szG, x + k * ldg)) * std::abs(hv);
                }
                for (int k = 0; k < K2; ++k) {
                    const LD<T> xv = conj_h(at(hX, szX, x + k * ldx)), yv = at(hY, szY, y + k * ldy);
                    gr += xv * yv;
                    sr += std::abs(xv) * std::abs(yv);
                }
                worst = std::max({worst, rel(o[2 * RC + x + y * R], gm, sg), rel(o[3 * RC + x + y * R], gr, sr)});
            }
        for (int j = 0; j < OC; ++j)
            for (int i = 0; i < OR; ++i) {
                const bool in = i >= I0 && i < I0 + R && j >= J0 && j < J0 + C;
                const T want = in ? o[3 * RC + (i - I0) + (j - J0) * R] : sentinel;
                exact += !same_bits(o[7 * RC + i + j * OR], want);
            }
    }
    EXPECT_EQ(exact, 0) << "laset or for_each placed a value wrong";
    EXPECT_EQ(hand, 0) << "trsm_llnu / gemm_sub / ger differ in bits from the legacy hand loops";
    EXPECT_VERIFY(T, Check::blas, K2, worst);
}

// ---- launch-sized work-group: distribution and LAPACK auxiliaries ---------------------------------

// The 4-deep batched column-major walk over rows != cols with a stride (96, 1024) that is no
// multiple of rows (dead slots must not write), trttp / tpttr of a column range for both uplos
// (packed slots outside the range stay NaN, entries outside the triangle keep their sentinel),
// tri_index over 4096 t per item plus the 2^29 end of its exact range.
TYPED_TEST(DeviceBlasV2PotriTest, LaunchSizedGroupAuxOps) {
    using T = TypeParam;
    using D = Dv<T>;
    using R_ = real_of<T>;
    constexpr int n = 37, cols = 29, lda = n + 3, c0 = 5, c1 = 33;
    constexpr int szM = lda * n, np = n * (n + 1) / 2, kTri = 4096;
    const T nan = T(std::numeric_limits<R_>::quiet_NaN());
    const T sentinel = mk<T>(R_(7), R_(-3));
    auto hA = batched<T>(szM, 32);
    auto hQl = batched<T>(np, 33), hQu = batched<T>(np, 34);
    const std::vector<T> pnan(std::size_t(kBatch) * np, nan), msent(std::size_t(kBatch) * szM, sentinel);
    const std::vector<int> ipoison(kBatch, -7);
    T* dA = this->dev.up(hA);
    T* dQl = this->dev.up(hQl);
    T* dQu = this->dev.up(hQu);
    T* dU = this->dev.up(msent);
    T* dPl = this->dev.up(pnan);
    T* dPu = this->dev.up(pnan);
    T* dWl = this->dev.up(msent);
    T* dWu = this->dev.up(msent);
    int* dBad = this->dev.up(ipoison);
    for (int wg : {96, 1024}) {
        if (wg > this->dev.max_wg()) continue;
        Dev& d = this->dev;
        d.put(dU, msent), d.put(dPl, pnan), d.put(dPu, pnan), d.put(dWl, msent), d.put(dWu, msent);
        d.put(dBad, ipoison);
        this->dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
            using Ar = v2::Arith<D>;
            const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
            const v2::RtGroup<> ex(it);
            auto dm = [&](T* p, std::ptrdiff_t sz) { return reinterpret_cast<D*>(p) + b * sz; };
            D* Ab = dm(dA, szM);
            struct Slot {
                bool live;
                int r, c;
                D v;
            };
            D* ub = dm(dU, szM);
            pv::for_each_owned_2d_batched<4>(
                ex, n, cols, [&](bool live, int r, int c) { return Slot{live, r, c, live ? Ab[r + c * lda] : D{}}; },
                [&](const Slot& e) {
                    if (e.live) ub[e.r + e.c * lda] = Ar::add(e.v, e.v);
                });

            auto m_of = [&](D* p) { return [p](int r, int c) -> D& { return p[r + std::ptrdiff_t(c) * lda]; }; };
            pv::tpcopy(ex, true, true, n, m_of(Ab), dm(dPl, np), c0, c1);
            pv::tpcopy(ex, false, true, n, m_of(Ab), dm(dPu, np), c0, c1);
            pv::tpcopy(ex, true, false, n, m_of(dm(dWl, szM)), dm(dQl, np), c0, c1);
            pv::tpcopy(ex, false, false, n, m_of(dm(dWu, szM)), dm(dQu, np), c0, c1);

            int bad = 0;
            v2::for_each_owned(ex, kTri, [&](int k) { bad += !tri_ok(static_cast<int>(b) * kTri + k); });
            if (ex.leader())
                for (int k = 0; k < 4; ++k)
                    bad += !tri_ok((1 << 29) - 1 - 4 * static_cast<int>(b) - k);
            bad = ex.group_reduce(bad, sycl::plus<int>());
            if (ex.leader()) dBad[b] = bad;
        });
        const auto Bad = this->dev.down(dBad, kBatch);
        const auto gU = this->dev.down(dU, std::size_t(kBatch) * szM);
        const auto gPl = this->dev.down(dPl, std::size_t(kBatch) * np),
                   gPu = this->dev.down(dPu, std::size_t(kBatch) * np);
        const auto gWl = this->dev.down(dWl, std::size_t(kBatch) * szM),
                   gWu = this->dev.down(dWu, std::size_t(kBatch) * szM);
        const std::string where = "wg " + std::to_string(wg);
        int tbad = 0;
        for (int b = 0; b < kBatch; ++b)
            tbad += Bad[b] != 0;
        EXPECT_EQ(tbad, 0) << where << ": tri_index";
        EXPECT_EQ(twin_diffs(gU, szM) + twin_diffs(gPl, np) + twin_diffs(gPu, np) + twin_diffs(gWl, szM) +
                      twin_diffs(gWu, szM),
                  0)
            << where << ": items differ from their representative";
        int walk = 0, pack = 0, unpack = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const auto A = [&](int r, int c) { return hA[std::size_t(b) * szM + r + c * lda]; };
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < lda; ++r)
                    walk += !same_bits(gU[std::size_t(b) * szM + r + c * lda],
                                       r < n && c < cols ? T(A(r, c) + A(r, c)) : sentinel);
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < n; ++r) {
                    const bool cin = c >= c0 && c < c1;
                    // LAPACK AP indices (0-based): lower i + (2n - j - 1) j / 2, upper i + j (j + 1) / 2.
                    if (r >= c) {
                        const T g = gPl[std::size_t(b) * np + r + (2 * n - c - 1) * c / 2];
                        pack += cin ? !same_bits(g, A(r, c)) : !std::isnan(std::real(g));
                    }
                    if (r <= c) {
                        const T g = gPu[std::size_t(b) * np + r + c * (c + 1) / 2];
                        pack += cin ? !same_bits(g, A(r, c)) : !std::isnan(std::real(g));
                    }
                    const T ql = r >= c ? hQl[std::size_t(b) * np + r + (2 * n - c - 1) * c / 2] : sentinel;
                    const T qu = r <= c ? hQu[std::size_t(b) * np + r + c * (c + 1) / 2] : sentinel;
                    unpack += !same_bits(gWl[std::size_t(b) * szM + r + c * lda], cin ? ql : sentinel);
                    unpack += !same_bits(gWu[std::size_t(b) * szM + r + c * lda], cin ? qu : sentinel);
                }
            for (int c = 0; c < n; ++c)
                for (int r = n; r < lda; ++r)
                    unpack += !same_bits(gWl[std::size_t(b) * szM + r + c * lda], sentinel);
        }
        EXPECT_EQ(walk, 0) << where << ": for_each_owned_2d_batched";
        EXPECT_EQ(pack, 0) << where << ": trttp";
        EXPECT_EQ(unpack, 0) << where << ": tpttr";
    }
}

} // namespace

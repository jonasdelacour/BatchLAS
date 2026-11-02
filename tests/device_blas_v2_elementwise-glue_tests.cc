// SPDX-License-Identifier: MIT
// Device LA v2 elementwise-glue ops (include/batchlas/blas/device/v2/ext/elementwise-glue.hh)
// against host references. Guards (docs/developer/agent-guide.md section 8): ld != rows on every
// operand with sentinels in the pad, complex data with nonzero imaginary parts, NaN-poisoned
// storage that must never be read (the other triangle, the unit diagonal, beta == Zero), batch
// 1024 with period-3 data where every item must equal its representative bit for bit (SLM tile
// and work-group reduction paths), the batch walked past a capped dimension-0 grid, shapes that
// are not multiples of the group shape, a scaled Frobenius norm straddling overflow.
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

#include <batchlas/blas/device/v2/ext/elementwise-glue.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// kPeriod representatives (edited by fix(item pointer)) copied to every item b from b % kPeriod.
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

// Items whose bits differ from their representative's.
template <class T> int twin_diffs(const std::vector<T>& g, std::size_t per_item) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (std::size_t i = 0; i < per_item; ++i) d += !same_bits(g[b * per_item + i], g[(b % kPeriod) * per_item + i]);
    return d;
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
    template <class F> void run3(sycl::nd_range<3> r, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(r, [=](sycl::nd_item<3> it) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    template <int WG, class F> void run1(int groups, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GlueTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GlueTest, Types);

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

// ---- Grid2D elementwise ---------------------------------------------------------------------

enum Slot2d { kLacpy, kLaset, kTrimask, kTriExpand, kGeam0, kGeam, kDgmmL, kDgmmR, kLapmtF, kLapmtB, kSlots2d };

// Every op on one launch shape. Shape {4, 4, 8} puts 8 items in a group and splits the matrix
// over several tiles; max_groups0 = 37 makes for_each_batch walk the batch.
TYPED_TEST(DeviceBlasV2GlueTest, Grid2DOpsMatchHostAndNeverReadUnreferencedStorage) {
    using T = TypeParam;
    constexpr int m = 13, n = 9, ld = m + 3, per = ld * n;
    const T al = val<T>(71), be = val<T>(72), sentinel = T(-777);
    const std::array<int, n> kperm = {4, 1, 9, 2, 7, 3, 8, 5, 6};  // 1-based
    auto hA = batched<T>(per, 11);
    // At: the upper triangle and the diagonal poisoned; tri_expand(Lower, Unit) must not read them.
    auto hAt = batched<T>(per, 11, [](T* p) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= std::min(j, m - 1); ++i) p[i + j * ld] = nan_v<T>();
    });
    auto hB = batched<T>(per, 12);
    std::vector<T> hBn(hB.size(), nan_v<T>());
    auto hdl = batched<T>(m, 13), hdr = batched<T>(n, 14);
    std::vector<int> hk(kperm.begin(), kperm.end());
    T* A = this->dev.up(hA);
    T* At = this->dev.up(hAt);
    T* B = this->dev.up(hB);
    T* Bn = this->dev.up(hBn);
    T* dl = this->dev.up(hdl);
    T* dr = this->dev.up(hdr);
    const int* kp = this->dev.up(hk);
    const std::size_t per_out = std::size_t(kSlots2d) * per;
    for (const auto& [shape, g0] : {std::pair{v2::batched2d_shape(m, n), 37}, std::pair{v2::Shape2D{4, 4, 8}, 65535}}) {
        T* o = this->dev.up(std::vector<T>(std::size_t(kBatch) * per_out, sentinel));
        this->dev.run3(v2::batched2d_range(kBatch, m, n, shape, g0), 0, [=](sycl::nd_item<3> it, unsigned char*) {
            const v2::Grid2D<> ex{it};
            ex.for_each_batch(kBatch, [&](int b) {
                const std::ptrdiff_t off = std::ptrdiff_t(b) * per;
                const auto a = v2::mat(A + off, m, n, ld);
                const auto O = [&](int s) { return v2::mat(o + std::ptrdiff_t(b) * per_out + s * per, m, n, ld); };
                v2::lacpy(ex, v2::tri(v2::UpperC{}), a, O(kLacpy));
                v2::laset(ex, v2::tri(Uplo::Lower), al, be, O(kLaset));
                v2::trimask(ex, v2::tri(v2::UpperC{}, -2), a, O(kTrimask));
                v2::tri_expand(ex, v2::LowerC{}, v2::DiagC<Diag::Unit>{}, v2::mat(At + off, m, n, ld), O(kTriExpand));
                v2::geam(ex, al, a, v2::Zero{}, v2::mat(Bn + off, m, n, ld), O(kGeam0));
                v2::geam(ex, al, a, be, v2::mat(B + off, m, n, ld), O(kGeam));
                v2::dgmm<Side::Left>(ex, v2::vec(dl + std::ptrdiff_t(b) * m, m), a, O(kDgmmL));
                v2::dgmm<Side::Right>(ex, v2::vec(dr + std::ptrdiff_t(b) * n, n), a, O(kDgmmR));
                v2::lapmt<true>(ex, v2::vec(kp, n), a, O(kLapmtF));
                v2::lapmt<false>(ex, v2::vec(kp, n), a, O(kLapmtB));
            });
        });
        const auto g = this->dev.down(o, std::size_t(kBatch) * per_out);
        EXPECT_EQ(twin_diffs(g, per_out), 0) << "items differ from their representative";
        int exact_bad = 0;
        double worst = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const T* a = hA.data() + std::size_t(b) * per;
            const T* G = g.data() + std::size_t(b) * per_out;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < ld; ++i) {
                    const auto at = [&](int s) { return G[s * per + i + j * ld]; };
                    if (i >= m) {  // the ld pad is never written
                        for (int s = 0; s < kSlots2d; ++s) exact_bad += !same_bits(at(s), sentinel);
                        continue;
                    }
                    const T aij = a[i + j * ld];
                    exact_bad += !same_bits(at(kLacpy), i <= j ? aij : sentinel);
                    exact_bad += !same_bits(at(kLaset), i >= j ? (i == j ? be : al) : sentinel);
                    exact_bad += !same_bits(at(kTrimask), j - i >= -2 ? aij : T(0));
                    exact_bad += !same_bits(at(kTriExpand), i == j ? T(1) : (i > j ? aij : T(0)));
                    exact_bad += !same_bits(at(kLapmtF), a[i + (kperm[j] - 1) * ld]);
                    exact_bad += !same_bits(G[kLapmtB * per + i + (kperm[j] - 1) * ld], aij);
                    const LD<T> A_ = wide(aij), AL = wide(al), BE = wide(be);
                    const LD<T> B_ = wide(hB[std::size_t(b) * per + i + j * ld]);
                    const LD<T> DL = wide(hdl[std::size_t(b) * m + i]), DR = wide(hdr[std::size_t(b) * n + j]);
                    const long double s = std::abs(AL) * std::abs(A_) + std::abs(BE) * std::abs(B_) + std::abs(A_) * (std::abs(DL) + std::abs(DR));
                    worst = std::max({worst, rel(at(kGeam0), AL * A_, s), rel(at(kGeam), AL * A_ + BE * B_, s),
                                      rel(at(kDgmmL), DL * A_, s), rel(at(kDgmmR), A_ * DR, s)});
                }
            }
        }
        EXPECT_EQ(exact_bad, 0) << "lacpy/laset/trimask/tri_expand/lapmt or an ld pad, shape lb=" << shape.lb;
        EXPECT_VERIFY(T, Check::blas, 2, worst) << "geam/dgmm, shape lb=" << shape.lb;
    }
}

// ---- flat walk and reductions on a WorkGroup ---------------------------------------------------

enum SlotL { kMax, kOne, kInf, kFro, kFroS, kLo, kHi, kSlotsL };

template <class T, int WG> void lange_case(Dev& dev, int m, int n, int ld, real_of<T> scale_in) {
    using R = real_of<T>;
    const int per = ld * n;
    auto hA = batched<T>(per, 31 + m, [&](T* p) {
        for (int k = 0; k < per; ++k) p[k] = p[k] * T(scale_in);
    });
    auto hx = batched<T>(n + 1, 41, [&](T* p) { p[n / 2] = nan_v<T>(); });  // absrange skips the NaN
    T* A = dev.up(hA);
    T* x = dev.up(hx);
    T* o = dev.up(std::vector<T>(std::size_t(kBatch) * kSlotsL, nan_v<T>()));
    T* w = dev.up(std::vector<T>(std::size_t(kBatch) * per, T(-777)));
    dev.template run1<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const auto a = v2::mat(A + std::ptrdiff_t(b) * per, m, n, ld);
        const R r0 = v2::lange<NormType::Max>(ex, a), r1 = v2::lange<NormType::One>(ex, a), r2 = v2::lange<NormType::Inf>(ex, a);
        const R r3 = v2::lange<NormType::Frobenius>(ex, a), r4 = v2::lange<NormType::Frobenius, v2::alg::Scaled>(ex, a);
        const auto rg = v2::absrange(ex, v2::vec(x + std::ptrdiff_t(b) * (n + 1), n + 1));
        v2::map2d(ex, v2::mat(w + std::ptrdiff_t(b) * per, m, n, ld), [](T v) { return v2::Arith<T>::add(v, v); }, a);
        T* ob = o + std::ptrdiff_t(b) * kSlotsL;
        if (ex.leader()) ob[kMax] = T(r0), ob[kOne] = T(r1), ob[kInf] = T(r2), ob[kFro] = T(r3), ob[kFroS] = T(r4), ob[kLo] = T(rg.lo), ob[kHi] = T(rg.hi);
    });
    const auto g = dev.down(o, std::size_t(kBatch) * kSlotsL);
    const auto gw = dev.down(w, std::size_t(kBatch) * per);
    EXPECT_EQ(twin_diffs(g, kSlotsL), 0) << "reductions differ from their representative, WG " << WG;
    int walk_bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* a = hA.data() + std::size_t(b) * per;
        long double mx = 0, one = 0, inf = 0, ss = 0;
        std::vector<long double> rows(m, 0);
        for (int j = 0; j < n; ++j) {
            long double cs = 0;
            for (int i = 0; i < m; ++i) {
                const long double v = std::abs(wide(a[i + j * ld]));
                mx = std::max(mx, v), cs += v, rows[i] += v, ss += v * v;
            }
            one = std::max(one, cs);
        }
        for (int i = 0; i < m; ++i) inf = std::max(inf, rows[i]);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ld; ++i) {
                const T got = gw[std::size_t(b) * per + i + j * ld];
                walk_bad += !same_bits(got, i < m ? T(a[i + j * ld] + a[i + j * ld]) : T(-777));  // every element once, pad untouched
            }
        const T* G = g.data() + std::size_t(b) * kSlotsL;
        const R fro = static_cast<R>(std::sqrt(ss));
        EXPECT_VERIFY(T, Check::blas, 2, rel(G[kMax], LD<T>(mx), mx)) << "Max, WG " << WG << " m " << m;
        EXPECT_VERIFY(T, Check::blas, m, rel(G[kOne], LD<T>(one), one)) << "One";
        EXPECT_VERIFY(T, Check::blas, n, rel(G[kInf], LD<T>(inf), inf)) << "Inf";
        EXPECT_VERIFY(T, Check::blas, m * n, rel(G[kFroS], LD<T>(std::sqrt(ss)), std::sqrt(ss))) << "Fro scaled";
        if (std::isfinite(static_cast<double>(ss)) && ss < static_cast<long double>(std::numeric_limits<R>::max()))
            EXPECT_VERIFY(T, Check::blas, m * n, rel(G[kFro], LD<T>(std::sqrt(ss)), std::sqrt(ss))) << "Fro unscaled";
        else
            EXPECT_TRUE(std::isinf(std::abs(G[kFro])) && std::isfinite(fro)) << "the straddle must overflow the unscaled form only";
        long double lo = std::numeric_limits<long double>::infinity(), hi = 0;
        for (int i = 0; i <= n; ++i) {
            const long double v = std::abs(wide(hx[std::size_t(b) * (n + 1) + i]));
            if (!std::isnan(v)) lo = std::min(lo, v), hi = std::max(hi, v);
        }
        EXPECT_VERIFY(T, Check::blas, 2, rel(G[kLo], LD<T>(lo), lo)) << "absrange lo (NaN skipped)";
        EXPECT_VERIFY(T, Check::blas, 2, rel(G[kHi], LD<T>(hi), hi)) << "absrange hi (NaN skipped)";
    }
    EXPECT_EQ(walk_bad, 0) << "map2d flat walk, WG " << WG << " m " << m;
}

// m = 13 on 64 items: a carry each step; m = 100: fewer than one column per step; m = 1: one row.
TYPED_TEST(DeviceBlasV2GlueTest, LangeAndFlatWalkOnWorkGroup) {
    using T = TypeParam;
    lange_case<T, 64>(this->dev, 13, 9, 16, 1);
    lange_case<T, 64>(this->dev, 100, 3, 103, 1);
    lange_case<T, 256>(this->dev, 1, 300, 2, 1);
    lange_case<T, 1024>(this->dev, 37, 41, 40, 1);  // the work-group ceiling
    // Each |a|^2 ~ max/4: the sum overflows, the scaled form must not.
    lange_case<T, 256>(this->dev, 17, 11, 19, std::sqrt(std::numeric_limits<real_of<T>>::max()) / 2);
}

// ---- SLM tiles: transpose and mirror ------------------------------------------------------

TYPED_TEST(DeviceBlasV2GlueTest, TileTransposeAndMirrorMatchHost) {
    using T = TypeParam;
    {
        constexpr int TL = 16, rows = 37, cols = 21, lda = rows + 3, ldb = cols + 3;
        const int pa = lda * cols, pb = ldb * rows;
        auto hA = batched<T>(pa, 51);
        T* A = this->dev.up(hA);
        T* Bo = this->dev.up(std::vector<T>(std::size_t(kBatch) * pb, T(-777)));
        const auto up = [](int x) { return std::size_t((x + TL - 1) / TL) * TL; };
        this->dev.run3(sycl::nd_range<3>({std::size_t(kBatch), up(cols), up(rows)}, {1, TL, TL}), sizeof(T) * TL * (TL + 1),
                       [=](sycl::nd_item<3> it, unsigned char* lm) {
                           const v2::Grid2D<TL, TL, 1> ex{it};
                           const int b = ex.batch();
                           const auto t = v2::mat<v2::Mem::Local>(reinterpret_cast<T*>(lm), TL, TL, TL + 1);
                           v2::tile_load_trans<TL>(ex, v2::mat(A + std::ptrdiff_t(b) * pa, rows, cols, lda), ex.row0(), ex.col0(), t);
                           ex.barrier();
                           v2::tile_store<TL>(ex, t, v2::mat(Bo + std::ptrdiff_t(b) * pb, cols, rows, ldb), ex.col0(), ex.row0());
                       });
        const auto g = this->dev.down(Bo, std::size_t(kBatch) * pb);
        EXPECT_EQ(twin_diffs(g, pb), 0);
        int bad = 0;
        for (int b = 0; b < kPeriod; ++b)
            for (int j = 0; j < rows; ++j)
                for (int i = 0; i < ldb; ++i)
                    bad += !same_bits(g[std::size_t(b) * pb + i + j * ldb], i < cols ? hA[std::size_t(b) * pa + j + i * lda] : T(-777));
        EXPECT_EQ(bad, 0) << "transpose";
    }
    // Mirror from each triangle (the unreferenced one NaN-poisoned): runtime Lower + conj, static
    // Upper without; 32 x 32 tiles, 8 columns per group, so each item stores 4 columns.
    constexpr int TM = 32, GC = 8, n = 45, lda = n + 2, ldo = n + 4, tiles = (n + TM - 1) / TM;
    const int pa = lda * n, po = ldo * n;
    for (const bool lower : {true, false}) {
        auto hA = batched<T>(pa, lower ? 61 : 62, [&](T* p) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (lower ? i < j : i > j) p[i + j * lda] = nan_v<T>();
        });
        T* A = this->dev.up(hA);
        T* O = this->dev.up(std::vector<T>(std::size_t(kBatch) * po, T(-777)));
        this->dev.run3(sycl::nd_range<3>({std::size_t(kBatch), std::size_t(tiles) * GC, std::size_t(tiles) * TM}, {1, GC, TM}),
                       sizeof(T) * TM * (TM + 1), [=](sycl::nd_item<3> it, unsigned char* lm) {
                           const v2::Grid2D<TM, GC, 1> ex{it};
                           const int ti = static_cast<int>(it.get_group(1)), tj = static_cast<int>(it.get_group(2));
                           if (ti < tj) return;
                           const int b = ex.batch();
                           const int r0 = (lower ? ti : tj) * TM, c0 = (lower ? tj : ti) * TM;
                           const auto a = v2::mat(A + std::ptrdiff_t(b) * pa, n, n, lda);
                           const auto o = v2::mat(O + std::ptrdiff_t(b) * po, n, n, ldo);
                           const auto t = v2::mat<v2::Mem::Local>(reinterpret_cast<T*>(lm), TM, TM, TM + 1);
                           v2::tile_load<TM>(ex, a, r0, c0, t);
                           sycl::group_barrier(it.get_group());
                           if (lower) {
                               if (ti == tj) v2::tile_store_diag<true, TM>(ex, Uplo::Lower, t, o, r0);
                               else v2::tile_store_pair<true, TM>(ex, t, o, r0, c0);
                           } else {
                               if (ti == tj) v2::tile_store_diag<false, TM>(ex, v2::UpperC{}, t, o, r0);
                               else v2::tile_store_pair<false, TM>(ex, t, o, r0, c0);
                           }
                       });
        const auto g = this->dev.down(O, std::size_t(kBatch) * po);
        EXPECT_EQ(twin_diffs(g, po), 0);
        int bad = 0;
        for (int b = 0; b < kPeriod; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < ldo; ++i) {
                    const T got = g[std::size_t(b) * po + i + j * ldo];
                    if (i >= n) {
                        bad += !same_bits(got, T(-777));
                        continue;
                    }
                    const bool ref_side = lower ? i >= j : i <= j;
                    T want = ref_side ? hA[std::size_t(b) * pa + i + j * lda] : hA[std::size_t(b) * pa + j + i * lda];
                    if (lower && !ref_side) want = conj_h(want);
                    if (lower && i == j) want = T(std::real(want));  // Hermitian: real diagonal
                    bad += !same_bits(got, want);
                }
        EXPECT_EQ(bad, 0) << (lower ? "hermitian mirror from Lower" : "symmetric mirror from Upper");
    }
}

// In-place mirror on a Grid2D: the other strict triangle from the referenced one.
TYPED_TEST(DeviceBlasV2GlueTest, InPlaceMirrorReadsOnlyTheReferencedTriangle) {
    using T = TypeParam;
    constexpr int n = 23, ld = n + 5, per = ld * n;
    auto hA = batched<T>(per, 81, [](T* p) {
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < n; ++i) p[i + j * ld] = nan_v<T>();  // Upper stored: strict lower poisoned
    });
    T* A = this->dev.up(hA);
    this->dev.run3(v2::batched2d_range(kBatch, n, n, v2::batched2d_shape(n, n)), 0, [=](sycl::nd_item<3> it, unsigned char*) {
        const v2::Grid2D<> ex{it};
        ex.for_each_batch(kBatch, [&](int b) { v2::mirror<true>(ex, v2::UpperC{}, v2::mat(A + std::ptrdiff_t(b) * per, n, n, ld)); });
    });
    const auto g = this->dev.down(A, std::size_t(kBatch) * per);
    EXPECT_EQ(twin_diffs(g, per), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T want = i <= j ? hA[std::size_t(b) * per + i + j * ld] : conj_h(hA[std::size_t(b) * per + j + i * ld]);
                if (i == j) want = T(std::real(want));
                bad += !same_bits(g[std::size_t(b) * per + i + j * ld], want);
            }
    EXPECT_EQ(bad, 0);
}

// ---- RowFrag moves, bcast, argmax_pair (8-lane chunks of a 32-lane sub-group) -------------------

TYPED_TEST(DeviceBlasV2GlueTest, RowFragMovesBcastAndArgmax) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int N = 8, nl = 6, ld = N + 3, per = ld * N, kd = 4 * N * N + 3 * N;  // dumps per item
    auto hA = batched<T>(per, 91, [](T* p) {
        p[2] = p[5] = T(2);  // the maximum of |A(:, 0)| (abs1) tied on rows 2 and 5: the lower key wins
    });
    T* A = this->dev.up(hA);
    T* D = this->dev.up(std::vector<T>(std::size_t(kBatch) * kd, T(-777)));
    T* S = this->dev.up(std::vector<T>(std::size_t(kBatch) * 2 * per, T(-777)));
    this->dev.template run1<32>(kBatch / 4, [=](sycl::nd_item<1> it) {
        const int lid = static_cast<int>(it.get_local_linear_id());
        const int lane = lid & (N - 1), b = static_cast<int>(it.get_group_linear_id()) * 4 + lid / N;
        const bool live = lane < nl;
        const T* ab = A + std::ptrdiff_t(b) * per;
        T* db = D + std::ptrdiff_t(b) * kd;
        T* sb = S + std::ptrdiff_t(b) * 2 * per;
        T a[N];
        v2::rowfrag_load_tri<Uplo::Lower, N>(a, ab, ld, lane, live, true);
        v2::rowfrag_store<N>(a, db, N, lane, N, true);  // dump 0: the register row, identity padded
        v2::rowfrag_store_tri<Uplo::Lower, N>(a, sb, ld, lane, live);
        v2::rowfrag_load_tri<Uplo::Upper, N>(a, ab, ld, lane, live, false);
        v2::rowfrag_store<N>(a, db + N * N, N, lane, N, true);  // dump 1
        v2::rowfrag_store_tri<Uplo::Upper, N>(a, sb + per, ld, lane, live);
        v2::rowfrag_load<N>(a, ab, ld, lane, nl, live);
        v2::rowfrag_store<N>(a, db + 2 * N * N, N, lane, N, true);  // dump 2
        for (int c = 0; c < N; ++c) db[3 * N * N + lane + c * N] = v2::load_or_identity(ab, lane, c, nl, ld, live);  // dump 3
        const auto sg = it.get_sub_group();
        db[4 * N * N + lane] = v2::bcast(sg, ab[lane + 3 * ld], static_cast<std::uint32_t>((lid & ~(N - 1)) + 3));
        R mag = v2::abs1(ab[lane]);
        int key = lane;
        v2::argmax_pair<N>(sg, mag, key);
        db[4 * N * N + N + lane] = T(R(key));
        db[4 * N * N + 2 * N + lane] = T(mag);
    });
    const auto gd = this->dev.down(D, std::size_t(kBatch) * kd);
    const auto gs = this->dev.down(S, std::size_t(kBatch) * 2 * per);
    EXPECT_EQ(twin_diffs(gd, kd), 0);
    EXPECT_EQ(twin_diffs(gs, 2 * std::size_t(per)), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* a = hA.data() + std::size_t(b) * per;
        const T* d = gd.data() + std::size_t(b) * kd;
        const T* s = gs.data() + std::size_t(b) * 2 * per;
        const auto at = [&](int i, int j) { return a[i + j * ld]; };
        const auto ident = [](int r, int c) { return r == c ? T(1) : T(0); };
        int best = 0;
        for (int r = 0; r < N; ++r) {
            const auto key1 = [&](int i) { return std::abs(std::real(at(i, 0))) + std::abs(std::imag(at(i, 0))); };
            if (key1(r) > key1(best)) best = r;
            for (int c = 0; c < N; ++c) {
                const bool in = r < nl && c <= r;
                const T lo = r == c ? T(std::real(at(r, r))) : at(r, c);
                bad += !same_bits(d[r + c * N], in ? lo : ident(r, c));
                bad += !same_bits(d[N * N + r + c * N], in ? conj_h(at(c, r)) : ident(r, c));
                bad += !same_bits(d[2 * N * N + r + c * N], r < nl && c < nl ? at(r, c) : ident(r, c));
                bad += !same_bits(d[3 * N * N + r + c * N], r < nl && c < nl ? at(r, c) : ident(r, c));
                bad += !same_bits(s[r + c * ld], in ? lo : T(-777));
                bad += !same_bits(s[per + c + r * ld], in ? at(c, r) : T(-777));  // conj twice
            }
        }
        for (int l = 0; l < N; ++l) {
            bad += !same_bits(d[4 * N * N + l], at(3, 3));
            bad += !same_bits(d[4 * N * N + N + l], T(R(best)));
        }
    }
    EXPECT_EQ(bad, 0);
}

// ---- scalar LAPACK ------------------------------------------------------------------------

TYPED_TEST(DeviceBlasV2GlueTest, ScalarLartgLarfgAnd2x2Kernels) {
    using T = TypeParam;
    using R = real_of<T>;
    const R big = std::sqrt(std::numeric_limits<R>::max()) * R(4), tiny = std::sqrt(std::numeric_limits<R>::min()) / R(4);
    // (f, g) pairs: in range, g == 0, f == 0, both beyond rtmax, both below rtmin, mixed.
    std::vector<T> hf = {val<T>(1), val<T>(2), T(0), val<T>(3) * T(big), val<T>(4) * T(tiny), val<T>(5) * T(big)};
    std::vector<T> hg = {val<T>(6), T(0), val<T>(7), val<T>(8) * T(big), val<T>(9) * T(tiny), val<T>(10) * T(tiny)};
    const int nl = static_cast<int>(hf.size());
    constexpr int xl = 3;
    std::vector<T> hx = batched<T>(xl, 101);
    hx.resize(xl);
    long double xn2 = 0;
    for (const T& v : hx) xn2 += std::norm(wide(v));
    const R xnorm = static_cast<R>(std::sqrt(xn2));
    const T alpha = val<T>(102);
    T* f = this->dev.up(hf);
    T* g = this->dev.up(hg);
    T* o = this->dev.up(std::vector<T>(std::size_t(3 * nl + 6 + 8), nan_v<T>()));
    this->dev.template run1<32>(1, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        for (int k = 0; k < nl; ++k) {
            const auto r = v2::lartg(f[k], g[k]);
            o[3 * k] = T(r.c), o[3 * k + 1] = r.s, o[3 * k + 2] = r.r;
        }
        const auto p = v2::larfg_scalars<v2::conv::Phase>(alpha, xnorm, xl + 1);
        const auto q = v2::larfg_scalars<v2::conv::RealBeta>(alpha, xnorm, xl + 1);
        T* s = o + 3 * nl;
        s[0] = p.tau, s[1] = p.beta, s[2] = p.scale, s[3] = q.tau, s[4] = q.beta, s[5] = q.scale;
        if constexpr (!is_cplx<T>) {
            const R a = f[0], bb = g[0], c = f[1];
            const auto e = v2::lae2(a, bb, c);
            const auto ev = v2::laev2(a, bb, c);
            const auto sv = v2::las2(a, bb, c);
            s[6] = e[0], s[7] = e[1], s[8] = ev[0], s[9] = ev[1], s[10] = ev[2], s[11] = ev[3], s[12] = sv[0], s[13] = sv[1];
        }
    });
    const auto h = this->dev.down(o, std::size_t(3 * nl + 14));
    for (int k = 0; k < nl; ++k) {
        const LD<T> F = wide(hf[k]), G = wide(hg[k]), S = wide(h[3 * k + 1]), Rr = wide(h[3 * k + 2]);
        const long double C = std::real(wide(h[3 * k]));
        const long double sc = std::abs(F) + std::abs(G);
        EXPECT_VERIFY(T, Check::blas, 4, rel(h[3 * k + 2], C * F + S * G, sc)) << "lartg r, case " << k;
        EXPECT_VERIFY(T, Check::blas, 4, rel(T(0), -conj_h(S) * F + C * G, sc)) << "lartg zero, case " << k;
        EXPECT_VERIFY(T, Check::blas, 4, double(std::abs(C * C + std::norm(S) - 1.0L))) << "lartg unitary, case " << k;
        if constexpr (!is_cplx<T>) {
            if (hf[k] != T(0)) EXPECT_EQ(std::signbit(std::real(h[3 * k + 2])), std::signbit(std::real(hf[k]))) << "r takes f's sign";
        }
        (void)Rr;
    }
    // H^H [alpha; x] = [beta; 0] with v = [1; scale x], H = I - tau v v^H.
    for (int c = 0; c < 2; ++c) {
        const T* s = h.data() + 3 * nl + 3 * c;
        std::vector<LD<T>> y = {wide(alpha)}, v = {LD<T>(1)};
        for (const T& xi : hx) y.push_back(wide(xi)), v.push_back(wide(s[2]) * wide(xi));
        LD<T> vy = 0;
        for (std::size_t i = 0; i < y.size(); ++i) vy += conj_h(v[i]) * y[i];
        const long double nrm = std::sqrt(std::norm(wide(alpha)) + xn2);
        for (std::size_t i = 0; i < y.size(); ++i) {
            const LD<T> hy = y[i] - conj_h(wide(s[0])) * v[i] * vy;
            EXPECT_VERIFY(T, Check::blas, xl + 1, rel(i == 0 ? s[1] : T(0), hy, nrm)) << (c ? "RealBeta" : "Phase") << " row " << i;
        }
        if (c == 1) EXPECT_EQ(std::imag(LD<T>(s[1])), 0.0L) << "RealBeta: beta is real";
    }
    if constexpr (!is_cplx<T>) {
        const long double a = hf[0], bb = hg[0], c = hf[1];
        const long double d = std::sqrt((a - c) * (a - c) + 4 * bb * bb);
        const long double l1 = (a + c + d) / 2, l2 = (a + c - d) / 2;
        const long double big1 = std::abs(l1) >= std::abs(l2) ? l1 : l2, small1 = std::abs(l1) >= std::abs(l2) ? l2 : l1;
        const R* s = reinterpret_cast<const R*>(h.data() + 3 * nl + 6);
        const long double sc = std::abs(a) + std::abs(bb) + std::abs(c);
        EXPECT_VERIFY(T, Check::blas, 4, rel(s[0], big1, sc)) << "lae2 rt1";
        EXPECT_VERIFY(T, Check::blas, 4, rel(s[1], small1, sc)) << "lae2 rt2";
        EXPECT_VERIFY(T, Check::blas, 4, rel(s[2], big1, sc)) << "laev2 rt1";
        EXPECT_VERIFY(T, Check::blas, 4, rel(s[3], small1, sc)) << "laev2 rt2";
        const long double cs = s[4], sn = s[5];
        EXPECT_VERIFY(T, Check::blas, 4, rel(R(0), (a - big1) * cs + bb * sn, sc)) << "laev2 eigenvector";
        // las2 of [[f, g], [0, h]] with (f, g, h) = (a, bb, c): smin smax = |f h|, smin^2 + smax^2 = f^2 + g^2 + h^2.
        EXPECT_VERIFY(T, Check::blas, 4, rel(R(s[6] * s[7]), std::abs(a * c), a * a + bb * bb + c * c)) << "las2 product";
        EXPECT_VERIFY(T, Check::blas, 4, rel(R(s[6] * s[6] + s[7] * s[7]), a * a + bb * bb + c * c, a * a + bb * bb + c * c)) << "las2 sum";
    }
}

// larfg on a WorkGroup, x with inc 2, both conventions; batch 1024 bit-identical.
TYPED_TEST(DeviceBlasV2GlueTest, LarfgOnWorkGroupAnnihilatesX) {
    using T = TypeParam;
    constexpr int WG = 64, n = 50, inc = 2, per = n * inc + 1;  // [alpha, x with inc]
    auto h0 = batched<T>(per, 111);
    for (int c = 0; c < 2; ++c) {
        T* X = this->dev.up(h0);
        T* tau = this->dev.up(std::vector<T>(kBatch, nan_v<T>()));
        this->dev.template run1<WG>(kBatch, [=](sycl::nd_item<1> it) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const v2::WorkGroup<WG> ex{it};
            T* p = X + std::ptrdiff_t(b) * per;
            T alpha = p[0];
            const auto x = v2::vec(p + 1, n, inc);
            const T t = c ? v2::larfg<v2::conv::RealBeta>(ex, alpha, x) : v2::larfg<v2::conv::Phase>(ex, alpha, x);
            if (ex.leader()) p[0] = alpha, tau[b] = t;
        });
        const auto g = this->dev.down(X, std::size_t(kBatch) * per);
        const auto gt = this->dev.down(tau, kBatch);
        EXPECT_EQ(twin_diffs(g, per), 0);
        EXPECT_EQ(twin_diffs(gt, 1), 0);
        for (int b = 0; b < kPeriod; ++b) {
            const T* y0 = h0.data() + std::size_t(b) * per;
            const T* y1 = g.data() + std::size_t(b) * per;
            LD<T> vy = wide(y0[0]);
            long double nrm2 = std::norm(wide(y0[0]));
            for (int i = 0; i < n; ++i) vy += conj_h(wide(y1[1 + i * inc])) * wide(y0[1 + i * inc]), nrm2 += std::norm(wide(y0[1 + i * inc]));
            const long double nrm = std::sqrt(nrm2);
            const LD<T> t = conj_h(wide(gt[b]));
            EXPECT_VERIFY(T, Check::blas, n + 1, rel(y1[0], wide(y0[0]) - t * vy, nrm)) << "beta, conv " << c;
            double worst = 0;
            for (int i = 0; i < n; ++i) worst = std::max(worst, rel(T(0), wide(y0[1 + i * inc]) - t * wide(y1[1 + i * inc]) * vy, nrm));
            EXPECT_VERIFY(T, Check::blas, n + 1, worst) << "annihilated x, conv " << c;
            int gap_bad = 0;
            for (int i = 0; i + 1 < n; ++i) gap_bad += !same_bits(y1[2 + i * inc], y0[2 + i * inc]);
            EXPECT_EQ(gap_bad, 0) << "the inc gap is untouched";
        }
    }
}

}  // namespace

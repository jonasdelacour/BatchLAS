// SPDX-License-Identifier: MIT
// The elementwise-glue family's private Device LA ops (src/kernels/common/elementwise_glue_v2.hh:
// map2d, trimask, tri_expand, geam, the SLM tile moves) and the core lange norm.cc calls, against
// host references. The core ops they build on (Grid2D, for_each_owned_2d, mat_wide, item) are covered by
// device_blas_v2_tests. Guards (docs/developer/agent-guide.md section 8): ld != rows on every
// operand with sentinels in the pad, rectangular shapes, complex data with nonzero imaginary
// parts, unreferenced storage poisoned with a large finite value the ops would accept (the other
// triangle, a unit diagonal) and B NaN-poisoned for beta == Zero, batch 1024 with period-3 data
// where every item must equal its representative bit for bit (SLM tiles, work-group
// reductions), the batch walked past a capped dimension-0 grid, both runtime and static Uplo,
// a WG = 1024 launch, a Frobenius norm straddling overflow. Deliberate breaks that turn each
// test red: tile_cols stride GC -> GC + 1 (tiles), in-region `>=` -> `>` (trimask), the unit
// test of tri_expand dropped (tri_expand), Zero geam reading B (geam), Inf summing columns (lange).
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

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/elementwise_glue_v2.hh>
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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
// In range and finite, so a read of it is not filtered out anywhere: only the bits tell.
template <class T> T poison() {
    if constexpr (is_cplx<T>) return T(real_of<T>(4097.5), real_of<T>(-8191.25));
    else return T(4097.5);
}

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
    // A 3-D launch with a typed local tile of local_elems E (as the kernels' local_accessor<T>).
    template <class E, class F> void run3(sycl::nd_range<3> r, std::size_t local_elems, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<E, 1> lm(sycl::range<1>(std::max<std::size_t>(local_elems, 1)), h);
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
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
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
// |got - ref| / scale, the blas relative error that EXPECT_VERIFY bounds.
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// ---- Grid2D elementwise ---------------------------------------------------------------------

enum Slot2d { kMap, kMaskU, kMaskL, kExpandLU, kExpandUN, kGeam0, kGeam, kSlots2d };

// Every op on two launch shapes: the kernels' own, and {4, 4, 8} (8 batch items per group, the
// matrix split over several tiles) with max_groups0 = 37 so for_each_batch walks the batch.
TYPED_TEST(DeviceBlasV2GlueTest, Grid2DOpsMatchHostAndNeverReadUnreferencedStorage) {
    using T = TypeParam;
    constexpr int m = 13, n = 9, ld = m + 3, per = ld * n;
    const T al = val<T>(71), be = val<T>(72), sentinel = T(-777);
    auto hA = batched<T>(per, 11);
    // Lower + Unit: the upper triangle and the diagonal are poisoned. Upper + NonUnit: the strict lower.
    auto hAl = batched<T>(per, 15, [](T* p) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= std::min(j, m - 1); ++i) p[i + j * ld] = poison<T>();
    });
    auto hAu = batched<T>(per, 16, [](T* p) {
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < m; ++i) p[i + j * ld] = poison<T>();
    });
    auto hB = batched<T>(per, 12);
    const std::vector<T> hBn(hB.size(), nan_v<T>());
    T* A = this->dev.up(hA);
    T* Al = this->dev.up(hAl);
    T* Au = this->dev.up(hAu);
    T* B = this->dev.up(hB);
    T* Bn = this->dev.up(hBn);
    const std::size_t per_out = std::size_t(kSlots2d) * per;
    for (const auto& [shape, g0] : {std::pair{device::batched2d_shape(m, n), 65535}, std::pair{device::Shape2D{4, 4, 8}, 37}}) {
        T* o = this->dev.up(std::vector<T>(std::size_t(kBatch) * per_out, sentinel));
        this->dev.template run3<T>(device::batched2d_range(kBatch, m, n, shape, g0), 0, [=](sycl::nd_item<3> it, T*) {
            const device::Grid2D<> ex{it};
            ex.for_each_batch(kBatch, [&](int b) {
                const std::ptrdiff_t off = std::ptrdiff_t(b) * per;
                const auto a = device::mat(A + off, m, n, ld);
                const auto O = [&](int s) { return device::mat(o + std::ptrdiff_t(b) * per_out + s * per, m, n, ld); };
                device::glue::map2d(ex, O(kMap), [](T x, T y) { return device::Arith<T>::add(x, y); }, a, device::mat(B + off, m, n, ld));
                device::glue::trimask(ex, device::glue::tri(device::glue::UpperC{}, -2), a, O(kMaskU));
                device::glue::trimask(ex, device::glue::tri(Uplo::Lower, 1), a, O(kMaskL));
                device::glue::tri_expand(ex, device::glue::LowerC{}, device::glue::DiagC<Diag::Unit>{}, device::mat_wide(Al + off, m, n, ld), O(kExpandLU));
                device::glue::tri_expand(ex, Uplo::Upper, Diag::NonUnit, device::mat_wide(Au + off, m, n, ld), O(kExpandUN));
                device::glue::geam(ex, al, a, device::Zero{}, device::mat(Bn + off, m, n, ld), O(kGeam0));
                device::glue::geam(ex, al, a, be, device::mat(B + off, m, n, ld), O(kGeam));
            });
        });
        const auto g = this->dev.down(o, std::size_t(kBatch) * per_out);
        EXPECT_EQ(twin_diffs(g, per_out), 0) << "items differ from their representative";
        int exact_bad = 0;
        double worst = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const std::size_t ib = std::size_t(b) * per;
            const T* G = g.data() + std::size_t(b) * per_out;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < ld; ++i) {
                    const auto at = [&](int s) { return G[s * per + i + j * ld]; };
                    if (i >= m) {  // the ld pad is never written
                        for (int s = 0; s < kSlots2d; ++s) exact_bad += !same_bits(at(s), sentinel);
                        continue;
                    }
                    const std::size_t e = ib + i + j * ld;
                    const T aij = hA[e];
                    exact_bad += !same_bits(at(kMap), T(aij + hB[e]));
                    exact_bad += !same_bits(at(kMaskU), j - i >= -2 ? aij : T(0));
                    exact_bad += !same_bits(at(kMaskL), j - i <= 1 ? aij : T(0));
                    exact_bad += !same_bits(at(kExpandLU), i == j ? T(1) : (i > j ? hAl[e] : T(0)));
                    exact_bad += !same_bits(at(kExpandUN), i <= j ? hAu[e] : T(0));
                    const LD<T> A_ = wide(aij), AL = wide(al), BE = wide(be), B_ = wide(hB[e]);
                    const long double s = std::abs(AL) * std::abs(A_) + std::abs(BE) * std::abs(B_);
                    worst = std::max({worst, rel(at(kGeam0), AL * A_, s), rel(at(kGeam), AL * A_ + BE * B_, s)});
                }
            }
        }
        EXPECT_EQ(exact_bad, 0) << "map2d/trimask/tri_expand or an ld pad, shape lb=" << shape.lb;
        EXPECT_VERIFY(T, Check::blas, 2, worst) << "geam, shape lb=" << shape.lb;
    }
}

// ---- lange and the flat walk on a WorkGroup ---------------------------------------------------

enum SlotL { kMax, kOne, kInf, kFro, kFroS, kSlotsL };

template <class T, int WG> void lange_case(Dev& dev, int m, int n, int ld, real_of<T> scale_in) {
    using R = real_of<T>;
    const int per = ld * n;
    auto hA = batched<T>(per, 31 + m, [&](T* p) {
        for (int k = 0; k < per; ++k) p[k] = p[k] * T(scale_in);
    });
    T* A = dev.up(hA);
    T* o = dev.up(std::vector<T>(std::size_t(kBatch) * kSlotsL, nan_v<T>()));
    T* w = dev.up(std::vector<T>(std::size_t(kBatch) * per, T(-777)));
    dev.template run1<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<WG> ex{it};
        const auto a = device::mat(A + std::ptrdiff_t(b) * per, m, n, ld);
        const R r0 = device::lange<NormType::Max>(ex, a), r1 = device::lange<NormType::One>(ex, a), r2 = device::lange<NormType::Inf>(ex, a);
        const R r3 = device::lange<NormType::Frobenius>(ex, a), r4 = device::lange<NormType::Frobenius, device::alg::Scaled>(ex, a);
        device::glue::map2d(ex, device::mat(w + std::ptrdiff_t(b) * per, m, n, ld), [](T v) { return device::Arith<T>::add(v, v); }, a);
        T* ob = o + std::ptrdiff_t(b) * kSlotsL;
        if (ex.leader()) ob[kMax] = T(r0), ob[kOne] = T(r1), ob[kInf] = T(r2), ob[kFro] = T(r3), ob[kFroS] = T(r4);
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
        EXPECT_VERIFY(T, Check::blas, 2, rel(G[kMax], LD<T>(mx), mx)) << "Max, WG " << WG << " m " << m;
        EXPECT_VERIFY(T, Check::blas, m, rel(G[kOne], LD<T>(one), one)) << "One, WG " << WG << " m " << m;
        EXPECT_VERIFY(T, Check::blas, n, rel(G[kInf], LD<T>(inf), inf)) << "Inf, WG " << WG << " m " << m;
        EXPECT_VERIFY(T, Check::blas, m * n, rel(G[kFroS], LD<T>(std::sqrt(ss)), std::sqrt(ss))) << "Fro scaled";
        if (ss < static_cast<long double>(std::numeric_limits<R>::max()))
            EXPECT_VERIFY(T, Check::blas, m * n, rel(G[kFro], LD<T>(std::sqrt(ss)), std::sqrt(ss))) << "Fro unscaled";
        else
            EXPECT_TRUE(std::isinf(std::abs(G[kFro])) && std::isfinite(std::abs(G[kFroS])))
                << "past the threshold only the unscaled form may overflow";
    }
    EXPECT_EQ(walk_bad, 0) << "map2d flat walk, WG " << WG << " m " << m;
}

// m = 13 on 64 items: a carry each step; m = 100: fewer than one column per step; m = 1: one row.
TYPED_TEST(DeviceBlasV2GlueTest, LangeAndFlatWalkOnWorkGroup) {
    using T = TypeParam;
    using R = real_of<T>;
    lange_case<T, 64>(this->dev, 13, 9, 16, 1);
    lange_case<T, 64>(this->dev, 100, 3, 103, 1);
    lange_case<T, 256>(this->dev, 1, 300, 2, 1);
    lange_case<T, 1024>(this->dev, 37, 41, 40, 1);  // the work-group ceiling
    // The Frobenius threshold from both sides: sum |a|^2 just under max and well over it.
    lange_case<T, 256>(this->dev, 17, 11, 19, std::sqrt(std::numeric_limits<R>::max()) / R(64));
    lange_case<T, 256>(this->dev, 17, 11, 19, std::sqrt(std::numeric_limits<R>::max()) / R(2));
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
        this->dev.template run3<T>(sycl::nd_range<3>({std::size_t(kBatch), up(cols), up(rows)}, {1, TL, TL}), TL * (TL + 1),
                                   [=](sycl::nd_item<3> it, T* lm) {
                                       const device::Grid2D<TL, TL, 1> ex{it};
                                       const int b = ex.batch();
                                       const int r0 = static_cast<int>(it.get_group(2)) * TL, c0 = static_cast<int>(it.get_group(1)) * TL;
                                       const auto t = device::mat<device::Mem::Local>(lm, TL, TL, TL + 1);
                                       device::glue::tile_load_trans<TL>(ex, device::mat(A + std::ptrdiff_t(b) * pa, rows, cols, lda), r0, c0, t);
                                       ex.barrier();
                                       device::glue::tile_store<TL>(ex, t, device::mat(Bo + std::ptrdiff_t(b) * pb, cols, rows, ldb), c0, r0);
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
    // Mirror from each triangle (the unreferenced one poisoned): runtime Lower + conj, static
    // Upper without; 32 x 32 tiles, 8 columns per group, so each item stores 4 columns.
    constexpr int TM = 32, GC = 8, n = 45, lda = n + 2, ldo = n + 4, tiles = (n + TM - 1) / TM;
    const int pa = lda * n, po = ldo * n;
    for (const bool lower : {true, false}) {
        auto hA = batched<T>(pa, lower ? 61 : 62, [&](T* p) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (lower ? i < j : i > j) p[i + j * lda] = poison<T>();
        });
        T* A = this->dev.up(hA);
        T* O = this->dev.up(std::vector<T>(std::size_t(kBatch) * po, T(-777)));
        this->dev.template run3<T>(
            sycl::nd_range<3>({std::size_t(kBatch), std::size_t(tiles) * GC, std::size_t(tiles) * TM}, {1, GC, TM}), TM * (TM + 1),
            [=](sycl::nd_item<3> it, T* lm) {
                const device::Grid2D<TM, GC, 1> ex{it};
                const int ti = static_cast<int>(it.get_group(1)), tj = static_cast<int>(it.get_group(2));
                if (ti < tj) return;
                const int b = ex.batch();
                const int r0 = (lower ? ti : tj) * TM, c0 = (lower ? tj : ti) * TM;
                const auto a = device::mat_wide(A + std::ptrdiff_t(b) * pa, n, n, lda);
                const auto o = device::mat_wide(O + std::ptrdiff_t(b) * po, n, n, ldo);
                const auto t = device::mat<device::Mem::Local>(lm, TM, TM, TM + 1);
                device::glue::tile_load<TM>(ex, a, r0, c0, t);
                sycl::group_barrier(it.get_group());
                if (lower) {
                    if (ti == tj) device::glue::tile_store_diag<true, TM>(ex, Uplo::Lower, t, o, r0);
                    else device::glue::tile_store_pair<true, TM>(ex, t, o, r0, c0);
                } else {
                    if (ti == tj) device::glue::tile_store_diag<false, TM>(ex, device::glue::UpperC{}, t, o, r0);
                    else device::glue::tile_store_pair<false, TM>(ex, t, o, r0, c0);
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

}  // namespace

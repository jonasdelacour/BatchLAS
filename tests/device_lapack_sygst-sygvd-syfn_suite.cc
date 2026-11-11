// SPDX-License-Identifier: MIT
// detail of lapack/sygst.hh (the ?sygs2 body's helpers) against host references; the core
// ops they compose with (device::load / store, arith::Fma) are covered by device_core_ops_tests.
// Guards (docs/developer/agent-guide.md section 8): pack-padded, non-natural ld and inc, complex
// data, every Uplo x {Trans, ConjTrans}, a NaN-poisoned unread triangle, pad and fragment tail,
// a poisoned Im(diag) the ops must not read, compile-time and run-time extents (n not a multiple
// of the pack), sentinels outside every written range, batch 1024 with period-3 data where every
// item must be bit-identical to its representative, a sweep where m does not divide the group.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/sygst.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

// Folded into device_lapack_sygst_tests (one binary per routine): its own namespace keeps the two
// translation units' helper and kernel names apart.
namespace device_lapack_sygst_sygvd_syfn_suite {

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
template <class T> T nan_of() {
    const real_of<T> q = std::numeric_limits<real_of<T>>::quiet_NaN();
    if constexpr (is_cplx<T>) return T(q, q);
    else return q;
}
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> T with_imag(real_of<T> re, real_of<T> im) {
    if constexpr (is_cplx<T>) return T(re, im);
    else return re;
}
bool in_tri(Uplo ul, int i, int j) { return ul == Uplo::Lower ? i >= j : i <= j; }

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

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

// Items whose block of `per` values differs from its representative's.
template <class T> int twin_diffs(const std::vector<T>& got, std::size_t per) {
    int d = 0;
    for (std::size_t b = kPeriod; b < std::size_t(kBatch); ++b)
        for (std::size_t i = 0; i < per; ++i) d += !same_bits(got[b * per + i], got[(b % kPeriod) * per + i]);
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
    template <int WG, class F> void run(int groups, F f) {
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

template <class T> class DeviceLapackSygstTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceLapackSygstTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }

template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

template <class NE> NE make_ne(int n) {
    if constexpr (device::static_extent_v<NE> > 0) return NE{};
    else return NE{n};
}

// ---- trsv_packed / trmv_packed ---------------------------------------------------------------

// One work-item per item: x := op(A)^-1 x or op(A) x, stored with inc 2. The Frag tail past n is
// NaN after the load; A is n x n with ld padded to whole packs plus one, its unread triangle and
// padding NaN, Im(diag) 1e3.
template <class T, Uplo UP, Transpose TR, bool Solve, class NE, int N>
void check_tri(Dev& dev, int n, const char* what) {
    constexpr int V = device::detail::pack16_elems<T>;
    const int ld = (n + V - 1) / V * V + V, per_a = ld * n;
    const auto upper = UP == Uplo::Upper;
    auto ha = batched<T>(per_a, 11, [&](T* a) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ld; ++i) {
                T& e = a[i + j * ld];
                if (i >= n || (upper ? i > j : i < j)) e = nan_of<T>();
                else if (i == j) e = with_imag<T>(real_of<T>(1.5) + real_of<T>(0.5) * std::real(e), real_of<T>(1e3));
                else e = e * real_of<T>(0.5 / n);
            }
    });
    auto hx = batched<T>(n, 12, [](T*) {});
    std::vector<T> hy(std::size_t(kBatch) * 2 * n, nan_of<T>());
    const T* A = dev.up(ha);
    const T* X = dev.up(hx);
    T* Y = dev.up(hy);
    dev.template run<32>(kBatch / 32, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id()) * 32 + static_cast<int>(it.get_local_linear_id());
        const device::Thread th{};
        const NE ne = make_ne<NE>(n);
        const device::Mat<const T, NE, NE, device::Dyn> a{A + std::ptrdiff_t(b) * per_a, ne, ne, device::Dyn{ld}};
        device::Frag<T, N> x;
        device::load(x, device::vec(X + std::ptrdiff_t(b) * n, n), n);
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (i >= n) x[i] = device::Arith<T>::make(std::numeric_limits<real_of<T>>::quiet_NaN(), real_of<T>(0));
        if constexpr (Solve) device::detail::trsv_packed<UP, TR>(th, a, x);
        else device::detail::trmv_packed<UP, TR>(th, a, x);
        device::store(x, device::vec(Y + std::ptrdiff_t(b) * 2 * n, n, 2), n);
    });
    const auto got = dev.down(Y, hy.size());
    EXPECT_EQ(twin_diffs(got, std::size_t(2) * n), 0) << what << ": items differ from their representative";
    double worst = 0;
    int pad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* a = ha.data() + std::size_t(b) * per_a;
        const T* x = hx.data() + std::size_t(b) * n;
        const T* y = got.data() + std::size_t(b) * 2 * n;
        // M = op(A) on A's triangle, real diagonal.
        auto m = [&](int i, int p) -> LD<T> {
            if (i == p) return LD<T>(std::real(wide(a[i + i * ld])));
            if (upper ? p > i : p < i) return LD<T>(0);
            const T apj = a[p + i * ld];
            return wide(TR == Transpose::ConjTrans ? conj_h(apj) : apj);
        };
        std::vector<LD<T>> ref(n);
        if constexpr (!Solve) {
            for (int i = 0; i < n; ++i) {
                LD<T> s = 0;
                long double sc = 0;
                for (int p = 0; p < n; ++p) s += m(i, p) * wide(x[p]), sc += std::abs(m(i, p)) * std::abs(wide(x[p]));
                worst = std::max(worst, rel(y[2 * i], s, sc));
            }
        } else {
            const bool fwd = upper;  // op(A) lower: forward substitution
            for (int s = 0; s < n; ++s) {
                const int i = fwd ? s : n - 1 - s;
                LD<T> acc = wide(x[i]);
                long double sc = std::abs(wide(x[i]));
                for (int p = 0; p < n; ++p)
                    if (p != i && (fwd ? p < i : p > i)) acc -= m(i, p) * ref[p], sc += std::abs(m(i, p)) * std::abs(ref[p]);
                ref[i] = acc / m(i, i);
                worst = std::max(worst, rel(y[2 * i], ref[i], sc / std::abs(m(i, i))));
            }
        }
        for (int i = 0; i < n; ++i) pad += !same_bits(y[2 * i + 1], nan_of<T>());
    }
    EXPECT_EQ(pad, 0) << what << ": inc-2 gaps written";
    EXPECT_VERIFY(T, Check::blas, n, worst) << what;
}

template <class T, Uplo UP, Transpose TR, bool Solve> void check_tri_extents(Dev& dev, const char* what) {
    check_tri<T, UP, TR, Solve, device::Ext<16>, 16>(dev, 16, what);  // compile-time order, whole packs
    check_tri<T, UP, TR, Solve, device::Dyn, 32>(dev, 13, what);       // run-time order < N, partial last pack
}

TYPED_TEST(DeviceLapackSygstTest, TrsvPackedEveryUploAndTranspose) {
    using T = TypeParam;
    check_tri_extents<T, Uplo::Upper, Transpose::Trans, true>(this->dev, "trsv U T");  // sygst itype 1
    check_tri_extents<T, Uplo::Upper, Transpose::ConjTrans, true>(this->dev, "trsv U C");
    check_tri_extents<T, Uplo::Lower, Transpose::Trans, true>(this->dev, "trsv L T");
    check_tri_extents<T, Uplo::Lower, Transpose::ConjTrans, true>(this->dev, "trsv L C");
}

TYPED_TEST(DeviceLapackSygstTest, TrmvPackedEveryUploAndTranspose) {
    using T = TypeParam;
    check_tri_extents<T, Uplo::Upper, Transpose::Trans, false>(this->dev, "trmv U T");
    check_tri_extents<T, Uplo::Upper, Transpose::ConjTrans, false>(this->dev, "trmv U C");
    check_tri_extents<T, Uplo::Lower, Transpose::Trans, false>(this->dev, "trmv L T");
    check_tri_extents<T, Uplo::Lower, Transpose::ConjTrans, false>(this->dev, "trmv L C");  // sygst itype 2, 3
}

// ---- herm / device::conj_view through device::load, and device::lacpy on an ElemAt ------------------------

// S stores one triangle (the other 1e6, Im(diag) 1e3). Per column j: O1(:, j) = column j of the
// Hermitian S, O2(:, j) = conj(row j of S), loaded into a Frag and stored; O3 = herm(S) by
// device::lacpy element by element. Rows n.. of O1, O2, O3 (ld n + 2) stay NaN.
template <class T, class NE, int N> void check_views(Dev& dev, Uplo ul, int n) {
    const int lds = n + 3, ldo = n + 2, per_s = lds * n, per_o = ldo * n;
    auto hs = batched<T>(per_s, 21, [&](T* s) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (i == j) s[i + j * lds] = with_imag<T>(std::real(s[i + j * lds]), real_of<T>(1e3));
                else if (!in_tri(ul, i, j)) s[i + j * lds] = with_imag<T>(real_of<T>(1e6), real_of<T>(-1e6));
            }
    });
    std::vector<T> ho(std::size_t(kBatch) * per_o, nan_of<T>());
    const T* S = dev.up(hs);
    T* O1 = dev.up(ho);
    T* O2 = dev.up(ho);
    T* O3 = dev.up(ho);
    dev.template run<32>(kBatch / 32, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id()) * 32 + static_cast<int>(it.get_local_linear_id());
        const NE ne = make_ne<NE>(n);
        const device::Mat<const T, NE, NE, device::Dyn> s{S + std::ptrdiff_t(b) * per_s, ne, ne, device::Dyn{lds}};
        const device::Mat<T, NE, NE, device::Dyn> o1{O1 + std::ptrdiff_t(b) * per_o, ne, ne, device::Dyn{ldo}};
        const device::Mat<T, NE, NE, device::Dyn> o2{O2 + std::ptrdiff_t(b) * per_o, ne, ne, device::Dyn{ldo}};
        const device::Mat<T, NE, NE, device::Dyn> o3{O3 + std::ptrdiff_t(b) * per_o, ne, ne, device::Dyn{ldo}};
        for (int j = 0; j < n; ++j) {
            device::Frag<T, N> x{};
            device::load(x, device::detail::herm(s, ul).col(j), n);
            device::store(x, o1.col(j), n);
            device::load(x, device::conj_view(s.row(j)), n);
            device::store(x, o2.col(j), n);
            for (int i = 0; i < n; ++i) device::lacpy(device::ElemAt<true>{i, j}, device::detail::herm(s, ul), o3);
        }
    });
    const auto g1 = dev.down(O1, ho.size()), g2 = dev.down(O2, ho.size()), g3 = dev.down(O3, ho.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* s = hs.data() + std::size_t(b) * per_s;
        const std::size_t o = std::size_t(b) * per_o;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ldo; ++i) {
                const std::size_t at = o + std::size_t(j) * ldo + i;
                if (i >= n) {
                    bad += !same_bits(g1[at], nan_of<T>()) + !same_bits(g2[at], nan_of<T>()) + !same_bits(g3[at], nan_of<T>());
                    continue;
                }
                const T h = i == j ? with_imag<T>(std::real(s[i + i * lds]), 0) : in_tri(ul, i, j) ? s[i + j * lds] : conj_h(s[j + i * lds]);
                bad += !same_bits(g1[at], h) + !same_bits(g2[at], conj_h(s[j + i * lds])) + !same_bits(g3[at], h);
            }
    }
    EXPECT_EQ(bad, 0) << (ul == Uplo::Lower ? "lower" : "upper") << " n=" << n;
}

TYPED_TEST(DeviceLapackSygstTest, HermAndConjViewsThroughLoadAndLacpyAt) {
    using T = TypeParam;
    for (Uplo ul : {Uplo::Lower, Uplo::Upper}) {
        check_views<T, device::Ext<8>, 8>(this->dev, ul, 8);
        check_views<T, device::Dyn, 16>(this->dev, ul, 11);
    }
}

// ---- for_each_tile_col ------------------------------------------------------------------------

// A WG of 128 sweeps `tiles` m x m tiles: every element at least once; exactly once when m
// divides 128; the row passed is lid % m.
TYPED_TEST(DeviceLapackSygstTest, ForEachTileColCoversEveryTileElement) {
    if constexpr (!std::is_same_v<TypeParam, float>) GTEST_SKIP() << "type-free; runs once";
    struct Case {
        int m, tiles;
    };
    for (const Case cs : {Case{16, 8}, Case{32, 4}, Case{24, 5}, Case{7, 18}}) {
        const int m = cs.m, tiles = cs.tiles, cells = tiles * m * m;
        std::vector<int> hc(cells + 1, 0);
        int* cnt = this->dev.up(hc);
        this->dev.template run<128>(1, [=](sycl::nd_item<1> it) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            device::detail::for_each_tile_col(lid, 128, m, tiles, [&](int g, int row, int c) {
                using Ref = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
                Ref(cnt[g * m * m + row + c * m]).fetch_add(1);
                if (row != lid % m) Ref(cnt[cells]).fetch_add(1);
            });
        });
        const auto got = this->dev.down(cnt, hc.size());
        int missed = 0, twice = 0;
        for (int e = 0; e < cells; ++e) missed += got[e] == 0, twice += got[e] > 1;
        EXPECT_EQ(missed, 0) << "m=" << m << " tiles=" << tiles;
        EXPECT_EQ(got[cells], 0) << "row != lid % m";
        if (128 % m == 0) EXPECT_EQ(twice, 0) << "m=" << m << " divides the group: one visit each";
    }
}

}  // namespace

}  // namespace device_lapack_sygst_sygvd_syfn_suite

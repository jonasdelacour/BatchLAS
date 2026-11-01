// SPDX-License-Identifier: MIT
// Device LA v2 sygst / sygvd / syfn ops (include/batchlas/blas/device/v2/ext/sygst-sygvd-syfn.hh)
// against host references. Guards (docs/developer/agent-guide.md section 8): pack-padded and
// non-natural ld / inc, complex data, ConjTrans, a NaN-poisoned unread triangle and fragment tail,
// a poisoned Im(diag) the RealDiag ops must not read, compile-time and run-time extents (n not a
// multiple of the pack), sentinels outside every written range, batch 1024 with period-3 data
// where every item must be bit-identical to its representative, a sweep where m does not divide
// the group (revisits allowed, misses not).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/sygst-sygvd-syfn.hh>
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

// kPeriod representatives (edited by fix(item pointer)) copied to every item b from b % kPeriod.
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

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

// Every item's block of `per` values bit-identical to its representative's.
template <class T> int twin_diffs(const std::vector<T>& got, std::size_t per) {
    int d = 0;
    for (std::size_t b = kPeriod; b < std::size_t(kBatch); ++b)
        for (std::size_t i = 0; i < per; ++i) d += !same_bits(got[b * per + i], got[(b % kPeriod) * per + i]);
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

template <class T> class DeviceBlasV2SygstTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2SygstTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }

template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

template <class NE> NE make_ne(int n) {
    if constexpr (v2::static_extent_v<NE> > 0) return NE{};
    else return NE{n};
}

// ---- trsv_frag / trmv_frag -------------------------------------------------------------------

// One work-item per item: x (NaN beyond n) := op(A)^-1 x or op(A) x, stored with inc 2. A is
// n x n with ld padded to whole packs plus one, its unread triangle and padding NaN, Im(diag) 1e3.
template <class T, Uplo UP, Transpose TR, bool Solve, class NE, int N>
void check_tri(Dev& dev, int n, const char* what) {
    constexpr int V = v2::pack16_elems<T>;
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
        const v2::Thread th{};
        const NE ne = make_ne<NE>(n);
        const v2::Mat<const T, NE, NE, v2::Dyn> a{A + std::ptrdiff_t(b) * per_a, ne, ne, v2::Dyn{ld}};
        v2::Frag<T, N> x;
#pragma unroll
        for (int i = 0; i < N; ++i) x[i] = v2::Arith<T>::make(std::numeric_limits<real_of<T>>::quiet_NaN(), real_of<T>(0));
        x = v2::frag_load(th, ne, v2::vec(X + std::ptrdiff_t(b) * n, n), x);
        if constexpr (Solve) x = v2::trsv_frag<UP, TR>(th, a, x);
        else x = v2::trmv_frag<UP, TR>(th, a, x);
        v2::frag_store(th, ne, x, v2::vec(Y + std::ptrdiff_t(b) * 2 * n, n, 2));
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
    check_tri<T, UP, TR, Solve, v2::Ext<16>, 16>(dev, 16, what);  // compile-time order, whole packs
    check_tri<T, UP, TR, Solve, v2::Dyn, 32>(dev, 13, what);       // run-time order < N, partial last pack
}

TYPED_TEST(DeviceBlasV2SygstTest, TrsvFragColumnDotEveryUploAndTranspose) {
    using T = TypeParam;
    check_tri_extents<T, Uplo::Upper, Transpose::Trans, true>(this->dev, "trsv U T");  // sygst itype 1
    check_tri_extents<T, Uplo::Upper, Transpose::ConjTrans, true>(this->dev, "trsv U C");
    check_tri_extents<T, Uplo::Lower, Transpose::Trans, true>(this->dev, "trsv L T");
    check_tri_extents<T, Uplo::Lower, Transpose::ConjTrans, true>(this->dev, "trsv L C");
}

TYPED_TEST(DeviceBlasV2SygstTest, TrmvFragColumnDotEveryUploAndTranspose) {
    using T = TypeParam;
    check_tri_extents<T, Uplo::Upper, Transpose::Trans, false>(this->dev, "trmv U T");
    check_tri_extents<T, Uplo::Upper, Transpose::ConjTrans, false>(this->dev, "trmv U C");
    check_tri_extents<T, Uplo::Lower, Transpose::Trans, false>(this->dev, "trmv L T");
    check_tri_extents<T, Uplo::Lower, Transpose::ConjTrans, false>(this->dev, "trmv L C");  // sygst itype 2, 3
}

// ---- frag_load / frag_store over herm and conj_vec views ---------------------------------------

// S stores one triangle (the other 1e6, Im(diag) 1e3); per column j: O1(:, j) = column j of the
// Hermitian S, O2(:, j) = conj(row j of S). Rows n.. of O1, O2 (ld n + 2) stay NaN.
template <class T, class NE, int N> void check_views(Dev& dev, Uplo ul, int n) {
    const int lds = n + 3, ldo = n + 2, per_s = lds * n, per_o = ldo * n;
    const bool lower = ul == Uplo::Lower;
    auto hs = batched<T>(per_s, 21, [&](T* s) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (i == j) s[i + j * lds] = with_imag<T>(std::real(s[i + j * lds]), real_of<T>(1e3));
                else if (!v2::in_triangle(ul, i, j)) s[i + j * lds] = with_imag<T>(real_of<T>(1e6), real_of<T>(-1e6));
            }
    });
    std::vector<T> ho(std::size_t(kBatch) * per_o, nan_of<T>());
    const T* S = dev.up(hs);
    T* O1 = dev.up(ho);
    T* O2 = dev.up(ho);
    dev.template run<32>(kBatch / 32, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id()) * 32 + static_cast<int>(it.get_local_linear_id());
        const v2::Thread th{};
        const NE ne = make_ne<NE>(n);
        const v2::Mat<const T, NE, NE, v2::Dyn> s{S + std::ptrdiff_t(b) * per_s, ne, ne, v2::Dyn{lds}};
        const v2::Mat<T, NE, NE, v2::Dyn> o1{O1 + std::ptrdiff_t(b) * per_o, ne, ne, v2::Dyn{ldo}};
        const v2::Mat<T, NE, NE, v2::Dyn> o2{O2 + std::ptrdiff_t(b) * per_o, ne, ne, v2::Dyn{ldo}};
        const Uplo u = lower ? Uplo::Lower : Uplo::Upper;
        for (int j = 0; j < n; ++j) {
            v2::Frag<T, N> x{};
            x = v2::frag_load(th, ne, v2::herm(s, u).col(j), x);
            v2::frag_store(th, ne, x, o1.col(j));
            x = v2::frag_load(th, ne, v2::conj_vec(s.row(j)), x);
            v2::frag_store(th, ne, x, o2.col(j));
        }
    });
    const auto g1 = dev.down(O1, ho.size()), g2 = dev.down(O2, ho.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* s = hs.data() + std::size_t(b) * per_s;
        const T* o1 = g1.data() + std::size_t(b) * per_o;
        const T* o2 = g2.data() + std::size_t(b) * per_o;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ldo; ++i) {
                if (i >= n) {
                    bad += !same_bits(o1[i + j * ldo], nan_of<T>()) + !same_bits(o2[i + j * ldo], nan_of<T>());
                    continue;
                }
                const T h = i == j ? with_imag<T>(std::real(s[i + i * lds]), 0)
                            : v2::in_triangle(ul, i, j) ? s[i + j * lds] : conj_h(s[j + i * lds]);
                bad += !same_bits(o1[i + j * ldo], h) + !same_bits(o2[i + j * ldo], conj_h(s[j + i * lds]));
            }
    }
    EXPECT_EQ(bad, 0) << (lower ? "lower" : "upper") << " n=" << n;
}

TYPED_TEST(DeviceBlasV2SygstTest, FragLoadStoreOverHermAndConjViews) {
    using T = TypeParam;
    for (Uplo ul : {Uplo::Lower, Uplo::Upper}) {
        check_views<T, v2::Ext<8>, 8>(this->dev, ul, 8);
        check_views<T, v2::Dyn, 16>(this->dev, ul, 11);
    }
}

// ---- for_each_tile_col ------------------------------------------------------------------------

// A WG of 128 sweeps `tiles` m x m tiles: every element at least once; exactly once when m
// divides 128; the row passed is lid % m (counted per (row, lid % m) match).
TYPED_TEST(DeviceBlasV2SygstTest, ForEachTileColCoversEveryTileElement) {
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
            v2::for_each_tile_col(lid, 128, m, tiles, [&](int g, int row, int c) {
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

// ---- laset_at / dgmm_at -----------------------------------------------------------------------

// One WG of 32 per item over the n x n elements (ld n + 3, NaN sentinels): L = laset(Lower,
// alpha, 1), U = laset(Upper, -1, beta), F = laset(full, 0, beta); Y1 = X diag(d real),
// Y2 = X diag(dt of X's type).
TYPED_TEST(DeviceBlasV2SygstTest, LasetAtAndDgmmAtElementOps) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 19, ld = n + 3, per = ld * n;
    const T alpha = val<T>(31), beta = val<T>(32);
    auto hx = batched<T>(per, 33, [](T*) {});
    auto hd = batched<R>(n, 34, [](R*) {});
    auto hdt = batched<T>(n, 35, [](T*) {});
    std::vector<T> hsent(std::size_t(kBatch) * per, nan_of<T>());
    const T* X = this->dev.up(hx);
    const R* Dr = this->dev.up(hd);
    const T* Dt = this->dev.up(hdt);
    T* L = this->dev.up(hsent);
    T* U = this->dev.up(hsent);
    T* F = this->dev.up(hsent);
    T* Y1 = this->dev.up(hsent);
    T* Y2 = this->dev.up(hsent);
    this->dev.template run<32>(kBatch, [=](sycl::nd_item<1> it) {
        const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        for (int e = static_cast<int>(it.get_local_linear_id()); e < n * n; e += 32) {
            const int i = e % n, j = e / n;
            const std::ptrdiff_t at = b * per + std::ptrdiff_t(j) * ld + i;
            v2::laset_at(Uplo::Lower, L[at], i, j, v2::Scalar{alpha}, v2::One{});
            v2::laset_at(Uplo::Upper, U[at], i, j, v2::NegOne{}, v2::Scalar{beta});
            v2::laset_at(F[at], i, j, v2::Zero{}, beta);
            Y1[at] = v2::dgmm_at(X[at], Dr[b * n + j]);
            Y2[at] = v2::dgmm_at(X[at], Dt[b * n + j]);
        }
    });
    const auto gl = this->dev.down(L, hsent.size()), gu = this->dev.down(U, hsent.size()),
               gf = this->dev.down(F, hsent.size()), g1 = this->dev.down(Y1, hsent.size()),
               g2 = this->dev.down(Y2, hsent.size());
    int bad = 0;
    double worst = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ld; ++i) {
                const std::size_t at = std::size_t(b) * per + std::size_t(j) * ld + i;
                if (i >= n) {
                    bad += !same_bits(gl[at], nan_of<T>()) + !same_bits(gu[at], nan_of<T>()) + !same_bits(gf[at], nan_of<T>()) +
                           !same_bits(g1[at], nan_of<T>()) + !same_bits(g2[at], nan_of<T>());
                    continue;
                }
                const T el = i == j ? T(1) : i > j ? alpha : nan_of<T>();
                const T eu = i == j ? beta : i < j ? -T(1) : nan_of<T>();  // NegOne: (-1, -0)
                const T ef = i == j ? beta : T(0);
                bad += !same_bits(gl[at], el) + !same_bits(gu[at], eu) + !same_bits(gf[at], ef);
                const LD<T> xw = wide(hx[at]), dt = wide(hdt[std::size_t(b) * n + j]);
                const long double dr = hd[std::size_t(b) * n + j];
                worst = std::max({worst, rel(g1[at], xw * dr, std::abs(xw) * std::abs(dr)),
                                  rel(g2[at], xw * dt, std::abs(xw) * std::abs(dt))});
            }
    EXPECT_EQ(bad, 0) << "laset_at values, untouched triangle or padding";
    EXPECT_EQ(twin_diffs(g1, per) + twin_diffs(g2, per), 0) << "dgmm_at items differ from their representative";
    EXPECT_VERIFY(T, Check::blas, 1, worst);
}

}  // namespace

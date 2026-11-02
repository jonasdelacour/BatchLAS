// SPDX-License-Identifier: MIT
// Device LA v2 steqr extension (include/batchlas/blas/device/v2/ext/steqr.hh) against host
// references. Guards (docs/developer/agent-guide.md section 8): non-natural inc and ld, complex
// data under real rotations, batch 1024 with period-3 data where every item must be
// bit-identical to its representative, the split test straddled (0.5x and 2x the threshold),
// windows that exclude a planted extreme on each side, an Inf next to identity rotations (only
// alg::SkipIdentity keeps it out of its neighbour), a NaN that lanst_max must ignore, every
// lane-register executor (lockstep Partition, masked SgpPartition, SubGroup), and a dense
// similarity G T G^T as the independent reference for tridiag_rot in both directions.
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

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/steqr.hh>
#include <src/kernels/common/dla_v2_partition.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

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
// Element i of item b: item b repeats representative b % kPeriod.
template <class T> T item_val(std::uint64_t seed, int b, int i) { return val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i); }

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

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

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

template <class T> class V2SteqrTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};
template <class T> class V2SteqrRotTest : public V2SteqrTest<T> {};
template <class T> class V2SteqrTridiagTest : public V2SteqrTest<T> {};

using RotTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(V2SteqrRotTest, RotTypes);
TYPED_TEST_SUITE(V2SteqrTridiagTest, RealTypes);

template <class R> v2::Givens<R> angle(std::uint64_t seed, int b, int j) {
    const R t = static_cast<R>(3.14159265358979 * double(item_val<R>(seed, b, j)));
    return {std::cos(t), std::sin(t)};
}

// ---- rotations -------------------------------------------------------------------------------

// rot on two strided vectors and lasr_rows on a matrix with ld > m, rows Cyclic on WG 64.
TYPED_TEST(V2SteqrRotTest, RotAndLasrRowsOnWorkGroup) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 64, n = 77, incx = 2, incy = 3, m = 45, nc = 9, ld = m + 5;
    std::vector<T> hx(std::size_t(kBatch) * n * incx), hy(std::size_t(kBatch) * n * incy), ha(std::size_t(kBatch) * ld * nc);
    std::vector<v2::Givens<R>> hg(std::size_t(kBatch) * nc);
    for (int b = 0; b < kBatch; ++b) {
        for (std::size_t i = 0; i < std::size_t(n * incx); ++i) hx[b * std::size_t(n * incx) + i] = item_val<T>(1, b, int(i));
        for (std::size_t i = 0; i < std::size_t(n * incy); ++i) hy[b * std::size_t(n * incy) + i] = item_val<T>(2, b, int(i));
        for (std::size_t i = 0; i < std::size_t(ld * nc); ++i) ha[b * std::size_t(ld * nc) + i] = item_val<T>(3, b, int(i));
        for (int j = 0; j < nc; ++j) hg[std::size_t(b) * nc + j] = angle<R>(4, b, j);
    }
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    T* a = this->dev.up(ha);
    const v2::Givens<R>* g = this->dev.up(hg);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const v2::Givens<R>* gb = g + std::ptrdiff_t(b) * nc;
        v2::rot(ex, gb[nc - 1], v2::vec(x + std::ptrdiff_t(b) * n * incx, n, incx), v2::vec(y + std::ptrdiff_t(b) * n * incy, n, incy));
        v2::lasr_rows(ex, [=](int j) { return gb[j]; }, v2::mat(a + std::ptrdiff_t(b) * ld * nc, m, nc, ld));
    });
    const auto gx = this->dev.down(x, hx.size()), gy = this->dev.down(y, hy.size()), ga = this->dev.down(a, ha.size());
    double worst_rot = 0, worst_lasr = 0;
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b) {
        const std::size_t r = std::size_t(b % kPeriod);
        for (int i = 0; i < n; ++i) {
            bitdiff += !same_bits(gx[b * std::size_t(n * incx) + i * incx], gx[r * (n * incx) + i * incx]);
            bitdiff += !same_bits(gy[b * std::size_t(n * incy) + i * incy], gy[r * (n * incy) + i * incy]);
        }
        for (int i = 0; i < ld * nc; ++i) bitdiff += !same_bits(ga[b * std::size_t(ld * nc) + i], ga[r * (ld * nc) + i]);
        if (b >= kPeriod) continue;
        const auto G = hg[std::size_t(b) * nc + nc - 1];
        for (int i = 0; i < n; ++i) {
            const LD<T> xi = wide(hx[b * std::size_t(n * incx) + i * incx]), yi = wide(hy[b * std::size_t(n * incy) + i * incy]);
            const long double c = G.c, s = G.s, sc = std::abs(xi) + std::abs(yi);
            worst_rot = std::max({worst_rot, rel(gx[b * std::size_t(n * incx) + i * incx], c * xi - s * yi, sc),
                                  rel(gy[b * std::size_t(n * incy) + i * incy], s * xi + c * yi, sc)});
        }
        for (int i = m; i < ld; ++i)  // rows past m are padding and stay untouched
            for (int j = 0; j < nc; ++j) bitdiff += !same_bits(ga[b * std::size_t(ld * nc) + i + j * ld], ha[b * std::size_t(ld * nc) + i + j * ld]);
        for (int i = 0; i < m; ++i) {
            std::vector<LD<T>> row(nc);
            long double nrm = 0;
            for (int j = 0; j < nc; ++j) {
                row[j] = wide(ha[b * std::size_t(ld * nc) + i + j * ld]);
                nrm += std::norm(row[j]);
            }
            for (int j = 0; j + 1 < nc; ++j) {  // ascending j: a descending sweep differs
                const long double c = hg[std::size_t(b) * nc + j].c, s = hg[std::size_t(b) * nc + j].s;
                const LD<T> u = row[j], w = row[j + 1];
                row[j] = c * u - s * w;
                row[j + 1] = s * u + c * w;
            }
            for (int j = 0; j < nc; ++j) worst_lasr = std::max(worst_lasr, rel(ga[b * std::size_t(ld * nc) + i + j * ld], row[j], std::sqrt(nrm)));
        }
    }
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative, or padding rows were written";
    EXPECT_VERIFY(T, Check::blas, 2, worst_rot) << "rot";
    EXPECT_VERIFY(T, Check::blas, nc, worst_lasr) << "lasr_rows";
}

// lasr on Thread with alg::SkipIdentity vs alg::Dense: x(5) = Inf and both rotations touching it
// are the identity, so only the skipping form keeps x(4) finite.
TYPED_TEST(V2SteqrRotTest, LasrSkipIdentityKeepsInfOutOfNeighbours) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 12, p = 5, inc = 2;
    const R inf = std::numeric_limits<R>::infinity();
    std::vector<T> hx(std::size_t(kBatch) * n * inc);
    std::vector<std::array<R, 2>> hg(std::size_t(kBatch) * (n - 1));
    for (int b = 0; b < kBatch; ++b) {
        for (int i = 0; i < n * inc; ++i) hx[std::size_t(b) * n * inc + i] = item_val<T>(5, b, i);
        hx[std::size_t(b) * n * inc + p * inc] = T(inf);
        for (int j = 0; j < n - 1; ++j) {
            const auto G = angle<R>(6, b, j);
            hg[std::size_t(b) * (n - 1) + j] = (j == p - 1 || j == p) ? std::array<R, 2>{R(1), R(0)} : std::array<R, 2>{G.c, G.s};
        }
    }
    std::vector<T> hx2 = hx;
    T* xs = this->dev.up(hx);
    T* xd = this->dev.up(hx2);
    const std::array<R, 2>* g = this->dev.up(hg);
    this->dev.template run<32>(kBatch, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;  // Thread: one item owns the vector
        const int b = static_cast<int>(it.get_group_linear_id());
        const std::array<R, 2>* gb = g + std::ptrdiff_t(b) * (n - 1);
        const auto rots = [=](int j) -> const std::array<R, 2>& { return gb[j]; };
        v2::lasr<v2::alg::SkipIdentity>(v2::Thread{}, n, rots, v2::vec(xs + std::ptrdiff_t(b) * n * inc, n, inc));
        v2::lasr(v2::Thread{}, n, rots, v2::vec(xd + std::ptrdiff_t(b) * n * inc, n, inc));
    });
    const auto gs = this->dev.down(xs, hx.size()), gd = this->dev.down(xd, hx.size());
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<LD<T>> ref(n);
        for (int i = 0; i < n; ++i) ref[i] = wide(hx[std::size_t(b) * n * inc + i * inc]);
        for (int j = 0; j + 1 < n; ++j) {
            if (j == p - 1 || j == p) continue;
            const long double c = hg[std::size_t(b) * (n - 1) + j][0], s = hg[std::size_t(b) * (n - 1) + j][1];
            const LD<T> u = ref[j], w = ref[j + 1];
            ref[j] = c * u - s * w;
            ref[j + 1] = s * u + c * w;
        }
        for (int i = 0; i < n; ++i) {
            const T got = gs[std::size_t(b) * n * inc + i * inc];
            if (i == p) bad += !(std::isinf(std::real(got)));
            else worst = std::max(worst, rel(got, ref[i], 2.0L));
        }
        bad += !std::isnan(std::real(gd[std::size_t(b) * n * inc + (p - 1) * inc]));  // 1 x - 0 Inf
        for (int i = 0; i < n * inc; i += 2) bad += !same_bits(gs[std::size_t(b) * n * inc + i + 1], hx[std::size_t(b) * n * inc + i + 1]);
    }
    EXPECT_EQ(bad, 0) << "Inf placement, Dense NaN, or a write between strided elements";
    EXPECT_VERIFY(T, Check::blas, n, worst) << "lasr SkipIdentity";
}

// ---- tridiagonal operations, memory form on Thread ---------------------------------------------

// lanst_max ignores a NaN and must read d(n-1); tridiag_split zeroes exactly the entries under
// the threshold and reports whether it did; d and e have inc 2 with a poisoned gap.
TYPED_TEST(V2SteqrTridiagTest, LanstMaxAndSplitOnThread) {
    using T = TypeParam;
    constexpr int n = 9, inc = 2;
    const v2::SplitTol<T> tol{T(std::numeric_limits<T>::epsilon() / 2) * T(std::numeric_limits<T>::epsilon() / 2),
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
        e[2 * inc] = T(2) * thr(2);                      // kept: 4x the threshold squared
        if (rep != 0) e[(3 + rep) * inc] = T(0.5) * thr(3 + rep);  // split: 1/4 of it
    }
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    T* out = this->dev.up(std::vector<T>(2 * kBatch));
    this->dev.template run<32>(kBatch, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::Thread th{};
        const auto dv = v2::vec(d + std::ptrdiff_t(b) * n * inc, n, inc);
        const auto ev = v2::vec(e + std::ptrdiff_t(b) * n * inc, n - 1, inc);
        out[2 * b] = v2::lanst_max(th, dv, ev);
        out[2 * b + 1] = v2::tridiag_split(th, dv, ev, tol) ? T(1) : T(0);
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
TYPED_TEST(V2SteqrTridiagTest, TridiagRotMatchesDenseSimilarity) {
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
        bo[q] = v2::tridiag_rot(v2::Thread{}, v2::vec(d + std::ptrdiff_t(q) * n, n), v2::vec(e + std::ptrdiff_t(q) * (n - 1), n - 1),
                                bo[q], i, i + 1, g[q], fwd);
    });
    const auto gd = this->dev.down(d, hd.size()), ge = this->dev.down(e, he.size()), gb = this->dev.down(bo, hb.size());
    double worst = 0;
    for (int q = 0; q < nprob; ++q) {
        const int i = q % (n - 1), j = i + 1;
        const bool fwd = (q / (n - 1)) % 2 == 0;
        const auto P = [&](int k) { return fwd ? k : n - 1 - k; };      // virtual -> physical d
        const auto Pe = [&](int k) { return fwd ? k : n - 2 - k; };     // virtual -> physical e
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
        for (int k = 0; k + 1 < n; ++k)
            worst = std::max(worst, rel(ge[std::size_t(q) * (n - 1) + Pe(k)], (long double)C[k + (k + 1) * n], sc));
        const long double bulge = j + 1 < n ? C[i + (j + 1) * n] : 0.0L;
        worst = std::max(worst, rel(gb[q], bulge, sc));
    }
    EXPECT_VERIFY(T, Check::blas, 4, worst) << "tridiag_rot";
}

// ---- tridiagonal operations, lane-register form -------------------------------------------------

// One problem of n = P - 1 per chunk (the last lane is padding): lanst_max on windows that exclude
// a planted extreme on each side, the split over the whole e, find_first past a leading zero,
// then reverse of d, and the paired reverse of d[2, n-1) with e (e only on even problems).
template <class T, int P, class Mk> void lane_case(Dev& dev, Mk mk, const char* where) {
    constexpr int WG = 32, chunks = WG / P, n = P - 1, nprob = kBatch;
    const v2::SplitTol<T> tol{T(std::numeric_limits<T>::epsilon() / 2) * T(std::numeric_limits<T>::epsilon() / 2),
                              std::numeric_limits<T>::min()};
    std::vector<T> hd(std::size_t(nprob) * n), he(std::size_t(nprob) * n, T(0));
    for (int q = 0; q < nprob; ++q) {
        T* d = hd.data() + std::size_t(q) * n;
        T* e = he.data() + std::size_t(q) * n;
        for (int i = 0; i < n; ++i) d[i] = item_val<T>(11, q, i);
        for (int i = 0; i < n - 1; ++i) e[i] = item_val<T>(12, q, i);
        d[0] = T(100);      // outside the d window [1, n-1)
        d[n - 2] = T(-5);   // inside it: the expected norm
        e[n - 2] = T(50);   // outside the e window [1, n-2)
        e[0] = T(0);        // outside the find_first window [1, n-1)
        const auto thr = [&](int i) { return std::sqrt(tol.eps2 * std::abs(d[i]) * std::abs(d[i + 1]) + tol.safmin); };
        e[1] = T(2) * thr(1);                          // kept
        e[2 + q % kPeriod] = T(0.5) * thr(2 + q % kPeriod);  // split
    }
    T* d = dev.up(hd);
    T* e = dev.up(he);
    T* nrm = dev.up(std::vector<T>(nprob));
    int* idx = dev.up(std::vector<int>(nprob));
    dev.template run<WG>(nprob / chunks, [=](sycl::nd_item<1> it) {
        const auto ex = mk(it);
        const int q = static_cast<int>(it.get_group_linear_id()) * chunks + static_cast<int>(it.get_sub_group().get_local_linear_id()) / P;
        const int l = ex.lid();
        T dv = l < n ? d[std::ptrdiff_t(q) * n + l] : T(0);
        T ev = l < n - 1 ? e[std::ptrdiff_t(q) * n + l] : T(0);
        const T an = v2::lanst_max(ex, v2::lane_vec(dv, 1, n - 1), v2::lane_vec(ev, 1, n - 2));
        v2::tridiag_split(ex, dv, v2::lane_vec(ev, 0, n - 1), tol);
        const int first = v2::find_first(ex, v2::lane_vec(ev, 1, n - 1), n - 1, [](T x) { return x == T(0); });
        v2::reverse(ex, v2::lane_vec(dv, 1, n - 1), true);
        v2::reverse(ex, v2::lane_vec(dv, 2, n - 1), v2::lane_vec(ev, 1, q % 2 == 0 ? n - 2 : 1), true);
        if (l < n) d[std::ptrdiff_t(q) * n + l] = dv;
        if (l < n - 1) e[std::ptrdiff_t(q) * n + l] = ev;
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
        // d reversed on [1, n-1), then [2, n-1) reversed again: a rotation of d[1, n-1).
        for (int i = 0; i < n; ++i)
            bad += !same_bits(gd[std::size_t(q) * n + i], i == 1 ? d0[n - 2] : (i >= 2 && i < n - 1) ? d0[i - 1] : d0[i]);
        for (int i = 0; i < n - 1; ++i)
            bad += !same_bits(ge[std::size_t(q) * n + i], (q % 2 == 0 && i >= 1 && i < n - 2) ? es[n - 2 - i] : es[i]);
    }
    EXPECT_EQ(bad, 0) << where;
}

TYPED_TEST(V2SteqrTridiagTest, LaneOpsOnEveryPartitionExecutor) {
    using T = TypeParam;
    lane_case<T, 8>(this->dev, [](sycl::nd_item<1> it) { return v2::Partition<8>{it.get_sub_group()}; }, "Partition<8> lockstep");
    lane_case<T, 8>(this->dev, [](sycl::nd_item<1> it) { return v2::SgpPartition<8, true>{it.get_sub_group()}; },
                    "SgpPartition<8, Masked>");
    lane_case<T, 16>(this->dev, [](sycl::nd_item<1> it) { return v2::SgpPartition<16, false>{it.get_sub_group()}; },
                     "SgpPartition<16, lockstep>");
    lane_case<T, 32>(this->dev, [](sycl::nd_item<1> it) { return v2::SubGroup<32>{it.get_sub_group()}; }, "SubGroup<32>");
}

}  // namespace

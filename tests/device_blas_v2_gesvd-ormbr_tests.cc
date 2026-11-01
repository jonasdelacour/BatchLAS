// SPDX-License-Identifier: MIT
// Device LA v2 gesvd/ormbr extension (include/batchlas/blas/device/v2/ext/gesvd-ormbr.hh) against
// host references. Guards (docs/developer/agent-guide.md section 8): non-natural ld / inc / stride
// on every operand, rectangular shapes, complex data with nonzero imaginary parts, NaN-poisoned
// outputs where an op must not write (ld padding, the untouched triangle, outside an ElemIn2
// block that is not square), every one of kBatch = 1024 items checked against its own host data
// (item() must index item b), threshold straddles for any_of and the sign scan (-0.0, ties), a
// WG = 1024 launch, both the element executors and the Thread / WorkGroup distributions.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/ext/gesvd-ormbr.hh>
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
// kBatch items of per_item values; item b repeats representative b % kPeriod.
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i);
    return h;
}
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> std::vector<T> poisoned(std::size_t n) { return std::vector<T>(n, nan_v<T>()); }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> bool is_nan(const T& v) {
    if constexpr (is_cplx<T>) return std::isnan(v.real());
    else return std::isnan(v);
}
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
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
    // One item per element of each item's m x n grid, decomposed as the gesvd glue kernels do.
    template <class F> void grid(int m, int n, F f) {
        q().parallel_for(sycl::range<2>(std::size_t(kBatch), std::size_t(m) * n), [=](sycl::id<2> id) {
               const int b = static_cast<int>(id[0]), lin = static_cast<int>(id[1]);
               const int j = lin / m;
               f(b, lin - j * m, j);
           }).wait_and_throw();
    }
    template <class F> void line(int n, F f) { grid(n, 1, [=](int b, int i, int) { f(b, i); }); }
    template <int WG, class F> void run(F f) {
        q().parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
               f(it, static_cast<int>(it.get_group_linear_id()));
           }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GesvdOrmbrTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "the WorkGroup cases launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GesvdOrmbrTest, Types);

std::size_t at(int b, int stride, int i, int j, int ld) { return std::size_t(b) * stride + i + std::size_t(j) * ld; }

// laset / lacpy (widening) / omatcopy on Elem2 over a v1 KernelMatrixView (item()) and raw pointers.
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, Elem2CopiesAreExact) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int m = 37, n = 29, lda = m + 3, sa = lda * n, ldc = n + 5, sc = ldc * m;
    const T alpha = val<T>(7);
    auto ha = batched<T>(sa, 11);
    auto hr = batched<R>(sa, 12);
    T* a = this->dev.up(ha);
    R* ar = this->dev.up(hr);
    T* lp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sa));
    T* bp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sa));
    T* wp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sa));
    T* cp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sc));
    T* tp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sc));
    const KernelMatrixView<T, MatrixFormat::Dense> ak(a, m, n, lda, sa, kBatch);
    this->dev.grid(m, n, [=](int b, int i, int j) {
        const v2::Elem2 ex{i, j};
        const auto A = v2::item(ak, b, m, n);
        const std::ptrdiff_t ob = std::ptrdiff_t(b) * sa, oc = std::ptrdiff_t(b) * sc;
        v2::laset(ex, alpha, v2::One{}, v2::strided(lp, ob, lda, m, n));
        v2::lacpy(ex, A, v2::strided(bp, ob, lda, m, n));
        v2::lacpy(ex, v2::strided(ar, ob, lda, m, n), v2::strided(wp, ob, lda, m, n));
        v2::omatcopy<Transpose::ConjTrans>(ex, v2::One{}, A, v2::strided(cp, oc, ldc, n, m));
        v2::omatcopy<Transpose::Trans>(ex, v2::One{}, A, v2::strided(tp, oc, ldc, n, m));
    });
    const auto gl = this->dev.down(lp, std::size_t(kBatch) * sa), gb = this->dev.down(bp, std::size_t(kBatch) * sa);
    const auto gw = this->dev.down(wp, std::size_t(kBatch) * sa), gc = this->dev.down(cp, std::size_t(kBatch) * sc);
    const auto gt = this->dev.down(tp, std::size_t(kBatch) * sc);
    int bad = 0, pad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const std::size_t k = at(b, sa, i, j, lda), kt = at(b, sc, j, i, ldc);
                bad += !same_bits(gl[k], i == j ? T(1) : alpha) + !same_bits(gb[k], ha[k]) + !same_bits(gw[k], T(hr[k]));
                bad += !same_bits(gc[kt], conj_h(ha[k])) + !same_bits(gt[kt], ha[k]);
            }
            for (int i = m; i < lda; ++i) pad += !is_nan(gl[at(b, sa, i, j, lda)]) + !is_nan(gb[at(b, sa, i, j, lda)]);
        }
        for (int i = 0; i < m; ++i)
            for (int j = n; j < ldc; ++j) pad += !is_nan(gc[at(b, sc, j, i, ldc)]);
    }
    EXPECT_EQ(bad, 0);
    EXPECT_EQ(pad, 0) << "ld padding was written";
}

// trcopy (the ormbr P panel), the ormbr AQ composition laset + lacpy_strict on ElemIn2, a
// non-square ElemIn2 block (its column bound is not implied), and the WorkGroup / Thread paths.
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, TriangularCopiesWriteExactlyTheirPart) {
    using T = TypeParam;
    constexpr int N = 40, lda = N + 3, sa = lda * N, i0 = 2, ib = 8, mv = N - i0 - 1, ldv = mv + 2, sv = ldv * ib;
    constexpr int ldq = N + 1, sq = ldq * N, Mf = 24, Nf = 12, ldf = Mf + 3, sf = ldf * Nf, rm = 20, rn = 5;
    auto ha = batched<T>(sa, 21);
    T* a = this->dev.up(ha);
    T* vp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sv));
    T* qp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sq));
    T* fp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sf));
    T* wp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sa));
    T* tp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sa));
    const KernelMatrixView<T, MatrixFormat::Dense> ak(a, N, N, lda, sa, kBatch), qk(qp, N, N, ldq, sq, kBatch);
    this->dev.grid(mv, ib, [=](int b, int r, int c) {
        const auto ap = v2::strided(static_cast<const T*>(a), std::ptrdiff_t(b) * sa, lda, N, N);
        v2::trcopy<Uplo::Lower, Transpose::ConjTrans, Diag::Unit>(v2::Elem2{r, c}, v2::shifted(ap, i0, i0 + 1, ib, mv),
                                                                  v2::strided(vp, std::ptrdiff_t(b) * sv, ldv, mv, ib));
    });
    this->dev.grid(N, N, [=](int b, int row, int col) {
        const v2::Elem2 ex{row, col};
        const auto aq = v2::item(qk, b, N, N);
        v2::laset(ex, v2::Zero{}, v2::Zero{}, aq);
        v2::lacpy_strict<Uplo::Lower, Transpose::ConjTrans>(ex.sub(1, 1), v2::shifted(v2::item(ak, b, N, N), 0, 1, N - 1, N - 1),
                                                            v2::shifted(aq, 1, 1, N - 1, N - 1));
    });
    this->dev.grid(Mf, Nf, [=](int b, int r, int c) {
        const auto src = v2::strided(static_cast<const T*>(a), std::ptrdiff_t(b) * sa, lda, rm, rn);
        const auto dst = v2::shifted(v2::strided(fp, std::ptrdiff_t(b) * sf, ldf, Mf, Nf), 1, 2, rm, rn);
        v2::lacpy_strict<Uplo::Lower, Transpose::NoTrans>(v2::Elem2{r, c}.sub(1, 2), src, dst);
    });
    this->dev.template run<64>([=](sycl::nd_item<1> it, int b) {
        const auto A = v2::strided(static_cast<const T*>(a), std::ptrdiff_t(b) * sa, lda, N, N);
        v2::lacpy_strict<Uplo::Lower, Transpose::ConjTrans>(v2::WorkGroup<64>{it}, A, v2::strided(wp, std::ptrdiff_t(b) * sa, lda, N, N));
        if (it.get_local_linear_id() == 0)
            v2::trcopy<Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>(v2::Thread{}, A, v2::strided(tp, std::ptrdiff_t(b) * sa, lda, N, N));
    });
    const auto gv = this->dev.down(vp, std::size_t(kBatch) * sv), gq = this->dev.down(qp, std::size_t(kBatch) * sq);
    const auto gf = this->dev.down(fp, std::size_t(kBatch) * sf), gw = this->dev.down(wp, std::size_t(kBatch) * sa);
    const auto gt = this->dev.down(tp, std::size_t(kBatch) * sa);
    int bad_v = 0, bad_q = 0, bad_f = 0, bad_w = 0, bad_t = 0;
    for (int b = 0; b < kBatch; ++b) {
        const auto A = [&](int i, int j) { return ha[at(b, sa, i, j, lda)]; };
        for (int c = 0; c < ib; ++c)
            for (int r = 0; r < mv; ++r)
                bad_v += !same_bits(gv[at(b, sv, r, c, ldv)], r == c ? T(1) : (r > c ? conj_h(A(i0 + c, i0 + 1 + r)) : T(0)));
        for (int c = 0; c < N; ++c)
            for (int r = 0; r < N; ++r)
                bad_q += !same_bits(gq[at(b, sq, r, c, ldq)], (c >= 1 && r > c) ? conj_h(A(c - 1, r)) : T(0));
        for (int c = 0; c < Nf; ++c)
            for (int r = 0; r < Mf; ++r) {
                const int i = r - 1, j = c - 2;
                const bool in = j >= 0 && j < rn && i > j && i < rm;
                const T g = gf[at(b, sf, r, c, ldf)];
                bad_f += in ? !same_bits(g, A(i, j)) : !is_nan(g);
            }
        for (int c = 0; c < N; ++c)
            for (int r = 0; r < N; ++r) {
                const T w = gw[at(b, sa, r, c, lda)], t = gt[at(b, sa, r, c, lda)];
                bad_w += r > c ? !same_bits(w, conj_h(A(c, r))) : !is_nan(w);
                bad_t += !same_bits(t, r >= c ? A(r, c) : T(0));
            }
    }
    EXPECT_EQ(bad_v, 0) << "trcopy<Lower, ConjTrans, Unit> on Elem2";
    EXPECT_EQ(bad_q, 0) << "laset + lacpy_strict on ElemIn2 (ormbr AQ)";
    EXPECT_EQ(bad_f, 0) << "lacpy_strict on a non-square ElemIn2 block";
    EXPECT_EQ(bad_w, 0) << "lacpy_strict on WorkGroup<64>";
    EXPECT_EQ(bad_t, 0) << "trcopy<Lower, NoTrans, NonUnit> on Thread";
}

// mirror: Hermitian Lower on Elem2, plain Upper on a WG = 1024 launch (the ceiling); diagonal kept.
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, MirrorFillsTheOtherTriangle) {
    using T = TypeParam;
    constexpr int n = 45, ld = n + 4, s = ld * n;
    auto h = batched<T>(s, 31);
    T* p = this->dev.up(h);
    T* u = this->dev.up(h);
    const KernelMatrixView<T, MatrixFormat::Dense> pk(p, n, n, ld, s, kBatch);
    this->dev.grid(n, n, [=](int b, int i, int j) { v2::mirror<Uplo::Lower>(v2::Elem2{i, j}, v2::item(pk, b, n, n)); });
    this->dev.template run<1024>([=](sycl::nd_item<1> it, int b) {
        v2::mirror<Uplo::Upper, false>(v2::WorkGroup<1024>{it}, v2::strided(u, std::ptrdiff_t(b) * s, ld, n, n));
    });
    const auto gp = this->dev.down(p, h.size()), gu = this->dev.down(u, h.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t k = at(b, s, i, j, ld), kt = at(b, s, j, i, ld);
                bad += !same_bits(gp[k], i < j ? conj_h(h[kt]) : h[k]);
                bad += !same_bits(gu[k], i > j ? h[kt] : h[k]);
            }
    EXPECT_EQ(bad, 0);
}

// dgmm Left with a real d (inc 2, item()) on Elem2, Right with a T d on WorkGroup<64>; downshift on Elem.
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, DgmmAndDownshift) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int m = 33, n = 21, ld = m + 2, s = ld * n, dinc = 2, dstr = m * dinc + 3, nx = 17, xinc = 3, xs = nx * xinc + 1;
    auto ha = batched<T>(s, 41);
    auto hd = batched<R>(dstr, 42);
    auto he = batched<T>(n, 43);
    auto hx = batched<T>(xs, 44);
    T* a = this->dev.up(ha);
    T* a2 = this->dev.up(ha);
    R* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    T* x = this->dev.up(hx);
    T* y1 = this->dev.up(poisoned<T>(std::size_t(kBatch) * (nx + 1)));
    T* y2 = this->dev.up(poisoned<T>(std::size_t(kBatch) * (nx + 4)));
    const VectorView<R> dv(d, m, kBatch, Inc{dinc}, Stride{dstr});
    const VectorView<T> xv(x, nx, kBatch, Inc{xinc}, Stride{xs});
    const VectorView<T> yv1(y1, nx + 1, kBatch), yv2(y2, nx + 4, kBatch);
    this->dev.grid(m, n, [=](int b, int i, int j) {
        v2::dgmm<Side::Left>(v2::Elem2{i, j}, v2::item(dv, b, m), v2::strided(a, std::ptrdiff_t(b) * s, ld, m, n));
    });
    this->dev.template run<64>([=](sycl::nd_item<1> it, int b) {
        v2::dgmm<Side::Right>(v2::WorkGroup<64>{it}, v2::vec(e + std::ptrdiff_t(b) * n, n), v2::strided(a2, std::ptrdiff_t(b) * s, ld, m, n));
    });
    this->dev.line(nx + 4, [=](int b, int i) {
        if (i < nx + 1) v2::downshift(v2::Elem{i}, v2::item(xv, b, nx), v2::item(yv1, b, nx + 1));
        v2::downshift(v2::Elem{i}, v2::item(xv, b, nx), v2::item(yv2, b, nx + 4));
    });
    const auto ga = this->dev.down(a, ha.size()), ga2 = this->dev.down(a2, ha.size());
    const auto g1 = this->dev.down(y1, std::size_t(kBatch) * (nx + 1)), g2 = this->dev.down(y2, std::size_t(kBatch) * (nx + 4));
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                const std::size_t k = at(b, s, i, j, ld);
                const LD<T> aij = LD<T>(ha[k]);
                const long double di = hd[std::size_t(b) * dstr + i * dinc];
                const LD<T> ej = LD<T>(he[std::size_t(b) * n + j]);
                worst = std::max({worst, rel(ga[k], aij * di, std::abs(aij) * std::abs(di)),
                                  rel(ga2[k], aij * ej, std::abs(aij) * std::abs(ej))});
            }
        for (int i = 0; i < nx + 4; ++i) {
            const T ref = (i >= 1 && i <= nx) ? hx[std::size_t(b) * xs + (i - 1) * xinc] : T(0);
            if (i < nx + 1) bad += !same_bits(g1[std::size_t(b) * (nx + 1) + i], ref);
            bad += !same_bits(g2[std::size_t(b) * (nx + 4) + i], ref);
        }
    }
    EXPECT_VERIFY(T, Check::blas, 1, worst);
    EXPECT_EQ(bad, 0) << "downshift";
}

// map over reversed() and gathered() operands (finalize_values_only / finalize_hermitian_values).
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, MapOverReversedAndGathered) {
    using T = TypeParam;
    constexpr int n = 26, inc = 2, stride = n * inc + 5;
    auto hv = batched<T>(stride, 51);
    std::vector<std::int32_t> hp(std::size_t(kBatch) * n);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n; ++i) hp[std::size_t(b) * n + i] = (7 * i + b) % n;  // 7 is prime to 26: a permutation
    T* v = this->dev.up(hv);
    std::int32_t* p = this->dev.up(hp);
    T* o1 = this->dev.up(poisoned<T>(std::size_t(kBatch) * n));
    T* o2 = this->dev.up(poisoned<T>(std::size_t(kBatch) * n));
    const VectorView<T> vv(v, n, kBatch, Inc{inc}, Stride{stride});
    const VectorView<std::int32_t> pv(p, n, kBatch);
    this->dev.line(n, [=](int b, int i) {
        const std::size_t base = std::size_t(b) * std::size_t(n);
        v2::map(v2::Elem{i}, v2::strided(o1, base, n), [](T t) { return v2::Arith<T>::neg(t); }, v2::reversed(v2::item(vv, b, n)));
        v2::map(v2::Elem{i}, v2::strided(o2, base, n), [](T t) { return t; }, v2::gathered(v2::item(vv, b, n), v2::item(pv, b, n)));
    });
    const auto g1 = this->dev.down(o1, std::size_t(kBatch) * n), g2 = this->dev.down(o2, std::size_t(kBatch) * n);
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n; ++i) {
            const auto x = [&](int k) { return hv[std::size_t(b) * stride + k * inc]; };
            bad += !same_bits(g1[std::size_t(b) * n + i], T(-x(n - 1 - i))) + !same_bits(g2[std::size_t(b) * n + i], x(hp[std::size_t(b) * n + i]));
        }
    EXPECT_EQ(bad, 0);
}

// Real bidiagonal ops: bdgram Right / Left (zero padding) on Elem and WorkGroup<64>, bdmv_div on
// Thread (live and dead sigma), the sign scan with -0.0 / 0 / negatives, any_of straddling tol.
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, BidiagonalOps) {
    using T = TypeParam;
    if constexpr (is_cplx<T>) {
        GTEST_SKIP() << "the bidiagonal of gebrd is real; the real types cover these ops";
    } else {
        constexpr int k = 23, order = 31, ldv = k + 4, sv = ldv * k;
        auto hd = batched<T>(k, 61), he = batched<T>(k - 1, 62), hv = batched<T>(sv, 63);
        std::vector<T> hsig(kBatch);
        for (int b = 0; b < kBatch; ++b) hsig[b] = (b % 4 == 3) ? T(0) : T(0.5) + T(b % kPeriod);  // every 4th: dead
        T *d = this->dev.up(hd), *e = this->dev.up(he), *v = this->dev.up(hv), *sig = this->dev.up(hsig);
        T *tdr = this->dev.up(poisoned<T>(std::size_t(kBatch) * k)), *ter = this->dev.up(poisoned<T>(std::size_t(kBatch) * (k - 1)));
        T *tdl = this->dev.up(poisoned<T>(std::size_t(kBatch) * order)), *tel = this->dev.up(poisoned<T>(std::size_t(kBatch) * (order - 1)));
        T *tdw = this->dev.up(poisoned<T>(std::size_t(kBatch) * order)), *tew = this->dev.up(poisoned<T>(std::size_t(kBatch) * (order - 1)));
        T* u = this->dev.up(poisoned<T>(std::size_t(kBatch) * order));
        const VectorView<T> dv(d, k, kBatch), ev(e, k - 1, kBatch);
        const VectorView<T> tdrv(tdr, k, kBatch), terv(ter, k - 1, kBatch), tdlv(tdl, order, kBatch), telv(tel, order - 1, kBatch);
        this->dev.line(order, [=](int b, int i) {
            if (i < k) v2::bdgram<Side::Right>(v2::Elem{i}, v2::item(dv, b, k), v2::item(ev, b, k - 1), v2::item(tdrv, b, k), v2::item(terv, b, k - 1));
            v2::bdgram<Side::Left>(v2::Elem{i}, v2::item(dv, b, k), v2::item(ev, b, k - 1), v2::item(tdlv, b, order), v2::item(telv, b, order - 1));
        });
        this->dev.template run<64>([=](sycl::nd_item<1> it, int b) {
            const v2::WorkGroup<64> ex{it};
            v2::bdgram<Side::Left>(ex, v2::vec(d + b * k, k), v2::vec(e + b * (k - 1), k - 1), v2::vec(tdw + b * order, order),
                                   v2::vec(tew + b * (order - 1), order - 1));
            if (ex.leader()) {
                const auto x = v2::strided(v, std::ptrdiff_t(b) * sv, ldv, k, k);
                v2::bdmv_div(v2::Thread{}, v2::vec(d + b * k, k), v2::vec(e + b * (k - 1), k - 1),
                             v2::vec(&x(0, 5), k), sig[b], T(1e-3), v2::vec(u + b * order, order));
            }
        });
        const auto gtdr = this->dev.down(tdr, std::size_t(kBatch) * k), gter = this->dev.down(ter, std::size_t(kBatch) * (k - 1));
        const auto gtdl = this->dev.down(tdl, std::size_t(kBatch) * order), gtel = this->dev.down(tel, std::size_t(kBatch) * (order - 1));
        const auto gtdw = this->dev.down(tdw, std::size_t(kBatch) * order), gtew = this->dev.down(tew, std::size_t(kBatch) * (order - 1));
        const auto gu = this->dev.down(u, std::size_t(kBatch) * order);
        double worst = 0;
        int bad = 0;
        for (int b = 0; b < kBatch; ++b) {
            const auto D = [&](int i) { return (long double)hd[std::size_t(b) * k + i]; };
            const auto E = [&](int i) { return (long double)he[std::size_t(b) * (k - 1) + i]; };
            for (int i = 0; i < k; ++i) {
                const long double tr = D(i) * D(i) + (i > 0 ? E(i - 1) * E(i - 1) : 0.0L);
                worst = std::max(worst, rel(gtdr[std::size_t(b) * k + i], tr, tr));
                if (i < k - 1) worst = std::max(worst, rel(gter[std::size_t(b) * (k - 1) + i], D(i) * E(i), std::fabs(D(i) * E(i))));
            }
            for (int i = 0; i < order; ++i) {
                const long double tl = i < k ? D(i) * D(i) + (i < k - 1 ? E(i) * E(i) : 0.0L) : 0.0L;
                const long double ol = i < k - 1 ? E(i) * D(i + 1) : 0.0L;
                for (const auto* g : {&gtdl, &gtdw}) worst = std::max(worst, rel((*g)[std::size_t(b) * order + i], tl, tl));
                if (i < order - 1)
                    for (const auto* g : {&gtel, &gtew}) worst = std::max(worst, rel((*g)[std::size_t(b) * (order - 1) + i], ol, std::fabs(ol)));
                const bool live = hsig[b] > T(1e-3);
                long double ui = 0, us = 0;
                if (live && i < k) {
                    const auto X = [&](int r) { return (long double)hv[at(b, sv, r, 5, ldv)]; };
                    ui = D(i) * X(i) + (i < k - 1 ? E(i) * X(i + 1) : 0.0L);
                    us = std::fabs(D(i) * X(i)) + (i < k - 1 ? std::fabs(E(i) * X(i + 1)) : 0.0L);
                    ui /= hsig[b], us /= hsig[b];
                }
                const T g = gu[std::size_t(b) * order + i];
                if (live && i < k) worst = std::max(worst, rel(g, ui, us));
                else bad += !same_bits(g, T(0));
            }
        }
        EXPECT_VERIFY(T, Check::blas, 2, worst);
        EXPECT_EQ(bad, 0) << "bdmv_div must write zeros past k and for a dead sigma";
    }
}

TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, SignScanAndAnyOf) {
    using T = TypeParam;
    if constexpr (is_cplx<T>) {
        GTEST_SKIP() << "real-only ops";
    } else {
        constexpr int n = 19, k = 40;
        auto he = batched<T>(n - 1, 71);
        for (int b = 0; b < kBatch; ++b) {
            he[std::size_t(b) * (n - 1) + 3] = T(-0.0);
            he[std::size_t(b) * (n - 1) + 4] = T(0);
            he[std::size_t(b) * (n - 1) + 5] = -std::fabs(he[std::size_t(b) * (n - 1) + 5]) - T(0.25);
        }
        // Item b: s(0) is the tol scale; b % 3 == 0 puts s(k-1) exactly at tol (counts), == 1 one ulp above.
        std::vector<T> hs(std::size_t(kBatch) * k);
        const T s0 = T(4), tol = std::numeric_limits<T>::epsilon() * s0;
        for (int b = 0; b < kBatch; ++b) {
            for (int i = 0; i < k; ++i) hs[std::size_t(b) * k + i] = s0 / T(i + 1);
            const T last = (b % 3 == 0) ? tol : (b % 3 == 1 ? std::nextafter(tol, T(1)) : T(1));
            hs[std::size_t(b) * k + k - 1] = last;
        }
        T* e = this->dev.up(he);
        T* sg = this->dev.up(poisoned<T>(std::size_t(kBatch) * n));
        T* s = this->dev.up(hs);
        std::int32_t* f1 = this->dev.up(std::vector<std::int32_t>(kBatch, -1));
        std::int32_t* f2 = this->dev.up(std::vector<std::int32_t>(kBatch, -1));
        const VectorView<T> evw(e, n - 1, kBatch), sgv(sg, n, kBatch);
        this->dev.template run<256>([=](sycl::nd_item<1> it, int b) {
            const v2::WorkGroup<256> ex{it};
            const auto sv = v2::vec(static_cast<const T*>(s) + std::size_t(b) * k, k);
            const T t = std::numeric_limits<T>::epsilon() * sv(0);
            const int any_g = v2::any_of<v2::Scope::Group>(ex, k, [&](int i) { return sv(i) <= t; });
            if (ex.leader()) {
                f2[b] = any_g;
                f1[b] = v2::any_of(v2::Thread{}, k, [&](int i) { return sv(i) <= t; });
                v2::tridiag_abs_offdiag(v2::Thread{}, v2::item(evw, b, n - 1), v2::item(sgv, b, n));
            }
        });
        const auto ge = this->dev.down(e, he.size()), gs = this->dev.down(sg, std::size_t(kBatch) * n);
        const auto g1 = this->dev.down(f1, kBatch), g2 = this->dev.down(f2, kBatch);
        int bad = 0;
        for (int b = 0; b < kBatch; ++b) {
            const int want = (b % 3 == 0) ? 1 : 0;
            bad += (g1[b] != want) + (g2[b] != want);
            T sign = T(1);
            bad += !same_bits(gs[std::size_t(b) * n], T(1));
            for (int i = 0; i < n - 1; ++i) {
                const T e0 = he[std::size_t(b) * (n - 1) + i];
                sign *= (e0 >= T(0)) ? T(1) : T(-1);
                bad += !same_bits(gs[std::size_t(b) * n + i + 1], sign) + !same_bits(ge[std::size_t(b) * (n - 1) + i], std::fabs(e0));
                bad += gs[std::size_t(b) * n + i] * e0 * gs[std::size_t(b) * n + i + 1] != std::fabs(e0);
            }
        }
        EXPECT_EQ(bad, 0);
    }
}

static_assert(v2::Elem::size == 1 && v2::Elem2::size == 1);

}  // namespace

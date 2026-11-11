// SPDX-License-Identifier: MIT
// The gesvd / ormbr private device pieces (src/ops/gesvd/gesvd-ormbr_v2.hh) against host references:
// core's Grid1D<true> / ElemAt<true> / ElemIn executors driving core ops in the compositions the
// glue kernels use, the item_n / col_of / row_of / sub_of adaptors with core flip / mapped /
// shifted over them, core dgmm, and bdgram, bdmv_div, tridiag_abs_offdiag. Core ops themselves are
// covered by the device_core tests.
// Guards (agent-guide section 8): non-natural ld / inc / stride, views longer than the n an
// adaptor is given, rectangular shapes and blocks, complex data with nonzero imaginary parts,
// NaN-poisoned outputs where nothing may be written, all kBatch = 1024 items checked against their
// own host data, threshold straddles (dead sigma, -0.0 / 0 signs), a WG = 1024 launch.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <src/ops/gesvd/gesvd-ormbr_v2.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;

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
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GesvdOrmbrTest, Types);

std::size_t at(int b, int stride, int i, int j, int ld) { return std::size_t(b) * stride + i + std::size_t(j) * ld; }
std::ptrdiff_t off(int b, int stride) { return std::ptrdiff_t(b) * stride; }

// ElemAt<true> driving core laset / lacpy (real -> T widening) / lacpy<ConjTrans> on the swapped element
// (adjoint_into) / mirror<Lower, Herm> on the swapped element (mirror_lower_to_upper).
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, Elem2DrivesCoreCopies) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int m = 37, n = 29, lda = m + 3, sa = lda * n, ldc = n + 5, sc = ldc * m, nh = 31, ldh = nh + 2, sh = ldh * nh;
    const T alpha = val<T>(7);
    auto ha = batched<T>(sa, 11);
    auto hr = batched<R>(sa, 12);
    auto hh = batched<T>(sh, 13);
    T* a = this->dev.up(ha);
    R* ar = this->dev.up(hr);
    T* hp = this->dev.up(hh);
    T* lp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sa));
    T* wp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sa));
    T* cp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sc));
    const KernelMatrixView<T, MatrixFormat::Dense> ak(a, m, n, lda, sa, kBatch), ck(cp, n, m, ldc, sc, kBatch);
    const KernelMatrixView<R, MatrixFormat::Dense> rk(ar, m, n, lda, sa, kBatch);
    const KernelMatrixView<T, MatrixFormat::Dense> hk(hp, nh, nh, ldh, sh, kBatch);
    this->dev.grid(m, n, [=](int b, int i, int j) {
        device::laset(device::ElemAt<true>{i, j}, alpha, device::One{}, device::mat(lp + off(b, sa), m, n, lda));
        device::lacpy(device::ElemAt<true>{i, j}, device::item(rk, b, m, n), device::mat(wp + off(b, sa), m, n, lda));
        device::lacpy<device::Part::Full, Transpose::ConjTrans>(device::ElemAt<true>{j, i}, device::item(ak, b, m, n), device::item(ck, b, n, m));
    });
    this->dev.grid(nh, nh, [=](int b, int i, int j) { device::mirror<Uplo::Lower, true>(device::ElemAt<true>{j, i}, device::item(hk, b, nh, nh)); });
    const auto gl = this->dev.down(lp, std::size_t(kBatch) * sa), gw = this->dev.down(wp, std::size_t(kBatch) * sa);
    const auto gc = this->dev.down(cp, std::size_t(kBatch) * sc), gh = this->dev.down(hp, hh.size());
    int bad = 0, pad = 0, bad_h = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const std::size_t k = at(b, sa, i, j, lda), kt = at(b, sc, j, i, ldc);
                bad += !same_bits(gl[k], i == j ? T(1) : alpha) + !same_bits(gw[k], T(hr[k])) + !same_bits(gc[kt], conj_h(ha[k]));
            }
            for (int i = m; i < lda; ++i) pad += !is_nan(gl[at(b, sa, i, j, lda)]) + !is_nan(gw[at(b, sa, i, j, lda)]);
        }
        for (int i = 0; i < m; ++i)
            for (int j = n; j < ldc; ++j) pad += !is_nan(gc[at(b, sc, j, i, ldc)]);
        for (int j = 0; j < nh; ++j)
            for (int i = 0; i < nh; ++i) {
                const T h0 = hh[at(b, sh, i, j, ldh)];
                T want = i < j ? conj_h(hh[at(b, sh, j, i, ldh)]) : h0;
                if constexpr (is_cplx<T>)
                    if (i == j) want = T(h0.real(), R(0));
                bad_h += !same_bits(gh[at(b, sh, i, j, ldh)], want);
            }
    }
    EXPECT_EQ(bad, 0);
    EXPECT_EQ(pad, 0) << "ld padding was written";
    EXPECT_EQ(bad_h, 0) << "mirror<Lower, Herm> on the swapped ElemAt";
}

// The ormbr compositions: AQ := 0 then its strict lower (1:, 1:) := A(0:, 1:)^H on ElemIn over
// sub_of(item()); a non-square ElemIn block (its far column bound is not implied by the
// triangle); the P panel V := unit lower of A(i0:, i0+1:)^H on ElemAt<true> (laset<Upper> + lacpy).
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, BlockCompositionsWriteExactlyTheirPart) {
    using T = TypeParam;
    constexpr int N = 40, lda = N + 3, sa = lda * N, i0 = 2, ib = 8, mv = N - i0 - 1, ldv = mv + 2, sv = ldv * ib;
    constexpr int ldq = N + 1, sq = ldq * N, Mf = 24, Nf = 12, ldf = Mf + 3, sf = ldf * Nf, rm = 20, rn = 5;
    auto ha = batched<T>(sa, 21);
    T* a = this->dev.up(ha);
    T* vp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sv));
    T* qp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sq));
    T* fp = this->dev.up(poisoned<T>(std::size_t(kBatch) * sf));
    const KernelMatrixView<T, MatrixFormat::Dense> ak(a, N, N, lda, sa, kBatch), qk(qp, N, N, ldq, sq, kBatch);
    this->dev.grid(N, N, [=](int b, int row, int col) {
        const device::ElemAt<true> ex{row, col};
        const auto aq = device::item(qk, b, N, N);
        device::laset(ex, device::Zero{}, device::Zero{}, aq);
        device::lacpy<device::Part::StrictLower, Transpose::ConjTrans>(ex.sub(1, 1), device::gesvd_glue::sub_of(device::item(ak, b, N, N), 0, 1, N - 1, N - 1),
                                                               device::gesvd_glue::sub_of(aq, 1, 1, N - 1, N - 1));
    });
    this->dev.grid(Mf, Nf, [=](int b, int r, int c) {
        const auto src = device::mat(static_cast<const T*>(a) + off(b, sa), rm, rn, lda);
        const auto dst = device::gesvd_glue::sub_of(device::mat(fp + off(b, sf), Mf, Nf, ldf), 1, 2, rm, rn);
        device::lacpy<device::Part::StrictLower>(device::ElemAt<true>{r, c}.sub(1, 2), src, dst);
    });
    this->dev.grid(mv, ib, [=](int b, int r, int c) {
        const device::ElemAt<true> ex{r, c};
        const auto ap = device::mat(static_cast<const T*>(a) + off(b, sa), N, N, lda);
        const auto v = device::mat(vp + off(b, sv), mv, ib, ldv);
        device::laset<device::Part::Upper>(ex, device::Zero{}, device::One{}, v);
        device::lacpy<device::Part::StrictLower, Transpose::ConjTrans>(ex, ap.sub(i0, i0 + 1, ib, mv), v);
    });
    const auto gv = this->dev.down(vp, std::size_t(kBatch) * sv), gq = this->dev.down(qp, std::size_t(kBatch) * sq);
    const auto gf = this->dev.down(fp, std::size_t(kBatch) * sf);
    int bad_v = 0, bad_q = 0, bad_f = 0;
    for (int b = 0; b < kBatch; ++b) {
        const auto A = [&](int i, int j) { return ha[at(b, sa, i, j, lda)]; };
        for (int c = 0; c < ib; ++c)
            for (int r = 0; r < mv; ++r)
                bad_v += !same_bits(gv[at(b, sv, r, c, ldv)], r == c ? T(1) : (r > c ? conj_h(A(i0 + c, i0 + 1 + r)) : T(0)));
        for (int c = 0; c < N; ++c)
            for (int r = 0; r < N; ++r) bad_q += !same_bits(gq[at(b, sq, r, c, ldq)], (c >= 1 && r > c) ? conj_h(A(c - 1, r)) : T(0));
        for (int c = 0; c < Nf; ++c)
            for (int r = 0; r < Mf; ++r) {
                const int i = r - 1, j = c - 2;
                const bool in = j >= 0 && j < rn && i > j && i < rm;
                const T g = gf[at(b, sf, r, c, ldf)];
                bad_f += in ? !same_bits(g, A(i, j)) : !is_nan(g);
            }
    }
    EXPECT_EQ(bad_v, 0) << "P panel: laset<Upper> + lacpy<StrictLower, ConjTrans> on ElemAt";
    EXPECT_EQ(bad_q, 0) << "AQ: laset + lacpy<StrictLower, ConjTrans> on ElemIn over sub_of";
    EXPECT_EQ(bad_f, 0) << "lacpy<StrictLower> on a non-square ElemIn block";
}

// map on Grid1D<true> over flip() / mapped() of item_n views longer than n (finalize_*_values);
// Thread copy / map through col_of / row_of, and a zero column through laset on sub_of
// (build_hermitian_vectors, build_bidiag_vectors, patch_zero_left_vectors).
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, AdaptorsAddressTheirElement) {
    using T = TypeParam;
    constexpr int n = 26, inc = 2, vn = n + 3, stride = vn * inc + 5, N = 21, ld = N + 3, s = ld * N;
    auto hv = batched<T>(stride, 51);
    auto hq = batched<T>(s, 52);
    std::vector<std::int32_t> hp(std::size_t(kBatch) * (n + 2));
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n + 2; ++i) hp[std::size_t(b) * (n + 2) + i] = (7 * i + b) % n;  // 7 is prime to 26
    T* v = this->dev.up(hv);
    T* q = this->dev.up(hq);
    std::int32_t* p = this->dev.up(hp);
    T* o1 = this->dev.up(poisoned<T>(std::size_t(kBatch) * n));
    T* o2 = this->dev.up(poisoned<T>(std::size_t(kBatch) * n));
    T* u = this->dev.up(poisoned<T>(std::size_t(kBatch) * s));
    T* w = this->dev.up(poisoned<T>(std::size_t(kBatch) * s));
    const VectorView<T> vv(v, vn, kBatch, Inc{inc}, Stride{stride});
    const VectorView<std::int32_t> pv(p, n + 2, kBatch);
    const KernelMatrixView<T, MatrixFormat::Dense> qk(q, N, N, ld, s, kBatch), uk(u, N, N, ld, s, kBatch), wk(w, N, N, ld, s, kBatch);
    this->dev.line(n, [=](int b, int i) {
        const std::ptrdiff_t base = std::ptrdiff_t(b) * n;
        device::map(device::Grid1D<true>{i}, device::vec(o1 + base, n), [](T t) { return device::Arith<T>::neg(t); }, device::flip(device::gesvd_glue::item_n(vv, b, n)));
        device::map(device::Grid1D<true>{i}, device::vec(o2 + base, n), [](T t) { return t; },
                device::mapped(device::gesvd_glue::item_n(pv, b, n), [x = device::gesvd_glue::item_n(vv, b, n)](int k) { return x(k); }));
    });
    this->dev.line(N, [=](int b, int tgt) {
        const int src = (N - 1) - tgt;
        const auto qc = device::gesvd_glue::col_of(device::item(qk, b, N, N), src);
        if (tgt % 3 == 2) device::laset(device::Thread{}, device::Zero{}, device::Zero{}, device::gesvd_glue::sub_of(device::item(uk, b, N, N), 0, tgt, N, 1));
        else device::copy(device::Thread{}, qc, device::gesvd_glue::col_of(device::item(uk, b, N, N), tgt));
        device::map(device::Thread{}, device::gesvd_glue::row_of(device::item(wk, b, N, N), tgt), [](T x) { return device::Arith<T>::conj(x); }, qc);
    });
    const auto g1 = this->dev.down(o1, std::size_t(kBatch) * n), g2 = this->dev.down(o2, std::size_t(kBatch) * n);
    const auto gu = this->dev.down(u, std::size_t(kBatch) * s), gw = this->dev.down(w, std::size_t(kBatch) * s);
    int bad = 0, bad_m = 0, pad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const auto x = [&](int k) { return hv[std::size_t(b) * stride + k * inc]; };
        for (int i = 0; i < n; ++i) {
            bad += !same_bits(g1[std::size_t(b) * n + i], T(-x(n - 1 - i)));
            bad += !same_bits(g2[std::size_t(b) * n + i], x(hp[std::size_t(b) * (n + 2) + i]));
        }
        for (int tgt = 0; tgt < N; ++tgt) {
            const int src = (N - 1) - tgt;
            for (int r = 0; r < N; ++r) {
                const T qv = hq[at(b, s, r, src, ld)];
                bad_m += !same_bits(gu[at(b, s, r, tgt, ld)], tgt % 3 == 2 ? T(0) : qv);
                bad_m += !same_bits(gw[at(b, s, tgt, r, ld)], conj_h(qv));
            }
            for (int r = N; r < ld; ++r) pad += !is_nan(gu[at(b, s, r, tgt, ld)]) + !is_nan(gw[at(b, s, r, tgt, ld)]);
        }
    }
    EXPECT_EQ(bad, 0) << "map over reversed / gathered item_n";
    EXPECT_EQ(bad_m, 0) << "col_of / row_of / sub_of";
    EXPECT_EQ(pad, 0) << "ld padding was written";
}

// dgmm Left with a real d (inc 2, item_n) on ElemAt<true> over item(), Right with a T d on a WG = 1024
// launch (the ceiling); copy of shifted(x, 1) on Grid1D<true> with x shorter than y and x's view
// longer than nx.
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, DgmmAndDownshift) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int m = 33, n = 21, ld = m + 2, s = ld * n, dinc = 2, dstr = m * dinc + 3, nx = 17, xinc = 3, xs = (nx + 2) * xinc + 1;
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
    const VectorView<T> xv(x, nx + 2, kBatch, Inc{xinc}, Stride{xs});
    const VectorView<T> yv1(y1, nx + 1, kBatch), yv2(y2, nx + 4, kBatch);
    const KernelMatrixView<T, MatrixFormat::Dense> ak(a, m, n, ld, s, kBatch);
    this->dev.grid(m, n, [=](int b, int i, int j) {
        device::dgmm<Side::Left>(device::ElemAt<true>{i, j}, device::gesvd_glue::item_n(dv, b, m), device::item(ak, b, m, n));
    });
    this->dev.template run<1024>([=](sycl::nd_item<1> it, int b) {
        device::dgmm<Side::Right>(device::WorkGroup<1024>{it}, device::vec(e + off(b, n), n), device::mat(a2 + off(b, s), m, n, ld));
    });
    this->dev.line(nx + 4, [=](int b, int i) {
        if (i < nx + 1) device::copy(device::Grid1D<true>{i}, device::shifted(device::gesvd_glue::item_n(xv, b, nx), 1, nx + 1), device::gesvd_glue::item_n(yv1, b, nx + 1));
        device::copy(device::Grid1D<true>{i}, device::shifted(device::gesvd_glue::item_n(xv, b, nx), 1, nx + 4), device::gesvd_glue::item_n(yv2, b, nx + 4));
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
    EXPECT_EQ(bad, 0) << "copy(shifted(x, 1))";
}

// Real bidiagonal ops: bdgram Right / Left (zero padding) on Grid1D<true> and WorkGroup<64>, bdmv_div on
// Thread through col_of (live and dead sigma: every 4th item straddles tol).
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, BidiagonalOps) {
    using T = TypeParam;
    if constexpr (is_cplx<T>) {
        GTEST_SKIP() << "the bidiagonal of gebrd is real; the real types cover these ops";
    } else {
        constexpr int k = 23, order = 31, ldv = k + 4, sv = ldv * k;
        auto hd = batched<T>(k, 61), he = batched<T>(k - 1, 62), hv = batched<T>(sv, 63);
        std::vector<T> hsig(kBatch);
        for (int b = 0; b < kBatch; ++b) hsig[b] = (b % 4 == 3) ? T(1e-3) : T(0.5) + T(b % kPeriod);  // every 4th: at tol, dead
        T *d = this->dev.up(hd), *e = this->dev.up(he), *v = this->dev.up(hv), *sig = this->dev.up(hsig);
        T *tdr = this->dev.up(poisoned<T>(std::size_t(kBatch) * k)), *ter = this->dev.up(poisoned<T>(std::size_t(kBatch) * (k - 1)));
        T *tdl = this->dev.up(poisoned<T>(std::size_t(kBatch) * order)), *tel = this->dev.up(poisoned<T>(std::size_t(kBatch) * (order - 1)));
        T *tdw = this->dev.up(poisoned<T>(std::size_t(kBatch) * order)), *tew = this->dev.up(poisoned<T>(std::size_t(kBatch) * (order - 1)));
        T* u = this->dev.up(poisoned<T>(std::size_t(kBatch) * order));
        const VectorView<T> dv(d, k, kBatch), ev(e, k - 1, kBatch);
        const VectorView<T> tdrv(tdr, k, kBatch), terv(ter, k - 1, kBatch), tdlv(tdl, order, kBatch), telv(tel, order - 1, kBatch);
        const KernelMatrixView<T, MatrixFormat::Dense> vk(v, k, k, ldv, sv, kBatch);
        this->dev.line(order, [=](int b, int i) {
            if (i < k)
                device::detail::bdgram<Side::Right>(device::Grid1D<true>{i}, device::gesvd_glue::item_n(dv, b, k), device::gesvd_glue::item_n(ev, b, k - 1), device::gesvd_glue::item_n(tdrv, b, k),
                                          device::gesvd_glue::item_n(terv, b, k - 1));
            device::detail::bdgram<Side::Left>(device::Grid1D<true>{i}, device::gesvd_glue::item_n(dv, b, k), device::gesvd_glue::item_n(ev, b, k - 1), device::gesvd_glue::item_n(tdlv, b, order),
                                     device::gesvd_glue::item_n(telv, b, order - 1));
        });
        this->dev.template run<64>([=](sycl::nd_item<1> it, int b) {
            const device::WorkGroup<64> ex{it};
            device::detail::bdgram<Side::Left>(ex, device::vec(d + b * k, k), device::vec(e + b * (k - 1), k - 1), device::vec(tdw + b * order, order),
                                     device::vec(tew + b * (order - 1), order - 1));
            if (ex.leader())
                device::detail::bdmv_div(device::Thread{}, device::vec(d + b * k, k), device::vec(e + b * (k - 1), k - 1), device::gesvd_glue::col_of(device::item(vk, b, k, k), 5),
                               sig[b], T(1e-3), device::vec(u + b * order, order));
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
        EXPECT_EQ(bad, 0) << "bdmv_div must write zeros past k and for a sigma at tol";
    }
}

// The sign scan with -0.0 / 0 / negatives; s's view is two longer than n (item_n bounds the scan).
TYPED_TEST(DeviceBlasV2GesvdOrmbrTest, SignScan) {
    using T = TypeParam;
    if constexpr (is_cplx<T>) {
        GTEST_SKIP() << "real-only op";
    } else {
        constexpr int n = 19;
        auto he = batched<T>(n - 1, 71);
        for (int b = 0; b < kBatch; ++b) {
            he[std::size_t(b) * (n - 1) + 3] = T(-0.0);
            he[std::size_t(b) * (n - 1) + 4] = T(0);
            he[std::size_t(b) * (n - 1) + 5] = -std::fabs(he[std::size_t(b) * (n - 1) + 5]) - T(0.25);
        }
        T* e = this->dev.up(he);
        T* sg = this->dev.up(poisoned<T>(std::size_t(kBatch) * (n + 2)));
        const VectorView<T> evw(e, n - 1, kBatch), sgv(sg, n + 2, kBatch);
        this->dev.line(1, [=](int b, int) { device::detail::tridiag_abs_offdiag(device::Thread{}, device::gesvd_glue::item_n(evw, b, n - 1), device::gesvd_glue::item_n(sgv, b, n)); });
        const auto ge = this->dev.down(e, he.size()), gs = this->dev.down(sg, std::size_t(kBatch) * (n + 2));
        int bad = 0;
        for (int b = 0; b < kBatch; ++b) {
            const auto S = [&](int i) { return gs[std::size_t(b) * (n + 2) + i]; };
            T sign = T(1);
            bad += !same_bits(S(0), T(1)) + !is_nan(S(n)) + !is_nan(S(n + 1));
            for (int i = 0; i < n - 1; ++i) {
                const T e0 = he[std::size_t(b) * (n - 1) + i];
                sign *= (e0 >= T(0)) ? T(1) : T(-1);
                bad += !same_bits(S(i + 1), sign) + !same_bits(ge[std::size_t(b) * (n - 1) + i], std::fabs(e0));
                bad += S(i) * e0 * S(i + 1) != std::fabs(e0);
            }
        }
        EXPECT_EQ(bad, 0);
    }
}

}  // namespace

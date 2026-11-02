// SPDX-License-Identifier: MIT
// Device LA v2 krylov/ortho/ritz extension (include/batchlas/blas/device/v2/ext/krylov-ortho-ritz.hh)
// against host references. Guards (docs/developer/agent-guide.md section 8): run-time widths
// that are not powers of two, below and above n, and a launch at the device ceiling; non-natural
// ld and inc; complex data with nonzero imaginary parts and ConjTrans; padding poisoned with a
// large finite value that the ops must not touch; batch 1024 with period-3 data where every item
// must be bit-identical to its representative (the SLM publish path).
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

#include <batchlas/blas/device/v2/ext/krylov-ortho-ritz.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kBatch = 1024;  // saturating batch: items repeat with period kPeriod
constexpr int kPeriod = 3;
constexpr double kPoison = 1.0e6;  // in range, finite: an op that writes padding changes it

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

// kPeriod representatives copied to every item b from representative b % kPeriod.
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(static_cast<std::size_t>(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> T poison() { return T(static_cast<real_of<T>>(kPoison)); }

// Items whose results differ from their representative's.
template <class T> int bit_twins(const std::vector<T>& h, std::size_t per_item) {
    int bad = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (std::size_t i = 0; i < per_item; ++i) bad += !same_bits(h[b * per_item + i], h[(b % kPeriod) * per_item + i]);
    return bad;
}

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    int max_wg() const { return static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)); }
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
    // A run-time local range and no required sub-group size: the launch shape of the ported kernels.
    template <class F> void run(int batch, int wg, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * wg, std::size_t(wg)),
                            [=](sycl::nd_item<1> it) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2KrylovTest : public ::testing::Test {
protected:
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2KrylovTest, Types);

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

// ---- DynGroup reductions ----------------------------------------------------------------------

enum RedSlot { kDotU, kDotc, kSsqF, kSsqU, kNrm2U, kSlots };

// Widths 37 and 100 (not powers of two, below n: the stride loop runs 2-4 times) and the device
// ceiling (above n: idle items contribute zero). Three launches so the ceiling launch fits.
TYPED_TEST(DeviceBlasV2KrylovTest, DynGroupReductions) {
    using T = TypeParam;
    constexpr int n = 133, inc = 3;
    const auto hx = batched<T>(n * inc, 11), hy = batched<T>(n, 12);
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    for (int wg : {37, 100, std::min(1024, this->dev.max_wg())}) {
        std::vector<T> ho(std::size_t(kBatch) * kSlots, T(std::numeric_limits<real_of<T>>::quiet_NaN()));
        T* o = this->dev.up(ho);
        auto launch = [&](auto part_c) {
            constexpr int P = decltype(part_c)::value;
            this->dev.run(kBatch, wg, 0, [=](sycl::nd_item<1> it, unsigned char*) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const v2::DynGroup ex{it};
                const auto xv = v2::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
                const auto yv = v2::vec(y + std::ptrdiff_t(b) * n, n);
                T* ob = o + std::ptrdiff_t(b) * kSlots;
                if constexpr (P == 0) {
                    const T r0 = v2::dot_unfused(ex, xv, yv), r1 = v2::dotc(ex, xv, yv);
                    if (ex.leader()) ob[kDotU] = r0, ob[kDotc] = r1;
                } else if constexpr (P == 1) {
                    const T r0 = T(v2::sumsq(ex, xv)), r1 = T(v2::sumsq<v2::alg::Unfused>(ex, xv));
                    if (ex.leader()) ob[kSsqF] = r0, ob[kSsqU] = r1;
                } else {
                    const T r0 = T(v2::nrm2<v2::alg::Unscaled>(ex, xv));
                    if (ex.leader()) ob[kNrm2U] = r0;
                }
            });
        };
        launch(std::integral_constant<int, 0>{});
        launch(std::integral_constant<int, 1>{});
        launch(std::integral_constant<int, 2>{});
        const auto got = this->dev.down(o, ho.size());
        EXPECT_EQ(bit_twins(got, kSlots), 0) << "wg " << wg << ": items differ from their representative";
        for (int b = 0; b < kPeriod; ++b) {
            const T* r = got.data() + std::size_t(b) * kSlots;
            LD<T> dot = 0, dotc = 0;
            long double sabs = 0, ssq = 0;
            for (int i = 0; i < n; ++i) {
                const LD<T> xi = wide(hx[std::size_t(b) * n * inc + std::size_t(i) * inc]), yi = wide(hy[std::size_t(b) * n + i]);
                dot += xi * yi, dotc += conj_h(xi) * yi;
                sabs += std::abs(xi) * std::abs(yi), ssq += std::norm(xi);
            }
            const double e = std::max({rel(r[kDotU], dot, sabs), rel(r[kDotc], dotc, sabs), rel(r[kSsqF], LD<T>(ssq), ssq),
                                       rel(r[kSsqU], LD<T>(ssq), ssq), rel(r[kNrm2U], LD<T>(std::sqrt(ssq)), std::sqrt(ssq))});
            EXPECT_VERIFY(T, Check::blas, n, e) << "wg " << wg << " item " << b << " (dot_unfused, dotc, sumsq x2, nrm2)";
        }
    }
}

// ---- DynGroup elementwise ops ------------------------------------------------------------------

// generate / axpy2 / scal_div / add_scalar on a width-100 executor (n = 133), then lacpy in the
// three transposes from an ld = m + 3 source into poisoned destinations with padded ld.
TYPED_TEST(DeviceBlasV2KrylovTest, DynGroupElementwise) {
    using T = TypeParam;
    using R = real_of<T>;
    using Ar = v2::Arith<T>;
    constexpr int wg = 100, n = 133, inc = 3, nout = 4;
    constexpr int m = 7, c = 19, lds = m + 3, ldn = m + 2, ldt = c + 4;
    static_assert(m * c == n);
    const T a = val<T>(91), bb = val<T>(92), s = val<T>(93);
    const R d = R(-1.7);
    const auto hx = batched<T>(n * inc, 21), hy = batched<T>(n, 22), hw = batched<T>(n, 23), hs = batched<T>(lds * c, 24);
    std::vector<T> ho(std::size_t(kBatch) * nout * n, T(std::numeric_limits<R>::quiet_NaN()));
    std::vector<T> hn(std::size_t(kBatch) * ldn * c, poison<T>()), ht(std::size_t(kBatch) * ldt * m, poison<T>()),
        hc(std::size_t(kBatch) * ldt * m, poison<T>());
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    T* w = this->dev.up(hw);
    T* src = this->dev.up(hs);
    T* o = this->dev.up(ho);
    T* on = this->dev.up(hn);
    T* ot = this->dev.up(ht);
    T* oc = this->dev.up(hc);
    auto gen = [](int i) { return Ar::make(R(i) * R(0.25) - R(3), is_cplx<T> ? R(1) - R(i) * R(0.5) : R(0)); };
    this->dev.run(kBatch, wg, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::DynGroup ex{it};
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * n, n);
        const auto wv = v2::vec(w + std::ptrdiff_t(b) * n, n);
        const auto om = v2::mat(o + std::ptrdiff_t(b) * nout * n, n, nout, n);
        v2::generate(ex, om.col(0), gen);
        v2::copy(ex, yv, om.col(1));
        v2::axpy2(ex, a, xv, bb, wv, om.col(1));  // y + (a x + b w)
        v2::scal_div(ex, d, xv, om.col(2));       // x / d
        v2::copy(ex, yv, om.col(3));
        v2::add_scalar(ex, s, om.col(3));         // y + s
        const auto A = v2::mat(src + std::ptrdiff_t(b) * lds * c, m, c, lds);
        // NoTrans and ConjTrans through the run-time-op form ortho's StridedCopyKernel uses.
        const bool ct = b >= 0;  // true, but not a constant the op can fold
        v2::lacpy(ex, ct ? Transpose::NoTrans : Transpose::ConjTrans, A, v2::mat(on + std::ptrdiff_t(b) * ldn * c, m, c, ldn));
        v2::lacpy<Transpose::Trans>(ex, A, v2::mat(ot + std::ptrdiff_t(b) * ldt * m, c, m, ldt));
        v2::lacpy(ex, ct ? Transpose::ConjTrans : Transpose::NoTrans, A, v2::mat(oc + std::ptrdiff_t(b) * ldt * m, c, m, ldt));
    });
    const auto go = this->dev.down(o, ho.size()), gn = this->dev.down(on, hn.size()), gt = this->dev.down(ot, ht.size()),
               gc = this->dev.down(oc, hc.size());
    EXPECT_EQ(bit_twins(go, std::size_t(nout) * n), 0) << "items differ from their representative";
    double worst = 0;
    int inexact = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* g = go.data() + std::size_t(b) * nout * n;
        for (int i = 0; i < n; ++i) {
            const T xi = hx[std::size_t(b) * n * inc + std::size_t(i) * inc], yi = hy[std::size_t(b) * n + i], wi = hw[std::size_t(b) * n + i];
            inexact += !same_bits(g[i], gen(i));
            inexact += !same_bits(g[3 * n + i], T(yi + s));  // one rounding per component
            const long double sc = std::abs(wide(yi)) + std::abs(wide(a)) * std::abs(wide(xi)) + std::abs(wide(bb)) * std::abs(wide(wi));
            worst = std::max({worst, rel(g[n + i], wide(yi) + (wide(a) * wide(xi) + wide(bb) * wide(wi)), sc),
                              rel(g[2 * n + i], wide(xi) / LD<T>(d), std::abs(wide(xi)) / std::abs(d))});
        }
    }
    EXPECT_EQ(inexact, 0) << "generate and add_scalar are exact";
    EXPECT_VERIFY(T, Check::blas, 3, worst) << "axpy2, scal_div";
    // lacpy: every copied element exact, every padding element still the poison.
    int wrong = 0, touched = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* A = hs.data() + std::size_t(b) * lds * c;
        const T* N = gn.data() + std::size_t(b) * ldn * c;
        const T* Tt = gt.data() + std::size_t(b) * ldt * m;
        const T* Ct = gc.data() + std::size_t(b) * ldt * m;
        for (int j = 0; j < c; ++j)
            for (int i = 0; i < ldn; ++i) {
                if (i < m) wrong += !same_bits(N[i + j * ldn], A[i + j * lds]);
                else touched += !same_bits(N[i + j * ldn], poison<T>());
            }
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < ldt; ++i) {
                if (i < c) {
                    wrong += !same_bits(Tt[i + j * ldt], A[j + i * lds]);
                    wrong += !same_bits(Ct[i + j * ldt], conj_h(A[j + i * lds]));
                } else {
                    touched += !same_bits(Tt[i + j * ldt], poison<T>()) + !same_bits(Ct[i + j * ldt], poison<T>());
                }
            }
    }
    EXPECT_EQ(wrong, 0) << "lacpy NoTrans / Trans / ConjTrans";
    EXPECT_EQ(touched, 0) << "lacpy wrote into the destination padding";
}

// ---- ItemGroup: diagonal ops and laqge with a published scale ---------------------------------

// k = 37 with ld = k + 2; WG = k, ItemGroup's contract (one index per item, no guard).
// trace via reduce over diag, add_scalar on diag, laqge(D, generated c) and laqge(D, D) with D
// staged in SLM and published. The scalings are exact in T, so they are compared bit for bit.
TYPED_TEST(DeviceBlasV2KrylovTest, ItemGroupDiagonalAndLaqge) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int k = 37, ld = k + 2, per = ld * k;
    const T s = val<T>(77);
    const auto hC = batched<T>(per, 31), hE = batched<T>(per, 32), hF = batched<T>(per, 33);
    auto hd = batched<R>(k, 34);
    for (auto& v : hd) v = R(0.5) + std::abs(v);  // positive scales, as rsqrt(diag) gives
    auto cgen = [](int j) { return R(1) + R(j) * R(0.125); };
    for (int wg : {k}) {
        // Padding rows i in [k, ld) carry the poison; the ops must leave them alone.
        auto pad = [&](std::vector<T> h) {
            for (int b = 0; b < kBatch; ++b)
                for (int j = 0; j < k; ++j)
                    for (int i = k; i < ld; ++i) h[std::size_t(b) * per + i + j * ld] = poison<T>();
            return h;
        };
        const auto hCp = pad(hC), hEp = pad(hE), hFp = pad(hF);
        T* C = this->dev.up(hCp);
        T* E = this->dev.up(hEp);
        T* F = this->dev.up(hFp);
        R* dg = this->dev.up(hd);
        std::vector<R> htr(kBatch, std::numeric_limits<R>::quiet_NaN());
        R* tr = this->dev.up(htr);
        this->dev.run(kBatch, wg, sizeof(R) * k, [=](sycl::nd_item<1> it, unsigned char* lm) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const v2::ItemGroup ex{it};
            const auto Cm = v2::mat(C + std::ptrdiff_t(b) * per, k, k, ld);
            const R t = v2::reduce(ex, k, [&](int i) { return v2::Arith<T>::re(v2::diag(Cm)(i)); });
            if (ex.leader()) tr[b] = t;
            v2::add_scalar(ex, s, v2::diag(Cm));
            const auto d_slm = v2::vec<v2::Mem::Local>(reinterpret_cast<R*>(lm), k);
            v2::copy(ex, v2::vec(dg + std::ptrdiff_t(b) * k, k), d_slm);
            const auto D = ex.publish(d_slm);
            v2::laqge(ex, D, v2::generated(k, cgen), v2::mat(E + std::ptrdiff_t(b) * per, k, k, ld));
            v2::laqge(ex, D, D, v2::mat(F + std::ptrdiff_t(b) * per, k, k, ld));
        });
        const auto gC = this->dev.down(C, hCp.size()), gE = this->dev.down(E, hEp.size()), gF = this->dev.down(F, hFp.size());
        const auto gtr = this->dev.down(tr, htr.size());
        EXPECT_EQ(bit_twins(gC, per) + bit_twins(gE, per) + bit_twins(gF, per) + bit_twins(gtr, 1), 0)
            << "wg " << wg << ": items differ from their representative";
        auto scaled = [](T a, R sc) {
            if constexpr (is_cplx<T>) return T(a.real() * sc, a.imag() * sc);
            else return T(a * sc);
        };
        int wrong = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const std::size_t o = std::size_t(b) * per;
            const R* db = hd.data() + std::size_t(b) * k;
            long double ref = 0, sc = 0;
            for (int i = 0; i < k; ++i) ref += std::real(wide(hC[o + i + i * ld])), sc += std::abs(std::real(wide(hC[o + i + i * ld])));
            EXPECT_VERIFY(T, Check::blas, k, rel(T(gtr[b]), LD<T>(ref), sc)) << "wg " << wg << " trace";
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < ld; ++i) {
                    const std::size_t e = o + i + j * ld;
                    if (i >= k) {
                        wrong += !same_bits(gC[e], poison<T>()) + !same_bits(gE[e], poison<T>()) + !same_bits(gF[e], poison<T>());
                        continue;
                    }
                    wrong += !same_bits(gC[e], i == j ? T(hC[e] + s) : hC[e]);
                    wrong += !same_bits(gE[e], scaled(hE[e], db[i] * cgen(j)));
                    wrong += !same_bits(gF[e], scaled(hF[e], db[i] * db[j]));
                }
        }
        EXPECT_EQ(wrong, 0) << "wg " << wg << ": add_scalar on diag / laqge (exact) or padding touched";
    }
}

}  // namespace

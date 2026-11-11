// SPDX-License-Identifier: MIT
// The band-reduction routine header's own ops (include/batchlas/device/lapack/sb2st.hh) against
// long-double host references; the core ops the kernels also use are covered by device_blas_v2_tests.
// Guards (docs/developer/agent-guide.md section 8): band storage with ld > kd + 1 whose padding
// rows hold a sentinel that must survive, windows that cross the band edge (reads 0, writes
// dropped), strided vectors (inc 2, 3, 5, 7), non-natural ld, complex data with a non-real
// diagonal, beta == Zero over a NaN-poisoned output, batch 1024 with period-3 data where every
// item must be bit-identical to its representative (all SLM paths), a WG = 1024 launch, both
// butterfly orders, rotation operands through fn_vec index maps (write-through references).
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

#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/lapack/sb2st.hh>
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

// kPeriod representatives (edited by fix(item pointer, item)) copied to every item b from b % kPeriod.
template <class T, class Fix> std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix) {
    std::vector<T> h(static_cast<std::size_t>(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item, b);
    }
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    return batched<T>(per_item, seed, [](T*, int) {});
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> int bitdiff_vs_rep(const std::vector<T>& got, int per_item) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i)
            d += !same_bits(got[std::size_t(b) * per_item + i], got[std::size_t(b % kPeriod) * per_item + i]);
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
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2BandTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2BandTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class V> V cj(V v) {
    if constexpr (std::is_arithmetic_v<V>) return v;
    else return std::conj(v);
}
template <class V> V re_only(V v) {
    if constexpr (std::is_arithmetic_v<V>) return v;
    else return V(v.real(), 0);
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}
template <class T> T nan_t() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// Host view of lower band storage s (ld, kd): the same mirror / zero rule as device::sb2st_detail::HermBandLower.
template <class T> struct HostBand {
    const T* s;
    int ld, kd;
    LD<T> get(int i, int j) const {
        if (i >= j) return (i - j <= kd) ? wide(s[(i - j) + std::size_t(j) * ld]) : LD<T>(0);
        return (j - i <= kd) ? cj(wide(s[(j - i) + std::size_t(i) * ld])) : LD<T>(0);
    }
    // Storage index of (i, j) when it is inside the band, else -1.
    long idx(int i, int j) const {
        if (i >= j) return (i - j <= kd) ? long((i - j) + std::size_t(j) * ld) : -1;
        return (j - i <= kd) ? long((j - i) + std::size_t(i) * ld) : -1;
    }
};
// Band storage with padding rows kd + 1 .. ld - 1 set to a sentinel that no op may touch.
template <class T> std::vector<T> make_band(int ld, int kd, int n, std::uint64_t seed) {
    return batched<T>(ld * n, seed, [=](T* p, int) {
        for (int j = 0; j < n; ++j)
            for (int r = kd + 1; r < ld; ++r) p[r + std::size_t(j) * ld] = T(7);
    });
}

// ---- hemv / her2 on a band window ---------------------------------------------------------------

// p = W x (beta == Zero over NaN), then W -= p x^H + x p^H on the lower triangle, for the window
// W = A[a : a + m, a : a + m] of a Hermitian band matrix; m > kd + 1 so the window crosses the band.
template <class T, int WG, class O> void run_band_hemv_her2(Dev& dev) {
    constexpr int n = 40, kd = 6, ld = kd + 4, a = 3, m = 11;
    auto hs = make_band<T>(ld, kd, n, 11);
    auto hx = batched<T>(m, 12);
    std::vector<T> hp(std::size_t(kBatch) * m, nan_t<T>());
    T* s = dev.up(hs);
    T* x = dev.up(hx);
    T* p = dev.up(hp);
    dev.template run<WG>(kBatch, 2 * m * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<WG> ex{it};
        const device::Teams<WG, 8, O> rows{it};
        T* vl = reinterpret_cast<T*>(lm);
        T* wl = vl + m;
        const device::sb2st_detail::HermBandLower<T> A{s + std::ptrdiff_t(b) * ld * n, ld, kd};
        device::for_each_owned(ex, m, [&](int i) {
            vl[i] = x[std::ptrdiff_t(b) * m + i];
            wl[i] = T(std::numeric_limits<real_of<T>>::quiet_NaN());
        });
        device::sync(ex, device::Mem::Global);
        const auto W = A.sub(a, a, m, m);
        const auto v = device::vec<device::Mem::Local>(vl, m);
        const auto w = device::vec<device::Mem::Local>(wl, m);
        device::sb2st_detail::hemv_band<Uplo::Lower>(rows, device::One{}, W, device::assume_published(v), device::Zero{}, w);
        device::sync(ex, device::Mem::Global);
        device::for_each_owned(ex, m, [&](int i) { p[std::ptrdiff_t(b) * m + i] = wl[i]; });
        device::her2<device::Part::Lower, device::alg::TeamPerRow, device::arith::Native>(rows, device::NegOne{}, device::assume_published(w),
                                                                          device::assume_published(v), W);
    });
    const auto gs = dev.down(s, hs.size());
    const auto gp = dev.down(p, hp.size());
    EXPECT_EQ(bitdiff_vs_rep(gs, ld * n), 0) << "band: items differ from their representative";
    EXPECT_EQ(bitdiff_vs_rep(gp, m), 0) << "p: items differ from their representative";
    double worst = 0;
    int untouched_bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const HostBand<T> A0{hs.data() + std::size_t(b) * ld * n, ld, kd};
        const T* xb = hx.data() + std::size_t(b) * m;
        std::vector<LD<T>> pr(m);
        std::vector<long double> ps(m);
        for (int r = 0; r < m; ++r) {
            LD<T> acc = 0;
            long double sc = 0;
            for (int j = 0; j < m; ++j) acc += A0.get(a + r, a + j) * wide(xb[j]), sc += std::abs(A0.get(a + r, a + j)) * std::abs(xb[j]);
            pr[r] = acc, ps[r] = sc;
            worst = std::max(worst, rel(gp[std::size_t(b) * m + r], acc, sc));
        }
        std::vector<char> hit(std::size_t(ld) * n, 0);
        for (int r = 0; r < m; ++r)
            for (int j = 0; j <= r; ++j) {
                const long k = A0.idx(a + r, a + j);
                if (k < 0) continue;
                hit[k] = 1;
                LD<T> e = A0.get(a + r, a + j) - pr[r] * cj(wide(xb[j])) - wide(xb[r]) * cj(pr[j]);
                if (r == j) e = re_only(e);
                const long double sc = std::abs(A0.get(a + r, a + j)) + ps[r] * std::abs(xb[j]) + std::abs(xb[r]) * ps[j];
                worst = std::max(worst, rel(gs[std::size_t(b) * ld * n + k], e, sc));
            }
        for (std::size_t k = 0; k < hit.size(); ++k)
            if (!hit[k]) untouched_bad += !same_bits(gs[std::size_t(b) * ld * n + k], hs[std::size_t(b) * ld * n + k]);
    }
    EXPECT_EQ(untouched_bad, 0) << "entries outside the window / band / padding changed";
    EXPECT_VERIFY(T, Check::blas, 2 * m, worst);
}

TYPED_TEST(DeviceBlasV2BandTest, HemvHer2OnBandWindowTeamsUp) { run_band_hemv_her2<TypeParam, 256, device::bfly::Asc>(this->dev); }
TYPED_TEST(DeviceBlasV2BandTest, HemvHer2OnBandWindowTeamsDown) { run_band_hemv_her2<TypeParam, 256, device::bfly::Down>(this->dev); }
TYPED_TEST(DeviceBlasV2BandTest, HemvHer2OnBandWindowCeilingLaunch) { run_band_hemv_her2<TypeParam, 1024, device::bfly::Asc>(this->dev); }

// ---- larf on a band, an SLM row-major tile and a global matrix ----------------------------------

// Host larf on an accessor pair: Left C := C - tau v (v^H C) (fold = ScaleV / ScaleW irrelevant in
// exact arithmetic), Right C := C - tau (C v) v^H. get(i, j) / put(i, j, value) / mag(i, j).
template <class T, class Get, class Put> void ref_larf(bool left, int mr, int nc, const std::vector<LD<T>>& v, LD<T> tau, Get get,
                                                       Put put, std::vector<long double>& scale) {
    const int nout = left ? nc : mr, len = left ? mr : nc;
    std::vector<LD<T>> res(std::size_t(mr) * nc);
    scale.assign(std::size_t(mr) * nc, 0);
    for (int o = 0; o < nout; ++o) {
        LD<T> w = 0;
        long double sw = 0;
        for (int k = 0; k < len; ++k) {
            const LD<T> c = left ? get(k, o) : get(o, k);
            w += left ? cj(v[k]) * c : c * v[k];
            sw += std::abs(v[k]) * std::abs(c);
        }
        for (int k = 0; k < len; ++k) {
            const int i = left ? k : o, j = left ? o : k;
            const LD<T> nv = left ? get(i, j) - tau * v[k] * w : get(i, j) - tau * w * cj(v[k]);
            res[std::size_t(i) * nc + j] = nv;
            scale[std::size_t(i) * nc + j] = std::abs(get(i, j)) + std::abs(tau) * std::abs(v[k]) * sw;
        }
    }
    for (int i = 0; i < mr; ++i)
        for (int j = 0; j < nc; ++j) put(i, j, res[std::size_t(i) * nc + j]);
}

// Right then Left on two windows of a band (Teams<256, 8, Asc>, ScaleW, v and tau complex); the
// windows are strictly below the diagonal and cross the band edge.
TYPED_TEST(DeviceBlasV2BandTest, LarfOnBandWindows) {
    using T = TypeParam;
    constexpr int n = 40, kd = 6, ld = kd + 3;
    constexpr int r0 = 14, c0 = 8, mr = 7, nc = 5;      // right: rows 14..20, cols 8..12
    constexpr int r1 = 22, c1 = 17, mr1 = 6, nc1 = 4;   // left: rows 22..27, cols 17..20
    auto hs = make_band<T>(ld, kd, n, 21);
    auto hv = batched<T>(nc + mr1, 22);
    const T tau = val<T>(23), tau1 = val<T>(24);
    T* s = this->dev.up(hs);
    T* vg = this->dev.up(hv);
    this->dev.template run<256>(kBatch, (nc + mr1) * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<256> ex{it};
        const device::Teams<256, 8, device::bfly::Asc> rows{it};
        T* vl = reinterpret_cast<T*>(lm);
        device::copy(ex, device::vec(vg + std::ptrdiff_t(b) * (nc + mr1), nc + mr1), device::vec(vl, nc + mr1));
        device::sync(ex, device::Mem::Global);
        const device::sb2st_detail::HermBandLower<T> A{s + std::ptrdiff_t(b) * ld * n, ld, kd};
        device::sb2st_detail::larf_right(rows, tau, device::assume_published(device::vec<device::Mem::Local>(vl, nc)), A.sub(r0, c0, mr, nc));
        device::sync(ex, device::Mem::Global);
        device::larf<Side::Left, device::arith::Native, device::larf_form::ScaleW, device::loop::Lockstep>(
            rows, tau1, device::assume_published(device::vec<device::Mem::Local>(vl, nc + mr1).tail(nc)), A.sub(r1, c1, mr1, nc1));
    });
    const auto gs = this->dev.down(s, hs.size());
    EXPECT_EQ(bitdiff_vs_rep(gs, ld * n), 0) << "items differ from their representative";
    double worst = 0;
    int untouched_bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<T> ref_store(hs.begin() + std::ptrdiff_t(b) * ld * n, hs.begin() + std::ptrdiff_t(b + 1) * ld * n);
        std::vector<LD<T>> exact(ref_store.size());
        for (std::size_t k = 0; k < exact.size(); ++k) exact[k] = wide(ref_store[k]);
        std::vector<long double> sc_all(ref_store.size(), -1);
        auto apply = [&](bool left, int i0, int j0, int m_, int n_, const T* v, T t) {
            std::vector<LD<T>> vv(left ? m_ : n_);
            for (std::size_t k = 0; k < vv.size(); ++k) vv[k] = wide(v[k]);
            HostBand<T> hb{nullptr, ld, kd};
            auto get = [&](int i, int j) -> LD<T> {
                const long k = hb.idx(i0 + i, j0 + j);
                return k < 0 ? LD<T>(0) : exact[k];
            };
            std::vector<long double> sc;
            std::vector<std::pair<long, LD<T>>> writes;
            ref_larf<T>(left, m_, n_, vv, wide(t), get, [&](int i, int j, LD<T> x) {
                const long k = hb.idx(i0 + i, j0 + j);
                if (k >= 0) writes.push_back({k, x});
            }, sc);
            for (auto& [k, x] : writes) exact[k] = x;
            for (int i = 0; i < m_; ++i)
                for (int j = 0; j < n_; ++j) {
                    const long k = hb.idx(i0 + i, j0 + j);
                    if (k >= 0) sc_all[k] = std::max(sc_all[k], sc[std::size_t(i) * n_ + j]);
                }
        };
        const T* vb = hv.data() + std::size_t(b) * (nc + mr1);
        apply(false, r0, c0, mr, nc, vb, tau);
        apply(true, r1, c1, mr1, nc1, vb + nc, tau1);
        for (std::size_t k = 0; k < exact.size(); ++k) {
            const T g = gs[std::size_t(b) * ld * n + k];
            if (sc_all[k] < 0) untouched_bad += !same_bits(g, ref_store[k]);
            else worst = std::max(worst, rel(g, exact[k], sc_all[k]));
        }
    }
    EXPECT_EQ(untouched_bad, 0) << "entries outside the windows / band / padding changed";
    EXPECT_VERIFY(T, Check::blas, 2 * nc, worst);
}

// Z := (I - tau v v^H) Z with ScaleV on: (1) a resident row-major SLM tile, StridedTeams<4> in one
// sub-group, rows [s, s + L), L > G lanes per column; (2) four sub-groups of StridedTeams<2, 128>,
// one reflector each on disjoint rows (the wave layout); (3) WorkGroup<32> on a global matrix with
// ld = rows + 5 and L > 32 (the streaming layout). V has ld = L + 3; tau is complex.
TYPED_TEST(DeviceBlasV2BandTest, LarfColumnTeamsOnTilesAndGlobal) {
    using T = TypeParam;
    constexpr int rows = 50, C = 4, s = 7, L = 37, ldv = L + 3;           // (1)
    constexpr int C2 = 2, L2 = 9, S2 = 4;                                 // (2) rows g*10 .. g*10+8
    constexpr int ncol3 = 3, ldz = rows + 5, s3 = 2, L3 = 45;             // (3)
    auto hz = batched<T>(rows * C, 31);
    auto hz2 = batched<T>(rows * C2, 32);
    auto hz3 = batched<T>(ldz * ncol3, 33);
    auto hv = batched<T>(ldv * S2, 34);
    auto hv3 = batched<T>(L3, 35);
    const T tau = val<T>(36);
    T* z = this->dev.up(hz);
    T* z2 = this->dev.up(hz2);
    T* z3 = this->dev.up(hz3);
    T* vg = this->dev.up(hv);
    T* v3 = this->dev.up(hv3);
    this->dev.template run<32>(kBatch, rows * C * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<32> ex{it};
        const device::sb2st_detail::StridedTeams<C> cols{it};
        T* zl = reinterpret_cast<T*>(lm);
        T* zb = z + std::ptrdiff_t(b) * rows * C;
        device::for_each_owned(ex, rows * C, [&](int i) { zl[i] = zb[i]; });
        device::sync(cols);
        const auto V = device::mat(vg + std::ptrdiff_t(b) * ldv * S2, L, S2, ldv);
        device::larf<Side::Left, device::arith::Native, device::larf_form::ScaleV, device::loop::Lockstep>(
            cols, tau, device::assume_published(V.col(1)), device::StridedMat<T, device::Ext<C>, device::Unit>{zl + s * C, L, C});
        device::sync(cols);
        device::for_each_owned(ex, rows * C, [&](int i) { zb[i] = zl[i]; });
        // (3) on global memory, the same work-group
        const auto Z3 = device::mat(z3 + std::ptrdiff_t(b) * ldz * ncol3, rows, ncol3, ldz);
        device::larf<Side::Left, device::arith::Native, device::larf_form::ScaleV, device::loop::Lockstep>(
            ex, tau, device::assume_published(device::vec(v3 + std::ptrdiff_t(b) * L3, L3)), Z3.sub(s3, 0, L3, ncol3));
    });
    this->dev.template run<128>(kBatch, rows * C2 * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<128> ex{it};
        const device::sb2st_detail::StridedTeams<C2, 128> cols{it};
        const int g = static_cast<int>(it.get_local_linear_id()) / 32;
        T* zl = reinterpret_cast<T*>(lm);
        T* zb = z2 + std::ptrdiff_t(b) * rows * C2;
        device::for_each_owned(ex, rows * C2, [&](int i) { zl[i] = zb[i]; });
        device::sync(ex, device::Mem::Global);
        const auto V = device::mat(vg + std::ptrdiff_t(b) * ldv * S2, L2, S2, ldv);
        device::larf<Side::Left, device::arith::Native, device::larf_form::ScaleV, device::loop::Lockstep>(
            cols, tau, device::assume_published(V.col(g)), device::StridedMat<T, device::Ext<C2>, device::Unit>{zl + g * 10 * C2, L2, C2});
        device::sync(ex, device::Mem::Global);
        device::for_each_owned(ex, rows * C2, [&](int i) { zb[i] = zl[i]; });
    });
    const auto gz = this->dev.down(z, hz.size());
    const auto gz2 = this->dev.down(z2, hz2.size());
    const auto gz3 = this->dev.down(z3, hz3.size());
    EXPECT_EQ(bitdiff_vs_rep(gz, rows * C) + bitdiff_vs_rep(gz2, rows * C2) + bitdiff_vs_rep(gz3, ldz * ncol3), 0)
        << "items differ from their representative";
    double worst = 0;
    int untouched_bad = 0;
    // Row-major (rs = ncols, cs = 1) or column-major (rs = 1, cs = ld) storage, rows [r0, r0 + len).
    auto check = [&](const std::vector<T>& h, const std::vector<T>& g, std::size_t per, std::size_t base, int rs, int cs, int r0,
                     int len, int ncols, const T* v) {
        std::vector<LD<T>> vv(len);
        for (int k = 0; k < len; ++k) vv[k] = wide(v[k]);
        std::vector<LD<T>> out(std::size_t(len) * ncols);
        std::vector<long double> sc;
        auto at = [&](int i, int j) { return base + std::size_t(r0 + i) * rs + std::size_t(j) * cs; };
        ref_larf<T>(true, len, ncols, vv, wide(tau), [&](int i, int j) { return wide(h[at(i, j)]); },
                    [&](int i, int j, LD<T> x) { out[std::size_t(i) * ncols + j] = x; }, sc);
        std::vector<char> hit(per, 0);
        for (int i = 0; i < len; ++i)
            for (int j = 0; j < ncols; ++j) {
                hit[at(i, j) - base] = 1;
                worst = std::max(worst, rel(g[at(i, j)], out[std::size_t(i) * ncols + j], sc[std::size_t(i) * ncols + j]));
            }
        return hit;
    };
    for (int b = 0; b < kPeriod; ++b) {
        const T* vb = hv.data() + std::size_t(b) * ldv * S2;
        auto hit = check(hz, gz, rows * C, std::size_t(b) * rows * C, C, 1, s, L, C, vb + ldv);
        for (int k = 0; k < rows * C; ++k)
            if (!hit[k]) untouched_bad += !same_bits(gz[std::size_t(b) * rows * C + k], hz[std::size_t(b) * rows * C + k]);
        std::vector<char> hit2(rows * C2, 0);
        for (int g = 0; g < S2; ++g) {
            auto h2 = check(hz2, gz2, rows * C2, std::size_t(b) * rows * C2, C2, 1, g * 10, L2, C2, vb + g * ldv);
            for (int k = 0; k < rows * C2; ++k) hit2[k] |= h2[k];
        }
        for (int k = 0; k < rows * C2; ++k)
            if (!hit2[k]) untouched_bad += !same_bits(gz2[std::size_t(b) * rows * C2 + k], hz2[std::size_t(b) * rows * C2 + k]);
        auto hit3 = check(hz3, gz3, ldz * ncol3, std::size_t(b) * ldz * ncol3, 1, ldz, s3, L3, ncol3, hv3.data() + std::size_t(b) * L3);
        for (int k = 0; k < ldz * ncol3; ++k)
            if (!hit3[k]) untouched_bad += !same_bits(gz3[std::size_t(b) * ldz * ncol3 + k], hz3[std::size_t(b) * ldz * ncol3 + k]);
    }
    EXPECT_EQ(untouched_bad, 0) << "rows outside the reflector's span changed";
    EXPECT_VERIFY(T, Check::blas, L3, worst);
}

// ---- plane rotations -------------------------------------------------------------------------------

// A deterministic stand-in for the caller's rotation kernels: the test checks the distribution and
// the operand strides, and the reference evaluates the same formulas in long double.
template <class T> struct TestRot {
    using R = real_of<T>;
    static void lartg(const T& f, const T& g, R& c, T& s, T& r) {
        c = R(0.6);
        s = g * R(0.8);
        r = f + g;
    }
    static void rot_pair(T& x, T& y, R c, const T& s) {
        const T xt = x, yt = y;
        x = T(c) * xt + s * yt;
        y = T(c) * yt - device::arith_t<device::arith::Native, T>::conj(s) * xt;
    }
    static void lar2v_pair(T& x, T& y, T& z, R c, const T& s) {
        const T xo = x, yo = y, zo = z;
        x = T(c) * xo + s * zo;
        y = T(c) * yo - device::arith_t<device::arith::Native, T>::conj(s) * zo;
        z = zo * T(c) + xo - yo;
    }
};

// largv, lartv, lar2v over nrot = 150 rotations (more than the 64 lanes) with operand strides
// 7, 5, 3, 2, 1, and rot_by with one (c, s) on a WorkGroup and, through fn_vec, on a Thread.
TYPED_TEST(DeviceBlasV2BandTest, PlaneRotationsDistributeOverStrides) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 64, nr = 150, ix = 7, iy = 5, ic = 3, is = 2, nrot = 77;
    constexpr int per = nr * ix;  // per-item length of every strided array (the widest stride)
    auto hx = batched<T>(per, 61), hy = batched<T>(per, 62), hz = batched<T>(per, 63), hs = batched<T>(per, 64);
    auto hc = batched<R>(per, 65);
    T* x = this->dev.up(hx);
    T* y = this->dev.up(hy);
    T* z = this->dev.up(hz);
    T* sv = this->dev.up(hs);
    R* c = this->dev.up(hc);
    T* x2 = this->dev.up(hx);
    T* y2 = this->dev.up(hy);
    T* x3 = this->dev.up(hx);
    T* y3 = this->dev.up(hy);
    T* x4 = this->dev.up(hx);
    T* y4 = this->dev.up(hy);
    T* z4 = this->dev.up(hz);
    const T s1 = val<T>(66);
    const R c1 = R(0.28);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const std::ptrdiff_t o = std::ptrdiff_t(it.get_group_linear_id()) * per;
        const device::WorkGroup<WG> ex{it};
        // largv: x (stride 7) <- r, y (stride 5) <- s, c (stride 3) <- c
        device::largv<TestRot<T>>(ex, device::vec(x + o, nr, ix), device::vec(y + o, nr, iy), device::vec(c + o, nr, ic));
        device::sync(ex, device::Mem::Global);
        // lartv with the c just written and an independent s (stride 2) on fresh x2 / y2
        device::lartv<TestRot<T>>(ex, device::vec(x2 + o, nr, ix), device::vec(y2 + o, nr, iy), device::vec(c + o, nr, ic), device::vec(sv + o, nr, is));
        device::lar2v<TestRot<T>>(ex, device::vec(x4 + o, nr, ix), device::vec(y4 + o, nr, iy), device::vec(z4 + o, nr), device::vec(c + o, nr, ic),
                              device::vec(sv + o, nr, is));
        device::sb2st_detail::rot_by<TestRot<T>>(ex, device::vec(x3 + o, nrot, ix), device::vec(y3 + o, nrot, iy), c1, s1);
        // the Thread pass reaches x3 / y3 through fn_vec index maps returning references
        const auto xs = device::sb2st_detail::fn_vec(nrot, [=](int i) -> T& { return x3[o + 1 + std::ptrdiff_t(i) * ix]; });
        const auto ys = device::sb2st_detail::fn_vec(nrot, [=](int i) -> T& { return y3[o + 1 + std::ptrdiff_t(i) * iy]; });
        if (ex.leader()) device::sb2st_detail::rot_by<TestRot<T>>(device::Thread{}, xs, ys, c1, s1);
    });
    const auto gx = this->dev.down(x, hx.size()), gy = this->dev.down(y, hy.size()), gx2 = this->dev.down(x2, hx.size()),
               gy2 = this->dev.down(y2, hy.size()), gx3 = this->dev.down(x3, hx.size()), gy3 = this->dev.down(y3, hy.size()),
               gx4 = this->dev.down(x4, hx.size()), gy4 = this->dev.down(y4, hy.size()), gz4 = this->dev.down(z4, hz.size());
    const auto gc = this->dev.down(c, hc.size());
    double worst = 0;
    int untouched_bad = 0, bitdiff = 0;
    for (const auto* gv : {&gx, &gy, &gx2, &gy2, &gx3, &gy3, &gx4, &gy4, &gz4}) bitdiff += bitdiff_vs_rep(*gv, per);
    bitdiff += bitdiff_vs_rep(gc, per);
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    for (int b = 0; b < kPeriod; ++b) {
        const std::size_t o = std::size_t(b) * per;
        auto W = [&](const std::vector<T>& h, std::size_t k) { return wide(h[o + k]); };
        std::vector<char> tx(per, 0), ty(per, 0), tc(per, 0), tz(per, 0), tx3(per, 0), ty3(per, 0);
        for (int t = 0; t < nr; ++t) {
            const std::size_t kx = std::size_t(t) * ix, ky = std::size_t(t) * iy, kc = std::size_t(t) * ic, ks = std::size_t(t) * is;
            tx[kx] = ty[ky] = tc[kc] = tz[t] = 1;
            const LD<T> f = W(hx, kx), gg = W(hy, ky);
            const long double sc = std::abs(f) + std::abs(gg);
            worst = std::max({worst, rel(gx[o + kx], f + gg, sc), rel(gy[o + ky], gg * (long double)0.8, sc)});
            worst = std::max(worst, rel(T(gc[o + kc]), LD<T>(R(0.6)), 1.0L));
            const LD<T> cc = LD<T>((long double)gc[o + kc]), ss = W(hs, ks);
            const LD<T> xx = W(hx, kx), yy = W(hy, ky), zz = W(hz, t);
            const long double s2 = std::abs(xx) + std::abs(yy) + std::abs(zz);
            worst = std::max({worst, rel(gx2[o + kx], cc * xx + ss * yy, s2), rel(gy2[o + ky], cc * yy - cj(ss) * xx, s2)});
            worst = std::max({worst, rel(gx4[o + kx], cc * xx + ss * zz, s2), rel(gy4[o + ky], cc * yy - cj(ss) * zz, s2),
                              rel(gz4[o + t], zz * cc + xx - yy, s2)});
        }
        // rot: the WorkGroup pass on elements i * stride, the Thread pass on 1 + i * stride
        for (int i = 0; i < nrot; ++i)
            for (int sh = 0; sh < 2; ++sh) {
                const std::size_t kx = std::size_t(i) * ix + sh, ky = std::size_t(i) * iy + sh;
                tx3[kx] = ty3[ky] = 1;
                const LD<T> xx = W(hx, kx), yy = W(hy, ky), cc = LD<T>((long double)c1), ss = wide(s1);
                const long double s2 = std::abs(xx) + std::abs(yy);
                worst = std::max({worst, rel(gx3[o + kx], cc * xx + ss * yy, s2), rel(gy3[o + ky], cc * yy - cj(ss) * xx, s2)});
            }
        for (int k = 0; k < per; ++k) {
            untouched_bad += !tx[k] && !(same_bits(gx[o + k], hx[o + k]) && same_bits(gx2[o + k], hx[o + k]) && same_bits(gx4[o + k], hx[o + k]));
            untouched_bad += !ty[k] && !(same_bits(gy[o + k], hy[o + k]) && same_bits(gy2[o + k], hy[o + k]) && same_bits(gy4[o + k], hy[o + k]));
            untouched_bad += !tc[k] && !same_bits(gc[o + k], hc[o + k]);
            untouched_bad += !tz[k] && !same_bits(gz4[o + k], hz[o + k]);
            untouched_bad += !tx3[k] && !same_bits(gx3[o + k], hx[o + k]);
            untouched_bad += !ty3[k] && !same_bits(gy3[o + k], hy[o + k]);
        }
    }
    EXPECT_EQ(untouched_bad, 0) << "elements between the strides changed";
    EXPECT_VERIFY(T, Check::blas, 4, worst);
}

}  // namespace

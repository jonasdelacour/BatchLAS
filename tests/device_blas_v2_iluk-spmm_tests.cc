// SPDX-License-Identifier: MIT
// The ILU(k) / SpMM private v2 ops (src/kernels/common/dla_v2_sparse.hh) against host references,
// each in its kernel's shape; core ops are covered by device_blas_v2_tests. Guards (agent guide
// section 8): CSR item strides above nnz, ld above the extent, offset bases, slot-major storage,
// complex data with nonzero imaginary parts, NoTrans / Trans / ConjTrans as a runtime argument,
// beta == 0 over NaN-poisoned C, large finite poison past a row bound, out-of-range scatter
// indices beside guard cells, width and bound straddled, a bit-exact replay of the fma chain on
// the sycl_device::Cx type the SpMM kernels use, batch 1024 with period-3 twins, WG 1024.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <batchlas/device.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <src/kernels/common/dla_v2_sparse.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;
template <class T> using dev_of = typename sycl_device::DevMap<T>::type;

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
template <class T> T poison_v() { return T(real_of<T>(1e30)); }

// kPeriod representatives, copied to every item b from representative b % kPeriod.
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

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
    template <int WG, class F> void run(std::size_t groups, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2SparseTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2SparseTest, Types);

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
template <class T> dev_of<T> to_dev(T v) {
    dev_of<T> d;
    std::memcpy(&d, &v, sizeof(T));
    return d;
}

// Host replay of arith::Fma, operation for operation (std::fma is correctly rounded like sycl::fma).
template <class T> void h_mac(T& acc, T a, T b) {
    if constexpr (is_cplx<T>) {
        auto re = std::fma(a.real(), b.real(), acc.real());
        re = std::fma(-a.imag(), b.imag(), re);
        auto im = std::fma(a.real(), b.imag(), acc.imag());
        im = std::fma(a.imag(), b.real(), im);
        acc = T(re, im);
    } else {
        acc = std::fma(a, b, acc);
    }
}
template <class T> T h_mul(T a, T b) {
    if constexpr (is_cplx<T>) return T(std::fma(a.real(), b.real(), -a.imag() * b.imag()), std::fma(a.real(), b.imag(), a.imag() * b.real()));
    else return a * b;
}

// Shared CSR pattern: row r keeps column j < cols when mix(r * 977 + j) % 3 == 0; row `empty` keeps none.
struct Csr {
    std::vector<int> ro, ci;
    int nnz = 0;
};
Csr pattern(int rows, int cols, int empty) {
    Csr a;
    a.ro.push_back(0);
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < cols && r != empty; ++j)
            if (mix(std::uint64_t(r) * 977u + j) % 3 == 0) a.ci.push_back(j);
        a.ro.push_back(static_cast<int>(a.ci.size()));
    }
    a.nnz = static_cast<int>(a.ci.size());
    return a;
}
std::vector<int> replicate(const std::vector<int>& v, int stride, int pad_value) {
    std::vector<int> h(std::size_t(kBatch) * stride, pad_value);
    for (int b = 0; b < kBatch; ++b) std::copy(v.begin(), v.end(), h.begin() + std::size_t(b) * stride);
    return h;
}

// ---- SpmmGather: usmm_row + store_cols over SpRow(LoadVec, BasedVec<int64>), RtOpMat, ColMat64 ----
template <class T, bool Pair> void check_usmm(Dev& dev, Transpose trans, bool beta_zero) {
    using D = dev_of<T>;
    using Ar = v2::arith_t<v2::arith::Fma, D>;
    constexpr int NC = 4;
    const int m = 13, k = 11, width = 7, cblocks = (width + NC - 1) / NC;
    const Csr A = pattern(m, k, 5);
    const int ms = A.nnz + 3, os = m + 3;
    const bool nt = trans == Transpose::NoTrans, cj = trans == Transpose::ConjTrans;
    const int brows = nt ? k : width, bcols = nt ? width : k, ldb = brows + 3, sb = ldb * bcols + 1;
    const int ldc = m + 5, sc = ldc * width + 2;
    const auto hv = batched<T>(ms, 11), hb = batched<T>(sb, 12);
    const auto hc = beta_zero ? std::vector<T>(std::size_t(kBatch) * sc, nan_v<T>()) : batched<T>(sc, 13);
    const T alpha = val<T>(14), beta = beta_zero ? T(0) : val<T>(15);
    const D alpha_d = to_dev(alpha), beta_d = to_dev(beta);
    const D* v = reinterpret_cast<const D*>(dev.up(hv));
    const int* ci = dev.up(replicate(A.ci, ms, 0));
    const int* ro = dev.up(replicate(A.ro, os, 0));
    const D* bp = reinterpret_cast<const D*>(dev.up(hb));
    T* cp_t = dev.up(hc);
    D* cp = reinterpret_cast<D*>(cp_t);
    const std::int64_t total = std::int64_t(kBatch) * m * cblocks;
    dev.template run<64>(std::size_t((total + 63) / 64), [=](sycl::nd_item<1> it) {
        const std::int64_t gid = static_cast<std::int64_t>(it.get_global_linear_id());
        if (gid >= total) return;
        const std::int64_t b = gid / (m * cblocks);
        const int rem = static_cast<int>(gid - b * m * cblocks);
        const int i = rem % m, c0 = (rem / m) * NC;
        const std::int64_t vb = b * ms;
        const auto arow = v2::sp_row(v2::LoadVec<const D, Pair>{v + vb}, v2::BasedVec<const int, std::int64_t>{ci, vb},
                                     ro[b * os + i], ro[b * os + i + 1]);
        const v2::RtOpMat<const D, Pair> opB{bp + b * sb, ldb, nt, cj};
        const v2::ColMat64<D, Pair> C{cp + b * sc, ldc};
        v2::Frag<D, NC> acc;
        v2::fill(acc, D{});
        v2::usmm_row<Ar>(v2::Thread{}, arow, opB, c0, width, acc);
        v2::store_cols<Ar>(v2::Thread{}, alpha_d, acc, beta_d, beta_zero, C, i, c0, width);
    });
    const auto got = dev.down(cp_t, hc.size());
    auto opb = [&](int b, int j, int c) {
        const T x = nt ? hb[std::size_t(b) * sb + std::size_t(c) * ldb + j] : hb[std::size_t(b) * sb + std::size_t(j) * ldb + c];
        return cj ? conj_h(x) : x;
    };
    double worst = 0;
    int bitdiff = 0, replay = 0, untouched = 0, maxlen = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int e = 0; e < sc; ++e) {
            const std::size_t at = std::size_t(b) * sc + e;
            const int c = e / ldc, i = e % ldc;
            if (c < width && i < m) bitdiff += !same_bits(got[at], got[std::size_t(b % kPeriod) * sc + e]);
            else untouched += !same_bits(got[at], hc[at]);
        }
        if (b >= kPeriod) continue;
        for (int i = 0; i < m; ++i) {
            maxlen = std::max(maxlen, A.ro[i + 1] - A.ro[i]);
            for (int c = 0; c < width; ++c) {
                T acc = T(0);
                LD<T> ref = 0;
                long double s = 0;
                for (int p = A.ro[i]; p < A.ro[i + 1]; ++p) {
                    const T av = hv[std::size_t(b) * ms + p], bv = opb(b, A.ci[p], c);
                    h_mac(acc, av, bv);
                    ref += wide(av) * wide(bv);
                    s += std::abs(wide(av)) * std::abs(wide(bv));
                }
                const T cold = hc[std::size_t(b) * sc + std::size_t(c) * ldc + i];
                T out = T(0);
                h_mac(out, alpha, acc);
                if (!beta_zero) h_mac(out, beta, cold);
                const T g = got[std::size_t(b) * sc + std::size_t(c) * ldc + i];
                replay += !same_bits(g, out);
                const LD<T> r = wide(alpha) * ref + (beta_zero ? LD<T>(0) : wide(beta) * wide(cold));
                const long double sr = std::abs(wide(alpha)) * s + (beta_zero ? 0.0L : std::abs(wide(beta)) * std::abs(wide(cold)));
                worst = std::max(worst, rel(g, r, sr));
            }
        }
    }
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_EQ(replay, 0) << "not the fma chain of sycl_device::fma_acc";
    EXPECT_EQ(untouched, 0) << "wrote outside C(0:m, 0:width)";
    EXPECT_VERIFY(T, Check::blas, maxlen + 2, worst);
}

TYPED_TEST(DeviceBlasV2SparseTest, UsmmRowStoreColsReplaysFmaChain) {
    using T = TypeParam;
    for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        for (bool bz : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "trans " << int(tr) << " beta_zero " << bz);
            check_usmm<T, false>(this->dev, tr, bz);
            check_usmm<T, true>(this->dev, tr, bz);
        }
    }
}

// ---- SpmmScatter: load_cols_scaled, then the kernel's row loop with atomic_accumulate ----------
template <class T> void check_scatter(Dev& dev, Transpose transB, bool conj_a) {
    using D = dev_of<T>;
    using Ar = v2::arith_t<v2::arith::Fma, D>;
    constexpr int NCS = 4, G = 4;
    const int rows_in = 9, rows_out = 12, width = 6, cblocks = (width + NCS - 1) / NCS;
    const int ldc = rows_out + 3, sc = G + ldc * width + G;
    // Each stored row also holds column rows_out - 1 (in range), rows_out (even rows) and -1 (rows
    // divisible by 3): the last two must be skipped, the -1 one would land in the guard before C.
    Csr A = pattern(rows_in, rows_out - 1, 4);
    {
        Csr x;
        x.ro.push_back(0);
        for (int r = 0; r < rows_in; ++r) {
            for (int p = A.ro[r]; p < A.ro[r + 1]; ++p) x.ci.push_back(A.ci[p]);
            x.ci.push_back(rows_out - 1);
            if (r % 2 == 0) x.ci.push_back(rows_out);
            if (r % 3 == 0) x.ci.push_back(-1);
            x.ro.push_back(static_cast<int>(x.ci.size()));
        }
        x.nnz = static_cast<int>(x.ci.size());
        A = x;
    }
    const int ms = A.nnz + 2, os = rows_in + 2;
    const bool nt = transB == Transpose::NoTrans, cjb = transB == Transpose::ConjTrans;
    const int brows = nt ? rows_in : width, bcols = nt ? width : rows_in, ldb = brows + 2, sb = ldb * bcols + 3;
    const auto hv = batched<T>(ms, 21), hb = batched<T>(sb, 22), hc = batched<T>(sc, 23);
    const T alpha = val<T>(24);
    const D alpha_d = to_dev(alpha);
    const D* v = reinterpret_cast<const D*>(dev.up(hv));
    const int* ci = dev.up(replicate(A.ci, ms, 0));
    const int* ro = dev.up(replicate(A.ro, os, 0));
    const D* bp = reinterpret_cast<const D*>(dev.up(hb));
    T* cp_t = dev.up(hc);
    D* cp = reinterpret_cast<D*>(cp_t);
    const int wn = cblocks * NCS;
    std::vector<T> hw(std::size_t(kBatch) * rows_in * wn, nan_v<T>());
    T* wp_t = dev.up(hw);
    D* wp = reinterpret_cast<D*>(wp_t);

    const std::int64_t items = std::int64_t(kBatch) * rows_in;
    dev.template run<64>(std::size_t((items + 63) / 64), [=](sycl::nd_item<1> it) {
        const std::int64_t gid = static_cast<std::int64_t>(it.get_global_linear_id());
        if (gid >= items) return;
        const std::int64_t b = gid / rows_in;
        const int i = static_cast<int>(gid - b * rows_in);
        const std::int64_t vb = b * ms;
        const v2::RtOpMat<const D> opB{bp + b * sb, ldb, nt, cjb};
        D* Cb = cp + b * sc + G;
        const int rs = ro[b * os + i], re = ro[b * os + i + 1];
        for (int c0 = 0; c0 < width; c0 += NCS) {
            v2::Frag<D, NCS> ab;
            v2::load_cols_scaled<Ar>(v2::Thread{}, alpha_d, opB, i, c0, width, ab);
            for (int t = 0; t < NCS; ++t) wp[gid * wn + c0 + t] = ab[t];
            for (int p = rs; p < re; ++p) {
                D av = v[vb + p];
                if constexpr (is_cplx<T>) {
                    if (conj_a) av = v2::Arith<D>::conj(av);
                }
                const int j = ci[vb + p];
                if (static_cast<unsigned>(j) >= static_cast<unsigned>(rows_out)) continue;
                for (int t = 0; t < NCS; ++t) {
                    const int c = c0 + t;
                    if (c < width) v2::atomic_accumulate(Cb + static_cast<std::int64_t>(c) * ldc + j, Ar::mul(av, ab[t]));
                }
            }
        }
    });
    const auto got = dev.down(cp_t, hc.size());
    const auto gw = dev.down(wp_t, hw.size());
    auto opb = [&](int b, int i, int c) {
        const T x = nt ? hb[std::size_t(b) * sb + std::size_t(c) * ldb + i] : hb[std::size_t(b) * sb + std::size_t(i) * ldb + c];
        return cjb ? conj_h(x) : x;
    };
    double worst = 0;
    int untouched = 0, maxcount = 0, wreplay = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int e = 0; e < sc; ++e) {
            const int f = e - G, c = f / ldc, j = f % ldc;
            const bool inside = f >= 0 && c < width && j < rows_out;
            const std::size_t at = std::size_t(b) * sc + e;
            if (!inside) untouched += !same_bits(got[at], hc[at]);
        }
        if (b >= kPeriod) continue;
        for (int i = 0; i < rows_in; ++i)
            for (int c = 0; c < wn; ++c) {
                const T want = c < width ? h_mul(alpha, opb(b, i, c)) : T(0);
                wreplay += !same_bits(gw[(std::size_t(b) * rows_in + i) * wn + c], want);
            }
        for (int c = 0; c < width; ++c)
            for (int j = 0; j < rows_out; ++j) {
                const T cold = hc[std::size_t(b) * sc + G + std::size_t(c) * ldc + j];
                LD<T> ref = wide(cold);
                long double s = std::abs(wide(cold));
                int count = 0;
                for (int i = 0; i < rows_in; ++i)
                    for (int p = A.ro[i]; p < A.ro[i + 1]; ++p) {
                        if (A.ci[p] != j) continue;
                        const T a = hv[std::size_t(b) * ms + p];
                        const LD<T> term = wide(conj_a ? conj_h(a) : a) * wide(alpha) * wide(opb(b, i, c));
                        ref += term;
                        s += std::abs(term);
                        ++count;
                    }
                maxcount = std::max(maxcount, count);
                worst = std::max(worst, rel(got[std::size_t(b) * sc + G + std::size_t(c) * ldc + j], ref, s));
            }
    }
    EXPECT_EQ(wreplay, 0) << "load_cols_scaled: not dev_mul's fma form, or a column past width is not +0";
    EXPECT_EQ(untouched, 0) << "an out-of-range index or column was written";
    EXPECT_VERIFY(T, Check::blas, maxcount + 3, worst);
}

TYPED_TEST(DeviceBlasV2SparseTest, ScatterArmSkipsOutOfRangeAndConjugates) {
    using T = TypeParam;
    for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
        for (bool ca : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "transB " << int(tr) << " conj_a " << ca);
            check_scatter<T>(this->dev, tr, ca);
        }
}

// ---- ILUKEliminate: InterleavedVec slots, the kernel's update and modulus-max loops -----------
TYPED_TEST(DeviceBlasV2SparseTest, InterleavedSlotsUpdateAndModulusMax) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int slots = 40, first = 8, len = 20, nupd = 5;
    const std::vector<int> src = {1, 4, 33, 37, 9}, dst = {0, 2, 30, 35, 39};  // dst outside [first, first + len)
    // [first, first + len) is shrunk below modulus 0.71, then slot 12 gets modulus 1 (abs1 1),
    // slot 20 modulus 0.85 (abs1 1.2) and slot 15 a NaN, which the kernel's `m > s` skips.
    auto im = batched<T>(slots, 31);
    for (int b = 0; b < kBatch; ++b) {
        T* w = im.data() + std::size_t(b) * slots;
        for (int s = first; s < first + len; ++s) w[s] *= R(0.5);
        w[12] = T(R(-1));
        if constexpr (is_cplx<T>) w[20] = T(R(0.6), R(-0.6));
        w[15] = nan_v<T>();
    }
    std::vector<T> hw(im.size());
    for (int b = 0; b < kBatch; ++b)
        for (int s = 0; s < slots; ++s) hw[std::size_t(s) * kBatch + b] = im[std::size_t(b) * slots + s];
    const auto ha = batched<T>(1, 32);
    T* w = this->dev.up(hw);
    const T* a = this->dev.up(ha);
    const int* ds = this->dev.up(src);
    const int* dd = this->dev.up(dst);
    std::vector<R> hm(kBatch, R(-7));
    R* mx = this->dev.up(hm);
    this->dev.template run<64>(kBatch / 64, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const v2::InterleavedVec<T> at{w, kBatch, b};
        for (int u = 0; u < nupd; ++u) at(dd[u]) -= a[b] * at(ds[u]);
        R s = R(0);
        for (int p = first; p < first + len; ++p) {
            const R m = v2::modulus(at(p));
            if (m > s) s = m;
        }
        mx[b] = s;
    });
    const auto got = this->dev.down(w, hw.size());
    const auto gm = this->dev.down(mx, hm.size());
    double worst = 0;
    int bitdiff = 0, untouched = 0, badmax = 0;
    for (int b = 0; b < kBatch; ++b) {
        badmax += gm[b] != R(1);
        for (int s = 0; s < slots; ++s) {
            const std::size_t at = std::size_t(s) * kBatch + b;
            bitdiff += !same_bits(got[at], got[std::size_t(s) * kBatch + b % kPeriod]);
            const bool written = std::find(dst.begin(), dst.end(), s) != dst.end();
            if (!written) untouched += !same_bits(got[at], hw[at]);
        }
        if (b >= kPeriod) continue;
        for (int u = 0; u < nupd; ++u) {
            const T x = hw[std::size_t(src[u]) * kBatch + b], y = hw[std::size_t(dst[u]) * kBatch + b];
            const LD<T> ref = wide(y) - wide(ha[b]) * wide(x);
            worst = std::max(worst, rel(got[std::size_t(dst[u]) * kBatch + b], ref, std::abs(wide(y)) + std::abs(wide(ha[b])) * std::abs(wide(x))));
        }
    }
    EXPECT_EQ(badmax, 0) << "not max |x| with the true modulus over the item's own slots, or a NaN was not skipped";
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_EQ(untouched, 0) << "a slot of another item or outside the destinations was written";
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

// ---- usdot_sub_below: stops at the first idx >= bound, reads nothing past it ------------------
TYPED_TEST(DeviceBlasV2SparseTest, RowResidualStopsAtBoundAndReadsNothingPast) {
    using T = TypeParam;
    using Ar = v2::arith_t<v2::arith::Native, T>;
    const std::vector<int> cols = {0, 2, 3, 5, 8, 9, 11, 14, 15};  // bound 9: 8 kept, 9 ends the row
    constexpr int bound = 9, kept = 5, nent = 9, ylen = 16, begin = 2, ms = nent + begin + 3, ys = ylen + 4;
    auto hv = batched<T>(ms, 51), hy = batched<T>(ys, 53), hacc = batched<T>(1, 54);
    for (int b = 0; b < kBatch; ++b) {
        for (int e = kept; e < nent; ++e) hv[std::size_t(b) * ms + begin + e] = poison_v<T>();
        for (int j = bound; j < ylen; ++j) hy[std::size_t(b) * ys + 1 + j] = poison_v<T>();
    }
    std::vector<int> hci(std::size_t(kBatch) * ms, 0);
    for (int b = 0; b < kBatch; ++b) std::copy(cols.begin(), cols.end(), hci.begin() + std::size_t(b) * ms + begin);
    const T* v = this->dev.up(hv);
    const T* y = this->dev.up(hy);
    const T* acc = this->dev.up(hacc);
    const int* ci = this->dev.up(hci);
    std::vector<T> ho(kBatch, nan_v<T>());
    T* o = this->dev.up(ho);
    this->dev.template run<64>(kBatch / 64, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto row = v2::sp_row(v2::BasedVec<const T>{v, b * ms}, v2::BasedVec<const int>{ci, b * ms}, begin, begin + nent);
        o[b] = v2::usdot_sub_below<Ar>(v2::Thread{}, acc[b], row, v2::BasedVec<const T>{y, b * ys + 1}, bound);
    });
    const auto got = this->dev.down(o, ho.size());
    double worst = 0;
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b) {
        bitdiff += !same_bits(got[b], got[b % kPeriod]);
        if (b >= kPeriod) continue;
        LD<T> ref = wide(hacc[b]);
        long double s = std::abs(ref);
        for (int e = 0; e < kept; ++e) {
            const LD<T> t = wide(hv[std::size_t(b) * ms + begin + e]) * wide(hy[std::size_t(b) * ys + 1 + cols[e]]);
            ref -= t;
            s += std::abs(t);
        }
        worst = std::max(worst, rel(got[b], ref, s));
    }
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_VERIFY(T, Check::blas, kept + 1, worst);
}

// ---- ILUKApply: level-scheduled L solve (for_each_owned + usdot_sub_below), U solve as the kernel ----
template <class T, int WG> void check_lu_solve(Dev& dev) {
    using R = real_of<T>;
    using Ar = v2::arith_t<v2::arith::Native, T>;
    constexpr int n = 40;
    // Sorted CSR rows: strict lower (mix % 4 == 0), diagonal, strict upper (mix % 5 == 0).
    std::vector<int> ro{0}, ci, dpos(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const std::uint64_t h = mix(std::uint64_t(i) * 131u + j);
            if (j == i) dpos[i] = static_cast<int>(ci.size());
            if (j == i || (j < i && h % 4 == 0) || (j > i && h % 5 == 0)) ci.push_back(j);
        }
        ro.push_back(static_cast<int>(ci.size()));
    }
    const int nnz = static_cast<int>(ci.size()), ms = nnz + 7, rs = n + 3, ys = n + 5;
    auto hv = batched<T>(ms, 61);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n; ++i)
            for (int p = ro[i]; p < ro[i + 1]; ++p) {
                T& e = hv[std::size_t(b) * ms + p];
                e = ci[p] == i ? T(R(2) + std::abs(e)) + e * R(0.25) : e * R(0.3);
            }
    auto schedule = [&](bool lower, std::vector<int>& rows, std::vector<int>& ptr) {
        std::vector<int> lev(n, 0);
        int nlev = 0;
        for (int t = 0; t < n; ++t) {
            const int i = lower ? t : n - 1 - t;
            for (int p = ro[i]; p < ro[i + 1]; ++p)
                if (lower ? ci[p] < i : ci[p] > i) lev[i] = std::max(lev[i], lev[ci[p]] + 1);
            nlev = std::max(nlev, lev[i] + 1);
        }
        ptr.assign(1, 0);
        rows.clear();
        for (int l = 0; l < nlev; ++l) {
            for (int i = 0; i < n; ++i)
                if (lev[i] == l) rows.push_back(i);
            ptr.push_back(static_cast<int>(rows.size()));
        }
        return nlev;
    };
    std::vector<int> lrows, lptr, urows, uptr;
    const int nl = schedule(true, lrows, lptr), nu = schedule(false, urows, uptr);
    const auto hb = batched<T>(rs, 62);
    std::vector<T> hy(std::size_t(kBatch) * ys, nan_v<T>()), hmid(hy);
    const T* v = dev.up(hv);
    const int* dci = dev.up(replicate(ci, ms, 0));
    const int* dro = dev.up(ro);
    const int* dlr = dev.up(lrows);
    const int* dlp = dev.up(lptr);
    const int* dur = dev.up(urows);
    const int* dup = dev.up(uptr);
    const int* ddp = dev.up(dpos);
    const T* rhs = dev.up(hb);
    T* y = dev.up(hy);
    T* mid = dev.up(hmid);
    dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const v2::BasedVec<const T> lv{v, b * ms};
        const v2::BasedVec<const int> lc{dci, b * ms};
        const v2::BasedVec<const T> bv{rhs, b * rs};
        const v2::BasedVec<T> yv{y, b * ys};
        for (int l = 0; l < nl; ++l) {
            const int begin = dlp[l];
            v2::for_each_owned(ex, dlp[l + 1] - begin, [&](int k) {
                const int i = dlr[begin + k];
                yv(i) = v2::usdot_sub_below<Ar>(v2::Thread{}, bv(i), v2::sp_row(lv, lc, dro[i], dro[i + 1]), yv, i);
            });
            ex.barrier(v2::Mem::Global);
        }
        v2::for_each_owned(ex, n, [&](int i) { mid[b * ys + i] = yv(i); });
        for (int l = 0; l < nu; ++l) {
            const int begin = dup[l];
            for (int k = ex.lid(), nk = dup[l + 1] - begin; k < nk; k += ex.size) {
                const int i = dur[begin + k];
                const auto row = v2::sp_row(lv, lc, ddp[i] + 1, dro[i + 1]);
                T acc = yv(i);
                for (int p = row.begin; p < row.end; ++p) Ar::msub(acc, row.val(p), yv(row.idx(p)));
                yv(i) = Ar::div(acc, lv(ddp[i]));
            }
            ex.barrier(v2::Mem::Global);
        }
    });
    const auto gx = dev.down(y, hy.size());
    const auto gy = dev.down(mid, hmid.size());
    double worst = 0;
    int bitdiff = 0, maxlen = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int i = 0; i < ys; ++i) bitdiff += !same_bits(gx[std::size_t(b) * ys + i], gx[std::size_t(b % kPeriod) * ys + i]);
        if (b >= kPeriod) continue;
        for (int i = 0; i < n; ++i) {
            maxlen = std::max(maxlen, ro[i + 1] - ro[i]);
            // Componentwise backward error of each stage: rhs = y + L_strict y, y = U x.
            LD<T> r1 = wide(hb[std::size_t(b) * rs + i]) - wide(gy[std::size_t(b) * ys + i]);
            long double s1 = std::abs(wide(hb[std::size_t(b) * rs + i])) + std::abs(wide(gy[std::size_t(b) * ys + i]));
            LD<T> r2 = wide(gy[std::size_t(b) * ys + i]);
            long double s2 = std::abs(r2);
            for (int p = ro[i]; p < ro[i + 1]; ++p) {
                const LD<T> a = wide(hv[std::size_t(b) * ms + p]);
                if (ci[p] < i) {
                    const LD<T> yj = wide(gy[std::size_t(b) * ys + ci[p]]);
                    r1 -= a * yj;
                    s1 += std::abs(a) * std::abs(yj);
                } else {
                    const LD<T> xj = wide(gx[std::size_t(b) * ys + ci[p]]);
                    r2 -= a * xj;
                    s2 += std::abs(a) * std::abs(xj);
                }
            }
            worst = std::max({worst, static_cast<double>(std::abs(r1) / s1), static_cast<double>(std::abs(r2) / s2)});
        }
    }
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_VERIFY(T, Check::blas, maxlen + 2, worst);
}

// WG 128 is ILUKApply's launch; 1024 is the work-group ceiling.
TYPED_TEST(DeviceBlasV2SparseTest, LevelScheduledLuSolveBackwardStable) {
    check_lu_solve<TypeParam, 128>(this->dev);
    check_lu_solve<TypeParam, 1024>(this->dev);
}

}  // namespace

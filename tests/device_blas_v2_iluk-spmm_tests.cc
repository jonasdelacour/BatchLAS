// SPDX-License-Identifier: MIT
// Device LA v2 sparse extension (include/batchlas/blas/device/v2/ext/iluk-spmm.hh) against host
// references. Guards (docs/developer/agent-guide.md section 8): non-natural strides everywhere
// (CSR item strides above nnz, ldb / ldc above the extent, offset bases, batch-interleaved
// slots), complex data with nonzero imaginary parts, runtime NoTrans / Trans / ConjTrans, beta
// != 0 and beta == 0 over a NaN-poisoned C, NaN in every entry a bounded op must not read, an
// out-of-range scatter index beside sentinels, bit-identity to a host replay of the fma chain,
// batch 1024 with period-3 twins that must be bit-identical, a WG = 1024 launch, and a modulus
// (not abs1) maximum.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/iluk-spmm.hh>
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
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
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

// Host replay of FmaChainArith, operation for operation (std::fma is correctly rounded like sycl::fma).
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
// Pattern arrays replicated per item at strides above their extent (offset stride os, value stride ms).
std::vector<int> replicate(const std::vector<int>& v, int stride, int pad_value) {
    std::vector<int> h(std::size_t(kBatch) * stride, pad_value);
    for (int b = 0; b < kBatch; ++b) std::copy(v.begin(), v.end(), h.begin() + std::size_t(b) * stride);
    return h;
}

// ---- CSR row x dense block: usmm_row + store_cols (the spmm gather arm) ----------------------
template <class T, bool Pair> void check_usmm(Dev& dev, Transpose trans, bool beta_zero) {
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
    const T* v = dev.up(hv);
    const int* ci = dev.up(replicate(A.ci, ms, 0));
    const int* ro = dev.up(replicate(A.ro, os, 0));
    const T* bp = dev.up(hb);
    T* cp = dev.up(hc);
    const std::int64_t total = std::int64_t(kBatch) * m * cblocks;
    dev.template run<64>(std::size_t((total + 63) / 64), [=](sycl::nd_item<1> it) {
        const std::int64_t gid = static_cast<std::int64_t>(it.get_global_linear_id());
        if (gid >= total) return;
        const std::int64_t b = gid / (m * cblocks);
        const int rem = static_cast<int>(gid - b * m * cblocks);
        const int i = rem % m, c0 = (rem / m) * NC;
        const auto arow = v2::sp_row(v2::LoadVec<const T, Pair>{v + b * ms}, v2::BasedVec<const int, std::int64_t>{ci, b * ms},
                                     ro[b * os + i], ro[b * os + i + 1]);
        const v2::RtOpMat<const T, Pair> opB{bp + b * sb, ldb, nt, cj};
        const v2::ColMat64<T, Pair> C{cp + b * sc, ldc};
        auto acc = v2::frag_fill<T, NC>(v2::Arith<T>::zero());
        v2::usmm_row<v2::FmaChainArith<T>>(v2::Thread{}, arow, opB, c0, width, acc);
        v2::store_cols<v2::FmaChainArith<T>>(v2::Thread{}, alpha, acc, beta, beta_zero, C, i, c0, width);
    });
    const auto got = dev.down(cp, hc.size());
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

// ---- transposed arm: scal_or_zero, then load_cols_scaled + usger_atomic -------------------
template <class T> void check_scatter(Dev& dev, Transpose transB, bool conj_a, bool beta_zero) {
    constexpr int NCS = 4, G = 4;
    const int rows_in = 9, rows_out = 12, width = 6, ldc = rows_out + 3, sc = G + ldc * width + G;
    // Each stored row also holds column rows_out (even rows) and -1 (rows divisible by 3): both
    // out of range and must be skipped, the -1 one would land in the guard before C.
    Csr A = pattern(rows_in, rows_out, 4);
    {
        Csr x;
        x.ro.push_back(0);
        for (int r = 0; r < rows_in; ++r) {
            for (int p = A.ro[r]; p < A.ro[r + 1]; ++p) x.ci.push_back(A.ci[p]);
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
    const auto hv = batched<T>(ms, 21), hb = batched<T>(sb, 22);
    const auto hc = beta_zero ? std::vector<T>(std::size_t(kBatch) * sc, nan_v<T>()) : batched<T>(sc, 23);
    const T alpha = val<T>(24), beta = beta_zero ? T(0) : val<T>(25);
    const T* v = dev.up(hv);
    const int* ci = dev.up(replicate(A.ci, ms, 0));
    const int* ro = dev.up(replicate(A.ro, os, 0));
    const T* bp = dev.up(hb);
    T* cp = dev.up(hc);
    using Ar = v2::FmaChainArith<T>;

    const std::int64_t elems = std::int64_t(kBatch) * rows_out * width;
    dev.template run<64>(std::size_t((elems + 63) / 64), [=](sycl::nd_item<1> it) {
        const std::int64_t gid = static_cast<std::int64_t>(it.get_global_linear_id());
        if (gid >= elems) return;
        const std::int64_t b = gid / (rows_out * width);
        const int r = static_cast<int>(gid - b * rows_out * width);
        v2::scal_or_zero<Ar>(beta, beta_zero, cp[b * sc + G + std::int64_t(r / rows_out) * ldc + r % rows_out]);
    });
    const auto scaled = dev.down(cp, hc.size());
    int scal_replay = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int c = 0; c < width; ++c)
            for (int j = 0; j < rows_out; ++j) {
                const std::size_t at = std::size_t(b) * sc + G + std::size_t(c) * ldc + j;
                scal_replay += !same_bits(scaled[at], beta_zero ? T(0) : h_mul(beta, hc[at]));
            }
    EXPECT_EQ(scal_replay, 0) << "scal_or_zero: not dev_mul's fma form, or beta == 0 is not +0";

    const std::int64_t items = std::int64_t(kBatch) * rows_in;
    dev.template run<64>(std::size_t((items + 63) / 64), [=](sycl::nd_item<1> it) {
        const std::int64_t gid = static_cast<std::int64_t>(it.get_global_linear_id());
        if (gid >= items) return;
        const std::int64_t b = gid / rows_in;
        const int i = static_cast<int>(gid - b * rows_in);
        const auto arow = v2::sp_row(v2::BasedVec<const T, std::int64_t>{v, b * ms}, v2::BasedVec<const int, std::int64_t>{ci, b * ms},
                                     ro[b * os + i], ro[b * os + i + 1]);
        const v2::RtOpMat<const T> opB{bp + b * sb, ldb, nt, cjb};
        for (int c0 = 0; c0 < width; c0 += NCS) {
            const auto ab = v2::load_cols_scaled<Ar, NCS>(v2::Thread{}, alpha, opB, i, c0, width);
            v2::usger_atomic<Ar>(v2::Thread{}, arow, conj_a, ab, cp + b * sc + G, std::int64_t(ldc), rows_out, c0, width);
        }
    });
    const auto got = dev.down(cp, hc.size());
    double worst = 0;
    int untouched = 0, maxcount = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int e = 0; e < sc; ++e) {
            const int f = e - G, c = f / ldc, j = f % ldc;
            const bool inside = f >= 0 && c < width && j < rows_out;
            const std::size_t at = std::size_t(b) * sc + e;
            if (!inside) untouched += !same_bits(got[at], hc[at]);
        }
        if (b >= kPeriod) continue;
        for (int c = 0; c < width; ++c)
            for (int j = 0; j < rows_out; ++j) {
                const T cold = hc[std::size_t(b) * sc + G + std::size_t(c) * ldc + j];
                LD<T> ref = beta_zero ? LD<T>(0) : wide(beta) * wide(cold);
                long double s = beta_zero ? 0.0L : std::abs(wide(beta)) * std::abs(wide(cold));
                int count = 0;
                for (int i = 0; i < rows_in; ++i)
                    for (int p = A.ro[i]; p < A.ro[i + 1]; ++p) {
                        if (A.ci[p] != j) continue;
                        const T a = hv[std::size_t(b) * ms + p];
                        const T bx = nt ? hb[std::size_t(b) * sb + std::size_t(c) * ldb + i] : hb[std::size_t(b) * sb + std::size_t(i) * ldb + c];
                        const LD<T> term = wide(conj_a ? conj_h(a) : a) * wide(alpha) * wide(cjb ? conj_h(bx) : bx);
                        ref += term;
                        s += std::abs(term);
                        ++count;
                    }
                maxcount = std::max(maxcount, count);
                worst = std::max(worst, rel(got[std::size_t(b) * sc + G + std::size_t(c) * ldc + j], ref, s));
            }
    }
    EXPECT_EQ(untouched, 0) << "an out-of-range index or column was written";
    EXPECT_VERIFY(T, Check::blas, maxcount + 3, worst);
}

TYPED_TEST(DeviceBlasV2SparseTest, ScatterArmSkipsOutOfRangeAndConjugates) {
    using T = TypeParam;
    for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
        for (bool ca : {false, true})
            for (bool bz : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "transB " << int(tr) << " conj_a " << ca << " beta_zero " << bz);
                check_scatter<T>(this->dev, tr, ca, bz);
            }
}

// ---- ILU elimination pieces: InterleavedVec, us_gather, axpy_sub, sp_range, norm_max ----------
TYPED_TEST(DeviceBlasV2SparseTest, InterleavedGatherAxpyAndModulusMax) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int slots = 40, first = 8, len = 20, nupd = 5;
    const std::vector<int> src = {1, 4, 33, 37, 9}, dst = {0, 2, 30, 35, 39};  // dst outside [first, first + len)
    // Item-major values, made slot-major below; [first, first + len) is shrunk below modulus 0.71,
    // then slot 12 gets modulus 1 (abs1 1) and slot 20 modulus 0.85 (abs1 1.2), slot 15 a NaN.
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
        const v2::InterleavedVec<T> at{w, std::size_t(kBatch), std::size_t(b)};
        v2::axpy_sub<v2::NativeArith<T>>(v2::Thread{}, a[b], v2::us_gather(at, ds, nupd), v2::us_gather(at, dd, nupd));
        mx[b] = v2::norm_max(v2::Thread{}, v2::sp_range(at, first, len));
    });
    const auto got = this->dev.down(w, hw.size());
    const auto gm = this->dev.down(mx, hm.size());
    double worst = 0;
    int bitdiff = 0, untouched = 0, badmax = 0;
    for (int b = 0; b < kBatch; ++b) {
        badmax += gm[b] != R(1);  // the modulus maximum; NaN skipped; abs1 would give 1.2 for complex
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
    EXPECT_EQ(badmax, 0) << "norm_max is not max |x| with the true modulus, or a NaN was not skipped";
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_EQ(untouched, 0) << "axpy_sub wrote a slot outside its gathered destination";
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

// norm_max on a WorkGroup at the 1024 ceiling (group reduction with sycl::maximum).
TYPED_TEST(DeviceBlasV2SparseTest, NormMaxWorkGroupCeiling) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 1024, n = 3001, stride = n + 5, items = 64;
    std::vector<T> hx(std::size_t(items) * stride);
    for (std::size_t i = 0; i < hx.size(); ++i) hx[i] = val<T>(41 + i);
    const T* x = this->dev.up(hx);
    std::vector<R> hm(items, R(-7));
    R* mx = this->dev.up(hm);
    this->dev.template run<WG>(items, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> ex{it};
        const R m = v2::norm_max(ex, v2::vec(x + std::ptrdiff_t(b) * stride + 2, n));
        if (ex.leader()) mx[b] = m;
    });
    const auto gm = this->dev.down(mx, hm.size());
    double worst = 0;
    for (int b = 0; b < items; ++b) {
        long double ref = 0;
        for (int i = 0; i < n; ++i) ref = std::max(ref, std::abs(wide(hx[std::size_t(b) * stride + 2 + i])));
        worst = std::max(worst, static_cast<double>(std::abs(static_cast<long double>(gm[b]) - ref) / ref));
    }
    EXPECT_VERIFY(T, Check::blas, 1, worst);
}

// ---- row residuals of a triangular solve: usdot_sub_below and usdot_sub ----------------------
TYPED_TEST(DeviceBlasV2SparseTest, RowResidualStopsAtBoundAndReadsNothingPast) {
    using T = TypeParam;
    const std::vector<int> cols = {0, 2, 3, 5, 8, 9, 11, 14, 15};  // sorted; bound 9 keeps the first five
    constexpr int bound = 9, nent = 9, ylen = 16, begin = 2, ms = nent + begin + 3, ys = ylen + 4;
    auto hv = batched<T>(ms, 51), hv2 = batched<T>(ms, 52), hy = batched<T>(ys, 53), hacc = batched<T>(1, 54);
    for (int b = 0; b < kBatch; ++b) {
        for (int e = 5; e < nent; ++e) hv[std::size_t(b) * ms + begin + e] = nan_v<T>();  // past the bound: never read
        for (int j = bound; j < ylen; ++j) hy[std::size_t(b) * ys + 1 + j] = nan_v<T>();
    }
    auto hy2 = batched<T>(ys, 55);
    std::vector<int> hci(std::size_t(kBatch) * ms, 0);
    for (int b = 0; b < kBatch; ++b) std::copy(cols.begin(), cols.end(), hci.begin() + std::size_t(b) * ms + begin);
    const T* v = this->dev.up(hv);
    const T* v2p = this->dev.up(hv2);
    const T* y = this->dev.up(hy);
    const T* y2 = this->dev.up(hy2);
    const T* acc = this->dev.up(hacc);
    const int* ci = this->dev.up(hci);
    std::vector<T> ho(std::size_t(kBatch) * 2, nan_v<T>());
    T* o = this->dev.up(ho);
    this->dev.template run<64>(kBatch / 64, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const v2::BasedVec<const int> idx{ci, b * ms};
        const auto row = v2::sp_row(v2::BasedVec<const T>{v, b * ms}, idx, begin, begin + nent);
        const auto row2 = v2::sp_row(v2::BasedVec<const T>{v2p, b * ms}, idx, begin, begin + nent);
        o[2 * b] = v2::usdot_sub_below<v2::NativeArith<T>>(v2::Thread{}, acc[b], row, v2::BasedVec<const T>{y, b * ys + 1}, bound);
        o[2 * b + 1] = v2::usdot_sub<v2::ArithMsub<T>>(v2::Thread{}, acc[b], row2, v2::BasedVec<const T>{y2, b * ys + 1});
    });
    const auto got = this->dev.down(o, ho.size());
    double worst = 0;
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b) {
        bitdiff += !same_bits(got[2 * b], got[2 * (b % kPeriod)]) + !same_bits(got[2 * b + 1], got[2 * (b % kPeriod) + 1]);
        if (b >= kPeriod) continue;
        for (int which = 0; which < 2; ++which) {
            const auto& vals = which == 0 ? hv : hv2;
            const auto& yy = which == 0 ? hy : hy2;
            LD<T> ref = wide(hacc[b]);
            long double s = std::abs(ref);
            for (int e = 0; e < (which == 0 ? 5 : nent); ++e) {
                const LD<T> t = wide(vals[std::size_t(b) * ms + begin + e]) * wide(yy[std::size_t(b) * ys + 1 + cols[e]]);
                ref -= t;
                s += std::abs(t);
            }
            worst = std::max(worst, rel(got[2 * b + which], ref, s));
        }
    }
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_VERIFY(T, Check::blas, nent + 1, worst);
}

// ---- the ILU(k) apply composition: level-scheduled L then U solve on a WorkGroup -------------
TYPED_TEST(DeviceBlasV2SparseTest, LevelScheduledLuSolveBackwardStable) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 128, n = 40;
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
    // Level schedules: L level(i) = 1 + max over lower j; U the same from the bottom.
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
    const T* v = this->dev.up(hv);
    const int* dci = this->dev.up(replicate(ci, ms, 0));
    const int* dro = this->dev.up(ro);
    const int* dlr = this->dev.up(lrows);
    const int* dlp = this->dev.up(lptr);
    const int* dur = this->dev.up(urows);
    const int* dup = this->dev.up(uptr);
    const int* ddp = this->dev.up(dpos);
    const T* rhs = this->dev.up(hb);
    T* y = this->dev.up(hy);
    T* mid = this->dev.up(hmid);
    using Ar = v2::NativeArith<T>;
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
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
            v2::for_each_owned(ex, dup[l + 1] - begin, [&](int k) {
                const int i = dur[begin + k];
                const T sum = yv(i);
                const T diag = lv(ddp[i]);
                yv(i) = Ar::div(v2::usdot_sub<Ar>(v2::Thread{}, sum, v2::sp_row(lv, lc, ddp[i] + 1, dro[i + 1]), yv), diag);
            });
            ex.barrier(v2::Mem::Global);
        }
    });
    const auto gx = this->dev.down(y, hy.size());
    const auto gy = this->dev.down(mid, hmid.size());
    double worst = 0;
    int bitdiff = 0, maxlen = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int i = 0; i < ys; ++i) bitdiff += !same_bits(gx[std::size_t(b) * ys + i], gx[std::size_t(b % kPeriod) * ys + i]);
        if (b >= kPeriod) continue;
        for (int i = 0; i < n; ++i) {
            maxlen = std::max(maxlen, ro[i + 1] - ro[i]);
            // Backward error of each stage, componentwise: rhs = y + L_strict y, y = U x.
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

}  // namespace

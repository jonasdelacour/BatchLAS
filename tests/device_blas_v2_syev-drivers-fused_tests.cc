// SPDX-License-Identifier: MIT
// Device LA v2 ext ops of the syev drivers (include/batchlas/blas/device/v2/ext/
// syev-drivers-fused.hh) against host references.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld everywhere, complex data with
// nonzero imaginary parts (also on the diagonal), NaN poison in every entry an op must not read
// or write (unread triangle, ld padding, reflector-store rows at and below the head, S when
// want_s == false), a zero off-diagonal and a tau == 0 reflector, batch 1024 with period-3
// data, partitions of 8 (four problems per sub-group) and 32, n < P so pad lanes exist.
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
#include <batchlas/blas/device/v2/ext/syev-drivers-fused.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class C> C conj_w(C v) {
    if constexpr (std::is_same_v<C, long double>) return v;
    else return std::conj(v);
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// Representative item b % kPeriod of a per-item array.
template <class T> std::vector<T> per_item(int per, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < per; ++i) h[std::size_t(b) * per + i] = val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i);
    return h;
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

template <class T> class DeviceBlasV2SyevDriversTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2SyevDriversTest, Types);

// ---- Hermitian completion ------------------------------------------------------------------

// herm_at (both stored triangles) and mirror_at<Upper, Keep> / <Lower, Real>: pure data
// movement, so every entry is checked bit for bit, the ld padding included.
TYPED_TEST(DeviceBlasV2SyevDriversTest, HermitianCompletion) {
    using T = TypeParam;
    constexpr int WG = 128, n = 13, ld = n + 4, ldo = n + 2;
    const int per = n * ld, pero = n * ldo;
    auto ha = per_item<T>(per, 1);  // diagonal keeps its imaginary part: herm_at / diag::Real drop it
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = n; i < ld; ++i) ha[std::size_t(b) * per + i + j * ld] = nan_v<T>();
    std::vector<T> hout(std::size_t(kBatch) * 2 * pero, nan_v<T>());
    T* a = this->dev.up(ha);
    T* m1 = this->dev.up(ha);
    T* m2 = this->dev.up(ha);
    T* o = this->dev.up(hout);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto A = v2::mat(a + std::ptrdiff_t(b) * per, n, n, ld);
        const auto M1 = v2::mat(m1 + std::ptrdiff_t(b) * per, n, n, ld);
        const auto M2 = v2::mat(m2 + std::ptrdiff_t(b) * per, n, n, ld);
        const auto Hu = v2::mat(o + std::ptrdiff_t(b) * 2 * pero, n, n, ldo);
        const auto Hl = v2::mat(o + std::ptrdiff_t(b) * 2 * pero + pero, n, n, ldo);
        for (int t = static_cast<int>(it.get_local_linear_id()); t < n * n; t += WG) {
            const int i = t % n, j = t / n;
            Hu(i, j) = v2::herm_at(A, true, i, j);
            Hl(i, j) = v2::herm_at<v2::NativeArith<T>>(A, false, i, j);
            v2::mirror_at<Uplo::Upper>(M1, i, j);
            v2::mirror_at<Uplo::Lower, v2::diag::Real, v2::NativeArith<T>>(M2, i, j);
        }
    });
    const auto go = this->dev.down(o, hout.size());
    const auto g1 = this->dev.down(m1, ha.size());
    const auto g2 = this->dev.down(m2, ha.size());
    int bad_h = 0, bad_m = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* A = ha.data() + std::size_t(b) * per;
        auto realdiag = [&](int i) { return is_cplx<T> ? T(std::real(A[i + i * ld])) : A[i + i * ld]; };
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const T hu = i == j ? realdiag(i) : (i < j ? A[i + j * ld] : conj_h(A[j + i * ld]));
                const T hl = i == j ? realdiag(i) : (i > j ? A[i + j * ld] : conj_h(A[j + i * ld]));
                bad_h += !same_bits(go[std::size_t(b) * 2 * pero + i + j * ldo], hu);
                bad_h += !same_bits(go[std::size_t(b) * 2 * pero + pero + i + j * ldo], hl);
                const T m1r = i <= j ? A[i + j * ld] : conj_h(A[j + i * ld]);
                const T m2r = i == j ? realdiag(i) : (i > j ? A[i + j * ld] : conj_h(A[j + i * ld]));
                bad_m += !same_bits(g1[std::size_t(b) * per + i + j * ld], m1r);
                bad_m += !same_bits(g2[std::size_t(b) * per + i + j * ld], m2r);
            }
            for (int i = n; i < ld; ++i) {
                bad_m += !same_bits(g1[std::size_t(b) * per + i + j * ld], A[i + j * ld]);
                bad_m += !same_bits(g2[std::size_t(b) * per + i + j * ld], A[i + j * ld]);
            }
        }
    }
    EXPECT_EQ(bad_h, 0) << "herm_at";
    EXPECT_EQ(bad_m, 0) << "mirror_at";
}

// ---- dgmm Left lift ------------------------------------------------------------------------

// C := diag(d) Z with Z real lifted (out of place), and Y := diag(d) Y in place, through the
// item_vec / item_mat adapters over v1-style (i, b) / (i, j, b) accessors.
template <class T> struct BatchedVec {
    T* p;
    int stride;
    T& operator()(int i, int b) const { return p[std::ptrdiff_t(b) * stride + i]; }
};
template <class T> struct BatchedMat {
    T* p;
    int ld, stride;
    T& operator()(int i, int j, int b) const { return p[std::ptrdiff_t(b) * stride + i + std::ptrdiff_t(j) * ld]; }
};

TYPED_TEST(DeviceBlasV2SyevDriversTest, DgmmLeftLift) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 64, n = 19, ldz = n + 3, ldc = n + 5;
    auto hd = per_item<T>(n, 11);
    auto hz = per_item<R>(n * ldz, 12);
    auto hy = per_item<T>(n * ldc, 13);
    std::vector<T> hc(std::size_t(kBatch) * n * ldc, nan_v<T>());
    T* d = this->dev.up(hd);
    R* z = this->dev.up(hz);
    T* y = this->dev.up(hy);
    T* c = this->dev.up(hc);
    const BatchedVec<T> D{d, n};
    const BatchedMat<R> Z{z, ldz, n * ldz};
    const BatchedMat<T> C{c, ldc, n * ldc}, Y{y, ldc, n * ldc};
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        for (int t = static_cast<int>(it.get_local_linear_id()); t < n * n; t += WG) {
            const int i = t % n, j = t / n;
            v2::dgmm_left_at(v2::item_vec(D, b), v2::item_mat(Z, b), v2::item_mat(C, b), i, j);
            v2::dgmm_left_at<v2::NativeArith<T>>(v2::item_vec(D, b), v2::item_mat(Y, b), v2::item_mat(Y, b), i, j);
        }
    });
    const auto gc = this->dev.down(c, hc.size());
    const auto gy = this->dev.down(y, hy.size());
    double worst = 0;
    int pad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const LD<T> di = wide(hd[std::size_t(b) * n + i]);
                const LD<T> zr = LD<T>(static_cast<long double>(hz[std::size_t(b) * n * ldz + i + j * ldz]));
                const LD<T> yr = wide(hy[std::size_t(b) * n * ldc + i + j * ldc]);
                worst = std::max({worst, rel(gc[std::size_t(b) * n * ldc + i + j * ldc], di * zr, std::abs(di) * std::abs(zr)),
                                  rel(gy[std::size_t(b) * n * ldc + i + j * ldc], di * yr, std::abs(di) * std::abs(yr))});
            }
            for (int i = n; i < ldc; ++i) pad += !std::isnan(std::real(LD<T>(gc[std::size_t(b) * n * ldc + i + j * ldc])));
        }
    }
    EXPECT_EQ(pad, 0) << "ld padding written";
    EXPECT_VERIFY(T, Check::blas, 1, worst);
}

// ---- tridiagonal similarities --------------------------------------------------------------

// phase_similarity, both Uplo: T' = S^H T S has the real off-diagonal |e|, |S| = 1, a zero e_i
// carries S over bit for bit; want_s == false leaves NaN-poisoned S untouched. Real types:
// sign_similarity gives e := |e| and S = +-1 exactly.
TYPED_TEST(DeviceBlasV2SyevDriversTest, TridiagonalSimilarities) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 32, n = 17, kZero = 5;
    auto hd = per_item<T>(n, 21);
    auto he = per_item<T>(n - 1, 22);
    for (int b = 0; b < kBatch; ++b) he[std::size_t(b) * (n - 1) + kZero] = T(0);
    std::vector<R> hdr(std::size_t(kBatch) * n * 3, std::numeric_limits<R>::quiet_NaN());
    std::vector<R> her(std::size_t(kBatch) * (n - 1) * 3, std::numeric_limits<R>::quiet_NaN());
    std::vector<T> hs(std::size_t(kBatch) * n * 3, nan_v<T>());
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    T* e2 = this->dev.up(he);
    R* dr = this->dev.up(hdr);
    R* er = this->dev.up(her);
    T* s = this->dev.up(hs);
    T* s2 = this->dev.up(std::vector<T>(std::size_t(kBatch) * n, nan_v<T>()));
    this->dev.template run<WG>(kBatch / WG, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto dv = v2::vec(d + std::ptrdiff_t(b) * n, n);
        const auto ev = v2::vec(e + std::ptrdiff_t(b) * (n - 1), n - 1);
        for (int k = 0; k < 3; ++k) {
            const Uplo u = k == 1 ? Uplo::Upper : Uplo::Lower;
            const auto drv = v2::vec(dr + (std::ptrdiff_t(b) * 3 + k) * n, n);
            const auto erv = v2::vec(er + (std::ptrdiff_t(b) * 3 + k) * (n - 1), n - 1);
            const auto sv = v2::vec(s + (std::ptrdiff_t(b) * 3 + k) * n, n);
            v2::phase_similarity<v2::NativeArith<T>>(v2::Thread{}, n, dv, ev, drv, erv, sv, u, /*want_s=*/k < 2);
        }
        if constexpr (!is_cplx<T>) {
            v2::sign_similarity(v2::Thread{}, n, v2::vec(e2 + std::ptrdiff_t(b) * (n - 1), n - 1), v2::vec(s2 + std::ptrdiff_t(b) * n, n));
        }
    });
    const auto gdr = this->dev.down(dr, hdr.size());
    const auto ger = this->dev.down(er, her.size());
    const auto gs = this->dev.down(s, hs.size());
    const auto ge2 = this->dev.down(e2, he.size());
    const auto gs2 = this->dev.down(s2, std::size_t(kBatch) * n);
    double worst = 0;
    int exact = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int k = 0; k < 3; ++k) {
            const R* gdrk = gdr.data() + (std::size_t(b) * 3 + k) * n;
            const R* gerk = ger.data() + (std::size_t(b) * 3 + k) * (n - 1);
            const T* gsk = gs.data() + (std::size_t(b) * 3 + k) * n;
            for (int i = 0; i < n; ++i) exact += !same_bits(gdrk[i], static_cast<R>(std::real(hd[std::size_t(b) * n + i])));
            for (int i = 0; i < n - 1; ++i) {
                const LD<T> ei = wide(he[std::size_t(b) * (n - 1) + i]);
                worst = std::max(worst, rel<R>(gerk[i], std::abs(ei), std::abs(ei)));
            }
            if (k == 2) {
                for (int i = 0; i < n; ++i) exact += !std::isnan(std::real(LD<T>(gsk[i])));
                continue;
            }
            exact += !same_bits(gsk[0], T(1));
            for (int i = 0; i < n - 1; ++i) {
                const LD<T> ei = wide(he[std::size_t(b) * (n - 1) + i]);
                const LD<T> si = wide(gsk[i]), sn = wide(gsk[i + 1]);
                worst = std::max(worst, rel<T>(T(std::abs(sn)), LD<T>(1), 1.0L));
                if (i == kZero) {
                    exact += !same_bits(gsk[i + 1], gsk[i]);
                    continue;
                }
                const LD<T> off = (k == 0) ? conj_w(sn) * ei * si : conj_w(si) * ei * sn;  // T'(i+1,i) / T'(i,i+1)
                worst = std::max(worst, rel<T>(static_cast<T>(off), LD<T>(std::abs(ei)), std::abs(ei)));
            }
        }
        if constexpr (!is_cplx<T>) {
            T sg = T(1);
            exact += !same_bits(gs2[std::size_t(b) * n], T(1));
            for (int i = 0; i < n - 1; ++i) {
                const T ei = he[std::size_t(b) * (n - 1) + i];
                sg = sg * (ei >= T(0) ? T(1) : T(-1));
                exact += !same_bits(ge2[std::size_t(b) * (n - 1) + i], std::fabs(ei));
                exact += !same_bits(gs2[std::size_t(b) * n + i + 1], sg);
            }
        }
    }
    EXPECT_EQ(exact, 0) << "dr / S(0) / zero-e carry / unwritten S / sign_similarity";
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// phase_replay on a masked SgpPartition<P>: lane l returns S(l) of the superdiagonal chain,
// the values phase_similarity<Upper> writes serially on lane 0 (same product order), so the
// two agree to rounding; a lane reading another chunk's e would not.
template <class T, int P> void run_phase_replay(Dev& dev, int n) {
    using R = real_of<T>;
    constexpr int WG = 64, chunks = WG / P;
    auto he = per_item<T>(P, 31);
    for (int b = 0; b < kBatch; ++b) he[std::size_t(b) * P + 2] = T(0);
    std::vector<T> ho(std::size_t(kBatch) * P * 2, nan_v<T>());
    T* e = dev.up(he);
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch / chunks, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::SgpPartition<P> pex{it.get_sub_group()};
        const int lane = pex.lid();
        const int b = static_cast<int>(it.get_group_linear_id()) * chunks + static_cast<int>(it.get_local_linear_id()) / P;
        const T el = lane < n - 1 ? e[std::ptrdiff_t(b) * P + lane] : T(0);
        const R al = lane < n - 1 ? sycl::hypot(v2::Arith<T>::re(el), v2::Arith<T>::im(el)) : R(0);
        o[std::ptrdiff_t(b) * 2 * P + lane] = v2::phase_replay<v2::NativeArith<T>>(pex, el, al, n);
        if (lane == 0) {
            // Serial reference on lane 0 (Thread): S written to the second half.
            T* sr = o + std::ptrdiff_t(b) * 2 * P + P;
            const auto ev = v2::vec(e + std::ptrdiff_t(b) * P, n - 1);
            R scratch[P];
            const auto rv = v2::Vec<R, v2::Ext<P>>{scratch, {}};
            v2::phase_similarity<v2::NativeArith<T>>(v2::Thread{}, n, ev, ev, rv, rv, v2::vec(sr, n), Uplo::Upper, true);
        }
    });
    const auto g = dev.down(o, ho.size());
    double worst = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int l = 0; l < n; ++l) worst = std::max(worst, rel(g[std::size_t(b) * 2 * P + l], wide(g[std::size_t(b) * 2 * P + P + l]), 1.0L));
    SCOPED_TRACE(::testing::Message() << "phase_replay<" << P << "> n=" << n);
    EXPECT_VERIFY(T, Check::blas, n, worst);
}
TYPED_TEST(DeviceBlasV2SyevDriversTest, PhaseReplayPartition) {
    run_phase_replay<TypeParam, 8>(this->dev, 7);
    run_phase_replay<TypeParam, 32>(this->dev, 32);
    run_phase_replay<TypeParam, 32>(this->dev, 20);
}

// ---- SYTRD reflector repack ----------------------------------------------------------------

// sytrd_reflector_pack_at with and without the folded TAUQ, and sytrd_tau_pack_at, both Uplo;
// exact. TAUQ starts NaN, so the no-tau form must leave it NaN.
TYPED_TEST(DeviceBlasV2SyevDriversTest, SytrdReflectorPack) {
    using T = TypeParam;
    constexpr int WG = 128, n = 11, lda = n + 2, ldq = n + 3;
    auto ha = per_item<T>(n * lda, 41);
    auto ht = per_item<T>(n - 1, 42);
    std::vector<T> hq(std::size_t(kBatch) * 4 * n * ldq, nan_v<T>());
    std::vector<T> htq(std::size_t(kBatch) * 6 * n, nan_v<T>());
    T* a = this->dev.up(ha);
    T* tau = this->dev.up(ht);
    T* aq = this->dev.up(hq);
    T* tq = this->dev.up(htq);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int lid = static_cast<int>(it.get_local_linear_id());
        const auto A = v2::mat(a + std::ptrdiff_t(b) * n * lda, n, n, lda);
        const auto tv = v2::vec(tau + std::ptrdiff_t(b) * (n - 1), n - 1);
        for (int k = 0; k < 4; ++k) {
            const Uplo u = (k & 1) ? Uplo::Upper : Uplo::Lower;
            const auto AQ = v2::mat(aq + (std::ptrdiff_t(b) * 4 + k) * n * ldq, n, n, ldq);
            const auto tqv = v2::vec(tq + (std::ptrdiff_t(b) * 6 + k) * n, n);
            for (int t = lid; t < n * n; t += WG) {
                if (k < 2) v2::sytrd_reflector_pack_at(A, AQ, u, n, t % n, t / n, tv, tqv);
                else v2::sytrd_reflector_pack_at(A, AQ, u, n, t % n, t / n);
            }
            if (k >= 2 && lid < n) v2::sytrd_tau_pack_at(tv, v2::vec(tq + (std::ptrdiff_t(b) * 6 + k + 2) * n, n), u, n, lid);
        }
    });
    const auto gq = this->dev.down(aq, hq.size());
    const auto gt = this->dev.down(tq, htq.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* A = ha.data() + std::size_t(b) * n * lda;
        const T* tv = ht.data() + std::size_t(b) * (n - 1);
        for (int k = 0; k < 4; ++k) {
            const bool upper = k & 1;
            const T* Q = gq.data() + (std::size_t(b) * 4 + k) * n * ldq;
            for (int c = 0; c < n; ++c) {
                for (int r = 0; r < n; ++r) {
                    T ref = T(0);
                    if (!upper && c >= 1 && r > c) ref = A[r + (c - 1) * lda];
                    if (upper && c <= n - 2 && r < c) ref = A[r + (c + 1) * lda];
                    bad += !same_bits(Q[r + c * ldq], ref);
                }
                for (int r = n; r < ldq; ++r) bad += !std::isnan(std::real(LD<T>(Q[r + c * ldq])));
                const T tref = upper ? (c < n - 1 ? tv[c] : T(0)) : (c >= 1 ? tv[c - 1] : T(0));
                const T* tq_fold = gt.data() + (std::size_t(b) * 6 + k) * n;
                if (k < 2) bad += !same_bits(tq_fold[c], tref);
                else bad += !std::isnan(std::real(LD<T>(tq_fold[c])));
                if (k >= 2) bad += !same_bits(gt[(std::size_t(b) * 6 + k + 2) * n + c], tref);
            }
        }
    }
    EXPECT_EQ(bad, 0);
}

// ---- register-column Householder family ----------------------------------------------------

// QL reflectors of a P-frame: reflector k (k < n-1) in column k+1 of R (rows < k), unitary
// H(k) = I - tau_k v v^H with tau_k = (1 - e^{i theta}) / (v^H v) (real: theta = pi), tau_2 = 0.
// Every other entry of R is NaN, so a read outside the support poisons the result.
template <class T> void ql_reflectors(int P, int ldr, int n, std::uint64_t seed, T* R, T* tau) {
    using RT = real_of<T>;
    for (int i = 0; i < P * ldr; ++i) R[i] = nan_v<T>();
    for (int k = 0; k < P; ++k) tau[k] = T(0);
    for (int k = 0; k < n - 1; ++k) {
        long double s = 1;
        for (int r = 0; r < k; ++r) {
            const T v = val<T>(seed + std::uint64_t(k) * 97 + r);
            R[r + (k + 1) * ldr] = v;
            s += std::norm(LD<T>(v));
        }
        if (k == 2) continue;
        if constexpr (is_cplx<T>) {
            const long double th = 0.7L + 0.3L * k;
            tau[k] = T(static_cast<RT>((1 - std::cos(th)) / s), static_cast<RT>(-std::sin(th) / s));
        } else {
            tau[k] = static_cast<T>(2 / s);
        }
    }
}
// M = H(n-2) ... H(0) (P x P, column-major), long double.
template <class T> std::vector<LD<T>> ql_product(int P, int ldr, int n, const T* R, const T* tau) {
    std::vector<LD<T>> M(std::size_t(P) * P, LD<T>(0));
    for (int i = 0; i < P; ++i) M[i + i * P] = LD<T>(1);
    for (int k = 0; k < n - 1; ++k) {
        std::vector<LD<T>> v(P, LD<T>(0));
        for (int r = 0; r < k; ++r) v[r] = wide(R[r + (k + 1) * ldr]);
        v[k] = LD<T>(1);
        const LD<T> t = wide(tau[k]);
        for (int j = 0; j < P; ++j) {
            LD<T> dot = 0;
            for (int r = 0; r < P; ++r) dot += conj_w(v[r]) * M[r + j * P];
            for (int r = 0; r < P; ++r) M[r + j * P] -= v[r] * (t * dot);
        }
    }
    return M;
}

// org2l (Arith) and orm2l (NativeArith) on SgpPartition<P>: lane l < n-1 holds column l of
// M = H(n-2)...H(0); orm2l gives M c on lanes l < n and leaves the pad lanes' columns alone.
template <class T, int P> void run_org2l_orm2l(Dev& dev, int n) {
    constexpr int WG = 64, chunks = WG / P, LDT = P + 1, LDR = P + 3, slm = P * LDT + P;
    std::vector<T> hr(std::size_t(kBatch) * P * LDR), ht(std::size_t(kBatch) * P);
    for (int b = 0; b < kBatch; ++b) {
        if (b < kPeriod) ql_reflectors<T>(P, LDR, n, 500 + b * 7919, hr.data() + std::size_t(b) * P * LDR, ht.data() + std::size_t(b) * P);
        else {
            std::copy_n(hr.data() + std::size_t(b % kPeriod) * P * LDR, P * LDR, hr.data() + std::size_t(b) * P * LDR);
            std::copy_n(ht.data() + std::size_t(b % kPeriod) * P, P, ht.data() + std::size_t(b) * P);
        }
    }
    auto hc = per_item<T>(P * P, 51);
    std::vector<T> hq(std::size_t(kBatch) * P * P, nan_v<T>()), ho(hq);
    T* rg = dev.up(hr);
    T* tg = dev.up(ht);
    T* cg = dev.up(hc);
    T* qg = dev.up(hq);
    T* og = dev.up(ho);
    dev.template run<WG>(kBatch / chunks, std::size_t(chunks) * slm * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const v2::SgpPartition<P> pex{it.get_sub_group()};
        const int lane = pex.lid();
        const int chunk = static_cast<int>(it.get_local_linear_id()) / P;
        const int b = static_cast<int>(it.get_group_linear_id()) * chunks + chunk;
        T* tile = reinterpret_cast<T*>(lm) + chunk * slm;
        const v2::Mat<T, v2::Ext<P>, v2::Ext<P>, v2::Ext<LDT>, v2::Mem::Local> tm{tile, {}, {}, {}};
        const v2::Vec<T, v2::Ext<P>, v2::Unit, v2::Mem::Local> vs{tile + P * LDT, {}};
        for (int c = 0; c < P; ++c) tm(lane, c) = rg[std::ptrdiff_t(b) * P * LDR + lane + c * LDR];
        group_barrier(pex.part);
        const T tau_lane = tg[std::ptrdiff_t(b) * P + lane];

        v2::Frag<T, P, v2::ColPerLane> q;
        v2::org2l(pex, q, vs, tm, tau_lane, n);
        v2::store(q, v2::vec(qg + std::ptrdiff_t(b) * P * P + lane * P, P));

        v2::Frag<T, P, v2::ColPerLane> c;
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < P; ++r) c[r] = cg[std::ptrdiff_t(b) * P * P + r + lane * P];
        v2::orm2l<v2::NativeArith<T>>(pex, c, vs, tm, tau_lane, n, lane < n);
        v2::store(c, v2::vec(og + std::ptrdiff_t(b) * P * P + lane * P, P));
    });
    const auto gq = dev.down(qg, hq.size());
    const auto go = dev.down(og, ho.size());
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const auto M = ql_product<T>(P, LDR, n, hr.data() + std::size_t(b) * P * LDR, ht.data() + std::size_t(b) * P);
        const T* Q = gq.data() + std::size_t(b) * P * P;
        const T* O = go.data() + std::size_t(b) * P * P;
        const T* C = hc.data() + std::size_t(b) * P * P;
        for (int l = 0; l < P; ++l) {
            for (int r = 0; r < P; ++r) {
                if (l < n - 1) worst = std::max(worst, rel(Q[r + l * P], M[r + l * P], 1.0L));
                if (l < n) {
                    LD<T> ref = 0;
                    long double sc = 0;
                    for (int k = 0; k < P; ++k) {
                        ref += M[r + k * P] * wide(C[k + l * P]);
                        sc += std::abs(M[r + k * P]) * std::abs(wide(C[k + l * P]));
                    }
                    worst = std::max(worst, rel(O[r + l * P], ref, sc));
                } else {
                    bad += !same_bits(O[r + l * P], C[r + l * P]);
                }
            }
        }
    }
    EXPECT_EQ(bad, 0) << "orm2l<" << P << "> touched a lane with own == false";
    EXPECT_VERIFY(T, Check::blas, n, worst);
}
TYPED_TEST(DeviceBlasV2SyevDriversTest, Org2lOrm2lPartition) {
    run_org2l_orm2l<TypeParam, 8>(this->dev, 6);
    run_org2l_orm2l<TypeParam, 32>(this->dev, 29);
    run_org2l_orm2l<TypeParam, 32>(this->dev, 32);
}

// Per-item fragment ops on one work-item each (Thread semantics): larf with both arithmetic
// policies, larf_unit, laset, fill, dgmm_lift (r < n only), store_n (r < n only).
TYPED_TEST(DeviceBlasV2SyevDriversTest, FragmentOps) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 32, N = 16, n = 11, k = 9, slots = 7;
    auto hv = per_item<T>(N, 61);
    auto hc = per_item<T>(N, 62);
    auto hz = per_item<R>(N, 63);
    const T tau = val<T>(64);
    std::vector<T> ho(std::size_t(kBatch) * slots * N, nan_v<T>());
    T* v = this->dev.up(hv);
    T* c = this->dev.up(hc);
    R* z = this->dev.up(hz);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch / WG, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const auto vv = v2::assume_published(v2::Vec<T, v2::Ext<N>>{v + std::ptrdiff_t(b) * N, {}});
        const auto out = [&](int s) { return v2::Vec<T, v2::Ext<N>>{o + (std::ptrdiff_t(b) * slots + s) * N, {}}; };
        v2::Frag<T, N, v2::ColPerLane> f, g;
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < N; ++r) f[r] = c[std::ptrdiff_t(b) * N + r];
        g = f;
        v2::larf(f, vv, tau);
        v2::store(f, out(0));
        v2::larf<v2::NativeArith<T>>(g, vv, tau);
        v2::store(g, out(1));
        v2::larf_unit(f, vv, tau, k);
        v2::store(f, out(2));
        v2::laset(f, k, T(2), T(-3));
        v2::store(f, out(3));
        v2::fill(f, T(5));
        v2::dgmm_lift(f, v2::Vec<T, v2::Ext<N>>{c + std::ptrdiff_t(b) * N, {}}, v2::Vec<R, v2::Ext<N>>{z + std::ptrdiff_t(b) * N, {}}, n);
        v2::store(f, out(4));
        v2::store_n(f, out(5), n);
    });
    const auto g = this->dev.down(o, ho.size());
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* V = hv.data() + std::size_t(b) * N;
        const T* C = hc.data() + std::size_t(b) * N;
        const R* Z = hz.data() + std::size_t(b) * N;
        const T* G = g.data() + std::size_t(b) * slots * N;
        LD<T> dot = 0;
        long double sd = 0;
        for (int r = 0; r < N; ++r) {
            dot += conj_w(wide(V[r])) * wide(C[r]);
            sd += std::abs(wide(V[r])) * std::abs(wide(C[r]));
        }
        const LD<T> t = wide(tau);
        std::vector<LD<T>> l1(N);
        for (int r = 0; r < N; ++r) {
            l1[r] = wide(C[r]) - wide(V[r]) * (t * dot);
            const long double sc = std::abs(wide(C[r])) + std::abs(wide(V[r])) * std::abs(t) * sd;
            worst = std::max({worst, rel(G[r], l1[r], sc), rel(G[N + r], l1[r], sc)});
            // larf_unit ignores c: e_k - tau v with v(k) = 1, v(r > k) = 0.
            const LD<T> lu = r < k ? -t * wide(V[r]) : (r == k ? LD<T>(1) - t : LD<T>(0));
            worst = std::max(worst, rel(G[2 * N + r], lu, std::abs(t) * std::abs(wide(V[r])) + 1));
            bad += !same_bits(G[3 * N + r], r == k ? T(-3) : T(2));
            const T lifted = r < n ? T(0) : T(5);
            if (r < n) {
                const LD<T> ref = wide(C[r]) * LD<T>(static_cast<long double>(Z[r]));
                worst = std::max(worst, rel(G[4 * N + r], ref, std::abs(ref)));
                bad += !same_bits(G[5 * N + r], G[4 * N + r]);
            } else {
                bad += !same_bits(G[4 * N + r], lifted);
                bad += !std::isnan(std::real(LD<T>(G[5 * N + r])));
            }
        }
    }
    EXPECT_EQ(bad, 0) << "laset / dgmm_lift beyond n / store_n beyond n";
    EXPECT_VERIFY(T, Check::blas, N, worst);
}

}  // namespace

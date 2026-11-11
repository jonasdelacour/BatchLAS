// SPDX-License-Identifier: MIT
// device_core_view_tests: view.hh (design A.2) against host index maths.
// Guards (agent-guide section 8): ld = m + 3 and inc = 2 (non-natural strides); complex data;
// the never-read triangle of herm_view poisoned with NaN, and a poisoned Im(diag) that DiagReal
// must mask; every (i, j) of PackedTri for n = 1 .. 40 in all four layouts, index and decode;
// a run-time flip / conj switch in both states; Packet through a 16-byte aligned pointer.
// Red set (each break was made once, 2026-11-04, and this suite failed): herm_view reading
// a(i, j) outside the stored triangle (the poisoned triangle surfaces); herm_view (band,
// shifted, refs): the Upper band row index kd - d replaced by d.
#include <tests/device_core_common.hh>
#include <batchlas/device/view.hh>

#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> class DeviceCoreViewTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreViewTest, AllTypes);

template <class T> T cj(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}

using batchlas::Uplo;
struct LC { using type = batchlas::device::PackedTri<Uplo::Lower, batchlas::device::PackOrder::ByCols>; static constexpr Uplo uplo = Uplo::Lower; static int size(int n) { return n * (n + 1) / 2; } };
struct UC { using type = batchlas::device::PackedTri<Uplo::Upper, batchlas::device::PackOrder::ByCols>; static constexpr Uplo uplo = Uplo::Upper; static int size(int n) { return n * (n + 1) / 2; } };
struct LR { using type = batchlas::device::PackedTri<Uplo::Lower, batchlas::device::PackOrder::ByRows>; static constexpr Uplo uplo = Uplo::Lower; static int size(int n) { return n * (n + 1) / 2; } };
struct UR { using type = batchlas::device::PackedTri<Uplo::Upper, batchlas::device::PackOrder::ByRows>; static constexpr Uplo uplo = Uplo::Upper; static int size(int n) { return n * (n + 1) / 2; } };

TEST(DeviceCoreViewHost, PackedTriIndexIsABijectionAndDecodeInvertsIt) {
    auto check = [](auto tag) {
        using P = decltype(tag);
        for (int n = 1; n <= 40; ++n) {
            std::vector<int> seen(P::size(n), 0);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    if (P::uplo == batchlas::Uplo::Lower ? i < j : i > j) continue;
                    const int k = P::type::index(i, j, n);
                    ASSERT_GE(k, 0);
                    ASSERT_LT(k, P::size(n));
                    ++seen[k];
                    const auto rc = P::type::decode(k, n);
                    EXPECT_EQ(rc.i, i) << "n " << n << " k " << k;
                    EXPECT_EQ(rc.j, j) << "n " << n << " k " << k;
                }
            for (int k = 0; k < P::size(n); ++k) EXPECT_EQ(seen[k], 1) << n;
        }
    };
    check(LC{});
    check(UC{});
    check(LR{});
    check(UR{});
    // LAPACK's AP layout (column-major packed): Upper (i, j) at i + j (j + 1) / 2, Lower at
    // i + j (2n - j - 1) / 2.
    EXPECT_EQ((batchlas::device::PackedTri<Uplo::Upper>::index(1, 2, 4)), 1 + 3);
    EXPECT_EQ((batchlas::device::PackedTri<Uplo::Lower>::index(3, 1, 4)), 3 + 1 * (8 - 1 - 1) / 2);
}

TYPED_TEST(DeviceCoreViewTest, ViewsMapEveryElementOnDevice) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int m = 7, n = 5, ld = m + 3, inc = 2;
    const T nanv = batchlas::device::qnan<T>();
    std::vector<T> a_h(std::size_t(ld) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < ld; ++i) a_h[i + j * ld] = i < m ? val<T>(100 + i + 31 * j) : nanv;  // pad rows poisoned
    // A square n x n Hermitian source: lower triangle stored, upper poisoned, Im(diag) poisoned.
    constexpr int s = 6, lds = s + 3;
    std::vector<T> h_h(std::size_t(lds) * s, nanv);
    for (int j = 0; j < s; ++j)
        for (int i = j; i < s; ++i) h_h[i + j * lds] = val<T>(500 + i + 17 * j);
    for (int j = 0; j < s; ++j)
        if constexpr (is_cplx<T>) h_h[j + j * lds] = T(std::real(h_h[j + j * lds]), std::numeric_limits<R>::quiet_NaN());
    const auto x_h = vals<T>(std::size_t(m) * inc, 9);
    T* a = d.up(a_h);
    T* h = d.up(h_h);
    T* x = d.up(x_h);
    constexpr int kVec = 5 * m, kMat = 4 * m * n, kHerm = 2 * s * s;
    T* vo = d.template alloc<T>(kVec, T(0));
    T* mo = d.template alloc<T>(kMat, T(0));
    T* ho = d.template alloc<T>(kHerm, T(0));
    d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
        const batchlas::device::Vec<T, batchlas::device::Dyn, batchlas::device::Dyn> xv{x, batchlas::device::Dyn{m}, batchlas::device::Dyn{inc}};
        const batchlas::device::Mat<T> av{a, batchlas::device::Dyn{m}, batchlas::device::Dyn{n}, batchlas::device::Dyn{ld}};
        const auto fx = batchlas::device::flip(xv);
        const auto fx_off = batchlas::device::flip(xv, false);
        const auto cx = batchlas::device::conj_if<true>(xv);
        const auto sl = batchlas::device::slice(xv, 1, 3, 2);
        const auto mp = batchlas::device::mapped(xv, [](T v) { return batchlas::device::Arith<T>::add(v, v); });
        for (int i = 0; i < m; ++i) {
            vo[0 * m + i] = fx(i);
            vo[1 * m + i] = fx_off(i);
            vo[2 * m + i] = cx(i);
            vo[3 * m + i] = i < sl.size() ? sl(i) : T(0);
            vo[4 * m + i] = mp(i);
        }
        const auto fr = batchlas::device::flip<batchlas::device::Axis::Rows>(av);
        const auto fc = batchlas::device::flip<batchlas::device::Axis::Cols>(av);
        const auto fb = batchlas::device::flip<batchlas::device::Axis::Both>(av);
        const auto tt = batchlas::device::op_view<batchlas::Transpose::ConjTrans>(av);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                mo[0 * m * n + i + j * m] = fr(i, j);
                mo[1 * m * n + i + j * m] = fc(i, j);
                mo[2 * m * n + i + j * m] = fb(i, j);
                mo[3 * m * n + j + i * n] = tt(j, i);  // op(A) is n x m
            }
        const batchlas::device::Mat<T> hv{h, batchlas::device::Dyn{s}, batchlas::device::Dyn{s}, batchlas::device::Dyn{lds}};
        const auto hh = batchlas::device::herm_view<batchlas::Uplo::Lower, true>(hv);
        const auto sy = batchlas::device::sym_view<batchlas::Uplo::Lower>(hv);
        for (int j = 0; j < s; ++j)
            for (int i = 0; i < s; ++i) {
                ho[i + j * s] = hh(i, j);
                ho[s * s + i + j * s] = i == j ? T(0) : sy(i, j);
            }
    });
    const auto V = d.down(vo, kVec), M = d.down(mo, kMat), H = d.down(ho, kHerm);
    for (int i = 0; i < m; ++i) {
        EXPECT_TRUE(same_bits(V[0 * m + i], x_h[(m - 1 - i) * inc])) << i;
        EXPECT_TRUE(same_bits(V[1 * m + i], x_h[i * inc])) << i;
        EXPECT_TRUE(same_bits(V[2 * m + i], cj(x_h[i * inc]))) << i;
        if (i < 3) EXPECT_TRUE(same_bits(V[3 * m + i], x_h[(1 + 2 * i) * inc])) << i;
        EXPECT_LE(err(V[4 * m + i], wide(x_h[i * inc]) * LD<T>(2)), tol<T>()) << i;
    }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            EXPECT_TRUE(same_bits(M[0 * m * n + i + j * m], a_h[(m - 1 - i) + j * ld])) << i << "," << j;
            EXPECT_TRUE(same_bits(M[1 * m * n + i + j * m], a_h[i + (n - 1 - j) * ld])) << i << "," << j;
            EXPECT_TRUE(same_bits(M[2 * m * n + i + j * m], a_h[(m - 1 - i) + (n - 1 - j) * ld])) << i << "," << j;
            EXPECT_TRUE(same_bits(M[3 * m * n + j + i * n], cj(a_h[i + j * ld]))) << i << "," << j;
        }
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < s; ++i) {
            const T lo = i >= j ? h_h[i + j * lds] : cj(h_h[j + i * lds]);
            const T want = i == j ? T(std::real(h_h[i + i * lds])) : lo;
            EXPECT_TRUE(same_bits(H[i + j * s], want)) << "herm " << i << "," << j;
            if (i != j) EXPECT_TRUE(same_bits(H[s * s + i + j * s], i >= j ? h_h[i + j * lds] : h_h[j + i * lds])) << "sym " << i << "," << j;
        }
}

TYPED_TEST(DeviceCoreViewTest, PacketMovesOneAlignedUnit) {
    using T = TypeParam;
    auto& d = this->dev;
    using P = batchlas::device::Packet<T, 16>;
    const auto src_h = vals<T>(64, 13);
    T* src = d.up(src_h);
    T* dst = d.template alloc<T>(64, T(0));
    d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
        for (int k = 0; k < 64; k += P::n) P::load(src + k).store(dst + k);
    });
    const auto got = d.down(dst, 64);
    for (int k = 0; k < 64; ++k) EXPECT_TRUE(same_bits(got[k], src_h[k])) << k;
}


// herm_view<UL, Storage::Band> of LAPACK band storage and herm_view<UL, Storage::Full,
// alg::TwoLoads>, against a host Hermitian built from the full matrix; shifted; refs. The band
// array's unused corner and the matrix's unstored triangle are NaN-poisoned.
TYPED_TEST(DeviceCoreViewTest, HermBandAndTwoLoadViewsShiftedRefs) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int n = 10, kd = 3, ld = n + 3, ldab = kd + 1 + 2;
    const T poison = [] {
        constexpr R q = std::numeric_limits<R>::quiet_NaN();
        if constexpr (is_cplx<T>) return T(q, q);
        else return T(q);
    }();
    auto a_h = vals<T>(std::size_t(ld) * n, 90);  // Im(diag) != 0: must be dropped
    std::vector<T> lo_h(a_h), up_h(a_h);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if (i < j) lo_h[j * ld + i] = poison;
            if (i > j) up_h[j * ld + i] = poison;
        }
    std::vector<T> abl(std::size_t(ldab) * n, poison), abu(std::size_t(ldab) * n, poison);
    for (int j = 0; j < n; ++j)
        for (int i = j; i <= std::min(n - 1, j + kd); ++i) abl[j * ldab + (i - j)] = a_h[j * ld + i];
    for (int j = 0; j < n; ++j)
        for (int i = std::max(0, j - kd); i <= j; ++i) abu[j * ldab + (kd + i - j)] = up_h[j * ld + i];
    T* lo = d.up(lo_h);
    T* ab = d.up(abl);
    T* au = d.up(abu);
    T* out = d.template alloc<T>(3 * n * n, T(0));
    const auto x_h = vals<T>(n, 91);
    T* x = d.up(x_h);
    T* sh = d.template alloc<T>(n + 2, T(0));
    d.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int g = static_cast<int>(it.get_local_linear_id());
        const auto two = batchlas::device::herm_view<Uplo::Lower, batchlas::device::Storage::Full, batchlas::device::alg::TwoLoads>(batchlas::device::mat(lo, n, n, ld));
        const auto bl = batchlas::device::herm_view<Uplo::Lower, batchlas::device::Storage::Band>(batchlas::device::mat(ab, ldab, n, ldab), n, kd);
        const auto bu = batchlas::device::herm_view<Uplo::Upper, batchlas::device::Storage::Band>(batchlas::device::mat(au, ldab, n, ldab), n, kd);
        for (int e = g; e < n * n; e += 32) {
            const int i = e % n, j = e / n;
            out[e] = two(i, j);
            out[n * n + e] = bl(i, j);
            out[2 * n * n + e] = bu(i, j);
        }
        const auto s = batchlas::device::shifted(batchlas::device::vec(x, n), 1, n + 2);
        if (g < n + 2) sh[g] = s(g);
    });
    const auto o = d.down(out, 3 * n * n), gs = d.down(sh, n + 2);
    auto herm = [&](const std::vector<T>& m, bool lower, int i, int j) {
        const bool stored = lower ? i >= j : i <= j;
        T v = stored ? m[j * ld + i] : m[i * ld + j];
        if constexpr (is_cplx<T>) {
            if (i == j) return T(v.real(), R(0));
            if (!stored) v = std::conj(v);
        }
        return v;
    };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const int e = j * n + i;
            EXPECT_TRUE(same_bits(o[e], herm(lo_h, true, i, j))) << "alg::TwoLoads " << i << "," << j;
            const bool in_band = std::abs(i - j) <= kd;
            EXPECT_TRUE(same_bits(o[n * n + e], in_band ? herm(lo_h, true, i, j) : T(0))) << "Band Lower " << i << "," << j;
            EXPECT_TRUE(same_bits(o[2 * n * n + e], in_band ? herm(up_h, false, i, j) : T(0))) << "Band Upper " << i << "," << j;
        }
    for (int i = 0; i < n + 2; ++i) EXPECT_TRUE(same_bits(gs[i], (i >= 1 && i <= n) ? x_h[i - 1] : T(0))) << "shifted " << i;
    T p = val<T>(1), q = val<T>(2), r = val<T>(3);
    const auto rf = batchlas::device::refs(p, q, r);
    EXPECT_EQ(rf.size(), 3);
    rf(1) = rf(2);
    EXPECT_TRUE(same_bits(q, r));
}

}  // namespace

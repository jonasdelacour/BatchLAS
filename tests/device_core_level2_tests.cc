// SPDX-License-Identifier: MIT
// device_core_level2_tests: level2.hh v2.2 additions (design A.2) against host references.
// Guards (agent-guide section 8): complex data with nonzero imaginary parts, Im(diag) != 0 on
// the Hermitian inputs (her2 must zero it on a triangle and keep it on Full); a non-natural
// ld = m + 3 whose pad is poisoned (NaN) and must not be read or written; rectangular m != n
// with m, n not multiples of the team (m = 21, n = 13); ColDot<4> so a butterfly crossing its
// 4-lane block is visible; trsv diagonals 2 .. 5 with Im(diag) != 0, so tri_diag::Real differs
// from Div by more than the tolerance; gbmv with KL != KU and a band that clips at both edges. Red set
// (each break made once, 2026-11-04, and this suite failed): her2 adding conj(alpha) to the
// first term instead of the second.
#include <tests/device_core_common.hh>
#include <batchlas/device/level2.hh>

#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> class DeviceCoreLevel2Test : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreLevel2Test, AllTypes);

template <class T> T nan_of() {
    using R = real_of<T>;
    if constexpr (is_cplx<T>) return T(std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN());
    else return std::numeric_limits<R>::quiet_NaN();
}
// m x n column-major with ld = m + 3; the pad rows poisoned.
template <class T> std::vector<T> padded(int m, int n, std::uint64_t seed) {
    const int ld = m + 3;
    std::vector<T> a(std::size_t(ld) * n, nan_of<T>());
    const auto v = vals<T>(std::size_t(m) * n, seed);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) a[j * ld + i] = v[j * m + i];
    return a;
}
template <class T> LD<T> cj(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}

TYPED_TEST(DeviceCoreLevel2Test, GemvTeamPerRowColDotAndAlphaRight) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int m = 21, n = 13, ld = m + 3, WG = 64;
    const auto a_h = padded<T>(m, n, 1);
    const auto xn_h = vals<T>(n, 2), xt_h = vals<T>(m, 3), y0n_h = vals<T>(m, 4), y0t_h = vals<T>(n, 5);
    const T alpha = val<T>(6), beta = val<T>(7);
    T* a = d.up(a_h);
    T* xn = d.up(xn_h);
    T* xt = d.up(xt_h);
    T* yn = d.up(y0n_h);
    T* yt = d.up(y0t_h);
    T* yc = d.up(y0t_h);
    T* yr = d.template alloc<T>(m, T(0));
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::WorkGroup<WG, 32> ex{it};
        const auto A = dv::mat(a, m, n, ld);
        dv::gemv<Transpose::NoTrans, dv::alg::TeamPerRow>(ex, alpha, A, dv::assume_published(dv::vec(xn, n)), beta, dv::vec(yn, m));
        dv::gemv<Transpose::Trans, dv::alg::ColDot<4>>(ex, alpha, A, dv::assume_published(dv::vec(xt, m)), beta, dv::vec(yt, n));
        dv::gemv<Transpose::ConjTrans, dv::alg::ColDot<4>>(ex, alpha, A, dv::assume_published(dv::vec(xt, m)), beta, dv::vec(yc, n));
        dv::gemv<Transpose::NoTrans, void, dv::arith::Explicit, dv::epi::Sum, dv::alpha_side::Right>(
            ex, alpha, A, dv::assume_published(dv::vec(xn, n)), dv::Zero{}, dv::vec(yr, m));
    });
    const auto gn = d.down(yn, m), gt = d.down(yt, n), gc = d.down(yc, n), gr = d.down(yr, m);
    for (int r = 0; r < m; ++r) {
        LD<T> acc = 0;
        for (int c = 0; c < n; ++c) acc += wide(a_h[c * ld + r]) * wide(xn_h[c]);
        EXPECT_LE(err(gn[r], wide(alpha) * acc + wide(beta) * wide(y0n_h[r])), tol<T>(8)) << "TeamPerRow " << r;
        EXPECT_LE(err(gr[r], acc * wide(alpha)), tol<T>(8)) << "alpha Right " << r;
    }
    for (int c = 0; c < n; ++c) {
        LD<T> at = 0, ac = 0;
        for (int r = 0; r < m; ++r) {
            at += wide(a_h[c * ld + r]) * wide(xt_h[r]);
            ac += cj<T>(wide(a_h[c * ld + r])) * wide(xt_h[r]);
        }
        EXPECT_LE(err(gt[c], wide(alpha) * at + wide(beta) * wide(y0t_h[c])), tol<T>(8)) << "ColDot T " << c;
        EXPECT_LE(err(gc[c], wide(alpha) * ac + wide(beta) * wide(y0t_h[c])), tol<T>(8)) << "ColDot C " << c;
    }
}

TYPED_TEST(DeviceCoreLevel2Test, GerFlatAndRegisterRow) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int m = 21, n = 13, ld = m + 3, WG = 64;
    const auto a_h = padded<T>(m, n, 11);
    const auto x_h = vals<T>(m, 12), y_h = vals<T>(n, 13);
    const T alpha = val<T>(14);
    T* a = d.up(a_h);
    T* b = d.up(a_h);
    T* x = d.up(x_h);
    T* y = d.up(y_h);
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::WorkGroup<WG, 32> ex{it};
        dv::gerc<dv::alg::Flat>(ex, alpha, dv::assume_published(dv::vec(x, m)), dv::assume_published(dv::vec(y, n)), dv::mat(a, m, n, ld));
        const int r = ex.lid();
        if (r < m) {  // the register-row form: item r holds row r of B
            dv::Frag<T, 16> row;
            dv::load_row(row, dv::mat(b, m, n, ld), r, n);
            dv::ger_row<true>(alpha, x[r], dv::vec(y, n), row, n);
            dv::store_row(row, dv::mat(b, m, n, ld), r, n);
        }
    });
    const auto ga = d.down(a, a_h.size()), gb = d.down(b, a_h.size());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < ld; ++i) {
            if (i >= m) {
                EXPECT_TRUE(std::isnan(static_cast<double>(std::abs(ga[j * ld + i])))) << "pad written";
                continue;
            }
            const LD<T> ref = wide(a_h[j * ld + i]) + wide(x_h[i]) * (wide(alpha) * cj<T>(wide(y_h[j])));
            EXPECT_LE(err(ga[j * ld + i], ref), tol<T>(4)) << i << "," << j;
            EXPECT_LE(err(gb[j * ld + i], ref), tol<T>(4)) << "row " << i << "," << j;
        }
}

TYPED_TEST(DeviceCoreLevel2Test, Her2OnATriangleAndOnTheFullSquare) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int n = 19, ld = n + 3, WG = 64;
    const auto a_h = padded<T>(n, n, 21);  // Im(diag) != 0 (random)
    const auto x_h = vals<T>(n, 22), y_h = vals<T>(n, 23);
    const T alpha = val<T>(24);
    T* al = d.up(a_h);
    T* af = d.up(a_h);
    T* x = d.up(x_h);
    T* y = d.up(y_h);
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::WorkGroup<WG, 32> ex{it};
        const auto X = dv::assume_published(dv::vec(x, n));
        const auto Y = dv::assume_published(dv::vec(y, n));
        dv::her2<dv::Part::Lower>(ex, alpha, X, Y, dv::mat(al, n, n, ld));
        dv::her2<dv::Part::Full, dv::alg::TeamPerRow>(ex, alpha, X, Y, dv::mat(af, n, n, ld));
    });
    const auto gl = d.down(al, a_h.size()), gf = d.down(af, a_h.size());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const LD<T> upd = wide(alpha) * wide(x_h[i]) * cj<T>(wide(y_h[j])) + cj<T>(wide(alpha)) * wide(y_h[i]) * cj<T>(wide(x_h[j]));
            LD<T> ref = wide(a_h[j * ld + i]) + upd;
            EXPECT_LE(err(gf[j * ld + i], ref), tol<T>(4)) << "Full " << i << "," << j;
            if (i < j) {
                EXPECT_TRUE(same_bits(gl[j * ld + i], a_h[j * ld + i])) << "upper written " << i << "," << j;
                continue;
            }
            if constexpr (is_cplx<T>)
                if (i == j) ref = LD<T>(ref.real(), 0);
            EXPECT_LE(err(gl[j * ld + i], ref), tol<T>(4)) << "Lower " << i << "," << j;
            if constexpr (is_cplx<T>)
                if (i == j) EXPECT_EQ(gl[j * ld + i].imag(), real_of<T>(0)) << "Im(diag) " << i;
        }
}

TYPED_TEST(DeviceCoreLevel2Test, TrsvDiagonalPoliciesAndLanePerRow) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int n = 13, ld = n + 3, N = 16;
    auto a_h = padded<T>(n, n, 31);
    for (int i = 0; i < n; ++i) a_h[i * ld + i] = T(R(2) + R(0.25) * R(i)) + val<T>(500 + i) * R(0.05);
    const auto b_h = vals<T>(n, 32);
    T* a = d.up(a_h);
    T* outs = d.template alloc<T>(5 * n, T(0));
    T* b = d.up(b_h);
    d.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto A = dv::mat(a, n, n, ld);
        const dv::Partition<16> part{it.get_sub_group()};
        const int l = part.lid();
        T xl = l < n ? b[l] : T(0);
        dv::trsv<Uplo::Lower, Transpose::NoTrans, dv::tri_diag::Div, dv::alg::LanePerRow>(part, A, xl, n);
        if (it.get_local_linear_id() < 16 && l < n) outs[4 * n + l] = xl;
        if (it.get_local_linear_id() == 0) {
            const dv::Thread th;
            dv::Frag<T, N> f0, f1, f2, f3;
            dv::load(f0, dv::vec(b, n), n);
            f1 = f0;
            f2 = f0;
            f3 = f0;
            dv::trsv<Uplo::Lower, Transpose::NoTrans, dv::tri_diag::Div>(th, A, f0, n);
            dv::trsv<Uplo::Lower, Transpose::NoTrans, dv::tri_diag::Recip>(th, A, f1, n);
            dv::trsv<Uplo::Lower, Transpose::ConjTrans, dv::tri_diag::Real>(th, A, f2, n);
            dv::trsv<Uplo::Lower, Transpose::NoTrans, dv::tri_diag::Unit>(th, A, f3, n);
            for (int i = 0; i < n; ++i) {
                outs[i] = f0[i];
                outs[n + i] = f1[i];
                outs[2 * n + i] = f2[i];
                outs[3 * n + i] = f3[i];
            }
        }
    });
    const auto g = d.down(outs, 5 * n);
    // Host references: forward substitution (policies Div / Recip / Unit), and op(A) = A^H with
    // the diagonal's real part (Real), back substitution.
    auto Aat = [&](int i, int j) { return wide(a_h[j * ld + i]); };
    std::vector<LD<T>> div(n), unit(n), real(n);
    for (int i = 0; i < n; ++i) {
        LD<T> s = wide(b_h[i]), su = wide(b_h[i]);
        for (int j = 0; j < i; ++j) {
            s -= Aat(i, j) * div[j];
            su -= Aat(i, j) * unit[j];
        }
        div[i] = s / Aat(i, i);
        unit[i] = su;
    }
    for (int i = n - 1; i >= 0; --i) {
        LD<T> s = wide(b_h[i]);
        for (int j = i + 1; j < n; ++j) s -= cj<T>(Aat(j, i)) * real[j];
        if constexpr (is_cplx<T>) real[i] = s / LD<T>(Aat(i, i).real());
        else real[i] = s / Aat(i, i);
    }
    for (int i = 0; i < n; ++i) {
        const double sc = std::max(1.0, static_cast<double>(std::abs(div[i])));
        EXPECT_LE(err(g[i], div[i]), tol<T>(64) * sc) << "Div " << i;
        EXPECT_LE(err(g[n + i], div[i]), tol<T>(64) * sc) << "Recip " << i;
        EXPECT_LE(err(g[2 * n + i], real[i]), tol<T>(64) * std::max(1.0, static_cast<double>(std::abs(real[i])))) << "Real " << i;
        EXPECT_LE(err(g[3 * n + i], unit[i]), tol<T>(64) * std::max(1.0, static_cast<double>(std::abs(unit[i])))) << "Unit " << i;
        EXPECT_LE(err(g[4 * n + i], div[i]), tol<T>(64) * sc) << "LanePerRow " << i;
    }
}

TYPED_TEST(DeviceCoreLevel2Test, DgmmBothSidesAndGbmv) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int m = 21, n = 13, ld = m + 3, WG = 64, KL = 1, KU = 2, ldb = KL + KU + 1 + 2;
    const auto b_h = padded<T>(m, n, 41);
    std::vector<R> dl_h(m), dr_h(n);
    for (int i = 0; i < m; ++i) dl_h[i] = uval<R>(4200 + i);
    for (int j = 0; j < n; ++j) dr_h[j] = uval<R>(4300 + j);
    const auto dc_h = vals<T>(n, 44);
    std::vector<T> ab_h(std::size_t(ldb) * n, nan_of<T>());
    const auto abv = vals<T>(std::size_t(ldb) * n, 45);
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < KL + KU + 1; ++k) {
            const int i = k - KU + j;  // A(i, j) = ab(KU + i - j, j)
            if (i >= 0 && i < m) ab_h[j * ldb + k] = abv[j * ldb + k];
        }
    const auto x_h = vals<T>(n, 46), y0_h = vals<T>(m, 47);
    const T alpha = val<T>(48), beta = val<T>(49);
    T* bl = d.up(b_h);
    T* br = d.up(b_h);
    T* cc = d.template alloc<T>(b_h.size(), nan_of<T>());
    R* dl = d.up(dl_h);
    R* dr = d.up(dr_h);
    T* dc = d.up(dc_h);
    T* ab = d.up(ab_h);
    T* x = d.up(x_h);
    T* y = d.up(y0_h);
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::WorkGroup<WG, 32> ex{it};
        dv::dgmm<Side::Left>(ex, dv::vec(dl, m), dv::mat(bl, m, n, ld));
        dv::dgmm<Side::Right>(ex, dv::vec(dr, n), dv::mat(br, m, n, ld));
        dv::dgmm<Side::Right>(ex, dv::vec(dc, n), dv::mat(bl, m, n, ld), dv::mat(cc, m, n, ld));
        dv::gbmv<KL, KU>(ex, alpha, dv::mat(ab, ldb, n, ldb), dv::assume_published(dv::vec(x, n)), beta, dv::vec(y, m));
    });
    const auto gl = d.down(bl, b_h.size()), gr = d.down(br, b_h.size()), gc = d.down(cc, b_h.size()), gy = d.down(y, m);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const LD<T> l = wide(b_h[j * ld + i]) * static_cast<long double>(dl_h[i]);
            EXPECT_LE(err(gl[j * ld + i], l), tol<T>()) << "Left " << i << "," << j;
            EXPECT_LE(err(gr[j * ld + i], wide(b_h[j * ld + i]) * static_cast<long double>(dr_h[j])), tol<T>()) << "Right";
            EXPECT_LE(err(gc[j * ld + i], wide(gl[j * ld + i]) * wide(dc_h[j])), tol<T>(2)) << "C := B D";
        }
    for (int i = 0; i < m; ++i) {
        LD<T> acc = 0;
        for (int j = std::max(0, i - KL); j <= std::min(n - 1, i + KU); ++j) acc += wide(ab_h[j * ldb + KU + i - j]) * wide(x_h[j]);
        EXPECT_LE(err(gy[i], wide(alpha) * acc + wide(beta) * wide(y0_h[i])), tol<T>(4)) << "gbmv " << i;
    }
}

}  // namespace

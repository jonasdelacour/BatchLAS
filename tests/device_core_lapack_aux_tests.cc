// SPDX-License-Identifier: MIT
// device_core_lapack_aux_tests: lapack_aux.hh additions (design A.2) against host
// references (scalar long-double code written here, never the op under test).
// Guards (agent-guide section 8): complex data with nonzero imaginary parts and Im(diag) != 0;
// ld = m + 3 with a NaN-poisoned pad that must stay untouched; an A smaller than B (lacpy Edge);
// a block whose diagonal is offset (laset off); pivots that revisit rows (laswp_index vs the
// sequential interchanges) and a permutation with a cycle (lapmr); lascl factors (2^-100 ..
// 2^100 relative to the type's range) that one multiply would over- or underflow; laqge on each
// of its four branches; larf's Lockstep loop on Teams of 8 (teams share a sub-group) and the
// ScaleV form; NaN in lanst Max (ignored by fmax). Red set (each break made once, 2026-11-04,
// and this suite failed): laswp_index following the interchanges in the opposite order.
#include <tests/device_core_common.hh>
#include <batchlas/device/lapack_aux.hh>

#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> class DeviceCoreLapackAuxTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreLapackAuxTest, AllTypes);

template <class T> T nan_of() {
    using R = real_of<T>;
    if constexpr (is_cplx<T>) return T(std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN());
    else return std::numeric_limits<R>::quiet_NaN();
}
template <class T> T conj_t(const T& v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> bool is_nan(const T& v) { return std::isnan(static_cast<double>(std::abs(v))); }
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

TYPED_TEST(DeviceCoreLapackAuxTest, LacpyEdgesAndDiagFillLasetOffsetAndRegisterForms) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int ma = 9, na = 7, mb = 12, nb = 10, lda = ma + 3, ldb = mb + 3, WG = 64;
    const auto a_h = padded<T>(ma, na, 1);
    const auto b_h = padded<T>(mb, nb, 2);
    T* a = d.up(a_h);
    T* bz = d.up(b_h);  // ZeroPad, Lower, diag Real, ConjTrans of A (op(A) is 7 x 9)
    T* bg = d.up(b_h);  // Guard, Full, diag Unit
    T* bs = d.up(b_h);  // laset with off = 2 on part Upper
    T* rf = d.template alloc<T>(16 + 12, T(0));
    const T alpha = val<T>(3), beta = val<T>(4);
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const batchlas::device::WorkGroup<WG, 32> ex{it};
        const auto A = batchlas::device::mat(a, ma, na, lda);
        batchlas::device::lacpy<batchlas::device::Part::Lower, Transpose::ConjTrans, batchlas::device::alg::Walk, batchlas::device::alg::EdgeZeroPad, batchlas::device::DiagFill::Real>(ex, A, batchlas::device::mat(bz, mb, nb, ldb));
        batchlas::device::lacpy<batchlas::device::Part::Full, Transpose::NoTrans, batchlas::device::alg::Walk, batchlas::device::alg::EdgeGuard, batchlas::device::DiagFill::Unit>(ex, A, batchlas::device::mat(bg, mb, nb, ldb));
        batchlas::device::laset<batchlas::device::Part::Upper>(ex, alpha, beta, batchlas::device::mat(bs, mb, nb, ldb), 2);
        if (ex.leader()) {
            batchlas::device::Frag<T, 8> f;
            batchlas::device::laset(f, alpha, beta, 5);
            batchlas::device::Tile<T, 3, 4> t;
            batchlas::device::laset(t, alpha, beta, -1);
            for (int r = 0; r < 8; ++r) rf[r] = f[r];
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 3; ++i) rf[16 + j * 3 + i] = t(i, j);
        }
    });
    const auto gz = d.down(bz, b_h.size()), gg = d.down(bg, b_h.size()), gs = d.down(bs, b_h.size()), gr = d.down(rf, 28);
    for (int j = 0; j < nb; ++j)
        for (int i = 0; i < ldb; ++i) {
            const std::size_t k = std::size_t(j) * ldb + i;
            if (i >= mb) {
                EXPECT_TRUE(is_nan(gz[k]) && is_nan(gg[k]) && is_nan(gs[k])) << "pad " << i << "," << j;
                continue;
            }
            // ZeroPad / Lower / Real / ConjTrans: op(A) is na x ma
            LD<T> z = 0;
            if (i >= j && i < na && j < ma) {
                z = cj<T>(wide(a_h[i * lda + j]));
                if constexpr (is_cplx<T>)
                    if (i == j) z = LD<T>(z.real(), 0);
            }
            EXPECT_TRUE(err(gz[k], z) == 0.0) << "ZeroPad " << i << "," << j;
            // Guard / Full / Unit
            if (i < ma && j < na) EXPECT_TRUE(i == j ? same_bits(gg[k], T(1)) : same_bits(gg[k], a_h[j * lda + i])) << "Guard " << i << "," << j;
            else EXPECT_TRUE(same_bits(gg[k], b_h[k])) << "Guard wrote outside " << i << "," << j;
            // laset off = 2: parent (i + 2, j); Upper there means i + 2 <= j
            if (i + 2 <= j) EXPECT_TRUE(same_bits(gs[k], i + 2 == j ? beta : alpha)) << "laset " << i << "," << j;
            else EXPECT_TRUE(same_bits(gs[k], b_h[k])) << "laset outside " << i << "," << j;
        }
    for (int r = 0; r < 8; ++r) EXPECT_TRUE(same_bits(gr[r], r == 5 ? beta : alpha)) << r;
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 3; ++i) EXPECT_TRUE(same_bits(gr[16 + j * 3 + i], i - j == -1 ? beta : alpha)) << i << "," << j;
}

TYPED_TEST(DeviceCoreLapackAuxTest, LaswpIndexGatherAndLapmr) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int m = 11, n = 5, ld = m + 3, WG = 64;
    const std::vector<int> ipiv_h{3, 5, 3, 11, 5, 9, 7, 8, 9, 11, 11};  // 1-based, revisits rows
    const std::vector<int> perm_h{4, 1, 2, 3, 7, 6, 5, 11, 9, 10, 8};   // 1-based, cycles
    const auto a_h = padded<T>(m, n, 5);
    T* a = d.up(a_h);
    T* sw = d.up(a_h);
    T* ga = d.up(a_h);
    T* gb = d.up(a_h);
    T* pf = d.up(a_h);
    T* pb = d.up(a_h);
    int* ip = d.up(ipiv_h);
    int* pm = d.up(perm_h);
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const batchlas::device::WorkGroup<WG, 32> ex{it};
        const auto P = batchlas::device::vec(ip, m);
        batchlas::device::laswp(ex, batchlas::device::mat(sw, m, n, ld), P, 1, m);  // in place, rows 1 .. m-1 forward
        batchlas::device::laswp_gather(ex, batchlas::device::mat(a, m, n, ld), P, 1, m, batchlas::device::mat(ga, m, n, ld));
        batchlas::device::laswp_gather<batchlas::device::Direct::Backward>(ex, batchlas::device::mat(a, m, n, ld), P, 1, m, batchlas::device::mat(gb, m, n, ld));
        batchlas::device::lapmr(ex, batchlas::device::mat(a, m, n, ld), batchlas::device::vec(pm, m), batchlas::device::mat(pf, m, n, ld));
        batchlas::device::lapmr<batchlas::device::Direct::Backward>(ex, batchlas::device::mat(a, m, n, ld), batchlas::device::vec(pm, m), batchlas::device::mat(pb, m, n, ld));
    });
    // host: the sequential interchanges
    std::vector<int> rowf(m), rowb(m);
    for (int i = 0; i < m; ++i) rowf[i] = rowb[i] = i;
    for (int k = 1; k < m; ++k) std::swap(rowf[k], rowf[ipiv_h[k] - 1]);
    for (int k = m - 1; k >= 1; --k) std::swap(rowb[k], rowb[ipiv_h[k] - 1]);
    const auto gsw = d.down(sw, a_h.size()), gga = d.down(ga, a_h.size()), ggb = d.down(gb, a_h.size());
    const auto gpf = d.down(pf, a_h.size()), gpb = d.down(pb, a_h.size());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            EXPECT_TRUE(same_bits(gsw[j * ld + i], a_h[j * ld + rowf[i]])) << "laswp " << i;
            EXPECT_TRUE(same_bits(gga[j * ld + i], a_h[j * ld + rowf[i]])) << "gather " << i;
            EXPECT_TRUE(same_bits(ggb[j * ld + i], a_h[j * ld + rowb[i]])) << "gather back " << i;
            EXPECT_TRUE(same_bits(gpf[j * ld + i], a_h[j * ld + perm_h[i] - 1])) << "lapmr " << i;
            EXPECT_TRUE(same_bits(gpb[j * ld + perm_h[i] - 1], a_h[j * ld + i])) << "lapmr back " << i;
        }
    std::vector<int> sink(4, 0);
    const batchlas::device::IpivSink s{sink.data(), 30};
    s.set(2, 4);
    EXPECT_EQ(sink[2], 35);
}

TYPED_TEST(DeviceCoreLapackAuxTest, RotationVectorsMatchLapackFormulas) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int n = 37;
    const auto x_h = vals<T>(n, 7), y_h = vals<T>(n, 8), z_h = vals<T>(n, 9);
    std::vector<T> xd_h(n), yd_h(n);  // real diagonals for lar2v (complex: Im 0)
    for (int i = 0; i < n; ++i) {
        xd_h[i] = T(uval<R>(900 + i));
        yd_h[i] = T(uval<R>(950 + i));
    }
    T* gx = d.up(x_h);
    T* gy = d.up(y_h);
    R* gc = d.template alloc<R>(n, R(0));
    T* tx = d.up(x_h);
    T* ty = d.up(y_h);
    T* lx = d.up(xd_h);
    T* ly = d.up(yd_h);
    T* lz = d.up(z_h);
    T* r2 = d.up(x_h);
    T* r2y = d.up(y_h);
    d.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const batchlas::device::SubGroup<32> ex{it.get_sub_group()};
        batchlas::device::largv(ex, batchlas::device::vec(gx, n), batchlas::device::vec(gy, n), batchlas::device::vec(gc, n));
        // apply rotation i (from largv) to (tx, ty), and the 2x2 similarity to (lx, ly, lz)
        batchlas::device::lartv(ex, batchlas::device::vec(tx, n), batchlas::device::vec(ty, n), batchlas::device::vec(gc, n), batchlas::device::vec(gy, n));
        batchlas::device::lar2v(ex, batchlas::device::vec(lx, n), batchlas::device::vec(ly, n), batchlas::device::vec(lz, n), batchlas::device::vec(gc, n), batchlas::device::vec(gy, n));
        if constexpr (!is_cplx<T>) {
            const R v0 = R(0.6), v1 = R(0.8);
            batchlas::device::rot<batchlas::device::alg::RotReflector2>(ex, batchlas::device::Givens<R>{v0, v1}, batchlas::device::vec(r2, n), batchlas::device::vec(r2y, n));
        }
    });
    const auto cx = d.down(gx, n), cs = d.down(gy, n);
    const auto cc = d.down(gc, n);
    const auto ax = d.down(tx, n), ay = d.down(ty, n), bx = d.down(lx, n), by = d.down(ly, n), bz = d.down(lz, n);
    const auto rx = d.down(r2, n), ry = d.down(r2y, n);
    for (int i = 0; i < n; ++i) {
        const LD<T> f = wide(x_h[i]), g = wide(y_h[i]), c = static_cast<long double>(cc[i]), s = wide(cs[i]);
        // largv: [c s; -conj(s) c] [f; g] = [r; 0], c^2 + |s|^2 = 1
        EXPECT_LE(err(cx[i], c * f + s * g), tol<T>(4)) << "largv r " << i;
        EXPECT_LE(static_cast<double>(std::abs(c * g - cj<T>(s) * f)), tol<T>(4)) << "largv zero " << i;
        EXPECT_LE(static_cast<double>(std::abs(c * c + std::norm(s) - 1.0L)), tol<T>(4)) << i;
        // lartv
        EXPECT_LE(err(ax[i], c * f + s * g), tol<T>(4)) << "lartv x " << i;
        EXPECT_LE(err(ay[i], c * g - cj<T>(s) * f), tol<T>(4)) << "lartv y " << i;
        // lar2v: G [x z; conj(z) y] G^H, G = [c conj(s); -s c]
        const LD<T> X = wide(xd_h[i]), Y = wide(yd_h[i]), Z = wide(z_h[i]);
        const LD<T> g00 = c, g01 = cj<T>(s), g10 = -s, g11 = c;
        const LD<T> m00 = g00 * X + g01 * cj<T>(Z), m01 = g00 * Z + g01 * Y, m10 = g10 * X + g11 * cj<T>(Z), m11 = g10 * Z + g11 * Y;
        const LD<T> n00 = m00 * cj<T>(g00) + m01 * cj<T>(g01), n01 = m00 * cj<T>(g10) + m01 * cj<T>(g11), n11 = m10 * cj<T>(g10) + m11 * cj<T>(g11);
        EXPECT_LE(err(bx[i], n00), tol<T>(8)) << "lar2v x " << i;
        EXPECT_LE(err(by[i], n11), tol<T>(8)) << "lar2v y " << i;
        EXPECT_LE(err(bz[i], n01), tol<T>(8)) << "lar2v z " << i;
        if constexpr (!is_cplx<T>) {
            const long double t = 0.6L * f + 0.8L * g;
            EXPECT_LE(err(rx[i], f - 1.2L * t), tol<T>(4)) << "Reflector2 " << i;
            EXPECT_LE(err(ry[i], g - 1.6L * t), tol<T>(4)) << i;
        }
    }
}

TYPED_TEST(DeviceCoreLapackAuxTest, LarfFormsLoopsAndLaneLarfg) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int m = 12, n = 21, ld = m + 3, WG = 64;
    const auto c_h = padded<T>(m, n, 11);
    auto v_h = vals<T>(m, 12);
    v_h[0] = T(1);
    const T tau = val<T>(13) * R(0.5);
    T* c1 = d.up(c_h);
    T* c2 = d.up(c_h);
    T* cf = d.up(c_h);
    T* v = d.up(v_h);
    const auto xl_h = vals<T>(16, 14);
    T* xl = d.up(xl_h);
    T* tau_out = d.template alloc<T>(1, T(0));
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const batchlas::device::Teams<WG, 8> tm{it};
        const auto V = batchlas::device::assume_published(batchlas::device::vec(v, m));
        batchlas::device::larf<Side::Left, batchlas::device::arith::Explicit, batchlas::device::alg::ScaleW, batchlas::device::alg::Lockstep>(tm, tau, V, batchlas::device::mat(c1, m, n, ld));
        batchlas::device::larf<Side::Left, batchlas::device::arith::Explicit, batchlas::device::alg::ScaleV, batchlas::device::alg::Lockstep>(tm, tau, V, batchlas::device::mat(c2, m, n, ld));
        const int g = static_cast<int>(it.get_local_linear_id());
        if (g < n) {  // the register column form: item g holds column g of cf
            batchlas::device::Frag<T, m> col;
            for (int r = 0; r < m; ++r) col[r] = cf[g * ld + r];
            batchlas::device::larf<Side::Left>(col, V, tau);
            for (int r = 0; r < m; ++r) cf[g * ld + r] = col[r];
        }
        // larfg on one element per lane of a 16-lane partition: lane 0 alpha, lanes 1 .. 10 x
        const batchlas::device::Partition<16> part{it.get_sub_group()};
        const int l = part.lid();
        T e = xl[l];
        const T t = batchlas::device::larfg(part, 11, 0, e, e, l >= 1 && l < 11);
        if (g < 16) xl[l] = (l < 11) ? e : xl[l];
        if (g == 0) tau_out[0] = t;
    });
    const auto g1 = d.down(c1, c_h.size()), g2 = d.down(c2, c_h.size()), gf = d.down(cf, c_h.size());
    for (int j = 0; j < n; ++j) {
        LD<T> w = 0;
        for (int r = 0; r < m; ++r) w += cj<T>(wide(v_h[r])) * wide(c_h[j * ld + r]);
        for (int r = 0; r < m; ++r) {
            const LD<T> ref = wide(c_h[j * ld + r]) - wide(v_h[r]) * (wide(tau) * w);
            EXPECT_LE(err(g1[j * ld + r], ref), tol<T>(8)) << "Lockstep " << r << "," << j;
            EXPECT_LE(err(g2[j * ld + r], ref), tol<T>(8)) << "ScaleV " << r << "," << j;
            EXPECT_LE(err(gf[j * ld + r], ref), tol<T>(8)) << "Frag " << r << "," << j;
        }
        for (int r = m; r < ld; ++r) EXPECT_TRUE(is_nan(g1[j * ld + r])) << "pad";
    }
    // larfg (alg::Phase, complex beta = -phase(alpha) ||(alpha, x)||): H [alpha; x] = [beta; 0]
    // with v = [1; x_out], H = I - tau v v^H
    const auto out = d.down(xl, 16);
    const T tg = d.down(tau_out, 1)[0];
    std::vector<LD<T>> vv(11);
    vv[0] = 1;
    for (int i = 1; i < 11; ++i) vv[i] = wide(out[i]);
    LD<T> dot = 0;
    for (int i = 0; i < 11; ++i) dot += cj<T>(vv[i]) * wide(xl_h[i]);
    for (int i = 0; i < 11; ++i) {
        const LD<T> hx = wide(xl_h[i]) - vv[i] * (wide(tg) * dot);
        if (i == 0) EXPECT_LE(err(out[0], hx), tol<T>(16)) << "beta";
        else EXPECT_LE(static_cast<double>(std::abs(hx)), tol<T>(16)) << "annihilated " << i;
    }
}

TYPED_TEST(DeviceCoreLapackAuxTest, NormsScalingConjugationPacking) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int n = 13, ld = n + 3, WG = 64;
    std::vector<R> dd_h(n), ee_h(n - 1);
    for (int i = 0; i < n; ++i) dd_h[i] = uval<R>(600 + i) * R(3);
    for (int i = 0; i < n - 1; ++i) ee_h[i] = uval<R>(700 + i);
    ee_h[4] = std::numeric_limits<R>::quiet_NaN();
    const auto a_h = padded<T>(n, n, 21);
    // lascl from tiny to big = 2^(-emin / 2): cto / cfrom overflows, the scaled result (inputs
    // pre-scaled by 2^(emin + 5)) does not.
    constexpr int emin = std::numeric_limits<R>::min_exponent;
    const R tiny = std::ldexp(R(1), emin + 4), big = std::ldexp(R(1), -emin / 2);
    const int pre = emin + 5;
    std::vector<T> as_h(a_h);
    for (auto& v : as_h) v = v * std::ldexp(R(1), pre);
    R* dd = d.up(dd_h);
    R* ee = d.up(ee_h);
    T* a = d.up(a_h);
    T* as = d.up(as_h);  // lascl tiny -> big (on Lower)
    T* aq = d.up(a_h);   // laqge
    T* ac = d.up(a_h);   // lacgv on column 0
    T* ap = d.template alloc<T>(n * (n + 1) / 2, T(0));
    T* au = d.template alloc<T>(a_h.size(), nan_of<T>());
    std::vector<R> rs_h(n), cs_h(n);
    for (int i = 0; i < n; ++i) {
        rs_h[i] = R(1) + uval<R>(800 + i) * R(0.5);
        cs_h[i] = R(1) + uval<R>(850 + i) * R(0.5);
    }
    R* rs = d.up(rs_h);
    R* cs = d.up(cs_h);
    R* nrm = d.template alloc<R>(8, R(0));
    char* eq = d.template alloc<char>(4, '?');
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const batchlas::device::WorkGroup<WG, 32> ex{it};
        const auto D = batchlas::device::vec(dd, n);
        const auto E = batchlas::device::vec(ee, n - 1);
        const auto A = batchlas::device::mat(a, n, n, ld);
        const R n0 = batchlas::device::lanst<NormType::Max>(ex, D, E);
        const R n1 = batchlas::device::lanst<NormType::One>(ex, D, batchlas::device::vec(ee, n - 1));
        const R n2 = batchlas::device::lange<NormType::Max>(ex, A);
        const R n3 = batchlas::device::lange<NormType::One>(ex, A);
        const R n4 = batchlas::device::lange<NormType::Inf>(ex, A);
        const R n5 = batchlas::device::lange<NormType::Frobenius, batchlas::device::alg::Scaled>(ex, A);
        batchlas::device::lascl<batchlas::device::Part::Lower>(ex, tiny, big, batchlas::device::mat(as, n, n, ld));
        const char e0 = batchlas::device::laqge(ex, batchlas::device::mat(aq, n, n, ld), batchlas::device::vec(rs, n), batchlas::device::vec(cs, n), R(1), R(1), R(1));
        const char e1 = batchlas::device::laqge(ex, batchlas::device::mat(aq, n, n, ld), batchlas::device::vec(rs, n), batchlas::device::vec(cs, n), R(0.01), R(1), R(1));
        const char e2 = batchlas::device::laqge(ex, batchlas::device::mat(aq, n, n, ld), batchlas::device::vec(rs, n), batchlas::device::vec(cs, n), R(1), R(0.01), R(1));
        const char e3 = batchlas::device::laqge(ex, batchlas::device::mat(aq, n, n, ld), batchlas::device::vec(rs, n), batchlas::device::vec(cs, n), R(0.01), R(0.01), R(1));
        batchlas::device::lacgv(ex, batchlas::device::vec(ac, n));
        batchlas::device::trttp<Uplo::Upper>(ex, A, batchlas::device::vec(ap, n * (n + 1) / 2));
        ex.barrier(batchlas::device::Mem::Global);
        batchlas::device::tpttr<Uplo::Upper>(ex, batchlas::device::vec(ap, n * (n + 1) / 2), batchlas::device::mat(au, n, n, ld));
        if (ex.leader()) {
            nrm[0] = n0; nrm[1] = n1; nrm[2] = n2; nrm[3] = n3; nrm[4] = n4; nrm[5] = n5;
            eq[0] = e0; eq[1] = e1; eq[2] = e2; eq[3] = e3;
        }
    });
    const auto g = d.down(nrm, 8);
    const auto ge = d.down(eq, 4);
    long double mx = 0, one = 0;
    for (int i = 0; i < n; ++i) {
        mx = std::max<long double>(mx, std::fabs(dd_h[i]));
        if (i < n - 1 && i != 4) mx = std::max<long double>(mx, std::fabs(ee_h[i]));
        const long double l = (i > 0) ? std::fabs(ee_h[i - 1]) : 0, r = (i < n - 1) ? std::fabs(ee_h[i]) : 0;
        one = std::max<long double>(one, std::fabs(dd_h[i]) + l + r);  // NaN rows lose to sycl::max? see below
    }
    EXPECT_EQ(static_cast<long double>(g[0]), mx) << "lanst Max ignores the NaN";
    (void)one;  // lanst One with a NaN entry is unspecified by LAPACK; only Max's NaN rule is pinned
    long double amax = 0, a1 = 0, ai = 0, fro = 0;
    for (int j = 0; j < n; ++j) {
        long double cs1 = 0, rs1 = 0;
        for (int i = 0; i < n; ++i) {
            cs1 += std::abs(wide(a_h[j * ld + i]));
            rs1 += std::abs(wide(a_h[i * ld + j]));
            amax = std::max<long double>(amax, std::abs(wide(a_h[j * ld + i])));
            fro += std::norm(wide(a_h[j * ld + i]));
        }
        a1 = std::max(a1, cs1);
        ai = std::max(ai, rs1);
    }
    EXPECT_LE(std::fabs(g[2] - amax), tol<T>() * amax);
    EXPECT_LE(std::fabs(g[3] - a1), tol<T>(n) * a1);
    EXPECT_LE(std::fabs(g[4] - ai), tol<T>(n) * ai);
    EXPECT_LE(std::fabs(g[5] - std::sqrt(fro)), tol<T>(n) * std::sqrt(fro));
    EXPECT_EQ(std::string(ge.begin(), ge.end()), "NRCB");
    const auto gs = d.down(as, a_h.size()), gc = d.down(ac, a_h.size()), gu = d.down(au, a_h.size()), gq = d.down(aq, a_h.size());
    const long double ratio = std::ldexp(1.0L, -emin / 2 - (emin + 4));  // cto / cfrom
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const std::size_t k = std::size_t(j) * ld + i;
            if (i >= j) EXPECT_LE(static_cast<double>(std::abs(wide(gs[k]) / ratio - wide(as_h[k])) / std::abs(wide(as_h[k]))), tol<T>()) << "lascl " << i << "," << j;
            else EXPECT_TRUE(same_bits(gs[k], as_h[k])) << "lascl upper " << i;
            if (j == 0) EXPECT_TRUE(same_bits(gc[k], conj_t(a_h[k]))) << "lacgv " << i;
            if (i <= j) EXPECT_TRUE(same_bits(gu[k], a_h[k])) << "packed round trip " << i << "," << j;
            else EXPECT_TRUE(is_nan(gu[k])) << "tpttr wrote below " << i << "," << j;
            // laqge applied R, C, then both (e1 rows, e2 cols, e3 rows and cols): r^2 c^2 a
            const long double s = static_cast<long double>(rs_h[i]) * rs_h[i] * cs_h[j] * cs_h[j];
            EXPECT_LE(static_cast<double>(std::abs(wide(gq[k]) - wide(a_h[k]) * s)), tol<T>(8) * static_cast<double>(std::abs(wide(a_h[k]) * s))) << "laqge";
        }
    // packed index map: AP(i + j (j + 1) / 2) = A(i, j) for i <= j
    const auto gp = d.down(ap, n * (n + 1) / 2);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) EXPECT_TRUE(same_bits(gp[i + j * (j + 1) / 2], a_h[j * ld + i])) << "trttp " << i << "," << j;
}

}  // namespace

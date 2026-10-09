// Host-only tests of batchlas::verify residuals (docs/design/verification.md#verification-acceptance).
// Every "exact" input is built from small integers and dyadic scalars, so the host reference is
// exact in every dtype and an exact result scores 0. Unread triangles and padding hold large finite
// poison, so a check that reads them fails the exact case.

#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using batchlas::Diag;
using batchlas::MatrixFormat;
using batchlas::MatrixView;
using batchlas::Side;
using batchlas::Transpose;
using batchlas::Uplo;
using batchlas::VectorView;
using batchlas::verify::Check;
using batchlas::verify::Shape;
using batchlas::verify::conj;
using batchlas::verify::make;
using batchlas::verify::promoted_t;
using batchlas::verify::real_t;
using batchlas::verify::up;

namespace {

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;
using Dtypes = ::testing::Types<float, double, cfloat, cdouble>;

constexpr int kB = 5;
const double kNaN = std::numeric_limits<double>::quiet_NaN();

template <class E> E poison() { return make<E>(3.0e3, -2.0e3); }
template <class E> E bump() { return make<E>(500.0, -300.0); }
template <class T> T down(const promoted_t<T>& x) {
    const cdouble z(x);
    return make<T>(z.real(), z.imag());
}
template <class T> T rand_int(batchlas::verify::Rng& rng, double amp) {
    const double re = std::round(amp * rng.next());
    const double im = std::round(amp * rng.next());
    return make<T>(re, im);
}
template <class T> promoted_t<T> alpha_of() { return up(make<T>(0.5, 0.5)); }
template <class T> promoted_t<T> beta_of() { return up(make<T>(-2.0, 1.0)); }
template <class T> T phase(int k) {
    if constexpr (batchlas::verify::is_complex<T>::value) {
        const int q = ((k % 4) + 4) % 4;
        return make<T>(q == 0 ? 1.0 : q == 2 ? -1.0 : 0.0, q == 1 ? 1.0 : q == 3 ? -1.0 : 0.0);
    } else {
        return T((k % 2 == 0) ? 1 : -1);
    }
}

// One batched operand at a non-natural layout (matrix: ld = rows+3, stride = ld*cols+5; vector:
// inc = 2, stride = 2*size+5), with its intended content per item kept aside so it can be re-stored
// at a wrong layout.
template <class E> struct Batched {
    using value_type = E;
    int rows = 0, cols = 0, inc = 1, ld = 0, batch = kB;
    long long stride = 0;
    bool vector = false;
    std::vector<E> buf;
    std::vector<std::vector<E>> exact;

    static Batched matrix(int rows, int cols) {
        Batched m;
        m.rows = rows, m.cols = cols, m.ld = rows + 3, m.stride = static_cast<long long>(m.ld) * cols + 5;
        m.init();
        return m;
    }
    static Batched vec(int size, int inc = 2) {
        Batched m;
        m.rows = size, m.cols = 1, m.inc = inc, m.ld = size * inc, m.stride = static_cast<long long>(size) * inc + 5, m.vector = true;
        m.init();
        return m;
    }
    void init() {
        buf.assign(static_cast<std::size_t>(stride * batch + ld * cols + 8), poison<E>());
        exact.assign(batch, std::vector<E>(static_cast<std::size_t>(rows * cols), poison<E>()));
    }
    E& ex(int b, int r, int c) { return exact[b][static_cast<std::size_t>(c * rows + r)]; }
    void store(int inc_, int ld_, long long stride_) {
        std::fill(buf.begin(), buf.end(), poison<E>());
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < cols; ++c)
                for (int r = 0; r < rows; ++r) buf[static_cast<std::size_t>(b * stride_ + static_cast<long long>(c) * ld_ + r * inc_)] = ex(b, r, c);
    }
    void store() { store(inc, ld, stride); }
    void store_wrong_ld() { store(1, rows, stride); }
    void store_wrong_stride() { store(inc, ld, vector ? static_cast<long long>(rows) * inc : static_cast<long long>(ld) * cols); }
    E& at(int b, int r, int c) { return buf[static_cast<std::size_t>(b * stride + static_cast<long long>(c) * ld + r * inc)]; }
    MatrixView<E, MatrixFormat::Dense> mview() { return MatrixView<E, MatrixFormat::Dense>(buf.data(), rows, cols, ld, static_cast<int>(stride), batch); }
    VectorView<E> vview() { return VectorView<E>(buf.data(), rows, batch, inc, static_cast<int>(stride)); }
};

template <class C> double bound_of(const C& c) { return batchlas::verify::bound<typename C::type>(C::kind, c.bound_n); }

template <class C> void expect_small(C& c) {
    const double v = c.value();
    EXPECT_LT(v, bound_of(c) / 100) << "value " << v;
}
template <class C> void expect_fails(C& c) {
    const double v = c.value();
    EXPECT_TRUE(std::isnan(v) || v > bound_of(c) * 100) << "value " << v << " bound " << bound_of(c);
}

template <class C> void exact_is_small() { C c; expect_small(c); }
template <class C> void perturbed_fails() {
    C c;
    c.out().at(kB / 2, c.hot_r, c.hot_c) += bump<typename std::remove_reference_t<decltype(c.out())>::value_type>();
    expect_fails(c);
}
template <class C> void wrong_ld_fails() { C c; c.out().store_wrong_ld(); expect_fails(c); }
template <class C> void wrong_stride_fails() { C c; c.out().store_wrong_stride(); expect_fails(c); }
template <class C> void last_item_only_fails() {
    C c;
    c.out().at(kB - 1, c.hot_r, c.hot_c) += bump<typename std::remove_reference_t<decltype(c.out())>::value_type>();
    expect_fails(c);
}
template <class C> void nan_fails() {
    C c;
    c.out().at(kB / 2, c.hot_r, c.hot_c) = make<typename std::remove_reference_t<decltype(c.out())>::value_type>(kNaN, 0.0);
    expect_fails(c);
}

}  // namespace

#define VERIFY_RESIDUAL_SUITE(Name, Case)                                                  \
    template <class T> class Name : public ::testing::Test {};                             \
    TYPED_TEST_SUITE(Name, Dtypes);                                                        \
    TYPED_TEST(Name, ExactIsSmall) { exact_is_small<Case<TypeParam>>(); }                  \
    TYPED_TEST(Name, PerturbedElementFails) { perturbed_fails<Case<TypeParam>>(); }        \
    TYPED_TEST(Name, WrongLdFails) { wrong_ld_fails<Case<TypeParam>>(); }                  \
    TYPED_TEST(Name, WrongStrideFails) { wrong_stride_fails<Case<TypeParam>>(); }          \
    TYPED_TEST(Name, LastItemOnlyFails) { last_item_only_fails<Case<TypeParam>>(); }       \
    TYPED_TEST(Name, NanFails) { nan_fails<Case<TypeParam>>(); }

namespace {

// ------------------------------------------------------------------ potrf

template <class T> struct PotrfCase {
    using type = T;
    static constexpr Check kind = Check::factorization;
    static constexpr int n = 8;
    int bound_n = n, hot_r, hot_c;
    Uplo uplo;
    Batched<T> A0 = Batched<T>::matrix(n, n), F = Batched<T>::matrix(n, n);

    explicit PotrfCase(Uplo u = Uplo::Lower) : hot_r(u == Uplo::Lower ? n - 1 : 0), hot_c(u == Uplo::Lower ? 0 : n - 1), uplo(u) {
        using D = promoted_t<T>;
        batchlas::verify::Rng rng(11);
        for (int b = 0; b < kB; ++b) {
            std::vector<D> L(n * n, D(0));
            for (int j = 0; j < n; ++j) {
                L[j * n + j] = up(make<T>(2.0 + std::round(2.0 * rng.next() + 2.0), 0.0));
                for (int i = j + 1; i < n; ++i) L[j * n + i] = up(rand_int<T>(rng, 3.0));
            }
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    D acc = D(0);
                    for (int k = 0; k < n; ++k) acc += L[k * n + i] * conj(L[k * n + j]);
                    A0.ex(b, i, j) = down<T>(acc);
                    if (u == Uplo::Lower && i >= j) F.ex(b, i, j) = down<T>(L[j * n + i]);
                    if (u == Uplo::Upper && i <= j) F.ex(b, i, j) = down<T>(conj(L[i * n + j]));
                }
        }
        A0.store();
        F.store();
    }
    Batched<T>& out() { return F; }
    double value() { return batchlas::verify::potrf_residual(A0.mview(), F.mview(), uplo); }
};

template <class T> struct PotrfUpperCase : PotrfCase<T> {
    PotrfUpperCase() : PotrfCase<T>(Uplo::Upper) {}
};

// ------------------------------------------------------------------ getrf

template <class T> struct GetrfCase {
    using type = T;
    static constexpr Check kind = Check::factorization;
    int m, n, mn, bound_n, hot_r, hot_c = 0;
    Batched<T> A0, F;
    std::vector<std::int32_t> piv;
    int pinc = 2, pstride;

    explicit GetrfCase(int m_ = 9, int n_ = 7)
        : m(m_), n(n_), mn(std::min(m_, n_)), bound_n(std::max(m_, n_)), hot_r(m_ - 1),
          A0(Batched<T>::matrix(m_, n_)), F(Batched<T>::matrix(m_, n_)), pstride(2 * std::min(m_, n_) + 3) {
        using D = promoted_t<T>;
        piv.assign(static_cast<std::size_t>(pstride * kB), -99);
        batchlas::verify::Rng rng(22);
        for (int b = 0; b < kB; ++b) {
            std::vector<D> LU(m * n, D(0));
            std::vector<D> L(m * mn, D(0)), U(mn * n, D(0));
            for (int k = 0; k < mn; ++k) {
                L[k * m + k] = D(1);
                for (int i = k + 1; i < m; ++i) L[k * m + i] = up(rand_int<T>(rng, 2.0));
            }
            for (int j = 0; j < n; ++j)
                for (int k = 0; k <= std::min(j, mn - 1); ++k) U[j * mn + k] = up(k == j ? make<T>(5.0, 1.0) : rand_int<T>(rng, 3.0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    for (int k = 0; k < mn; ++k) LU[j * m + i] += L[k * m + i] * U[j * mn + k];
                    F.ex(b, i, j) = down<T>(i > j ? L[j * m + i] : U[j * mn + i]);
                }
            std::vector<int> ip(mn);
            for (int k = 0; k < mn; ++k) {
                ip[k] = k + int((rng.next() * 0.5 + 0.5) * double(m - k)) % (m - k);
                piv[static_cast<std::size_t>(b * pstride + k * pinc)] = ip[k] + 1;
            }
            for (int k = mn - 1; k >= 0; --k)
                for (int c = 0; c < n; ++c) std::swap(LU[c * m + k], LU[c * m + ip[k]]);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) A0.ex(b, i, j) = down<T>(LU[j * m + i]);
        }
        A0.store();
        F.store();
    }
    VectorView<std::int32_t> pview() { return VectorView<std::int32_t>(piv.data(), mn, kB, pinc, pstride); }
    Batched<T>& out() { return F; }
    double value() { return batchlas::verify::getrf_residual(A0.mview(), F.mview(), pview()); }
};

// ------------------------------------------------------------------ geqrf

// Reflectors with three unit-phase entries below the unit, so v^H v = 4 and tau = (1+i)/4 (real:
// 1/2) makes H unitary with dyadic entries: Q R is exact in every dtype.
template <class T> struct QrCase {
    using type = T;
    static constexpr Check kind = Check::factorization;
    static constexpr int m = 9, n = 6, k = 6;
    int bound_n = m, hot_r = 0, hot_c = n - 1;
    Batched<T> A0 = Batched<T>::matrix(m, n), F = Batched<T>::matrix(m, n), tau = Batched<T>::vec(k);

    QrCase() {
        using D = promoted_t<T>;
        batchlas::verify::Rng rng(33);
        const T t = batchlas::verify::is_complex<T>::value ? make<T>(0.25, 0.25) : make<T>(0.5, 0.0);
        for (int b = 0; b < kB; ++b) {
            for (int i = 0; i < k; ++i) {
                tau.ex(b, i, 0) = t;
                const int avail = m - i - 1;
                for (int r = i + 1; r < m; ++r) F.ex(b, r, i) = make<T>(0.0, 0.0);
                const int s = int((rng.next() * 0.5 + 0.5) * double(avail)) % avail;
                for (int q = 0; q < 3; ++q) F.ex(b, i + 1 + (s + q) % avail, i) = phase<T>(i + b + q);
            }
            for (int j = 0; j < n; ++j)
                for (int r = 0; r <= j; ++r) F.ex(b, r, j) = r == j ? make<T>(4.0, -1.0) : rand_int<T>(rng, 3.0);
            for (int j = 0; j < n; ++j) {
                std::vector<D> x(m, D(0));
                for (int r = 0; r <= j; ++r) x[r] = up(F.ex(b, r, j));
                for (int i = k - 1; i >= 0; --i) {
                    D s = x[i];
                    for (int r = i + 1; r < m; ++r) s += conj(up(F.ex(b, r, i))) * x[r];
                    s *= up(t);
                    x[i] -= s;
                    for (int r = i + 1; r < m; ++r) x[r] -= s * up(F.ex(b, r, i));
                }
                for (int r = 0; r < m; ++r) A0.ex(b, r, j) = down<T>(x[r]);
            }
        }
        A0.store();
        F.store();
        tau.store();
    }
    Batched<T>& out() { return F; }
    double value() { return batchlas::verify::qr_residual(A0.mview(), F.mview(), tau.vview()); }
};

// ------------------------------------------------------------------ solve

template <class T> struct SolveCase {
    using type = T;
    static constexpr Check kind = Check::solve;
    static constexpr int n = 8, nrhs = 5;
    int bound_n = n, hot_r = n - 1, hot_c = nrhs - 1;
    Batched<T> A0 = Batched<T>::matrix(n, n), X = Batched<T>::matrix(n, nrhs), B0 = Batched<T>::matrix(n, nrhs);
    double limit = double(1 << 28);

    SolveCase() {
        using D = promoted_t<T>;
        batchlas::verify::Rng rng(44);
        for (int b = 0; b < kB; ++b) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) A0.ex(b, i, j) = rand_int<T>(rng, 4.0);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) X.ex(b, i, j) = rand_int<T>(rng, 4.0);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) {
                    D acc = D(0);
                    for (int l = 0; l < n; ++l) acc += up(A0.ex(b, i, l)) * up(X.ex(b, l, j));
                    B0.ex(b, i, j) = down<T>(acc);
                }
        }
        A0.store();
        X.store();
        B0.store();
    }
    Batched<T>& out() { return X; }
    double value() { return batchlas::verify::detail::solve_residual(A0.mview(), X.mview(), B0.mview(), {}, limit); }
};

// getri: A0 = D P (unit phases times a permutation), X = A0^{-1} = P^T D^{-1}, B0 empty.
template <class T> struct InverseCase {
    using type = T;
    static constexpr Check kind = Check::solve;
    static constexpr int n = 7;
    int bound_n = n, hot_r = n - 1, hot_c = n - 1;
    Batched<T> A0 = Batched<T>::matrix(n, n), X = Batched<T>::matrix(n, n);

    InverseCase() {
        for (int b = 0; b < kB; ++b) {
            for (int j = 0; j < n; ++j)
                for (int r = 0; r < n; ++r) A0.ex(b, r, j) = X.ex(b, r, j) = make<T>(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                const int i = (3 * j + b) % n;
                const T d = phase<T>(i + j + b);
                A0.ex(b, i, j) = d;
                X.ex(b, j, i) = down<T>(promoted_t<T>(1) / up(d));
            }
        }
        A0.store();
        X.store();
    }
    Batched<T>& out() { return X; }
    double value() {
        MatrixView<T, MatrixFormat::Dense> none(nullptr, 0, 0, 1, 0, kB);
        return batchlas::verify::solve_residual(A0.mview(), X.mview(), none);
    }
};

// ------------------------------------------------------------------ gemm family

// The logical matrix a Shape describes, built independently of the library's reader.
template <class T> std::vector<T> logical(Shape s, int rows, int cols, batchlas::verify::Rng& rng) {
    std::vector<T> M(rows * cols, make<T>(0.0, 0.0));
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) M[j * rows + i] = rand_int<T>(rng, 3.0);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) {
            T& x = M[j * rows + i];
            switch (s) {
                case Shape::general: break;
                case Shape::lower: if (i < j) x = T(0); break;
                case Shape::upper: if (i > j) x = T(0); break;
                case Shape::unit_lower: x = i == j ? T(1) : (i < j ? T(0) : x); break;
                case Shape::unit_upper: x = i == j ? T(1) : (i > j ? T(0) : x); break;
                case Shape::hermitian_lower: case Shape::hermitian_upper:
                    if (i == j) x = make<T>(cdouble(up(x)).real(), 0.0);
                    if (i < j) x = down<T>(conj(up(M[i * rows + j])));
                    break;
                case Shape::symmetric_lower: case Shape::symmetric_upper:
                    if (i < j) x = M[i * rows + j];
                    break;
            }
        }
    return M;
}

// Which elements a Shape reads; the rest of the stored operand is poison.
bool stored(Shape s, int i, int j) {
    switch (s) {
        case Shape::general: return true;
        case Shape::lower: case Shape::hermitian_lower: case Shape::symmetric_lower: return i >= j;
        case Shape::upper: case Shape::hermitian_upper: case Shape::symmetric_upper: return i <= j;
        case Shape::unit_lower: return i > j;
        case Shape::unit_upper: return i < j;
    }
    return true;
}

template <class T> promoted_t<T> op_at(const std::vector<T>& M, int rows, Transpose t, int i, int j) {
    if (t == Transpose::NoTrans) return up(M[j * rows + i]);
    const auto x = up(M[i * rows + j]);
    return t == Transpose::ConjTrans ? conj(x) : x;
}

struct GemmConfig {
    Shape sa;
    Transpose ta;
    Shape sb;
    Transpose tb;
    Shape sc;
    int m, n, k;
    bool zero_beta = false;
};

template <class T> struct GemmCase {
    using type = T;
    static constexpr Check kind = Check::blas;
    GemmConfig g;
    int bound_n, hot_r, hot_c;
    Batched<T> A, B, C0, C;
    promoted_t<T> alpha = alpha_of<T>(), beta = beta_of<T>();

    // Default: syrk-like, C's strict upper triangle unchecked (sc = lower), op(A) = A^H.
    explicit GemmCase(GemmConfig cfg = {Shape::general, Transpose::ConjTrans, Shape::general, Transpose::NoTrans, Shape::lower, 7, 7, 6})
        : g(cfg), bound_n(cfg.k), hot_r(cfg.sc == Shape::upper ? 0 : cfg.m - 1), hot_c(cfg.sc == Shape::upper ? cfg.n - 1 : 0),
          A(Batched<T>::matrix(cfg.ta == Transpose::NoTrans ? cfg.m : cfg.k, cfg.ta == Transpose::NoTrans ? cfg.k : cfg.m)),
          B(Batched<T>::matrix(cfg.tb == Transpose::NoTrans ? cfg.k : cfg.n, cfg.tb == Transpose::NoTrans ? cfg.n : cfg.k)),
          C0(Batched<T>::matrix(cfg.m, cfg.n)), C(Batched<T>::matrix(cfg.m, cfg.n)) {
        using D = promoted_t<T>;
        if (g.zero_beta) beta = D(0);
        batchlas::verify::Rng rng(55);
        for (int b = 0; b < kB; ++b) {
            const auto a = logical<T>(g.sa, A.rows, A.cols, rng);
            const auto bb = logical<T>(g.sb, B.rows, B.cols, rng);
            for (int j = 0; j < A.cols; ++j)
                for (int i = 0; i < A.rows; ++i) A.ex(b, i, j) = stored(g.sa, i, j) ? a[j * A.rows + i] : poison<T>();
            for (int j = 0; j < B.cols; ++j)
                for (int i = 0; i < B.rows; ++i) B.ex(b, i, j) = stored(g.sb, i, j) ? bb[j * B.rows + i] : poison<T>();
            for (int j = 0; j < g.n; ++j)
                for (int i = 0; i < g.m; ++i) {
                    C0.ex(b, i, j) = g.zero_beta ? make<T>(kNaN, kNaN) : rand_int<T>(rng, 3.0);
                    D acc = D(0);
                    for (int l = 0; l < g.k; ++l) acc += op_at(a, A.rows, g.ta, i, l) * op_at(bb, B.rows, g.tb, l, j);
                    const bool sel = g.sc == Shape::general || (g.sc == Shape::lower ? i >= j : i <= j);
                    C.ex(b, i, j) = sel ? down<T>(alpha * acc + (g.zero_beta ? D(0) : beta * up(C0.ex(b, i, j)))) : poison<T>();
                }
        }
        A.store();
        B.store();
        C0.store();
        C.store();
    }
    Batched<T>& out() { return C; }
    double value(Shape sa_read) {
        return batchlas::verify::gemm_backward_error(A.mview(), sa_read, g.ta, B.mview(), g.sb, g.tb, C0.mview(), C.mview(), g.sc, alpha, beta);
    }
    double value() { return value(g.sa); }
};

template <class T> struct GemvCase {
    using type = T;
    static constexpr Check kind = Check::blas;
    static constexpr int ar = 7, ac = 5;
    int bound_n = ar, hot_r = ac - 1, hot_c = 0;
    Batched<T> A = Batched<T>::matrix(ar, ac), x = Batched<T>::vec(ar), y0 = Batched<T>::vec(ac), y = Batched<T>::vec(ac);
    promoted_t<T> alpha = alpha_of<T>(), beta = beta_of<T>();

    GemvCase() {
        using D = promoted_t<T>;
        batchlas::verify::Rng rng(66);
        for (int b = 0; b < kB; ++b) {
            for (int j = 0; j < ac; ++j)
                for (int i = 0; i < ar; ++i) A.ex(b, i, j) = rand_int<T>(rng, 3.0);
            for (int i = 0; i < ar; ++i) x.ex(b, i, 0) = rand_int<T>(rng, 3.0);
            for (int i = 0; i < ac; ++i) {
                y0.ex(b, i, 0) = rand_int<T>(rng, 3.0);
                D acc = D(0);
                for (int l = 0; l < ar; ++l) acc += conj(up(A.ex(b, l, i))) * up(x.ex(b, l, 0));
                y.ex(b, i, 0) = down<T>(alpha * acc + beta * up(y0.ex(b, i, 0)));
            }
        }
        A.store();
        x.store();
        y0.store();
        y.store();
    }
    Batched<T>& out() { return y; }
    double value() {
        return batchlas::verify::gemv_backward_error(A.mview(), Transpose::ConjTrans, x.vview(), y0.vview(), y.vview(), alpha, beta);
    }
};

template <class T> struct TrsmCase {
    using type = T;
    static constexpr Check kind = Check::solve;
    Side side;
    Uplo uplo;
    Transpose ta;
    Diag diag;
    int na, xr, xc, bound_n, hot_r, hot_c;
    Batched<T> A, X, B0;
    promoted_t<T> alpha = alpha_of<T>();

    explicit TrsmCase(Side s = Side::Left, Uplo u = Uplo::Lower, Transpose t = Transpose::ConjTrans, Diag d = Diag::NonUnit)
        : side(s), uplo(u), ta(t), diag(d), na(6), xr(s == Side::Left ? 6 : 4), xc(s == Side::Left ? 4 : 6), bound_n(6),
          hot_r(xr - 1), hot_c(xc - 1), A(Batched<T>::matrix(6, 6)), X(Batched<T>::matrix(xr, xc)), B0(Batched<T>::matrix(xr, xc)) {
        using D = promoted_t<T>;
        const Shape sh = u == Uplo::Lower ? (d == Diag::Unit ? Shape::unit_lower : Shape::lower) : (d == Diag::Unit ? Shape::unit_upper : Shape::upper);
        batchlas::verify::Rng rng(77);
        for (int b = 0; b < kB; ++b) {
            auto a = logical<T>(sh, na, na, rng);
            if (d == Diag::NonUnit)
                for (int i = 0; i < na; ++i) a[i * na + i] = make<T>(3.0, 1.0);
            for (int j = 0; j < na; ++j)
                for (int i = 0; i < na; ++i) A.ex(b, i, j) = stored(sh, i, j) ? a[j * na + i] : poison<T>();
            for (int j = 0; j < xc; ++j)
                for (int i = 0; i < xr; ++i) X.ex(b, i, j) = rand_int<T>(rng, 3.0);
            for (int j = 0; j < xc; ++j)
                for (int i = 0; i < xr; ++i) {
                    D acc = D(0);
                    for (int l = 0; l < na; ++l)
                        acc += s == Side::Left ? op_at(a, na, t, i, l) * up(X.ex(b, l, j)) : up(X.ex(b, i, l)) * op_at(a, na, t, l, j);
                    B0.ex(b, i, j) = down<T>(acc / alpha);
                }
        }
        A.store();
        X.store();
        B0.store();
    }
    Batched<T>& out() { return X; }
    double value() { return batchlas::verify::trsm_residual(A.mview(), side, uplo, ta, diag, X.mview(), B0.mview(), alpha); }
};

template <class T> struct TrsmRightCase : TrsmCase<T> {
    TrsmRightCase() : TrsmCase<T>(Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::Unit) {}
};

// ------------------------------------------------------------------ spmm

template <class T> struct SpmmCase {
    using type = T;
    static constexpr Check kind = Check::blas;
    static constexpr int am = 6, ak = 7, n = 5;
    Transpose ta, tb;
    int cm, ck, bound_n, hot_r, hot_c;
    int nnz_max = am * ak, mstride = am * ak + 4, ostride = am + 3;
    std::vector<T> vals;
    std::vector<int> offs, cols;
    Batched<T> B, C0, C;
    promoted_t<T> alpha = alpha_of<T>(), beta = beta_of<T>();

    explicit SpmmCase(Transpose a = Transpose::NoTrans, Transpose tb_ = Transpose::ConjTrans)
        : ta(a), tb(tb_), cm(a == Transpose::NoTrans ? am : ak), ck(a == Transpose::NoTrans ? ak : am), bound_n(ck), hot_r(cm - 1), hot_c(n - 1),
          B(Batched<T>::matrix(tb_ == Transpose::NoTrans ? ck : n, tb_ == Transpose::NoTrans ? n : ck)),
          C0(Batched<T>::matrix(cm, n)), C(Batched<T>::matrix(cm, n)) {
        using D = promoted_t<T>;
        vals.assign(static_cast<std::size_t>(mstride * kB), poison<T>());
        cols.assign(static_cast<std::size_t>(mstride * kB), 1 << 20);
        offs.assign(static_cast<std::size_t>(ostride * kB), -7);
        batchlas::verify::Rng rng(88);
        for (int b = 0; b < kB; ++b) {
            std::vector<T> dense(am * ak, make<T>(0.0, 0.0));
            int p = 0;
            for (int r = 0; r < am; ++r) {
                offs[b * ostride + r] = p;
                for (int c = 0; c < ak; ++c)
                    if (rng.next() > 0.0 || c == (r + b) % ak) {
                        dense[c * am + r] = rand_int<T>(rng, 3.0);
                        vals[b * mstride + p] = dense[c * am + r];
                        cols[b * mstride + p] = c;
                        ++p;
                    }
            }
            offs[b * ostride + am] = p;
            for (int j = 0; j < B.cols; ++j)
                for (int i = 0; i < B.rows; ++i) B.ex(b, i, j) = rand_int<T>(rng, 3.0);
            const std::vector<T> bq(B.exact[b]);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < cm; ++i) {
                    C0.ex(b, i, j) = rand_int<T>(rng, 3.0);
                    D acc = D(0);
                    for (int l = 0; l < ck; ++l) acc += op_at(dense, am, ta, i, l) * op_at(bq, B.rows, tb, l, j);
                    C.ex(b, i, j) = down<T>(alpha * acc + beta * up(C0.ex(b, i, j)));
                }
        }
        B.store();
        C0.store();
        C.store();
    }
    MatrixView<T, MatrixFormat::CSR> aview() {
        return MatrixView<T, MatrixFormat::CSR>(vals.data(), offs.data(), cols.data(), am, ak, batchlas::NonZeros{nnz_max}, mstride, ostride, kB);
    }
    Batched<T>& out() { return C; }
    double value() { return batchlas::verify::spmm_backward_error(aview(), ta, B.mview(), tb, C0.mview(), C.mview(), alpha, beta); }
};

template <class T> struct SpmmTransCase : SpmmCase<T> {
    SpmmTransCase() : SpmmCase<T>(Transpose::ConjTrans, Transpose::NoTrans) {}
};

// ------------------------------------------------------------------ orthogonality and eigen

// Sylvester Hadamard H16 / 4 with unit phases on the rows: unitary, entries +-1/4 or +-i/4.
template <class T> T hadamard(int i, int j, int b) {
    const int h = __builtin_popcount(unsigned(i & j)) % 2 == 0 ? 1 : -1;
    return down<T>(up(phase<T>(i + b)) * (0.25 * h));
}

template <class T> struct OrthoCase {
    using type = T;
    static constexpr Check kind = Check::orthogonality;
    static constexpr int m = 16, k = 12;
    int bound_n = m, hot_r = m - 1, hot_c = k - 1;
    Batched<T> Q = Batched<T>::matrix(m, k);

    OrthoCase() {
        for (int b = 0; b < kB; ++b)
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < m; ++i) Q.ex(b, i, j) = hadamard<T>(i, j, b);
        Q.store();
    }
    Batched<T>& out() { return Q; }
    double value() { return batchlas::verify::orthogonality(Q.mview()); }
};

template <class T> struct EigenCase {
    using type = T;
    using R = real_t<T>;
    static constexpr Check kind = Check::eigen_residual;
    static constexpr int n = 16;
    int bound_n = n, hot_r = n - 1, hot_c = 0;
    Batched<T> A = Batched<T>::matrix(n, n), V = Batched<T>::matrix(n, n);
    Batched<R> w = Batched<R>::vec(n);

    EigenCase() {
        using D = promoted_t<T>;
        for (int b = 0; b < kB; ++b) {
            for (int j = 0; j < n; ++j) {
                w.ex(b, j, 0) = R(j - 7 + b);
                for (int i = 0; i < n; ++i) V.ex(b, i, j) = hadamard<T>(i, j, b);
            }
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    D acc = D(0);
                    for (int l = 0; l < n; ++l) acc += up(V.ex(b, i, l)) * double(w.ex(b, l, 0)) * conj(up(V.ex(b, j, l)));
                    A.ex(b, i, j) = i >= j ? down<T>(acc) : poison<T>();
                }
        }
        A.store();
        V.store();
        w.store();
    }
    Batched<T>& out() { return V; }
    double value() { return batchlas::verify::eigen_residual(A.mview(), V.mview(), w.vview()); }
};

template <class T> struct ValuesCase {
    using type = T;
    using R = real_t<T>;
    static constexpr Check kind = Check::values;
    static constexpr int n = 9;
    int bound_n = n, hot_r = n - 1, hot_c = 0;
    Batched<R> w = Batched<R>::vec(n, 3);
    std::vector<std::vector<double>> ref;

    ValuesCase() {
        ref.assign(kB, std::vector<double>(n));
        for (int b = 0; b < kB; ++b)
            for (int i = 0; i < n; ++i) w.ex(b, i, 0) = R(ref[b][i] = (i - 4) / 8.0 + b);
        w.store();
    }
    Batched<R>& out() { return w; }
    double value() { return batchlas::verify::values_error(w.vview(), ref, 1.0); }
};

}  // namespace

VERIFY_RESIDUAL_SUITE(PotrfResidual, PotrfCase)
VERIFY_RESIDUAL_SUITE(PotrfUpperResidual, PotrfUpperCase)
VERIFY_RESIDUAL_SUITE(GetrfResidual, GetrfCase)
VERIFY_RESIDUAL_SUITE(QrResidual, QrCase)
VERIFY_RESIDUAL_SUITE(SolveResidual, SolveCase)
VERIFY_RESIDUAL_SUITE(InverseSolveResidual, InverseCase)
VERIFY_RESIDUAL_SUITE(GemmBackwardError, GemmCase)
VERIFY_RESIDUAL_SUITE(GemvBackwardError, GemvCase)
VERIFY_RESIDUAL_SUITE(TrsmResidual, TrsmCase)
VERIFY_RESIDUAL_SUITE(TrsmRightResidual, TrsmRightCase)
VERIFY_RESIDUAL_SUITE(SpmmBackwardError, SpmmCase)
VERIFY_RESIDUAL_SUITE(SpmmTransBackwardError, SpmmTransCase)
VERIFY_RESIDUAL_SUITE(Orthogonality, OrthoCase)
VERIFY_RESIDUAL_SUITE(EigenResidual, EigenCase)
VERIFY_RESIDUAL_SUITE(ValuesError, ValuesCase)

// ------------------------------------------------------------------ further axes

TYPED_TEST(GetrfResidual, WideExactIsSmall) {
    GetrfCase<TypeParam> c(6, 9);
    expect_small(c);
    c.F.at(kB - 1, 0, 8) += bump<TypeParam>();
    expect_fails(c);
}

TYPED_TEST(GetrfResidual, BadPivotIsNan) {
    GetrfCase<TypeParam> c;
    c.piv[static_cast<std::size_t>((kB - 1) * c.pstride + 2 * c.pinc)] = c.m + 1;
    EXPECT_TRUE(std::isnan(c.value()));
    c.piv[static_cast<std::size_t>((kB - 1) * c.pstride + 2 * c.pinc)] = 0;
    EXPECT_TRUE(std::isnan(c.value()));
}

TYPED_TEST(GetrfResidual, DroppedInterchangeFails) {
    GetrfCase<TypeParam> c;
    int k = 0;
    while (k < c.mn && c.piv[static_cast<std::size_t>(k * c.pinc)] == k + 1) ++k;
    ASSERT_LT(k, c.mn);
    c.piv[static_cast<std::size_t>(k * c.pinc)] = k + 1;
    expect_fails(c);
}

TYPED_TEST(SolveResidual, ProbeColumnsSeeAWrongEntry) {
    SolveCase<TypeParam> c;
    c.limit = 0.0;
    EXPECT_LT(c.value(), bound_of(c));
    c.X.at(kB - 1, 3, 2) += bump<TypeParam>();
    expect_fails(c);
}

TYPED_TEST(GemmBackwardError, ShapesExactIsSmall) {
    const GemmConfig cases[] = {
        {Shape::hermitian_lower, Transpose::NoTrans, Shape::general, Transpose::Trans, Shape::general, 6, 5, 6},
        {Shape::symmetric_upper, Transpose::NoTrans, Shape::general, Transpose::NoTrans, Shape::general, 6, 5, 6},
        {Shape::unit_upper, Transpose::ConjTrans, Shape::general, Transpose::NoTrans, Shape::general, 6, 4, 6},
        {Shape::lower, Transpose::Trans, Shape::general, Transpose::NoTrans, Shape::general, 6, 4, 6},
        {Shape::general, Transpose::NoTrans, Shape::unit_lower, Transpose::ConjTrans, Shape::general, 5, 6, 6},
        {Shape::general, Transpose::NoTrans, Shape::hermitian_upper, Transpose::NoTrans, Shape::upper, 6, 6, 6},
        {Shape::general, Transpose::ConjTrans, Shape::general, Transpose::NoTrans, Shape::general, 5, 4, 7, true},
    };
    for (const GemmConfig& g : cases) {
        GemmCase<TypeParam> c(g);
        SCOPED_TRACE(int(g.sa) * 100 + int(g.sb) * 10 + int(g.sc));
        expect_small(c);
    }
}

TYPED_TEST(GemmBackwardError, WrongShapeFails) {
    GemmCase<TypeParam> c({Shape::hermitian_lower, Transpose::NoTrans, Shape::general, Transpose::NoTrans, Shape::general, 6, 5, 6});
    EXPECT_GT(c.value(Shape::hermitian_upper), bound_of(c) * 100);
    EXPECT_GT(c.value(Shape::general), bound_of(c) * 100);
}

TEST(GemmBackwardError, ZeroDenominatorUsesNumerator) {
    std::vector<double> z(8, 0.0), c(8, 0.0);
    MatrixView<double, MatrixFormat::Dense> Z(z.data(), 2, 2, 2, 4, 1), C(c.data(), 2, 2, 2, 4, 1);
    EXPECT_EQ(batchlas::verify::gemm_backward_error(Z, Shape::general, Transpose::NoTrans, Z, Shape::general, Transpose::NoTrans, Z, C,
                                                    Shape::general, 1.0, 1.0), 0.0);
    c[3] = 0.25;
    EXPECT_EQ(batchlas::verify::gemm_backward_error(Z, Shape::general, Transpose::NoTrans, Z, Shape::general, Transpose::NoTrans, Z, C,
                                                    Shape::general, 1.0, 1.0), 0.25);
}

TYPED_TEST(SpmmBackwardError, BadColumnIsNan) {
    SpmmCase<TypeParam> c;
    c.cols[static_cast<std::size_t>((kB - 1) * c.mstride)] = SpmmCase<TypeParam>::ak;
    EXPECT_TRUE(std::isnan(c.value()));
}

TYPED_TEST(ValuesError, MismatchedRefThrows) {
    ValuesCase<TypeParam> c;
    c.ref[kB / 2].pop_back();
    EXPECT_THROW(c.value(), std::invalid_argument);
}

TEST(Residuals, ExplicitItemsAreTheOnesRead) {
    PotrfCase<double> c;
    c.F.at(1, c.hot_r, c.hot_c) += 500.0;  // item 1 is not a default item
    EXPECT_LT(c.value(), bound_of(c) / 100);
    const std::vector<int> all = batchlas::verify::all_items(kB);
    EXPECT_GT(batchlas::verify::potrf_residual(c.A0.mview(), c.F.mview(), Uplo::Lower, all), bound_of(c) * 100);
}

TEST(Residuals, HeterogeneousViewThrows) {
    PotrfCase<double> c;
    std::vector<int> rows(kB, 4);
    batchlas::KernelMatrixView<double, MatrixFormat::Dense> F(c.F.buf.data(), c.F.rows, c.F.cols, c.F.ld, int(c.F.stride), kB);
    F.active_rows_ = rows.data();
    EXPECT_THROW(batchlas::verify::potrf_residual(c.A0.mview(), F, Uplo::Lower), std::invalid_argument);
}

TEST(PivotsValid, RangeAndLayout) {
    std::vector<std::int32_t> p(3 * 11, 0);
    for (int b = 0; b < 3; ++b)
        for (int k = 0; k < 4; ++k) p[b * 11 + 2 * k] = 1 + (k + b) % 5;
    VectorView<std::int32_t> v(p.data(), 4, 3, 2, 11);
    EXPECT_TRUE(batchlas::verify::pivots_valid(v, 5));
    EXPECT_FALSE(batchlas::verify::pivots_valid(v, 4));
    p[2 * 11 + 2 * 3] = 0;  // last item, last pivot
    EXPECT_FALSE(batchlas::verify::pivots_valid(v, 5));
    p[2 * 11 + 2 * 3] = 3;
    VectorView<std::int32_t> natural(p.data(), 4, 3, 1, 11);  // inc 1 reads the zero padding
    EXPECT_FALSE(batchlas::verify::pivots_valid(natural, 5));
}

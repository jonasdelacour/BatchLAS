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
    std::vector<std::vector<T>> la, lb;  // logical op operands per item
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
            la.push_back(a);
            lb.push_back(bb);
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
    std::vector<std::vector<T>> la;
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
            la.push_back(a);
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
    std::vector<std::vector<T>> dense_items;
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
            dense_items.push_back(dense);
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

// A = [1 -2; 3 4], B = [1 0; -1 2], C0 = [5 -6; 0 1], alpha 2, beta -1, in item 1 of a padded batch
// (ld 3, stride 7) whose item 0 holds 1e6: |A||B| = [3 4; 7 8], so the denominators are [11 14; 14 17].
TEST(GemmMaxDenominator, KnownValues) {
    const auto fill = [](std::vector<double>& v, std::initializer_list<double> col_major) {
        v.assign(14, 1e6);
        int i = 0;
        for (double x : col_major) v[7 + (i / 2) * 3 + i % 2] = x, ++i;
    };
    std::vector<double> a, b, c0;
    fill(a, {1, 3, -2, 4});
    fill(b, {1, -1, 0, 2});
    fill(c0, {5, 0, -6, 1});
    const auto A = batchlas::verify::view(a.data(), 2, 2, 3, 7, 2), B = batchlas::verify::view(b.data(), 2, 2, 3, 7, 2),
               C0 = batchlas::verify::view(c0.data(), 2, 2, 3, 7, 2);
    using batchlas::verify::gemm_max_denominator;
    const auto g = Shape::general;
    EXPECT_EQ(gemm_max_denominator(A, g, Transpose::NoTrans, B, g, Transpose::NoTrans, C0, 2.0, -1.0, 1), 17.0);
    // Unit lower triangle of A: [1 0; 3 1] gives [7 6; 8 5]; op(A) = A^T: [13 18; 12 17].
    EXPECT_EQ(gemm_max_denominator(A, Shape::unit_lower, Transpose::NoTrans, B, g, Transpose::NoTrans, C0, 2.0, -1.0, 1), 8.0);
    EXPECT_EQ(gemm_max_denominator(A, g, Transpose::Trans, B, g, Transpose::NoTrans, C0, 2.0, -1.0, 1), 18.0);
    // beta == 0 never reads C0 (NaN there is ignored); a NaN in A is the result.
    c0[7 + 3 + 1] = std::nan("");
    EXPECT_EQ(gemm_max_denominator(A, g, Transpose::NoTrans, B, g, Transpose::NoTrans, C0, 2.0, 0.0, 1), 16.0);
    EXPECT_TRUE(std::isnan(gemm_max_denominator(A, g, Transpose::NoTrans, B, g, Transpose::NoTrans, C0, 2.0, -1.0, 1)));
    a[7] = std::nan("");
    EXPECT_TRUE(std::isnan(gemm_max_denominator(A, g, Transpose::NoTrans, B, g, Transpose::NoTrans, C0, 2.0, 0.0, 1)));
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

// ------------------------------------------------------------------ known values
// One dyadic delta on one element of the last item, the expected value worked out by hand from the
// fixture's own data: these pin each denominator (the exact cases score 0 whatever it is).

namespace {

template <class T> T delta() { return make<T>(0.5, -0.25); }
template <class T> double nrm2(const T& x) { return std::norm(cdouble(up(x))); }
template <class T> double mod(const T& x) { return std::abs(cdouble(up(x))); }

template <class C> void expect_value(C& c, double expected) {
    const double v = c.value();
    EXPECT_NEAR(v, expected, 1e-12 * expected) << "value " << v << " expected " << expected;
}

template <class E> double frob2(Batched<E>& M, int b) {
    double s = 0;
    for (int j = 0; j < M.cols; ++j)
        for (int i = 0; i < M.rows; ++i) s += nrm2(M.ex(b, i, j));
    return s;
}

// Only (n-1, n-1) of L L^H (or U^H U) moves, by |l + d|^2 - |l|^2; the denominator is the norm of
// A0's triangle.
template <class T> void potrf_known(Uplo u) {
    PotrfCase<T> c(u);
    const int n = c.n, b = kB - 1;
    const T d = delta<T>();
    const cdouble l(up(c.F.ex(b, n - 1, n - 1)));
    c.F.at(b, n - 1, n - 1) += d;
    double den = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (u == Uplo::Lower ? i >= j : i <= j) den += nrm2(c.A0.ex(b, i, j));
    expect_value(c, std::abs(std::norm(l + cdouble(up(d))) - std::norm(l)) / std::sqrt(den));
}

template <class C> void trsm_known() {
    using T = typename C::type;
    C c;
    const int b = kB - 1, i0 = c.xr - 1, j0 = c.xc - 1;
    const T d = delta<T>();
    const double x2 = nrm2(c.X.ex(b, i0, j0));
    const double xp2 = nrm2(c.X.ex(b, i0, j0) + d);
    c.X.at(b, i0, j0) += d;
    // Left: op(A) X moves by d op(A)(:, i0) in column j0. Right: X op(A) moves by d op(A)(j0, :) in row i0.
    double hit = 0;
    for (int l = 0; l < c.na; ++l)
        hit += std::norm(cdouble(c.side == Side::Left ? op_at(c.la[b], c.na, c.ta, l, i0) : op_at(c.la[b], c.na, c.ta, j0, l)));
    double a2 = 0;
    for (const T& x : c.la[b]) a2 += nrm2(x);
    const double den = std::sqrt(a2) * std::sqrt(frob2(c.X, b) - x2 + xp2) + mod(c.alpha) * std::sqrt(frob2(c.B0, b));
    expect_value(c, mod(d) * std::sqrt(hit) / den);
}

}  // namespace

TYPED_TEST(PotrfResidual, KnownValue) { potrf_known<TypeParam>(Uplo::Lower); }
TYPED_TEST(PotrfUpperResidual, KnownValue) { potrf_known<TypeParam>(Uplo::Upper); }

// L(m-1, 0) += d moves row m-1 of L U by d U(0, :).
TYPED_TEST(GetrfResidual, KnownValue) {
    GetrfCase<TypeParam> c;
    const int b = kB - 1;
    const TypeParam d = delta<TypeParam>();
    c.F.at(b, c.m - 1, 0) += d;
    double u0 = 0;
    for (int j = 0; j < c.n; ++j) u0 += nrm2(c.F.ex(b, 0, j));
    expect_value(c, mod(d) * std::sqrt(u0) / std::sqrt(frob2(c.A0, b)));
}

// R(0, n-1) += d moves column n-1 of Q R by d Q e_0, a unit vector.
TYPED_TEST(QrResidual, KnownValue) {
    QrCase<TypeParam> c;
    const int b = kB - 1;
    const TypeParam d = delta<TypeParam>();
    c.F.at(b, 0, c.n - 1) += d;
    expect_value(c, mod(d) / std::sqrt(frob2(c.A0, b)));
}

// X(i0, j0) += d moves column j0 of A0 X by d A0(:, i0); the denominator is ||A0|| ||X + d||.
TYPED_TEST(SolveResidual, KnownValue) {
    SolveCase<TypeParam> c;
    const int b = kB - 1, i0 = c.n - 1, j0 = c.nrhs - 1;
    const TypeParam d = delta<TypeParam>();
    const double x2 = nrm2(c.X.ex(b, i0, j0)), xp2 = nrm2(c.X.ex(b, i0, j0) + d);
    c.X.at(b, i0, j0) += d;
    double col = 0;
    for (int r = 0; r < c.n; ++r) col += nrm2(c.A0.ex(b, r, i0));
    expect_value(c, mod(d) * std::sqrt(col) / (std::sqrt(frob2(c.A0, b)) * std::sqrt(frob2(c.X, b) - x2 + xp2)));
}

// getri form: A0 is a phased permutation, so ||A0||_F^2 = n and every column of A0 has norm 1.
TYPED_TEST(InverseSolveResidual, KnownValue) {
    InverseCase<TypeParam> c;
    const int b = kB - 1, n = c.n;
    const TypeParam d = delta<TypeParam>();
    const double x2 = nrm2(c.X.ex(b, n - 1, n - 1)), xp2 = nrm2(c.X.ex(b, n - 1, n - 1) + d);
    c.X.at(b, n - 1, n - 1) += d;
    expect_value(c, mod(d) / (std::sqrt(double(n)) * std::sqrt(double(n) - x2 + xp2)));
}

namespace {

// Componentwise: only C(i0, j0) is off, by d, over |alpha| sum |opA||opB| + |beta||C0|.
template <class T> void gemm_known(const GemmConfig& g) {
    GemmCase<T> c(g);
    const int b = kB - 1, i0 = g.m - 1, j0 = 0;
    const T d = delta<T>();
    c.C.at(b, i0, j0) += d;
    double mag = 0;
    for (int l = 0; l < g.k; ++l)
        mag += std::abs(cdouble(op_at(c.la[b], c.A.rows, g.ta, i0, l))) * std::abs(cdouble(op_at(c.lb[b], c.B.rows, g.tb, l, j0)));
    expect_value(c, mod(d) / (mod(c.alpha) * mag + mod(c.beta) * mod(c.C0.ex(b, i0, j0))));
}

}  // namespace

TYPED_TEST(GemmBackwardError, KnownValue) {
    gemm_known<TypeParam>({Shape::general, Transpose::ConjTrans, Shape::general, Transpose::NoTrans, Shape::lower, 7, 7, 6});
}
TYPED_TEST(GemmBackwardError, HermitianKnownValue) {
    gemm_known<TypeParam>({Shape::hermitian_lower, Transpose::NoTrans, Shape::unit_upper, Transpose::ConjTrans, Shape::general, 6, 6, 6});
}

TYPED_TEST(GemvBackwardError, KnownValue) {
    GemvCase<TypeParam> c;
    const int b = kB - 1, i0 = c.ac - 1;
    const TypeParam d = delta<TypeParam>();
    c.y.at(b, i0, 0) += d;
    double mag = 0;
    for (int l = 0; l < c.ar; ++l) mag += mod(c.A.ex(b, l, i0)) * mod(c.x.ex(b, l, 0));
    expect_value(c, mod(d) / (mod(c.alpha) * mag + mod(c.beta) * mod(c.y0.ex(b, i0, 0))));
}

TYPED_TEST(TrsmResidual, KnownValue) { trsm_known<TrsmCase<TypeParam>>(); }
TYPED_TEST(TrsmRightResidual, KnownValue) { trsm_known<TrsmRightCase<TypeParam>>(); }

TYPED_TEST(SpmmBackwardError, KnownValue) {
    SpmmCase<TypeParam> c;
    const int b = kB - 1, i0 = c.cm - 1, j0 = c.n - 1;
    const TypeParam d = delta<TypeParam>();
    c.C.at(b, i0, j0) += d;
    double mag = 0;
    for (int l = 0; l < c.ck; ++l)
        mag += std::abs(cdouble(op_at(c.dense_items[b], c.am, c.ta, i0, l))) * std::abs(cdouble(op_at(c.B.exact[b], c.B.rows, c.tb, l, j0)));
    expect_value(c, mod(d) / (mod(c.alpha) * mag + mod(c.beta) * mod(c.C0.ex(b, i0, j0))));
}

// Column c += d e_r: (Q^H Q)(a, c) and (c, a) move by d conj(Q(r, a)) and its conjugate for a != c,
// and (c, c) by 2 Re(conj(q) d) + |d|^2.
TYPED_TEST(Orthogonality, KnownValue) {
    OrthoCase<TypeParam> c;
    const int b = kB - 1, r = c.m - 1, col = c.k - 1;
    const TypeParam d = delta<TypeParam>();
    const cdouble q(up(c.Q.ex(b, r, col))), dd(up(d));
    double off = 0;
    for (int a = 0; a < c.k; ++a)
        if (a != col) off += nrm2(c.Q.ex(b, r, a));
    const double diag = 2.0 * (std::conj(q) * dd).real() + std::norm(dd);
    c.Q.at(b, r, col) += d;
    expect_value(c, std::sqrt(2.0 * std::norm(dd) * off + diag * diag));
}

// V(i0, j0) += d moves column j0 of A V - V diag(w) by d (A(:, i0) - w_j0 e_i0); A is full Hermitian.
TYPED_TEST(EigenResidual, KnownValue) {
    EigenCase<TypeParam> c;
    const int b = kB - 1, n = c.n, i0 = n - 1, j0 = 0;
    const TypeParam d = delta<TypeParam>();
    c.V.at(b, i0, j0) += d;
    auto full = [&](int r, int s) { return r >= s ? cdouble(up(c.A.ex(b, r, s))) : std::conj(cdouble(up(c.A.ex(b, s, r)))); };
    double hit = 0, a2 = 0;
    for (int r = 0; r < n; ++r) {
        hit += std::norm(full(r, i0) - (r == i0 ? double(c.w.ex(b, j0, 0)) : 0.0));
        for (int s = 0; s < n; ++s) a2 += std::norm(full(r, s));
    }
    expect_value(c, mod(d) * std::sqrt(hit) / std::sqrt(a2));
}

TYPED_TEST(ValuesError, KnownValue) {
    ValuesCase<TypeParam> c;
    c.w.at(kB - 1, c.n - 1, 0) += real_t<TypeParam>(0.5);
    EXPECT_DOUBLE_EQ(batchlas::verify::values_error(c.w.vview(), c.ref, 4.0), 0.125);
}

// ------------------------------------------------------------------ form_q, qr_reconstruction

namespace {

template <class T> T unit_tau() { return batchlas::verify::is_complex<T>::value ? make<T>(0.25, 0.25) : make<T>(0.5, 0.0); }

// QrCase's reflectors; F's diagonal and upper triangle stay poison, which form_q must not read.
template <class T> void unit_reflectors(Batched<T>& F, Batched<T>& tau, int k, batchlas::verify::Rng& rng) {
    for (int b = 0; b < kB; ++b)
        for (int i = 0; i < k; ++i) {
            tau.ex(b, i, 0) = unit_tau<T>();
            const int avail = F.rows - i - 1;
            for (int r = i + 1; r < F.rows; ++r) F.ex(b, r, i) = make<T>(0.0, 0.0);
            const int s = int((rng.next() * 0.5 + 0.5) * double(avail)) % avail;
            for (int q = 0; q < 3; ++q) F.ex(b, i + 1 + (s + q) % avail, i) = phase<T>(i + b + q);
        }
}

// Q = H_0 ... H_{k-1} by right-multiplication Q <- Q H_j: not form_q's order.
template <class T> std::vector<promoted_t<T>> q_by_rows(Batched<T>& F, Batched<T>& tau, int b, int k) {
    using D = promoted_t<T>;
    const int m = F.rows;
    std::vector<D> Q(m * m, D(0));
    for (int i = 0; i < m; ++i) Q[i * m + i] = D(1);
    for (int j = 0; j < k; ++j) {
        std::vector<D> v(m, D(0));
        v[j] = D(1);
        for (int i = j + 1; i < m; ++i) v[i] = up(F.ex(b, i, j));
        const D t = up(tau.ex(b, j, 0));
        for (int r = 0; r < m; ++r) {
            D w = D(0);
            for (int i = 0; i < m; ++i) w += Q[i * m + r] * v[i];
            for (int i = 0; i < m; ++i) Q[i * m + r] -= t * w * conj(v[i]);
        }
    }
    return Q;
}

template <class T> struct FormQCase {
    using type = T;
    static constexpr Check kind = Check::orthogonality;
    static constexpr int m = 9, k = 5;
    int cols = m, bound_n = m, hot_r = m - 1, hot_c = 0;
    Batched<T> F = Batched<T>::matrix(m, k), tau = Batched<T>::vec(k);
    std::vector<std::vector<promoted_t<T>>> ref;

    FormQCase() {
        batchlas::verify::Rng rng(99);
        unit_reflectors(F, tau, k, rng);
        for (int b = 0; b < kB; ++b) ref.push_back(q_by_rows(F, tau, b, k));
        F.store();
        tau.store();
    }
    Batched<T>& out() { return F; }
    double value() {
        double worst = 0;
        for (int b : batchlas::verify::default_items(kB)) {
            const auto Q = batchlas::verify::form_q(F.mview(), tau.vview(), b, cols);
            if (int(Q.size()) != m * cols) return kNaN;
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < m; ++i) worst = batchlas::verify::nanmax(worst, std::abs(cdouble(Q[j * m + i] - ref[b][j * m + i])));
        }
        return worst;
    }
};

// Q explicit (Hadamard columns, a different column set per item), R in geqrf storage (poison below
// the diagonal), A0 = Q triu(R) exact.
template <class T> struct QrReconCase {
    using type = T;
    static constexpr Check kind = Check::factorization;
    static constexpr int m = 16, n = 6, k = 6;
    int bound_n = m, hot_r = 0, hot_c = n - 1;
    Batched<T> A0 = Batched<T>::matrix(m, n), Q = Batched<T>::matrix(m, k), R = Batched<T>::matrix(m, n);

    QrReconCase() {
        using D = promoted_t<T>;
        batchlas::verify::Rng rng(111);
        for (int b = 0; b < kB; ++b) {
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < m; ++i) Q.ex(b, i, j) = hadamard<T>(i, (j + 3 * b) % m, b);
            for (int j = 0; j < n; ++j)
                for (int r = 0; r <= j && r < k; ++r) R.ex(b, r, j) = rand_int<T>(rng, 3.0);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    D acc = D(0);
                    for (int r = 0; r <= j && r < k; ++r) acc += up(Q.ex(b, i, r)) * up(R.ex(b, r, j));
                    A0.ex(b, i, j) = down<T>(acc);
                }
        }
        A0.store();
        Q.store();
        R.store();
    }
    Batched<T>& out() { return R; }
    double value() {
        const auto a0 = batchlas::verify::view(static_cast<const T*>(A0.buf.data()), m, n, A0.ld, int(A0.stride), kB);
        return batchlas::verify::qr_reconstruction(a0, Q.mview(), R.mview());
    }
};

}  // namespace

VERIFY_RESIDUAL_SUITE(FormQ, FormQCase)
VERIFY_RESIDUAL_SUITE(QrReconstruction, QrReconCase)

TYPED_TEST(FormQ, LeadingColumnsOnly) {
    FormQCase<TypeParam> c;
    c.cols = c.k;
    expect_small(c);
    c.cols = 0;
    EXPECT_EQ(c.value(), 0.0);
    c.cols = c.m + 1;
    EXPECT_THROW(c.value(), std::invalid_argument);
}

TYPED_TEST(FormQ, PerturbedTauFails) {
    FormQCase<TypeParam> c;
    c.tau.at(kB - 1, c.k - 1, 0) += bump<TypeParam>();
    expect_fails(c);
}

// m = 3, one reflector v = (1, 1, -1 + i) (real: -1) at ld 4, tau t: Q = I - t v v^H, by hand.
TYPED_TEST(FormQ, KnownValue) {
    using T = TypeParam;
    constexpr bool cx = batchlas::verify::is_complex<T>::value;
    std::vector<T> f(8, poison<T>()), t{make<T>(0.5, 0.25)};
    f[1] = make<T>(1.0, 0.0);
    f[2] = make<T>(-1.0, 1.0);
    const auto Q = batchlas::verify::form_q(batchlas::verify::view(static_cast<const T*>(f.data()), 3, 1, 4), VectorView<T>(t.data(), 1, 1), 0, 3);
    ASSERT_EQ(Q.size(), 9u);
    const auto at = [&](int i, int j) { return cdouble(Q[j * 3 + i]); };
    EXPECT_EQ(at(0, 0), cx ? cdouble(0.5, -0.25) : cdouble(0.5));
    EXPECT_EQ(at(1, 0), cx ? cdouble(-0.5, -0.25) : cdouble(-0.5));
    EXPECT_EQ(at(2, 0), cx ? cdouble(0.75, -0.25) : cdouble(0.5));
    EXPECT_EQ(at(1, 2), cx ? cdouble(0.25, 0.75) : cdouble(0.5));
    EXPECT_EQ(at(2, 2), cx ? cdouble(0.0, -0.5) : cdouble(0.5));
}

// R(0, n-1) += d moves column n-1 of Q R by d Q(:, 0), a unit vector.
TYPED_TEST(QrReconstruction, KnownValue) {
    QrReconCase<TypeParam> c;
    const int b = kB - 1;
    const TypeParam d = delta<TypeParam>();
    c.R.at(b, 0, c.n - 1) += d;
    expect_value(c, mod(d) / std::sqrt(frob2(c.A0, b)));
}

TYPED_TEST(QrReconstruction, SwappedQFails) {
    QrReconCase<TypeParam> c;
    std::swap(c.Q.exact[0], c.Q.exact[kB - 1]);
    c.Q.store();
    expect_fails(c);
}

TYPED_TEST(QrReconstruction, PerturbedQFails) {
    QrReconCase<TypeParam> c;
    c.Q.at(kB / 2, c.m - 1, 0) += bump<TypeParam>();
    expect_fails(c);
}

TYPED_TEST(QrReconstruction, CompactRAccepted) {
    QrReconCase<TypeParam> c;
    Batched<TypeParam> R = Batched<TypeParam>::matrix(c.k, c.n);
    for (int b = 0; b < kB; ++b)
        for (int j = 0; j < c.n; ++j)
            for (int r = 0; r < c.k; ++r) R.ex(b, r, j) = c.R.ex(b, r, j);
    R.store();
    const double v = batchlas::verify::qr_reconstruction(c.A0.mview(), c.Q.mview(), R.mview());
    EXPECT_LT(v, bound_of(c) / 100);
    Batched<TypeParam> short_r = Batched<TypeParam>::matrix(c.k - 1, c.n);
    EXPECT_THROW(batchlas::verify::qr_reconstruction(c.A0.mview(), c.Q.mview(), short_r.mview()), std::invalid_argument);
}

// ------------------------------------------------------------------ structured solve

namespace {

struct SolveConfig {
    Shape sa;
    Transpose ta;
    int m, k;  // op(A) is m x k
};

template <class T> struct StructSolveCase {
    using type = T;
    static constexpr Check kind = Check::solve;
    static constexpr int nrhs = 4;
    SolveConfig g;
    int bound_n, hot_r, hot_c = nrhs - 1;
    Batched<T> A, X, B0;
    std::vector<std::vector<T>> la;
    double limit = double(1 << 28);

    explicit StructSolveCase(SolveConfig cfg = {Shape::unit_lower, Transpose::ConjTrans, 7, 7})
        : g(cfg), bound_n(std::max(cfg.m, cfg.k)), hot_r(cfg.k - 1),
          A(Batched<T>::matrix(cfg.ta == Transpose::NoTrans ? cfg.m : cfg.k, cfg.ta == Transpose::NoTrans ? cfg.k : cfg.m)),
          X(Batched<T>::matrix(cfg.k, nrhs)), B0(Batched<T>::matrix(cfg.m, nrhs)) {
        using D = promoted_t<T>;
        batchlas::verify::Rng rng(121);
        for (int b = 0; b < kB; ++b) {
            const auto a = logical<T>(g.sa, A.rows, A.cols, rng);
            la.push_back(a);
            for (int j = 0; j < A.cols; ++j)
                for (int i = 0; i < A.rows; ++i) A.ex(b, i, j) = stored(g.sa, i, j) ? a[j * A.rows + i] : poison<T>();
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < g.k; ++i) X.ex(b, i, j) = rand_int<T>(rng, 4.0);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < g.m; ++i) {
                    D acc = D(0);
                    for (int l = 0; l < g.k; ++l) acc += op_at(a, A.rows, g.ta, i, l) * up(X.ex(b, l, j));
                    B0.ex(b, i, j) = down<T>(acc);
                }
        }
        A.store();
        X.store();
        B0.store();
    }
    Batched<T>& out() { return X; }
    double value(Shape sa, Transpose ta) {
        return batchlas::verify::detail::solve_residual(A.mview(), sa, ta, X.mview(), B0.mview(), {}, limit);
    }
    double value() { return value(g.sa, g.ta); }
};

}  // namespace

VERIFY_RESIDUAL_SUITE(StructuredSolveResidual, StructSolveCase)

TYPED_TEST(StructuredSolveResidual, ShapesExactIsSmall) {
    const SolveConfig cases[] = {
        {Shape::general, Transpose::Trans, 6, 5},       {Shape::general, Transpose::ConjTrans, 5, 7},
        {Shape::symmetric_lower, Transpose::NoTrans, 6, 6}, {Shape::hermitian_upper, Transpose::Trans, 6, 6},
        {Shape::upper, Transpose::NoTrans, 6, 6},       {Shape::lower, Transpose::ConjTrans, 6, 6},
        {Shape::unit_upper, Transpose::Trans, 6, 6},    {Shape::general, Transpose::NoTrans, 7, 4},
    };
    for (const SolveConfig& g : cases) {
        StructSolveCase<TypeParam> c(g);
        SCOPED_TRACE(int(g.sa) * 10 + int(g.ta));
        expect_small(c);
        c.limit = 0.0;
        EXPECT_LT(c.value(), bound_of(c));
    }
}

TYPED_TEST(StructuredSolveResidual, WrongStructureFails) {
    StructSolveCase<TypeParam> c({Shape::lower, Transpose::Trans, 6, 6});
    EXPECT_GT(c.value(Shape::lower, Transpose::NoTrans), bound_of(c) * 100);
    EXPECT_GT(c.value(Shape::upper, Transpose::Trans), bound_of(c) * 100);
    EXPECT_GT(c.value(Shape::general, Transpose::Trans), bound_of(c) * 100);
    if constexpr (batchlas::verify::is_complex<TypeParam>::value) {
        StructSolveCase<TypeParam> h({Shape::general, Transpose::ConjTrans, 6, 5});
        EXPECT_GT(h.value(Shape::general, Transpose::Trans), bound_of(h) * 100);
    }
}

TYPED_TEST(StructuredSolveResidual, PublicOverloadIsGeneralNoTrans) {
    StructSolveCase<TypeParam> c({Shape::general, Transpose::NoTrans, 7, 4});
    EXPECT_EQ(batchlas::verify::solve_residual(c.A.mview(), c.X.mview(), c.B0.mview()),
              batchlas::verify::solve_residual(c.A.mview(), Shape::general, Transpose::NoTrans, c.X.mview(), c.B0.mview()));
    c.X.at(kB - 1, 0, 0) += delta<TypeParam>();
    const double v = batchlas::verify::solve_residual(c.A.mview(), c.X.mview(), c.B0.mview());
    EXPECT_GT(v, 0.0);
    EXPECT_EQ(v, batchlas::verify::solve_residual(c.A.mview(), Shape::general, Transpose::NoTrans, c.X.mview(), c.B0.mview()));
}

TYPED_TEST(StructuredSolveResidual, ProbeColumnsSeeAWrongEntry) {
    StructSolveCase<TypeParam> c;
    c.limit = 0.0;
    EXPECT_LT(c.value(), bound_of(c));
    c.X.at(kB - 1, 2, 1) += bump<TypeParam>();
    expect_fails(c);
}

// op(A0) = A0^H for a phased permutation A0 has the inverse A0 itself; A0 A0 is not I.
TYPED_TEST(StructuredSolveResidual, InverseOfConjTransposedUnitary) {
    InverseCase<TypeParam> c;
    MatrixView<TypeParam, MatrixFormat::Dense> none(nullptr, 0, 0, 1, 0, kB);
    EXPECT_LT(batchlas::verify::solve_residual(c.A0.mview(), Shape::general, Transpose::ConjTrans, c.A0.mview(), none), bound_of(c) / 100);
    EXPECT_GT(batchlas::verify::solve_residual(c.A0.mview(), Shape::general, Transpose::NoTrans, c.A0.mview(), none), bound_of(c) * 100);
}

// X(i0, j0) += d moves column j0 of op(A) X by d op(A)(:, i0); the denominator is ||A|| ||X + d||.
TYPED_TEST(StructuredSolveResidual, KnownValue) {
    StructSolveCase<TypeParam> c({Shape::hermitian_lower, Transpose::ConjTrans, 6, 6});
    const int b = kB - 1, i0 = c.g.k - 1, j0 = c.nrhs - 1;
    const TypeParam d = delta<TypeParam>();
    const double x2 = nrm2(c.X.ex(b, i0, j0)), xp2 = nrm2(c.X.ex(b, i0, j0) + d);
    c.X.at(b, i0, j0) += d;
    double col = 0, a2 = 0;
    for (int r = 0; r < c.g.m; ++r) col += std::norm(cdouble(op_at(c.la[b], c.A.rows, c.g.ta, r, i0)));
    for (const TypeParam& x : c.la[b]) a2 += nrm2(x);
    expect_value(c, mod(d) * std::sqrt(col) / (std::sqrt(a2) * std::sqrt(frob2(c.X, b) - x2 + xp2)));
}

// ------------------------------------------------------------------ rank-2k

namespace {

struct Rank2kConfig {
    bool herm;
    Transpose trans;
    Uplo uplo;
    bool zero_beta = false;
};

template <class T> struct Rank2kCase {
    using type = T;
    static constexpr Check kind = Check::blas;
    static constexpr int n = 6, k = 5;
    Rank2kConfig g;
    int bound_n = k, hot_r, hot_c;
    Batched<T> A, B, C0, C;
    std::vector<std::vector<T>> la, lb;
    promoted_t<T> alpha = alpha_of<T>(), beta;

    // op(X)(j, l) taken back: conjugated for her2k, plain for syr2k.
    promoted_t<T> back(const std::vector<T>& M, int j, int l) const {
        const auto x = op_at(M, g.trans == Transpose::NoTrans ? n : k, g.trans, j, l);
        return g.herm ? conj(x) : x;
    }

    explicit Rank2kCase(Rank2kConfig cfg = {true, Transpose::NoTrans, Uplo::Lower})
        : g(cfg), hot_r(cfg.uplo == Uplo::Lower ? n - 1 : 0), hot_c(cfg.uplo == Uplo::Lower ? 0 : n - 1),
          A(Batched<T>::matrix(cfg.trans == Transpose::NoTrans ? n : k, cfg.trans == Transpose::NoTrans ? k : n)),
          B(Batched<T>::matrix(cfg.trans == Transpose::NoTrans ? n : k, cfg.trans == Transpose::NoTrans ? k : n)),
          C0(Batched<T>::matrix(n, n)), C(Batched<T>::matrix(n, n)) {
        using D = promoted_t<T>;
        beta = g.zero_beta ? D(0) : g.herm ? D(-2.0) : beta_of<T>();
        const D alpha2 = g.herm ? conj(alpha) : alpha;
        batchlas::verify::Rng rng(131);
        for (int b = 0; b < kB; ++b) {
            for (int j = 0; j < A.cols; ++j)
                for (int i = 0; i < A.rows; ++i) {
                    A.ex(b, i, j) = rand_int<T>(rng, 3.0);
                    B.ex(b, i, j) = rand_int<T>(rng, 3.0);
                }
            la.push_back(A.exact[b]);
            lb.push_back(B.exact[b]);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    if (g.uplo == Uplo::Lower ? i < j : i > j) continue;
                    // A Hermitian C0's diagonal imaginary part is not read: poison it.
                    C0.ex(b, i, j) = g.zero_beta ? make<T>(kNaN, kNaN) : g.herm && i == j ? make<T>(std::round(3.0 * rng.next()), 1000.0) : rand_int<T>(rng, 3.0);
                    D s1 = D(0), s2 = D(0);
                    for (int l = 0; l < k; ++l) {
                        s1 += op_at(la[b], A.rows, g.trans, i, l) * back(lb[b], j, l);
                        s2 += op_at(lb[b], B.rows, g.trans, i, l) * back(la[b], j, l);
                    }
                    D c0 = g.zero_beta ? D(0) : up(C0.ex(b, i, j));
                    if (g.herm && i == j) c0 = D(std::real(c0));
                    C.ex(b, i, j) = down<T>(alpha * s1 + alpha2 * s2 + beta * c0);
                }
        }
        A.store();
        B.store();
        C0.store();
        C.store();
    }
    Batched<T>& out() { return C; }
    double value(bool herm, Uplo uplo) {
        return batchlas::verify::rank2k_backward_error(A.mview(), B.mview(), g.trans, C0.mview(), C.mview(), uplo, alpha, beta, herm);
    }
    double value() { return value(g.herm, g.uplo); }
};

template <class T> struct Rank2kSymCase : Rank2kCase<T> {
    Rank2kSymCase() : Rank2kCase<T>({false, Transpose::Trans, Uplo::Upper}) {}
};

}  // namespace

VERIFY_RESIDUAL_SUITE(Rank2kBackwardError, Rank2kCase)
VERIFY_RESIDUAL_SUITE(Rank2kSymBackwardError, Rank2kSymCase)

TYPED_TEST(Rank2kBackwardError, VariantsExactIsSmall) {
    const Rank2kConfig cases[] = {
        {true, Transpose::ConjTrans, Uplo::Upper}, {false, Transpose::NoTrans, Uplo::Upper},
        {false, Transpose::Trans, Uplo::Lower},    {true, Transpose::NoTrans, Uplo::Upper, true},
    };
    for (const Rank2kConfig& g : cases) {
        Rank2kCase<TypeParam> c(g);
        SCOPED_TRACE(int(g.herm) * 100 + int(g.trans) * 10 + int(g.uplo));
        expect_small(c);
    }
}

TYPED_TEST(Rank2kBackwardError, WrongTriangleFails) {
    Rank2kCase<TypeParam> c;
    EXPECT_GT(c.value(true, Uplo::Upper), bound_of(c) * 100);
}

TYPED_TEST(Rank2kBackwardError, WrongVariantFails) {
    if constexpr (batchlas::verify::is_complex<TypeParam>::value) {
        Rank2kCase<TypeParam> s({false, Transpose::NoTrans, Uplo::Lower, true});
        EXPECT_GT(s.value(true, Uplo::Lower), bound_of(s) * 100);
        Rank2kCase<TypeParam> h({true, Transpose::NoTrans, Uplo::Lower, true});
        EXPECT_GT(h.value(false, Uplo::Lower), bound_of(h) * 100);
    } else {
        Rank2kCase<TypeParam> c;  // real: her2k and syr2k coincide
        EXPECT_LT(c.value(false, Uplo::Lower), bound_of(c) / 100);
    }
}

TYPED_TEST(Rank2kBackwardError, InvalidCombinationsThrow) {
    Rank2kCase<TypeParam> c;
    if constexpr (batchlas::verify::is_complex<TypeParam>::value) {
        Rank2kCase<TypeParam> t({false, Transpose::Trans, Uplo::Lower});
        EXPECT_THROW(batchlas::verify::rank2k_backward_error(t.A.mview(), t.B.mview(), Transpose::Trans, t.C0.mview(), t.C.mview(), Uplo::Lower,
                                                             t.alpha, promoted_t<TypeParam>(1.0), true),
                     std::invalid_argument);
        EXPECT_THROW(batchlas::verify::rank2k_backward_error(t.A.mview(), t.B.mview(), Transpose::ConjTrans, t.C0.mview(), t.C.mview(), Uplo::Lower,
                                                             t.alpha, t.beta, false),
                     std::invalid_argument);
        EXPECT_THROW(batchlas::verify::rank2k_backward_error(c.A.mview(), c.B.mview(), Transpose::NoTrans, c.C0.mview(), c.C.mview(), Uplo::Lower,
                                                             c.alpha, beta_of<TypeParam>(), true),
                     std::invalid_argument);
    }
    Batched<TypeParam> wide = Batched<TypeParam>::matrix(c.n, c.n + 1);
    EXPECT_THROW(batchlas::verify::rank2k_backward_error(c.A.mview(), c.B.mview(), Transpose::NoTrans, c.C0.mview(), wide.mview(), Uplo::Lower,
                                                         c.alpha, c.beta, true),
                 std::invalid_argument);
}

namespace {

// Only C(i0, j0) is off, by d, over |alpha| (sum |opA(i0,:)||opB(j0,:)| + |opB(i0,:)||opA(j0,:)|) + |beta||C0|.
template <class T> void rank2k_known(const Rank2kConfig& g) {
    Rank2kCase<T> c(g);
    const int b = kB - 1, i0 = c.hot_r, j0 = c.hot_c;
    const T d = delta<T>();
    c.C.at(b, i0, j0) += d;
    double mag = 0;
    for (int l = 0; l < c.k; ++l)
        mag += std::abs(cdouble(op_at(c.la[b], c.A.rows, g.trans, i0, l))) * std::abs(cdouble(op_at(c.lb[b], c.B.rows, g.trans, j0, l))) +
               std::abs(cdouble(op_at(c.lb[b], c.B.rows, g.trans, i0, l))) * std::abs(cdouble(op_at(c.la[b], c.A.rows, g.trans, j0, l)));
    expect_value(c, mod(d) / (mod(c.alpha) * mag + mod(c.beta) * mod(c.C0.ex(b, i0, j0))));
}

}  // namespace

TYPED_TEST(Rank2kBackwardError, KnownValue) { rank2k_known<TypeParam>({true, Transpose::NoTrans, Uplo::Lower}); }
TYPED_TEST(Rank2kSymBackwardError, KnownValue) { rank2k_known<TypeParam>({false, Transpose::Trans, Uplo::Upper}); }

// A Hermitian diagonal: |C0(i,i)| counts its real part only (the imaginary part is poison).
TYPED_TEST(Rank2kBackwardError, HermitianDiagonalKnownValue) {
    Rank2kCase<TypeParam> c;
    const int b = kB - 1, i0 = c.n - 1;
    const TypeParam d = delta<TypeParam>();
    c.C.at(b, i0, i0) += d;
    double mag = 0;
    for (int l = 0; l < c.k; ++l) mag += 2.0 * std::abs(cdouble(op_at(c.la[b], c.A.rows, c.g.trans, i0, l))) * std::abs(cdouble(op_at(c.lb[b], c.B.rows, c.g.trans, i0, l)));
    expect_value(c, mod(d) / (mod(c.alpha) * mag + mod(c.beta) * std::abs(cdouble(up(c.C0.ex(b, i0, i0))).real())));
}

// ------------------------------------------------------------------ LU solve from factors

namespace {

template <class T> struct LuSolveCase {
    using type = T;
    static constexpr Check kind = Check::solve;
    static constexpr int n = 7, nrhs = 4;
    Transpose trans;
    int bound_n = n, hot_r = n - 1, hot_c = 0;
    Batched<T> F = Batched<T>::matrix(n, n), X = Batched<T>::matrix(n, nrhs), B0 = Batched<T>::matrix(n, nrhs);
    std::vector<std::int32_t> piv;
    int pinc = 2, pstride = 2 * n + 3;
    std::vector<std::vector<promoted_t<T>>> M;  // P^T L U per item, column-major

    promoted_t<T> op_m(int b, int i, int j) const {
        if (trans == Transpose::NoTrans) return M[b][j * n + i];
        return trans == Transpose::ConjTrans ? conj(M[b][i * n + j]) : M[b][i * n + j];
    }

    explicit LuSolveCase(Transpose t = Transpose::ConjTrans) : trans(t) {
        using D = promoted_t<T>;
        piv.assign(static_cast<std::size_t>(pstride * kB), -99);
        batchlas::verify::Rng rng(141);
        for (int b = 0; b < kB; ++b) {
            std::vector<D> L(n * n, D(0)), U(n * n, D(0)), A(n * n, D(0));
            for (int j = 0; j < n; ++j) {
                L[j * n + j] = D(1);
                for (int i = j + 1; i < n; ++i) L[j * n + i] = up(rand_int<T>(rng, 2.0));
                for (int i = 0; i <= j; ++i) U[j * n + i] = up(i == j ? make<T>(5.0, 1.0) : rand_int<T>(rng, 3.0));
            }
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    for (int l = 0; l < n; ++l) A[j * n + i] += L[l * n + i] * U[j * n + l];
                    F.ex(b, i, j) = down<T>(i > j ? L[j * n + i] : U[j * n + i]);
                }
            std::vector<int> ip(n);
            for (int k = 0; k < n; ++k) {
                ip[k] = k + int((rng.next() * 0.5 + 0.5) * double(n - k)) % (n - k);
                piv[static_cast<std::size_t>(b * pstride + k * pinc)] = ip[k] + 1;
            }
            for (int k = n - 1; k >= 0; --k)
                for (int c = 0; c < n; ++c) std::swap(A[c * n + k], A[c * n + ip[k]]);
            M.push_back(A);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) X.ex(b, i, j) = rand_int<T>(rng, 3.0);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) {
                    D acc = D(0);
                    for (int l = 0; l < n; ++l) acc += op_m(b, i, l) * up(X.ex(b, l, j));
                    B0.ex(b, i, j) = down<T>(acc);
                }
        }
        F.store();
        X.store();
        B0.store();
    }
    VectorView<std::int32_t> pview() { return VectorView<std::int32_t>(piv.data(), n, kB, pinc, pstride); }
    Batched<T>& out() { return F; }
    double value() { return batchlas::verify::lu_solve_residual(F.mview(), pview(), trans, X.mview(), B0.mview()); }
};

}  // namespace

VERIFY_RESIDUAL_SUITE(LuSolveResidual, LuSolveCase)

TYPED_TEST(LuSolveResidual, TransposeFormsExactIsSmall) {
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        LuSolveCase<TypeParam> c(t);
        SCOPED_TRACE(int(t));
        expect_small(c);
    }
}

TYPED_TEST(LuSolveResidual, WrongTransposeFails) {
    LuSolveCase<TypeParam> c(Transpose::NoTrans);
    c.trans = Transpose::Trans;
    expect_fails(c);
}

TYPED_TEST(LuSolveResidual, PerturbedXFails) {
    LuSolveCase<TypeParam> c;
    c.X.at(kB - 1, c.n - 1, c.nrhs - 1) += bump<TypeParam>();
    expect_fails(c);
}

TYPED_TEST(LuSolveResidual, WrongPivotFails) {
    LuSolveCase<TypeParam> c;
    int& p = c.piv[static_cast<std::size_t>((kB - 1) * c.pstride + 1 * c.pinc)];
    p = p == c.n ? 2 : p + 1;
    expect_fails(c);
}

TYPED_TEST(LuSolveResidual, BadPivotIsNan) {
    LuSolveCase<TypeParam> c;
    c.piv[static_cast<std::size_t>((kB - 1) * c.pstride + 3 * c.pinc)] = c.n + 1;
    EXPECT_TRUE(std::isnan(c.value()));
    c.piv[static_cast<std::size_t>((kB - 1) * c.pstride + 3 * c.pinc)] = 0;
    EXPECT_TRUE(std::isnan(c.value()));
}

// X(i0, j0) += d moves column j0 of op(M) X by d op(M)(:, i0); the denominator is ||M|| ||X + d||.
TYPED_TEST(LuSolveResidual, KnownValue) {
    LuSolveCase<TypeParam> c;
    const int b = kB - 1, i0 = c.n - 1, j0 = c.nrhs - 1;
    const TypeParam d = delta<TypeParam>();
    const double x2 = nrm2(c.X.ex(b, i0, j0)), xp2 = nrm2(c.X.ex(b, i0, j0) + d);
    c.X.at(b, i0, j0) += d;
    double col = 0, m2 = 0;
    for (int r = 0; r < c.n; ++r) col += std::norm(cdouble(c.op_m(b, r, i0)));
    for (const auto& x : c.M[b]) m2 += std::norm(cdouble(x));
    expect_value(c, mod(d) * std::sqrt(col) / (std::sqrt(m2) * std::sqrt(frob2(c.X, b) - x2 + xp2)));
}

// getri from factors: L = I, U = unit phases, so M = P^T U is unitary and op(M)^{-1} = op(M)^H.
TYPED_TEST(LuSolveResidual, InverseFromFactors) {
    using T = TypeParam;
    LuSolveCase<T> c;
    const int n = c.n;
    Batched<T> Xi = Batched<T>::matrix(n, n);
    for (int b = 0; b < kB; ++b) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) c.F.ex(b, i, j) = i == j ? phase<T>(i + 2 * b + 1) : make<T>(0.0, 0.0);
        auto& Mb = c.M[b];
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) Mb[j * n + i] = up(c.F.ex(b, i, j));
        for (int k = n - 1; k >= 0; --k) {
            const int ip = c.piv[static_cast<std::size_t>(b * c.pstride + k * c.pinc)] - 1;
            for (int col = 0; col < n; ++col) std::swap(Mb[col * n + k], Mb[col * n + ip]);
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) Xi.ex(b, i, j) = down<T>(conj(c.op_m(b, j, i)));
    }
    c.F.store();
    Xi.store();
    MatrixView<T, MatrixFormat::Dense> none(nullptr, 0, 0, 1, 0, kB);
    EXPECT_LT(batchlas::verify::lu_solve_residual(c.F.mview(), c.pview(), c.trans, Xi.mview(), none), bound_of(c) / 100);
    Xi.at(kB - 1, n - 1, 0) += bump<T>();
    EXPECT_GT(batchlas::verify::lu_solve_residual(c.F.mview(), c.pview(), c.trans, Xi.mview(), none), bound_of(c) * 100);
}

// ------------------------------------------------------------------ pivot ratio

namespace {

// U(k,k) = 4 * phase, L(i,k) = a / U(k,k) with cabs1(a) <= 4: a cabs1 partial-pivoting factor whose
// ratio cabs1(L U(k,k)) / cabs1(U(k,k)) = cabs1(a) / 4 exactly.
template <class T> struct PivotCase {
    static constexpr int n = 6;
    Batched<T> F = Batched<T>::matrix(n, n);

    PivotCase() {
        batchlas::verify::Rng rng(151);
        for (int b = 0; b < kB; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const T u = down<T>(up(phase<T>(j + b)) * 4.0);
                    if (i < j) F.ex(b, i, j) = rand_int<T>(rng, 5.0);
                    else if (i == j) F.ex(b, i, j) = u;
                    else F.ex(b, i, j) = down<T>(up(rand_int<T>(rng, 2.0)) / up(u));
                }
        F.store();
    }
    void set_a(int b, int i, int j, T a) { F.at(b, i, j) = down<T>(up(a) / up(F.ex(b, j, j))); }
    double value() { return batchlas::verify::pivot_ratio(F.mview()); }
};

template <class T> void expect_ratio_fails(double v) {
    EXPECT_TRUE(std::isnan(v) || v > 1.1 * batchlas::verify::pivot_ratio_bound<T>()) << "ratio " << v;
}

}  // namespace

template <class T> class PivotRatio : public ::testing::Test {};
TYPED_TEST_SUITE(PivotRatio, Dtypes);

TYPED_TEST(PivotRatio, ExactPasses) {
    PivotCase<TypeParam> c;
    const double v = c.value();
    EXPECT_LE(v, 1.0);
    EXPECT_GT(v, 0.0);
}

TYPED_TEST(PivotRatio, KnownValue) {
    PivotCase<TypeParam> c;
    c.set_a(kB - 1, c.n - 1, 0, make<TypeParam>(5.0, -2.0));
    EXPECT_EQ(c.value(), batchlas::verify::is_complex<TypeParam>::value ? 7.0 / 4.0 : 5.0 / 4.0);
}

TYPED_TEST(PivotRatio, PerturbedElementFails) {
    PivotCase<TypeParam> c;
    c.F.at(kB / 2, c.n - 1, 0) += bump<TypeParam>();
    expect_ratio_fails<TypeParam>(c.value());
}
TYPED_TEST(PivotRatio, WrongLdFails) {
    PivotCase<TypeParam> c;
    c.F.store_wrong_ld();
    expect_ratio_fails<TypeParam>(c.value());
}
TYPED_TEST(PivotRatio, WrongStrideFails) {
    PivotCase<TypeParam> c;
    c.F.store_wrong_stride();
    expect_ratio_fails<TypeParam>(c.value());
}
TYPED_TEST(PivotRatio, LastItemOnlyFails) {
    PivotCase<TypeParam> c;
    c.F.at(kB - 1, c.n - 1, c.n - 2) += bump<TypeParam>();
    expect_ratio_fails<TypeParam>(c.value());
}
TYPED_TEST(PivotRatio, NanFails) {
    PivotCase<TypeParam> c;
    c.F.at(kB / 2, 3, 2) = make<TypeParam>(kNaN, 0.0);
    EXPECT_TRUE(std::isnan(c.value()));
}

// |a| = 3.54 < |U(k,k)| = 4 but cabs1(a) = 5 > 4: a modulus pivot rule passes, cabs1 does not.
TYPED_TEST(PivotRatio, ModulusPivotFails) {
    if constexpr (batchlas::verify::is_complex<TypeParam>::value) {
        PivotCase<TypeParam> c;
        c.set_a(kB - 1, 4, 1, make<TypeParam>(2.5, 2.5));
        EXPECT_EQ(c.value(), 1.25);
    }
}

// U(0,0) = 2 + 2i (cabs1 4, modulus 2.83) and L(n-1,0) U(0,0) = 3 - i: the ratio is 4 / 4, not 4 / 2.83.
TYPED_TEST(PivotRatio, DenominatorIsCabs1) {
    if constexpr (batchlas::verify::is_complex<TypeParam>::value) {
        PivotCase<TypeParam> c;
        const int b = kB - 1;
        c.F.ex(b, 0, 0) = make<TypeParam>(2.0, 2.0);
        c.F.at(b, 0, 0) = c.F.ex(b, 0, 0);
        for (int i = 1; i < c.n; ++i) c.F.at(b, i, 0) = make<TypeParam>(0.0, 0.0);
        c.set_a(b, c.n - 1, 0, make<TypeParam>(3.0, -1.0));
        EXPECT_EQ(c.value(), 1.0);
    }
}

TYPED_TEST(PivotRatio, ZeroPivotColumnSkipped) {
    PivotCase<TypeParam> c;
    c.F.at(kB - 1, 2, 2) = make<TypeParam>(0.0, 0.0);
    c.F.at(kB - 1, 4, 2) = bump<TypeParam>();
    EXPECT_LE(c.value(), 1.0);
}

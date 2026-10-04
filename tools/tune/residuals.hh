#pragma once

// Inputs and host verification for the tuner's specs, ported from benchmarks/factor_bench.cc
// (fill_spd, potrf_residual, getrs_residual and their bounds) so a tuned time is checked by the
// same residuals as a factor_bench sweep row: double promotion, items 0 AND batch-1 (item 0
// alone is blind to a wrong batch stride), NaN-propagating maxima.

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace batchlas::tune {

template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };

inline double ab(double x) { return std::fabs(x); }
inline double ab(std::complex<double> x) { return std::abs(x); }
inline double cj(double x) { return x; }
inline std::complex<double> cj(std::complex<double> x) { return std::conj(x); }
inline double up(float x) { return double(x); }
inline double up(double x) { return x; }
inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
inline std::complex<double> up(std::complex<double> x) { return x; }

// std::max(a, NaN) returns a, so a poisoned probe would read as perfect.
inline double nanmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return a > b ? a : b;
}

template <class T> inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) { return {float(re), float(im)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) { return {re, im}; }

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {  // uniform in [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(std::int32_t(std::uint32_t(s >> 32))) / 2147483648.0;
    }
};

// The inputs are conditioned O(1) by construction, so these bounds are generous, not tuned.
template <class T> struct Tol;
template <> struct Tol<float> { static constexpr double v = 1e-4; };
template <> struct Tol<std::complex<float>> { static constexpr double v = 1e-4; };
template <> struct Tol<double> { static constexpr double v = 1e-11; };
template <> struct Tol<std::complex<double>> { static constexpr double v = 1e-11; };

// Diagonally dominant SPD with condition number near 1 (factor_bench's input): a nonzero info
// is the implementation, never the input.
template <class T>
void fill_spd(T* A0, int n, int ld, std::size_t stride, int batch) {
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const double v = (r == c) ? double(n) + 1.0 : 0.5 / (1.0 + std::abs(r - c));
                A0[std::size_t(b) * stride + std::size_t(c) * std::size_t(ld) + std::size_t(r)] = mk<T>(v, 0.0);
            }
}

// || A0 - L L^H ||_F / || A0 ||_F over the factored triangle (Upper: U^H U, U(k,i) at F[i*ld+k]).
template <class T>
double potrf_residual(const T* F, const T* A0, int n, int ld, std::size_t stride, int batch, bool upper) {
    using D = typename Prom<T>::type;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        const std::size_t o = std::size_t(b) * stride;
        auto at = [&](const T* M, int r, int c) { return up(M[o + std::size_t(c) * std::size_t(ld) + std::size_t(r)]); };
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = (upper ? 0 : j); upper ? (i <= j) : (i < n); ++i) {
                D acc = D(0);
                if (upper)
                    for (int k = 0; k <= i; ++k) acc += cj(at(F, k, i)) * at(F, k, j);
                else
                    for (int k = 0; k <= j; ++k) acc += at(F, i, k) * cj(at(F, j, k));
                const D a = at(A0, i, j);
                const double d = ab(acc - a), r = ab(a);
                num += d * d;
                den += r * r;
            }
        if (std::isnan(num) || std::isnan(den)) return std::numeric_limits<double>::quiet_NaN();
        worst = nanmax(worst, den > 0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num));
    }
    return worst;
}

// || A0 X - B0 ||_F / (|| A0 ||_F || X ||_F), A0 held in full.
template <class T>
double solve_residual(const T* X, const T* B0, const T* A0, int n, int nrhs, int lda, std::size_t sa, int ldb,
                      std::size_t sb, int batch) {
    using D = typename Prom<T>::type;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        const std::size_t oa = std::size_t(b) * sa, ob = std::size_t(b) * sb;
        auto a = [&](int r, int c) { return up(A0[oa + std::size_t(c) * std::size_t(lda) + std::size_t(r)]); };
        auto x = [&](int r, int c) { return up(X[ob + std::size_t(c) * std::size_t(ldb) + std::size_t(r)]); };
        double na = 0, nx = 0, num = 0;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) na += ab(a(r, c)) * ab(a(r, c));
        for (int c = 0; c < nrhs; ++c)
            for (int r = 0; r < n; ++r) {
                nx += ab(x(r, c)) * ab(x(r, c));
                D acc = D(0);
                for (int k = 0; k < n; ++k) acc += a(r, k) * x(k, c);
                const double d = ab(acc - up(B0[ob + std::size_t(c) * std::size_t(ldb) + std::size_t(r)]));
                num += d * d;
            }
        if (std::isnan(num) || std::isnan(na) || std::isnan(nx)) return std::numeric_limits<double>::quiet_NaN();
        const double den = std::sqrt(na) * std::sqrt(nx);
        worst = nanmax(worst, den > 0 ? std::sqrt(num) / den : std::sqrt(num));
    }
    return worst;
}

}  // namespace batchlas::tune

#pragma once

// LU inputs and host verification for the getrf, getrs, getri and gesv specs, ported from
// benchmarks/factor_bench.cc (fill_lu, getrf_residual) on top of residuals.hh: double promotion,
// items 0 and batch-1, NaN-propagating maxima. Pivots are the packed 1-based int32 that every
// GPU family writes into the int64 span's first half, n per item (src/extensions/getrf_native.hh).

#include "residuals.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace batchlas::tune {

// Diagonally dominant with complex noise (nonzero imaginary parts), THEN row-permuted per item:
// on the dominant matrix alone partial pivoting picks the diagonal, the pivots are the identity
// and a dropped interchange leaves the residual unchanged (factor_bench's recorded break).
template <class T>
void fill_lu(T* A0, int n, int ld, std::size_t stride, int batch, std::uint64_t seed) {
    Rng rg(seed);
    std::vector<T> col(static_cast<std::size_t>(n));
    std::vector<int> perm(static_cast<std::size_t>(n));
    for (int b = 0; b < batch; ++b) {
        T* a = A0 + std::size_t(b) * stride;
        auto at = [&](int r, int c) -> T& { return a[std::size_t(c) * std::size_t(ld) + std::size_t(r)]; };
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) at(r, c) = mk<T>(rg.next(), rg.next());
        for (int i = 0; i < n; ++i) at(i, i) = at(i, i) + mk<T>(double(n), 0.0);
        for (int i = 0; i < n; ++i) perm[std::size_t(i)] = i;
        for (int i = n - 1; i > 0; --i) {
            const int j = int((rg.next() * 0.5 + 0.5) * double(i + 1)) % (i + 1);
            std::swap(perm[std::size_t(i)], perm[std::size_t(j)]);
        }
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < n; ++i) col[std::size_t(i)] = at(perm[std::size_t(i)], c);
            for (int i = 0; i < n; ++i) at(i, c) = col[std::size_t(i)];
        }
    }
}

// || P A0 - L U ||_F / || A0 ||_F, P rebuilt from the device pivots; an out-of-range pivot is NaN.
template <class T>
double getrf_residual(const T* F, const T* A0, const std::int32_t* piv, int n, int ld, std::size_t stride,
                      int batch) {
    using D = typename Prom<T>::type;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double worst = 0;
    std::vector<D> PA(std::size_t(n) * std::size_t(n));
    for (int b : {0, batch - 1}) {
        const std::size_t o = std::size_t(b) * stride;
        auto f = [&](int r, int c) { return up(F[o + std::size_t(c) * std::size_t(ld) + std::size_t(r)]); };
        auto pa = [&](int r, int c) -> D& { return PA[std::size_t(c) * std::size_t(n) + std::size_t(r)]; };
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) pa(r, c) = up(A0[o + std::size_t(c) * std::size_t(ld) + std::size_t(r)]);
        const std::int32_t* pv = piv + std::size_t(b) * std::size_t(n);
        for (int k = 0; k < n; ++k) {
            const int ip = pv[k] - 1;
            if (ip < 0 || ip >= n) return nan;
            if (ip != k)
                for (int c = 0; c < n; ++c) std::swap(pa(k, c), pa(ip, c));
        }
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                for (int k = 0; k <= std::min(i, j); ++k) acc += (k == i ? D(1) : f(i, k)) * f(k, j);
                const double d = ab(acc - pa(i, j)), r = ab(pa(i, j));
                num += d * d;
                den += r * r;
            }
        if (std::isnan(num) || std::isnan(den)) return nan;
        worst = nanmax(worst, den > 0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num));
    }
    return worst;
}

// || A0 X V - B0 V ||_F / (|| A0 ||_F || X V ||_F) with B0 = I when null (getri: X = inv(A0)).
// V is the identity (the full residual) while n^2 * ncols <= 2^28, else 4 random dense probe
// columns: O(n^2) host work at n = 4096, and a dense probe still sees any wrong entry of X.
template <class T>
double lu_solve_residual(const T* X, const T* B0, const T* A0, int n, int ncols, int lda, std::size_t sa, int ldx,
                         std::size_t sx, int batch) {
    using D = typename Prom<T>::type;
    const bool full = double(n) * double(n) * double(ncols) <= double(1 << 28);
    const int nv = full ? ncols : 4;
    std::vector<D> V(full ? 0 : std::size_t(ncols) * std::size_t(nv));
    std::vector<D> xv(static_cast<std::size_t>(n)), bv(static_cast<std::size_t>(n));
    Rng rg(4242);
    for (D& v : V) v = up(mk<T>(rg.next(), rg.next()));
    double worst = 0;
    for (int b : {0, batch - 1}) {
        const std::size_t oa = std::size_t(b) * sa, ox = std::size_t(b) * sx;
        auto a = [&](int r, int c) { return up(A0[oa + std::size_t(c) * std::size_t(lda) + std::size_t(r)]); };
        auto x = [&](int r, int c) { return up(X[ox + std::size_t(c) * std::size_t(ldx) + std::size_t(r)]); };
        auto rhs = [&](int r, int c) {
            return B0 ? up(B0[ox + std::size_t(c) * std::size_t(ldx) + std::size_t(r)]) : D(r == c ? 1 : 0);
        };
        auto v = [&](int c, int j) { return full ? D(c == j ? 1 : 0) : V[std::size_t(j) * std::size_t(ncols) + std::size_t(c)]; };
        double na = 0, nx = 0, num = 0;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) na += ab(a(r, c)) * ab(a(r, c));
        for (int j = 0; j < nv; ++j) {
            std::fill(xv.begin(), xv.end(), D(0));
            std::fill(bv.begin(), bv.end(), D(0));
            for (int c = 0; c < ncols; ++c) {
                const D w = v(c, j);
                if (w == D(0)) continue;
                for (int r = 0; r < n; ++r) xv[std::size_t(r)] += x(r, c) * w, bv[std::size_t(r)] += rhs(r, c) * w;
            }
            for (int r = 0; r < n; ++r) {
                nx += ab(xv[std::size_t(r)]) * ab(xv[std::size_t(r)]);
                D acc = D(0);
                for (int k = 0; k < n; ++k) acc += a(r, k) * xv[std::size_t(k)];
                const double d = ab(acc - bv[std::size_t(r)]);
                num += d * d;
            }
        }
        if (std::isnan(num) || std::isnan(na) || std::isnan(nx)) return std::numeric_limits<double>::quiet_NaN();
        const double den = std::sqrt(na) * std::sqrt(nx);
        worst = nanmax(worst, den > 0 ? std::sqrt(num) / den : std::sqrt(num));
    }
    return worst;
}

}  // namespace batchlas::tune

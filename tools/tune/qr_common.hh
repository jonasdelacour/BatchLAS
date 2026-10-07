#pragma once

// Host Householder helpers for the geqrf, orgqr and ormqr specs. Reflectors are LAPACK storage:
// v_i is 1 at row i, A(r, i) below it and 0 above, H_i = I - tau_i v_i v_i^H. The inputs of
// orgqr and ormqr come from host_reflectors (larfg on random columns), never from a tuned geqrf,
// and every reference is a host walk over the same storage in double / complex<double>
// (docs/developer/agent-guide.md §8 rule 7: no self-referential reference path).

#include "residuals.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace batchlas::tune {

// geqrf and orgqr have no batch key, so a cell runs at the largest power-of-two batch whose input
// fits kQrItemBudget, within [2, 16384]: saturating at small n (agent guide §10), memory-bounded at
// large n. Two items keep 0 and batch-1 distinct, and batch 1 would send geqrf's vendor arm to
// cusolverDnXgeqrf instead of the geqrfBatched every other cell times.
inline constexpr double kQrItemBudget = 256.0 * 1024 * 1024;
inline int qr_batch(double item_bytes) {
    int b = 2;
    while (b < 16384 && double(2 * b) * item_bytes <= kQrItemBudget) b *= 2;
    return b;
}

// The bound grows like sqrt(m) with the reflector length (m reaches 2^18 on tall cells).
template <class T>
double qr_tol(int m) {
    return Tol<T>::v * std::max(1.0, std::sqrt(double(m) / 1024.0));
}

// Up to `cap` indices of [0, n): all of them, or spread with the first and last always kept.
inline std::vector<int> sample_idx(int n, int cap) {
    std::vector<int> out;
    for (int i = 0; i < std::min(n, cap); ++i)
        out.push_back(n <= cap ? i : int(std::int64_t(i) * (n - 1) / (cap - 1)));
    return out;
}

template <class T>
struct Reflectors {  // item `o`'s reflectors, read in promoted precision
    const T* A;
    const T* tau;
    std::size_t o, to;
    int ld, m;
    using D = typename Prom<T>::type;
    D v(int r, int i) const { return r == i ? D(1) : up(A[o + std::size_t(i) * std::size_t(ld) + std::size_t(r)]); }
    D t(int i) const { return up(tau[to + std::size_t(i)]); }
    // x := H_i x (herm: H_i^H x) for a column x in C^m.
    void left(std::vector<D>& x, int i, bool herm) const {
        D s = D(0);
        for (int r = i; r < m; ++r) s += cj(v(r, i)) * x[std::size_t(r)];
        s *= herm ? cj(t(i)) : t(i);
        for (int r = i; r < m; ++r) x[std::size_t(r)] -= s * v(r, i);
    }
    // |2 Re tau - |tau|^2 ||v||^2|: zero exactly when H_i is unitary.
    double unitarity(int i) const {
        double vv = 0;
        for (int r = i; r < m; ++r) vv += ab(v(r, i)) * ab(v(r, i));
        const D ti = t(i);
        return std::fabs(2.0 * std::real(ti) - ab(ti) * ab(ti) * vv);
    }
};

// Q e_c = H_0 ... H_{k-1} e_c; only H_0..H_c touch e_c.
template <class T>
std::vector<typename Prom<T>::type> q_column(const Reflectors<T>& R, int k, int c) {
    std::vector<typename Prom<T>::type> x(std::size_t(R.m), typename Prom<T>::type(0));
    x[std::size_t(c)] = 1;
    for (int i = std::min(c, k - 1); i >= 0; --i) R.left(x, i, false);
    return x;
}

// larfg on a random column per reflector (the safmin rescale aside): a reflector set a real geqrf
// could return, complex tau off the real axis, tau = 0 where nothing is annihilated. The diagonal
// and above hold large finite values the reflectors must never read.
template <class T>
void host_reflectors(T* A, T* tau, int m, int k, int ld, std::size_t stride, int batch, std::uint64_t seed) {
    using D = typename Prom<T>::type;
    Rng rg(seed);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < k; ++i) {
            T* col = A + std::size_t(b) * stride + std::size_t(i) * std::size_t(ld);
            for (int r = 0; r <= i; ++r) col[r] = mk<T>(64.0 + 8.0 * rg.next(), 32.0 * rg.next());
            const D alpha = up(mk<T>(rg.next(), rg.next()));
            double xx = 0;
            std::vector<D> x(std::size_t(m - i - 1));
            for (D& e : x) e = up(mk<T>(rg.next(), rg.next())), xx += ab(e) * ab(e);
            D t = D(0);
            if (xx > 0 || std::imag(alpha) != 0) {
                const double beta = -std::copysign(std::sqrt(ab(alpha) * ab(alpha) + xx), std::real(alpha));
                t = (D(beta) - alpha) / D(beta);
                const D scale = D(1) / (alpha - D(beta));
                for (std::size_t r = 0; r < x.size(); ++r) x[r] *= scale;
            }
            for (std::size_t r = 0; r < x.size(); ++r)
                col[std::size_t(i) + 1 + r] = mk<T>(std::real(x[r]), std::imag(x[r]));
            tau[std::size_t(b) * std::size_t(k) + std::size_t(i)] = mk<T>(std::real(t), std::imag(t));
        }
}

}  // namespace batchlas::tune

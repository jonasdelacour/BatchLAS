// SPDX-License-Identifier: MIT
// Input generators of batchlas::verify (docs/design/verification.md). Host only, seeded, and written
// through ld and stride: padding (rows >= rows() up to ld(), gaps between items) is never touched.
#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/verify/norms.hh>
#include <batchlas/verify/scalar.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace batchlas::verify {
namespace detail {

template <class View>
using value_of_t = std::remove_const_t<std::remove_pointer_t<decltype(item_of(std::declval<const View&>(), 0).data)>>;

// item_of hands out const data for the read-only norms; the view itself owns mutable storage.
template <class T>
T* writable(const Item<T>& m) { return const_cast<T*>(m.data); }

template <class T>
T& at(T* base, int ld, int r, int c) { return base[static_cast<long long>(c) * ld + r]; }

template <class View>
void require_square(const View& A, const char* who) {
    if (A.rows() != A.cols()) throw std::invalid_argument(std::string("batchlas::verify::") + who + ": square matrices only");
}

// make<T>(rng.next(), rng.next()) leaves the call order unspecified; factor_bench's goldens are
// clang's left-to-right, so sequence the draws explicitly.
template <class T>
T draw(Rng& rng) {
    const double re = rng.next();
    const double im = rng.next();
    return make<T>(re, im);
}

template <class T>
T draw_normal(Rng& rng) {
    const double re = rng.normal();
    const double im = rng.normal();
    return make<T>(re, im);
}

}  // namespace detail

/// Diagonally dominant SPD: diagonal n+1, off-diagonal 0.5/(1+|r-c|). Deterministic, no seed.
template <class View> void fill_spd(const View& A) {
    using T = detail::value_of_t<View>;
    detail::require_square(A, "fill_spd");
    const int n = A.rows();
    for (int b = 0; b < A.batch_size(); ++b) {
        T* p = detail::writable(detail::item_of(A, b));
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r)
                detail::at(p, A.ld(), r, c) = make<T>((r == c) ? double(n) + 1.0 : 0.5 / (1.0 + std::abs(r - c)), 0.0);
    }
}

/// Dominant random matrix, then row-permuted per item, so partial pivoting is not the identity.
template <class View> void fill_lu(const View& A, std::uint64_t seed) {
    using T = detail::value_of_t<View>;
    detail::require_square(A, "fill_lu");
    const int n = A.rows();
    const int ld = A.ld();
    Rng rng(seed);
    std::vector<T> col(static_cast<std::size_t>(n));
    std::vector<int> perm(static_cast<std::size_t>(n));
    for (int b = 0; b < A.batch_size(); ++b) {
        T* p = detail::writable(detail::item_of(A, b));
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) detail::at(p, ld, r, c) = detail::draw<T>(rng);
        for (int i = 0; i < n; ++i) detail::at(p, ld, i, i) = detail::at(p, ld, i, i) + make<T>(double(n), 0.0);
        for (int i = 0; i < n; ++i) perm[static_cast<std::size_t>(i)] = i;
        for (int i = n - 1; i > 0; --i) {
            const int j = int((rng.next() * 0.5 + 0.5) * double(i + 1)) % (i + 1);
            std::swap(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(j)]);
        }
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < n; ++i) col[static_cast<std::size_t>(i)] = detail::at(p, ld, perm[static_cast<std::size_t>(i)], c);
            for (int i = 0; i < n; ++i) detail::at(p, ld, i, c) = col[static_cast<std::size_t>(i)];
        }
    }
}

/// Gaussian entries (real and imaginary parts drawn separately; the imaginary draw is made for real T too).
template <class View> void fill_gauss(const View& A, std::uint64_t seed) {
    using T = detail::value_of_t<View>;
    Rng rng(seed);
    for (int b = 0; b < A.batch_size(); ++b) {
        T* p = detail::writable(detail::item_of(A, b));
        for (int c = 0; c < A.cols(); ++c)
            for (int r = 0; r < A.rows(); ++r) detail::at(p, A.ld(), r, c) = detail::draw_normal<T>(rng);
    }
}

/// Uniform [-1, 1) entries, item-major, column-major within an item.
template <class View> void fill_random(const View& A, std::uint64_t seed) {
    using T = detail::value_of_t<View>;
    Rng rng(seed);
    for (int b = 0; b < A.batch_size(); ++b) {
        T* p = detail::writable(detail::item_of(A, b));
        for (int c = 0; c < A.cols(); ++c)
            for (int r = 0; r < A.rows(); ++r) detail::at(p, A.ld(), r, c) = detail::draw<T>(rng);
    }
}

/// Q diag(lambda) Q^H with lambda geometric from 1 down to 10^-log10_cond and Q a product of n
/// Householder reflectors of random vectors. Both triangles are written; the diagonal is real.
template <class View> void fill_graded_hermitian(const View& A, double log10_cond, std::uint64_t seed) {
    using T = detail::value_of_t<View>;
    using D = std::complex<double>;
    detail::require_square(A, "fill_graded_hermitian");
    const int n = A.rows();
    const auto N = static_cast<std::size_t>(n);
    Rng rng(seed);
    std::vector<D> M(N * N), v(N), w(N);
    for (int b = 0; b < A.batch_size(); ++b) {
        std::fill(M.begin(), M.end(), D(0));
        for (int i = 0; i < n; ++i)
            M[static_cast<std::size_t>(i) * (N + 1)] = n == 1 ? 1.0 : std::pow(10.0, -log10_cond * double(i) / double(n - 1));
        for (int k = 0; k < n; ++k) {
            double vv = 0;
            for (auto& e : v) {
                const double re = rng.next();
                // A real matrix needs real reflectors, or the complex Q leaves a non-real result.
                const double im = is_complex<T>::value ? rng.next() : 0.0;
                e = D(re, im);
                vv += std::norm(e);
            }
            if (vv == 0) continue;
            for (auto& e : v) e /= std::sqrt(vv);
            // M <- H M H with H = I - 2 v v^H: two rank-one updates, O(n^2) each.
            for (int c = 0; c < n; ++c) {
                D s = 0;
                for (int r = 0; r < n; ++r) s += std::conj(v[static_cast<std::size_t>(r)]) * M[static_cast<std::size_t>(c) * N + static_cast<std::size_t>(r)];
                for (int r = 0; r < n; ++r) M[static_cast<std::size_t>(c) * N + static_cast<std::size_t>(r)] -= 2.0 * v[static_cast<std::size_t>(r)] * s;
            }
            for (int r = 0; r < n; ++r) {
                D s = 0;
                for (int c = 0; c < n; ++c) s += M[static_cast<std::size_t>(c) * N + static_cast<std::size_t>(r)] * v[static_cast<std::size_t>(c)];
                for (int c = 0; c < n; ++c) M[static_cast<std::size_t>(c) * N + static_cast<std::size_t>(r)] -= 2.0 * s * std::conj(v[static_cast<std::size_t>(c)]);
            }
        }
        T* p = detail::writable(detail::item_of(A, b));
        for (int c = 0; c < n; ++c) {
            detail::at(p, A.ld(), c, c) = make<T>(M[static_cast<std::size_t>(c) * (N + 1)].real(), 0.0);
            for (int r = c + 1; r < n; ++r) {
                const D x = M[static_cast<std::size_t>(c) * N + static_cast<std::size_t>(r)];
                detail::at(p, A.ld(), r, c) = make<T>(x.real(), x.imag());
                detail::at(p, A.ld(), c, r) = make<T>(x.real(), -x.imag());
            }
        }
    }
}

/// LAPACK geqrf storage built on the host: column i holds the larfg reflector of a random column
/// below the diagonal, tau[i] its scalar (tau = 0 where nothing is annihilated, complex tau off the
/// real axis). The diagonal and above hold large finite values the consumers must never read.
template <class ViewA, class T>
void fill_reflectors(const ViewA& A, const VectorView<T>& tau, std::uint64_t seed) {
    static_assert(std::is_same_v<detail::value_of_t<ViewA>, T>, "fill_reflectors: A and tau element types differ");
    using D = promoted_t<T>;
    const int m = A.rows();
    const int k = A.cols();
    if (k > m || tau.size() < k || tau.batch_size() != A.batch_size())
        throw std::invalid_argument("batchlas::verify::fill_reflectors: need cols <= rows, tau.size() >= cols and equal batch sizes");
    Rng rng(seed);
    for (int b = 0; b < A.batch_size(); ++b) {
        T* p = detail::writable(detail::item_of(A, b));
        T* t = tau.data_ptr() + static_cast<long long>(b) * tau.stride();
        for (int i = 0; i < k; ++i) {
            for (int r = 0; r <= i; ++r) {
                const double re = 64.0 + 8.0 * rng.next();
                const double im = 32.0 * rng.next();
                detail::at(p, A.ld(), r, i) = make<T>(re, im);
            }
            const D alpha = up(detail::draw<T>(rng));
            double xx = 0;
            std::vector<D> x(static_cast<std::size_t>(m - i - 1));
            for (D& e : x) {
                e = up(detail::draw<T>(rng));
                xx += abs(e) * abs(e);
            }
            D ti = D(0);
            if (xx > 0 || std::imag(std::complex<double>(alpha)) != 0) {
                const double beta = -std::copysign(std::sqrt(abs(alpha) * abs(alpha) + xx), std::real(std::complex<double>(alpha)));
                ti = (D(beta) - alpha) / D(beta);
                const D scale = D(1) / (alpha - D(beta));
                for (D& e : x) e *= scale;
            }
            for (std::size_t r = 0; r < x.size(); ++r) {
                const std::complex<double> e(x[r]);
                detail::at(p, A.ld(), i + 1 + static_cast<int>(r), i) = make<T>(e.real(), e.imag());
            }
            const std::complex<double> tc(ti);
            t[static_cast<long long>(i) * tau.inc()] = make<T>(tc.real(), tc.imag());
        }
    }
}

}  // namespace batchlas::verify

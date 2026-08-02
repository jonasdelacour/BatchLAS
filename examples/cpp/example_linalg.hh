#pragma once

// Small host-side linear algebra used to CHECK the library's results.
//
// Everything here runs on the host in plain C++ and never calls BatchLAS, so a
// passing check means the library agreed with an independent computation.
// Deliberately naive — correctness and readability over speed.
//
// Matrices are BatchLAS matrices: column-major, element (i, j) of batch item b
// at data[b*stride + j*ld + i]. This header is not part of the BatchLAS API.

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <blas/matrix.hh>

namespace examples {

using batchlas::float_t;
using batchlas::MatrixFormat;
using batchlas::MatrixView;
using batchlas::Transpose;
using batchlas::Uplo;

// --- scalar helpers --------------------------------------------------------

template <typename T>
T conjugate(const T& x) {
    if constexpr (batchlas::is_std_complex_v<T>) {
        return std::conj(x);
    } else {
        return x;
    }
}

template <typename T>
double magnitude(const T& x) {
    return static_cast<double>(std::abs(x));
}

template <typename T>
double real_part(const T& x) {
    if constexpr (batchlas::is_std_complex_v<T>) {
        return static_cast<double>(x.real());
    } else {
        return static_cast<double>(x);
    }
}

// --- a plain host matrix ---------------------------------------------------

// Column-major, single matrix. Used as the reference side of every check.
template <typename T>
struct HostMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<T> data;

    HostMatrix() = default;
    HostMatrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, T(0)) {}

    T& operator()(int i, int j) { return data[static_cast<size_t>(j) * rows + i]; }
    const T& operator()(int i, int j) const { return data[static_cast<size_t>(j) * rows + i]; }
};

// Pull batch item `b` of a BatchLAS matrix onto the host.
template <typename T>
HostMatrix<T> to_host(const MatrixView<T, MatrixFormat::Dense>& A, int b = 0) {
    HostMatrix<T> out(A.rows(), A.cols());
    for (int j = 0; j < A.cols(); ++j) {
        for (int i = 0; i < A.rows(); ++i) out(i, j) = A.at(i, j, b);
    }
    return out;
}

template <typename T>
HostMatrix<T> to_host(const batchlas::Matrix<T, MatrixFormat::Dense>& A, int b = 0) {
    return to_host(MatrixView<T, MatrixFormat::Dense>(A), b);
}

// Write a host matrix into batch item `b` of a BatchLAS matrix.
template <typename T>
void from_host(const HostMatrix<T>& src, batchlas::Matrix<T, MatrixFormat::Dense>& dst, int b = 0) {
    for (int j = 0; j < src.cols; ++j) {
        for (int i = 0; i < src.rows; ++i) dst(i, j, b) = src(i, j);
    }
}

// --- host reference kernels ------------------------------------------------

template <typename T>
HostMatrix<T> matmul(const HostMatrix<T>& A, const HostMatrix<T>& B, Transpose ta = Transpose::NoTrans,
                     Transpose tb = Transpose::NoTrans) {
    auto dim = [](const HostMatrix<T>& M, Transpose t) {
        return t == Transpose::NoTrans ? std::pair<int, int>{M.rows, M.cols} : std::pair<int, int>{M.cols, M.rows};
    };
    auto get = [](const HostMatrix<T>& M, int i, int j, Transpose t) -> T {
        if (t == Transpose::NoTrans) return M(i, j);
        if (t == Transpose::Trans) return M(j, i);
        return conjugate(M(j, i));
    };
    const auto [m, k] = dim(A, ta);
    const auto [k2, n] = dim(B, tb);
    (void)k2;
    HostMatrix<T> C(m, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T acc = T(0);
            for (int p = 0; p < k; ++p) acc += get(A, i, p, ta) * get(B, p, j, tb);
            C(i, j) = acc;
        }
    }
    return C;
}

template <typename T>
HostMatrix<T> transposed(const HostMatrix<T>& A, bool conj = false) {
    HostMatrix<T> out(A.cols, A.rows);
    for (int j = 0; j < A.cols; ++j) {
        for (int i = 0; i < A.rows; ++i) out(j, i) = conj ? conjugate(A(i, j)) : A(i, j);
    }
    return out;
}

template <typename T>
HostMatrix<T> identity(int n) {
    HostMatrix<T> I(n, n);
    for (int i = 0; i < n; ++i) I(i, i) = T(1);
    return I;
}

// Largest absolute entry of A - B.
template <typename T>
double max_abs_diff(const HostMatrix<T>& A, const HostMatrix<T>& B) {
    double worst = 0.0;
    for (size_t i = 0; i < A.data.size(); ++i) worst = std::max(worst, magnitude(A.data[i] - B.data[i]));
    return worst;
}

template <typename T>
double max_abs(const HostMatrix<T>& A) {
    double worst = 0.0;
    for (const auto& v : A.data) worst = std::max(worst, magnitude(v));
    return worst;
}

// Same, across every item of a batch.
template <typename T>
double max_abs_diff_batched(const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B) {
    double worst = 0.0;
    for (int b = 0; b < A.batch_size(); ++b) worst = std::max(worst, max_abs_diff(to_host(A, b), to_host(B, b)));
    return worst;
}

// max |Q^H Q - I| — how far the columns of Q are from orthonormal.
template <typename T>
double orthogonality_error(const HostMatrix<T>& Q) {
    auto QhQ = matmul(Q, Q, Transpose::ConjTrans, Transpose::NoTrans);
    return max_abs_diff(QhQ, identity<T>(QhQ.rows));
}

// max |A V - V diag(w)| — the eigenpair residual.
template <typename T, typename R>
double eigen_residual(const HostMatrix<T>& A, const HostMatrix<T>& V, const std::vector<R>& w) {
    auto AV = matmul(A, V);
    double worst = 0.0;
    for (int j = 0; j < V.cols; ++j) {
        for (int i = 0; i < V.rows; ++i) {
            worst = std::max(worst, magnitude(AV(i, j) - V(i, j) * T(w[j])));
        }
    }
    return worst;
}

// --- host eigenvalues ------------------------------------------------------

// Cyclic two-sided Jacobi on the host: eigenvalues of a real symmetric matrix,
// ascending. Slow and simple on purpose — it is the independent reference the
// library's eigensolvers are checked against.
inline std::vector<double> jacobi_eigenvalues(HostMatrix<double> A) {
    const int n = A.rows;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += A(p, q) * A(p, q);
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (std::abs(A(p, q)) < 1e-300) continue;
                const double theta = (A(q, q) - A(p, p)) / (2.0 * A(p, q));
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int i = 0; i < n; ++i) {
                    const double aip = A(i, p), aiq = A(i, q);
                    A(i, p) = c * aip - s * aiq;
                    A(i, q) = s * aip + c * aiq;
                }
                for (int j = 0; j < n; ++j) {
                    const double apj = A(p, j), aqj = A(q, j);
                    A(p, j) = c * apj - s * aqj;
                    A(q, j) = s * apj + c * aqj;
                }
            }
        }
    }
    std::vector<double> vals(n);
    for (int i = 0; i < n; ++i) vals[i] = A(i, i);
    std::sort(vals.begin(), vals.end());
    return vals;
}

// Eigenvalues of the symmetric tridiagonal matrix with diagonal d and
// off-diagonal e, ascending. A reduction to tridiagonal form is correct
// exactly when these match the original matrix's spectrum.
inline std::vector<double> tridiagonal_eigenvalues(const std::vector<double>& d, const std::vector<double>& e) {
    const int n = static_cast<int>(d.size());
    HostMatrix<double> T(n, n);
    for (int i = 0; i < n; ++i) T(i, i) = d[i];
    for (int i = 0; i + 1 < n && i < static_cast<int>(e.size()); ++i) {
        T(i + 1, i) = e[i];
        T(i, i + 1) = e[i];
    }
    return jacobi_eigenvalues(T);
}

// --- band storage ----------------------------------------------------------

// LAPACK band storage, lower convention: AB is (kd+1) x n with
// AB(i - j, j) = A(i, j) for j <= i <= min(n-1, j+kd). Expand it back to a
// full symmetric matrix so it can be checked like any other.
inline HostMatrix<double> band_to_dense(const HostMatrix<double>& AB, int kd, int n) {
    HostMatrix<double> A(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i <= std::min(n - 1, j + kd); ++i) {
            const double v = AB(i - j, j);
            A(i, j) = v;
            A(j, i) = v;
        }
    }
    return A;
}

// --- host generators (deterministic) ---------------------------------------

template <typename T>
T random_scalar(std::mt19937& rng, std::uniform_real_distribution<double>& dist) {
    if constexpr (batchlas::is_std_complex_v<T>) {
        using R = float_t<T>;
        return T(static_cast<R>(dist(rng)), static_cast<R>(dist(rng)));
    } else {
        return static_cast<T>(dist(rng));
    }
}

template <typename T>
HostMatrix<T> random_host(int rows, int cols, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    HostMatrix<T> A(rows, cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) A(i, j) = random_scalar<T>(rng, dist);
    }
    return A;
}

// Symmetric (Hermitian for complex T), full storage.
template <typename T>
HostMatrix<T> random_symmetric_host(int n, unsigned seed = 42) {
    auto A = random_host<T>(n, n, seed);
    HostMatrix<T> S(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) S(i, j) = (A(i, j) + conjugate(A(j, i))) * T(0.5);
    }
    for (int i = 0; i < n; ++i) S(i, i) = T(real_part(S(i, i)));
    return S;
}

// Symmetric positive definite: A^H A + n*I.
template <typename T>
HostMatrix<T> random_spd_host(int n, unsigned seed = 42) {
    auto A = random_host<T>(n, n, seed);
    auto S = matmul(A, A, Transpose::ConjTrans, Transpose::NoTrans);
    for (int i = 0; i < n; ++i) S(i, i) += T(static_cast<float_t<T>>(n));
    return S;
}

// Orthonormal columns, by modified Gram-Schmidt on the host.
template <typename T>
HostMatrix<T> random_orthonormal_host(int m, int k, unsigned seed = 42) {
    auto A = random_host<T>(m, k, seed);
    for (int j = 0; j < k; ++j) {
        for (int p = 0; p < j; ++p) {
            T dot = T(0);
            for (int i = 0; i < m; ++i) dot += conjugate(A(i, p)) * A(i, j);
            for (int i = 0; i < m; ++i) A(i, j) -= dot * A(i, p);
        }
        double nrm = 0.0;
        for (int i = 0; i < m; ++i) nrm += magnitude(A(i, j)) * magnitude(A(i, j));
        nrm = std::sqrt(nrm);
        for (int i = 0; i < m; ++i) A(i, j) /= T(static_cast<float_t<T>>(nrm));
    }
    return A;
}

// A matrix with exactly the requested singular values: U diag(s) V^H.
// Ill-conditioning built this way is spread across the whole matrix, unlike
// plain column scaling, which most algorithms shrug off.
template <typename T>
HostMatrix<T> with_singular_values(int m, const std::vector<double>& s, unsigned seed = 42) {
    const int k = static_cast<int>(s.size());
    auto U = random_orthonormal_host<T>(m, k, seed);
    auto V = random_orthonormal_host<T>(k, k, seed + 1000);
    HostMatrix<T> US(m, k);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i) US(i, j) = U(i, j) * T(static_cast<float_t<T>>(s[j]));
    return matmul(US, V, Transpose::NoTrans, Transpose::ConjTrans);
}

// A symmetric/Hermitian matrix with exactly the requested eigenvalues:
// Q diag(lambda) Q^H. The spectrum is then known exactly, so an eigensolver
// can be checked against it rather than against another eigensolver.
template <typename T, typename R>
HostMatrix<T> symmetric_with_eigenvalues(const std::vector<R>& lambda, unsigned seed = 42) {
    const int n = static_cast<int>(lambda.size());
    auto Q = random_orthonormal_host<T>(n, n, seed);
    HostMatrix<T> QL(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) QL(i, j) = Q(i, j) * T(static_cast<float_t<T>>(lambda[j]));
    auto A = matmul(QL, Q, Transpose::NoTrans, Transpose::ConjTrans);
    // Symmetrise to kill the last rounding asymmetry.
    HostMatrix<T> S(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) S(i, j) = (A(i, j) + conjugate(A(j, i))) * T(0.5);
    return S;
}

// Zero out the triangle a routine is told not to read, so a check really
// proves the routine honoured `uplo`.
template <typename T>
HostMatrix<T> keep_triangle(const HostMatrix<T>& A, Uplo uplo) {
    HostMatrix<T> out(A.rows, A.cols);
    for (int j = 0; j < A.cols; ++j) {
        for (int i = 0; i < A.rows; ++i) {
            const bool keep = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
            if (keep) out(i, j) = A(i, j);
        }
    }
    return out;
}

// Fill every item of a batch with the same host matrix.
template <typename T>
batchlas::Matrix<T, MatrixFormat::Dense> broadcast(const HostMatrix<T>& src, int batch_size) {
    batchlas::Matrix<T, MatrixFormat::Dense> M(src.rows, src.cols, batch_size);
    for (int b = 0; b < batch_size; ++b) from_host(src, M, b);
    return M;
}

// Vary each batch item a little, so a batched check is not the same problem
// solved N times.
template <typename T>
batchlas::Matrix<T, MatrixFormat::Dense> batch_of(int rows, int cols, int batch_size,
                                                  HostMatrix<T> (*make)(int, int, unsigned), unsigned seed = 42) {
    batchlas::Matrix<T, MatrixFormat::Dense> M(rows, cols, batch_size);
    for (int b = 0; b < batch_size; ++b) from_host(make(rows, cols, seed + b), M, b);
    return M;
}

// Sorted ascending, for comparing spectra computed different ways.
template <typename R>
std::vector<R> sorted(std::vector<R> v) {
    std::sort(v.begin(), v.end());
    return v;
}

template <typename R>
double max_abs_diff(const std::vector<R>& a, const std::vector<R>& b) {
    double worst = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) worst = std::max(worst, std::abs(static_cast<double>(a[i] - b[i])));
    return worst;
}

// Largest |a_i - b_i| / max(|b_i|, floor) — what notebook 10 is about.
template <typename R>
double max_rel_diff(const std::vector<R>& a, const std::vector<R>& b, double floor = 0.0) {
    double worst = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double denom = std::max(std::abs(static_cast<double>(b[i])), floor);
        if (denom > 0.0) worst = std::max(worst, std::abs(static_cast<double>(a[i] - b[i])) / denom);
    }
    return worst;
}

}  // namespace examples

// SPDX-License-Identifier: MIT
// The checks libbatchlas_verify runs as SYCL kernels (docs/design/verification.md#verification-device-engine),
// on type-erased operands. Internal to residuals.hh.
#pragma once

#include <batchlas/blas/enums.hh>

#include <complex>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace batchlas::verify {

/// How an operand is read: the stored triangle, an implicit unit diagonal, or a mirrored triangle.
enum class Shape { general, lower, upper, unit_lower, unit_upper, hermitian_lower, hermitian_upper, symmetric_lower, symmetric_upper };

namespace detail {

enum class Dtype : std::uint8_t { f32, f64, c32, c64, i32 };  // i32: pivots, read on the host only

template <class T> constexpr Dtype dtype_of() {
    if constexpr (std::is_same_v<T, float>) return Dtype::f32;
    else if constexpr (std::is_same_v<T, double>) return Dtype::f64;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return Dtype::c32;
    else {
        static_assert(std::is_same_v<T, std::complex<double>>, "batchlas::verify: float, double, complex<float> or complex<double> only");
        return Dtype::c64;
    }
}

// Element (r, c) of item b at data + b * stride + c * ld + r * inc (a vector: cols = 1, inc = its inc).
struct Operand {
    const void* data = nullptr;
    Dtype dtype = Dtype::f64;
    int rows = 0, cols = 0, ld = 1, inc = 1;
    long long stride = 0;
    int batch = 0;
};

// Every entry point checks the items given (none empty), runs on the engine's queue and waits.
// `complex` selects complex<double> arithmetic; real operands are then read with a zero imaginary part.
double potrf(const Operand& A0, const Operand& F, bool upper, bool complex, std::span<const int> items);
double similarity(const Operand& A0, const Operand& Q, const Operand& B, bool complex, std::span<const int> items);
double getrf(const Operand& A0, const Operand& F, const Operand& piv, bool complex, std::span<const int> items);
double qr(const Operand& A0, const Operand& F, const Operand& tau, bool complex, std::span<const int> items);
std::vector<double> form_q(const Operand& F, const Operand& tau, bool complex, int item, int cols);
double qr_reconstruction(const Operand& A0, const Operand& Q, const Operand& R, bool complex, std::span<const int> items);

// ||op(A) X V - B0 V||_F / (||op(A)||_F ||X V||_F): A read through (shape, trans), or formed as
// P L U from getrf factors when `lu` (shape ignored). An empty B0 (rows 0) is the identity.
double solve(const Operand& A, Shape shape, Transpose trans, const Operand* piv, const Operand& X, const Operand& B0,
             bool complex, std::span<const int> items, double full_limit);

// gemm_backward_error over the elements `sc` selects; `denominator_only` gives the largest denominator.
double componentwise(const Operand& A, Shape sa, Transpose ta, const Operand& B, Shape sb, Transpose tb, const Operand* C0,
                     const Operand* C, Shape sc, std::complex<double> alpha, std::complex<double> beta, bool complex,
                     std::span<const int> items, bool denominator_only);
double rank2k(const Operand& A, const Operand& B, Transpose trans, const Operand* C0, const Operand& C, Uplo uplo,
              std::complex<double> alpha, std::complex<double> beta, bool hermitian, bool complex, std::span<const int> items);
double trsm(const Operand& A, Shape s, Transpose ta, bool left, const Operand& X, const Operand& B0, std::complex<double> alpha,
            bool complex, std::span<const int> items);
double orthogonality(const Operand& Q, bool complex, std::span<const int> items);
// w is real: one eigenvalue per column of V, read as double.
double eigen(const Operand& A, const Operand& V, const Operand& w, bool complex, std::span<const int> items);

}  // namespace detail
}  // namespace batchlas::verify

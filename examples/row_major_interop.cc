// Row-major interop: a consumer that holds row-major batches (a LinAlg::MatView-style
// pointer with m, n, lda and an item stride) runs syev and Trans-flag gemm on them
// without copying. docs/cpp-api.md ("Row-major data") states the rule shown here.
//
// A row-major m x n buffer with row stride lda is the column-major n x m matrix with
// ld = lda over the same bytes, i.e. its transpose. So:
//   * symmetric syev: the row-major UPPER triangle is the column-major LOWER triangle,
//     so Uplo::Lower reads it. The row-major lower triangle is poisoned and must not be
//     read. The eigenvectors land in the same bytes as the columns of the column-major
//     V, i.e. the ROWS of the row-major buffer.
//   * general gemm: a row-major operand enters as the transposed column-major view, and
//     gemm's Trans flags select the product the row-major caller meant.
// Every buffer has a padded ld and item stride, filled with a large finite poison outside
// the matrix. Results are checked with batchlas::verify and a host oracle in row-major
// indexing that shares no view code.

#include <batchlas.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

using batchlas::Matrix;
using batchlas::MatrixFormat;
using batchlas::MatrixView;
using batchlas::Transpose;
using batchlas::UnifiedVector;
using batchlas::Uplo;
using batchlas::VectorView;
using batchlas::verify::Check;

using T = double;
using MV = MatrixView<T, MatrixFormat::Dense>;

constexpr int kN = 6;              // syev order
constexpr int kM = 5;              // gemm: C (m x n) = alpha A (m x k) B^T + beta C0, B is n x k
constexpr int kNc = 3;
constexpr int kK = 4;
constexpr int kBatch = 4;
constexpr int kPad = 3;            // ld = logical columns + kPad
constexpr double kPoison = 1.0e7;  // large, finite and wrong: any read of unused storage shows
constexpr T kAlpha1 = 1.25, kBeta1 = -0.5;  // C = A B^T with transA = Trans
constexpr T kAlpha2 = 0.5, kBeta2 = 0.75;   // D = A^T A with transB = Trans

// What the consumer holds: element (b, i, j) at data[b * stride + i * ld + j].
struct RowMajorBatch {
    T* data = nullptr;
    int rows = 0, cols = 0, ld = 0, stride = 0, batch = 0;
    T& at(int b, int i, int j) const {
        return data[std::size_t(b) * std::size_t(stride) + std::size_t(i) * std::size_t(ld) + std::size_t(j)];
    }
    std::size_t count() const {
        return std::size_t(batch - 1) * std::size_t(stride) + std::size_t(rows) * std::size_t(ld);
    }
};

RowMajorBatch layout(int rows, int cols, int batch) {
    return {nullptr, rows, cols, cols + kPad, rows * (cols + kPad) + 2, batch};
}

// Owns a consumer buffer, poisoned outside the matrix entries the test writes.
struct RowMajorStore {
    RowMajorBatch r;
    UnifiedVector<T> mem;
    RowMajorStore(int rows, int cols, int batch) : r(layout(rows, cols, batch)), mem(r.count()) {
        r.data = mem.data();
        std::fill_n(r.data, r.count(), kPoison);
    }
};

// The same bytes as a column-major BatchLAS view of the TRANSPOSE: cols x rows, ld unchanged.
MV transposed_view(const RowMajorBatch& r) {
    return MV(r.data, r.cols, r.rows, r.ld, r.stride, r.batch);
}

double sym_val(int i, int j, int b) {
    const int lo = std::min(i, j), hi = std::max(i, j);
    return std::sin(1.0 + 0.7 * lo + 1.3 * hi + 0.5 * b);
}
double a_val(int i, int p, int b) { return std::sin(0.3 + 0.9 * i + 0.4 * p + 0.2 * b); }
double b_val(int j, int p, int b) { return std::cos(1.1 + 0.5 * j - 0.8 * p + 0.3 * b); }
double c_val(int i, int j, int b) { return std::cos(0.7 + 0.4 * i + 0.6 * j + 0.25 * b); }
double d_val(int p, int q, int b) { return std::sin(0.2 + 0.3 * p + 0.8 * q + 0.4 * b); }

bool report(const char* what, Check kind, int n, double value) {
    const bool ok = batchlas::verify::pass<T>(kind, n, value);
    std::printf("%-22s value = %.3e, bound = %.3e -> %s\n", what, value, batchlas::verify::bound<T>(kind, n),
                ok ? "PASS" : "FAIL");
    return ok;
}

bool report_host(const char* what, double value, double tol) {
    const bool ok = value < tol;
    std::printf("%-22s max = %.3e, tol = %.3e -> %s\n", what, value, tol, ok ? "PASS" : "FAIL");
    return ok;
}

bool check_syev(batchlas::Queue& ctx) {
    RowMajorStore A(kN, kN, kBatch);
    std::vector<T> full(std::size_t(kN) * kN * kBatch);  // full symmetric reference, row-major per item
    Matrix<T, MatrixFormat::Dense> reference(kN, kN, kBatch);  // column-major copy for eigen_residual
    auto R = reference.view();
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < kN; ++i)
            for (int j = 0; j < kN; ++j) {
                full[(std::size_t(b) * kN + i) * kN + j] = sym_val(i, j, b);
                R(i, j, b) = sym_val(i, j, b);
                if (i <= j) A.r.at(b, i, j) = sym_val(i, j, b);  // upper triangle only
            }

    // Square, so this view is both the transpose and the column-major V after syev.
    const MV V = transposed_view(A.r);
    const batchlas::SyevOptions opts{.jobz = batchlas::JobType::EigenVectors, .uplo = Uplo::Lower};
    UnifiedVector<T> W(std::size_t(kN) * kBatch);
    UnifiedVector<std::byte> work(batchlas::syev_buffer_size(ctx, V, W.to_span(), opts.jobz, opts.uplo));
    (void)batchlas::syev(ctx, V, W.to_span(), opts, work.to_span());
    ctx.wait();

    // Row j of item b (A.r.at(b, j, .)) is eigenvector j: |A x - w x| on the row-major buffer.
    double resid = 0.0, orth = 0.0;
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < kN; ++j) {
            const double w = W.data()[std::size_t(b) * kN + j];
            for (int i = 0; i < kN; ++i) {
                double ax = 0.0;
                for (int p = 0; p < kN; ++p) ax += full[(std::size_t(b) * kN + i) * kN + p] * A.r.at(b, j, p);
                resid = std::max(resid, std::fabs(ax - w * A.r.at(b, j, i)));
            }
            for (int k = 0; k < kN; ++k) {
                double dot = 0.0;
                for (int p = 0; p < kN; ++p) dot += A.r.at(b, j, p) * A.r.at(b, k, p);
                orth = std::max(orth, std::fabs(dot - (j == k ? 1.0 : 0.0)));
            }
        }

    const VectorView<T> w(W.data(), kN, kBatch);
    bool ok = report_host("syev row residual", resid, 1e-10);
    ok = report_host("syev row orthonormal", orth, 1e-10) && ok;
    ok = report("syev eigen_residual", Check::eigen_residual, kN, MV::eigen_residual(ctx, R, V, w, R.all_items())) && ok;
    ok = report("syev orthogonality", Check::orthogonality_rotations, kN, MV::orthogonality(ctx, V, R.all_items())) && ok;
    return ok;
}

bool check_gemm(batchlas::Queue& ctx) {
    RowMajorStore A(kM, kK, kBatch), B(kNc, kK, kBatch), C(kM, kNc, kBatch), C0(kM, kNc, kBatch);
    RowMajorStore D(kK, kK, kBatch), D0(kK, kK, kBatch);
    for (int b = 0; b < kBatch; ++b) {
        for (int i = 0; i < kM; ++i)
            for (int p = 0; p < kK; ++p) A.r.at(b, i, p) = a_val(i, p, b);
        for (int j = 0; j < kNc; ++j)
            for (int p = 0; p < kK; ++p) B.r.at(b, j, p) = b_val(j, p, b);
        for (int i = 0; i < kM; ++i)
            for (int j = 0; j < kNc; ++j) C.r.at(b, i, j) = C0.r.at(b, i, j) = c_val(i, j, b);
        for (int p = 0; p < kK; ++p)
            for (int q = 0; q < kK; ++q) D.r.at(b, p, q) = D0.r.at(b, p, q) = d_val(p, q, b);
    }

    // C = alpha A B^T + beta C0 in row-major, computed as C^T = alpha B A^T + beta C0^T in
    // column-major: op(Bv) = B (n x k), Av = A^T (k x m), Cv = C^T (n x m).
    const MV Av = transposed_view(A.r), Bv = transposed_view(B.r);
    const MV Cv = transposed_view(C.r), C0v = transposed_view(C0.r);
    (void)batchlas::gemm(ctx, Bv, Av, Cv, {.alpha = kAlpha1, .beta = kBeta1, .transA = Transpose::Trans});
    ctx.wait();

    const double err = MV::gemm_backward_error(ctx, Bv, Transpose::Trans, Av, Transpose::NoTrans, C0v, Cv,
                                               MV::up(kAlpha1), MV::up(kBeta1), Cv.all_items());
    bool ok = report("gemm backward error", Check::blas, kK, err);

    double worst = 0.0;
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < kM; ++i)
            for (int j = 0; j < kNc; ++j) {
                double ref = 0.0;
                for (int p = 0; p < kK; ++p) ref += a_val(i, p, b) * b_val(j, p, b);
                const double want = kAlpha1 * ref + kBeta1 * c_val(i, j, b);
                worst = std::max(worst, std::fabs(C.r.at(b, i, j) - want));
            }
    ok = report_host("gemm host oracle", worst, 1e-12 * kK) && ok;

    // D = alpha A^T A + beta D0 with transB = Trans on the same operand: Av Av^T.
    const MV Dv = transposed_view(D.r), D0v = transposed_view(D0.r);
    (void)batchlas::gemm(ctx, Av, Av, Dv, {.alpha = kAlpha2, .beta = kBeta2, .transB = Transpose::Trans});
    ctx.wait();

    const double err2 = MV::gemm_backward_error(ctx, Av, Transpose::NoTrans, Av, Transpose::Trans, D0v, Dv,
                                                MV::up(kAlpha2), MV::up(kBeta2), Dv.all_items());
    ok = report("gemm^T backward error", Check::blas, kM, err2) && ok;

    double worst2 = 0.0;
    for (int b = 0; b < kBatch; ++b)
        for (int p = 0; p < kK; ++p)
            for (int q = 0; q < kK; ++q) {
                double ref = 0.0;
                for (int i = 0; i < kM; ++i) ref += a_val(i, p, b) * a_val(i, q, b);
                const double want = kAlpha2 * ref + kBeta2 * d_val(p, q, b);
                worst2 = std::max(worst2, std::fabs(D.r.at(b, p, q) - want));
            }
    return report_host("gemm^T host oracle", worst2, 1e-12 * kM) && ok;
}

}  // namespace

int main() try {
    batchlas::Queue ctx(batchlas::Device::default_device());
    std::printf("backend = %d\n", static_cast<int>(ctx.backend()));
    const bool s = check_syev(ctx);
    const bool g = check_gemm(ctx);
    const bool ok = s && g;
    std::printf("row-major interop -> %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "row-major interop example failed: %s\n", e.what());
    return 1;
}

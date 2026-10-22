// Row-major interop: a consumer that holds row-major batches (a LinAlg::MatView-style
// span with m, n, lda) runs syev and a Trans-flag gemm on them without copying.
// docs/cpp-api.md ("Row-major data") states the rule this example demonstrates.
//
// A row-major m x n buffer with row stride lda is, byte for byte, the column-major
// n x m matrix with ld = lda: the transpose. So
//   * symmetric input: A^T == A, so syev on the transposed view solves the same
//     matrix. Its eigenvectors are the columns of the column-major V, and read
//     row-major from the same bytes they are the ROWS of the buffer.
//   * general input: a row-major operand enters as the transposed column-major view,
//     and gemm's Trans flags select the product the row-major caller meant.
//
// Results are checked with batchlas::verify (MatrixView statics and tolerance kinds),
// plus a host oracle written in row-major indexing that shares no view code.

#include <batchlas.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>

namespace {

using batchlas::Matrix;
using batchlas::MatrixFormat;
using batchlas::MatrixView;
using batchlas::Transpose;
using batchlas::VectorView;
using batchlas::verify::Check;

using T = double;
using MV = MatrixView<T, MatrixFormat::Dense>;

constexpr int kN = 6;        // syev order
constexpr int kM = 5;        // gemm: C (m x n) = A (m x k) * B^T, B is n x k
constexpr int kNc = 3;
constexpr int kK = 4;
constexpr int kBatch = 4;

// What the consumer holds: element (b, i, j) at data[b * stride + i * ld + j].
struct RowMajorBatch {
    T* data;
    int rows, cols, ld, stride, batch;
    T& at(int b, int i, int j) const {
        return data[std::size_t(b) * std::size_t(stride) + std::size_t(i) * std::size_t(ld) + std::size_t(j)];
    }
};

// Packed row-major storage in USM, owned by `store` (its element count is batch * rows * cols).
RowMajorBatch make_row_major(Matrix<T, MatrixFormat::Dense>& store, int rows, int cols, int batch) {
    return {store.view().data_ptr(), rows, cols, cols, rows * cols, batch};
}

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

bool report(const char* what, Check kind, int n, double value) {
    const bool ok = batchlas::verify::pass<T>(kind, n, value);
    std::printf("%-22s value = %.3e, bound = %.3e -> %s\n", what, value, batchlas::verify::bound<T>(kind, n),
                ok ? "PASS" : "FAIL");
    return ok;
}

bool check_syev(batchlas::Queue& ctx) {
    Matrix<T, MatrixFormat::Dense> store(kN, kN, kBatch);
    const RowMajorBatch A = make_row_major(store, kN, kN, kBatch);
    Matrix<T, MatrixFormat::Dense> reference(kN, kN, kBatch);  // column-major copy, kept for the residual
    auto R = reference.view();
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < kN; ++i)
            for (int j = 0; j < kN; ++j) {
                A.at(b, i, j) = sym_val(i, j, b);
                R(i, j, b) = sym_val(i, j, b);
            }

    // Square, so this view is both the transpose and the column-major V after syev.
    const MV V = transposed_view(A);
    batchlas::UnifiedVector<T> W(std::size_t(kN) * kBatch);
    batchlas::UnifiedVector<std::byte> work(
        batchlas::syev_buffer_size(ctx, V, W.to_span(), batchlas::JobType::EigenVectors, batchlas::Uplo::Lower));
    (void)batchlas::syev(ctx, V, W.to_span(), batchlas::SyevOptions{.jobz = batchlas::JobType::EigenVectors}, work.to_span());
    ctx.wait();

    // Eigenvectors now sit in A's bytes: row j of the row-major buffer is eigenvector j.
    const VectorView<T> w(W.data(), kN, kBatch);
    const bool a = report("syev eigen_residual", Check::eigen_residual, kN, MV::eigen_residual(ctx, R, V, w, R.all_items()));
    const bool b = report("syev orthogonality", Check::orthogonality_rotations, kN, MV::orthogonality(ctx, V, R.all_items()));
    return a && b;
}

bool check_gemm(batchlas::Queue& ctx) {
    Matrix<T, MatrixFormat::Dense> aStore(kM, kK, kBatch), bStore(kNc, kK, kBatch), cStore(kM, kNc, kBatch),
        c0Store(kM, kNc, kBatch);
    const RowMajorBatch A = make_row_major(aStore, kM, kK, kBatch);
    const RowMajorBatch B = make_row_major(bStore, kNc, kK, kBatch);
    const RowMajorBatch C = make_row_major(cStore, kM, kNc, kBatch);
    const RowMajorBatch C0 = make_row_major(c0Store, kM, kNc, kBatch);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < kM; ++i)
            for (int p = 0; p < kK; ++p) A.at(b, i, p) = a_val(i, p, b);
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < kNc; ++j)
            for (int p = 0; p < kK; ++p) B.at(b, j, p) = b_val(j, p, b);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < kM; ++i)
            for (int j = 0; j < kNc; ++j) C.at(b, i, j) = C0.at(b, i, j) = 0.0;  // beta = 0: C is not read

    // C = A * B^T in row-major, computed as C^T = B * A^T in column-major:
    // op(Bv) with Bv = B^T (k x n) gives B (n x k), and Av = A^T (k x m).
    const MV Av = transposed_view(A);
    const MV Bv = transposed_view(B);
    const MV Cv = transposed_view(C);
    const MV C0v = transposed_view(C0);
    (void)batchlas::gemm(ctx, Bv, Av, Cv, {.alpha = T(1), .beta = T(0), .transA = Transpose::Trans});
    ctx.wait();

    const double err = MV::gemm_backward_error(ctx, Bv, Transpose::Trans, Av, Transpose::NoTrans, C0v, Cv, MV::up(T(1)),
                                               MV::up(T(0)), Cv.all_items());
    bool ok = report("gemm backward error", Check::blas, kK, err);

    double worst = 0.0;
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < kM; ++i)
            for (int j = 0; j < kNc; ++j) {
                double ref = 0.0;
                for (int p = 0; p < kK; ++p) ref += a_val(i, p, b) * b_val(j, p, b);
                worst = std::max(worst, std::fabs(C.at(b, i, j) - ref));
            }
    const bool oracle = worst < 1e-12 * kK;
    std::printf("%-22s max |C - ref| = %.3e -> %s\n", "gemm host oracle", worst, oracle ? "PASS" : "FAIL");
    return ok && oracle;
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

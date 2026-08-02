// 2. Batched dense BLAS
//
// The BLAS-2 and BLAS-3 surface: gemm, gemv, symm, syrk, syr2k, trmm, trsm,
// plus heterogeneous batches and mixed-precision compute.
//
// Every result is checked against a host reference in example_linalg.hh.

#include <complex>
#include <vector>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_common.hh"
#include "example_linalg.hh"
#include "example_runner.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 4;

template <Backend B>
struct Example {
    // `Matrix<T>` defaults to MatrixFormat::Dense, so this is the same type
    // 01 spelled out in full.
    using Mat = Matrix<double>;

    static void run(Queue& ctx) {
        gemm_section(ctx);
        gemv_section(ctx);
        symm_section(ctx);
        rank_k_section(ctx);
        triangular_section(ctx);
        heterogeneous_section(ctx);
        mixed_precision_section(ctx);
    }

    // -----------------------------------------------------------------------
    // gemm — general matrix product, C <- alpha*op(A)*op(B) + beta*C.
    //
    // The workhorse. op() is NoTrans/Trans/ConjTrans and is folded into the
    // kernel. Shapes are checked against the op'd operands, so with
    // transA=Trans an m-by-k A means op(A) is k-by-m.
    // -----------------------------------------------------------------------
    static void gemm_section(Queue& ctx) {
        section("gemm - general matrix product");

        const int m = 5, k = 4, n = 3;
        auto A = batch_of<double>(m, k, kBatch, random_host<double>, 1);
        auto Bm = batch_of<double>(k, n, kBatch, random_host<double>, 100);
        auto C = Mat::Zeros(m, n, kBatch);

        gemm<B>(ctx, A, Bm, C, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            worst = std::max(worst, max_abs_diff(to_host(C, b), matmul(to_host(A, b), to_host(Bm, b))));
        }
        report_error("gemm error", worst, 1e-12);

        // Transposed operands, without materialising a transpose.
        auto At = batch_of<double>(k, m, kBatch, random_host<double>, 7);
        auto Ct = Mat::Zeros(m, n, kBatch);
        gemm<B>(ctx, At, Bm, Ct, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
        ctx.wait();

        worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            worst = std::max(worst, max_abs_diff(to_host(Ct, b), matmul(to_host(At, b), to_host(Bm, b),
                                                                       Transpose::Trans, Transpose::NoTrans)));
        }
        report_error("gemm transA error", worst, 1e-12);

        // Complex input takes ConjTrans as well.
        auto Ac = batch_of<std::complex<double>>(m, k, 2, random_host<std::complex<double>>, 3);
        auto Bc = batch_of<std::complex<double>>(m, n, 2, random_host<std::complex<double>>, 9);
        auto Cc = Matrix<std::complex<double>>::Zeros(k, n, 2);
        gemm<B>(ctx, Ac, Bc, Cc, std::complex<double>(1.0), std::complex<double>(0.0), Transpose::ConjTrans,
                Transpose::NoTrans);
        ctx.wait();

        worst = 0.0;
        for (int b = 0; b < 2; ++b) {
            worst = std::max(worst, max_abs_diff(to_host(Cc, b), matmul(to_host(Ac, b), to_host(Bc, b),
                                                                       Transpose::ConjTrans, Transpose::NoTrans)));
        }
        report_error("gemm conjTransA error", worst, 1e-12);
    }

    // -----------------------------------------------------------------------
    // gemv — matrix-vector product, y <- alpha*op(A)*x + beta*y.
    //
    // Vectors are `Vector<T>` / `VectorView<T>`: a pointer plus a length, an
    // element stride `inc`, and a batch stride. A column of a matrix is a
    // vector view with inc=1; a row is one with inc=ld — either can be passed
    // straight to gemv without a copy.
    // -----------------------------------------------------------------------
    static void gemv_section(Queue& ctx) {
        section("gemv - matrix-vector product");

        const int m = 6, n = 4;
        auto A = batch_of<double>(m, n, kBatch, random_host<double>, 11);
        Vector<double> x(n, kBatch);
        Vector<double> y(m, kBatch);
        for (int b = 0; b < kBatch; ++b) {
            for (int i = 0; i < n; ++i) x(i, b) = 0.5 * (i + 1) + b;
            for (int i = 0; i < m; ++i) y(i, b) = 0.0;
        }

        gemv<B>(ctx, A.view(), x, y, 1.0, 0.0, Transpose::NoTrans);
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto Ah = to_host(A, b);
            for (int i = 0; i < m; ++i) {
                double acc = 0.0;
                for (int j = 0; j < n; ++j) acc += Ah(i, j) * x(j, b);
                worst = std::max(worst, std::abs(y(i, b) - acc));
            }
        }
        report_error("gemv error", worst, 1e-12);

        // A column of a matrix, used directly as a vector.
        auto Acol = A.view()(Slice{}, 0);
        report_check("column view has inc=1", Acol.inc() == 1 && Acol.size() == m);
    }

    // -----------------------------------------------------------------------
    // symm — symmetric matrix product, C <- alpha*A*Bm + beta*C with A
    // symmetric. Only the `uplo` triangle of A is read, so the other triangle
    // can hold anything. Real scalars only (the signature is constrained to
    // RealScalar).
    // -----------------------------------------------------------------------
    static void symm_section(Queue& ctx) {
        section("symm - symmetric matrix product");

        const int n = 5, nrhs = 3;
        auto Sh = random_symmetric_host<double>(n, 21);

        // Fill only the lower triangle on the device; symm must still behave
        // as if the full symmetric matrix were there.
        auto A = broadcast(keep_triangle(Sh, Uplo::Lower), kBatch);
        auto Bm = batch_of<double>(n, nrhs, kBatch, random_host<double>, 31);
        auto C = Mat::Zeros(n, nrhs, kBatch);

        symm<B>(ctx, A, Bm, C, 1.0, 0.0, Side::Left, Uplo::Lower);
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) worst = std::max(worst, max_abs_diff(to_host(C, b), matmul(Sh, to_host(Bm, b))));
        report_error("symm(lower, half-filled) error", worst, 1e-12);

        // Side::Right computes C <- alpha*Bm*A + beta*C.
        auto Br = batch_of<double>(nrhs, n, kBatch, random_host<double>, 41);
        auto Cr = Mat::Zeros(nrhs, n, kBatch);
        symm<B>(ctx, A, Br, Cr, 1.0, 0.0, Side::Right, Uplo::Lower);
        ctx.wait();

        worst = 0.0;
        for (int b = 0; b < kBatch; ++b) worst = std::max(worst, max_abs_diff(to_host(Cr, b), matmul(to_host(Br, b), Sh)));
        report_error("symm(right) error", worst, 1e-12);
    }

    // -----------------------------------------------------------------------
    // syrk  — C <- alpha*A*A^T + beta*C   (symmetric rank-k update)
    // syr2k — C <- alpha*(A*B^T + B*A^T) + beta*C
    //
    // Both write only the `uplo` triangle of C; the other triangle is left
    // untouched, so the checks below compare against a reference restricted to
    // the same triangle.
    // -----------------------------------------------------------------------
    static void rank_k_section(Queue& ctx) {
        section("syrk / syr2k - symmetric rank-k updates");

        const int n = 5, k = 3;
        auto A = batch_of<double>(n, k, kBatch, random_host<double>, 51);
        auto C = Mat::Zeros(n, n, kBatch);

        syrk<B>(ctx, A, C, 1.0, 0.0, Uplo::Lower, Transpose::NoTrans);
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto want = keep_triangle(matmul(to_host(A, b), to_host(A, b), Transpose::NoTrans, Transpose::Trans),
                                      Uplo::Lower);
            worst = std::max(worst, max_abs_diff(keep_triangle(to_host(C, b), Uplo::Lower), want));
        }
        report_error("syrk error (lower triangle)", worst, 1e-12);

        auto Bm = batch_of<double>(n, k, kBatch, random_host<double>, 61);
        auto C2 = Mat::Zeros(n, n, kBatch);
        syr2k<B>(ctx, A, Bm, C2, 1.0, 0.0, Uplo::Lower, Transpose::NoTrans);
        ctx.wait();

        worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto Ah = to_host(A, b);
            auto Bh = to_host(Bm, b);
            auto abt = matmul(Ah, Bh, Transpose::NoTrans, Transpose::Trans);
            auto bat = matmul(Bh, Ah, Transpose::NoTrans, Transpose::Trans);
            HostMatrix<double> want(n, n);
            for (int j = 0; j < n; ++j) {
                for (int i = j; i < n; ++i) want(i, j) = abt(i, j) + bat(i, j);
            }
            worst = std::max(worst, max_abs_diff(keep_triangle(to_host(C2, b), Uplo::Lower), want));
        }
        report_error("syr2k error (lower triangle)", worst, 1e-12);
    }

    // -----------------------------------------------------------------------
    // trmm — C <- alpha*op(A)*Bm with A triangular
    // trsm — solves op(A)*X = alpha*Bm in place, overwriting Bm with X
    //
    // `diag` says whether the diagonal is stored (NonUnit) or implicitly 1
    // (Unit). trsm has no separate output: the right-hand side is overwritten.
    // -----------------------------------------------------------------------
    static void triangular_section(Queue& ctx) {
        section("trmm / trsm - triangular multiply and solve");

        const int n = 5, nrhs = 3;

        // A well-conditioned lower-triangular matrix.
        HostMatrix<double> Lh(n, n);
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) Lh(i, j) = (i == j) ? 2.0 + 0.1 * i : 0.3 / (1 + i - j);
        }

        auto L = broadcast(Lh, kBatch);
        auto X = batch_of<double>(n, nrhs, kBatch, random_host<double>, 71);
        auto C = Mat::Zeros(n, nrhs, kBatch);

        trmm<B>(ctx, L, X, C, 1.0, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) worst = std::max(worst, max_abs_diff(to_host(C, b), matmul(Lh, to_host(X, b))));
        report_error("trmm error", worst, 1e-12);

        // Solve L*Y = C. Since C = L*X, Y must come back equal to X.
        // trsm overwrites its right-hand side, so copy first.
        auto Y = C.clone();
        trsm<B>(ctx, L, Y, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0);
        ctx.wait();

        report_error("trsm recovers the solution", max_abs_diff_batched(Y.view(), X.view()), 1e-10);
        report_check("trsm wrote in place", Y.data().data() != C.data().data());

        // Diag::Unit ignores whatever is stored on the diagonal.
        auto Lu = broadcast(Lh, 1);
        auto Rhs = batch_of<double>(n, nrhs, 1, random_host<double>, 81);
        auto Sol = Rhs.clone();
        trsm<B>(ctx, Lu, Sol, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit, 1.0);
        ctx.wait();

        HostMatrix<double> Lunit = Lh;
        for (int i = 0; i < n; ++i) Lunit(i, i) = 1.0;
        report_error("trsm(unit diag) residual", max_abs_diff(matmul(Lunit, to_host(Sol, 0)), to_host(Rhs, 0)), 1e-10);
    }

    // -----------------------------------------------------------------------
    // Heterogeneous batches
    //
    // The Python facade takes a list of differently shaped arrays. In C++ you
    // allocate one batch at the maximum shape and declare the *active*
    // dimensions per item with set_active_dims. Items keep their own row and
    // column counts; a count of 0 is legal and that item is skipped.
    //
    // A.rows() is then the capacity and A.rows(b) the active count for item b.
    // -----------------------------------------------------------------------
    static void heterogeneous_section(Queue& ctx) {
        section("Heterogeneous batches");

        const int batch = 3;
        const int max_m = 4, max_k = 3, max_n = 5;

        auto A = Mat::Zeros(max_m, max_k, batch);
        auto Bm = Mat::Zeros(max_k, max_n, batch);
        auto C = Mat::Zeros(max_m, max_n, batch);

        UnifiedVector<int> a_rows(batch), a_cols(batch);
        UnifiedVector<int> b_rows(batch), b_cols(batch);
        UnifiedVector<int> c_rows(batch), c_cols(batch);

        // item 0: (4x3)(3x5) -> 4x5   item 1: (2x3)(3x2) -> 2x2   item 2: empty
        const int ms[batch] = {4, 2, 0};
        const int ks[batch] = {3, 3, 3};
        const int ns[batch] = {5, 2, 4};
        for (int b = 0; b < batch; ++b) {
            a_rows[b] = ms[b]; a_cols[b] = ks[b];
            b_rows[b] = ks[b]; b_cols[b] = ns[b];
            c_rows[b] = ms[b]; c_cols[b] = ns[b];
        }

        A.set_active_dims(a_rows.to_span(), a_cols.to_span());
        Bm.set_active_dims(b_rows.to_span(), b_cols.to_span());
        C.set_active_dims(c_rows.to_span(), c_cols.to_span());

        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < A.cols(b); ++j)
                for (int i = 0; i < A.rows(b); ++i) A(i, j, b) = 1.0 + i + 2.0 * j + 10.0 * b;
            for (int j = 0; j < Bm.cols(b); ++j)
                for (int i = 0; i < Bm.rows(b); ++i) Bm(i, j, b) = 1.0 + i + j + 7.0 * b;
        }

        report_check("is_heterogeneous", A.is_heterogeneous());
        report_check("capacity vs active", A.rows() == max_m && A.rows(1) == 2);

        gemm_heterogeneous<B>(ctx, A, Bm, C, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < batch; ++b) {
            if (C.rows(b) == 0 || C.cols(b) == 0) continue;
            HostMatrix<double> Ah(A.rows(b), A.cols(b)), Bh(Bm.rows(b), Bm.cols(b));
            for (int j = 0; j < A.cols(b); ++j)
                for (int i = 0; i < A.rows(b); ++i) Ah(i, j) = A(i, j, b);
            for (int j = 0; j < Bm.cols(b); ++j)
                for (int i = 0; i < Bm.rows(b); ++i) Bh(i, j) = Bm(i, j, b);
            auto want = matmul(Ah, Bh);
            for (int j = 0; j < C.cols(b); ++j)
                for (int i = 0; i < C.rows(b); ++i) worst = std::max(worst, std::abs(C(i, j, b) - want(i, j)));
        }
        report_error("heterogeneous gemm error", worst, 1e-12);
    }

    // -----------------------------------------------------------------------
    // Mixed precision
    //
    // ComputePrecision asks the backend to run the multiply at a lower
    // internal precision while keeping the input and output type. Not every
    // backend honours every setting; Default always works and means "same as
    // the input type".
    // -----------------------------------------------------------------------
    static void mixed_precision_section(Queue& ctx) {
        section("Mixed precision");

        const int n = 8;
        auto A = batch_of<float>(n, n, 1, random_host<float>, 91);
        auto Bm = batch_of<float>(n, n, 1, random_host<float>, 92);
        auto ref = Matrix<float>::Zeros(n, n);
        gemm<B>(ctx, A, Bm, ref, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
        ctx.wait();
        report_error("float gemm (Default) error", max_abs_diff(to_host(ref, 0), matmul(to_host(A, 0), to_host(Bm, 0))),
                     1e-4);

        // TF32 keeps float in and out but truncates the mantissa internally,
        // so the answer is close, not equal. Backends that do not implement it
        // throw; that is informative rather than a failure.
        auto tf32 = Matrix<float>::Zeros(n, n);
        try {
            gemm<B>(ctx, A, Bm, tf32, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::TF32);
            ctx.wait();
            report_error("TF32 gemm vs float gemm", max_abs_diff(to_host(tf32, 0), to_host(ref, 0)), 1e-2);
        } catch (const std::exception& e) {
            report_skip("TF32 gemm", std::string("not available: ") + e.what());
        }
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("2. Batched dense BLAS")

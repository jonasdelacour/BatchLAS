// 2. Batched dense BLAS
//
// gemm, gemv, symm, syrk, syr2k, trmm, trsm, plus heterogeneous batches and
// mixed-precision compute.

#include <complex>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 4;

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        gemm_section(ctx);
        gemv_section(ctx);
        symm_section(ctx);
        rank_k_section(ctx);
        triangular_section(ctx);
        heterogeneous_section(ctx);
        mixed_precision_section(ctx);
    }

    // gemm — C <- alpha*op(A)*op(B) + beta*C.
    //
    // The workhorse. op() is NoTrans/Trans/ConjTrans and is folded into the
    // kernel, so shapes are checked against the op'd operands: with
    // transA = Trans, an m-by-k A means op(A) is k-by-m.
    static void gemm_section(Queue& ctx) {
        section("gemm - general matrix product");

        auto A = Matrix<double>::Random(5, 4, false, kBatch, 1);
        auto Bm = Matrix<double>::Random(4, 3, false, kBatch, 2);
        auto C = Matrix<double>::Zeros(5, 3, kBatch);
        gemm<B>(ctx, A, Bm, C, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();
        std::cout << "(5x4)(4x3) -> 5x3, item 0:\n";
        C.view()[0].print();

        // Transposed operands, no copy made.
        auto At = Matrix<double>::Random(4, 5, false, kBatch, 3);
        auto Ct = Matrix<double>::Zeros(5, 3, kBatch);
        gemm<B>(ctx, At, Bm, Ct, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
        ctx.wait();
        print("transA=Trans: op(A) is 5x4 from a 4x5 A", Ct.rows() == 5 && Ct.cols() == 3);

        // Complex input takes ConjTrans as well.
        using C64 = std::complex<double>;
        auto Ac = Matrix<C64>::Random(5, 4, false, 2, 4);
        auto Bc = Matrix<C64>::Random(5, 3, false, 2, 5);
        auto Cc = Matrix<C64>::Zeros(4, 3, 2);
        gemm<B>(ctx, Ac, Bc, Cc, C64(1.0), C64(0.0), Transpose::ConjTrans, Transpose::NoTrans);
        ctx.wait();
        std::cout << "A^H B for complex input, item 0:\n";
        Cc.view()[0].print();
    }

    // gemv — y <- alpha*op(A)*x + beta*y.
    //
    // Vectors are `Vector<T>` / `VectorView<T>`: a pointer, a length, an
    // element stride `inc`, and a batch stride. A column of a matrix is a view
    // with inc=1 and a row is one with inc=ld, so either can be passed straight
    // in without a copy.
    static void gemv_section(Queue& ctx) {
        section("gemv - matrix-vector product");

        const int m = 6, n = 4;
        auto A = Matrix<double>::Random(m, n, false, kBatch, 11);
        auto x = Vector<double>::ones(n, kBatch);
        auto y = Vector<double>::zeros(m, kBatch);

        gemv<B>(ctx, A.view(), x, y, 1.0, 0.0, Transpose::NoTrans);
        ctx.wait();
        std::cout << "A * ones, item 0: ";
        y.batch_item(0).print();

        // A column of a matrix, used directly as a vector.
        auto column = A.view()(Slice{}, 0);
        print("column view length", column.size());
        print("column view inc", column.inc());
        auto row = A.view()(0, Slice{});
        print("row view inc (= ld)", row.inc());
    }

    // symm — C <- alpha*A*B + beta*C with A symmetric.
    //
    // Only the `uplo` triangle of A is read, so the other one can hold
    // anything. Real scalars only: the signature is constrained to RealScalar.
    static void symm_section(Queue& ctx) {
        section("symm - symmetric matrix product");

        const int n = 5, nrhs = 3;
        auto A = Matrix<double>::Random(n, n, /*hermitian=*/true, kBatch, 21);
        auto Bm = Matrix<double>::Random(n, nrhs, false, kBatch, 22);
        auto C = Matrix<double>::Zeros(n, nrhs, kBatch);

        symm<B>(ctx, A, Bm, C, 1.0, 0.0, Side::Left, Uplo::Lower);
        ctx.wait();
        std::cout << "A * B with A symmetric (lower triangle read), item 0:\n";
        C.view()[0].print();

        // Side::Right computes C <- alpha*B*A + beta*C instead.
        auto Br = Matrix<double>::Random(nrhs, n, false, kBatch, 23);
        auto Cr = Matrix<double>::Zeros(nrhs, n, kBatch);
        symm<B>(ctx, A, Br, Cr, 1.0, 0.0, Side::Right, Uplo::Lower);
        ctx.wait();
        print("Side::Right output shape", std::to_string(Cr.rows()) + "x" + std::to_string(Cr.cols()));
    }

    // syrk  — C <- alpha*A*A^T + beta*C
    // syr2k — C <- alpha*(A*B^T + B*A^T) + beta*C
    //
    // Both write only the `uplo` triangle of C and leave the other untouched.
    static void rank_k_section(Queue& ctx) {
        section("syrk / syr2k - symmetric rank-k updates");

        const int n = 5, k = 3;
        auto A = Matrix<double>::Random(n, k, false, kBatch, 31);
        auto Bm = Matrix<double>::Random(n, k, false, kBatch, 32);

        auto C = Matrix<double>::Zeros(n, n, kBatch);
        syrk<B>(ctx, A, C, 1.0, 0.0, Uplo::Lower, Transpose::NoTrans);
        ctx.wait();
        std::cout << "A A^T, lower triangle written (upper left at zero), item 0:\n";
        C.view()[0].print();

        auto C2 = Matrix<double>::Zeros(n, n, kBatch);
        syr2k<B>(ctx, A, Bm, C2, 1.0, 0.0, Uplo::Lower, Transpose::NoTrans);
        ctx.wait();
        print("syr2k C(0,0)", C2(0, 0, 0));
    }

    // trmm — C <- alpha*op(A)*B with A triangular
    // trsm — solves op(A)*X = alpha*B in place, overwriting B with X
    //
    // `diag` says whether the diagonal is stored (NonUnit) or implicitly 1
    // (Unit). trsm has no separate output — the right-hand side is the result.
    static void triangular_section(Queue& ctx) {
        section("trmm / trsm - triangular multiply and solve");

        const int n = 5, nrhs = 3;
        auto L = Matrix<double>::RandomTriangular(n, Uplo::Lower, Diag::NonUnit, kBatch, 41);
        auto X = Matrix<double>::Random(n, nrhs, false, kBatch, 42);

        auto C = Matrix<double>::Zeros(n, nrhs, kBatch);
        trmm<B>(ctx, L, X, C, 1.0, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
        ctx.wait();
        std::cout << "L * X, item 0:\n";
        C.view()[0].print();

        // Solve L*Y = C in place. Since C = L*X, Y comes back equal to X.
        auto Y = C.clone();
        trsm<B>(ctx, L, Y, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0);
        ctx.wait();
        std::cout << "solving L*Y = L*X recovers X, item 0:\n";
        Y.view()[0].print();

        // Diag::Unit ignores whatever is stored on the diagonal.
        auto Z = C.clone();
        trsm<B>(ctx, L, Z, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit, 1.0);
        ctx.wait();
        print("Diag::Unit gives a different solve", Z(0, 0, 0) != Y(0, 0, 0));
    }

    // Heterogeneous batches
    //
    // The Python facade takes a list of differently shaped arrays. In C++ you
    // allocate one batch at the maximum shape and declare the *active*
    // dimensions per item with set_active_dims. A count of 0 is legal and that
    // item is skipped. A.rows() is then the capacity, A.rows(b) the active
    // count for item b.
    static void heterogeneous_section(Queue& ctx) {
        section("Heterogeneous batches");

        const int batch = 3;
        auto A = Matrix<double>::Zeros(4, 3, batch);
        auto Bm = Matrix<double>::Zeros(3, 5, batch);
        auto C = Matrix<double>::Zeros(4, 5, batch);

        UnifiedVector<int> a_rows(batch), a_cols(batch), b_rows(batch), b_cols(batch), c_rows(batch), c_cols(batch);
        // item 0: (4x3)(3x5) -> 4x5   item 1: (2x3)(3x2) -> 2x2   item 2: empty
        const int ms[batch] = {4, 2, 0}, ks[batch] = {3, 3, 3}, ns[batch] = {5, 2, 4};
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
                for (int i = 0; i < A.rows(b); ++i) A(i, j, b) = 1.0 + i + 2.0 * j;
            for (int j = 0; j < Bm.cols(b); ++j)
                for (int i = 0; i < Bm.rows(b); ++i) Bm(i, j, b) = 1.0 + i + j;
        }

        gemm_heterogeneous<B>(ctx, A, Bm, C, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        print("is_heterogeneous", A.is_heterogeneous());
        print("capacity A.rows()", A.rows());
        print("active A.rows(1)", A.rows(1));
        std::cout << "item 1 result (2x2 inside a 4x5 allocation):\n";
        C.view()[1].print();
    }

    // Mixed precision
    //
    // ComputePrecision asks the backend to run the multiply at a lower internal
    // precision while keeping the input and output type. Default always works
    // and means "same as the input type"; not every backend implements the
    // others.
    static void mixed_precision_section(Queue& ctx) {
        section("Mixed precision");

        const int n = 8;
        auto A = Matrix<float>::Random(n, n, false, 1, 51);
        auto Bm = Matrix<float>::Random(n, n, false, 1, 52);

        auto ref = Matrix<float>::Zeros(n, n);
        gemm<B>(ctx, A, Bm, ref, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
        ctx.wait();
        print("float gemm, C(0,0)", ref(0, 0, 0));

        auto tf32 = Matrix<float>::Zeros(n, n);
        try {
            gemm<B>(ctx, A, Bm, tf32, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::TF32);
            ctx.wait();
            print("TF32 gemm, C(0,0)", tf32(0, 0, 0));
        } catch (const std::exception& e) {
            skip("TF32 gemm", e.what());
        }
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("2. Batched dense BLAS")

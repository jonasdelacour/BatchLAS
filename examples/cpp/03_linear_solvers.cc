// 3. Factorizations and linear solves
//
// potrf (Cholesky), getrf/getrs (LU with partial pivoting), getri and inv
// (explicit inverses), and solving with a factor directly.
//
// This is where the workspace contract from example 01 gets used in earnest:
// every routine here takes a `Span<std::byte>` sized by its `*_buffer_size`
// companion.

#include <complex>
#include <cstddef>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 3;

// A symmetric positive definite batch: A^T A + n*I.
template <Backend B>
Matrix<double> make_spd(Queue& ctx, int n, unsigned seed) {
    auto R = Matrix<double>::Random(n, n, false, kBatch, seed);
    auto A = Matrix<double>::Identity(n, kBatch);
    gemm<B>(ctx, R, R, A, 1.0, static_cast<double>(n), Transpose::Trans, Transpose::NoTrans);
    ctx.wait();
    return A;
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        cholesky_section(ctx);
        cholesky_solve_section(ctx);
        lu_section(ctx);
        inverse_section(ctx);
        complex_section(ctx);
    }

    // potrf — Cholesky factorization of a symmetric positive definite matrix.
    //
    // Factors in place: the `uplo` triangle of A is replaced by L (or U) and
    // the other triangle is left as it was. A = L*L^H for Uplo::Lower.
    static void cholesky_section(Queue& ctx) {
        section("potrf - Cholesky factorization");

        const int n = 5;
        auto A = make_spd<B>(ctx, n, 11);

        UnifiedVector<std::byte> ws(potrf_buffer_size<B>(ctx, A, Uplo::Lower));
        potrf<B>(ctx, A, Uplo::Lower, ws.to_span());
        ctx.wait();

        std::cout << "L in the lower triangle (upper triangle untouched), item 0:\n";
        A.view()[0].print();
    }

    // Solving with the Cholesky factor
    //
    // There is no `potrs`. Once you have L, solve A x = b as two triangular
    // solves: L y = b, then L^T x = y. Both overwrite their right-hand side,
    // so one buffer carries b -> y -> x.
    static void cholesky_solve_section(Queue& ctx) {
        section("Solving with the Cholesky factor");

        const int n = 5, nrhs = 2;
        auto A = make_spd<B>(ctx, n, 21);
        auto X = Matrix<double>::Random(n, nrhs, false, kBatch, 22);

        // b = A * x, so the solve should give x back.
        auto Bm = Matrix<double>::Zeros(n, nrhs, kBatch);
        gemm<B>(ctx, A, X, Bm, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        UnifiedVector<std::byte> ws(potrf_buffer_size<B>(ctx, A, Uplo::Lower));
        potrf<B>(ctx, A, Uplo::Lower, ws.to_span());
        trsm<B>(ctx, A, Bm, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0);
        trsm<B>(ctx, A, Bm, Side::Left, Uplo::Lower, Transpose::Trans, Diag::NonUnit, 1.0);
        ctx.wait();

        std::cout << "x used to build b, item 0:\n";
        X.view()[0].print();
        std::cout << "x recovered by the two solves:\n";
        Bm.view()[0].print();
    }

    // getrf / getrs — LU with partial pivoting.
    //
    // getrf factors A in place into L and U and writes the pivot sequence into
    // a `Span<int64_t>` of length n per batch item, which you allocate. getrs
    // then takes the factored A *and* the same pivots and solves A X = B,
    // overwriting B with X.
    //
    // Treat the pivots as an opaque token to hand back to getrs: what actually
    // lands in that buffer is int32 on the CUDA path and int64 on the netlib
    // one. See the known issues in the README.
    static void lu_section(Queue& ctx) {
        section("getrf / getrs - LU with partial pivoting");

        const int n = 5, nrhs = 2;
        auto A = Matrix<double>::Random(n, n, false, kBatch, 31);
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) += static_cast<double>(n);  // well conditioned

        auto X = Matrix<double>::Random(n, nrhs, false, kBatch, 32);
        auto Bm = Matrix<double>::Zeros(n, nrhs, kBatch);
        gemm<B>(ctx, A, X, Bm, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        // One pivot per row, per batch item.
        UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(getrf_buffer_size<B>(ctx, A));
        getrf<B>(ctx, A, pivots.to_span(), ws.to_span());
        ctx.wait();
        print("pivot array length (n per item)", pivots.size());

        UnifiedVector<std::byte> ws2(getrs_buffer_size<B>(ctx, A, Bm, Transpose::NoTrans));
        getrs<B>(ctx, A, Bm, Transpose::NoTrans, pivots.to_span(), ws2.to_span());
        ctx.wait();

        std::cout << "x used to build b, item 0:\n";
        X.view()[0].print();
        std::cout << "x recovered by getrs:\n";
        Bm.view()[0].print();

        // Transpose::Trans solves A^T X = B with the same factors and pivots.
        auto Bt = Matrix<double>::Random(n, nrhs, false, kBatch, 33);
        getrs<B>(ctx, A, Bt, Transpose::Trans, pivots.to_span(), ws2.to_span());
        ctx.wait();
        print("A^T solve done with the same factorization", true);
    }

    // getri and inv — explicit inverses.
    //
    // getri takes an already-factored A plus its pivots and writes A^-1 into a
    // separate output. `inv` is the one-call version and factors internally;
    // `inv_matrix` also allocates the result for you.
    //
    // Forming an inverse to solve a system is slower and less accurate than
    // getrs — do it when you need the entries of A^-1 themselves.
    static void inverse_section(Queue& ctx) {
        section("getri / inv - explicit inverses");

        const int n = 4;
        auto A = Matrix<double>::Random(n, n, false, kBatch, 41);
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) += static_cast<double>(n);
        auto original = A.clone();

        // inv: factor and invert in one call.
        auto Ainv = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<std::byte> ws(inv_buffer_size<B>(ctx, A));
        inv<B>(ctx, A, Ainv, ws.to_span());
        ctx.wait();

        // A * A^-1 should be the identity.
        auto I = Matrix<double>::Zeros(n, n, kBatch);
        gemm<B>(ctx, original, Ainv, I, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();
        std::cout << "A * inv(A), item 0:\n";
        I.view()[0].print();

        // getri: the same result from a factorization you already have.
        auto F = original.clone();
        auto Ainv2 = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws_f(getrf_buffer_size<B>(ctx, F));
        getrf<B>(ctx, F, pivots.to_span(), ws_f.to_span());
        UnifiedVector<std::byte> ws_i(getri_buffer_size<B>(ctx, F));
        getri<B>(ctx, F, Ainv2, pivots.to_span(), ws_i.to_span());
        ctx.wait();
        print("getri agrees with inv at (0,0)", std::abs(Ainv2(0, 0, 0) - Ainv(0, 0, 0)) < 1e-12);

        // The allocating convenience overload — no workspace to manage.
        auto Ainv3 = inv_matrix<B>(ctx, original);
        ctx.wait();
        print("inv_matrix allocates its own output", Ainv3.rows() == n && Ainv3.batch_size() == kBatch);

        // A batch of one is the awkward case: `Matrix` allocates its array of
        // per-item base pointers only when batch_size > 1, and the batched
        // vendor routine has nothing to read. Build the view yourself with an
        // explicit pointer array. See the known issues in the README.
        auto One = Matrix<double>::Identity(n, 1);
        for (int i = 0; i < n; ++i) One(i, i, 0) = 2.0 + i;
        auto OneInv = Matrix<double>::Zeros(n, n, 1);
        UnifiedVector<double*> pa(1), pc(1);
        pa[0] = One.data().data();
        pc[0] = OneInv.data().data();
        MatrixView<double> one_v(One.data().data(), n, n, One.ld(), One.stride(), 1, pa.data());
        MatrixView<double> one_inv_v(OneInv.data().data(), n, n, OneInv.ld(), OneInv.stride(), 1, pc.data());

        UnifiedVector<int64_t> piv1(n);
        UnifiedVector<std::byte> w1(getrf_buffer_size<B>(ctx, one_v));
        UnifiedVector<std::byte> w2(getri_buffer_size<B>(ctx, one_v));
        getrf<B>(ctx, one_v, piv1.to_span(), w1.to_span());
        getri<B>(ctx, one_v, one_inv_v, piv1.to_span(), w2.to_span());
        ctx.wait();
        std::cout << "inverse of diag(2,3,4,5) as a batch of one:\n";
        OneInv.print();
    }

    // Complex input
    //
    // The same routines take std::complex<float> and std::complex<double>. For
    // Hermitian positive definite input potrf gives A = L*L^H — note the
    // conjugate transpose.
    static void complex_section(Queue& ctx) {
        section("Complex input");

        using C64 = std::complex<double>;
        const int n = 4;

        auto R = Matrix<C64>::Random(n, n, false, 2, 51);
        auto A = Matrix<C64>::Identity(n, 2);
        gemm<B>(ctx, R, R, A, C64(1.0), C64(n), Transpose::ConjTrans, Transpose::NoTrans);
        ctx.wait();

        UnifiedVector<std::byte> ws(potrf_buffer_size<B>(ctx, A, Uplo::Lower));
        potrf<B>(ctx, A, Uplo::Lower, ws.to_span());
        ctx.wait();
        std::cout << "Cholesky factor of a Hermitian positive definite matrix, item 0:\n";
        A.view()[0].print();
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("3. Factorizations and linear solves")

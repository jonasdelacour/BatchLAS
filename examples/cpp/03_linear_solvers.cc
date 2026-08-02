// 3. Factorizations and linear solves
//
// potrf (Cholesky), getrf/getrs (LU with partial pivoting), getri and inv
// (explicit inverses), and solving with a factor directly.
//
// This is where the workspace contract from example 01 gets used in anger:
// every routine here takes a `Span<std::byte>` you sized with its
// `*_buffer_size` companion.

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

constexpr int kBatch = 3;

// A `Matrix` allocates its array of per-batch-item base pointers only when
// batch_size > 1 (src/matrix.cc, the dense constructors). Vendor routines that
// take a pointer array — cuBLAS's batched getrf/getri — therefore throw
// "data_ptrs target is null" on a *single* matrix. Building the view yourself
// with an explicit pointer array is the workaround. See the known issues in
// examples/cpp/README.md.
template <typename T>
MatrixView<T> with_pointer_array(const Matrix<T>& M, UnifiedVector<T*>& ptrs) {
    for (int b = 0; b < M.batch_size(); ++b) ptrs[b] = M.data().data() + b * M.stride();
    return MatrixView<T>(M.data().data(), M.rows(), M.cols(), M.ld(), M.stride(), M.batch_size(), ptrs.data());
}

// getrf writes its pivots into a `Span<int64_t>`, but what lands there depends
// on the backend: the netlib path widens LAPACK's int32 to int64, while cuBLAS
// writes int32 straight into the buffer. Passing the span back to getrs is
// always fine — only *reading* the values needs this.
template <Backend B>
int pivot_at(const UnifiedVector<int64_t>& pivots, size_t index) {
    if constexpr (B == Backend::NETLIB) {
        return static_cast<int>(pivots[index]);
    } else {
        return pivots.to_span().template as_span<int>()[index];
    }
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

    // -----------------------------------------------------------------------
    // potrf — Cholesky factorization of a symmetric positive definite matrix.
    //
    // Factors in place: the `uplo` triangle of A is replaced by L (or U), and
    // the other triangle is left as it was. A = L*L^H for Uplo::Lower.
    // -----------------------------------------------------------------------
    static void cholesky_section(Queue& ctx) {
        section("potrf - Cholesky factorization");

        const int n = 6;
        std::vector<HostMatrix<double>> originals;
        Matrix<double> A(n, n, kBatch);
        for (int b = 0; b < kBatch; ++b) {
            originals.push_back(random_spd_host<double>(n, 100 + b));
            from_host(originals.back(), A, b);
        }

        const size_t bytes = potrf_buffer_size<B>(ctx, A, Uplo::Lower);
        UnifiedVector<std::byte> ws(bytes);
        potrf<B>(ctx, A, Uplo::Lower, ws.to_span());
        ctx.wait();

        // Check L*L^T == the original, using only the lower triangle of the
        // result — potrf leaves the upper triangle untouched.
        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto L = keep_triangle(to_host(A, b), Uplo::Lower);
            worst = std::max(worst, max_abs_diff(matmul(L, L, Transpose::NoTrans, Transpose::Trans), originals[b]));
        }
        report_error("|L L^T - A|", worst, 1e-10);
    }

    // -----------------------------------------------------------------------
    // Solving with the Cholesky factor
    //
    // There is no `potrs`. Once you have L, solve A x = b as two triangular
    // solves with trsm: L y = b, then L^T x = y. Both overwrite their
    // right-hand side, so the same buffer carries b -> y -> x.
    // -----------------------------------------------------------------------
    static void cholesky_solve_section(Queue& ctx) {
        section("Solving with the Cholesky factor");

        const int n = 6, nrhs = 2;
        auto Ah = random_spd_host<double>(n, 7);
        auto A = broadcast(Ah, kBatch);
        auto Xtrue = batch_of<double>(n, nrhs, kBatch, random_host<double>, 11);

        // Right-hand side b = A * x_true.
        auto Bm = Matrix<double>::Zeros(n, nrhs, kBatch);
        gemm<B>(ctx, A, Xtrue, Bm, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        UnifiedVector<std::byte> ws(potrf_buffer_size<B>(ctx, A, Uplo::Lower));
        potrf<B>(ctx, A, Uplo::Lower, ws.to_span());
        trsm<B>(ctx, A, Bm, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0);
        trsm<B>(ctx, A, Bm, Side::Left, Uplo::Lower, Transpose::Trans, Diag::NonUnit, 1.0);
        ctx.wait();

        report_error("Cholesky solve error", max_abs_diff_batched(Bm.view(), Xtrue.view()), 1e-9);
    }

    // -----------------------------------------------------------------------
    // getrf / getrs — LU with partial pivoting.
    //
    // getrf factors A in place into L and U and writes the pivot sequence into
    // a `Span<int64_t>` of length n per batch item, which you allocate. getrs
    // then consumes the factored A *and* the same pivots to solve A X = B,
    // overwriting B with X.
    // -----------------------------------------------------------------------
    static void lu_section(Queue& ctx) {
        section("getrf / getrs - LU with partial pivoting");

        const int n = 6, nrhs = 2;
        auto A = batch_of<double>(n, n, kBatch, random_host<double>, 21);

        // Push the diagonal up so the systems are well conditioned.
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) += static_cast<double>(n);

        std::vector<HostMatrix<double>> originals;
        for (int b = 0; b < kBatch; ++b) originals.push_back(to_host(A, b));

        auto Xtrue = batch_of<double>(n, nrhs, kBatch, random_host<double>, 31);
        auto Bm = Matrix<double>::Zeros(n, nrhs, kBatch);
        gemm<B>(ctx, A, Xtrue, Bm, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        // One pivot per row, per batch item.
        UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(getrf_buffer_size<B>(ctx, A));
        getrf<B>(ctx, A, pivots.to_span(), ws.to_span());
        ctx.wait();
        report("pivot array length", pivots.size());

        UnifiedVector<std::byte> ws2(getrs_buffer_size<B>(ctx, A, Bm, Transpose::NoTrans));
        getrs<B>(ctx, A, Bm, Transpose::NoTrans, pivots.to_span(), ws2.to_span());
        ctx.wait();

        report_error("LU solve error", max_abs_diff_batched(Bm.view(), Xtrue.view()), 1e-9);

        // P*A = L*U: reconstruct from the factored matrix and the pivots.
        // Pivots are 1-based (LAPACK convention): row i was swapped with
        // row pivots[i]-1.
        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto LU = to_host(A, b);
            HostMatrix<double> L(n, n), U(n, n);
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    if (i > j) L(i, j) = LU(i, j);
                    else U(i, j) = LU(i, j);
                }
            }
            for (int i = 0; i < n; ++i) L(i, i) = 1.0;

            auto PA = originals[b];
            for (int i = 0; i < n; ++i) {
                const int p = pivot_at<B>(pivots, static_cast<size_t>(b) * n + i) - 1;  // 1-based
                if (p != i && p >= 0 && p < n) {
                    for (int j = 0; j < n; ++j) std::swap(PA(i, j), PA(p, j));
                }
            }
            worst = std::max(worst, max_abs_diff(matmul(L, U), PA));
        }
        report_error("|L U - P A|", worst, 1e-10);

        // Transpose::Trans solves A^T X = B with the same factors.
        auto Bt = Matrix<double>::Zeros(n, nrhs, kBatch);
        for (int b = 0; b < kBatch; ++b) from_host(matmul(transposed(originals[b]), to_host(Xtrue, b)), Bt, b);
        getrs<B>(ctx, A, Bt, Transpose::Trans, pivots.to_span(), ws2.to_span());
        ctx.wait();
        report_error("LU solve (transposed) error", max_abs_diff_batched(Bt.view(), Xtrue.view()), 1e-9);
    }

    // -----------------------------------------------------------------------
    // getri and inv — explicit inverses.
    //
    // getri takes an already-factored A plus its pivots and writes A^-1 into a
    // separate output. `inv` is the one-call version: it factors internally.
    // There is also an allocating overload, `inv_matrix`, when you would
    // rather not size a workspace at all.
    //
    // Forming an inverse to solve a system is slower and less accurate than
    // getrs; do it when you need the entries of A^-1 themselves.
    // -----------------------------------------------------------------------
    static void inverse_section(Queue& ctx) {
        section("getri / inv - explicit inverses");

        const int n = 5;
        auto A = batch_of<double>(n, n, kBatch, random_host<double>, 41);
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) += static_cast<double>(n);

        std::vector<HostMatrix<double>> originals;
        for (int b = 0; b < kBatch; ++b) originals.push_back(to_host(A, b));

        // inv: factor and invert in one call.
        auto Ainv = Matrix<double>::Zeros(n, n, kBatch);
        {
            UnifiedVector<std::byte> ws(inv_buffer_size<B>(ctx, A));
            inv<B>(ctx, A, Ainv, ws.to_span());
            ctx.wait();
        }

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            worst = std::max(worst, max_abs_diff(matmul(originals[b], to_host(Ainv, b)), identity<double>(n)));
        }
        report_error("|A A^-1 - I| (inv)", worst, 1e-9);

        // getri: same result from a factorization you already have. It needs
        // the pivots getrf produced, and consumes the factored A rather than
        // the original.
        auto Af = batch_of<double>(n, n, kBatch, random_host<double>, 41);
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < n; ++i) Af(i, i, b) += static_cast<double>(n);

        auto Ainv2 = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws_f(getrf_buffer_size<B>(ctx, Af));
        getrf<B>(ctx, Af, pivots.to_span(), ws_f.to_span());
        UnifiedVector<std::byte> ws_i(getri_buffer_size<B>(ctx, Af));
        getri<B>(ctx, Af, Ainv2, pivots.to_span(), ws_i.to_span());
        ctx.wait();

        worst = 0.0;
        for (int b = 0; b < kBatch; ++b)
            worst = std::max(worst, max_abs_diff(matmul(originals[b], to_host(Ainv2, b)), identity<double>(n)));
        report_error("|A A^-1 - I| (getri)", worst, 1e-9);

        // A batch of one is the awkward case: `Matrix` skips allocating its
        // pointer array, and the batched vendor routine has nothing to read.
        // Supply the array yourself and it works.
        auto A1 = broadcast(originals[0], 1);
        auto Ainv1 = Matrix<double>::Zeros(n, n, 1);
        UnifiedVector<double*> pa(1), pc(1);
        auto A1v = with_pointer_array(A1, pa);
        auto Ainv1v = with_pointer_array(Ainv1, pc);
        UnifiedVector<int64_t> piv1(n);
        UnifiedVector<std::byte> ws1(getrf_buffer_size<B>(ctx, A1v));
        UnifiedVector<std::byte> ws2(getri_buffer_size<B>(ctx, A1v));
        getrf<B>(ctx, A1v, piv1.to_span(), ws1.to_span());
        getri<B>(ctx, A1v, Ainv1v, piv1.to_span(), ws2.to_span());
        ctx.wait();
        report_error("|A A^-1 - I| (batch of one, explicit pointer array)",
                     max_abs_diff(matmul(originals[0], to_host(Ainv1, 0)), identity<double>(n)), 1e-9);

        // The allocating convenience overload — no workspace to manage.
        auto A3 = broadcast(originals[1], kBatch);
        auto Ainv3 = inv_matrix<B>(ctx, A3);
        ctx.wait();
        report_error("|A A^-1 - I| (inv_matrix)",
                     max_abs_diff(matmul(originals[1], to_host(Ainv3, 0)), identity<double>(n)), 1e-9);
    }

    // -----------------------------------------------------------------------
    // Complex input
    //
    // The same routines take std::complex<float> and std::complex<double>. For
    // Hermitian positive definite input, potrf gives A = L*L^H — note the
    // conjugate transpose.
    // -----------------------------------------------------------------------
    static void complex_section(Queue& ctx) {
        section("Complex input");

        using C = std::complex<double>;
        const int n = 5;
        auto Ah = random_spd_host<C>(n, 51);  // Hermitian positive definite
        auto A = broadcast(Ah, 2);

        UnifiedVector<std::byte> ws(potrf_buffer_size<B>(ctx, A, Uplo::Lower));
        potrf<B>(ctx, A, Uplo::Lower, ws.to_span());
        ctx.wait();

        auto L = keep_triangle(to_host(A, 0), Uplo::Lower);
        report_error("|L L^H - A| (complex)", max_abs_diff(matmul(L, L, Transpose::NoTrans, Transpose::ConjTrans), Ah),
                     1e-10);

        // And an LU solve in complex arithmetic.
        auto Ac = batch_of<C>(n, n, 2, random_host<C>, 61);
        for (int b = 0; b < 2; ++b)
            for (int i = 0; i < n; ++i) Ac(i, i, b) += C(static_cast<double>(n), 0.0);

        std::vector<HostMatrix<C>> originals;
        for (int b = 0; b < 2; ++b) originals.push_back(to_host(Ac, b));

        auto Xtrue = batch_of<C>(n, 1, 2, random_host<C>, 71);
        auto Bm = Matrix<C>::Zeros(n, 1, 2);
        gemm<B>(ctx, Ac, Xtrue, Bm, C(1.0), C(0.0), Transpose::NoTrans, Transpose::NoTrans);
        ctx.wait();

        UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * 2);
        UnifiedVector<std::byte> ws_f(getrf_buffer_size<B>(ctx, Ac));
        getrf<B>(ctx, Ac, pivots.to_span(), ws_f.to_span());
        UnifiedVector<std::byte> ws_s(getrs_buffer_size<B>(ctx, Ac, Bm, Transpose::NoTrans));
        getrs<B>(ctx, Ac, Bm, Transpose::NoTrans, pivots.to_span(), ws_s.to_span());
        ctx.wait();

        report_error("complex LU solve error", max_abs_diff_batched(Bm.view(), Xtrue.view()), 1e-9);
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("3. Factorizations and linear solves")

// 9. Sparse matrices and iterative eigensolvers
//
// CSR matrices, spmm, and the iterative eigensolvers built on them: syevx
// (a few extreme eigenpairs), ritz_values, lanczos, and ILU(k) preconditioning.
//
// Sparsity is a *format*, not a separate type: `Matrix<T, MatrixFormat::CSR>`
// alongside the dense default, and the routines that take either are templated
// on the format.

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 2;
constexpr int kN = 64;

using Csr = Matrix<double, MatrixFormat::CSR>;

// A symmetric sparse batch with a dominant diagonal, built by hand so the
// sparsity pattern is explicit.
//
// The CSR constructor takes the three arrays plus the strides between batch
// items: `matrix_stride` between value/column arrays and `offset_stride`
// between row-offset arrays. Row offsets are per item and restart at 0.
//
// (`Matrix<T, CSR>::RandomSparseHermitian` generates one for you, but is not
// actually symmetric at these sizes — see the known issues in the README.)
Csr make_matrix(double diagonal = 4.0) {
    const int far = 7;
    std::vector<int> offsets, cols;
    std::vector<double> vals;

    offsets.push_back(0);
    for (int i = 0; i < kN; ++i) {
        if (i - far >= 0) cols.push_back(i - far);
        if (i - 1 >= 0) cols.push_back(i - 1);
        cols.push_back(i);
        if (i + 1 < kN) cols.push_back(i + 1);
        if (i + far < kN) cols.push_back(i + far);
        offsets.push_back(static_cast<int>(cols.size()));
    }
    const int nnz = static_cast<int>(cols.size());

    std::vector<int> all_offsets, all_cols;
    for (int b = 0; b < kBatch; ++b) {
        all_offsets.insert(all_offsets.end(), offsets.begin(), offsets.end());
        all_cols.insert(all_cols.end(), cols.begin(), cols.end());
        for (int i = 0; i < kN; ++i) {
            for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
                const int j = cols[k];
                vals.push_back(i == j ? diagonal + 0.1 * b : (std::abs(i - j) == 1 ? -1.0 : 0.3));
            }
        }
    }

    return Csr(vals.data(), all_offsets.data(), all_cols.data(), nnz, kN, kN, /*matrix_stride=*/nnz,
               /*offset_stride=*/kN + 1, kBatch);
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        build_section(ctx);
        spmm_section(ctx);
        syevx_section(ctx);
        history_section(ctx);
        ritz_section(ctx);
        lanczos_section(ctx);
        iluk_section(ctx);
        combined_section(ctx);
    }

    // Building a batch of sparse symmetric matrices
    static void build_section(Queue& ctx) {
        section("Building a batch of sparse symmetric matrices");

        auto A = make_matrix();
        ctx.wait();

        print("shape", std::to_string(A.rows()) + " x " + std::to_string(A.cols()) + ", batch " +
                           std::to_string(A.batch_size()));
        print("stored non-zeros per item", A.nnz());
        print("matrix_stride (between value arrays)", A.matrix_stride());
        print("offset_stride (between row-offset arrays)", A.offset_stride());
    }

    // spmm — sparse times dense.
    //
    // C <- alpha*op(A)*op(B) + beta*C with A sparse and B, C dense. Same shape
    // as gemm plus a workspace: the vendor sparse libraries need scratch and
    // sometimes a preprocessing pass.
    static void spmm_section(Queue& ctx) {
        section("spmm - sparse times dense");

        auto A = make_matrix();
        auto X = Matrix<double>::Identity(kN, kBatch);
        auto C = Matrix<double>::Zeros(kN, kN, kBatch);

        UnifiedVector<std::byte> ws(spmm_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), X.view(), C.view(), 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans));
        spmm<B, double, MatrixFormat::CSR>(ctx, A.view(), X.view(), C.view(), 1.0, 0.0, Transpose::NoTrans,
                                           Transpose::NoTrans, ws.to_span());
        ctx.wait();

        std::cout << "A * I, top-left corner — the sparse matrix made dense:\n";
        C.view()[0].print(std::cout, 6, 6);
    }

    // syevx — a few extreme eigenpairs.
    //
    // A block iterative solver: it finds `neigs` eigenvalues at one end of the
    // spectrum without touching the rest. `SyevxParams::find_largest` picks the
    // end, `iterations` caps the work, and the tolerances decide convergence.
    //
    // Unlike syev it takes A in either format — the MFormat template parameter
    // — and does not overwrite it.
    //
    // Ask for eigenvectors even when you only want values: the values-only path
    // faults on the SYCL native-CPU device. See the known issues in the README.
    static void syevx_section(Queue& ctx) {
        section("syevx - a few extreme eigenpairs");

        auto A = make_matrix();
        const size_t neigs = 4;

        SyevxParams<double> params;
        params.find_largest = true;
        params.iterations = 200;
        params.extra_directions = 4;  // a larger search block converges more reliably

        UnifiedVector<double> W(neigs * kBatch);
        auto V = Matrix<double>::Zeros(kN, static_cast<int>(neigs), kBatch);
        UnifiedVector<std::byte> ws(syevx_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), W.to_span(), neigs, JobType::EigenVectors, V.view(), params));
        syevx<B, double, MatrixFormat::CSR>(ctx, A.view(), W.to_span(), neigs, ws.to_span(), JobType::EigenVectors,
                                            V.view(), params);
        ctx.wait();
        print_values("largest 4 eigenvalues", W.to_span(), 4);

        params.find_largest = false;
        UnifiedVector<double> W2(neigs * kBatch);
        auto V2 = Matrix<double>::Zeros(kN, static_cast<int>(neigs), kBatch);
        UnifiedVector<std::byte> ws2(syevx_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), W2.to_span(), neigs, JobType::EigenVectors, V2.view(), params));
        syevx<B, double, MatrixFormat::CSR>(ctx, A.view(), W2.to_span(), neigs, ws2.to_span(), JobType::EigenVectors,
                                            V2.view(), params);
        ctx.wait();
        print_values("smallest 4 eigenvalues", W2.to_span(), 4);
    }

    // Convergence history
    //
    // `SyevxInstrumentation` is an optional sink you point at your own buffers;
    // syevx writes a residual (and optionally Ritz value) history into them as
    // it iterates. The layout is [iteration][batch][eigenvalue], and
    // `iterations_done` reports how many iterations each item needed.
    static void history_section(Queue& ctx) {
        section("Convergence history");

        auto A = make_matrix();
        const size_t neigs = 4;
        const size_t max_iterations = 100;

        UnifiedVector<double> residuals(max_iterations * kBatch * neigs, 0.0);
        UnifiedVector<int32_t> iterations_done(kBatch, 0);

        SyevxInstrumentation<double> instr;
        instr.best_residual_history = residuals.to_span();
        instr.iterations_done = iterations_done.data();
        instr.max_iterations = max_iterations;
        instr.store_every = 1;

        SyevxParams<double> params;
        params.iterations = max_iterations;
        params.extra_directions = 4;
        params.instrumentation = &instr;

        UnifiedVector<double> W(neigs * kBatch);
        auto V = Matrix<double>::Zeros(kN, static_cast<int>(neigs), kBatch);
        UnifiedVector<std::byte> ws(syevx_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), W.to_span(), neigs, JobType::EigenVectors, V.view(), params));
        syevx<B, double, MatrixFormat::CSR>(ctx, A.view(), W.to_span(), neigs, ws.to_span(), JobType::EigenVectors,
                                            V.view(), params);
        ctx.wait();

        print("iterations used (item 0)", iterations_done[0]);
        const size_t iter_stride = kBatch * neigs;
        std::cout << "best residual, first eigenvalue, every 10 iterations:";
        for (size_t it = 0; it < static_cast<size_t>(iterations_done[0]); it += 10)
            std::cout << " " << residuals[it * iter_stride];
        std::cout << "\n";
    }

    // ritz_values — Rayleigh quotients for a subspace you already have.
    //
    // Given an orthonormal V, returns diag(V^H A V): the best eigenvalue
    // estimates that subspace supports. This is the projection step inside
    // every block iterative method, exposed on its own. The convenience
    // overload allocates both the output and its workspace.
    static void ritz_section(Queue& ctx) {
        section("ritz_values - Rayleigh quotients for a given subspace");

        auto A = make_matrix();
        const int k = 4;

        auto V = Matrix<double>::Random(kN, k, false, kBatch, 21);
        UnifiedVector<std::byte> ows(ortho_buffer_size<B>(ctx, V, Transpose::NoTrans));
        ortho<B>(ctx, V, Transpose::NoTrans, ows.to_span());
        ctx.wait();

        auto vals = ritz_values<B, double, MatrixFormat::CSR>(ctx, A.view(), V.view());
        ctx.wait();
        std::cout << "Ritz values of a random 4-dimensional subspace: ";
        vals.batch_item(0).print(std::cout, k);
    }

    // lanczos — Krylov subspace eigenvalues.
    //
    // Builds a Krylov basis and solves the small tridiagonal problem it
    // produces. Cheaper per iteration than syevx but less robust — the classic
    // method rather than the recommended one, and it does not converge on the
    // NETLIB backend (see the README).
    //
    // Note the sizing: it fills the whole spectrum of the Krylov tridiagonal,
    // so W is n per batch item, not "the number you want".
    static void lanczos_section(Queue& ctx) {
        section("lanczos - Krylov subspace eigenvalues");

        auto A = make_matrix();

        LanczosParams<double> params;
        params.ortho_algorithm = OrthoAlgorithm::CGS2;
        params.sort_order = SortOrder::Ascending;

        UnifiedVector<double> W(static_cast<size_t>(kN) * kBatch);
        // A real V, not a default-constructed view: the values-only path trips
        // an internal sort that reads V.batch_size(). See the README.
        auto V = Matrix<double>::Zeros(kN, kN, kBatch);
        UnifiedVector<std::byte> ws(lanczos_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), W.to_span(), JobType::EigenVectors, V.view(), params));
        lanczos<B, double, MatrixFormat::CSR>(ctx, A.view(), W.to_span(), ws.to_span(), JobType::EigenVectors,
                                              V.view(), params);
        ctx.wait();
        print_values("lanczos, smallest 6 of the full spectrum", W.to_span(), 6);
    }

    // ILU(k) preconditioning
    //
    // `iluk_factorize` returns a preconditioner object — the one place in this
    // API where a routine hands back a handle rather than filling something you
    // allocated. `iluk_apply` applies it, and syevx takes a pointer to it in
    // its params. levels_of_fill = 0 is ILU(0): the factors keep A's pattern.
    static void iluk_section(Queue& ctx) {
        section("ILU(k) preconditioning");

        auto A = make_matrix();

        ILUKParams<double> params;
        params.levels_of_fill = 0;
        auto M = iluk_factorize<B>(ctx, A.view(), params);
        ctx.wait();

        print("preconditioner n", M.n);
        print("preconditioner batch", M.batch_size);
        print("levels of fill", M.levels_of_fill);
        print("factor non-zeros", M.lu.nnz());

        // Applying M^-1 to A x should land near x — that is what
        // preconditioning means. ILU(0) is approximate, so "near", not "at".
        auto X = Matrix<double>::Random(kN, 2, false, kBatch, 31);
        auto Bm = Matrix<double>::Zeros(kN, 2, kBatch);
        UnifiedVector<std::byte> sws(spmm_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), X.view(), Bm.view(), 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans));
        spmm<B, double, MatrixFormat::CSR>(ctx, A.view(), X.view(), Bm.view(), 1.0, 0.0, Transpose::NoTrans,
                                           Transpose::NoTrans, sws.to_span());
        ctx.wait();

        auto Y = Matrix<double>::Zeros(kN, 2, kBatch);
        UnifiedVector<std::byte> aws(iluk_apply_buffer_size<B>(ctx, M, Bm.view(), Y.view()));
        iluk_apply<B>(ctx, M, Bm.view(), Y.view(), aws.to_span());
        ctx.wait();

        std::cout << "first entries of x:          ";
        for (int i = 0; i < 4; ++i) std::cout << " " << X(i, 0, 0);
        std::cout << "\nfirst entries of M^-1(A x):  ";
        for (int i = 0; i < 4; ++i) std::cout << " " << Y(i, 0, 0);
        std::cout << "\n";
    }

    // Combining the two
    //
    // Point SyevxParams::preconditioner at the factorization and syevx uses it.
    // Whether it pays off is a property of the problem — measure before
    // adopting one.
    static void combined_section(Queue& ctx) {
        section("Combining the two");

        auto A = make_matrix(/*diagonal=*/2.2);  // a harder, less separated spectrum
        const size_t neigs = 4;

        auto solve = [&](const ILUKPreconditioner<double>* M, const char* label) {
            UnifiedVector<int32_t> iterations_done(kBatch, 0);
            SyevxInstrumentation<double> instr;
            instr.iterations_done = iterations_done.data();
            instr.max_iterations = 200;

            SyevxParams<double> params;
            params.find_largest = false;
            params.iterations = 200;
            params.extra_directions = 4;
            params.preconditioner = M;
            params.instrumentation = &instr;

            UnifiedVector<double> W(neigs * kBatch);
            auto V = Matrix<double>::Zeros(kN, static_cast<int>(neigs), kBatch);
            UnifiedVector<std::byte> ws(syevx_buffer_size<B, double, MatrixFormat::CSR>(
                ctx, A.view(), W.to_span(), neigs, JobType::EigenVectors, V.view(), params));
            syevx<B, double, MatrixFormat::CSR>(ctx, A.view(), W.to_span(), neigs, ws.to_span(),
                                                JobType::EigenVectors, V.view(), params);
            ctx.wait();
            print_values(std::string(label) + ", smallest 4", W.to_span(), 4);
            print(std::string(label) + ", iterations", iterations_done[0]);
        };

        solve(nullptr, "unpreconditioned");

        ILUKParams<double> ip;
        ip.levels_of_fill = 1;
        auto M = iluk_factorize<B>(ctx, A.view(), ip);
        ctx.wait();
        solve(&M, "ILU(1) preconditioned");
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("9. Sparse matrices and iterative eigensolvers")

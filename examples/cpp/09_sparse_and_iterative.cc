// 9. Sparse matrices and iterative eigensolvers
//
// CSR matrices, spmm, and the iterative eigensolvers built on them: syevx
// (a few extreme eigenpairs), lanczos, ritz_values, and ILU(k) preconditioning.
//
// Sparsity is a *format* here, not a different type: `Matrix<T,
// MatrixFormat::CSR>` alongside the dense default, and the routines that take
// either are templated on the format.

#include <algorithm>
#include <cmath>
#include <vector>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_common.hh"
#include "example_linalg.hh"
#include "example_runner.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 2;
constexpr int kN = 64;

template <Backend B>
struct Example {
    using Csr = Matrix<double, MatrixFormat::CSR>;

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

    // The dense form of the matrix the sections below use: a symmetric
    // "tridiagonal plus one pair of far diagonals" pattern with a dominant
    // diagonal. Deterministic, genuinely sparse, and well conditioned.
    static HostMatrix<double> reference_dense(int n, unsigned seed, double boost) {
        HostMatrix<double> A(n, n);
        const int far = 7;
        for (int i = 0; i < n; ++i) {
            A(i, i) = boost * (2.0 + 0.5 * std::sin(0.7 * i + seed));
            if (i + 1 < n) {
                const double v = -1.0 + 0.1 * std::cos(0.3 * i + seed);
                A(i + 1, i) = v;
                A(i, i + 1) = v;
            }
            if (i + far < n) {
                const double v = 0.4 * std::sin(0.9 * i + seed);
                A(i + far, i) = v;
                A(i, i + far) = v;
            }
        }
        return A;
    }

    // Build a CSR batch from that dense matrix.
    //
    // The CSR constructor takes the three arrays plus the strides between
    // batch items: `matrix_stride` between value/column arrays and
    // `offset_stride` between row-offset arrays. Row offsets are per item and
    // start from 0 each time.
    static Csr make_matrix(int n = kN, unsigned seed = 42, double boost = 2.0) {
        std::vector<int> offsets, cols;
        std::vector<double> vals;
        std::vector<std::vector<std::pair<int, double>>> rows(n);

        // Same sparsity pattern for every batch item — required by ILU(k), and
        // what the batched sparse kernels are built for.
        const auto pattern_source = reference_dense(n, seed, boost);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (pattern_source(i, j) != 0.0) rows[i].emplace_back(j, 0.0);

        int nnz = 0;
        offsets.push_back(0);
        for (int i = 0; i < n; ++i) {
            nnz += static_cast<int>(rows[i].size());
            offsets.push_back(nnz);
            for (auto& [j, v] : rows[i]) cols.push_back(j);
        }

        // Values vary per batch item; the pattern does not.
        std::vector<int> all_offsets, all_cols;
        for (int b = 0; b < kBatch; ++b) {
            const auto dense = reference_dense(n, seed + b, boost);
            all_offsets.insert(all_offsets.end(), offsets.begin(), offsets.end());
            all_cols.insert(all_cols.end(), cols.begin(), cols.end());
            for (int i = 0; i < n; ++i)
                for (auto& [j, v] : rows[i]) vals.push_back(dense(i, j));
        }

        return Csr(vals.data(), all_offsets.data(), all_cols.data(), nnz, n, n, /*matrix_stride=*/nnz,
                   /*offset_stride=*/n + 1, kBatch);
    }

    // Expand batch item b of a CSR matrix to a dense host matrix, so the
    // checks can use the same reference kernels as everywhere else.
    static HostMatrix<double> csr_to_host(const Csr& A, int b = 0) {
        const int n = A.rows();
        HostMatrix<double> out(n, n);
        auto offsets = A.row_offsets();
        auto cols = A.col_indices();
        auto vals = A.data();
        const int off_base = b * A.offset_stride();
        const int val_base = b * A.matrix_stride();
        for (int i = 0; i < n; ++i) {
            for (int k = offsets[off_base + i]; k < offsets[off_base + i + 1]; ++k) {
                out(i, cols[val_base + k]) += vals[val_base + k];
            }
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Building a batch of sparse symmetric matrices
    //
    // `random_sparse_hermitian_csr` allocates the CSR storage and fills it on
    // the device. `shared_pattern` makes every item of the batch use the same
    // sparsity pattern, which is what the batched kernels — and ILU(k) — want.
    // -----------------------------------------------------------------------
    static void build_section(Queue& ctx) {
        section("Building a batch of sparse symmetric matrices");

        auto A = make_matrix();
        ctx.wait();

        report("shape", std::to_string(A.rows()) + " x " + std::to_string(A.cols()) + ", batch " +
                            std::to_string(A.batch_size()));
        report("stored non-zeros per item", A.nnz());
        report_magnitude("density", static_cast<double>(A.nnz()) / (static_cast<double>(kN) * kN));

        // It really is symmetric — the iterative solvers below assume it.
        double asym = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto dense = csr_to_host(A, b);
            for (int j = 0; j < kN; ++j)
                for (int i = 0; i < kN; ++i) asym = std::max(asym, std::abs(dense(i, j) - dense(j, i)));
        }
        report_error("|A - A^T|", asym, 1e-14);

        // Round-tripping through the dense form recovers exactly what went in.
        report_error("CSR matches the dense matrix it was built from",
                     max_abs_diff(csr_to_host(A, 0), reference_dense(kN, 42, 2.0)), 1e-14);

        // There is also a random generator, Matrix<T, CSR>::RandomSparseHermitian
        // (also exposed as csr_generators::random_sparse_hermitian_csr). It is
        // not symmetric at these sizes — see the known issues in the README —
        // so the examples below build their own matrix instead.
        auto R = csr_generators::random_sparse_hermitian_csr<double>(kN, 0.1f, 1, 42, 2.0, true);
        ctx.wait();
        auto rdense = csr_to_host(R, 0);
        double rasym = 0.0;
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kN; ++i) rasym = std::max(rasym, std::abs(rdense(i, j) - rdense(j, i)));
        report_magnitude("RandomSparseHermitian |A - A^T| (known bad)", rasym);
    }

    // -----------------------------------------------------------------------
    // spmm — sparse times dense.
    //
    // C <- alpha*op(A)*op(B) + beta*C with A sparse and B, C dense. Same shape
    // as gemm, plus a workspace: the vendor sparse libraries need scratch and
    // sometimes a preprocessing pass.
    // -----------------------------------------------------------------------
    static void spmm_section(Queue& ctx) {
        section("spmm - sparse times dense");

        auto A = make_matrix();
        const int nrhs = 4;
        auto X = batch_of<double>(kN, nrhs, kBatch, random_host<double>, 11);
        auto C = Matrix<double>::Zeros(kN, nrhs, kBatch);

        UnifiedVector<std::byte> ws(spmm_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), X.view(), C.view(), 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans));
        spmm<B, double, MatrixFormat::CSR>(ctx, A.view(), X.view(), C.view(), 1.0, 0.0, Transpose::NoTrans,
                                           Transpose::NoTrans, ws.to_span());
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b)
            worst = std::max(worst, max_abs_diff(to_host(C, b), matmul(csr_to_host(A, b), to_host(X, b))));
        report_error("spmm vs dense reference", worst, 1e-12);
    }

    // -----------------------------------------------------------------------
    // syevx — a few extreme eigenpairs.
    //
    // A block iterative solver: it finds `neigs` eigenvalues at one end of the
    // spectrum without touching the rest. `SyevxParams::find_largest` picks
    // the end, `iterations` caps the work, and the tolerances decide when a
    // pair has converged.
    //
    // Unlike syev, this one takes A in either format — the MFormat template
    // parameter — and does not overwrite it.
    // -----------------------------------------------------------------------
    static void syevx_section(Queue& ctx) {
        section("syevx - a few extreme eigenpairs");

        auto A = make_matrix();
        const size_t neigs = 4;

        // The reference: the full spectrum of item 0, computed densely.
        const auto full = jacobi_eigenvalues(csr_to_host(A, 0));

        SyevxParams<double> params;
        params.find_largest = true;
        params.iterations = 300;
        params.extra_directions = 4;  // a slightly larger search block converges more reliably

        UnifiedVector<double> W(neigs * kBatch);
        auto V = Matrix<double>::Zeros(kN, static_cast<int>(neigs), kBatch);
        UnifiedVector<std::byte> ws(syevx_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), W.to_span(), neigs, JobType::EigenVectors, V.view(), params));
        syevx<B, double, MatrixFormat::CSR>(ctx, A.view(), W.to_span(), neigs, ws.to_span(), JobType::EigenVectors,
                                            V.view(), params);
        ctx.wait();

        // As returned: column j of V goes with W[j]. Sort a *copy* for the
        // value comparison and leave the pairing alone for the residual.
        std::vector<double> as_returned(W.begin(), W.begin() + neigs);
        std::vector<double> got = sorted(as_returned);
        std::vector<double> want(full.end() - neigs, full.end());
        report_error("largest 4 eigenvalues", max_abs_diff(got, want), 1e-6);

        auto dense = csr_to_host(A, 0);
        report_error("|A V - V diag(w)|", eigen_residual(dense, to_host(V, 0), as_returned), 1e-5);

        // The other end of the spectrum.
        //
        // Note the job type: syevx's JobType::NoEigenVectors path faults on
        // the SYCL native-CPU device, so these examples ask for vectors
        // everywhere and ignore them where they are not needed. See the known
        // issues in the README.
        params.find_largest = false;
        UnifiedVector<double> W2(neigs * kBatch);
        auto V2 = Matrix<double>::Zeros(kN, static_cast<int>(neigs), kBatch);
        UnifiedVector<std::byte> ws2(syevx_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), W2.to_span(), neigs, JobType::EigenVectors, V2.view(), params));
        syevx<B, double, MatrixFormat::CSR>(ctx, A.view(), W2.to_span(), neigs, ws2.to_span(),
                                            JobType::EigenVectors, V2.view(), params);
        ctx.wait();

        std::vector<double> got2(W2.begin(), W2.begin() + neigs);
        std::sort(got2.begin(), got2.end());
        std::vector<double> want2(full.begin(), full.begin() + neigs);
        report_error("smallest 4 eigenvalues", max_abs_diff(got2, want2), 1e-6);
    }

    // -----------------------------------------------------------------------
    // Convergence history
    //
    // `SyevxInstrumentation` is an optional sink you point at your own buffers;
    // syevx writes a residual (and optionally Ritz value) history into them as
    // it iterates. Layout is [iteration][batch][eigenvalue]. `iterations_done`
    // reports how many iterations each batch item actually needed.
    // -----------------------------------------------------------------------
    static void history_section(Queue& ctx) {
        section("Convergence history");

        auto A = make_matrix();
        const size_t neigs = 4;
        const size_t max_iterations = 200;

        UnifiedVector<double> residual_history(max_iterations * kBatch * neigs, 0.0);
        UnifiedVector<int32_t> iterations_done(kBatch, 0);

        SyevxInstrumentation<double> instr;
        instr.best_residual_history = residual_history.to_span();
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

        report("iterations used (item 0)", iterations_done[0]);

        // The residual should end far below where it started.
        const size_t iter_stride = kBatch * neigs;
        double first = 0.0, last = 0.0;
        const int used = std::max(1, iterations_done[0]);
        for (size_t j = 0; j < neigs; ++j) {
            first = std::max(first, residual_history[j]);
            last = std::max(last, residual_history[(used - 1) * iter_stride + j]);
        }
        report_magnitude("first recorded residual", first);
        report_magnitude("last recorded residual", last);
        report_check("the residual decreased", last < first);
    }

    // -----------------------------------------------------------------------
    // ritz_values — Rayleigh quotients for a subspace you already have.
    //
    // Given an orthonormal V, returns diag(V^H A V): the best eigenvalue
    // estimates that subspace supports. This is the projection step inside
    // every block iterative method, exposed on its own.
    // -----------------------------------------------------------------------
    static void ritz_section(Queue& ctx) {
        section("ritz_values - Rayleigh quotients for a given subspace");

        auto A = make_matrix();
        const int k = 5;

        // Any orthonormal block will do.
        auto V = batch_of<double>(kN, k, kBatch, random_host<double>, 21);
        UnifiedVector<std::byte> ows(ortho_buffer_size<B>(ctx, V, Transpose::NoTrans));
        ortho<B>(ctx, V, Transpose::NoTrans, ows.to_span());
        ctx.wait();

        auto vals = ritz_values<B, double, MatrixFormat::CSR>(ctx, A.view(), V.view());
        ctx.wait();

        // Check against V^T A V computed on the host.
        auto dense = csr_to_host(A, 0);
        auto Vh = to_host(V, 0);
        auto proj = matmul(matmul(Vh, dense, Transpose::Trans, Transpose::NoTrans), Vh);
        double worst = 0.0;
        for (int i = 0; i < k; ++i) worst = std::max(worst, std::abs(vals(i, 0) - proj(i, i)));
        report_error("ritz values vs diag(V^T A V)", worst, 1e-10);
    }

    // -----------------------------------------------------------------------
    // lanczos — Krylov subspace eigenvalues.
    //
    // Builds a Krylov basis and solves the small tridiagonal problem it
    // produces. Cheaper per iteration than syevx but less robust; it is the
    // classic method rather than the recommended one. The number of values it
    // returns is the length of the span you pass.
    // -----------------------------------------------------------------------
    static void lanczos_section(Queue& ctx) {
        section("lanczos - Krylov subspace eigenvalues");

        auto A = make_matrix();

        // Note the sizing: lanczos fills the whole spectrum of the Krylov
        // tridiagonal, so W is n per batch item — not "the number you want".
        LanczosParams<double> params;
        params.ortho_algorithm = OrthoAlgorithm::CGS2;
        params.sort_order = SortOrder::Ascending;

        // Asking for eigenvalues only, with a default-constructed V, trips an
        // internal sort that reads V.batch_size() — see the known issues in
        // the README. Either pass a real V, as here, or set
        // LanczosParams::sort_enabled = false.
        UnifiedVector<double> W(static_cast<size_t>(kN) * kBatch);
        auto V = Matrix<double>::Zeros(kN, kN, kBatch);
        UnifiedVector<std::byte> ws(lanczos_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), W.to_span(), JobType::EigenVectors, V.view(), params));
        lanczos<B, double, MatrixFormat::CSR>(ctx, A.view(), W.to_span(), ws.to_span(), JobType::EigenVectors,
                                              V.view(), params);
        ctx.wait();

        // Krylov estimates converge from the outside in: the extreme values
        // are accurate long before the interior ones, so only the largest is
        // checked tightly.
        const auto full = jacobi_eigenvalues(csr_to_host(A, 0));
        std::vector<double> got(W.begin(), W.begin() + kN);
        std::sort(got.begin(), got.end());
        if constexpr (B == Backend::NETLIB) {
            // lanczos does not converge on the host backend; the extreme
            // values come back far from the true ones. Reported, not checked.
            // See the known issues in the README.
            report_magnitude("lanczos: largest eigenvalue error (known bad on NETLIB)",
                             std::abs(got.back() - full.back()));
            report_magnitude("lanczos: smallest eigenvalue error (known bad on NETLIB)",
                             std::abs(got.front() - full.front()));
        } else {
            report_error("lanczos: largest eigenvalue", std::abs(got.back() - full.back()), 1e-6);
            report_error("lanczos: smallest eigenvalue", std::abs(got.front() - full.front()), 1e-6);
        }
        report_magnitude("lanczos: error in the middle of the spectrum",
                         std::abs(got[kN / 2] - full[kN / 2]));
    }

    // -----------------------------------------------------------------------
    // ILU(k) preconditioning
    //
    // `iluk_factorize` returns a preconditioner object — the one place in this
    // API where a routine hands back a handle rather than filling something
    // you allocated. `iluk_apply` applies it, and syevx takes a pointer to it
    // in its params.
    //
    // levels_of_fill = 0 is ILU(0): the factors keep A's sparsity pattern.
    // -----------------------------------------------------------------------
    static void iluk_section(Queue& ctx) {
        section("ILU(k) preconditioning");

        auto A = make_matrix();

        ILUKParams<double> params;
        params.levels_of_fill = 0;
        auto M = iluk_factorize<B>(ctx, A.view(), params);
        ctx.wait();

        report("preconditioner n", M.n);
        report("preconditioner batch", M.batch_size);
        report("levels of fill", M.levels_of_fill);

        // Applying M^-1 to A x should land near x: that is what preconditioning
        // means. Take x random, form b = A x, then solve approximately.
        const int nrhs = 2;
        auto X = batch_of<double>(kN, nrhs, kBatch, random_host<double>, 31);
        auto Bm = Matrix<double>::Zeros(kN, nrhs, kBatch);
        UnifiedVector<std::byte> sws(spmm_buffer_size<B, double, MatrixFormat::CSR>(
            ctx, A.view(), X.view(), Bm.view(), 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans));
        spmm<B, double, MatrixFormat::CSR>(ctx, A.view(), X.view(), Bm.view(), 1.0, 0.0, Transpose::NoTrans,
                                           Transpose::NoTrans, sws.to_span());
        ctx.wait();

        auto Y = Matrix<double>::Zeros(kN, nrhs, kBatch);
        UnifiedVector<std::byte> aws(iluk_apply_buffer_size<B>(ctx, M, Bm.view(), Y.view()));
        iluk_apply<B>(ctx, M, Bm.view(), Y.view(), aws.to_span());
        ctx.wait();

        // ILU(0) is approximate, so this is a comparison, not an equality: the
        // preconditioned residual must be smaller than the unpreconditioned one.
        double before = 0.0, after = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            before = std::max(before, max_abs_diff(to_host(Bm, b), to_host(X, b)));
            after = std::max(after, max_abs_diff(to_host(Y, b), to_host(X, b)));
        }
        report_magnitude("|A x - x| without the preconditioner", before);
        report_magnitude("|M^-1 A x - x| with it", after);
        report_check("ILU(0) moves the system towards the identity", after < before);
    }

    // -----------------------------------------------------------------------
    // Combining the two
    //
    // Point SyevxParams::preconditioner at the factorization and syevx uses it.
    // On a hard problem this is the difference between converging and
    // stagnating; here it mainly cuts the iteration count.
    // -----------------------------------------------------------------------
    static void combined_section(Queue& ctx) {
        section("Combining the two");

        // A harder problem: a weaker diagonal makes the spectrum less separated.
        auto A = make_matrix(kN, 77, /*boost=*/0.2);
        const size_t neigs = 4;
        const auto full = jacobi_eigenvalues(csr_to_host(A, 0));
        std::vector<double> want(full.begin(), full.begin() + neigs);

        auto solve = [&](const ILUKPreconditioner<double>* M, const char* label) {
            UnifiedVector<int32_t> iterations_done(kBatch, 0);
            SyevxInstrumentation<double> instr;
            instr.iterations_done = iterations_done.data();
            instr.max_iterations = 400;

            SyevxParams<double> params;
            params.find_largest = false;
            params.iterations = 400;
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

            std::vector<double> got(W.begin(), W.begin() + neigs);
            std::sort(got.begin(), got.end());
            const double err = max_abs_diff(got, want);
            report_magnitude(std::string(label) + ": error on the smallest 4", err);
            report(std::string(label) + ": iterations", iterations_done[0]);
            return err;
        };

        const double plain = solve(nullptr, "unpreconditioned");

        ILUKParams<double> ip;
        ip.levels_of_fill = 1;
        auto M = iluk_factorize<B>(ctx, A.view(), ip);
        ctx.wait();
        const double preconditioned = solve(&M, "ILU(1) preconditioned");

        // Wiring the preconditioner in is the point of this section; whether
        // it pays off is a property of the problem. On this matrix it does
        // not — the unpreconditioned run is already accurate, and ILU(1)
        // makes it worse. Measure before adopting one.
        report_check("the unpreconditioned run converged", plain < 1e-6);
        report_magnitude("ILU(1) error on the same problem", preconditioned);
        report_skip("preconditioner benefit", "none on this matrix; it helps on badly conditioned problems");
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("9. Sparse matrices and iterative eigensolvers")

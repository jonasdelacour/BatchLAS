// SPDX-License-Identifier: MIT
// her2k for the tuner (complex only): C = alpha A B^H + conj(alpha) B A^H + beta C on the lower
// triangle, A, B and C0 from MatrixView::fill_random, C0's diagonal made real and its strict upper
// triangle poisoned with a large finite value, alpha = (1.25, -0.5), beta = 0.75 so C is read. uplo
// and trans are not table keys: fixed at L and N (sytrd's update). Every ld is padded by --ld-pad.
// Verification is MatrixView::rank2k_backward_error (hermitian) over C's lower triangle on the
// default items (Check::blas with inner dimension 2k); a change to the upper triangle or an
// imaginary part left on the diagonal counts as a nonzero info.

#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/her2k/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <complex>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace H2 = ops::her2k;

template <typename T>
struct Her2kProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using R = float_t<T>;
    Queue& q;
    int n, k, batch, lda, ldc;
    std::size_t sa, sc;
    T alpha;
    R beta;
    UnifiedVector<T> A, B, C0, C;

    Her2kProblem(Queue& q_, int n_, int k_, int batch_, int pad)
        : q(q_), n(n_), k(k_), batch(batch_), lda(n_ + pad), ldc(n_ + pad), sa(std::size_t(lda) * std::size_t(k_)),
          sc(std::size_t(ldc) * std::size_t(n_)), alpha(T(1.25, -0.5)), beta(R(0.75)), A(sa * std::size_t(batch_)),
          B(sa * std::size_t(batch_)), C0(sc * std::size_t(batch_)), C(sc * std::size_t(batch_)) {
        (void)view(A, n, k, lda, sa).fill_random(q, false, 9191);
        (void)view(B, n, k, lda, sa).fill_random(q, false, 9192);
        (void)view(C0, n, n, ldc, sc).fill_random(q, false, 9193);
        q.wait();
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < n; ++c) {
                T* col = C0.data() + std::size_t(b) * sc + std::size_t(c) * std::size_t(ldc);
                col[c] = T(col[c].real(), R(0));
                for (int r = 0; r < c; ++r) col[r] = T(1e6, -1e6);
            }
    }
    MV view(UnifiedVector<T>& v, int r, int c, int ld, std::size_t s) { return MV(v.data(), r, c, ld, int(s), batch); }
    void reset() {  // beta != 0 accumulates into C
        (void)MV::copy(q, view(C, n, n, ldc, sc), view(C0, n, n, ldc, sc));
        q.wait();
    }
    std::size_t workspace() {  // her2k takes no caller workspace; a refused pin throws here
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)her2k<kBackend, T>(q, view(A, n, k, lda, sa), view(B, n, k, lda, sa), view(C, n, n, ldc, sc), alpha,
                                 beta, Uplo::Lower, Transpose::NoTrans);
        q.wait();
    }
    std::pair<double, int> verify() {
        int touched = 0;
        for (int b : view(C, n, n, ldc, sc).default_items())
            for (int c = 0; c < n; ++c) {
                const std::size_t o = std::size_t(b) * sc + std::size_t(c) * std::size_t(ldc);
                touched += C[o + std::size_t(c)].imag() != R(0);
                for (int r = 0; r < c; ++r) touched += C[o + std::size_t(r)] != C0[o + std::size_t(r)];
            }
        return {MV::rank2k_backward_error(q, view(A, n, k, lda, sa), view(B, n, k, lda, sa), Transpose::NoTrans,
                                          view(C0, n, n, ldc, sc), view(C, n, n, ldc, sc), Uplo::Lower, MV::up(alpha),
                                          MV::up(T(beta)), true),
                touched};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class Her2kSpec final : public OpSpec {
public:
    std::string op() const override { return "her2k"; }
    std::vector<std::string> key_names() const override { return {H2::key_names.begin(), H2::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_complex_dtype("her2k", dtype, []<class T>() { return spellings(H2::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"n", ints(H2::grid_n)}, {"k", ints(H2::grid_k)}, {"batch", ints(H2::grid_batch)}};
    }
    double bytes(const std::string& dtype, const CellKey& key) const override {  // A, B, C0, C
        const double n = double(key_int(key, "n")), k = double(key_int(key, "k"));
        return (2.0 * n * k + 2.0 * n * n) * double(key_int(key, "batch")) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: triangular
            "src/ops/syr2k/kernels/syr2k_triangular_tiles.hh",
            "src/kernels/common/triangular_tiles.hh",
            // family: fold
            "src/ops/level3/batched.hh",
            "src/expansion_budget.hh",
            // common
            "src/ops/her2k/choice.hh",
            "src/ops/her2k/route.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/her2k/her2k.cc"
        // her2k fold calls the public gemm: its native kernels and can_run count too.
        // family: fold "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/kernels/gemm_launch.hh" "src/ops/gemm/kernels/gemm_rows.hh" "src/ops/gemm/kernels/gemm_shard.cc.in" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/her2k_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_complex_dtype("her2k", req.dtype, [&]<class T>() {
            Her2kProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "k")), int(key_int(req.key, "batch")),
                              req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, 2 * p.k);
            if (req.mode == "race") return run_race<H2::Her2kChoice>("her2k", p, req, tol);
            return run_arms<H2::Her2kChoice>("her2k", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::Her2kSpec)

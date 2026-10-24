// SPDX-License-Identifier: MIT
// herk for the tuner (complex only): C = alpha op(A) op(A)^H + beta C on the lower triangle, A and
// C0 from MatrixView::fill_random, C0's diagonal made real, its strict upper triangle poisoned with a
// large finite value, alpha = 1.25, beta = 0.75 so C is read. uplo is not a table key: fixed at L.
// Every ld is padded by --ld-pad. Verification is MatrixView::gemm_backward_error over C's lower
// triangle on the default items (Check::blas, docs/design/verification.md); a change to the upper
// triangle or an imaginary part left on the diagonal counts as a nonzero info.

#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/herk/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace H = ops::herk;

template <class T>
struct HerkProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using R = float_t<T>;
    Queue& q;
    Transpose trans;
    int n, k, batch, ar, ac, lda, ldc;
    std::size_t sa, sc;
    R alpha, beta;
    UnifiedVector<T> A, C0, C;

    HerkProblem(Queue& q_, Transpose t, int n_, int k_, int batch_, int pad)
        : q(q_), trans(t), n(n_), k(k_), batch(batch_), ar(t == Transpose::NoTrans ? n_ : k_),
          ac(t == Transpose::NoTrans ? k_ : n_), lda(ar + pad), ldc(n_ + pad), sa(std::size_t(lda) * std::size_t(ac)),
          sc(std::size_t(ldc) * std::size_t(n_)), alpha(R(1.25)), beta(R(0.75)), A(sa * std::size_t(batch_)),
          C0(sc * std::size_t(batch_)), C(sc * std::size_t(batch_)) {
        (void)view(A, ar, ac, lda, sa).fill_random(q, false, 7171);
        (void)view(C0, n, n, ldc, sc).fill_random(q, false, 7172);
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
    std::size_t workspace() {  // herk takes no caller workspace; a refused pin throws here
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)herk<kBackend, T>(q, view(A, ar, ac, lda, sa), view(C, n, n, ldc, sc), alpha, beta, Uplo::Lower, trans);
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
        const Transpose other = trans == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans;
        const MV a = view(A, ar, ac, lda, sa);
        return {MV::gemm_backward_error(q, a, trans, a, other, view(C0, n, n, ldc, sc), view(C, n, n, ldc, sc),
                                        MV::up(T(alpha)), MV::up(T(beta)), a.default_items(), Uplo::Lower),
                touched};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class HerkSpec final : public OpSpec {
public:
    std::string op() const override { return "herk"; }
    std::vector<std::string> key_names() const override { return {H::key_names.begin(), H::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_complex_dtype("herk", dtype, []<class T>() { return spellings(H::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"trans", {H::grid_trans.begin(), H::grid_trans.end()}}, {"n", ints(H::grid_n)},
                {"k", ints(H::grid_k)}, {"batch", ints(H::grid_batch)}};
    }
    double bytes(const std::string& dtype, const CellKey& key) const override {  // A, C0, C
        const double n = double(key_int(key, "n")), k = double(key_int(key, "k"));
        return (n * k + 2.0 * n * n) * double(key_int(key, "batch")) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: gram
            "src/ops/syrk/kernels/syrk_gram_tiles.hh",
            // family: triangular
            "src/ops/syrk/kernels/syrk_triangular_tiles.hh",
            // family: fold
            "src/ops/level3/batched.hh",
            "src/expansion_budget.hh",
            // common
            "src/kernels/common/triangular_tiles.hh",
            "src/ops/herk/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/herk/herk.cc"
        // herk fold calls the public gemm: its native kernels and can_run count too.
        // family: fold "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/kernels/gemm_launch.hh" "src/ops/gemm/kernels/gemm_rows.hh" "src/ops/gemm/kernels/gemm_shard.cc.in" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/herk_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_complex_dtype("herk", req.dtype, [&]<class T>() {
            const Transpose trans = *key_get(req.key, "trans") == "N" ? Transpose::NoTrans : Transpose::ConjTrans;
            HerkProblem<T> p(q, trans, int(key_int(req.key, "n")), int(key_int(req.key, "k")),
                             int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, p.k);
            if (req.mode == "race") return run_race<H::HerkChoice>("herk", p, req, tol);
            return run_arms<H::HerkChoice>("herk", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::HerkSpec)

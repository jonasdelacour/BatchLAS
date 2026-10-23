// hemm for the tuner (complex only): C = alpha A B + beta C with A Hermitian m x m read from its
// lower triangle (Side::Left), A, B and C0 from MatrixView::fill_random, A's strict upper triangle
// poisoned with a large finite value and its diagonal given an imaginary part hemm must ignore,
// alpha = (1.25, -0.5), beta = (0.75, 0.25) so C is read. side and uplo are not table keys: fixed
// at Left and L. Every ld is padded by --ld-pad. Verification is MatrixView::gemm_backward_error
// against the hermitized A on the default items (Check::blas, docs/design/verification.md).

#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/hemm/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <complex>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace HM = ops::hemm;

template <class T>
struct HemmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int m, n, batch, lda, ldb;
    std::size_t sa, sb;
    T alpha, beta;
    UnifiedVector<T> A, B, C0, C;

    HemmProblem(Queue& q_, int m_, int n_, int batch_, int pad)
        : q(q_), m(m_), n(n_), batch(batch_), lda(m_ + pad), ldb(m_ + pad), sa(std::size_t(lda) * std::size_t(m_)),
          sb(std::size_t(ldb) * std::size_t(n_)), alpha(T(1.25, -0.5)), beta(T(0.75, 0.25)),
          A(sa * std::size_t(batch_)), B(sb * std::size_t(batch_)), C0(sb * std::size_t(batch_)),
          C(sb * std::size_t(batch_)) {
        (void)view(A, m, m, lda, sa).fill_random(q, false, 5051);
        (void)view(B, m, n, ldb, sb).fill_random(q, false, 5052);
        (void)view(C0, m, n, ldb, sb).fill_random(q, false, 5053);
        q.wait();
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < m; ++c) {
                T* col = A.data() + std::size_t(b) * sa + std::size_t(c) * std::size_t(lda);
                col[c] = T(col[c].real(), 1e3);
                for (int r = 0; r < c; ++r) col[r] = T(1e6, -1e6);
            }
    }
    MV view(UnifiedVector<T>& v, int r, int c, int ld, std::size_t s) { return MV(v.data(), r, c, ld, int(s), batch); }
    void reset() {  // beta != 0 accumulates into C
        (void)MV::copy(q, view(C, m, n, ldb, sb), view(C0, m, n, ldb, sb));
        q.wait();
    }
    std::size_t workspace() {  // hemm takes no caller workspace; a refused pin throws here
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)hemm<kBackend, T>(q, view(A, m, m, lda, sa), view(B, m, n, ldb, sb), view(C, m, n, ldb, sb), alpha,
                                beta, Side::Left, Uplo::Lower);
        q.wait();
    }
    std::pair<double, int> verify() {
        const MV av = view(A, m, m, lda, sa);
        const auto items = av.default_items();
        auto a = av.template astype<T>(items);
        auto b = view(B, m, n, ldb, sb).template astype<T>(items), c0 = view(C0, m, n, ldb, sb).template astype<T>(items),
             c = view(C, m, n, ldb, sb).template astype<T>(items);
        (void)a.hermitize(q, Uplo::Lower);
        q.wait();
        const Transpose nt = Transpose::NoTrans;
        return {MV::gemm_backward_error(q, a, nt, b, nt, c0, c, MV::up(alpha), MV::up(beta), a.all_items()), 0};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class HemmSpec final : public OpSpec {
public:
    std::string op() const override { return "hemm"; }
    std::vector<std::string> key_names() const override { return {HM::key_names.begin(), HM::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_complex_dtype("hemm", dtype, []<class T>() { return spellings(HM::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"m", ints(HM::grid_mn)}, {"n", ints(HM::grid_mn)}, {"batch", ints(HM::grid_batch)}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A, B, C0, C
        const double m = double(key_int(k, "m")), n = double(key_int(k, "n"));
        return (m * m + 3.0 * m * n) * double(key_int(k, "batch")) * dtype_bytes(dtype);
    }
    std::string refine_key() const override { return "m"; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: expand
            "src/kernels/common/triangular_expand.hh",
            "src/ops/level3/batched.hh",
            "src/expansion_budget.hh",
            // common
            "src/ops/hemm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/hemm/hemm.cc"
        // hemm expand calls the public gemm: its native kernels and can_run count too.
        // family: expand "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/hemm_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_complex_dtype("hemm", req.dtype, [&]<class T>() {
            HemmProblem<T> p(q, int(key_int(req.key, "m")), int(key_int(req.key, "n")), int(key_int(req.key, "batch")),
                             req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, p.m);
            if (req.mode == "race") return run_race<HM::HemmChoice>("hemm", p, req, tol);
            return run_arms<HM::HemmChoice>("hemm", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::HemmSpec)

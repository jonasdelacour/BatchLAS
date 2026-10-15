// symm for the tuner (real only): C = alpha A B + beta C, A symmetric m x m held in its lower
// triangle (MatrixView::fill_random, the strict upper triangle then poisoned with a large finite
// value), B and C0 fill_random, beta = 0.75 so C is read. side and uplo are not table keys: fixed at
// L and L. Every ld is padded by --ld-pad. Verification is MatrixView::gemm_backward_error against
// A's mirrored lower triangle (symmetrize on a packed copy of the default items),
// |C - Cref| / (|alpha| |A| |B| + |beta| |C0|) componentwise (Check::blas,
// docs/design/verification.md). expand's times include the public gemm it calls.
//
// form is derived from (m, n) (choice.hh form_of), so the grid is the lattice's consistent cells
// only: a cell whose extents contradict its form cannot be timed. Refinement keeps form and the
// other extent fixed, and the extents of one form along one axis are an interval, so a midpoint
// stays consistent.

#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/symm/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace Y = ops::symm;

template <class T>
struct SymmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int m, n, batch, lda, ldb;
    std::size_t sa, sb;
    T alpha, beta;
    UnifiedVector<T> A, B, C0, C;

    SymmProblem(Queue& q_, int m_, int n_, int batch_, int pad)
        : q(q_), m(m_), n(n_), batch(batch_), lda(m_ + pad), ldb(m_ + pad), sa(std::size_t(lda) * std::size_t(m_)),
          sb(std::size_t(ldb) * std::size_t(n_)), alpha(T(1.25)), beta(T(0.75)), A(sa * std::size_t(batch_)),
          B(sb * std::size_t(batch_)), C0(sb * std::size_t(batch_)), C(sb * std::size_t(batch_)) {
        (void)view(A, m, m, lda, sa).fill_random(q, false, 6061);
        (void)view(B, m, n, ldb, sb).fill_random(q, false, 6062);
        (void)view(C0, m, n, ldb, sb).fill_random(q, false, 6063);
        q.wait();
        for (int b = 0; b < batch; ++b)
            for (int c = 1; c < m; ++c)
                for (int r = 0; r < c; ++r) A[std::size_t(b) * sa + std::size_t(c) * std::size_t(lda) + std::size_t(r)] = T(1e6);
    }
    MV view(UnifiedVector<T>& v, int r, int c, int ld, std::size_t s) { return MV(v.data(), r, c, ld, int(s), batch); }
    void reset() {  // beta != 0 accumulates into C
        (void)MV::copy(q, view(C, m, n, ldb, sb), view(C0, m, n, ldb, sb));
        q.wait();
    }
    // symm takes no caller workspace; probing a pin means running once (a refused pin throws first).
    std::size_t workspace() {
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)symm<kBackend, T>(q, view(A, m, m, lda, sa), view(B, m, n, ldb, sb), view(C, m, n, ldb, sb), alpha,
                                beta, Side::Left, Uplo::Lower);
        q.wait();
    }
    std::pair<double, int> verify() {
        const MV av = view(A, m, m, lda, sa);
        const auto items = av.default_items();
        auto a = av.template astype<T>(items);
        auto b = view(B, m, n, ldb, sb).template astype<T>(items), c0 = view(C0, m, n, ldb, sb).template astype<T>(items),
             c = view(C, m, n, ldb, sb).template astype<T>(items);
        (void)a.symmetrize(q, Uplo::Lower);
        q.wait();
        const Transpose nt = Transpose::NoTrans;
        return {MV::gemm_backward_error(q, a, nt, b, nt, c0, c, MV::up(alpha), MV::up(beta),
                                        a.all_items()),
                0};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class SymmSpec final : public OpSpec {
public:
    std::string op() const override { return "symm"; }
    std::vector<std::string> key_names() const override { return {Y::key_names.begin(), Y::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_real_dtype("symm", dtype, []<class T>() { return spellings(Y::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"form", {Y::grid_form.begin(), Y::grid_form.end()}},
                {"m", ints(Y::grid_mn)},
                {"n", ints(Y::grid_mn)},
                {"batch", ints(Y::grid_batch)}};
    }
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        static_cast<void>(dtype);
        std::vector<CellKey> cells;
        for (int m : Y::grid_mn)
            for (int n : Y::grid_mn)
                for (int b : Y::grid_batch)
                    cells.push_back({{"form", std::string(Y::form_of(m, n))}, {"m", std::to_string(m)},
                                     {"n", std::to_string(n)}, {"batch", std::to_string(b)}});
        return filter_cells(*this, std::move(cells), overrides);
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
            "src/expansion_budget.hh",
            // common
            "src/ops/symm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/symm/symm.cc"
        // symm expand calls the public gemm: its native kernels and can_run count too.
        // family: expand "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh" "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/symm_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_real_dtype("symm", req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const int m = int(key_int(req.key, "m")), n = int(key_int(req.key, "n"));
            if (*key_get(req.key, "form") != Y::form_of(m, n))
                throw std::invalid_argument("symm: form " + *key_get(req.key, "form") + " contradicts m, n");
            SymmProblem<T> p(*q, m, n, int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, p.m);
            if (req.mode == "race") return run_race<Y::SymmChoice>("symm", p, req, tol);
            return run_arms<Y::SymmChoice>("symm", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SymmSpec)

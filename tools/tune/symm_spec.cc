// symm for the tuner (real only): C = alpha A B + beta C, A symmetric m x m held in its lower
// triangle (the strict upper triangle poisoned with a large finite value), B and C0 random,
// beta = 0.75 so C is read. side and uplo are not table keys: fixed at L and L. Every ld is
// padded by --ld-pad. Verification is the componentwise error |C - Cref| / (|alpha| |A| |B| +
// |beta| |C0|) in double on items 0 and batch-1, over up to 256 sampled rows of up to 64 sampled
// columns. expand's times include the public gemm it calls.
//
// form is derived from (m, n) (choice.hh form_of), so the grid is the lattice's consistent cells
// only: a cell whose extents contradict its form cannot be timed. Refinement keeps form and the
// other extent fixed, and the extents of one form along one axis are an interval, so a midpoint
// stays consistent.

#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/symm/choice.hh"
#include "cell_runner.hh"
#include "residuals.hh"

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
        Rng rg(6061);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < m; ++c)
                for (int r = 0; r < m; ++r)
                    A[std::size_t(b) * sa + std::size_t(c) * std::size_t(lda) + std::size_t(r)] =
                        r >= c ? T(rg.next()) : T(1e6);
        for (auto* v : {&B, &C0})
            for (std::size_t i = 0; i < v->size(); ++i) (*v)[i] = T(rg.next());
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
        double worst = 0;
        for (int b : {0, batch - 1}) {
            const std::size_t oa = std::size_t(b) * sa, ob = std::size_t(b) * sb;
            auto a = [&](int r, int c) {  // the lower triangle, mirrored
                return double(r >= c ? A[oa + std::size_t(c) * std::size_t(lda) + std::size_t(r)]
                                     : A[oa + std::size_t(r) * std::size_t(lda) + std::size_t(c)]);
            };
            auto at = [&](const UnifiedVector<T>& v, int r, int c) {
                return double(v[ob + std::size_t(c) * std::size_t(ldb) + std::size_t(r)]);
            };
            for (int j : sample_indices(n, 64))
                for (int i : sample_indices(m, 256)) {
                    double acc = 0, mag = 0;
                    for (int k = 0; k < m; ++k) {
                        const double t = a(i, k) * at(B, k, j);
                        acc += t;
                        mag += ab(t);
                    }
                    const double c0 = at(C0, i, j);
                    const double num = ab(at(C, i, j) - (double(alpha) * acc + double(beta) * c0)),
                                 den = ab(double(alpha)) * mag + ab(double(beta)) * ab(c0);
                    worst = nanmax(worst, den > 0 ? num / den : num);
                }
        }
        return {worst, 0};
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
            // family: expand
            "src/backends/triangular_expand.hh",
            "src/expansion_budget.hh",
            // common
            "src/ops/symm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/symm/symm.cc"
        // family: expand "src/backends/route_common.hh"
        // symm expand calls the public gemm: its native kernels and can_run count too.
        // family: expand "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/symm_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<Y::SymmChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_real_dtype("symm", req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const int m = int(key_int(req.key, "m")), n = int(key_int(req.key, "n"));
            if (*key_get(req.key, "form") != Y::form_of(m, n))
                throw std::invalid_argument("symm: form " + *key_get(req.key, "form") + " contradicts m, n");
            SymmProblem<T> p(*q, m, n, int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<Y::SymmChoice>("symm", p, req, Tol<T>::v);
            return run_arms<Y::SymmChoice>("symm", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SymmSpec)

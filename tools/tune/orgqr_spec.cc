// orgqr for the tuner (src/ops/orgqr/choice.hh): reflectors and tau from host_reflectors (larfg on
// random columns, never a tuned geqrf), the diagonal and above holding values orgqr must ignore,
// ld = m + ld_pad. Verified on items 0 and batch-1 over sampled columns: Q against H_0 ... H_{n-1}
// e_j applied on the host from the input reflectors, and Q^H Q - I (qr_common.hh).
//
// The key has no batch: a cell runs at qr_batch (qr_common.hh). The grid is the m x n lattice with
// n <= m (Q's columns live in C^m). blocked's time includes the public ormqr it applies to an
// identity, and that ormqr's own gemm and trmm, each choosing for itself.

#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/orgqr/choice.hh"
#include "cell_runner.hh"
#include "qr_common.hh"
#include "residuals.hh"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

constexpr double kCheckBudget = 4e8;  // host multiply-adds per verified item

template <class T>
struct OrgqrProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using D = typename Prom<T>::type;
    Queue& q;
    int m, n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, A, tau;
    UnifiedVector<T*> pA0, pA;
    MV A0v, Av;

    OrgqrProblem(Queue& q_, int m_, int n_, int batch_, int ld_pad)
        : q(q_), m(m_), n(n_), batch(batch_), ld(m_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)), tau(std::size_t(n_) * std::size_t(batch_)),
          pA0(std::size_t(batch_)), pA(std::size_t(batch_)), A0v(A0.data(), m, n, ld, int(stride), batch, pA0.data()),
          Av(A.data(), m, n, ld, int(stride), batch, pA.data()) {
        host_reflectors<T>(A0.data(), tau.data(), m, n, ld, stride, batch, 5151);
    }
    std::size_t workspace() { return orgqr_buffer_size<kBackend, T>(q, Av, tau.to_span()); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {}
    void run(Span<std::byte> ws) {
        (void)orgqr<kBackend, T>(q, Av, tau.to_span(), ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        const auto cols = sample_idx(n, int(std::clamp(kCheckBudget / (double(m) * double(n)), 2.0, 32.0)));
        double worst = 0;
        for (int b : {0, batch - 1}) {
            const std::size_t o = std::size_t(b) * stride;
            const Reflectors<T> R{A0.data(), tau.data(), o, std::size_t(b) * std::size_t(n), ld, m};
            auto at = [&](int r, int c) { return up(A[o + std::size_t(c) * std::size_t(ld) + std::size_t(r)]); };
            double diff = 0, orth = 0;
            for (int j : cols) {
                const auto ref = q_column<T>(R, n, j);
                for (int r = 0; r < m; ++r) {
                    const double d = ab(at(r, j) - ref[std::size_t(r)]);
                    diff += d * d;
                }
                for (int i : cols) {
                    D g = D(i == j ? -1 : 0);
                    for (int r = 0; r < m; ++r) g += cj(at(r, i)) * at(r, j);
                    orth += ab(g) * ab(g);
                }
            }
            if (std::isnan(diff) || std::isnan(orth)) return {std::nan(""), 0};
            worst = nanmax(worst, std::sqrt(std::max(diff, orth) / double(cols.size())));
        }
        return {worst, 0};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class OrgqrSpec final : public OpSpec {
public:
    std::string op() const override { return "orgqr"; }
    std::vector<std::string> key_names() const override {
        return {ops::orgqr::key_names.begin(), ops::orgqr::key_names.end()};
    }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(ops::orgqr::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"m", ints(ops::orgqr::grid)}, {"n", ints(ops::orgqr::grid)}};
    }
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        auto cells = OpSpec::grid(dtype, overrides);
        std::erase_if(cells, [](const CellKey& k) { return key_int(k, "n") > key_int(k, "m"); });
        if (cells.empty()) throw std::invalid_argument("orgqr: the --grid values leave no cell with n <= m");
        return cells;
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A0, A (tau is n per item)
        const double item = double(key_int(k, "m")) * double(key_int(k, "n")) * dtype_bytes(dtype);
        return 2.0 * item * double(qr_batch(item));
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            // family: blocked
            "src/extensions/orgqr_blocked.cc",
            // common
            "src/extensions/orgqr_native.hh",
            "src/ops/orgqr/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/orgqr/orgqr.cc"
        // orgqr blocked applies the public ormqr to an identity: ormqr's kernels and selection count, and
        // through ormqr blocked the public gemm and trmm too.
        // family: blocked "src/extensions/ormqr_blocked.cc" "include/batchlas/internal/ormqr_blocked.hh" "src/extensions/larft_wy.hh" "src/ops/ormqr/choice.hh" "src/ops/ormqr/ormqr.cc"
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh"
        // family: blocked "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // family: blocked "src/ops/trmm/choice.hh" "src/ops/trmm/trmm.cc" "src/backends/trmm_triangular_tiles.hh" "src/backends/triangular_tiles.hh" "src/backends/triangular_expand.hh" "src/backends/level3_shape.hh" "src/expansion_budget.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/orgqr_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<ops::orgqr::OrgqrChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const int m = int(key_int(req.key, "m")), n = int(key_int(req.key, "n"));
            if (n > m) throw std::invalid_argument("orgqr: n > m has no native family to tune");
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            OrgqrProblem<T> p(*q, m, n, qr_batch(double(m) * double(n) * sizeof(T)), req.ld_pad);
            if (req.mode == "race") return run_race<ops::orgqr::OrgqrChoice>("orgqr", p, req, qr_tol<T>(m));
            return run_arms<ops::orgqr::OrgqrChoice>("orgqr", p, req, qr_tol<T>(m));
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::OrgqrSpec)

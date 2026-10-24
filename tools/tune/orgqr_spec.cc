// SPDX-License-Identifier: MIT
// orgqr for the tuner (src/ops/orgqr/choice.hh): reflectors and tau from one untimed public geqrf
// (Auto) of a MatrixView::fill_random M, R on and above the diagonal (values orgqr must ignore),
// ld = m + ld_pad. Verified on the default items by MatrixView::qr_reconstruction, ||M - Q R||_F /
// ||M||_F with R from the geqrf factor (factorization), and MatrixView::orthogonality (orthogonality),
// each as a fraction of its bound at n = m (docs/design/verification.md), so the cell's residual is
// the worse of the two and its tolerance 1. No orgqr path forms M's reflectors, so the check is not
// self-referential (agent guide §8 rule 7).
// The key has no batch: a cell runs at qr_batch (spec.hh). The grid is the m x n lattice with
// n <= m (Q's columns live in C^m). blocked's time includes the public ormqr it applies to an
// identity, and that ormqr's own gemm and trmm, each choosing for itself.

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/orgqr/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

template <typename T>
struct OrgqrProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int m, n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> M, A0, A, tau;
    UnifiedVector<T*> pM, pA0, pA;
    MV Mv, A0v, Av;

    OrgqrProblem(Queue& q_, int m_, int n_, int batch_, int ld_pad)
        : q(q_), m(m_), n(n_), batch(batch_), ld(m_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          M(stride * std::size_t(batch_)), A0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)),
          tau(std::size_t(n_) * std::size_t(batch_)), pM(std::size_t(batch_)), pA0(std::size_t(batch_)),
          pA(std::size_t(batch_)), Mv(M.data(), m, n, ld, int(stride), batch, pM.data()),
          A0v(A0.data(), m, n, ld, int(stride), batch, pA0.data()), Av(A.data(), m, n, ld, int(stride), batch, pA.data()) {
        (void)Mv.fill_random(q, false, 5151);
        (void)MV::copy(q, A0v, Mv);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, geqrf_buffer_size<kBackend, T>(q, A0v, tau.to_span())));
        (void)geqrf<kBackend, T>(q, A0v, tau.to_span(), ws.to_span());
        q.wait();
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
        const double rec = MV::qr_reconstruction(q, Mv, Av, A0v) / verify::bound<T>(verify::Check::factorization, m);
        return {verify::nanmax(rec, MV::orthogonality(q, Av) / verify::bound<T>(verify::Check::orthogonality, m)), 0};
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
        return with_dtype(dtype, []<typename T>() { return spellings(ops::orgqr::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"m", ints(ops::orgqr::grid)}, {"n", ints(ops::orgqr::grid)}};
    }
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        auto cells = OpSpec::grid(dtype, overrides);
        std::erase_if(cells, [](const CellKey& k) { return key_int(k, "n") > key_int(k, "m"); });
        if (cells.empty()) throw batchlas::invalid_argument("orgqr: the --grid values leave no cell with n <= m");
        return cells;
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A0, A (tau is n per item; M is the check's)
        const double item = double(key_int(k, "m")) * double(key_int(k, "n")) * dtype_bytes(dtype);
        return 2.0 * item * double(qr_batch(item));
    }
    FrontierPlan frontier() const override { return {1}; }  // p99.9 fails at every k. evidence: docs/design/tiered-tuning.md#engine-round-planning-by-flip-frontier
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            // family: blocked
            "src/ops/orgqr/kernels/orgqr_blocked.cc",
            // common
            "src/ops/orgqr/orgqr_native.hh",
            "src/ops/orgqr/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/orgqr/orgqr.cc"
        // blocked's batch chunk comes from tuning_params.hh (blocked_chunk_items).
        // family: blocked "include/batchlas/tuning_params.hh"
        // orgqr blocked applies the public ormqr to an identity: ormqr's kernels and selection count, and
        // through ormqr blocked the public gemm and trmm too.
        // family: blocked "src/ops/ormqr/kernels/ormqr_blocked.cc" "include/batchlas/internal/ormqr_blocked.hh" "src/kernels/common/larft_wy.hh" "src/ops/ormqr/choice.hh" "src/ops/ormqr/ormqr.cc"
        // family: blocked "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/kernels/gemm_launch.hh" "src/ops/gemm/kernels/gemm_rows.hh" "src/ops/gemm/kernels/gemm_shard.cc.in" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/register_256x128.hh" "src/ops/gemm/kernels/copy_async.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh"
        // family: blocked "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // family: blocked "src/ops/trmm/choice.hh" "src/ops/trmm/trmm.cc" "src/ops/trmm/kernels/trmm_triangular_tiles.hh" "src/kernels/common/triangular_tiles.hh" "src/kernels/common/triangular_expand.hh" "src/backends/level3_shape.hh" "src/expansion_budget.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/orgqr_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<typename T>() {
            const int m = int(key_int(req.key, "m")), n = int(key_int(req.key, "n"));
            if (n > m) throw batchlas::invalid_argument("orgqr: n > m has no native family to tune");
            OrgqrProblem<T> p(q, m, n, qr_batch(double(m) * double(n) * sizeof(T)), req.ld_pad);
            if (req.mode == "race") return run_race<ops::orgqr::OrgqrChoice>("orgqr", p, req, 1.0);
            return run_arms<ops::orgqr::OrgqrChoice>("orgqr", p, req, 1.0);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::OrgqrSpec)

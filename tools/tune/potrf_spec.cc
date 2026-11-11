// SPDX-License-Identifier: MIT
// potrf for the tuner (docs/design/flat-kernel-selection.md §6.1): SPD inputs, the MatrixView::potrf_residual
// factorization residual (docs/design/verification.md), every candidates<T>() entry pinned through
// the public potrf.

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/potrf/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace P = ops::potrf;

template <typename T>
struct PotrfProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, batch, ld;
    std::size_t stride;
    Uplo uplo;
    UnifiedVector<T> A0, A;
    UnifiedVector<T*> pA0, pA;
    MV A0v, Av;
    UnifiedVector<std::int32_t> info;

    PotrfProblem(Queue& q_, int n_, int batch_, int ld_pad, Uplo u)
        : q(q_), n(n_), batch(batch_), ld(n_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)), uplo(u),
          A0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)), pA0(std::size_t(batch_)),
          pA(std::size_t(batch_)), A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()),
          Av(A.data(), n, n, ld, int(stride), batch, pA.data()), info(std::size_t(batch_), 0) {
        (void)A0v.fill_random_spd(q, 12345);
        q.wait();
    }
    std::size_t workspace() { return potrf_buffer_size<kBackend, T>(q, Av, uplo); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)potrf<kBackend, T>(q, Av, uplo, ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        return {MV::potrf_residual(q, A0v, Av, uplo), bad};
    }
};

class PotrfSpec final : public OpSpec {
public:
    std::string op() const override { return "potrf"; }
    std::vector<std::string> key_names() const override { return {P::key_names.begin(), P::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<typename T>() { return spellings(P::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, b;
        for (int v : P::grid_n) n.push_back(std::to_string(v));
        for (int v : P::grid_batch) b.push_back(std::to_string(v));
        return {{"uplo", {"L", "U"}}, {"n", n}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "include/batchlas/device/detail/kernel_attrs.hh",
            "include/batchlas/device/detail/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            "include/batchlas/device/detail/sg_partition/backend_amdgcn.hh",
            "include/batchlas/device/detail/sg_partition/backend_amdgcn_core.hh",
            "include/batchlas/device/detail/sg_partition/backend_generic.hh",
            "include/batchlas/device/detail/sg_partition/backend_nvptx.hh",
            "include/batchlas/device/detail/sg_partition/backend_spirv.hh",
            "include/batchlas/device/detail/sg_partition/backend_sscp.hh",
            "include/batchlas/device/detail/sg_partition/sg_partition.hh",
            // family: tiny
            "src/ops/potrf/kernels/potrf_tiny.cc",
            // family: cta
            "src/ops/potrf/kernels/potrf_cta.cc",
            // family: lpanel
            "src/ops/potrf/kernels/potrf_lpanel.cc",
            "src/ops/potrf/kernels/potrf_lpanel_device.hh",
            // family: blocked
            "src/ops/potrf/kernels/potrf_blocked.cc",
            // common
            "src/ops/potrf/potrf_native.hh",
            "src/util/resident_capacity.hh",
            "src/ops/potrf/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/potrf/potrf.cc"
        // family: tiny "src/kernels/common/tiny_device.hh"
        // family: blocked "src/ops/potrf/kernels/symmetric_product_fold.hh" "src/ops/trsm/trsm_native.hh"
        // family: cta "include/batchlas/device/arith.hh"
        // family: lpanel "include/batchlas/device/arith.hh"
        // blocked's batch chunk comes from tuning_params.hh (blocked_chunk_items).
        // family: blocked "include/batchlas/tuning_params.hh"
        // The include/batchlas/device/ headers each family's kernels include.
        // family: blocked "include/batchlas/device/arith.hh" "include/batchlas/device/core.hh"
        // family: cta "include/batchlas/device/core.hh" "include/batchlas/device/exec.hh"
        // family: cta "include/batchlas/device/interop.hh" "include/batchlas/device/lapack/potrf.hh"
        // family: cta "include/batchlas/device/lapack_aux.hh" "include/batchlas/device/level1.hh"
        // family: cta "include/batchlas/device/level3/cfg.hh" "include/batchlas/device/level3/copy_async.hh"
        // family: cta "include/batchlas/device/level3/epilogue.hh" "include/batchlas/device/level3/group.hh"
        // family: cta "include/batchlas/device/level3/mma.hh" "include/batchlas/device/level3/packet.hh"
        // family: cta "include/batchlas/device/level3/pipeline.hh" "include/batchlas/device/level3/schedule.hh"
        // family: cta "include/batchlas/device/level3/stage.hh" "include/batchlas/device/operand.hh"
        // family: cta "include/batchlas/device/reduce.hh" "include/batchlas/device/scalar.hh"
        // family: cta "include/batchlas/device/tier.hh" "include/batchlas/device/view.hh"
        // family: cta "include/batchlas/device/workspace.hh"
        // family: lpanel "include/batchlas/device/core.hh" "include/batchlas/device/exec.hh"
        // family: lpanel "include/batchlas/device/interop.hh" "include/batchlas/device/lapack/potrf.hh"
        // family: lpanel "include/batchlas/device/lapack_aux.hh" "include/batchlas/device/level1.hh"
        // family: lpanel "include/batchlas/device/level3/cfg.hh" "include/batchlas/device/level3/copy_async.hh"
        // family: lpanel "include/batchlas/device/level3/epilogue.hh" "include/batchlas/device/level3/group.hh"
        // family: lpanel "include/batchlas/device/level3/mma.hh" "include/batchlas/device/level3/packet.hh"
        // family: lpanel "include/batchlas/device/level3/pipeline.hh" "include/batchlas/device/level3/schedule.hh"
        // family: lpanel "include/batchlas/device/level3/stage.hh" "include/batchlas/device/operand.hh"
        // family: lpanel "include/batchlas/device/reduce.hh" "include/batchlas/device/scalar.hh"
        // family: lpanel "include/batchlas/device/tier.hh" "include/batchlas/device/view.hh"
        // family: lpanel "include/batchlas/device/workspace.hh"
        // family: tiny "include/batchlas/device/arith.hh" "include/batchlas/device/core.hh"
        // family: tiny "include/batchlas/device/exec.hh" "include/batchlas/device/interop.hh"
        // family: tiny "include/batchlas/device/lapack/potrf.hh" "include/batchlas/device/lapack_aux.hh"
        // family: tiny "include/batchlas/device/level1.hh" "include/batchlas/device/level3/cfg.hh"
        // family: tiny "include/batchlas/device/level3/copy_async.hh" "include/batchlas/device/level3/epilogue.hh"
        // family: tiny "include/batchlas/device/level3/group.hh" "include/batchlas/device/level3/mma.hh"
        // family: tiny "include/batchlas/device/level3/packet.hh" "include/batchlas/device/level3/pipeline.hh"
        // family: tiny "include/batchlas/device/level3/schedule.hh" "include/batchlas/device/level3/stage.hh"
        // family: tiny "include/batchlas/device/operand.hh" "include/batchlas/device/reduce.hh"
        // family: tiny "include/batchlas/device/scalar.hh" "include/batchlas/device/sg_partition.hh"
        // family: tiny "include/batchlas/device/tier.hh" "include/batchlas/device/view.hh"
        // family: tiny "include/batchlas/device/workspace.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/potrf_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<typename T>() {
            const Uplo uplo = *key_get(req.key, "uplo") == "U" ? Uplo::Upper : Uplo::Lower;
            PotrfProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad, uplo);
            const double tol = batchlas::verify::bound<T>(batchlas::verify::Check::factorization, p.n);
            if (req.mode == "race") return run_race<P::PotrfChoice>("potrf", p, req, tol);
            return run_arms<P::PotrfChoice>("potrf", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::PotrfSpec)

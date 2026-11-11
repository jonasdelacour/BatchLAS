// SPDX-License-Identifier: MIT
// posv for the tuner (flat-kernel-selection.md#phase-3-posv): SPD A, random B, the
// MatrixView::solve_residual (docs/design/verification.md). posv's cta and blocked times
// include the public potrf and trsm the composition calls, each choosing for itself (#phase-3-posv).

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/posv/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::posv;

template <typename T>
struct PosvProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, nrhs, batch, lda, ldb;
    std::size_t sa, sb;
    Uplo uplo;
    UnifiedVector<T> A0, A, B0, X;
    UnifiedVector<T*> pA0, pA, pB0, pX;
    MV A0v, Av, B0v, Xv;
    UnifiedVector<std::int32_t> info;

    PosvProblem(Queue& q_, int n_, int nrhs_, int batch_, int ld_pad, Uplo u)
        : q(q_), n(n_), nrhs(nrhs_), batch(batch_), lda(n_ + ld_pad), ldb(n_ + ld_pad),
          sa(std::size_t(lda) * std::size_t(n_)), sb(std::size_t(ldb) * std::size_t(nrhs_)), uplo(u),
          A0(sa * std::size_t(batch_)), A(sa * std::size_t(batch_)), B0(sb * std::size_t(batch_)),
          X(sb * std::size_t(batch_)), pA0(std::size_t(batch_)), pA(std::size_t(batch_)), pB0(std::size_t(batch_)),
          pX(std::size_t(batch_)), A0v(A0.data(), n, n, lda, int(sa), batch, pA0.data()),
          Av(A.data(), n, n, lda, int(sa), batch, pA.data()), B0v(B0.data(), n, nrhs, ldb, int(sb), batch, pB0.data()),
          Xv(X.data(), n, nrhs, ldb, int(sb), batch, pX.data()), info(std::size_t(batch_), 0) {
        (void)A0v.fill_random_spd(q, 12345);
        (void)B0v.fill_random(q, false, 777);
        q.wait();
    }
    std::size_t workspace() { return posv_buffer_size<kBackend, T>(q, Av, Xv, uplo); }
    void reset() {  // the solve overwrites A with the factor and B with X
        (void)MV::copy(q, Av, A0v);
        (void)MV::copy(q, Xv, B0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)posv<kBackend, T>(q, Av, Xv, uplo, ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        return {MV::solve_residual(q, A0v, Xv, B0v), bad};
    }
};

class PosvSpec final : public OpSpec {
public:
    std::string op() const override { return "posv"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<typename T>() { return spellings(S::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, r, b;
        for (int v : S::grid_n) n.push_back(std::to_string(v));
        for (int v : S::grid_nrhs) r.push_back(std::to_string(v));
        for (int v : S::grid_batch) b.push_back(std::to_string(v));
        return {{"uplo", {"L", "U"}}, {"n", n}, {"nrhs", r}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return (2.0 * n * n + 2.0 * n * double(key_int(k, "nrhs"))) * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, A, B0, X
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
            "src/ops/posv/kernels/posv_tiny.cc",
            "src/ops/posv/posv_native.hh",
            // family: cta
            "src/ops/getrs/kernels/getrs_fused.cc",
            "src/ops/getrs/getrs_native.hh",
            // common
            "src/ops/posv/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/posv/posv.cc"
        // family: tiny "src/kernels/common/tiny_device.hh" "src/ops/potrf/potrf_native.hh"
        // posv cta and blocked call the public potrf and trsm: their native kernels count too.
        // family: cta "src/ops/potrf/kernels/potrf_tiny.cc" "src/ops/potrf/kernels/potrf_cta.cc" "include/batchlas/device/lapack/potrf.hh" "src/ops/potrf/kernels/potrf_lpanel.cc" "src/ops/potrf/kernels/potrf_lpanel_device.hh" "src/ops/potrf/kernels/potrf_blocked.cc" "src/ops/potrf/potrf_native.hh" "src/util/resident_capacity.hh" "src/ops/potrf/choice.hh"
        // family: cta "include/batchlas/device/arith.hh"
        // family: cta "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // family: blocked "src/ops/potrf/kernels/potrf_tiny.cc" "src/ops/potrf/kernels/potrf_cta.cc" "include/batchlas/device/lapack/potrf.hh" "src/ops/potrf/kernels/potrf_lpanel.cc" "src/ops/potrf/kernels/potrf_lpanel_device.hh" "src/ops/potrf/kernels/potrf_blocked.cc" "src/ops/potrf/potrf_native.hh" "src/util/resident_capacity.hh" "src/ops/potrf/choice.hh"
        // family: blocked "include/batchlas/device/arith.hh"
        // family: blocked "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // The include/batchlas/device/ headers each family's kernels include.
        // family: blocked "include/batchlas/device.hh" "include/batchlas/device/core.hh"
        // family: blocked "include/batchlas/device/exec.hh" "include/batchlas/device/grid.hh"
        // family: blocked "include/batchlas/device/interop.hh" "include/batchlas/device/lapack/potrf.hh"
        // family: blocked "include/batchlas/device/lapack_aux.hh" "include/batchlas/device/level1.hh"
        // family: blocked "include/batchlas/device/level2.hh" "include/batchlas/device/level3.hh"
        // family: blocked "include/batchlas/device/level3/cfg.hh" "include/batchlas/device/level3/copy_async.hh"
        // family: blocked "include/batchlas/device/level3/epilogue.hh" "include/batchlas/device/level3/group.hh"
        // family: blocked "include/batchlas/device/level3/mma.hh" "include/batchlas/device/level3/packet.hh"
        // family: blocked "include/batchlas/device/level3/pipeline.hh" "include/batchlas/device/level3/schedule.hh"
        // family: blocked "include/batchlas/device/level3/stage.hh" "include/batchlas/device/operand.hh"
        // family: blocked "include/batchlas/device/reduce.hh" "include/batchlas/device/scalar.hh"
        // family: blocked "include/batchlas/device/sg_partition.hh" "include/batchlas/device/tier.hh"
        // family: blocked "include/batchlas/device/view.hh" "include/batchlas/device/workspace.hh"
        // family: cta "include/batchlas/device.hh" "include/batchlas/device/core.hh" "include/batchlas/device/exec.hh"
        // family: cta "include/batchlas/device/grid.hh" "include/batchlas/device/interop.hh"
        // family: cta "include/batchlas/device/lapack/getrs.hh" "include/batchlas/device/lapack/potrf.hh"
        // family: cta "include/batchlas/device/lapack_aux.hh" "include/batchlas/device/level1.hh"
        // family: cta "include/batchlas/device/level2.hh" "include/batchlas/device/level3.hh"
        // family: cta "include/batchlas/device/level3/cfg.hh" "include/batchlas/device/level3/copy_async.hh"
        // family: cta "include/batchlas/device/level3/epilogue.hh" "include/batchlas/device/level3/group.hh"
        // family: cta "include/batchlas/device/level3/mma.hh" "include/batchlas/device/level3/packet.hh"
        // family: cta "include/batchlas/device/level3/pipeline.hh" "include/batchlas/device/level3/schedule.hh"
        // family: cta "include/batchlas/device/level3/stage.hh" "include/batchlas/device/operand.hh"
        // family: cta "include/batchlas/device/reduce.hh" "include/batchlas/device/scalar.hh"
        // family: cta "include/batchlas/device/sg_partition.hh" "include/batchlas/device/tier.hh"
        // family: cta "include/batchlas/device/view.hh" "include/batchlas/device/workspace.hh"
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
    std::string spec_file() const override { return "tools/tune/posv_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<typename T>() {
            const Uplo uplo = *key_get(req.key, "uplo") == "U" ? Uplo::Upper : Uplo::Lower;
            PosvProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "nrhs")),
                             int(key_int(req.key, "batch")), req.ld_pad, uplo);
            const double tol = batchlas::verify::bound<T>(batchlas::verify::Check::solve, p.n);
            if (req.mode == "race") return run_race<S::PosvChoice>("posv", p, req, tol);
            return run_arms<S::PosvChoice>("posv", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::PosvSpec)

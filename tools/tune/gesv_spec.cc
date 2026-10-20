// gesv for the tuner (docs/design/flat-kernel-selection.md §6.1): MatrixView::fill_random_pivoted A,
// fill_random B, verified on the default items by MatrixView::solve_residual against the unfactored A
// (Check::solve, docs/design/verification.md). blocked's time
// includes the public getrf and getrs it composes, each choosing for itself (and through them
// trsm and gemm), so gesv is tuned after those ops.
//
// The table has no batch key (choice.hh: the router reads n and nrhs only), so every cell runs at
// gesv_batch(): saturating, 16384 items at small n, shrinking so A stays near 1 GiB.

#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/gesv/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace G = ops::gesv;

int gesv_batch(const std::string& dtype, int n) {
    const double per = double(n) * double(n) * dtype_bytes(dtype);
    int b = 16384;
    while (b > 1 && per * double(b) > double(1 << 30)) b /= 2;
    return b;
}

template <class T>
struct GesvProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, nrhs, batch, lda, ldb;
    std::size_t sa, sb;
    UnifiedVector<T> A0, A, B0, X;
    UnifiedVector<T*> pA0, pA, pB0, pX;
    MV A0v, Av, B0v, Xv;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<std::int32_t> info;

    GesvProblem(Queue& q_, int n_, int nrhs_, int batch_, int ld_pad)
        : q(q_), n(n_), nrhs(nrhs_), batch(batch_), lda(n_ + ld_pad), ldb(n_ + ld_pad),
          sa(std::size_t(lda) * std::size_t(n_)), sb(std::size_t(ldb) * std::size_t(nrhs_)),
          A0(sa * std::size_t(batch_)), A(sa * std::size_t(batch_)), B0(sb * std::size_t(batch_)),
          X(sb * std::size_t(batch_)), pA0(std::size_t(batch_)), pA(std::size_t(batch_)), pB0(std::size_t(batch_)),
          pX(std::size_t(batch_)), A0v(A0.data(), n, n, lda, int(sa), batch, pA0.data()),
          Av(A.data(), n, n, lda, int(sa), batch, pA.data()), B0v(B0.data(), n, nrhs, ldb, int(sb), batch, pB0.data()),
          Xv(X.data(), n, nrhs, ldb, int(sb), batch, pX.data()), piv(std::size_t(n_) * std::size_t(batch_), 0),
          info(std::size_t(batch_), 0) {
        (void)A0v.fill_random_pivoted(q, 12345);
        (void)B0v.fill_random(q, false, 777);
        q.wait();
    }
    std::size_t workspace() { return gesv_buffer_size<kBackend, T>(q, Av, Xv); }
    void reset() {  // the solve overwrites A with the factor and B with X
        (void)MV::copy(q, Av, A0v);
        (void)MV::copy(q, Xv, B0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)gesv<kBackend, T>(q, Av, Xv, piv.to_span(), ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        return {MV::solve_residual(q, A0v, Xv, B0v), bad};
    }
};

class GesvSpec final : public OpSpec {
public:
    std::string op() const override { return "gesv"; }
    std::vector<std::string> key_names() const override { return {G::key_names.begin(), G::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(G::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, r;
        for (int v : G::grid_n) n.push_back(std::to_string(v));
        for (int v : G::grid_nrhs) r.push_back(std::to_string(v));
        return {{"n", n}, {"nrhs", r}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const int n = int(key_int(k, "n"));
        const double nn = double(n);
        return (2.0 * nn * nn + 2.0 * nn * double(key_int(k, "nrhs"))) * double(gesv_batch(dtype, n)) * dtype_bytes(dtype);  // A0, A, B0, X
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "src/kernels/common/sg_partition/backend_amdgcn.hh",
            "src/kernels/common/sg_partition/backend_amdgcn_core.hh",
            "src/kernels/common/sg_partition/backend_generic.hh",
            "src/kernels/common/sg_partition/backend_nvptx.hh",
            "src/kernels/common/sg_partition/backend_spirv.hh",
            "src/kernels/common/sg_partition/backend_sscp.hh",
            "src/kernels/common/sg_partition/sg_partition.hh",
            // family: tiny
            "src/ops/gesv/kernels/gesv_tiny.cc",
            // common
            "src/kernels/common/solve_native.hh",
            "src/ops/gesv/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/gesv/gesv.cc"
        // family: tiny "src/ops/getrf/getrf_native.hh" "src/ops/getrf/getrf_cta_device.hh" "src/kernels/common/tiny_device.hh"
        // blocked calls the public getrf and getrs (and through them trsm and gemm): their kernels and can_run count too.
        // family: blocked "src/ops/getrf/kernels/getrf_tiny.cc" "src/ops/getrf/kernels/getrf_blocked.cc" "src/ops/getrf/kernels/getrf_panel_reg.cc" "src/ops/getrf/kernels/getrf_panel_reg_device.hh" "src/ops/getrf/kernels/getrf_cta.cc" "src/ops/getrf/getrf_cta_device.hh" "src/ops/getrf/getrf_native.hh" "src/kernels/common/tiny_device.hh" "src/ops/getrf/choice.hh" "src/ops/getrf/getrf.cc"
        // family: blocked "src/ops/getrs/kernels/getrs_fused.cc" "src/ops/getrs/kernels/getrs_native.cc" "src/ops/getrs/getrs_native.hh" "src/kernels/common/lu_laswp.hh" "src/ops/getrs/choice.hh" "src/ops/getrs/getrs.cc"
        // family: blocked "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // family: blocked "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/copy_async.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh" "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh" "src/ops/gemm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/gesv_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const int n = int(key_int(req.key, "n"));
            GesvProblem<T> p(q, n, int(key_int(req.key, "nrhs")), gesv_batch(req.dtype, n), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::solve, p.n);
            if (req.mode == "race") return run_race<G::GesvChoice>("gesv", p, req, tol);
            return run_arms<G::GesvChoice>("gesv", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GesvSpec)

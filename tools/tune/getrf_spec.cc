// SPDX-License-Identifier: MIT
// getrf for the tuner (docs/design/flat-kernel-selection.md §6.1): MatrixView::fill_random_pivoted
// inputs (row-permuted dominant, complex parts both), verified on the default items by
// MatrixView::getrf_residual (factorization) and MatrixView::pivot_ratio (the partial-pivoting rule
// no residual sees), each as a fraction of its bound (docs/design/verification.md), so the cell's
// residual is the worse of the two and its tolerance 1. Every candidates<T>() entry is pinned through
// the public getrf. blocked's time includes the public trsm and gemm it calls, each choosing for itself.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/getrf/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace F = ops::getrf;

template <class T>
struct GetrfProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, A;
    UnifiedVector<T*> pA0, pA;
    MV A0v, Av;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<std::int32_t> info;

    GetrfProblem(Queue& q_, int n_, int batch_, int ld_pad)
        : q(q_), n(n_), batch(batch_), ld(n_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)), pA0(std::size_t(batch_)),
          pA(std::size_t(batch_)), A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()),
          Av(A.data(), n, n, ld, int(stride), batch, pA.data()), piv(std::size_t(n_) * std::size_t(batch_), 0),
          info(std::size_t(batch_), 0) {
        (void)A0v.fill_random_pivoted(q, 12345);
        q.wait();
    }
    std::size_t workspace() { return getrf_buffer_size<kBackend, T>(q, Av); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    // Before the verifying run only: a host write here would migrate pages inside a timed run.
    // Pivot 0 is out of range, so an arm that writes no pivots scores NaN.
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
        for (std::int64_t& p : piv) p = 0;
    }
    void run(Span<std::byte> ws) {
        (void)getrf<kBackend, T>(q, Av, piv.to_span(), ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        const VectorView<std::int32_t> p32(reinterpret_cast<std::int32_t*>(piv.data()), n, batch, 1, n);
        const double res = MV::getrf_residual(q, A0v, Av, p32) / verify::bound<T>(verify::Check::factorization, n);
        return {verify::nanmax(res, MV::pivot_ratio(Av) / verify::pivot_ratio_bound<T>()), bad};
    }
};

class GetrfSpec final : public OpSpec {
public:
    std::string op() const override { return "getrf"; }
    std::vector<std::string> key_names() const override { return {F::key_names.begin(), F::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(F::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, b;
        for (int v : F::grid_n) n.push_back(std::to_string(v));
        for (int v : F::grid_batch) b.push_back(std::to_string(v));
        return {{"n", n}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return 2.0 * n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, A
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
            "src/ops/getrf/kernels/getrf_tiny.cc",
            // family: blocked
            "src/ops/getrf/kernels/getrf_blocked.cc",
            "src/ops/getrf/kernels/getrf_panel_reg.cc",
            "src/ops/getrf/kernels/getrf_panel_reg_device.hh",
            // common
            "src/ops/getrf/kernels/getrf_cta.cc",
            "src/ops/getrf/getrf_cta_device.hh",
            "src/ops/getrf/getrf_native.hh",
            "src/ops/getrf/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/getrf/getrf.cc"
        // getrf_cta.cc is common: cta's kernel and blocked's panel leaf (getrf_panel_factorize) live there.
        // family: tiny "src/kernels/common/tiny_device.hh"
        // blocked calls the public trsm and gemm: their native kernels count too.
        // family: blocked "src/kernels/common/lu_laswp.hh"
        // family: blocked "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // family: blocked "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/register_256x128.hh" "src/ops/gemm/kernels/copy_async.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh" "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh" "src/ops/gemm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/getrf_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            GetrfProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<F::GetrfChoice>("getrf", p, req, 1.0);
            return run_arms<F::GetrfChoice>("getrf", p, req, 1.0);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GetrfSpec)

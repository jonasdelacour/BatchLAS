// getrf for the tuner (docs/design/flat-kernel-selection.md §6.1): row-permuted dominant inputs
// with complex noise, the P*L*U reconstruction residual on items 0 and batch-1, every
// candidates<T>() entry pinned through the public getrf. blocked's time includes the public
// trsm and gemm it calls, each choosing for itself.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/getrf/choice.hh"
#include "cell_runner.hh"
#include "lu_residuals.hh"

#include <memory>
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
        fill_lu<T>(A0.data(), n, ld, stride, batch, 12345);
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
        const auto* p32 = reinterpret_cast<const std::int32_t*>(piv.data());
        return {getrf_residual<T>(A.data(), A0.data(), p32, n, ld, stride, batch), bad};
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
            // family: tiny
            "src/extensions/getrf_tiny.cc",
            // family: blocked
            "src/extensions/getrf_blocked.cc",
            "src/extensions/getrf_panel_reg.cc",
            "src/extensions/getrf_panel_reg_device.hh",
            // common
            "src/extensions/getrf_cta.cc",
            "src/extensions/getrf_cta_device.hh",
            "src/extensions/getrf_native.hh",
            "src/ops/getrf/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/getrf/getrf.cc"
        // getrf_cta.cc is common: cta's kernel and blocked's panel leaf (getrf_panel_factorize) live there.
        // family: tiny "src/extensions/tiny_device.hh"
        // blocked calls the public trsm and gemm: their native kernels count too.
        // family: blocked "src/extensions/lu_laswp.hh"
        // family: blocked "src/sycl/trsm_native.cc" "src/sycl/trsm_native.hh" "src/sycl/trsm_sg_left.cc" "src/sycl/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh" "src/ops/gemm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/getrf_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<F::GetrfChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            GetrfProblem<T> p(*q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<F::GetrfChoice>("getrf", p, req, Tol<T>::v);
            return run_arms<F::GetrfChoice>("getrf", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GetrfSpec)

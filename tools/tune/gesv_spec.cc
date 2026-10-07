// gesv for the tuner (docs/design/flat-kernel-selection.md §6.1): row-permuted dominant A with
// complex noise, random complex B, the solve residual on items 0 and batch-1. blocked's time
// includes the public getrf and getrs it composes, each choosing for itself (and through them
// trsm and gemm), so gesv is tuned after those ops.
//
// The table has no batch key (choice.hh: the router reads n and nrhs only), so every cell runs at
// gesv_batch(): saturating, 16384 items at small n, shrinking so A stays near 1 GiB.

#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/gesv/choice.hh"
#include "cell_runner.hh"
#include "lu_residuals.hh"

#include <memory>
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
        fill_lu<T>(A0.data(), n, lda, sa, batch, 12345);
        Rng rg(777);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < nrhs; ++c)
                for (int r = 0; r < n; ++r)
                    B0[std::size_t(b) * sb + std::size_t(c) * std::size_t(ldb) + std::size_t(r)] =
                        mk<T>(rg.next(), rg.next());
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
        return {lu_solve_residual<T>(X.data(), B0.data(), A0.data(), n, nrhs, lda, sa, ldb, sb, batch), bad};
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
            // family: tiny
            "src/extensions/gesv_tiny.cc",
            // common
            "src/extensions/solve_native.hh",
            "src/ops/gesv/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/gesv/gesv.cc"
        // family: tiny "src/extensions/getrf_native.hh" "src/extensions/getrf_cta_device.hh" "src/extensions/tiny_device.hh"
        // blocked calls the public getrf and getrs (and through them trsm and gemm): their kernels and can_run count too.
        // family: blocked "src/extensions/getrf_tiny.cc" "src/extensions/getrf_blocked.cc" "src/extensions/getrf_panel_reg.cc" "src/extensions/getrf_panel_reg_device.hh" "src/extensions/getrf_cta.cc" "src/extensions/getrf_cta_device.hh" "src/extensions/getrf_native.hh" "src/extensions/tiny_device.hh" "src/ops/getrf/choice.hh" "src/ops/getrf/getrf.cc"
        // family: blocked "src/extensions/getrs_fused.cc" "src/extensions/getrs_native.cc" "src/extensions/getrs_native.hh" "src/extensions/lu_laswp.hh" "src/ops/getrs/choice.hh" "src/ops/getrs/getrs.cc"
        // family: blocked "src/sycl/trsm_native.cc" "src/sycl/trsm_native.hh" "src/sycl/trsm_sg_left.cc" "src/sycl/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh" "src/ops/gemm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/gesv_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<G::GesvChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const int n = int(key_int(req.key, "n"));
            GesvProblem<T> p(*q, n, int(key_int(req.key, "nrhs")), gesv_batch(req.dtype, n), req.ld_pad);
            if (req.mode == "race") return run_race<G::GesvChoice>("gesv", p, req, Tol<T>::v);
            return run_arms<G::GesvChoice>("gesv", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GesvSpec)

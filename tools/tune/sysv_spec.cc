// SPDX-License-Identifier: MIT
// sysv for the tuner (docs/design/flat-kernel-selection.md §6.1): sytrf_spec.cc's symmetric A0 and a
// random n x nrhs B0, every candidates<T>() entry pinned through the public sysv, verified on the default
// items by MatrixView::solve_residual (Check::solve). compose's time includes the public sytrf (which
// selects its own kernel) and sytrs. Only Lower is tuned, as for sytrf. evidence: docs/perf/sytrf.md

#include <batchlas/blas/functions/sysv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/sysv/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::sysv;

template <typename T>
struct SysvProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, nrhs, batch, ld;
    std::size_t sa, sb;
    UnifiedVector<T> A0, A, B0, X;
    UnifiedVector<T*> pA0, pA, pB0, pX;
    MV A0v, Av, B0v, Xv;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<std::int32_t> info;

    SysvProblem(Queue& q_, int n_, int nrhs_, int batch_, int ld_pad)
        : q(q_), n(n_), nrhs(nrhs_), batch(batch_), ld(n_ + ld_pad), sa(std::size_t(ld) * std::size_t(n_)),
          sb(std::size_t(ld) * std::size_t(nrhs_)), A0(sa * std::size_t(batch_)), A(sa * std::size_t(batch_)),
          B0(sb * std::size_t(batch_)), X(sb * std::size_t(batch_)), pA0(std::size_t(batch_)), pA(std::size_t(batch_)),
          pB0(std::size_t(batch_)), pX(std::size_t(batch_)), A0v(A0.data(), n, n, ld, int(sa), batch, pA0.data()),
          Av(A.data(), n, n, ld, int(sa), batch, pA.data()), B0v(B0.data(), n, nrhs, ld, int(sb), batch, pB0.data()),
          Xv(X.data(), n, nrhs, ld, int(sb), batch, pX.data()), piv(std::size_t(n_) * std::size_t(batch_), 0),
          info(std::size_t(batch_), 0) {
        (void)A0v.fill_random(q, false, 12345);
        (void)A0v.symmetrize(q, Uplo::Lower);
        (void)B0v.fill_random(q, false, 777);
        q.wait();
    }
    std::size_t workspace() { return sysv_buffer_size<kBackend, T>(q, Av, Xv, Uplo::Lower); }
    void reset() {  // the solve overwrites A with the factor and B with X
        (void)MV::copy(q, Av, A0v);
        (void)MV::copy(q, Xv, B0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)sysv<kBackend, T>(q, Av, Xv, Uplo::Lower, piv.to_span(), ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        return {MV::solve_residual(q, A0v, Xv, B0v), bad};
    }
};

class SysvSpec final : public OpSpec {
public:
    std::string op() const override { return "sysv"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<typename T>() { return spellings(S::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, r, b;
        for (int v : S::grid_n) n.push_back(std::to_string(v));
        for (int v : S::grid_nrhs) r.push_back(std::to_string(v));
        for (int v : S::grid_batch) b.push_back(std::to_string(v));
        return {{"n", n}, {"nrhs", r}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return (2.0 * n * n + 2.0 * n * double(key_int(k, "nrhs"))) * double(key_int(k, "batch")) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/device_scalar.hh",
            "include/batchlas/util/group-collectives.hh",
            // cta is sytrf_kernels.cc's fused launch with sytrs cta's solve; compose's sytrf and sytrs live there too
            // common
            "src/ops/sytrf/kernels/sytrf_kernels.cc",
            "src/ops/sytrf/sytrf_device.hh",
            "src/ops/sytrf/sytrs_device.hh",
            "src/ops/sytrf/sytrf_native.hh",
            "src/ops/sysv/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/sysv/sysv.cc"
        // compose calls the public sytrf and the routed sytrs, which select for themselves: their routing,
        // kernels and vendor arm count too.
        // family: compose "src/ops/sytrf/sytrf.cc" "src/ops/sytrf/choice.hh" "src/backends/cusolver_sytrf.cc" "include/batchlas/tuning_params.hh"
        // family: compose "src/ops/sytrs/sytrs.hh" "src/ops/sytrs/choice.hh" "src/ops/sytrf/kernels/sytrs_kernels.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/sysv_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<typename T>() {
            SysvProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "nrhs")), int(key_int(req.key, "batch")),
                             req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::solve, p.n);
            if (req.mode == "race") return run_race<S::SysvChoice>("sysv", p, req, tol);
            return run_arms<S::SysvChoice>("sysv", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SysvSpec)

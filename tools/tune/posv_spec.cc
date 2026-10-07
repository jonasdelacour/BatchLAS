// posv for the tuner (flat-kernel-selection-phase3-plan.md §1.1): SPD A, random B, the
// factor_bench solve residual on items 0 and batch-1. posv's cta and blocked times include the
// public potrf and trsm the composition calls, each choosing for itself (plan §2 "Coupling").

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/posv/choice.hh"
#include "cell_runner.hh"
#include "residuals.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::posv;

template <class T>
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
        fill_spd<T>(A0.data(), n, lda, sa, batch);
        Rng rg(777);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < nrhs; ++c)
                for (int r = 0; r < n; ++r)
                    B0[std::size_t(b) * sb + std::size_t(c) * std::size_t(ldb) + std::size_t(r)] =
                        mk<T>(rg.next(), rg.next());
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
        return {solve_residual<T>(X.data(), B0.data(), A0.data(), n, nrhs, lda, sa, ldb, sb, batch), bad};
    }
};

class PosvSpec final : public OpSpec {
public:
    std::string op() const override { return "posv"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(S::candidates<T>()); });
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
            // family: tiny
            "src/extensions/posv_tiny.cc",
            "src/extensions/solve_native.hh",
            // family: cta
            "src/extensions/getrs_fused.cc",
            "src/extensions/getrs_native.hh",
            // common
            "src/ops/posv/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/posv/posv.cc"
        // family: tiny "src/extensions/tiny_device.hh" "src/extensions/potrf_native.hh"
        // posv cta and blocked call the public potrf and trsm: their native kernels count too.
        // family: cta "src/extensions/potrf_tiny.cc" "src/extensions/potrf_cta.cc" "src/extensions/potrf_cta_device.hh" "src/extensions/potrf_lpanel.cc" "src/extensions/potrf_lpanel_device.hh" "src/extensions/potrf_blocked.cc" "src/extensions/potrf_native.hh" "src/extensions/potrf_slm_hole.hh" "src/ops/potrf/choice.hh"
        // family: cta "src/sycl/trsm_native.cc" "src/sycl/trsm_native.hh" "src/sycl/trsm_sg_left.cc" "src/sycl/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // family: blocked "src/extensions/potrf_tiny.cc" "src/extensions/potrf_cta.cc" "src/extensions/potrf_cta_device.hh" "src/extensions/potrf_lpanel.cc" "src/extensions/potrf_lpanel_device.hh" "src/extensions/potrf_blocked.cc" "src/extensions/potrf_native.hh" "src/extensions/potrf_slm_hole.hh" "src/ops/potrf/choice.hh"
        // family: blocked "src/sycl/trsm_native.cc" "src/sycl/trsm_native.hh" "src/sycl/trsm_sg_left.cc" "src/sycl/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/posv_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<S::PosvChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const Uplo uplo = *key_get(req.key, "uplo") == "U" ? Uplo::Upper : Uplo::Lower;
            PosvProblem<T> p(*q, int(key_int(req.key, "n")), int(key_int(req.key, "nrhs")),
                             int(key_int(req.key, "batch")), req.ld_pad, uplo);
            if (req.mode == "race") return run_race<S::PosvChoice>("posv", p, req, Tol<T>::v);
            return run_arms<S::PosvChoice>("posv", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::PosvSpec)

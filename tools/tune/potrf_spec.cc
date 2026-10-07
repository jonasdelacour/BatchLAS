// potrf for the tuner (docs/design/flat-kernel-selection.md §6.1): SPD inputs, the factor_bench
// residual on items 0 and batch-1, every candidates<T>() entry pinned through the public potrf.

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/potrf/choice.hh"
#include "cell_runner.hh"
#include "residuals.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace P = ops::potrf;

template <class T>
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
        fill_spd<T>(A0.data(), n, ld, stride, batch);
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
        return {potrf_residual<T>(A.data(), A0.data(), n, ld, stride, batch, uplo == Uplo::Upper), bad};
    }
};

class PotrfSpec final : public OpSpec {
public:
    std::string op() const override { return "potrf"; }
    std::vector<std::string> key_names() const override { return {P::key_names.begin(), P::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(P::candidates<T>()); });
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
            // family: tiny
            "src/extensions/potrf_tiny.cc",
            // family: cta
            "src/extensions/potrf_cta.cc",
            "src/extensions/potrf_cta_device.hh",
            // family: lpanel
            "src/extensions/potrf_lpanel.cc",
            "src/extensions/potrf_lpanel_device.hh",
            // family: blocked
            "src/extensions/potrf_blocked.cc",
            // common
            "src/extensions/potrf_native.hh",
            "src/extensions/potrf_slm_hole.hh",
            "src/ops/potrf/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // family: tiny "src/extensions/tiny_device.hh"
        // family: blocked "src/extensions/symmetric_product_fold.hh" "src/sycl/trsm_native.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/potrf_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<P::PotrfChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const Uplo uplo = *key_get(req.key, "uplo") == "U" ? Uplo::Upper : Uplo::Lower;
            PotrfProblem<T> p(*q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad, uplo);
            return run_arms<P::PotrfChoice>("potrf", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::PotrfSpec)

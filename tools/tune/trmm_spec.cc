// trmm for the tuner: C = alpha op(A) B (Left) or alpha B op(A) (Right), out of place. A is
// lower triangular with random entries and its strict upper triangle poisoned with a large finite
// value; B is random; alpha has an imaginary part; C is re-poisoned before every run, so an arm
// that skips a tile cannot inherit the previous arm's answer. uplo, trans and diag are not table
// keys: fixed at L, N and N. Every ld is padded by --ld-pad. Verification is the componentwise
// error |C - Cref| / (|alpha| |op(A)| |B|) in double / complex<double> on items 0 and batch-1, over
// up to 256 sampled rows of up to 64 sampled columns (always the first and last of each).
// expand's times include the public gemm it calls.

#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/trmm/choice.hh"
#include "cell_runner.hh"
#include "residuals.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace M = ops::trmm;

template <class T>
struct TrmmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using D = typename Prom<T>::type;
    Queue& q;
    Side side;
    int order, nrhs, batch, rows, cols, lda, ldb;
    std::size_t sa, sb;
    T alpha;
    UnifiedVector<T> A, B, Cp, C;

    TrmmProblem(Queue& q_, Side s, int order_, int q_rhs, int batch_, int pad)
        : q(q_), side(s), order(order_), nrhs(q_rhs), batch(batch_), rows(s == Side::Left ? order_ : q_rhs),
          cols(s == Side::Left ? q_rhs : order_), lda(order_ + pad), ldb(rows + pad),
          sa(std::size_t(lda) * std::size_t(order_)), sb(std::size_t(ldb) * std::size_t(cols)), alpha(mk<T>(1.5, -0.5)),
          A(sa * std::size_t(batch_)), B(sb * std::size_t(batch_)), Cp(sb * std::size_t(batch_), mk<T>(-999, 777)),
          C(sb * std::size_t(batch_)) {
        Rng rg(5150);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < order; ++c)
                for (int r = 0; r < order; ++r)
                    A[std::size_t(b) * sa + std::size_t(c) * std::size_t(lda) + std::size_t(r)] =
                        r >= c ? mk<T>(rg.next(), rg.next()) : mk<T>(1e6, 0.0);
        for (std::size_t i = 0; i < B.size(); ++i) B[i] = mk<T>(rg.next(), rg.next());
    }
    MV view(UnifiedVector<T>& v, int r, int c, int ld, std::size_t s) { return MV(v.data(), r, c, ld, int(s), batch); }
    void reset() {
        (void)MV::copy(q, view(C, rows, cols, ldb, sb), view(Cp, rows, cols, ldb, sb));
        q.wait();
    }
    // trmm takes no caller workspace; probing a pin means running once (a refused pin throws first).
    std::size_t workspace() {
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)trmm<kBackend, T>(q, view(A, order, order, lda, sa), view(B, rows, cols, ldb, sb),
                                view(C, rows, cols, ldb, sb), alpha, side, Uplo::Lower, Transpose::NoTrans,
                                Diag::NonUnit);
        q.wait();
    }
    std::pair<double, int> verify() {
        double worst = 0;
        const double aal = ab(up(alpha));
        for (int b : {0, batch - 1}) {
            const std::size_t oa = std::size_t(b) * sa, ob = std::size_t(b) * sb;
            auto a = [&](int r, int c) {
                return r >= c ? up(A[oa + std::size_t(c) * std::size_t(lda) + std::size_t(r)]) : D(0);
            };
            auto at = [&](const UnifiedVector<T>& v, int r, int c) {
                return up(v[ob + std::size_t(c) * std::size_t(ldb) + std::size_t(r)]);
            };
            for (int j : sample_indices(cols, 64))
                for (int i : sample_indices(rows, 256)) {
                    D acc = D(0);
                    double mag = 0;
                    for (int k = 0; k < order; ++k) {
                        const D t = side == Side::Left ? a(i, k) * at(B, k, j) : at(B, i, k) * a(k, j);
                        acc += t;
                        mag += ab(t);
                    }
                    const double num = ab(at(C, i, j) - up(alpha) * acc), den = aal * mag;
                    worst = nanmax(worst, den > 0 ? num / den : num);
                }
        }
        return {worst, 0};
    }
};

class TrmmSpec final : public OpSpec {
public:
    std::string op() const override { return "trmm"; }
    std::vector<std::string> key_names() const override { return {M::key_names.begin(), M::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(M::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> o, r, b;
        for (int v : M::grid_order) o.push_back(std::to_string(v));
        for (int v : M::grid_q) r.push_back(std::to_string(v));
        for (int v : M::grid_batch) b.push_back(std::to_string(v));
        return {{"side", {"L", "R"}}, {"order", o}, {"q", r}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A, B, Cp, C
        const double o = double(key_int(k, "order"));
        return (o * o + 3.0 * o * double(key_int(k, "q"))) * double(key_int(k, "batch")) * dtype_bytes(dtype);
    }
    std::string refine_key() const override { return "order"; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            // family: triangular
            "src/backends/trmm_triangular_tiles.hh",
            "src/backends/triangular_tiles.hh",
            // family: expand
            "src/backends/triangular_expand.hh",
            "src/expansion_budget.hh",
            // common
            "src/ops/trmm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/trmm/trmm.cc"
        // family: expand "src/backends/route_common.hh"
        // trmm expand calls the public gemm: its native kernels and can_run count too.
        // family: expand "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/trmm_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<M::TrmmChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const Side side = *key_get(req.key, "side") == "R" ? Side::Right : Side::Left;
            TrmmProblem<T> p(*q, side, int(key_int(req.key, "order")), int(key_int(req.key, "q")),
                             int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<M::TrmmChoice>("trmm", p, req, Tol<T>::v);
            return run_arms<M::TrmmChoice>("trmm", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::TrmmSpec)

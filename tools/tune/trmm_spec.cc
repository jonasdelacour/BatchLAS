// trmm for the tuner: C = alpha op(A) B (Left) or alpha B op(A) (Right), out of place. A is
// lower triangular, MatrixView::fill_random with its strict upper triangle then poisoned with a
// large finite value; B is fill_random; alpha has an imaginary part; C is re-poisoned before every
// run, so an arm that skips a tile cannot inherit the previous arm's answer. uplo, trans and diag
// are not table keys: fixed at L, N and N. Every ld is padded by --ld-pad. Verification is
// MatrixView::gemm_backward_error against A's triangle (triangularize on a packed copy of the
// default items), |C - Cref| / (|alpha| |op(A)| |B|) componentwise (Check::blas,
// docs/design/verification.md). expand's times include the public gemm it calls.

#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/trmm/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace M = ops::trmm;

template <class T>
struct TrmmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    Side side;
    int order, nrhs, batch, rows, cols, lda, ldb;
    std::size_t sa, sb;
    T alpha;
    UnifiedVector<T> A, B, Cp, C;

    TrmmProblem(Queue& q_, Side s, int order_, int q_rhs, int batch_, int pad)
        : q(q_), side(s), order(order_), nrhs(q_rhs), batch(batch_), rows(s == Side::Left ? order_ : q_rhs),
          cols(s == Side::Left ? q_rhs : order_), lda(order_ + pad), ldb(rows + pad),
          sa(std::size_t(lda) * std::size_t(order_)), sb(std::size_t(ldb) * std::size_t(cols)), alpha(MV::make(1.5, -0.5)),
          A(sa * std::size_t(batch_)), B(sb * std::size_t(batch_)), Cp(sb * std::size_t(batch_), MV::make(-999, 777)),
          C(sb * std::size_t(batch_)) {
        (void)view(A, order, order, lda, sa).fill_random(q, false, 5150);
        (void)view(B, rows, cols, ldb, sb).fill_random(q, false, 5151);
        q.wait();
        for (int b = 0; b < batch; ++b)
            for (int c = 1; c < order; ++c)
                for (int r = 0; r < c; ++r)
                    A[std::size_t(b) * sa + std::size_t(c) * std::size_t(lda) + std::size_t(r)] = MV::make(1e6, 0.0);
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
        const MV av = view(A, order, order, lda, sa);
        const auto items = av.default_items();
        auto a = av.template astype<T>(items);
        auto b = view(B, rows, cols, ldb, sb).template astype<T>(items), c = view(C, rows, cols, ldb, sb).template astype<T>(items);
        (void)a.triangularize(q, Uplo::Lower, Diag::NonUnit);
        q.wait();
        const auto all = a.all_items();
        const Transpose nt = Transpose::NoTrans;
        const auto al = MV::up(alpha), zero = MV::up(T(0));
        return {side == Side::Left ? MV::gemm_backward_error(q, a, nt, b, nt, b, c, al, zero, all)
                                   : MV::gemm_backward_error(q, b, nt, a, nt, b, c, al, zero, all),
                0};
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
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: triangular
            "src/ops/trmm/kernels/trmm_triangular_tiles.hh",
            "src/kernels/common/triangular_tiles.hh",
            // family: expand
            "src/kernels/common/triangular_expand.hh",
            "src/expansion_budget.hh",
            // common
            "src/ops/trmm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/trmm/trmm.cc"
        // trmm expand calls the public gemm: its native kernels and can_run count too.
        // family: expand "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh" "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/trmm_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const Side side = *key_get(req.key, "side") == "R" ? Side::Right : Side::Left;
            TrmmProblem<T> p(q, side, int(key_int(req.key, "order")), int(key_int(req.key, "q")),
                             int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, p.order);
            if (req.mode == "race") return run_race<M::TrmmChoice>("trmm", p, req, tol);
            return run_arms<M::TrmmChoice>("trmm", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::TrmmSpec)

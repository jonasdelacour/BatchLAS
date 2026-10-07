// trsm for the tuner (flat-kernel-selection-phase3-plan.md §1.2, §3): a strongly diagonally
// dominant triangular A (other triangle poisoned with a large finite value), random B, alpha with
// an imaginary part, and the componentwise backward error of op(A) X = alpha B (Left) or X op(A) = alpha B (Right) on
// items 0 and batch-1. trans=T times ConjTrans for a complex scalar (what the library's callers
// issue; the key folds C into T). blocked's times include the public gemm it calls.
//
// uplo and diag are not table keys (plan §1.2). They are hidden grid axes fixed at L and N, so
// the invariance A/B is a flag: `--grid uplo=L:U --grid diag=N:U --no-refine` with --raw only.
// scripts/sweep_to_table.py refuses to make a table from such a run.

#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/trsm/choice.hh"
#include "cell_runner.hh"
#include "residuals.hh"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::trsm;

constexpr int kMaxCheckedRhs = 32;  // host check cost is order^2 per rhs; sample the rest

template <class T>
struct TrsmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using D = typename Prom<T>::type;
    Queue& q;
    Side side;
    Uplo uplo;
    Transpose trans;
    Diag diag;
    int order, nrhs, batch, brows, bcols, lda, ldb;
    std::size_t sa, sb;
    T alpha;
    UnifiedVector<T> A, B0, X;
    UnifiedVector<T*> pA, pB0, pX;
    MV Av, B0v, Xv;

    TrsmProblem(Queue& q_, Side s, Uplo u, Transpose t, Diag d, int order_, int q_rhs, int batch_, int ld_pad)
        : q(q_), side(s), uplo(u), trans(t), diag(d), order(order_), nrhs(q_rhs), batch(batch_),
          brows(s == Side::Left ? order_ : q_rhs), bcols(s == Side::Left ? q_rhs : order_), lda(order_ + ld_pad),
          ldb(brows + ld_pad), sa(std::size_t(lda) * std::size_t(order_)), sb(std::size_t(ldb) * std::size_t(bcols)),
          alpha(mk<T>(1.5, -0.5)), A(sa * std::size_t(batch_)), B0(sb * std::size_t(batch_)), X(sb * std::size_t(batch_)),
          pA(std::size_t(batch_)), pB0(std::size_t(batch_)), pX(std::size_t(batch_)),
          Av(A.data(), order, order, lda, int(sa), batch, pA.data()),
          B0v(B0.data(), brows, bcols, ldb, int(sb), batch, pB0.data()),
          Xv(X.data(), brows, bcols, ldb, int(sb), batch, pX.data()) {
        // Off-diagonal row mass <= 0.56*(pi^2/6 - 1) ~ 0.36 against a diagonal of 1 or ~2.06: well
        // conditioned for NonUnit and Unit alike. The coupling must not shrink with order, or a
        // lost trailing block (blocked's gemm) stays under Tol<float> from order 256 up.
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < order; ++c)
                for (int r = 0; r < order; ++r) {
                    const bool in_tri = uplo == Uplo::Lower ? r >= c : r <= c;
                    const double w = 0.5 / ((1.0 + std::abs(r - c)) * (1.0 + std::abs(r - c)));
                    A[std::size_t(b) * sa + std::size_t(c) * std::size_t(lda) + std::size_t(r)] =
                        r == c ? mk<T>(2.0, 0.5) : (in_tri ? mk<T>(w, 0.5 * w) : mk<T>(1e6, 0.0));
                }
        Rng rg(4242);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < bcols; ++c)
                for (int r = 0; r < brows; ++r)
                    B0[std::size_t(b) * sb + std::size_t(c) * std::size_t(ldb) + std::size_t(r)] =
                        mk<T>(rg.next(), rg.next());
    }
    void reset() {  // the solve overwrites B with X
        (void)MV::copy(q, Xv, B0v);
        q.wait();
    }
    // trsm takes no workspace; probing a pin means running once (a refused pin throws first).
    std::size_t workspace() {
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)trsm<kBackend, T>(q, Av, Xv, alpha, side, uplo, trans, diag);
        q.wait();
    }
    D opA(std::size_t o, int r, int c) const {
        const int sr = trans == Transpose::NoTrans ? r : c, sc = trans == Transpose::NoTrans ? c : r;
        if (uplo == Uplo::Lower ? sr < sc : sr > sc) return D(0);
        if (sr == sc && diag == Diag::Unit) return D(1);
        const D v = up(A[o + std::size_t(sc) * std::size_t(lda) + std::size_t(sr)]);
        return trans == Transpose::ConjTrans ? cj(v) : v;
    }
    // Componentwise backward error, max over sampled rhs and rows i of
    // |op(A) X - alpha B|_i / (|op(A)| |X| + |alpha| |B|)_i. A normwise residual averages a lost
    // 32-wide trailing block over the whole matrix and passes Tol<float> from order ~1024 up
    // (2e-4); componentwise it is ~0.3 against ~5e-8 for a correct float solve at every order.
    std::pair<double, int> verify() {
        std::vector<int> rhs;
        for (int i = 0; i < std::min(nrhs, kMaxCheckedRhs); ++i)
            rhs.push_back(nrhs <= kMaxCheckedRhs ? i : int(std::int64_t(i) * (nrhs - 1) / (kMaxCheckedRhs - 1)));
        double worst = 0;
        const double aal = ab(up(alpha));
        for (int b : {0, batch - 1}) {
            const std::size_t oa = std::size_t(b) * sa, ob = std::size_t(b) * sb;
            auto x = [&](int r, int c) { return up(X[ob + std::size_t(c) * std::size_t(ldb) + std::size_t(r)]); };
            auto b0 = [&](int r, int c) { return up(B0[ob + std::size_t(c) * std::size_t(ldb) + std::size_t(r)]); };
            for (int j : rhs)
                for (int i = 0; i < order; ++i) {
                    // Left: rhs j is column j of X; Right: rhs j is row j.
                    const int xr = side == Side::Left ? i : j, xc = side == Side::Left ? j : i;
                    D acc = D(0);
                    double mag = 0;
                    for (int k = 0; k < order; ++k) {
                        const D t = side == Side::Left ? opA(oa, i, k) * x(k, j) : x(j, k) * opA(oa, k, i);
                        acc += t;
                        mag += ab(t);
                    }
                    const double num = ab(acc - up(alpha) * b0(xr, xc)), den = mag + aal * ab(b0(xr, xc));
                    worst = nanmax(worst, den > 0 ? num / den : num);
                }
        }
        return {worst, 0};
    }
};

class TrsmSpec final : public OpSpec {
public:
    std::string op() const override { return "trsm"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(S::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> o, r, b;
        for (int v : S::grid_order) o.push_back(std::to_string(v));
        for (int v : S::grid_q) r.push_back(std::to_string(v));
        for (int v : S::grid_batch) b.push_back(std::to_string(v));
        return {{"side", {"L", "R"}}, {"trans", {"N", "T"}}, {"order", o}, {"q", r}, {"batch", b},
                {"uplo", {"L"}}, {"diag", {"N"}}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double o = double(key_int(k, "order"));
        return (o * o + 2.0 * o * double(key_int(k, "q"))) * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A, B0, X
    }
    std::string refine_key() const override { return "order"; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/trsm_native.cc",
            "src/sycl/trsm_native.hh",
            // family: sg_left
            "src/sycl/trsm_sg_left.cc",
            // common
            "src/sycl/trsm_canonical.hh",
            "src/ops/trsm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/trsm/trsm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/trsm_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<S::TrsmChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            auto get = [&](std::string_view name, const char* dflt) {
                const std::string* v = key_get(req.key, name);
                return v ? *v : std::string(dflt);
            };
            const Side side = get("side", "L") == "R" ? Side::Right : Side::Left;
            constexpr bool kComplex = !std::is_same_v<T, typename base_type<T>::type>;
            const Transpose trans = get("trans", "N") == "N" ? Transpose::NoTrans
                                    : kComplex                ? Transpose::ConjTrans
                                                              : Transpose::Trans;
            const Uplo uplo = get("uplo", "L") == "U" ? Uplo::Upper : Uplo::Lower;
            const Diag diag = get("diag", "N") == "U" ? Diag::Unit : Diag::NonUnit;
            TrsmProblem<T> p(*q, side, uplo, trans, diag, int(key_int(req.key, "order")), int(key_int(req.key, "q")),
                             int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<S::TrsmChoice>("trsm", p, req, Tol<T>::v);
            return run_arms<S::TrsmChoice>("trsm", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::TrsmSpec)

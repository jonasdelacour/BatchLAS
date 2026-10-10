// trsm for the tuner (flat-kernel-selection-phase3-plan.md §1.2, §3): a strongly diagonally
// dominant triangular A (other triangle poisoned with a large finite value), random B, alpha with
// an imaginary part, and the componentwise backward error of op(A) X = alpha B (Left) or
// X op(A) = alpha B (Right) through batchlas::verify (docs/design/verification.md). trans=T times
// ConjTrans for a complex scalar (what the library's callers issue; the key folds C into T).
// blocked's times include the public gemm it calls.
//
// uplo and diag are not table keys (plan §1.2). They are hidden grid axes fixed at L and N, so
// the invariance A/B is a flag: `--grid uplo=L:U --grid diag=N:U --no-refine` with --raw only.
// scripts/sweep_to_table.py refuses to make a table from such a run.

#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/inputs.hh>
#include <batchlas/verify/items.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include "../../src/ops/trsm/choice.hh"
#include "cell_runner.hh"

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
          alpha(batchlas::verify::make<T>(1.5, -0.5)), A(sa * std::size_t(batch_)), B0(sb * std::size_t(batch_)),
          X(sb * std::size_t(batch_)), pA(std::size_t(batch_)), pB0(std::size_t(batch_)), pX(std::size_t(batch_)),
          Av(A.data(), order, order, lda, int(sa), batch, pA.data()),
          B0v(B0.data(), brows, bcols, ldb, int(sb), batch, pB0.data()),
          Xv(X.data(), brows, bcols, ldb, int(sb), batch, pX.data()) {
        // Off-diagonal row mass <= 0.56*(pi^2/6 - 1) ~ 0.36 against a diagonal of 1 or ~2.06: well
        // conditioned for NonUnit and Unit alike. The coupling must not shrink with order, or a
        // lost trailing block (blocked's gemm) stays under the float bound from order 256 up.
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < order; ++c)
                for (int r = 0; r < order; ++r) {
                    const bool in_tri = uplo == Uplo::Lower ? r >= c : r <= c;
                    const double w = 0.5 / ((1.0 + std::abs(r - c)) * (1.0 + std::abs(r - c)));
                    A[std::size_t(b) * sa + std::size_t(c) * std::size_t(lda) + std::size_t(r)] =
                        r == c ? batchlas::verify::make<T>(2.0, 0.5)
                        : batchlas::verify::make<T>(in_tri ? w : 1e6, in_tri ? 0.5 * w : 0.0);
                }
        batchlas::verify::fill_random(B0v, 4242);
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
    // Componentwise backward error, max over sampled rhs and rows i of
    // |op(A) X - alpha B|_i / (|op(A)| |X| + |alpha| |B|)_i: gemm_backward_error with C = 0, C0 = B0,
    // beta = -alpha, over packed copies of the sampled rhs (Left: columns of X; Right: rows). A
    // normwise residual averages a lost 32-wide trailing block over the whole matrix and passes a
    // float bound from order ~1024 up (2e-4); componentwise it is ~0.3 against ~5e-8 for a correct
    // float solve at every order.
    std::pair<double, int> verify() {
        using V = MatrixView<T, MatrixFormat::Dense>;
        using batchlas::verify::Shape;
        const bool unit = diag == Diag::Unit, left = side == Side::Left;
        const Shape shape = uplo == Uplo::Lower ? (unit ? Shape::unit_lower : Shape::lower)
                                                : (unit ? Shape::unit_upper : Shape::upper);
        std::vector<int> rhs;
        for (int i = 0; i < std::min(nrhs, kMaxCheckedRhs); ++i)
            rhs.push_back(nrhs <= kMaxCheckedRhs ? i : int(std::int64_t(i) * (nrhs - 1) / (kMaxCheckedRhs - 1)));
        const int nr = int(rhs.size());
        const std::size_t len = std::size_t(order) * std::size_t(nr);
        std::vector<T> xs(len), b0s(len), zero(len, T(0));
        // Element (t, i) of rhs i in the packed copy: Left order x nr, Right nr x order.
        auto at = [&](int t, int i) { return left ? std::size_t(i) * std::size_t(order) + std::size_t(t)
                                                  : std::size_t(t) * std::size_t(nr) + std::size_t(i); };
        const int ldp = left ? order : nr;
        double worst = 0;
        for (int b : batchlas::verify::default_items(batch)) {
            const std::size_t ob = std::size_t(b) * sb;
            for (int i = 0; i < nr; ++i)
                for (int t = 0; t < order; ++t) {
                    const int r = left ? t : rhs[std::size_t(i)], c = left ? rhs[std::size_t(i)] : t;
                    const std::size_t o = ob + std::size_t(c) * std::size_t(ldb) + std::size_t(r);
                    xs[at(t, i)] = X[o];
                    b0s[at(t, i)] = B0[o];
                }
            const V a(A.data() + std::size_t(b) * sa, order, order, lda);
            const V x = left ? V(xs.data(), order, nr, ldp) : V(xs.data(), nr, order, ldp);
            const V b0 = left ? V(b0s.data(), order, nr, ldp) : V(b0s.data(), nr, order, ldp);
            const V z = left ? V(zero.data(), order, nr, ldp) : V(zero.data(), nr, order, ldp);
            const auto one = batchlas::verify::up(T(1)), nal = -batchlas::verify::up(alpha);
            const Transpose nt = Transpose::NoTrans;
            const double e = left ? batchlas::verify::gemm_backward_error(a, shape, trans, x, Shape::general, nt, b0, z,
                                                                          Shape::general, one, nal)
                                  : batchlas::verify::gemm_backward_error(x, Shape::general, nt, a, shape, trans, b0, z,
                                                                          Shape::general, one, nal);
            worst = batchlas::verify::nanmax(worst, e);
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
            const double tol = batchlas::verify::bound<T>(batchlas::verify::Check::blas, p.order);
            if (req.mode == "race") return run_race<S::TrsmChoice>("trsm", p, req, tol);
            return run_arms<S::TrsmChoice>("trsm", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::TrsmSpec)

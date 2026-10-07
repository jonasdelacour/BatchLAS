// gemv for the tuner: random A, x and y0 in [-1, 1) (complex parts both), alpha with an imaginary
// part and beta = 0.75, so the epilogue reads y. trans=T times ConjTrans for a complex scalar (the
// key folds C into T). A's ld and x's and y's item strides are padded by --ld-pad. Verification is
// the componentwise error |y - yref| / (|alpha| |op(A)| |x| + |beta| |y0|) against a double /
// complex<double> host reference on items 0 and batch-1, over up to 256 sampled outputs (always
// the first and last).

#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/gemv/choice.hh"
#include "cell_runner.hh"
#include "residuals.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace V = ops::gemv;

template <class T>
struct GemvProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using D = typename Prom<T>::type;
    Queue& q;
    Transpose trans;
    int out, red, batch, rows, cols, lda, sx, sy;
    std::size_t sa;
    T alpha, beta;
    UnifiedVector<T> A, X, Y0, Y;

    GemvProblem(Queue& q_, Transpose t, int out_, int red_, int batch_, int pad)
        : q(q_), trans(t), out(out_), red(red_), batch(batch_), rows(t == Transpose::NoTrans ? out_ : red_),
          cols(t == Transpose::NoTrans ? red_ : out_), lda(rows + pad), sx(red_ + pad), sy(out_ + pad),
          sa(std::size_t(lda) * std::size_t(cols)), alpha(mk<T>(1.25, -0.5)), beta(mk<T>(0.75, 0.0)),
          A(sa * std::size_t(batch_)), X(std::size_t(sx) * std::size_t(batch_)), Y0(std::size_t(sy) * std::size_t(batch_)),
          Y(std::size_t(sy) * std::size_t(batch_)) {
        Rng rg(919);
        for (auto* v : {&A, &X, &Y0})
            for (std::size_t i = 0; i < v->size(); ++i) (*v)[i] = mk<T>(rg.next(), rg.next());
    }
    MV Av() { return MV(A.data(), rows, cols, lda, int(sa), batch); }
    void reset() {  // beta != 0 accumulates into y
        (void)MV::copy(q, MV(Y.data(), sy, 1, sy, sy, batch), MV(Y0.data(), sy, 1, sy, sy, batch));
        q.wait();
    }
    // gemv takes no workspace; probing a pin means running once (a refused pin throws first).
    std::size_t workspace() {
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)gemv<kBackend, T>(q, Av(), VectorView<T>(X.data(), red, batch, 1, sx),
                                VectorView<T>(Y.data(), out, batch, 1, sy), alpha, beta, trans);
        q.wait();
    }
    std::pair<double, int> verify() {
        double worst = 0;
        const double aal = ab(up(alpha)), abe = ab(up(beta));
        for (int b : {0, batch - 1}) {
            const std::size_t oa = std::size_t(b) * sa, ox = std::size_t(b) * std::size_t(sx),
                              oy = std::size_t(b) * std::size_t(sy);
            for (int i : sample_indices(out, 256)) {
                D acc = D(0);
                double mag = 0;
                for (int j = 0; j < red; ++j) {
                    const int r = trans == Transpose::NoTrans ? i : j, c = trans == Transpose::NoTrans ? j : i;
                    D a = up(A[oa + std::size_t(c) * std::size_t(lda) + std::size_t(r)]);
                    if (trans == Transpose::ConjTrans) a = cj(a);
                    const D t = a * up(X[ox + std::size_t(j)]);
                    acc += t;
                    mag += ab(t);
                }
                const D y0 = up(Y0[oy + std::size_t(i)]);
                const double num = ab(up(Y[oy + std::size_t(i)]) - (up(alpha) * acc + up(beta) * y0)),
                             den = aal * mag + abe * ab(y0);
                worst = nanmax(worst, den > 0 ? num / den : num);
            }
        }
        return {worst, 0};
    }
};

class GemvSpec final : public OpSpec {
public:
    std::string op() const override { return "gemv"; }
    std::vector<std::string> key_names() const override { return {V::key_names.begin(), V::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(V::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> o, r, b;
        for (int v : V::grid_out) o.push_back(std::to_string(v));
        for (int v : V::grid_red) r.push_back(std::to_string(v));
        for (int v : V::grid_batch) b.push_back(std::to_string(v));
        return {{"trans", {"N", "T"}}, {"out", o}, {"red", r}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A, x, y0, y
        const double o = double(key_int(k, "out")), r = double(key_int(k, "red"));
        return (o * r + r + 2.0 * o) * double(key_int(k, "batch")) * dtype_bytes(dtype);
    }
    std::string refine_key() const override { return "red"; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            // cta and direct are both in gemv_native.cc: common only.
            "src/sycl/gemv_native.cc",
            "src/sycl/gemv_native.hh",
            "src/ops/gemv/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/gemv/gemv.cc" "src/sycl/device_scalar.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/gemv_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<V::GemvChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            constexpr bool kComplex = !std::is_same_v<T, typename base_type<T>::type>;
            const bool n = *key_get(req.key, "trans") == "N";
            const Transpose t = n ? Transpose::NoTrans : kComplex ? Transpose::ConjTrans : Transpose::Trans;
            GemvProblem<T> p(*q, t, int(key_int(req.key, "out")), int(key_int(req.key, "red")),
                             int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<V::GemvChoice>("gemv", p, req, Tol<T>::v);
            return run_arms<V::GemvChoice>("gemv", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GemvSpec)

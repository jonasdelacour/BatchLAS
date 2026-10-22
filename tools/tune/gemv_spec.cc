// SPDX-License-Identifier: MIT
// gemv for the tuner: MatrixView::fill_random A, x and y0 (complex parts both), alpha with an
// imaginary part and beta = 0.75, so the epilogue reads y. trans=T times ConjTrans for a complex
// scalar (the key folds C into T). A's ld and x's and y's item strides are padded by --ld-pad.
// Verification is MatrixView::gemv_backward_error, |y - yref| / (|alpha| |op(A)| |x| + |beta| |y0|)
// componentwise on the default items (Check::blas, docs/design/verification.md).

#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/gemv/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace V = ops::gemv;

template <typename T>
struct GemvProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    Transpose trans;
    int out, red, batch, rows, cols, lda, sx, sy;
    std::size_t sa;
    T alpha, beta;
    UnifiedVector<T> A, X, Y0, Y;

    GemvProblem(Queue& q_, Transpose t, int out_, int red_, int batch_, int pad)
        : q(q_), trans(t), out(out_), red(red_), batch(batch_), rows(t == Transpose::NoTrans ? out_ : red_),
          cols(t == Transpose::NoTrans ? red_ : out_), lda(rows + pad), sx(red_ + pad), sy(out_ + pad),
          sa(std::size_t(lda) * std::size_t(cols)), alpha(MV::make(1.25, -0.5)), beta(MV::make(0.75, 0.0)),
          A(sa * std::size_t(batch_)), X(std::size_t(sx) * std::size_t(batch_)), Y0(std::size_t(sy) * std::size_t(batch_)),
          Y(std::size_t(sy) * std::size_t(batch_)) {
        (void)Av().fill_random(q, false, 919);
        (void)MV(X.data(), sx, 1, sx, sx, batch).fill_random(q, false, 920);
        (void)MV(Y0.data(), sy, 1, sy, sy, batch).fill_random(q, false, 921);
        q.wait();
    }
    MV Av() { return MV(A.data(), rows, cols, lda, int(sa), batch); }
    VectorView<T> xv() { return VectorView<T>(X.data(), red, batch, 1, sx); }
    VectorView<T> yv(UnifiedVector<T>& y) { return VectorView<T>(y.data(), out, batch, 1, sy); }
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
        (void)gemv<kBackend, T>(q, Av(), xv(), yv(Y), alpha, beta, trans);
        q.wait();
    }
    std::pair<double, int> verify() {
        return {MV::gemv_backward_error(q, Av(), trans, xv(), yv(Y0), yv(Y), MV::up(alpha), MV::up(beta)), 0};
    }
};

class GemvSpec final : public OpSpec {
public:
    std::string op() const override { return "gemv"; }
    std::vector<std::string> key_names() const override { return {V::key_names.begin(), V::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<typename T>() { return spellings(V::candidates<T>()); });
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
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            // cta and direct are both in gemv_native.cc: common only.
            "src/ops/gemv/kernels/gemv_native.cc",
            "src/ops/gemv/gemv_native.hh",
            "src/ops/gemv/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/gemv/gemv.cc" "src/sycl/device_scalar.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/gemv_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<typename T>() {
            constexpr bool kComplex = !std::is_same_v<T, typename base_type<T>::type>;
            const bool n = *key_get(req.key, "trans") == "N";
            const Transpose t = n ? Transpose::NoTrans : kComplex ? Transpose::ConjTrans : Transpose::Trans;
            GemvProblem<T> p(q, t, int(key_int(req.key, "out")), int(key_int(req.key, "red")),
                             int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, p.red);
            if (req.mode == "race") return run_race<V::GemvChoice>("gemv", p, req, tol);
            return run_arms<V::GemvChoice>("gemv", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GemvSpec)

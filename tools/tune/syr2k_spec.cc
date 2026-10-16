// syr2k for the tuner (real only): C = alpha (A B^T + B A^T) + beta C on the lower triangle, A, B
// and C0 from MatrixView::fill_random, C0's strict upper triangle then poisoned with a large finite
// value, beta = 0.75 so C is read. uplo and trans are not table keys: fixed at L and N (the
// sytrd-style update). Every ld is padded by --ld-pad. Verification is
// MatrixView::rank2k_backward_error over C's lower triangle on the default items (Check::blas with
// inner dimension 2k, docs/design/verification.md); any change to the upper triangle counts as a
// nonzero info. double has the vendor arm only (choice.hh R7).

#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/syr2k/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace R2 = ops::syr2k;

template <class T>
struct Syr2kProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, k, batch, lda, ldc;
    std::size_t sa, sc;
    T alpha, beta;
    UnifiedVector<T> A, B, C0, C;

    Syr2kProblem(Queue& q_, int n_, int k_, int batch_, int pad)
        : q(q_), n(n_), k(k_), batch(batch_), lda(n_ + pad), ldc(n_ + pad), sa(std::size_t(lda) * std::size_t(k_)),
          sc(std::size_t(ldc) * std::size_t(n_)), alpha(T(1.25)), beta(T(0.75)), A(sa * std::size_t(batch_)),
          B(sa * std::size_t(batch_)), C0(sc * std::size_t(batch_)), C(sc * std::size_t(batch_)) {
        (void)view(A, n, k, lda, sa).fill_random(q, false, 8282);
        (void)view(B, n, k, lda, sa).fill_random(q, false, 8283);
        (void)view(C0, n, n, ldc, sc).fill_random(q, false, 8284);
        q.wait();
        for (int b = 0; b < batch; ++b)
            for (int c = 1; c < n; ++c)
                for (int r = 0; r < c; ++r) C0[std::size_t(b) * sc + std::size_t(c) * std::size_t(ldc) + std::size_t(r)] = T(1e6);
    }
    MV view(UnifiedVector<T>& v, int r, int c, int ld, std::size_t s) { return MV(v.data(), r, c, ld, int(s), batch); }
    void reset() {  // beta != 0 accumulates into C
        (void)MV::copy(q, view(C, n, n, ldc, sc), view(C0, n, n, ldc, sc));
        q.wait();
    }
    // syr2k takes no workspace; probing a pin means running once (a refused pin throws first).
    std::size_t workspace() {
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)syr2k<kBackend, T>(q, view(A, n, k, lda, sa), view(B, n, k, lda, sa), view(C, n, n, ldc, sc), alpha, beta,
                                 Uplo::Lower, Transpose::NoTrans);
        q.wait();
    }
    std::pair<double, int> verify() {
        int touched = 0;  // the strict upper triangle of C must stay bit for bit as C0 left it
        for (int b : view(C, n, n, ldc, sc).default_items())
            for (int c = 1; c < n; ++c)
                for (int r = 0; r < c; ++r) {
                    const std::size_t o = std::size_t(b) * sc + std::size_t(c) * std::size_t(ldc) + std::size_t(r);
                    touched += C[o] != C0[o];
                }
        return {MV::rank2k_backward_error(q, view(A, n, k, lda, sa), view(B, n, k, lda, sa), Transpose::NoTrans,
                                          view(C0, n, n, ldc, sc), view(C, n, n, ldc, sc), Uplo::Lower, MV::up(alpha),
                                          MV::up(beta), false),
                touched};
    }
};

class Syr2kSpec final : public OpSpec {
public:
    std::string op() const override { return "syr2k"; }
    std::vector<std::string> key_names() const override { return {R2::key_names.begin(), R2::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_real_dtype("syr2k", dtype, []<class T>() { return spellings(R2::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, k, b;
        for (int v : R2::grid_n) n.push_back(std::to_string(v));
        for (int v : R2::grid_k) k.push_back(std::to_string(v));
        for (int v : R2::grid_batch) b.push_back(std::to_string(v));
        return {{"n", n}, {"k", k}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& key) const override {  // A, B, C0, C
        const double n = double(key_int(key, "n")), k = double(key_int(key, "k"));
        return (2.0 * n * k + 2.0 * n * n) * double(key_int(key, "batch")) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: triangular
            "src/ops/syr2k/kernels/syr2k_triangular_tiles.hh",
            "src/kernels/common/triangular_tiles.hh",
            // common
            "src/ops/syr2k/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/syr2k/syr2k.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/syr2k_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_real_dtype("syr2k", req.dtype, [&]<class T>() {
            Syr2kProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "k")), int(key_int(req.key, "batch")),
                              req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, 2 * p.k);
            if (req.mode == "race") return run_race<R2::Syr2kChoice>("syr2k", p, req, tol);
            return run_arms<R2::Syr2kChoice>("syr2k", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::Syr2kSpec)

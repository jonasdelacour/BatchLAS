// syr2k for the tuner (real only): C = alpha (A B^T + B A^T) + beta C on the lower triangle, A, B
// and C0's lower triangle random, C0's strict upper triangle poisoned with a large finite value,
// beta = 0.75 so C is read. uplo and trans are not table keys: fixed at L and N (the sytrd-style
// update). Every ld is padded by --ld-pad. Verification is the componentwise error
// |C - Cref| / (|alpha| (|A| |B^T| + |B| |A^T|) + |beta| |C0|) in double on items 0 and batch-1,
// over the lower-triangle entries of up to 256 sampled rows of up to 64 sampled columns; any change
// to the upper triangle counts as a nonzero info. double has the vendor arm only (choice.hh R7).

#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/syr2k/choice.hh"
#include "cell_runner.hh"
#include "residuals.hh"

#include <memory>
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
        Rng rg(8282);
        for (auto* v : {&A, &B})
            for (std::size_t i = 0; i < v->size(); ++i) (*v)[i] = T(rg.next());
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < n; ++r)
                    C0[std::size_t(b) * sc + std::size_t(c) * std::size_t(ldc) + std::size_t(r)] =
                        r >= c ? T(rg.next()) : T(1e6);
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
        double worst = 0;
        int touched = 0;
        for (int b : {0, batch - 1}) {
            const std::size_t oa = std::size_t(b) * sa, oc = std::size_t(b) * sc;
            auto in = [&](const UnifiedVector<T>& v, int r, int p) {
                return double(v[oa + std::size_t(p) * std::size_t(lda) + std::size_t(r)]);
            };
            auto at = [&](const UnifiedVector<T>& v, int r, int c) {
                return v[oc + std::size_t(c) * std::size_t(ldc) + std::size_t(r)];
            };
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < c; ++r) touched += at(C, r, c) != at(C0, r, c);
            for (int j : sample_indices(n, 64))
                for (int i : sample_indices(n, 256)) {
                    if (i < j) continue;
                    double acc = 0, mag = 0;
                    for (int p = 0; p < k; ++p) {
                        const double t1 = in(A, i, p) * in(B, j, p), t2 = in(B, i, p) * in(A, j, p);
                        acc += t1 + t2;
                        mag += ab(t1) + ab(t2);
                    }
                    const double c0 = double(at(C0, i, j));
                    const double num = ab(double(at(C, i, j)) - (double(alpha) * acc + double(beta) * c0)),
                                 den = ab(double(alpha)) * mag + ab(double(beta)) * ab(c0);
                    worst = nanmax(worst, den > 0 ? num / den : num);
                }
        }
        return {worst, touched};
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
            // family: triangular
            "src/backends/syr2k_triangular_tiles.hh",
            "src/backends/triangular_tiles.hh",
            // common
            "src/ops/syr2k/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/syr2k/syr2k.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/syr2k_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<R2::Syr2kChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_real_dtype("syr2k", req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            Syr2kProblem<T> p(*q, int(key_int(req.key, "n")), int(key_int(req.key, "k")), int(key_int(req.key, "batch")),
                              req.ld_pad);
            if (req.mode == "race") return run_race<R2::Syr2kChoice>("syr2k", p, req, Tol<T>::v);
            return run_arms<R2::Syr2kChoice>("syr2k", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::Syr2kSpec)

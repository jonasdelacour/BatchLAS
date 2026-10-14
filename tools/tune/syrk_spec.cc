// syrk for the tuner (real only): C = alpha op(A) op(A)^T + beta C on the lower triangle, A and
// C0 from MatrixView::fill_random, C0's strict upper triangle and ld padding then poisoned with a
// large finite value, beta = 0.75 so C is read. trans C is the real Trans spelled C (the key keeps
// it exact). uplo is not a table key: fixed at L. Every ld is padded by --ld-pad. Verification is
// MatrixView::gemm_backward_error over C's lower triangle on the default items,
// |C - Cref| / (|alpha| |op(A)| |op(A)^T| + |beta| |C0|) componentwise (Check::blas,
// docs/design/verification.md); any change to the upper triangle counts as a nonzero info (the arm
// wrote the half it must not).
//
// form is derived from (n, k) (choice.hh form_of), so the grid holds only consistent cells:
// trans N and T take the n x k x batch product, C only the n axis at k = batch = 1 (choice.hh).

#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include "../../src/ops/syrk/choice.hh"
#include "cell_runner.hh"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace K = ops::syrk;

template <class T>
struct SyrkProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    Transpose trans;
    int n, k, batch, ar, ac, lda, ldc;
    std::size_t sa, sc;
    T alpha, beta;
    UnifiedVector<T> A, C0, C;

    SyrkProblem(Queue& q_, Transpose t, int n_, int k_, int batch_, int pad)
        : q(q_), trans(t), n(n_), k(k_), batch(batch_), ar(t == Transpose::NoTrans ? n_ : k_),
          ac(t == Transpose::NoTrans ? k_ : n_), lda(ar + pad), ldc(n_ + pad), sa(std::size_t(lda) * std::size_t(ac)),
          sc(std::size_t(ldc) * std::size_t(n_)), alpha(T(1.25)), beta(T(0.75)), A(sa * std::size_t(batch_)),
          C0(sc * std::size_t(batch_)), C(sc * std::size_t(batch_)) {
        (void)view(A, ar, ac, lda, sa).fill_random(q, false, 7373);
        (void)view(C0, n, n, ldc, sc).fill_random(q, false, 7374);
        q.wait();
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < ldc; ++r)
                    if (r < c || r >= n) C0[std::size_t(b) * sc + std::size_t(c) * std::size_t(ldc) + std::size_t(r)] = T(1e6);
    }
    MV view(UnifiedVector<T>& v, int r, int c, int ld, std::size_t s) { return MV(v.data(), r, c, ld, int(s), batch); }
    void reset() {  // beta != 0 accumulates into C
        (void)MV::copy(q, view(C, n, n, ldc, sc), view(C0, n, n, ldc, sc));
        q.wait();
    }
    // syrk takes no workspace; probing a pin means running once (a refused pin throws first).
    std::size_t workspace() {
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)syrk<kBackend, T>(q, view(A, ar, ac, lda, sa), view(C, n, n, ldc, sc), alpha, beta, Uplo::Lower, trans);
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
        const MV a = view(A, ar, ac, lda, sa);
        const Transpose tt = trans == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
        return {MV::gemm_backward_error(q, a, trans, a, tt, view(C0, n, n, ldc, sc), view(C, n, n, ldc, sc), MV::up(alpha),
                                        MV::up(beta), {}, Uplo::Lower),
                touched};
    }
};

class SyrkSpec final : public OpSpec {
public:
    std::string op() const override { return "syrk"; }
    std::vector<std::string> key_names() const override { return {K::key_names.begin(), K::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_real_dtype("syrk", dtype, []<class T>() { return spellings(K::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        auto ints = [](const auto& xs) {
            std::vector<std::string> out;
            for (int v : xs) out.push_back(std::to_string(v));
            return out;
        };
        return {{"form", {"sq", "tall", "wide"}}, {"trans", {K::grid_trans.begin(), K::grid_trans.end()}},
                {"n", ints(K::grid_n)}, {"k", ints(K::grid_k)}, {"batch", ints(K::grid_batch)}};
    }
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        static_cast<void>(dtype);
        std::vector<CellKey> cells;
        auto add = [&](std::string_view t, int n, int k, int b) {
            cells.push_back({{"form", std::string(K::form_of(n, k))}, {"trans", std::string(t)}, {"n", std::to_string(n)},
                             {"k", std::to_string(k)}, {"batch", std::to_string(b)}});
        };
        for (std::string_view t : K::grid_trans)
            for (int n : K::grid_n) {
                if (t == "C") {
                    add(t, n, 1, 1);
                    continue;
                }
                for (int k : K::grid_k)
                    for (int b : K::grid_batch) add(t, n, k, b);
            }
        return filter_cells(*this, std::move(cells), overrides);
    }
    double bytes(const std::string& dtype, const CellKey& key) const override {  // A, C0, C
        const double n = double(key_int(key, "n")), k = double(key_int(key, "k"));
        return (n * k + 2.0 * n * n) * double(key_int(key, "batch")) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            // family: gram
            "src/backends/syrk_gram_tiles.hh",
            // family: triangular
            "src/backends/syrk_triangular_tiles.hh",
            // common
            "src/backends/triangular_tiles.hh",
            "src/ops/syrk/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/syrk/syrk.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/syrk_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<K::SyrkChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_real_dtype("syrk", req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const std::string& t = *key_get(req.key, "trans");
            const Transpose trans = t == "N" ? Transpose::NoTrans : t == "T" ? Transpose::Trans : Transpose::ConjTrans;
            const int n = int(key_int(req.key, "n")), k = int(key_int(req.key, "k"));
            if (*key_get(req.key, "form") != K::form_of(n, k))
                throw std::invalid_argument("syrk: form " + *key_get(req.key, "form") + " contradicts n, k");
            SyrkProblem<T> p(*q, trans, n, k, int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, p.k);
            if (req.mode == "race") return run_race<K::SyrkChoice>("syrk", p, req, tol);
            return run_arms<K::SyrkChoice>("syrk", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SyrkSpec)

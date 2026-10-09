// gemm for the tuner (flat-kernel-selection-phase3-plan.md §1.3, §3): random A, B and C0 in
// [-1, 1) (complex parts both), alpha with an imaginary part and beta = 1, so the epilogue reads C
// (docs/developer/agent-guide.md §10: confirm a GEMM at beta = 1). layout=strided cells pad every ld by
// max(1, --ld-pad) (an odd ld fails every aligned leg, as the parent-ld panel updates do);
// layout=packed cells are contiguous with the allocator's aligned bases. Verification is
// batchlas::verify::gemm_backward_error (docs/design/verification.md) on the default items, over
// every row of up to 64 sampled columns (always the first and last), so a lost tile edge or an
// ignored transpose is caught.
//
// The grid is demand-driven, not a lattice (plan §3): squares for every form and both layouts,
// panels and skinny shapes for the issued forms. `--grid name=v1:v2` FILTERS that grid.

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/items.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/scalar.hh>
#include <batchlas/verify/tolerance.hh>

#include "../../src/ops/gemm/choice.hh"
#include "cell_runner.hh"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace G = ops::gemm;

constexpr int kMaxCheckedCols = 64;

Transpose trans_of(const std::string& w, bool complex) {
    if (w == "N") return Transpose::NoTrans;
    if (w == "C" && complex) return Transpose::ConjTrans;
    return Transpose::Trans;  // a real scalar's C is its T (the key folds it)
}

template <class T>
struct GemmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    Transpose ta, tb;
    int m, n, k, batch;
    int ar, ac, br, bc, lda, ldb, ldc;
    std::size_t sa, sb, sc;
    T alpha, beta;
    UnifiedVector<T> A, B, C0, C;
    MV Av, Bv, C0v, Cv;

    GemmProblem(Queue& q_, Transpose ta_, Transpose tb_, int m_, int n_, int k_, int batch_, int pad)
        : q(q_), ta(ta_), tb(tb_), m(m_), n(n_), k(k_), batch(batch_),
          ar(ta_ == Transpose::NoTrans ? m_ : k_), ac(ta_ == Transpose::NoTrans ? k_ : m_),
          br(tb_ == Transpose::NoTrans ? k_ : n_), bc(tb_ == Transpose::NoTrans ? n_ : k_),
          lda(ar + pad), ldb(br + pad), ldc(m_ + pad),
          sa(std::size_t(lda) * std::size_t(ac)), sb(std::size_t(ldb) * std::size_t(bc)),
          sc(std::size_t(ldc) * std::size_t(n_)), alpha(batchlas::verify::make<T>(1.25, -0.5)),
          beta(batchlas::verify::make<T>(1.0, 0.0)),
          A(sa * std::size_t(batch_)), B(sb * std::size_t(batch_)), C0(sc * std::size_t(batch_)),
          C(sc * std::size_t(batch_)), Av(A.data(), ar, ac, lda, int(sa), batch_),
          Bv(B.data(), br, bc, ldb, int(sb), batch_), C0v(C0.data(), m_, n_, ldc, int(sc), batch_),
          Cv(C.data(), m_, n_, ldc, int(sc), batch_) {
        batchlas::verify::Rng rg(777);
        for (auto* v : {&A, &B, &C0})
            for (std::size_t i = 0; i < v->size(); ++i) {
                const double re = rg.next();
                (*v)[i] = batchlas::verify::make<T>(re, rg.next());
            }
    }
    void reset() {  // beta = 1 accumulates into C
        (void)MV::copy(q, Cv, C0v);
        q.wait();
    }
    // gemm takes no workspace; probing a pin means running once (a refused pin throws first).
    std::size_t workspace() {
        reset();
        run({});
        return 1;
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)gemm<kBackend, T>(q, Av, Bv, Cv, alpha, beta, ta, tb);
        q.wait();
    }
    // The sampled columns of op(B), C0 and C are gathered into packed host copies, so A is read
    // once per item. op(B)'s column j is B's row j when B is transposed: gather rows then.
    std::pair<double, int> verify() {
        using V = MatrixView<T, MatrixFormat::Dense>;
        using batchlas::verify::Shape;
        std::vector<int> cols;
        for (int i = 0; i < std::min(n, kMaxCheckedCols); ++i)
            cols.push_back(n <= kMaxCheckedCols ? i : int(std::int64_t(i) * (n - 1) / (kMaxCheckedCols - 1)));
        const int nc = int(cols.size());
        const bool bt = tb != Transpose::NoTrans;
        std::vector<T> bs(std::size_t(k) * std::size_t(nc)), c0s(std::size_t(m) * std::size_t(nc)), cs(c0s.size());
        double worst = 0;
        for (int b : batchlas::verify::default_items(batch)) {
            const std::size_t ob = std::size_t(b) * sb, oc = std::size_t(b) * sc;
            for (int i = 0; i < nc; ++i) {
                const std::size_t j = std::size_t(cols[std::size_t(i)]), ii = std::size_t(i);
                for (std::size_t p = 0; p < std::size_t(k); ++p)
                    bs[bt ? p * std::size_t(nc) + ii : ii * std::size_t(k) + p] =
                        bt ? B[ob + p * std::size_t(ldb) + j] : B[ob + j * std::size_t(ldb) + p];
                for (std::size_t r = 0; r < std::size_t(m); ++r) {
                    c0s[ii * std::size_t(m) + r] = C0[oc + j * std::size_t(ldc) + r];
                    cs[ii * std::size_t(m) + r] = C[oc + j * std::size_t(ldc) + r];
                }
            }
            worst = batchlas::verify::nanmax(
                worst, batchlas::verify::gemm_backward_error(
                           V(A.data() + std::size_t(b) * sa, ar, ac, lda), Shape::general, ta,
                           bt ? V(bs.data(), nc, k, nc) : V(bs.data(), k, nc, k), Shape::general, tb,
                           V(c0s.data(), m, nc, m), V(cs.data(), m, nc, m), Shape::general,
                           batchlas::verify::up(alpha), batchlas::verify::up(beta)));
        }
        return {worst, 0};
    }
};

std::vector<std::string> strs(const auto& xs) {
    std::vector<std::string> out;
    for (auto v : xs) out.push_back(std::string(v));
    return out;
}

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class GemmSpec final : public OpSpec {
public:
    std::string op() const override { return "gemm"; }
    std::vector<std::string> key_names() const override { return {G::key_names.begin(), G::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(G::candidates<T>()); });
    }
    // Every value a grid cell may take, per axis (the --grid filter's vocabulary).
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::set<int> mn(G::grid_square.begin(), G::grid_square.end());
        mn.insert(G::grid_panel_mn.begin(), G::grid_panel_mn.end());
        mn.insert(G::grid_skinny_mn.begin(), G::grid_skinny_mn.end());
        std::set<int> kk(mn);
        kk.insert(G::grid_panel_k.begin(), G::grid_panel_k.end());
        kk.insert(G::grid_skinny_k.begin(), G::grid_skinny_k.end());
        return {{"ta", {"N", "T", "C"}}, {"tb", {"N", "T", "C"}}, {"layout", {"packed", "strided"}},
                {"m", ints(mn)},         {"n", ints(mn)},         {"k", ints(kk)},
                {"batch", ints(G::grid_batch)}};
    }
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        const bool cplx = dtype == "cfloat" || dtype == "cdouble";
        const auto forms = cplx ? strs(G::grid_complex_forms) : strs(G::grid_real_forms);
        const auto panel = cplx ? strs(G::grid_complex_panel_forms) : strs(G::grid_real_panel_forms);
        std::vector<std::tuple<std::string, bool, int, int, int>> shapes;
        std::set<std::tuple<std::string, bool, int, int, int>> seen;
        auto add = [&](const std::string& f, bool packed, int m, int n, int k) {
            if (seen.insert({f, packed, m, n, k}).second) shapes.push_back({f, packed, m, n, k});
        };
        for (const std::string& f : forms) {
            for (bool packed : {false, true})
                for (int s : G::grid_square) add(f, packed, s, s, s);
            if (std::find(panel.begin(), panel.end(), f) == panel.end()) continue;
            for (int m : G::grid_panel_mn)
                for (int n : G::grid_panel_mn)
                    for (int k : G::grid_panel_k) {
                        add(f, false, m, n, k);
                        if (m >= G::grid_packed_panel_min && n >= G::grid_packed_panel_min) add(f, true, m, n, k);
                    }
            for (int mn : G::grid_skinny_mn)
                for (int k : G::grid_skinny_k) {
                    add(f, false, mn, 32, k);
                    add(f, false, 32, mn, k);
                }
        }
        const auto ax = axes();
        for (const auto& [name, values] : overrides)
            if (std::none_of(ax.begin(), ax.end(), [&](const auto& a) { return a.first == name; }))
                throw std::invalid_argument("gemm has no grid axis '" + name + "'");
        auto keep = [&](const std::string& name, const std::string& v) {
            const auto it = overrides.find(name);
            return it == overrides.end() || std::find(it->second.begin(), it->second.end(), v) != it->second.end();
        };
        std::vector<CellKey> out;
        for (const auto& [f, packed, m, n, k] : shapes)
            for (int b : G::grid_batch) {
                CellKey key{{"ta", f.substr(0, 1)}, {"tb", f.substr(1, 1)}, {"layout", packed ? "packed" : "strided"},
                            {"m", std::to_string(m)}, {"n", std::to_string(n)}, {"k", std::to_string(k)},
                            {"batch", std::to_string(b)}};
                if (std::all_of(key.begin(), key.end(), [&](const auto& kv) { return keep(kv.name, kv.value); }))
                    out.push_back(std::move(key));
            }
        if (out.empty()) throw std::invalid_argument("gemm: the --grid filters leave no cell of the declared grid");
        return out;
    }
    double bytes(const std::string& dtype, const CellKey& key) const override {  // A, B, C0, C
        const double m = double(key_int(key, "m")), n = double(key_int(key, "n")), k = double(key_int(key, "k"));
        return (m * k + k * n + 2.0 * m * n) * double(key_int(key, "batch")) * dtype_bytes(dtype);
    }
    std::string refine_key() const override { return "k"; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/gemm_kernels.cc",
            "src/sycl/gemm_kernels.hh",
            "src/sycl/gemm/accessors.hh",
            "src/sycl/gemm/epilogue_linear.hh",
            // family: reg
            "src/sycl/gemm/load_policies.hh",
            "src/sycl/gemm/register_128x128.hh",
            // family: wide
            "src/sycl/gemm/register_64x64_k16_wide.hh",
            // family: reg
            "src/sycl/gemm/register_launchers.hh",
            "src/sycl/gemm/register_tiled_common.hh",
            // family: wide
            "src/sycl/gemm/register_wide_transposed.hh",
            // family: small
            "src/sycl/gemm/small_batched.hh",
            // family: tiled
            "src/sycl/gemm/tiled_general.hh",
            // common
            "src/ops/gemm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/gemm_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<G::GemmChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            auto get = [&](std::string_view name, const char* dflt) {
                const std::string* v = key_get(req.key, name);
                return v ? *v : std::string(dflt);
            };
            constexpr bool kComplex = G::is_complex_v<T>;
            const bool packed = get("layout", "strided") == "packed";
            GemmProblem<T> p(*q, trans_of(get("ta", "N"), kComplex), trans_of(get("tb", "N"), kComplex),
                             int(key_int(req.key, "m")), int(key_int(req.key, "n")), int(key_int(req.key, "k")),
                             int(key_int(req.key, "batch")), packed ? 0 : std::max(1, req.ld_pad));
            const double tol = batchlas::verify::bound<T>(batchlas::verify::Check::blas, p.k);
            if (req.mode == "race") return run_race<G::GemmChoice>("gemm", p, req, tol);
            return run_arms<G::GemmChoice>("gemm", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GemmSpec)

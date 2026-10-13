// ormqr for the tuner (src/ops/ormqr/choice.hh): reflectors and tau from one untimed public geqrf
// (Auto) of a MatrixView::fill_random m x k matrix, MatrixView::fill_random C (complex parts both),
// every ld padded by ld_pad, block_size_hint 0 so blocked derives its WY width as a library caller's
// does. Verified on the default items: Q (m x m) formed by MatrixView::form_q from the input
// reflectors, independently of any ormqr, then op(Q) C0 (Left) or C0 op(Q) (Right) judged by
// MatrixView::gemm_backward_error (Check::blas at n = m, docs/design/verification.md).
//
// The grid is the lattice with k <= m; complex trans=T is an argument error (throw_if_undefined),
// so complex grids leave it out. blocked's time includes the public gemm and trmm of its WY
// updates, each choosing for itself.

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include "../../src/ops/ormqr/choice.hh"
#include "cell_runner.hh"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

Transpose trans_of(const std::string& w) {
    return w == "N" ? Transpose::NoTrans : (w == "T" ? Transpose::Trans : Transpose::ConjTrans);
}

template <class T>
struct OrmqrProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    Side side;
    Transpose trans;
    int m, k, nq, batch, lda, crows, ccols, ldc;
    std::size_t sa, sc;
    UnifiedVector<T> A0, A, tau, C0, C;
    UnifiedVector<T*> pA0, pA, pC0, pC;
    MV A0v, Av, C0v, Cv;

    OrmqrProblem(Queue& q_, Side s, Transpose t, int m_, int k_, int q_rhs, int batch_, int ld_pad)
        : q(q_), side(s), trans(t), m(m_), k(k_), nq(q_rhs), batch(batch_), lda(m_ + ld_pad),
          crows(s == Side::Left ? m_ : q_rhs), ccols(s == Side::Left ? q_rhs : m_), ldc(crows + ld_pad),
          sa(std::size_t(lda) * std::size_t(k_)), sc(std::size_t(ldc) * std::size_t(ccols)),
          A0(sa * std::size_t(batch_)), A(sa * std::size_t(batch_)), tau(std::size_t(k_) * std::size_t(batch_)),
          C0(sc * std::size_t(batch_)), C(sc * std::size_t(batch_)), pA0(std::size_t(batch_)), pA(std::size_t(batch_)),
          pC0(std::size_t(batch_)), pC(std::size_t(batch_)), A0v(A0.data(), m, k, lda, int(sa), batch, pA0.data()),
          Av(A.data(), m, k, lda, int(sa), batch, pA.data()),
          C0v(C0.data(), crows, ccols, ldc, int(sc), batch, pC0.data()),
          Cv(C.data(), crows, ccols, ldc, int(sc), batch, pC.data()) {
        (void)A0v.fill_random(q, false, 6161);
        (void)C0v.fill_random(q, false, 777);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, geqrf_buffer_size<kBackend, T>(q, A0v, tau.to_span())));
        (void)geqrf<kBackend, T>(q, A0v, tau.to_span(), ws.to_span());
        q.wait();
    }
    std::size_t workspace() { return ormqr_buffer_size<kBackend, T>(q, Av, Cv, side, trans, tau.to_span()); }
    void reset() {  // A too: a family that edits its reflectors in place must not hand the next arm its edit
        (void)MV::copy(q, Av, A0v);
        (void)MV::copy(q, Cv, C0v);
        q.wait();
    }
    void clear_info() {}
    void run(Span<std::byte> ws) {
        (void)ormqr<kBackend, T>(q, Av, Cv, side, trans, tau.to_span(), ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        using P = typename MV::promoted_type;
        using PV = MatrixView<P, MatrixFormat::Dense>;
        const auto items = Cv.default_items();
        auto Q = MV::form_q(q, A0v, VectorView<T>(tau.data(), k, batch, 1, k), items, m);
        auto c0 = C0v.template astype<P>(items), c = Cv.template astype<P>(items);
        const Transpose nt = Transpose::NoTrans;
        const auto all = Q.view().all_items();
        return {side == Side::Left ? PV::gemm_backward_error(q, Q.view(), trans, c0.view(), nt, c0.view(), c.view(), P(1), P(0), all)
                                   : PV::gemm_backward_error(q, c0.view(), nt, Q.view(), trans, c0.view(), c.view(), P(1), P(0), all),
                0};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class OrmqrSpec final : public OpSpec {
public:
    std::string op() const override { return "ormqr"; }
    std::vector<std::string> key_names() const override {
        return {ops::ormqr::key_names.begin(), ops::ormqr::key_names.end()};
    }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(ops::ormqr::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"side", {"L", "R"}},         {"trans", {"N", "T", "C"}},
                {"m", ints(ops::ormqr::grid_m)}, {"k", ints(ops::ormqr::grid_m)},
                {"q", ints(ops::ormqr::grid_q)}, {"batch", ints(ops::ormqr::grid_batch)}};
    }
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        const bool cplx = dtype == "cfloat" || dtype == "cdouble";
        auto cells = OpSpec::grid(dtype, overrides);
        std::erase_if(cells, [&](const CellKey& c) {
            return key_int(c, "k") > key_int(c, "m") || (cplx && *key_get(c, "trans") == "T");
        });
        if (cells.empty()) throw std::invalid_argument("ormqr: the --grid values leave no cell with k <= m");
        return cells;
    }
    double bytes(const std::string& dtype, const CellKey& c) const override {  // A0, A, C0, C
        const double m = double(key_int(c, "m")), k = double(key_int(c, "k")), q = double(key_int(c, "q"));
        return (2.0 * m * k + 2.0 * m * q) * double(key_int(c, "batch")) * dtype_bytes(dtype);
    }
    std::string refine_key() const override { return "m"; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: blocked
            "src/extensions/ormqr_blocked.cc",
            "include/batchlas/internal/ormqr_blocked.hh",
            "src/extensions/larft_wy.hh",
            // common
            "src/ops/ormqr/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/ormqr/ormqr.cc"
        // family: vendor "src/ops/ormqr/vendor.hh"
        // ormqr blocked's WY width comes from tuning_params.hh; its updates call the public gemm and trmm.
        // family: blocked "include/batchlas/tuning_params.hh"
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh"
        // family: blocked "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // family: blocked "src/ops/trmm/choice.hh" "src/ops/trmm/trmm.cc" "src/backends/trmm_triangular_tiles.hh" "src/backends/triangular_tiles.hh" "src/backends/triangular_expand.hh" "src/backends/level3_shape.hh" "src/expansion_budget.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/ormqr_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<ops::ormqr::OrmqrChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const int m = int(key_int(req.key, "m")), k = int(key_int(req.key, "k"));
            if (k > m) throw std::invalid_argument("ormqr: k > m reflectors do not fit Q's order");
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const Side side = *key_get(req.key, "side") == "R" ? Side::Right : Side::Left;
            OrmqrProblem<T> p(*q, side, trans_of(*key_get(req.key, "trans")), m, k, int(key_int(req.key, "q")),
                              int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, m);
            if (req.mode == "race") return run_race<ops::ormqr::OrmqrChoice>("ormqr", p, req, tol);
            return run_arms<ops::ormqr::OrmqrChoice>("ormqr", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::OrmqrSpec)

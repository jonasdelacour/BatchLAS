// SPDX-License-Identifier: MIT
// potrs for the tuner (docs/design/flat-kernel-selection.md §6.1): MatrixView::fill_random_spd A factored once,
// untimed, by the public potrf under Auto in the cell's uplo triangle; fill_random B; verified on the default
// items by MatrixView::solve_residual against the unfactored A (Check::solve, docs/design/verification.md).
// trsm's time includes the two public trsm calls (each choosing its own kernel); no candidate's time includes
// potrf. evidence: docs/perf/potrs.md#potrs-selection

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/potrs/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::potrs;

template <class T>
struct PotrsProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, nrhs, batch, lda, ldb;
    std::size_t sa, sb;
    Uplo uplo;
    UnifiedVector<T> A0, L, B0, X;
    UnifiedVector<T*> pA0, pL, pB0, pX;
    MV A0v, Lv, B0v, Xv;

    PotrsProblem(Queue& q_, int n_, int nrhs_, int batch_, int ld_pad, Uplo u)
        : q(q_), n(n_), nrhs(nrhs_), batch(batch_), lda(n_ + ld_pad), ldb(n_ + ld_pad),
          sa(std::size_t(lda) * std::size_t(n_)), sb(std::size_t(ldb) * std::size_t(nrhs_)), uplo(u),
          A0(sa * std::size_t(batch_)), L(sa * std::size_t(batch_)), B0(sb * std::size_t(batch_)),
          X(sb * std::size_t(batch_)), pA0(std::size_t(batch_)), pL(std::size_t(batch_)), pB0(std::size_t(batch_)),
          pX(std::size_t(batch_)), A0v(A0.data(), n, n, lda, int(sa), batch, pA0.data()),
          Lv(L.data(), n, n, lda, int(sa), batch, pL.data()), B0v(B0.data(), n, nrhs, ldb, int(sb), batch, pB0.data()),
          Xv(X.data(), n, nrhs, ldb, int(sb), batch, pX.data()) {
        (void)A0v.fill_random_spd(q, 12345);
        (void)B0v.fill_random(q, false, 777);
        (void)MV::copy(q, Lv, A0v);
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, potrf_buffer_size<kBackend, T>(q, Lv, uplo)));
        UnifiedVector<std::int32_t> info(std::size_t(batch_), 0);
        (void)potrf<kBackend, T>(q, Lv, uplo, fws.to_span(), info.to_span());
        q.wait();
        for (int b = 0; b < batch; ++b)
            if (info[std::size_t(b)] != 0) throw batchlas::convergence_error("potrs input: potrf reported a non-SPD item");
    }
    std::size_t workspace() {  // potrs takes no workspace; a refused pin throws here, before any timing
        reset();
        run({});
        return 1;
    }
    void reset() {  // the solve overwrites B with X; the factor is read only
        (void)MV::copy(q, Xv, B0v);
        q.wait();
    }
    void clear_info() {}
    void run(Span<std::byte>) {
        (void)potrs<kBackend, T>(q, Lv, Xv, uplo);
        q.wait();
    }
    std::pair<double, int> verify() { return {MV::solve_residual(q, A0v, Xv, B0v), 0}; }
};

class PotrsSpec final : public OpSpec {
public:
    std::string op() const override { return "potrs"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(S::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, r, b;
        for (int v : S::grid_n) n.push_back(std::to_string(v));
        for (int v : S::grid_nrhs) r.push_back(std::to_string(v));
        for (int v : S::grid_batch) b.push_back(std::to_string(v));
        return {{"uplo", {"L", "U"}}, {"n", n}, {"nrhs", r}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return (2.0 * n * n + 2.0 * n * double(key_int(k, "nrhs"))) * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, L, B0, X
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: fused
            "src/ops/getrs/kernels/getrs_fused.cc",
            "src/ops/getrs/getrs_native.hh",
            // common
            "src/ops/potrs/potrs.hh",
            "src/ops/potrs/kernels/potrs_trsm.cc",
            "src/ops/potrs/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/potrs/potrs.cc"
        // trsm calls the public trsm: its native kernels count too.
        // family: trsm "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/potrs_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const Uplo uplo = *key_get(req.key, "uplo") == "U" ? Uplo::Upper : Uplo::Lower;
            PotrsProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "nrhs")),
                              int(key_int(req.key, "batch")), req.ld_pad, uplo);
            const double tol = verify::bound<T>(verify::Check::solve, p.n);
            if (req.mode == "race") return run_race<S::PotrsChoice>("potrs", p, req, tol);
            return run_arms<S::PotrsChoice>("potrs", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::PotrsSpec)

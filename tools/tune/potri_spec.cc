// potri for the tuner (docs/design/flat-kernel-selection.md §6.1): MatrixView::fill_random_spd A factored once,
// untimed, by the public potrf under Auto; the timed call turns the Lower factor into the Lower triangle of the
// inverse, which is mirrored onto a copy and checked on the default items by MatrixView::solve_residual with an
// empty B (|| A0 X - I ||; Check::solve, docs/design/verification.md). A is restored from the factor before every
// run, so an arm that writes nothing cannot inherit a correct inverse. Both families' times include the public
// trsm (and, for fused, the one-launch solve) they call; no candidate's time includes potrf.
// Only Lower is tuned: Upper runs the same kernels on the transposed sweep and the table has no uplo key.

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/error.hh>
#include <batchlas/blas/functions/potri.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/potri/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace I = ops::potri;

template <class T>
struct PotriProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, L, A, X;
    UnifiedVector<T*> pA0, pL, pA, pX;
    MV A0v, Lv, Av, Xv;
    UnifiedVector<std::int32_t> info;

    PotriProblem(Queue& q_, int n_, int batch_, int ld_pad)
        : q(q_), n(n_), batch(batch_), ld(n_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), L(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)),
          X(stride * std::size_t(batch_)), pA0(std::size_t(batch_)), pL(std::size_t(batch_)), pA(std::size_t(batch_)),
          pX(std::size_t(batch_)), A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()),
          Lv(L.data(), n, n, ld, int(stride), batch, pL.data()), Av(A.data(), n, n, ld, int(stride), batch, pA.data()),
          Xv(X.data(), n, n, ld, int(stride), batch, pX.data()), info(std::size_t(batch_), 0) {
        (void)A0v.fill_random_spd(q, 12345);
        (void)MV::copy(q, Lv, A0v);
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, potrf_buffer_size<kBackend, T>(q, Lv, Uplo::Lower)));
        (void)potrf<kBackend, T>(q, Lv, Uplo::Lower, fws.to_span(), info.to_span());
        q.wait();
        for (int b = 0; b < batch; ++b)
            if (info[std::size_t(b)] != 0) throw batchlas::convergence_error("potri input: potrf reported a non-SPD item");
    }
    std::size_t workspace() { return potri_buffer_size<kBackend, T>(q, Av, Uplo::Lower); }
    void reset() {  // the call overwrites A with the inverse
        (void)MV::copy(q, Av, Lv);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)potri<kBackend, T>(q, Av, Uplo::Lower, false, ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        (void)MV::copy(q, Xv, Av);
        if constexpr (is_std_complex_v<T>) (void)Xv.hermitize(q, Uplo::Lower);
        else (void)Xv.symmetrize(q, Uplo::Lower);
        q.wait();
        return {MV::solve_residual(q, A0v, Xv, MV(nullptr, 0, 0, 1)), bad};
    }
};

class PotriSpec final : public OpSpec {
public:
    std::string op() const override { return "potri"; }
    std::vector<std::string> key_names() const override { return {I::key_names.begin(), I::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(I::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, b;
        for (int v : I::grid_n) n.push_back(std::to_string(v));
        for (int v : I::grid_batch) b.push_back(std::to_string(v));
        return {{"n", n}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return 5.0 * n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, L, A, X and the identity workspace
    }
    FrontierPlan frontier() const override { return {1}; }  // as getri: the two families cross once per row
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
            "src/ops/potri/kernels/potri_native.cc",
            "src/ops/potri/potri_native.hh",
            "src/ops/potrs/potrs.hh",
            "src/ops/potri/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/potri/potri.cc"
        // blocked calls the public trsm: its native kernels count too.
        // family: blocked "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/potri_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            PotriProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::solve, p.n);
            if (req.mode == "race") return run_race<I::PotriChoice>("potri", p, req, tol);
            return run_arms<I::PotriChoice>("potri", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::PotriSpec)

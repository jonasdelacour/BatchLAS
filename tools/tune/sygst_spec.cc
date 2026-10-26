// sygst for the tuner (docs/design/flat-kernel-selection.md §6.1): a Hermitian A0 (fill_random) and an
// SPD B factored once, untimed, by the public potrf under Auto. The timed call is itype 1, Lower:
// A := L^-1 A0 L^-H. It is checked on the default items by ||A0 - L C L^H|| / ||A0||
// (MatrixView::similarity_residual, Check::factorization) with C the result mirrored and L the factor
// with its upper triangle zeroed. A is restored from A0 before every run. Only itype 1, Lower is tuned:
// the table has no itype or uplo key. evidence: docs/perf/sygvd.md#sygst-measured

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/sygst/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace I = ops::sygst;

template <class T>
struct SygstProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, L, A, X;
    UnifiedVector<T*> pA0, pL, pA, pX;
    MV A0v, Lv, Av, Xv;
    UnifiedVector<std::int32_t> info;

    SygstProblem(Queue& q_, int n_, int batch_, int ld_pad)
        : q(q_), n(n_), batch(batch_), ld(n_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), L(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)),
          X(stride * std::size_t(batch_)), pA0(std::size_t(batch_)), pL(std::size_t(batch_)), pA(std::size_t(batch_)),
          pX(std::size_t(batch_)), A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()),
          Lv(L.data(), n, n, ld, int(stride), batch, pL.data()), Av(A.data(), n, n, ld, int(stride), batch, pA.data()),
          Xv(X.data(), n, n, ld, int(stride), batch, pX.data()), info(std::size_t(batch_), 0) {
        (void)A0v.fill_random(q, true, 4242);
        (void)Lv.fill_random_spd(q, 12345);
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, potrf_buffer_size<kBackend, T>(q, Lv, Uplo::Lower)));
        (void)potrf<kBackend, T>(q, Lv, Uplo::Lower, fws.to_span(), info.to_span());
        (void)Lv.triangularize(q, Uplo::Lower, Diag::NonUnit);
        q.wait();
        for (int b = 0; b < batch; ++b)
            if (info[std::size_t(b)] != 0) throw std::runtime_error("sygst input: potrf reported a non-SPD item");
    }
    std::size_t workspace() { return sygst_buffer_size<kBackend, T>(q, Av, Lv, 1, Uplo::Lower); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {}
    void run(Span<std::byte> ws) {
        (void)sygst<kBackend, T>(q, Av, Lv, 1, Uplo::Lower, ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        (void)MV::copy(q, Xv, Av);
        (void)Xv.hermitize(q, Uplo::Lower);
        q.wait();
        return {MV::similarity_residual(q, A0v, Lv, Xv), 0};
    }
};

class SygstSpec final : public OpSpec {
public:
    std::string op() const override { return "sygst"; }
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
        return 4.0 * n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, L, A, X
    }
    FrontierPlan frontier() const override { return {1}; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/ops/sygst/choice.hh",  // vendor (LAPACKE, host only) has no GPU source of its own
            // family: blocked
            "src/ops/sygst/sygst_native.hh",
            "src/ops/sygst/kernels/sygst_diag.cc",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/sygst/sygst.cc"
        // twosided calls the public trsm (itype 1) and trmm (2, 3): their native kernels count too.
        // family: twosided "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // blocked also calls the public hemm/symm, her2k/syr2k and trmm (and, through them, gemm).
        // family: blocked "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh" "src/ops/hemm/hemm.cc" "src/ops/hemm/choice.hh" "src/ops/symm/symm.cc" "src/ops/symm/choice.hh" "src/ops/her2k/her2k.cc" "src/ops/her2k/choice.hh" "src/ops/syr2k/syr2k.cc" "src/ops/syr2k/choice.hh" "src/ops/syr2k/kernels/syr2k_triangular_tiles.hh" "src/ops/trmm/trmm.cc" "src/ops/trmm/choice.hh" "src/ops/trmm/kernels/trmm_triangular_tiles.hh" "src/kernels/common/triangular_expand.hh" "src/kernels/common/triangular_tiles.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/sygst_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            SygstProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::factorization, p.n);
            if (req.mode == "race") return run_race<I::SygstChoice>("sygst", p, req, tol);
            return run_arms<I::SygstChoice>("sygst", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SygstSpec)

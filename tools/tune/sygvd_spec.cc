// sygvd for the tuner (docs/design/flat-kernel-selection.md §6.1): a Hermitian A0 (fill_random) and an SPD
// B0 (fill_random_spd). The timed call is itype 1, Upper (gfn2-xTB's dsygvd call; uplo is not a key, and
// potrf routes Upper apart from Lower), with the jobz of the cell's key: it factors B, so both A and B are
// restored from A0 and B0 before every run, and every candidate's time includes potrf.
// jobz = V is checked on the default items by ||A0 X - B0 X diag(w)|| / (||A0|| ||X||) (the reference
// product and MatrixView::solve_residual, Check::eigen_residual); jobz = N by info only.

#include <batchlas/blas/functions/sygvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/sygvd/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace I = ops::sygvd;

template <class T>
struct SygvdProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using R = typename base_type<T>::type;
    Queue& q;
    int n, batch, ld;
    JobType jobz;
    std::size_t stride;
    UnifiedVector<T> A0, B0, A, B, P;
    UnifiedVector<T*> pA0, pB0, pA, pB, pP;
    MV A0v, B0v, Av, Bv, Pv;
    UnifiedVector<R> w;
    UnifiedVector<std::int32_t> info;

    SygvdProblem(Queue& q_, int n_, int batch_, int ld_pad, JobType jobz_)
        : q(q_), n(n_), batch(batch_), ld(n_ + ld_pad), jobz(jobz_), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), B0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)),
          B(stride * std::size_t(batch_)), P(stride * std::size_t(batch_)), pA0(std::size_t(batch_)),
          pB0(std::size_t(batch_)), pA(std::size_t(batch_)), pB(std::size_t(batch_)), pP(std::size_t(batch_)),
          A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()),
          B0v(B0.data(), n, n, ld, int(stride), batch, pB0.data()), Av(A.data(), n, n, ld, int(stride), batch, pA.data()),
          Bv(B.data(), n, n, ld, int(stride), batch, pB.data()), Pv(P.data(), n, n, ld, int(stride), batch, pP.data()),
          w(std::size_t(n_) * std::size_t(batch_)), info(std::size_t(batch_), 0) {
        (void)A0v.fill_random(q, true, 4242);
        (void)B0v.fill_random_spd(q, 12345);
        q.wait();
    }
    std::size_t workspace() { return sygvd_buffer_size<kBackend, T>(q, Av, Bv, w.to_span(), 1, jobz, Uplo::Upper); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        (void)MV::copy(q, Bv, B0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)sygvd<kBackend, T>(q, Av, Bv, w.to_span(), 1, jobz, Uplo::Upper, ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        if (jobz != JobType::EigenVectors) return {0.0, bad};
        (void)MV::product(q, B0v, Transpose::NoTrans, Av, Transpose::NoTrans, Pv);
        q.wait();
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) Pv(i, j, b) *= T(w[std::size_t(b) * n + j]);
        return {MV::solve_residual(q, A0v, Av, Pv), bad};
    }
};

class SygvdSpec final : public OpSpec {
public:
    std::string op() const override { return "sygvd"; }
    std::vector<std::string> key_names() const override { return {I::key_names.begin(), I::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(I::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, b;
        for (int v : I::grid_n) n.push_back(std::to_string(v));
        for (int v : I::grid_batch) b.push_back(std::to_string(v));
        return {{"jobz", {"N", "V"}}, {"n", n}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return 5.0 * n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, B0, A, B, P
    }
    FrontierPlan frontier() const override { return {1}; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/ops/sygvd/choice.hh",
            // family: compose
            "src/ops/sygvd/kernels/sygvd_native.cc",
            "src/ops/sygvd/sygvd_native.hh",
            // family: vendor
            "src/backends/cusolver_sygvd.cc",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/sygvd/sygvd.cc"
        // compose calls the public potrf, sygst, syev, trsm and trmm, each selecting its own kernel.
        // family: compose "src/ops/sygvd/kernels/sygvd_compose.cc" "src/ops/sygst/sygst.cc" "src/ops/sygst/kernels/sygst_blocked.cc" "src/ops/sygst/kernels/sygst_twosided.cc" "src/ops/syev/syev.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/sygvd_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const JobType jobz = *key_get(req.key, "jobz") == "V" ? JobType::EigenVectors : JobType::NoEigenVectors;
            SygvdProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad, jobz);
            const double tol = verify::bound<T>(verify::Check::eigen_residual, p.n);
            if (req.mode == "race") return run_race<I::SygvdChoice>("sygvd", p, req, tol);
            return run_arms<I::SygvdChoice>("sygvd", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SygvdSpec)

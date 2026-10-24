// SPDX-License-Identifier: MIT
// getrs for the tuner (docs/design/flat-kernel-selection.md §6.1): MatrixView::fill_random_pivoted A
// factored once, untimed, by the public getrf under Auto (factor_bench's protocol), fill_random B,
// NoTrans, verified on the default items by MatrixView::lu_solve_residual against those factors
// (Check::solve, docs/design/verification.md). blocked's time includes the public trsm it calls; no
// candidate's time includes getrf.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/getrs/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::getrs;

template <typename T>
struct GetrsProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, nrhs, batch, lda, ldb;
    std::size_t sa, sb;
    UnifiedVector<T> A0, A, B0, X;
    UnifiedVector<T*> pA0, pA, pB0, pX;
    MV A0v, Av, B0v, Xv;
    UnifiedVector<std::int64_t> piv;

    GetrsProblem(Queue& q_, int n_, int nrhs_, int batch_, int ld_pad)
        : q(q_), n(n_), nrhs(nrhs_), batch(batch_), lda(n_ + ld_pad), ldb(n_ + ld_pad),
          sa(std::size_t(lda) * std::size_t(n_)), sb(std::size_t(ldb) * std::size_t(nrhs_)),
          A0(sa * std::size_t(batch_)), A(sa * std::size_t(batch_)), B0(sb * std::size_t(batch_)),
          X(sb * std::size_t(batch_)), pA0(std::size_t(batch_)), pA(std::size_t(batch_)), pB0(std::size_t(batch_)),
          pX(std::size_t(batch_)), A0v(A0.data(), n, n, lda, int(sa), batch, pA0.data()),
          Av(A.data(), n, n, lda, int(sa), batch, pA.data()), B0v(B0.data(), n, nrhs, ldb, int(sb), batch, pB0.data()),
          Xv(X.data(), n, nrhs, ldb, int(sb), batch, pX.data()), piv(std::size_t(n_) * std::size_t(batch_), 0) {
        (void)A0v.fill_random_pivoted(q, 12345);
        (void)B0v.fill_random(q, false, 777);
        (void)MV::copy(q, Av, A0v);
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, getrf_buffer_size<kBackend, T>(q, Av)));
        UnifiedVector<std::int32_t> info(std::size_t(batch_), 0);
        (void)getrf<kBackend, T>(q, Av, piv.to_span(), fws.to_span(), info.to_span());
        q.wait();
        for (int b = 0; b < batch; ++b)
            if (info[std::size_t(b)] != 0) throw batchlas::convergence_error("getrs input: getrf reported a singular item");
    }
    std::size_t workspace() { return getrs_buffer_size<kBackend, T>(q, Av, Xv, Transpose::NoTrans); }
    void reset() {  // the solve overwrites B with X; A and the pivots are read only
        (void)MV::copy(q, Xv, B0v);
        q.wait();
    }
    void clear_info() {}
    void run(Span<std::byte> ws) {
        (void)getrs<kBackend, T>(q, Av, Xv, Transpose::NoTrans, piv.to_span(), ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        const VectorView<std::int32_t> p32(reinterpret_cast<std::int32_t*>(piv.data()), n, batch, 1, n);
        return {MV::lu_solve_residual(q, Av, p32, Transpose::NoTrans, Xv, B0v), 0};
    }
};

class GetrsSpec final : public OpSpec {
public:
    std::string op() const override { return "getrs"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<typename T>() { return spellings(S::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, r, b;
        for (int v : S::grid_n) n.push_back(std::to_string(v));
        for (int v : S::grid_nrhs) r.push_back(std::to_string(v));
        for (int v : S::grid_batch) b.push_back(std::to_string(v));
        return {{"n", n}, {"nrhs", r}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return (2.0 * n * n + 2.0 * n * double(key_int(k, "nrhs"))) * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, A, B0, X
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            // family: cta
            "src/ops/getrs/kernels/getrs_fused.cc",
            // family: blocked
            "src/ops/getrs/kernels/getrs_native.cc",
            // common
            "src/ops/getrs/getrs_native.hh",
            "src/ops/getrs/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/getrs/getrs.cc"
        // blocked calls the public trsm: its native kernels count too.
        // family: blocked "src/kernels/common/lu_laswp.hh"
        // family: blocked "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/getrs_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<typename T>() {
            GetrsProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "nrhs")),
                              int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::solve, p.n);
            if (req.mode == "race") return run_race<S::GetrsChoice>("getrs", p, req, tol);
            return run_arms<S::GetrsChoice>("getrs", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GetrsSpec)

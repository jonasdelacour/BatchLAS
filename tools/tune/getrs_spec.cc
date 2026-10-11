// getrs for the tuner (docs/design/flat-kernel-selection.md §6.1): A factored once, untimed, by
// the public getrf under Auto (factor_bench's protocol), random complex B, NoTrans, the solve
// residual against the unfactored A on items 0 and batch-1. blocked's time includes the public
// trsm it calls; no candidate's time includes getrf.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/getrs/choice.hh"
#include "cell_runner.hh"
#include "lu_residuals.hh"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::getrs;

template <class T>
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
        fill_lu<T>(A0.data(), n, lda, sa, batch, 12345);
        Rng rg(777);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < nrhs; ++c)
                for (int r = 0; r < n; ++r)
                    B0[std::size_t(b) * sb + std::size_t(c) * std::size_t(ldb) + std::size_t(r)] =
                        mk<T>(rg.next(), rg.next());
        (void)MV::copy(q, Av, A0v);
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, getrf_buffer_size<kBackend, T>(q, Av)));
        UnifiedVector<std::int32_t> info(std::size_t(batch_), 0);
        (void)getrf<kBackend, T>(q, Av, piv.to_span(), fws.to_span(), info.to_span());
        q.wait();
        for (int b = 0; b < batch; ++b)
            if (info[std::size_t(b)] != 0) throw std::runtime_error("getrs input: getrf reported a singular item");
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
        return {lu_solve_residual<T>(X.data(), B0.data(), A0.data(), n, nrhs, lda, sa, ldb, sb, batch), 0};
    }
};

class GetrsSpec final : public OpSpec {
public:
    std::string op() const override { return "getrs"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(S::candidates<T>()); });
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
            // family: cta
            "src/extensions/getrs_fused.cc",
            // family: blocked
            "src/extensions/getrs_native.cc",
            // common
            "src/extensions/getrs_native.hh",
            "src/ops/getrs/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/getrs/getrs.cc"
        // blocked calls the public trsm: its native kernels count too.
        // family: blocked "src/extensions/lu_laswp.hh"
        // family: blocked "src/sycl/trsm_native.cc" "src/sycl/trsm_native.hh" "src/sycl/trsm_sg_left.cc" "src/sycl/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/getrs_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<S::GetrsChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            GetrsProblem<T> p(*q, int(key_int(req.key, "n")), int(key_int(req.key, "nrhs")),
                              int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<S::GetrsChoice>("getrs", p, req, Tol<T>::v);
            return run_arms<S::GetrsChoice>("getrs", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GetrsSpec)

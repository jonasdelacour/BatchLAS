// getri for the tuner (docs/design/flat-kernel-selection.md §6.1): A factored once, untimed, by
// the public getrf under Auto, then C = inv(A) checked as || A0 C - I || on items 0 and batch-1
// (random probe columns past n = 645, lu_residuals.hh). C is reset to a large finite poison before
// every run, so an arm that writes nothing cannot inherit a correct inverse. blocked's time
// includes the public trsm it calls; no candidate's time includes getrf.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/getri/choice.hh"
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

namespace I = ops::getri;

template <class T>
struct GetriProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, LU, A, C0, C;
    UnifiedVector<T*> pLU, pA, pC0, pC;
    MV LUv, Av, C0v, Cv;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<std::int32_t> info;

    GetriProblem(Queue& q_, int n_, int batch_, int ld_pad)
        : q(q_), n(n_), batch(batch_), ld(n_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), LU(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)),
          C0(stride * std::size_t(batch_)), C(stride * std::size_t(batch_)), pLU(std::size_t(batch_)),
          pA(std::size_t(batch_)), pC0(std::size_t(batch_)), pC(std::size_t(batch_)),
          LUv(LU.data(), n, n, ld, int(stride), batch, pLU.data()), Av(A.data(), n, n, ld, int(stride), batch, pA.data()),
          C0v(C0.data(), n, n, ld, int(stride), batch, pC0.data()), Cv(C.data(), n, n, ld, int(stride), batch, pC.data()),
          piv(std::size_t(n_) * std::size_t(batch_), 0), info(std::size_t(batch_), 0) {
        fill_lu<T>(A0.data(), n, ld, stride, batch, 12345);
        std::copy(A0.data(), A0.data() + stride * std::size_t(batch), LU.data());
        std::fill(C0.data(), C0.data() + stride * std::size_t(batch), mk<T>(1e4, -1e4));
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, getrf_buffer_size<kBackend, T>(q, LUv)));
        (void)getrf<kBackend, T>(q, LUv, piv.to_span(), fws.to_span(), info.to_span());
        q.wait();
        for (int b = 0; b < batch; ++b)
            if (info[std::size_t(b)] != 0) throw std::runtime_error("getri input: getrf reported a singular item");
    }
    std::size_t workspace() { return getri_buffer_size<kBackend, T>(q, Av); }
    void reset() {  // A is read only, restored anyway so one arm cannot corrupt the next one's input
        (void)MV::copy(q, Av, LUv);
        (void)MV::copy(q, Cv, C0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)getri<kBackend, T>(q, Av, Cv, piv.to_span(), ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        return {lu_solve_residual<T>(C.data(), nullptr, A0.data(), n, n, ld, stride, ld, stride, batch), bad};
    }
};

class GetriSpec final : public OpSpec {
public:
    std::string op() const override { return "getri"; }
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
        return 5.0 * n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, LU, A, C0, C
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            // family: blocked
            "src/extensions/getri_blocked.cc",
            // common
            "src/extensions/getri_native.hh",
            "src/ops/getri/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/getri/getri.cc"
        // blocked calls the public trsm: its native kernels count too.
        // family: blocked "src/sycl/trsm_native.cc" "src/sycl/trsm_native.hh" "src/sycl/trsm_sg_left.cc" "src/sycl/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/getri_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<I::GetriChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            GetriProblem<T> p(*q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<I::GetriChoice>("getri", p, req, Tol<T>::v);
            return run_arms<I::GetriChoice>("getri", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GetriSpec)

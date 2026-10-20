// syev for the tuner: a graded Hermitian A from MatrixView::fill_graded_hermitian (condition 1e6, a
// different eigenbasis per item, so item 0 and batch-1 differ), Lower named, the ld padding poisoned
// with a large finite value. The upper triangle holds the mirror, not poison: blocked (n <= 32) and
// two_stage read it (known defect 14,
// docs/design/known-defects.md#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle), and a
// poisoned run would drop them from those rows for a defect, not a time. Verified on the default
// items (docs/design/verification.md): the eigenvalues against host LAPACKE (verify::eigenvalues,
// MatrixView::values_error relative to max |lambda|; values), and with jobz=V
// MatrixView::eigen_residual (eigen_residual) and MatrixView::orthogonality (orthogonality_rotations),
// each as a fraction of its bound, so the cell's residual is the worst and its tolerance 1. Without
// LAPACKE every arm is bad. blocked's and two_stage's times include the public gemm, syr2k and trmm
// (sytrd_blocked, stedc, ormqr_blocked) and, two_stage, geqrf and ormqr (sytrd_sy2sb), each choosing
// for itself.

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/syev/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::syev;

constexpr double kPoison = 1e6;

template <class T>
struct SyevProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using R = typename base_type<T>::type;
    Queue& q;
    JobType jobz;
    int n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, A;
    UnifiedVector<T*> pA0, pA;
    MV A0v, Av;
    UnifiedVector<R> w;
    UnifiedVector<std::int32_t> info;
    std::vector<std::vector<double>> ref;  // host eigenvalues of the default items (indexed by item), lazily

    SyevProblem(Queue& q_, JobType j, int n_, int batch_, int ld_pad)
        : q(q_), jobz(j), n(n_), batch(batch_), ld(n_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)), pA0(std::size_t(batch_)),
          pA(std::size_t(batch_)), A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()),
          Av(A.data(), n, n, ld, int(stride), batch, pA.data()), w(std::size_t(n_) * std::size_t(batch_)),
          info(std::size_t(batch_), 0) {
        (void)MV(A0.data(), ld, n, ld, int(stride), batch).fill(q, MV::make(kPoison, 0.0));
        (void)A0v.fill_graded_hermitian(q, 6.0, 1000);
        (void)MV::copy(q, MV(A.data(), ld, n, ld, int(stride), batch), MV(A0.data(), ld, n, ld, int(stride), batch));
        q.wait();  // the padding keeps its poison
    }
    std::size_t workspace() { return syev_buffer_size<kBackend, T>(q, Av, w.to_span(), jobz, Uplo::Lower); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {  // an earlier arm's w must not pass for this one's
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
        for (R& x : w) x = R(kPoison);
    }
    void run(Span<std::byte> ws) {
        (void)syev<kBackend, T>(q, Av, w.to_span(), jobz, Uplo::Lower, ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        const auto items = A0v.default_items();
        if (ref.empty()) {
            ref.assign(std::size_t(batch), {});
            for (int b : items)
                if (!verify::eigenvalues(A0v, b, ref[std::size_t(b)])) {
                    ref.clear();
                    return {std::nan(""), bad};
                }
        }
        double scale = 0;
        for (int b : items)
            for (double x : ref[std::size_t(b)]) scale = std::max(scale, std::fabs(x));
        const VectorView<R> wv(w.data(), n, batch);
        double worst = MV::values_error(wv, ref, scale) / verify::bound<T>(verify::Check::values, n);
        if (jobz != JobType::EigenVectors) return {worst, bad};
        worst = verify::nanmax(worst, MV::eigen_residual(q, A0v, Av, wv) / verify::bound<T>(verify::Check::eigen_residual, n));
        const double orth = MV::orthogonality(q, Av) / verify::bound<T>(verify::Check::orthogonality_rotations, n);
        return {verify::nanmax(worst, orth), bad};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class SyevSpec final : public OpSpec {
public:
    std::string op() const override { return "syev"; }
    std::vector<std::string> key_names() const override { return {S::key_names.begin(), S::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(S::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"jobz", {"N", "V"}}, {"n", ints(S::grid_n)}, {"batch", ints(S::grid_batch)}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A0, A
        const double n = double(key_int(k, "n"));
        return 2.0 * n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            "src/kernels/common/sg_partition/backend_amdgcn.hh",
            "src/kernels/common/sg_partition/backend_amdgcn_core.hh",
            "src/kernels/common/sg_partition/backend_generic.hh",
            "src/kernels/common/sg_partition/backend_nvptx.hh",
            "src/kernels/common/sg_partition/backend_spirv.hh",
            "src/kernels/common/sg_partition/backend_sscp.hh",
            "src/kernels/common/sg_partition/sg_partition.hh",
            // family: cta
            "src/ops/syev/kernels/syev_cta.cc",
            "src/ops/ormqr/kernels/ormqr_cta.cc",
            // family: cta_fused
            "src/ops/syev/kernels/syev_cta_fused.cc",
            // family: jacobi
            "src/ops/syev/kernels/syev_jacobi_cta.cc",
            // family: blocked
            "src/ops/syev/kernels/syev_blocked.cc",
            "src/ops/syev/sytrd/sytrd_blocked.cc",
            "src/ops/syev/sytrd/latrd_lower_panel.cc",
            // family: two_stage
            "src/ops/syev/kernels/syev_two_stage.cc",
            "src/kernels/common/two_stage_common.hh",
            "src/ops/syev/sytrd/sytrd_sy2sb.cc",
            "src/ops/syev/sytrd/sytrd_sb2st.cc",
            "src/ops/syev/sytrd/sytrd_sb2st_cta.cc",
            "src/ops/syev/sytrd/sytrd_sb2st_cta.hh",
            "src/ops/syev/sytrd/sytrd_sb2st_hh.cc",
            "src/ops/syev/sytrd/sytrd_sb2st_hh.hh",
            // common
            "src/ops/syev/sytrd/sytrd_cta.cc",
            "src/ops/syev/sytrd/sytrd_cta_device.hh",
            "src/ops/syev/steqr/steqr.cc",
            "src/ops/syev/steqr/steqr_cta.cc",
            "src/ops/syev/steqr/steqr_cta_device.hh",
            "src/ops/syev/steqr/steqr_wg.cc",
            "src/ops/syev/steqr/steqr_internal.hh",
            "src/ops/syev/stedc/stedc.cc",
            "src/ops/syev/stedc/stedc_internal.hh",
            "src/ops/syev/stedc/stedc_levels_plan.hh",
            "src/ops/syev/stedc/stedc_merge_cta.cc",
            "src/ops/syev/stedc/stedc_merge_kernels.cc",
            "src/ops/syev/stedc/stedc_merge_kernels.hh",
            "src/ops/syev/stedc/stedc_secular.cc",
            "src/ops/syev/stedc/stedc_secular.hh",
            "src/ops/ormqr/kernels/ormqr_blocked.cc",
            "src/kernels/common/larft_wy.hh",
            "src/ops/syev/kernels/uplo_mirror.cc",
            "src/ops/syev/uplo_mirror.hh",
            "src/ops/syev/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/syev/syev.cc"
        // blocked and two_stage call the public gemm and trmm (stedc, ormqr_blocked); blocked syr2k, two_stage geqrf.
        // family: blocked "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/ops/trmm/trmm.cc" "src/ops/syr2k/syr2k.cc"
        // family: blocked "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/copy_async.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh" "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh"
        // family: two_stage "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/ops/trmm/trmm.cc"
        // family: two_stage "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/copy_async.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh" "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh"
        // family: two_stage "src/ops/geqrf/kernels/geqrf_blocked.cc" "src/ops/geqrf/kernels/geqrf_cta.cc" "src/ops/geqrf/kernels/geqrf_tiny.cc" "src/ops/geqrf/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/syev_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const JobType jobz = *key_get(req.key, "jobz") == "V" ? JobType::EigenVectors : JobType::NoEigenVectors;
            const int n = int(key_int(req.key, "n"));
            SyevProblem<T> p(q, jobz, n, int(key_int(req.key, "batch")), req.ld_pad);
            if (req.mode == "race") return run_race<S::SyevChoice>("syev", p, req, 1.0);
            return run_arms<S::SyevChoice>("syev", p, req, 1.0);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SyevSpec)

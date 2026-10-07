// syev for the tuner: a graded random Hermitian A = D S D (S uniform in [-1, 1), complex parts both;
// D = diag(10^(-3 i / (n-1)))), Lower named, the ld padding poisoned with a large finite value. The
// upper triangle holds the mirror, not poison: blocked (n <= 32) and two_stage read it (known defect
// 14, docs/design/known-defects.md#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle),
// and a poisoned run would drop them from those rows for a defect, not a time. jobz=V: max_j ||A v_j - w_j v_j|| / ||A||_F and max |V^H V - I| over up to
// 32 sampled columns; jobz=N: the sorted eigenvalues against host LAPACK, max |w - w_ref| / max
// |w_ref|; both on items 0 and batch-1. Items cycle through four matrices, item 0 and batch-1 differ.
// blocked's and two_stage's times include the public gemm, syr2k and trmm (sytrd_blocked, stedc,
// ormqr_blocked) and, two_stage, geqrf and ormqr (sytrd_sy2sb), each choosing for itself.

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/syev/choice.hh"
#include "cell_runner.hh"
#include "host_reference.hh"
#include "residuals.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace S = ops::syev;

constexpr int kVariants = 4;
constexpr int kMaxChecked = 32;
constexpr double kPoison = 1e6;

// item b's matrix: item 0 and batch-1 always differ, so a wrong batch stride shows.
inline int variant_of(int b, int batch) { return b == 0 ? 0 : b == batch - 1 ? 1 : 2 + (b & 1); }

template <class T>
struct SyevProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using R = typename base_type<T>::type;
    using D = typename Prom<T>::type;
    Queue& q;
    JobType jobz;
    int n, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, A;
    UnifiedVector<T*> pA0, pA;
    MV A0v, Av;
    UnifiedVector<R> w;
    UnifiedVector<std::int32_t> info;
    std::vector<std::vector<double>> ref;  // jobz=N: host eigenvalues of items 0 and batch-1, lazily

    SyevProblem(Queue& q_, JobType j, int n_, int batch_, int ld_pad)
        : q(q_), jobz(j), n(n_), batch(batch_), ld(n_ + ld_pad), stride(std::size_t(ld) * std::size_t(n_)),
          A0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)), pA0(std::size_t(batch_)),
          pA(std::size_t(batch_)), A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()),
          Av(A.data(), n, n, ld, int(stride), batch, pA.data()), w(std::size_t(n_) * std::size_t(batch_)),
          info(std::size_t(batch_), 0) {
        std::vector<std::vector<T>> var(kVariants, std::vector<T>(stride, mk<T>(kPoison, 0.0)));
        for (int v = 0; v < kVariants; ++v) {
            Rng rg(1000 + std::uint64_t(v));
            for (int c = 0; c < n; ++c)
                for (int r = c; r < n; ++r) {
                    const double g = std::pow(10.0, -3.0 * double(r + c) / double(std::max(1, n - 1)));
                    const double re = g * rg.next(), im = r == c ? 0.0 : g * rg.next();
                    var[v][std::size_t(c) * std::size_t(ld) + std::size_t(r)] = mk<T>(re, im);
                    var[v][std::size_t(r) * std::size_t(ld) + std::size_t(c)] = mk<T>(re, -im);
                }
        }
        for (int b = 0; b < batch; ++b) std::copy(var[variant_of(b, batch)].begin(), var[variant_of(b, batch)].end(),
                                                  A0.data() + std::size_t(b) * stride);
        std::copy(A0.data(), A0.data() + A0.size(), A.data());  // the padding keeps its poison
    }
    std::size_t workspace() { return syev_buffer_size<kBackend, T>(q, Av, w.to_span(), jobz, Uplo::Lower); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        (void)syev<kBackend, T>(q, Av, w.to_span(), jobz, Uplo::Lower, ws, info.to_span());
        q.wait();
    }
    // The Hermitian matrix item b represents, from its Lower triangle.
    std::vector<D> full(int b) const {
        std::vector<D> a(std::size_t(n) * std::size_t(n));
        const std::size_t o = std::size_t(b) * stride;
        for (int c = 0; c < n; ++c)
            for (int r = c; r < n; ++r) {
                const D v = up(A0[o + std::size_t(c) * std::size_t(ld) + std::size_t(r)]);
                a[std::size_t(c) * n + r] = v;
                a[std::size_t(r) * n + c] = cj(v);
            }
        return a;
    }
    double vectors_residual(int b) const {
        const std::vector<D> a = full(b);
        double fro = 0;
        for (const D& x : a) fro += ab(x) * ab(x);
        fro = std::sqrt(fro);
        const std::size_t o = std::size_t(b) * stride;
        auto v = [&](int r, int c) { return up(A[o + std::size_t(c) * std::size_t(ld) + std::size_t(r)]); };
        const std::vector<int> cols = sample_indices(n, kMaxChecked);
        double worst = 0;
        for (int j : cols) {
            const double wj = double(w[std::size_t(b) * n + j]);
            double num = 0;
            for (int r = 0; r < n; ++r) {
                D acc = D(0);
                for (int k = 0; k < n; ++k) acc += a[std::size_t(k) * n + r] * v(k, j);
                num += ab(acc - wj * v(r, j)) * ab(acc - wj * v(r, j));
            }
            worst = nanmax(worst, fro > 0 ? std::sqrt(num) / fro : std::sqrt(num));
            for (int i : cols) {
                D dot = D(0);
                for (int r = 0; r < n; ++r) dot += cj(v(r, i)) * v(r, j);
                worst = nanmax(worst, ab(dot - D(i == j ? 1.0 : 0.0)));
            }
        }
        return worst;
    }
    double values_residual(int slot, int b) {
        if (ref.empty()) ref.resize(2);
        if (ref[slot].empty()) {
            std::vector<D> a = full(b);
            if (!host_eigenvalues(n, a, ref[slot])) return std::nan("");
        }
        std::vector<double> got(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) got[std::size_t(i)] = double(w[std::size_t(b) * n + i]);
        std::sort(got.begin(), got.end());
        double scale = 0, worst = 0;
        for (double x : ref[slot]) scale = std::max(scale, std::fabs(x));
        for (int i = 0; i < n; ++i) worst = nanmax(worst, std::fabs(got[std::size_t(i)] - ref[slot][std::size_t(i)]));
        return scale > 0 ? worst / scale : worst;
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        double worst = 0;
        for (int slot : {0, 1}) {
            const int b = slot == 0 ? 0 : batch - 1;
            worst = nanmax(worst, jobz == JobType::EigenVectors ? vectors_residual(b) : values_residual(slot, b));
        }
        return {worst, bad};
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
            // family: cta
            "src/extensions/syev_cta.cc",
            "src/extensions/ormqr_cta.cc",
            // family: cta_fused
            "src/extensions/syev_cta_fused.cc",
            // family: jacobi
            "src/extensions/syev_jacobi_cta.cc",
            // family: blocked
            "src/extensions/syev_blocked.cc",
            "src/extensions/sytrd_blocked.cc",
            "src/extensions/latrd_lower_panel.cc",
            // family: two_stage
            "src/extensions/syev_two_stage.cc",
            "src/extensions/two_stage_common.hh",
            "src/extensions/sytrd_sy2sb.cc",
            "src/extensions/sytrd_sb2st.cc",
            "src/extensions/sytrd_sb2st_cta.cc",
            "src/extensions/sytrd_sb2st_cta.hh",
            "src/extensions/sytrd_sb2st_hh.cc",
            "src/extensions/sytrd_sb2st_hh.hh",
            // common
            "src/extensions/sytrd_cta.cc",
            "src/extensions/sytrd_cta_device.hh",
            "src/extensions/steqr.cc",
            "src/extensions/steqr_cta.cc",
            "src/extensions/steqr_cta_device.hh",
            "src/extensions/steqr_wg.cc",
            "src/extensions/steqr_internal.hh",
            "src/extensions/stedc.cc",
            "src/extensions/stedc_internal.hh",
            "src/extensions/stedc_levels_plan.hh",
            "src/extensions/stedc_merge_cta.cc",
            "src/extensions/stedc_merge_kernels.cc",
            "src/extensions/stedc_merge_kernels.hh",
            "src/extensions/stedc_secular.cc",
            "src/extensions/stedc_secular.hh",
            "src/extensions/ormqr_blocked.cc",
            "src/extensions/larft_wy.hh",
            "src/extensions/uplo_mirror.cc",
            "src/extensions/uplo_mirror.hh",
            "src/ops/syev/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/syev/syev.cc"
        // blocked and two_stage call the public gemm and trmm (stedc, ormqr_blocked); blocked syr2k, two_stage geqrf.
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/extensions/trmm.cc" "src/extensions/syr2k.cc"
        // family: blocked "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh"
        // family: two_stage "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/extensions/trmm.cc"
        // family: two_stage "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh"
        // family: two_stage "src/extensions/geqrf_blocked.cc" "src/extensions/geqrf_cta.cc" "src/extensions/geqrf_tiny.cc" "src/ops/geqrf/choice.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/syev_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<S::SyevChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const JobType jobz = *key_get(req.key, "jobz") == "V" ? JobType::EigenVectors : JobType::NoEigenVectors;
            const int n = int(key_int(req.key, "n"));
            SyevProblem<T> p(*q, jobz, n, int(key_int(req.key, "batch")), req.ld_pad);
            // Backward error grows like n eps: scale the O(1)-input bound, which a wrong answer (O(1)) still fails.
            const double tol = Tol<T>::v * std::max(1.0, double(n) / 32.0);
            if (req.mode == "race") return run_race<S::SyevChoice>("syev", p, req, tol);
            return run_arms<S::SyevChoice>("syev", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SyevSpec)

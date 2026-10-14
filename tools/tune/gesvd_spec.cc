// gesvd for the tuner. herm=N: MatrixView::fill_random m x n A (complex parts both); herm=L|U:
// MatrixView::fill_graded_hermitian (condition 100) with both triangles valid (the vendor family
// reads all of A). Every item differs. Verified on the default items (docs/design/verification.md):
// the singular values against host LAPACKE (verify::singular_values, MatrixView::values_error
// relative to s_max; values at n = min(m, n)), and with vectors MatrixView::svd_residual
// (factorization at n = max(m, n)) and the orthogonality_rotations of U's columns and of Vh's rows
// (the columns of Vh^H, formed by MatrixView::product with an identity), each as a fraction of its
// bound, so the cell's residual is the worst and its tolerance 1. Without LAPACKE every arm is bad.
//
// batch is no key (choice.hh), so it is derived from the shape, never a grid axis: the largest
// power of two in [128, 16384] with m n batch <= 2^24, saturating per docs/developer/agent-guide.md
// §10 (gesvd saturates from batch ~4096 at n = 32, docs/perf/gesvd.md). herm=L|U cells are square
// only, and vec=thin only where it survives canonicalisation (m != n): the key is canonical.
// cta's and blocked's times include the syev_cta / syev_blocked they call, and blocked's the public
// gemm, trmm and syr2k underneath.

#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/tolerance.hh>

#include "../../src/ops/gesvd/choice.hh"
#include "cell_runner.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace G = ops::gesvd;

int gesvd_batch(std::int64_t m, std::int64_t n) {
    std::int64_t b = 16384;
    while (b > 128 && b * m * n > (std::int64_t(1) << 24)) b /= 2;
    return int(b);
}

SvdVectors job_of(const std::string& vec) {
    return vec == "none" ? SvdVectors::None : vec == "thin" ? SvdVectors::Thin : SvdVectors::All;
}

template <class T>
struct GesvdProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using R = typename base_type<T>::type;
    Queue& q;
    std::optional<Uplo> herm;
    SvdVectors job;
    int m, n, k, batch, ld, urows, ucols, vrows, vcols, ldu, ldv;
    std::size_t sa, su, sv;
    UnifiedVector<T> A0, A, U, Vh;
    UnifiedVector<T*> pA0, pA, pU, pVh;
    MV A0v, Av, Uv, Vhv;
    UnifiedVector<R> s;
    UnifiedVector<std::int32_t> info;
    std::vector<std::vector<double>> ref;  // host singular values of the default items (indexed by item), lazily

    GesvdProblem(Queue& q_, std::optional<Uplo> h, SvdVectors j, int m_, int n_, int ld_pad)
        : q(q_), herm(h), job(j), m(m_), n(n_), k(std::min(m_, n_)), batch(gesvd_batch(m_, n_)), ld(m_ + ld_pad),
          urows(j == SvdVectors::None ? 1 : m_), ucols(j == SvdVectors::All ? m_ : j == SvdVectors::Thin ? k : 1),
          vrows(j == SvdVectors::All ? n_ : j == SvdVectors::Thin ? k : 1), vcols(j == SvdVectors::None ? 1 : n_),
          ldu(urows + ld_pad), ldv(vrows + ld_pad), sa(std::size_t(ld) * std::size_t(n_)),
          su(std::size_t(ldu) * std::size_t(ucols)), sv(std::size_t(ldv) * std::size_t(vcols)),
          A0(sa * std::size_t(batch)), A(sa * std::size_t(batch)), U(su * std::size_t(batch)),
          Vh(sv * std::size_t(batch)), pA0(std::size_t(batch)), pA(std::size_t(batch)), pU(std::size_t(batch)),
          pVh(std::size_t(batch)), A0v(A0.data(), m, n, ld, int(sa), batch, pA0.data()),
          Av(A.data(), m, n, ld, int(sa), batch, pA.data()), Uv(U.data(), urows, ucols, ldu, int(su), batch, pU.data()),
          Vhv(Vh.data(), vrows, vcols, ldv, int(sv), batch, pVh.data()), s(std::size_t(k) * std::size_t(batch)),
          info(std::size_t(batch), 0) {
        if (herm) (void)A0v.fill_graded_hermitian(q, 2.0, 2000);
        else (void)A0v.fill_random(q, false, 2000);
        q.wait();
    }
    // The vendor binding throws `unsupported` for a shape its batched call cannot take (m or n
    // above 32, Thin) although can_run admits it: that is a refused pin here, not a crashed arm.
    // Values-only non-square input faults the device instead (CUDA_ERROR_ILLEGAL_ADDRESS, a worker
    // restart per cell), so that pin is refused up front. Auto never sends the shape there.
    // evidence: docs/design/known-defects.md#defect-15-cusolver-gesvdjbatched-faults-on-values-only-non-square-input
    std::size_t workspace() {
        std::string src;
        if (job == SvdVectors::None && m != n && select::detail::pin_text("gesvd", &src) == std::optional<std::string>("vendor"))
            throw std::invalid_argument("gesvd vendor: values-only non-square faults (known defect 15)");
        try {
            return herm ? gesvd_buffer_size<kBackend, T>(q, Av, s.to_span(), Uv, Vhv, job, job, *herm)
                        : gesvd_buffer_size<kBackend, T>(q, Av, s.to_span(), Uv, Vhv, job, job);
        } catch (const batchlas::unsupported& e) {
            throw std::invalid_argument(e.what());
        }
    }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {
        for (int b = 0; b < batch; ++b) info[std::size_t(b)] = 0;
    }
    void run(Span<std::byte> ws) {
        if (herm)
            (void)gesvd<kBackend, T>(q, Av, s.to_span(), Uv, Vhv, job, job, *herm, ws, info.to_span());
        else
            (void)gesvd<kBackend, T>(q, Av, s.to_span(), Uv, Vhv, job, job, ws, info.to_span());
        q.wait();
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        const auto items = A0v.default_items();
        if (ref.empty()) {
            ref.assign(std::size_t(batch), {});
            for (int b : items)
                if (!verify::singular_values(A0v, b, ref[std::size_t(b)])) {
                    ref.clear();
                    return {std::nan(""), bad};
                }
        }
        double scale = 0;
        for (int b : items) scale = std::max(scale, ref[std::size_t(b)].empty() ? 0.0 : ref[std::size_t(b)][0]);
        const VectorView<R> sval(s.data(), k, batch);
        double worst = MV::values_error(sval, ref, scale) / verify::bound<T>(verify::Check::values, k);
        if (job == SvdVectors::None) return {worst, bad};
        const double rec = MV::svd_residual(q, A0v, Uv, sval, Vhv) / verify::bound<T>(verify::Check::factorization, std::max(m, n));
        const double ou = MV::orthogonality(q, Uv) / verify::bound<T>(verify::Check::orthogonality_rotations, ucols);
        auto vh = Vhv.template astype<T>(items);
        Matrix<T, MatrixFormat::Dense> vhh(vcols, vrows, int(items.size()));
        auto eye = Matrix<T, MatrixFormat::Dense>::Identity(vrows, int(items.size()));
        (void)MV::product(q, vh.view(), Transpose::ConjTrans, eye.view(), Transpose::NoTrans, vhh.view());
        q.wait();
        const double ov = MV::orthogonality(q, vhh.view(), vhh.view().all_items()) /
                          verify::bound<T>(verify::Check::orthogonality_rotations, vrows);
        worst = verify::nanmax(worst, verify::nanmax(rec, verify::nanmax(ou, ov)));
        return {worst, bad};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class GesvdSpec final : public OpSpec {
public:
    std::string op() const override { return "gesvd"; }
    std::vector<std::string> key_names() const override { return {G::key_names.begin(), G::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(G::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> herm(G::grid_herm.begin(), G::grid_herm.end()), vec(G::grid_vec.begin(), G::grid_vec.end());
        return {{"herm", herm}, {"vec", vec}, {"m", ints(G::grid_mn)}, {"n", ints(G::grid_mn)}};
    }
    // The lattice less the cells no call can key: Hermitian is square, and square Thin keys as all.
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        std::vector<CellKey> out;
        for (CellKey& c : OpSpec::grid(dtype, overrides)) {
            const bool square = key_int(c, "m") == key_int(c, "n");
            if ((*key_get(c, "herm") != "N" && !square) || (*key_get(c, "vec") == "thin" && (square || *key_get(c, "herm") != "N")))
                continue;
            out.push_back(std::move(c));
        }
        if (out.empty()) throw std::invalid_argument("gesvd: the --grid values leave no keyable cell");
        return out;
    }
    double bytes(const std::string& dtype, const CellKey& key) const override {  // A0, A, U, Vh at the derived batch
        const double m = double(key_int(key, "m")), n = double(key_int(key, "n"));
        const std::string vec = *key_get(key, "vec");
        const double uv = vec == "none" ? 2.0 : vec == "thin" ? (m + n) * std::min(m, n) : m * m + n * n;
        return (2.0 * m * n + uv) * double(gesvd_batch(std::int64_t(m), std::int64_t(n))) * dtype_bytes(dtype);
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/sycl/sscp_target.hh",
            "include/batchlas/util/group-collectives.hh",
            "src/extensions/sg_compat.hh",
            "src/extensions/sg_partition/backend_amdgcn.hh",
            "src/extensions/sg_partition/backend_amdgcn_core.hh",
            "src/extensions/sg_partition/backend_generic.hh",
            "src/extensions/sg_partition/backend_nvptx.hh",
            "src/extensions/sg_partition/backend_spirv.hh",
            "src/extensions/sg_partition/backend_sscp.hh",
            "src/extensions/sg_partition/sg_partition.hh",
            // family: jacobi
            "src/extensions/gesvdj_cta.cc",
            // family: cta
            "src/extensions/gebrd_cta.cc",
            "src/extensions/ormqr_cta.cc",
            // family: blocked
            "src/extensions/gebrd_blocked.cc",
            "src/extensions/bdsdc.cc",
            "src/extensions/ormbr.cc",
            // common
            "src/extensions/gesvd_blocked.cc",
            "src/extensions/gesvd_native.hh",
            "src/extensions/gebrd.cc",
            "src/ops/gesvd/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/gesvd/gesvd.cc"
        // cta runs syev_cta (steqr); blocked syev_blocked, stedc (bdsdc) and the public gemm, trmm and syr2k.
        // family: cta "src/extensions/syev_cta.cc" "src/extensions/sytrd_cta.cc" "src/extensions/sytrd_cta_device.hh" "src/extensions/steqr.cc" "src/extensions/steqr_cta.cc" "src/extensions/steqr_cta_device.hh" "src/extensions/steqr_wg.cc"
        // family: blocked "src/extensions/syev_blocked.cc" "src/extensions/sytrd_blocked.cc" "src/extensions/latrd_lower_panel.cc" "src/extensions/sytrd_cta.cc" "src/extensions/steqr.cc" "src/extensions/steqr_wg.cc"
        // family: blocked "src/extensions/stedc.cc" "src/extensions/stedc_merge_cta.cc" "src/extensions/stedc_merge_kernels.cc" "src/extensions/stedc_secular.cc" "src/extensions/ormqr_blocked.cc" "src/extensions/uplo_mirror.cc"
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/ops/trmm/trmm.cc" "src/ops/syr2k/syr2k.cc"
        // family: blocked "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/gesvd_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const std::string h = *key_get(req.key, "herm");
            const std::optional<Uplo> herm =
                h == "N" ? std::nullopt : std::optional<Uplo>(h == "U" ? Uplo::Upper : Uplo::Lower);
            const int m = int(key_int(req.key, "m")), n = int(key_int(req.key, "n"));
            GesvdProblem<T> p(*q, herm, job_of(*key_get(req.key, "vec")), m, n, req.ld_pad);
            if (req.mode == "race") return run_race<G::GesvdChoice>("gesvd", p, req, 1.0);
            return run_arms<G::GesvdChoice>("gesvd", p, req, 1.0);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GesvdSpec)

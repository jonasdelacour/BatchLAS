// gesvd for the tuner. herm=N: a random m x n A (uniform in [-1, 1), complex parts both) with graded
// columns, A(:, j) scaled by 10^(-j / (n-1)); herm=L|U: a graded random Hermitian D S D with both
// triangles valid (the vendor family reads all of A). Verification on items 0 and batch-1: the
// singular values against host LAPACK, max |s - s_ref| / s_ref[0], and with vectors the
// reconstruction ||A(:, j) - U S Vh(:, j)|| / ||A||_F plus max |U^H U - I| and |Vh Vh^H - I| over up
// to 32 sampled columns and rows. Items cycle through four matrices; item 0 and batch-1 differ.
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

#include "../../src/ops/gesvd/choice.hh"
#include "cell_runner.hh"
#include "host_reference.hh"
#include "residuals.hh"

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

constexpr int kVariants = 4;
constexpr int kMaxChecked = 32;

inline int variant_of(int b, int batch) { return b == 0 ? 0 : b == batch - 1 ? 1 : 2 + (b & 1); }

int gesvd_batch(std::int64_t m, std::int64_t n) {
    std::int64_t b = 16384;
    while (b > 128 && b * m * n > (std::int64_t(1) << 24)) b /= 2;
    return int(b);
}

template <class T>
T conj_of(T x) {
    if constexpr (std::is_same_v<T, typename base_type<T>::type>) return x;
    else return std::conj(x);
}

SvdVectors job_of(const std::string& vec) {
    return vec == "none" ? SvdVectors::None : vec == "thin" ? SvdVectors::Thin : SvdVectors::All;
}

template <class T>
struct GesvdProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using R = typename base_type<T>::type;
    using D = typename Prom<T>::type;
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
    std::vector<std::vector<double>> ref;  // host singular values of items 0 and batch-1, lazily

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
        std::vector<std::vector<T>> var(kVariants, std::vector<T>(sa, mk<T>(0.0, 0.0)));
        for (int v = 0; v < kVariants; ++v) {
            Rng rg(2000 + std::uint64_t(v));
            auto at = [&](int r, int c) -> T& { return var[v][std::size_t(c) * std::size_t(ld) + std::size_t(r)]; };
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < m; ++r) {
                    if (herm && r < c) continue;
                    const double g = herm ? std::pow(10.0, -double(r + c) / double(std::max(1, n - 1)))
                                          : std::pow(10.0, -double(c) / double(std::max(1, n - 1)));
                    at(r, c) = herm && r == c ? mk<T>(g * rg.next(), 0.0) : mk<T>(g * rg.next(), g * rg.next());
                }
            if (herm)
                for (int c = 0; c < n; ++c)
                    for (int r = 0; r < c; ++r) at(r, c) = conj_of(at(c, r));
        }
        for (int b = 0; b < batch; ++b)
            std::copy(var[variant_of(b, batch)].begin(), var[variant_of(b, batch)].end(), A0.data() + std::size_t(b) * sa);
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
    D a0(int b, int r, int c) const { return up(A0[std::size_t(b) * sa + std::size_t(c) * std::size_t(ld) + std::size_t(r)]); }
    D u(int b, int r, int c) const { return up(U[std::size_t(b) * su + std::size_t(c) * std::size_t(ldu) + std::size_t(r)]); }
    D vh(int b, int r, int c) const { return up(Vh[std::size_t(b) * sv + std::size_t(c) * std::size_t(ldv) + std::size_t(r)]); }
    double values_residual(int slot, int b) {
        if (ref.empty()) ref.resize(2);
        if (ref[slot].empty()) {
            std::vector<D> a(std::size_t(m) * std::size_t(n));
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < m; ++r) a[std::size_t(c) * m + r] = a0(b, r, c);
            if (!host_singular_values(m, n, a, ref[slot])) return std::nan("");
        }
        double worst = 0;
        for (int i = 0; i < k; ++i)
            worst = nanmax(worst, std::fabs(double(s[std::size_t(b) * k + i]) - ref[slot][std::size_t(i)]));
        return ref[slot][0] > 0 ? worst / ref[slot][0] : worst;
    }
    double vectors_residual(int b) const {
        double fro = 0;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < m; ++r) fro += ab(a0(b, r, c)) * ab(a0(b, r, c));
        fro = std::sqrt(fro);
        double worst = 0;
        for (int j : sample_indices(n, kMaxChecked)) {
            double num = 0;
            for (int r = 0; r < m; ++r) {
                D acc = D(0);
                for (int l = 0; l < k; ++l) acc += u(b, r, l) * double(s[std::size_t(b) * k + l]) * vh(b, l, j);
                num += ab(acc - a0(b, r, j)) * ab(acc - a0(b, r, j));
            }
            worst = nanmax(worst, fro > 0 ? std::sqrt(num) / fro : std::sqrt(num));
        }
        const auto uc = sample_indices(ucols, kMaxChecked), vr = sample_indices(vrows, kMaxChecked);
        for (int i : uc)
            for (int j : uc) {
                D dot = D(0);
                for (int r = 0; r < m; ++r) dot += cj(u(b, r, i)) * u(b, r, j);
                worst = nanmax(worst, ab(dot - D(i == j ? 1.0 : 0.0)));
            }
        for (int i : vr)
            for (int j : vr) {
                D dot = D(0);
                for (int c = 0; c < n; ++c) dot += vh(b, i, c) * cj(vh(b, j, c));
                worst = nanmax(worst, ab(dot - D(i == j ? 1.0 : 0.0)));
            }
        return worst;
    }
    std::pair<double, int> verify() {
        int bad = 0;
        for (int b = 0; b < batch; ++b) bad += info[std::size_t(b)] != 0;
        double worst = 0;
        for (int slot : {0, 1}) {
            const int b = slot == 0 ? 0 : batch - 1;
            worst = nanmax(worst, values_residual(slot, b));
            if (job != SvdVectors::None) worst = nanmax(worst, vectors_residual(b));
        }
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
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/ops/gemm/choice.hh" "src/extensions/trmm.cc" "src/extensions/syr2k.cc"
        // family: blocked "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh" "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/gesvd_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<G::GesvdChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            const std::string h = *key_get(req.key, "herm");
            const std::optional<Uplo> herm =
                h == "N" ? std::nullopt : std::optional<Uplo>(h == "U" ? Uplo::Upper : Uplo::Lower);
            const int m = int(key_int(req.key, "m")), n = int(key_int(req.key, "n"));
            GesvdProblem<T> p(*q, herm, job_of(*key_get(req.key, "vec")), m, n, req.ld_pad);
            // cta solves the normal equations, so its U loses eps kappa^2 (1.1e-4 for float at 32 x 32,
            // evidence: docs/perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa): a 10x looser
            // bound than syev's, which a 1% error in one entry still fails (4.7e-3 at 16 x 16).
            const double tol = 10.0 * Tol<T>::v * std::max(1.0, double(std::max(m, n)) / 32.0);
            if (req.mode == "race") return run_race<G::GesvdChoice>("gesvd", p, req, tol);
            return run_arms<G::GesvdChoice>("gesvd", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GesvdSpec)

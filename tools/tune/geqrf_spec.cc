// geqrf for the tuner (docs/design/flat-kernel-selection.md#phase-5-geqrf): random complex A in
// [-1, 1), ld = m + ld_pad. Verified on items 0 and batch-1 by || A0 - Q R ||_F / || A0 ||_F over
// sampled columns, Q applied on the host from the returned reflectors and tau, plus the unitarity
// of every reflector (qr_common.hh). tau is poisoned before the verified run.
//
// The key has no batch: a cell runs at qr_batch (qr_common.hh). Its grid is not a lattice (sq
// x grid_n, tall x grid_n x grid_aspect, wide x grid_wide_n x grid_wide_aspect), and a key maps to
// the smallest m of its aspect bucket; a key no (m, n) reproduces is left out. `--grid` FILTERS it.
// blocked's time includes the public gemm of its trailing updates, which chooses for itself.

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/geqrf/choice.hh"
#include "cell_runner.hh"
#include "qr_common.hh"
#include "residuals.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

constexpr double kCheckBudget = 4e8;  // host multiply-adds per verified item

// The rows of a (form, n, aspect) key: the smallest m whose aspect_of is `aspect`.
std::optional<int> rows_of(const std::string& form, int n, int aspect) {
    long m = n;
    if (form == "tall") m = aspect >= 2 ? long(n) * aspect : n + std::max(1, n / 2);
    else if (form == "wide") m = aspect >= 2 ? n / aspect : n - std::max(1, n / 4);
    else if (aspect != 1) return std::nullopt;
    const bool shape_ok = m >= 1 && (form == "tall") == (m > n) && (form == "wide") == (m < n);
    if (!shape_ok || ops::geqrf::aspect_of(m, long(n)) != aspect) return std::nullopt;
    return int(m);
}

template <class T>
struct GeqrfProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using D = typename Prom<T>::type;
    Queue& q;
    int m, n, k, batch, ld;
    std::size_t stride;
    UnifiedVector<T> A0, A, tau;
    UnifiedVector<T*> pA0, pA;
    MV A0v, Av;

    GeqrfProblem(Queue& q_, int m_, int n_, int batch_, int ld_pad)
        : q(q_), m(m_), n(n_), k(std::min(m_, n_)), batch(batch_), ld(m_ + ld_pad),
          stride(std::size_t(ld) * std::size_t(n_)), A0(stride * std::size_t(batch_)), A(stride * std::size_t(batch_)),
          tau(std::size_t(k) * std::size_t(batch_)), pA0(std::size_t(batch_)), pA(std::size_t(batch_)),
          A0v(A0.data(), m, n, ld, int(stride), batch, pA0.data()),
          Av(A.data(), m, n, ld, int(stride), batch, pA.data()) {
        Rng rg(4242);
        for (std::size_t i = 0; i < A0.size(); ++i) A0[i] = mk<T>(rg.next(), rg.next());
    }
    std::size_t workspace() { return geqrf_buffer_size<kBackend, T>(q, Av, tau.to_span()); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {  // a tau left by an earlier arm must not pass for this one's
        for (std::size_t i = 0; i < tau.size(); ++i) tau[i] = mk<T>(7.0, -3.0);
    }
    void run(Span<std::byte> ws) {
        (void)geqrf<kBackend, T>(q, Av, tau.to_span(), ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        const int cap = int(std::clamp(kCheckBudget / (double(m) * double(n)), 2.0, 32.0));
        double worst = 0;
        for (int b : {0, batch - 1}) {
            const std::size_t o = std::size_t(b) * stride;
            const Reflectors<T> R{A.data(), tau.data(), o, std::size_t(b) * std::size_t(k), ld, m};
            double num = 0, den = 0, unit = 0;
            for (int i = 0; i < k; ++i) unit = nanmax(unit, R.unitarity(i));
            for (int j : sample_idx(n, cap)) {
                std::vector<D> x(std::size_t(m), D(0));
                for (int r = 0; r <= std::min(j, m - 1); ++r) x[std::size_t(r)] = up(A[o + std::size_t(j) * ld + r]);
                for (int i = std::min(j, k - 1); i >= 0; --i) R.left(x, i, false);
                for (int r = 0; r < m; ++r) {
                    const D a = up(A0[o + std::size_t(j) * ld + r]);
                    num += ab(x[std::size_t(r)] - a) * ab(x[std::size_t(r)] - a);
                    den += ab(a) * ab(a);
                }
            }
            if (std::isnan(num) || std::isnan(den)) return {std::nan(""), 0};
            worst = nanmax(worst, nanmax(den > 0 ? std::sqrt(num / den) : std::sqrt(num), unit));
        }
        return {worst, 0};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class GeqrfSpec final : public OpSpec {
public:
    std::string op() const override { return "geqrf"; }
    std::vector<std::string> key_names() const override {
        return {ops::geqrf::key_names.begin(), ops::geqrf::key_names.end()};
    }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(ops::geqrf::candidates<T>()); });
    }
    // grid_wide_n and grid_wide_aspect are subsets of grid_n and grid_aspect.
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"form", {"sq", "tall", "wide"}},
                {"n", ints(ops::geqrf::grid_n)},
                {"aspect", ints(ops::geqrf::grid_aspect)}};
    }
    std::vector<CellKey> grid(const std::string& dtype,
                              const std::map<std::string, std::vector<std::string>>& overrides) const override {
        static_cast<void>(dtype);
        const auto ax = axes();
        for (const auto& [name, values] : overrides)
            if (std::none_of(ax.begin(), ax.end(), [&](const auto& a) { return a.first == name; }))
                throw std::invalid_argument("geqrf has no grid axis '" + name + "'");
        auto keep = [&](const std::string& name, const std::string& v) {
            const auto it = overrides.find(name);
            return it == overrides.end() || std::find(it->second.begin(), it->second.end(), v) != it->second.end();
        };
        std::vector<CellKey> out;
        auto add = [&](const char* form, int n, int aspect) {
            const CellKey key{{"form", form}, {"n", std::to_string(n)}, {"aspect", std::to_string(aspect)}};
            if (rows_of(form, n, aspect) &&
                std::all_of(key.begin(), key.end(), [&](const auto& kv) { return keep(kv.name, kv.value); }))
                out.push_back(key);
        };
        for (int n : ops::geqrf::grid_n) add("sq", n, 1);
        for (int n : ops::geqrf::grid_n)
            for (int a : ops::geqrf::grid_aspect) add("tall", n, a);
        for (int n : ops::geqrf::grid_wide_n)
            for (int a : ops::geqrf::grid_wide_aspect) add("wide", n, a);
        if (out.empty()) throw std::invalid_argument("geqrf: the --grid filters leave no cell of the declared grid");
        return out;
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A0, A (tau is k per item)
        const int n = int(key_int(k, "n"));
        const double m = double(rows_of(*key_get(k, "form"), n, int(key_int(k, "aspect"))).value_or(n));
        const double item = m * double(n) * dtype_bytes(dtype);
        return 2.0 * item * double(qr_batch(item));
    }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            // family: tiny
            "src/extensions/geqrf_tiny.cc",
            "src/extensions/geqrf_tiny_device.hh",
            // family: cta
            "src/extensions/geqrf_cta.cc",
            // family: blocked
            "src/extensions/geqrf_blocked.cc",
            "src/extensions/larft_wy.hh",
            // common
            "src/extensions/geqrf_native.hh",
            "src/extensions/geqrf_cta_device.hh",
            "src/extensions/geqrf_panel_reg_device.hh",
            "src/ops/geqrf/can_run.hh",
            "src/ops/geqrf/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/geqrf/geqrf.cc" "src/util/resident_capacity.hh"
        // family: tiny "src/extensions/tiny_device.hh"
        // geqrf blocked's trailing updates call the public gemm: its kernels and selection count too.
        // family: blocked "src/sycl/gemm_kernels.cc" "src/sycl/gemm_kernels.hh" "src/sycl/gemm/accessors.hh" "src/sycl/gemm/epilogue_linear.hh" "src/sycl/gemm/load_policies.hh" "src/sycl/gemm/register_128x128.hh" "src/sycl/gemm/register_64x64_k16_wide.hh"
        // family: blocked "src/sycl/gemm/register_launchers.hh" "src/sycl/gemm/register_tiled_common.hh" "src/sycl/gemm/register_wide_transposed.hh" "src/sycl/gemm/small_batched.hh" "src/sycl/gemm/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/geqrf_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<ops::geqrf::GeqrfChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const int n = int(key_int(req.key, "n"));
            const auto m = rows_of(*key_get(req.key, "form"), n, int(key_int(req.key, "aspect")));
            if (!m) throw std::invalid_argument("geqrf: no (m, n) has key " + key_arg(req.key));
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            GeqrfProblem<T> p(*q, *m, n, qr_batch(double(*m) * double(n) * sizeof(T)), req.ld_pad);
            if (req.mode == "race") return run_race<ops::geqrf::GeqrfChoice>("geqrf", p, req, qr_tol<T>(*m));
            return run_arms<ops::geqrf::GeqrfChoice>("geqrf", p, req, qr_tol<T>(*m));
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GeqrfSpec)

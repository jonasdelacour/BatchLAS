// geqrf for the tuner (docs/design/flat-kernel-selection.md#phase-5-geqrf): MatrixView::fill_random A
// (complex parts both), ld = m + ld_pad. Verified on the default items by MatrixView::qr_residual
// (factorization) and the orthogonality of Q's first min(m, n) columns from MatrixView::form_q
// (orthogonality), each as a fraction of its bound at n = m (docs/design/verification.md), so the
// cell's residual is the worse of the two and its tolerance 1. tau is poisoned before the verified run.
//
// The key has no batch: a cell runs at qr_batch (spec.hh). Its grid is not a lattice (sq
// x grid_n, tall x grid_n x grid_aspect, wide x grid_wide_n x grid_wide_aspect), and a key maps to
// the smallest m of its aspect bucket; a key no (m, n) reproduces is left out. `--grid` FILTERS it.
// blocked's time includes the public gemm of its trailing updates, which chooses for itself.

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/geqrf/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

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
        (void)A0v.fill_random(q, false, 4242);
        q.wait();
    }
    std::size_t workspace() { return geqrf_buffer_size<kBackend, T>(q, Av, tau.to_span()); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {  // a tau left by an earlier arm must not pass for this one's
        for (std::size_t i = 0; i < tau.size(); ++i) tau[i] = MV::make(7.0, -3.0);
    }
    void run(Span<std::byte> ws) {
        (void)geqrf<kBackend, T>(q, Av, tau.to_span(), ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        using PV = MatrixView<typename MV::promoted_type, MatrixFormat::Dense>;
        const VectorView<T> tv(tau.data(), k, batch, 1, k);
        const double res = MV::qr_residual(q, A0v, Av, tv) / verify::bound<T>(verify::Check::factorization, m);
        auto Q = MV::form_q(q, Av, tv, Av.default_items(), k);
        const double orth = PV::orthogonality(q, Q, Q.all_items()) / verify::bound<T>(verify::Check::orthogonality, m);
        return {verify::nanmax(res, orth), 0};
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
    std::vector<std::int64_t> dims(const std::string&, const CellKey& k) const override {  // m derives from (form, n, aspect)
        const int n = int(key_int(k, "n"));
        return {rows_of(*key_get(k, "form"), n, int(key_int(k, "aspect"))).value_or(n), n};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {  // A0, A (tau is k per item)
        const int n = int(key_int(k, "n"));
        const double m = double(rows_of(*key_get(k, "form"), n, int(key_int(k, "aspect"))).value_or(n));
        const double item = m * double(n) * dtype_bytes(dtype);
        return 2.0 * item * double(qr_batch(item));
    }
    FrontierPlan frontier() const override { return {1}; }  // p99.9 fails at every k. evidence: docs/design/tiered-tuning.md#engine-round-planning-by-flip-frontier
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
            // family: tiny
            "src/ops/geqrf/kernels/geqrf_tiny.cc",
            "src/ops/geqrf/kernels/geqrf_tiny_device.hh",
            // family: cta
            "src/ops/geqrf/kernels/geqrf_cta.cc",
            // family: blocked
            "src/ops/geqrf/kernels/geqrf_blocked.cc",
            "src/kernels/common/larft_wy.hh",
            // common
            "src/ops/geqrf/geqrf_native.hh",
            "src/ops/geqrf/geqrf_cta_device.hh",
            "src/ops/geqrf/geqrf_panel_reg_device.hh",
            "src/ops/geqrf/can_run.hh",
            "src/ops/geqrf/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/geqrf/geqrf.cc" "src/util/resident_capacity.hh"
        // family: tiny "src/kernels/common/tiny_device.hh"
        // geqrf blocked's trailing updates call the public gemm: its kernels and selection count too.
        // family: blocked "src/ops/gemm/kernels/gemm_kernels.cc" "src/ops/gemm/gemm_kernels.hh" "src/ops/gemm/kernels/accessors.hh" "src/ops/gemm/kernels/epilogue_linear.hh" "src/ops/gemm/kernels/load_policies.hh" "src/ops/gemm/kernels/register_128x128.hh" "src/ops/gemm/kernels/copy_async.hh" "src/ops/gemm/kernels/register_64x64_k16_wide.hh"
        // family: blocked "src/ops/gemm/kernels/register_launchers.hh" "src/ops/gemm/kernels/register_tiled_common.hh" "src/ops/gemm/kernels/register_wide_transposed.hh" "src/ops/gemm/kernels/small_batched.hh" "src/ops/gemm/kernels/tiled_general.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/geqrf_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const int n = int(key_int(req.key, "n"));
            const auto m = rows_of(*key_get(req.key, "form"), n, int(key_int(req.key, "aspect")));
            if (!m) throw std::invalid_argument("geqrf: no (m, n) has key " + key_arg(req.key));
            GeqrfProblem<T> p(q, *m, n, qr_batch(double(*m) * double(n) * sizeof(T)), req.ld_pad);
            if (req.mode == "race") return run_race<ops::geqrf::GeqrfChoice>("geqrf", p, req, 1.0);
            return run_arms<ops::geqrf::GeqrfChoice>("geqrf", p, req, 1.0);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::GeqrfSpec)

// Descriptor workspace()/launch() definitions and the select()/run() of the NEW potrf path.
// Host code only: every launch forwards to an existing library launcher.

#include "potrf_select.hh"

#include "../../../src/backends/potrf_profile_constants.hh"
#include "../../../src/sycl/trsm_native.hh"

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/device_facts.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/settings.hh>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

namespace batchlas::potrf_routes {

template <class T>
std::size_t Tiny<T>::workspace(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return sycl_potrf::potrf_tiny_buffer_size<T>(q, *a.A);
}
template <class T>
Event Tiny<T>::launch(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return sycl_potrf::potrf_tiny_dispatch<T>(q, *a.A, a.uplo, a.ws, a.info);
}

template <class T>
std::size_t Cta<T>::workspace(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return sycl_potrf::potrf_cta_buffer_size<T>(q, *a.A);
}
template <class T>
Event Cta<T>::launch(Queue& q, const PotrfArgs<T>& a, const Geometry& g) {
    return sycl_potrf::potrf_cta_dispatch_geometry<T>(q, *a.A, a.uplo, a.ws, a.info, g);
}

template <class T>
std::size_t LPanel<T>::workspace(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return sycl_potrf::potrf_lpanel_buffer_size<T>(q, *a.A);
}
template <class T>
Event LPanel<T>::launch(Queue& q, const PotrfArgs<T>& a, const Geometry& g) {
    return sycl_potrf::potrf_lpanel_dispatch<T>(q, *a.A, a.uplo, a.ws, a.info,
                                                resident::kMinBlocksPerSm, g.nb);
}

template <Backend B, class T>
std::size_t Blocked<B, T>::workspace(Queue& q, const PotrfArgs<T>& a, const Geometry& g) {
    return sycl_potrf::potrf_blocked_buffer_size_params<T>(q, *a.A, g.p);
}
template <Backend B, class T>
Event Blocked<B, T>::launch(Queue& q, const PotrfArgs<T>& a, const Geometry& g) {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    return sycl_potrf::potrf_blocked_dispatch_params<T>(
        q, *a.A, a.uplo, a.ws, a.info, g.p,
        [](Queue& c, const MV& ga, const MV& gb, const MV& gc, T al, T be, Transpose ta,
           Transpose tb, ComputePrecision p) { return gemm<B, T>(c, ga, gb, gc, al, be, ta, tb, p); },
        [](Queue& c, const MV& ta, const MV& tb, T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
            return trsm<B, T>(c, ta, tb, al, sd, ul, tr, dg);
        });
}

template <class T>
std::size_t CtaWg<T>::workspace(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return sycl_potrf::potrf_cta_buffer_size<T>(q, *a.A);
}
template <class T>
Event CtaWg<T>::launch(Queue& q, const PotrfArgs<T>& a, const Geometry& g) {
    return sycl_potrf::potrf_cta_dispatch_geometry<T>(q, *a.A, a.uplo, a.ws, a.info, g);
}

template <Backend B, class T>
std::size_t Cusolver<B, T>::workspace(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return backend::potrf_vendor_buffer_size<B, T>(q, *a.A, a.uplo);
}
template <Backend B, class T>
Event Cusolver<B, T>::launch(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return backend::potrf_vendor<B, T>(q, *a.A, a.uplo, a.ws, a.info);
}

}  // namespace batchlas::potrf_routes

namespace batchlas::potrf_v2 {

namespace {

template <class T>
constexpr int dtype_index() {
    if constexpr (std::is_same_v<T, float>) return 0;
    else if constexpr (std::is_same_v<T, double>) return 1;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return 2;
    else return 3;
}

// Migration shim: the positional generated profile re-keyed by NAME. The generator would emit
// these rows directly; nothing below this function knows the positional layout.
constexpr std::string_view kProfileKeys[5] = {"native:tiny", "native:cta", "native:lpanel",
                                              "native:blocked", "vendor:cusolver"};

template <class T>
const ds::CostBook* cost_book(arch::RoutingProfile prof) {
    struct Books {
        std::vector<ds::CostRow> rows[3];
        ds::CostBook book[3];
    };
    static const Books* books = [] {
        auto* b = new Books;   // leaked: no static-destruction ordering hazard
        for (auto p : {arch::RoutingProfile::sm_89, arch::RoutingProfile::sm_120}) {
            const auto* P = potrf_profile::profile_for(p);
            const int pi = static_cast<int>(p);
            for (int i = 0; i < 5; ++i) {
                for (int u = 0; u < 2; ++u) {
                    const auto& e = P->e[i][dtype_index<T>()][u];
                    if (!e.present) continue;
                    b->rows[pi].push_back({kProfileKeys[i], static_cast<std::uint8_t>(u), e.c,
                                           {e.box.n_min, e.box.n_max, e.box.batch_min, e.box.batch_max},
                                           e.box.rows});
                }
            }
            b->book[pi] = {P->model_enabled, P->margin, b->rows[pi]};
        }
        return b;
    }();
    const int pi = static_cast<int>(prof);
    return (pi >= 1 && pi <= 2) ? &books->book[pi] : nullptr;
}

std::optional<ds::PinPolicy> g_policy_override;
std::atomic<std::size_t> g_pin_warnings{0};

ds::PinPolicy pin_policy() {
    if (g_policy_override) return *g_policy_override;
    const char* v = std::getenv("BATCHLAS_PIN_POLICY");
    return (v && std::strcmp(v, "strict") == 0) ? ds::PinPolicy::Strict : ds::PinPolicy::Warn;
}

std::string_view intern(const std::string& s) {
    static std::mutex mu;
    static auto* pool = new std::set<std::string>;   // leaked on purpose
    const std::lock_guard<std::mutex> lock(mu);
    return *pool->insert(s).first;
}

void warn_pin_once(std::string_view pin, const std::string& text) {
    static std::mutex mu;
    static auto* seen = new std::set<std::string>;
    const std::lock_guard<std::mutex> lock(mu);
    if (!seen->insert(std::string(pin)).second) return;
    g_pin_warnings.fetch_add(1);
    std::fprintf(stderr, "batchlas: potrf pin '%.*s' not honoured:\n%s", static_cast<int>(pin.size()),
                 pin.data(), text.c_str());
}

// Exact key first ("native:cta_wg"), then the shared route grammar ("native", "cta", ...).
template <class Table>
std::optional<ds::Pin> parse_pin() {
    const char* raw = settings().routing.canonical_route(dispatch::Op::potrf).get();
    const bool canonical = raw && *raw;
    const std::string text = canonical ? raw : std::string();
    if (canonical) {
        for (int i = 0; i < Table::size; ++i) {
            if (Table::keys[i] == text) return ds::Pin{intern(text), false, {}, Table::keys[i]};
        }
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::potrf);
    if (!parsed.found) return std::nullopt;
    const std::string_view shown = intern(canonical ? text : std::string(parsed.source.value));
    if (parsed.route.algo == dispatch::Algorithm::Auto) {
        return ds::Pin{shown, true, parsed.route.origin, {}};
    }
    for (int i = 0; i < Table::size; ++i) {
        if (Table::routes[i] == parsed.route) return ds::Pin{shown, false, {}, Table::keys[i]};
    }
    return ds::Pin{shown, false, {}, shown};   // names no row: recorded as not honoured
}

template <class T>
pr::PotrfShape make_shape(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                          const potrf_profile::Profile* P) {
    pr::PotrfShape s;
    s.n = static_cast<int>(A.rows());
    s.batch = A.batch_size();
    s.uplo = uplo;
    s.square = A.rows() == A.cols();
    s.homogeneous = !A.is_heterogeneous();
    const Device dev = q.device();
    const dispatch::DeviceFacts f = dispatch::device_facts(dev);   // memoised per device
    s.is_gpu = f.is_gpu;
    s.has_sg32 = dev.supports_sub_group_size(32);
    s.dev = sycl_potrf::potrf_device_facts(dev);
    s.dev.compute_units = f.compute_units;
    if (P) {
        s.dev.max_threads_per_cu = P->max_threads_per_cu;
        s.dev.max_groups_per_cu = P->max_groups_per_cu;
        s.regs = P->regs[dtype_index<T>()];
    }
    s.leaf_trsm = sycl_trsm::trsm_cta_max_n<T>();
    sycl_potrf::potrf_blocked_overrides(s.nb_knob, s.w_knob);
    return s;
}

}  // namespace

std::size_t pin_warnings_emitted() { return g_pin_warnings.load(); }
void set_pin_policy(std::optional<ds::PinPolicy> p) { g_policy_override = p; }

template <Backend B, class T, bool Vendor>
Sel<B, T, Vendor> select(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                         Span<std::byte> ws, Span<int32_t> info, Mode mode, StrategyOverride so) {
    using Table = Tbl<B, T, Vendor>;
    static_assert(pr::windows_name_rows<Table, T>(), "a potrf window names no route in the table");

    const std::uint64_t epoch = detail::settings_epoch();
    const arch::RoutingProfile prof = dispatch::routing_profile(q.device()).profile;
    const potrf_profile::Profile* P = potrf_profile::profile_for(prof);
    const pr::PotrfShape s = make_shape<T>(q, A, uplo, P);

    typename Table::Geoms g;
    Sel<B, T, Vendor> out;
    out.candidates = Table::candidates(s, g);
    const ds::CostBook* book = (so == StrategyOverride::WindowsOnly) ? nullptr : cost_book<T>(prof);
    const ds::ModelChooser model{book, s.dev};
    const ds::WindowChooser<pr::PotrfShape> windows{
        so == StrategyOverride::ModelOnly ? std::span<const ds::Window<pr::PotrfShape>>{}
                                          : pr::potrf_windows<T>(),
        &s};
    const std::optional<ds::Pin> pin = parse_pin<Table>();
    out.decision = ds::decide(out.candidates, pin, model, windows);
    out.settings_epoch = epoch;

    const auto& d = out.decision;
    if (d.index < 0) {   // a vendor-free table on a shape no native serves
        if (mode == Mode::DecideOnly) return out;
        throw std::runtime_error("potrf: no eligible route\n" + ds::explain(out.candidates, d));
    }
    out.plan = Table::plan_at(d.index, g);
    if (!d.pin.empty() && !d.pin_honoured) {
        const std::string text = explain<B, T, Vendor>(out);
        if (pin_policy() == ds::PinPolicy::Strict) {
            throw std::invalid_argument("potrf pin '" + std::string(d.pin) + "' not honoured:\n" + text);
        }
        warn_pin_once(d.pin, text);
    }
    if (mode == Mode::Full) {
        const pr::PotrfArgs<T> a{&A, uplo, ws, info};
        out.workspace = Table::with_row(out.plan, [&]<class R>(const typename R::Geometry& geo) {
            return R::workspace(q, a, geo);
        });
        if (dispatch::coverage::dynamic_enabled()) {   // the one coverage record, backend = B
            dispatch::OpShape row;
            row.op = dispatch::Op::potrf;
            row.scalar = dispatch::scalar_kind_of<T>;
            row.backend = B;
            row.m = A.rows(); row.n = A.cols(); row.k = A.rows(); row.batch = A.batch_size();
            row.uplo = uplo;
            dispatch::fill_device_facts(row, q);
            row.cost_extrapolated = d.extrapolated;
            bool native_supported = false;
            for (int i = 0; i < out.candidates.size; ++i) {
                native_supported |= dispatch::is_native(out.candidates.row[i].route) &&
                                    out.candidates.row[i].legal.ok;
            }
            dispatch::coverage::record_if_enabled(dispatch::Op::potrf, row.scalar, B, row,
                                                  out.candidates.row[d.index].route, true,
                                                  native_supported);
        }
    }
    return out;
}

template <Backend B, class T, bool Vendor>
Event run(Queue& q, const Sel<B, T, Vendor>& sel, const MatrixView<T, MatrixFormat::Dense>& A,
          Uplo uplo, Span<std::byte> ws, Span<int32_t> info) {
    if (sel.settings_epoch != detail::settings_epoch()) {
        throw std::logic_error("potrf: settings changed between select() and run()");
    }
    if (ws.size() < sel.workspace) {
        throw std::length_error("potrf: workspace " + std::to_string(ws.size()) + " B < " +
                                std::to_string(sel.workspace) + " B needed by " +
                                std::string(sel.key()));
    }
    const pr::PotrfArgs<T> a{&A, uplo, ws, info};
    return Tbl<B, T, Vendor>::with_row(sel.plan, [&]<class R>(const typename R::Geometry& geo) {
        return R::launch(q, a, geo);
    });
}

template <Backend B, class T>
std::size_t potrf_buffer_size(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    return select<B, T>(q, A, uplo).workspace;
}

template <Backend B, class T>
Event potrf(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
            Span<std::byte> ws, Span<int32_t> info) {
    return run<B, T>(q, select<B, T>(q, A, uplo, ws, info), A, uplo, ws, info);
}

template <Backend B, class T, bool Vendor>
std::string explain(const Sel<B, T, Vendor>& sel) {
    if (sel.decision.index < 0) return ds::explain(sel.candidates, sel.decision);
    const std::string kernel = Tbl<B, T, Vendor>::with_row(
        sel.plan, [&]<class R>(const typename R::Geometry& geo) { return R::kernel(geo); });
    return ds::explain(sel.candidates, sel.decision, kernel);
}

#define BATCHLAS_POTRF_V2_INSTANTIATE(B, T)                                                       \
    template Sel<B, T, true> select<B, T, true>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                                Uplo, Span<std::byte>, Span<int32_t>, Mode,        \
                                                StrategyOverride);                                  \
    template Sel<B, T, false> select<B, T, false>(Queue&,                                         \
                                                  const MatrixView<T, MatrixFormat::Dense>&, Uplo, \
                                                  Span<std::byte>, Span<int32_t>, Mode,            \
                                                  StrategyOverride);                                \
    template Event run<B, T, true>(Queue&, const Sel<B, T, true>&,                                \
                                   const MatrixView<T, MatrixFormat::Dense>&, Uplo,               \
                                   Span<std::byte>, Span<int32_t>);                               \
    template Event run<B, T, false>(Queue&, const Sel<B, T, false>&,                              \
                                    const MatrixView<T, MatrixFormat::Dense>&, Uplo,              \
                                    Span<std::byte>, Span<int32_t>);                              \
    template std::string explain<B, T, true>(const Sel<B, T, true>&);                             \
    template std::string explain<B, T, false>(const Sel<B, T, false>&);                           \
    template std::size_t potrf_buffer_size<B, T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                                 Uplo);                                            \
    template Event potrf<B, T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo,           \
                               Span<std::byte>, Span<int32_t>);

BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, float)
BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, double)
BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, std::complex<float>)
BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, std::complex<double>)

#undef BATCHLAS_POTRF_V2_INSTANTIATE

}  // namespace batchlas::potrf_v2

// potrf on the descriptor registry: the descriptors' workspace()/launch() bodies, the cost-book
// shim, the shape builder and the thin select/run wrappers over src/dispatch/selection/run.hh.
// Host code only: every launch forwards to an existing launcher.

#include <batchlas/backend_config.h>

#if BATCHLAS_HAS_CUDA_BACKEND

#include "../backends/potrf_select.hh"

#include "../backends/potrf_profile_constants.hh"
#include "../sycl/trsm_native.hh"

#include <batchlas/blas/dispatch/device_facts.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>

#if BATCHLAS_HAS_CUSOLVER
#include <batchlas/blas/functions/potrf.hh>
#endif

#include <atomic>
#include <cstdio>
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

// Defined only where the vendor TU is linked; the Vendor = false table never names Cusolver.
#if BATCHLAS_HAS_CUSOLVER
template <Backend B, class T>
std::size_t Cusolver<B, T>::workspace(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return backend::potrf_vendor_buffer_size<B, T>(q, *a.A, a.uplo);
}
template <Backend B, class T>
Event Cusolver<B, T>::launch(Queue& q, const PotrfArgs<T>& a, const Geometry&) {
    return backend::potrf_vendor<B, T>(q, *a.A, a.uplo, a.ws, a.info);
}
#endif

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

// A profile measured on another architecture routes by hypotheses: say so once.
void warn_nearest_profile(const arch::ProfileChoice& pc) {
    static std::atomic<bool> warned{false};
    if (!pc.nearest || warned.exchange(true)) return;
    std::fprintf(stderr,
                 "batchlas: potrf routes with the %.*s profile, which was not measured on this "
                 "device's architecture; its windows and cost constants are hypotheses here\n",
                 static_cast<int>(arch::to_string(pc.profile).size()), arch::to_string(pc.profile).data());
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

std::size_t pin_warnings_emitted() { return ds::pin_warnings().load(); }
void set_pin_policy(std::optional<ds::PinPolicy> p) { ds::pin_policy_override() = p; }

template <Backend B, class T, bool Vendor>
Sel<B, T, Vendor> select(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                         Mode mode, StrategyOverride so) {
    using Table = Tbl<B, T, Vendor>;
    static_assert(pr::windows_name_rows<Table, T>(), "a potrf window names no route in the table");

    const arch::ProfileChoice pc = dispatch::routing_profile(q.device());
    warn_nearest_profile(pc);
    const potrf_profile::Profile* P = potrf_profile::profile_for(pc.profile);
    const pr::PotrfShape s = make_shape<T>(q, A, uplo, P);

    const ds::CostBook* book = (so == StrategyOverride::WindowsOnly) ? nullptr : cost_book<T>(pc.profile);
    const ds::ModelChooser model{book, s.dev};
    const ds::WindowChooser<pr::PotrfShape> windows{
        so == StrategyOverride::ModelOnly ? std::span<const ds::Window<pr::PotrfShape>>{}
                                          : pr::potrf_windows<T>(),
        &s};
    Sel<B, T, Vendor> out = ds::decide_selection<Table>(s, ds::parse_pin<Table>(), model, windows);
    if (out.decision.index < 0) {   // a vendor-free table on a shape no native serves
        if (mode == Mode::DecideOnly) return out;
        dispatch::throw_no_vendor_route<T>(dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
    }
    ds::enforce_pin(out);
    if (mode == Mode::Full) {
        ds::size_selection(q, out, pr::PotrfArgs<T>{&A, uplo, {}, {}});
        dispatch::OpShape row;
        row.op = dispatch::Op::potrf;
        row.scalar = dispatch::scalar_kind_of<T>;
        row.backend = B;
        row.m = A.rows(); row.n = A.cols(); row.k = A.rows(); row.batch = A.batch_size();
        row.uplo = uplo;
        row.heterogeneous_batch = A.is_heterogeneous();
        dispatch::fill_device_facts(row, q);
        ds::record_coverage(out, row);
    }
    return out;
}

template <Backend B, class T, bool Vendor>
Event run(Queue& q, const Sel<B, T, Vendor>& sel, const MatrixView<T, MatrixFormat::Dense>& A,
          Uplo uplo, Span<std::byte> ws, Span<int32_t> info) {
    return ds::run_selection(q, sel, pr::PotrfArgs<T>{&A, uplo, ws, info}, ws.size());
}

template <Backend B, class T, bool Vendor>
std::string explain(const Sel<B, T, Vendor>& sel) {
    return ds::explain(sel);
}

template <Backend B, class T>
std::size_t buffer_size(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    return select<B, T>(q, A, uplo).workspace;
}

template <Backend B, class T>
Event potrf(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
            Span<std::byte> ws, Span<int32_t> info) {
    return run<B, T>(q, select<B, T>(q, A, uplo), A, uplo, ws, info);
}

#define BATCHLAS_POTRF_V2_INSTANTIATE_TABLE(B, T, V)                                              \
    template Sel<B, T, V> select<B, T, V>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,      \
                                          Uplo, Mode, StrategyOverride);                          \
    template Event run<B, T, V>(Queue&, const Sel<B, T, V>&,                                      \
                                const MatrixView<T, MatrixFormat::Dense>&, Uplo,                  \
                                Span<std::byte>, Span<int32_t>);                                  \
    template std::string explain<B, T, V>(const Sel<B, T, V>&);

#define BATCHLAS_POTRF_V2_INSTANTIATE(B, T)                                                       \
    BATCHLAS_POTRF_V2_INSTANTIATE_TABLE(B, T, false)                                              \
    template std::size_t buffer_size<B, T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,     \
                                           Uplo);                                                 \
    template Event potrf<B, T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo,           \
                               Span<std::byte>, Span<int32_t>);

BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, float)
BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, double)
BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, std::complex<float>)
BATCHLAS_POTRF_V2_INSTANTIATE(Backend::CUDA, std::complex<double>)

#if BATCHLAS_HAS_CUSOLVER
BATCHLAS_POTRF_V2_INSTANTIATE_TABLE(Backend::CUDA, float, true)
BATCHLAS_POTRF_V2_INSTANTIATE_TABLE(Backend::CUDA, double, true)
BATCHLAS_POTRF_V2_INSTANTIATE_TABLE(Backend::CUDA, std::complex<float>, true)
BATCHLAS_POTRF_V2_INSTANTIATE_TABLE(Backend::CUDA, std::complex<double>, true)
#endif

#undef BATCHLAS_POTRF_V2_INSTANTIATE
#undef BATCHLAS_POTRF_V2_INSTANTIATE_TABLE

}  // namespace batchlas::potrf_v2

#endif  // BATCHLAS_HAS_CUDA_BACKEND

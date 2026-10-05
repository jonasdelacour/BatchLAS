// gesvd: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public gesvd() -> choose() -> std::visit -> launch. The kernel for a shape is the first
// runnable entry of the nearest row in tuned/gesvd.<dtype>.<device>.txt; can_run() below only
// removes entries that cannot run. Jacobi is one-sided Jacobi in local memory; Cta forms the
// normal equations and solves them with syev_cta; Blocked bidiagonalises (or, Hermitian, runs
// syev_blocked); Vendor is the solver library (cuSOLVER gesvdjBatched, LAPACKE on the host).

#include <batchlas/backend_config.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/settings.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/gesvd_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::gesvd {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <class T>
using SV = Span<typename base_type<T>::type>;

// The call after canonicalisation: Thin is rewritten to All wherever the two coincide, once,
// so sizing and running see the same jobs (enums.hh).
struct Job {
    SvdVectors jobu, jobvh;
    std::optional<Uplo> herm;
    bool thin() const { return jobu == SvdVectors::Thin || jobvh == SvdVectors::Thin; }
    bool vectors() const { return jobu != SvdVectors::None || jobvh != SvdVectors::None; }
};

template <class T>
Job canonical(const MV<T>& A, SvdVectors jobu, SvdVectors jobvh, std::optional<Uplo> herm) {
    const std::int64_t k = std::min<std::int64_t>(A.rows(), A.cols());
    return {canonical_jobu(jobu, A.rows(), k), canonical_jobvh(jobvh, A.cols(), k), herm};
}

template <class T>
select::Key key_of(const MV<T>& A, const Job& j) {
    const char* herm = !j.herm ? "N" : (*j.herm == Uplo::Lower ? "L" : "U");
    const char* vec = j.thin() ? "thin" : (j.vectors() ? "all" : "none");
    return {{"herm", herm}, {"vec", vec}, {"m", A.rows()}, {"n", A.cols()}};
}

// Correctness only (R3): false means the driver throws or answers wrongly. Jacobi has no
// Hermitian form; Cta and Blocked run complex input only through the Hermitian shortcut
// (gesvd_native_impl throws `unsupported` for complex general); Cta cannot write a genuinely thin
// U; Blocked's Hermitian arm is Lower-only, as the old router had it (Upper is unverified).
template <class T>
bool can_run(const GesvdChoice& c, const select::Device& d, const MV<T>& A, const Job& j) {
    constexpr bool kReal = std::is_same_v<T, typename base_type<T>::type>;
    const std::int64_t m = A.rows(), n = A.cols(), md = std::max(m, n);
    const bool native = d.is_gpu && m >= 1 && n >= 1 && A.batch_size() >= 1;
    const bool square = m == n;
    return std::visit(overloaded{
        [&](Jacobi) {
            return native && d.has_sg32 && !j.herm && md <= sycl_gesvd::gesvd_jacobi_max_dim<T>(j.vectors());
        },
        [&](Cta) {
            const bool form = j.herm ? square : kReal;
            return native && d.has_sg32 && md <= sycl_gesvd::kGesvdCtaMaxDim && !j.thin() && form;
        },
        [&](Blocked) { return native && (j.herm ? square && *j.herm == Uplo::Lower : kReal); },
        [&](Vendor) { return d.has_vendor_solver; },
    }, c);
}

// BATCHLAS_GESVD_PROVIDER, the pre-route spelling, still pins when neither a ScopedPin nor
// BATCHLAS_GESVD_ROUTE does. It is applied as a pin for this one choose().
inline std::optional<std::string> legacy_pin() {
    std::string source;
    if (select::detail::pin_text("gesvd", &source)) return std::nullopt;
    const char* raw = settings().routing.legacy_route(dispatch::Op::gesvd).get();
    if (!raw || !*raw) return std::nullopt;
    return std::string(raw);
}

template <Backend B, class T>
GesvdChoice choose(Queue& q, const MV<T>& A, const Job& j) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const GesvdChoice& c) { return can_run<T>(c, d, A, j); };
    std::optional<select::ScopedPin<GesvdChoice>> legacy;
    if (auto word = legacy_pin()) legacy.emplace("gesvd", std::string_view(*word));
    try {
        return select::choose("gesvd", select::dtype_name<T>(), d, key_of<T>(A, j), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!dispatch::solver_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            dispatch::throw_no_vendor_route<T>(dispatch::Op::gesvd, B, dispatch::kSolverLibrary<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const Job& j) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(), [&](const GesvdChoice& c) { return can_run<T>(c, d, A, j); });
}

// Exactly the chosen family's need (R5).
template <Backend B, class T>
std::size_t workspace(Queue& q, const GesvdChoice& c, const MV<T>& A, SV<T> s, const MV<T>& U, const MV<T>& Vh,
                      const Job& j) {
    return std::visit(overloaded{
        [&](Jacobi) { return gesvdj_cta_buffer_size<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh); },
        [&](Cta) {
            return j.herm ? gesvd_cta_buffer_size<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm)
                          : gesvd_cta_buffer_size<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh);
        },
        [&](Blocked) {
            return j.herm ? gesvd_blocked_buffer_size<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm)
                          : gesvd_blocked_buffer_size<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh);
        },
        [&](Vendor) -> std::size_t {
            if constexpr (dispatch::solver_vendor_available<B>)
                return backend::gesvd_vendor_buffer_size<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::gesvd, B, dispatch::kSolverLibrary<B>);
        },
    }, c);
}

// Every driver clears or stores `info` itself, and exactly one runs. Jacobi gets default
// GesvdjParams: no sweep cap is reachable from the public entry point.
template <Backend B, class T>
Event launch(Queue& q, const GesvdChoice& c, const MV<T>& A, SV<T> s, const MV<T>& U, const MV<T>& Vh,
             const Job& j, Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Jacobi) { return gesvdj_cta<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, GesvdjParams<T>(), info); },
        [&](Cta) {
            return j.herm ? gesvd_cta<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm, ws, info)
                          : gesvd_cta<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, info);
        },
        [&](Blocked) {
            return j.herm ? gesvd_blocked<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm, ws, info)
                          : gesvd_blocked<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, info);
        },
        [&](Vendor) -> Event {
            if constexpr (dispatch::solver_vendor_available<B>)
                return backend::gesvd_vendor<B, T>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, info);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::gesvd, B, dispatch::kSolverLibrary<B>);
        },
    }, c);
}

template <Backend B, class T>
Event run(Queue& ctx, const MV<T>& A, SV<T> s, const MV<T>& U, const MV<T>& Vh, SvdVectors jobu, SvdVectors jobvh,
          std::optional<Uplo> herm, Span<std::byte> ws, Span<int32_t> info) {
    const Job j = canonical<T>(A, jobu, jobvh, herm);
    const GesvdChoice c = choose<B, T>(ctx, A, j);
    // The coverage row's key: m, n, k = min(m, n); uplo is the Hermitian triangle (general: Lower).
    // Opened before sizing, so a driver that refuses the shape still leaves its row, as before.
    auto shape = select::square_shape<B, T>(std::min<std::int64_t>(A.rows(), A.cols()), A.batch_size());
    shape.m = A.rows();
    shape.n = A.cols();
    shape.uplo = herm.value_or(Uplo::Lower);
    select::TraceScope trace("gesvd", c, shape, native_facts<B, T>(ctx, A, j), key_of<T>(A, j));
    if (ws.size() < workspace<B, T>(ctx, c, A, s, U, Vh, j))
        throw batchlas::workspace_error("gesvd: insufficient workspace for chosen provider");
    // The native drivers need an in-order queue. std::optional, not a Queue: the default
    // constructor builds a real sycl::queue on the default device.
    Queue* run_q = &ctx;
    std::optional<Queue> in_order_q;
    if (!ctx.in_order()) {
        in_order_q.emplace(ctx, true);
        Event dep = ctx.get_event();
        in_order_q->enqueue(dep);
        run_q = &*in_order_q;
    }
    return launch<B, T>(*run_q, c, A, s, U, Vh, j, ws, info);
}

template <Backend B, class T>
std::size_t size(Queue& ctx, const MV<T>& A, SV<T> s, const MV<T>& U, const MV<T>& Vh, SvdVectors jobu,
                 SvdVectors jobvh, std::optional<Uplo> herm) {
    const Job j = canonical<T>(A, jobu, jobvh, herm);
    return workspace<B, T>(ctx, choose<B, T>(ctx, A, j), A, s, U, Vh, j);
}

}  // namespace ops::gesvd

template <Backend B, typename T>
Event gesvd(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U, const MatrixView<T, MatrixFormat::Dense>& Vh, SvdVectors jobu,
            SvdVectors jobvh, Span<std::byte> workspace, Span<int32_t> info) {
    return ops::gesvd::run<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, std::nullopt, workspace, info);
}

template <Backend B, typename T>
Event gesvd(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U, const MatrixView<T, MatrixFormat::Dense>& Vh, SvdVectors jobu,
            SvdVectors jobvh, Uplo hermitian_uplo, Span<std::byte> workspace, Span<int32_t> info) {
    return ops::gesvd::run<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo, workspace, info);
}

template <Backend B, typename T>
size_t gesvd_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values, const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh, SvdVectors jobu, SvdVectors jobvh) {
    return ops::gesvd::size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, std::nullopt);
}

template <Backend B, typename T>
size_t gesvd_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values, const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh, SvdVectors jobu, SvdVectors jobvh,
                         Uplo hermitian_uplo) {
    return ops::gesvd::size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo);
}

#define GESVD_INSTANTIATE(B_, fp)                                                               \
    BATCHLAS_INSTANTIATE(sig::gesvd<fp>, gesvd, B_, fp)                                         \
    BATCHLAS_INSTANTIATE(sig::gesvd_hermitian<fp>, gesvd, B_, fp)                               \
    BATCHLAS_INSTANTIATE(sig::gesvd_buffer_size<fp>, gesvd_buffer_size, B_, fp)                 \
    BATCHLAS_INSTANTIATE(sig::gesvd_buffer_size_hermitian<fp>, gesvd_buffer_size, B_, fp)

#define GESVD_ALL(B_)                          \
    GESVD_INSTANTIATE(B_, float)               \
    GESVD_INSTANTIATE(B_, double)              \
    GESVD_INSTANTIATE(B_, std::complex<float>) \
    GESVD_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
GESVD_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
GESVD_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
GESVD_ALL(Backend::NETLIB)
#endif

#undef GESVD_ALL
#undef GESVD_INSTANTIATE

}  // namespace batchlas

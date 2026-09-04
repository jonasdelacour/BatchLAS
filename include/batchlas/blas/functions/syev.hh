#pragma once

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <string_view>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>

#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/extensions.hh>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using syev = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   Span<typename base_type<T>::type>,
                   JobType, Uplo, Span<std::byte>);

template <typename T>
using syev_buffer_size = size_t(Queue&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                Span<typename base_type<T>::type>,
                                JobType, Uplo);

// backend::syev_vendor / syev_vendor_buffer_size share these signatures.
template <typename T> using syev_vendor = syev<T>;
template <typename T> using syev_vendor_buffer_size = syev_buffer_size<T>;
}  // namespace sig


template <Backend B, typename T>
Event syev(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& descrA, // A is overwritten with eigenvectors
           Span<typename base_type<T>::type> eigenvalues,
           JobType jobtype,
           Uplo uplo,
           Span<std::byte> workspace);

template <Backend B, typename T>
size_t syev_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobtype,
                        Uplo uplo);

template <Backend B, typename T>
inline Event syev(Queue& ctx,
                  const Matrix<T, MatrixFormat::Dense>& descrA,
                  Span<typename base_type<T>::type> eigenvalues,
                  JobType jobtype,
                  Uplo uplo,
                  Span<std::byte> workspace) {
    return syev<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(descrA), eigenvalues, jobtype, uplo, workspace);
}

template <Backend B, typename T>
inline size_t syev_buffer_size(Queue& ctx,
                               const Matrix<T, MatrixFormat::Dense>& A,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo) {
    return syev_buffer_size<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), eigenvalues, jobtype, uplo);
}

} // namespace batchlas

namespace batchlas::backend {

// Implemented by backend wrapper TUs (e.g. cuSOLVER / rocSOLVER / LAPACKE).
template <Backend B, typename T>
Event syev_vendor(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& descrA,
                  Span<typename base_type<T>::type> eigenvalues,
                  JobType jobtype,
                  Uplo uplo,
                  Span<std::byte> workspace);

template <Backend B, typename T>
size_t syev_vendor_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo);

} // namespace batchlas::backend


namespace batchlas::blas::dispatch::detail {

// The vendor call, gated on the vendor being compiled in: as an `if constexpr` it
// is not compiled at all when the library is absent, so there is no symbol to link.
template <Backend B, typename T, typename... Args>
Event syev_vendor_or_throw(Args&&... args) {
    if constexpr (!batchlas::dispatch::solver_vendor_available<B>) {
        batchlas::dispatch::throw_no_vendor_route<T>(
            batchlas::dispatch::Op::syev, B, batchlas::dispatch::kSolverLibrary<B>);
    } else {
        return batchlas::backend::syev_vendor<B, T>(std::forward<Args>(args)...);
    }
}

template <Backend B, typename T, typename... Args>
size_t syev_vendor_buffer_size_or_throw(Args&&... args) {
    if constexpr (!batchlas::dispatch::solver_vendor_available<B>) {
        batchlas::dispatch::throw_no_vendor_route<T>(
            batchlas::dispatch::Op::syev, B, batchlas::dispatch::kSolverLibrary<B>);
    } else {
        return batchlas::backend::syev_vendor_buffer_size<B, T>(std::forward<Args>(args)...);
    }
}

} // namespace batchlas::blas::dispatch::detail

namespace batchlas::blas::dispatch {

namespace detail {

template <typename T>
inline SteqrParams<T> syev_cta_steqr_params(JobType jobtype) {
    SteqrParams<T> params{};
    // Deliberately slower and more robust than the CTA STEQR defaults: syev runs
    // inside syevx, where an inaccurate Ritz solve stagnates the outer iteration.
    params.max_sweeps = 400;
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
    return params;
}

// Thin wrappers over RouteTable<Op::syev, T>::supports, for the Python binding.
template <typename T>
inline bool syev_supports_cta(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);
template <typename T>
inline bool syev_supports_blocked(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);
template <typename T>
inline bool syev_supports_two_stage(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

// Superseded by the saturated per-n rules below; preferred() now reaches this only
// for n <= 32, where it returns false.
inline bool syev_prefer_vendor(bool is_gpu, int64_t n, int64_t batch) {
    if (!is_gpu) return false;
    if (n <= 32) return false;
    if (n >= 320 && n <= 640 && batch >= 128) return false;
    return true;
}

// --- Small n: where does the vendor overtake the CTA solver? ---------------
// CTA supports every n <= 32 and is checked first, but with eigenvectors the
// vendor is faster over the top of that range. BATCHLAS_SYEV_CTA_MAX_N=<n>
// (0..32) overrides the per-type default below; a forced native:cta still wins.
template <typename T>
inline constexpr int64_t syev_cta_max_n_default_for() {
    using Real = typename base_type<T>::type;
    constexpr bool kReal = std::is_same_v<T, Real>;
    constexpr bool kDouble = std::is_same_v<Real, double>;
    // complex<double> ONLY: the vendor takes over at n = 25, not 33 -- an FP64-rate
    // artifact (this card runs FP64 at 1/64). complex<float> does NOT cross over, so
    // this deliberately does not generalise to "complex".
    if constexpr (!kReal && kDouble) return 24;
    return 32;
}

template <typename T>
inline int64_t syev_cta_max_n_for_vectors() {
    // Default 32 == OFF for every type but complex<double>. Lowering it is a measured
    // 1.03x - 1.15x for LOBPCG's projected solve, but it also flips a marginal case in
    // ILUKTests.SyevxInstrumentationAndPreconditioner, so it stays off by default.
    constexpr int64_t kDefault = syev_cta_max_n_default_for<T>();
    const char* v = std::getenv("BATCHLAS_SYEV_CTA_MAX_N");
    if (!v || !*v) return kDefault;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed < 0 || parsed > 32) return kDefault;
    return static_cast<int64_t>(parsed);
}

// --- Which small-n kernel? -------------------------------------------------
// Chosen per type and n; BATCHLAS_SYEV_SMALL_KERNEL=cta|fused|jacobi overrides it
// wholesale. FP64 runs at 1/64 rate on this card, which inflates Jacobi's margin,
// so re-measure the `double` rule on a 1:2 FP64 GPU.
// evidence: SYEV_RETUNE_RESULTS.md
enum class SyevSmallKernel { Cta, CtaFused, Jacobi };

inline SyevSmallKernel syev_small_kernel_env(bool& forced) {
    forced = true;
    const char* v = std::getenv("BATCHLAS_SYEV_SMALL_KERNEL");
    if (v && *v) {
        const std::string_view s(v);
        if (s == "cta") return SyevSmallKernel::Cta;
        if (s == "fused" || s == "cta_fused") return SyevSmallKernel::CtaFused;
        if (s == "jacobi") return SyevSmallKernel::Jacobi;
    }
    forced = false;
    return SyevSmallKernel::Cta;
}

template <typename T>
inline SyevSmallKernel syev_choose_small_kernel(const MatrixView<T, MatrixFormat::Dense>& A) {
    bool forced = false;
    const SyevSmallKernel env = syev_small_kernel_env(forced);
    if (forced) return env;

    // `internal::is_complex` (src/math-helpers.hh) is not visible from this public
    // header, so detect complex via base_type: for a real T, base_type<T>::type IS T.
    constexpr bool kReal = std::is_same_v<T, typename base_type<T>::type>;
    if constexpr (!kReal) {
        // Complex keeps the historical Cta from n >= 9: the float rule below is wrong
        // here, jacobi running 4x - 6x off the pace at n >= 20. complex<double> is not
        // split at all.
        constexpr bool is_double_c = std::is_same_v<typename base_type<T>::type, double>;
        if constexpr (is_double_c) {
            return SyevSmallKernel::Cta;
        } else {
            return A.rows() <= 8 ? SyevSmallKernel::CtaFused : SyevSmallKernel::Cta;
        }
    } else {
        constexpr bool is_double = std::is_same_v<typename base_type<T>::type, double>;
        if constexpr (is_double) {
            return SyevSmallKernel::Jacobi;      // wins at every measured n <= 32
        } else {
            return A.rows() <= 8 ? SyevSmallKernel::Jacobi   // 2.2x - 4.6x
                                 : SyevSmallKernel::CtaFused; // 1.03x - 1.25x
        }
    }
}

// Takes the shape facts directly rather than a DeviceCaps + MatrixView, so it can
// be called from RouteTable<Op::syev, T>::preferred, which is pure.
template <typename T>
inline bool syev_prefer_vendor_over_cta(bool is_gpu,
                                        int64_t n,
                                        int max_sub_group,
                                        JobType jobtype) {
    if (!is_gpu) return false;
    if (jobtype != JobType::EigenVectors) return false;
    if (n < 1 || n > 32 || max_sub_group < 32) return false;   // == supports(CTA)
    return n > syev_cta_max_n_for_vectors<T>();
}


// --- Eigenvector routing, decided AT SATURATION and keyed on n alone ----------
// Keyed on n only: the earlier batch-keyed rules were measured on ladders that
// never reached saturation and routed five of nine sizes wrong. Boundaries are per
// scalar type; complex<double>'s is the most hardware-specific (FP64 at 1/64 here).
// evidence: tag perf-evidence/vendor-independence (see docs/perf/README.md)
template <typename T>
inline batchlas::dispatch::Algorithm syev_saturated_algorithm_for_n(int64_t n) {
    // Algorithm::Auto here means "no native algorithm is preferred at this n"; the
    // resolver, not this function, decides the origin.
    using A = batchlas::dispatch::Algorithm;
    using Real = typename base_type<T>::type;
    constexpr bool kReal = std::is_same_v<T, Real>;
    constexpr bool kDouble = std::is_same_v<Real, double>;

    // complex<double>: blocked only to 256.
    if constexpr (!kReal && kDouble) {
        return n <= 256 ? A::Blocked : A::Auto;
    } else if constexpr (!kReal) {
        // complex<float>: blocked to 512.
        return n <= 512 ? A::Blocked : A::Auto;
    } else {
        // Real types: blocked to 448.
        if (n <= 448) return A::Blocked;
        if constexpr (!kDouble) {
            if (n <= 1024) return A::TwoStage;
            return A::Auto;                        // 2048+
        } else {
            return A::Auto;                        // double
        }
    }
}

// --- The same, for EIGENVALUES-ONLY -----------------------------------------
// Keyed on n alone. The predicate this replaced also required batch >= 256, so an
// n = 1024 solve at batch 254 fell through to the vendor and paid 2.75x.
// evidence: tag perf-evidence/vendor-independence (see docs/perf/README.md)
inline batchlas::dispatch::Algorithm syev_saturated_algorithm_for_n_values(int64_t n) {
    using A = batchlas::dispatch::Algorithm;
    if (n <= 320) return A::Blocked;   // 64..320, 1.05x - 1.21x
    return A::TwoStage;                // 512..2048, 1.29x - 2.75x
}

// The routing inputs, in one place so the call and its buffer-size query cannot
// build different ones. syev's routing reads `jobtype`; OpShape has no field for it.
struct SyevShape : batchlas::dispatch::OpShape {
    JobType jobtype = JobType::EigenVectors;
};

template <typename T>
inline SyevShape syev_op_shape(const Queue& ctx,
                               Backend backend,
                               const MatrixView<T, MatrixFormat::Dense>& A,
                               Uplo uplo,
                               JobType jobtype) {
    SyevShape s;
    s.op = batchlas::dispatch::Op::syev;
    s.scalar = batchlas::dispatch::scalar_kind_of<T>;
    s.backend = backend;
    s.m = A.rows();
    s.n = A.cols();
    s.k = A.rows();
    s.batch = A.batch_size();
    s.uplo = uplo;
    s.jobtype = jobtype;
    try {
        s.is_gpu = ctx.device().type == DeviceType::GPU;
    } catch (...) {
        // best-effort: query_caps never threw, and that contract is kept.
    }
    try {
        s.max_sub_group =
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    } catch (...) {
        // leave default
    }
    return s;
}

} // namespace detail
} // namespace batchlas::blas::dispatch

namespace batchlas::dispatch {

// SYEV's routing table. Unlike gemm/ormqr/gesvd it lives in the op's own header
// rather than in dispatch/route_syev.hh, next to the predicates `preferred` uses.
inline constexpr Route kSyevOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Native, Algorithm::TwoStage},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::syev, T> {
    using Shape = batchlas::blas::dispatch::detail::SyevShape;

    // ---- CORRECTNESS ------------------------------------------------------
    static bool supports(Route r, const Shape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;
        if (s.m != s.n) return false;
        if (!s.is_gpu) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                if (s.n < 1 || s.n > 32) return false;
                return s.max_sub_group >= 32;
            case Algorithm::Blocked:
            case Algorithm::TwoStage:
                // Uplo::Upper is supported: both paths mirror the upper triangle into
                // the lower one (src/extensions/uplo_mirror.hh) and run the Lower path.
                return s.n >= 1 && s.batch >= 1;
            default:
                // Including Auto: syev has three native routes, so a bare "native"
                // names none of them; resolve_route walks the order.
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    static bool preferred(Route r, const Shape& s) {
        namespace det = batchlas::blas::dispatch::detail;
        if (!is_native(r)) return false;
        if (!supports(r, s)) return false;

        // CUDA only: the grids were measured on no other backend.
        if (s.backend == Backend::CUDA) {
            if (s.n > 32) {
                // Eigenvalues-only first: its grid overlaps the vendor-preferred region.
                const Algorithm want = (s.jobtype != JobType::EigenVectors)
                    ? det::syev_saturated_algorithm_for_n_values(s.n)
                    : det::syev_saturated_algorithm_for_n<T>(s.n);
                // Auto == no native route preferred, so the resolver takes the vendor.
                return r.algo == want;
            }

            // Only reachable for n <= 32, where it returns false by construction.
            // Kept so that changing the branches above cannot silently drop it.
            if (det::syev_prefer_vendor(s.is_gpu, s.n, s.batch)) return false;

            // Small n with eigenvectors: this declines the CALL, not just the CTA
            // route, so no native route is preferred and the vendor runs.
            if (det::syev_prefer_vendor_over_cta<T>(s.is_gpu, s.n, s.max_sub_group,
                                                    s.jobtype)) {
                return false;
            }
        }

        return true;
    }

    static constexpr const Route* order_begin() { return kSyevOrder; }
    static constexpr const Route* order_end() {
        return kSyevOrder + (sizeof(kSyevOrder) / sizeof(kSyevOrder[0]));
    }
};

} // namespace batchlas::dispatch

namespace batchlas::blas::dispatch {
namespace detail {

// Introspection wrappers: they ask the resolver's `supports` and cannot drift.
template <typename T>
inline bool syev_supports_cta(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    namespace d = batchlas::dispatch;
    return d::RouteTable<d::Op::syev, T>::supports(
        {d::Origin::Native, d::Algorithm::CTA},
        syev_op_shape<T>(ctx, Backend::AUTO, A, Uplo::Lower, JobType::EigenVectors));
}

template <typename T>
inline bool syev_supports_blocked(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    namespace d = batchlas::dispatch;
    return d::RouteTable<d::Op::syev, T>::supports(
        {d::Origin::Native, d::Algorithm::Blocked},
        syev_op_shape<T>(ctx, Backend::AUTO, A, uplo, JobType::EigenVectors));
}

template <typename T>
inline bool syev_supports_two_stage(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    namespace d = batchlas::dispatch;
    return d::RouteTable<d::Op::syev, T>::supports(
        {d::Origin::Native, d::Algorithm::TwoStage},
        syev_op_shape<T>(ctx, Backend::AUTO, A, uplo, JobType::EigenVectors));
}

// One resolution per call, shared by syev_dispatch and its buffer-size query.
template <Backend B, typename T>
inline batchlas::dispatch::Route syev_route(const Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& A,
                                            Uplo uplo,
                                            JobType jobtype) {
    namespace d = batchlas::dispatch;
    const auto parsed = d::parse_route_env(d::Op::syev);
    const d::Route forced = parsed.found ? parsed.route : d::legacy_unset_default(d::Op::syev);
    return d::resolve_route<d::Op::syev, T>(forced, syev_op_shape<T>(ctx, B, A, uplo, jobtype));
}

} // namespace detail

// Route resolution plus orchestration; the vendor call is `backend::syev_vendor`.
template <Backend B, typename T>
inline Event syev_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& descrA,
                           Span<typename base_type<T>::type> eigenvalues,
                           JobType jobtype,
                           Uplo uplo,
                           Span<std::byte> workspace) {
    namespace d = batchlas::dispatch;
    // NETLIB has no native syev route; skip resolution rather than override it.
    const d::Route chosen = (B == Backend::NETLIB)
        ? d::Route{d::Origin::Vendor, d::Algorithm::Auto}
        : detail::syev_route<B, T>(ctx, descrA, uplo, jobtype);

    size_t need_ws = 0;
    if (d::is_vendor(chosen)) {
        need_ws = detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, descrA, eigenvalues, jobtype, uplo);
    } else if (chosen.algo == d::Algorithm::CTA) {
        switch (detail::syev_choose_small_kernel<T>(descrA)) {
            case detail::SyevSmallKernel::Jacobi:
                need_ws = syev_jacobi_cta_buffer_size<B, T>(ctx, descrA, jobtype);
                break;
            case detail::SyevSmallKernel::CtaFused:
                need_ws = syev_cta_fused_buffer_size<B, T>(ctx, descrA, jobtype,
                                                           detail::syev_cta_steqr_params<T>(jobtype));
                break;
            default:
                need_ws = syev_cta_buffer_size<B, T>(ctx, descrA, jobtype,
                                                     detail::syev_cta_steqr_params<T>(jobtype));
                break;
        }
    } else if (chosen.algo == d::Algorithm::TwoStage) {
        need_ws = syev_two_stage_buffer_size<B, T>(ctx,
                                                   descrA,
                                                   jobtype,
                                                   uplo,
                                                   StedcParams<typename base_type<T>::type>{});
    } else if (chosen.algo == d::Algorithm::Blocked) {
        need_ws = syev_blocked_buffer_size<B, T>(ctx,
                                                 descrA,
                                                 jobtype,
                                                 uplo,
                                                 StedcParams<typename base_type<T>::type>{});
    } else {
        // Unreachable. A hard failure rather than a silent reset to Vendor, which is
        // what let ormqr's buffer size and call disagree (see route_ormqr.hh).
        throw std::logic_error("syev: resolver returned a route with no dispatch arm");
    }

    if (workspace.size() < need_ws) {
        throw std::runtime_error("syev: insufficient workspace for chosen provider");
    }

    // std::optional, not a plain `Queue`: the default Queue constructor is not inert -- it
    // builds a real sycl::queue on Device::default_device(), so a by-value declaration would
    // pay that (and touch device 0) on every call. It also cannot be sunk into the
    // if-block: run_q escapes to the calls below.
    Queue* run_q = &ctx;
    std::optional<Queue> in_order_q;
    if (!ctx.in_order()) {
        in_order_q.emplace(ctx, true);
        Event dep = ctx.get_event();
        in_order_q->enqueue(dep);
        run_q = &*in_order_q;
    }

    Event e;
    if (d::is_vendor(chosen)) {
        e = detail::syev_vendor_or_throw<B, T>(*run_q, descrA, eigenvalues, jobtype, uplo, workspace);
    } else if (chosen.algo == d::Algorithm::CTA) {
        // The workspace query above MUST take the same branch: the selector reads its
        // env override fresh, so it must not be flipped between the query and the call.
        switch (detail::syev_choose_small_kernel<T>(descrA)) {
            case detail::SyevSmallKernel::Jacobi:
                e = syev_jacobi_cta<B, T>(*run_q, descrA, eigenvalues, jobtype, uplo, workspace);
                break;
            case detail::SyevSmallKernel::CtaFused:
                e = syev_cta_fused<B, T>(*run_q,
                                         descrA,
                                         eigenvalues,
                                         jobtype,
                                         uplo,
                                         workspace,
                                         detail::syev_cta_steqr_params<T>(jobtype),
                                         /*cta_wg_size_multiplier=*/1);
                break;
            default:
                e = syev_cta<B, T>(*run_q,
                                   descrA,
                                   eigenvalues,
                                   jobtype,
                                   uplo,
                                   workspace,
                                   detail::syev_cta_steqr_params<T>(jobtype),
                                   /*cta_wg_size_multiplier=*/1);
                break;
        }
    } else if (chosen.algo == d::Algorithm::TwoStage) {
        e = syev_two_stage<B, T>(*run_q,
                                 descrA,
                                 eigenvalues,
                                 jobtype,
                                 uplo,
                                 workspace,
                                 StedcParams<typename base_type<T>::type>{});
    } else {
        e = syev_blocked<B, T>(*run_q,
                               descrA,
                               eigenvalues,
                               jobtype,
                               uplo,
                               workspace,
                               StedcParams<typename base_type<T>::type>{});
    }

    return e;
}

template <Backend B, typename T>
inline size_t syev_buffer_size_dispatch(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& descrA,
                                        Span<typename base_type<T>::type> eigenvalues,
                                        JobType jobtype,
                                        Uplo uplo) {
    namespace d = batchlas::dispatch;
    // NETLIB has no native syev route; skip resolution rather than override it.
    const d::Route chosen = (B == Backend::NETLIB)
        ? d::Route{d::Origin::Vendor, d::Algorithm::Auto}
        : detail::syev_route<B, T>(ctx, descrA, uplo, jobtype);

    if (d::is_vendor(chosen)) {
        return detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, descrA, eigenvalues, jobtype, uplo);
    }
    if (chosen.algo == d::Algorithm::CTA) {
        // Must mirror syev_dispatch: sizing here and running a different small-n
        // kernel would under-allocate.
        switch (detail::syev_choose_small_kernel<T>(descrA)) {
            case detail::SyevSmallKernel::Jacobi:
                return syev_jacobi_cta_buffer_size<B, T>(ctx, descrA, jobtype);
            case detail::SyevSmallKernel::CtaFused:
                return syev_cta_fused_buffer_size<B, T>(ctx, descrA, jobtype,
                                                        detail::syev_cta_steqr_params<T>(jobtype));
            default:
                return syev_cta_buffer_size<B, T>(ctx, descrA, jobtype,
                                                  detail::syev_cta_steqr_params<T>(jobtype));
        }
    }
    if (chosen.algo == d::Algorithm::TwoStage) {
        return syev_two_stage_buffer_size<B, T>(ctx,
                                                descrA,
                                                jobtype,
                                                uplo,
                                                StedcParams<typename base_type<T>::type>{});
    }
    return syev_blocked_buffer_size<B, T>(ctx,
                                          descrA,
                                          jobtype,
                                          uplo,
                                          StedcParams<typename base_type<T>::type>{});
}

} // namespace batchlas::blas::dispatch

namespace batchlas {

template <Backend B, typename T>
inline Event syev(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& descrA,
                  Span<typename base_type<T>::type> eigenvalues,
                  JobType jobtype,
                  Uplo uplo,
                  Span<std::byte> workspace) {
    return blas::dispatch::syev_dispatch<B, T>(ctx, descrA, eigenvalues, jobtype, uplo, workspace);
}

template <Backend B, typename T>
inline size_t syev_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo) {
    return blas::dispatch::syev_buffer_size_dispatch<B, T>(ctx, descrA, eigenvalues, jobtype, uplo);
}

} // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(syev)
BATCHLAS_DISPATCH_ON_QUEUE(syev_buffer_size)

}  // namespace batchlas

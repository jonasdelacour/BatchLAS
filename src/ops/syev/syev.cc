// syev (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-syev): select::run
// takes the first entry of the nearest tuned/syev.<dtype>.<device>.txt row that can_run() admits. Cta,
// CtaFused and Jacobi are the three n <= 32 sub-group solvers; Blocked (sytrd_blocked + stedc) and
// TwoStage (sy2sb + sb2st + stedc) serve every n and call their sub-ops directly.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/settings.hh>

#include "choice.hh"
#include "vendor.hh"
#include "../../select/select.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::syev {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <class T>
using Real = typename base_type<T>::type;

// The sub-group solvers' own ceiling (syev_cta.cc, syev_cta_fused.cc, syev_jacobi_cta.cc all
// throw above it): one matrix per 32-lane partition, the CTA invariant of AGENTS.md §11.
inline constexpr std::int64_t kSmallMaxN = 32;

template <class T>
select::Key key_of(const MV<T>& A, JobType jobz) {
    return {{"jobz", jobz == JobType::EigenVectors ? "V" : "N"}, {"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or compute garbage. The native
// drivers are GPU kernels on a GPU backend (a NETLIB instantiation would call netlib on device
// memory) and need a square A with n >= 1; the small solvers add n <= 32 and a 32-wide
// sub-group, Blocked and TwoStage a batch >= 1 (the small drivers do not check it). Blocked and
// TwoStage take Upper by mirroring into Lower (uplo_mirror.hh). No term reads heterogeneity:
// neither the old router nor any driver checks it.
template <Backend B, class T>
bool can_run(const SyevChoice& c, const select::Device& d, const MV<T>& A) {
    const std::int64_t n = A.rows();
    const bool native = B != Backend::NETLIB && d.is_gpu && A.rows() == A.cols() && n >= 1;
    const bool small = native && n <= kSmallMaxN && d.has_sg32;
    const bool large = native && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Cta) { return small; },
        [&](CtaFused) { return small; },
        [&](Jacobi) { return small; },
        [&](Blocked) { return large; },
        [&](TwoStage) { return large; },
        [&](Vendor) { return d.has_vendor; },
    }, c);
}

// Deliberately slower and more robust than the CTA STEQR defaults: syev runs inside syevx,
// where an inaccurate Ritz solve stagnates the outer iteration.
template <class T>
SteqrParams<T> cta_steqr_params() {
    SteqrParams<T> params{};
    params.max_sweeps = 400;
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
    return params;
}

// Exactly the chosen family's need (R5).
template <Backend B, class T>
std::size_t workspace(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo) {
    return std::visit(overloaded{
        [&](Cta) { return syev_cta_buffer_size<B, T>(q, A, jobz, cta_steqr_params<T>()); },
        [&](CtaFused) { return syev_cta_fused_buffer_size<B, T>(q, A, jobz, cta_steqr_params<T>()); },
        [&](Jacobi) { return syev_jacobi_cta_buffer_size<B, T>(q, A, jobz); },
        [&](Blocked) { return syev_blocked_buffer_size<B, T>(q, A, jobz, uplo, StedcParams<Real<T>>{}); },
        [&](TwoStage) { return syev_two_stage_buffer_size<B, T>(q, A, jobz, uplo, StedcParams<Real<T>>{}); },
        [&](Vendor) {
            return blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(q, A, w, jobz, uplo);
        },
    }, c);
}

// Every arm clears or stores `info` itself, so the launch adds no clear of its own.
template <Backend B, class T>
Event launch(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo,
             Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Cta) { return syev_cta<B, T>(q, A, w, jobz, uplo, ws, cta_steqr_params<T>(), 0, info); },
        [&](CtaFused) { return syev_cta_fused<B, T>(q, A, w, jobz, uplo, ws, cta_steqr_params<T>(), 0, info); },
        [&](Jacobi) { return syev_jacobi_cta<B, T>(q, A, w, jobz, uplo, ws, JacobiParams<T>(), info); },
        [&](Blocked) { return syev_blocked<B, T>(q, A, w, jobz, uplo, ws, StedcParams<Real<T>>{}, info); },
        [&](TwoStage) { return syev_two_stage<B, T>(q, A, w, jobz, uplo, ws, StedcParams<Real<T>>{}, info); },
        [&](Vendor) { return blas::dispatch::detail::syev_vendor_or_throw<B, T>(q, A, w, jobz, uplo, ws, info); },
    }, c);
}

// Every driver refuses a non-square A; the old router sent one to the vendor, which then read
// rows() as the order. Thrown before choose() so neither Auto nor a pin can take it.
template <class T>
void validate(const MV<T>& A, const char* who) {
    if (A.rows() != A.cols())
        throw batchlas::invalid_argument(std::string(who) + ": A must be square, got " + std::to_string(A.rows()) +
                                         " x " + std::to_string(A.cols()));
}

// Capability only, for the Python binding's introspection: the same can_run, no backend.
template <class T>
bool supports(const Queue& q, const MV<T>& A, const SyevChoice& c) {
    return can_run<Backend::AUTO, T>(c, select::describe(q.device(), Backend::AUTO, false), A);
}

}  // namespace ops::syev

template <Backend Back, typename T>
Event syev(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<typename base_type<T>::type> eigenvalues,
           JobType jobtype, Uplo uplo, Span<std::byte> workspace, Span<int32_t> info) {
    namespace o = ops::syev;
    o::validate<T>(A, "syev");
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = o::key_of<T>(A, jobtype);
    return select::run<Back, T>(
        o::spec, ctx, key, o::candidates<T>(), [&](const auto& c, const auto& d) { return o::can_run<Back, T>(c, d, A); },
        shape, key, [&](const auto& c) {
            if (workspace.size() < o::workspace<Back, T>(ctx, c, A, eigenvalues, jobtype, uplo))
                throw batchlas::workspace_error("syev: insufficient workspace for chosen provider");
            // Blocked and TwoStage require an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return o::launch<Back, T>(q, c, A, eigenvalues, jobtype, uplo, workspace, info);
            });
        });
}

template <Backend Back, typename T>
size_t syev_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues, JobType jobtype, Uplo uplo) {
    namespace o = ops::syev;
    o::validate<T>(A, "syev_buffer_size");
    const auto c = select::pick<Back, T>(o::spec, ctx, o::key_of<T>(A, jobtype), o::candidates<T>(),
                                         [&](const auto& k, const auto& d) { return o::can_run<Back, T>(k, d, A); });
    return o::workspace<Back, T>(ctx, c, A, eigenvalues, jobtype, uplo);
}

namespace blas::dispatch::detail {

template <typename T>
bool syev_supports_cta(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    return ops::syev::supports<T>(ctx, A, ops::syev::Cta{});
}
template <typename T>
bool syev_supports_blocked(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo) {
    return ops::syev::supports<T>(ctx, A, ops::syev::Blocked{});
}
template <typename T>
bool syev_supports_two_stage(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo) {
    return ops::syev::supports<T>(ctx, A, ops::syev::TwoStage{});
}

#define SYEV_SUPPORTS_INSTANTIATE(fp)                                                                      \
    template bool syev_supports_cta<fp>(const Queue&, const MatrixView<fp, MatrixFormat::Dense>&);         \
    template bool syev_supports_blocked<fp>(const Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo); \
    template bool syev_supports_two_stage<fp>(const Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo);
SYEV_SUPPORTS_INSTANTIATE(float)
SYEV_SUPPORTS_INSTANTIATE(double)
SYEV_SUPPORTS_INSTANTIATE(std::complex<float>)
SYEV_SUPPORTS_INSTANTIATE(std::complex<double>)
#undef SYEV_SUPPORTS_INSTANTIATE

}  // namespace blas::dispatch::detail

#define SYEV_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syev) BATCHLAS_INSTANTIATE_OP(B_, fp, syev_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_INSTANTIATE)
#undef SYEV_INSTANTIATE

}  // namespace batchlas

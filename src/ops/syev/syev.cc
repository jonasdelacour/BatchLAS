// SPDX-License-Identifier: MIT
// syev (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-syev): select::run
// takes the first entry of the nearest tuned/syev.<device>.txt row that can_run() admits. Cta,
// CtaFused and Jacobi are the three n <= 32 sub-group solvers; Blocked (sytrd_blocked + stedc) and
// TwoStage (sy2sb + sb2st + stedc) serve every n and call their sub-ops directly.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/settings.hh>

#include <src/ops/syev/choice.hh>
#include <src/ops/syev/vendor.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

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

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;
template <typename Scalar>
using Real = typename base_type<Scalar>::type;

// The sub-group solvers' own ceiling (syev_cta.cc, syev_cta_fused.cc, syev_jacobi_cta.cc all
// throw above it): one matrix per 32-lane partition, the CTA invariant of docs/developer/agent-guide.md §11.
inline constexpr std::int64_t kSmallMaxN = 32;

template <typename Scalar>
select::Key key_of(const MV<Scalar>& A, JobType jobz) {
    return {{"jobz", jobz == JobType::EigenVectors ? "V" : "N"}, {"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or compute garbage. The native
// drivers are GPU kernels on a GPU backend (a NETLIB instantiation would call netlib on device
// memory) and need a square A with n >= 1; the small solvers add n <= 32 and a 32-wide
// sub-group, Blocked and TwoStage a batch >= 1 (the small drivers do not check it). Blocked and
// TwoStage take Upper by mirroring into Lower (uplo_mirror.hh). Every family, the vendor included,
// runs each item at the top-level order, so none takes a heterogeneous batch. Cta and CtaFused also
// exclude ROCm, where steqr_cta gives wrong eigenvalues (gfx1200). A batch mask runs on every native
// family and the NETLIB vendor; the device vendors cannot skip an item.
template <Backend B, typename Scalar>
bool can_run(const SyevChoice& c, const select::Device& d, const MV<Scalar>& A) {
    const std::int64_t n = A.rows();
    const bool masked = batchlas::detail::any_batch_mask(A);
    const bool homogeneous = !A.is_heterogeneous();
    const bool native = B != Backend::NETLIB && d.is_gpu && homogeneous && A.rows() == A.cols() && n >= 1;
    const bool small = native && n <= kSmallMaxN && d.has_sg32;
    const bool cta_steqr = small && B != Backend::ROCM;
    const bool large = native && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Cta) { return cta_steqr; },
        [&](CtaFused) { return cta_steqr; },
        [&](Jacobi) { return small; },
        [&](Blocked) { return large; },
        [&](TwoStage) { return large; },
        [&](Vendor) { return d.has_vendor && homogeneous && (!masked || B == Backend::NETLIB); },
    }, c);
}

// Deliberately slower and more robust than the CTA STEQR defaults: syev runs inside syevx,
// where an inaccurate Ritz solve stagnates the outer iteration.
template <typename Scalar>
SteqrParams<Scalar> cta_steqr_params() {
    SteqrParams<Scalar> params{};
    params.max_sweeps = 400;
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
    return params;
}

// Exactly the chosen family's need (R5).
template <Backend B, typename Scalar>
std::size_t workspace(Queue& q, const SyevChoice& c, const MV<Scalar>& A, Span<Real<Scalar>> w, JobType jobz,
                      Uplo uplo) {
    return std::visit(overloaded{
        [&](Cta) { return syev_cta_buffer_size<B, Scalar>(q, A, jobz, cta_steqr_params<Scalar>()); },
        [&](CtaFused) { return syev_cta_fused_buffer_size<B, Scalar>(q, A, jobz, cta_steqr_params<Scalar>()); },
        [&](Jacobi) { return syev_jacobi_cta_buffer_size<B, Scalar>(q, A, jobz); },
        [&](Blocked) { return syev_blocked_buffer_size<B, Scalar>(q, A, jobz, uplo, StedcParams<Real<Scalar>>{}); },
        [&](TwoStage) { return syev_two_stage_buffer_size<B, Scalar>(q, A, jobz, uplo, StedcParams<Real<Scalar>>{}); },
        [&](Vendor) {
            return ops::syev::detail::syev_vendor_buffer_size_or_throw<B, Scalar>(q, A, w, jobz, uplo);
        },
    }, c);
}

// Every arm clears or stores `info` itself, so the launch adds no clear of its own.
template <Backend B, typename Scalar>
Event launch(Queue& q, const SyevChoice& c, const MV<Scalar>& A, Span<Real<Scalar>> w, JobType jobz, Uplo uplo,
             Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Cta) { return syev_cta<B, Scalar>(q, A, w, jobz, uplo, ws, cta_steqr_params<Scalar>(), 0, info); },
        [&](CtaFused) { return syev_cta_fused<B, Scalar>(q, A, w, jobz, uplo, ws, cta_steqr_params<Scalar>(), 0,
                                                         info); },
        [&](Jacobi) { return syev_jacobi_cta<B, Scalar>(q, A, w, jobz, uplo, ws, JacobiParams<Scalar>(), info); },
        [&](Blocked) { return syev_blocked<B, Scalar>(q, A, w, jobz, uplo, ws, StedcParams<Real<Scalar>>{}, info); },
        [&](TwoStage) { return syev_two_stage<B, Scalar>(q, A, w, jobz, uplo, ws, StedcParams<Real<Scalar>>{}, info); },
        [&](Vendor) { return ops::syev::detail::syev_vendor_or_throw<B, Scalar>(q, A, w, jobz, uplo, ws, info); },
    }, c);
}

// Every driver refuses a non-square A, and the vendor would read rows() as the order.
// Thrown before choose() so neither Auto nor a pin can take it.
template <typename Scalar>
void validate(const MV<Scalar>& A, const char* who) {
    (void)batchlas::detail::call_mask(who, A);
    if (A.rows() != A.cols())
        throw batchlas::invalid_argument(std::string(who) + ": A must be square, got " + std::to_string(A.rows()) +
                                         " x " + std::to_string(A.cols()));
}

// Capability only (the Python binding, the LOBPCG accuracy pin): the same can_run, where the only
// backend term is the ROCm exclusion.
template <typename Scalar>
bool supports(const Queue& q, const MV<Scalar>& A, const SyevChoice& c) {
    const select::Device& d = select::describe(q.device(), Backend::Auto, false);
    if (q.backend() == Backend::ROCM) return can_run<Backend::ROCM, Scalar>(c, d, A);
    return can_run<Backend::Auto, Scalar>(c, d, A);
}

}  // namespace ops::syev

template <Backend Back, typename Scalar>
Event syev(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A,
           Span<typename base_type<Scalar>::type> eigenvalues,
           JobType jobtype, Uplo uplo, Span<std::byte> workspace, Span<int32_t> info) {
    ops::syev::validate<Scalar>(A, "syev");
    // An empty batch is a no-op under any pin, as gemm's is: it never reaches selection.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = ops::syev::key_of<Scalar>(A, jobtype);
    return select::run<Back, Scalar>(
        ops::syev::spec, ctx, key, ops::syev::candidates<Scalar>(),
        [&](const auto& c, const auto& d) { return ops::syev::can_run<Back, Scalar>(c, d, A); }, shape, key,
        [&](const auto& c) {
            if (workspace.size() < ops::syev::workspace<Back, Scalar>(ctx, c, A, eigenvalues, jobtype, uplo))
                throw batchlas::workspace_error("syev: insufficient workspace for chosen provider");
            // Blocked and TwoStage require an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return ops::syev::launch<Back, Scalar>(q, c, A, eigenvalues, jobtype, uplo, workspace, info);
            });
        });
}

template <Backend Back, typename Scalar>
size_t syev_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A,
                        Span<typename base_type<Scalar>::type> eigenvalues, JobType jobtype, Uplo uplo) {
    ops::syev::validate<Scalar>(A, "syev_buffer_size");
    if (A.batch_size() == 0) return 0;
    const auto c = select::pick<Back, Scalar>(
        ops::syev::spec, ctx, ops::syev::key_of<Scalar>(A, jobtype), ops::syev::candidates<Scalar>(),
        [&](const auto& k, const auto& d) { return ops::syev::can_run<Back, Scalar>(k, d, A); });
    return ops::syev::workspace<Back, Scalar>(ctx, c, A, eigenvalues, jobtype, uplo);
}

namespace ops::syev::detail {

template <typename Scalar>
bool syev_supports_cta(const Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A) {
    return ops::syev::supports<Scalar>(ctx, A, ops::syev::Cta{});
}
template <typename Scalar>
bool syev_supports_blocked(const Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Uplo) {
    return ops::syev::supports<Scalar>(ctx, A, ops::syev::Blocked{});
}
template <typename Scalar>
bool syev_supports_two_stage(const Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Uplo) {
    return ops::syev::supports<Scalar>(ctx, A, ops::syev::TwoStage{});
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

}  // namespace ops::syev::detail

#define SYEV_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syev) BATCHLAS_INSTANTIATE_OP(B_, fp, syev_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_INSTANTIATE)
#undef SYEV_INSTANTIATE

}  // namespace batchlas

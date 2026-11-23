// SPDX-License-Identifier: MIT
// gesvd (flat-kernel-selection.md §4.3, R1): select::run takes the first entry of the nearest
// tuned/gesvd.<device>.txt row that can_run() admits. Jacobi is one-sided Jacobi in local memory;
// Cta forms the normal equations and solves them with syev_cta; Blocked bidiagonalises (or, Hermitian,
// runs syev_blocked); Vendor is the solver library (cuSOLVER gesvdjBatched, LAPACKE on the host).

#include <batchlas/backend_config.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/settings.hh>

#include <src/ops/gesvd/choice.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/ops/gesvd/gesvd_native.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::gesvd {

using select::overloaded;

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;
template <typename Scalar>
using SV = Span<typename base_type<Scalar>::type>;

// The call after canonicalisation: Thin is rewritten to All wherever the two coincide, once,
// so sizing and running see the same jobs (enums.hh).
struct Job {
    SvdVectors jobu, jobvh;
    std::optional<Uplo> herm;
    bool thin() const { return jobu == SvdVectors::Thin || jobvh == SvdVectors::Thin; }
    bool vectors() const { return jobu != SvdVectors::None || jobvh != SvdVectors::None; }
};

template <typename Scalar>
Job canonical(const MV<Scalar>& A, SvdVectors jobu, SvdVectors jobvh, std::optional<Uplo> herm) {
    const std::int64_t k = std::min<std::int64_t>(A.rows(), A.cols());
    return {canonical_jobu(jobu, A.rows(), k), canonical_jobvh(jobvh, A.cols(), k), herm};
}

template <typename Scalar>
select::Key key_of(const MV<Scalar>& A, const Job& j) {
    const char* herm = !j.herm ? "N" : (*j.herm == Uplo::Lower ? "L" : "U");
    const char* vec = j.thin() ? "thin" : (j.vectors() ? "all" : "none");
    return {{"herm", herm}, {"vec", vec}, {"m", A.rows()}, {"n", A.cols()}};
}

// A batch whose items sit ld * cols apart: gesvdjBatched takes no stride argument.
template <typename Scalar>
bool packed(const MV<Scalar>& M) {
    return M.batch_size() == 1 || M.stride() == static_cast<std::int64_t>(M.ld()) * M.cols();
}

// Correctness only (R3): false means the driver throws or answers wrongly. Jacobi and Cta need a
// GPU; Blocked runs wherever natives do (a vendor-free host too). Jacobi has no
// Hermitian form; Cta runs complex input only through the Hermitian shortcut and cannot write a
// genuinely thin U; Blocked takes complex general (complex gebrd to a real bidiagonal) and either
// Hermitian triangle (syev_blocked mirrors Upper into Lower). Vendor is exactly what its binding
// runs: cuSOLVER gesvdjBatched (m, n <= 32, packed A and U, no thin; a values-only non-square call
// faults inside it, docs/perf/gesvd.md), nothing on ROCm, LAPACKE on the host. Every family runs each item at the top-level (m, n): no heterogeneous batch.
// Masked: every native family and NETLIB's host loop skip an inactive item (Blocked wherever every
// child it reaches does, gesvd_blocked_honours_mask); a strided vendor call cannot.
template <Backend B, typename Scalar>
bool can_run(const GesvdChoice& c, const select::Device& d, const MV<Scalar>& A, const MV<Scalar>& U,
             const MV<Scalar>& Vh,
             const Job& j) {
    constexpr bool kReal = std::is_same_v<Scalar, typename base_type<Scalar>::type>;
    const std::int64_t m = A.rows(), n = A.cols(), md = std::max(m, n);
    const bool masked = detail::any_batch_mask(A, U, Vh);
    const bool homogeneous = !A.is_heterogeneous();
    const bool shape_ok = homogeneous && m >= 1 && n >= 1 && A.batch_size() >= 1;
    // Jacobi and Cta are sub-group kernels with GPU-only entry guards; Blocked also runs on a
    // vendor-free host (select::natives_run_on), where its NETLIB arms take the unblocked gebrd.
    const bool native = d.is_gpu && shape_ok;
    const bool blocked_native = select::natives_run_on(d) && shape_ok;
    const bool square = m == n;
    const bool vendor_binds = B == Backend::CUDA
        ? md <= ops::gesvd::kGesvdVendorCudaMaxDim && packed(A) && !j.thin() && (j.vectors() || square) &&
              (j.jobu == SvdVectors::None || packed(U))
        : B != Backend::ROCM;
    return std::visit(overloaded{
        [&](Jacobi) {
            const auto budget = static_cast<std::size_t>(std::max<std::int64_t>(0, d.slm_budget));
            return native && d.has_sg32 && !j.herm && md <= ops::gesvd::gesvd_jacobi_max_dim<Scalar>(j.vectors(),
                                                                                                     budget);
        },
        [&](Cta) {
            const bool form = j.herm ? square : kReal;
            return native && d.has_sg32 && md <= ops::gesvd::kGesvdCtaMaxDim && !j.thin() && form;
        },
        [&](Blocked) {
            const bool mask_ok = !masked || ops::gesvd::gesvd_blocked_honours_mask<B, Scalar>(
                                                m, n, j.jobu != SvdVectors::None, j.jobvh != SvdVectors::None);
            return blocked_native && mask_ok && (!j.herm || square);
        },
        [&](Vendor) { return d.has_vendor && homogeneous && (!masked || B == Backend::NETLIB) && vendor_binds; },
    }, c);
}

// Exactly the chosen family's need (R5).
template <Backend B, typename Scalar>
std::size_t workspace(Queue& q, const GesvdChoice& c, const MV<Scalar>& A, SV<Scalar> s, const MV<Scalar>& U,
                      const MV<Scalar>& Vh,
                      const Job& j) {
    return std::visit(overloaded{
        [&](Jacobi) { return gesvdj_cta_buffer_size<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh); },
        [&](Cta) {
            return j.herm ? gesvd_cta_buffer_size<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm)
                          : gesvd_cta_buffer_size<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh);
        },
        [&](Blocked) {
            return j.herm ? gesvd_blocked_buffer_size<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm)
                          : gesvd_blocked_buffer_size<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh);
        },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::gesvd_vendor_buffer_size<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

// Every driver clears or stores `info` itself, and exactly one runs. Jacobi gets default
// GesvdjParams: no sweep cap is reachable from the public entry point.
template <Backend B, typename Scalar>
Event launch(Queue& q, const GesvdChoice& c, const MV<Scalar>& A, SV<Scalar> s, const MV<Scalar>& U,
             const MV<Scalar>& Vh,
             const Job& j, Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Jacobi) { return gesvdj_cta<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, GesvdjParams<Scalar>(),
                                                   info); },
        [&](Cta) {
            return j.herm ? gesvd_cta<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm, ws, info)
                          : gesvd_cta<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, info);
        },
        [&](Blocked) {
            return j.herm ? gesvd_blocked<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, *j.herm, ws, info)
                          : gesvd_blocked<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, info);
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::gesvd_vendor<B, Scalar>(q, A, s, U, Vh, j.jobu, j.jobvh, ws, info);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

template <Backend B, typename Scalar>
Event run(Queue& ctx, const MV<Scalar>& A, SV<Scalar> s, const MV<Scalar>& U, const MV<Scalar>& Vh, SvdVectors jobu,
          SvdVectors jobvh,
          std::optional<Uplo> herm, Span<std::byte> ws, Span<int32_t> info) {
    // An empty batch is a no-op under any pin, as gemm's is: it never reaches selection.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    (void)detail::call_mask("gesvd", A, U, Vh);
    const Job j = canonical<Scalar>(A, jobu, jobvh, herm);
    // The coverage row's key: m, n, k = min(m, n); uplo is the Hermitian triangle (general: Lower).
    // The scope opens before sizing, so a driver that refuses the shape still leaves its row, as before.
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = std::min<std::int64_t>(A.rows(), A.cols()),
                                .batch = A.batch_size(), .uplo = herm.value_or(Uplo::Lower)};
    const select::Key key = key_of<Scalar>(A, j);
    return select::run<B, Scalar>(
        spec, ctx, key, candidates<Scalar>(), [&](const auto& c,
                                                  const auto& d) { return can_run<B, Scalar>(c, d, A, U, Vh, j); },
        shape, key, [&](const auto& c) {
            if (ws.size() < workspace<B, Scalar>(ctx, c, A, s, U, Vh, j))
                throw batchlas::workspace_error("gesvd: insufficient workspace for chosen provider");
            // The native drivers need an in-order queue.
            return select::on_in_order_queue(ctx,
                                             [&](Queue& q) { return launch<B, Scalar>(q, c, A, s, U, Vh, j, ws,
                                                                                      info); });
        });
}

template <Backend B, typename Scalar>
std::size_t size(Queue& ctx, const MV<Scalar>& A, SV<Scalar> s, const MV<Scalar>& U, const MV<Scalar>& Vh,
                 SvdVectors jobu,
                 SvdVectors jobvh, std::optional<Uplo> herm) {
    if (A.batch_size() == 0) return 0;
    (void)detail::call_mask("gesvd", A, U, Vh);
    const Job j = canonical<Scalar>(A, jobu, jobvh, herm);
    const auto c = select::pick<B, Scalar>(spec, ctx, key_of<Scalar>(A, j), candidates<Scalar>(),
                                           [&](const auto& k,
                                               const auto& d) { return can_run<B, Scalar>(k, d, A, U, Vh, j); });
    return workspace<B, Scalar>(ctx, c, A, s, U, Vh, j);
}

}  // namespace ops::gesvd

template <Backend B, typename Scalar>
Event gesvd(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A,
            Span<typename base_type<Scalar>::type> singular_values,
            const MatrixView<Scalar, MatrixFormat::Dense>& U, const MatrixView<Scalar, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh, Span<std::byte> workspace, Span<int32_t> info) {
    return ops::gesvd::run<B, Scalar>(ctx, A, singular_values, U, Vh, jobu, jobvh, std::nullopt, workspace, info);
}

template <Backend B, typename Scalar>
Event gesvd(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A,
            Span<typename base_type<Scalar>::type> singular_values,
            const MatrixView<Scalar, MatrixFormat::Dense>& U, const MatrixView<Scalar, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh, Uplo hermitian_uplo, Span<std::byte> workspace, Span<int32_t> info) {
    return ops::gesvd::run<B, Scalar>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo, workspace, info);
}

template <Backend B, typename Scalar>
size_t gesvd_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A,
                         Span<typename base_type<Scalar>::type> singular_values, const MatrixView<Scalar,
                         MatrixFormat::Dense>& U,
                         const MatrixView<Scalar, MatrixFormat::Dense>& Vh, SvdVectors jobu, SvdVectors jobvh) {
    return ops::gesvd::size<B, Scalar>(ctx, A, singular_values, U, Vh, jobu, jobvh, std::nullopt);
}

template <Backend B, typename Scalar>
size_t gesvd_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A,
                         Span<typename base_type<Scalar>::type> singular_values, const MatrixView<Scalar,
                         MatrixFormat::Dense>& U,
                         const MatrixView<Scalar, MatrixFormat::Dense>& Vh, SvdVectors jobu, SvdVectors jobvh,
                         Uplo hermitian_uplo) {
    return ops::gesvd::size<B, Scalar>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo);
}

#define GESVD_INSTANTIATE(B_, fp)                                                                       \
    BATCHLAS_INSTANTIATE_OP(B_, fp, gesvd) BATCHLAS_INSTANTIATE_OP(B_, fp, gesvd_buffer_size)             \
    BATCHLAS_INSTANTIATE(sig::gesvd_hermitian<BATCHLAS_UNPAREN fp>, gesvd, B_, BATCHLAS_UNPAREN fp)     \
    BATCHLAS_INSTANTIATE(sig::gesvd_buffer_size_hermitian<BATCHLAS_UNPAREN fp>, gesvd_buffer_size, B_, \
                         BATCHLAS_UNPAREN fp)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GESVD_INSTANTIATE)
#undef GESVD_INSTANTIATE

}  // namespace batchlas

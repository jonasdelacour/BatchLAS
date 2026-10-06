// orgqr (flat-kernel-selection.md §4.3, R1): select::run takes the first entry of the nearest
// tuned/orgqr.<dtype>.<device>.txt row that can_run() admits. Blocked writes an identity and applies Q to
// it through the public ormqr, which picks its own kernel; Vendor is the library's per-item orgqr loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/orgqr_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::orgqr {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A) {
    return {{"m", A.rows()}, {"n", A.cols()}};
}

// Correctness only (R3): false means the driver would throw. Blocked's clauses are
// orgqr_blocked_dispatch's own checks (one identity and one apply serve the whole batch, and
// Q's columns live in C^m, so n <= m); its tau-length check is an argument error, left to the
// driver. The vendor accepts every shape, a heterogeneous batch included, as before.
template <class T>
bool can_run(const OrgqrChoice& c, const select::Device& d, const MV<T>& A) {
    return std::visit(overloaded{
        [&](Blocked) {
            return d.is_gpu && sycl_orgqr::orgqr_blocked_available<T>() && !A.is_heterogeneous() &&
                   A.rows() >= 1 && A.cols() >= 1 && A.batch_size() >= 1 && A.cols() <= A.rows();
        },
        [&](Vendor) { return d.has_vendor; },
    }, c);
}

// The apply seams: the public ormqr and its sizing, so the apply picks its own kernel and its
// workspace comes from the same routed query as the call.
template <Backend B, class T>
Event apply_q(Queue& c, const MV<T>& oa, const MV<T>& oc, Side os, Transpose ot, Span<T> otau,
              Span<std::byte> ows, int32_t obs) {
    return ormqr<B, T>(c, oa, oc, os, ot, otau, ows, obs);
}
template <Backend B, class T>
std::size_t apply_q_size(Queue& c, const MV<T>& oa, const MV<T>& oc, Side os, Transpose ot, Span<T> otau,
                         int32_t obs) {
    return ormqr_buffer_size<B, T>(c, oa, oc, os, ot, otau, obs);
}

template <Backend B, class T>
Event launch(Queue& q, const OrgqrChoice& c, const MV<T>& A, Span<T> tau, Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](Blocked) {
            return sycl_orgqr::orgqr_blocked_dispatch<T>(q, A, tau, ws, apply_q<B, T>, apply_q_size<B, T>);
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::orgqr_vendor<B, T>(q, A, tau, ws);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5). The vendor's is batch-linear (a per-item loop), so a
// native call must never be sized by it: at cdouble n=64 batch=8192 that is ~4.6 GB.
// evidence: docs/perf/qr.md#the-orgqr_buffer_size-latent-defect
template <Backend B, class T>
std::size_t workspace(Queue& q, const OrgqrChoice& c, const MV<T>& A, Span<T> tau) {
    return std::visit(overloaded{
        [&](Blocked) { return sycl_orgqr::orgqr_blocked_buffer_size<T>(q, A, tau, apply_q_size<B, T>); },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::orgqr_vendor_buffer_size<B, T>(q, A, tau);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::orgqr

template <Backend Back, typename T>
Event orgqr(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau, Span<std::byte> workspace) {
    orgqr_validate_params<T>(A);
    // The coverage row's key, as the old builder set it: k = min(m, n) reflectors, (Left, NoTrans).
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = std::min(A.rows(), A.cols()),
                                .batch = A.batch_size(), .transA = Transpose::NoTrans, .side = Side::Left};
    select::Key trace_key = ops::orgqr::key_of<T>(A);
    trace_key.emplace_back("batch", A.batch_size());
    return select::run<Back, T>(
        ops::orgqr::spec, ctx, ops::orgqr::key_of<T>(A), ops::orgqr::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::orgqr::can_run<T>(c, d, A); }, shape, trace_key,
        [&](const auto& c) { return ops::orgqr::launch<Back, T>(ctx, c, A, tau, workspace); });
}

template <Backend Back, typename T>
size_t orgqr_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau) {
    orgqr_validate_params<T>(A);
    const auto c = select::pick<Back, T>(
        ops::orgqr::spec, ctx, ops::orgqr::key_of<T>(A), ops::orgqr::candidates<T>(),
        [&](const auto& k, const auto& d) { return ops::orgqr::can_run<T>(k, d, A); });
    return ops::orgqr::workspace<Back, T>(ctx, c, A, tau);
}

#define ORGQR_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, orgqr) BATCHLAS_INSTANTIATE_OP(B_, fp, orgqr_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(ORGQR_INSTANTIATE)
#undef ORGQR_INSTANTIATE

}  // namespace batchlas

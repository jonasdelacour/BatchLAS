// orgqr: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public orgqr() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/orgqr.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// Blocked writes an identity and applies Q to it through the public ormqr, which picks its own
// kernel; Vendor is the library's per-item orgqr loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

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

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A) {
    return {{"m", A.rows()}, {"n", A.cols()}};
}

// orgqr's vendor is the factorization group (cuBLAS and cuSOLVER on CUDA), the same gate that
// compiles the Vendor arm below, so its Device carries that group in has_vendor_solver.
template <Backend B>
const select::Device& device(const Queue& q) {
    return select::device_of<B>(q, dispatch::factorization_vendor_available<B>);
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
        [&](Vendor) { return d.has_vendor_solver; },
    }, c);
}

template <Backend B, class T>
OrgqrChoice choose(Queue& q, const MV<T>& A) {
    const select::Device& d = device<B>(q);
    auto ok = [&](const OrgqrChoice& c) { return can_run<T>(c, d, A); };
    try {
        return select::choose("orgqr", select::dtype_name<T>(), d, key_of<T>(A), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!dispatch::factorization_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            dispatch::throw_no_vendor_route<T>(dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = device<B>(q);
    return select::native_facts(candidates<T>(), [&](const OrgqrChoice& c) { return can_run<T>(c, d, A); });
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
            if constexpr (dispatch::factorization_vendor_available<B>)
                return backend::orgqr_vendor<B, T>(q, A, tau, ws);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
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
            if constexpr (dispatch::factorization_vendor_available<B>)
                return backend::orgqr_vendor_buffer_size<B, T>(q, A, tau);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
        },
    }, c);
}

}  // namespace ops::orgqr

template <Backend Back, typename T>
Event orgqr(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau, Span<std::byte> workspace) {
    orgqr_validate_params<T>(A);
    const auto c = ops::orgqr::choose<Back, T>(ctx, A);
    // The coverage row's key, as the old builder set it: k = min(m, n) reflectors, (Left, NoTrans).
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.n = A.cols();
    shape.k = std::min(A.rows(), A.cols());
    shape.side = Side::Left;
    shape.transA = Transpose::NoTrans;
    select::Key trace_key = ops::orgqr::key_of<T>(A);
    trace_key.emplace_back("batch", A.batch_size());
    select::TraceScope trace("orgqr", c, shape, ops::orgqr::native_facts<Back, T>(ctx, A), trace_key);
    return ops::orgqr::launch<Back, T>(ctx, c, A, tau, workspace);
}

template <Backend Back, typename T>
size_t orgqr_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau) {
    orgqr_validate_params<T>(A);
    return ops::orgqr::workspace<Back, T>(ctx, ops::orgqr::choose<Back, T>(ctx, A), A, tau);
}

#define ORGQR_INSTANTIATE(B_, fp)                                    \
    BATCHLAS_INSTANTIATE(sig::orgqr<fp>, orgqr, B_, fp)              \
    BATCHLAS_INSTANTIATE(sig::orgqr_buffer_size<fp>, orgqr_buffer_size, B_, fp)

#define ORGQR_ALL(B_)                          \
    ORGQR_INSTANTIATE(B_, float)               \
    ORGQR_INSTANTIATE(B_, double)              \
    ORGQR_INSTANTIATE(B_, std::complex<float>) \
    ORGQR_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
ORGQR_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
ORGQR_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
ORGQR_ALL(Backend::NETLIB)
#endif

#undef ORGQR_ALL
#undef ORGQR_INSTANTIATE

}  // namespace batchlas

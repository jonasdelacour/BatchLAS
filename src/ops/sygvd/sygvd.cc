// sygvd: the generalized Hermitian-definite eigensolver. select::run takes the first entry of the nearest
// tuned/sygvd.<device>.txt row that can_run() admits. Compose is potrf(B) -> sygst -> syev -> back-transform,
// each child a public routed op; Vendor is cuSOLVER ?sygvd per item or LAPACKE on the host. sygvd_factored
// takes potrf's factor of B and always composes (no vendor library takes a factored B).
// evidence: docs/perf/sygvd.md#sygvd-routes

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/functions/sygvd.hh>

#include <src/ops/sygvd/choice.hh>
#include <src/ops/sygvd/sygvd_native.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/select/vendor.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::sygvd {

using select::overloaded;

template <Backend B>
inline constexpr bool vendor_linked =
    (B == Backend::CUDA && select::kHasCudaVendor) || (B == Backend::NETLIB && select::kHasNetlib);

template <class T>
select::Key key_of(const Args<T>& a) {
    return {{"jobz", a.jobz == JobType::EigenVectors ? "V" : "N"}, {"n", a.A.rows()}, {"batch", a.A.batch_size()}};
}

// Correctness only. The vendor loops run each item at the top-level order, and no library takes a
// factored B; compose's children refuse what they cannot run themselves.
template <Backend B, class T>
bool can_run(const SygvdChoice& c, const select::Device& d, const Args<T>& a) {
    const bool homogeneous = !a.A.is_heterogeneous() && !a.B.is_heterogeneous() && a.A.rows() >= 1;
    // The NETLIB vendor honours a batch mask (cuSOLVER cannot skip an item); Compose keeps refusing it.
    const bool masked = batchlas::detail::any_batch_mask(a.A, a.B);
    return std::visit(overloaded{
        [&](Compose) { return homogeneous && !masked; },  // mask: lift after potrf (W2a), syev
        [&](Vendor) {
            return vendor_linked<B> && d.has_vendor && homogeneous && !a.factored && (!masked || B == Backend::NETLIB);
        },
    }, c);
}

template <Backend B, class T>
std::size_t workspace(Queue& q, const SygvdChoice& c, const Args<T>& a) {
    return std::visit(overloaded{
        [&](Compose) { return compose_workspace<B, T>(q, a); },
        [&](Vendor) -> std::size_t {
            if constexpr (vendor_linked<B>)
                return backend::sygvd_vendor_buffer_size<B, T>(q, a.A, a.B, a.w, a.itype, a.jobz, a.uplo);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const SygvdChoice& c, const Args<T>& a, Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Compose) { return compose<B, T>(q, a, ws, info); },
        [&](Vendor) -> Event {
            if constexpr (vendor_linked<B>)
                return backend::sygvd_vendor<B, T>(q, a.A, a.B, a.w, a.itype, a.jobz, a.uplo, ws, info);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

template <class T>
void validate(const char* who, const Args<T>& a) {
    sygst_validate_params<T>(who, a.A, a.B, a.itype, a.uplo);
    (void)batchlas::detail::call_mask(who, a.A, a.B);
    const auto need = static_cast<std::size_t>(a.A.rows()) * static_cast<std::size_t>(a.A.batch_size());
    if (a.w.size() < need)
        throw batchlas::invalid_argument(std::string(who) + ": eigenvalues holds " + std::to_string(a.w.size()) +
                                         " values, the call needs n * batch = " + std::to_string(need));
}

template <Backend B, class T>
Event run(Queue& ctx, const Args<T>& a, Span<std::byte> ws, Span<int32_t> info, const char* who) {
    validate<T>(who, a);
    if (a.A.batch_size() == 0 || a.A.rows() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = a.A.rows(), .n = a.A.rows(), .k = a.A.rows(), .batch = a.A.batch_size(),
                                .uplo = a.uplo};
    const select::Key key = key_of<T>(a);
    return select::run<B, T>(
        spec, ctx, key, candidates<T>(), [&](const auto& c, const auto& d) { return can_run<B, T>(c, d, a); }, shape,
        key, [&](const auto& c) {
            if (ws.size() < workspace<B, T>(ctx, c, a))
                throw batchlas::workspace_error(std::string(who) + ": workspace is shorter than " + who +
                                                "_buffer_size()");
            return select::on_in_order_queue(ctx, [&](Queue& q) { return launch<B, T>(q, c, a, ws, info); });
        });
}

template <Backend B, class T>
std::size_t buffer_size(Queue& ctx, const Args<T>& a, const char* who) {
    validate<T>(who, a);
    if (a.A.batch_size() == 0 || a.A.rows() == 0) return 0;
    const auto c = select::pick<B, T>(spec, ctx, key_of<T>(a), candidates<T>(),
                                      [&](const auto& k, const auto& d) { return can_run<B, T>(k, d, a); });
    return workspace<B, T>(ctx, c, a);
}

}  // namespace ops::sygvd

template <Backend Back, typename T>
Event sygvd(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz, Uplo uplo, Span<std::byte> work_space,
            Span<int32_t> info) {
    return ops::sygvd::run<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, false}, work_space, info, "sygvd");
}

template <Backend Back, typename T>
size_t sygvd_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, Span<typename base_type<T>::type> eigenvalues,
                         int itype, JobType jobz, Uplo uplo) {
    return ops::sygvd::buffer_size<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, false}, "sygvd_buffer_size");
}

template <Backend Back, typename T>
Event sygvd_factored(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                     Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz, Uplo uplo,
                     Span<std::byte> work_space, Span<int32_t> info) {
    return ops::sygvd::run<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, true}, work_space, info,
                                    "sygvd_factored");
}

template <Backend Back, typename T>
size_t sygvd_factored_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B,
                                  Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz, Uplo uplo) {
    return ops::sygvd::buffer_size<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, true},
                                            "sygvd_factored_buffer_size");
}

#define SYGVD_INSTANTIATE(B_, fp)                                                                          \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd) BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd_buffer_size)              \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd_factored) BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd_factored_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYGVD_INSTANTIATE)
#undef SYGVD_INSTANTIATE

}  // namespace batchlas

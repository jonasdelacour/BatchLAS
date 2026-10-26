// sygvd: the generalized Hermitian-definite eigensolver. select::run takes the first entry of the nearest
// tuned/sygvd.<device>.txt row that can_run() admits. Compose is potrf(B) -> sygst -> syev -> back-transform,
// each child a public routed op; Vendor is cuSOLVER ?sygvd per item or LAPACKE on the host. sygvd_factored
// takes potrf's factor of B and always composes (no vendor library takes a factored B).
// evidence: docs/perf/sygvd.md#sygvd-routes

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/functions/sygvd.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/util/mempool.hh>

#include <src/ops/sygvd/choice.hh>
#include <src/ops/sygvd/sygvd_native.hh>
#include <src/select/select.hh>
#include <src/select/vendor.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::sygvd {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <class T>
using Real = typename base_type<T>::type;

template <Backend B>
inline constexpr bool vendor_linked =
    (B == Backend::CUDA && select::kHasCudaVendor) || (B == Backend::NETLIB && select::kHasNetlib);

// Everything one call reads besides the queue and the workspace.
template <class T>
struct Args {
    MV<T> A, B;
    Span<Real<T>> w;
    int itype;
    JobType jobz;
    Uplo uplo;
    bool factored;  ///< sygvd_factored: B holds potrf's factor
};

template <class T>
select::Key key_of(const Args<T>& a) {
    return {{"jobz", a.jobz == JobType::EigenVectors ? "V" : "N"}, {"n", a.A.rows()}, {"batch", a.A.batch_size()}};
}

// Correctness only. The vendor loops run each item at the top-level order, and no library takes a
// factored B; compose's children refuse what they cannot run themselves.
template <Backend B, class T>
bool can_run(const SygvdChoice& c, const select::Device& d, const Args<T>& a) {
    const bool homogeneous = !a.A.is_heterogeneous() && !a.B.is_heterogeneous() && a.A.rows() >= 1;
    return std::visit(overloaded{
        [&](Compose) { return homogeneous; },
        [&](Vendor) { return vendor_linked<B> && d.has_vendor && homogeneous && !a.factored; },
    }, c);
}

template <class T>
constexpr Transpose kH = is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;

// Compose's workspace: potrf's per-item status (kept to the end), itype 3's back-transform product, and
// one scratch region the children take in turn (they run one after another on an in-order queue).
template <class T>
struct ComposeLayout {
    Span<int32_t> finfo;
    Span<T> prod;
    Span<T*> prod_ptr;
    Span<std::byte> child;
};

template <Backend B, class T>
std::size_t child_bytes(Queue& q, const Args<T>& a) {
    std::size_t s = std::max(::batchlas::sygst_buffer_size<B, T>(q, a.A, a.B, a.itype, a.uplo),
                             ::batchlas::syev_buffer_size<B, T>(q, a.A, a.w, a.jobz, a.uplo));
    if (!a.factored) s = std::max(s, ::batchlas::potrf_buffer_size<B, T>(q, a.B, a.uplo));
    return s;
}

template <Backend B, class T>
ComposeLayout<T> compose_layout(Queue& q, BumpAllocator& pool, const Args<T>& a) {
    ComposeLayout<T> l;
    const auto n = static_cast<std::size_t>(a.A.rows()), batch = static_cast<std::size_t>(a.A.batch_size());
    if (!a.factored) l.finfo = pool.allocate<int32_t>(q, batch);
    if (a.itype == 3 && a.jobz == JobType::EigenVectors) {
        l.prod = pool.allocate<T>(q, n * n * batch);
        l.prod_ptr = pool.allocate<T*>(q, batch);
    }
    l.child = pool.allocate<std::byte>(q, child_bytes<B, T>(q, a));
    return l;
}

template <Backend B, class T>
std::size_t workspace(Queue& q, const SygvdChoice& c, const Args<T>& a) {
    return std::visit(overloaded{
        [&](Compose) { return workspace_bytes([&](BumpAllocator& p) { return compose_layout<B, T>(q, p, a); }); },
        [&](Vendor) -> std::size_t {
            if constexpr (vendor_linked<B>)
                return backend::sygvd_vendor_buffer_size<B, T>(q, a.A, a.B, a.w, a.itype, a.jobz, a.uplo);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

template <Backend B, class T>
Event compose(Queue& q, const Args<T>& a, Span<std::byte> ws, Span<int32_t> info) {
    BumpAllocator pool(ws);
    const auto l = compose_layout<B, T>(q, pool, a);
    const bool lower = a.uplo == Uplo::Lower;
    const T one(1);
    if (!a.factored) {
        (void)::batchlas::potrf<B, T>(q, a.B, a.uplo, l.child, l.finfo);
        // A failed item would carry a partial (possibly NaN) factor into every later child.
        (void)sycl_sygvd::sygvd_identity_failed<T>(q, a.B, a.uplo, Span<const int32_t>(l.finfo.data(), l.finfo.size()));
    }
    (void)::batchlas::sygst<B, T>(q, a.A, a.B, a.itype, a.uplo, l.child);
    Event ev = ::batchlas::syev<B, T>(q, a.A, a.w, a.jobz, a.uplo, l.child, info);
    if (a.jobz == JobType::EigenVectors) {
        if (a.itype != 3) {  // x = L^-H y (Lower) or U^-1 y (Upper)
            ev = ::batchlas::trsm<B, T>(q, a.B, a.A, one, Side::Left, a.uplo, lower ? kH<T> : Transpose::NoTrans, Diag::NonUnit);
        } else {  // x = L y (Lower) or U^H y (Upper), formed out of place then copied back
            const int n = a.A.rows(), batch = a.A.batch_size();
            const MV<T> P(l.prod.data(), n, n, n, n * n, batch, l.prod_ptr.data());
            (void)::batchlas::trmm<B, T>(q, a.B, a.A, P, one, Side::Left, a.uplo, lower ? Transpose::NoTrans : kH<T>, Diag::NonUnit);
            ev = MV<T>::copy(q, a.A, P);
        }
    }
    if (!a.factored && info.size() >= static_cast<std::size_t>(a.A.batch_size()))
        ev = sycl_sygvd::sygvd_merge_info(q, a.A.rows(), Span<const int32_t>(l.finfo.data(), l.finfo.size()),
                                          Span<int32_t>(info.data(), static_cast<std::size_t>(a.A.batch_size())));
    return ev;
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

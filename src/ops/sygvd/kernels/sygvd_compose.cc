// sygvd's Compose family (sygvd_native.hh): potrf(B) -> sygst -> syev -> back-transform, each child a
// public routed op on the caller's in-order queue. evidence: docs/perf/sygvd.md#sygvd-routes

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/util/mempool.hh>

#include <src/ops/sygvd/sygvd_native.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>

namespace batchlas::ops::sygvd {

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

// The call's mask on A and on B (unless B is broadcast), whichever carried it: every child, and every
// workspace query, sees the masked operands.
template <class T>
Args<T> masked_args(const Args<T>& a) {
    const Span<const std::uint8_t> mask = ::batchlas::detail::call_mask("sygvd", a.A, a.B);
    Args<T> m = a;
    m.A = a.A.with_batch_mask(mask);
    if (!::batchlas::detail::mask_view_is_broadcast(a.B, a.A.batch_size())) m.B = a.B.with_batch_mask(mask);
    return m;
}

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
std::size_t compose_workspace(Queue& q, const Args<T>& a) {
    const Args<T> m = masked_args(a);
    return workspace_bytes([&](BumpAllocator& p) { return compose_layout<B, T>(q, p, m); });
}

template <Backend B, class T>
Event compose(Queue& q, const Args<T>& a_in, Span<std::byte> ws, Span<int32_t> info) {
    const Args<T> a = masked_args(a_in);
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(a.A.batch_mask());
    BumpAllocator pool(ws);
    const auto l = compose_layout<B, T>(q, pool, a);
    const bool lower = a.uplo == Uplo::Lower;
    const T one(1);
    if (!a.factored) {
        (void)::batchlas::potrf<B, T>(q, a.B, a.uplo, l.child, l.finfo);
        // A failed item would carry a partial (possibly NaN) factor into every later child.
        (void)sycl_sygvd::sygvd_identity_failed<T>(q, a.B, a.uplo, Span<const int32_t>(l.finfo.data(), l.finfo.size()),
                                                   mask);
    }
    (void)::batchlas::sygst<B, T>(q, a.A, a.B, a.itype, a.uplo, l.child);
    Event ev = ::batchlas::syev<B, T>(q, a.A, a.w, a.jobz, a.uplo, l.child, info);
    if (a.jobz == JobType::EigenVectors) {
        if (a.itype != 3) {  // x = L^-H y (Lower) or U^-1 y (Upper)
            ev = ::batchlas::trsm<B, T>(q, a.B, a.A, one, Side::Left, a.uplo, lower ? kH<T> : Transpose::NoTrans, Diag::NonUnit);
        } else {  // x = L y (Lower) or U^H y (Upper), formed out of place then copied back
            const int n = a.A.rows(), batch = a.A.batch_size();
            const MV<T> P =
                MV<T>(l.prod.data(), n, n, n, n * n, batch, l.prod_ptr.data()).with_batch_mask(a.A.batch_mask());
            (void)::batchlas::trmm<B, T>(q, a.B, a.A, P, one, Side::Left, a.uplo, lower ? Transpose::NoTrans : kH<T>, Diag::NonUnit);
            ev = MV<T>::copy(q, a.A, P);
        }
    }
    if (!a.factored && info.size() >= static_cast<std::size_t>(a.A.batch_size()))
        ev = sycl_sygvd::sygvd_merge_info(q, a.A.rows(), Span<const int32_t>(l.finfo.data(), l.finfo.size()),
                                          Span<int32_t>(info.data(), static_cast<std::size_t>(a.A.batch_size())), mask);
    return ev;
}

#define SYGVD_COMPOSE_INSTANTIATE(B_, fp)                                                                     \
    template std::size_t compose_workspace<B_, BATCHLAS_UNPAREN fp>(Queue&, const Args<BATCHLAS_UNPAREN fp>&);  \
    template Event compose<B_, BATCHLAS_UNPAREN fp>(Queue&, const Args<BATCHLAS_UNPAREN fp>&, Span<std::byte>,  \
                                                    Span<int32_t>);
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYGVD_COMPOSE_INSTANTIATE)
#undef SYGVD_COMPOSE_INSTANTIATE

}  // namespace batchlas::ops::sygvd

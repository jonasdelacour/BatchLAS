// sygst: the two-sided triangular reduction of sygvd. select::run takes the first entry of the nearest
// tuned/sygst.<device>.txt row that can_run() admits. TwoSided mirrors A's uplo triangle and applies the
// factor from both sides with the public trsm (itype 1) or trmm (itypes 2 and 3), so its children select
// their own kernels; Vendor is LAPACKE ?sygst/?hegst on the host (cuSOLVER has none).
// evidence: docs/perf/sygvd.md#sygvd-routes

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/util/mempool.hh>

#include <src/ops/sygst/choice.hh>
#include <src/select/select.hh>
#include <src/select/vendor.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::sygst {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <Backend B>
inline constexpr bool vendor_linked = B == Backend::NETLIB && select::kHasNetlib;

template <class T>
select::Key key_of(const MV<T>& A) {
    return {{"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only. The children refuse what they cannot run; a heterogeneous batch has no family.
template <Backend B, class T>
bool can_run(const SygstChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& Bf) {
    const bool homogeneous = !A.is_heterogeneous() && !Bf.is_heterogeneous() && A.rows() >= 1;
    return std::visit(overloaded{
        [&](TwoSided) { return homogeneous; },
        [&](Vendor) { return vendor_linked<B> && d.has_vendor && homogeneous; },
    }, c);
}

template <class T>
constexpr Transpose kH = is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;

// itypes 2 and 3 form one product out of place: an n x n buffer and its pointer array per item.
template <class T>
std::size_t twosided_workspace(Queue& q, const MV<T>& A, int itype) {
    if (itype == 1) return 0;
    auto pool = BumpAllocator::measuring();
    (void)pool.allocate<T>(q, static_cast<std::size_t>(A.rows()) * A.rows() * A.batch_size());
    (void)pool.allocate<T*>(q, static_cast<std::size_t>(A.batch_size()));
    return pool.required_bytes();
}

template <Backend B, class T>
std::size_t workspace(Queue& q, const SygstChoice& c, const MV<T>& A, int itype) {
    return std::visit(overloaded{
        [&](TwoSided) { return twosided_workspace<T>(q, A, itype); },
        [&](Vendor) -> std::size_t { return 0; },
    }, c);
}

template <Backend B, class T>
Event twosided(Queue& q, const MV<T>& A, const MV<T>& Bf, int itype, Uplo uplo, Span<std::byte> ws) {
    const bool lower = uplo == Uplo::Lower;
    const T one(1);
    (void)A.hermitize(q, uplo);
    if (itype == 1) {  // Lower: L^-1 A L^-H. Upper: U^-H A U^-1.
        (void)::batchlas::trsm<B, T>(q, Bf, A, one, Side::Left, uplo, lower ? Transpose::NoTrans : kH<T>, Diag::NonUnit);
        return ::batchlas::trsm<B, T>(q, Bf, A, one, Side::Right, uplo, lower ? kH<T> : Transpose::NoTrans, Diag::NonUnit);
    }
    const int n = A.rows(), batch = A.batch_size();
    BumpAllocator pool(ws);
    const Span<T> buf = pool.allocate<T>(q, static_cast<std::size_t>(n) * n * batch);
    const Span<T*> ptr = pool.allocate<T*>(q, static_cast<std::size_t>(batch));
    const MV<T> W(buf.data(), n, n, n, n * n, batch, ptr.data());
    // Lower: W := L^H A, A := W L. Upper: W := U A, A := W U^H.
    (void)::batchlas::trmm<B, T>(q, Bf, A, W, one, Side::Left, uplo, lower ? kH<T> : Transpose::NoTrans, Diag::NonUnit);
    return ::batchlas::trmm<B, T>(q, Bf, W, A, one, Side::Right, uplo, lower ? Transpose::NoTrans : kH<T>, Diag::NonUnit);
}

template <Backend B, class T>
Event launch(Queue& q, const SygstChoice& c, const MV<T>& A, const MV<T>& Bf, int itype, Uplo uplo,
             Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](TwoSided) { return twosided<B, T>(q, A, Bf, itype, uplo, ws); },
        [&](Vendor) -> Event {
            if constexpr (vendor_linked<B>) return backend::sygst_vendor<B, T>(q, A, Bf, itype, uplo, ws);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

template <Backend B, class T>
SygstChoice pick(Queue& ctx, const MV<T>& A, const MV<T>& Bf) {
    return select::pick<B, T>(spec, ctx, key_of<T>(A), candidates<T>(),
                              [&](const auto& k, const auto& d) { return can_run<B, T>(k, d, A, Bf); });
}

}  // namespace ops::sygst

template <Backend Back, typename T>
Event sygst(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            int itype, Uplo uplo, Span<std::byte> work_space) {
    sygst_validate_params<T>("sygst", A, B, itype, uplo);
    if (A.batch_size() == 0 || A.rows() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = ops::sygst::key_of<T>(A);
    return select::run<Back, T>(
        ops::sygst::spec, ctx, key, ops::sygst::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::sygst::can_run<Back, T>(c, d, A, B); }, shape, key,
        [&](const auto& c) {
            if (work_space.size() < ops::sygst::workspace<Back, T>(ctx, c, A, itype))
                throw batchlas::workspace_error("sygst: workspace is shorter than sygst_buffer_size()");
            // The mirror and both triangular products chain, so they need an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return ops::sygst::launch<Back, T>(q, c, A, B, itype, uplo, work_space);
            });
        });
}

template <Backend Back, typename T>
size_t sygst_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, int itype, Uplo uplo) {
    sygst_validate_params<T>("sygst_buffer_size", A, B, itype, uplo);
    if (A.batch_size() == 0 || A.rows() == 0) return 0;
    return ops::sygst::workspace<Back, T>(ctx, ops::sygst::pick<Back, T>(ctx, A, B), A, itype);
}

#define SYGST_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, sygst) BATCHLAS_INSTANTIATE_OP(B_, fp, sygst_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYGST_INSTANTIATE)
#undef SYGST_INSTANTIATE

}  // namespace batchlas

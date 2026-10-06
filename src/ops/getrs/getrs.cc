// getrs (flat-kernel-selection.md §4.3, R1): select::run takes the first entry of the nearest
// tuned/getrs.<dtype>.<device>.txt row that can_run() admits. Cta permutes and solves both triangles in
// one kernel with the right-hand sides resident in local memory; Blocked applies the interchanges and
// calls the public trsm twice.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/trsm.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/getrs_native.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::getrs {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B) {
    return {{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or answer wrongly. Both natives
// re-check exactly this at their entry. A non-conforming pair (A not square, rows or batch
// disagreeing) is left to the vendor, as before. NETLIB writes true int64 pivots where the
// natives read packed 1-based int32, a silent wrong answer, so no native runs under it.
template <Backend B, class T>
bool can_run(const GetrsChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& Bm) {
    const std::int64_t n = A.rows(), nrhs = Bm.cols();
    const bool conforming = A.cols() == n && Bm.rows() == n && Bm.batch_size() == A.batch_size();
    const bool native = B != Backend::NETLIB && d.is_gpu && d.has_sg32 && conforming && !A.is_heterogeneous() &&
                        !Bm.is_heterogeneous() && n >= 1 && nrhs >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Cta) {
            // The fused work-group never drops below 32 lanes (getrs_fused_wg).
            const auto cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(static_cast<std::size_t>(d.slm_budget));
            return native && sycl_getrs::getrs_fused_available<T>() && d.max_wg >= 32 &&
                   nrhs <= sycl_getrs::kGetrsFusedMaxRhs &&
                   static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs) <= cap;
        },
        [&](Blocked) { return native && sycl_getrs::getrs_blocked_available<T>(); },
        [&](Vendor) { return d.has_vendor; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const GetrsChoice& c, const MV<T>& A, const MV<T>& Bm, Transpose transA,
             Span<std::int64_t> pivots, Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_getrs::getrs_fused_dispatch<T>(q, A, Bm, transA, pivots, ws); },
        [&](Blocked) {
            return sycl_getrs::getrs_blocked_dispatch<T>(
                q, A, Bm, transA, pivots, ws,
                [](Queue& c2, const MV<T>& ta, const MV<T>& tb, T talpha, Side tside, Uplo tuplo, Transpose ttrans,
                   Diag tdiag) { return trsm<B, T>(c2, ta, tb, talpha, tside, tuplo, ttrans, tdiag); });
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::getrs_vendor<B, T>(q, A, Bm, transA, pivots, ws);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Nothing here dereferences A or B: a measuring pass
// hands null data pointers.
template <Backend B, class T>
std::size_t workspace(Queue& q, const GetrsChoice& c, const MV<T>& A, const MV<T>& Bm, Transpose transA) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_getrs::getrs_fused_buffer_size<T>(q, A, Bm, transA); },
        [&](Blocked) { return sycl_getrs::getrs_blocked_buffer_size<T>(q, A, Bm, transA); },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::getrs_vendor_buffer_size<B, T>(q, A, Bm, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::getrs

template <Backend Back, typename T>
Event getrs(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            Transpose transA, Span<int64_t> pivots, Span<std::byte> work_space) {
    namespace o = ops::getrs;
    getrs_validate_params<T>(A, B);
    // The coverage row's key: m = k = order, n = nrhs, and transA (the field separating rows).
    const coverage::Shape shape{.m = A.rows(), .n = B.cols(), .k = A.rows(), .batch = A.batch_size(),
                                .transA = transA};
    const select::Key key = o::key_of<T>(A, B);
    return select::run<Back, T>(
        o::spec, ctx, key, o::candidates<T>(),
        [&](const auto& c, const auto& d) { return o::can_run<Back, T>(c, d, A, B); }, shape, key,
        [&](const auto& c) { return o::launch<Back, T>(ctx, c, A, B, transA, pivots, work_space); });
}

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, Transpose transA) {
    namespace o = ops::getrs;
    getrs_validate_params<T>(A, B);
    const auto c = select::pick<Back, T>(o::spec, ctx, o::key_of<T>(A, B), o::candidates<T>(),
                                         [&](const auto& k, const auto& d) { return o::can_run<Back, T>(k, d, A, B); });
    return o::workspace<Back, T>(ctx, c, A, B, transA);
}

#define GETRS_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, getrs) BATCHLAS_INSTANTIATE_OP(B_, fp, getrs_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GETRS_INSTANTIATE)
#undef GETRS_INSTANTIATE

}  // namespace batchlas

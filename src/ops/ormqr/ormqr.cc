// ormqr (flat-kernel-selection.md §4.3, R1): select::run takes the first entry of the nearest
// tuned/ormqr.<dtype>.<device>.txt row that can_run() admits. Blocked is the WY apply (larft + level-3
// updates) in ormqr_blocked.cc; Vendor is cuSOLVER / rocSOLVER / LAPACKE ormqr.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/settings.hh>
#include <batchlas/tuning_params.hh>

#include "choice.hh"
#include "vendor.hh"
#include "../../select/select.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::ormqr {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// C's extent that Q does not act on: its columns for Side::Left, its rows for Side::Right.
template <class T>
std::int64_t rhs_count(const MV<T>& C, Side side) {
    return side == Side::Left ? C.cols() : C.rows();
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Side side, Transpose trans) {
    const char* t = trans == Transpose::NoTrans ? "N" : (trans == Transpose::Trans ? "T" : "C");
    return {{"side", side == Side::Left ? "L" : "R"}, {"trans", t}, {"m", A.rows()},
            {"k", std::min(A.rows(), A.cols())}, {"q", rhs_count<T>(C, side)}, {"batch", A.batch_size()}};
}

// Correctness only (R3). Neither family serves complex Trans: ormqr_blocked throws `unsupported`
// and cuSOLVER / LAPACKE / rocSOLVER unmqr reject 'T' (cuSOLVER status 3). Blocked is also kept
// off non-GPU queues, as the old router did; dimension checks are left to the drivers.
template <class T>
bool can_run(const OrmqrChoice& c, const select::Device& d, Transpose trans) {
    const bool defined = !(is_std_complex_v<T> && trans == Transpose::Trans);
    return std::visit(overloaded{
        [&](Blocked) { return defined && d.is_gpu; },
        [&](Vendor) { return defined && d.has_vendor; },
    }, c);
}

// Complex Trans has no kernel anywhere, so it is an argument error, thrown before choose() so
// that no pin and no vendor-free build reports it as a routing miss.
template <class T>
void throw_if_undefined(Transpose trans, const char* who) {
    if (is_std_complex_v<T> && trans == Transpose::Trans)
        throw batchlas::invalid_argument(std::string(who) + ": Transpose::Trans of a complex Q is not an ormqr "
                                         "operation (unmqr takes N or C); use Transpose::ConjTrans");
}

// Blocked's WY width: a positive hint (a caller that knows its reflector count) clamped to
// [1, k], else the tuning table keyed on A.rows(). Derived from A alone, so sizing and the call
// agree.
template <class T>
std::int32_t block_size(const MV<T>& A, std::int32_t hint) {
    const std::int32_t k = static_cast<std::int32_t>(std::min(A.rows(), A.cols()));
    if (hint > 0) return std::max<std::int32_t>(1, std::min<std::int32_t>(hint, std::max<std::int32_t>(1, k)));
    return tuning::ormqr_block_size_for_n(static_cast<std::int32_t>(A.rows()));
}

// The vendor's spelling: real ormqr takes N or T only (cuSOLVER status 3 on C), and for a real
// Q, ConjTrans is Trans.
template <class T>
Transpose vendor_trans(Transpose trans) {
    return !is_std_complex_v<T> && trans == Transpose::ConjTrans ? Transpose::Trans : trans;
}

// Exactly the chosen family's need (R5).
template <Backend B, class T>
std::size_t workspace(Queue& q, const OrmqrChoice& c, const MV<T>& A, const MV<T>& C, Side side, Transpose trans,
                      Span<T> tau, std::int32_t hint) {
    return std::visit(overloaded{
        [&](Blocked) { return ormqr_blocked_buffer_size<B, T>(q, A, C, side, trans, tau, block_size<T>(A, hint)); },
        [&](Vendor) -> std::size_t {
            return blas::dispatch::detail::ormqr_vendor_buffer_size_or_throw<B, T>(q, A, C, side, vendor_trans<T>(trans),
                                                                                  tau);
        },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const OrmqrChoice& c, const MV<T>& A, const MV<T>& C, Side side, Transpose trans,
             Span<T> tau, Span<std::byte> ws, std::int32_t hint) {
    return std::visit(overloaded{
        [&](Blocked) { return ormqr_blocked<B, T>(q, A, C, side, trans, tau, ws, block_size<T>(A, hint)); },
        [&](Vendor) {
            return blas::dispatch::detail::ormqr_vendor_or_throw<B, T>(q, A, C, side, vendor_trans<T>(trans), tau, ws);
        },
    }, c);
}

}  // namespace ops::ormqr

template <Backend Back, typename T>
Event ormqr(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& C,
            Side side, Transpose trans, Span<T> tau, Span<std::byte> workspace, int32_t block_size_hint) {
    namespace o = ops::ormqr;
    o::throw_if_undefined<T>(trans, "ormqr");
    // The coverage row's key, as the old builder had it (m, n = A's extents, k = reflectors), plus
    // the backend, which that builder never set.
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = std::min(A.rows(), A.cols()),
                                .batch = A.batch_size(), .transA = trans, .side = side};
    const select::Key key = o::key_of<T>(A, C, side, trans);
    return select::run<Back, T>(
        o::spec, ctx, key, o::candidates<T>(),
        [&](const auto& c, const auto& d) { return o::can_run<T>(c, d, trans); }, shape, key, [&](const auto& c) {
            if (workspace.size() < o::workspace<Back, T>(ctx, c, A, C, side, trans, tau, block_size_hint))
                throw batchlas::workspace_error("ormqr: insufficient workspace for chosen provider");
            // Both families sequence several launches.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return o::launch<Back, T>(q, c, A, C, side, trans, tau, workspace, block_size_hint);
            });
        });
}

template <Backend Back, typename T>
size_t ormqr_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C, Side side, Transpose trans, Span<T> tau,
                         int32_t block_size_hint) {
    namespace o = ops::ormqr;
    o::throw_if_undefined<T>(trans, "ormqr_buffer_size");
    const auto c = select::pick<Back, T>(o::spec, ctx, o::key_of<T>(A, C, side, trans), o::candidates<T>(),
                                         [&](const auto& k, const auto& d) { return o::can_run<T>(k, d, trans); });
    return o::workspace<Back, T>(ctx, c, A, C, side, trans, tau, block_size_hint);
}

#define ORMQR_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, ormqr) BATCHLAS_INSTANTIATE_OP(B_, fp, ormqr_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(ORMQR_INSTANTIATE)
#undef ORMQR_INSTANTIATE

}  // namespace batchlas

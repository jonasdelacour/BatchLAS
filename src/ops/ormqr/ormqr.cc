// ormqr: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public ormqr() -> choose() -> std::visit -> launch. The kernel for a shape is the first
// runnable entry of the nearest row in tuned/ormqr.<dtype>.<device>.txt; can_run() below only
// removes entries that cannot run. Blocked is the WY apply (larft + level-3 updates) in
// ormqr_blocked.cc; Vendor is cuSOLVER / rocSOLVER / LAPACKE ormqr.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/no_route.hh>
#include "../../select/vendor.hh"
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

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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
        [&](Vendor) { return defined && d.has_vendor_solver; },
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

// ormqr's vendor is cuBLAS + cuSOLVER (factorization_vendor_available), not the potrf solver set.
template <Backend B>
const select::Device& device(Queue& q) {
    return select::device_of<B>(q, select::factorization_vendor_available<B>, select::level3_vendor_available<B>);
}

template <Backend B, class T>
OrmqrChoice choose(Queue& q, const MV<T>& A, const MV<T>& C, Side side, Transpose trans) {
    const select::Device& d = device<B>(q);
    auto ok = [&](const OrmqrChoice& c) { return can_run<T>(c, d, trans); };
    try {
        return select::choose("ormqr", select::dtype_name<T>(), d, key_of<T>(A, C, side, trans), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::factorization_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::ormqr, B, select::kFactorizationLibrary<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, Transpose trans) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = device<B>(q);
    return select::native_facts(candidates<T>(), [&](const OrmqrChoice& c) { return can_run<T>(c, d, trans); });
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
    ops::ormqr::throw_if_undefined<T>(trans, "ormqr");
    const auto c = ops::ormqr::choose<Back, T>(ctx, A, C, side, trans);
    const std::size_t need = ops::ormqr::workspace<Back, T>(ctx, c, A, C, side, trans, tau, block_size_hint);
    if (workspace.size() < need) throw batchlas::workspace_error("ormqr: insufficient workspace for chosen provider");
    // The coverage row's key, as the old builder had it (m, n = A's extents, k = reflectors), plus
    // the backend, which that builder never set.
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.n = A.cols();
    shape.k = std::min(A.rows(), A.cols());
    shape.side = side;
    shape.transA = trans;
    const select::Key trace_key = ops::ormqr::key_of<T>(A, C, side, trans);
    select::TraceScope trace("ormqr", c, shape, ops::ormqr::native_facts<Back, T>(ctx, trans), trace_key);
    // Both families sequence several launches; an out-of-order caller gets an in-order queue that
    // waits on its pending work. std::optional: a default Queue would build a real sycl::queue.
    Queue* run_q = &ctx;
    std::optional<Queue> in_order_q;
    if (!ctx.in_order()) {
        in_order_q.emplace(ctx, true);
        Event dep = ctx.get_event();
        in_order_q->enqueue(dep);
        run_q = &*in_order_q;
    }
    return ops::ormqr::launch<Back, T>(*run_q, c, A, C, side, trans, tau, workspace, block_size_hint);
}

template <Backend Back, typename T>
size_t ormqr_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C, Side side, Transpose trans, Span<T> tau,
                         int32_t block_size_hint) {
    ops::ormqr::throw_if_undefined<T>(trans, "ormqr_buffer_size");
    const auto c = ops::ormqr::choose<Back, T>(ctx, A, C, side, trans);
    return ops::ormqr::workspace<Back, T>(ctx, c, A, C, side, trans, tau, block_size_hint);
}

#define ORMQR_INSTANTIATE(B_, fp)                                   \
    BATCHLAS_INSTANTIATE(sig::ormqr<fp>, ormqr, B_, fp)             \
    BATCHLAS_INSTANTIATE(sig::ormqr_buffer_size<fp>, ormqr_buffer_size, B_, fp)

#define ORMQR_ALL(B_)                          \
    ORMQR_INSTANTIATE(B_, float)               \
    ORMQR_INSTANTIATE(B_, double)              \
    ORMQR_INSTANTIATE(B_, std::complex<float>) \
    ORMQR_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
ORMQR_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
ORMQR_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
ORMQR_ALL(Backend::NETLIB)
#endif

#undef ORMQR_ALL
#undef ORMQR_INSTANTIATE

}  // namespace batchlas

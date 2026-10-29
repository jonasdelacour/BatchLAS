// syfn: matrix functions of Hermitian matrices (fullerenes plan P2.2). A pure composition with no
// kernel choice of its own, so no Op and no route: syev, herk/syrk and gemm select for themselves,
// and its element-wise kernels (weights, info, column scale) live in kernels/syfn_kernels.cc.
// evidence: docs/perf/syfn.md#syfn-composition

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/syfn.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/util/mempool.hh>

#include <src/util/batch_mask.hh>
#include <src/ops/syfn/syfn_native.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas {
namespace ops::syfn {

// InvSqrt, Sqrt and Exp are >= 0 everywhere, so Y = V sqrt(f) and F = Y Y^H (herk) halves the
// flops. Inverse and Weights may be negative: Y = V f and F = Y V^H (gemm).
inline bool rank_k_form(SyfnKind k) { return k == SyfnKind::InvSqrt || k == SyfnKind::Sqrt || k == SyfnKind::Exp; }

template <class T>
Core<T> carve_core(const Device& dev, BumpAllocator& pool, int64_t n, int64_t m, int64_t batch) {
    Core<T> c;
    c.y = pool.allocate<T>(dev, size_t(n * m * batch));
    c.g = pool.allocate<float_t<T>>(dev, size_t(m * batch));
    return c;
}

template <class T>
struct Full {
    Span<T> v;             // n x n per item: the copy of A syev overwrites with V
    Span<float_t<T>> w;    // n per item
    Span<std::byte> syev;  // syev's own workspace
    Core<T> core;
};

// V (the copy syev overwrites) as a view: the workspace rebuilt for the call, carrying its mask.
template <class T>
MV<T> v_view(Span<T> v, int64_t n, int64_t batch, Span<const std::uint8_t> mask) {
    return MV<T>(v.data(), int(n), int(n), int(n), int(n * n), int(batch)).with_batch_mask(mask);
}

template <Backend B, class T>
Full<T> carve_full(Queue& ctx, BumpAllocator& pool, int64_t n, int64_t batch, Uplo uplo,
                   Span<const std::uint8_t> mask) {
    Full<T> f;
    f.v = pool.allocate<T>(ctx.device(), size_t(n * n * batch));
    f.w = pool.allocate<float_t<T>>(ctx.device(), size_t(n * batch));
    const MV<T> Vv = v_view(f.v, n, batch, mask);
    f.syev = pool.allocate<std::byte>(ctx.device(), syev_buffer_size<B, T>(ctx, Vv, f.w, JobType::EigenVectors, uplo));
    f.core = carve_core<T>(ctx.device(), pool, n, n, batch);
    return f;
}

// F := V diag(f(w)) V^H on the in-order queue q, for the items `mask` keeps. With `syev_info`
// nonempty, info[b] is n + its value where it is nonzero.
template <Backend B, class T>
Event from_eigen(Queue& q, const MV<T>& V, Span<float_t<T>> W, const MV<T>& F_in, SyfnFunction<float_t<T>> fn,
                 const Core<T>& ws, Span<int32_t> info, Span<int32_t> syev_info, Span<const std::uint8_t> mask) {
    using R = float_t<T>;
    const MV<T> F = F_in.with_batch_mask(mask);  // hermitize skips by F's own mask
    const int64_t n = V.rows(), m = V.cols(), batch = V.batch_size();
    const SyfnKind kind = fn.kind;
    const std::uint8_t* mp = detail::mask_ptr(mask);
    if (!info.empty()) launch_info<B, T>(q, V, W, kind, info, syev_info, mp);
    if (m == 0) return launch_zero<B, T>(q, V, F, mp);
    launch_scale<B, T>(q, V, W, fn, ws, mp);
    T* yp = ws.y.data();
    const MV<T> Y = MV<T>(yp, int(n), int(m), int(n), int(n * m), int(batch)).with_batch_mask(mask);
    if (rank_k_form(kind)) {
        if constexpr (is_std_complex_v<T>) herk<B, T>(q, Y, F, R(1), R(0), Uplo::Lower, Transpose::NoTrans);
        else syrk<B, T>(q, Y, F, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
    } else {
        constexpr Transpose herm = is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
        gemm<B, T>(q, Y, V, F, T(1), T(0), Transpose::NoTrans, herm);
    }
    // herk writes one triangle and gemm's two halves round differently: mirror for exact Hermiticity.
    return F.hermitize(q, Uplo::Lower);
}

// A sub-op can throw (NoRouteError vendor-free) after this op's own kernels are enqueued on the
// caller's workspace: drain them before the exception lets the caller free it.
template <class F>
Event drain_on_throw(Queue& q, F&& body) {
    try {
        return body();
    } catch (...) {
        q.wait();
        throw;
    }
}

// The weights travel inside SyfnFunction, which the dispatch spelling's pointer check cannot see.
template <class R>
void check_call(const Queue& q, const char* op, Span<int32_t> info, int64_t batch, SyfnFunction<R> fn) {
    if (!info.empty() && info.size() < size_t(batch)) {
        throw batchlas::invalid_argument(std::string(op) + ": info holds " + std::to_string(info.size()) +
                                         " values, need batch = " + std::to_string(batch));
    }
    if (fn.kind == SyfnKind::Weights && detail::pointer_checks_enabled())
        detail::require_arg_accessible(q, fn.weights, std::string(op) + ": weights");
}

// The host backend's syrk and herk are its BLAS, which would refuse a mask only after syev has
// written info: refuse up front there.
template <Backend B, typename... V>
void refuse_host_mask(const char* op, const V&... v) {
    if constexpr (B == Backend::NETLIB) ::batchlas::detail::refuse_batch_mask(op, v...);
}

}  // namespace ops::syfn

template <Backend B, typename T>
Event syfn(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& F,
           SyfnFunction<typename base_type<T>::type> fn, Uplo uplo, Span<std::byte> workspace, Span<int32_t> info) {
    const Span<const std::uint8_t> mask = detail::call_mask("syfn", A, F);
    ops::syfn::refuse_host_mask<B>("syfn", A, F);
    syfn_validate_params<T>(A, F, fn, uplo);
    ops::syfn::check_call(ctx, "syfn", info, A.batch_size(), fn);
    const int64_t n = A.rows(), batch = A.batch_size();
    if (batch == 0 || n == 0) return ctx.create_event_after_external_work();
    // The copy, syev and the core chain through the workspace: they need an in-order queue.
    return detail::on_in_order_queue(ctx, [&](Queue& q) {
        BumpAllocator pool(workspace);
        const ops::syfn::Full<T> ws = ops::syfn::carve_full<B, T>(q, pool, n, batch, uplo, mask);
        const ops::syfn::MV<T> V = ops::syfn::v_view(ws.v, n, batch, mask);
        return ops::syfn::drain_on_throw(q, [&] {
            ops::syfn::MV<T>::copy(q, V, A);
            syev<B, T>(q, V, ws.w, JobType::EigenVectors, uplo, ws.syev, info);
            return ops::syfn::from_eigen<B, T>(q, V, ws.w, F, fn, ws.core, info, info, mask);
        });
    });
}

template <Backend B, typename T>
size_t syfn_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& F, SyfnFunction<typename base_type<T>::type> fn,
                        Uplo uplo) {
    const Span<const std::uint8_t> mask = detail::call_mask("syfn", A, F);
    ops::syfn::refuse_host_mask<B>("syfn", A, F);
    syfn_validate_params<T>(A, F, fn, uplo);
    if (A.batch_size() == 0 || A.rows() == 0) return 0;
    BumpAllocator pool = BumpAllocator::measuring();
    (void)ops::syfn::carve_full<B, T>(ctx, pool, A.rows(), A.batch_size(), uplo, mask);
    return pool.required_bytes();
}

template <Backend B, typename T>
Event syfn_from_eigen(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& V,
                      Span<typename base_type<T>::type> eigenvalues, const MatrixView<T, MatrixFormat::Dense>& F,
                      SyfnFunction<typename base_type<T>::type> fn, Span<std::byte> workspace, Span<int32_t> info) {
    const Span<const std::uint8_t> mask = detail::call_mask("syfn_from_eigen", V, F);
    ops::syfn::refuse_host_mask<B>("syfn_from_eigen", V, F);
    syfn_from_eigen_validate_params<T>(V, eigenvalues, F, fn);
    ops::syfn::check_call(ctx, "syfn_from_eigen", info, V.batch_size(), fn);
    const int64_t n = V.rows(), m = V.cols(), batch = V.batch_size();
    if (batch == 0 || n == 0) return ctx.create_event_after_external_work();
    return detail::on_in_order_queue(ctx, [&](Queue& q) {
        BumpAllocator pool(workspace);
        const ops::syfn::Core<T> ws = ops::syfn::carve_core<T>(q.device(), pool, n, m, batch);
        return ops::syfn::drain_on_throw(
            q, [&] { return ops::syfn::from_eigen<B, T>(q, V, eigenvalues, F, fn, ws, info, Span<int32_t>{}, mask); });
    });
}

template <Backend B, typename T>
size_t syfn_from_eigen_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& V,
                                   Span<typename base_type<T>::type> eigenvalues,
                                   const MatrixView<T, MatrixFormat::Dense>& F,
                                   SyfnFunction<typename base_type<T>::type> fn) {
    ops::syfn::refuse_host_mask<B>("syfn_from_eigen", V, F);
    syfn_from_eigen_validate_params<T>(V, eigenvalues, F, fn);
    if (V.batch_size() == 0 || V.rows() == 0) return 0;
    BumpAllocator pool = BumpAllocator::measuring();
    (void)ops::syfn::carve_core<T>(ctx.device(), pool, V.rows(), V.cols(), V.batch_size());
    return pool.required_bytes();
}

#define SYFN_INSTANTIATE(B_, fp)                                                                     \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syfn) BATCHLAS_INSTANTIATE_OP(B_, fp, syfn_buffer_size)          \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syfn_from_eigen) BATCHLAS_INSTANTIATE_OP(B_, fp, syfn_from_eigen_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYFN_INSTANTIATE)
#undef SYFN_INSTANTIATE

}  // namespace batchlas

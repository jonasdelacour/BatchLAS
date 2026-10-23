// sysv (flat-kernel-selection.md §4.3): select::run takes the first entry of the nearest
// tuned/sysv.<device>.txt row that can_run() admits. Cta is the fused factor-and-solve kernel; Compose
// is the public sytrf (which selects its own kernel) then sytrs. evidence: docs/perf/sytrf.md

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/sysv.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/util/mempool.hh>

#include <src/ops/sysv/choice.hh>
#include <src/ops/sytrf/sytrf_native.hh>
#include <src/select/select.hh>
#include <src/select/vendor.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::sysv {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B) {
    return {{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}};
}

// Correctness only (R3); Compose's children decide for themselves, except on a heterogeneous batch,
// which no sytrf family serves.
template <class T>
bool can_run(const SysvChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& B) {
    const bool homogeneous = !A.is_heterogeneous() && !B.is_heterogeneous();
    return std::visit(overloaded{
        [&](Cta) {
            return d.is_gpu && d.has_sg32 && homogeneous && d.max_wg >= sycl_sytrf::kSytrfCtaWg && A.rows() >= 1 &&
                   A.rows() <= sycl_sytrf::kSytrfCtaMaxN;
        },
        [&](Compose) { return homogeneous; },
    }, c);
}

// sytrs skips an item whose sytrf status is nonzero (LAPACK ?sysv keeps that B), so Compose always
// needs the status: the caller's span when it is long enough, else one drawn here.
template <Backend Back, class T>
Span<int32_t> compose_layout(Queue& q, BumpAllocator& pool, const MV<T>& A, Uplo uplo, Span<int32_t> info,
                             Span<std::byte>* rest) {
    const auto batch = static_cast<std::size_t>(A.batch_size());
    Span<int32_t> st = info.size() >= batch ? info : pool.allocate<int32_t>(q, batch);
    const std::size_t child = sytrf_buffer_size<Back, T>(q, A, uplo);
    Span<std::byte> ws = pool.allocate<std::byte>(q, child);
    if (rest) *rest = ws;
    return st;
}

template <Backend Back, class T>
Event launch(Queue& q, const SysvChoice& c, const MV<T>& A, const MV<T>& Bm, Uplo uplo, Span<int64_t> pivots,
             Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_sytrf::sysv_cta_dispatch<T>(q, A, Bm, uplo, pivots, ws, info); },
        [&](Compose) {
            BumpAllocator pool(ws);
            Span<std::byte> child;
            const Span<int32_t> st = compose_layout<Back, T>(q, pool, A, uplo, info, &child);
            (void)sytrf<Back, T>(q, A, uplo, pivots, child, st);
            if constexpr (Back == Backend::NETLIB) {
                if constexpr (select::kHasNetlib) return backend::sytrs_vendor<Back, T>(q, A, Bm, uplo, pivots, st.data());
                else return sytrs<Back, T>(q, A, Bm, uplo, pivots, Span<std::byte>{});
            } else {
                return sycl_sytrf::sytrs_native_dispatch<T>(q, A, Bm, uplo, pivots, st.data());
            }
        },
    }, c);
}

template <Backend Back, class T>
std::size_t workspace(Queue& q, const SysvChoice& c, const MV<T>& A, Uplo uplo) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_sytrf::sytrf_native_buffer_size<T>(q, A); },
        [&](Compose) {
            return workspace_bytes([&](BumpAllocator& p) {
                return compose_layout<Back, T>(q, p, A, uplo, Span<int32_t>{}, nullptr);
            });
        },
    }, c);
}

}  // namespace ops::sysv

template <Backend Back, typename T>
Event sysv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           Uplo uplo, Span<int64_t> pivots, Span<std::byte> work_space, Span<int32_t> info) {
    sytrs_validate_params<T>(A, B, uplo);
    if (A.batch_size() == 0 || A.rows() == 0) return ctx.create_event_after_external_work();
    if (A.is_heterogeneous() || B.is_heterogeneous())
        throw batchlas::internal_error("sysv: a heterogeneous batch, which no sysv kernel serves.");
    const coverage::Shape shape{.m = A.rows(), .n = B.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = ops::sysv::key_of<T>(A, B);
    return select::run<Back, T>(
        ops::sysv::spec, ctx, key, ops::sysv::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::sysv::can_run<T>(c, d, A, B); }, shape, key,
        [&](const auto& c) {
            // Compose chains sytrf into sytrs, so it needs an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return ops::sysv::launch<Back, T>(q, c, A, B, uplo, pivots, work_space, info);
            });
        });
}

template <Backend Back, typename T>
size_t sysv_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo) {
    sytrs_validate_params<T>(A, B, uplo);
    if (A.batch_size() == 0 || A.rows() == 0) return 0;
    if (A.is_heterogeneous() || B.is_heterogeneous())
        throw batchlas::internal_error("sysv_buffer_size: a heterogeneous batch, which no sysv kernel serves.");
    const auto c = select::pick<Back, T>(
        ops::sysv::spec, ctx, ops::sysv::key_of<T>(A, B), ops::sysv::candidates<T>(),
        [&](const auto& k, const auto& d) { return ops::sysv::can_run<T>(k, d, A, B); });
    return ops::sysv::workspace<Back, T>(ctx, c, A, uplo);
}

#define SYSV_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, sysv) BATCHLAS_INSTANTIATE_OP(B_, fp, sysv_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYSV_INSTANTIATE)
#undef SYSV_INSTANTIATE

}  // namespace batchlas

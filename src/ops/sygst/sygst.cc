// sygst: the two-sided triangular reduction of sygvd. select::run takes the first entry of the nearest
// tuned/sygst.<device>.txt row that can_run() admits. Blocked is LAPACK's dsygst/zhegst schedule on the
// uplo triangles only: the SLM diagonal kernel (kernels/sygst_diag.cc) and the public trsm, gemm (for
// LAPACK's hemm/symm), her2k/syr2k and trmm, each selecting its own kernel. TwoSided mirrors A and applies the factor from
// both sides with two full trsm or trmm; Vendor is LAPACKE ?sygst/?hegst on the host (cuSOLVER has none).
// evidence: docs/perf/sygvd.md#sygvd-routes, docs/perf/sygvd.md#sygst-the-blocked-family

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/sygst.hh>

#include <src/ops/sygst/choice.hh>
#include <src/ops/sygst/sygst_native.hh>
#include <src/select/select.hh>
#include <src/select/vendor.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::sygst {

using select::overloaded;

template <Backend B>
inline constexpr bool vendor_linked = B == Backend::NETLIB && select::kHasNetlib;

template <class T>
select::Key key_of(const MV<T>& A, int itype, Uplo uplo) {
    return {{"itype", itype}, {"uplo", uplo == Uplo::Lower ? "L" : "U"}, {"n", A.rows()}, {"batch", A.batch_size()}};
}

inline bool blocked_nb_compiled(int nb) { return nb == 32 || nb == 64 || nb == 128 || nb == 256; }

// Correctness only. The children refuse what they cannot run; a heterogeneous batch has no family.
// Blocked's children are GPU level-3 natives, and its diagonal kernel must fit the device's SLM.
template <Backend B, class T>
bool can_run(const SygstChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& Bf) {
    const bool homogeneous = !A.is_heterogeneous() && !Bf.is_heterogeneous() && A.rows() >= 1;
    return std::visit(overloaded{
        [&](const Blocked& b) {
            const int kb = std::min(sycl_sygst::kDiagMaxOrder, A.rows());
            return homogeneous && d.is_gpu && blocked_nb_compiled(b.nb) && d.max_wg >= sycl_sygst::kDiagWg &&
                   static_cast<std::int64_t>(sycl_sygst::diag_slm_bytes<T>(kb)) <= d.slm_budget;
        },
        [&](TwoSided) { return homogeneous; },
        [&](Vendor) { return vendor_linked<B> && d.has_vendor && homogeneous; },
    }, c);
}

// ---- dispatch -------------------------------------------------------------------------------------

template <Backend B, class T>
std::size_t workspace(Queue& q, const SygstChoice& c, const MV<T>& A, int itype) {
    return std::visit(overloaded{
        [&](const Blocked& b) { return blocked_workspace<T>(q, A, b.nb, itype); },
        [&](TwoSided) { return twosided_workspace<T>(q, A, itype); },
        [&](Vendor) -> std::size_t { return 0; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const SygstChoice& c, const MV<T>& A, const MV<T>& Bf, int itype, Uplo uplo,
             Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](const Blocked& b) { return blocked<B, T>(q, A, Bf, b.nb, itype, uplo, ws); },
        [&](TwoSided) { return twosided<B, T>(q, A, Bf, itype, uplo, ws); },
        [&](Vendor) -> Event {
            if constexpr (vendor_linked<B>) return backend::sygst_vendor<B, T>(q, A, Bf, itype, uplo, ws);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

template <Backend B, class T>
SygstChoice pick(Queue& ctx, const MV<T>& A, const MV<T>& Bf, int itype, Uplo uplo) {
    return select::pick<B, T>(spec, ctx, key_of<T>(A, itype, uplo), candidates<T>(),
                              [&](const auto& k, const auto& d) { return can_run<B, T>(k, d, A, Bf); });
}

}  // namespace ops::sygst

template <Backend Back, typename T>
Event sygst(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            int itype, Uplo uplo, Span<std::byte> work_space) {
    sygst_validate_params<T>("sygst", A, B, itype, uplo);
    if (A.batch_size() == 0 || A.rows() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = ops::sygst::key_of<T>(A, itype, uplo);
    return select::run<Back, T>(
        ops::sygst::spec, ctx, key, ops::sygst::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::sygst::can_run<Back, T>(c, d, A, B); }, shape, key,
        [&](const auto& c) {
            if (work_space.size() < ops::sygst::workspace<Back, T>(ctx, c, A, itype))
                throw batchlas::workspace_error("sygst: workspace is shorter than sygst_buffer_size()");
            // Every child reads the previous one's output, so they need an in-order queue.
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
    return ops::sygst::workspace<Back, T>(ctx, ops::sygst::pick<Back, T>(ctx, A, B, itype, uplo), A, itype);
}

#define SYGST_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, sygst) BATCHLAS_INSTANTIATE_OP(B_, fp, sygst_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYGST_INSTANTIATE)
#undef SYGST_INSTANTIATE

}  // namespace batchlas

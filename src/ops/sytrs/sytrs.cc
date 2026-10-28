// The public sytrs: validation, the empty problem, then sytrs.hh's routed solve.
// evidence: docs/perf/sytrf.md

#include <batchlas/blas/functions/sytrf.hh>

#include <src/ops/sytrs/sytrs.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas {

template <Backend B, typename T>
Event sytrs(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& Bm,
            Uplo uplo, Span<int64_t> pivots, Span<std::byte> work_space) {
    sytrs_validate_params<T>(A, Bm, uplo);
    (void)detail::call_mask("sytrs", A, Bm);
    // An empty problem is a no-op under any pin: it never reaches selection.
    if (A.batch_size() == 0 || A.rows() == 0 || Bm.cols() == 0) return ctx.create_event_after_external_work();
    return ops::sytrs::run<B, T>(ctx, A, Bm, uplo, pivots, work_space, nullptr);
}

template <Backend B, typename T>
size_t sytrs_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& Bm, Uplo uplo) {
    sytrs_validate_params<T>(A, Bm, uplo);
    (void)detail::call_mask("sytrs", A, Bm);
    if (A.batch_size() == 0 || A.rows() == 0 || Bm.cols() == 0) return 0;
    return ops::sytrs::buffer_size<B, T>(ctx, A, Bm);
}

#define SYTRS_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sytrs) BATCHLAS_INSTANTIATE_OP(B_, fp, sytrs_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYTRS_INSTANTIATE)
#undef SYTRS_INSTANTIATE

}  // namespace batchlas

// SPDX-License-Identifier: MIT
// sygvd_masked and sygvd_factored_masked: the public sygvd / sygvd_factored on packed copies of the active
// items, scattered back (as syev_masked). evidence: docs/design/masked-batches.md#masked-batches-design

#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/functions/sygvd.hh>
#include <batchlas/util/mempool.hh>

#include <src/kernels/common/batch_mask.hh>
#include <src/select/select.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>

namespace batchlas {
namespace {

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <class T>
using Real = typename base_type<T>::type;

template <class T>
struct MaskedArgs {
    MV<T> A, B;
    Span<Real<T>> w;
    int itype;
    JobType jobz;
    Uplo uplo;
    bool factored;
};

template <Backend B, class T>
std::size_t solve_bytes(Queue& q, const MaskedArgs<T>& a) {
    return a.factored ? sygvd_factored_buffer_size<B, T>(q, a.A, a.B, a.w, a.itype, a.jobz, a.uplo)
                      : sygvd_buffer_size<B, T>(q, a.A, a.B, a.w, a.itype, a.jobz, a.uplo);
}

template <Backend B, class T>
Event masked_solve(Queue& q, const MaskedArgs<T>& a, Span<std::byte> ws, Span<int32_t> info) {
    return a.factored ? sygvd_factored<B, T>(q, a.A, a.B, a.w, a.itype, a.jobz, a.uplo, ws, info)
                      : sygvd<B, T>(q, a.A, a.B, a.w, a.itype, a.jobz, a.uplo, ws, info);
}

template <class T>
struct MaskedPacked {
    Span<std::int32_t> idx;
    MaskedArgs<T> p;
    Span<int32_t> info;
    Span<std::byte> child;
};

// One layout serves the size query and the run.
template <Backend B, class T>
MaskedPacked<T> masked_layout(Queue& q, BumpAllocator& pool, const MaskedArgs<T>& a, int k) {
    MaskedPacked<T> l;
    l.idx = pool.allocate<std::int32_t>(q, static_cast<std::size_t>(k));
    l.p = a;
    l.p.A = batch_mask::packed_like<T>(q, pool, a.A, k);
    l.p.B = batch_mask::packed_like<T>(q, pool, a.B, k);
    l.p.w = pool.allocate<Real<T>>(q, static_cast<std::size_t>(a.A.rows()) * static_cast<std::size_t>(k));
    l.info = pool.allocate<int32_t>(q, static_cast<std::size_t>(k));
    l.child = pool.allocate<std::byte>(q, solve_bytes<B, T>(q, l.p));
    return l;
}

template <Backend B, class T>
std::size_t packed_bytes(Queue& q, const MaskedArgs<T>& a, int k) {
    return workspace_bytes([&](BumpAllocator& p) { return masked_layout<B, T>(q, p, a, k); });
}

template <class T>
void masked_validate(const char* who, const MaskedArgs<T>& a) {
    sygst_validate_params<T>(who, a.A, a.B, a.itype, a.uplo);
    const auto need = static_cast<std::size_t>(a.A.rows()) * static_cast<std::size_t>(a.A.batch_size());
    if (a.w.size() < need)
        throw batchlas::invalid_argument(std::string(who) + ": eigenvalues holds " + std::to_string(a.w.size()) +
                                         " values, the call needs n * batch = " + std::to_string(need));
}

template <Backend B, class T>
Event run_masked(Queue& ctx, const MaskedArgs<T>& a, Span<std::byte> ws, Span<int32_t> info, Span<const uint8_t> mask,
                 const char* who) {
    masked_validate<T>(who, a);
    if (a.A.batch_size() == 0 || a.A.rows() == 0) return ctx.create_event_after_external_work();
    const auto idx = batch_mask::active_items(who, mask, a.A.batch_size());
    if (idx.empty()) return ctx.create_event_after_external_work();
    if (idx.size() == static_cast<std::size_t>(a.A.batch_size())) return masked_solve<B, T>(ctx, a, ws, info);
    const int k = static_cast<int>(idx.size()), n = a.A.rows();
    if (ws.size() < packed_bytes<B, T>(ctx, a, k))
        throw batchlas::workspace_error(std::string(who) + ": workspace is shorter than " + who + "_buffer_size()");
    return select::on_in_order_queue(ctx, [&](Queue& q) {
        BumpAllocator pool(ws);
        const auto l = masked_layout<B, T>(q, pool, a, k);
        batch_mask::upload(q, l.idx, idx);
        const Span<const std::int32_t> ix(l.idx.data(), l.idx.size());
        (void)batch_mask::gather<T>(q, a.A, l.p.A, ix);
        (void)batch_mask::gather<T>(q, a.B, l.p.B, ix);
        const bool want_info = info.size() >= static_cast<std::size_t>(a.A.batch_size());
        (void)masked_solve<B, T>(q, l.p, l.child, want_info ? l.info : Span<int32_t>{});
        (void)batch_mask::scatter<T>(q, l.p.A, a.A, ix);
        if (!a.factored) (void)batch_mask::scatter<T>(q, l.p.B, a.B, ix);
        if (want_info) (void)batch_mask::move_vectors<int32_t>(q, info.data(), l.info.data(), 1, ix, false);
        return batch_mask::move_vectors<Real<T>>(q, a.w.data(), l.p.w.data(), n, ix, false);
    });
}

template <Backend B, class T>
std::size_t masked_buffer_size(Queue& ctx, const MaskedArgs<T>& a, Span<const uint8_t> mask, const char* who) {
    masked_validate<T>(who, a);
    if (a.A.batch_size() == 0 || a.A.rows() == 0) return 0;
    (void)batch_mask::active_items(who, mask, a.A.batch_size());
    return std::max(solve_bytes<B, T>(ctx, a), batch_mask::largest_over_counts(a.A.batch_size() - 1, [&](int k) {
                        return packed_bytes<B, T>(ctx, a, k);
                    }));
}

}  // namespace

template <Backend Back, typename T>
Event sygvd_masked(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                   Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz, Uplo uplo,
                   Span<std::byte> work_space, Span<int32_t> info, Span<const uint8_t> mask) {
    return run_masked<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, false}, work_space, info, mask,
                               "sygvd_masked");
}

template <Backend Back, typename T>
size_t sygvd_masked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B, Span<typename base_type<T>::type> eigenvalues,
                                int itype, JobType jobz, Uplo uplo, Span<const uint8_t> mask) {
    return masked_buffer_size<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, false}, mask,
                                       "sygvd_masked_buffer_size");
}

template <Backend Back, typename T>
Event sygvd_factored_masked(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B, Span<typename base_type<T>::type> eigenvalues,
                            int itype, JobType jobz, Uplo uplo, Span<std::byte> work_space, Span<int32_t> info,
                            Span<const uint8_t> mask) {
    return run_masked<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, true}, work_space, info, mask,
                               "sygvd_factored_masked");
}

template <Backend Back, typename T>
size_t sygvd_factored_masked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                         const MatrixView<T, MatrixFormat::Dense>& B,
                                         Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz, Uplo uplo,
                                         Span<const uint8_t> mask) {
    return masked_buffer_size<Back, T>(ctx, {A, B, eigenvalues, itype, jobz, uplo, true}, mask,
                                       "sygvd_factored_masked_buffer_size");
}

#define SYGVD_MASKED_INSTANTIATE(B_, fp)                                                                \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd_masked) BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd_masked_buffer_size) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd_factored_masked)                                              \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sygvd_factored_masked_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYGVD_MASKED_INSTANTIATE)
#undef SYGVD_MASKED_INSTANTIATE

}  // namespace batchlas

// SPDX-License-Identifier: MIT
// syev_masked: the public syev on a packed copy of the active items, scattered back. A TU of its own so
// syev.cc (which the syev and sygvd tuner specs hash) is untouched. evidence: docs/design/masked-batches.md#masked-batches-design

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/util/mempool.hh>

#include <src/kernels/common/batch_mask.hh>
#include <src/select/select.hh>
#include <src/util/template-instantiations.hh>

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
struct MaskedPacked {
    Span<std::int32_t> idx;
    MV<T> A;
    Span<Real<T>> w;
    Span<int32_t> info;
    Span<std::byte> child;
};

// One layout serves the size query and the run.
template <Backend B, class T>
MaskedPacked<T> masked_layout(Queue& q, BumpAllocator& pool, const MV<T>& A, int k, JobType jobz, Uplo uplo) {
    MaskedPacked<T> l;
    l.idx = pool.allocate<std::int32_t>(q, static_cast<std::size_t>(k));
    l.A = batch_mask::packed_like<T>(q, pool, A, k);
    l.w = pool.allocate<Real<T>>(q, static_cast<std::size_t>(A.rows()) * static_cast<std::size_t>(k));
    l.info = pool.allocate<int32_t>(q, static_cast<std::size_t>(k));
    l.child = pool.allocate<std::byte>(q, syev_buffer_size<B, T>(q, l.A, l.w, jobz, uplo));
    return l;
}

template <Backend B, class T>
std::size_t packed_bytes(Queue& q, const MV<T>& A, int k, JobType jobz, Uplo uplo) {
    return workspace_bytes([&](BumpAllocator& p) { return masked_layout<B, T>(q, p, A, k, jobz, uplo); });
}

template <class T>
void masked_validate(const char* who, const MV<T>& A, std::size_t nw) {
    if (A.rows() != A.cols())
        throw batchlas::invalid_argument(std::string(who) + ": A must be square, got " + std::to_string(A.rows()) +
                                         " x " + std::to_string(A.cols()));
    const auto need = static_cast<std::size_t>(A.rows()) * static_cast<std::size_t>(A.batch_size());
    if (nw < need)
        throw batchlas::invalid_argument(std::string(who) + ": eigenvalues holds " + std::to_string(nw) +
                                         " values, the call needs n * batch = " + std::to_string(need));
}

}  // namespace

template <Backend Back, typename T>
Event syev_masked(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<typename base_type<T>::type> w,
                  JobType jobtype, Uplo uplo, Span<std::byte> workspace, Span<int32_t> info, Span<const uint8_t> mask) {
    masked_validate<T>("syev_masked", A, w.size());
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    const auto idx = batch_mask::active_items("syev_masked", mask, A.batch_size());
    if (idx.empty()) return ctx.create_event_after_external_work();
    if (idx.size() == static_cast<std::size_t>(A.batch_size()))
        return syev<Back, T>(ctx, A, w, jobtype, uplo, workspace, info);
    const int k = static_cast<int>(idx.size()), n = A.rows();
    if (workspace.size() < packed_bytes<Back, T>(ctx, A, k, jobtype, uplo))
        throw batchlas::workspace_error("syev_masked: workspace is shorter than syev_masked_buffer_size()");
    return select::on_in_order_queue(ctx, [&](Queue& q) {
        BumpAllocator pool(workspace);
        const auto l = masked_layout<Back, T>(q, pool, A, k, jobtype, uplo);
        batch_mask::upload(q, l.idx, idx);
        const Span<const std::int32_t> ix(l.idx.data(), l.idx.size());
        (void)batch_mask::gather<T>(q, A, l.A, ix);
        const bool want_info = info.size() >= static_cast<std::size_t>(A.batch_size());
        (void)syev<Back, T>(q, l.A, l.w, jobtype, uplo, l.child, want_info ? l.info : Span<int32_t>{});
        (void)batch_mask::scatter<T>(q, l.A, A, ix);
        if (want_info) (void)batch_mask::move_vectors<int32_t>(q, info.data(), l.info.data(), 1, ix, false);
        return batch_mask::move_vectors<Real<T>>(q, w.data(), l.w.data(), n, ix, false);
    });
}

template <Backend Back, typename T>
size_t syev_masked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                               Span<typename base_type<T>::type> w, JobType jobtype, Uplo uplo, Span<const uint8_t> mask) {
    masked_validate<T>("syev_masked_buffer_size", A, w.size());
    if (A.batch_size() == 0) return 0;
    (void)batch_mask::active_items("syev_masked_buffer_size", mask, A.batch_size());
    return std::max(syev_buffer_size<Back, T>(ctx, A, w, jobtype, uplo),
                    batch_mask::largest_over_counts(A.batch_size() - 1,
                                                    [&](int k) { return packed_bytes<Back, T>(ctx, A, k, jobtype, uplo); }));
}

#define SYEV_MASKED_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syev_masked) BATCHLAS_INSTANTIATE_OP(B_, fp, syev_masked_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_MASKED_INSTANTIATE)
#undef SYEV_MASKED_INSTANTIATE

}  // namespace batchlas

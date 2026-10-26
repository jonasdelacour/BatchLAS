#pragma once  // Masked batches by compaction. evidence: docs/design/masked-batches.md#masked-batches-design

#include <src/util/internal-api.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace batchlas::batch_mask {

/// Moves `rows x cols` items between two strided batches: item k of @p packed <-> item idx[k] of @p full.
/// `to_packed` gathers, otherwise it scatters.
template <typename T>
BATCHLAS_INTERNAL_API Event move_items(Queue& ctx, T* full, std::int64_t ld, std::int64_t stride, T* packed,
                                       std::int64_t pld, std::int64_t pstride, int rows, int cols,
                                       Span<const std::int32_t> idx, bool to_packed);

/// Copies the host index list into @p dst (device-accessible, idx.size() entries) and waits for it.
BATCHLAS_INTERNAL_API void upload(Queue& q, Span<std::int32_t> dst, const std::vector<std::int32_t>& idx);

/// The active items of a host-readable mask (nonzero = active), in order.
/// @throws batchlas::invalid_argument if the mask holds fewer than @p batch entries
inline std::vector<std::int32_t> active_items(const char* who, Span<const std::uint8_t> mask, int batch) {
    if (mask.size() < static_cast<std::size_t>(batch))
        throw batchlas::invalid_argument(std::string(who) + ": mask holds " + std::to_string(mask.size()) +
                                         " entries, the batch has " + std::to_string(batch));
    std::vector<std::int32_t> idx;
    for (int b = 0; b < batch; ++b)
        if (mask[static_cast<std::size_t>(b)] != 0) idx.push_back(b);
    return idx;
}

/// The largest @p size(k) over k = 1 .. @p kmax (0 when kmax < 1): a masked *_buffer_size bound that
/// holds for every mask, whichever family the packed call of k items selects.
template <typename F>
std::size_t largest_over_counts(int kmax, F&& size) {
    std::size_t s = 0;
    for (int k = 1; k <= kmax; ++k) s = std::max(s, static_cast<std::size_t>(size(k)));
    return s;
}

/// A workspace batch of @p k items laid out as @p like (same rows, cols, ld and stride), so a family that
/// branches on the layout (a vendor's packed-batch path) takes the branch it would take on @p like.
template <typename T>
MatrixView<T, MatrixFormat::Dense> packed_like(Queue& q, BumpAllocator& pool, const MatrixView<T, MatrixFormat::Dense>& like,
                                               int k) {
    const Span<T> buf = pool.allocate<T>(q, static_cast<std::size_t>(like.stride()) * static_cast<std::size_t>(k));
    const Span<T*> ptr = pool.allocate<T*>(q, static_cast<std::size_t>(k));
    return MatrixView<T, MatrixFormat::Dense>(buf.data(), like.rows(), like.cols(), like.ld(), like.stride(), k,
                                              ptr.data());
}

template <typename T>
Event gather(Queue& q, const MatrixView<T, MatrixFormat::Dense>& full, const MatrixView<T, MatrixFormat::Dense>& packed,
             Span<const std::int32_t> idx) {
    return move_items<T>(q, full.data_ptr(), full.ld(), full.stride(), packed.data_ptr(), packed.ld(), packed.stride(),
                         full.rows(), full.cols(), idx, true);
}

template <typename T>
Event scatter(Queue& q, const MatrixView<T, MatrixFormat::Dense>& packed, const MatrixView<T, MatrixFormat::Dense>& full,
              Span<const std::int32_t> idx) {
    return move_items<T>(q, full.data_ptr(), full.ld(), full.stride(), packed.data_ptr(), packed.ld(), packed.stride(),
                         full.rows(), full.cols(), idx, false);
}

/// Per-item vectors of @p len entries packed back to back (eigenvalues, info).
template <typename T>
Event move_vectors(Queue& q, T* full, T* packed, int len, Span<const std::int32_t> idx, bool to_packed) {
    return move_items<T>(q, full, len, len, packed, len, len, len, 1, idx, to_packed);
}

}  // namespace batchlas::batch_mask

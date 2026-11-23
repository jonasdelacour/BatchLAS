// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Level 3 stagers of the tuned gemm: `op(X)` blocks into local memory, zero outside the
/// matrix, and the element read they share (@ref api_device_lowlevel, not covered by the
/// compatibility promise).
///
/// `l3::stage_block_read`, `l3::stage_block`, `l3::stage_block_packet`,
/// `l3::stage_block_packet_aligned`, `l3::stage_storage`, and `l3::OperandAccessor`.
/// @ingroup api_device_lowlevel
// The tuned gemm code moved behind a signature: same loops, pragmas, predicate order and
// helpers, so a kernel built from them keeps its SASS.

#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/level3/packet.hh>

namespace batchlas::device::l3 {

/// @brief Element `(row, col)` of `op(X)` for a column-major `X`, conjugated only for a complex `ConjTrans`.
/// @ingroup api_device_lowlevel
/// @param ptr the first element of the batch.
/// @param ld the leading dimension.
/// @param batch_offset the item's offset from `ptr`.
/// @param row the row of `op(X)`.
/// @param col the column of `op(X)`.
/// @param trans `op`, at run time.
/// @return the element.
template <class Scalar>
inline Scalar operand_value(const Scalar* ptr,
                            int ld,
                            int batch_offset,
                            int row,
                            int col,
                            Transpose trans) {
    const bool transpose = trans != Transpose::NoTrans;
    const bool conjugate = trans == Transpose::ConjTrans;
    const int source_row = transpose ? col : row;
    const int source_col = transpose ? row : col;
    const Scalar& value = ptr[batch_offset + source_col * ld + source_row];
    return conjugate ? device::conj(value) : value;
}

/// @brief The element read of `op(X)` with `op` fixed at compile time.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the element type.
/// @tparam Op `op`.
template <class Scalar, Transpose Op>
struct OperandAccessor {
    /// @brief `operand_value(ptr, ld, batch_offset, row, col, Op)`.
    /// @param ptr the first element of the batch.
    /// @param ld the leading dimension.
    /// @param batch_offset the item's offset from `ptr`.
    /// @param row the row of `op(X)`.
    /// @param col the column of `op(X)`.
    static Scalar load(const Scalar* ptr, int ld, int batch_offset, int row, int col) {
        return operand_value(ptr, ld, batch_offset, row, col, Op);
    }
};

// The bounds are references for the same reason as mma_slab's k (the reg tiles' REG and LDG/STS).
/// @brief The staging loop: `dst(r, c) := read(r0 + r, c0 + c)` for the `R` x `C` block, zero
/// where the row is at or past `mr` or the column at or past `mc`.
///
/// The operand adapters (a plain `op(X)`, a mirrored or a triangular operand) are the `read`.
/// Items `lid, lid + Threads, ...` share the block.
/// @ingroup api_device_lowlevel
/// @tparam RowFastest `true`: the items walk `r` fastest (else `c`). Name the operand's
///   contiguous index, so a warp reads consecutive addresses whatever `dst`'s slot map is.
/// @tparam R block rows.
/// @tparam C block columns.
/// @tparam Threads the number of items sharing the block.
/// @tparam T the element type (for the zero).
/// @param lid this item's index.
/// @param r0 the block's first row.
/// @param c0 the block's first column.
/// @param mr the matrix's row count.
/// @param mc the matrix's column count.
/// @param read called as `read(int row, int col)`.
/// @param dst called as `dst(int r, int c)`, assignable.
template <bool RowFastest, int R, int C, int Threads, class T, class Read, class Dst>
BATCHLAS_DEVICE_INLINE void stage_block_read(int lid, int r0, int c0, const int& mr, const int& mc, Read&& read, Dst&& dst) {
    for (int index = lid; index < R * C; index += Threads) {
        const int r = RowFastest ? index % R : index / C;
        const int c = RowFastest ? index / R : index % C;
        const int row = r0 + r;
        const int col = c0 + c;
        dst(r, c) = (row < mr && col < mc) ? read(row, col) : T(0);
    }
}

// x and ld are references: the closure fields of the reg tiles' kernel lambda, read where the
// tuned loop read them.
/// @brief `stage_block_read` of `op(X)(row, col)` for a column-major `X`.
/// @ingroup api_device_lowlevel
/// @tparam Op `op`; the items walk `X`'s contiguous index.
/// @tparam R block rows.
/// @tparam C block columns.
/// @tparam Threads the number of items sharing the block.
/// @param lid this item's index.
/// @param x `X`'s first element.
/// @param ld `X`'s leading dimension.
/// @param r0 the block's first row.
/// @param c0 the block's first column.
/// @param mr `op(X)`'s row count.
/// @param mc `op(X)`'s column count.
/// @param dst called as `dst(int r, int c)`, assignable.
template <Transpose Op, int R, int C, int Threads, class Scalar, class Dst>
BATCHLAS_DEVICE_INLINE void stage_block(int lid, const Scalar* const& x, const int& ld, int r0, int c0, const int& mr, const int& mc,
                                     Dst&& dst) {
    stage_block_read<Op == Transpose::NoTrans, R, C, Threads, Scalar>(
        lid, r0, c0, mr, mc, [&](int row, int col) BATCHLAS_DEVICE_BODY { return OperandAccessor<Scalar, Op>::load(x, ld, 0, row, col); },
        dst);
}

/// @brief `stage_block` with `V`-element aligned packet loads along `X`'s contiguous index.
///
/// A packet that crosses the matrix edge falls back to predicated scalar reads.
/// @ingroup api_device_lowlevel
/// @pre every in-range packet's address is aligned to the packet (the caller's `ld` and base test,
///   `supports_aligned_packet_loads`).
/// @tparam Op `op`.
/// @tparam R block rows.
/// @tparam C block columns.
/// @tparam V the packet width.
/// @tparam Threads the number of items sharing the block.
/// @param lid this item's index.
/// @param x `X`'s first element.
/// @param ld `X`'s leading dimension.
/// @param r0 the block's first row.
/// @param c0 the block's first column.
/// @param mr `op(X)`'s row count.
/// @param mc `op(X)`'s column count.
/// @param dst called as `dst(int r, int c)`, assignable.
template <Transpose Op, int R, int C, int V, int Threads, class Real, class Dst>
BATCHLAS_DEVICE_INLINE void stage_block_packet(int lid, const Real* const& x, const int& ld, int r0, int c0,
                                               const int& mr,
                                            const int& mc, Dst&& dst) {
    if constexpr (Op == Transpose::NoTrans) {
        constexpr int PacketsPerCol = R / V;
        constexpr int PacketCount = PacketsPerCol * C;
#pragma unroll 4
        for (int packet = lid; packet < PacketCount; packet += Threads) {
            const int r = (packet % PacketsPerCol) * V;
            const int c = packet / PacketsPerCol;
            const int row = r0 + r;
            const int col = c0 + c;
            if (row + (V - 1) < mr && col < mc) {
                const auto packet_values = packet_load_aligned<Real, V>(x, col * ld + row);
                for (int lane = 0; lane < V; ++lane) dst(r + lane, c) = packet_values[lane];
            } else {
                for (int lane = 0; lane < V; ++lane) {
                    const int rr = row + lane;
                    dst(r + lane, c) = (rr < mr && col < mc) ? OperandAccessor<Real, Op>::load(x, ld, 0, rr,
                                                                                               col) : Real(0);
                }
            }
        }
    } else {
        constexpr int PacketsPerRow = C / V;
        constexpr int PacketCount = R * PacketsPerRow;
#pragma unroll 4
        for (int packet = lid; packet < PacketCount; packet += Threads) {
            const int r = packet / PacketsPerRow;
            const int c = (packet % PacketsPerRow) * V;
            const int row = r0 + r;
            const int col = c0 + c;
            if (row < mr && col + (V - 1) < mc) {
                const auto packet_values = packet_load_aligned<Real, V>(x, row * ld + col);
                for (int lane = 0; lane < V; ++lane) dst(r, c + lane) = packet_values[lane];
            } else {
                for (int lane = 0; lane < V; ++lane) {
                    const int cc = col + lane;
                    dst(r, c + lane) = (row < mr && cc < mc) ? OperandAccessor<Real, Op>::load(x, ld, 0, row,
                                                                                               cc) : Real(0);
                }
            }
        }
    }
}

/// @brief The unpredicated `NoTrans` packet stager: one statically unrolled pass per `Threads` packets.
/// @ingroup api_device_lowlevel
/// @pre the block lies inside `X` and every packet is aligned.
/// @tparam R block rows, a multiple of `V`.
/// @tparam C block columns.
/// @tparam V the packet width.
/// @tparam Threads the number of items sharing the block.
/// @param lid this item's index.
/// @param x `X`'s first element.
/// @param ld `X`'s leading dimension.
/// @param r0 the block's first row.
/// @param c0 the block's first column.
/// @param dst called as `dst(int r, int c)`, assignable.
template <int R, int C, int V, int Threads, class Real, class Dst>
BATCHLAS_DEVICE_INLINE void stage_block_packet_aligned(int lid, const Real* const& x, const int& ld, int r0, int c0, Dst&& dst) {
    constexpr int PacketsPerCol = R / V;
    constexpr int PacketCount = PacketsPerCol * C;
    constexpr int Iterations = (PacketCount + Threads - 1) / Threads;
    static_for<Iterations>([&](auto iter) {
        constexpr int iteration = iter;
        const int packet = lid + iteration * Threads;
        if (packet < PacketCount) {
            const int r = (packet % PacketsPerCol) * V;
            const int c = packet / PacketsPerCol;
            const auto packet_values = packet_load_aligned<Real, V>(x, (c0 + c) * ld + (r0 + r));
            for (int lane = 0; lane < V; ++lane) dst(r + lane, c) = packet_values[lane];
        }
    });
}

/// @brief `dst(i, j) := op(X)(i, j)` over the `s0n` x `s1n` storage block of `X`, walked in storage order.
///
/// Items `start, start + Step, ...` walk `s0` fastest, so the reads coalesce either way;
/// `(i, j) = (s1, s0)` when `trans`. Zero where `!live` or outside `[0, mi) x [0, mj)`. `trans` and
/// `conj` are run-time values, so one kernel serves every form; a constant argument folds.
/// @ingroup api_device_lowlevel
/// @tparam Step the stride between items.
/// @param start this item's first storage index.
/// @param s0n the storage block's fast extent.
/// @param s1n the storage block's slow extent.
/// @param live `false` writes zeros.
/// @param x `X`'s first element.
/// @param ld `X`'s leading dimension.
/// @param trans transpose the block.
/// @param conj conjugate the elements.
/// @param mi `op(X)`'s row bound.
/// @param mj `op(X)`'s column bound.
/// @param dst called as `dst(int i, int j)`, assignable.
template <int Step, class S0, class S1, class Scalar, class Ld, class Dst>
BATCHLAS_DEVICE_INLINE void stage_storage(int start, S0 s0n, S1 s1n, bool live, const Scalar* x, Ld ld, bool trans, bool conj, int mi,
                                       int mj, Dst&& dst) {
    for (int e = start; e < int(s0n) * int(s1n); e += Step) {
        const int s0 = e % int(s0n), s1 = e / int(s0n);
        const int i = trans ? s1 : s0;
        const int j = trans ? s0 : s1;
        Scalar v{};
        if (live && i < mi && j < mj) v = x[s0 + s1 * ld];
        dst(i, j) = conj ? Arith<Scalar>::conj(v) : v;
    }
}

}  // namespace batchlas::device::l3

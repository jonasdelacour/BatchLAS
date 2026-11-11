// SPDX-License-Identifier: MIT
#pragma once
// Level 3 stagers (v22 design B.2): op(X) blocks into local memory, zero outside the matrix, and
// the element read they share (OperandAccessor). The tuned gemm code moved behind a signature:
// same loops, pragmas, predicate order and helpers, so a kernel built from them keeps its SASS.
//   stage_block_read, stage_block, stage_block_packet, stage_block_packet_aligned, stage_storage

#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/level3/packet.hh>

namespace batchlas::device::l3 {

// op(X)(row, col) of a column-major X: the conjugate (core conj: std::conj, by reference) only
// for a complex ConjTrans.
template <typename T>
inline T operand_value(const T* ptr,
                       int ld,
                       int batch_offset,
                       int row,
                       int col,
                       Transpose trans) {
    const bool transpose = trans != Transpose::NoTrans;
    const bool conjugate = trans == Transpose::ConjTrans;
    const int source_row = transpose ? col : row;
    const int source_col = transpose ? row : col;
    const T& value = ptr[batch_offset + source_col * ld + source_row];
    return conjugate ? device::conj(value) : value;
}

template <typename T, Transpose Op>
struct OperandAccessor {
    static T load(const T* ptr, int ld, int batch_offset, int row, int col) {
        return operand_value(ptr, ld, batch_offset, row, col, Op);
    }
};

// dst(r, c) := read(r0 + r, c0 + c) for the R x C block, zero where row >= mr or col >= mc:
// THE staging loop; the operand adapters (plain op(X), a mirrored or triangular operand) are
// the read. Items lid, lid + Threads, ... walk r fastest when RowFastest (c otherwise): name the
// operand's contiguous index so a warp reads consecutive addresses whatever dst's slot map is.
// The bounds are references for the same reason as mma_slab's k (the reg tiles' REG and LDG/STS).
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

// stage_block_read of op(X)(row, col) of a column-major X. x and ld are references (the
// closure fields of the reg tiles' kernel lambda, read where the tuned loop read them).
template <Transpose Op, int R, int C, int Threads, class T, class Dst>
BATCHLAS_DEVICE_INLINE void stage_block(int lid, const T* const& x, const int& ld, int r0, int c0, const int& mr, const int& mc,
                                     Dst&& dst) {
    stage_block_read<Op == Transpose::NoTrans, R, C, Threads, T>(
        lid, r0, c0, mr, mc, [&](int row, int col) BATCHLAS_DEVICE_BODY { return OperandAccessor<T, Op>::load(x, ld, 0, row, col); },
        dst);
}

// stage_block with V-element aligned packets along X's contiguous index (x + offset 16-byte
// aligned for every in-range packet: the caller's ld and base test); a packet that crosses the
// matrix edge falls back to predicated scalars.
template <Transpose Op, int R, int C, int V, int Threads, class T, class Dst>
BATCHLAS_DEVICE_INLINE void stage_block_packet(int lid, const T* const& x, const int& ld, int r0, int c0, const int& mr,
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
                const auto packet_values = packet_load_aligned<T, V>(x, col * ld + row);
                for (int lane = 0; lane < V; ++lane) dst(r + lane, c) = packet_values[lane];
            } else {
                for (int lane = 0; lane < V; ++lane) {
                    const int rr = row + lane;
                    dst(r + lane, c) = (rr < mr && col < mc) ? OperandAccessor<T, Op>::load(x, ld, 0, rr, col) : T(0);
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
                const auto packet_values = packet_load_aligned<T, V>(x, row * ld + col);
                for (int lane = 0; lane < V; ++lane) dst(r, c + lane) = packet_values[lane];
            } else {
                for (int lane = 0; lane < V; ++lane) {
                    const int cc = col + lane;
                    dst(r, c + lane) = (row < mr && cc < mc) ? OperandAccessor<T, Op>::load(x, ld, 0, row, cc) : T(0);
                }
            }
        }
    }
}

// The unpredicated NoTrans form: the block lies inside X and every packet is aligned. One
// statically unrolled pass per Threads packets (core static_for).
template <int R, int C, int V, int Threads, class T, class Dst>
BATCHLAS_DEVICE_INLINE void stage_block_packet_aligned(int lid, const T* const& x, const int& ld, int r0, int c0, Dst&& dst) {
    constexpr int PacketsPerCol = R / V;
    constexpr int PacketCount = PacketsPerCol * C;
    constexpr int Iterations = (PacketCount + Threads - 1) / Threads;
    static_for<Iterations>([&](auto iter) {
        constexpr int iteration = iter;
        const int packet = lid + iteration * Threads;
        if (packet < PacketCount) {
            const int r = (packet % PacketsPerCol) * V;
            const int c = packet / PacketsPerCol;
            const auto packet_values = packet_load_aligned<T, V>(x, (c0 + c) * ld + (r0 + r));
            for (int lane = 0; lane < V; ++lane) dst(r + lane, c) = packet_values[lane];
        }
    });
}

// dst(i, j) := op(X)(i, j) over the S0 x S1 STORAGE block of X, walked in storage order (s0
// fastest: coalesced either way) by items start, start + Step, ...; (i, j) = (s1, s0) when trans.
// Zero where !live or outside [0, mi) x [0, mj); conj conjugates. trans and
// conj are runtime values (one kernel serves every form); a constant folds.
template <int Step, class S0, class S1, class T, class Ld, class Dst>
BATCHLAS_DEVICE_INLINE void stage_storage(int start, S0 s0n, S1 s1n, bool live, const T* x, Ld ld, bool trans, bool conj, int mi,
                                       int mj, Dst&& dst) {
    for (int e = start; e < int(s0n) * int(s1n); e += Step) {
        const int s0 = e % int(s0n), s1 = e / int(s0n);
        const int i = trans ? s1 : s0;
        const int j = trans ? s0 : s1;
        T v{};
        if (live && i < mi && j < mj) v = x[s0 + s1 * ld];
        dst(i, j) = conj ? Arith<T>::conj(v) : v;
    }
}

}  // namespace batchlas::device::l3

// SPDX-License-Identifier: MIT
#pragma once
// The tuned gemm kernels' private components (src/ops/gemm/kernels): stagers, register
// micro-kernels, k loops and the staged C epilogue, so a kernel reads as stage -> k loop of
// rank-1 updates -> epilogue. Gemm-private by design (one implementation, not v2 core). Each op
// is the tuned code moved behind a signature: same loops, pragmas, predicate order, helpers
// (OperandAccessor, packet_load_aligned, LinearEpilogue, sycl_device) and arithmetic, so a kernel
// built from them keeps its SASS. Shared vocabulary comes from v2 core: Tile, Frag, fill, load,
// the arith policies (Ops tags; dla_v2_scalar.hh covers the POD Cx) and any executor with barrier().
//   epilogue_store_fused    C := alpha acc + beta C with a host beta_zero flag
//   outer_mac, mma_slab, dot_slab, gemv_t_frag   register micro-kernels
//   stage_block, stage_block_packet, stage_block_packet_aligned, stage_storage   stagers
//   kloop, kloop_prefetch2, kloop_async   k loops (the barriers are the pipeline's contract)
//   store_tile_staged       a register tile through local memory to C, coalesced at any ldc

#include <cstddef>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <src/ops/gemm/kernels/accessors.hh>
#include <src/ops/gemm/kernels/epilogue_linear.hh>
#include <src/ops/gemm/kernels/load_policies.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::sycl_gemm::components {

namespace v2 = batchlas::device;

// ---- epilogue ------------------------------------------------------------------------------
// *dst := alpha acc + beta *dst with the host's beta_zero flag (beta_zero never reads *dst). A
// complex takes Ops' product and accumulate (arith::Fma: sycl_device's dev_mul and fma_acc), a
// real scalar the operators.
template <class Ops, class T> BATCHLAS_DLA_INLINE void epilogue_store_fused(T* dst, T alpha, T beta, bool beta_zero, T acc) {
    using Ar = v2::arith_t<Ops, T>;
    if constexpr (Ar::is_complex) {
        T out = Ar::mul(alpha, acc);
        if (!beta_zero) Ar::mac(out, beta, *dst);
        *dst = out;
    } else {
        *dst = beta_zero ? alpha * acc : alpha * acc + beta * *dst;
    }
}

// ---- register micro-kernels ----------------------------------------------------------------
// acc += a b^T on a TR x TC register tile. Full: the full-unroll pragmas; otherwise none (the
// register-tiled kernel's own loops, unrolled as the compiler decides).
template <bool Full = true, class Ops = v2::arith::Native, class T, int TR, int TC>
BATCHLAS_DLA_INLINE void outer_mac(v2::Tile<T, TR, TC>& acc, const T (&a)[TR], const T (&b)[TC]) {
    using A = v2::arith_t<Ops, T>;
    if constexpr (Full) {
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TR; ++i) {
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TC; ++j) A::mac(acc(i, j), a[i], b[j]);
        }
    } else {
        for (int i = 0; i < TR; ++i) {
            for (int j = 0; j < TC; ++j) A::mac(acc(i, j), a[i], b[j]);
        }
    }
}

// One staged k slab: for each t < TileK (and kk0 + t < k unless Exact) a(i) = a_at(t, i),
// b(j) = b_at(t, j), acc += a b^T; UnrollK steps per outer iteration, no pragmas. k by
// reference, like the tuned lambda's closure read: by value, SimplifyCFG merged the two exits
// first and the TileK = 8 slab stayed rolled (32x32x8: 34 -> 6 FFMA).
template <int TileK, int UnrollK, bool Exact, class Ops = v2::arith::Native, class T, int TR, int TC, class FA, class FB>
BATCHLAS_DLA_INLINE void mma_slab(v2::Tile<T, TR, TC>& acc, int kk0, const int& k, FA&& a_at, FB&& b_at) {
    auto step = [&](int t) BATCHLAS_DLA_BODY {
        T a_frag[TR];
        T b_frag[TC];
        for (int i = 0; i < TR; ++i) a_frag[i] = a_at(t, i);
        for (int j = 0; j < TC; ++j) b_frag[j] = b_at(t, j);
        outer_mac<false, Ops>(acc, a_frag, b_frag);
    };
    if constexpr (Exact) {
        for (int t0 = 0; t0 < TileK; t0 += UnrollK) {
            for (int unroll = 0; unroll < UnrollK; ++unroll) step(t0 + unroll);
        }
    } else {
        for (int t0 = 0; t0 < TileK && kk0 + t0 < k; t0 += UnrollK) {
            for (int unroll = 0; unroll < UnrollK && t0 + unroll < TileK && kk0 + t0 + unroll < k; ++unroll)
                step(t0 + unroll);
        }
    }
}

// acc += x(t) y(t) for t < N with kk0 + t < k, in index order: one item's dot over a staged slab.
template <int N, class Ops = v2::arith::Native, class T, class VX, class VY>
BATCHLAS_DLA_INLINE void dot_slab(T& acc, const VX& x, const VY& y, int kk0, int k) {
    using A = v2::arith_t<Ops, T>;
    for (int t = 0; t < N && kk0 + t < k; ++t) A::mac(acc, x(t), y(t));
}

namespace detail {
template <int L0, int Lc, class Ar, class T, int NA, int NC, class FB>
BATCHLAS_DLA_INLINE void gemv_t_frag_chunk(v2::Frag<T, NC>& acc, const v2::Frag<T, NA>& a, FB& b_at, int k) {
BATCHLAS_UNROLL_FULL
    for (int l = L0; l < L0 + Lc; ++l) {
        if (l >= k) continue;  // uniform; `continue` keeps the unroll, so a[] stays in registers
BATCHLAS_UNROLL_FULL
        for (int j = 0; j < NC; ++j) Ar::mac(acc[j], a[l], b_at(l, j));
    }
}
template <class Fn, int... I> BATCHLAS_DLA_INLINE void static_for(Fn&& fn, std::integer_sequence<int, I...>) {
    (fn(std::integral_constant<int, I>{}), ...);
}
}  // namespace detail

// acc(j) += sum_{l < k} a(l) B(l, j), l outermost: gemv^T with x and y in registers. Chunk = NA
// is one unrolled loop; Chunk = NA / 4 is four sequential ones (one complex loop over 64 l passes
// LLVM's unroll threshold and a[] spills; a lambda per chunk put 160-184 B on the stack).
template <int Chunk, class Ops, class T, int NA, int NC, class FB>
BATCHLAS_DLA_INLINE void gemv_t_frag(v2::Frag<T, NC>& acc, const v2::Frag<T, NA>& a, FB&& b_at, int k) {
    using Ar = v2::arith_t<Ops, T>;
    static_assert(Chunk == NA || Chunk * 4 == NA, "gemv_t_frag: one loop, or four");
    detail::gemv_t_frag_chunk<0, Chunk, Ar>(acc, a, b_at, k);
    if constexpr (NA / Chunk > 1) {
        detail::gemv_t_frag_chunk<Chunk, Chunk, Ar>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<2 * Chunk, Chunk, Ar>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<3 * Chunk, Chunk, Ar>(acc, a, b_at, k);
    }
}

// ---- stagers: op(X) blocks into local memory, zero outside the matrix ------------------------
// dst(r, c) := op(X)(r0 + r, c0 + c) for the R x C block, zero where row >= mr or col >= mc.
// Items lid, lid + Threads, ... walk X's contiguous index fastest (r for NoTrans, c otherwise),
// so a warp reads consecutive addresses whatever dst's slot map is. x, ld and the bounds are
// references for the same reason as mma_slab's k (the reg tiles' REG and LDG/STS).
template <Transpose Op, int R, int C, int Threads, class T, class Dst>
BATCHLAS_DLA_INLINE void stage_block(int lid, const T* const& x, const int& ld, int r0, int c0, const int& mr, const int& mc,
                                     Dst&& dst) {
    for (int index = lid; index < R * C; index += Threads) {
        const int r = Op == Transpose::NoTrans ? index % R : index / C;
        const int c = Op == Transpose::NoTrans ? index / R : index % C;
        const int row = r0 + r;
        const int col = c0 + c;
        dst(r, c) = (row < mr && col < mc) ? OperandAccessor<T, Op>::load(x, ld, 0, row, col) : T(0);
    }
}

// stage_block with V-element aligned packets along X's contiguous index (x + offset 16-byte
// aligned for every in-range packet: the caller's ld and base test); a packet that crosses the
// matrix edge falls back to predicated scalars.
template <Transpose Op, int R, int C, int V, int Threads, class T, class Dst>
BATCHLAS_DLA_INLINE void stage_block_packet(int lid, const T* const& x, const int& ld, int r0, int c0, const int& mr,
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
// statically unrolled pass per Threads packets.
template <int R, int C, int V, int Threads, class T, class Dst>
BATCHLAS_DLA_INLINE void stage_block_packet_aligned(int lid, const T* const& x, const int& ld, int r0, int c0, Dst&& dst) {
    constexpr int PacketsPerCol = R / V;
    constexpr int PacketCount = PacketsPerCol * C;
    constexpr int Iterations = (PacketCount + Threads - 1) / Threads;
    detail::static_for(
        [&](auto iter) {
            constexpr int iteration = iter;
            const int packet = lid + iteration * Threads;
            if (packet < PacketCount) {
                const int r = (packet % PacketsPerCol) * V;
                const int c = packet / PacketsPerCol;
                const auto packet_values = packet_load_aligned<T, V>(x, (c0 + c) * ld + (r0 + r));
                for (int lane = 0; lane < V; ++lane) dst(r + lane, c) = packet_values[lane];
            }
        },
        std::make_integer_sequence<int, Iterations>{});
}

// dst(i, j) := op(X)(i, j) over the S0 x S1 STORAGE block of X, walked in storage order (s0
// fastest: coalesced either way) by items start, start + Step, ...; (i, j) = (s1, s0) when trans.
// Zero where !live or outside [0, mi) x [0, mj); conj conjugates. trans and
// conj are runtime values (one kernel serves every form); a constant folds.
template <int Step, class S0, class S1, class T, class Ld, class Dst>
BATCHLAS_DLA_INLINE void stage_storage(int start, S0 s0n, S1 s1n, bool live, const T* x, Ld ld, bool trans, bool conj, int mi,
                                       int mj, Dst&& dst) {
    for (int e = start; e < int(s0n) * int(s1n); e += Step) {
        const int s0 = e % int(s0n), s1 = e / int(s0n);
        const int i = trans ? s1 : s0;
        const int j = trans ? s0 : s1;
        T v{};
        if (live && i < mi && j < mj) v = x[s0 + s1 * ld];
        dst(i, j) = conj ? v2::Arith<T>::conj(v) : v;
    }
}

// ---- k loops -------------------------------------------------------------------------------
// load(kk0, stage) fills local stage `stage` with k slab kk0; compute(stage, kk0) consumes it.
// Stages 1: load, barrier, compute, barrier per slab. Stages 2: slab t + 1 loads while slab t
// computes, one barrier per slab. Exact: no tail, and the first load sits behind tiles > 0.
template <int Stages, bool Exact, int TileK, class Ex, class Load, class Compute>
BATCHLAS_DLA_INLINE void kloop(const Ex& ex, int tile_count, Load&& load, Compute&& compute) {
    static_assert(Stages == 1 || Stages == 2, "kloop: one or two local stages");
    if constexpr (Stages == 1) {
        for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
            const int kk0 = tile_idx * TileK;
            load(kk0, 0);
            ex.barrier();
            compute(0, kk0);
            ex.barrier();
        }
    } else if constexpr (Exact) {
        if (tile_count > 0) {
            load(0, 0);
            ex.barrier();
            for (int tile_idx = 0; tile_idx + 1 < tile_count; ++tile_idx) {
                const int current_stage = tile_idx & 1;
                const int next_stage = current_stage ^ 1;
                load((tile_idx + 1) * TileK, next_stage);
                compute(current_stage, tile_idx * TileK);
                ex.barrier();
            }
            const int final_stage = (tile_count - 1) & 1;
            compute(final_stage, (tile_count - 1) * TileK);
            ex.barrier();
        }
    } else {
        load(0, 0);
        ex.barrier();
        for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
            const int kk0 = tile_idx * TileK;
            const int current_stage = tile_idx & 1;
            const int next_tile_idx = tile_idx + 1;
            if (next_tile_idx < tile_count) load(next_tile_idx * TileK, current_stage ^ 1);
            compute(current_stage, kk0);
            ex.barrier();
        }
    }
}

// Register-prefetched double buffer: gload(k0) reads slab k0 into registers, sstore(buf) writes
// them to local buffer buf, compute(buf) consumes one. One barrier per slab: slab s + 1 goes into
// the buffer slab s - 1 was read from, and that read ended before the barrier closing iteration
// s - 1. Unrolled by two so buf is a constant. Ends without a barrier.
template <int TileK, class Ex, class GLoad, class SStore, class Compute>
BATCHLAS_DLA_INLINE void kloop_prefetch2(const Ex& ex, int slabs, GLoad&& gload, SStore&& sstore, Compute&& compute) {
    gload(0);
    sstore(0);
    ex.barrier();
    int s = 0;
    for (; s + 1 < slabs; s += 2) {
        gload((s + 1) * TileK);
        compute(0);
        sstore(1);
        ex.barrier();
        const bool more = s + 2 < slabs;
        if (more) gload((s + 2) * TileK);
        compute(1);
        if (more) sstore(0);
        ex.barrier();
    }
    if (s < slabs) compute(0);  // an odd slab count leaves the last slab in buffer 0
}

// Stages - 1 slabs in flight through asynchronous copies: issue(k0, buf) starts slab k0's copies
// into buffer buf (an integral_constant), commit() closes a copy group, wait() waits until at most
// Stages - 2 groups are pending, drain() until none. Step s waits for slab s, barriers, refills the
// buffer slab s - 1 was read from and computes; one commit per step (empty past the end).
template <int Stages, int TileK, class Ex, class Issue, class Commit, class Wait, class Drain, class Compute>
BATCHLAS_DLA_INLINE void kloop_async(const Ex& ex, int slabs, Issue&& issue, Commit&& commit, Wait&& wait, Drain&& drain,
                                     Compute&& compute) {
    [&]<int... J>(std::integer_sequence<int, J...>) BATCHLAS_DLA_BODY {
        ((J < slabs ? issue(J * TileK, std::integral_constant<int, J>{}) : void(), commit()), ...);
    }(std::make_integer_sequence<int, Stages - 1>{});
    auto step = [&](int s, auto buf) BATCHLAS_DLA_BODY {
        constexpr int b = decltype(buf)::value;
        wait();
        ex.barrier();
        if (s + Stages - 1 < slabs) issue((s + Stages - 1) * TileK, std::integral_constant<int, (b + Stages - 1) % Stages>{});
        commit();
        compute(b);
    };
    for (int s0 = 0; s0 < slabs; s0 += Stages) {
        [&]<int... J>(std::integer_sequence<int, J...>) BATCHLAS_DLA_BODY {
            ((s0 + J < slabs && (step(s0 + J, std::integral_constant<int, J>{}), true)) && ...);
        }(std::make_integer_sequence<int, Stages>{});
    }
    drain();
}

// ---- C epilogue through local memory -------------------------------------------------------
// C(m0 + r, n0 + c) := alpha acc + beta C for a TileM x TileN tile whose item (local_row,
// local_col) holds rows local_row * TR + i, cols local_col * TC + j. Columns pass through local
// memory (lo: LoElems, then hi: HiElems; odd stride TileM + 1), then lane = row stores run down
// C's columns (coalesced at any ldc), Batch priors in flight per item. Precondition: the caller's
// last read of lo and hi ended before a barrier. beta == 0 never reads C.
template <int TileM, int TileN, int Threads, int LoElems, int HiElems, int Batch = 2, class Ex, class T, int TR, int TC>
BATCHLAS_DLA_INLINE void store_tile_staged(const Ex& ex, int lid, int local_row, int local_col, const v2::Tile<T, TR, TC>& acc,
                                           T* lo, T* hi, T* c_item, int ldc, int m0, int n0, int m, int n, T alpha, T beta) {
    constexpr int CS = TileM + 1;
    constexpr int LoCols = LoElems / CS;
    constexpr int HiCols = HiElems / CS;
    constexpr int CW = LoCols + HiCols;
    constexpr int Passes = (TileN + CW - 1) / CW;
    static_assert(CW >= 1, "store_tile_staged: the stage must hold one C column");
    auto slot = [&](int c, int r) -> T* { return c < LoCols ? lo + c * CS + r : hi + (c - LoCols) * CS + r; };
    const bool beta_zero = beta == T(0);
#pragma unroll 1
    for (int pass = 0; pass < Passes; ++pass) {
        const int c0 = pass * CW;
        const int cw = TileN - c0 < CW ? TileN - c0 : CW;
#pragma unroll
        for (int j = 0; j < TC; ++j) {
            const int c = local_col * TC + j - c0;
            if (c < 0 || c >= cw) continue;
#pragma unroll
            for (int i = 0; i < TR; ++i) *slot(c, local_row * TR + i) = acc(i, j);
        }
        ex.barrier();
        const int count = TileM * cw;
        auto at = [&](int e, bool& ok) {
            const int row = m0 + e % TileM, col = n0 + c0 + e / TileM;
            ok = e < count && row < m && col < n;
            return row + static_cast<std::ptrdiff_t>(col) * ldc;
        };
#pragma unroll 1
        for (int e0 = lid; e0 < count; e0 += Batch * Threads) {
            T prior[Batch];
#pragma unroll
            for (int b = 0; b < Batch; ++b) {
                bool ok;
                const std::ptrdiff_t off = at(e0 + b * Threads, ok);
                prior[b] = (ok && !beta_zero) ? c_item[off] : T(0);
            }
#pragma unroll
            for (int b = 0; b < Batch; ++b) {
                const int e = e0 + b * Threads;
                bool ok;
                const std::ptrdiff_t off = at(e, ok);
                if (!ok) continue;
                const T v = *slot(e / TileM, e % TileM);
                c_item[off] = beta_zero ? alpha * v : LinearEpilogue<T>::apply(alpha, beta, v, prior[b]);
            }
        }
        if (pass + 1 < Passes) ex.barrier();
    }
}

}  // namespace batchlas::sycl_gemm::components

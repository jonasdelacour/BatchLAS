// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, level-3 components: the tuned gemm kernels' stager, micro-kernel, k-loop and
// epilogue (src/ops/gemm/kernels), exported so a kernel reads as stage -> k loop of register
// rank-1 updates -> epilogue (synthesis D10: one implementation, never a second one). Each op is
// the tuned code moved behind a signature: same loop shapes, pragmas, predicate order and
// arithmetic, so a kernel built from them keeps its SASS.
//   WorkGroupNd<WG, D>    a work-group of an nd_range<D> launch (the gemm grids are 3-D)
//   OpArith, FmaArith     the C++ operators (Annex G for std::complex) and the fused form
//   op_elem, conj_value   op(X)(i, j) read from column-major storage
//   epilogue_apply/store, epilogue_store_fused, axpby_value   runtime-beta C updates
//   frag_fill, frag_load, outer_mac, gemv_t_frag, mma_slab, dot_slab   register micro-kernels
//   stage_block, stage_block_packet, stage_block_packet_aligned, stage_storage   stagers
//   kloop, kloop_prefetch2, kloop_async   k loops (the barriers are the pipeline's contract)
//   store_tile_staged     a register tile through local memory to C, one column per warp store
//   sum_seq               x(0) + ... + x(n-1) from x(0) (no +0 start: -0 survives)

#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/dla.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL  // src/sycl/kernel_attrs.hh's BATCHLAS_UNROLL_FULL, which include/ cannot name
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

// ---- executor ------------------------------------------------------------------------------
// The work-group of an nd_range<D> launch. Only barrier() is offered: a gemm kernel computes its
// own thread coordinates (they are part of its tuned lane map), so lid() is not a member.
template <int WG, int D = 3> struct WorkGroupNd {
    static constexpr int size = WG;
    sycl::nd_item<D> it;
    BATCHLAS_DLA_INLINE void barrier(Mem m = Mem::Local) const {
        if (m == Mem::Local) it.barrier(sycl::access::fence_space::local_space);
        else it.barrier(sycl::access::fence_space::global_and_local);
    }
};

// ---- arithmetic ----------------------------------------------------------------------------
namespace detail {
template <class T> concept GcMemberComplex = requires(T t) {
    t.re;
    t.im;
};
template <class T> inline constexpr bool gc_std_complex_v = false;
template <class R> inline constexpr bool gc_std_complex_v<std::complex<R>> = true;
}  // namespace detail

template <class T> BATCHLAS_DLA_INLINE T conj_value(T v) {
    if constexpr (detail::gc_std_complex_v<T>) return std::conj(v);
    else if constexpr (detail::GcMemberComplex<T>) return T{v.re, -v.im};
    else return v;
}

// The C++ operators. For std::complex that is Annex G (a __mulsc3 libcall): parity only, for a
// kernel whose base multiplied with operator*; moving it to Arith<T> is an algo change.
template <class T> struct OpArith {
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return a + b; }
};

// One fma per real product: acc = fma(a, b, acc); a complex takes four fmas in the order
// re += a.re b.re, re -= a.im b.im, im += a.re b.im, im += a.im b.re (sycl_device::fma_acc).
template <class T> struct FmaArith {
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        if constexpr (detail::GcMemberComplex<T>) {
            return T{sycl::fma(a.re, b.re, -a.im * b.im), sycl::fma(a.re, b.im, a.im * b.re)};
        } else if constexpr (detail::gc_std_complex_v<T>) {
            return T(sycl::fma(a.real(), b.real(), -a.imag() * b.imag()), sycl::fma(a.real(), b.imag(), a.imag() * b.real()));
        } else {
            return a * b;
        }
    }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        if constexpr (detail::GcMemberComplex<T>) {
            acc.re = sycl::fma(a.re, b.re, acc.re);
            acc.re = sycl::fma(-a.im, b.im, acc.re);
            acc.im = sycl::fma(a.re, b.im, acc.im);
            acc.im = sycl::fma(a.im, b.re, acc.im);
        } else if constexpr (detail::gc_std_complex_v<T>) {
            auto re = sycl::fma(a.real(), b.real(), acc.real());
            re = sycl::fma(-a.imag(), b.imag(), re);
            auto im = sycl::fma(a.real(), b.imag(), acc.imag());
            im = sycl::fma(a.imag(), b.real(), im);
            acc = T(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
};

// op(X)(row, col) of a column-major X: X(col, row) when transposed, conjugated for ConjTrans.
template <Transpose Op, class T> BATCHLAS_DLA_INLINE T op_elem(const T* x, int ld, int row, int col) {
    const bool transpose = Op != Transpose::NoTrans;
    const int source_row = transpose ? col : row;
    const int source_col = transpose ? row : col;
    // The index sum carries no nsw (unsigned add, as the tuned OperandAccessor's reassociated
    // sum): with nsw LLVM splits it into hoisted 64-bit addresses, +3 to +8 REG on the reg tiles.
    const int index = static_cast<int>(static_cast<unsigned>(source_col * ld) + static_cast<unsigned>(source_row));
    const T v = x[index];
    if constexpr (Op == Transpose::ConjTrans) return conj_value(v);
    else return v;
}

// ---- epilogues (beta is a kernel argument here, so a runtime test stands in for the tag) ----
// alpha acc + beta prior; beta == 0 is alpha acc alone (0 * NaN is NaN, -0 + 0 is +0).
template <class T> inline T epilogue_apply(T alpha, T beta, T acc, T prior) {
    return beta == T(0) ? alpha * acc : alpha * acc + beta * prior;
}
// *dst := alpha acc + beta *dst; beta == 0 never reads *dst.
template <class T> inline void epilogue_store(T* dst, T alpha, T beta, T acc) {
    *dst = beta == T(0) ? alpha * acc : epilogue_apply(alpha, beta, acc, *dst);
}
// alpha acc + beta prior with beta known nonzero (a prefetched prior).
template <class T> BATCHLAS_DLA_INLINE T axpby_value(T alpha, T acc, T beta, T prior) { return alpha * acc + beta * prior; }
// *dst := alpha acc + beta *dst with the host's beta_zero flag; a complex takes Ar's fused
// product and accumulate, a real scalar the operators.
template <class Ar, class T> BATCHLAS_DLA_INLINE void epilogue_store_fused(T* dst, T alpha, T beta, bool beta_zero, T acc) {
    if constexpr (detail::GcMemberComplex<T> || detail::gc_std_complex_v<T>) {
        T out = Ar::mul(alpha, acc);
        if (!beta_zero) Ar::mac(out, beta, *dst);
        *dst = out;
    } else {
        *dst = beta_zero ? alpha * acc : alpha * acc + beta * *dst;
    }
}

// ---- register fragments and micro-kernels ----------------------------------------------------
enum class Unroll { Compiler, Full };  // Full = the full-unroll pragma; Compiler = no pragma

template <Unroll U = Unroll::Full, class T, int N> BATCHLAS_DLA_INLINE void frag_fill(T (&f)[N], T v) {
    if constexpr (U == Unroll::Full) {
BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < N; ++i) f[i] = v;
    } else {
        for (int i = 0; i < N; ++i) f[i] = v;
    }
}
template <Unroll U = Unroll::Full, class T, int TR, int TC> BATCHLAS_DLA_INLINE void frag_fill(T (&f)[TR][TC], T v) {
    if constexpr (U == Unroll::Full) {
BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < TR; ++i) {
BATCHLAS_DLA_UNROLL_FULL
            for (int j = 0; j < TC; ++j) f[i][j] = v;
        }
    } else {
        for (int i = 0; i < TR; ++i) {
            for (int j = 0; j < TC; ++j) f[i][j] = v;
        }
    }
}

// f[i] := at(i) for i < N.
template <Unroll U = Unroll::Full, class T, int N, class At> BATCHLAS_DLA_INLINE void frag_load(T (&f)[N], At&& at) {
    if constexpr (U == Unroll::Full) {
BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < N; ++i) f[i] = at(i);
    } else {
        for (int i = 0; i < N; ++i) f[i] = at(i);
    }
}

// acc += a b^T on a TR x TC register tile: one rank-1 update, the register micro-kernel's step.
template <Unroll U = Unroll::Full, class Ar = void, class T, int TR, int TC>
BATCHLAS_DLA_INLINE void outer_mac(T (&acc)[TR][TC], const T (&a)[TR], const T (&b)[TC]) {
    using A = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    if constexpr (U == Unroll::Full) {
BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < TR; ++i) {
BATCHLAS_DLA_UNROLL_FULL
            for (int j = 0; j < TC; ++j) A::mac(acc[i][j], a[i], b[j]);
        }
    } else {
        for (int i = 0; i < TR; ++i) {
            for (int j = 0; j < TC; ++j) A::mac(acc[i][j], a[i], b[j]);
        }
    }
}

// One staged k slab of the register-tiled micro-kernel: for each t < TileK (and kk0 + t < k
// unless Exact) load a(i) = a_at(t, i), b(j) = b_at(t, j) and acc += a b^T. UnrollK k steps per
// outer iteration; no pragmas (the compiler's own unroll, as tuned). k by reference, like the
// tuned lambda's closure read: by value, SimplifyCFG merged the two exits first and the TileK = 8
// slab stayed rolled (32x32x8: 34 -> 6 FFMA).
template <int TileK, int UnrollK, bool Exact, class Ar = void, class T, int TR, int TC, class FA, class FB>
BATCHLAS_DLA_INLINE void mma_slab(T (&acc)[TR][TC], int kk0, const int& k, FA&& a_at, FB&& b_at) {
    auto step = [&](int t) __attribute__((always_inline)) {
        T a_frag[TR];
        T b_frag[TC];
        for (int i = 0; i < TR; ++i) a_frag[i] = a_at(t, i);
        for (int j = 0; j < TC; ++j) b_frag[j] = b_at(t, j);
        outer_mac<Unroll::Compiler, Ar>(acc, a_frag, b_frag);
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
template <int N, class Ar = void, class T, class VX, class VY>
BATCHLAS_DLA_INLINE void dot_slab(T& acc, const VX& x, const VY& y, int kk0, int k) {
    using A = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    for (int t = 0; t < N && kk0 + t < k; ++t) A::mac(acc, x(t), y(t));
}

namespace detail {
template <int L0, int Lc, class Ar, class T, int NA, int NC, class FB>
BATCHLAS_DLA_INLINE void gemv_t_frag_chunk(T (&acc)[NC], const T (&a)[NA], FB& b_at, int k) {
BATCHLAS_DLA_UNROLL_FULL
    for (int l = L0; l < L0 + Lc; ++l) {
        if (l >= k) continue;  // uniform; `continue` keeps the unroll, so a[] stays in registers
BATCHLAS_DLA_UNROLL_FULL
        for (int j = 0; j < NC; ++j) Ar::mac(acc[j], a[l], b_at(l, j));
    }
}
}  // namespace detail

// acc(j) += sum_{l < k} a(l) B(l, j) for j < NC, l outermost (one register row times a staged
// B): gemv^T with x and y in registers. Chunk = NA is one unrolled loop; Chunk = NA / 4 is four
// sequential ones (one complex loop over 64 l passes LLVM's unroll threshold: a[] spills).
template <int Chunk, class Ar = void, class T, int NA, int NC, class FB>
BATCHLAS_DLA_INLINE void gemv_t_frag(T (&acc)[NC], const T (&a)[NA], FB&& b_at, int k) {
    using A = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    static_assert(Chunk == NA || Chunk * 4 == NA, "gemv_t_frag: one loop, or four");
    detail::gemv_t_frag_chunk<0, Chunk, A>(acc, a, b_at, k);
    if constexpr (NA / Chunk > 1) {
        detail::gemv_t_frag_chunk<Chunk, Chunk, A>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<2 * Chunk, Chunk, A>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<3 * Chunk, Chunk, A>(acc, a, b_at, k);
    }
}

// ---- stagers: op(X) blocks into local memory, zero outside the matrix --------------------------
// dst(r, c) := op(X)(r0 + r, c0 + c) for the R x C block, zero where row >= mr or col >= mc.
// Items lid, lid + Threads, ... walk it with X's contiguous index fastest (r for NoTrans, c for
// Trans), so a warp reads consecutive addresses whatever the slot map dst is. x, ld and the
// bounds are references for the same reason as mma_slab's k (the reg tiles' REG and LDG/STS).
template <Transpose Op, int R, int C, int Threads, class T, class Dst>
BATCHLAS_DLA_INLINE void stage_block(int lid, const T* const& x, const int& ld, int r0, int c0, const int& mr, const int& mc, Dst&& dst) {
    for (int index = lid; index < R * C; index += Threads) {
        const int r = Op == Transpose::NoTrans ? index % R : index / C;
        const int c = Op == Transpose::NoTrans ? index / R : index % C;
        const int row = r0 + r;
        const int col = c0 + c;
        dst(r, c) = (row < mr && col < mc) ? op_elem<Op>(x, ld, row, col) : T(0);
    }
}

namespace detail {
template <class T, int V> BATCHLAS_DLA_INLINE sycl::vec<T, V> packet_at(const T* x, int offset) {
    return *reinterpret_cast<const sycl::vec<T, V>*>(x + offset);
}
template <class Fn, int... I> BATCHLAS_DLA_INLINE void gc_static_for(Fn&& fn, std::integer_sequence<int, I...>) {
    (fn(std::integral_constant<int, I>{}), ...);
}
}  // namespace detail

// stage_block with V-element aligned packets along X's contiguous index (x + offset must be
// 16-byte aligned for every in-range packet: the caller's ld and base test); a packet that
// crosses the matrix edge falls back to predicated scalars.
template <Transpose Op, int R, int C, int V, int Threads, class T, class Dst>
BATCHLAS_DLA_INLINE void stage_block_packet(int lid, const T* const& x, const int& ld, int r0, int c0, const int& mr, const int& mc, Dst&& dst) {
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
                const int base = col * ld + row;
                const auto packet_values = detail::packet_at<T, V>(x, base);
                for (int lane = 0; lane < V; ++lane) dst(r + lane, c) = packet_values[lane];
            } else {
                for (int lane = 0; lane < V; ++lane) {
                    const int rr = row + lane;
                    dst(r + lane, c) = (rr < mr && col < mc) ? op_elem<Op>(x, ld, rr, col) : T(0);
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
                const int base = row * ld + col;
                const auto packet_values = detail::packet_at<T, V>(x, base);
                for (int lane = 0; lane < V; ++lane) dst(r, c + lane) = packet_values[lane];
            } else {
                for (int lane = 0; lane < V; ++lane) {
                    const int cc = col + lane;
                    dst(r, c + lane) = (row < mr && cc < mc) ? op_elem<Op>(x, ld, row, cc) : T(0);
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
    detail::gc_static_for(
        [&](auto iter) {
            constexpr int iteration = iter;
            const int packet = lid + iteration * Threads;
            if (packet < PacketCount) {
                const int r = (packet % PacketsPerCol) * V;
                const int c = packet / PacketsPerCol;
                const int row = r0 + r;
                const int col = c0 + c;
                const int base = col * ld + row;
                const auto packet_values = detail::packet_at<T, V>(x, base);
                for (int lane = 0; lane < V; ++lane) dst(r + lane, c) = packet_values[lane];
            }
        },
        std::make_integer_sequence<int, Iterations>{});
}

// dst(i, j) := op(X)(i, j) over the S0 x S1 STORAGE block of X, walked in storage order (s0
// fastest: coalesced either way), items start, start + Step, ...; (i, j) = (s1, s0) when trans.
// Zero where !live or outside [0, mi) x [0, mj); conj conjugates. trans / conj are runtime values
// (one kernel serves every form); a constant folds.
template <int Step, class S0, class S1, class T, class Ld, class Dst>
BATCHLAS_DLA_INLINE void stage_storage(int start, S0 s0n, S1 s1n, bool live, const T* x, Ld ld, bool trans, bool conj,
                                       int mi, int mj, Dst&& dst) {
    for (int e = start; e < int(s0n) * int(s1n); e += Step) {
        const int s0 = e % int(s0n), s1 = e / int(s0n);
        const int i = trans ? s1 : s0;
        const int j = trans ? s0 : s1;
        T v{};
        if (live && i < mi && j < mj) v = x[s0 + s1 * ld];
        dst(i, j) = conj ? conj_value(v) : v;
    }
}

// ---- k loops ---------------------------------------------------------------------------------
// load(kk0, stage) fills local stage `stage` with k slab kk0; compute(stage, kk0) consumes it.
// Stages 1: load, barrier, compute, barrier per slab. Stages 2: slab t + 1 loads while slab t
// computes, one barrier per slab. Exact: tiles >= 0 slabs of TileK each and no tail (the
// aligned form, whose first load sits behind a tiles > 0 test).
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
// them to local buffer buf, compute(buf) consumes a buffer. One barrier per slab: slab s + 1 is
// stored into the buffer slab s - 1 was read from, and that read ended before the barrier that
// closed iteration s - 1. Unrolled by two so buf is a constant. Ends without a barrier.
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
    [&]<int... J>(std::integer_sequence<int, J...>) __attribute__((always_inline)) {
        ((J < slabs ? issue(J * TileK, std::integral_constant<int, J>{}) : void(), commit()), ...);
    }(std::make_integer_sequence<int, Stages - 1>{});
    auto step = [&](int s, auto buf) __attribute__((always_inline)) {
        constexpr int b = decltype(buf)::value;
        wait();
        ex.barrier();
        if (s + Stages - 1 < slabs) issue((s + Stages - 1) * TileK, std::integral_constant<int, (b + Stages - 1) % Stages>{});
        commit();
        compute(b);
    };
    for (int s0 = 0; s0 < slabs; s0 += Stages) {
        [&]<int... J>(std::integer_sequence<int, J...>) __attribute__((always_inline)) {
            ((s0 + J < slabs && (step(s0 + J, std::integral_constant<int, J>{}), true)) && ...);
        }(std::make_integer_sequence<int, Stages>{});
    }
    drain();
}

// ---- C epilogue through local memory -----------------------------------------------------------
// C(m0 + r, n0 + c) := alpha acc + beta C for a TileM x TileN tile whose item (local_row,
// local_col) holds rows local_row * TR + i, cols local_col * TC + j. Columns go through local
// memory (lo: LoElems, then hi: HiElems; odd stride TileM + 1) in passes, then lane = row stores
// run down C's columns (coalesced at any ldc), Batch priors in flight per item. Precondition:
// the caller's last read of lo and hi ended before a barrier. beta == 0 never reads C.
template <int TileM, int TileN, int TR, int TC, int Threads, int LoElems, int HiElems, int Batch = 2, class Ex, class T>
BATCHLAS_DLA_INLINE void store_tile_staged(const Ex& ex, int lid, int local_row, int local_col, const T (&acc)[TR][TC],
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
            for (int i = 0; i < TR; ++i) *slot(c, local_row * TR + i) = acc[i][j];
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
                c_item[off] = beta_zero ? alpha * v : epilogue_apply(alpha, beta, v, prior[b]);
            }
        }
        if (pass + 1 < Passes) ex.barrier();
    }
}

// ---- elementwise ---------------------------------------------------------------------------------
// x(0) + x(stride) + ... + x((n - 1) stride), left to right from x(0); n >= 1.
template <class Ar = void, class T, class S> BATCHLAS_DLA_INLINE T sum_seq(const T* x, int n, S stride) {
    using A = std::conditional_t<std::is_void_v<Ar>, Arith<T>, Ar>;
    T acc = x[0];
    for (int q = 1; q < n; ++q) {
        const T v = x[q * stride];
        acc = A::add(acc, v);
    }
    return acc;
}

}  // namespace batchlas::device::v2

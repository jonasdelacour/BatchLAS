#pragma once

// A 128x128x8 register-tiled GEMM with 8x8 accumulators per thread.
//
// Shallow K (8) with M = N = 128 buys an 8x8 thread tile, so a k-step is 4
// vectorized shared loads for 64 FFMAs. The shared tiles use stride exactly
// TileM / TileN (an odd stride defeats the 16-byte alignment the LDS.128 form
// needs), and B is staged [k][n] so a thread's 8 B values are contiguous.
//
// Four more things are load-bearing, and each fails silently if undone:
//   * prefetch registers are clang ext_vector values, NOT a struct (LLVM turns a
//     struct copy into a memcpy sunk to the store: no prefetch at all);
//   * the lane swizzle: m takes lane bit 0, n lane bit 1 (an LDS.128 whose address
//     depends on both bits costs twice the wavefronts); warp sub-tile 64 x 32;
//   * the L2::128B fetch hint on the k-contiguous loads (B, and A^T);
//   * the launch bound, two groups per CU (128-register cap), is a guard: the
//     kernel fits at 127, one register more is one group per SM, 3/CU spills.
// evidence: docs/perf/gemm.md#the-128x128-float-kernel
// evidence: docs/perf/gemm.md#gemm-the-128x128-kernel-in-every-real-transpose-form

#include <src/ops/gemm/kernels/accessors.hh>
#include <src/ops/gemm/kernels/copy_async.hh>
#include <src/ops/gemm/kernels/epilogue_linear.hh>
#include <src/ops/gemm/kernels/vec4.hh>

#include <src/ops/gemm/gemm_kernels.hh>

#include <src/linalg-impl.hh>
#include <src/sycl/kernel_attrs.hh>

#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

template <typename T, bool AlignedFastPath, bool StagedEpilogue, Transpose OpA, Transpose OpB, int Stages>
class GemmRegister128x128Kernel;

template <typename T>
inline const Vec4<T>& vec4_ref(const T* p) {
    return *reinterpret_cast<const Vec4<T>*>(__builtin_assume_aligned(p, 4 * sizeof(T)));
}

template <typename T>
inline Vec4<T>& vec4_ref(T* p) {
    return *reinterpret_cast<Vec4<T>*>(__builtin_assume_aligned(p, 4 * sizeof(T)));
}

// A 16-byte global load that asks L2 to fetch the whole 128-byte line. PTX has
// no portable spelling, so every other target takes the plain load.
template <typename T>
inline Vec4<T> load4_l2_line(const T* p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    if constexpr (std::is_same_v<T, float>) {
        Vec4<T> r;
        asm("ld.global.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
            : "l"(p));
        return r;
    }
#elif BATCHLAS_SYCL_IMPL_ACPP && defined(__x86_64__)
    // SSCP: the asm passes x86 Sema (no "=f"; `{` is a dialect brace, hence %{ %}).
    // An arm64 host (macOS) rejects "l" and has no CUDA device: it takes the plain load.
    if constexpr (std::is_same_v<T, float>) {
        if (sycl_impl::on_ptx()) {
            unsigned x = 0, y = 0, z = 0, w = 0;
            __acpp_if_target_sscp(asm("ld.global.L2::128B.v4.b32 %{%0, %1, %2, %3%}, [%4];"
                                      : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                                      : "l"(p));)
            return Vec4<T>{__builtin_bit_cast(T, x), __builtin_bit_cast(T, y),
                           __builtin_bit_cast(T, z), __builtin_bit_cast(T, w)};
        }
    }
#endif
    return vec4_ref(p);
}

inline constexpr int kStagedEpilogueMaxK = 64;

// The scalar form, for the predicated leg's B loads.
template <typename T>
inline T load1_l2_line(const T* p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    if constexpr (std::is_same_v<T, float>) {
        T r;
        asm("ld.global.L2::128B.f32 %0, [%1];" : "=f"(r) : "l"(p));
        return r;
    }
#elif BATCHLAS_SYCL_IMPL_ACPP && defined(__x86_64__)
    if constexpr (std::is_same_v<T, float>) {
        if (sycl_impl::on_ptx()) {
            unsigned r = 0;
            __acpp_if_target_sscp(asm("ld.global.L2::128B.b32 %0, [%1];" : "=r"(r) : "l"(p));)
            return __builtin_bit_cast(T, r);
        }
    }
#endif
    return *p;
}

// Does this problem satisfy everything the unpredicated path assumes? A leg predicate only:
// it picks aligned vs predicated inside the launcher and is never a routing term.
template <typename T>
inline bool can_use_128x128_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      const MatrixView<T, MatrixFormat::Dense>& C,
                                      Transpose transA, Transpose transB) {
    constexpr int TileM = 128, TileN = 128, TileK = 8;
    const auto [m, k] = get_effective_dims(A, transA);
    const auto n = get_effective_dims(B, transB).second;
    // k == 0 would pass the modulus test, and the prefetch of slab 0 runs before the loop.
    if (k < TileK || (m % TileM) != 0 || (n % TileN) != 0 || (k % TileK) != 0) {
        return false;
    }
    // Every 128-bit access this kernel makes is at a multiple of 4 elements
    // from the base pointer, so the base itself must be 4-element aligned and
    // the leading dimensions must preserve that. A transposed operand's packet
    // runs down k from a multiple of 4 instead of down m or n: the same terms.
    auto aligned = [](const T* p, int ld, int stride) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % (4 * sizeof(T))) == 0 &&
            (ld % 4) == 0 && (stride % 4) == 0;
    };
    return aligned(A.data_ptr(), A.ld(), A.stride()) &&
        aligned(B.data_ptr(), B.ld(), B.stride()) &&
        aligned(C.data_ptr(), C.ld(), C.stride());
}

// Each kernel functor (Body, launched directly on aligned u1; Gemm128x128Inlined, predicated u1;
// Gemm128x128Pipelined, u4) also declares its work-group size: ptxas sees .reqntid and holds the two-group cap (128 registers); the launch bound alone is
// dropped (known defect #20) and left u1 at up to 139 and staged u4 at 155-159 (one group per SM).
// NVPTX device code only, like BATCHLAS_LAUNCH_BOUNDS (none under acpp: speed only). Local, not in
// kernel_attrs.hh: that header is in every op's tuning hash. It replaces BATCHLAS_LAUNCH_BOUNDS:
// .maxntid next to .reqntid is a ptxas error ("Conflicting directives") on CUDA 13.2 sm_89.
#if !BATCHLAS_SYCL_IMPL_ACPP && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define BATCHLAS_GEMM_REQD_WG(n, min_groups) \
    [[sycl::reqd_work_group_size(1, 1, n), intel::min_work_groups_per_cu(min_groups)]]
#else
#define BATCHLAS_GEMM_REQD_WG(n, min_groups)
#endif

// A FUNCTOR, not a lambda, only so the launch bound can be spelled.
// StagedEpilogue routes C through local memory so each warp stores whole
// 128-row columns, on either leg; see launch_register_128x128_k8.
// A real scalar only: a transposed operand is read as Trans (ConjTrans folds to it).
// Stages > 2 (predicated leg only) fills Stages local slabs by zero-filled copy_async elements
// instead of the register prefetch.
template <typename T, bool AlignedFastPath, bool StagedEpilogue, Transpose OpA = Transpose::NoTrans,
          Transpose OpB = Transpose::NoTrans, int Stages = 2>
struct Gemm128x128Body {
    static constexpr int TileM = 128, TileN = 128, TileK = 8;
    static constexpr int ThreadTile = 8, Band = 4, Threads = 256;
    static constexpr int BandM = 32, BandN = 16;  // second band's offset
    // Which operands are contiguous in k: A^T and B (the transposing loader), else down m or n.
    static constexpr bool AK = OpA != Transpose::NoTrans, BK = OpB == Transpose::NoTrans;
    // The predicated pipeline copies one element per lane: a packet operand's lanes take
    // consecutive m (n), 32 apart per copy, so each copy is one 128-byte warp access at any ld. A
    // k-contiguous operand's lanes take k fastest (8 lanes = one 32-byte row segment, 4 rows per copy),
    // and the shared stride is padded by 4 so those 8 lanes' transposed writes hit distinct banks
    // (both operands, so the local memory is one size per config: RegCfg::slm_bytes).
    // evidence: docs/perf/gemm.md#gemm-wp-f-float-mid-size-parity-on-sm_120
    static constexpr bool kLaneMap = Stages > 2;
    static constexpr int AStride = TileM + (kLaneMap ? 4 : 0), BStride = TileN + (kLaneMap ? 4 : 0);
    static constexpr int kMinGroupsPerCu = 2;
    static constexpr int StageStride = TileM + 4;  // staged C column stride
    static constexpr int kStages = Stages;
    static constexpr int TileAElems = StagedEpilogue && 32 * StageStride > Stages * TileK * AStride
                                          ? 32 * StageStride
                                          : Stages * TileK * AStride;
    static constexpr int TileBElems = Stages * TileK * BStride;
    static_assert(Stages == 2 || (Stages == 4 && !AlignedFastPath && sizeof(T) == 4),
                  "the copy_async pipeline is the float predicated leg at four stages (trace names say s4)");
    static constexpr KernelVariant kVariant =
        Stages == 2 ? (!AK ? (BK ? KernelVariant::Tiled128x128RegisterK8 : KernelVariant::Tiled128x128RegisterK8NT)
                           : (BK ? KernelVariant::Tiled128x128RegisterK8TN : KernelVariant::Tiled128x128RegisterK8TT))
                    : (!AK ? (BK ? KernelVariant::Tiled128x128RegisterK8S4 : KernelVariant::Tiled128x128RegisterK8S4NT)
                           : (BK ? KernelVariant::Tiled128x128RegisterK8S4TN
                                 : KernelVariant::Tiled128x128RegisterK8S4TT));

    const T* a_ptr;
    const T* b_ptr;
    T* c_ptr;
    int m, n, k, lda, ldb, ldc;
    std::ptrdiff_t stride_a, stride_b, stride_c;
    int batch;
    T alpha, beta;
    sycl::local_accessor<T, 1> tile_a, tile_b;

    BATCHLAS_GEMM_REQD_WG(Threads, kMinGroupsPerCu)
    void operator()(sycl::nd_item<3> item) const {
        const int bid = static_cast<int>(item.get_group(0));
        if (bid >= batch) {
            return;
        }
        const int tid = static_cast<int>(item.get_local_id(2));
        const int lane = tid & 31, warp = tid >> 5;
        const int ml = (lane & 1) | (((lane >> 2) & 3) << 1);
        const int nl = ((lane >> 1) & 1) | (((lane >> 4) & 1) << 1);
        const int mb = (warp & 1) * 64 + ml * Band;
        const int nb = (warp >> 1) * 32 + nl * Band;

        const int m0 = static_cast<int>(item.get_group(1)) * TileM;
        const int n0 = static_cast<int>(item.get_group(2)) * TileN;
        const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
        const T* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
        T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
        T* sa = tile_a.template get_multi_ptr<sycl::access::decorated::no>().get();
        T* sb = tile_b.template get_multi_ptr<sycl::access::decorated::no>().get();
        [[maybe_unused]] std::uint32_t sa_l = 0, sb_l = 0;  // shared-window byte addresses (cp.async)
#if BATCHLAS_GEMM_CP_ASYNC
        if constexpr (Stages > 2) {
            // The shared-window byte address cp.async takes, from the generic local pointer.
            asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(sa_l) : "l"(sa));
            asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(sb_l) : "l"(sb));
        }
#endif

        T accum[ThreadTile][ThreadTile];
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < ThreadTile; ++i) {
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < ThreadTile; ++j) {
                accum[i][j] = T(0);
            }
        }

        // Stages == 2 (the Stages > 2 lane map is described at kLaneMap):
        // one packet of each operand per thread per slab, both coalesced. Down m or n (A, B^T):
        // a warp reads 128 contiguous elements of one k row and stores one vector each. Down k
        // (B, A^T): two lanes per column read 4 k each and transpose into shared (2-way bank
        // conflict); the L2::128B hint fetches the line the next three slabs read.
        const int a_m = AK ? (kLaneMap ? tid / 8 : tid / 2) : (tid % 32) * (kLaneMap ? 1 : 4);
        const int a_k = AK ? (kLaneMap ? tid % 8 : (tid % 2) * 4) : tid / 32;
        const int b_n = BK ? (kLaneMap ? tid / 8 : tid / 2) : (tid % 32) * (kLaneMap ? 1 : 4);
        const int b_k = BK ? (kLaneMap ? tid % 8 : (tid % 2) * 4) : tid / 32;
        const T* ga = AK ? Ab + a_k + static_cast<std::ptrdiff_t>(m0 + a_m) * lda
                         : Ab + (m0 + a_m) + static_cast<std::ptrdiff_t>(a_k) * lda;
        const T* gb = BK ? Bb + b_k + static_cast<std::ptrdiff_t>(n0 + b_n) * ldb
                         : Bb + (n0 + b_n) + static_cast<std::ptrdiff_t>(b_k) * ldb;

        // B^T bumps gb one slab per call (gload runs once per slab, in order): an int k0 * ldb
        // overflows on a large item, and a 64-bit product spilled the staged NT leg.
        Vec4<T> ra, rb;
        auto gload = [&](int k0, auto edge) __attribute__((always_inline)) {
            if constexpr (AlignedFastPath) {
                if constexpr (AK) ra = load4_l2_line(ga + k0);
                else ra = vec4_ref(ga + static_cast<std::ptrdiff_t>(k0) * lda);
                if constexpr (BK) rb = load4_l2_line(gb + k0);
                else rb = vec4_ref(gb);
            } else if constexpr (!decltype(edge)::value) {
                // An interior group: rows and columns are in range, so only the k tail is bounded.
                const bool kfull = k0 + TileK <= k;
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < 4; ++i) {
                    if (kfull) {
                        if constexpr (AK) ra[i] = load1_l2_line(ga + k0 + i);
                        else ra[i] = ga[static_cast<std::ptrdiff_t>(k0) * lda + i];
                        if constexpr (BK) rb[i] = load1_l2_line(gb + k0 + i);
                        else rb[i] = gb[i];
                    } else {
                        if constexpr (AK) ra[i] = k0 + a_k + i < k ? load1_l2_line(ga + k0 + i) : T(0);
                        else ra[i] = k0 + a_k < k ? ga[static_cast<std::ptrdiff_t>(k0) * lda + i] : T(0);
                        if constexpr (BK) rb[i] = k0 + b_k + i < k ? load1_l2_line(gb + k0 + i) : T(0);
                        else rb[i] = k0 + b_k < k ? gb[i] : T(0);
                    }
                }
            } else {
                // Zero outside the matrix, so the inner loop needs no bounds. Same lane maps as
                // the aligned leg, so an odd ld stays coalesced. Register-cap choices (ptxas at 128):
                // B^T's bound is one live int, not four hoisted row predicates (with A's four, NT
                // spilled); NN's A takes one int too (its four spilled the staged NN leg), NT's
                // keeps its four (two ints spilled NT).
                constexpr bool kOneIntA = !AK && BK;
                [[maybe_unused]] const int lim_a = kOneIntA ? (k0 + a_k < k ? m - (m0 + a_m) : 0) : 0;
                [[maybe_unused]] const int lim_b = BK ? 0 : (k0 + b_k < k ? n - (n0 + b_n) : 0);
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < 4; ++i) {
                    if constexpr (AK)
                        ra[i] = (k0 + a_k + i < k && m0 + a_m < m) ? load1_l2_line(ga + k0 + i) : T(0);
                    else if constexpr (kOneIntA)
                        ra[i] = i < lim_a ? ga[static_cast<std::ptrdiff_t>(k0) * lda + i] : T(0);
                    else
                        ra[i] = (m0 + a_m + i < m && k0 + a_k < k) ? ga[static_cast<std::ptrdiff_t>(k0) * lda + i]
                                                                   : T(0);
                    if constexpr (BK)
                        rb[i] = (k0 + b_k + i < k && n0 + b_n < n) ? load1_l2_line(gb + k0 + i) : T(0);
                    else
                        rb[i] = i < lim_b ? gb[i] : T(0);
                }
            }
            if constexpr (!BK) gb += static_cast<std::ptrdiff_t>(TileK) * ldb;
        };
        auto sstore = [&](int buf) __attribute__((always_inline)) {
            if constexpr (AK) {
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < 4; ++i) {
                    sa[buf * TileK * AStride + (a_k + i) * AStride + a_m] = ra[i];
                }
            } else {
                vec4_ref(&sa[buf * TileK * AStride + a_k * AStride + a_m]) = ra;
            }
            if constexpr (BK) {
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < 4; ++i) {
                    sb[buf * TileK * BStride + (b_k + i) * BStride + b_n] = rb[i];
                }
            } else {
                vec4_ref(&sb[buf * TileK * BStride + b_k * BStride + b_n]) = rb;
            }
        };
        auto compute = [&](int buf) __attribute__((always_inline)) {
            const T* xa = sa + buf * TileK * AStride;
            const T* xb = sb + buf * TileK * BStride;
BATCHLAS_UNROLL_FULL
            for (int kk = 0; kk < TileK; ++kk) {
                const Vec4<T> a0 = vec4_ref(&xa[kk * AStride + mb]);
                const Vec4<T> a1 = vec4_ref(&xa[kk * AStride + BandM + mb]);
                const Vec4<T> b0 = vec4_ref(&xb[kk * BStride + nb]);
                const Vec4<T> b1 = vec4_ref(&xb[kk * BStride + BandN + nb]);
                const T af[ThreadTile] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const T bf[ThreadTile] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < ThreadTile; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < ThreadTile; ++j) {
                        accum[i][j] += af[i] * bf[j];
                    }
                }
            }
        };

        // One barrier per slab is enough: slab s+1 is stored into the buffer
        // slab s-1 was read from, and every read of that finished before the
        // barrier that ended iteration s-1. Unrolled by two so the buffer
        // index is a constant; a runtime one costs a spill at the 128 cap.
        const int slabs = (k + TileK - 1) / TileK;
        auto mainloop = [&](auto edge) __attribute__((always_inline)) {
            gload(0, edge);
            sstore(0);
            item.barrier(sycl::access::fence_space::local_space);
            int s = 0;
            for (; s + 1 < slabs; s += 2) {
                gload((s + 1) * TileK, edge);
                compute(0);
                sstore(1);
                item.barrier(sycl::access::fence_space::local_space);
                const bool more = s + 2 < slabs;
                if (more) gload((s + 2) * TileK, edge);
                compute(1);
                if (more) sstore(0);
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (s < slabs) {
                compute(0);  // an odd slab count leaves the last slab in buffer 0
            }
        };
        if constexpr (Stages > 2) {
            // Stages - 1 slabs in flight. Step s waits for slab s (its own copies, then the
            // barrier for everyone's), refills the buffer slab s - 1 was read from (every read of it ended
            // before that barrier) and computes. One commit per step, empty past the end, keeps the wait
            // count at Stages - 2. Element copies, in range iff the row (column) and the k index are; each
            // operand pointer bumps one slab per issue (a k0 * ld product spilled staged NN and NT at the
            // 128 cap). Unrolled by Stages: the buffer is a constant. A zero-size copy names the
            // operand's batch base, not its own address (up to 127 m/n or 7 k indices past the operand).
            // 32 ld, rebuilt per issue: hoisted and held across the mainloop it spilled 16 B in staged u4
            // NN, TN and TT on sm_89 at the 128 cap.
            auto ld32 = [](int ld) __attribute__((always_inline)) {
#if BATCHLAS_GEMM_CP_ASYNC
                asm volatile("" : "+r"(ld));
#endif
                return static_cast<std::ptrdiff_t>(32) * ld;
            };
            auto issue = [&](int k0, auto buf) __attribute__((always_inline)) {
                constexpr int oa = decltype(buf)::value * TileK * AStride, ob = decltype(buf)::value * TileK * BStride;
                if constexpr (AK) {
                    const int lim = k0 + a_k < k ? m - (m0 + a_m) : 0;
                    const std::ptrdiff_t lda32 = ld32(lda);
BATCHLAS_UNROLL_FULL
                    for (int i = 0; i < 4; ++i) {
                        const int o = oa + a_k * AStride + a_m + 32 * i;
                        copy_async_zfill<4>(sa + o, sa_l + 4u * o, 32 * i < lim ? ga + lda32 * i : a_ptr, 32 * i < lim ? 4u : 0u);
                    }
                    ga += TileK;
                } else {
                    const int lim = k0 + a_k < k ? m - (m0 + a_m) : 0;
BATCHLAS_UNROLL_FULL
                    for (int i = 0; i < 4; ++i) {
                        const int o = oa + a_k * AStride + a_m + 32 * i;
                        copy_async_zfill<4>(sa + o, sa_l + 4u * o, 32 * i < lim ? ga + 32 * i : a_ptr, 32 * i < lim ? 4u : 0u);
                    }
                    ga += static_cast<std::ptrdiff_t>(TileK) * lda;
                }
                if constexpr (BK) {
                    const int lim = k0 + b_k < k ? n - (n0 + b_n) : 0;
                    const std::ptrdiff_t ldb32 = ld32(ldb);
BATCHLAS_UNROLL_FULL
                    for (int i = 0; i < 4; ++i) {
                        const int o = ob + b_k * BStride + b_n + 32 * i;
                        copy_async_zfill<4>(sb + o, sb_l + 4u * o, 32 * i < lim ? gb + ldb32 * i : b_ptr, 32 * i < lim ? 4u : 0u);
                    }
                    gb += TileK;
                } else {
                    const int lim = k0 + b_k < k ? n - (n0 + b_n) : 0;
BATCHLAS_UNROLL_FULL
                    for (int i = 0; i < 4; ++i) {
                        const int o = ob + b_k * BStride + b_n + 32 * i;
                        copy_async_zfill<4>(sb + o, sb_l + 4u * o, 32 * i < lim ? gb + 32 * i : b_ptr, 32 * i < lim ? 4u : 0u);
                    }
                    gb += static_cast<std::ptrdiff_t>(TileK) * ldb;
                }
            };
            [&]<int... J>(std::integer_sequence<int, J...>) __attribute__((always_inline)) {
                ((J < slabs ? issue(J * TileK, std::integral_constant<int, J>{}) : void(), copy_async_commit()), ...);
            }(std::make_integer_sequence<int, Stages - 1>{});
            auto step = [&](int s, auto buf) __attribute__((always_inline)) {
                constexpr int b = decltype(buf)::value;
                copy_async_wait<Stages - 2>();
                item.barrier(sycl::access::fence_space::local_space);
                if (s + Stages - 1 < slabs)
                    issue((s + Stages - 1) * TileK, std::integral_constant<int, (b + Stages - 1) % Stages>{});
                copy_async_commit();
                compute(b);
            };
            for (int s0 = 0; s0 < slabs; s0 += Stages) {
                [&]<int... J>(std::integer_sequence<int, J...>) __attribute__((always_inline)) {
                    ((s0 + J < slabs && (step(s0 + J, std::integral_constant<int, J>{}), true)) && ...);
                }(std::make_integer_sequence<int, Stages>{});
            }
            copy_async_wait<0>();
        } else if constexpr (AlignedFastPath) {
            mainloop(std::true_type{});
        } else if (m0 + TileM > m || n0 + TileN > n) {
            // The predicated leg's interior groups (uniform test) run a copy of the loop with no row
            // or column predicates. At the 128 cap a per-slab branch spilled 4-24 B, and so did this
            // test written interior-first (4 B); ptxas is chaotic here, so re-probe any edit.
            // evidence: docs/perf/gemm.md#gemm-wp-i-an-unpredicated-interior-loop-on-the-128x128-predicated-leg
            mainloop(std::true_type{});
        } else {
            mainloop(std::false_type{});
        }

        if constexpr (StagedEpilogue) {
            // Pass p stages the 32 tile columns c with c % 4 == p, slot c / 4,
            // then each warp stores 4 of them as whole 128-row columns: 16-byte
            // stores on the aligned leg, predicated scalars at rows lane + 32 i
            // on the other (a per-group 16-byte branch there spilled at the cap).
            T* sc = sa;
            int tid2 = static_cast<int>(item.get_local_id(2));
#if BATCHLAS_GEMM_CP_ASYNC
            // Opaque re-read keeps every epilogue address out of the mainloop (no hoist) on u4 and on
            // u1 aligned (u4 staged NN spilled 20 B at the 128 cap with it hoisted; u1 aligned staged
            // TT 32 B on sm_120).
            if constexpr (Stages > 2 || AlignedFastPath) asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid2));
#endif
            item.barrier(sycl::access::fence_space::local_space);  // the odd tail read sa
BATCHLAS_UNROLL_FULL
            for (int p = 0; p < Band; ++p) {
BATCHLAS_UNROLL_FULL
                for (int half = 0; half < 2; ++half) {
                    const int slot = (nb >> 2) + half * (BandN / 4);
BATCHLAS_UNROLL_FULL
                    for (int band = 0; band < 2; ++band) {
                        Vec4<T> v;
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) {
                            v[i] = accum[band * Band + i][p + half * Band];
                        }
                        vec4_ref(&sc[slot * StageStride + band * BandM + mb]) = v;
                    }
                }
                // Predicated leg: the pass's 16 priors are loaded after the stage write (its
                // accumulators are dead) and before the barrier, all in flight at once. Read inline, each
                // load waited before its store: 0.84x u4's time at 256x512x78 strided.
                // evidence: docs/perf/gemm.md#gemm-wp-f-float-mid-size-parity-on-sm_120
                [[maybe_unused]] T prior[4][4];
                if constexpr (!AlignedFastPath) {
                    const bool bz = beta == T(0);
                    const int rows_left = m - m0 - (tid2 & 31);
BATCHLAS_UNROLL_FULL
                    for (int q = 0; q < 4; ++q) {
                        const int col = n0 + ((tid2 >> 5) * 4 + q) * 4 + p;
                        const T* src = &Cb[m0 + (tid2 & 31) + static_cast<std::ptrdiff_t>(col) * ldc];
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i)
                            prior[q][i] = !bz && col < n && 32 * i < rows_left ? src[32 * i] : T(0);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
BATCHLAS_UNROLL_FULL
                for (int q = 0; q < 4; ++q) {
                    const int slot = (tid2 >> 5) * 4 + q;
                    const int col = n0 + slot * 4 + p;
                    if constexpr (AlignedFastPath) {
                        const int row = (tid2 & 31) * 4;
                        T* dst = &Cb[m0 + row + static_cast<std::ptrdiff_t>(col) * ldc];
                        const Vec4<T> v = vec4_ref(&sc[slot * StageStride + row]);
                        Vec4<T> out;
                        if (beta == T(0)) {
                            out = alpha * v;
                        } else {
                            const Vec4<T> prior = vec4_ref(const_cast<const T*>(dst));
BATCHLAS_UNROLL_FULL
                            for (int i = 0; i < 4; ++i) {
                                out[i] = LinearEpilogue<T>::apply(alpha, beta, v[i], prior[i]);
                            }
                        }
                        vec4_ref(dst) = out;
                    } else {
                        // A warp store is a whole 128-byte piece of the column at any ldc.
                        if (col >= n) continue;
                        const int rows_left = m - m0 - (tid2 & 31);
                        T* dst = &Cb[m0 + (tid2 & 31) + static_cast<std::ptrdiff_t>(col) * ldc];
                        const T* src = &sc[slot * StageStride + (tid2 & 31)];
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i)
                            if (32 * i < rows_left)  // beta = 0 is alpha * acc alone: -0 + 0 would be +0
                                dst[32 * i] = beta == T(0) ? alpha * src[32 * i]
                                                           : LinearEpilogue<T>::apply(alpha, beta, src[32 * i],
                                                                                      prior[q][i]);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            return;
        }

        // Within a band the four rows are consecutive in m, the contiguous
        // direction of a column-major C, so a band is one 128-bit access.
BATCHLAS_UNROLL_FULL
        for (int band = 0; band < 2; ++band) {
            const int gm = m0 + band * BandM + mb;
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < ThreadTile; ++j) {
                const int gn = n0 + (j < Band ? nb + j : BandN + nb + j - Band);
                if constexpr (AlignedFastPath) {
                    T* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                    Vec4<T> out;
                    if (beta == T(0)) {
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) {
                            out[i] = alpha * accum[band * Band + i][j];
                        }
                    } else {
                        const Vec4<T> prior = vec4_ref(const_cast<const T*>(p));
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) {
                            out[i] = LinearEpilogue<T>::apply(alpha, beta, accum[band * Band + i][j], prior[i]);
                        }
                    }
                    vec4_ref(p) = out;
                } else {
                    if (gn >= n) {
                        continue;
                    }
BATCHLAS_UNROLL_FULL
                    for (int i = 0; i < 4; ++i) {
                        const int row = gm + i;
                        if (row >= m) {
                            continue;
                        }
                        T* p = &Cb[row + static_cast<std::ptrdiff_t>(gn) * ldc];
                        *p = LinearEpilogue<T>::apply(alpha, beta, accum[band * Band + i][j],
                                                      beta == T(0) ? T(0) : *p);
                    }
                }
            }
        }
    }
};

// The predicated leg's two loop copies push the body past the inliner's threshold, and out of
// line it spilled behind the call. Forcing it on Body's own operator() changed the aligned leg's
// code too, so only the predicated leg launches through this wrapper.
template <typename Body>
struct Gemm128x128Inlined {
    Body body;

    BATCHLAS_GEMM_REQD_WG(Body::Threads, Body::kMinGroupsPerCu)
    __attribute__((always_inline)) void operator()(sycl::nd_item<3> item) const {
        [[clang::always_inline]] body(item);
    }
};

template <typename Body>
struct Gemm128x128Pipelined {
    Body body;

    BATCHLAS_GEMM_REQD_WG(Body::Threads, Body::kMinGroupsPerCu)
    __attribute__((always_inline)) void operator()(sycl::nd_item<3> item) const {
        [[clang::always_inline]] body(item);
    }
};

template <typename T, bool AlignedFastPath, bool StagedEpilogue, Transpose OpA, Transpose OpB, int Stages>
Event launch_register_128x128_k8_leg(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    using Body = Gemm128x128Body<T, AlignedFastPath, StagedEpilogue, OpA, OpB, Stages>;
    BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(Body::kVariant));

    const auto [m, k] = get_effective_dims(A, OpA);
    const int n = get_effective_dims(B, OpB).second;
    const int group_rows = (m + Body::TileM - 1) / Body::TileM;
    const int group_cols = (n + Body::TileN - 1) / Body::TileN;

    const sycl::range<3> local(1, 1, Body::Threads);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()), static_cast<size_t>(group_rows),
                                static_cast<size_t>(group_cols) * Body::Threads);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Body::TileAElems), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(Body::TileBElems), h);
        Body body{A.data_ptr(), B.data_ptr(), C.data_ptr(), m, n, k, A.ld(), B.ld(), C.ld(),
                  A.stride(), B.stride(), C.stride(), A.batch_size(), alpha, beta, tile_a, tile_b};
        using Name = GemmRegister128x128Kernel<T, AlignedFastPath, StagedEpilogue, OpA, OpB, Stages>;
        // Out of line, the copy_async asm's memory clobber reloads every member through `this` (a
        // stack copy of the functor) after each copy: 23-33x slower. Stages > 2 always inlines.
        // evidence: docs/perf/gemm.md#gemm-wp-n2-the-u4-stack-frame-and-zero-filled-predicated-copies
        if constexpr (Stages > 2) {
            batchlas::impl::parallel_for<Name>(h, sycl::nd_range<3>(global, local), Gemm128x128Pipelined<Body>{body});
        } else if constexpr (AlignedFastPath) {
            batchlas::impl::parallel_for<Name>(h, sycl::nd_range<3>(global, local), body);
        } else {
            batchlas::impl::parallel_for<Name>(h, sycl::nd_range<3>(global, local), Gemm128x128Inlined<Body>{body});
        }
    });

    return ctx.get_event();
}

// The aligned leg: short k is store-bound, and there the direct epilogue's four 128-byte column
// pieces per store run DRAM measurably slower than whole columns; at long k the staging round trip
// costs more than it saves. The predicated leg stages at every k: its direct epilogue's scalar
// stores touched 3.7x the sectors of a staged one. An epilogue choice, never a routing term.
// evidence: docs/perf/gemm.md#the-staged-epilogue
// evidence: docs/perf/gemm.md#gemm-wp-f-float-mid-size-parity-on-sm_120
template <typename T, bool AlignedFastPath = false, Transpose OpA = Transpose::NoTrans,
          Transpose OpB = Transpose::NoTrans, int Stages = 2>
Event launch_register_128x128_k8(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    if constexpr (!AlignedFastPath) {
        return launch_register_128x128_k8_leg<T, false, true, OpA, OpB, Stages>(ctx, A, B, C, alpha, beta,
                                                                                kernel_trace_name);
    } else {
        if (get_effective_dims(A, OpA).second <= kStagedEpilogueMaxK)
            return launch_register_128x128_k8_leg<T, true, true, OpA, OpB, Stages>(ctx, A, B, C, alpha, beta,
                                                                                   kernel_trace_name);
        return launch_register_128x128_k8_leg<T, true, false, OpA, OpB, Stages>(ctx, A, B, C, alpha, beta,
                                                                                kernel_trace_name);
    }
}

} // namespace batchlas::sycl_gemm

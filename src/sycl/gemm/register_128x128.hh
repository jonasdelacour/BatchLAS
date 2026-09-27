#pragma once

// A 128x128x8 register-tiled GEMM with 8x8 accumulators per thread.
//
// Shallow K (8) with M = N = 128 buys an 8x8 thread tile, so a k-step is 4
// vectorized shared loads for 64 FFMAs. The shared tiles use stride exactly
// TileM / TileN (an odd stride defeats the 16-byte alignment the LDS.128 form
// needs), and B is staged [k][n] so a thread's 8 B values are contiguous.
//
// Four things beyond that are load-bearing, each measured:
//
//   * Register prefetch into a double-buffered shared tile, one barrier per
//     slab. The prefetch registers are clang ext_vector values, NOT a struct:
//     a struct copy global -> shared is folded into a memcpy that LLVM sinks
//     to the store, which silently turns the prefetch back into a stall.
//   * The lane swizzle. An LDS.128 costs 4 shared wavefronts when its address
//     depends on BOTH lane bits 0 and 1, and 2 otherwise, so m takes lane bit
//     0 and n takes lane bit 1. Each warp owns a 64 (m) x 32 (n) sub-tile.
//   * An L2::128B fetch hint on the B loads. A thread reads 32 bytes of each
//     B column per slab; without the hint DRAM sees 32-byte requests strided
//     by ldb, and the memory-bound shapes run at 0.8x of the vendor.
//   * The launch bound (two work-groups per CU, a 128-register cap) is a
//     guard: the kernel compiles to 127 today, and one register more drops
//     it to one group per SM. Three per CU spills and runs ~9x slower.
//
// evidence: docs/perf/gemm.md#the-128x128-float-kernel

#include "accessors.hh"
#include "epilogue_linear.hh"

#include "../gemm_kernels.hh"

#include "../../linalg-impl.hh"

#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

template <typename T, bool AlignedFastPath, bool StagedEpilogue>
class GemmRegister128x128Kernel;

template <typename T>
using Vec4 = T __attribute__((ext_vector_type(4)));

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
#endif
    return *p;
}

// Does this problem satisfy everything the unpredicated path assumes?
template <typename T>
inline bool can_use_128x128_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      const MatrixView<T, MatrixFormat::Dense>& C) {
    constexpr int TileM = 128, TileN = 128, TileK = 8;
    const auto m = A.rows();
    const auto k = A.cols();
    const auto n = B.cols();
    // k == 0 would pass the modulus test, and the prefetch of slab 0 runs before the loop.
    if (k < TileK || (m % TileM) != 0 || (n % TileN) != 0 || (k % TileK) != 0) {
        return false;
    }
    // Every 128-bit access this kernel makes is at a multiple of 4 elements
    // from the base pointer, so the base itself must be 4-element aligned and
    // the leading dimensions must preserve that.
    auto aligned = [](const T* p, int ld, int stride) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % (4 * sizeof(T))) == 0 &&
            (ld % 4) == 0 && (stride % 4) == 0;
    };
    return aligned(A.data_ptr(), A.ld(), A.stride()) &&
        aligned(B.data_ptr(), B.ld(), B.stride()) &&
        aligned(C.data_ptr(), C.ld(), C.stride());
}

// A FUNCTOR, not a lambda, only so the launch bound can be spelled.
// StagedEpilogue (aligned leg only) routes C through local memory so each
// warp stores whole 128-row columns; see launch_register_128x128_k8.
template <typename T, bool AlignedFastPath, bool StagedEpilogue>
struct Gemm128x128Body {
    static constexpr int TileM = 128, TileN = 128, TileK = 8;
    static constexpr int ThreadTile = 8, Band = 4, Threads = 256;
    static constexpr int BandM = 32, BandN = 16;  // second band's offset
    static constexpr int AStride = TileM, BStride = TileN;
    static constexpr int kMinGroupsPerCu = 2;
    static constexpr int StageStride = TileM + 4;  // staged C column stride
    static constexpr int TileAElems = StagedEpilogue ? 32 * StageStride : 2 * TileK * AStride;
    static_assert(!StagedEpilogue || AlignedFastPath, "the staged epilogue stores unpredicated");

    const T* a_ptr;
    const T* b_ptr;
    T* c_ptr;
    int m, n, k, lda, ldb, ldc;
    std::ptrdiff_t stride_a, stride_b, stride_c;
    int batch;
    T alpha, beta;
    sycl::local_accessor<T, 1> tile_a, tile_b;

    [[intel::max_work_group_size(1, 1, Threads), intel::min_work_groups_per_cu(kMinGroupsPerCu)]]
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

        T accum[ThreadTile][ThreadTile];
#pragma unroll
        for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
            for (int j = 0; j < ThreadTile; ++j) {
                accum[i][j] = T(0);
            }
        }

        // A is read down m, B down k and transposed into shared, both
        // coalesced; one packet of each per thread per slab.
        const int a_m = (tid % 32) * 4;
        const int a_k = tid / 32;
        const int b_k = (tid % 2) * 4;
        const int b_n = tid / 2;
        const T* ga = Ab + (m0 + a_m) + static_cast<std::ptrdiff_t>(a_k) * lda;
        const T* gb = Bb + b_k + static_cast<std::ptrdiff_t>(n0 + b_n) * ldb;

        Vec4<T> ra, rb;
        auto gload = [&](int k0) __attribute__((always_inline)) {
            if constexpr (AlignedFastPath) {
                ra = vec4_ref(ga + static_cast<std::ptrdiff_t>(k0) * lda);
                rb = load4_l2_line(gb + k0);
            } else {
                // Zero outside the matrix, so the inner loop needs no bounds.
                const int gk_a = k0 + a_k;
                const int gn_b = n0 + b_n;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int gm = m0 + a_m + i;
                    ra[i] = (gm < m && gk_a < k) ? ga[static_cast<std::ptrdiff_t>(k0) * lda + i] : T(0);
                    rb[i] = (k0 + b_k + i < k && gn_b < n) ? load1_l2_line(gb + k0 + i) : T(0);
                }
            }
        };
        auto sstore = [&](int buf) __attribute__((always_inline)) {
            vec4_ref(&sa[buf * TileK * AStride + a_k * AStride + a_m]) = ra;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sb[buf * TileK * BStride + (b_k + i) * BStride + b_n] = rb[i];
            }
        };
        auto compute = [&](int buf) __attribute__((always_inline)) {
            const T* xa = sa + buf * TileK * AStride;
            const T* xb = sb + buf * TileK * BStride;
#pragma unroll
            for (int kk = 0; kk < TileK; ++kk) {
                const Vec4<T> a0 = vec4_ref(&xa[kk * AStride + mb]);
                const Vec4<T> a1 = vec4_ref(&xa[kk * AStride + BandM + mb]);
                const Vec4<T> b0 = vec4_ref(&xb[kk * BStride + nb]);
                const Vec4<T> b1 = vec4_ref(&xb[kk * BStride + BandN + nb]);
                const T af[ThreadTile] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const T bf[ThreadTile] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
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
        gload(0);
        sstore(0);
        item.barrier(sycl::access::fence_space::local_space);
        int s = 0;
        for (; s + 1 < slabs; s += 2) {
            gload((s + 1) * TileK);
            compute(0);
            sstore(1);
            item.barrier(sycl::access::fence_space::local_space);
            const bool more = s + 2 < slabs;
            if (more) gload((s + 2) * TileK);
            compute(1);
            if (more) sstore(0);
            item.barrier(sycl::access::fence_space::local_space);
        }
        if (s < slabs) {
            compute(0);  // an odd slab count leaves the last slab in buffer 0
        }

        if constexpr (StagedEpilogue) {
            // Pass p stages the 32 tile columns c with c % 4 == p, slot c / 4,
            // then each warp stores 4 of them as whole 128-row columns.
            T* sc = sa;
            const int tid2 = static_cast<int>(item.get_local_id(2));
            item.barrier(sycl::access::fence_space::local_space);  // the odd tail read sa
#pragma unroll
            for (int p = 0; p < Band; ++p) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int slot = (nb >> 2) + half * (BandN / 4);
#pragma unroll
                    for (int band = 0; band < 2; ++band) {
                        Vec4<T> v;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            v[i] = accum[band * Band + i][p + half * Band];
                        }
                        vec4_ref(&sc[slot * StageStride + band * BandM + mb]) = v;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int slot = (tid2 >> 5) * 4 + q;
                    const int row = (tid2 & 31) * 4;
                    T* dst = &Cb[m0 + row + static_cast<std::ptrdiff_t>(n0 + slot * 4 + p) * ldc];
                    const Vec4<T> v = vec4_ref(&sc[slot * StageStride + row]);
                    Vec4<T> out;
                    if (beta == T(0)) {
                        out = alpha * v;
                    } else {
                        const Vec4<T> prior = vec4_ref(const_cast<const T*>(dst));
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            out[i] = LinearEpilogue<T>::apply(alpha, beta, v[i], prior[i]);
                        }
                    }
                    vec4_ref(dst) = out;
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            return;
        }

        // Within a band the four rows are consecutive in m, the contiguous
        // direction of a column-major C, so a band is one 128-bit access.
#pragma unroll
        for (int band = 0; band < 2; ++band) {
            const int gm = m0 + band * BandM + mb;
#pragma unroll
            for (int j = 0; j < ThreadTile; ++j) {
                const int gn = n0 + (j < Band ? nb + j : BandN + nb + j - Band);
                if constexpr (AlignedFastPath) {
                    T* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                    Vec4<T> out;
                    if (beta == T(0)) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            out[i] = alpha * accum[band * Band + i][j];
                        }
                    } else {
                        const Vec4<T> prior = vec4_ref(const_cast<const T*>(p));
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            out[i] = LinearEpilogue<T>::apply(alpha, beta, accum[band * Band + i][j], prior[i]);
                        }
                    }
                    vec4_ref(p) = out;
                } else {
                    if (gn >= n) {
                        continue;
                    }
#pragma unroll
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

template <typename T, bool AlignedFastPath, bool StagedEpilogue>
Event launch_register_128x128_k8_leg(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(KernelVariant::Tiled128x128RegisterK8));
    using Body = Gemm128x128Body<T, AlignedFastPath, StagedEpilogue>;

    const int m = static_cast<int>(A.rows());
    const int k = static_cast<int>(A.cols());
    const int n = static_cast<int>(B.cols());
    const int group_rows = (m + Body::TileM - 1) / Body::TileM;
    const int group_cols = (n + Body::TileN - 1) / Body::TileN;

    const sycl::range<3> local(1, 1, Body::Threads);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()), static_cast<size_t>(group_rows),
                                static_cast<size_t>(group_cols) * Body::Threads);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Body::TileAElems), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(2 * Body::TileK * Body::BStride), h);
        Body body{A.data_ptr(), B.data_ptr(), C.data_ptr(), m, n, k, A.ld(), B.ld(), C.ld(),
                  A.stride(), B.stride(), C.stride(), A.batch_size(), alpha, beta, tile_a, tile_b};
        h.parallel_for<GemmRegister128x128Kernel<T, AlignedFastPath, StagedEpilogue>>(
            sycl::nd_range<3>(global, local), body);
    });

    return ctx.get_event();
}

// Short k is store-bound, and there the direct epilogue's four 128-byte
// column pieces per store run DRAM measurably slower than whole columns; at
// long k the staging round trip costs more than it saves.
// evidence: docs/perf/gemm.md#the-staged-epilogue
template <typename T, bool AlignedFastPath = false>
Event launch_register_128x128_k8(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    if constexpr (AlignedFastPath) {
        if (A.cols() <= kStagedEpilogueMaxK) {
            return launch_register_128x128_k8_leg<T, true, true>(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
    }
    return launch_register_128x128_k8_leg<T, AlignedFastPath, false>(ctx, A, B, C, alpha, beta,
                                                                     kernel_trace_name);
}

} // namespace batchlas::sycl_gemm

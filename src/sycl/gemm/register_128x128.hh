#pragma once

// A 128x128x8 register-tiled GEMM with 8x8 accumulators per thread.
//
// This is a different shape from the rest of the register-tiled family, and
// the difference is deliberate. Every other kernel in this directory keeps a
// deep K (16 or 32) and a narrow N (32 or 64), which caps the thread tile at
// 4x4 or 8x4 -- 16 or 32 accumulators. That makes the inner loop issue 8 to 12
// shared-memory loads for every 16 to 32 FFMAs, a ratio of 2.0-2.7:1, and the
// shared-memory pipe becomes the limit long before the FP32 pipe does.
//
// Here K is shallow (8) and both M and N are 128, which buys a 8x8 thread tile
// with 64 accumulators. Per k-step a thread issues 4 vectorized shared loads
// and 64 FFMAs -- a ratio of 16:1, six times better, and enough ILP that the
// kernel does not need double buffering to hide shared-memory latency.
//
// Two layout details make the 16:1 real rather than nominal, and both are
// things the older family gets wrong:
//
//   * The shared tiles are stored with stride exactly TileM / TileN, not
//     TileM+1 / TileK+1. An odd stride means the compiler can never prove
//     16-byte alignment, so every fragment load degrades to scalar
//     ld.shared.b32. With an aligned stride they become ld.shared.v4.
//   * B is staged as [k][n], not [n][k], so a thread's 8 B values are
//     contiguous and vectorize. In the [n][k] layout they stride by TileK+1
//     and cannot.
//
// The thread's 8 rows and 8 columns are split into two 4-wide bands
// (ty*4 and 64+ty*4). That split is what keeps the vectorized loads
// bank-conflict free: an LDS.128 is serviced 8 lanes at a time, and 8 lanes x
// 4 floats is exactly the 32 banks.
//
// Measured on RTX 4090 / sm_89 at 512^3 batch 512: 43.6 TFLOP/s, against
// cuBLAS SGEMM's 43.9 under the same timing, where the 128x64x32 kernel that
// this supersedes reaches 21. See docs/perf/gemm.md#the-128x128-float-kernel.

#include "accessors.hh"
#include "epilogue_linear.hh"

#include "../gemm_kernels.hh"

#include "../../linalg-impl.hh"

#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

template <typename T, bool AlignedFastPath>
class GemmRegister128x128Kernel;

// A vector of four T with the alignment the 128-bit load/store forms need.
// Using a plain struct rather than sycl::vec keeps the generated code the same
// as the equivalent CUDA float4 access.
template <typename T>
struct alignas(4 * sizeof(T)) Packet4 {
    T v[4];
};

template <typename T>
inline const Packet4<T>& packet4_ref(const T* p) {
    return *reinterpret_cast<const Packet4<T>*>(p);
}

template <typename T>
inline Packet4<T>& packet4_ref(T* p) {
    return *reinterpret_cast<Packet4<T>*>(p);
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
    if ((m % TileM) != 0 || (n % TileN) != 0 || (k % TileK) != 0) {
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

template <typename T, bool AlignedFastPath = false>
Event launch_register_128x128_k8(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(KernelVariant::Tiled128x128RegisterK8));

    constexpr int TileM = 128;
    constexpr int TileN = 128;
    constexpr int TileK = 8;
    constexpr int ThreadTile = 8;          // rows and cols per thread
    constexpr int Band = ThreadTile / 2;   // 4: the vectorized band width
    constexpr int LocalRows = TileM / ThreadTile;  // 16
    constexpr int LocalCols = TileN / ThreadTile;  // 16
    constexpr int Threads = LocalRows * LocalCols; // 256

    // No padding: an aligned stride is what lets the fragment loads vectorize.
    // A is staged [k][m] and B as [k][n], so both fragments are contiguous.
    constexpr int AStride = TileM;
    constexpr int BStride = TileN;

    static_assert(TileM * TileK == 4 * Threads, "A staging assumes one packet per thread");
    static_assert(TileK * TileN == 4 * Threads, "B staging assumes one packet per thread");

    const int m = static_cast<int>(A.rows());
    const int k = static_cast<int>(A.cols());
    const int n = static_cast<int>(B.cols());

    const int group_rows = (m + TileM - 1) / TileM;
    const int group_cols = (n + TileN - 1) / TileN;

    const sycl::range<3> local(1, LocalRows, LocalCols);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(group_rows * LocalRows),
                                static_cast<size_t>(group_cols * LocalCols));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(TileK * AStride), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(TileK * BStride), h);

        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();

        h.parallel_for<GemmRegister128x128Kernel<T, AlignedFastPath>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) {
                    return;
                }
                // The m index is the fastest-varying one. That is what keeps
                // the epilogue coalesced: C is column-major, so lanes that
                // differ in m touch consecutive addresses, while lanes that
                // differ in n would stride by ldc. Getting this backwards
                // costs nothing when beta == 0 (stores are fire-and-forget)
                // but is catastrophic when beta != 0, because the read of C
                // then issues one scattered transaction per lane.
                const int ty = static_cast<int>(item.get_local_id(2));  // 0..15, m
                const int tx = static_cast<int>(item.get_local_id(1));  // 0..15, n
                const int tid = tx * LocalRows + ty;

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

                // Staging coordinates. A is read down m (the contiguous
                // direction of a column-major A) so the warp is coalesced; B is
                // read down k and transposed into shared so that the n
                // direction ends up contiguous.
                const int a_m = (tid % 32) * 4;
                const int a_k = tid / 32;
                const int b_k = (tid % 2) * 4;
                const int b_n = tid / 2;

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    if constexpr (AlignedFastPath) {
                        packet4_ref(&sa[a_k * AStride + a_m]) =
                            packet4_ref(Ab + (m0 + a_m) +
                                        static_cast<std::ptrdiff_t>(k0 + a_k) * lda);
                        const Packet4<T> vb =
                            packet4_ref(Bb + (k0 + b_k) +
                                        static_cast<std::ptrdiff_t>(n0 + b_n) * ldb);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            sb[(b_k + i) * BStride + b_n] = vb.v[i];
                        }
                    } else {
                        // Predicated staging. The shared tile is always filled
                        // to its full 128x8, with zeros outside the matrix, so
                        // the inner loop below needs no bounds checks at all.
                        const int gk_a = k0 + a_k;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int gm = m0 + a_m + i;
                            sa[a_k * AStride + a_m + i] =
                                (gm < m && gk_a < k)
                                ? Ab[gm + static_cast<std::ptrdiff_t>(gk_a) * lda]
                                : T(0);
                        }
                        const int gn_b = n0 + b_n;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int gk = k0 + b_k + i;
                            sb[(b_k + i) * BStride + b_n] =
                                (gk < k && gn_b < n)
                                ? Bb[gk + static_cast<std::ptrdiff_t>(gn_b) * ldb]
                                : T(0);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
                        const Packet4<T> a0 = packet4_ref(&sa[kk * AStride + ty * Band]);
                        const Packet4<T> a1 = packet4_ref(&sa[kk * AStride + 64 + ty * Band]);
                        const Packet4<T> b0 = packet4_ref(&sb[kk * BStride + tx * Band]);
                        const Packet4<T> b1 = packet4_ref(&sb[kk * BStride + 64 + tx * Band]);
                        const T af[ThreadTile] = {a0.v[0], a0.v[1], a0.v[2], a0.v[3],
                                                  a1.v[0], a1.v[1], a1.v[2], a1.v[3]};
                        const T bf[ThreadTile] = {b0.v[0], b0.v[1], b0.v[2], b0.v[3],
                                                  b1.v[0], b1.v[1], b1.v[2], b1.v[3]};
#pragma unroll
                        for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
                            for (int j = 0; j < ThreadTile; ++j) {
                                accum[i][j] += af[i] * bf[j];
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Epilogue. Within a band the four rows are consecutive in m,
                // which is the contiguous direction of a column-major C, so a
                // whole band is one 128-bit store.
#pragma unroll
                for (int band = 0; band < 2; ++band) {
                    const int gm = m0 + band * 64 + ty * Band;
#pragma unroll
                    for (int j = 0; j < ThreadTile; ++j) {
                        const int gn = n0 + (j < Band ? tx * Band + j : 64 + tx * Band + j - Band);
                        if constexpr (AlignedFastPath) {
                            T* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                            Packet4<T> out;
                            if (beta == T(0)) {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    out.v[i] = alpha * accum[band * Band + i][j];
                                }
                            } else {
                                // One 128-bit read rather than four scalar
                                // ones; the four rows are contiguous in m.
                                const Packet4<T> prior = packet4_ref(const_cast<const T*>(p));
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    out.v[i] = LinearEpilogue<T>::apply(
                                        alpha, beta, accum[band * Band + i][j], prior.v[i]);
                                }
                            }
                            packet4_ref(p) = out;
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
                                *p = LinearEpilogue<T>::apply(alpha, beta,
                                                              accum[band * Band + i][j],
                                                              beta == T(0) ? T(0) : *p);
                            }
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

} // namespace batchlas::sycl_gemm

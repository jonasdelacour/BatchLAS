#pragma once

// Batched GEMM for orders <= 32: several matrices per work-group, op(B) staged in local
// memory, each lane holding one row of op(A) in registers and a column strip of C.
// Lanes run DOWN the rows, so the A loads and the C stores are coalesced; Direct runs
// them across columns, ld apart. evidence: docs/perf/gemm.md#the-small-batched-kernel

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <type_traits>

namespace batchlas::sycl_gemm_small {

template <typename T, int NB>
class GemmSmallBatchedKernel;

inline constexpr int kSmallWg = 128;

// The largest order the kernel serves; max(m, n, k) picks the bucket.
inline constexpr int kSmallMaxDim = 64;

inline int small_bucket(int max_dim) {
    return max_dim <= 8 ? 8 : max_dim <= 16 ? 16 : max_dim <= 32 ? 32 : 64;
}

template <typename T, int NB>
Event launch_small_batched(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha, T beta, Transpose transA, Transpose transB,
                           int m, int n, int k) {
    static_assert(!std::is_class_v<T>, "real scalars only: std::complex's operator* is Annex G");
    constexpr int kTpm = 2 * NB;          // lanes per matrix: NB rows x 2 column halves
    constexpr int kCpt = NB / 2;          // columns of C per lane
    constexpr int kG = kSmallWg / kTpm;   // matrices per work-group
    constexpr int kLdb = NB + 1;          // odd: a transposed stage spreads over the banks
    static_assert(kG >= 1 && kG * kTpm == kSmallWg, "the work-group must hold whole matrices");

    const int batch = static_cast<int>(A.batch_size());
    const int num_wg = (batch + kG - 1) / kG;

    const T* a_ptr = A.data_ptr();
    const T* b_ptr = B.data_ptr();
    T* c_ptr = C.data_ptr();
    const std::ptrdiff_t lda = A.ld(), ldb = B.ld(), ldc = C.ld();
    const std::ptrdiff_t sa = A.stride(), sb = B.stride(), sc = C.stride();
    const bool ta = (transA != Transpose::NoTrans);
    const bool tb = (transB != Transpose::NoTrans);
    const bool beta_zero = (beta == T(0));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> slm(sycl::range<1>(static_cast<std::size_t>(kG) * NB * kLdb), h);
        h.parallel_for<GemmSmallBatchedKernel<T, NB>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) * kSmallWg),
                              sycl::range<1>(kSmallWg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int t = static_cast<int>(it.get_local_id(0));
                const int slot = t / kTpm;
                const int tt = t % kTpm;
                const int mat = static_cast<int>(it.get_group(0)) * kG + slot;
                // Clamped, never returned: every lane must reach the barrier below.
                const bool live = mat < batch;
                const std::ptrdiff_t bi = live ? mat : 0;
                const int r = tt % NB;
                const int c0 = (tt / NB) * kCpt;
                T* const sB = &slm[0] + static_cast<std::ptrdiff_t>(slot) * NB * kLdb;
                const T* const Ab = a_ptr + bi * sa;
                const T* const Bb = b_ptr + bi * sb;

                // op(B) into sB[l + c * kLdb], walked in STORAGE order so the global
                // read is coalesced either way; zero outside k x n.
                for (int e = tt; e < NB * NB; e += kTpm) {
                    const int s0 = e % NB, s1 = e / NB;
                    const int l = tb ? s1 : s0;
                    const int c = tb ? s0 : s1;
                    T v = T(0);
                    if (live && l < k && c < n) v = Bb[s0 + s1 * ldb];
                    sB[l + c * kLdb] = v;
                }

                // Row r of op(A), unrolled so the array stays in registers.
                T a[NB];
#pragma unroll
                for (int l = 0; l < NB; ++l) {
                    T v = T(0);
                    if (live && r < m && l < k) v = ta ? Ab[l + r * lda] : Ab[r + l * lda];
                    a[l] = v;
                }
                it.barrier(sycl::access::fence_space::local_space);

                T acc[kCpt];
#pragma unroll
                for (int j = 0; j < kCpt; ++j) acc[j] = T(0);
#pragma unroll
                for (int l = 0; l < NB; ++l) {
                    if (l >= k) continue;   // uniform; `continue` keeps the unroll, so a[] stays put
#pragma unroll
                    for (int j = 0; j < kCpt; ++j) acc[j] += a[l] * sB[l + (c0 + j) * kLdb];
                }

                if (live && r < m) {
                    T* const Cb = c_ptr + bi * sc;
#pragma unroll
                    for (int j = 0; j < kCpt; ++j) {
                        const int c = c0 + j;
                        if (c < n) {
                            T* const dst = Cb + r + c * ldc;
                            *dst = beta_zero ? alpha * acc[j] : alpha * acc[j] + beta * *dst;
                        }
                    }
                }
            });
    });
    return ctx.get_event();
}

template <typename T, int NB, bool PrefetchC>
class GemmSmallTiledKernel;

// NN, 32 < max(m, n, k) <= kSmallTiledMaxDim: one matrix per work-group, A
// ([k][m]) and B (as stored) both staged in local memory, and a 4x4 register
// tile of C per lane, so an FMA costs 1/8 of a shared load instead of one.
// With beta != 0 the lane reads its C tile BEFORE the barrier so that latency
// overlaps the staging; at beta == 0 that prefetch is compiled out, because
// its registers cost more than it saves. evidence: docs/perf/gemm.md#the-small-tiled-kernel
inline constexpr int kSmallTiledMaxDim = 56;

template <typename T, int NB, bool PrefetchC>
Event launch_small_tiled(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         T alpha, T beta, int m, int n, int k) {
    using V4 = T __attribute__((ext_vector_type(4)));
    constexpr int kTr = NB / 4;          // lanes per row and per column of 4x4 tiles
    constexpr int kWg = kTr * kTr;
    constexpr int kLd = NB + 4;          // keeps the V4 reads 16-byte aligned
    static_assert(NB % 4 == 0, "4x4 lane tiles");

    const int batch = static_cast<int>(A.batch_size());
    const T* a_ptr = A.data_ptr();
    const T* b_ptr = B.data_ptr();
    T* c_ptr = C.data_ptr();
    const std::ptrdiff_t lda = A.ld(), ldb = B.ld(), ldc = C.ld();
    const std::ptrdiff_t sa = A.stride(), sb = B.stride(), sc = C.stride();
    const bool beta_zero = (beta == T(0));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> slm(sycl::range<1>(static_cast<std::size_t>(2 * NB * kLd)), h);
        h.parallel_for<GemmSmallTiledKernel<T, NB, PrefetchC>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * kWg),
                              sycl::range<1>(kWg)),
            [=](sycl::nd_item<1> it) {
                const int t = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t bi = static_cast<std::ptrdiff_t>(it.get_group(0));
                const int kp = (k + 3) & ~3;
                T* const sA = &slm[0];           // sA[l * kLd + r]
                T* const sB = sA + NB * kLd;     // sB[c * kLd + l]
                const T* const Ab = a_ptr + bi * sa;
                const T* const Bb = b_ptr + bi * sb;
                T* const Cb = c_ptr + bi * sc;
                // Storage order, so both reads are coalesced; zero-padded.
                for (int e = t; e < NB * kp; e += kWg) {
                    const int r = e % NB, l = e / NB;
                    sA[l * kLd + r] = (r < m && l < k) ? Ab[r + l * lda] : T(0);
                }
                for (int e = t; e < kp * NB; e += kWg) {
                    const int l = e % kp, c = e / kp;
                    sB[c * kLd + l] = (l < k && c < n) ? Bb[l + c * ldb] : T(0);
                }
                const int tr = t % kTr, tc = t / kTr;
                T prior[4][4];
                if constexpr (PrefetchC) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int r = tr * 4 + i, c = tc * 4 + j;
                            prior[i][j] = (r < m && c < n) ? Cb[r + c * ldc] : T(0);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                T acc[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
                }
                for (int l0 = 0; l0 < kp; l0 += 4) {
                    V4 bv[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        bv[j] = *reinterpret_cast<const V4*>(&sB[(tc * 4 + j) * kLd + l0]);
                    }
#pragma unroll
                    for (int ll = 0; ll < 4; ++ll) {
                        const V4 av = *reinterpret_cast<const V4*>(&sA[(l0 + ll) * kLd + tr * 4]);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
#pragma unroll
                            for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j][ll];
                        }
                    }
                }

#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = tc * 4 + j;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int r = tr * 4 + i;
                        if (r < m && c < n) {
                            T* const dst = Cb + r + c * ldc;
                            if constexpr (PrefetchC) {
                                *dst = alpha * acc[i][j] + beta * prior[i][j];
                            } else {
                                *dst = beta_zero ? alpha * acc[i][j] : alpha * acc[i][j] + beta * *dst;
                            }
                        }
                    }
                }
            });
    });
    return ctx.get_event();
}

template <typename T, int NB>
Event small_tiled(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha, T beta, int m, int n, int k) {
    if (beta == T(0)) {
        return launch_small_tiled<T, NB, false>(ctx, A, B, C, alpha, beta, m, n, k);
    }
    return launch_small_tiled<T, NB, true>(ctx, A, B, C, alpha, beta, m, n, k);
}

template <typename T>
Event small_batched(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    const MatrixView<T, MatrixFormat::Dense>& B,
                    const MatrixView<T, MatrixFormat::Dense>& C,
                    T alpha, T beta, Transpose transA, Transpose transB,
                    int m, int n, int k) {
    const int max_dim = m > n ? (m > k ? m : k) : (n > k ? n : k);
    if constexpr (std::is_same_v<T, float>) {
        if (transA == Transpose::NoTrans && transB == Transpose::NoTrans && max_dim > 32 &&
            max_dim <= kSmallTiledMaxDim) {
            return max_dim <= 48 ? small_tiled<T, 48>(ctx, A, B, C, alpha, beta, m, n, k)
                                 : small_tiled<T, 56>(ctx, A, B, C, alpha, beta, m, n, k);
        }
    }
    switch (small_bucket(max_dim)) {
    case 8:
        return launch_small_batched<T, 8>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k);
    case 16:
        return launch_small_batched<T, 16>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k);
    case 32:
        return launch_small_batched<T, 32>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k);
    default:
        return launch_small_batched<T, 64>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k);
    }
}

}  // namespace batchlas::sycl_gemm_small

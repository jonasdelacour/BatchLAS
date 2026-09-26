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

template <typename T>
Event small_batched(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    const MatrixView<T, MatrixFormat::Dense>& B,
                    const MatrixView<T, MatrixFormat::Dense>& C,
                    T alpha, T beta, Transpose transA, Transpose transB,
                    int m, int n, int k) {
    const int max_dim = m > n ? (m > k ? m : k) : (n > k ? n : k);
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

#pragma once

// Batched GEMM for orders <= 64: several matrices per work-group, op(B) staged in local
// memory, each lane holding one row of op(A) in registers and a column strip of C.
// Lanes run DOWN the rows, so the A loads and the C stores are coalesced; Direct runs
// them across columns, ld apart. evidence: docs/perf/gemm.md#the-small-batched-kernel
// Complex runs on the POD Cx (fma_acc, dev_mul): std::complex's operator* is Annex G.
// evidence: docs/perf/gemm.md#gemm-complex-small-and-the-staged-transposed-a-row

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/ops/gemm/kernels/vec4.hh>

namespace batchlas::sycl_gemm_small {

template <typename T, int NB>
class GemmSmallBatchedKernel;

inline constexpr int kSmallWg = 128;

// The largest order the kernel serves; max(m, n, k) picks the bucket.
inline constexpr int kSmallMaxDim = 64;

// A 16-byte scalar stops at 32: its 64 bucket stages 64 x 65 x 16 B = 66.5 KB and holds a
// 64-entry row plus 32 accumulators, 384 registers. choice.hh's small_max_dim mirrors it.
template <typename T>
inline constexpr int kSmallMaxDimFor = sizeof(T) >= 16 ? 32 : kSmallMaxDim;

inline int small_bucket(int max_dim) {
    return max_dim <= 8 ? 8 : max_dim <= 16 ? 16 : max_dim <= 32 ? 32 : 64;
}

// Elements of local memory the batched kernel stages per work-group (choice.hh's small_slm_bytes).
template <int NB>
inline constexpr std::int64_t kSmallBatchedSlmElems = std::int64_t(kSmallWg / (2 * NB)) * NB * (NB + 1);

template <typename T, int NB>
Event launch_small_batched(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha, T beta, Transpose transA, Transpose transB,
                           int m, int n, int k) {
    using D = typename sycl_device::DevMap<T>::type;
    constexpr bool kCx = sycl_device::dev_is_complex_v<D>;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(NB <= kSmallMaxDimFor<T>, "no bucket above the scalar's ceiling");
    constexpr int kTpm = 2 * NB;          // lanes per matrix: NB rows x 2 column halves
    constexpr int kCpt = NB / 2;          // columns of C per lane
    constexpr int kG = kSmallWg / kTpm;   // matrices per work-group
    constexpr int kLdb = NB + 1;          // odd: a transposed stage spreads over the banks
    constexpr int kLc = kCx && NB == 64 ? 16 : NB;  // l per unrolled loop of the inner product
    static_assert(kLc == NB || kLc * 4 == NB, "one loop, or the four the inner product writes out");
    static_assert(kG >= 1 && kG * kTpm == kSmallWg, "the work-group must hold whole matrices");
    static_assert(std::int64_t(kG) * NB * kLdb == kSmallBatchedSlmElems<NB>, "choice.hh's bytes");

    const int batch = static_cast<int>(A.batch_size());
    const int num_wg = (batch + kG - 1) / kG;

    const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
    const D* b_ptr = reinterpret_cast<const D*>(B.data_ptr());
    D* c_ptr = reinterpret_cast<D*>(C.data_ptr());
    const std::ptrdiff_t lda = A.ld(), ldb = B.ld(), ldc = C.ld();
    const std::ptrdiff_t sa = A.stride(), sb = B.stride(), sc = C.stride();
    const bool ta = (transA != Transpose::NoTrans);
    const bool tb = (transB != Transpose::NoTrans);
    const bool ca = kCx && transA == Transpose::ConjTrans;
    const bool cb = kCx && transB == Transpose::ConjTrans;
    const bool beta_zero = (beta == T(0));
    D alpha_d, beta_d;
    __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
    __builtin_memcpy(&beta_d, &beta, sizeof(D));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(static_cast<std::size_t>(kG) * NB * kLdb), h);
        batchlas::impl::parallel_for<GemmSmallBatchedKernel<T, NB>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) * kSmallWg),
                              sycl::range<1>(kSmallWg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                using sycl_device::dev_conj;
                const int t = static_cast<int>(it.get_local_id(0));
                const int slot = t / kTpm;
                const int tt = t % kTpm;
                const int mat = static_cast<int>(it.get_group(0)) * kG + slot;
                // Clamped, never returned: every lane must reach the barriers below.
                const bool live = mat < batch;
                const std::ptrdiff_t bi = live ? mat : 0;
                const int r = tt % NB;
                const int c0 = (tt / NB) * kCpt;
                D* const sB = &slm[0] + static_cast<std::ptrdiff_t>(slot) * NB * kLdb;
                const D* const Ab = a_ptr + bi * sa;
                const D* const Bb = b_ptr + bi * sb;

                // Row r of op(A), unrolled so the array stays in registers. A transposed A is
                // stored k x m, so a lane's row is a stored column, lda from its neighbours':
                // it goes through sB in storage order (coalesced) as sB[r + l * kLdb], and the
                // second barrier keeps it there until every lane of the slot has read its row.
                D a[NB];
                if (ta) {
                    for (int e = tt; e < NB * NB; e += kTpm) {
                        const int l = e % NB, rr = e / NB;
                        D v{};
                        if (live && l < k && rr < m) v = Ab[l + rr * lda];
                        sB[rr + l * kLdb] = ca ? dev_conj(v) : v;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
BATCHLAS_UNROLL_FULL
                    for (int l = 0; l < NB; ++l) a[l] = sB[r + l * kLdb];
                    it.barrier(sycl::access::fence_space::local_space);
                } else {
BATCHLAS_UNROLL_FULL
                    for (int l = 0; l < NB; ++l) {
                        D v{};
                        if (live && r < m && l < k) v = Ab[r + l * lda];
                        a[l] = v;
                    }
                }

                // op(B) into sB[l + c * kLdb], walked in STORAGE order so the global
                // read is coalesced either way; zero outside k x n.
                for (int e = tt; e < NB * NB; e += kTpm) {
                    const int s0 = e % NB, s1 = e / NB;
                    const int l = tb ? s1 : s0;
                    const int c = tb ? s0 : s1;
                    D v{};
                    if (live && l < k && c < n) v = Bb[s0 + s1 * ldb];
                    sB[l + c * kLdb] = cb ? dev_conj(v) : v;
                }
                it.barrier(sycl::access::fence_space::local_space);

                D acc[kCpt];
BATCHLAS_UNROLL_FULL
                for (int j = 0; j < kCpt; ++j) acc[j] = D{};
                // Sequential (not nested) loops of kLc l each: one complex loop over all 64 l is
                // past LLVM's pragma unroll threshold, so a[] went to the stack (512 B). A macro: a
                // lambda per chunk put 160-184 B of stack on every bucket, real ones included.
#define BATCHLAS_SMALL_DOT(L0)                                                                  \
    BATCHLAS_UNROLL_FULL                                                                        \
    for (int l = (L0); l < (L0) + kLc; ++l) {                                                   \
        if (l >= k) continue; /* uniform; `continue` keeps the unroll, so a[] stays put */       \
        BATCHLAS_UNROLL_FULL                                                                    \
        for (int j = 0; j < kCpt; ++j) sycl_device::fma_acc(acc[j], a[l], sB[l + (c0 + j) * kLdb]); \
    }
                BATCHLAS_SMALL_DOT(0)
                if constexpr (NB / kLc > 1) {
                    BATCHLAS_SMALL_DOT(kLc)
                    BATCHLAS_SMALL_DOT(2 * kLc)
                    BATCHLAS_SMALL_DOT(3 * kLc)
                }
#undef BATCHLAS_SMALL_DOT

                if (live && r < m) {
                    D* const Cb = c_ptr + bi * sc;
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < kCpt; ++j) {
                        const int c = c0 + j;
                        if (c < n) {
                            D* const dst = Cb + r + c * ldc;
                            if constexpr (kCx) {
                                D out = sycl_device::dev_mul(alpha_d, acc[j]);
                                if (!beta_zero) sycl_device::fma_acc(out, beta_d, *dst);
                                *dst = out;
                            } else {
                                *dst = beta_zero ? alpha_d * acc[j] : alpha_d * acc[j] + beta_d * *dst;
                            }
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
template <int NB>
inline constexpr std::int64_t kSmallTiledSlmElems = std::int64_t(2) * NB * (NB + 4);

template <typename T, int NB, bool PrefetchC>
Event launch_small_tiled(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         T alpha, T beta, int m, int n, int k) {
    using V4 = sycl_gemm::Vec4<T>;
    constexpr int kTr = NB / 4;          // lanes per row and per column of 4x4 tiles
    constexpr int kWg = kTr * kTr;
    constexpr int kLd = NB + 4;          // keeps the V4 reads 16-byte aligned
    static_assert(NB % 4 == 0, "4x4 lane tiles");
    static_assert(2 * NB * kLd == kSmallTiledSlmElems<NB>, "choice.hh's bytes");

    const int batch = static_cast<int>(A.batch_size());
    const T* a_ptr = A.data_ptr();
    const T* b_ptr = B.data_ptr();
    T* c_ptr = C.data_ptr();
    const std::ptrdiff_t lda = A.ld(), ldb = B.ld(), ldc = C.ld();
    const std::ptrdiff_t sa = A.stride(), sb = B.stride(), sc = C.stride();
    const bool beta_zero = (beta == T(0));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> slm(sycl::range<1>(static_cast<std::size_t>(2 * NB * kLd)), h);
        batchlas::impl::parallel_for<GemmSmallTiledKernel<T, NB, PrefetchC>>(h,
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
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < 4; ++j) {
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) {
                            const int r = tr * 4 + i, c = tc * 4 + j;
                            prior[i][j] = (r < m && c < n) ? Cb[r + c * ldc] : T(0);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                T acc[4][4];
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < 4; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
                }
                for (int l0 = 0; l0 < kp; l0 += 4) {
                    V4 bv[4];
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < 4; ++j) {
                        bv[j] = *reinterpret_cast<const V4*>(&sB[(tc * 4 + j) * kLd + l0]);
                    }
BATCHLAS_UNROLL_FULL
                    for (int ll = 0; ll < 4; ++ll) {
                        const V4 av = *reinterpret_cast<const V4*>(&sA[(l0 + ll) * kLd + tr * 4]);
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) {
BATCHLAS_UNROLL_FULL
                            for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j][ll];
                        }
                    }
                }

BATCHLAS_UNROLL_FULL
                for (int j = 0; j < 4; ++j) {
                    const int c = tc * 4 + j;
BATCHLAS_UNROLL_FULL
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
        if constexpr (kSmallMaxDimFor<T> >= 64)
            return launch_small_batched<T, 64>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k);
        else throw ::batchlas::internal_error("gemm small: max(m, n, k) above this scalar's ceiling reached the kernel");
    }
}

}  // namespace batchlas::sycl_gemm_small

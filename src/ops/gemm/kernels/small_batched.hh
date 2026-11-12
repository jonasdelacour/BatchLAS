// SPDX-License-Identifier: MIT
#pragma once

// Batched GEMM for orders \f$\le 64\f$: several matrices per work-group, op(B) staged in local
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
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device.hh>
#include <batchlas/device/level3/epilogue.hh>
#include <batchlas/device/level3/mma.hh>
#include <batchlas/device/level3/packet.hh>
#include <batchlas/device/level3/stage.hh>
#include <src/ops/gemm/choice.hh>
#include <src/ops/gemm/gemm_kernels.hh>

namespace batchlas::ops::gemm {


template <typename T, int NB, bool Masked>
class GemmSmallBatchedKernel;

// kSmallWg, kSmallMaxDim (the largest order served; \f$\max(m, n, k)\f$ picks the bucket) and
// kSmallTiledMaxDim are choice.hh's.

// A 16-byte scalar stops at 32: its 64 bucket stages \f$64 \times 65 \times 16\f$ B = 66.5 KB and holds a
// 64-entry row plus 32 accumulators, 384 registers. choice.hh's small_max_dim mirrors it.
template <typename T>
inline constexpr int kSmallMaxDimFor = sizeof(T) >= 16 ? 32 : kSmallMaxDim;

inline int small_bucket(int max_dim) {
    return max_dim <= 8 ? 8 : max_dim <= 16 ? 16 : max_dim <= 32 ? 32 : 64;
}

// Elements of local memory the batched kernel stages per work-group (choice.hh's small_slm_bytes).
template <int NB>
inline constexpr std::int64_t kSmallBatchedSlmElems = std::int64_t(kSmallWg / (2 * NB)) * NB * (NB + 1);

// Masked: the batch mask folds into `live` (several items share a work-group's barriers, so an
// inactive one is clamped like a tail slot, never returned). A template so the unmasked instance
// is the kernel as it was: the fold cost 2-4.5% unmasked in a packed kernel (potrf tiny).
// evidence: docs/design/masked-batches.md#masked-batches-kernel-contract
template <typename T, int NB, bool Masked>
Event launch_small_batched(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha, T beta, Transpose transA, Transpose transB,
                           int m, int n, int k, const std::uint8_t* mask) {
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
        batchlas::impl::parallel_for<GemmSmallBatchedKernel<T, NB, Masked>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) * kSmallWg),
                              sycl::range<1>(kSmallWg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int t = static_cast<int>(it.get_local_id(0));
                const int slot = t / kTpm;
                const int tt = t % kTpm;
                const int mat = static_cast<int>(it.get_group(0)) * kG + slot;
                // Clamped, never returned: every lane must reach the barriers below.
                bool live = mat < batch;
                if constexpr (Masked) live = live && mask[mat] != 0;
                const std::ptrdiff_t bi = live ? mat : 0;
                const int r = tt % NB;
                const int c0 = (tt / NB) * kCpt;
                D* const sB = &slm[0] + static_cast<std::ptrdiff_t>(slot) * NB * kLdb;
                const D* const Ab = a_ptr + bi * sa;
                const D* const Bb = b_ptr + bi * sb;

                // Row r of op(A), unrolled so the array stays in registers. A transposed A is
                // stored \f$k \times m\f$, so a lane's row is a stored column, lda from its neighbours':
                // it goes through sB in storage order (coalesced) as sB[r + l * kLdb], and the
                // second barrier keeps it there until every lane of the slot has read its row.
                const device::WorkGroup<kSmallWg> ex{it};
                using Ops = device::arith::Fma;  // sycl_device's fma_acc / dev_mul, bit for bit
                device::Frag<D, NB> a;
                if (ta) {
                    device::l3::stage_storage<kTpm>(tt, device::Ext<NB>{}, device::Ext<NB>{}, live, Ab, lda, false, ca, k, m,
                                            [&](int l, int rr) -> D& { return sB[rr + l * kLdb]; });
                    ex.barrier();
                    device::load(a, [&](int l) { return sB[r + l * kLdb]; });
                    ex.barrier();
                } else {
                    device::load(a, [&](int l) {
                        D v{};
                        if (live && r < m && l < k) v = Ab[r + l * lda];
                        return v;
                    });
                }

                // op(B) into sB[l + c * kLdb], walked in STORAGE order so the global
                // read is coalesced either way; zero outside k x n.
                device::l3::stage_storage<kTpm>(tt, device::Ext<NB>{}, device::Ext<NB>{}, live, Bb, ldb, tb, cb, k, n,
                                        [&](int l, int c) -> D& { return sB[l + c * kLdb]; });
                ex.barrier();

                // acc(j) := op(A)(r, :) op(B)(:, c0 + j) in kLc-wide unrolled chunks: one complex loop
                // over all 64 l is past LLVM's pragma unroll threshold, so a[] went to the stack (512 B).
                device::Frag<D, kCpt> acc;
                device::fill(acc, D{});
                device::l3::gemv_t_frag<kLc, Ops>(acc, a, [&](int l, int j) { return sB[l + (c0 + j) * kLdb]; }, k);

                if (live && r < m) {
                    D* const Cb = c_ptr + bi * sc;
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < kCpt; ++j) {
                        const int c = c0 + j;
                        if (c < n) device::store_epi<device::epi::MulMac, Ops>(alpha_d, acc[j], device::MaybeZero<D>{beta_d, beta_zero}, *(Cb + r + c * ldc));
                    }
                }
            });
    });
    return ctx.get_event();
}

template <typename T, int NB, bool PrefetchC>
class GemmSmallTiledKernel;

// NN, \f$32 < \max(m, n, k) \le\f$ kSmallTiledMaxDim: one matrix per work-group, A
// ([k][m]) and B (as stored) both staged in local memory, and a 4x4 register
// tile of C per lane, so an FMA costs 1/8 of a shared load instead of one.
// With \f$\beta \ne 0\f$ the lane reads its C tile BEFORE the barrier so that latency
// overlaps the staging; at \f$\beta = 0\f$ that prefetch is compiled out, because
// its registers cost more than it saves. evidence: docs/perf/gemm.md#the-small-tiled-kernel
template <int NB>
inline constexpr std::int64_t kSmallTiledSlmElems = std::int64_t(2) * NB * (NB + 4);

template <typename T, int NB, bool PrefetchC>
Event launch_small_tiled(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         T alpha, T beta, int m, int n, int k, const std::uint8_t* mask) {
    using V4 = device::l3::Vec4<T>;
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
                if (ops::gemm::item_off(mask, bi)) return;  // one item per work-group: before the barrier
                const int kp = (k + 3) & ~3;
                T* const sA = &slm[0];           // sA[l * kLd + r]
                T* const sB = sA + NB * kLd;     // sB[c * kLd + l]
                const T* const Ab = a_ptr + bi * sa;
                const T* const Bb = b_ptr + bi * sb;
                T* const Cb = c_ptr + bi * sc;
                // Storage order, so both reads are coalesced; zero-padded.
                device::l3::stage_storage<kWg>(t, device::Ext<NB>{}, device::Dyn{kp}, true, Ab, lda, false, false, m, k,
                                       [&](int r, int l) -> T& { return sA[l * kLd + r]; });
                device::l3::stage_storage<kWg>(t, device::Dyn{kp}, device::Ext<NB>{}, true, Bb, ldb, false, false, k, n,
                                       [&](int l, int c) -> T& { return sB[c * kLd + l]; });
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
                it.barrier(sycl::access::fence_space::local_space);  // WG 144 / 196: not a v2::WorkGroup

                // acc += A(:, l) B(l, :) on the lane's 4x4 tile, four l per 16-byte read of each operand.
                device::Tile<T, 4, 4> acc;
                device::fill(acc, T(0));
                for (int l0 = 0; l0 < kp; l0 += 4) {
                    V4 bv[4];
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < 4; ++j) {
                        bv[j] = *reinterpret_cast<const V4*>(&sB[(tc * 4 + j) * kLd + l0]);
                    }
BATCHLAS_UNROLL_FULL
                    for (int ll = 0; ll < 4; ++ll) {
                        const V4 av = *reinterpret_cast<const V4*>(&sA[(l0 + ll) * kLd + tr * 4]);
                        const T a4[4] = {av[0], av[1], av[2], av[3]};
                        const T b4[4] = {bv[0][ll], bv[1][ll], bv[2][ll], bv[3][ll]};
                        device::l3::outer_mac(acc, a4, b4);
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
                            // acc.v, not acc(i, j): a direct operator() call in the kernel lambda
                            // would become SemaSYCL's force-inlined "body" (scripts/check_kernel_entry_calls.py).
                            if constexpr (PrefetchC) *dst = alpha * acc.v[i][j] + beta * prior[i][j];
                            else device::store_epi<device::epi::MulMac, device::arith::Fma>(alpha, acc.v[i][j], device::MaybeZero<T>{beta, beta_zero}, *dst);
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
                  T alpha, T beta, int m, int n, int k, const std::uint8_t* mask) {
    if (beta == T(0)) {
        return launch_small_tiled<T, NB, false>(ctx, A, B, C, alpha, beta, m, n, k, mask);
    }
    return launch_small_tiled<T, NB, true>(ctx, A, B, C, alpha, beta, m, n, k, mask);
}

template <typename T, int NB>
Event launch_small_batched_leg(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                               const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                               T alpha, T beta, Transpose transA, Transpose transB, int m, int n, int k,
                               const std::uint8_t* mask) {
    if (mask) return launch_small_batched<T, NB, true>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k, mask);
    return launch_small_batched<T, NB, false>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k, nullptr);
}

template <typename T>
Event small_batched(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    const MatrixView<T, MatrixFormat::Dense>& B,
                    const MatrixView<T, MatrixFormat::Dense>& C,
                    T alpha, T beta, Transpose transA, Transpose transB,
                    int m, int n, int k) {
    const int max_dim = m > n ? (m > k ? m : k) : (n > k ? n : k);
    const std::uint8_t* mask = ops::gemm::item_mask(A, B, C);
    if constexpr (std::is_same_v<T, float>) {
        if (transA == Transpose::NoTrans && transB == Transpose::NoTrans && max_dim > 32 &&
            max_dim <= kSmallTiledMaxDim) {
            return max_dim <= 48 ? small_tiled<T, 48>(ctx, A, B, C, alpha, beta, m, n, k, mask)
                                 : small_tiled<T, 56>(ctx, A, B, C, alpha, beta, m, n, k, mask);
        }
    }
    switch (small_bucket(max_dim)) {
    case 8:
        return launch_small_batched_leg<T, 8>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k, mask);
    case 16:
        return launch_small_batched_leg<T, 16>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k, mask);
    case 32:
        return launch_small_batched_leg<T, 32>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k, mask);
    default:
        if constexpr (kSmallMaxDimFor<T> >= 64)
            return launch_small_batched_leg<T, 64>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k, mask);
        else throw ::batchlas::internal_error("gemm small: max(m, n, k) above this scalar's ceiling reached the kernel");
    }
}

}  // namespace batchlas::ops::gemm

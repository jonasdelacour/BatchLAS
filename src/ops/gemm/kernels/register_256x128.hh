#pragma once

// Float 256x128 and 128x256 tiles, k = 8 slabs, 4 copy_async stages, one 256-lane group per SM: the
// shape of cuBLAS's cutlass_80_simt_sgemm_{256x128,128x256}_8x4 on sm_120. A thread holds TM x TN
// accumulators (16x8 or 8x16) as 4-row/4-column bands, read by LDS.128 from padded [k][m] / [k][n]
// slabs. Every copy is one zero-filled 4-byte element (any ld, any base): a packet operand (A, B^T)
// takes consecutive m (n) per lane, a k-contiguous one (B, A^T) k fastest, 8 lanes per 32-byte row
// segment; the stride pad of 4 keeps those transposed writes conflict-free.
// evidence: docs/perf/gemm.md#gemm-wp-w-a-256x128-float-tile-with-four-copy-async-stages

#include <src/ops/gemm/kernels/copy_async.hh>
#include <src/ops/gemm/kernels/epilogue_linear.hh>
#include <src/ops/gemm/kernels/register_128x128.hh>
#include <src/ops/gemm/kernels/vec4.hh>

#include <src/ops/gemm/gemm_kernels.hh>

#include <src/linalg-impl.hh>
#include <src/sycl/kernel_attrs.hh>

#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

template <int TM, int TN, Transpose OpA, Transpose OpB>
class GemmRegisterK8S4LargeKernel;

template <int TM, int TN, Transpose OpA, Transpose OpB>
struct GemmK8S4LargeBody {
    static constexpr int TileM = 16 * TM, TileN = 16 * TN, TileK = 8, Threads = 256, Stages = 4;
    static constexpr int BM = TM / 4, BN = TN / 4;            // 4-wide bands per thread
    static constexpr int AS = TileM + 4, BS = TileN + 4;       // padded slab strides
    static constexpr int SlabA = TileK * AS, SlabB = TileK * BS;
    static constexpr int SC = TileM + 4, Slots = TileN / 4;    // staged C: one pass = TileN / 4 columns
    static constexpr int Elems = Stages * (SlabA + SlabB);
    static_assert(Slots * SC <= Elems, "a staged C pass fits the slabs");
    static constexpr bool AK = OpA != Transpose::NoTrans, BK = OpB == Transpose::NoTrans;
    static constexpr KernelVariant kVariant =
        TM == 16 ? (!AK ? (BK ? KernelVariant::Tiled256x128RegisterK8S4 : KernelVariant::Tiled256x128RegisterK8S4NT)
                        : (BK ? KernelVariant::Tiled256x128RegisterK8S4TN : KernelVariant::Tiled256x128RegisterK8S4TT))
                 : (!AK ? (BK ? KernelVariant::Tiled128x256RegisterK8S4 : KernelVariant::Tiled128x256RegisterK8S4NT)
                        : (BK ? KernelVariant::Tiled128x256RegisterK8S4TN : KernelVariant::Tiled128x256RegisterK8S4TT));

    const float* a_ptr;
    const float* b_ptr;
    float* c_ptr;
    int m, n, k, lda, ldb, ldc;
    std::ptrdiff_t stride_a, stride_b, stride_c;
    int batch;
    float alpha, beta;
    sycl::local_accessor<float, 1> tile;

    // One operand's slab: E elements of extent `ext` from e0, packet (down E) or k-contiguous.
    template <int E, int S, bool KC>
    struct Copier {
        const float* g;
        int e, kk;  // this lane's offset along E and along k
        __attribute__((always_inline)) void issue(float* s, std::uint32_t s_l, int off, int k0, int k, int lim,
                                                  int ld) {
            const bool kin = k0 + kk < k;
            const int left = kin ? lim : 0;
            std::ptrdiff_t ld32 = 0;
            if constexpr (KC) {
                int l = ld;
#if BATCHLAS_GEMM_CP_ASYNC
                asm volatile("" : "+r"(l));
#endif
                ld32 = static_cast<std::ptrdiff_t>(32) * l;
            }
BATCHLAS_UNROLL_FULL
            for (int i = 0; i < E / 32; ++i) {
                const int o = off + kk * S + e + 32 * i;
                copy_async_zfill<4>(s + o, s_l + 4u * o, KC ? g + ld32 * i : g + 32 * i, 32 * i < left ? 4u : 0u);
            }
            if constexpr (KC) g += TileK;
            else g += static_cast<std::ptrdiff_t>(TileK) * ld;
        }
    };

    __attribute__((always_inline)) void operator()(sycl::nd_item<3> item) const {
        const int bid = static_cast<int>(item.get_group(0));
        if (bid >= batch) return;
        const int tid = static_cast<int>(item.get_local_id(2));
        const int lane = tid & 31, warp = tid >> 5;
        // The 128x128 kernel's lane swizzle and 2 x 4 warp grid, bands 32 rows / 16 columns apart.
        const int ml = (lane & 1) | (((lane >> 2) & 3) << 1);
        const int nl = ((lane >> 1) & 1) | (((lane >> 4) & 1) << 1);
        const int mb = (warp & 1) * (TileM / 2) + ml * 4;
        const int nb = (warp >> 1) * (TileN / 4) + nl * 4;
        const int m0 = static_cast<int>(item.get_group(1)) * TileM;
        const int n0 = static_cast<int>(item.get_group(2)) * TileN;
        const float* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
        const float* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
        float* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
        float* sa = tile.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* sb = sa + Stages * SlabA;
        std::uint32_t sa_l = 0;
#if BATCHLAS_GEMM_CP_ASYNC
        asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(sa_l) : "l"(sa));
#endif
        const std::uint32_t sb_l = sa_l + 4u * Stages * SlabA;

        const int a_e = AK ? tid / 8 : tid % 32, a_k = AK ? tid % 8 : tid / 32;
        const int b_e = BK ? tid / 8 : tid % 32, b_k = BK ? tid % 8 : tid / 32;
        Copier<TileM, AS, AK> ca{AK ? Ab + a_k + static_cast<std::ptrdiff_t>(m0 + a_e) * lda
                                    : Ab + (m0 + a_e) + static_cast<std::ptrdiff_t>(a_k) * lda,
                                 a_e, a_k};
        Copier<TileN, BS, BK> cb{BK ? Bb + b_k + static_cast<std::ptrdiff_t>(n0 + b_e) * ldb
                                    : Bb + (n0 + b_e) + static_cast<std::ptrdiff_t>(b_k) * ldb,
                                 b_e, b_k};
        const int lim_a = m - (m0 + a_e), lim_b = n - (n0 + b_e);

        float acc[TM][TN];
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TM; ++i)
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

        auto issue = [&](int k0, auto buf) __attribute__((always_inline)) {
            constexpr int b = decltype(buf)::value;
            ca.issue(sa, sa_l, b * SlabA, k0, k, lim_a, lda);
            cb.issue(sb, sb_l, b * SlabB, k0, k, lim_b, ldb);
        };
        auto compute = [&](auto buf) __attribute__((always_inline)) {
            const float* xa = sa + decltype(buf)::value * SlabA + mb;
            const float* xb = sb + decltype(buf)::value * SlabB + nb;
BATCHLAS_UNROLL_FULL
            for (int kk = 0; kk < TileK; ++kk) {
                float af[TM], bf[TN];
BATCHLAS_UNROLL_FULL
                for (int q = 0; q < BM; ++q) {
                    const Vec4<float> v = vec4_ref(&xa[kk * AS + 32 * q]);
                    af[4 * q] = v.x, af[4 * q + 1] = v.y, af[4 * q + 2] = v.z, af[4 * q + 3] = v.w;
                }
BATCHLAS_UNROLL_FULL
                for (int q = 0; q < BN; ++q) {
                    const Vec4<float> v = vec4_ref(&xb[kk * BS + 16 * q]);
                    bf[4 * q] = v.x, bf[4 * q + 1] = v.y, bf[4 * q + 2] = v.z, bf[4 * q + 3] = v.w;
                }
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < TM; ++i)
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < TN; ++j) acc[i][j] += af[i] * bf[j];
            }
        };

        // Stages - 1 slabs in flight: step s waits for slab s (own copies, then the barrier for
        // everyone's), refills the buffer slab s - 1 was read from, commits (empty past the end) and
        // computes. Unrolled by Stages so the buffer is a constant.
        const int slabs = (k + TileK - 1) / TileK;
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
            compute(buf);
        };
        for (int s0 = 0; s0 < slabs; s0 += Stages) {
            [&]<int... J>(std::integer_sequence<int, J...>) __attribute__((always_inline)) {
                ((s0 + J < slabs && (step(s0 + J, std::integral_constant<int, J>{}), true)) && ...);
            }(std::make_integer_sequence<int, Stages>{});
        }
        copy_async_wait<0>();

        // Staged epilogue: pass p stages each thread's column p of every n band (slot = column / 4),
        // then each warp stores Slots / 8 whole columns, rows lane + 32 i (one 128-byte piece per
        // warp store at any ldc). The pass's priors load after the stage write, all in flight; none
        // at beta = 0, where the store is alpha * acc alone (keeps -0.0).
        // evidence: docs/perf/gemm.md#gemm-wp-f-float-mid-size-parity-on-sm_120
        constexpr int CW = Slots / 8, RW = TileM / 32;
        float* sc = sa;
        int t2 = static_cast<int>(item.get_local_id(2));
#if BATCHLAS_GEMM_CP_ASYNC
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t2));
#endif
        const int row = t2 & 31, wslot = (t2 >> 5) * CW;
        const int rows_left = m - m0 - row;
        const bool bz = beta == 0.f;
        item.barrier(sycl::access::fence_space::local_space);
BATCHLAS_UNROLL_FULL
        for (int p = 0; p < 4; ++p) {
BATCHLAS_UNROLL_FULL
            for (int bn = 0; bn < BN; ++bn) {
                const int slot = (nb >> 2) + 4 * bn;
BATCHLAS_UNROLL_FULL
                for (int bm = 0; bm < BM; ++bm) {
                    Vec4<float> v;
BATCHLAS_UNROLL_FULL
                    for (int i = 0; i < 4; ++i) v[i] = acc[4 * bm + i][4 * bn + p];
                    vec4_ref(&sc[slot * SC + 32 * bm + mb]) = v;
                }
            }
            float prior[CW][RW];
BATCHLAS_UNROLL_FULL
            for (int q = 0; q < CW; ++q) {
                const int col = n0 + (wslot + q) * 4 + p;
                const float* src = &Cb[m0 + row + static_cast<std::ptrdiff_t>(col) * ldc];
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < RW; ++i) prior[q][i] = !bz && col < n && 32 * i < rows_left ? src[32 * i] : 0.f;
            }
            item.barrier(sycl::access::fence_space::local_space);
BATCHLAS_UNROLL_FULL
            for (int q = 0; q < CW; ++q) {
                const int col = n0 + (wslot + q) * 4 + p;
                if (col >= n) continue;
                float* dst = &Cb[m0 + row + static_cast<std::ptrdiff_t>(col) * ldc];
                const float* src = &sc[(wslot + q) * SC + row];
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < RW; ++i)
                    if (32 * i < rows_left)
                        dst[32 * i] = bz ? alpha * src[32 * i]
                                         : LinearEpilogue<float>::apply(alpha, beta, src[32 * i], prior[q][i]);
            }
            item.barrier(sycl::access::fence_space::local_space);
        }
    }
};

// The kernel functor: the body always inlines into it (out of line, the copy_async asm's memory
// clobber reloaded every member through a stack copy of the functor, WP-N2), and the work-group
// size is declared so ptxas sees .reqntid and applies the one-group bound (known defect #20).
template <typename Body>
struct GemmK8S4LargeKernel {
    Body body;

    BATCHLAS_LAUNCH_BOUNDS(Body::Threads, 1)
    BATCHLAS_GEMM_REQD_WG(Body::Threads)
    __attribute__((always_inline)) void operator()(sycl::nd_item<3> item) const {
        [[clang::always_inline]] body(item);
    }
};

template <int TM, int TN, Transpose OpA, Transpose OpB>
Event launch_register_k8_s4_large(Queue& ctx, const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& B,
                                  const MatrixView<float, MatrixFormat::Dense>& C, float alpha, float beta,
                                  const char* (*kernel_trace_name)(KernelVariant)) {
    using Body = GemmK8S4LargeBody<TM, TN, OpA, OpB>;
    BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(Body::kVariant));
    const auto [m, k] = get_effective_dims(A, OpA);
    const int n = get_effective_dims(B, OpB).second;
    const int group_rows = (m + Body::TileM - 1) / Body::TileM;
    const int group_cols = (n + Body::TileN - 1) / Body::TileN;
    const sycl::range<3> local(1, 1, Body::Threads);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()), static_cast<size_t>(group_rows),
                                static_cast<size_t>(group_cols) * Body::Threads);
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> tile(sycl::range<1>(Body::Elems), h);
        Body body{A.data_ptr(), B.data_ptr(), C.data_ptr(), m, n, k, A.ld(), B.ld(), C.ld(),
                  A.stride(), B.stride(), C.stride(), A.batch_size(), alpha, beta, tile};
        batchlas::impl::parallel_for<GemmRegisterK8S4LargeKernel<TM, TN, OpA, OpB>>(
            h, sycl::nd_range<3>(global, local), GemmK8S4LargeKernel<Body>{body});
    });
    return ctx.get_event();
}

}  // namespace batchlas::sycl_gemm

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

template <int TM, int TN, Transpose OpA, Transpose OpB, bool Vec16>
class GemmRegisterK8S4LargeKernel;

// Vec16: the packet operands copy 16-byte granules (their bases, ld and stride are 4-element aligned).
template <int TM, int TN, Transpose OpA, Transpose OpB, bool Vec16>
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

    // One operand's copies of a slab, packet (down E) or k-contiguous (KC): E / 32 elements per lane,
    // or on the Vec leg (packet only) E / 128 granules, 1024 / E k rows apart, edge granules clipped.
    template <int E, int S, bool KC>
    struct Copier {
        static constexpr bool Vec = Vec16 && !KC;
        static constexpr int R = 1024 / E, N = Vec ? E / 128 : E / 32;
        static constexpr int lanes_e(int tid) { return Vec ? 4 * (tid % (E / 4)) : KC ? tid / 8 : tid % 32; }
        static constexpr int lanes_k(int tid) { return Vec ? tid / (E / 4) : KC ? tid % 8 : tid / 32; }
        const float* g;
        int e, kk, lim;  // this lane's offsets along E and k; elements left along E
        int left = 0;    // per slab: lim (0 past k: every copy zero-filled); Vec: k rows left
        __attribute__((always_inline)) void begin(int k0, int k) {
            if constexpr (Vec) left = k - k0 - kk;
            else left = k0 + kk < k ? lim : 0;
        }
        __attribute__((always_inline)) void copy(int i, float* s, std::uint32_t s_l, int off, int ld) const {
            if constexpr (Vec) {
                const int o = off + (kk + R * i) * S + e;
                const std::uint32_t bytes = lim >= 4 ? 16u : lim > 0 ? 4u * lim : 0u;
                copy_async_zfill<16>(s + o, s_l + 4u * o, g + static_cast<std::ptrdiff_t>(R * i) * ld,
                                     R * i < left ? bytes : 0u);
            } else {
                const int o = off + kk * S + e + 32 * i;
                const float* src = KC ? g + static_cast<std::ptrdiff_t>(32 * i) * ld : g + 32 * i;
                copy_async_zfill<4>(s + o, s_l + 4u * o, src, 32 * i < left ? 4u : 0u);
            }
        }
        __attribute__((always_inline)) void bump(int ld) {
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

        using CA = Copier<TileM, AS, AK>;
        using CB = Copier<TileN, BS, BK>;
        const int a_e = CA::lanes_e(tid), a_k = CA::lanes_k(tid);
        const int b_e = CB::lanes_e(tid), b_k = CB::lanes_k(tid);
        CA ca{AK ? Ab + a_k + static_cast<std::ptrdiff_t>(m0 + a_e) * lda
                                    : Ab + (m0 + a_e) + static_cast<std::ptrdiff_t>(a_k) * lda,
                                 a_e, a_k, m - (m0 + a_e)};
        CB cb{BK ? Bb + b_k + static_cast<std::ptrdiff_t>(n0 + b_e) * ldb
                                    : Bb + (n0 + b_e) + static_cast<std::ptrdiff_t>(b_k) * ldb,
                                 b_e, b_k, n - (n0 + b_e)};

        float acc[TM][TN];
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TM; ++i)
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

        // A slab's copies issue back to back after the step's barrier: spread over the k steps of the
        // compute they queued behind the fragment reads instead (2-4% slower; WP-W).
        auto issue = [&](int k0, auto buf) __attribute__((always_inline)) {
            constexpr int b = decltype(buf)::value;
            ca.begin(k0, k);
            cb.begin(k0, k);
BATCHLAS_UNROLL_FULL
            for (int i = 0; i < CA::N; ++i) ca.copy(i, sa, sa_l, b * SlabA, lda);
BATCHLAS_UNROLL_FULL
            for (int i = 0; i < CB::N; ++i) cb.copy(i, sb, sb_l, b * SlabB, ldb);
            ca.bump(lda);
            cb.bump(ldb);
            copy_async_commit();
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
        // everyone's), refills the buffer slab s - 1 was read from, commits and computes. A slab past
        // k is issued anyway: every copy is zero-filled (src-size 0, nothing read) into a buffer no
        // step reads, so no branch guards it. Unrolled by Stages so the buffer is a constant.
        const int slabs = (k + TileK - 1) / TileK;
        [&]<int... J>(std::integer_sequence<int, J...>) __attribute__((always_inline)) {
            (issue(J * TileK, std::integral_constant<int, J>{}), ...);
        }(std::make_integer_sequence<int, Stages - 1>{});
        auto step = [&](int s, auto buf) __attribute__((always_inline)) {
            constexpr int b = decltype(buf)::value;
            copy_async_wait<Stages - 2>();
            item.barrier(sycl::access::fence_space::local_space);
            issue((s + Stages - 1) * TileK, std::integral_constant<int, (b + Stages - 1) % Stages>{});
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
        // warp store at any ldc). Pass p + 1's priors load while pass p stores, so one prior latency
        // is exposed, not four; none at beta = 0, where the store is alpha * acc alone (keeps -0.0).
        // evidence: docs/perf/gemm.md#gemm-wp-w-a-256x128-float-tile-with-four-copy-async-stages
        constexpr int CW = Slots / 8, RW = TileM / 32;
        float* sc = sa;
        int t2 = static_cast<int>(item.get_local_id(2));
#if BATCHLAS_GEMM_CP_ASYNC
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t2));
#endif
        const int row = t2 & 31, wslot = (t2 >> 5) * CW;
        const int rows_left = m - m0 - row;
        const bool bz = beta == 0.f;
        float prior[2][CW][RW];
        auto load_priors = [&](int p, float (&pr)[CW][RW]) __attribute__((always_inline)) {
BATCHLAS_UNROLL_FULL
            for (int q = 0; q < CW; ++q) {
                const int col = n0 + (wslot + q) * 4 + p;
                const float* src = &Cb[m0 + row + static_cast<std::ptrdiff_t>(col) * ldc];
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < RW; ++i) pr[q][i] = !bz && col < n && 32 * i < rows_left ? src[32 * i] : 0.f;
            }
        };
        load_priors(0, prior[0]);
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
            item.barrier(sycl::access::fence_space::local_space);
            if (p + 1 < 4) load_priors(p + 1, prior[(p + 1) & 1]);
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
                                         : LinearEpilogue<float>::apply(alpha, beta, src[32 * i], prior[p & 1][q][i]);
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

template <int TM, int TN, Transpose OpA, Transpose OpB, bool Vec16>
Event launch_register_k8_s4_large_leg(Queue& ctx, const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& B,
                                  const MatrixView<float, MatrixFormat::Dense>& C, float alpha, float beta,
                                  const char* (*kernel_trace_name)(KernelVariant)) {
    using Body = GemmK8S4LargeBody<TM, TN, OpA, OpB, Vec16>;
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
        batchlas::impl::parallel_for<GemmRegisterK8S4LargeKernel<TM, TN, OpA, OpB, Vec16>>(
            h, sycl::nd_range<3>(global, local), GemmK8S4LargeKernel<Body>{body});
    });
    return ctx.get_event();
}

// The leg is derived: 16-byte packet copies whenever every packet operand allows them (TN has none).
template <int TM, int TN, Transpose OpA, Transpose OpB>
Event launch_register_k8_s4_large(Queue& ctx, const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& B,
                                  const MatrixView<float, MatrixFormat::Dense>& C, float alpha, float beta,
                                  const char* (*kernel_trace_name)(KernelVariant),
                                  const char* (*aligned_trace_name)(KernelVariant)) {
    constexpr bool pa = OpA == Transpose::NoTrans, pb = OpB != Transpose::NoTrans;
    auto aligned = [](const float* p, int ld, std::ptrdiff_t stride) {
        return (reinterpret_cast<std::uintptr_t>(p) % 16) == 0 && ld % 4 == 0 && stride % 4 == 0;
    };
    const char* e = std::getenv("BATCHLAS_W_NOVEC");  // WP-W EXPLORATION ONLY (remove before shipping)
    if constexpr (pa || pb) {
        if (!(e && *e == '1') && (!pa || aligned(A.data_ptr(), A.ld(), A.stride())) &&
            (!pb || aligned(B.data_ptr(), B.ld(), B.stride())))
            return launch_register_k8_s4_large_leg<TM, TN, OpA, OpB, true>(ctx, A, B, C, alpha, beta,
                                                                           aligned_trace_name);
    }
    return launch_register_k8_s4_large_leg<TM, TN, OpA, OpB, false>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

}  // namespace batchlas::sycl_gemm

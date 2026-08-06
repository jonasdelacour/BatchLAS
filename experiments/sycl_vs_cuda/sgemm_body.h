// Shared SGEMM kernel body, compiled verbatim by BOTH nvcc and DPC++.
//
// The point of this file: every arithmetic instruction, every shared-memory
// index and every load width is identical between the CUDA and the SYCL build.
// The only things that differ are how a thread learns its coordinates, how it
// reaches shared memory, and how it barriers -- and those are macros supplied
// by the includer. Any performance gap between the two builds is therefore
// attributable to the toolchain, not to the kernel design.
//
// Tile:  128x128 block tile, K-step 8, 256 threads, 8x8 accumulators/thread.
// Each thread owns two 4-row bands (ty*4 and 64+ty*4) and two 4-col bands
// (tx*4 and 64+tx*4). That split is what makes the LDS.128 fragment loads
// bank-conflict free: 8 lanes x 4 floats = exactly 32 banks per phase.
//
// Operands are column-major, as in BLAS: A is m x k with leading dimension
// lda, B is k x n with ldb, C is m x n with ldc. NN only, alpha/beta applied
// in the epilogue.

#ifndef SGEMM_BODY_H
#define SGEMM_BODY_H

#define SG_BM 128
#define SG_BN 128
#define SG_BK 8
#define SG_THREADS 256

// A 16-byte vector type with identical layout under both compilers, so that
// *reinterpret_cast<const F4*>(p) lowers to a 128-bit load in both.
struct alignas(16) F4 { float x, y, z, w; };

#define SG_LD4(p) (*reinterpret_cast<const F4*>(p))
#define SG_ST4(p, v) (*reinterpret_cast<F4*>(p) = (v))

// The kernel body. Included inside a __global__ function (CUDA) or inside a
// parallel_for lambda (SYCL), with the coordinate/shared/barrier macros bound.
//
// Requires from the includer:
//   SG_TX, SG_TY        thread coords, each 0..15
//   SG_BM_ID, SG_BN_ID  block tile coords
//   SG_BATCH_ID         batch index
//   SG_SA, SG_SB        float* to the two shared tiles (each SG_BK*128 floats)
//   SG_BARRIER()        work-group barrier

#define SGEMM_BODY(M, N, K, Ag, lda, strideA, Bg, ldb, strideB,                \
                   Cg, ldc, strideC, alpha, beta)                              \
{                                                                              \
    const int tx = SG_TX;               /* 0..15, n direction */               \
    const int ty = SG_TY;               /* 0..15, m direction */               \
    const int tid = ty * 16 + tx;                                              \
                                                                               \
    const float* Ab = (Ag) + (size_t)(SG_BATCH_ID) * (strideA);                \
    const float* Bb = (Bg) + (size_t)(SG_BATCH_ID) * (strideB);                \
    float*       Cb = (Cg) + (size_t)(SG_BATCH_ID) * (strideC);                \
                                                                               \
    const int m0 = (SG_BM_ID) * SG_BM;                                         \
    const int n0 = (SG_BN_ID) * SG_BN;                                         \
                                                                               \
    float acc[8][8];                                                           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 8; ++i) {                                              \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;                          \
    }                                                                          \
                                                                               \
    /* Global->shared staging coordinates.                                     \
       A tile is 128 rows x 8 k-slices; one float4 per thread, coalesced       \
       along m (32 lanes x 4 floats = 128 consecutive elements).               \
       B tile is 8 k-slices x 128 cols; one float4 per thread along k, then    \
       scattered into Bs[k][n] so the n direction is contiguous in shared. */  \
    const int a_m = (tid % 32) * 4;     /* 0..127 */                           \
    const int a_k = tid / 32;           /* 0..7   */                           \
    const int b_k = (tid % 2) * 4;      /* 0 or 4 */                           \
    const int b_n = tid / 2;            /* 0..127 */                           \
                                                                               \
    for (int k0 = 0; k0 < (K); k0 += SG_BK) {                                  \
        {                                                                      \
            const float* gA = Ab + (m0 + a_m) + (size_t)(k0 + a_k) * (lda);    \
            SG_ST4(&SG_SA[a_k * SG_BM + a_m], SG_LD4(gA));                     \
        }                                                                      \
        {                                                                      \
            const float* gB = Bb + (k0 + b_k) + (size_t)(n0 + b_n) * (ldb);    \
            const F4 vb = SG_LD4(gB);                                          \
            SG_SB[(b_k + 0) * SG_BN + b_n] = vb.x;                             \
            SG_SB[(b_k + 1) * SG_BN + b_n] = vb.y;                             \
            SG_SB[(b_k + 2) * SG_BN + b_n] = vb.z;                             \
            SG_SB[(b_k + 3) * SG_BN + b_n] = vb.w;                             \
        }                                                                      \
        SG_BARRIER();                                                          \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int kk = 0; kk < SG_BK; ++kk) {                                   \
            const F4 a0 = SG_LD4(&SG_SA[kk * SG_BM + ty * 4]);                 \
            const F4 a1 = SG_LD4(&SG_SA[kk * SG_BM + 64 + ty * 4]);            \
            const F4 b0 = SG_LD4(&SG_SB[kk * SG_BN + tx * 4]);                 \
            const F4 b1 = SG_LD4(&SG_SB[kk * SG_BN + 64 + tx * 4]);            \
            const float af[8] = {a0.x, a0.y, a0.z, a0.w,                       \
                                 a1.x, a1.y, a1.z, a1.w};                      \
            const float bf[8] = {b0.x, b0.y, b0.z, b0.w,                       \
                                 b1.x, b1.y, b1.z, b1.w};                      \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 8; ++i) {                                      \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 8; ++j) acc[i][j] += af[i] * bf[j];        \
            }                                                                  \
        }                                                                      \
        SG_BARRIER();                                                          \
    }                                                                          \
                                                                               \
    /* Epilogue. A thread owns two 4-row bands, and within a band the four      \
       rows are consecutive in m -- which is the contiguous direction of a      \
       column-major C. So each band/column pair is one 128-bit store rather     \
       than four column-strided scalar ones. */                                 \
    _Pragma("unroll")                                                          \
    for (int band = 0; band < 2; ++band) {                                     \
        const int gm = m0 + band * 64 + ty * 4;                                \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 8; ++j) {                                          \
            const int gn = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);    \
            float* p = &Cb[gm + (size_t)gn * (ldc)];                           \
            F4 out;                                                            \
            out.x = (alpha) * acc[band * 4 + 0][j];                            \
            out.y = (alpha) * acc[band * 4 + 1][j];                            \
            out.z = (alpha) * acc[band * 4 + 2][j];                            \
            out.w = (alpha) * acc[band * 4 + 3][j];                            \
            if ((beta) != 0.0f) {                                              \
                const F4 prev = SG_LD4(p);                                     \
                out.x += (beta) * prev.x;                                      \
                out.y += (beta) * prev.y;                                      \
                out.z += (beta) * prev.z;                                      \
                out.w += (beta) * prev.w;                                      \
            }                                                                  \
            SG_ST4(p, out);                                                    \
        }                                                                      \
    }                                                                          \
}

#endif // SGEMM_BODY_H
